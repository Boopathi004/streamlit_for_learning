"""End-to-end question answering with layered hallucination controls.

Guards, in order:
  1. Retrieval threshold   - chunks below `min_score` are discarded; none left -> "I don't know" (LLM never called).
  2. Strict context prompt - answer only from numbered excerpts, JSON output, "I don't know" fallback.
  3. Citation enforcement  - an answer without a valid [n] citation is rejected.
  4. Number verification   - figures in the answer must appear in the cited excerpts, else confidence drops.
  5. Confidence scoring    - min(retrieval strength, model self-rating, verification); optional refusal on "low".
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .config import IDK_MESSAGE, Settings
from .llm import GroqLLM, LLMError, parse_llm_json
from .prompts import build_condense_messages, build_context, build_messages
from .retriever import Retrieval, retrieve

_RANK = {"low": 0, "medium": 1, "high": 2}
_NAME = {0: "low", 1: "medium", 2: "high"}

_SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks?|thank you|ok(ay)?|cool|great|bye|goodbye)[\s!.?]*$", re.I)


@dataclass
class Answer:
    text: str
    answerable: bool
    confidence: str = "none"            # high | medium | low | none
    citations: list[int] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)      # cited chunks, numbered like the answer
    retrieved: list[dict] = field(default_factory=list)    # every chunk Pinecone returned (debug / calibration)
    standalone_query: str = ""
    warnings: list[str] = field(default_factory=list)
    refusal_reason: str = ""
    latency_s: float = 0.0


def _src(i: int, m, kept: bool = True) -> dict:
    md = m.metadata
    return {
        "n": i, "id": m.id, "score": round(m.score, 3), "kept": kept,
        "filename": md.get("filename"), "doc_id": md.get("doc_id"), "version": md.get("version_id"),
        "updated_at": md.get("updated_at"), "department": md.get("department"), "doc_type": md.get("doc_type"),
        "location": md.get("location"), "section": md.get("section"), "is_latest": md.get("is_latest", True),
        "ocr": md.get("ocr", False), "text": md.get("text", ""),
    }


def _numbers(text: str) -> set[str]:
    text = re.sub(r"\[\d+\]", " ", text)                       # citation markers
    text = re.sub(r"(?m)^\s*\d+[.)]\s", " ", text)             # list numbering
    out = set()
    for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        tok = tok.replace(",", "").rstrip(".")
        if tok:
            out.add(tok.lstrip("0") or "0")
    return out


def unverified_numbers(answer: str, context: str) -> list[str]:
    """Numbers stated in the answer that appear nowhere in the cited context."""
    ctx = _numbers(context)
    return sorted(n for n in _numbers(answer) if n not in ctx)


def _retrieval_confidence(top: float, threshold: float) -> int:
    margin = top - threshold
    return 2 if margin >= 0.25 else 1 if margin >= 0.10 else 0


class RAGPipeline:
    def __init__(self, store, embedder, llm: GroqLLM):
        self.store, self.embedder, self.llm = store, embedder, llm

    # ------------------------------------------------------------------
    def _standalone_query(self, question: str, history: list[dict], s: Settings) -> str:
        if not history:
            return question
        try:
            q = self.llm.chat(build_condense_messages(question, history, s.history_turns), s,
                              temperature=0.0, max_tokens=80)
            q = q.strip().strip('"').splitlines()[0] if q.strip() else question
            return q if 3 <= len(q) <= 400 else question
        except LLMError:
            return question                     # fall back to the raw question

    def _refuse(self, reason: str, standalone: str, ret: Retrieval | None, t0: float, text: str = IDK_MESSAGE,
                warnings: list[str] | None = None) -> Answer:
        retrieved = [_src(i, m, m in (ret.kept if ret else [])) for i, m in enumerate(ret.all, 1)] if ret else []
        return Answer(text=text, answerable=False, confidence="none", standalone_query=standalone,
                      retrieved=retrieved, refusal_reason=reason, warnings=warnings or [],
                      latency_s=round(time.time() - t0, 2))

    # ------------------------------------------------------------------
    def ask(self, question: str, history: list[dict], s: Settings) -> Answer:
        t0 = time.time()
        if _SMALL_TALK.match(question):
            return Answer(text="Hello! Ask me anything about your company documents: policies, contracts, finance data or scanned forms.",
                          answerable=True, confidence="none", latency_s=0.0)

        standalone = self._standalone_query(question, history, s)
        ret = retrieve(self.store, self.embedder, standalone, s)

        # Guard 1: retrieval threshold
        if not ret.kept:
            best = f" (best similarity {ret.all[0].score:.2f}, threshold {s.min_score:.2f})" if ret.all else ""
            return self._refuse("no chunk above similarity threshold" + best, standalone, ret, t0)

        context = build_context(ret.kept)
        messages = build_messages(question, history, context, s.history_turns)
        raw = self.llm.chat(messages, s, json_mode=True)
        obj = parse_llm_json(raw)
        if obj is None:                          # one strict retry at temperature 0
            raw = self.llm.chat(messages, s, json_mode=True, temperature=0.0)
            obj = parse_llm_json(raw)
        if obj is None:
            return self._refuse("model did not return the required format", standalone, ret, t0)

        answerable = bool(obj.get("answerable", False))
        answer = str(obj.get("answer", "")).strip()
        llm_conf = _RANK.get(str(obj.get("confidence", "low")).lower(), 0)

        if not answerable or not answer:
            msg = answer if answer.lower().startswith("i don't know") else f"{IDK_MESSAGE} {answer}".strip()
            return self._refuse("model found the answer is not in the context", standalone, ret, t0, text=msg)

        # Guard 3: citations must exist and point at real excerpts
        n = len(ret.kept)
        cited = {int(c) for c in re.findall(r"\[(\d+)\]", answer) if 1 <= int(c) <= n}
        for c in obj.get("citations", []) or []:
            try:
                if 1 <= int(c) <= n:
                    cited.add(int(c))
            except (TypeError, ValueError):
                pass
        if not cited:
            return self._refuse("answer had no valid source citation", standalone, ret, t0)
        answer = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", answer)
        citations = sorted(cited)

        # Guard 4: numbers must be traceable to the cited text
        cited_text = "\n".join(ret.kept[c - 1].metadata.get("text", "") for c in citations)
        warnings = []
        missing = unverified_numbers(answer, cited_text)
        ver = 2
        if missing:
            ver = 0 if len(missing) >= 2 else 1
            warnings.append("These figures were not found verbatim in the cited sources: "
                            + ", ".join(missing) + ". Verify them in the source document.")
        if any(not ret.kept[c - 1].metadata.get("is_latest", True) for c in citations):
            warnings.append("The answer uses a superseded (outdated) document version.")
        if any(ret.kept[c - 1].metadata.get("ocr") for c in citations):
            warnings.append("Part of this answer comes from OCR text of a scanned file; check the original for exact values.")

        # Guard 5: confidence
        conf = min(_retrieval_confidence(ret.kept[0].score, s.min_score), llm_conf, ver)
        confidence = _NAME[conf]
        if confidence == "low" and s.refuse_low_confidence:
            return self._refuse("confidence too low", standalone, ret, t0, warnings=warnings,
                                text=f"{IDK_MESSAGE} The sources I found were not reliable enough to answer confidently.")

        sources = [_src(c, ret.kept[c - 1]) for c in citations]
        retrieved = [_src(i, m) for i, m in enumerate(ret.kept, 1)] + \
                    [_src(len(ret.kept) + j, m, False) for j, m in enumerate(x for x in ret.all if x not in ret.kept)]
        return Answer(text=answer, answerable=True, confidence=confidence, citations=citations, sources=sources,
                      retrieved=retrieved, standalone_query=standalone, warnings=warnings,
                      latency_s=round(time.time() - t0, 2))
