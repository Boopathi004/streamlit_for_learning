"""Streamlit UI for the enterprise RAG assistant.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import pathlib
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag import llm as llm_mod
from rag.config import (DEPARTMENTS, DOC_TYPES, EMBEDDERS, GROQ_MODELS, IDK_MESSAGE,
                        SUPPORTED_EXTENSIONS, Settings)
from rag.embeddings import build_embedder
from rag.ingest import delete_document, guess_metadata, ingest_file, list_documents, slugify_doc_id
from rag.pipeline import RAGPipeline
from rag.vectorstore import InMemoryStore, PineconeStore

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚", layout="wide")

SAMPLE_DIR = pathlib.Path(__file__).parent / "sample_docs"
LOG_DIR = pathlib.Path(__file__).parent / "logs"


def secret(name: str, default: str = "") -> str:
    """Environment variable first, then Streamlit secrets (if configured)."""
    if os.getenv(name):
        return os.getenv(name, default)
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# ----------------------------------------------------------------------------
# cached resources (heavy objects live across Streamlit reruns)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder(key: str, pinecone_key: str):
    pc = None
    if key == "pinecone-hosted":
        from pinecone import Pinecone
        pc = Pinecone(api_key=pinecone_key)
    return build_embedder(key, pc)


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_pinecone_store(api_key: str, index: str, dim: int, cloud: str, region: str):
    return PineconeStore(api_key, index, dim, cloud, region)


@st.cache_resource
def get_memory_store(embedder_key: str, dim: int):
    return InMemoryStore(dim)


def audit(entry: dict) -> None:
    """Append every Q&A to a JSONL file so answers stay traceable."""
    try:
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_DIR / "qa_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ----------------------------------------------------------------------------
# session state
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex[:8])
ss.setdefault("messages", [])          # [{"role", "content", "answer": Answer|None}]
ss.setdefault("ingest_log", [])

# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Connections")
    groq_key = st.text_input("Groq API key", value=secret("GROQ_API_KEY"), type="password", key="groq_key")
    store_kind = st.radio("Vector database", ["Pinecone", "In-memory (demo, no Pinecone)"], key="store_kind")
    pinecone_key, index_name, cloud, region = "", "", "aws", "us-east-1"
    if store_kind == "Pinecone":
        pinecone_key = st.text_input("Pinecone API key", value=secret("PINECONE_API_KEY"), type="password", key="pc_key")
        index_name = st.text_input("Index name", value=secret("PINECONE_INDEX", "enterprise-rag"), key="pc_index")
        c1, c2 = st.columns(2)
        cloud = c1.selectbox("Cloud", ["aws", "gcp", "azure"], index=0, key="pc_cloud")
        region = c2.text_input("Region", value=secret("PINECONE_REGION", "us-east-1"), key="pc_region")

    emb_key = st.selectbox("Embedding model", list(EMBEDDERS), format_func=lambda k: EMBEDDERS[k]["label"],
                           index=0, key="emb_key",
                           help="The index dimension is fixed by the model. Switching models needs a new index name.")
    if emb_key == "hash-demo":
        st.caption("⚠️ Demo embedder: matches shared words only. Use a real model for real work.")

    st.divider()
    st.header("🧠 Generation (Groq)")
    model = st.selectbox("Model", GROQ_MODELS, index=0, key="model")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, key="temp",
                            help="Low = factual and repeatable. Keep ≤ 0.2 for policy answers.")
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9, 0.05, key="top_p")
    top_k = st.slider("Top-k (chunks retrieved)", 1, 15, 5, key="top_k",
                      help="Groq's API has no sampling top-k, so top-k here is the number of chunks pulled from Pinecone.")
    max_tokens = st.slider("Max answer tokens", 100, 2000, 700, 50, key="max_tokens")

    st.header("🛡️ Hallucination controls")
    min_score = st.slider("Retrieval threshold (min similarity)", 0.0, 0.9,
                          float(EMBEDDERS[emb_key]["default_threshold"]), 0.01, key=f"thr_{emb_key}",
                          help="Chunks scoring below this are dropped. If nothing passes, the assistant says "
                               "'I don't know' without calling the LLM. Check the scores in 'Retrieval details'.")
    refuse_low = st.checkbox("Refuse when confidence is low", value=False, key="refuse_low")
    include_old = st.checkbox("Search superseded versions too (audit)", value=False, key="include_old")
    f_dept = st.multiselect("Limit to departments", DEPARTMENTS, key="f_dept")
    f_type = st.multiselect("Limit to document types", DOC_TYPES, key="f_type")

    settings = Settings(model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens, top_k=top_k,
                        min_score=min_score, include_superseded=include_old, departments=f_dept, doc_types=f_type,
                        refuse_low_confidence=refuse_low)

# ----------------------------------------------------------------------------
# connect
# ----------------------------------------------------------------------------
store = embedder = None
conn_error = ""
try:
    if store_kind == "Pinecone" and not pinecone_key:
        conn_error = "Enter your Pinecone API key in the sidebar (or choose the in-memory demo store)."
    elif emb_key == "pinecone-hosted" and store_kind != "Pinecone":
        conn_error = "Pinecone-hosted embeddings need a Pinecone API key. Pick another embedding model for the demo store."
    else:
        embedder = get_embedder(emb_key, pinecone_key)
        if store_kind == "Pinecone":
            store = get_pinecone_store(pinecone_key, index_name, embedder.dim, cloud, region)
        else:
            store = get_memory_store(emb_key, embedder.dim)
except Exception as e:  # surfaced to the user instead of a stack trace
    conn_error = f"{type(e).__name__}: {e}"

# ----------------------------------------------------------------------------
# sidebar: upload + documents (needs the store)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.header("📤 Documents")
    supersede = st.radio("When a document is re-uploaded", ["Deprecate old version", "Delete old version"],
                         key="supersede", help="Either way, answers always use the newest version.")
    files = st.file_uploader("Upload files", type=SUPPORTED_EXTENSIONS, accept_multiple_files=True, key="uploader")

    meta_df = None
    if files:
        rows = []
        for f in files:
            d, t = guess_metadata(f.name)
            rows.append({"file": f.name, "doc_id": slugify_doc_id(f.name), "department": d, "doc_type": t})
        meta_df = st.data_editor(
            pd.DataFrame(rows), hide_index=True, width="stretch", key="meta_editor",
            column_config={
                "file": st.column_config.TextColumn(disabled=True),
                "doc_id": st.column_config.TextColumn("doc_id", help="Same doc_id = new version of that document."),
                "department": st.column_config.SelectboxColumn(options=DEPARTMENTS, required=True),
                "doc_type": st.column_config.SelectboxColumn("doc_type", options=DOC_TYPES, required=True),
            })

    def run_ingest(items: list[tuple[str, bytes, str, str, str]]):
        mode = "delete" if supersede.startswith("Delete") else "deprecate"
        bar, out = st.progress(0.0), []
        for i, (name, data, doc_id, dept, dtype) in enumerate(items, 1):
            try:
                r = ingest_file(store, embedder, name, data, dept, dtype, doc_id=doc_id, supersede_mode=mode,
                                chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
            except Exception as e:
                r = None
                out.append({"file": name, "status": "error", "version": "", "chunks": "", "note": f"{type(e).__name__}: {e}"})
            if r is not None:
                note = r.message + (" " + " ".join(r.warnings) if r.warnings else "") + (" (OCR used)" if r.ocr_used else "")
                out.append({"file": name, "status": r.status, "version": r.version or "", "chunks": r.n_chunks or "", "note": note})
            bar.progress(i / len(items))
        ss["ingest_log"] = out
        bar.empty()

    disabled = store is None or embedder is None
    if st.button("Ingest uploaded files", type="primary", disabled=disabled or not files, key="btn_ingest"):
        with st.spinner("Parsing, OCR, embedding and indexing..."):
            payload = []
            for f in files:
                row = meta_df[meta_df["file"] == f.name].iloc[0]
                payload.append((f.name, f.getvalue(), row["doc_id"] or slugify_doc_id(f.name), row["department"], row["doc_type"]))
            run_ingest(payload)
    if SAMPLE_DIR.exists() and st.button("Load bundled sample documents", disabled=disabled, key="btn_samples"):
        with st.spinner("Indexing sample documents (includes OCR of 2 scans)..."):
            payload = []
            for p in sorted(SAMPLE_DIR.iterdir()):
                if p.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS and p.suffix.lower() != ".md":
                    d, t = guess_metadata(p.name)
                    payload.append((p.name, p.read_bytes(), slugify_doc_id(p.name), d, t))
            run_ingest(payload)

    if ss["ingest_log"]:
        st.dataframe(pd.DataFrame(ss["ingest_log"]), hide_index=True, width="stretch")

    with st.expander("📚 Indexed documents", expanded=False):
        if store is not None:
            try:
                docs = list_documents(store)
            except Exception as e:
                docs = []
                st.error(f"Could not list documents: {e}")
            if docs:
                st.dataframe(pd.DataFrame([{
                    "doc_id": d["doc_id"], "file": d["filename"], "version": int(d["latest_version"]),
                    "dept": d["department"], "type": d["doc_type"], "updated": str(d["updated_at"])[:16].replace("T", " "),
                    "chunks": int(d.get("n_chunks", 0))} for d in docs]), hide_index=True, width="stretch")
                victim = st.selectbox("Remove a document (all versions)", [""] + [d["doc_id"] for d in docs], key="victim")
                if victim and st.button(f"Delete '{victim}'", key="btn_delete"):
                    n = delete_document(store, victim)
                    st.success(f"Removed {n} vectors.")
                    st.rerun()
            else:
                st.caption("Nothing indexed yet.")

    st.divider()
    if st.button("🗑️ New chat", key="btn_clear"):
        ss["messages"] = []
        ss["session_id"] = uuid.uuid4().hex[:8]
        st.rerun()
    if ss["messages"]:
        transcript = [{"role": m["role"], "content": m["content"],
                       "sources": [{k: s[k] for k in ("filename", "version", "location", "score")} for s in m["answer"].sources]
                       if m.get("answer") else []} for m in ss["messages"]]
        st.download_button("⬇️ Download chat (JSON)", json.dumps(transcript, indent=2, ensure_ascii=False),
                           file_name=f"chat_{ss['session_id']}.json", mime="application/json", key="dl_chat")

# ----------------------------------------------------------------------------
# main panel
# ----------------------------------------------------------------------------
st.title("📚 Enterprise Knowledge Assistant")
st.caption("Answers come only from your uploaded documents, with sources. If the documents don't say, it says 'I don't know'.")

CONF_BADGE = {"high": "🟢 High confidence", "medium": "🟡 Medium confidence", "low": "🔴 Low confidence — verify in the source"}


def render_sources(a, key: str):
    if a.sources:
        with st.expander(f"📎 Sources ({len(a.sources)})", expanded=True):
            for s in a.sources:
                status = "" if s["is_latest"] else " · ⚠️ SUPERSEDED"
                ocr = " · OCR" if s["ocr"] else ""
                where = s["location"] + (f" › {s['section']}" if s["section"] else "")
                st.markdown(f"**[{s['n']}] {s['filename']}** · v{s['version']} · {s['department']} · {s['doc_type']} · "
                            f"updated {str(s['updated_at'])[:10]} · {where} · similarity {s['score']}{status}{ocr}")
                st.caption(s["text"][:500] + ("…" if len(s["text"]) > 500 else ""))
    if a.retrieved:
        with st.expander("🔍 Retrieval details (for tuning the threshold)", expanded=False):
            if a.standalone_query:
                st.write(f"**Search query:** {a.standalone_query}")
            if a.refusal_reason:
                st.write(f"**Why no answer:** {a.refusal_reason}")
            st.dataframe(pd.DataFrame([{"#": r["n"], "similarity": r["score"], "used": r["kept"], "file": r["filename"],
                                        "v": r["version"], "location": r["location"]} for r in a.retrieved]),
                         hide_index=True, width="stretch", key=f"ret_{key}")


for i, m in enumerate(ss["messages"]):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        a = m.get("answer")
        if a:
            if a.confidence in CONF_BADGE:
                st.caption(CONF_BADGE[a.confidence])
            for w in a.warnings:
                st.warning(w)
            render_sources(a, f"hist{i}")

if conn_error:
    st.info(f"⚙️ {conn_error}")
elif store is not None and not list_documents(store):
    st.info("👈 Upload documents (or load the bundled samples) to get started.")

prompt = st.chat_input("Ask about your documents…", disabled=bool(conn_error))
if prompt:
    ss["messages"].append({"role": "user", "content": prompt, "answer": None})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            pipeline = RAGPipeline(store, embedder, llm_mod.GroqLLM(groq_key))
            history = [{"role": m["role"], "content": m["content"]} for m in ss["messages"][:-1]]
            with st.spinner("Searching documents…"):
                answer = pipeline.ask(prompt, history, settings)
            st.markdown(answer.text)
            if answer.confidence in CONF_BADGE:
                st.caption(CONF_BADGE[answer.confidence])
            for w in answer.warnings:
                st.warning(w)
            render_sources(answer, f"new{len(ss['messages'])}")
            ss["messages"].append({"role": "assistant", "content": answer.text, "answer": answer})
            audit({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "session": ss["session_id"],
                   "question": prompt, "search_query": answer.standalone_query, "answer": answer.text,
                   "answerable": answer.answerable, "confidence": answer.confidence, "refusal": answer.refusal_reason,
                   "sources": [{"id": s["id"], "file": s["filename"], "version": s["version"], "score": s["score"]} for s in answer.sources],
                   "model": settings.model, "temperature": settings.temperature, "top_p": settings.top_p,
                   "top_k": settings.top_k, "min_score": settings.min_score, "latency_s": answer.latency_s})
        except llm_mod.LLMError as e:
            st.error(str(e))
            ss["messages"].pop()               # let the user retry the same question
        except Exception as e:
            st.error(f"Something went wrong: {type(e).__name__}: {e}")
            ss["messages"].pop()
