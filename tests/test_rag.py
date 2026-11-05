"""Offline tests: no Pinecone, no Groq, no model downloads.

They run the real parsers/OCR/chunker/ingest/retrieval/pipeline code against the
in-memory vector store with the dependency-free hash embedder and a scripted LLM.
Run:  pytest -q
"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunking import chunk_blocks                                   # noqa: E402
from rag.config import DOC_NAMESPACE, IDK_MESSAGE, Settings             # noqa: E402
from rag.embeddings import HashEmbedder                                 # noqa: E402
from rag.ingest import (delete_document, guess_metadata, ingest_file,   # noqa: E402
                        list_documents, slugify_doc_id)
from rag.parsers import ParseError, parse_file                          # noqa: E402
from rag.pipeline import RAGPipeline, unverified_numbers                # noqa: E402
from rag.retriever import build_filter, retrieve                        # noqa: E402
from rag.vectorstore import InMemoryStore, match_filter                 # noqa: E402

SAMPLES = ROOT / "sample_docs"


# ---------------------------------------------------------------- helpers
class ScriptedLLM:
    """Stands in for Groq. `mode` decides how it misbehaves."""

    def __init__(self, mode="grounded"):
        self.mode, self.calls, self.last_messages = mode, 0, None

    def chat(self, messages, settings, json_mode=False, temperature=None, max_tokens=None):
        self.calls += 1
        self.last_messages = messages
        if messages[0]["content"].startswith("Rewrite the user's latest message"):
            latest = messages[-1]["content"].split("LATEST MESSAGE: ", 1)[1].split("\n\nSTANDALONE", 1)[0]
            # a real model resolves references like "and ..."; standalone questions pass through unchanged
            return "annual leave carry forward days" if latest.lower().startswith("and ") else latest
        user = messages[-1]["content"]
        first = user.split("\n", 3)[3].split("\n", 1)[1].split("\n")[0] if "[1]" in user else ""
        if self.mode == "grounded":
            return json.dumps({"answerable": True, "answer": f"{first} [1]", "citations": [1], "confidence": "high"})
        if self.mode == "no_cite":
            return json.dumps({"answerable": True, "answer": "Employees get plenty of leave.", "citations": [], "confidence": "high"})
        if self.mode == "hallucinate":
            return json.dumps({"answerable": True, "answer": "Employees get 99999 days of leave [1].", "citations": [1], "confidence": "high"})
        if self.mode == "idk":
            return json.dumps({"answerable": False, "answer": "I don't know based on the available documents.", "citations": [], "confidence": "low"})
        if self.mode == "garbage":
            return "not json at all"
        raise AssertionError(self.mode)


@pytest.fixture()
def system():
    store, emb = InMemoryStore(512), HashEmbedder()
    return store, emb


@pytest.fixture(scope="module")
def loaded():
    """Sample corpus ingested once per module (OCR is slow); tests must treat it as read-only."""
    store, emb = InMemoryStore(512), HashEmbedder()
    for f in sorted(SAMPLES.iterdir()):
        if f.suffix.lower() in (".md",):
            continue
        dept, dtype = guess_metadata(f.name)
        r = ingest_file(store, emb, f.name, f.read_bytes(), dept, dtype)
        assert r.status == "new", (f.name, r.message)
    return store, emb


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("name,needles", [
    ("HR-POL-001_Leave_and_Attendance_Policy.pdf", ["24 days", "Up to 10 days", "60 days"]),
    ("FIN-POL-007_Travel_and_Expense_Policy.docx", ["INR 6,000", "30 days", "INR 25,000"]),
    ("MSA-2026-014_Techspan_Master_Services_Agreement.rtf", ["24 months", "INR 50,00,000", "Chennai"]),
    ("NDA-2026-031_Orbit_Mutual_NDA.txt", ["5 years", "Coimbatore"]),
    ("Q3_FY2026_Budget_vs_Actual.xlsx", ["Operations", "3720000", "Over budget"]),
    ("Vendor_Payments_Q3_FY2026.csv", ["INV-2026-0466", "Disputed", "Vista Print and Media"]),
])
def test_text_formats_parse(name, needles):
    doc = parse_file(name, (SAMPLES / name).read_bytes())
    for n in needles:
        assert n in doc.full_text, f"{n!r} missing from {name}"
    assert not doc.ocr_used


def test_pdf_table_rows_stay_together():
    doc = parse_file("p.pdf", (SAMPLES / "HR-POL-001_Leave_and_Attendance_Policy.pdf").read_bytes())
    assert any("Annual (earned) leave" in l and "24 days" in l and "Up to 10 days" in l
               for l in doc.full_text.splitlines())


@pytest.mark.parametrize("name,needles", [
    ("Scanned_Invoice_INV-2026-0417.png", ["INV-2026-0417", "57,017.60", "PO-OPS-2026-0188"]),
    ("Scanned_Travel_Expense_Claim.jpg", ["6,400", "Priya", "None attached"]),
])
def test_ocr_images(name, needles):
    doc = parse_file(name, (SAMPLES / name).read_bytes())
    assert doc.ocr_used
    for n in needles:
        assert n in doc.full_text, f"OCR missed {n!r}"


def test_bad_inputs():
    with pytest.raises(ParseError):
        parse_file("x.exe", b"abc")
    with pytest.raises(ParseError):
        parse_file("x.pdf", b"")
    with pytest.raises(ParseError):
        parse_file("x.pdf", b"not a pdf")


# ---------------------------------------------------------------- chunking
def test_chunks_carry_sections_and_respect_size():
    doc = parse_file("n.txt", (SAMPLES / "NDA-2026-031_Orbit_Mutual_NDA.txt").read_bytes())
    chunks = chunk_blocks(doc.blocks, 600, 100)
    assert all(len(c.text) <= 700 for c in chunks)
    assert any("TERM AND SURVIVAL" in c.section for c in chunks)
    term = next(c for c in chunks if "TERM AND SURVIVAL" in c.section)
    assert "5 years" in term.text


def test_table_chunks_repeat_prefix():
    doc = parse_file("v.csv", (SAMPLES / "Vendor_Payments_Q3_FY2026.csv").read_bytes())
    chunks = chunk_blocks(doc.blocks, 900, 150)
    assert len(chunks) >= 2 and all(c.text.startswith("Table: v.csv") for c in chunks)


# ---------------------------------------------------------------- retrieval over the sample corpus
@pytest.mark.parametrize("question,expected_file", [
    ("How many annual leave days can be carried forward?", "HR-POL-001"),
    ("What is the hotel cap per night in metro cities?", "FIN-POL-007"),
    ("Which vendor invoices are disputed?", "Vendor_Payments"),
    ("What is the liability cap in the Techspan agreement?", "MSA-2026-014"),
    ("How long do the Orbit NDA confidentiality obligations last?", "NDA-2026-031"),
    ("What is the total payable on invoice INV-2026-0417?", "Scanned_Invoice"),
    ("What hotel rate per night did Priya Venkatesh claim?", "Scanned_Travel"),
    ("Which department overspent its Q3 budget the most?", "Q3_FY2026"),
])
def test_retrieval_finds_right_document(loaded, question, expected_file):
    store, emb = loaded
    r = retrieve(store, emb, question, Settings(top_k=4, min_score=0.12))
    assert any(expected_file in m.metadata["filename"] for m in r.kept[:3]), \
        [(m.metadata["filename"], round(m.score, 2)) for m in r.all]


def test_metadata_attached(loaded):
    store, _ = loaded
    rec = next(iter(store._data[DOC_NAMESPACE].values()))[1]
    for k in ["doc_id", "filename", "doc_type", "department", "version_id", "updated_at", "upload_date", "is_latest", "text"]:
        assert k in rec
    assert rec["version_id"] == 1 and rec["is_latest"] is True


def test_department_filter(loaded):
    store, emb = loaded
    r = retrieve(store, emb, "approval required", Settings(top_k=8, min_score=0.0, departments=["Legal"]))
    assert r.all and all(m.metadata["department"] == "Legal" for m in r.all)


def test_filter_builder_and_matcher():
    f = build_filter(Settings(departments=["HR"], doc_types=["Policy"]))
    assert f == {"$and": [{"is_latest": {"$eq": True}}, {"department": {"$in": ["HR"]}}, {"doc_type": {"$in": ["Policy"]}}]}
    assert match_filter({"is_latest": True, "department": "HR", "doc_type": "Policy"}, f)
    assert not match_filter({"is_latest": False, "department": "HR", "doc_type": "Policy"}, f)


# ---------------------------------------------------------------- versioning
V1 = b"LEAVE POLICY\n\n1. ANNUAL LEAVE\nEmployees receive 24 days of annual leave per year.\n"
V2 = b"LEAVE POLICY\n\n1. ANNUAL LEAVE\nEmployees receive 30 days of annual leave per year.\n"


def test_reupload_creates_new_version_and_replaces_old(system):
    store, emb = system
    r1 = ingest_file(store, emb, "Leave_Policy.txt", V1, "HR", "Policy")
    assert (r1.status, r1.version, r1.doc_id) == ("new", 1, "leave-policy")
    same = ingest_file(store, emb, "Leave_Policy.txt", V1, "HR", "Policy")
    assert same.status == "unchanged" and store.count() == r1.n_chunks

    r2 = ingest_file(store, emb, "Leave_Policy_v2.txt", V2, "HR", "Policy")
    assert (r2.status, r2.version, r2.doc_id) == ("updated", 2, "leave-policy")

    s = Settings(top_k=5, min_score=0.0)
    kept = retrieve(store, emb, "how many days of annual leave", s).kept
    assert kept and all(m.metadata["version_id"] == 2 and m.metadata["is_latest"] for m in kept)
    assert any("30 days" in m.metadata["text"] for m in kept)
    assert not any("24 days" in m.metadata["text"] for m in kept)

    audit = retrieve(store, emb, "how many days of annual leave", Settings(top_k=5, min_score=0.0, include_superseded=True)).kept
    old = [m for m in audit if m.metadata["version_id"] == 1]
    assert old and old[0].metadata["is_latest"] is False and old[0].metadata["superseded_by"] == 2

    info = list_documents(store)[0]
    hist = json.loads(info["history"])
    assert info["latest_version"] == 2 and [h["status"] for h in hist] == ["deprecated", "current"]


def test_delete_mode_removes_old_vectors(system):
    store, emb = system
    ingest_file(store, emb, "Leave_Policy.txt", V1, "HR", "Policy")
    ingest_file(store, emb, "Leave_Policy.txt", V2, "HR", "Policy", supersede_mode="delete")
    assert store.list_ids("leave-policy::v1::") == []
    assert store.list_ids("leave-policy::v2::")


def test_metadata_only_change_updates_without_new_version(system):
    store, emb = system
    ingest_file(store, emb, "Leave_Policy.txt", V1, "General", "Other")
    r = ingest_file(store, emb, "Leave_Policy.txt", V1, "HR", "Policy")
    assert r.status == "metadata-updated" and r.version == 1
    assert all(m.metadata["department"] == "HR" for m in retrieve(store, emb, "annual leave", Settings(min_score=0)).all)


def test_delete_document(loaded):
    import copy
    store = copy.deepcopy(loaded[0])
    n_docs = len(list_documents(store))
    assert delete_document(store, "nda-2026-031-orbit-mutual-nda") > 0
    assert len(list_documents(store)) == n_docs - 1


def test_failed_upsert_leaves_no_partial_version(system):
    store, emb = system
    ingest_file(store, emb, "Leave_Policy.txt", V1, "HR", "Policy")
    orig = store.upsert
    store.upsert = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        ingest_file(store, emb, "Leave_Policy.txt", V2, "HR", "Policy")
    store.upsert = orig
    assert store.list_ids("leave-policy::v2::") == []
    assert all(m.metadata["is_latest"] for m in retrieve(store, emb, "annual leave", Settings(min_score=0)).all)


def test_slug_and_guess():
    assert slugify_doc_id("Leave Policy (1).pdf") == "leave-policy"
    assert slugify_doc_id("leave-policy-final.pdf") == "leave-policy"
    assert slugify_doc_id("Leave_Policy_v2.1.docx") == "leave-policy"
    assert slugify_doc_id("MSA-2026-014_Techspan.rtf") == "msa-2026-014-techspan"
    assert guess_metadata("HR-POL-001_Leave_and_Attendance_Policy.pdf") == ("HR", "Policy")
    assert guess_metadata("NDA-2026-031_Orbit_Mutual_NDA.txt") == ("Legal", "Contract")


# ---------------------------------------------------------------- pipeline guards
def make(loaded, mode="grounded"):
    store, emb = loaded
    llm = ScriptedLLM(mode)
    return RAGPipeline(store, emb, llm), llm


S = Settings(top_k=4, min_score=0.12)


def test_grounded_answer_has_sources(loaded):
    p, llm = make(loaded)
    a = p.ask("How many annual leave days can be carried forward?", [], S)
    assert a.answerable and a.citations == [1] and a.sources[0]["filename"]
    assert "[1]" in a.text and a.sources[0]["version"] == 1


def test_out_of_scope_question_never_reaches_llm(loaded):
    p, llm = make(loaded)
    a = p.ask("What is the stock options vesting schedule?", [], S)
    assert not a.answerable and a.text == IDK_MESSAGE and llm.calls == 0
    assert "threshold" in a.refusal_reason


def test_uncited_answer_is_rejected(loaded):
    p, _ = make(loaded, "no_cite")
    a = p.ask("How many annual leave days can be carried forward?", [], S)
    assert not a.answerable and "citation" in a.refusal_reason


def test_invented_number_is_flagged(loaded):
    p, _ = make(loaded, "hallucinate")
    a = p.ask("How many annual leave days can be carried forward?", [], S)
    assert a.answerable and a.confidence in ("low", "medium") and "99999" in " ".join(a.warnings)


def test_low_confidence_refusal_option(loaded):
    p, _ = make(loaded, "hallucinate")
    s = Settings(top_k=4, min_score=0.12, refuse_low_confidence=True)
    a = p.ask("How many annual leave days can be carried forward?", [], s)
    assert a.answerable in (True, False)
    if a.confidence == "none":
        assert a.text.startswith(IDK_MESSAGE)


def test_model_says_unknown(loaded):
    p, _ = make(loaded, "idk")
    a = p.ask("How many annual leave days can be carried forward?", [], S)
    assert not a.answerable and a.text.startswith("I don't know")


def test_garbage_output_is_safe(loaded):
    p, _ = make(loaded, "garbage")
    a = p.ask("How many annual leave days can be carried forward?", [], S)
    assert not a.answerable and a.text == IDK_MESSAGE


def test_followup_is_condensed_with_history(loaded):
    p, llm = make(loaded)
    hist = [{"role": "user", "content": "Tell me about annual leave."},
            {"role": "assistant", "content": "Employees get 24 days [1]."}]
    a = p.ask("and how many can be carried forward?", hist, S)
    assert a.standalone_query == "annual leave carry forward days"
    assert a.answerable
    assert any(m["role"] == "assistant" for m in llm.last_messages)     # history passed to the answer prompt


def test_small_talk_skips_retrieval(loaded):
    p, llm = make(loaded)
    a = p.ask("thanks!", [], S)
    assert a.answerable and llm.calls == 0


def test_prompt_contains_grounding_rules_and_metadata(loaded):
    p, llm = make(loaded)
    p.ask("How many annual leave days can be carried forward?", [], S)
    system = llm.last_messages[0]["content"]
    assert "Answer only from the provided context" in system and "I don't know" in system
    user = llm.last_messages[-1]["content"]
    assert "Version: 1" in user and "Status: CURRENT" in user and "Updated:" in user


def test_number_verifier():
    ctx = "Hotel cap INR 6,000 per night. Claims within 30 days."
    assert unverified_numbers("The cap is INR 6,000 [1] within 30 days.", ctx) == []
    assert unverified_numbers("The cap is INR 7,500 [1].", ctx) == ["7500"]
    assert unverified_numbers("1. First point [2] 30 days", ctx) == []
