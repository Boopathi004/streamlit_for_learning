"""Ingestion with document versioning.

Re-uploading a document never leaves stale answers behind:

1. The extracted text is hashed. Identical content -> nothing to do ("unchanged").
2. Changed content -> version_id + 1, new chunks are written with is_latest=True.
3. Only after the new chunks are safely stored, the previous version is deprecated
   (is_latest=False, superseded_by=...) or deleted. Retrieval always filters on
   is_latest=True, so users see the newest version only.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .chunking import chunk_blocks
from .config import DOC_NAMESPACE
from .parsers import ParseError, parse_file


@dataclass
class IngestResult:
    filename: str
    doc_id: str = ""
    status: str = "error"          # new | updated | unchanged | metadata-updated | error
    version: int = 0
    n_chunks: int = 0
    ocr_used: bool = False
    warnings: list[str] = field(default_factory=list)
    message: str = ""


# ----------------------------------------------------------------------------
_VERSION_SUFFIX = re.compile(
    r"(?:[-_ .]+(?:v\d+(?:\.\d+)*|final|updated|revised|new|latest|copy|draft|\d{4}[-_]?\d{2}[-_]?\d{2}))+$", re.I)
_COPY_SUFFIX = re.compile(r"\s*\(\d+\)$")


def slugify_doc_id(filename: str) -> str:
    """'Leave Policy (1).pdf' / 'Leave_Policy_v2.pdf' / 'leave-policy-final.pdf' -> 'leave-policy'."""
    stem = filename.rsplit(".", 1)[0]
    stem = _COPY_SUFFIX.sub("", stem)
    stem = _VERSION_SUFFIX.sub("", stem)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
    return slug or "document"


def guess_metadata(filename: str) -> tuple[str, str]:
    """Cheap guess of (department, doc_type) from the file name; the user can edit it."""
    n = filename.lower()
    dept = "General"
    for words, d in [
        (("hr", "leave", "attendance", "recruit", "payroll", "employee"), "HR"),
        (("fin", "budget", "invoice", "expense", "vendor", "payment", "tax", "travel", "receipt"), "Finance"),
        (("legal", "nda", "msa", "agreement", "contract", "terms"), "Legal"),
        (("sop", "ops", "operation", "maintenance", "production"), "Operations"),
        (("it-", "security", "network", "software"), "IT"),
        (("sales", "crm", "quote"), "Sales"),
        (("marketing", "brand", "campaign"), "Marketing"),
    ]:
        if any(re.search(rf"(?<![a-z]){re.escape(w)}", n) for w in words):
            dept = d
            break
    dtype = "Other"
    for words, t in [
        (("policy", "pol-", "handbook"), "Policy"), (("sop", "procedure"), "SOP"),
        (("nda", "msa", "agreement", "contract"), "Contract"),
        (("invoice", "receipt", "claim"), "Invoice/Receipt"),
        (("budget", "payments", "ledger", "vendor"), "Financial data"), (("report", "summary"), "Report"),
    ]:
        if any(w in n for w in words):
            dtype = t
            break
    return dept, dtype


def _hash_text(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------------------
def ingest_file(store, embedder, filename: str, data: bytes, department: str, doc_type: str,
                doc_id: str | None = None, supersede_mode: str = "deprecate",
                chunk_size: int = 900, chunk_overlap: int = 150, now: datetime | None = None) -> IngestResult:
    """Parse -> chunk -> embed -> upsert (with metadata) -> retire the old version."""
    now = now or _now()
    doc_id = doc_id or slugify_doc_id(filename)
    res = IngestResult(filename=filename, doc_id=doc_id)
    try:
        parsed = parse_file(filename, data)
    except ParseError as e:
        res.message = str(e)
        return res
    res.warnings, res.ocr_used = list(parsed.warnings), parsed.ocr_used

    chunks = chunk_blocks(parsed.blocks, chunk_size, chunk_overlap)
    if not chunks:
        res.message = "No text could be extracted, nothing to index."
        return res

    content_hash = _hash_text(parsed.full_text)
    prev = store.get_registry(doc_id)

    # ---- unchanged content: skip (but honour edited department / type)
    if prev and prev.get("content_hash") == content_hash:
        res.version, res.n_chunks = int(prev["latest_version"]), int(prev.get("n_chunks", 0))
        if prev.get("department") != department or prev.get("doc_type") != doc_type:
            ids = store.list_ids(f"{doc_id}::v{res.version}::", DOC_NAMESPACE)
            store.set_metadata(ids, {"department": department, "doc_type": doc_type}, DOC_NAMESPACE)
            store.put_registry(doc_id, {**prev, "department": department, "doc_type": doc_type})
            res.status, res.message = "metadata-updated", "Content identical; department/type updated."
        else:
            res.status, res.message = "unchanged", "Identical to the current version; nothing re-indexed."
        return res

    version = int(prev["latest_version"]) + 1 if prev else 1
    texts = [f"{filename} | {doc_type} | {department} | {c.location}\n{c.text}" for c in chunks]
    vectors = embedder.embed_documents(texts)

    base = {
        "doc_id": doc_id, "filename": filename, "file_type": parsed.file_type, "doc_type": doc_type,
        "department": department, "version_id": version, "updated_at": now.isoformat(timespec="seconds"),
        "upload_date": now.date().isoformat(), "is_latest": True, "content_hash": content_hash,
        "ocr": bool(parsed.ocr_used),
    }
    records = []
    new_ids = []
    for c, v in zip(chunks, vectors):
        rid = f"{doc_id}::v{version}::{c.index:04d}"
        new_ids.append(rid)
        records.append((rid, v, {**base, "chunk_index": c.index, "location": c.location,
                                 "section": c.section, "text": c.text}))
    try:
        store.upsert(records, DOC_NAMESPACE)
    except Exception:
        try:
            store.delete(new_ids, DOC_NAMESPACE)      # no half-written versions
        finally:
            raise

    # ---- retire older versions only now that the new one is safely stored
    history = json.loads(prev["history"]) if prev and prev.get("history") else []
    for h in history:
        if h.get("status") != "current":
            continue
        old_ids = store.list_ids(f"{doc_id}::v{h['version']}::", DOC_NAMESPACE)
        if supersede_mode == "delete":
            store.delete(old_ids, DOC_NAMESPACE)
            h["status"] = "deleted"
        else:
            store.set_metadata(old_ids, {"is_latest": False, "superseded_by": version,
                                         "deprecated_at": base["updated_at"]}, DOC_NAMESPACE)
            h["status"] = "deprecated"
    history.append({"version": version, "updated_at": base["updated_at"], "content_hash": content_hash[:12],
                    "n_chunks": len(chunks), "filename": filename, "status": "current"})

    store.put_registry(doc_id, {
        "doc_id": doc_id, "filename": filename, "file_type": parsed.file_type, "department": department,
        "doc_type": doc_type, "latest_version": version, "updated_at": base["updated_at"],
        "first_uploaded_at": prev.get("first_uploaded_at", base["updated_at"]) if prev else base["updated_at"],
        "content_hash": content_hash, "n_chunks": len(chunks), "history": json.dumps(history),
    })
    res.status = "updated" if prev else "new"
    res.version, res.n_chunks = version, len(chunks)
    res.message = (f"Version {version} indexed ({len(chunks)} chunks)."
                   + (f" Version {version - 1} was {'deleted' if supersede_mode == 'delete' else 'deprecated'}." if prev else ""))
    return res


# ----------------------------------------------------------------------------
def list_documents(store) -> list[dict]:
    docs = store.list_registry()
    return sorted(docs, key=lambda d: d.get("doc_id", ""))


def delete_document(store, doc_id: str) -> int:
    """Remove every version of a document (vectors + registry entry)."""
    ids = store.list_ids(f"{doc_id}::", DOC_NAMESPACE)
    store.delete(ids, DOC_NAMESPACE)
    store.delete_registry(doc_id)
    return len(ids)
