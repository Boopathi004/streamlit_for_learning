"""Retrieval: query -> embedding -> similarity search (Pinecone only) -> threshold filter."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import DOC_NAMESPACE, Settings
from .vectorstore import Match


@dataclass
class Retrieval:
    query: str
    filter: dict | None
    all: list[Match] = field(default_factory=list)     # everything Pinecone returned
    kept: list[Match] = field(default_factory=list)    # above the similarity threshold


def build_filter(settings: Settings) -> dict | None:
    clauses = []
    if not settings.include_superseded:
        clauses.append({"is_latest": {"$eq": True}})   # latest version only
    if settings.departments:
        clauses.append({"department": {"$in": list(settings.departments)}})
    if settings.doc_types:
        clauses.append({"doc_type": {"$in": list(settings.doc_types)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def retrieve(store, embedder, query: str, settings: Settings) -> Retrieval:
    flt = build_filter(settings)
    vec = embedder.embed_query(query)
    matches = store.query(vec, settings.top_k, flt, DOC_NAMESPACE)

    # Latest version wins: if several versions of a document come back, keep the newest only.
    if not settings.include_superseded:
        newest: dict[str, int] = {}
        for m in matches:
            d = m.metadata.get("doc_id", "")
            newest[d] = max(newest.get(d, 0), int(m.metadata.get("version_id", 0)))
        matches = [m for m in matches if int(m.metadata.get("version_id", 0)) == newest.get(m.metadata.get("doc_id", ""), 0)]

    kept = [m for m in matches if m.score >= settings.min_score]
    # Order by similarity, with current versions ahead of superseded ones.
    kept.sort(key=lambda m: (bool(m.metadata.get("is_latest", True)), m.score), reverse=True)
    return Retrieval(query=query, filter=flt, all=matches, kept=kept)
