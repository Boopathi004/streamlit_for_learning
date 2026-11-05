"""Vector store layer: Pinecone (production) and an in-memory twin (tests/demo).

Both expose the same small interface:

    upsert(records, namespace)            records = [(id, vector, metadata)]
    query(vector, top_k, flt, namespace)  -> [Match]
    set_metadata(ids, meta, namespace)    deprecate old versions without deleting
    delete(ids, namespace)
    list_ids(prefix, namespace)
    put_registry / get_registry / list_registry / delete_registry
        a per-document manifest (latest version, history, hash) kept in its own
        namespace, so the system needs no local database and survives restarts.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .config import DOC_NAMESPACE, REGISTRY_NAMESPACE


@dataclass
class Match:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _g(obj: Any, key: str, default=None):
    """Read a field from an SDK object or a dict (SDK versions differ)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _batches(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ----------------------------------------------------------------------------
# Metadata filter evaluation for the in-memory store (subset of Pinecone syntax)
# ----------------------------------------------------------------------------
def match_filter(meta: dict, flt: dict | None) -> bool:
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$and":
            if not all(match_filter(meta, c) for c in cond):
                return False
            continue
        if key == "$or":
            if not any(match_filter(meta, c) for c in cond):
                return False
            continue
        val = meta.get(key)
        if not isinstance(cond, dict):
            cond = {"$eq": cond}
        for op, arg in cond.items():
            ok = {
                "$eq": lambda: val == arg, "$ne": lambda: val != arg,
                "$in": lambda: val in arg, "$nin": lambda: val not in arg,
                "$gt": lambda: val is not None and val > arg, "$gte": lambda: val is not None and val >= arg,
                "$lt": lambda: val is not None and val < arg, "$lte": lambda: val is not None and val <= arg,
            }[op]()
            if not ok:
                return False
    return True


# ----------------------------------------------------------------------------
class InMemoryStore:
    """Process-local stand-in for Pinecone. Same behaviour, no network."""

    def __init__(self, dim: int):
        self.dim = dim
        self._data: dict[str, dict[str, tuple[np.ndarray, dict]]] = {}

    def _ns(self, ns): return self._data.setdefault(ns, {})

    def upsert(self, records, namespace=DOC_NAMESPACE):
        for rid, vec, meta in records:
            self._ns(namespace)[rid] = (np.asarray(vec, dtype=np.float32), dict(meta))

    def query(self, vector, top_k, flt=None, namespace=DOC_NAMESPACE) -> list[Match]:
        q = np.asarray(vector, dtype=np.float32)
        scored = []
        for rid, (vec, meta) in self._ns(namespace).items():
            if match_filter(meta, flt):
                scored.append(Match(rid, float(np.dot(q, vec)), dict(meta)))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def set_metadata(self, ids, meta, namespace=DOC_NAMESPACE):
        for rid in ids:
            if rid in self._ns(namespace):
                self._ns(namespace)[rid][1].update(meta)

    def delete(self, ids, namespace=DOC_NAMESPACE):
        for rid in ids:
            self._ns(namespace).pop(rid, None)

    def list_ids(self, prefix, namespace=DOC_NAMESPACE) -> list[str]:
        return sorted(i for i in self._ns(namespace) if i.startswith(prefix))

    def count(self, namespace=DOC_NAMESPACE) -> int:
        return len(self._ns(namespace))

    # ---- registry
    def put_registry(self, doc_id, info):
        self.upsert([(f"reg::{doc_id}", [1.0] + [0.0] * (self.dim - 1), info)], REGISTRY_NAMESPACE)

    def get_registry(self, doc_id):
        rec = self._ns(REGISTRY_NAMESPACE).get(f"reg::{doc_id}")
        return dict(rec[1]) if rec else None

    def list_registry(self):
        return [dict(m) for _, m in self._ns(REGISTRY_NAMESPACE).values()]

    def delete_registry(self, doc_id):
        self.delete([f"reg::{doc_id}"], REGISTRY_NAMESPACE)


# ----------------------------------------------------------------------------
class PineconeStore:
    """Pinecone serverless index with metadata, filters and versioned records."""

    def __init__(self, api_key: str, index_name: str, dim: int, cloud: str = "aws",
                 region: str = "us-east-1", metric: str = "cosine"):
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as e:
            raise RuntimeError("Install the Pinecone SDK: pip install pinecone") from e
        self.pc = Pinecone(api_key=api_key)
        self.dim, self.index_name = dim, index_name
        if not self.pc.has_index(index_name):
            self.pc.create_index(name=index_name, dimension=dim, metric=metric,
                                 spec=ServerlessSpec(cloud=cloud, region=region))
            for _ in range(60):                       # wait until the index is ready
                if _g(_g(self.pc.describe_index(index_name), "status"), "ready"):
                    break
                time.sleep(2)
        existing = _g(self.pc.describe_index(index_name), "dimension")
        if existing and int(existing) != dim:
            raise RuntimeError(
                f"Index '{index_name}' has dimension {existing}, but the selected embedding model "
                f"produces {dim}. Use another index name or switch the embedding model."
            )
        self.index = self.pc.Index(index_name)

    # ---- vectors
    def upsert(self, records, namespace=DOC_NAMESPACE):
        recs = [{"id": i, "values": list(map(float, v)), "metadata": m} for i, v, m in records]
        for b in _batches(recs, 100):
            self.index.upsert(vectors=b, namespace=namespace)

    def query(self, vector, top_k, flt=None, namespace=DOC_NAMESPACE) -> list[Match]:
        res = self.index.query(vector=list(map(float, vector)), top_k=top_k, filter=flt or None,
                               include_metadata=True, namespace=namespace)
        return [Match(_g(m, "id"), float(_g(m, "score", 0.0)), dict(_g(m, "metadata", {}) or {}))
                for m in _g(res, "matches", [])]

    def set_metadata(self, ids, meta, namespace=DOC_NAMESPACE):
        def one(rid):
            self.index.update(id=rid, set_metadata=meta, namespace=namespace)
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(one, ids))

    def delete(self, ids, namespace=DOC_NAMESPACE):
        for b in _batches(list(ids), 500):
            self.index.delete(ids=b, namespace=namespace)

    def list_ids(self, prefix, namespace=DOC_NAMESPACE) -> list[str]:
        ids: list[str] = []
        for page in self.index.list(prefix=prefix, namespace=namespace):
            items = _g(page, "vectors", page)          # new SDK: page.vectors[*].id ; old SDK: list[str]
            for it in items:
                ids.append(it if isinstance(it, str) else _g(it, "id"))
        return ids

    def count(self, namespace=DOC_NAMESPACE) -> int:
        stats = self.index.describe_index_stats()
        ns = (_g(stats, "namespaces", {}) or {}).get(namespace)
        return int(_g(ns, "vector_count", 0) or 0) if ns is not None else 0

    # ---- registry
    def put_registry(self, doc_id, info):
        self.upsert([(f"reg::{doc_id}", [1.0] + [0.0] * (self.dim - 1), info)], REGISTRY_NAMESPACE)

    def get_registry(self, doc_id):
        res = self.index.fetch(ids=[f"reg::{doc_id}"], namespace=REGISTRY_NAMESPACE)
        vecs = _g(res, "vectors", {}) or {}
        v = vecs.get(f"reg::{doc_id}")
        return dict(_g(v, "metadata", {}) or {}) if v is not None else None

    def list_registry(self):
        out = []
        ids = self.list_ids("reg::", REGISTRY_NAMESPACE)
        for b in _batches(ids, 100):
            res = self.index.fetch(ids=b, namespace=REGISTRY_NAMESPACE)
            for v in (_g(res, "vectors", {}) or {}).values():
                out.append(dict(_g(v, "metadata", {}) or {}))
        return out

    def delete_registry(self, doc_id):
        self.delete([f"reg::{doc_id}"], REGISTRY_NAMESPACE)
