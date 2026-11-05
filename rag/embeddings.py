"""Embedding back-ends. All return L2-normalised vectors so cosine == dot product."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .config import EMBEDDERS


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


# ----------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    """Local model: free, no API key. Downloads the model on first use (~90 MB)."""

    def __init__(self, key: str = "local-minilm"):
        cfg = EMBEDDERS[key]
        self.name, self.dim = cfg["model"], cfg["dim"]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("Install `sentence-transformers` (pip install sentence-transformers).") from e
        self._model = SentenceTransformer(self.name)

    def embed_documents(self, texts):
        return self._model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()


# ----------------------------------------------------------------------------
class PineconeEmbedder:
    """Embeddings hosted by Pinecone (same API key as the vector database)."""

    def __init__(self, pc, key: str = "pinecone-hosted"):
        cfg = EMBEDDERS[key]
        self._pc, self.name, self.dim = pc, cfg["model"], cfg["dim"]

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            res = self._pc.inference.embed(
                model=self.name,
                inputs=texts[i:i + 64],
                parameters={"input_type": input_type, "truncate": "END"},
            )
            for item in res:
                vals = item["values"] if isinstance(item, dict) else item.values
                out.append(_normalise(list(vals)))
        return out

    def embed_documents(self, texts):
        return self._embed(texts, "passage")

    def embed_query(self, text):
        return self._embed([text], "query")[0]


# ----------------------------------------------------------------------------
_STOP = set("""a an the and or but if then of to in on at by for with from as is are was were be been being it its this that
these those i you he she we they them his her our your their my me do does did done can could should would will shall may might
must not no yes what which who whom whose when where why how there here about into over under per than also any all each
some such so very just only more most other""".split())


def _normalise(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashEmbedder:
    """Dependency-free demo embedder: hashed bag of words (+ stemmed tokens).

    It only matches shared words, it does not understand meaning. It exists so
    the whole pipeline can be run and tested offline; use a real model in practice.
    """

    def __init__(self, key: str = "hash-demo"):
        cfg = EMBEDDERS[key]
        self.name, self.dim = cfg["model"], cfg["dim"]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        toks = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]
        out = []
        for t in toks:
            out.append(t)
            if len(t) > 4 and t.endswith("s"):
                out.append(t[:-1])
            elif len(t) > 5 and t.endswith("ing"):
                out.append(t[:-3])
        return out

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for t in self._tokens(text):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        v = [1 + math.log(x) if x > 0 else 0.0 for x in v]
        return _normalise(v)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def build_embedder(key: str, pc=None) -> Embedder:
    if key == "local-minilm":
        return SentenceTransformerEmbedder(key)
    if key == "pinecone-hosted":
        if pc is None:
            raise RuntimeError("Pinecone-hosted embeddings need a Pinecone API key.")
        return PineconeEmbedder(pc, key)
    if key == "hash-demo":
        return HashEmbedder(key)
    raise ValueError(f"Unknown embedder: {key}")
