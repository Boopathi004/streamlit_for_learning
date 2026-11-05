"""Tests the thin Pinecone and Groq wrappers against stand-ins that mimic the SDK call shapes
(both the new object-style responses and the older dict/list-style ones)."""
import pathlib
import sys
import types
from types import SimpleNamespace as NS

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.config import Settings                                  # noqa: E402
from rag.embeddings import HashEmbedder                          # noqa: E402
from rag.ingest import ingest_file, list_documents               # noqa: E402
from rag.llm import GroqLLM, LLMError, parse_llm_json            # noqa: E402
from rag.retriever import retrieve                               # noqa: E402
from rag.vectorstore import InMemoryStore, PineconeStore         # noqa: E402


def make_fake_pinecone(style: str, dim: int, exists: bool = False):
    """style='new' -> attribute objects (SDK v6+), 'old' -> dicts / plain id lists."""
    backing = InMemoryStore(dim)
    calls = {"create_index": 0, "update": 0}

    class FakeIndex:
        def upsert(self, *, vectors, namespace=""):
            backing.upsert([(v["id"], v["values"], v["metadata"]) for v in vectors], namespace)

        def query(self, *, vector, top_k, filter=None, include_metadata=False, namespace=""):
            ms = backing.query(vector, top_k, filter, namespace)
            if style == "new":
                return NS(matches=[NS(id=m.id, score=m.score, metadata=m.metadata) for m in ms])
            return {"matches": [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in ms]}

        def update(self, *, id, set_metadata, namespace=""):
            calls["update"] += 1
            backing.set_metadata([id], set_metadata, namespace)

        def delete(self, *, ids, namespace=""):
            backing.delete(ids, namespace)

        def list(self, *, prefix, namespace=""):
            ids = backing.list_ids(prefix, namespace)
            for i in range(0, len(ids), 3):                       # several pages
                page = ids[i:i + 3]
                yield NS(vectors=[NS(id=x) for x in page]) if style == "new" else page

        def fetch(self, *, ids, namespace=""):
            vecs = {}
            for i in ids:
                rec = backing._ns(namespace).get(i)
                if rec:
                    vecs[i] = NS(metadata=rec[1]) if style == "new" else {"metadata": rec[1]}
            return NS(vectors=vecs) if style == "new" else {"vectors": vecs}

        def describe_index_stats(self):
            ns = {k: (NS(vector_count=len(v)) if style == "new" else {"vector_count": len(v)})
                  for k, v in backing._data.items()}
            return NS(namespaces=ns)

    class FakePinecone:
        def __init__(self, api_key):
            self.api_key, self.created = api_key, exists

        def has_index(self, name):
            return self.created

        def create_index(self, **kw):
            calls["create_index"] += 1
            self.created = True
            assert kw["dimension"] == dim and kw["metric"] == "cosine" and kw["spec"].cloud == "aws"

        def describe_index(self, name):
            st = NS(ready=True) if style == "new" else {"ready": True}
            return NS(status=st, dimension=dim)

        def Index(self, name):
            return FakeIndex()

    mod = types.ModuleType("pinecone")
    mod.Pinecone = FakePinecone
    mod.ServerlessSpec = lambda cloud, region: NS(cloud=cloud, region=region)
    return mod, backing, calls


@pytest.mark.parametrize("style", ["new", "old"])
def test_pinecone_store_full_flow(monkeypatch, style):
    mod, backing, calls = make_fake_pinecone(style, 512)
    monkeypatch.setitem(sys.modules, "pinecone", mod)
    store, emb = PineconeStore("key", "idx", 512), HashEmbedder()
    assert calls["create_index"] == 1

    v1 = b"POLICY\n\n1. LEAVE\nEmployees receive 24 days of annual leave.\n"
    v2 = b"POLICY\n\n1. LEAVE\nEmployees receive 30 days of annual leave.\n"
    assert ingest_file(store, emb, "policy.txt", v1, "HR", "Policy").status == "new"
    assert ingest_file(store, emb, "policy.txt", v1, "HR", "Policy").status == "unchanged"
    r = ingest_file(store, emb, "policy_v2.txt", v2, "HR", "Policy")
    assert (r.status, r.version) == ("updated", 2)
    assert calls["update"] >= 1                                    # old chunks deprecated via update()

    kept = retrieve(store, emb, "annual leave days", Settings(top_k=5, min_score=0.0)).kept
    assert kept and all(m.metadata["version_id"] == 2 for m in kept)
    docs = list_documents(store)
    assert len(docs) == 1 and docs[0]["latest_version"] == 2
    assert store.count() >= 1


def test_pinecone_dimension_mismatch_is_reported(monkeypatch):
    mod, _, _ = make_fake_pinecone("new", 384, exists=True)
    monkeypatch.setitem(sys.modules, "pinecone", mod)
    with pytest.raises(RuntimeError, match="dimension 384"):
        PineconeStore("key", "idx", 1024)


# ---------------------------------------------------------------- Groq
def fake_groq(monkeypatch, replies):
    seen = []

    class Completions:
        def create(self, **kw):
            seen.append(kw)
            r = replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return NS(choices=[NS(message=NS(content=r))])

    class Groq:
        def __init__(self, api_key, timeout, max_retries):
            self.chat = NS(completions=Completions())

    mod = types.ModuleType("groq")
    mod.Groq = Groq
    monkeypatch.setitem(sys.modules, "groq", mod)
    return seen


def test_groq_receives_generation_controls(monkeypatch):
    seen = fake_groq(monkeypatch, ['{"answerable": false}'])
    GroqLLM("k").chat([{"role": "user", "content": "x"}],
                      Settings(model="llama-3.3-70b-versatile", temperature=0.05, top_p=0.8, max_tokens=321, seed=11),
                      json_mode=True)
    kw = seen[0]
    assert kw["temperature"] == 0.05 and kw["top_p"] == 0.8 and kw["max_completion_tokens"] == 321
    assert kw["seed"] == 11 and kw["response_format"] == {"type": "json_object"}
    assert "top_k" not in kw                                       # unsupported by Groq


def test_groq_json_mode_fallback_and_errors(monkeypatch):
    seen = fake_groq(monkeypatch, [Exception("400 response_format not supported"), "plain reply"])
    assert GroqLLM("k").chat([{"role": "user", "content": "x"}], Settings(), json_mode=True) == "plain reply"
    assert "response_format" not in seen[1]

    fake_groq(monkeypatch, [Exception("Error code: 401 invalid api key")])
    with pytest.raises(LLMError, match="rejected the API key"):
        GroqLLM("k").chat([{"role": "user", "content": "x"}], Settings())
    fake_groq(monkeypatch, [Exception("429 rate limit exceeded")])
    with pytest.raises(LLMError, match="rate limit"):
        GroqLLM("k").chat([{"role": "user", "content": "x"}], Settings())
    with pytest.raises(LLMError, match="API key"):
        GroqLLM("")


def test_parse_llm_json_tolerance():
    assert parse_llm_json('{"a": 1}') == {"a": 1}
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_llm_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert parse_llm_json("nothing") is None and parse_llm_json("") is None
