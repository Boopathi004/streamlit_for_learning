"""Headless UI test of the Streamlit app (Streamlit's AppTest), with the Groq call scripted."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from streamlit.testing.v1 import AppTest   # noqa: E402

import rag.llm                              # noqa: E402
from test_rag import ScriptedLLM            # noqa: E402


class FakeGroq(ScriptedLLM):
    def __init__(self, api_key):
        if not api_key:
            raise rag.llm.LLMError("Enter your Groq API key in the sidebar.")
        super().__init__("grounded")


def _texts(elements):
    return [e.value for e in elements]


def test_app_end_to_end(monkeypatch):
    monkeypatch.setattr(rag.llm, "GroqLLM", FakeGroq)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    # Pinecone is the default store and no key was given -> friendly hint, chat disabled
    assert any("Pinecone API key" in v for v in _texts(at.info))
    assert at.chat_input[0].disabled

    # switch to the offline demo setup
    at.radio(key="store_kind").set_value("In-memory (demo, no Pinecone)")
    at.selectbox(key="emb_key").set_value("hash-demo")
    at.text_input(key="groq_key").set_value("test-key")
    at.run()
    assert not at.exception, at.exception
    assert not at.chat_input[0].disabled

    # load the bundled documents (runs parsers + OCR + versioning through the UI path)
    at.button(key="btn_samples").click().run()
    assert not at.exception, at.exception
    log = at.session_state["ingest_log"]
    assert len(log) == 8 and all(r["status"] == "new" for r in log), log
    assert any("OCR used" in r["note"] for r in log)

    # loading again must not duplicate anything
    at.button(key="btn_samples").click().run()
    assert all(r["status"] == "unchanged" for r in at.session_state["ingest_log"])

    # ask a question
    at.chat_input[0].set_value("How many annual leave days can be carried forward?").run()
    assert not at.exception, at.exception
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    ans = msgs[1]["answer"]
    assert ans.answerable and ans.sources and ans.sources[0]["filename"].startswith("HR-POL-001")

    # follow-up uses chat history
    at.chat_input[0].set_value("and what about casual leave?").run()
    assert not at.exception, at.exception
    assert len(at.session_state["messages"]) == 4

    # out-of-scope question -> "I don't know", no crash
    at.chat_input[0].set_value("What is the stock options vesting schedule?").run()
    assert not at.exception, at.exception
    last = at.session_state["messages"][-1]
    assert last["content"].startswith("I don't know") and not last["answer"].answerable

    # new chat clears the session
    at.button(key="btn_clear").click().run()
    assert at.session_state["messages"] == []


def test_missing_groq_key_shows_error(monkeypatch):
    monkeypatch.setattr(rag.llm, "GroqLLM", FakeGroq)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.radio(key="store_kind").set_value("In-memory (demo, no Pinecone)")
    at.selectbox(key="emb_key").set_value("hash-demo")
    at.run()
    at.chat_input[0].set_value("hello there, what is the policy?").run()
    assert not at.exception
    assert any("Groq API key" in e.value for e in at.error)
    assert at.session_state["messages"] == []
