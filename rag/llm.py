"""Groq LLM wrapper with controlled generation.

Controls exposed: temperature (low = factual), top_p (nucleus sampling), max tokens, seed.

Note on top-k: Groq's chat API does not accept a *sampling* top_k parameter
(its documented parameters are temperature, top_p, seed, max_completion_tokens, ...).
In this system "top-k" is therefore the *retrieval* top-k: how many chunks are pulled
from Pinecone. That is the top-k that matters most for grounding in a RAG pipeline.
"""
from __future__ import annotations

import json
import re

from .config import Settings


class LLMError(Exception):
    pass


class GroqLLM:
    def __init__(self, api_key: str):
        if not api_key:
            raise LLMError("Enter your Groq API key in the sidebar.")
        try:
            from groq import Groq
        except ImportError as e:
            raise LLMError("Install the Groq SDK: pip install groq") from e
        self._client = Groq(api_key=api_key, timeout=60.0, max_retries=2)

    def chat(self, messages: list[dict], settings: Settings, json_mode: bool = False,
             temperature: float | None = None, max_tokens: int | None = None) -> str:
        kwargs = dict(
            model=settings.model,
            messages=messages,
            temperature=settings.temperature if temperature is None else temperature,
            top_p=settings.top_p,
            max_completion_tokens=max_tokens or settings.max_tokens,
        )
        if settings.seed is not None:
            kwargs["seed"] = settings.seed
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            if json_mode and "response_format" in msg.lower():   # model without JSON mode: retry plain
                kwargs.pop("response_format", None)
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as e2:
                    raise LLMError(_friendly(e2)) from e2
            else:
                raise LLMError(_friendly(e)) from e
        return (resp.choices[0].message.content or "").strip()


def _friendly(e: Exception) -> str:
    s = str(e)
    low = s.lower()
    if "401" in s or "invalid api key" in low or "authentication" in low:
        return "Groq rejected the API key (401). Check the key in the sidebar."
    if "429" in s or "rate limit" in low:
        return "Groq rate limit reached (429). Wait a moment, or choose a smaller model."
    if "model" in low and ("not found" in low or "decommissioned" in low or "does not exist" in low):
        return f"The selected Groq model is unavailable: {s[:200]}"
    if "connection" in low or "timed out" in low:
        return "Could not reach Groq (network issue or timeout). Try again."
    return f"Groq error: {s[:300]}"


def parse_llm_json(raw: str) -> dict | None:
    """Parse the model's JSON reply, tolerating code fences and surrounding prose."""
    if not raw:
        return None
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    for candidate in (txt, _first_object(txt)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _first_object(s: str) -> str | None:
    start = s.find("{")
    end = s.rfind("}")
    return s[start:end + 1] if start != -1 and end > start else None
