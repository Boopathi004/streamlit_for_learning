"""Central configuration: model catalogues, defaults and the Settings object."""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Embedding back-ends. `default_threshold` is the minimum cosine similarity a
# chunk needs to be used as context. Score scales differ between models, so the
# UI lets you tune it and shows the scores of everything that was retrieved.
# --------------------------------------------------------------------------
EMBEDDERS: dict[str, dict] = {
    "local-minilm": {
        "label": "Local sentence-transformers: all-MiniLM-L6-v2 (384-d)",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "default_threshold": 0.30,
    },
    "pinecone-hosted": {
        "label": "Pinecone Inference: llama-text-embed-v2 (1024-d)",
        "model": "llama-text-embed-v2",
        "dim": 1024,
        "default_threshold": 0.30,
    },
    "hash-demo": {
        "label": "Offline demo: hashed bag-of-words (512-d, keyword matching only)",
        "model": "hashing-bow",
        "dim": 512,
        "default_threshold": 0.12,
    },
}

# Production chat models on Groq (checked against console.groq.com/docs/models).
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

DEPARTMENTS = ["HR", "Finance", "Legal", "Operations", "IT", "Sales", "Marketing", "General"]
DOC_TYPES = ["Policy", "SOP", "Report", "Contract", "Financial data", "Invoice/Receipt", "Form", "Other"]

SUPPORTED_EXTENSIONS = ["pdf", "docx", "rtf", "txt", "md", "xlsx", "csv", "png", "jpg", "jpeg"]

DOC_NAMESPACE = "docs"
REGISTRY_NAMESPACE = "registry"

IDK_MESSAGE = "I don't know based on the available documents."


@dataclass
class Settings:
    # ---- LLM generation (Groq). Low temperature = factual, repeatable answers.
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 700
    seed: int | None = 7

    # ---- Retrieval
    top_k: int = 5                      # number of chunks fetched from Pinecone
    min_score: float = 0.30             # retrieval threshold filter
    include_superseded: bool = False    # search old versions too (audit use)
    departments: list[str] = field(default_factory=list)   # [] = all
    doc_types: list[str] = field(default_factory=list)     # [] = all

    # ---- Answering policy
    refuse_low_confidence: bool = False
    history_turns: int = 4              # past turns used for follow-up questions

    # ---- Chunking
    chunk_size: int = 900
    chunk_overlap: int = 150
