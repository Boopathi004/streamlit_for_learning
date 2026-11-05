# Enterprise Knowledge Assistant (RAG)

Employees ask questions in a chat; the assistant answers **only from your internal documents**, cites the exact
file, version and location, and says *"I don't know"* when the documents don't contain the answer.

```
 Upload (pdf, docx, rtf, txt, xlsx, csv, png, jpg)
        │
        ▼
 Parsers + OCR (Tesseract) ─► structure-aware chunks ─► embeddings ─► Pinecone
        │                         (section / page / sheet-rows)        │  metadata: doc_id, filename, doc_type,
        │                                                              │  department, version_id, updated_at,
        │                                                              │  upload_date, is_latest, location, text
 Question ─► (follow-up? rewrite with chat history) ─► embed ─► top-k search, filter is_latest=true
                                                                          │
                              similarity threshold ◄──────────────────────┘
                                   │ nothing passes → "I don't know" (LLM is never called)
                                   ▼
                  Groq LLM (low temperature, top-p) with a strict grounded prompt, JSON output
                                   │
              citation check → number check → confidence score → answer + sources
```

## 1. Setup

```bash
cd rag_system
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# OCR engine (required for images / scanned PDFs)
#   Ubuntu/Debian: sudo apt install tesseract-ocr     macOS: brew install tesseract
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki  (add to PATH)

cp .env.example .env          # optional: put GROQ_API_KEY / PINECONE_API_KEY here, or type them in the sidebar
streamlit run app.py
```

Keys: Groq → https://console.groq.com/keys · Pinecone → https://app.pinecone.io (free Starter plan is enough).
The index is created for you (serverless, cosine). **No Pinecone key yet?** Choose *In-memory (demo)* in the sidebar to try everything locally.

Quick try: sidebar → **Load bundled sample documents** → ask the questions in `sample_docs/TEST_QUESTIONS_with_answers.md`.

Deploy on Streamlit Community Cloud: push the repo, add the keys under *Secrets*; `packages.txt` installs Tesseract.

## 2. How each requirement is met

| Requirement | Implementation | File |
|---|---|---|
| Multi-format upload | pdf, docx, rtf, txt/md, xlsx, csv, png, jpg parsed to text blocks with a location (page, section, sheet rows) | `rag/parsers.py` |
| OCR | Tesseract on images; scanned PDF pages are rasterised (pypdfium2) and OCR'd; low-confidence scans raise a warning | `rag/parsers.py` |
| Structured data | Excel/CSV rows become `column: value; ...` lines, so each row is self-describing; sheet titles and notes kept | `rag/parsers.py` |
| Chunking | Heading-aware, ~900 chars, 150 overlap for prose, none for tables; table prefix repeated per chunk | `rag/chunking.py` |
| Embeddings | Local MiniLM (default), Pinecone-hosted `llama-text-embed-v2`, or an offline demo embedder | `rag/embeddings.py` |
| Pinecone + metadata | Every vector carries doc type, upload date, department, version, `updated_at`, `is_latest`, location, section, text | `rag/vectorstore.py`, `rag/ingest.py` |
| Retrieval | query → embedding → top-k similarity search, metadata filter, nothing but Pinecone is consulted | `rag/retriever.py` |
| LLM control | Groq: temperature (default 0.1), top-p (0.9), max tokens, seed, JSON mode | `rag/llm.py` |
| Top-k | Retrieval top-k (slider). Groq's API has **no sampling top-k**, so none is sent; this is the top-k that governs grounding | `rag/retriever.py` |
| Prompting | "Answer only from context", "if unknown say I don't know", mandatory `[n]` citations, conflict and prompt-injection rules | `rag/prompts.py` |
| Hallucination reduction | five layers, see §3 | `rag/pipeline.py` |
| Updates / versioning | `version_id`, `updated_at`, `is_latest`; old versions deprecated or deleted; retrieval filters on latest | `rag/ingest.py` |
| Chat memory | `st.session_state` history; follow-ups rewritten into standalone queries; last N turns sent to the model | `app.py`, `rag/pipeline.py` |
| UI | Sidebar: keys, upload (editable metadata), model/retrieval controls. Main: chat, confidence, sources, retrieval debug | `app.py` |
| Traceability | each answer lists source file, version, date, location, similarity; every Q&A is appended to `logs/qa_log.jsonl` | `app.py` |

## 3. Hallucination controls (in the order they fire)

1. **Retrieval threshold.** Chunks under the similarity threshold are dropped. If none remain, the answer is *"I don't know based on the available documents."* and the LLM is not called.
2. **Strict grounded prompt.** Numbered excerpts with version / date / status; the model must answer only from them, cite `[n]`, and return JSON `{answerable, answer, citations, confidence}`. Retrieved text is declared *data, not instructions* (prompt-injection defence).
3. **Citation enforcement.** An answer with no valid `[n]` pointing at a real excerpt is rejected.
4. **Number verification.** Every figure in the answer must appear in the cited excerpts; otherwise a warning is shown and confidence drops.
5. **Confidence.** `min(retrieval strength, model self-rating, number check)` → high / medium / low. Optional *Refuse when confidence is low*.

Conflicting or outdated documents: only the newest version is searched (`is_latest = true`); if two *different* documents disagree, the prompt makes the model say so and name both. *Search superseded versions* is available for audits and flags answers that use outdated text.

## 4. Updating documents

Upload a file whose `doc_id` matches an existing one (the editable `doc_id` column defaults to the file name without suffixes like `_v2`, `_final`, `(1)`):

* identical text → skipped ("unchanged"); only department/type edits are applied to the existing vectors
* changed text → `version_id + 1`; new chunks are stored first, **then** the old version is marked `is_latest=false` (or deleted). A failed upload never leaves half a version behind.
* the per-document manifest lives in Pinecone (namespace `registry`), so state survives restarts and is shared by all users.

## 5. Tuning the threshold

Similarity scales differ per embedding model. Open **Retrieval details** under any answer: it lists every chunk Pinecone returned with its score and whether it was used. Raise the threshold if irrelevant chunks pass; lower it if good chunks are cut. Defaults: MiniLM and Pinecone-hosted 0.30, demo embedder 0.12.

## 6. Tests

```bash
pytest -q        # 48 tests, ~20 s, no network
```
Covers every parser including OCR on the two scanned samples, chunking, retrieval over the sample corpus, versioning (new / unchanged / updated / delete mode / failed upload), each hallucination guard, the Pinecone and Groq wrappers (against SDK stand-ins) and the Streamlit app end to end (`streamlit.testing`).

## 7. Known limits

* Live Groq, Pinecone and MiniLM calls are exercised only through stand-ins in the test suite; do one real run with your keys before relying on it.
* OCR quality follows image quality. Handwriting is not supported by Tesseract.
* Excel formulas are read from cached values; a workbook never opened in Excel may lack them (a warning is shown).
* Number verification flags derived figures (sums, percentages) too; treat the warning as "check this", not "wrong".
* No user authentication or per-user document permissions. Add them (e.g. Pinecone namespaces or a `department` filter bound to the login) before multi-team use.
* Never commit `.env`; keys typed in the sidebar are kept only in the session.
