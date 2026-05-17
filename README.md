# PDF Q&A — RAG System

Ask questions about any PDF and get grounded, cited answers. If the answer isn't in the document, it says so — no hallucinations.

Built as a learning project to understand Retrieval-Augmented Generation (RAG) from scratch, one component at a time.

---

## What it does

- Upload any PDF through a web UI or REST API
- Ask questions in plain English
- Get answers with exact page citations
- See the specific text chunks the LLM used to answer
- Returns "I don't know" when the answer isn't in the document

---

## How it works

RAG has two phases:

**Indexing** — runs once when you upload a PDF:
```
PDF → pymupdf4llm (Markdown) → MarkdownTextSplitter → chunks
                                                          ↓
                                              OpenAI text-embedding-3-small
                                                          ↓
                                              ChromaDB (cosine distance)
```

**Querying** — runs on every question:
```
Question → embed → ChromaDB similarity search → top-N chunks
                                                      ↓
                               relevance filter (distance threshold)
                                                      ↓
                               gpt-4.1-nano (temperature=0) → cited answer
```

The retrieval guardrail is the key piece: if no chunk scores close enough to the question, the system returns a fixed "I don't know" response and never calls the LLM. This is what prevents hallucination.

---

## Tech stack

| Concern | Choice | Why |
|---------|--------|-----|
| PDF extraction | `pymupdf4llm` | Structure-aware — keeps figures as figures, not garbage text |
| Text splitting | `MarkdownTextSplitter` | Splits on headers first, not mid-sentence |
| Embeddings | OpenAI `text-embedding-3-small` | 1536-dim, $0.02/M tokens, strong quality |
| Vector store | ChromaDB | Local persistence, cosine distance, metadata filtering |
| LLM | OpenAI `gpt-4.1-nano` | Cheaper and larger context than gpt-4o-mini (1M tokens) |
| API | FastAPI + uvicorn | REST endpoints with automatic docs at `/docs` |
| UI | Streamlit | Simple web interface, no frontend build step |
| Python | 3.12 | 3.14 lacks wheels for native dependencies |

---

## Project structure

```
pdf-rag/
├── pdf_processor.py   # PDF → chunks (pymupdf4llm + MarkdownTextSplitter)
├── embedder.py        # Wraps OpenAI text-embedding-3-small
├── vector_store.py    # ChromaDB: store, upsert, similarity search
├── retriever.py       # Applies relevance threshold, returns empty list if nothing passes
├── generator.py       # Builds prompt, calls gpt-4.1-nano, enforces NO_ANSWER guard
├── main.py            # FastAPI: POST /upload, POST /ask
├── streamlit_app.py   # Web UI: upload sidebar + question/answer main area
└── uploads/           # PDFs saved here on upload
```

Each file is one component. They are deliberately kept separate so each layer can be understood and tested independently.

---

## Setup

**Requirements:** Python 3.12, Linux x86_64 (ChromaDB's `onnxruntime` dependency does not have wheels for Intel Mac)

```bash
# Clone and install
git clone <repo-url>
cd pdf-rag

# Install dependencies
uv sync

# Add your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

**.env.example:**
```
OPENAI_API_KEY=sk-...
```

---

## Running

### Web UI (Streamlit)

```bash
uv run streamlit run streamlit_app.py
```

Open `http://localhost:8501` — upload a PDF from the sidebar, ask questions in the main area.

### REST API (FastAPI)

```bash
uv run uvicorn main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API explorer.

**Upload a PDF:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your-document.pdf"
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main argument?", "source": "uploads/your-document.pdf"}'
```

The `source` field is optional — omit it to search across all indexed PDFs.

---

## Key design decisions

**Cosine distance, not L2.** ChromaDB defaults to Euclidean (L2) distance, which is length-sensitive and wrong for text. Cosine distance measures semantic direction only. Must be set at collection creation — cannot be changed after the fact.

**`upsert` not `add`.** Re-uploading the same PDF would crash with `add`. `upsert` with source-scoped chunk IDs (`filename::chunk_0`) handles re-indexing cleanly.

**Relevance threshold.** Every retrieved chunk has a cosine distance score. Chunks above the threshold (too far from the question) are filtered out before the LLM sees them. If nothing passes, the system returns "I don't know" without calling the LLM at all. This is the hallucination guardrail.

**Chunk size matches document type.** Academic papers use `chunk_size=1000, overlap=200`. Resumes use `chunk_size=300, overlap=50`. A chunk that's too large dilutes the embedding — specific facts get buried and never retrieved.

**Temperature 0.** Factual Q&A requires deterministic answers. Any temperature above 0 adds randomness that serves no purpose and risks paraphrasing facts incorrectly.

---

## Cost

For a 15-page academic PDF:
- Indexing (embed all chunks): ~$0.0003
- Per question (embed + LLM): ~$0.0002

A 300-page textbook with 100 questions costs a few cents total.
