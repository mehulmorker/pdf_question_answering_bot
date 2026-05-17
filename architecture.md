# PDF Q&A RAG System — Architecture

## Stack

| Concern | Choice | Notes |
|---------|--------|-------|
| Embeddings | OpenAI `text-embedding-3-small` | 1536-dim vectors, ~$0.02/M tokens |
| Vector store | ChromaDB | Local, embedded in-process, persists to disk |
| LLM (generation) | OpenAI `gpt-4.1-nano` | Newer, cheaper than gpt-4o-mini, 1M token context window |
| PDF extraction | PyMuPDF (`pymupdf`) | Best text quality for real-world PDFs |
| API layer | FastAPI | Async, Python |

---

## Two Phases

### Indexing (once, at upload time)
```
PDF → [Document Processor] → chunks
                               ↓
                        [Embedding Model] → vectors
                               ↓
                        [Vector Store] ← stored on disk
```

### Querying (every question)
```
Question → [Embedding Model] → query vector
                                    ↓
                            [Vector Store] → top-N chunks
                                    ↓
Question + chunks → [LLM] → Answer
```

**Critical constraint:** the same embedding model must be used in both phases.

---

## Components

### 1. Document Processor (`pdf_processor.py`)
- **Input:** PDF file path
- **Output:** `list[str]` — text chunks
- Extracts text page-by-page with PyMuPDF
- Splits using recursive strategy: tries `\n\n` → `\n` → `.` → ` ` → character
- **Chunk size:** 1000 characters | **Overlap:** 200 characters
- Pure function: `load_and_chunk_pdf(file_path: str) -> list[str]`

### 2. Embedding Model (`embedder.py`)
- Wraps OpenAI `text-embedding-3-small`
- Used in both indexing and querying phases
- Produces 1536-dimension vectors

### 3. Vector Store (`vector_store.py`)
- ChromaDB collection persisted to `./chroma_db/`
- Stores: vector + chunk text + metadata (filename, page number)
- Operations: `add_chunks()`, `query()`

### 4. Retriever (`retriever.py`)
- Embeds incoming question
- Queries vector store for top-N chunks (default N=3)
- Returns chunk texts as context

### 5. Generator (`generator.py`)
- Builds prompt: system instructions + retrieved chunks + user question
- Calls `gpt-4.1-nano`
- Returns answer string

### 6. FastAPI Layer (`main.py`)
- `POST /upload` — accepts PDF, runs indexing pipeline
- `POST /ask` — accepts question, runs query pipeline, returns answer

---

## Key Design Decisions

| Decision | Choice | Tradeoff |
|----------|--------|----------|
| Chunk size | 1000 chars / 200 overlap | Precision vs. context richness |
| Embeddings | API (OpenAI) | Quality vs. data leaving your machine |
| Vector store | ChromaDB local | Simple vs. not production-scalable |
| Top-N retrieval | N=3 (default) | Speed/cost vs. recall completeness |

---

## Build Order
1. Document Processor — PDF → chunks
2. Embedding — chunks → vectors
3. Vector Store — store and search vectors
4. Retriever — query-time chunk lookup
5. Generator — LLM answer synthesis
6. FastAPI — wire everything into an API
