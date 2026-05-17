# PDF Q&A RAG System — Architecture

## Final Stack

| Concern | Choice | Notes |
|---------|--------|-------|
| PDF extraction | `pymupdf4llm` | Markdown output, preserves structure, no garbage figure text |
| Text splitting | `MarkdownTextSplitter` | Splits on headers first, then falls back to character split |
| Embeddings | OpenAI `text-embedding-3-small` | 1536-dim vectors, ~$0.02/M tokens |
| Vector store | ChromaDB (Linux) | Cosine distance, persists to `./chroma_db/`, Linux x86_64 only |
| LLM (generation) | OpenAI `gpt-4.1-nano` | $0.10/$0.40 per 1M tokens, 1M token context window |
| API layer | FastAPI + uvicorn | `POST /upload`, `POST /ask` |
| Env vars | `python-dotenv` | `OPENAI_API_KEY` via `.env` file |
| Python | 3.12 | 3.14 lacks wheels for native deps |

---

## Two Phases

### Indexing (once, at upload time)
```
PDF
 ↓
pymupdf4llm (page_chunks=True)        → Markdown text + page_number per page
 ↓
MarkdownTextSplitter                   → chunks (size by doc type, 20% overlap)
 ↓
embed_texts() [single API call]        → 1536-dim vectors
 ↓
ChromaDB upsert                        → stored with source + page metadata
```

### Querying (every question)
```
Question
 ↓
embed_text()                           → query vector
 ↓
ChromaDB similarity_search             → top-N chunks (filtered by source if specified)
 ↓
Retrieval guardrail                    → distance > threshold → "I don't know" (no LLM call)
 ↓
Prompt: system rules + chunks + question
 ↓
gpt-4.1-nano (temperature=0)           → answer with citations
 ↓
AskResponse(answer, found_in_document)
```

**Critical constraint:** the same embedding model must be used in both phases.

---

## Components

### 1. Document Processor (`pdf_processor.py`)
- **Input:** PDF file path, optional `chunk_size`, `chunk_overlap`
- **Output:** `list[Document]` — chunks with `source` + `page` metadata
- Extracts Markdown via `pymupdf4llm.to_markdown(path, page_chunks=True)`
- Splits with `MarkdownTextSplitter` — respects `##` headers before falling back
- Metadata key from pymupdf4llm is `page_number` (not `page`)
- Default: chunk_size=1000, overlap=200 for prose; use 300/50 for resumes

### 2. Embedding Model (`embedder.py`)
- Wraps OpenAI `text-embedding-3-small`
- `embed_text(str)` — single string, used at query time
- `embed_texts(list)` — batch call, used at index time (one API call for all chunks)

### 3. Vector Store (`vector_store.py`)
- ChromaDB collection persisted to `./chroma_db/`
- **Distance:** cosine (`hnsw:space: cosine`) — set at creation, cannot change
- **IDs:** `{source_filename}::chunk_{i}` — scoped so different PDFs never collide
- **`upsert` not `add`** — supports re-uploading the same PDF without crashing
- `similarity_search(question, n_results, source)` — `source` param filters to one document

### 4. Retriever (`retriever.py`)
- Calls `similarity_search()` then filters by `RELEVANCE_THRESHOLD`
- Returns empty list if nothing passes — caller treats this as "no answer"
- Threshold: 0.5 for academic papers, 0.6 for resumes/short structured docs

### 5. Generator (`generator.py`)
- Guards first: empty chunk list → return `NO_ANSWER` string, no LLM call
- Builds numbered context blocks: `[Chunk N | Source: file.pdf | Page: X]`
- System prompt instructs: only use context, cite source, exact phrase if unknown
- Calls `gpt-4.1-nano` at `temperature=0` (deterministic)

### 6. FastAPI Layer (`main.py`)
- `POST /upload` — saves PDF to `./uploads/`, indexes chunks, returns count
- `POST /ask` — accepts `question` + optional `source` filter, returns `answer` + `found_in_document`
- Pydantic models for automatic validation and `/docs` UI

---

## Retrieval Strategy

### Semantic search (what we built)
Good for conceptual questions. Fails on specific keywords, proper nouns, and numbers buried in large chunks.

### Hybrid search (next evolution)
Combines semantic + BM25 keyword search via Reciprocal Rank Fusion:

```
Query
  ├── BM25              → exact keyword match (names, acronyms, numbers)
  └── Semantic (vector) → meaning match (concepts, paraphrase)
          ↓
   Reciprocal Rank Fusion → unified ranking
```

| Document type | BM25 weight | Semantic weight |
|---------------|-------------|-----------------|
| Academic papers | 0.3 | 0.7 |
| Resumes / CVs | 0.5 | 0.5 |
| Legal / medical | 0.6 | 0.4 |

LangChain: `EnsembleRetriever([BM25Retriever, vectorstore.as_retriever()], weights=[0.4, 0.6])`

---

## Guardrails

```
User question
     ↓
[1. Input guardrail]      — validate/block harmful input
     ↓
[2. Retrieval guardrail]  — threshold filter ← BUILT
     ↓
[3. Output guardrail]     — validate LLM answer
     ↓
Response
```

---

## Key Design Decisions

| Decision | Choice | Tradeoff |
|----------|--------|----------|
| PDF loader | `pymupdf4llm` | Structure-aware vs. raw text (garbage figure content) |
| Chunk size | Doc-dependent (300–1500) | Precision vs. context — must match document type |
| Chunk overlap | 20% of chunk_size | Prevents boundary splits; only triggers on forced cuts |
| Embeddings | OpenAI API | Quality vs. data leaving your machine |
| Vector store | ChromaDB local | Simple setup vs. not production-scalable |
| Distance metric | Cosine | Direction only (meaning) vs. L2 (length-sensitive) |
| Retrieval | Semantic only → Hybrid | Simple vs. handles keywords/proper nouns |
| Top-N | N=3 default | Cost/speed vs. recall completeness |
| Threshold | 0.5–0.6 | Fewer hallucinations vs. more "I don't know" responses |
| LLM temperature | 0 | Deterministic factual answers vs. creative variation |
