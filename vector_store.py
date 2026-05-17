import json
import chromadb
from pathlib import Path
from langchain_core.documents import Document
from embedder import embed_text, embed_texts

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_chunks"
BM25_DIR = Path("./bm25_store")


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Must be set at creation time — cannot change later.
        # Tells ChromaDB to use cosine distance instead of the default L2.
        metadata={"hnsw:space": "cosine"},
    )


def _save_chunks(chunks: list[Document]) -> None:
    """Persist chunks to disk so BM25 can rebuild its index at query time.

    BM25 is not a vector index — it needs the raw text. ChromaDB stores
    vectors, not text suitable for BM25 tokenisation, so we keep a separate
    JSON copy per source document.
    """
    BM25_DIR.mkdir(exist_ok=True)
    source = chunks[0].metadata.get("source", "unknown")
    filename = Path(source).name + ".json"
    data = [{"text": c.page_content, "metadata": c.metadata} for c in chunks]
    (BM25_DIR / filename).write_text(json.dumps(data))


def load_chunks(source: str | None = None) -> list[Document]:
    """Load persisted chunks for BM25 index construction.

    Pass source=filepath to load one document's chunks only.
    Omit to load all indexed documents (used when no source filter is set).
    """
    if not BM25_DIR.exists():
        return []
    if source:
        path = BM25_DIR / (Path(source).name + ".json")
        files = [path] if path.exists() else []
    else:
        files = list(BM25_DIR.glob("*.json"))

    chunks = []
    for f in files:
        data = json.loads(f.read_text())
        chunks.extend([
            Document(page_content=d["text"], metadata=d["metadata"])
            for d in data
        ])
    return chunks


def add_chunks(chunks: list[Document]) -> int:
    """Embed all chunks in one API call, then upsert them into ChromaDB.
    Also saves chunks to disk for BM25 retrieval.
    """
    collection = _get_collection()

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    source = chunks[0].metadata.get("source", "unknown")
    ids = [f"{source}::chunk_{i}" for i in range(len(chunks))]

    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    _save_chunks(chunks)
    return len(chunks)


def similarity_search(question: str, n_results: int = 3, source: str | None = None) -> list[dict]:
    """Embed the question, find the n_results closest chunks, return with scores.

    Pass source=filename to restrict results to a single indexed document.
    Without it, search runs across every PDF in the collection.
    """
    collection = _get_collection()

    query_embedding = embed_text(question)

    # where filter is optional — only apply it when a source is specified
    where = {"source": source} if source else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # collection.query always returns lists-of-lists (one list per query).
    # We send one query at a time, so we always take index [0].
    retrieved = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": text,
            "metadata": meta,
            "distance": round(distance, 4),
            "similarity": round(1 - distance, 4),
        })

    return retrieved


if __name__ == "__main__":
    import sys
    from pdf_processor import load_and_chunk_pdf

    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <path-to-pdf>")
        sys.exit(1)

    # --- Indexing ---
    print("Loading and chunking PDF...")
    chunks = load_and_chunk_pdf(sys.argv[1])
    print(f"Chunked into {len(chunks)} pieces. Embedding and storing...")
    add_chunks(chunks)

    # --- Querying ---
    SAMPLE_QUERY = "What is the attention mechanism?"
    print(f"\nQuery: '{SAMPLE_QUERY}'\n")

    results = similarity_search(SAMPLE_QUERY, n_results=3)

    NOISE_THRESHOLD = 0.5  # distance above this → chunk is probably irrelevant

    for i, result in enumerate(results):
        flag = " *** LOW RELEVANCE" if result["distance"] > NOISE_THRESHOLD else ""
        print(f"--- Result {i + 1} | similarity: {result['similarity']} | distance: {result['distance']}{flag}")
        print(f"    Source: {result['metadata'].get('source', '?')} | Page: {result['metadata'].get('page', '?')}")
        print(f"    {result['text'][:300]}...")
        print()
