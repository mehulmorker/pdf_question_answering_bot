import chromadb
from langchain_core.documents import Document
from embedder import embed_text, embed_texts

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_chunks"


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Must be set at creation time — cannot change later.
        # Tells ChromaDB to use cosine distance instead of the default L2.
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Document]) -> int:
    """Embed all chunks in one API call, then upsert them into ChromaDB.

    Uses upsert (not add) so re-uploading the same PDF updates existing chunks
    rather than crashing on duplicate IDs. IDs are scoped to the source filename
    so two different PDFs never overwrite each other's chunks.
    """
    collection = _get_collection()

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Include source filename in the ID so different PDFs don't collide
    source = chunks[0].metadata.get("source", "unknown")
    ids = [f"{source}::chunk_{i}" for i in range(len(chunks))]

    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
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
