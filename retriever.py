from langchain_community.retrievers import BM25Retriever
from vector_store import similarity_search, load_chunks

RELEVANCE_THRESHOLD = 0.6  # applied to semantic-only results


def _rrf_merge(bm25_docs: list, semantic_results: list, n: int, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: combine BM25 and semantic results by rank position.

    RRF score = sum of 1/(rank + k) across both systems.
    k=60 is the standard constant — dampens the impact of very high ranks.
    Using rank (not raw score) means BM25 and cosine scores never need to be
    on the same scale.
    """
    scores: dict[str, dict] = {}

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        if key not in scores:
            scores[key] = {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "rrf_score": 0.0,
                "from_bm25": True,
            }
        scores[key]["rrf_score"] += 1.0 / (rank + k)

    for rank, result in enumerate(semantic_results):
        key = result["text"]
        if key not in scores:
            scores[key] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "rrf_score": 0.0,
                "distance": result["distance"],
                "from_bm25": False,
            }
        else:
            # chunk appeared in both — record distance for reference
            scores[key]["distance"] = result["distance"]
        scores[key]["rrf_score"] += 1.0 / (rank + k)

    merged = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return merged[:n]


def retrieve(question: str, n_results: int = 3, source: str | None = None) -> list[dict]:
    """Hybrid retrieval: BM25 + semantic search merged with RRF.

    BM25 matches pass the guardrail unconditionally — if the keyword is
    literally present in a chunk, the chunk is relevant by definition.
    Semantic-only results are still filtered by RELEVANCE_THRESHOLD.

    Falls back to semantic-only if no BM25 index exists for the source.
    """
    semantic_results = similarity_search(question, n_results=n_results, source=source)
    chunks = load_chunks(source)

    if chunks:
        bm25 = BM25Retriever.from_documents(chunks, k=n_results)
        bm25_docs = bm25.invoke(question)
        merged = _rrf_merge(bm25_docs, semantic_results, n_results)

        return [
            r for r in merged
            if r.get("from_bm25")                          # BM25 match → always relevant
            or r.get("distance", 1.0) <= RELEVANCE_THRESHOLD  # semantic match → threshold
        ]

    # Fallback: no BM25 index available
    return [r for r in semantic_results if r["distance"] <= RELEVANCE_THRESHOLD]
