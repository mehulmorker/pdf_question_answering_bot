from vector_store import similarity_search

RELEVANCE_THRESHOLD = 0.5  # distance above this → chunk is noise, not signal


def retrieve(question: str, n_results: int = 3) -> list[dict]:
    """Return the top relevant chunks for a question, filtered by threshold.

    Returns an empty list if nothing scores well enough — callers should
    treat this as "the document doesn't contain an answer" rather than
    passing bad context to the LLM.
    """
    results = similarity_search(question, n_results=n_results)
    relevant = [r for r in results if r["distance"] <= RELEVANCE_THRESHOLD]
    return relevant
