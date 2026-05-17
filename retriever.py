from vector_store import similarity_search

RELEVANCE_THRESHOLD = 0.6  # resumes and short docs need a looser threshold than papers


def retrieve(question: str, n_results: int = 3, source: str | None = None) -> list[dict]:
    """Return the top relevant chunks for a question, filtered by threshold.

    Pass source=filename to restrict search to one document.
    Returns an empty list if nothing scores well enough.
    """
    results = similarity_search(question, n_results=n_results, source=source)
    relevant = [r for r in results if r["distance"] <= RELEVANCE_THRESHOLD]
    return relevant
