from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Reads OPENAI_API_KEY automatically from the environment.
# Set it before running: export OPENAI_API_KEY="sk-..."
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"  # produces 1536-dimension vectors


def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns a vector of 1536 floats."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple strings in one API call — cheaper than looping embed_text."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # The API returns results in the same order as the input
    return [item.embedding for item in response.data]


if __name__ == "__main__":
    import sys
    from pdf_processor import load_and_chunk_pdf

    if len(sys.argv) < 2:
        print("Usage: python embedder.py <path-to-pdf>")
        sys.exit(1)

    chunks = load_and_chunk_pdf(sys.argv[1])
    first_chunk = chunks[0]

    print(f"Chunk text (first 200 chars):\n{first_chunk.page_content[:200]}\n")

    vector = embed_text(first_chunk.page_content)

    print(f"Vector dimensions : {len(vector)}")
    print(f"First 10 values   : {[round(v, 6) for v in vector[:10]]}")
    print()

    # Demonstrate cosine similarity between two chunks
    import math

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a))
        mag_b = math.sqrt(sum(x ** 2 for x in b))
        return dot / (mag_a * mag_b)

    if len(chunks) > 1:
        second_chunk = chunks[1]
        vec_a = embed_text(first_chunk.page_content)
        vec_b = embed_text(second_chunk.page_content)

        sim = cosine_similarity(vec_a, vec_b)
        print(f"Cosine similarity between chunk 0 and chunk 1: {sim:.4f}")
        print("(Adjacent chunks often score 0.7–0.9 — they share topic and vocabulary)")
