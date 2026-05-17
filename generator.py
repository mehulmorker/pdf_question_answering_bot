from openai import OpenAI
from dotenv import load_dotenv
from retriever import retrieve

load_dotenv()
client = OpenAI()

LLM_MODEL = "gpt-4.1-nano"
NO_ANSWER = "I don't know based on the provided document."

# ── Prompt template ──────────────────────────────────────────────────────────
#
# This is the exact text sent to the LLM. Two parts:
#
# SYSTEM — standing rules that constrain every response:
#   - Only use the provided context (no training knowledge)
#   - Say the exact NO_ANSWER phrase when the context doesn't help
#   - Cite the source so the user can verify
#
# USER — the context blocks + the question.
#   Each chunk is labelled with its chunk number, source file, and page so the
#   LLM can refer to them specifically ("According to page 3...").
#
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise Q&A assistant. Your job is to answer questions \
using ONLY the document context provided by the user.

Rules you must follow:
1. Base your answer exclusively on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, respond with exactly: \
"{no_answer}"
3. When answering, cite which chunk, section, or page your answer comes from.
4. Be concise. Do not pad the answer with unnecessary explanation.""".format(
    no_answer=NO_ANSWER
)


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page", "?")
        parts.append(
            f"[Chunk {i} | Source: {source} | Page: {page}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def answer_question(question: str) -> str:
    """Full RAG pipeline: retrieve → check → prompt → generate."""
    chunks = retrieve(question)

    # Guard: if nothing is relevant, say so rather than hallucinating
    if not chunks:
        return NO_ANSWER

    context = _build_context(chunks)

    user_message = f"Context from the document:\n\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,          # deterministic — factual Q&A, not creative writing
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    questions = [
        "What is the attention mechanism?",
        "What BLEU score did the model achieve on English-to-German translation?",
        "What is the capital of France?",   # unrelated — should return NO_ANSWER
    ]

    for q in questions:
        print(f"Q: {q}")
        print(f"A: {answer_question(q)}")
        print()
