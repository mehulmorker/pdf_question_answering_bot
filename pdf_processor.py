import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document


def load_and_chunk_pdf(file_path: str) -> list[Document]:
    # Extract as Markdown: preserves headers, sections, and document structure.
    # Handles multi-column layouts and figures without leaking figure text as prose.
    md_text = pymupdf4llm.to_markdown(file_path)

    # MarkdownTextSplitter splits on headers first (##, ###), then falls back
    # to RecursiveCharacterTextSplitter behaviour. Keeps sections coherent.
    splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.create_documents(
        texts=[md_text],
        metadatas=[{"source": file_path}],
    )

    print(f"Extracted {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <path-to-pdf>")
        sys.exit(1)

    chunks = load_and_chunk_pdf(sys.argv[1])
    print(f"Avg chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars\n")

    for i, chunk in enumerate(chunks[:5]):
        print(f"--- Chunk {i} ---")
        print(chunk.page_content)
        print()
