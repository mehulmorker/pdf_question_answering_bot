import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document


def load_and_chunk_pdf(file_path: str) -> list[Document]:
    # page_chunks=True returns one dict per page, each with text + metadata
    # (including page number). Tradeoff: a section spanning two pages will be
    # split at the boundary — acceptable because MarkdownTextSplitter respects
    # headers, so splits happen at section edges rather than mid-sentence.
    pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)

    splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    all_chunks: list[Document] = []
    for page in pages:
        chunks = splitter.create_documents(
            texts=[page["text"]],
            metadatas=[{
                "source": file_path,
                "page": page["metadata"]["page"],  # 0-indexed
            }],
        )
        all_chunks.extend(chunks)

    print(f"Extracted {len(all_chunks)} chunks.")
    return all_chunks


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
