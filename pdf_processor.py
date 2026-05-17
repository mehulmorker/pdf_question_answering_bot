import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document


def load_and_chunk_pdf(file_path: str) -> list[Document]:
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)

    try:
        # page_chunks=True gives per-page metadata including page number.
        # Fails on some PDFs due to a pymupdf4llm bug (empty range in page_filter).
        pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        all_chunks: list[Document] = []
        for page in pages:
            chunks = splitter.create_documents(
                texts=[page["text"]],
                metadatas=[{
                    "source": file_path,
                    "page": page["metadata"]["page_number"],
                }],
            )
            all_chunks.extend(chunks)

    except IndexError:
        # Fallback: extract full document without page tracking.
        # Loses per-chunk page numbers but avoids the pymupdf4llm bug.
        print("Warning: page_chunks failed, falling back to full-document extraction.")
        md_text = pymupdf4llm.to_markdown(file_path)
        all_chunks = splitter.create_documents(
            texts=[md_text],
            metadatas=[{"source": file_path, "page": "?"}],
        )

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
