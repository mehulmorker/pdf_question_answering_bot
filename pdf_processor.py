from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_chunk_pdf(file_path: str) -> list[Document]:
    # Load: one Document per page, each with metadata: {"source": path, "page": int}
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # max characters per chunk
        chunk_overlap=200,   # characters shared between adjacent chunks
        separators=[
            "\n\n",  # paragraph break — try this first
            "\n",    # line break
            ".",     # sentence end
            " ",     # word boundary
            "",      # character-level last resort
        ],
    )

    chunks = splitter.split_documents(pages)
    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <path-to-pdf>")
        sys.exit(1)

    chunks = load_and_chunk_pdf(sys.argv[1])

    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    print()

    # Show first 3 chunks so you can see what the splitter produced
    for i, chunk in enumerate(chunks[:10]):
        print(f"--- Chunk {i} (page {chunk.metadata.get('page', '?')}) ---")
        print(chunk.page_content)
        # print(chunk)
        print()
