from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_chunk_pdf(file_path: str) -> list[Document]:
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    # Join all pages before chunking so the splitter can cross page boundaries.
    # Chunking per-page severs any concept that continues onto the next page.
    # Tradeoff: chunks no longer carry an exact page number in their metadata.
    full_text = "\n\n".join(page.page_content for page in pages)
    full_doc = Document(
        page_content=full_text,
        metadata={"source": file_path},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_documents([full_doc])


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
