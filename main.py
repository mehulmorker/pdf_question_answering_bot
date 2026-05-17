from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from pdf_processor import load_and_chunk_pdf
from vector_store import add_chunks
from generator import answer_question, NO_ANSWER

UPLOADS_DIR = Path("./uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PDF Q&A", description="Upload a PDF, ask questions about it.")


# ── Request / response shapes ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    source: str | None = None  # optional: restrict search to one uploaded PDF

class AskResponse(BaseModel):
    answer: str
    found_in_document: bool   # False when the retriever returned nothing relevant

class UploadResponse(BaseModel):
    message: str
    chunks_indexed: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save to disk — pymupdf4llm needs a file path, not a byte stream
    save_path = UPLOADS_DIR / file.filename
    save_path.write_bytes(file.file.read())

    chunks = load_and_chunk_pdf(str(save_path))
    count = add_chunks(chunks)

    return UploadResponse(
        message=f"'{file.filename}' indexed successfully.",
        chunks_indexed=count,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer = answer_question(request.question, source=request.source)
    return AskResponse(
        answer=answer,
        found_in_document=(answer != NO_ANSWER),
    )
