import time
import shutil
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from pathlib import Path
from fastapi import HTTPException
from pydantic import BaseModel, Field
from backend.logging_config import logger
from backend.answer_engine import answer_question
from backend.orchestrator import retrieve_context
from backend.hallucination_checker import check_hallucination

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed_chunks"


UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Enterprise RAG API",
    description="Grounded RAG system with citations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def measure_latency(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start

    logger.info(
        "request_completed",
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency * 1000, 2)
    )
    return response

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)

class Source(BaseModel):
    id: str
    chunk_file: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]

@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):

    
# ✅ Structured log at request entry
    logger.info("query_received", question=req.question)

    retrieved = retrieve_context(req.question)

    # ✅ Build context text from retrieved docs for verification
    context_text = "\n\n".join(
        [f"[S{i}] {doc.page_content}" for i, (doc, _) in enumerate(retrieved, start=1)]
    )
    
    logger.info("retrieval_completed", chunks=len(retrieved))

    answer = answer_question(req.question)

    # ✅ Hallucination check (AFTER answer, using retrieved sources)
    is_supported = check_hallucination(answer, context_text)
    logger.info(f"Hallucination check passed: {is_supported}")

    if not is_supported:
        answer = "⚠️ The retrieved documents do not fully support a confident answer."

    logger.info("Answer successfully generated")

    sources = []
    for i, (doc, _) in enumerate(retrieved, start=1):
        sources.append({
            "id": f"S{i}",
            "chunk_file": doc.metadata.get("chunk_file")
        })

    return {
        "question": req.question,
        "answer": answer,
        "sources": sources
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/chunk/{chunk_file}")
def get_chunk(chunk_file: str):
    path = CHUNKS_DIR / chunk_file
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chunk file not found")
    return {
        "chunk_file": chunk_file,
        "text": path.read_text(encoding="utf-8")
    }

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # Save file to uploads folder
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # IMPORTANT:
    # Here you should trigger your ingestion pipeline to:
    # - extract text from the PDF
    # - chunk
    # - embed
    # - store into chroma + chunk folder
    #
    # If you already have an ingest function/script, call it here.
    # Example (pseudo):
    # run_ingestion(dest)

    return {"status": "uploaded", "filename": file.filename}
