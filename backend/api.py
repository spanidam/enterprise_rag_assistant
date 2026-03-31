import time
import shutil
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from pathlib import Path
from fastapi import HTTPException
from pydantic import BaseModel, Field
from backend.main import run_pipeline
from backend.logging_config import logger
from backend.answer_engine import answer_question
from backend.orchestrator import retrieve_context
from backend.hallucination_checker import check_hallucination

app = FastAPI(
    title="Enterprise RAG API",
    description="Grounded RAG system with confidence scoring and abstention",
    version="1.0.0"
)

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request latency logging
# -----------------------------
@app.middleware("http")
async def measure_latency(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = (time.time() - start) * 1000

    logger.info(
        "request_completed",
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency, 2)
    )
    return response

# -----------------------------
# Models
# -----------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)

class AskResponse(BaseModel):
    question: str
    answer: str
    verdict: str
    confidence_score: float
    citation_coverage: float
    used_sources: list[str]
    abstained: bool
    abstain_reason: str

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"status": "Enterprise RAG Backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Main RAG endpoint
# -----------------------------
@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    logger.info("query_received", question=req.question)

    result = run_pipeline(req.question)

    return {
        "question": result["question"],
        "answer": result["answer"],
        "verdict": result["verdict"],
        "confidence_score": result["confidence_score"],
        "citation_coverage": result["citation_coverage"],
        "used_sources": result["used_sources"],
        "abstained": result["abstained"],
        "abstain_reason": result["abstain_reason"],
    }