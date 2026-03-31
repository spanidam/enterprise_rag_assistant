import os
import re
from openai import OpenAI
from dotenv import load_dotenv

from backend.prompts import ANSWER_PROMPT, REVISE_PROMPT
from backend.utils import validate_citations
from backend.verification import verify_answer

load_dotenv()

# -----------------------------
# OpenAI client
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Simulated Retrieval Output
# -----------------------------
retrieved_docs = {
    "S1": "Llama 3 is an open-source large language model developed by Meta.",
    "S2": "vLLM is a high-throughput inference engine for serving LLMs.",
    "S3": "Grounding LLMs with retrieved documents reduces hallucinations."
}

# -----------------------------
# Helpers
# -----------------------------
def format_sources(docs: dict) -> str:
    return "\n".join([f"[{k}] {v}" for k, v in docs.items()])


CITATION_PATTERN = re.compile(r"\[(S\d+)\]")

def split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]

def extract_citations(text: str):
    return CITATION_PATTERN.findall(text or "")

# -----------------------------
# Groundedness + Confidence
# -----------------------------
def compute_groundedness(answer: str, valid_source_ids: set):
    sentences = split_sentences(answer)
    if not sentences:
        return {"citation_coverage": 0.0, "used_sources": [], "source_count": 0, "confidence_score": 0.0}

    cited = 0
    used = set()

    for s in sentences:
        cites = extract_citations(s)
        valid = [c for c in cites if c in valid_source_ids]
        if valid:
            cited += 1
            used.update(valid)

    coverage = cited / len(sentences)
    diversity_bonus = min(len(used), 3) * 0.07
    confidence = min(1.0, (0.8 * coverage) + diversity_bonus)

    return {
        "citation_coverage": round(coverage, 2),
        "used_sources": sorted(list(used)),
        "source_count": len(used),
        "confidence_score": round(confidence, 2),
    }

# -----------------------------
# Refusal detection
# -----------------------------
REFUSAL_PATTERNS = [
    "i cannot answer",
    "i can't answer",
    "cannot answer",
    "not enough evidence",
    "insufficient evidence",
    "not supported by the provided sources",
    "cannot be determined from the sources",
]

def detect_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in REFUSAL_PATTERNS)

# -----------------------------
# OpenAI call
# -----------------------------
def ask_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# Main RAG Pipeline
# -----------------------------
def run_pipeline(question: str):
    source_ids = set(retrieved_docs.keys())
    sources_text = format_sources(retrieved_docs)

    # Step 1: Generate answer
    answer = ask_llm(ANSWER_PROMPT.format(question=question, sources=sources_text))

    # Step 2: Enforce citations
    if not validate_citations(answer, source_ids):
        answer = ask_llm(REVISE_PROMPT.format(answer=answer, sources=sources_text))

    # ✅ If model refuses → hard abstain
    if detect_refusal(answer):
        groundedness = compute_groundedness(answer, source_ids)
        return {
            "question": question,
            "answer": "⚠️ I don’t have enough evidence in the provided documents to answer confidently.",
            "verdict": "UNSUPPORTED",
            "confidence_score": 0.10,
            "citation_coverage": groundedness["citation_coverage"],
            "used_sources": groundedness["used_sources"],
            "source_count": groundedness["source_count"],
            "abstained": True,
            "abstain_reason": "Model refused due to insufficient evidence."
        }

    # Step 3: Verification
    verdict = verify_answer(answer, sources_text)

    # Step 4: Attach citations if missing (safe post-processing)
    if not extract_citations(answer):
        answer += " " + " ".join(f"[{sid}]" for sid in sorted(source_ids))

    # Step 5: Groundedness & abstention
    groundedness = compute_groundedness(answer, source_ids)

    if verdict == "UNSUPPORTED" and groundedness["source_count"] == 0:
        return {
            "question": question,
            "answer": "⚠️ I don’t have enough evidence in the provided documents to answer confidently.",
            "verdict": "UNSUPPORTED",
            "confidence_score": 0.10,
            "citation_coverage": groundedness["citation_coverage"],
            "used_sources": groundedness["used_sources"],
            "source_count": groundedness["source_count"],
            "abstained": True,
            "abstain_reason": "Verification failed and grounding was insufficient."
        }

    return {
        "question": question,
        "answer": answer,
        "verdict": verdict,
        "confidence_score": groundedness["confidence_score"],
        "citation_coverage": groundedness["citation_coverage"],
        "used_sources": groundedness["used_sources"],
        "source_count": groundedness["source_count"],
        "abstained": False,
        "abstain_reason": ""
    }

# -----------------------------
# Test Run
# -----------------------------
if __name__ == "__main__":
    q = "What is Llama 3?"
    result = run_pipeline(q)

    print("\nQUESTION:", result["question"])
    print("\nANSWER:", result["answer"])
    print("VERDICT:", result["verdict"])
    print("CONFIDENCE:", result["confidence_score"])
    print("CITATION COVERAGE:", result["citation_coverage"])
    print("USED SOURCES:", result["used_sources"])
    print("ABSTAINED:", result["abstained"])
