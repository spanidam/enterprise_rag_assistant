from backend.orchestrator import retrieve_context
from backend.llm_answering import generate_answer
from backend.verification import verify_answer
from backend.logging_config import logger

# OPTIONAL: If you also want hallucination verification here (instead of api.py)
# from backend.hallucination_checker import check_hallucination

def answer_question(question, top_k=5, model="gpt-4o"):
    retrieved = retrieve_context(question, top_n=top_k)

    # generate answer + token usage
    answer, usage = generate_answer(question, retrieved, model=model)

    # ✅ Log LLM usage (Phase 4.2 complete)
    if usage:
        logger.info(
            "llm_usage",
            model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    # Verification: citations exist / basic evidence policy
    is_valid = verify_answer(answer, len(retrieved))

    if not is_valid:
        answer = (
            "The retrieved documents do not provide sufficient evidence "
            "to answer this question reliably."
        )

    return answer