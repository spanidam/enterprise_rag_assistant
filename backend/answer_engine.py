from orchestrator import retrieve_context
from llm_answering import generate_answer
from verification import verify_answer

def answer_question(question, top_k=5):
    retrieved = retrieve_context(question, top_n=top_k)

    answer = generate_answer(question, retrieved)

    is_valid = verify_answer(answer, len(retrieved))

    if not is_valid:
        answer = (
            "The retrieved documents do not provide sufficient evidence "
            "to answer this question reliably."
        )

    return answer