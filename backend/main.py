import openai
from prompts import ANSWER_PROMPT, REVISE_PROMPT
from utils import validate_citations
from verification import verify_answer

openai.api_key = "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"

# ✅ SIMULATED RETRIEVAL OUTPUT
retrieved_docs = {
    "S1": "Llama 3 is an open-source large language model developed by Meta.",
    "S2": "vLLM is a high-throughput inference engine for serving LLMs.",
    "S3": "Grounding LLMs with retrieved documents reduces hallucinations."
}

def format_sources(docs):
    return "\n".join([f"[{k}] {v}" for k, v in docs.items()])

def ask_llm(prompt): 
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def run_pipeline(question):
    sources_text = format_sources(retrieved_docs)
    source_ids = set(retrieved_docs.keys())

    # Step 1: Initial Answer
    answer = ask_llm(
        ANSWER_PROMPT.format(
            question=question,
            sources=sources_text
        )
    )

    # Step 2: Rule-based citation check
    if not validate_citations(answer, source_ids):
        answer = ask_llm(
            REVISE_PROMPT.format(
                answer=answer,
                sources=sources_text
            )
        )

    # Step 3: LLM Verification
    verdict = verify_answer(answer, sources_text)

    if verdict == "UNSUPPORTED":
        answer = ask_llm(
            REVISE_PROMPT.format(
                answer=answer,
                sources=sources_text
            )
        )

    return answer

if __name__ == "__main__":
    question = "What is Llama 3 and how does vLLM help serve it?"
    final_answer = run_pipeline(question)
    print("\nFINAL ANSWER:\n")
    print(final_answer)