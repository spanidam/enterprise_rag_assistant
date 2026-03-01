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