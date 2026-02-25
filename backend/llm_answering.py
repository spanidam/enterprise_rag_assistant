from openai import OpenAI
from prompt_builder import build_grounded_prompt

client = OpenAI()

def generate_answer(question, retrieved_docs):
    prompt = build_grounded_prompt(question, retrieved_docs)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-4 / gpt-3.5-turbo
        messages=[
            {"role": "system", "content": "You are a grounded RAG assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0  # IMPORTANT: reduces hallucinations
    )

    return response.choices[0].message.content.strip()