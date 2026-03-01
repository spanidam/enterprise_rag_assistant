import os
from openai import OpenAI
from dotenv import load_dotenv

from backend.prompt_builder import build_grounded_prompt

load_dotenv()

# Optional safety check
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found. Put it in your .env file.")

client = OpenAI()

def generate_answer(question, retrieved_docs, model="gpt-4o"):
    
    """
    Returns:
       answer_text: str
       usage: dict (prompt_tokens, completion_tokens, total_tokens)
    """

    prompt = build_grounded_prompt(question, retrieved_docs)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a grounded RAG assistant. Use only the provided sources and cite them."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    answer_text = response.choices[0].message.content.strip()

    usage = {}
    if getattr(response, "usage", None):
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    return answer_text, usage