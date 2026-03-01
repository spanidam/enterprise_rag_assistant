from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class HallucinationGrade(BaseModel):
    supported: bool

grader_llm = ChatOpenAI(model="gpt-4o", temperature=0)

def check_hallucination(answer: str, sources: str) -> bool:
    prompt = f"""
You are a strict fact checker.

Answer:
{answer}

Retrieved Sources:
{sources}

Question:
Is the answer fully supported by the sources?

Reply ONLY with true or false.
"""
    result = grader_llm.with_structured_output(HallucinationGrade).invoke(prompt)
    return result.supported