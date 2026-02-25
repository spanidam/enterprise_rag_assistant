def build_grounded_prompt(question, retrieved_docs):
    """
    Builds a grounded prompt with numbered sources.
    """

    sources = []
    for i, (doc, score) in enumerate(retrieved_docs, start=1):
        sources.append(f"[S{i}] {doc.page_content}")

    sources_text = "\n\n".join(sources)

    prompt = f"""
You are an AI assistant answering questions using ONLY the provided sources.

RULES:
- Use ONLY the information in the sources.
- Cite sources using [S1], [S2], etc.
- If the answer is not supported by the sources, say:
  "The provided documents do not contain enough information to answer this question."

SOURCES:
{sources_text}

QUESTION:
{question}

ANSWER (with citations):
"""

    return prompt