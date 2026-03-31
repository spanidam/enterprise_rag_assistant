ANSWER_PROMPT = """
You are a QA system.

RULES:
- Use ONLY the provided sources.
- Every factual sentence MUST have a citation like [S1].
- If the answer is not fully supported, say:
  "I cannot answer based on the provided sources."

QUESTION:
{question}

SOURCES:
{sources}
"""

VERIFY_PROMPT = """
You are verifying whether an answer is fully supported by sources.

ANSWER:
{answer}

SOURCES:
{sources}

If ALL claims are supported, reply ONLY:
SUPPORTED

Otherwise reply ONLY:
UNSUPPORTED
"""

REVISE_PROMPT = """
Revise the answer using ONLY supported information from the sources.
Remove any unsupported claims.
If nothing remains, say:
"I cannot answer based on the provided sources."

ANSWER:
{answer}

SOURCES:
{sources}
"""

"""

IMPORTANT RULES:
- Every factual sentence MUST include at least one citation in the format [S1], [S2], etc.
- Use ONLY the provided sources.
- If the sources do not clearly support the answer, explicitly say that the information is not available in the documents.
- Do NOT answer without citations.

"""