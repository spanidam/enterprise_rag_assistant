import re

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")

REFUSAL_PATTERNS = [
    "i cannot answer",
    "i can't answer",
    "cannot answer",
    "not enough evidence",
    "insufficient evidence",
    "not supported by the provided sources",
    "cannot be determined from the sources",
]

def verify_answer(answer_text: str, sources_text: str):
    """
    Returns "SUPPORTED" if:
    - Answer is not a refusal
    - Answer contains citations [S#]
    - All cited sources exist in sources_text
    Otherwise returns "UNSUPPORTED".
    """

    # 0) Refusal answers are unsupported by definition
    t = (answer_text or "").lower()
    if any(p in t for p in REFUSAL_PATTERNS):
        return "UNSUPPORTED"

    # 1) Valid source IDs from retrieved context
    valid_source_ids = set(CITATION_PATTERN.findall(sources_text or ""))
    if not valid_source_ids:
        return "UNSUPPORTED"

    # 2) Citations used in answer
    cited = CITATION_PATTERN.findall(answer_text or "")
    if not cited:
        return "UNSUPPORTED"

    # 3) All citations must be valid
    for c in cited:
        if c not in valid_source_ids:
            return "UNSUPPORTED"

    return "SUPPORTED"