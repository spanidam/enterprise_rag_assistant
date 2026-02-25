import re

def extract_citations(text):
    return re.findall(r"\[S\d+\]", text)

def validate_citations(answer, source_ids):
    sentences = answer.split(".")
    for s in sentences:
        if s.strip() == "":
            continue
        cites = extract_citations(s)
        if not cites:
            return False
        for c in cites:
            if c[1:-1] not in source_ids:
                return False
    return True