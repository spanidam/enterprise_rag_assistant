import re

def verify_answer(answer_text, num_sources):
    """
    Checks whether the answer contains valid citations.
    """
    citations = re.findall(r"\[S(\d+)\]", answer_text)

    if not citations:
        return False

    for c in citations:
        if int(c) < 1 or int(c) > num_sources:
            return False

    return True