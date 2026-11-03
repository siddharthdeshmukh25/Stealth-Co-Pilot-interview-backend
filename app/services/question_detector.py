"""Pick the question the interviewer actually just asked out of the caption stream."""
import re

QUESTION_START = re.compile(
    r"^\s*(what|what's|whats|why|how|when|where|which|who|whose|whom|"
    r"tell me|explain|describe|walk me through|talk me through|talk about|"
    r"can you|could you|would you|will you|do you|did you|does|have you|has|"
    r"are you|is there|give me|share|show me|suppose|imagine|difference between)\b",
    re.IGNORECASE,
)
INTERROGATIVE = re.compile(
    r"\b(what|why|how|when|where|which|who|tell me|explain|describe|"
    r"walk me through|can you|could you|would you|do you|did you|have you|"
    r"are you|is there|give me|share|difference between|talk about)\b",
    re.IGNORECASE,
)


def detect_question(segments: list[str], lookback: int = 8) -> dict:
    """Return {"question", "is_question", "index"} for the newest captions.

    Priority: newest segment with a question mark, then newest segment that reads
    like a question, then the last one or two segments (a question split by a pause).
    """
    cleaned = [segment.strip() for segment in segments if segment and segment.strip()]
    if not cleaned:
        return {"question": "", "is_question": False, "index": None}

    window = cleaned[-lookback:]
    offset = len(cleaned) - len(window)

    for position in range(len(window) - 1, -1, -1):
        if "?" in window[position]:
            return {"question": window[position], "is_question": True, "index": offset + position}

    for position in range(len(window) - 1, -1, -1):
        if QUESTION_START.search(window[position]) or INTERROGATIVE.search(window[position]):
            return {"question": window[position], "is_question": True, "index": offset + position}

    question = " ".join(window[-2:]) if len(window) > 1 else window[0]
    return {"question": question, "is_question": False, "index": None}
