"""Collect the interviewer's current question or set of questions from captions."""
import re

QUESTION_START = re.compile(
    r"^\s*(what|what's|whats|why|how|when|where|which|who|whose|whom|"
    r"tell me|explain|describe|walk me through|talk me through|talk about|"
    r"can you|could you|would you|will you|do you|did you|does|have you|has|"
    r"are you|is there|give me|share|show me|suppose|imagine|difference between|"
    r"write|implement|create|develop|build|solve)\b",
    re.IGNORECASE,
)
CODING_REQUEST = re.compile(
    r"\b(write|implement|create|develop|build|solve)\b.{0,140}\b("
    r"program|function|method|script|solution|algorithm|snippet|code)\b",
    re.IGNORECASE,
)
CODING_FOLLOW_UP = re.compile(
    r"\b(explain|describe|walk me through|talk me through)\b.{0,100}\b("
    r"approach|solution|code|program|logic|complexity|time|space)\b|"
    r"\b(time|space) complexity\b|\bcomplexity\b.{0,40}\b(time|space)\b",
    re.IGNORECASE,
)


def _key(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _questions_in_segment(segment: str) -> list[str]:
    # A single transcription can contain several complete questions.
    pieces = re.split(r"(?<=[?])\s+", segment.strip())
    found: list[str] = []
    for piece in pieces:
        cleaned = piece.strip(" \t\r\n.,;:-")
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
        if not cleaned:
            continue
        if "?" in piece or QUESTION_START.search(cleaned):
            found.append(cleaned)
    return found


def detect_question(segments: list[str], lookback: int = 10, exclude: list[str] | None = None) -> dict:
    """Return all distinct questions in recent captions, in the order asked."""
    cleaned = [" ".join(segment.split()) for segment in segments if segment and segment.strip()]
    if not cleaned:
        return {"question": "", "questions": [], "is_question": False, "index": None}

    window = cleaned[-lookback:]
    offset = len(cleaned) - len(window)
    excluded = {_key(part) for item in (exclude or []) for part in _questions_in_segment(item)}
    questions: list[str] = []
    seen = set(excluded)
    last_index = None

    for position, segment in enumerate(window):
        candidates = _questions_in_segment(segment)
        for candidate in candidates:
            key = _key(candidate)
            if key and key not in seen:
                questions.append(candidate)
                seen.add(key)
                last_index = offset + position

    if not questions:
        # Speech recognition sometimes drops punctuation. Use the newest caption
        # that reads like a question, or join the last two pieces of a split turn.
        for position in range(len(window) - 1, -1, -1):
            if QUESTION_START.search(window[position]):
                questions = [window[position]]
                last_index = offset + position
                break
        if not questions:
            questions = [" ".join(window[-2:]) if len(window) > 1 else window[0]]

    bundled = questions[0] if len(questions) == 1 else "\n".join(
        f"{index}. {question}" for index, question in enumerate(questions, 1)
    )
    return {
        "question": bundled,
        "questions": questions,
        "is_question": bool(last_index is not None),
        "index": last_index,
    }


def select_answer_question(questions: list[str]) -> str:
    """Keep a coding task with its immediate explanation/complexity follow-ups."""
    if not questions:
        return ""
    for index, question in enumerate(questions):
        if not CODING_REQUEST.search(question):
            continue
        end = index + 1
        while end < len(questions) and CODING_FOLLOW_UP.search(questions[end]):
            end += 1
        return "\n".join(questions[index:end])
    return questions[-1]
