import json
from collections.abc import Iterator
from typing import Any

from groq import Groq

from ..config import get_groq_api_key, settings

MAX_ANSWER_TOKENS = {"standard": 900, "explain_deeper": 1400, "code_example": 1200}

INSTRUCTIONS = """You are a live interview answer coach. The candidate is in a real interview right now and reads your answer on a phone.

Answer every new interviewer question listed at the very end of this message, in the same order. Never answer an earlier question from the transcript.

Style:
- Use simple, everyday words that are easy to say aloud. Speak in first person, naturally and confidently.
- For one question, give a useful answer in about 3-5 short sentences (roughly 50-90 words). For a numbered bundle, answer every item in order, with 2-3 short sentences for each item and a separate paragraph per answer.
- Answer the point directly first, then add one relevant example or detail from the candidate's real experience. Avoid vague one-line answers and unnecessary background.
- Use plain sentences: no headings, markdown, emoji, or stage directions. Keep the separate paragraphs for bundled answers.
- Never summarise the whole resume and never list all skills unless that is exactly what was asked.
- Ground every claim in the resume or job description below. Never invent employers, dates, metrics, skills, or preferences.
- If the question asks for a personal fact the resume does not contain, say briefly that you do not know it and give a short honest sentence the candidate can finish. Do not dodge it or switch topic.
- If the question asks about experience the resume does not support, be candid in one sentence and offer the closest true experience.
- For "tell me about yourself", give 2-3 sentences built from the most relevant resume facts.
- For a technical question, answer the question first, then one line connecting it to the candidate's real work."""

MODE_INSTRUCTIONS = {
    "explain_deeper": "The candidate tapped Explain deeper: add one concrete detail or trade-off, but stay under 110 words.",
    "code_example": "The candidate tapped Code example: always return a short, correct code snippet in a fenced block (markdown triple backticks) with the language name, then one sentence explaining it. Keep the snippet under 25 lines. If the candidate supplied a topic below, write the snippet for that topic.",
}


def _resume_block(context: dict) -> str:
    structured: dict[str, Any] = {}
    try:
        structured = json.loads(context.get("structured_json") or "{}")
    except (TypeError, ValueError):
        structured = {}
    public = {key: value for key, value in structured.items() if not str(key).startswith("_") and key != "note"}
    parts = ["Resume facts (JSON):", json.dumps(public, ensure_ascii=False)[:6000]]
    raw_text = str(context.get("resume_text") or "").strip()
    if raw_text:
        parts.extend(("Resume text, verbatim (may be truncated):", raw_text[:6000]))
    return "\n".join(parts)


def build_prompt(question: str, context: dict, action: str, history: list[str] | None = None, topic: str = "") -> str:
    job_description = str(context.get("job_description") or "").strip()[:2500]
    transcript = "\n".join(history[-6:]) if history else ""
    sections = [
        INSTRUCTIONS,
        MODE_INSTRUCTIONS.get(action, ""),
        f"Candidate role: {context.get('job_title') or 'the role being interviewed for'}",
        f"Job description:\n{job_description}",
        _resume_block(context),
    ]
    if transcript:
        sections.append(f"Earlier interviewer transcript, context only (oldest first):\n{transcript}")
    if topic.strip():
        sections.append(f"The candidate asked for a code example of:\n{topic.strip()[:200]}")
    sections.append(f"Interviewer question to answer now:\n{question}")
    return "\n\n".join(section for section in sections if section.strip())


def stream_answer(question: str, context: dict, action: str = "standard", history: list[str] | None = None, topic: str = ""):
    """Return (token_iterator, status). status['finish_reason'] is filled while consuming the iterator."""
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for answer generation.")
    client = Groq(api_key=api_key, timeout=60.0, max_retries=1)
    status: dict[str, Any] = {"finish_reason": None, "model": settings.answer_model}

    request: dict[str, Any] = {
        "model": settings.answer_model,
        # Groq recommends a single user message for GPT-OSS reasoning models.
        "messages": [{"role": "user", "content": build_prompt(question, context, action, history, topic)}],
        "temperature": settings.answer_temperature,
        "top_p": 0.95,
        "max_completion_tokens": MAX_ANSWER_TOKENS.get(action, MAX_ANSWER_TOKENS["standard"]),
        "stream": True,
    }
    if settings.is_reasoning_model:
        # Low reasoning keeps answers fast; the token budget above still leaves room for them.
        request["reasoning_effort"] = "low"
        request["include_reasoning"] = False

    completion = client.chat.completions.create(**request)

    def tokens() -> Iterator[str]:
        for chunk in completion:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                status["finish_reason"] = choice.finish_reason
            content = choice.delta.content if choice.delta else None
            if content:
                yield content

    return tokens(), status
