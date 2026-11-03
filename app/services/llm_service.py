import json
import re
from collections.abc import Iterator
from typing import Any

from groq import Groq

from ..config import get_groq_api_key, settings
from .pdf_parser import _resume_identity

MAX_ANSWER_TOKENS = {"standard": 900, "explain_deeper": 1400, "code_example": 1200}

INSTRUCTIONS = """You are a live interview answer coach. The candidate is in a real interview right now and reads your answer on a phone.

Answer ONLY the exact interviewer question at the very end of this message. Other questions in the transcript are context only; do not answer or mix them into this response.

Style:
- Use simple, everyday words that are easy to say aloud. Speak in first person, naturally and confidently.
- Give a useful answer in about 3-5 short sentences (roughly 50-90 words). Stay focused on this question alone.
- Answer the point directly first, then add one relevant example or detail from the candidate's real experience. Avoid vague one-line answers and unnecessary background.
- Use plain sentences: no headings, markdown, emoji, or stage directions.
- Never summarise the whole resume and never list all skills unless that is exactly what was asked.
- Ground every claim in the resume or job description below. Never invent employers, dates, metrics, skills, or preferences.
- If you use the candidate's name, copy it exactly from the authoritative resume identity below. Never guess, alter, or substitute a name. If no name is available, omit it and begin with relevant experience. Do not mention an email address unless asked.
- If the question asks for a personal fact the resume does not contain, say briefly that you do not know it and give a short honest sentence the candidate can finish. Do not dodge it or switch topic.
- If the question asks about experience the resume does not support, be candid in one sentence and offer the closest true experience.
- For "tell me about yourself", give 2-3 sentences built from the most relevant resume facts.
- For a technical question, answer the question first, then one line connecting it to the candidate's real work."""

MODE_INSTRUCTIONS = {
    "explain_deeper": "The candidate tapped Explain deeper: add one concrete detail or trade-off, but stay under 110 words.",
    "code_example": "The candidate tapped Code example: always return a short, correct code snippet in a fenced block (markdown triple backticks) with the language name, then one sentence explaining it. Keep the snippet under 25 lines. If the candidate supplied a topic below, write the snippet for that topic.",
}


def _resume_block(context: dict, question: str) -> str:
    structured: dict[str, Any] = {}
    try:
        structured = json.loads(context.get("structured_json") or "{}")
    except (TypeError, ValueError):
        structured = {}
    raw_text = str(context.get("resume_text") or "").strip()
    literal_name, literal_email = _resume_identity(raw_text)
    identity = {
        "name": literal_name,
        "email": literal_email,
    }
    question_lower = question.lower()
    categories: set[str] = set()
    if any(term in question_lower for term in ("introduce yourself", "tell me about yourself", "background")):
        categories.update(("education", "core_skills", "projects", "experience"))
    if any(term in question_lower for term in ("education", "degree", "college", "university", "academic")):
        categories.add("education")
    if any(term in question_lower for term in ("skill", "technical", "technology", "tech stack")):
        categories.add("core_skills")
    if any(term in question_lower for term in ("project", "built", "developed", "worked on")):
        categories.add("projects")
    if any(term in question_lower for term in ("experience", "work history", "company", "role")):
        categories.add("experience")
    if not categories:
        categories.update(("core_skills", "projects", "experience", "education"))

    selected = {"name": identity["name"], "email": identity["email"]}
    for category in categories:
        value = structured.get(category)
        if value:
            selected[category] = value

    parts = [
        "Authoritative resume identity (copy the name exactly; email is private unless asked):",
        json.dumps(identity, ensure_ascii=False),
        "Question-relevant structured resume facts (retrieved from the uploaded resume):",
        json.dumps(selected, ensure_ascii=False)[:7000],
    ]

    if raw_text:
        question_terms = {
            term for term in re.findall(r"[a-z0-9+#.]+", question_lower)
            if len(term) > 2 and term not in {"the", "and", "for", "you", "your", "with", "about", "could", "please", "briefly"}
        }
        category_terms = {
            "education": {"education", "degree", "college", "university", "school", "bachelor", "master"},
            "core_skills": {"skill", "skills", "technical", "technology", "tools", "language", "framework"},
            "projects": {"project", "projects", "built", "developed", "application", "app"},
            "experience": {"experience", "company", "role", "work", "employment"},
        }
        retrieval_terms = question_terms.union(*(category_terms[key] for key in categories))
        lines = [" ".join(line.split()) for line in raw_text.splitlines() if line.strip()]
        scored: list[tuple[int, int, str]] = []
        for index, line in enumerate(lines):
            line_terms = set(re.findall(r"[a-z0-9+#.]+", line.lower()))
            score = len(question_terms & line_terms) * 3 + len(retrieval_terms & line_terms)
            if index < 6:
                score += 1
            if score:
                scored.append((score, index, line))
        selected_lines = [line for _, _, line in sorted(scored, key=lambda item: (-item[0], item[1]))[:24]]
        if selected_lines:
            parts.append("Relevant resume text retrieved for this question:\n" + "\n".join(selected_lines)[:5000])
    return "\n".join(parts)


def build_prompt(question: str, context: dict, action: str, history: list[str] | None = None, topic: str = "") -> str:
    job_description = str(context.get("job_description") or "").strip()[:2500]
    transcript = "\n".join(history[-40:])[:9000] if history else ""
    sections = [
        INSTRUCTIONS,
        MODE_INSTRUCTIONS.get(action, ""),
        f"Candidate role: {context.get('job_title') or 'the role being interviewed for'}",
        f"Job description:\n{job_description}",
        _resume_block(context, question),
    ]
    if transcript:
        sections.append(
            "Complete interviewer transcript part captured since the last Ask AI click "
            f"(oldest first; use all of it as context):\n{transcript}"
        )
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
