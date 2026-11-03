import json
import logging
import re
from typing import Any

import fitz
from groq import Groq

from ..config import get_groq_api_key, settings

logger = logging.getLogger(__name__)

RESUME_KEYS = ("name", "core_skills", "projects", "experience")

EXTRACTION_INSTRUCTIONS = """You convert a resume into strict JSON. Return only a JSON object, no prose and no markdown fences.

Use exactly these keys:
{
  "name": "candidate full name",
  "core_skills": ["skill", "skill"],
  "projects": [{"name": "project", "tech_stack": ["technology"], "impact_and_metrics": "one short line"}],
  "experience": [{"company": "company", "role": "title", "duration": "dates", "key_achievements": ["achievement"]}]
}

Rules:
- Copy facts only from the resume text. Never invent a skill, project, employer, date, or metric.
- Use empty arrays when the resume genuinely does not contain that section.
- Keep every list item under 200 characters.
- Output must be valid JSON that a json.loads() call can parse."""


class ResumeParseError(ValueError):
    """A PDF problem the user can fix; the message is shown as-is in the desktop app."""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item.strip()[:200] for item in value if isinstance(item, str) and item.strip()][:30]


def _normalize(structured: dict) -> dict:
    name = structured.get("name")
    normalized = {
        "name": (name.strip()[:120] if isinstance(name, str) and name.strip() else "Candidate"),
        "core_skills": _as_text_list(structured.get("core_skills")),
        "projects": [],
        "experience": [],
    }
    for project in structured.get("projects") or []:
        if not isinstance(project, dict):
            continue
        normalized["projects"].append({
            "name": str(project.get("name") or "").strip()[:160],
            "tech_stack": _as_text_list(project.get("tech_stack")),
            "impact_and_metrics": str(project.get("impact_and_metrics") or "").strip()[:300],
        })
    for job in structured.get("experience") or []:
        if not isinstance(job, dict):
            continue
        normalized["experience"].append({
            "company": str(job.get("company") or "").strip()[:160],
            "role": str(job.get("role") or "").strip()[:160],
            "duration": str(job.get("duration") or "").strip()[:80],
            "key_achievements": _as_text_list(job.get("key_achievements")),
        })
    return normalized


def _has_resume_detail(structured: dict) -> bool:
    return bool(structured["core_skills"] or structured["projects"] or structured["experience"])


def _extract_text(pdf_bytes: bytes) -> str:
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ResumeParseError(
            "This file is not a readable PDF. Export your resume as a PDF and upload it again."
        ) from exc
    try:
        if document.needs_pass:
            raise ResumeParseError(
                "This PDF is password-protected. Remove the password, save it again, and upload it."
            )
        raw_text = "\n".join(page.get_text("text") for page in document).strip()
    finally:
        document.close()
    if len(re.sub(r"\s+", "", raw_text)) < 40:
        raise ResumeParseError(
            "This resume has no selectable text (a scanned or image-only PDF). "
            "Export a text-based PDF from Word, Google Docs, or Naukri and upload that file."
        )
    return raw_text


def _basic_resume_structure(raw_text: str, note: str) -> dict:
    name = next((line.strip() for line in raw_text.splitlines() if len(line.strip()) > 2), "Candidate")
    return {"name": name[:120], "core_skills": [], "projects": [], "experience": [], "note": note}


def _request_extraction(raw_text: str, api_key: str) -> dict:
    client = Groq(api_key=api_key, timeout=60.0, max_retries=1)
    request: dict[str, Any] = {
        "model": settings.resume_model,
        # Groq recommends keeping all instructions in the user message for GPT-OSS reasoning models.
        "messages": [
            {
                "role": "user",
                "content": f"{EXTRACTION_INSTRUCTIONS}\n\nResume text:\n\"\"\"\n{raw_text[:30000]}\n\"\"\"",
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.6,
        "max_completion_tokens": 4096,
    }
    if "gpt-oss" in settings.resume_model.lower():
        request["include_reasoning"] = False
    response = client.chat.completions.create(**request)
    content = _strip_code_fences(response.choices[0].message.content or "")
    if not content:
        raise ValueError("Groq returned an empty resume extraction.")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Groq returned resume JSON that is not an object.")
    structured = _normalize(parsed)
    if not _has_resume_detail(structured):
        raise ValueError("Groq returned a resume JSON without skills, projects, or experience.")
    return structured


def parse_resume(pdf_bytes: bytes) -> tuple[str, dict, dict]:
    """Return (raw_text, structured, extraction_info). Text is always kept, AI detail is best effort."""
    raw_text = _extract_text(pdf_bytes)
    api_key = get_groq_api_key()

    if not api_key:
        structured = _basic_resume_structure(
            raw_text, "GROQ_API_KEY is missing; only the resume text was saved."
        )
        info = {
            "mode": "text_only",
            "warning": (
                "GROQ_API_KEY is missing, so the resume was not structured with AI. "
                "The saved resume text is still used for answers. Add the key to backend/.env and restart."
            ),
        }
        structured["_extraction"] = info
        return raw_text, structured, info

    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            structured = _request_extraction(raw_text, api_key)
            info = {"mode": "ai", "warning": None}
            structured["_extraction"] = info
            return raw_text, structured, info
        except Exception as exc:  # noqa: BLE001 - reported to the user and logged below
            last_error = exc
            logger.warning("Groq resume extraction attempt %s failed: %s: %s", attempt, type(exc).__name__, exc)

    reason = f"{type(last_error).__name__}: {str(last_error)[:220]}" if last_error else "unknown error"
    structured = _basic_resume_structure(
        raw_text, "AI extraction failed; the resume text was saved and is still used for answers."
    )
    info = {
        "mode": "failed",
        "warning": (
            f"AI resume extraction failed ({reason}). The full resume text was saved and is still "
            "used to answer questions, but skills/projects were not split out."
        ),
    }
    structured["_extraction"] = info
    return raw_text, structured, info
