"""Interview-day self-check.

Run from the backend folder:  .venv\\Scripts\\python.exe preflight.py
Add --quick to skip paid model calls.
"""
from __future__ import annotations

import argparse
import io
import os
import socket
import subprocess
import sys
import wave
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

RESULTS: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    line = f"[{status.upper():4}] {name}"
    print(f"{line} — {detail}" if detail else line)


def run(name: str, func) -> None:
    try:
        status, detail = func()
        record(status, name, str(detail or ""))
    except Exception as exc:  # noqa: BLE001 - a check must never abort the report
        record("fail", name, f"{type(exc).__name__}: {str(exc)[:220]}")


def _sample_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 8000)
    return buffer.getvalue()


def _sample_pdf() -> bytes:
    import fitz

    document = fitz.open()
    page = document.new_page()
    lines = [
        "Priya Sharma",
        "Backend Engineer",
        "Skills: Python, FastAPI, PostgreSQL, Docker, AWS",
        "Experience:",
        "Acme Corp - Backend Engineer - 2022 to 2025",
        "- Built FastAPI services handling two million requests per day",
        "- Cut p95 latency by 40 percent with Redis caching",
        "Projects:",
        "Interview Copilot - Python, FastAPI, WebSockets",
    ]
    position = 72
    for line in lines:
        page.insert_text((72, position), line, fontsize=11)
        position += 16
    data = document.tobytes()
    document.close()
    return data


def _firewall_rule_present(ports: tuple[str, ...] = ("8000", "3000")) -> bool | None:
    if os.name != "nt":
        return None
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        output = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule", "name=all", "dir=in"],
            capture_output=True, text=True, timeout=10, creationflags=flags,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if not output:
        return None
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if "LocalPort:" not in line or not any(port in line for port in ports):
            continue
        window = "\n".join(lines[max(0, index - 6):index])
        if "Allow" in window:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the local interview co-pilot setup.")
    parser.add_argument("--quick", action="store_true", help="skip paid model calls")
    args = parser.parse_args()

    from app.config import ENV_FILE, get_groq_api_key, get_openai_api_key, settings

    def python_version():
        version = ".".join(str(part) for part in sys.version_info[:3])
        return ("ok" if sys.version_info >= (3, 11) else "fail", version)

    def api_key():
        if get_groq_api_key():
            return ("ok", "found")
        return ("fail", f"missing; add GROQ_API_KEY to {ENV_FILE}")

    def openai_api_key():
        if get_openai_api_key():
            return ("ok", "found")
        return ("fail", f"missing; add OPENAI_API_KEY to {ENV_FILE}")

    def database():
        from sqlmodel import Session, select

        from app.database import engine, init_db
        from app.models import Candidate

        init_db()
        with Session(engine) as db:
            db.exec(select(Candidate).limit(1)).first()
        path = settings.database_path
        if path and not os.access(path.parent, os.W_OK):
            return ("fail", f"{path.parent} is not writable")
        return ("ok", str(path or settings.database_url))

    def network():
        from app.network import lan_urls

        urls = lan_urls(8000)
        if not urls:
            return ("fail", "no private LAN address; the phone cannot connect. Join Wi-Fi or Ethernet.")
        return ("ok", ", ".join(urls))

    def firewall():
        state = _firewall_rule_present()
        if state is None:
            return ("warn", "could not inspect Windows Firewall rules")
        if state:
            return ("ok", "inbound rule for 8000/3000 found")
        return (
            "warn",
            "no inbound rule for 8000/3000. Run scripts\\allow_lan_firewall.bat as administrator, "
            "otherwise the phone shows 'Reconnecting'.",
        )

    def backend_running():
        try:
            with urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
                import json

                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, ValueError):
            return ("warn", "backend is not running on port 8000 (start it, then rerun preflight)")
        if not payload.get("ai_configured"):
            return ("fail", "backend is running but reports ai_configured=false")
        return ("ok", f"health ok; LAN URLs: {', '.join(payload.get('lan_urls') or []) or 'none'}")

    key = get_groq_api_key()
    client = None
    models: set[str] = set()

    def configured_models():
        nonlocal client, models
        if not key:
            return ("skip", "no API key")
        from groq import Groq

        client = Groq(api_key=key, timeout=40.0, max_retries=1)
        models = {model.id for model in client.models.list().data}
        wanted = {settings.resume_model, "whisper-large-v3-turbo"}
        missing = sorted(wanted - models)
        if missing:
            return ("fail", f"not available on Groq right now: {', '.join(missing)}")
        return ("ok", f"{settings.resume_model}, whisper-large-v3-turbo")

    def answer_generation():
        if not get_openai_api_key():
            return ("skip", "no API key")
        from app.services.llm_service import stream_answer

        context = {
            "job_title": "Backend Engineer",
            "job_description": "Build Python APIs and work with Postgres.",
            "structured_json": '{"name": "Priya Sharma", "core_skills": ["Python", "FastAPI"]}',
            "resume_text": "Priya Sharma - Backend Engineer - Python, FastAPI, PostgreSQL",
        }
        tokens, status = stream_answer("What is a Python decorator?", context, "standard", [])
        text = "".join(tokens).strip()
        finish = status.get("finish_reason")
        if not text:
            return ("fail", f"OpenAI returned an empty answer (finish_reason={finish})")
        return ("ok", f"{len(text.split())} words, finish_reason={finish}")

    def transcription():
        if not key:
            return ("skip", "no API key")
        from app.services.stt_service import transcribe_audio

        text = transcribe_audio(_sample_wav())
        return ("ok", f"whisper responded (text: {text[:30]!r})" if text else "whisper responded (silence, empty text)")

    def resume_parsing():
        if not key:
            return ("skip", "no API key")
        from app.services.pdf_parser import parse_resume

        _raw, structured, info = parse_resume(_sample_pdf())
        if info.get("mode") != "ai":
            return ("fail", str(info.get("warning") or info.get("mode")))
        return (
            "ok",
            f"extracted name={structured.get('name')!r}, "
            f"{len(structured.get('core_skills') or [])} skills, "
            f"{len(structured.get('experience') or [])} roles",
        )

    print("\nStealth Interview Co-Pilot — preflight check\n" + "-" * 52)
    run("Python version", python_version)
    run("GROQ_API_KEY", api_key)
    run("OPENAI_API_KEY", openai_api_key)
    run("Local database", database)
    run("LAN addresses", network)
    run("Windows Firewall", firewall)
    run("Backend health", backend_running)
    run("Groq models", configured_models)
    if args.quick:
        record("skip", "OpenAI answer call", "--quick")
        record("skip", "Groq transcription", "--quick")
        record("skip", "Resume extraction", "--quick")
    else:
        run("OpenAI answer call", answer_generation)
        run("Groq transcription", transcription)
        run("Resume extraction", resume_parsing)

    failures = [name for status, name, _ in RESULTS if status == "fail"]
    warnings = [name for status, name, _ in RESULTS if status == "warn"]
    print("-" * 52)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("Ready for the interview." + (f" Check warnings: {', '.join(warnings)}" if warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
