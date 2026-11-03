from groq import APIConnectionError, APIStatusError, AuthenticationError, BadRequestError, Groq, RateLimitError
from ..config import get_groq_api_key
import io
import wave


def transcribe_audio(wav_bytes: bytes) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "Groq is not configured. Set GROQ_API_KEY in backend/.env, "
            "%LOCALAPPDATA%\\StealthInterviewCoPilot\\.env, or the process environment."
        )

    if not wav_bytes or not wav_bytes.startswith(b"RIFF") or b"WAVE" not in wav_bytes[:16]:
        raise ValueError("Desktop sent an invalid WAV recording. Restart speaker capture and try again.")
    if len(wav_bytes) > 25 * 1024 * 1024:
        raise ValueError("Recording exceeds Groq's 25 MB upload limit. Wait for a short pause in speech and retry.")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as recording:
            if recording.getframerate() <= 0 or recording.getnframes() / recording.getframerate() < 0.01:
                raise ValueError("Recording is too short for transcription. Speak for at least a moment and retry.")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Desktop sent a damaged WAV recording. Restart speaker capture and try again.") from exc

    client = Groq(api_key=api_key, timeout=35.0, max_retries=2)
    try:
        result = client.audio.transcriptions.create(
            file=("interview-audio.wav", wav_bytes, "audio/wav"),
            model="whisper-large-v3-turbo",
            response_format="json",
            temperature=0.0,
            language="en",
        )
    except AuthenticationError as exc:
        raise RuntimeError(
            "Groq rejected the API key (401). Update GROQ_API_KEY in backend/.env "
            "or %LOCALAPPDATA%\\StealthInterviewCoPilot\\.env, then restart the desktop app."
        ) from exc
    except RateLimitError as exc:
        raise RuntimeError(
            "Groq rate limit or quota reached (429). Check your Groq account limits and try again."
        ) from exc
    except APIConnectionError as exc:
        raise RuntimeError(
            "Cannot reach Groq for transcription. Check the internet connection and access to api.groq.com."
        ) from exc
    except BadRequestError as exc:
        detail = getattr(exc, "message", "The WAV recording was rejected.")
        raise RuntimeError(f"Groq could not process this recording (400): {detail}") from exc
    except APIStatusError as exc:
        raise RuntimeError(f"Groq transcription failed (HTTP {exc.status_code}). Please retry in a moment.") from exc

    return (result.text or "").strip()

