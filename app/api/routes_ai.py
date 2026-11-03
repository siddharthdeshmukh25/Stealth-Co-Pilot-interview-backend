from fastapi import APIRouter, HTTPException
from groq import Groq

from ..config import get_groq_api_key, settings

router = APIRouter(prefix="/api/ai", tags=["ai"])
NON_CHAT_MODEL_HINTS = ("whisper", "transcribe", "speech", "tts", "guard", "embed", "moderation", "orpheus")


@router.get("/models")
def list_answer_models():
    """Fetch active text-generation models directly from the configured Groq account."""
    api_key = get_groq_api_key()
    if not api_key:
        raise HTTPException(503, "GROQ_API_KEY is not configured on the backend.")
    try:
        response = Groq(api_key=api_key, timeout=15.0, max_retries=1).models.list()
    except Exception as exc:  # noqa: BLE001 - pass a useful provider error to settings UI
        raise HTTPException(502, f"Could not load Groq models: {str(exc)[:240]}") from exc

    models = []
    for model in response.data:
        model_id = str(getattr(model, "id", "") or "").strip()
        if not model_id or getattr(model, "active", True) is False:
            continue
        if any(hint in model_id.lower() for hint in NON_CHAT_MODEL_HINTS):
            continue
        models.append({
            "id": model_id,
            "owned_by": str(getattr(model, "owned_by", "") or "Groq"),
            "context_window": getattr(model, "context_window", None),
        })
    models.sort(key=lambda item: item["id"].lower())
    available_ids = {model["id"] for model in models}
    default_model = settings.answer_model if settings.answer_model in available_ids else (models[0]["id"] if models else "")
    return {"models": models, "default_model": default_model}
