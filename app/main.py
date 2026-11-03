import asyncio
import base64
import binascii
import logging
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from .api.routes_resume import router as resume_router
from .api.routes_session import router as session_router
from .config import get_groq_api_key, settings
from .database import engine, init_db
from .models import QALog
from .network import lan_urls
from .services.llm_service import stream_answer
from .services.question_detector import detect_question
from .services.stt_service import transcribe_audio
from .state_manager import state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("copilot")
app = FastAPI(title="Stealth Interview Co-Pilot", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(resume_router)
app.include_router(session_router)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health():
    database_path = settings.database_path
    return {
        "status": "ok",
        "ai_configured": bool(get_groq_api_key()),
        "bind_host": settings.host,
        "port": settings.port,
        "lan_urls": lan_urls(settings.port),
        "database_path": str(database_path) if database_path else settings.database_url,
        "answer_model": settings.answer_model,
        "resume_model": settings.resume_model,
        "lan_cors": True,
    }


def _ai_error_message(exc: Exception) -> str:
    if not get_groq_api_key():
        return (
            "Groq API key is missing. Configure GROQ_API_KEY in backend/.env or "
            "%LOCALAPPDATA%\\StealthInterviewCoPilot\\.env, then restart the desktop app."
        )
    return str(exc)[:400] or "Groq could not process the request. Check the local desktop log for details."


def _save_answer(session_id: str, question: str, answer: str, action: str) -> int | None:
    with Session(engine) as db:
        entry = QALog(session_id=session_id, interviewer_question=question, ai_answer=answer, action_type=action)
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry.id


async def answer_question(session_id: str, question: str, action: str = "standard", history: list[str] | None = None, topic: str = "", batch: str = "") -> None:
    context = await state.context_for(session_id)
    if not context:
        await state.broadcast(session_id, {
            "event": "error",
            "message": "This interview room is not in the local database. Create a new room from the desktop app.",
        })
        await state.broadcast(session_id, {"event": "status", "state": "listening"})
        return
    question = question.strip()
    if not question:
        return

    generation, stop = state.begin_answer(session_id)
    await state.broadcast(session_id, {"event": "answer_started", "text": question, "batch": batch})
    await state.broadcast(session_id, {"event": "status", "state": "thinking"})

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def pump() -> None:
        """Run the blocking Groq stream off the event loop and feed tokens into the queue."""
        try:
            tokens, status = stream_answer(question, context, action, history, topic)
            for token in tokens:
                if stop.is_set():
                    return
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", status))
        except Exception as exc:  # noqa: BLE001 - reported to the socket below
            if not stop.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

    threading.Thread(target=pump, name="copilot-answer", daemon=True).start()

    parts: list[str] = []
    finish_reason = None
    try:
        await state.broadcast(session_id, {"event": "status", "state": "streaming"})
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=95)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("The AI answer timed out. Check the internet connection and tap Ask AI again.") from exc
            if not state.is_current(session_id, generation):
                return
            if kind == "token":
                parts.append(payload)
                state.append_answer(session_id, payload)
                await state.broadcast(session_id, {"event": "answer_token", "token": payload})
            elif kind == "done":
                finish_reason = (payload or {}).get("finish_reason")
                break
            else:
                raise payload

        answer = "".join(parts)
        qa_id = await asyncio.to_thread(_save_answer, session_id, question, answer, action) if answer else None
        await state.broadcast(session_id, {
            "event": "answer_done",
            "qa_id": qa_id,
            "status": "truncated" if finish_reason == "length" else "ok",
            "finish_reason": finish_reason,
        })
    except Exception as exc:  # noqa: BLE001 - surfaced to both devices
        logger.exception("Answer generation failed")
        if state.is_current(session_id, generation):
            await state.broadcast(session_id, {"event": "error", "message": _ai_error_message(exc)})
            await state.broadcast(session_id, {"event": "answer_done", "qa_id": None, "status": "error", "finish_reason": None})
    finally:
        if state.is_current(session_id, generation):
            await state.broadcast(session_id, {"event": "status", "state": "listening"})


async def _handle_ai_request(websocket: WebSocket, session_id: str, event: str, message: dict) -> None:
    action = "standard" if event == "ask_ai" else message.get("action")
    if action not in {"standard", "explain_deeper", "code_example"}:
        await websocket.send_json({"event": "error", "message": "Unknown AI action."})
        return
    segments = await state.recent_segments(session_id, limit=40)
    if not segments:
        await websocket.send_json({"event": "error", "message": "Waiting for interviewer speech before asking AI."})
        return
    batch_segments = state.take_pending_transcript(
        session_id,
        segments,
        repeat_last=event == "quick_action",
    )
    if not batch_segments:
        await websocket.send_json({"event": "error", "message": "Waiting for new interviewer speech before asking AI."})
        return
    detected = detect_question(batch_segments, lookback=max(10, len(batch_segments)))
    question = detected["question"]
    if not question:
        state.restore_pending_transcript(session_id, batch_segments)
        await websocket.send_json({"event": "error", "message": "Waiting for interviewer speech before asking AI."})
        return
    topic = str(message.get("topic") or "").strip()[:200]
    batch = "\n".join(batch_segments)
    history = batch_segments
    state.remember_question(session_id, question, batch)
    await state.broadcast(session_id, {
        "event": "question_detected",
        "text": question,
        "batch": batch,
        "is_question": detected["is_question"],
    })
    asyncio.create_task(answer_question(session_id, question, action, history, topic, batch))



_transcription_queues: dict[str, asyncio.Queue] = {}


def _transcription_queue(session_id: str) -> asyncio.Queue:
    """One ordered worker per room: a slow Whisper call must never block new audio."""
    queue = _transcription_queues.get(session_id)
    if queue is None:
        queue = asyncio.Queue(maxsize=6)
        _transcription_queues[session_id] = queue
        asyncio.create_task(_transcription_worker(session_id, queue))
    return queue


async def _transcription_worker(session_id: str, queue: asyncio.Queue) -> None:
    while True:
        wav_bytes = await queue.get()
        try:
            await state.broadcast(session_id, {"event": "status", "state": "transcribing"})
            caption = await asyncio.to_thread(transcribe_audio, wav_bytes)
            if caption:
                if await state.append_caption(session_id, caption):
                    await state.broadcast(session_id, {"event": "caption", "text": caption})
            else:
                await state.broadcast(session_id, {
                    "event": "transcription_empty",
                    "message": "No clear speech was detected. Raise speaker volume or check the loopback device.",
                })
        except Exception as exc:  # noqa: BLE001 - surfaced to both devices
            logger.exception("Transcription failed")
            await state.broadcast(session_id, {"event": "error", "message": _ai_error_message(exc)})
        finally:
            await state.broadcast(session_id, {"event": "status", "state": "listening"})
            queue.task_done()


@app.websocket("/ws/{session_id}")
async def websocket_room(websocket: WebSocket, session_id: str):
    await websocket.accept()
    await state.add_socket(session_id, websocket)
    if not await state.context_for(session_id):
        await websocket.send_json({
            "event": "error",
            "message": "This interview room was not found in the local database. Create the room again from the desktop app.",
        })
    await websocket.send_json({"event": "status", "state": "listening"})
    await websocket.send_json(await state.snapshot(session_id))
    try:
        while True:
            message = await websocket.receive_json()
            event = message.get("event")
            if event in {"ask_ai", "quick_action"}:
                await _handle_ai_request(websocket, session_id, event, message)
            elif event == "clear":
                await state.clear_room(session_id)
                await state.broadcast(session_id, {"event": "cleared"})
                await state.broadcast(session_id, {"event": "status", "state": "listening"})
            elif event == "transcript":
                # The desktop app transcribes on its own machine and sends text only.
                text = " ".join(str(message.get("text") or "").split())[:4000]
                if not text:
                    await websocket.send_json({"event": "error", "message": "Transcript text is empty."})
                    continue
                if await state.append_caption(session_id, text):
                    await state.broadcast(session_id, {"event": "caption", "text": text})
            elif event == "audio_utterance":
                encoded = message.get("audio_base64", "")
                if not encoded:
                    await websocket.send_json({"event": "error", "message": "Audio payload is empty."})
                    continue
                try:
                    wav_bytes = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError):
                    await websocket.send_json({"event": "error", "message": "Audio payload must be base64-encoded WAV data."})
                    continue
                queue = _transcription_queue(session_id)
                while queue.full():
                    try:
                        queue.get_nowait()
                        queue.task_done()
                    except asyncio.QueueEmpty:
                        break
                queue.put_nowait(wav_bytes)
            else:
                await websocket.send_json({"event": "error", "message": "Unsupported event."})
    except WebSocketDisconnect:
        pass
    finally:
        await state.remove_socket(session_id, websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
