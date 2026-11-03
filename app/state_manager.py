import asyncio
import difflib
import logging
import secrets
import threading
import time
from collections import defaultdict
from typing import Any

from fastapi import WebSocket
from sqlmodel import Session, select

from .database import engine
from .models import Candidate, InterviewSession, TranscriptSegment

logger = logging.getLogger(__name__)

TRANSCRIPT_WINDOW = 40


def _load_context(session_id: str) -> dict[str, Any] | None:
    """Rebuild room context from SQLite so a backend restart does not break answers."""
    with Session(engine) as db:
        record = db.get(InterviewSession, session_id)
        if not record:
            return None
        candidate = db.get(Candidate, record.candidate_id)
        return {
            "candidate_name": candidate.name if candidate else "",
            "job_title": record.job_title,
            "job_description": record.job_description,
            "structured_json": candidate.structured_json if candidate else "{}",
            "resume_text": candidate.raw_resume_text if candidate else "",
        }


def _save_segment(session_id: str, text: str, seq: int) -> None:
    with Session(engine) as db:
        db.add(TranscriptSegment(session_id=session_id, text=text, seq=seq))
        db.commit()


def _load_segments(session_id: str, limit: int = TRANSCRIPT_WINDOW) -> list[str]:
    with Session(engine) as db:
        rows = db.exec(
            select(TranscriptSegment)
            .where(TranscriptSegment.session_id == session_id)
            .order_by(TranscriptSegment.id.desc())
            .limit(limit)
        ).all()
    return [row.text for row in reversed(rows)]


def _delete_segments(session_id: str) -> None:
    with Session(engine) as db:
        rows = db.exec(select(TranscriptSegment).where(TranscriptSegment.session_id == session_id)).all()
        for row in rows:
            db.delete(row)
        db.commit()


def _active_session_for_code(code: str) -> str | None:
    """Look a pairing code up in SQLite so priming a room survives a backend restart."""
    with Session(engine) as db:
        record = db.exec(
            select(InterviewSession).where(
                InterviewSession.pairing_code == code,
                InterviewSession.status == "active",
            )
        ).first()
        return record.session_id if record else None


def _similar(first: str, second: str) -> bool:
    left = " ".join(first.lower().split())
    right = " ".join(second.lower().split())
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True
    return difflib.SequenceMatcher(None, left, right).ratio() >= 0.9


class LocalStateManager:
    """Room state: pairing codes, sockets, transcripts, and answer generation control.

    Pairing codes stay process-local, but transcripts and room context are also
    readable from SQLite so restarts and reconnects keep working.
    """

    def __init__(self) -> None:
        self.pairing_map: dict[str, tuple[str, float]] = {}
        self.active_sockets: dict[str, set[WebSocket]] = defaultdict(set)
        self.session_context_cache: dict[str, dict[str, Any]] = {}
        self.session_transcript_cache: dict[str, list[str]] = defaultdict(list)
        self._pending_transcript: dict[str, list[str]] = defaultdict(list)
        self._batch_requested: set[str] = set()
        self._lock = asyncio.Lock()
        self._send_locks: dict[int, asyncio.Lock] = {}
        self._answer_generation: dict[str, int] = {}
        self._answer_stop: dict[str, threading.Event] = {}
        self._answer_buffer: dict[str, str] = {}
        self._last_question: dict[str, str] = {}
        self._last_batch: dict[str, str] = {}

    # -- pairing ---------------------------------------------------------
    async def create_pairing_code(self, session_id: str, ttl_seconds: int = 600) -> str:
        async with self._lock:
            self._prune_expired()
            while True:
                code = f"{secrets.randbelow(1_000_000):06d}"
                if code not in self.pairing_map:
                    self.pairing_map[code] = (session_id, time.monotonic() + ttl_seconds)
                    return code

    async def resolve_pairing_code(self, code: str) -> str | None:
        normalized = "".join(character for character in code if character.isdigit())
        if not normalized:
            return None
        async with self._lock:
            self._prune_expired()
            match = self.pairing_map.get(normalized)
            if match:
                return match[0]
        # The in-memory map dies with the process (backend restart, or the app being
        # started before the room). The code is stored with its room, so keep pairing
        # working while that room is active instead of failing with a bare 404.
        return await asyncio.to_thread(_active_session_for_code, normalized)

    async def release_pairing_code(self, code: str) -> None:
        async with self._lock:
            self.pairing_map.pop(code, None)

    def _prune_expired(self) -> None:
        now = time.monotonic()
        self.pairing_map = {code: data for code, data in self.pairing_map.items() if data[1] > now}

    # -- sockets ---------------------------------------------------------
    async def add_socket(self, session_id: str, websocket: WebSocket) -> None:
        self.active_sockets[session_id].add(websocket)
        self._send_locks.setdefault(id(websocket), asyncio.Lock())

    async def remove_socket(self, session_id: str, websocket: WebSocket) -> None:
        sockets = self.active_sockets.get(session_id)
        if sockets:
            sockets.discard(websocket)
            if not sockets:
                self.active_sockets.pop(session_id, None)
        self._send_locks.pop(id(websocket), None)

    async def broadcast(self, session_id: str, message: dict[str, Any]) -> None:
        sockets = list(self.active_sockets.get(session_id, set()))
        if not sockets:
            return
        await asyncio.gather(*(self._send(socket, message) for socket in sockets))

    async def _send(self, socket: WebSocket, message: dict[str, Any]) -> None:
        """One lock per socket: concurrent answer/status writes used to raise and drop the client."""
        lock = self._send_locks.setdefault(id(socket), asyncio.Lock())
        try:
            async with lock:
                await socket.send_json(message)
        except Exception as exc:  # noqa: BLE001 - a dead socket must not break the room
            logger.debug("Skipping a websocket send: %s", exc)

    # -- transcript ------------------------------------------------------
    async def append_caption(self, session_id: str, text: str) -> bool:
        """Keep a bounded transcript in memory and in SQLite; drop near-duplicate captions."""
        text = " ".join(text.split())
        if not text:
            return False
        transcript = self.session_transcript_cache[session_id]
        if transcript and _similar(transcript[-1], text):
            return False
        normalized = text.lower()
        if "welcome to" in normalized and "interview" in normalized:
            # A fresh interview greeting starts a new captured part even if the
            # room still contains captions from an earlier practice run.
            self._pending_transcript[session_id].clear()
        transcript.append(text)
        del transcript[:-TRANSCRIPT_WINDOW]
        pending = self._pending_transcript[session_id]
        pending.append(text)
        del pending[:-TRANSCRIPT_WINDOW]
        await asyncio.to_thread(_save_segment, session_id, text, len(transcript))
        return True

    async def recent_segments(self, session_id: str, limit: int = 8) -> list[str]:
        cached = self.session_transcript_cache.get(session_id)
        if cached:
            return list(cached[-limit:])
        loaded = await asyncio.to_thread(_load_segments, session_id)
        if loaded:
            self.session_transcript_cache[session_id] = loaded
        return loaded[-limit:]

    def take_pending_transcript(
        self,
        session_id: str,
        fallback: list[str],
        repeat_last: bool = False,
    ) -> list[str]:
        """Take all captions since the previous Ask AI click as one turn batch."""
        pending = self._pending_transcript.pop(session_id, [])
        if pending:
            self._batch_requested.add(session_id)
            return pending
        if session_id not in self._batch_requested:
            self._batch_requested.add(session_id)
            return list(fallback)
        if repeat_last and self._last_batch.get(session_id):
            return self._last_batch[session_id].splitlines()
        return []

    def restore_pending_transcript(self, session_id: str, segments: list[str]) -> None:
        pending = self._pending_transcript[session_id]
        self._pending_transcript[session_id] = (segments + pending)[-TRANSCRIPT_WINDOW:]

    async def clear_room(self, session_id: str) -> None:
        self.session_transcript_cache.pop(session_id, None)
        self._pending_transcript.pop(session_id, None)
        self._batch_requested.discard(session_id)
        self._answer_buffer.pop(session_id, None)
        self._last_question.pop(session_id, None)
        self._last_batch.pop(session_id, None)
        await asyncio.to_thread(_delete_segments, session_id)

    async def close_room(self, session_id: str) -> None:
        """Stop live work and forget all in-memory room/resume context."""
        stop = self._answer_stop.pop(session_id, None)
        if stop:
            stop.set()
        self._answer_generation.pop(session_id, None)
        await self.broadcast(session_id, {"event": "room_closed"})
        self._answer_buffer.pop(session_id, None)
        self._last_question.pop(session_id, None)
        self._last_batch.pop(session_id, None)
        self.session_context_cache.pop(session_id, None)
        self.session_transcript_cache.pop(session_id, None)
        self._pending_transcript.pop(session_id, None)
        self._batch_requested.discard(session_id)
        for code, (mapped_session, _) in list(self.pairing_map.items()):
            if mapped_session == session_id:
                self.pairing_map.pop(code, None)
        for websocket in list(self.active_sockets.get(session_id, set())):
            try:
                await websocket.close(code=1000, reason="Interview room closed")
            except Exception:
                pass
            await self.remove_socket(session_id, websocket)

    def latest_transcript(self, session_id: str) -> str:
        return "\n".join(self.session_transcript_cache.get(session_id, [])[-12:])

    # -- context ---------------------------------------------------------
    async def context_for(self, session_id: str) -> dict[str, Any] | None:
        context = self.session_context_cache.get(session_id)
        if context:
            return context
        context = await asyncio.to_thread(_load_context, session_id)
        if context:
            self.session_context_cache[session_id] = context
        return context

    # -- answer generation ----------------------------------------------
    def begin_answer(self, session_id: str) -> tuple[int, threading.Event]:
        """Supersede any answer still streaming for this room."""
        previous = self._answer_stop.get(session_id)
        if previous is not None:
            previous.set()
        stop = threading.Event()
        self._answer_stop[session_id] = stop
        generation = self._answer_generation.get(session_id, 0) + 1
        self._answer_generation[session_id] = generation
        self._answer_buffer[session_id] = ""
        return generation, stop

    def is_current(self, session_id: str, generation: int) -> bool:
        return self._answer_generation.get(session_id) == generation

    def append_answer(self, session_id: str, token: str) -> None:
        self._answer_buffer[session_id] = self._answer_buffer.get(session_id, "") + token

    def answer_snapshot(self, session_id: str) -> str:
        return self._answer_buffer.get(session_id, "")

    def clear_answer(self, session_id: str) -> None:
        self._answer_buffer[session_id] = ""

    def remember_question(self, session_id: str, question: str, batch: str = "") -> None:
        self._last_question[session_id] = question
        self._last_batch[session_id] = batch

    def last_question(self, session_id: str) -> str:
        return self._last_question.get(session_id, "")

    async def snapshot(self, session_id: str) -> dict[str, Any]:
        return {
            "event": "snapshot",
            "captions": await self.recent_segments(session_id, limit=12),
            "question": self._last_question.get(session_id, ""),
            "batch": self._last_batch.get(session_id, ""),
            "answer": self.answer_snapshot(session_id),
        }


state = LocalStateManager()
