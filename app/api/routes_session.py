import uuid
import hmac

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from ..config import settings
from ..database import get_session
from ..models import Candidate, InterviewSession, QALog, TranscriptSegment
from ..schemas import PairRequest, PairResponse, SessionCloseRequest, SessionCreate, SessionResponse
from ..state_manager import state

router = APIRouter(prefix="/api/session", tags=["session"])


def _room_context(candidate: Candidate, record: InterviewSession) -> dict:
    return {
        "candidate_name": candidate.name,
        "job_title": record.job_title,
        "job_description": record.job_description,
        "structured_json": candidate.structured_json,
        "resume_text": candidate.raw_resume_text,
    }


@router.post("/create", response_model=SessionResponse)
async def create_session(payload: SessionCreate, db: Session = Depends(get_session)):
    candidate = db.get(Candidate, payload.candidate_id)
    if not candidate:
        raise HTTPException(404, "Upload a resume before starting a session.")
    session_id = str(uuid.uuid4())
    record: InterviewSession | None = None
    for _ in range(6):
        code = await state.create_pairing_code(session_id, settings.pairing_ttl_seconds)
        record = InterviewSession(
            session_id=session_id,
            candidate_id=candidate.id,
            job_title=payload.job_title.strip(),
            job_description=payload.job_description.strip(),
            pairing_code=code,
        )
        db.add(record)
        try:
            db.commit()
            break
        except IntegrityError:
            # A 6-digit code already exists in the database: roll back and draw another.
            db.rollback()
            await state.release_pairing_code(code)
            record = None
    if record is None:
        raise HTTPException(503, "Could not allocate a pairing code. Try creating the room again.")
    db.refresh(record)
    state.session_context_cache[session_id] = _room_context(candidate, record)
    return SessionResponse(session_id=session_id, pairing_code=record.pairing_code, status=record.status)


@router.post("/pair", response_model=PairResponse)
async def pair_desktop(payload: PairRequest, db: Session = Depends(get_session)):
    session_id = await state.resolve_pairing_code(payload.pairing_code)
    if not session_id:
        raise HTTPException(404, "This pairing code is not linked to an active interview room. Create the room again to get a fresh code.")
    record = db.get(InterviewSession, session_id)
    if not record or record.status != "active":
        raise HTTPException(409, "This interview room is no longer active. Create a new room to keep going.")
    record.is_exe_paired = True
    db.add(record)
    db.commit()
    # Rebuild room context if the backend restarted since the room was created.
    await state.context_for(session_id)
    await state.broadcast(session_id, {"event": "status", "state": "exe_connected"})
    return PairResponse(session_id=session_id, paired=True)


@router.get("/pairing/{pairing_code}")
async def resolve_pairing(pairing_code: str, db: Session = Depends(get_session)):
    """Look up a room from the code printed in the desktop app, without pairing the desktop."""
    session_id = await state.resolve_pairing_code(pairing_code)
    if not session_id:
        raise HTTPException(404, "This pairing code is not linked to an active interview room.")
    record = db.get(InterviewSession, session_id)
    if not record or record.status != "active":
        raise HTTPException(409, "This interview room is no longer active. Create a new room in the desktop app.")
    return {
        "session_id": session_id,
        "pairing_code": record.pairing_code,
        "status": record.status,
        "job_title": record.job_title,
    }


@router.get("/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_session)):
    record = db.get(InterviewSession, session_id)
    if not record:
        raise HTTPException(404, "Session not found.")
    return {
        "session_id": record.session_id,
        "job_title": record.job_title,
        "status": record.status,
        "is_exe_paired": record.is_exe_paired,
    }


@router.delete("/{session_id}", status_code=204)
async def close_session(
    session_id: str,
    payload: SessionCloseRequest,
    db: Session = Depends(get_session),
):
    """Delete a closed room's transcript, answers, and unshared resume data."""
    record = db.get(InterviewSession, session_id)
    if not record:
        return Response(status_code=204)
    if not hmac.compare_digest(record.pairing_code, payload.pairing_code):
        raise HTTPException(403, "The pairing code does not match this interview room.")

    candidate_id = record.candidate_id
    for item in db.exec(select(QALog).where(QALog.session_id == session_id)).all():
        db.delete(item)
    for item in db.exec(select(TranscriptSegment).where(TranscriptSegment.session_id == session_id)).all():
        db.delete(item)
    db.delete(record)
    db.flush()

    # A resume may be shared by another room; retain it until the final room closes.
    remaining_room = db.exec(
        select(InterviewSession).where(InterviewSession.candidate_id == candidate_id)
    ).first()
    if not remaining_room:
        candidate = db.get(Candidate, candidate_id)
        if candidate:
            db.delete(candidate)
    db.commit()

    await state.close_room(session_id)
    return Response(status_code=204)
