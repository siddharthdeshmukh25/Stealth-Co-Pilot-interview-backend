"""Interview history: ended rooms keep their captions and AI answers.

The desktop app lists these rooms and shows the questions and answers captured
during each interview, so nothing has to be re-typed after the call.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..database import get_session
from ..models import Candidate, InterviewSession, QALog, TranscriptSegment

router = APIRouter(prefix="/api/history", tags=["history"])


def _count(db: Session, model, session_id: str) -> int:
    return len(db.exec(select(model.id).where(model.session_id == session_id)).all())


def _payload(record: InterviewSession, candidate: Candidate | None) -> dict:
    name = candidate.name if candidate else ""
    job_title = record.job_title or "Interview"
    return {
        "session_id": record.session_id,
        "candidate_name": name,
        "job_title": job_title,
        "status": record.status,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "ended_at": record.ended_at.isoformat() if record.ended_at else None,
        "duration_seconds": record.duration_seconds,
    }


@router.get("")
def list_history(db: Session = Depends(get_session)):
    records = db.exec(
        select(InterviewSession)
        .where(InterviewSession.status == "ended")
        .order_by(InterviewSession.ended_at.desc())
    ).all()
    interviews = []
    for record in records:
        item = _payload(record, db.get(Candidate, record.candidate_id))
        item["questions"] = _count(db, QALog, record.session_id)
        item["captions"] = _count(db, TranscriptSegment, record.session_id)
        interviews.append(item)
    return {"interviews": interviews}


@router.get("/{session_id}")
def history_detail(session_id: str, db: Session = Depends(get_session)):
    record = db.get(InterviewSession, session_id)
    if not record or record.status != "ended":
        raise HTTPException(404, "This interview is not in your history.")
    detail = _payload(record, db.get(Candidate, record.candidate_id))
    detail["job_description"] = record.job_description
    detail["answers"] = [
        {
            "id": entry.id,
            "question": entry.interviewer_question,
            "answer": entry.ai_answer,
            "action": entry.action_type,
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
        }
        for entry in db.exec(
            select(QALog).where(QALog.session_id == session_id).order_by(QALog.id)
        ).all()
    ]
    detail["transcript"] = [
        {
            "text": segment.text,
            "timestamp": segment.created_at.isoformat() if segment.created_at else None,
        }
        for segment in db.exec(
            select(TranscriptSegment)
            .where(TranscriptSegment.session_id == session_id)
            .order_by(TranscriptSegment.id)
        ).all()
    ]
    return detail
