import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlmodel import Session

from ..database import get_session
from ..models import Candidate
from ..schemas import ResumeResponse
from ..services.pdf_parser import ResumeParseError, parse_resume

router = APIRouter(prefix="/api/resume", tags=["resume"])


def _to_response(candidate: Candidate) -> ResumeResponse:
    structured = json.loads(candidate.structured_json)
    return ResumeResponse(
        id=candidate.id,
        name=candidate.name,
        structured=structured,
        extraction=structured.get("_extraction"),
        created_at=candidate.created_at,
    )


@router.post("/upload", response_model=ResumeResponse)
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_session)):
    if file.content_type != "application/pdf" and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(415, "Upload a PDF file.")
    data = await file.read()
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(413, "PDF must be 12 MB or smaller.")
    try:
        raw_text, structured, _info = parse_resume(data)
    except ResumeParseError as exc:
        raise HTTPException(422, str(exc)) from exc
    candidate = Candidate(name=str(structured.get("name") or "Candidate"), raw_resume_text=raw_text, structured_json=json.dumps(structured))
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return _to_response(candidate)


@router.get("/{candidate_id}", response_model=ResumeResponse)
def get_resume(candidate_id: int, db: Session = Depends(get_session)):
    candidate = db.get(Candidate, candidate_id)
    if not candidate:
        raise HTTPException(404, "Resume not found.")
    return _to_response(candidate)
