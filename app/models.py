from datetime import datetime, timezone
from sqlmodel import Field, SQLModel

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Candidate(SQLModel, table=True):
    __tablename__ = "candidates"
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(nullable=False)
    raw_resume_text: str = Field(nullable=False)
    structured_json: str = Field(nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)

class InterviewSession(SQLModel, table=True):
    __tablename__ = "interview_sessions"
    session_id: str = Field(primary_key=True)
    candidate_id: int = Field(foreign_key="candidates.id", nullable=False)
    job_title: str = Field(nullable=False)
    job_description: str = Field(nullable=False)
    pairing_code: str = Field(index=True, unique=True, nullable=False)
    is_exe_paired: bool = Field(default=False, nullable=False)
    status: str = Field(default="active", nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)

class QALog(SQLModel, table=True):
    __tablename__ = "qa_logs"
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="interview_sessions.session_id", index=True, nullable=False)
    interviewer_question: str = Field(nullable=False)
    ai_answer: str = Field(nullable=False)
    action_type: str = Field(default="standard", nullable=False)
    timestamp: datetime = Field(default_factory=utc_now, nullable=False)

class TranscriptSegment(SQLModel, table=True):
    """Durable interviewer captions so Ask AI survives a backend restart."""
    __tablename__ = "transcript_segments"
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="interview_sessions.session_id", index=True, nullable=False)
    seq: int = Field(default=0, nullable=False)
    text: str = Field(nullable=False)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
