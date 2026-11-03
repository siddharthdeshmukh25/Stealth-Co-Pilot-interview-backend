from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, field_validator

class ResumeResponse(BaseModel):
    id: int
    name: str
    structured: dict[str, Any]
    extraction: dict[str, Any] | None = None
    created_at: datetime

class SessionCreate(BaseModel):
    candidate_id: int
    job_title: str = Field(min_length=1, max_length=160)
    job_description: str = Field(min_length=20)

class SessionResponse(BaseModel):
    session_id: str
    pairing_code: str
    status: str

class PairRequest(BaseModel):
    pairing_code: str = Field(min_length=6, max_length=12)

    @field_validator("pairing_code")
    @classmethod
    def _six_digits(cls, value: str) -> str:
        """Accept the code as shown in the app (e.g. "184-149") as well as plain digits."""
        digits = "".join(character for character in value if character.isdigit())
        if len(digits) != 6:
            raise ValueError("Pairing code must be six digits.")
        return digits


class SessionCloseRequest(PairRequest):
    """Require the desktop's pairing code before deleting a room and its data."""

class PairResponse(BaseModel):
    session_id: str
    paired: bool
