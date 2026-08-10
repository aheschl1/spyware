"""Response models for recording sessions."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from database.schema.sessions import RecordingSession


class SessionRead(BaseModel):
    """A recording session as served over HTTP.

    Carries no ``user_id``; every session a caller can see is their own.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    device: str | None
    label: str | None
    started_at: datetime
    ended_at: datetime | None
    is_open: bool = Field(description="True while the session has no end time.")
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, session: RecordingSession) -> "SessionRead":
        return cls(
            id=session.id,
            device=session.device,
            label=session.label,
            started_at=session.started_at,
            ended_at=session.ended_at,
            is_open=session.is_open,
            metadata=session.metadata,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
