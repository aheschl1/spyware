"""Pydantic models for the ``recording_sessions`` table."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RecordingSession(BaseModel):
    """A capture run: one device recording for one user, holding many segments."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    user_id: UUID
    device: str | None = None
    label: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @property
    def is_open(self) -> bool:
        return self.ended_at is None


class SessionCreate(BaseModel):
    """Input for starting a session."""

    model_config = ConfigDict(frozen=True)

    user_id: UUID
    device: str | None = None
    label: str | None = None
    started_at: datetime | None = None  # defaults to now() in the database
    metadata: dict[str, Any] = Field(default_factory=dict)
