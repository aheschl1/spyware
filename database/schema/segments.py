"""Pydantic models for the ``audio_segments`` table.

A row is metadata plus a pointer; the audio lives in the blob store under
``object_key``.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AudioSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    session_id: UUID
    user_id: UUID
    sequence: int
    ingested_at: datetime
    captured_at: datetime | None = None
    offset_ms: int | None = None
    duration_ms: int | None = None

    bucket: str
    object_key: str
    byte_size: int
    content_type: str
    checksum_sha256: bytes | None = None

    codec: str | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def checksum_hex(self) -> str | None:
        return self.checksum_sha256.hex() if self.checksum_sha256 else None


class SegmentCreate(BaseModel):
    """Input for registering a segment whose blob has already been written.

    ``sequence`` may be left ``None``, in which case the repository assigns the
    next one within the session.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    session_id: UUID
    user_id: UUID
    bucket: str
    object_key: str
    byte_size: int
    content_type: str = "application/octet-stream"
    sequence: int | None = None
    captured_at: datetime | None = None
    offset_ms: int | None = None
    duration_ms: int | None = None
    checksum_sha256: bytes | None = None
    codec: str | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserUsage(BaseModel):
    """How much audio one user has stored."""

    model_config = ConfigDict(frozen=True)

    segments: int
    total_bytes: int
