"""Response models for audio segments."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from database.schema.segments import ResourceSegment
from resources.audio import AudioAttrs


class SegmentRead(BaseModel):
    """An audio segment's metadata as served over HTTP.

    Carries no ``bucket`` or ``object_key``; audio is fetched from
    ``GET /v1/segments/{id}/audio``.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    session_id: UUID
    sequence: int
    ingested_at: datetime
    captured_at: datetime | None
    offset_ms: int | None
    duration_ms: int | None
    byte_size: int
    content_type: str
    checksum_sha256: str | None = Field(description="Hex-encoded SHA-256 of the audio.")
    codec: str | None
    sample_rate_hz: int | None
    channels: int | None
    metadata: dict[str, Any]

    @classmethod
    def from_model(cls, segment: ResourceSegment) -> "SegmentRead":
        attrs = AudioAttrs.from_attrs(segment.attrs)
        return cls(
            id=segment.id,
            session_id=segment.session_id,
            sequence=segment.sequence,
            ingested_at=segment.ingested_at,
            captured_at=segment.captured_at,
            offset_ms=segment.offset_ms,
            duration_ms=segment.duration_ms,
            byte_size=segment.byte_size,
            content_type=segment.content_type,
            checksum_sha256=segment.checksum_hex,
            codec=attrs.codec,
            sample_rate_hz=attrs.sample_rate_hz,
            channels=attrs.channels,
            metadata=segment.metadata,
        )
