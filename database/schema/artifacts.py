"""Pydantic models for the ``pipeline_artifacts`` table."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PipelineArtifact(BaseModel):
    """One pipeline output: an optional blob pointer plus pipeline-defined
    ``kind``, ``links``, and ``metadata``. Consumers query these to find
    upstream results (a transcript, a summary, ...) without knowing how the
    producing pipeline stores its data.

    ``start_ms``/``end_ms`` place the artifact on the session's timeline
    (milliseconds from session start); both NULL means the whole session.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    pipeline: str
    session_id: UUID | None = None
    kind: str
    start_ms: int | None = None
    end_ms: int | None = None
    bucket: str | None = None
    object_key: str | None = None
    links: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


def parse_uuid(value: Any) -> UUID | None:
    """A link value as a UUID; None when missing or malformed."""
    try:
        return UUID(str(value)) if value else None
    except ValueError:
        return None


def host_link(utterance: PipelineArtifact) -> dict[str, str]:
    """The ``host_utterance`` link to copy onto anything derived from an
    interjection utterance (empty for ordinary utterances)."""
    host = utterance.links.get("host_utterance")
    return {"host_utterance": host} if host else {}


class ArtifactCreate(BaseModel):
    """Input for recording an artifact."""

    model_config = ConfigDict(frozen=True)

    pipeline: str
    kind: str
    session_id: UUID | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    bucket: str | None = None
    object_key: str | None = None
    links: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
