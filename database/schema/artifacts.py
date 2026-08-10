"""Pydantic models for the ``pipeline_artifacts`` table."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PipelineArtifact(BaseModel):
    """One pipeline output: an optional blob pointer plus pipeline-defined
    ``kind``, ``links``, and ``metadata``. Consumers query these to find
    upstream results (a transcript, a summary, ...) without knowing how the
    producing pipeline stores its data."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    pipeline: str
    session_id: UUID | None = None
    kind: str
    bucket: str | None = None
    object_key: str | None = None
    links: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ArtifactCreate(BaseModel):
    """Input for recording an artifact."""

    model_config = ConfigDict(frozen=True)

    pipeline: str
    kind: str
    session_id: UUID | None = None
    bucket: str | None = None
    object_key: str | None = None
    links: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
