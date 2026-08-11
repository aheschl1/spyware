"""Response models for pipeline artifacts."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from database.schema.artifacts import PipelineArtifact


class ArtifactRead(BaseModel):
    """A pipeline output attached to a session, as served over HTTP.

    ``start_ms``/``end_ms`` place it on the session timeline; both null means
    the whole session. Carries no ``bucket``/``object_key`` — large payloads
    live in the blob store and are surfaced through pipeline-specific routes
    or ``metadata``.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    pipeline: str
    kind: str
    start_ms: int | None = Field(description="Span start on the session timeline, ms.")
    end_ms: int | None = Field(description="Span end (exclusive) on the session timeline, ms.")
    has_blob: bool = Field(description="Whether a stored payload backs this artifact.")
    links: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_model(cls, artifact: PipelineArtifact) -> "ArtifactRead":
        return cls(
            id=artifact.id,
            pipeline=artifact.pipeline,
            kind=artifact.kind,
            start_ms=artifact.start_ms,
            end_ms=artifact.end_ms,
            has_blob=artifact.object_key is not None,
            links=artifact.links,
            metadata=artifact.metadata,
            created_at=artifact.created_at,
        )
