"""Response models for text->audio search."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api.schema.timeline import AudioTagLabel
from database.repos.embeddings import AudioSearchHit


class AudioSearchRead(BaseModel):
    """One audio window matching the query, closest first."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID = Field(description="The window's ``audio-tag`` artifact.")
    session_id: UUID
    start_ms: int = Field(description="Start of the matching window, ms.")
    end_ms: int = Field(description="End (exclusive) of the matching window, ms.")
    distance: float = Field(
        description="Cosine distance between the query and the window's audio "
        "embedding (0 = identical direction; lower is closer)."
    )
    labels: tuple[AudioTagLabel, ...] = Field(
        description="The window's tag scores, for orientation alongside the match."
    )

    @classmethod
    def from_model(cls, hit: AudioSearchHit) -> "AudioSearchRead":
        return cls(
            artifact_id=hit.artifact_id,
            session_id=hit.session_id,
            start_ms=hit.start_ms,
            end_ms=hit.end_ms,
            distance=hit.distance,
            labels=tuple(hit.metadata.get("labels", ())),
        )


class AudioSearchResponse(BaseModel):
    """The ranked matches plus the query's embedding model, for provenance."""

    model_config = ConfigDict(frozen=True)

    query: str
    model: str = Field(description="The embedding model that encoded the query.")
    items: list[AudioSearchRead]
