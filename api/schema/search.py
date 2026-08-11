"""Response models for the two audio searches: contrastive (CLAP embedding
similarity over free text) and tag filtering (exact class scores from the
tagger)."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api.schema.timeline import AudioTagLabel
from database.repos.embeddings import AudioSearchHit
from database.repos.tags import TagLabelCount, TagWindowHit


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


class TagSearchRead(BaseModel):
    """One window where the class scored at or above the floor."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID = Field(description="The window's ``audio-tag`` artifact.")
    session_id: UUID
    start_ms: int
    end_ms: int
    label: str = Field(description="The matched class (canonical AudioSet name).")
    score: float = Field(description="The tagger's sigmoid score for that class.")
    labels: tuple[AudioTagLabel, ...] = Field(
        description="The window's full tag list, for context."
    )

    @classmethod
    def from_model(cls, hit: TagWindowHit) -> "TagSearchRead":
        return cls(
            artifact_id=hit.artifact_id,
            session_id=hit.session_id,
            start_ms=hit.start_ms,
            end_ms=hit.end_ms,
            label=hit.label,
            score=hit.score,
            labels=tuple(hit.metadata.get("labels", ())),
        )


class TagSearchResponse(BaseModel):
    """Windows matching the class filter, best score first."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(description="The filter as given (matched as substring).")
    min_score: float
    items: list[TagSearchRead]


class TagLabelRead(BaseModel):
    """One class present in the caller's windows."""

    model_config = ConfigDict(frozen=True)

    label: str
    windows: int = Field(description="Windows where the class scored above the store floor.")
    best: float = Field(description="The class's highest score anywhere.")

    @classmethod
    def from_model(cls, row: TagLabelCount) -> "TagLabelRead":
        return cls(label=row.label, windows=row.windows, best=row.best)


class TagLabelsResponse(BaseModel):
    """Every class heard in the caller's audio, most frequent first."""

    model_config = ConfigDict(frozen=True)

    labels: list[TagLabelRead]
