"""Response models for the speakers (global voice clusters) routes."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from database.schema.speakers import SpeakerSummary, SpeakerTranscript


class SpeakerRead(BaseModel):
    """One global speaker cluster. ``name`` is the user-given label — null
    means nobody has named it yet. Names are not unique: an imperfectly split
    voice may briefly be two clusters sharing a name, and fetch-by-name
    unions them."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    name: str | None = Field(None, description="User-given label; null = unlabeled.")
    model: str = Field(description="The embedding model this cluster lives in.")
    embeddings: int = Field(description="Voice-prints currently assigned to the cluster.")
    sessions: int = Field(description="Distinct sessions the cluster was heard in.")
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, speaker: SpeakerSummary) -> "SpeakerRead":
        return cls(
            id=speaker.id,
            name=speaker.name,
            model=speaker.model,
            embeddings=speaker.embeddings,
            sessions=speaker.sessions,
            created_at=speaker.created_at,
            updated_at=speaker.updated_at,
        )


class SpeakerLabelRequest(BaseModel):
    """Body of ``POST /speakers/{id}/label``. ``name`` is required so an
    empty body is a 422, never a silent clear; send an explicit null to
    remove the label."""

    model_config = ConfigDict(frozen=True)

    name: str | None = Field(
        description="The label to set, or null to clear it.", max_length=200
    )


class SpeakerMergeRequest(BaseModel):
    """Body of ``POST /speakers/{id}/merge``: fold the path speaker into
    this one. The survivor keeps its id (and its name, unless it has none
    and the merged-away cluster does)."""

    model_config = ConfigDict(frozen=True)

    into_speaker_id: UUID = Field(description="The cluster that survives the merge.")


class SimilarSpeakerRead(SpeakerRead):
    """A merge candidate: another cluster in the same embedding model,
    with its centroid's cosine distance from the reference speaker."""

    distance: float = Field(
        description="Cosine distance between centroids; ~0.6 within one "
        "voice, ~0.9 between different voices."
    )

    @classmethod
    def from_pair(
        cls, speaker: SpeakerSummary, distance: float
    ) -> "SimilarSpeakerRead":
        return cls(
            id=speaker.id,
            name=speaker.name,
            model=speaker.model,
            embeddings=speaker.embeddings,
            sessions=speaker.sessions,
            created_at=speaker.created_at,
            updated_at=speaker.updated_at,
            distance=distance,
        )


class SimilarSpeakersResponse(BaseModel):
    """Every other cluster of yours in the same model, closest first."""

    model_config = ConfigDict(frozen=True)

    items: list[SimilarSpeakerRead]


class SpeakerTranscriptRead(BaseModel):
    """One utterance's transcript, resolved through a speaker cluster."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID
    session_id: UUID
    speaker_id: UUID
    speaker: str = Field(description="Block-local diarization label (provenance).")
    start_ms: int
    end_ms: int
    text: str
    model: str | None = Field(None, description="The transcription model.")

    @classmethod
    def from_model(cls, row: SpeakerTranscript) -> "SpeakerTranscriptRead":
        return cls(
            artifact_id=row.artifact_id,
            session_id=row.session_id,
            speaker_id=row.speaker_id,
            speaker=row.speaker,
            start_ms=row.start_ms,
            end_ms=row.end_ms,
            text=row.text,
            model=row.model,
        )


class SessionSpeakerRead(BaseModel):
    """One voice heard in a session: a cluster (possibly unlabeled), or a
    not-yet-clustered local label (``speaker_id`` null) — nothing is hidden."""

    model_config = ConfigDict(frozen=True)

    speaker_id: UUID | None = Field(
        None, description="Global cluster id; null when not (yet) clustered."
    )
    name: str | None = Field(None, description="The cluster's label, if any.")
    local_labels: list[str] = Field(
        description="Block-local diarization labels resolved to this voice."
    )
    talk_ms: int = Field(description="Total speech attributed to this voice, ms.")
