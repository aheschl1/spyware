"""Response models for the speakers (global voice clusters) routes."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from database.schema.speakers import SpeakerMember, SpeakerSummary, SpeakerTranscript


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


class ClusterParamsValues(BaseModel):
    """A complete, resolved set of clustering knobs."""

    model_config = ConfigDict(frozen=True)

    cluster_distance: float = Field(
        description="Agglomeration stops at this cosine distance."
    )
    min_talk_ms: int = Field(
        description="Voice-prints from less speech than this are not clustered."
    )


class ClusterParamsOverrides(BaseModel):
    """Per-field overrides; null = inherit the server default for that field.

    Doubles as the ``POST /speakers/cluster-params`` body — every key is
    required so a partial body is a 422, never a silent clear."""

    model_config = ConfigDict(frozen=True)

    cluster_distance: float | None = Field(
        description="Override, or null to inherit the default.", gt=0.0, le=2.0
    )
    min_talk_ms: int | None = Field(
        description="Override, or null to inherit the default.", ge=0
    )


class ClusterParamsRead(BaseModel):
    """The user's clustering configuration: what applies, what they pinned,
    and what the server would use on its own."""

    model_config = ConfigDict(frozen=True)

    effective: ClusterParamsValues
    overrides: ClusterParamsOverrides
    defaults: ClusterParamsValues


class ReclusterResponse(BaseModel):
    """Counts from one batch rebuild."""

    model_config = ConfigDict(frozen=True)

    clusters: int = Field(description="Resulting cluster count.")
    assigned: int = Field(description="Voice-prints placed in a cluster.")
    pinned: int = Field(description="Voice-prints held by a user pin.")
    skipped_short: int = Field(description="Voice-prints under the talk gate.")


class SpeakerMemberRead(BaseModel):
    """One voice-print of a cluster, for the inspection view. ``distance``
    is to the cluster centroid — outliers (likely wrong voices) rank high.
    The clip is the voice's longest utterance in its block, playable
    directly; null when diarize produced no utterance for it."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID
    session_id: UUID
    started_at: datetime = Field(description="The session's start time.")
    speaker: str = Field(description="Block-local diarization label (provenance).")
    talk_ms: int | None = None
    distance: float
    pinned: bool = Field(description="Held to this identity by a user pin.")
    clip_start_ms: int | None = None
    clip_end_ms: int | None = None

    @classmethod
    def from_model(cls, row: SpeakerMember) -> "SpeakerMemberRead":
        return cls(
            artifact_id=row.artifact_id,
            session_id=row.session_id,
            started_at=row.started_at,
            speaker=row.speaker,
            talk_ms=row.talk_ms,
            distance=row.distance,
            pinned=row.pinned,
            clip_start_ms=row.clip_start_ms,
            clip_end_ms=row.clip_end_ms,
        )


class SpeakerReassignRequest(BaseModel):
    """Body of ``POST /speakers/{id}/members/{artifact_id}/reassign``: move
    one voice-print to another identity and pin it there. A null target
    ejects it into a fresh cluster (named via ``name`` — recommended: an
    unnamed pair of clusters may legitimately re-merge on the next rebuild)."""

    model_config = ConfigDict(frozen=True)

    into_speaker_id: UUID | None = Field(
        description="Target cluster, or null to eject into a new one."
    )
    name: str | None = Field(
        None, description="Label for the new cluster when ejecting.", max_length=200
    )


class SpeakerReassignResponse(BaseModel):
    """Both sides after a move; ``source`` is null when the move emptied an
    unnamed cluster and it was dissolved."""

    model_config = ConfigDict(frozen=True)

    source: "SpeakerRead | None"
    target: "SpeakerRead"


class SpeakerUnpinResponse(BaseModel):
    """Acknowledges a pin removal; the assignment itself stands until the
    next rebuild decides."""

    model_config = ConfigDict(frozen=True)

    unpinned: bool


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
