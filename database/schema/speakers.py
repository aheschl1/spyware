"""Pydantic models for the ``speakers`` table and its joins.

A speaker is a global voice cluster over ``speaker_embeddings``, scoped to one
user and one embedding model. ``name`` is the user-given label — NULL means
the cluster exists but nobody has named it yet. Names are deliberately not
unique: an imperfect split leaves two clusters of the same person, both
carrying the name, and fetch-by-name unions them.
"""

import json
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


def _parse_vector(value: object) -> object:
    # pgvector's text form is '[1,2,3]' — valid JSON.
    if isinstance(value, str):
        return json.loads(value)
    return value


class Speaker(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    user_id: UUID
    name: str | None = None
    model: str
    centroid: tuple[float, ...]
    created_at: datetime
    updated_at: datetime

    _parse_centroid = field_validator("centroid", mode="before")(_parse_vector)


class SpeakerCreate(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UUID
    model: str
    centroid: tuple[float, ...]
    name: str | None = None


class SpeakerSummary(Speaker):
    """A speaker plus its membership counts, for listings."""

    embeddings: int
    sessions: int


class CorpusEmbedding(BaseModel):
    """One voice-print as the batch clusterer consumes it: the vector plus
    its previous assignment (identity continuity) and pin (constraint)."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID
    session_id: UUID
    speaker: str  # block-local label
    model: str
    embedding: tuple[float, ...]
    talk_ms: int | None = None  # None on pre-talk_ms rows: passes the gate
    speaker_id: UUID | None = None  # previous assignment, if any
    pinned_to: UUID | None = None  # user-asserted identity, if any

    _parse_embedding = field_validator("embedding", mode="before")(_parse_vector)


class SpeakerMember(BaseModel):
    """One voice-print of a cluster as the inspection view renders it."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID
    session_id: UUID
    started_at: datetime  # the session's start
    speaker: str  # block-local label
    talk_ms: int | None = None
    distance: float  # to the cluster centroid
    pinned: bool
    clip_start_ms: int | None = None  # longest utterance of this voice
    clip_end_ms: int | None = None


class SessionSpeakerLabel(BaseModel):
    """One block-local label of a session with its cluster resolution."""

    model_config = ConfigDict(frozen=True)

    speaker: str  # block-local label
    speaker_id: UUID | None = None  # NULL: not (yet) clustered
    name: str | None = None
    talk_ms: int | None = None


class SpeakerTranscript(BaseModel):
    """One transcript row resolved through a speaker cluster."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID
    session_id: UUID
    speaker_id: UUID
    speaker: str  # block-local label, provenance
    start_ms: int
    end_ms: int
    text: str
    model: str | None = None  # transcription model
