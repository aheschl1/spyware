"""Row models for conversation reads."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ConversationTranscript(BaseModel):
    """One transcript inside a conversation, resolved through its speaker cluster."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID
    utterance_id: UUID
    start_ms: int
    end_ms: int
    speaker: str | None = None  # block-local label, provenance
    speaker_id: UUID | None = None
    name: str | None = None
    text: str
    interjection_of: UUID | None = None


class ConversationUtterance(BaseModel):
    """One utterance row, enough to recompute conversation stats."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    start_ms: int
    end_ms: int
    metadata: dict
