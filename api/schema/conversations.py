"""Response models for conversations: runs of utterances grouped by the
conversation tier."""

from collections.abc import Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from database.schema.artifacts import PipelineArtifact, parse_uuid
from database.schema.conversations import ConversationTranscript
from database.schema.speakers import SessionSpeakerLabel


class ConversationSpeaker(BaseModel):
    """One voice heard in the conversation, with its cluster resolution."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(description="Block-local diarizer label (provenance).")
    speaker_id: UUID | None = Field(None, description="Global cluster; null if unclustered.")
    name: str | None = Field(None, description="The cluster's user-given label.")


class ExcludedUtterance(BaseModel):
    model_config = ConfigDict(frozen=True)

    utterance_id: UUID
    reason: str | None = None
    source: str = Field(description="Who excluded it: ``manual`` or a filter name.")


class ConversationRead(BaseModel):
    """A run of utterances the tier judged to be one conversation."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    session_id: UUID
    start_ms: int
    end_ms: int
    turns: int = Field(description="Member utterances.")
    alternations: int = Field(
        description="Same-block speaker changes between consecutive members — "
        "0 means nobody demonstrably replied."
    )
    speakers: list[ConversationSpeaker]
    opening: str = Field(description="Why it started here: ``gap`` or ``session_start``.")
    closure: str = Field(description="Why it ended here: ``gap`` or ``session_end``.")
    gap_before_ms: int | None = None
    gap_after_ms: int | None = None
    utterance_ids: list[UUID] = Field(description="Members, in timeline order.")
    excluded: list[ExcludedUtterance]

    @classmethod
    def from_artifact(
        cls, artifact: PipelineArtifact, labels: Mapping[str, SessionSpeakerLabel]
    ) -> "ConversationRead":
        assert artifact.session_id is not None
        assert artifact.start_ms is not None and artifact.end_ms is not None
        meta = artifact.metadata
        return cls(
            id=artifact.id,
            session_id=artifact.session_id,
            start_ms=artifact.start_ms,
            end_ms=artifact.end_ms,
            turns=int(meta.get("turns", 0)),
            alternations=int(meta.get("alternations", 0)),
            speakers=[
                ConversationSpeaker(
                    label=label,
                    speaker_id=labels[label].speaker_id if label in labels else None,
                    name=labels[label].name if label in labels else None,
                )
                for label in meta.get("speakers", ())
            ],
            opening=str(meta.get("opening", "gap")),
            closure=str(meta.get("closure", "gap")),
            gap_before_ms=meta.get("gap_before_ms"),
            gap_after_ms=meta.get("gap_after_ms"),
            utterance_ids=[u for raw in meta.get("utterances", ()) if (u := parse_uuid(raw))],
            excluded=[
                ExcludedUtterance(
                    utterance_id=u, reason=item.get("reason"), source=str(item.get("source", ""))
                )
                for item in meta.get("excluded", ())
                if isinstance(item, Mapping) and (u := parse_uuid(item.get("utterance")))
            ],
        )


class ConversationTranscriptRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: UUID
    utterance_id: UUID
    start_ms: int
    end_ms: int
    speaker: str | None = None
    speaker_id: UUID | None = None
    speaker_name: str | None = None
    text: str
    interjection_of: UUID | None = None

    @classmethod
    def from_model(cls, row: ConversationTranscript) -> "ConversationTranscriptRead":
        return cls(
            artifact_id=row.artifact_id,
            utterance_id=row.utterance_id,
            start_ms=row.start_ms,
            end_ms=row.end_ms,
            speaker=row.speaker,
            speaker_id=row.speaker_id,
            speaker_name=row.name,
            text=row.text,
            interjection_of=row.interjection_of,
        )


class ConversationDetail(ConversationRead):
    """A conversation with its transcript in order."""

    transcripts: list[ConversationTranscriptRead]


class ConversationMemberRequest(BaseModel):
    """Body for exclude/include: which utterance, and (for exclude) why."""

    model_config = ConfigDict(frozen=True)

    utterance_id: UUID
    reason: str | None = Field(None, max_length=500)

