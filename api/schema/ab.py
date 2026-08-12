"""Schemas for the transcription A/B routes.

Candidates are served BLIND: no model/strategy on ``AbCandidateRead`` — the
reveal lives only on the vote, which the server derives from the winning
candidate's metadata (the client never asserts it).
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from database.schema.ab_votes import AbTallyRow, AbVote


class AbEnrollResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    queued: bool


class AbCandidateRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: UUID
    text: str
    chars: int


class AbVoteRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: UUID | None = None
    model: str
    strategy: str

    @classmethod
    def from_model(cls, vote: AbVote) -> "AbVoteRead":
        return cls(
            candidate_id=vote.candidate_artifact_id,
            model=vote.model,
            strategy=vote.strategy,
        )


class AbUtteranceRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    utterance_artifact_id: UUID
    start_ms: int
    end_ms: int
    speaker: str | None = None
    candidates: list[AbCandidateRead]
    vote: AbVoteRead | None = None


class AbSessionRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str  # none | queued | running | succeeded | dead
    total: int
    voted: int
    candidates: int
    expected: int  # 4 per utterance; degraded runs may finish below it
    utterances: list[AbUtteranceRead]


class AbVoteRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    utterance_artifact_id: UUID
    candidate_artifact_id: UUID


class AbVoteResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    strategy: str
    text: str


class AbTallyRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    strategy: str
    wins: int

    @classmethod
    def from_model(cls, row: AbTallyRow) -> "AbTallyRead":
        return cls(model=row.model, strategy=row.strategy, wins=row.wins)


class AbSessionState(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: UUID
    votes: int
    status: str  # queued | running | succeeded | dead
    candidates: int
    expected: int


class AbResultsRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    tally: list[AbTallyRead]
    sessions: list[AbSessionState]  # every enrolled session, live run state
