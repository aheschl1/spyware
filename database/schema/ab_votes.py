"""Pydantic models for the ``ab_votes`` table.

One row per utterance: the human-chosen winner of the transcription A/B.
``model``/``strategy`` are copied off the winning candidate so the tally
survives candidate republication (the FK goes NULL, the vote stays).
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AbVote(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    user_id: UUID
    session_id: UUID
    utterance_artifact_id: UUID
    candidate_artifact_id: UUID | None = None
    model: str
    strategy: str
    created_at: datetime


class AbTallyRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    strategy: str
    wins: int
