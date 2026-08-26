"""Conversations: runs of utterances the conversation tier grouped.

Membership is curated in place (exclude/include), like transcript text
edits: the edit lives in the artifact and is replaced when the session is
rediarized.
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Response, status

from api.deps import CurrentUser, Pipe
from api.schema.conversations import (
    ConversationDetail,
    ConversationMemberRequest,
    ConversationRead,
    ConversationTranscriptRead,
)
from database.pipe import DatabasePipe
from database.schema.artifacts import PipelineArtifact, parse_uuid
from processing.config import get_settings
from processing.pipelines.conversation import read_turn, stats_metadata, summarize

router = APIRouter(prefix="/conversations", tags=["conversations"])


async def _owned(pipe: DatabasePipe, conversation_id: UUID, user_id: UUID) -> PipelineArtifact:
    artifact = await pipe.conversations.get_owned(conversation_id, user_id)
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="conversation not found")
    return artifact


async def _read(pipe: DatabasePipe, artifact: PipelineArtifact) -> ConversationRead:
    assert artifact.session_id is not None
    labels = await pipe.speakers.labels_for_session(artifact.session_id)
    return ConversationRead.from_artifact(artifact, {label.speaker: label for label in labels})


@router.get("/{conversation_id}", summary="One conversation with its transcript")
async def get_conversation(
    conversation_id: UUID, user: CurrentUser, pipe: Pipe
) -> ConversationDetail:
    artifact = await _owned(pipe, conversation_id, user.id)
    read = await _read(pipe, artifact)
    rows = await pipe.conversations.transcripts_for(read.utterance_ids)
    return ConversationDetail(
        **read.model_dump(),
        transcripts=[ConversationTranscriptRead.from_model(row) for row in rows],
    )


async def _rewrite(
    pipe: DatabasePipe,
    artifact: PipelineArtifact,
    members: list[UUID],
    excluded: list[dict],
) -> PipelineArtifact | None:
    """Recompute the conversation over ``members``; None when too few remain."""
    rows = await pipe.conversations.utterances(members)
    turns = [
        turn
        for row in rows
        if (turn := read_turn(row.id, row.start_ms, row.end_ms, row.metadata)) is not None
    ]
    if len(turns) < get_settings().conversation_min_turns:
        await pipe.artifacts.delete(artifact.id)
        return None
    stats = summarize(turns)
    order = {turn.id: turn.start_ms for turn in turns}
    return await pipe.conversations.set_membership(
        artifact.id,
        start_ms=stats.start_ms,
        end_ms=stats.end_ms,
        patch={
            "utterances": [str(u) for u in sorted(order, key=order.get)],
            "excluded": excluded,
            **stats_metadata(stats),
        },
    )


def _membership(artifact: PipelineArtifact) -> tuple[list[UUID], list[dict]]:
    members = [u for raw in artifact.metadata.get("utterances", ()) if (u := parse_uuid(raw))]
    excluded = [
        dict(item)
        for item in artifact.metadata.get("excluded", ())
        if isinstance(item, dict) and parse_uuid(item.get("utterance"))
    ]
    return members, excluded


@router.post(
    "/{conversation_id}/exclude",
    summary="Drop an utterance from a conversation",
    responses={204: {"description": "Too few members remained; the conversation is gone."}},
)
async def exclude_utterance(
    conversation_id: UUID, body: ConversationMemberRequest, user: CurrentUser, pipe: Pipe
) -> ConversationRead:
    """Bounds and counts are recomputed from the remaining members; the
    conversation is never split. The exclusion is reversible via ``include``."""
    artifact = await _owned(pipe, conversation_id, user.id)
    members, excluded = _membership(artifact)
    if body.utterance_id not in members:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not a member")
    members.remove(body.utterance_id)
    excluded.append(
        {"utterance": str(body.utterance_id), "reason": body.reason, "source": "manual"}
    )
    updated = await _rewrite(pipe, artifact, members, excluded)
    if updated is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)  # type: ignore[return-value]
    return await _read(pipe, updated)


@router.post("/{conversation_id}/include", summary="Restore an excluded utterance")
async def include_utterance(
    conversation_id: UUID, body: ConversationMemberRequest, user: CurrentUser, pipe: Pipe
) -> ConversationRead:
    artifact = await _owned(pipe, conversation_id, user.id)
    members, excluded = _membership(artifact)
    remaining = [e for e in excluded if parse_uuid(e.get("utterance")) != body.utterance_id]
    if len(remaining) == len(excluded):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not excluded")
    members.append(body.utterance_id)
    updated = await _rewrite(pipe, artifact, members, remaining)
    assert updated is not None  # including can only grow the member set
    return await _read(pipe, updated)
