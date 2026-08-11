"""Speakers: the global voice clusters the speaker-cluster tier maintains.

Clusters either have a user-given label or they don't; labeling is the only
mutation — membership is derived data owned by the pipeline (and rebuilt by
``cli speakers recluster``). Names are deliberately non-unique, so the
by-name transcript route unions every cluster sharing the name.
"""

from fastapi import APIRouter, Query

from api.deps import CurrentUser, OwnedSpeaker, Paging, Pipe
from api.schema.common import Page
from api.schema.speakers import (
    SpeakerLabelRequest,
    SpeakerRead,
    SpeakerTranscriptRead,
)

router = APIRouter(prefix="/speakers", tags=["speakers"])


@router.get("", summary="List your speaker clusters")
async def list_speakers(
    user: CurrentUser,
    pipe: Pipe,
    paging: Paging,
    name: str | None = Query(None, description="Only clusters with exactly this label."),
) -> Page[SpeakerRead]:
    """Labeled and unlabeled clusters alike — an empty cluster stays listed:
    it anchors a named identity across diarize re-runs."""
    rows = await pipe.speakers.list_for_user(
        user.id, name=name, limit=paging.probe_limit, offset=paging.offset
    )
    return Page.build(rows, paging, SpeakerRead.from_model)


# Declared before /{speaker_id} so the literal path segment wins.
@router.get("/transcripts", summary="Everything said by one name")
async def list_transcripts_by_name(
    user: CurrentUser,
    pipe: Pipe,
    paging: Paging,
    name: str = Query(description="The label to fetch; unions all clusters sharing it."),
) -> Page[SpeakerTranscriptRead]:
    """Transcripts across every session and every cluster carrying this
    label, newest session first, in timeline order within a session."""
    rows = await pipe.speakers.transcripts_for_name(
        user.id, name, limit=paging.probe_limit, offset=paging.offset
    )
    return Page.build(rows, paging, SpeakerTranscriptRead.from_model)


@router.get("/{speaker_id}", summary="Fetch one speaker cluster")
async def get_speaker(speaker: OwnedSpeaker, pipe: Pipe) -> SpeakerRead:
    summary = await pipe.speakers.summarize(speaker.id)
    assert summary is not None  # the ownership dependency just loaded it
    return SpeakerRead.from_model(summary)


@router.post("/{speaker_id}/label", summary="Label (or unlabel) a speaker")
async def label_speaker(
    speaker: OwnedSpeaker, pipe: Pipe, body: SpeakerLabelRequest
) -> SpeakerRead:
    """Set the cluster's name, or clear it with an explicit null. The name
    survives re-clustering: a labeled cluster is an identity anchor."""
    await pipe.speakers.set_name(speaker.id, body.name)
    return await get_speaker(speaker, pipe)


@router.get("/{speaker_id}/transcripts", summary="Everything one cluster said")
async def list_speaker_transcripts(
    speaker: OwnedSpeaker, pipe: Pipe, paging: Paging
) -> Page[SpeakerTranscriptRead]:
    """Newest session first, timeline order within a session."""
    rows = await pipe.speakers.transcripts_for_speaker(
        speaker.id, limit=paging.probe_limit, offset=paging.offset
    )
    return Page.build(rows, paging, SpeakerTranscriptRead.from_model)
