"""Speakers: the global voice clusters the speaker-cluster tier maintains.

Membership is derived data, rebuilt wholesale by every batch clustering run;
what persists is user curation — labels on identities, and pins asserting a
voice-print's identity (created by merges and member moves, honored by every
rebuild). Names are deliberately non-unique, so the by-name transcript route
unions every cluster sharing the name.

The mutating routes here run on the request's transaction; the recluster
route funnels through the same ``rebuild_user_clusters`` (advisory-locked
per user) as the worker job and the CLI. Both the API and the worker read
``PROCESSING_*`` defaults from the same ``.env`` — the no-override behavior
assumes they share it.
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from api.deps import CurrentUser, OwnedSpeaker, Paging, Pipe
from api.schema.common import Page
from database.schema.speakers import SpeakerCreate
from api.schema.speakers import (
    ClusterParamsOverrides,
    ClusterParamsRead,
    ClusterParamsValues,
    ReclusterResponse,
    SimilarSpeakerRead,
    SimilarSpeakersResponse,
    SpeakerLabelRequest,
    SpeakerMemberRead,
    SpeakerMergeRequest,
    SpeakerRead,
    SpeakerReassignRequest,
    SpeakerReassignResponse,
    SpeakerTranscriptRead,
    SpeakerUnpinResponse,
)

router = APIRouter(prefix="/speakers", tags=["speakers"])


def _cluster_defaults() -> ClusterParamsValues:
    # Lazy: keep processing imports out of API startup (the CLI precedent).
    from processing.config import get_settings

    settings = get_settings()
    return ClusterParamsValues(
        cluster_distance=settings.cluster_distance,
        min_talk_ms=settings.cluster_min_talk_ms,
    )


async def _params_read(pipe: Pipe, user_id: UUID) -> ClusterParamsRead:
    defaults = _cluster_defaults()
    row = await pipe.cluster_params.get(user_id)
    overrides = ClusterParamsOverrides(
        cluster_distance=row.cluster_distance if row else None,
        min_talk_ms=row.min_talk_ms if row else None,
    )
    return ClusterParamsRead(
        effective=ClusterParamsValues(
            cluster_distance=overrides.cluster_distance or defaults.cluster_distance,
            min_talk_ms=(
                overrides.min_talk_ms
                if overrides.min_talk_ms is not None
                else defaults.min_talk_ms
            ),
        ),
        overrides=overrides,
        defaults=defaults,
    )


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


@router.get("/cluster-params", summary="Your clustering knobs")
async def get_cluster_params(user: CurrentUser, pipe: Pipe) -> ClusterParamsRead:
    return await _params_read(pipe, user.id)


@router.post("/cluster-params", summary="Override clustering knobs")
async def set_cluster_params(
    user: CurrentUser, pipe: Pipe, body: ClusterParamsOverrides
) -> ClusterParamsRead:
    """Persist per-field overrides (null = keep inheriting the default).
    They apply to every future rebuild — the worker's per-session runs, the
    CLI, and the recluster route alike."""
    await pipe.cluster_params.set(user.id, body.cluster_distance, body.min_talk_ms)
    return await _params_read(pipe, user.id)


@router.post("/cluster-params/reset", summary="Drop all overrides")
async def reset_cluster_params(user: CurrentUser, pipe: Pipe) -> ClusterParamsRead:
    await pipe.cluster_params.clear(user.id)
    return await _params_read(pipe, user.id)


@router.post("/recluster", summary="Rebuild your clusters now")
async def recluster(user: CurrentUser, pipe: Pipe) -> ReclusterResponse:
    """The same batch every session-end runs, on demand — named identities
    and pinned voice-prints survive; unnamed cluster ids churn."""
    from processing.config import get_settings
    from processing.pipelines.speaker_cluster import rebuild_user_clusters

    counts = await rebuild_user_clusters(pipe, user.id, get_settings())
    return ReclusterResponse(
        clusters=counts["clusters"],
        assigned=counts["assigned"],
        pinned=counts["pinned"],
        skipped_short=counts["skipped_short"],
    )


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


@router.get("/{speaker_id}/similar", summary="Merge candidates, closest first")
async def list_similar_speakers(
    speaker: OwnedSpeaker, pipe: Pipe
) -> SimilarSpeakersResponse:
    """Your other clusters in the same embedding model ranked by centroid
    distance — the shortlist for healing a split voice, with the numbers
    visible so a suspiciously far merge looks suspicious."""
    pairs = await pipe.speakers.similar(
        speaker.user_id, speaker.model, speaker.centroid, exclude_id=speaker.id
    )
    return SimilarSpeakersResponse(
        items=[SimilarSpeakerRead.from_pair(row, distance) for row, distance in pairs]
    )


@router.post("/{speaker_id}/merge", summary="Merge this cluster into another")
async def merge_speaker(
    speaker: OwnedSpeaker, pipe: Pipe, body: SpeakerMergeRequest
) -> SpeakerRead:
    """Fold the path speaker's voice-prints into the target, then delete the
    path speaker; the target's centroid becomes the mean over the combined
    members. A label is never lost — an unnamed survivor inherits the
    merged-away cluster's name. Same-model only: pgvector's distance and
    ``avg()`` error out across dimensions."""
    if body.into_speaker_id == speaker.id:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="a cluster cannot be merged into itself",
        )
    survivor = await pipe.speakers.get(body.into_speaker_id)
    if survivor is None or survivor.user_id != speaker.user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="speaker not found")
    if survivor.model != speaker.model:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="clusters live in different embedding models",
        )
    if survivor.name is None and speaker.name is not None:
        await pipe.speakers.set_name(survivor.id, speaker.name)
    # Pin both clusters' members to the survivor so batch rebuilds honor
    # the merge instead of re-deriving the split from geometry.
    await pipe.speakers.pin_members_of(survivor.id, survivor.id)
    await pipe.speakers.pin_members_of(speaker.id, survivor.id)
    await pipe.speakers.merge(speaker.id, survivor.id)
    summary = await pipe.speakers.summarize(survivor.id)
    assert summary is not None  # just merged into it, same transaction
    return SpeakerRead.from_model(summary)


@router.get("/{speaker_id}/members", summary="Inspect a cluster's voice-prints")
async def list_speaker_members(
    speaker: OwnedSpeaker, pipe: Pipe, paging: Paging
) -> Page[SpeakerMemberRead]:
    """Farthest-from-centroid first, so likely wrong voices surface on top;
    each row carries a playable utterance span when diarize produced one."""
    rows = await pipe.speakers.members(
        speaker.id, limit=paging.probe_limit, offset=paging.offset
    )
    return Page.build(rows, paging, SpeakerMemberRead.from_model)


@router.post(
    "/{speaker_id}/members/{artifact_id}/reassign",
    summary="Move a voice-print to another identity",
)
async def reassign_member(
    speaker: OwnedSpeaker,
    artifact_id: UUID,
    pipe: Pipe,
    body: SpeakerReassignRequest,
) -> SpeakerReassignResponse:
    """Pin + move in one step: the print lands in the target cluster and a
    pin keeps it there through every rebuild. A null target ejects it into a
    fresh cluster. An emptied unnamed source dissolves."""
    if body.into_speaker_id == speaker.id:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="the voice-print is already in that cluster",
        )
    if body.into_speaker_id is None:
        # Seed with the source centroid; the post-move recompute makes the
        # single-member cluster's centroid exactly the moved embedding.
        target = await pipe.speakers.create(
            SpeakerCreate(
                user_id=speaker.user_id,
                model=speaker.model,
                centroid=speaker.centroid,
                name=body.name,
            )
        )
    else:
        target = await pipe.speakers.get(body.into_speaker_id)
        if target is None or target.user_id != speaker.user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="speaker not found")
        if target.model != speaker.model:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="clusters live in different embedding models",
            )
    if not await pipe.speakers.reassign(artifact_id, speaker.id, target.id):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="cluster membership changed — reload and retry",
        )
    await pipe.speakers.pin(artifact_id, target.id)
    await pipe.speakers.delete_empty_unlabeled(speaker.user_id, speaker.model)
    source_summary = await pipe.speakers.summarize(speaker.id)
    target_summary = await pipe.speakers.summarize(target.id)
    assert target_summary is not None  # it just gained a member
    return SpeakerReassignResponse(
        source=(
            SpeakerRead.from_model(source_summary) if source_summary else None
        ),
        target=SpeakerRead.from_model(target_summary),
    )


@router.post(
    "/{speaker_id}/members/{artifact_id}/unpin",
    summary="Release a voice-print's pin",
)
async def unpin_member(
    speaker: OwnedSpeaker, artifact_id: UUID, pipe: Pipe
) -> SpeakerUnpinResponse:
    """The assignment stands until the next rebuild decides freely."""
    if not await pipe.speakers.unpin_member(artifact_id, speaker.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="pin not found")
    return SpeakerUnpinResponse(unpinned=True)


@router.get("/{speaker_id}/transcripts", summary="Everything one cluster said")
async def list_speaker_transcripts(
    speaker: OwnedSpeaker, pipe: Pipe, paging: Paging
) -> Page[SpeakerTranscriptRead]:
    """Newest session first, timeline order within a session."""
    rows = await pipe.speakers.transcripts_for_speaker(
        speaker.id, limit=paging.probe_limit, offset=paging.offset
    )
    return Page.build(rows, paging, SpeakerTranscriptRead.from_model)
