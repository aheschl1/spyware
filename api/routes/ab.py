"""Transcription A/B: enroll a session, serve blinded candidates, take votes.

Enrollment inserts a `transcribe-ab` job directly (the pipeline is
chained-only); re-enrolling deletes the job history first — the same redo
primitive as `sessions retranscribe`. A vote both records the winner (the
`ab_votes` tally row, model/strategy denormalized so republication can't
erase it) and promotes the candidate's text/words into the canonical
transcript artifact — the single location the timeline and search read.
"""

import hashlib
from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from api.deps import CurrentUser, OwnedSession, Pipe
from api.schema.ab import (
    AbCandidateRead,
    AbEnrollResponse,
    AbResultsRead,
    AbSessionRead,
    AbSessionState,
    AbTallyRead,
    AbUtteranceRead,
    AbVoteRead,
    AbVoteRequest,
    AbVoteResponse,
)
from database.repos.pipelines.transcribe_ab import AbQueries
from database.schema.artifacts import ArtifactCreate, host_link
from database.schema.jobs import JobCreate

router = APIRouter(tags=["ab"])

_PIPELINE = "transcribe-ab"


@router.get("/ab/results", summary="Global A/B tally + per-session run state")
async def ab_results(user: CurrentUser, pipe: Pipe) -> AbResultsRead:
    tally = await pipe.ab_votes.tally(user.id)
    by_session = await pipe.ab_votes.counts_by_session(user.id)
    states = await AbQueries(pipe.connection).session_states(user.id)
    return AbResultsRead(
        total=sum(row.wins for row in tally),
        tally=[AbTallyRead.from_model(row) for row in tally],
        sessions=[
            AbSessionState(
                session_id=row["session_id"],
                votes=by_session.get(row["session_id"], 0),
                status=row["status"],
                candidates=row["candidates"],
                votable=row["votable"],
                expected=row["expected"],
            )
            for row in states
        ],
    )


@router.post("/sessions/{session_id}/ab", summary="Generate A/B candidates")
async def enroll(session: OwnedSession, pipe: Pipe) -> AbEnrollResponse:
    """Queue (or re-queue) the candidate run. Regeneration republishes the
    candidate set; votes survive it by design."""
    await pipe.jobs.delete_for_session(session.id, _PIPELINE)
    await pipe.jobs.enqueue(
        JobCreate(
            pipeline=_PIPELINE,
            session_id=session.id,
            dedup_key=f"{_PIPELINE}:session:{session.id}",
            max_attempts=3,
        )
    )
    return AbEnrollResponse(queued=True)


def _blind_order(utterance_id: UUID, candidate_id: UUID) -> str:
    # Deterministic (stable across reloads) but not positionally biased.
    return hashlib.md5(f"{utterance_id}:{candidate_id}".encode()).hexdigest()


@router.get("/sessions/{session_id}/ab", summary="Blinded voting payload")
async def ab_session(session: OwnedSession, pipe: Pipe) -> AbSessionRead:
    utterances = await pipe.artifacts.list_for_session(
        session.id, pipeline="diarize", kind="utterance", limit=1_000_000
    )
    candidates = await pipe.artifacts.list_for_session(
        session.id, pipeline=_PIPELINE, kind="transcript-candidate", limit=1_000_000
    )
    votes = {v.utterance_artifact_id: v for v in await pipe.ab_votes.for_session(session.id)}
    jobs = [j for j in await pipe.jobs.list_for_session(session.id, limit=200) if j.pipeline == _PIPELINE]

    by_utterance: dict[str, list] = {}
    for c in candidates:
        by_utterance.setdefault(c.links.get("utterance", ""), []).append(c)

    rows = []
    for u in utterances:
        mine = sorted(
            by_utterance.get(str(u.id), ()), key=lambda c: _blind_order(u.id, c.id)
        )
        vote = votes.get(u.id)
        rows.append(
            AbUtteranceRead(
                utterance_artifact_id=u.id,
                start_ms=u.start_ms,
                end_ms=u.end_ms,
                speaker=u.metadata.get("speaker"),
                candidates=[
                    AbCandidateRead(
                        candidate_id=c.id,
                        text=c.metadata.get("text", ""),
                        chars=c.metadata.get("chars", 0),
                    )
                    for c in mine
                ],
                vote=AbVoteRead.from_model(vote) if vote else None,
            )
        )
    return AbSessionRead(
        status=jobs[-1].status.value if jobs else "none",
        total=len(rows),
        voted=sum(1 for r in rows if r.vote is not None),
        candidates=len(candidates),
        expected=len(utterances) * 4,
        utterances=rows,
    )


@router.post("/sessions/{session_id}/ab/votes", summary="Vote the best candidate")
async def vote(
    session: OwnedSession, user: CurrentUser, pipe: Pipe, body: AbVoteRequest
) -> AbVoteResponse:
    candidate = await pipe.artifacts.get(body.candidate_artifact_id)
    if (
        candidate is None
        or candidate.session_id != session.id
        or (candidate.pipeline, candidate.kind) != (_PIPELINE, "transcript-candidate")
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="candidate not found")
    if candidate.links.get("utterance") != str(body.utterance_artifact_id):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="candidate does not belong to that utterance",
        )
    utterance = await pipe.artifacts.get(body.utterance_artifact_id)
    if utterance is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="utterance not found")

    model = candidate.metadata.get("model", "")
    strategy = candidate.metadata.get("strategy", "")
    text = candidate.metadata.get("text", "")
    words = candidate.metadata.get("words", [])

    await pipe.ab_votes.upsert(
        user.id, session.id, utterance.id, candidate.id, model, strategy
    )

    patch = {
        "text": text,
        "chars": len(text),
        "words": words,
        "ab_source": {"model": model, "strategy": strategy},
    }
    transcript = await pipe.artifacts.find_by_link(
        "transcribe", "transcript", "utterance", str(utterance.id)
    )
    if transcript is not None:
        await pipe.artifacts.merge_metadata(transcript.id, patch, drop=("edited",))
    else:
        await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="transcribe",
                kind="transcript",
                session_id=session.id,
                start_ms=utterance.start_ms,
                end_ms=utterance.end_ms,
                links={"utterance": str(utterance.id), **host_link(utterance)},
                metadata={
                    **patch,
                    "model": model,
                    "speaker": utterance.metadata.get("speaker"),
                    "overlap_ms": utterance.metadata.get("overlap_ms"),
                },
            )
        )
    return AbVoteResponse(model=model, strategy=strategy, text=text)
