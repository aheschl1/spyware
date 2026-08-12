"""Transcription A/B end to end: enrollment generates four blinded candidates
per utterance (2 models x 2 strategies), a vote promotes the winner into the
canonical transcript and lands in the tally, and curation survives
regeneration.

The stub answers ?model=whisper with distinct text, so candidates are
tellable-apart. The e2e session is one 300ms block with two utterances
(0-150, 150-300); every stub word's midpoint falls under 100ms, so block
candidates carry all words for utterance one and none for utterance two.
"""

import httpx

from database.pipe import DatabasePipe
from database.schema.jobs import JobStatus
from tests.e2e.conftest import (
    Account,
    ingest,
    make_account,
    make_session,
    wait_for_job,
)
from tests.e2e.stub_audio_services import (
    STUB_TEXT,
    STUB_WHISPER_TEXT,
    STUB_WHISPER_WORDS,
)
from tests.wav import wav_bytes
from uuid import uuid4


async def _diarized_session(account: Account):
    session = await make_session(account)
    for _ in range(3):
        await ingest(session.id, wav_bytes(seconds=0.1), duration_ms=100)
    async with DatabasePipe() as pipe:
        await pipe.sessions.end(session.id)
    await wait_for_job(session.id, "transcribe", JobStatus.SUCCEEDED)
    return session


async def _enrolled(client: httpx.AsyncClient, account: Account):
    session = await _diarized_session(account)
    queued = await client.post(
        f"/v1/sessions/{session.id}/ab", headers=account.headers
    )
    assert queued.status_code == 200 and queued.json()["queued"] is True
    await wait_for_job(session.id, "transcribe-ab", JobStatus.SUCCEEDED)
    return session


async def _payload(client: httpx.AsyncClient, account: Account, session_id) -> dict:
    response = await client.get(f"/v1/sessions/{session_id}/ab", headers=account.headers)
    assert response.status_code == 200
    return response.json()


async def test_enrollment_generates_four_blinded_candidates(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    account = await make_account()
    session = await _enrolled(client, account)

    body = await _payload(client, account, session.id)
    assert body["status"] == "succeeded"
    assert body["total"] == 2 and body["voted"] == 0
    assert body["candidates"] == 8 and body["expected"] == 8
    first, second = body["utterances"]

    assert (first["start_ms"], first["end_ms"]) == (0, 150)
    assert len(first["candidates"]) == 4
    # Blind: no model/strategy anywhere on a candidate.
    assert all(
        set(c) == {"candidate_id", "text", "chars"} for c in first["candidates"]
    )
    texts = sorted(c["text"] for c in first["candidates"])
    # chunk parakeet, chunk whisper, block parakeet, block whisper — the
    # block words all land in this utterance, so texts pair up.
    assert texts == sorted([STUB_TEXT, STUB_WHISPER_TEXT, STUB_TEXT, STUB_WHISPER_TEXT])

    # No stub word's midpoint reaches 150ms: block candidates are empty here.
    second_texts = sorted(c["text"] for c in second["candidates"])
    assert second_texts == sorted([STUB_TEXT, STUB_WHISPER_TEXT, "", ""])

    async with DatabasePipe() as pipe:
        candidates = await pipe.artifacts.list_for_session(
            session.id, pipeline="transcribe-ab", kind="transcript-candidate",
            limit=100,
        )
    assert len(candidates) == 8
    assert {
        (c.metadata["model"], c.metadata["strategy"]) for c in candidates
    } == {("parakeet", "chunk"), ("whisper", "chunk"), ("parakeet", "block"), ("whisper", "block")}
    # Block words were rebased to session time before assignment.
    block_whisper = next(
        c for c in candidates
        if (c.metadata["model"], c.metadata["strategy"]) == ("whisper", "block")
        and c.start_ms == 0
    )
    assert block_whisper.metadata["words"] == [
        {"w": w["word"], "s": w["start_ms"], "e": w["end_ms"]} for w in STUB_WHISPER_WORDS
    ]


async def test_vote_promotes_winner_and_tallies(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    account = await make_account()
    session = await _enrolled(client, account)
    body = await _payload(client, account, session.id)
    first = body["utterances"][0]

    def candidate_with(text: str) -> str:
        return next(c["candidate_id"] for c in first["candidates"] if c["text"] == text)

    # The blinded payload identifies nothing; the vote response reveals.
    voted = await client.post(
        f"/v1/sessions/{session.id}/ab/votes",
        headers=account.headers,
        json={
            "utterance_artifact_id": first["utterance_artifact_id"],
            "candidate_artifact_id": candidate_with(STUB_WHISPER_TEXT),
        },
    )
    assert voted.status_code == 200
    reveal = voted.json()
    assert reveal["model"] == "whisper" and reveal["text"] == STUB_WHISPER_TEXT

    # Ground truth: the canonical transcript now serves the winner.
    timeline = await client.get(
        f"/v1/sessions/{session.id}/timeline", headers=account.headers
    )
    transcripts = [e for e in timeline.json()["items"] if e["type"] == "transcript"]
    winner = next(e for e in transcripts if e["start_ms"] == 0)
    assert winner["text"] == STUB_WHISPER_TEXT
    async with DatabasePipe() as pipe:
        transcript = await pipe.artifacts.find_by_link(
            "transcribe", "transcript", "utterance", first["utterance_artifact_id"]
        )
    assert transcript.metadata["ab_source"] == {"model": "whisper", "strategy": reveal["strategy"]}
    assert transcript.metadata["words"]

    # The payload now carries the reveal; voted counts.
    body = await _payload(client, account, session.id)
    assert body["voted"] == 1
    assert body["utterances"][0]["vote"]["model"] == "whisper"

    # Revote replaces, never double-counts.
    revoted = await client.post(
        f"/v1/sessions/{session.id}/ab/votes",
        headers=account.headers,
        json={
            "utterance_artifact_id": first["utterance_artifact_id"],
            "candidate_artifact_id": candidate_with(STUB_TEXT),
        },
    )
    assert revoted.status_code == 200 and revoted.json()["model"] == "parakeet"

    results = await client.get("/v1/ab/results", headers=account.headers)
    body = results.json()
    assert body["total"] == 1
    assert body["tally"][0]["model"] == "parakeet" and body["tally"][0]["wins"] == 1
    # Every enrolled session reports its live run state for the overview.
    assert body["sessions"] == [
        {
            "session_id": str(session.id),
            "votes": 1,
            "status": "succeeded",
            "candidates": 8,
            "expected": 8,
        }
    ]


async def test_votes_survive_regeneration(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    account = await make_account()
    session = await _enrolled(client, account)
    body = await _payload(client, account, session.id)
    first = body["utterances"][0]
    await client.post(
        f"/v1/sessions/{session.id}/ab/votes",
        headers=account.headers,
        json={
            "utterance_artifact_id": first["utterance_artifact_id"],
            "candidate_artifact_id": first["candidates"][0]["candidate_id"],
        },
    )
    old_ids = {c["candidate_id"] for c in first["candidates"]}

    requeued = await client.post(f"/v1/sessions/{session.id}/ab", headers=account.headers)
    assert requeued.status_code == 200
    await wait_for_job(session.id, "transcribe-ab", JobStatus.SUCCEEDED)

    body = await _payload(client, account, session.id)
    fresh = {c["candidate_id"] for c in body["utterances"][0]["candidates"]}
    assert fresh.isdisjoint(old_ids)
    # The vote row outlived its candidate (FK went NULL, tally intact).
    assert body["voted"] == 1 and body["utterances"][0]["vote"]["model"]
    results = (await client.get("/v1/ab/results", headers=account.headers)).json()
    assert results["total"] == 1


async def test_ab_guards_and_scoping(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    account = await make_account()
    other = await make_account()
    session = await _enrolled(client, account)
    body = await _payload(client, account, session.id)
    first, second = body["utterances"]

    for method, path, payload in (
        ("get", f"/v1/sessions/{session.id}/ab", None),
        ("post", f"/v1/sessions/{session.id}/ab", None),
        ("post", f"/v1/sessions/{session.id}/ab/votes", {
            "utterance_artifact_id": first["utterance_artifact_id"],
            "candidate_artifact_id": first["candidates"][0]["candidate_id"],
        }),
    ):
        response = await getattr(client, method)(
            path, headers=other.headers, **({"json": payload} if payload else {})
        )
        assert response.status_code == 404, path

    mismatched = await client.post(
        f"/v1/sessions/{session.id}/ab/votes",
        headers=account.headers,
        json={
            "utterance_artifact_id": first["utterance_artifact_id"],
            "candidate_artifact_id": second["candidates"][0]["candidate_id"],
        },
    )
    assert mismatched.status_code == 422

    unknown = await client.post(
        f"/v1/sessions/{session.id}/ab/votes",
        headers=account.headers,
        json={
            "utterance_artifact_id": first["utterance_artifact_id"],
            "candidate_artifact_id": str(uuid4()),
        },
    )
    assert unknown.status_code == 404
