"""The purity audit end to end: a mixed label splits into sub-labels.

A ≥1s session trips the stub diarizer's split scenario
(``stub_audio_services.STUB_SPLIT_TURNS``): SPEAKER_00's per-turn embeddings
form two orthogonal voice groups — the shape of a label pyannote wrongly gave
to two people — while SPEAKER_01 stays an old-style turn with no per-turn
fields. The diarize tier must mint ``SPEAKER_00.0``/``.1`` before publishing,
so utterances, transcripts, voice-prints, and clustering all see the
corrected labels; the old-style label must flow through untouched.
"""

import asyncio
import time

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
from tests.wav import wav_bytes


async def _long_session(account: Account):
    """One second of tone: enough bytes to trip the stub's split response."""
    session = await make_session(account)
    await ingest(session.id, wav_bytes(seconds=1.0), duration_ms=1_000)
    async with DatabasePipe() as pipe:
        await pipe.sessions.end(session.id)
    return session


async def test_mixed_label_splits_into_sub_labels(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    account = await make_account()
    session = await _long_session(account)

    job = await wait_for_job(session.id, "diarize", JobStatus.SUCCEEDED)
    assert job.result == {"blocks": 1, "turns": 5, "utterances": 3, "speakers": 3}

    async with DatabasePipe() as pipe:
        turns = await pipe.artifacts.list_for_session(session.id, kind="speaker-turn")
        utterances = await pipe.artifacts.list_for_session(session.id, kind="utterance")
        embeddings = await pipe.artifacts.list_for_session(
            session.id, kind="speaker-embedding"
        )
        vectors = await pipe.embeddings.list_for_session(session.id)

    # Turns: the mixed label's four turns are re-labeled by voice group
    # (.0 = more clean talk), the old-style label is untouched. (Equal
    # start_ms rows order by random id: sort.)
    assert sorted(
        (t.start_ms, t.end_ms, t.metadata["speaker"], t.metadata["overlap_ms"])
        for t in turns
    ) == [
        (0, 150, "b0:SPEAKER_01", 0),
        (0, 250, "b0:SPEAKER_00.0", 50),
        (250, 500, "b0:SPEAKER_00.0", 0),
        (500, 700, "b0:SPEAKER_00.1", 0),
        (750, 950, "b0:SPEAKER_00.1", 0),
    ]

    # Utterances merge per final label — the two voices never share one.
    assert sorted(
        (u.start_ms, u.end_ms, u.metadata["speaker"], u.metadata["overlap_ms"])
        for u in utterances
    ) == [
        (0, 150, "b0:SPEAKER_01", 0),
        (0, 500, "b0:SPEAKER_00.0", 50),
        (500, 950, "b0:SPEAKER_00.1", 0),
    ]

    # Voice-prints: one per final label. The sub-labels pool their own clean
    # turn vectors (never the blended aggregate); the old-style label falls
    # back to the service aggregate and records no clean talk at all.
    by_label = {e.metadata["speaker"]: e for e in embeddings}
    assert set(by_label) == {"b0:SPEAKER_00.0", "b0:SPEAKER_00.1", "b0:SPEAKER_01"}
    stored = {v.artifact_id: v.embedding for v in vectors}

    zero = by_label["b0:SPEAKER_00.0"]
    assert zero.metadata["split_of"] == "SPEAKER_00"
    assert zero.metadata["talk_ms"] == 500 and zero.metadata["clean_talk_ms"] == 450
    assert [round(x) for x in stored[zero.id]] == [1, 0, 0, 0]

    one = by_label["b0:SPEAKER_00.1"]
    assert one.metadata["split_of"] == "SPEAKER_00"
    assert one.metadata["talk_ms"] == 400 and one.metadata["clean_talk_ms"] == 400
    assert [round(x) for x in stored[one.id]] == [0, 1, 0, 0]

    other = by_label["b0:SPEAKER_01"]
    assert "split_of" not in other.metadata
    assert "clean_talk_ms" not in other.metadata
    assert list(stored[other.id]) == [0.0, 0.0, 1.0, 0.0]


async def test_split_labels_flow_into_transcripts_and_clusters(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    account = await make_account()
    session = await _long_session(account)

    await wait_for_job(session.id, "diarize", JobStatus.SUCCEEDED)
    await wait_for_job(session.id, "speaker-cluster", JobStatus.SUCCEEDED)
    await wait_for_job(session.id, "transcribe", JobStatus.SUCCEEDED)

    async with DatabasePipe() as pipe:
        utterances = await pipe.artifacts.list_for_session(session.id, kind="utterance")
    transcripts = []
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and len(transcripts) < len(utterances):
        async with DatabasePipe() as pipe:
            transcripts = await pipe.artifacts.list_for_session(
                session.id, kind="transcript"
            )
        await asyncio.sleep(0.1)

    # Transcripts attribute to the sub-label and carry its crosstalk.
    by_speaker = {t.metadata["speaker"]: t for t in transcripts}
    assert set(by_speaker) == {"b0:SPEAKER_00.0", "b0:SPEAKER_00.1", "b0:SPEAKER_01"}
    assert by_speaker["b0:SPEAKER_00.0"].metadata["overlap_ms"] == 50
    assert by_speaker["b0:SPEAKER_00.1"].metadata["overlap_ms"] == 0

    # Clustering: the two sub-label prints pass the clean-talk gate and land
    # in different clusters (orthogonal voices); the old-style label reported
    # no clean time, so the gate falls back to its talk_ms and it clusters
    # too — as its own voice, orthogonal to both sub-labels.
    listing = await client.get(
        f"/v1/sessions/{session.id}/speakers", headers=account.headers
    )
    assert listing.status_code == 200
    by_label = {
        label: row for row in listing.json() for label in row["local_labels"]
    }
    assert set(by_label) == {"b0:SPEAKER_00.0", "b0:SPEAKER_00.1", "b0:SPEAKER_01"}
    zero, one = by_label["b0:SPEAKER_00.0"], by_label["b0:SPEAKER_00.1"]
    other = by_label["b0:SPEAKER_01"]
    assert zero["speaker_id"] is not None and one["speaker_id"] is not None
    assert zero["speaker_id"] != one["speaker_id"]
    assert other["speaker_id"] is not None
    assert other["speaker_id"] not in (zero["speaker_id"], one["speaker_id"])
