"""The sound-span tier end to end: windows merged into long single-class
spans, the map summarised, and the spans on the timeline."""

import httpx
import pytest

from database.pipe import DatabasePipe
from database.schema.artifacts import ArtifactCreate
from database.schema.jobs import JobStatus
from tests.e2e.conftest import Account, ingest, make_account, make_session, wait_for_job
from tests.e2e.stub_audio_services import STUB_TAGGER_MODEL
from tests.wav import wav_bytes

WINDOW_MS = 10_000
HOP_MS = 5_000


async def _ended_session(account: Account, count: int = 3):
    session = await make_session(account)
    for _ in range(count):
        await ingest(session.id, wav_bytes(seconds=0.1), duration_ms=100)
    async with DatabasePipe() as pipe:
        await pipe.sessions.end(session.id)
    return session


async def _seed_tagged_session(account: Account, windows: list[dict[str, float]]):
    """A session carrying hand-built audio-tag windows on the production grid.

    Left unended on purpose: audio-tag discovers ended sessions, so an open one
    is never republished over. The sound-span tier discovers the map artifact
    and does not care about ``ended_at``.
    """
    session = await make_session(account)
    async with DatabasePipe() as pipe:
        await pipe.artifacts.create_many(
            [
                ArtifactCreate(
                    pipeline="audio-tag",
                    kind="audio-tag",
                    session_id=session.id,
                    start_ms=index * HOP_MS,
                    end_ms=index * HOP_MS + WINDOW_MS,
                    metadata={
                        "labels": [
                            {"label": label, "score": score}
                            for label, score in scores.items()
                        ],
                        "model": STUB_TAGGER_MODEL,
                    },
                )
                for index, scores in enumerate(windows)
            ]
        )
        # The map is the completion marker: writing it last is what mints the
        # sound-span job.
        await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="audio-tag",
                kind="audio-tag-map",
                session_id=session.id,
                metadata={"windows": len(windows), "models": {"tagger": STUB_TAGGER_MODEL}},
            )
        )
    return session


async def test_stub_windows_merge_into_one_span(worker: None, clean_state) -> None:
    """The full chain from ingested audio: Music held across both stub windows
    and became a span; Speech opened at exactly `enter` but never got a second
    window, so `min_windows` dropped it."""
    account = await make_account()
    session = await _ended_session(account)

    job = await wait_for_job(session.id, "sound-span", JobStatus.SUCCEEDED)
    assert job.result == {"spans": 1, "classes": 1, "windows": 2}

    async with DatabasePipe() as pipe:
        spans = await pipe.artifacts.list_for_session(session.id, kind="sound-span")
        maps = await pipe.artifacts.list_for_session(session.id, kind="sound-span-map")

    (span,) = spans
    assert (span.start_ms, span.end_ms) == (0, 300)
    assert span.metadata["label"] == "Music"
    assert span.metadata["peak"] == 0.9
    assert span.metadata["mean"] == pytest.approx(0.85)
    assert span.metadata["windows"] == 2
    assert span.metadata["model"] == STUB_TAGGER_MODEL
    # Postgres-only: the row is the store.
    assert span.bucket is None and span.object_key is None

    (span_map,) = maps
    assert span_map.metadata["spans"] == 1
    assert span_map.metadata["windows"] == 2
    assert span_map.metadata["classes"] == [
        {"label": "Music", "spans": 1, "total_ms": 300, "peak": 0.9}
    ]
    # Against in-process settings, not a literal: conftest pins stub-scale
    # thresholds and the worker must have used the same ones.
    from processing.config import get_settings

    assert span_map.metadata["params"]["enter"] == get_settings().sound_span_enter_score
    assert span.links["audio_tag_map"] == span_map.metadata["source_map"]


async def test_spans_merge_bridge_and_exclude_low_confidence(
    worker: None, clean_state
) -> None:
    """A realistic grid: two classes overlapping in time, a bridged dropout in
    one of them, and a class that never reaches `enter`."""
    account = await make_account()
    session = await _seed_tagged_session(
        account,
        [
            {"Music": 0.9, "Fan": 0.25},
            {"Music": 0.9, "Fan": 0.25},
            {"Fan": 0.25},  # Music drops out for one window; the span bridges it
            {"Music": 0.9, "Speech": 0.8, "Fan": 0.25},
            {"Music": 0.9, "Speech": 0.8, "Fan": 0.25},
            {"Speech": 0.8, "Fan": 0.25},
        ],
    )

    job = await wait_for_job(session.id, "sound-span", JobStatus.SUCCEEDED)
    assert job.result == {"spans": 2, "classes": 2, "windows": 6}

    async with DatabasePipe() as pipe:
        spans = await pipe.artifacts.list_for_session(session.id, kind="sound-span")
        (span_map,) = await pipe.artifacts.list_for_session(
            session.id, kind="sound-span-map"
        )

    shape = [(s.metadata["label"], s.start_ms, s.end_ms) for s in spans]
    assert shape == [
        ("Music", 0, 4 * HOP_MS + WINDOW_MS),
        ("Speech", 3 * HOP_MS, 5 * HOP_MS + WINDOW_MS),
    ]
    # Different classes overlap; the dropout window is inside Music's span but
    # is not a member of it.
    music, speech = spans
    assert speech.start_ms < music.end_ms
    assert music.metadata["windows"] == 4

    # Fan sat above sustain but never reached enter, so it has no span at all.
    assert "Fan" not in {s.metadata["label"] for s in spans}
    assert [c["label"] for c in span_map.metadata["classes"]] == ["Music", "Speech"]


async def test_spans_appear_on_the_timeline(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    session = await _seed_tagged_session(
        account,
        [
            {"Music": 0.9},
            {"Music": 0.9, "Speech": 0.7},
            {"Speech": 0.7},
        ],
    )
    await wait_for_job(session.id, "sound-span", JobStatus.SUCCEEDED)

    response = await client.get(
        f"/v1/sessions/{session.id}/timeline", headers=account.headers
    )
    assert response.status_code == 200
    events = [e for e in response.json()["items"] if e["type"] == "sound-span"]
    assert [(e["label"], e["start_ms"], e["end_ms"]) for e in events] == [
        ("Music", 0, HOP_MS + WINDOW_MS),
        ("Speech", HOP_MS, 2 * HOP_MS + WINDOW_MS),
    ]
    assert events[0]["peak"] == 0.9
    assert events[0]["windows"] == 2
    assert events[0]["model"] == STUB_TAGGER_MODEL
