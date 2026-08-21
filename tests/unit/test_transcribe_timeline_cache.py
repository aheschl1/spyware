"""The transcribe tier's per-session timeline reuse.

load_timeline costs several times what transcribing a short utterance does,
and consecutive jobs usually share a session. Reuse is only safe while the
cached timeline already covers the range: byte_range() clamps to the audio it
knows about, so a stale one truncates the clip rather than failing.
"""

from collections import OrderedDict
from uuid import uuid4

import pytest

from processing.pipelines import transcribe as tier
from services import stitch
from services.timeline import SessionTimeline


def _timeline(session_id, total_ms: int) -> SessionTimeline:
    data_bytes = total_ms * 16_000 * 2 // 1000
    return SessionTimeline(
        session_id=session_id,
        plan=stitch.StitchPlan(
            pieces=(
                stitch.Piece(
                    object_key="k", start=stitch.WAV_HEADER_BYTES, data_length=data_bytes
                ),
            ),
            total_size=stitch.WAV_HEADER_BYTES + data_bytes,
        ),
        sample_rate_hz=16_000,
        channels=1,
    )


@pytest.fixture
def pipeline(monkeypatch):
    """A pipeline whose loader is a counter, with no Transcriber or DB."""
    p = tier.TranscribePipeline()
    p._timelines = OrderedDict()  # setup() also builds a Transcriber we do not need
    calls = []

    async def fake_load(session_id):
        calls.append(session_id)
        return _timeline(session_id, total_ms=10_000)

    monkeypatch.setattr(tier.timeline, "load_timeline", fake_load)
    return p, calls


async def test_second_job_in_range_reuses_the_timeline(pipeline) -> None:
    p, calls = pipeline
    session = uuid4()
    assert await p._timeline_for(session, 5_000) is not None
    assert await p._timeline_for(session, 9_000) is not None
    assert len(calls) == 1


async def test_a_range_past_the_cached_end_reloads(pipeline) -> None:
    p, calls = pipeline
    session = uuid4()
    await p._timeline_for(session, 5_000)
    # The session grew; the cached timeline stops at 10s and would truncate.
    await p._timeline_for(session, 12_000)
    assert len(calls) == 2


async def test_a_range_exactly_at_the_cached_end_is_reused(pipeline) -> None:
    p, calls = pipeline
    session = uuid4()
    await p._timeline_for(session, 5_000)
    await p._timeline_for(session, 10_000)
    assert len(calls) == 1


async def test_interleaved_sessions_stay_cached(pipeline) -> None:
    p, calls = pipeline
    a, b = uuid4(), uuid4()
    for session in (a, b, a, b):
        await p._timeline_for(session, 1_000)
    assert len(calls) == 2


async def test_the_cache_is_bounded(pipeline) -> None:
    p, calls = pipeline
    sessions = [uuid4() for _ in range(tier._TIMELINE_CACHE + 1)]
    for session in sessions:
        await p._timeline_for(session, 1_000)
    assert len(p._timelines) == tier._TIMELINE_CACHE
    # The oldest was evicted, so it reloads.
    await p._timeline_for(sessions[0], 1_000)
    assert len(calls) == len(sessions) + 1


async def test_a_missing_session_is_not_cached(pipeline, monkeypatch) -> None:
    p, calls = pipeline

    async def gone(session_id):
        calls.append(session_id)
        return None

    monkeypatch.setattr(tier.timeline, "load_timeline", gone)
    session = uuid4()
    assert await p._timeline_for(session, 1_000) is None
    assert await p._timeline_for(session, 1_000) is None
    assert len(calls) == 2
