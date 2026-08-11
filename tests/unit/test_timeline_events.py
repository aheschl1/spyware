"""Timeline assembly: expansion, ordering, and window clipping. No stores."""

import random
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from api.timeline_events import assemble
from database.schema.artifacts import PipelineArtifact
from database.schema.sessions import RecordingSession

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _session(ended_after_ms: int | None = None) -> RecordingSession:
    return RecordingSession(
        id=uuid4(),
        user_id=uuid4(),
        started_at=_NOW,
        ended_at=None
        if ended_after_ms is None
        else _NOW + timedelta(milliseconds=ended_after_ms),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _artifact(
    pipeline: str,
    kind: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> PipelineArtifact:
    return PipelineArtifact(
        id=uuid4(),
        pipeline=pipeline,
        kind=kind,
        session_id=uuid4(),
        start_ms=start_ms,
        end_ms=end_ms,
        metadata=metadata or {},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _span(start_ms: int, end_ms: int, confidence: float = 0.9) -> PipelineArtifact:
    return _artifact(
        "speech-detect", "speech-span", start_ms, end_ms, {"confidence": confidence}
    )


def _transcript(
    start_ms: int, end_ms: int, text: str = "hello", speaker: str | None = "b0:SPEAKER_00"
) -> PipelineArtifact:
    return _artifact(
        "transcribe",
        "transcript",
        start_ms,
        end_ms,
        {"text": text, "chars": len(text), "model": "stt-1", "speaker": speaker},
    )


def _shape(events) -> list[tuple[str, int]]:
    return [(event.type, event.at_ms) for event in events]


def test_orders_session_frames_spans_and_transcripts() -> None:
    span_a, span_b = _span(0, 5_000), _span(5_000, 9_000, confidence=0.8)
    rows = [span_a, _transcript(0, 5_000), span_b, _transcript(5_000, 9_000)]

    events = assemble(_session(ended_after_ms=9_000), rows)

    # Adjacent spans: A ends before B starts at 5000; the transcript follows
    # its span's start; session-end wins the tie at 9000.
    assert _shape(events) == [
        ("session-start", 0),
        ("speech-start", 0),
        ("transcript", 0),
        ("speech-end", 5_000),
        ("speech-start", 5_000),
        ("transcript", 5_000),
        ("speech-end", 9_000),
        ("session-end", 9_000),
    ]
    assert events[0].started_at == _NOW
    assert events[-1].ended_at == _NOW + timedelta(milliseconds=9_000)
    assert events[1].artifact_id == span_a.id and events[1].confidence == 0.9
    assert events[3].artifact_id == span_a.id
    assert events[4].artifact_id == span_b.id and events[4].confidence == 0.8


def test_open_session_has_no_end_frame_and_unknown_kinds_are_skipped() -> None:
    rows = [
        _artifact("speech-detect", "speech-map", metadata={"spans": 0}),
        _artifact("session-stats", "session-stats"),
        _artifact("future", "sentiment", 1_000, 2_000),
    ]
    assert _shape(assemble(_session(), rows)) == [("session-start", 0)]


def test_window_keeps_only_events_positioned_inside_it() -> None:
    rows = [_span(3_000, 8_000), _transcript(3_000, 8_000)]

    inside = assemble(_session(), rows, from_ms=4_000, to_ms=10_000)
    assert _shape(inside) == [("speech-end", 8_000)]  # start/transcript/session-start clipped

    assert assemble(_session(), rows, from_ms=4_000, to_ms=8_000) == []  # half-open
    assert assemble(_session(), rows, from_ms=10_000, to_ms=20_000) == []


def test_adjacent_windows_partition_the_stream() -> None:
    session = _session(ended_after_ms=9_000)
    rows = [_span(0, 5_000), _transcript(0, 5_000), _span(5_000, 9_000), _transcript(5_000, 9_000)]

    first = assemble(session, rows, from_ms=0, to_ms=5_000)
    second = assemble(session, rows, from_ms=5_000, to_ms=10_000)
    assert first + second == assemble(session, rows)


def test_orphan_transcript_still_appears() -> None:
    events = assemble(_session(), [_transcript(2_000, 4_000, text="alone")])
    assert _shape(events) == [("session-start", 0), ("transcript", 2_000)]
    assert events[1].text == "alone" and events[1].end_ms == 4_000


def test_speaker_map_stamps_global_identity_on_transcripts() -> None:
    resolved = _transcript(0, 1_000, speaker="b0:SPEAKER_00")
    unresolved = _transcript(2_000, 3_000, speaker="b0:SPEAKER_01")
    speaker_id = uuid4()

    events = assemble(
        _session(),
        [resolved, unresolved],
        speakers={"b0:SPEAKER_00": (speaker_id, "Mom")},
    )

    stamped, plain = events[1:]
    assert stamped.speaker_id == speaker_id and stamped.speaker_name == "Mom"
    assert stamped.speaker == "b0:SPEAKER_00"  # provenance survives
    assert plain.speaker_id is None and plain.speaker_name is None

    # No map at all: nothing is stamped, nothing breaks.
    bare = assemble(_session(), [resolved])
    assert bare[1].speaker_id is None and bare[1].speaker_name is None


def test_speaker_map_names_can_be_null() -> None:
    # An unlabeled cluster still resolves the id; the name stays null.
    events = assemble(
        _session(),
        [_transcript(0, 1_000, speaker="b0:SPEAKER_00")],
        speakers={"b0:SPEAKER_00": (uuid4(), None)},
    )
    assert events[1].speaker_id is not None and events[1].speaker_name is None


def test_transcript_speaker_and_missing_metadata() -> None:
    attributed = _artifact(
        "transcribe",
        "transcript",
        0,
        1_000,
        {"text": "hi", "chars": 2, "speaker": "b0:SPEAKER_01"},
    )
    bare = _artifact("transcribe", "transcript", 2_000, 3_000)

    spoken, empty = assemble(_session(), [attributed, bare])[1:]

    assert spoken.speaker == "b0:SPEAKER_01" and spoken.chars == 2
    assert empty.text == "" and empty.chars == 0 and empty.speaker is None
    assert empty.model is None


def test_missing_confidence_and_null_spans_are_tolerated() -> None:
    no_confidence = _artifact("speech-detect", "speech-span", 0, 1_000)
    null_span = _artifact("speech-detect", "speech-span")

    events = assemble(_session(), [no_confidence, null_span])

    assert _shape(events) == [("session-start", 0), ("speech-start", 0), ("speech-end", 1_000)]
    assert events[1].confidence is None


def test_order_is_deterministic_under_input_shuffle() -> None:
    session = _session(ended_after_ms=9_000)
    # Includes a duplicated span (reprocessing) sharing its whole range.
    rows = [_span(0, 5_000), _span(0, 5_000), _transcript(0, 5_000), _span(5_000, 9_000)]
    expected = assemble(session, rows)

    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)
    assert assemble(session, shuffled) == expected
    assert assemble(session, list(reversed(rows))) == expected


def test_end_frame_never_precedes_the_start_frame() -> None:
    # A device clock stepping backwards must not produce a negative position.
    session = RecordingSession(
        id=uuid4(),
        user_id=uuid4(),
        started_at=_NOW,
        ended_at=_NOW - timedelta(seconds=5),
        created_at=_NOW,
        updated_at=_NOW,
    )
    assert _shape(assemble(session, [])) == [("session-start", 0), ("session-end", 0)]


def test_no_artifacts_serves_only_session_frames() -> None:
    assert _shape(assemble(_session(), [])) == [("session-start", 0)]
