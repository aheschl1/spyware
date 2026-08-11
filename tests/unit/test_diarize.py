"""Block/utterance assembly and the diarizer response contract: pure logic."""

import pytest

from processing.diarizer import DiarizerError, parse_response
from processing.pipelines.diarize import (
    Block,
    Utterance,
    blocks_from_spans,
    utterances_from_turns,
)


def _blocks(spans, gap=30_000, cap=1_800_000):
    return blocks_from_spans(spans, merge_gap_ms=gap, max_block_ms=cap)


def test_close_spans_merge_into_one_block() -> None:
    assert _blocks([(0, 20_000), (25_000, 60_000), (85_000, 90_000)]) == [
        Block(0, 90_000)
    ]


def test_long_silence_starts_a_new_block() -> None:
    assert _blocks([(0, 20_000), (120_000, 150_000)]) == [
        Block(0, 20_000),
        Block(120_000, 150_000),
    ]


def test_cap_closes_a_block_at_a_span_boundary() -> None:
    # Three 25-min spans, 10s apart: merging all would blow the 30-min cap, so
    # the block closes at the previous span's end — never mid-span.
    minute = 60_000
    spans = [
        (0, 25 * minute),
        (25 * minute + 10_000, 50 * minute),
        (50 * minute + 10_000, 75 * minute),
    ]
    blocks = _blocks(spans, cap=30 * minute)
    assert blocks == [
        Block(0, 25 * minute),
        Block(25 * minute + 10_000, 50 * minute),
        Block(50 * minute + 10_000, 75 * minute),
    ]


def test_empty_spans_yield_no_blocks() -> None:
    assert _blocks([]) == []


def _utterances(turns, gap=1_500, cap=30_000):
    return utterances_from_turns(turns, merge_gap_ms=gap, max_utterance_ms=cap)


def test_same_speaker_turns_within_gap_merge() -> None:
    assert _utterances([(0, 1_000, "a"), (2_000, 3_000, "a")]) == [
        Utterance(0, 3_000, "a", 2)
    ]


def test_gap_beyond_limit_splits_utterances() -> None:
    assert _utterances([(0, 1_000, "a"), (5_000, 6_000, "a")]) == [
        Utterance(0, 1_000, "a", 1),
        Utterance(5_000, 6_000, "a", 1),
    ]


def test_merge_spans_another_speakers_backchannel() -> None:
    # b's interjection sits inside a's pause; a's sentence must not split.
    assert _utterances(
        [(0, 2_000, "a"), (2_100, 2_400, "b"), (2_500, 4_000, "a")]
    ) == [
        Utterance(0, 4_000, "a", 2),
        Utterance(2_100, 2_400, "b", 1),
    ]


def test_speakers_never_merge_together() -> None:
    assert _utterances([(0, 1_000, "a"), (1_100, 2_000, "b")]) == [
        Utterance(0, 1_000, "a", 1),
        Utterance(1_100, 2_000, "b", 1),
    ]


def test_out_of_order_and_overlapping_turns_normalize() -> None:
    # Service order is untrusted; powerset overlap collapses.
    assert _utterances([(500, 2_000, "a"), (0, 1_000, "a")]) == [
        Utterance(0, 2_000, "a", 2)
    ]


def test_zero_length_turns_are_dropped() -> None:
    assert _utterances([(1_000, 1_000, "a"), (2_000, 1_500, "a")]) == []


def test_cap_refuses_a_merge_at_the_previous_turn_boundary() -> None:
    assert _utterances(
        [(0, 20_000, "a"), (21_000, 40_000, "a")], cap=30_000
    ) == [
        Utterance(0, 20_000, "a", 1),
        Utterance(21_000, 40_000, "a", 1),
    ]


def test_single_overlong_turn_splits_evenly() -> None:
    # 70s at a 30s cap: three near-equal pieces, never a sliver tail.
    pieces = _utterances([(0, 70_000, "a")], cap=30_000)
    assert [(u.start_ms, u.end_ms) for u in pieces] == [
        (0, 23_333),
        (23_333, 46_666),
        (46_666, 70_000),
    ]
    assert all(u.speaker == "a" for u in pieces)


def test_empty_turns_yield_no_utterances() -> None:
    assert _utterances([]) == []


def test_parse_response_round_trips() -> None:
    result = parse_response(
        {
            "turns": [{"start_ms": 0, "end_ms": 1200, "speaker": "SPEAKER_00"}],
            "embeddings": {"SPEAKER_00": [0.1, 0.2]},
            "model": "m",
        }
    )
    assert result.turns[0].end_ms == 1200
    assert result.embeddings == {"SPEAKER_00": [0.1, 0.2]}
    assert result.model == "m"


def test_parse_response_tolerates_missing_embeddings() -> None:
    result = parse_response({"turns": []})
    assert result.turns == () and result.embeddings == {}


@pytest.mark.parametrize(
    "body",
    [
        "not a dict",
        {},
        {"turns": [{"start_ms": 0}]},  # missing fields
        {"turns": [], "embeddings": "not-a-dict"},
    ],
)
def test_parse_response_rejects_malformed(body) -> None:
    with pytest.raises(DiarizerError):
        parse_response(body)


@pytest.mark.parametrize("bad_vector", ["not-a-vector", [1.0, "x"], [1.0, None]])
def test_malformed_embedding_is_dropped_not_fatal(bad_vector) -> None:
    """Turns gate transcription now; one bad vector must not kill the job."""
    result = parse_response(
        {
            "turns": [{"start_ms": 0, "end_ms": 1200, "speaker": "SPEAKER_00"}],
            "embeddings": {"SPEAKER_00": bad_vector, "SPEAKER_01": [0.5]},
            "model": "m",
        }
    )
    assert len(result.turns) == 1
    assert result.embeddings == {"SPEAKER_01": [0.5]}
