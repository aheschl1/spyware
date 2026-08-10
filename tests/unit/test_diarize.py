"""Block assembly and the diarizer response contract: pure logic, no service."""

import pytest

from processing.diarizer import DiarizerError, parse_response
from processing.pipelines.diarize import Block, blocks_from_spans


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
        {"turns": [], "embeddings": {"a": "not-a-vector"}},
        {"turns": [], "embeddings": {"a": [1.0, "x"]}},
    ],
)
def test_parse_response_rejects_malformed(body) -> None:
    with pytest.raises(DiarizerError):
        parse_response(body)
