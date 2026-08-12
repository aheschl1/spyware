"""Block/utterance assembly, the purity audit, and the diarizer contract."""

from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from database.schema.jobs import Job, JobStatus
from processing.config import ProcessingSettings
from processing.diarizer import (
    DiarizationResult,
    DiarizerError,
    Turn,
    parse_response,
)
from processing.pipelines.diarize import (
    Block,
    DiarizePipeline,
    Utterance,
    blocks_from_spans,
    split_labels,
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


# --------------------------------------------------------- crosstalk gating


def _gated(turns, crosstalk_turns=None, budget=300):
    return utterances_from_turns(
        turns,
        merge_gap_ms=1_500,
        max_utterance_ms=30_000,
        crosstalk_max_ms=budget,
        crosstalk_turns=crosstalk_turns,
    )


def test_interjection_in_gap_refuses_the_merge() -> None:
    # b holds 500ms of a's pause: merging would put b's words in a's span.
    assert _gated(
        [(0, 2_000, "a"), (2_100, 2_600, "b"), (2_700, 4_000, "a")]
    ) == [
        Utterance(0, 2_000, "a", 1),
        Utterance(2_100, 2_600, "b", 1),
        Utterance(2_700, 4_000, "a", 1),
    ]


def test_backchannel_within_budget_still_merges() -> None:
    assert _gated(
        [(0, 2_000, "a"), (2_100, 2_350, "b"), (2_500, 4_000, "a")]
    ) == [
        Utterance(0, 4_000, "a", 2),
        Utterance(2_100, 2_350, "b", 1),
    ]


def test_subminimum_interjection_gates_via_raw_turns() -> None:
    # b's turn was too short to keep as an artifact, but its speech is still
    # in the audio: the raw list must gate the merge anyway.
    assert _gated(
        [(0, 2_000, "a"), (2_700, 4_000, "a")],
        crosstalk_turns=[
            (0, 2_000, "a"),
            (2_100, 2_600, "b"),
            (2_700, 4_000, "a"),
        ],
    ) == [
        Utterance(0, 2_000, "a", 1),
        Utterance(2_700, 4_000, "a", 1),
    ]


def test_overlap_during_the_turn_itself_does_not_gate() -> None:
    # b talks over a's first turn, not in the pause: boundaries can't excise
    # simultaneous speech, so the merge must not be punished for it.
    assert _gated(
        [(0, 2_000, "a"), (500, 1_900, "b"), (2_500, 4_000, "a")]
    ) == [
        Utterance(0, 4_000, "a", 2),
        Utterance(500, 1_900, "b", 1),
    ]


def test_stacked_foreign_turns_count_once() -> None:
    # b and c overlap each other inside the gap: union 250ms, not 500.
    assert _gated(
        [
            (0, 2_000, "a"),
            (2_100, 2_350, "b"),
            (2_150, 2_350, "c"),
            (2_500, 4_000, "a"),
        ]
    ) == [
        Utterance(0, 4_000, "a", 2),
        Utterance(2_100, 2_350, "b", 1),
        Utterance(2_150, 2_350, "c", 1),
    ]


def test_none_budget_preserves_unconditional_merging() -> None:
    assert utterances_from_turns(
        [(0, 2_000, "a"), (2_100, 2_600, "b"), (2_700, 4_000, "a")],
        merge_gap_ms=1_500,
        max_utterance_ms=30_000,
    ) == [
        Utterance(0, 4_000, "a", 2),
        Utterance(2_100, 2_600, "b", 1),
    ]


# ----------------------------------------------------------- purity audit


def _turn(start, end, speaker, embedding=None, clean=None, overlap=0):
    return Turn(
        start, end, speaker, overlap_ms=overlap, clean_ms=clean, embedding=embedding
    )


def _split(turns, distance=0.7, floor=2_000, vote_floor=1_000):
    return split_labels(
        turns,
        distance=distance,
        split_min_clean_ms=floor,
        turn_min_clean_ms=vote_floor,
    )


A, A2 = (1.0, 0.0), (0.995, 0.1)  # one voice, slight spread
B, B2 = (0.0, 1.0), (0.1, 0.995)  # a clearly different voice
FAR = (-1.0, 0.0)  # far from both, nearer to B


def test_two_voices_under_one_label_split() -> None:
    labels = _split(
        [
            _turn(0, 3_000, "SPEAKER_10", A, 3_000),
            _turn(4_000, 7_000, "SPEAKER_10", A2, 3_000),
            _turn(8_000, 11_000, "SPEAKER_10", B, 3_000),
            _turn(12_000, 15_000, "SPEAKER_10", B2, 3_000),
        ]
    )
    assert labels == [
        "SPEAKER_10.0",
        "SPEAKER_10.0",
        "SPEAKER_10.1",
        "SPEAKER_10.1",
    ]


def test_one_voice_keeps_its_plain_label() -> None:
    labels = _split(
        [
            _turn(0, 3_000, "SPEAKER_10", A, 3_000),
            _turn(4_000, 7_000, "SPEAKER_10", A2, 3_000),
        ]
    )
    assert labels == ["SPEAKER_10", "SPEAKER_10"]


def test_sub_labels_number_by_descending_clean_talk() -> None:
    labels = _split(
        [
            _turn(0, 3_000, "SPEAKER_10", A, 2_500),
            _turn(8_000, 11_000, "SPEAKER_10", B, 3_000),
            _turn(12_000, 15_000, "SPEAKER_10", B2, 3_000),
        ]
    )
    # B's group carries more clean talk: it becomes .0 despite voting later.
    assert labels == ["SPEAKER_10.1", "SPEAKER_10.0", "SPEAKER_10.0"]


def test_below_floor_group_folds_into_nearest_surviving() -> None:
    labels = _split(
        [
            _turn(0, 3_000, "SPEAKER_10", A, 3_000),
            _turn(4_000, 7_000, "SPEAKER_10", A2, 3_000),
            _turn(8_000, 11_000, "SPEAKER_10", B, 3_000),
            _turn(12_000, 15_000, "SPEAKER_10", B2, 3_000),
            _turn(16_000, 17_500, "SPEAKER_10", FAR, 1_500),
        ]
    )
    # FAR is its own cluster but under the floor; nearest centroid is B's.
    assert labels == [
        "SPEAKER_10.0",
        "SPEAKER_10.0",
        "SPEAKER_10.1",
        "SPEAKER_10.1",
        "SPEAKER_10.1",
    ]


def test_voteless_turn_attaches_to_temporally_nearest_group() -> None:
    labels = _split(
        [
            _turn(0, 3_000, "SPEAKER_10", A, 3_000),
            _turn(4_000, 7_000, "SPEAKER_10", A2, 3_000),
            _turn(8_000, 11_000, "SPEAKER_10", B, 3_000),
            _turn(12_000, 15_000, "SPEAKER_10", B2, 3_000),
            _turn(11_200, 11_800, "SPEAKER_10"),  # no embedding: too short
        ]
    )
    assert labels[-1] == "SPEAKER_10.1"


def test_below_vote_floor_embeddings_do_not_vote() -> None:
    labels = _split(
        [
            _turn(0, 3_000, "SPEAKER_10", A, 3_000),
            _turn(4_000, 7_000, "SPEAKER_10", A2, 3_000),
            _turn(8_000, 8_500, "SPEAKER_10", B, 500),  # under vote floor
        ]
    )
    assert labels == ["SPEAKER_10", "SPEAKER_10", "SPEAKER_10"]


def test_single_voter_cannot_split() -> None:
    labels = _split(
        [
            _turn(0, 3_000, "SPEAKER_10", A, 3_000),
            _turn(4_000, 7_000, "SPEAKER_10"),
        ]
    )
    assert labels == ["SPEAKER_10", "SPEAKER_10"]


def test_labels_audit_independently() -> None:
    labels = _split(
        [
            _turn(0, 3_000, "SPEAKER_00", A, 3_000),
            _turn(4_000, 7_000, "SPEAKER_00", B, 3_000),
            _turn(8_000, 11_000, "SPEAKER_01", A, 3_000),
            _turn(12_000, 15_000, "SPEAKER_01", A2, 3_000),
        ]
    )
    assert labels == [
        "SPEAKER_00.0",
        "SPEAKER_00.1",
        "SPEAKER_01",
        "SPEAKER_01",
    ]


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


@pytest.mark.parametrize(
    "bad_vector",
    ["not-a-vector", [1.0, "x"], [1.0, None], [1.0, float("nan")], [float("inf")]],
)
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


def test_parse_response_round_trips_per_turn_fields() -> None:
    result = parse_response(
        {
            "turns": [
                {
                    "start_ms": 0,
                    "end_ms": 1200,
                    "speaker": "SPEAKER_00",
                    "overlap_ms": 150,
                    "clean_ms": 1050,
                    "embedding": [0.1, 0.2],
                }
            ],
            "embeddings": {},
            "model": "m",
        }
    )
    turn = result.turns[0]
    assert turn.overlap_ms == 150
    assert turn.clean_ms == 1050
    assert turn.embedding == (0.1, 0.2)


def test_parse_response_defaults_per_turn_fields() -> None:
    """An older service omits the fields; the tier must behave as before."""
    turn = parse_response(
        {"turns": [{"start_ms": 0, "end_ms": 1200, "speaker": "SPEAKER_00"}]}
    ).turns[0]
    assert turn.overlap_ms == 0
    assert turn.clean_ms is None
    assert turn.embedding is None


@pytest.mark.parametrize(
    "bad_vector", ["not-a-vector", [1.0, float("nan")], [float("inf")], 7]
)
def test_malformed_turn_embedding_degrades_to_none(bad_vector) -> None:
    turn = parse_response(
        {
            "turns": [
                {
                    "start_ms": 0,
                    "end_ms": 1200,
                    "speaker": "SPEAKER_00",
                    "clean_ms": 1200,
                    "embedding": bad_vector,
                }
            ]
        }
    ).turns[0]
    assert turn.embedding is None
    assert turn.clean_ms == 1200


# ------------------------------------------------- artifact/embedding rows


def _pipeline() -> DiarizePipeline:
    pipeline = DiarizePipeline()
    pipeline._settings = ProcessingSettings(_env_file=None)
    return pipeline


def _job() -> Job:
    now = datetime.now(timezone.utc)
    return Job(
        id=uuid4(),
        pipeline="diarize",
        session_id=uuid4(),
        artifact_id=uuid4(),
        priority=0,
        status=JobStatus.RUNNING,
        attempts=1,
        max_attempts=5,
        run_at=now,
        created_at=now,
        updated_at=now,
    )


def _result(turns, embeddings) -> DiarizationResult:
    return DiarizationResult(
        turns=tuple(turns), embeddings=embeddings, model="m"
    )


def test_voice_print_pools_clean_turns_not_the_aggregate() -> None:
    turns = [
        _turn(0, 3_000, "SPEAKER_00", (1.0, 0.0), 3_000),
        _turn(4_000, 7_000, "SPEAKER_00", (1.0, 0.0), 3_000),
    ]
    rows = _pipeline()._embedding_rows(
        _job(), Block(0, 7_000), _result(turns, {"SPEAKER_00": [0.0, 1.0]}), turns
    )
    (artifact, vector), = rows
    assert vector == pytest.approx([1.0, 0.0])
    assert artifact.metadata["speaker"] == "b0:SPEAKER_00"
    assert artifact.metadata["talk_ms"] == 6_000
    assert artifact.metadata["clean_talk_ms"] == 6_000
    assert "split_of" not in artifact.metadata


def test_voice_print_weights_by_clean_talk() -> None:
    turns = [
        _turn(0, 3_000, "SPEAKER_00", (1.0, 0.0), 3_000),
        _turn(4_000, 5_000, "SPEAKER_00", (0.0, 1.0), 1_000),
    ]
    (_, vector), = _pipeline()._embedding_rows(
        _job(), Block(0, 5_000), _result(turns, {}), turns
    )
    assert vector == pytest.approx([0.9487, 0.3162], abs=1e-3)


def test_voice_print_falls_back_to_aggregate_without_clean_turns() -> None:
    # All-overlap label: clean time is known (0) but nothing is poolable —
    # the aggregate keeps the label visible, and clean_talk_ms=0 lets the
    # clustering gate skip exactly this print.
    turns = [_turn(0, 3_000, "SPEAKER_00", None, 0, overlap=3_000)]
    (artifact, vector), = _pipeline()._embedding_rows(
        _job(), Block(0, 3_000), _result(turns, {"SPEAKER_00": [0.5, 0.5]}), turns
    )
    assert vector == [0.5, 0.5]
    assert artifact.metadata["clean_talk_ms"] == 0


def test_degraded_service_writes_no_clean_talk_ms() -> None:
    # Old service: no clean_ms anywhere. Writing clean_talk_ms=0 would make
    # the clustering gate skip every print — it must be absent instead.
    turns = [_turn(0, 3_000, "SPEAKER_00")]
    (artifact, vector), = _pipeline()._embedding_rows(
        _job(), Block(0, 3_000), _result(turns, {"SPEAKER_00": [0.5, 0.5]}), turns
    )
    assert vector == [0.5, 0.5]
    assert "clean_talk_ms" not in artifact.metadata
    assert artifact.metadata["talk_ms"] == 3_000


def test_split_labels_record_their_origin() -> None:
    original = [
        _turn(0, 3_000, "SPEAKER_00", (1.0, 0.0), 3_000),
        _turn(4_000, 7_000, "SPEAKER_00", (0.0, 1.0), 3_000),
    ]
    final = [
        replace(original[0], speaker="SPEAKER_00.0"),
        replace(original[1], speaker="SPEAKER_00.1"),
    ]
    rows = _pipeline()._embedding_rows(
        _job(), Block(0, 7_000), _result(original, {}), final
    )
    assert [artifact.metadata["speaker"] for artifact, _ in rows] == [
        "b0:SPEAKER_00.0",
        "b0:SPEAKER_00.1",
    ]
    assert all(
        artifact.metadata["split_of"] == "SPEAKER_00" for artifact, _ in rows
    )


def test_utterance_overlap_is_prorated_from_turns() -> None:
    pipeline = _pipeline()
    job = _job()
    turns = [
        _turn(0, 2_000, "SPEAKER_00", overlap=400, clean=1_600),
        _turn(2_500, 4_500, "SPEAKER_00", overlap=100, clean=1_900),
    ]
    turn_rows = pipeline._turn_artifacts(job, Block(0, 5_000), turns)
    assert [row.metadata["overlap_ms"] for row in turn_rows] == [400, 100]
    raw = [(row.start_ms, row.end_ms, row.metadata["speaker"]) for row in turn_rows]
    utterance_rows = pipeline._utterance_artifacts(job, turn_rows, raw)
    (utterance,) = utterance_rows
    assert utterance.start_ms == 0 and utterance.end_ms == 4_500
    assert utterance.metadata["overlap_ms"] == 500


def test_overcap_split_prorates_overlap_between_pieces() -> None:
    pipeline = _pipeline()
    job = _job()
    turns = [_turn(0, 60_000, "SPEAKER_00", overlap=600, clean=59_400)]
    turn_rows = pipeline._turn_artifacts(job, Block(0, 60_000), turns)
    raw = [(row.start_ms, row.end_ms, row.metadata["speaker"]) for row in turn_rows]
    pieces = pipeline._utterance_artifacts(job, turn_rows, raw)
    assert len(pieces) == 2
    assert [piece.metadata["overlap_ms"] for piece in pieces] == [300, 300]
