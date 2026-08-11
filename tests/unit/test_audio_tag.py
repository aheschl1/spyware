"""The audio-tag tier's pure logic: ontology suppression, session-tag
aggregation, span stepping, window dedup, and the classifier response parse."""

import pytest

from processing.classifier import ClassifierError, parse_response
from processing.pipelines.audio_tag import (
    TaggedWindow,
    ancestors,
    dedupe_windows,
    session_tags,
    span_starts,
    suppress_ancestors,
)

# A miniature ontology DAG: Sounds -> Music -> Guitar, Sounds -> Vehicle ->
# Car, and Alarm with two parents (the DAG case).
PARENTS = {
    "Music": ["Sounds"],
    "Guitar": ["Music"],
    "Vehicle": ["Sounds"],
    "Car": ["Vehicle"],
    "Alarm": ["Vehicle", "Music"],
}


def _window(start: int, labels, embedding=(1.0, 0.0)) -> TaggedWindow:
    return TaggedWindow(
        start_ms=start,
        end_ms=start + 10_000,
        labels=tuple(labels),
        embedding=tuple(embedding),
    )


class TestAncestors:
    def test_transitive(self) -> None:
        assert ancestors("Guitar", PARENTS) == {"Music", "Sounds"}

    def test_multi_parent(self) -> None:
        assert ancestors("Alarm", PARENTS) == {"Vehicle", "Music", "Sounds"}

    def test_root_has_none(self) -> None:
        assert ancestors("Sounds", PARENTS) == set()

    def test_unknown_label(self) -> None:
        assert ancestors("Not a class", PARENTS) == set()


class TestSuppressAncestors:
    def test_descendant_beats_ancestor(self) -> None:
        kept = suppress_ancestors([("Music", 0.8), ("Guitar", 0.9)], PARENTS)
        assert kept == [("Guitar", 0.9)]

    def test_tie_drops_the_ancestor(self) -> None:
        # The more specific label carries the information.
        kept = suppress_ancestors([("Music", 0.8), ("Guitar", 0.8)], PARENTS)
        assert kept == [("Guitar", 0.8)]

    def test_stronger_ancestor_survives(self) -> None:
        kept = suppress_ancestors([("Music", 0.9), ("Guitar", 0.5)], PARENTS)
        assert kept == [("Music", 0.9), ("Guitar", 0.5)]

    def test_transitive_suppression(self) -> None:
        kept = suppress_ancestors(
            [("Sounds", 0.7), ("Music", 0.7), ("Guitar", 0.9)], PARENTS
        )
        assert kept == [("Guitar", 0.9)]

    def test_unrelated_labels_untouched(self) -> None:
        labels = [("Car", 0.6), ("Guitar", 0.5)]
        assert suppress_ancestors(labels, PARENTS) == labels

    def test_empty(self) -> None:
        assert suppress_ancestors([], PARENTS) == []


class TestSessionTags:
    def test_max_across_windows(self) -> None:
        windows = [
            _window(0, [("Music", 0.5)]),
            _window(5_000, [("Music", 0.9)]),
        ]
        assert session_tags(windows, threshold=0.3, min_consecutive=2, top_k=15) == [
            ("Music", 0.9)
        ]

    def test_single_window_blip_is_dropped(self) -> None:
        windows = [
            _window(0, [("Music", 0.5), ("Car", 0.8)]),
            _window(5_000, [("Music", 0.5)]),
            _window(10_000, [("Music", 0.5)]),
        ]
        assert session_tags(windows, threshold=0.3, min_consecutive=2, top_k=15) == [
            ("Music", 0.5)
        ]

    def test_run_must_be_consecutive(self) -> None:
        # Present in windows 1 and 3 but not 2: never two in a row.
        windows = [
            _window(0, [("Car", 0.8)]),
            _window(5_000, []),
            _window(10_000, [("Car", 0.8)]),
        ]
        assert session_tags(windows, threshold=0.3, min_consecutive=2, top_k=15) == []

    def test_short_session_caps_the_requirement(self) -> None:
        # One window total: min_consecutive=2 must not make tagging impossible.
        windows = [_window(0, [("Music", 0.9)])]
        assert session_tags(windows, threshold=0.3, min_consecutive=2, top_k=15) == [
            ("Music", 0.9)
        ]

    def test_below_threshold_never_qualifies(self) -> None:
        windows = [
            _window(0, [("Music", 0.2)]),
            _window(5_000, [("Music", 0.2)]),
        ]
        assert session_tags(windows, threshold=0.3, min_consecutive=2, top_k=15) == []

    def test_top_k_and_ordering(self) -> None:
        windows = [
            _window(0, [("Music", 0.5), ("Car", 0.9), ("Guitar", 0.7)]),
            _window(5_000, [("Music", 0.5), ("Car", 0.9), ("Guitar", 0.7)]),
        ]
        assert session_tags(windows, threshold=0.3, min_consecutive=2, top_k=2) == [
            ("Car", 0.9),
            ("Guitar", 0.7),
        ]

    def test_no_windows(self) -> None:
        assert session_tags([], threshold=0.3, min_consecutive=2, top_k=15) == []


class TestSpanStarts:
    def test_short_session_is_one_span(self) -> None:
        assert span_starts(30_000, span_ms=120_000, overlap_ms=5_000) == [0]

    def test_spans_overlap_by_window_minus_hop(self) -> None:
        starts = span_starts(300_000, span_ms=120_000, overlap_ms=5_000)
        assert starts == [0, 115_000, 230_000]
        # Every span but the last is full-length; consecutive spans overlap.
        assert all(b - a == 115_000 for a, b in zip(starts, starts[1:]))

    def test_full_coverage(self) -> None:
        total = 987_654
        starts = span_starts(total, span_ms=120_000, overlap_ms=5_000)
        assert starts[0] == 0 and starts[-1] + 120_000 >= total


class TestDedupeWindows:
    def test_span_seam_duplicates_collapse(self) -> None:
        first = _window(115_000, [("Music", 0.9)])
        duplicate = _window(115_000, [("Music", 0.8)])
        kept = dedupe_windows([_window(0, []), first, duplicate])
        assert [w.start_ms for w in kept] == [0, 115_000]
        assert kept[1].labels == first.labels

    def test_orders_by_start(self) -> None:
        kept = dedupe_windows([_window(5_000, []), _window(0, [])])
        assert [w.start_ms for w in kept] == [0, 5_000]


class TestParseResponse:
    def _body(self, **overrides):
        body = {
            "windows": [
                {
                    "start_ms": 0,
                    "end_ms": 10_000,
                    "labels": [{"label": "Music", "score": 0.9}],
                    "embedding": [0.1, 0.2],
                }
            ],
            "models": {"tagger": "t", "clap": "c"},
        }
        body.update(overrides)
        return body

    def test_valid(self) -> None:
        analysis = parse_response(self._body())
        assert analysis.tagger_model == "t" and analysis.embedding_model == "c"
        (window,) = analysis.windows
        assert window.labels == (("Music", 0.9),)
        assert window.embedding == (0.1, 0.2)

    def test_non_finite_embedding_rejected(self) -> None:
        # json.loads parses literal NaN; one stored NaN poisons pgvector math.
        bad = self._body()
        bad["windows"][0]["embedding"] = [0.1, float("nan")]
        with pytest.raises(ClassifierError, match="non-finite"):
            parse_response(bad)

    def test_non_finite_score_rejected(self) -> None:
        bad = self._body()
        bad["windows"][0]["labels"] = [{"label": "Music", "score": float("inf")}]
        with pytest.raises(ClassifierError):
            parse_response(bad)

    def test_missing_models_rejected(self) -> None:
        with pytest.raises(ClassifierError):
            parse_response(self._body(models=None))

    def test_empty_embedding_rejected(self) -> None:
        bad = self._body()
        bad["windows"][0]["embedding"] = []
        with pytest.raises(ClassifierError):
            parse_response(bad)

    def test_inverted_window_rejected(self) -> None:
        bad = self._body()
        bad["windows"][0]["end_ms"] = 0
        with pytest.raises(ClassifierError):
            parse_response(bad)

    def test_no_windows_is_valid(self) -> None:
        assert parse_response(self._body(windows=[])).windows == ()
