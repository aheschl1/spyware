"""The sound-span tier's pure logic: window reading, hysteresis span building,
and class selection."""

import random

from processing.pipelines.sound_span import (
    ClassSpan,
    ScoredWindow,
    build_spans,
    read_window,
    select_spans,
)

# The production grid: 10s windows on a 5s hop, so consecutive windows overlap.
WINDOW_MS = 10_000
HOP_MS = 5_000

DEFAULTS = {
    "enter": 0.35,
    "sustain": 0.20,
    "bridge_gap_ms": 5_000,
    "min_windows": 2,
}


def _w(index: int, **scores: float) -> ScoredWindow:
    """The ``index``-th window of the standard grid, scoring ``scores``."""
    start = index * HOP_MS
    return ScoredWindow(
        start_ms=start,
        end_ms=start + WINDOW_MS,
        labels=tuple(scores.items()),
    )


def _build(windows, **overrides) -> list[ClassSpan]:
    return build_spans(windows, **{**DEFAULTS, **overrides})


class TestReadWindow:
    def test_valid_row(self) -> None:
        window = read_window(0, 10_000, {"labels": [{"label": "Music", "score": 0.9}]})
        assert window == ScoredWindow(0, 10_000, (("Music", 0.9),))

    def test_null_bounds_are_not_windows(self) -> None:
        assert read_window(None, 10_000, {"labels": []}) is None
        assert read_window(0, None, {"labels": []}) is None

    def test_inverted_span_is_not_a_window(self) -> None:
        assert read_window(10_000, 10_000, {"labels": []}) is None
        assert read_window(10_000, 5_000, {"labels": []}) is None

    def test_missing_labels_key(self) -> None:
        assert read_window(0, 10_000, {}) == ScoredWindow(0, 10_000, ())

    def test_null_labels(self) -> None:
        assert read_window(0, 10_000, {"labels": None}) == ScoredWindow(0, 10_000, ())

    def test_malformed_entries_are_dropped_not_raised(self) -> None:
        window = read_window(
            0,
            10_000,
            {
                "labels": [
                    {"label": "Music", "score": 0.9},
                    {"label": "NoScore"},
                    {"score": 0.5},
                    {"label": "Bad", "score": "high"},
                    {"label": "", "score": 0.5},
                    {"label": "NaN", "score": float("nan")},
                    {"label": "Inf", "score": float("inf")},
                    "not a mapping",
                ]
            },
        )
        assert window is not None
        assert window.labels == (("Music", 0.9),)

    def test_int_scores_become_floats(self) -> None:
        window = read_window(0, 10_000, {"labels": [{"label": "Music", "score": 1}]})
        assert window is not None
        assert window.labels == (("Music", 1.0),)


class TestBuildSpans:
    def test_consecutive_high_windows_merge_into_one(self) -> None:
        spans = _build([_w(i, Music=0.9) for i in range(4)])
        assert len(spans) == 1
        assert spans[0].label == "Music"
        assert (spans[0].start_ms, spans[0].end_ms) == (0, 3 * HOP_MS + WINDOW_MS)
        assert spans[0].windows == 4

    def test_two_classes_overlap_in_time(self) -> None:
        # Music over windows 0-3, Speech over 2-5: the spans must overlap, and
        # both must survive. This is the whole point of the tier.
        windows = [
            _w(0, Music=0.9),
            _w(1, Music=0.9),
            _w(2, Music=0.9, Speech=0.8),
            _w(3, Music=0.9, Speech=0.8),
            _w(4, Speech=0.8),
            _w(5, Speech=0.8),
        ]
        spans = _build(windows)
        assert [span.label for span in spans] == ["Music", "Speech"]
        music, speech = spans
        assert music.start_ms < speech.start_ms < music.end_ms < speech.end_ms

    def test_low_confidence_never_opens_a_span(self) -> None:
        # Above sustain, below enter, forever: no span at all.
        assert _build([_w(i, Music=0.25) for i in range(10)]) == []

    def test_sustain_holds_a_span_open_below_enter(self) -> None:
        windows = [_w(0, Music=0.9)] + [_w(i, Music=0.25) for i in range(1, 4)]
        spans = _build(windows)
        assert len(spans) == 1
        assert spans[0].windows == 4
        assert spans[0].end_ms == 3 * HOP_MS + WINDOW_MS

    def test_reopening_requires_enter_not_sustain(self) -> None:
        # Two strong windows, a long silence, then sustain-only scores. The
        # trailing run must not reopen the span.
        windows = [_w(0, Music=0.9), _w(1, Music=0.9)]
        windows += [_w(i, Music=0.25) for i in range(10, 14)]
        spans = _build(windows)
        assert len(spans) == 1
        assert spans[0].end_ms == HOP_MS + WINDOW_MS

    def test_single_dropout_is_bridged(self) -> None:
        # Window 1 does not mention Music at all; the span survives it.
        windows = [_w(0, Music=0.9), _w(1, Speech=0.9), _w(2, Music=0.9)]
        spans = [span for span in _build(windows) if span.label == "Music"]
        assert len(spans) == 1
        assert spans[0].windows == 2
        assert (spans[0].start_ms, spans[0].end_ms) == (0, 2 * HOP_MS + WINDOW_MS)

    def test_dropout_beyond_the_bridge_splits(self) -> None:
        windows = [_w(0, Music=0.9), _w(1, Music=0.9)]
        windows += [_w(i) for i in range(2, 6)]
        windows += [_w(6, Music=0.9), _w(7, Music=0.9)]
        spans = _build(windows)
        assert len(spans) == 2
        assert spans[0].end_ms < spans[1].start_ms

    def test_gap_is_milliseconds_not_window_count(self) -> None:
        # An irregular grid with rows simply missing: two windows 30s apart are
        # a split at bridge_gap_ms=5000 no matter that they are adjacent rows.
        windows = [
            ScoredWindow(0, 10_000, (("Music", 0.9),)),
            ScoredWindow(5_000, 15_000, (("Music", 0.9),)),
            ScoredWindow(45_000, 55_000, (("Music", 0.9),)),
            ScoredWindow(50_000, 60_000, (("Music", 0.9),)),
        ]
        spans = _build(windows)
        assert [(s.start_ms, s.end_ms) for s in spans] == [(0, 15_000), (45_000, 60_000)]

    def test_wide_bridge_joins_the_same_windows(self) -> None:
        windows = [
            ScoredWindow(0, 10_000, (("Music", 0.9),)),
            ScoredWindow(45_000, 55_000, (("Music", 0.9),)),
        ]
        spans = _build(windows, bridge_gap_ms=40_000)
        assert len(spans) == 1
        assert (spans[0].start_ms, spans[0].end_ms) == (0, 55_000)

    def test_single_window_span_is_dropped(self) -> None:
        assert _build([_w(0, Music=0.9), _w(5, Speech=0.9)]) == []

    def test_min_windows_one_keeps_a_singleton(self) -> None:
        spans = _build([_w(0, Music=0.9)], min_windows=1)
        assert len(spans) == 1
        assert spans[0].windows == 1

    def test_absent_label_scores_zero(self) -> None:
        # An empty window behaves exactly like one scoring 0.0.
        empty = _build([_w(0, Music=0.9), _w(1), _w(2, Music=0.9)], bridge_gap_ms=0)
        explicit = _build(
            [_w(0, Music=0.9), _w(1, Music=0.0), _w(2, Music=0.9)], bridge_gap_ms=0
        )
        assert empty == explicit

    def test_span_bounds_are_the_union_including_a_short_tail(self) -> None:
        # The session ends mid-window, so the last window is clipped short and
        # max(end_ms) is not the last member's end.
        windows = [
            _w(0, Music=0.9),
            _w(1, Music=0.9),
            ScoredWindow(10_000, 12_000, (("Music", 0.9),)),
        ]
        spans = _build(windows)
        assert len(spans) == 1
        assert (spans[0].start_ms, spans[0].end_ms) == (0, 15_000)

    def test_peak_and_mean(self) -> None:
        spans = _build([_w(0, Music=0.9), _w(1, Music=0.5), _w(2, Music=0.4)])
        assert spans[0].peak == 0.9
        assert spans[0].mean == (0.9 + 0.5 + 0.4) / 3

    def test_duplicate_windows_do_not_inflate_the_member_count(self) -> None:
        # The same window twice is one window's evidence, not two.
        assert _build([_w(0, Music=0.9), _w(0, Music=0.9)]) == []

    def test_output_is_ordered_and_deterministic_under_shuffle(self) -> None:
        windows = [_w(i, Music=0.9, Speech=0.9) for i in range(4)]
        windows += [_w(i, Rain=0.9) for i in range(4, 8)]
        expected = _build(windows)
        shuffled = list(windows)
        random.Random(0).shuffle(shuffled)
        assert _build(shuffled) == expected
        assert [(s.start_ms, s.end_ms, s.label) for s in expected] == sorted(
            (s.start_ms, s.end_ms, s.label) for s in expected
        )

    def test_empty_input(self) -> None:
        assert _build([]) == []

    def test_windows_with_no_labels_at_all(self) -> None:
        assert _build([_w(i) for i in range(5)]) == []


def _span(label: str, start: int, end: int, peak: float = 0.9) -> ClassSpan:
    return ClassSpan(
        label=label, start_ms=start, end_ms=end, windows=2, peak=peak, mean=peak
    )


class TestSelectSpans:
    def test_ranks_by_covered_time_not_peak(self) -> None:
        spans = [_span("Blip", 0, 15_000, peak=0.99), _span("Music", 0, 600_000, peak=0.4)]
        _, ranked = select_spans(spans, top_k=8, max_spans=2_000)
        assert [summary.label for summary in ranked] == ["Music", "Blip"]

    def test_top_k_keeps_whole_classes(self) -> None:
        spans = [
            _span("Music", 0, 600_000),
            _span("Music", 700_000, 800_000),
            _span("Speech", 0, 10_000),
        ]
        kept, ranked = select_spans(spans, top_k=1, max_spans=2_000)
        assert [span.label for span in kept] == ["Music", "Music"]
        assert [summary.label for summary in ranked] == ["Music"]

    def test_summary_totals(self) -> None:
        spans = [_span("Music", 0, 10_000, peak=0.5), _span("Music", 20_000, 50_000, peak=0.8)]
        _, ranked = select_spans(spans, top_k=8, max_spans=2_000)
        assert len(ranked) == 1
        assert (ranked[0].spans, ranked[0].total_ms, ranked[0].peak) == (2, 40_000, 0.8)

    def test_max_spans_drops_trailing_classes(self) -> None:
        spans = [_span("Music", i * 20_000, i * 20_000 + 10_000) for i in range(3)]
        spans.append(_span("Speech", 0, 5_000))
        kept, ranked = select_spans(spans, top_k=8, max_spans=3)
        assert {span.label for span in kept} == {"Music"}
        assert [summary.label for summary in ranked] == ["Music"]

    def test_max_spans_below_the_first_class_keeps_its_longest(self) -> None:
        spans = [
            _span("Music", 0, 5_000),
            _span("Music", 10_000, 60_000),
            _span("Music", 70_000, 200_000),
        ]
        kept, ranked = select_spans(spans, top_k=8, max_spans=2)
        assert [(span.start_ms, span.end_ms) for span in kept] == [
            (10_000, 60_000),
            (70_000, 200_000),
        ]
        assert ranked[0].spans == 2

    def test_ties_break_on_label(self) -> None:
        spans = [_span("Zebra", 0, 10_000), _span("Apple", 20_000, 30_000)]
        _, ranked = select_spans(spans, top_k=8, max_spans=2_000)
        assert [summary.label for summary in ranked] == ["Apple", "Zebra"]

    def test_kept_spans_stay_in_timeline_order(self) -> None:
        spans = [_span("Speech", 5_000, 15_000), _span("Music", 0, 30_000)]
        kept, _ = select_spans(spans, top_k=8, max_spans=2_000)
        assert [span.start_ms for span in kept] == [0, 5_000]

    def test_empty(self) -> None:
        assert select_spans([], top_k=8, max_spans=2_000) == ([], [])
