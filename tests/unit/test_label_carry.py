"""Label carry-over: span overlap, old→new label mapping, edit re-application."""

from processing.label_carry import (
    EditRecord,
    carried_text,
    edit_records,
    label_records,
    map_labels,
    overlap_ms,
)


class TestOverlap:
    def test_disjoint(self) -> None:
        assert overlap_ms([(0, 10)], [(10, 20)]) == 0

    def test_partial_and_multi(self) -> None:
        assert overlap_ms([(0, 10), (20, 30)], [(5, 25)]) == 10

    def test_containment(self) -> None:
        assert overlap_ms([(0, 100)], [(10, 20), (30, 40)]) == 20


class TestMapLabels:
    def test_renamed_blocks_map_by_speech(self) -> None:
        # Two blocks merged into one: b0 and b40000 labels both land in b0.
        old = {
            "b0:SPEAKER_00": [(0, 5000)],
            "b0:SPEAKER_01": [(5000, 9000)],
            "b40000:SPEAKER_00": [(40_000, 44_000)],  # same voice as b0:SPEAKER_01
            "b40000:SPEAKER_01": [(44_000, 50_000)],  # same voice as b0:SPEAKER_00
        }
        new = {
            "b0:SPEAKER_00": [(0, 5000), (44_000, 50_000)],
            "b0:SPEAKER_01": [(5000, 9000), (40_000, 44_000)],
        }
        assert map_labels(old, new) == {
            "b0:SPEAKER_00": "b0:SPEAKER_00",
            "b0:SPEAKER_01": "b0:SPEAKER_01",
            "b40000:SPEAKER_00": "b0:SPEAKER_01",
            "b40000:SPEAKER_01": "b0:SPEAKER_00",
        }

    def test_label_with_no_overlap_is_dropped(self) -> None:
        assert map_labels({"a": [(0, 10)]}, {"x": [(20, 30)]}) == {}

    def test_split_label_follows_the_larger_share(self) -> None:
        old = {"b0:SPEAKER_00": [(0, 10_000)]}
        new = {"b0:SPEAKER_00.0": [(0, 3000)], "b0:SPEAKER_00.1": [(3000, 10_000)]}
        assert map_labels(old, new) == {"b0:SPEAKER_00": "b0:SPEAKER_00.1"}


class TestCarriedText:
    def test_edit_reapplies_to_the_overlapping_utterance(self) -> None:
        edits = [EditRecord(1000, 4000, "fixed")]
        assert carried_text(edits, 900, 4200) == "fixed"

    def test_edit_needs_half_of_itself_covered(self) -> None:
        edits = [EditRecord(1000, 4000, "fixed")]
        assert carried_text(edits, 3000, 6000) is None  # 1 s of 3 s
        assert carried_text(edits, 2400, 6000) == "fixed"  # 1.6 s of 3 s

    def test_best_overlap_wins(self) -> None:
        edits = [EditRecord(0, 2000, "first"), EditRecord(1500, 4000, "second")]
        assert carried_text(edits, 1000, 4000) == "second"


class TestRecords:
    def test_malformed_entries_are_skipped(self) -> None:
        meta = {
            "labels": [
                {"speaker": "b0:S", "model": "m", "speaker_id": "x", "spans": [[0, 1]]},
                {"speaker": "b0:T"},
            ],
            "edits": [{"start_ms": 0, "end_ms": 1, "text": "t"}, {"start_ms": "?"}],
        }
        (record,) = label_records(meta)
        assert record.spans == ((0, 1),) and record.pinned is False
        assert edit_records(meta) == [EditRecord(0, 1, "t")]
