"""Carry hand curation across a re-diarize.

Labels are ``b{block_start}:SPEAKER_XX``, so a re-diarize under new block
parameters renames every voice; pins and cluster assignments keyed on the
old names would match nothing. Voices do not move, though: an old label
maps to whichever new label covers the most of its speech. Transcript
edits carry the same way, by span overlap onto the new utterance.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Span = tuple[int, int]

# An edit must cover at least this much of its own span in the new utterance
# to be re-applied; below it the utterances no longer describe the same speech.
_EDIT_MIN_COVER = 0.5


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """One old label's identity, and where it spoke."""

    speaker: str
    model: str
    speaker_id: str
    pinned: bool
    spans: tuple[Span, ...]


@dataclass(frozen=True, slots=True)
class EditRecord:
    start_ms: int
    end_ms: int
    text: str


def overlap_ms(a: Sequence[Span], b: Sequence[Span]) -> int:
    """Total intersection of two span lists (each sorted, non-overlapping)."""
    total = i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def map_labels(
    old: Mapping[str, Sequence[Span]], new: Mapping[str, Sequence[Span]]
) -> dict[str, str]:
    """Each old label → the new label sharing the most speech with it.

    Labels with no overlap at all are left out (speech that vanished or was
    reassigned wholesale). Ties break on label name for determinism.
    """
    out: dict[str, str] = {}
    for label, spans in old.items():
        best, best_ms = None, 0
        for candidate, candidate_spans in sorted(new.items()):
            shared = overlap_ms(sorted(spans), sorted(candidate_spans))
            if shared > best_ms:
                best, best_ms = candidate, shared
        if best is not None:
            out[label] = best
    return out


def carried_text(edits: Sequence[EditRecord], start_ms: int, end_ms: int) -> str | None:
    """The edited text to re-apply to an utterance at ``[start_ms, end_ms)``."""
    best, best_ms = None, 0
    for edit in edits:
        shared = overlap_ms([(edit.start_ms, edit.end_ms)], [(start_ms, end_ms)])
        if shared > best_ms and shared >= _EDIT_MIN_COVER * (edit.end_ms - edit.start_ms):
            best, best_ms = edit, shared
    return None if best is None else best.text


def label_records(metadata: Mapping[str, Any]) -> list[LabelRecord]:
    """Snapshot metadata → records; malformed entries are dropped, not raised."""
    out = []
    for item in metadata.get("labels", ()):
        try:
            out.append(
                LabelRecord(
                    speaker=str(item["speaker"]),
                    model=str(item["model"]),
                    speaker_id=str(item["speaker_id"]),
                    pinned=bool(item.get("pinned", False)),
                    spans=tuple((int(s), int(e)) for s, e in item.get("spans", ())),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def edit_records(metadata: Mapping[str, Any]) -> list[EditRecord]:
    out = []
    for item in metadata.get("edits", ()):
        try:
            out.append(EditRecord(int(item["start_ms"]), int(item["end_ms"]), str(item["text"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out
