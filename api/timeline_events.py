"""Expand a session and its pipeline artifacts into the ordered event timeline.

Pure planning over rows, no I/O. Each ``(pipeline, kind)`` with a registered
expander contributes events; everything else (``speech-map``,
``session-stats``, kinds from tiers not yet surfaced) is skipped.
Registering one expander is all a new tier needs to appear on the timeline.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import timedelta
from itertools import chain

from api.schema.timeline import (
    AudioTagEvent,
    SessionEndEvent,
    SessionStartEvent,
    SpeechEndEvent,
    SpeechStartEvent,
    TimelineEvent,
    TranscriptEvent,
)
from database.schema.artifacts import PipelineArtifact
from database.schema.sessions import RecordingSession
from database.schema.speakers import SessionSpeakerLabel

Expander = Callable[[PipelineArtifact], Iterator[TimelineEvent]]

_EXPANDERS: dict[tuple[str, str], Expander] = {}

# At one timestamp: the session opens first; span A ends before adjacent
# span B starts; a transcript follows its span's start; the session closes
# last even on an exact tie. Types without a rank land between.
_TYPE_RANK = {
    "session-start": 0,
    "speech-end": 1,
    "speech-start": 2,
    "transcript": 3,
    "audio-tag": 4,
    "session-end": 100,
}
_DEFAULT_RANK = 50


def expander(pipeline: str, kind: str) -> Callable[[Expander], Expander]:
    """Register the expander for one ``(pipeline, kind)`` — the plug-in point."""

    def register(fn: Expander) -> Expander:
        _EXPANDERS[(pipeline, kind)] = fn
        return fn

    return register


@expander("speech-detect", "speech-span")
def _speech_span_events(artifact: PipelineArtifact) -> Iterator[TimelineEvent]:
    if artifact.start_ms is None or artifact.end_ms is None:
        return
    confidence = artifact.metadata.get("confidence")
    yield SpeechStartEvent(
        at_ms=artifact.start_ms, artifact_id=artifact.id, confidence=confidence
    )
    yield SpeechEndEvent(at_ms=artifact.end_ms, artifact_id=artifact.id, confidence=confidence)


@expander("transcribe", "transcript")
def _transcript_events(artifact: PipelineArtifact) -> Iterator[TimelineEvent]:
    if artifact.start_ms is None or artifact.end_ms is None:
        return
    text = artifact.metadata.get("text", "")
    yield TranscriptEvent(
        at_ms=artifact.start_ms,
        artifact_id=artifact.id,
        start_ms=artifact.start_ms,
        end_ms=artifact.end_ms,
        text=text,
        chars=artifact.metadata.get("chars", len(text)),
        model=artifact.metadata.get("model"),
        speaker=artifact.metadata.get("speaker"),
    )


@expander("audio-tag", "audio-tag")
def _audio_tag_events(artifact: PipelineArtifact) -> Iterator[TimelineEvent]:
    if artifact.start_ms is None or artifact.end_ms is None:
        return
    yield AudioTagEvent(
        at_ms=artifact.start_ms,
        artifact_id=artifact.id,
        start_ms=artifact.start_ms,
        end_ms=artifact.end_ms,
        labels=tuple(artifact.metadata.get("labels", ())),
        model=artifact.metadata.get("model"),
    )


def _nothing(artifact: PipelineArtifact) -> Iterator[TimelineEvent]:
    return iter(())


def _session_events(session: RecordingSession) -> Iterator[TimelineEvent]:
    yield SessionStartEvent(at_ms=0, started_at=session.started_at)
    if session.ended_at is not None:
        elapsed = (session.ended_at - session.started_at) // timedelta(milliseconds=1)
        yield SessionEndEvent(at_ms=max(0, elapsed), ended_at=session.ended_at)


def _sort_key(event: TimelineEvent) -> tuple[int, int, str, str]:
    artifact_id = getattr(event, "artifact_id", None)
    return (
        event.at_ms,
        _TYPE_RANK.get(event.type, _DEFAULT_RANK),
        "" if artifact_id is None else str(artifact_id),
        event.type,
    )


def assemble(
    session: RecordingSession,
    artifacts: Sequence[PipelineArtifact],
    *,
    from_ms: int | None = None,
    to_ms: int | None = None,
    speakers: Mapping[str, SessionSpeakerLabel] | None = None,
) -> list[TimelineEvent]:
    """Every event derivable from the rows, in timeline order.

    ``from_ms``/``to_ms`` keep only events positioned in ``[from_ms, to_ms)``:
    an artifact overlapping the window contributes just the events inside it,
    so adjacent windows partition the stream — no duplicates, no gaps.

    ``speakers`` maps block-local labels to their clustering-tier resolution;
    matching transcript events get stamped with the global identity and the
    voice-print behind it. Unresolvable labels stay null — clustering is
    eventually consistent with diarization.
    """
    events: Iterator[TimelineEvent] = chain(
        _session_events(session),
        (
            event
            for artifact in artifacts
            for event in _EXPANDERS.get((artifact.pipeline, artifact.kind), _nothing)(artifact)
        ),
    )
    if from_ms is not None:
        events = (event for event in events if event.at_ms >= from_ms)
    if to_ms is not None:
        events = (event for event in events if event.at_ms < to_ms)
    ordered = sorted(events, key=_sort_key)
    if not speakers:
        return ordered
    return [
        event.model_copy(
            update={
                "speaker_id": ref.speaker_id,
                "speaker_name": ref.name,
                "voiceprint_id": ref.artifact_id,
            }
        )
        if event.type == "transcript"
        and (ref := speakers.get(getattr(event, "speaker", None))) is not None
        else event
        for event in ordered
    ]
