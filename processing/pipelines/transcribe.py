"""Tier: transcribe every diarized utterance.

Consumes ``utterance`` artifacts (the diarize tier's ASR units — same-speaker
turns merged to the model's input window), renders each from the session
timeline, sends it to the transcription service (processing/transcriber.py —
model and protocol are environment-swappable), and records a ``transcript``
artifact on the same range. Transcripts live entirely in the artifact row —
utterances are short, so the full text fits in metadata and no blob is
written.

Diarize invalidates this tier's output when it republishes a session: it
deletes the transcripts in the same transaction that replaces the utterances,
and the queued jobs pointing at deleted utterances skip themselves.
"""

from collections import OrderedDict
from collections.abc import Sequence
import logging
from typing import Any
from uuid import UUID

from database.pipe import DatabasePipe
from database.repos.pipelines.transcribe import TranscribeQueries
from database.schema.artifacts import ArtifactCreate
from database.schema.jobs import Job, JobCreate
from processing.base import Pipeline
from processing.config import get_settings
from processing.label_carry import carried_text, edit_records
from processing.transcriber import Transcriber
from resources import Resource
from services import label_carry, stitch, timeline

_SOURCE_PIPELINE = "diarize"
# Timelines to keep between jobs. Discovery interleaves sessions, so one slot
# would thrash; a handful covers a pass without holding stale audio for long.
_TIMELINE_CACHE = 4


class TranscribePipeline(Pipeline):
    name = "transcribe"
    resource = Resource.AUDIO

    async def setup(self) -> None:
        self._settings = get_settings()
        self._transcriber = Transcriber(self._settings)
        self._timelines: OrderedDict[UUID, timeline.SessionTimeline] = OrderedDict()

    async def teardown(self) -> None:
        await self._transcriber.close()

    async def _timeline_for(
        self, session_id: UUID, end_ms: int
    ) -> timeline.SessionTimeline | None:
        """The session's timeline, reused across jobs in the same pass.

        load_timeline is a segment query plus (sometimes) a blob header read --
        200-460 ms on real sessions, several times what transcribing the clip
        costs. Reuse is gated on the cached timeline already covering end_ms:
        audio is append-only, but byte_range() clamps silently, so reusing one
        that stops short would truncate the clip instead of failing.
        """
        cached = self._timelines.get(session_id)
        if cached is not None and end_ms <= cached.total_ms:
            self._timelines.move_to_end(session_id)
            return cached
        line = await timeline.load_timeline(session_id)
        if line is not None:
            self._timelines[session_id] = line
            self._timelines.move_to_end(session_id)
            while len(self._timelines) > _TIMELINE_CACHE:
                self._timelines.popitem(last=False)
        return line

    async def discover(self, limit: int) -> Sequence[JobCreate]:
        # A new pass may follow an ingest that extended a session, and a
        # deleted session must not stay cached indefinitely.
        self._timelines.clear()
        async with DatabasePipe() as pipe:
            utterances = await TranscribeQueries(pipe.connection).utterances_without_jobs(
                self.name, _SOURCE_PIPELINE, limit
            )
        logging.debug(f"pipeline {self.name} found {len(utterances)} jobs")
        return tuple(
            JobCreate(
                pipeline=self.name,
                session_id=utterance.session_id,
                artifact_id=utterance.id,
                payload={
                    "start_ms": utterance.start_ms,
                    "end_ms": utterance.end_ms,
                    "speaker": utterance.metadata.get("speaker"),
                    "overlap_ms": utterance.metadata.get("overlap_ms"),
                    "host_utterance": utterance.links.get("host_utterance"),
                },
                dedup_key=f"{self.name}:artifact:{utterance.id}",
            )
            for utterance in utterances
        )

    async def process(self, job: Job) -> dict[str, Any]:
        assert job.session_id is not None
        if job.artifact_id is None:
            # Diarize republished the session and deleted this utterance
            # (the FK nulls artifact_id); the replacement utterance has its
            # own job.
            return {"skipped": "utterance deleted before transcription"}
        start_ms, end_ms = job.payload["start_ms"], job.payload["end_ms"]
        speaker = job.payload.get("speaker")
        overlap_ms = job.payload.get("overlap_ms")
        # Set when this utterance was spoken entirely inside another
        # speaker's (the host's) — the host transcript also carries these
        # words; the link is what lets a reader nest this one under it.
        links = {"utterance": str(job.artifact_id)}
        if host := job.payload.get("host_utterance"):
            links["host_utterance"] = host

        try:
            line = await self._timeline_for(job.session_id, end_ms)
            if line is None:
                return {"skipped": "session audio is gone"}
            clip = await timeline.render_range(line, start_ms, end_ms)
        except (stitch.NotStitchable, timeline.NotRenderable) as exc:
            # Retrying cannot make the audio renderable.
            return {"skipped": str(exc)}

        # TranscriberError raises through: the worker retries with backoff,
        # which is exactly right for a briefly-down service.
        result = await self._transcriber.transcribe(
            clip, filename=f"{job.session_id}-{start_ms}-{end_ms}.wav"
        )

        metadata: dict[str, Any] = {
            "text": result.text,
            "chars": len(result.text),
            "model": self._settings.transcriber_model,
            "speaker": speaker,
            # Copied from the utterance: overlapped speech is
            # always transcribed (overlap never gates ASR), but
            # "this transcript contains crosstalk" must stay
            # queryable without joining back to the utterance.
            "overlap_ms": overlap_ms,
        }
        if result.words is not None:
            # Session-absolute ms: the clip was rendered from start_ms.
            metadata["words"] = [
                {"w": w.word, "s": start_ms + w.start_ms, "e": start_ms + w.end_ms}
                for w in result.words
            ]
        if result.language is not None:
            metadata["language"] = result.language

        async with DatabasePipe() as pipe:
            # Re-check inside the insert transaction: diarize may have
            # replaced the utterance while we were transcribing. (A commit
            # racing diarize's delete by microseconds could still slip an
            # orphan through; add FOR KEY SHARE here if one is ever observed.)
            if await pipe.artifacts.get(job.artifact_id) is None:
                return {"skipped": "utterance deleted during transcription"}
            # retranscribe deletes job history and can race an in-flight
            # job into a second run for the same utterance; job dedup no
            # longer guards that, so the data does.
            if await TranscribeQueries(pipe.connection).transcript_exists(
                job.artifact_id
            ):
                return {"skipped": "transcript already exists"}
            snapshot = await label_carry.pending(pipe, job.session_id)
            if snapshot is not None:
                edited = carried_text(edit_records(snapshot.metadata), start_ms, end_ms)
                if edited is not None:
                    # Same shape as an edit made through the API; the model's
                    # word timings no longer describe the text.
                    metadata.pop("words", None)
                    metadata.update({"text": edited, "chars": len(edited), "edited": True})
            await pipe.artifacts.create(
                ArtifactCreate(
                    pipeline=self.name,
                    kind="transcript",
                    session_id=job.session_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    links=links,
                    metadata=metadata,
                )
            )
        return {"chars": len(result.text), "span_ms": end_ms - start_ms}
