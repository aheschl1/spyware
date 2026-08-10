"""Tier 2: transcribe every detected speech span.

Consumes ``speech-span`` artifacts (the speech-detect tier's output), renders
each span from the session timeline, sends it to the transcription service
(processing/transcriber.py — model and protocol are environment-swappable),
and records a ``transcript`` artifact on the same span. The full service
response is persisted as a blob under this pipeline's prefix; the artifact
row carries the text.
"""

import json
from collections.abc import Sequence
from typing import Any

from database.pipe import DatabasePipe
from database.repos.pipelines.transcribe import TranscribeQueries
from database.schema.artifacts import ArtifactCreate
from database.schema.jobs import Job, JobCreate
from processing.base import Pipeline
from processing.config import get_settings
from processing.transcriber import Transcriber
from services import stitch, timeline
from storage.keys import pipeline_key
from storage.pipe import BlobPipe

_SOURCE_PIPELINE = "speech-detect"

# Artifact metadata keeps the row light; longer text lives in the blob only.
_TEXT_PREVIEW_CHARS = 500


class TranscribePipeline(Pipeline):
    name = "transcribe"

    async def setup(self) -> None:
        self._settings = get_settings()
        self._transcriber = Transcriber(self._settings)

    async def teardown(self) -> None:
        await self._transcriber.close()

    async def discover(self, limit: int) -> Sequence[JobCreate]:
        async with DatabasePipe() as pipe:
            spans = await TranscribeQueries(pipe.connection).spans_without_jobs(
                self.name, _SOURCE_PIPELINE, limit
            )
        return tuple(
            JobCreate(
                pipeline=self.name,
                session_id=span.session_id,
                artifact_id=span.id,
                payload={"start_ms": span.start_ms, "end_ms": span.end_ms},
                dedup_key=f"{self.name}:artifact:{span.id}",
            )
            for span in spans
        )

    async def process(self, job: Job) -> dict[str, Any]:
        assert job.session_id is not None and job.artifact_id is not None
        start_ms, end_ms = job.payload["start_ms"], job.payload["end_ms"]

        try:
            line = await timeline.load_timeline(job.session_id)
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

        key = pipeline_key(
            self.name, job.session_id, f"transcript-{start_ms:>09d}-{end_ms:>09d}.json"
        )
        document = {
            "text": result.text,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "model": self._settings.transcriber_model,
            "response": result.raw,
        }
        async with BlobPipe() as blobs:
            info = await blobs.put(
                key, json.dumps(document).encode(), "application/json"
            )
        async with DatabasePipe() as pipe:
            await pipe.artifacts.create(
                ArtifactCreate(
                    pipeline=self.name,
                    kind="transcript",
                    session_id=job.session_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    bucket=info.bucket,
                    object_key=info.key,
                    links={"speech_span": str(job.artifact_id)},
                    metadata={
                        "text": result.text[:_TEXT_PREVIEW_CHARS],
                        "chars": len(result.text),
                        "model": self._settings.transcriber_model,
                    },
                )
            )
        return {"chars": len(result.text), "span_ms": end_ms - start_ms}
