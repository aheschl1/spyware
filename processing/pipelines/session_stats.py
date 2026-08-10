"""Demo pipeline: per-session aggregates over every segment.

Exists to prove the plumbing — self-discovery, whole-session consumption, the
pipeline blob space, and artifact registration. Real pipelines (transcription,
...) follow this exact shape.
"""

import json
from collections.abc import Sequence
from typing import Any

from database.pipe import DatabasePipe
from database.repos.pipelines.session_stats import SessionStatsQueries
from database.schema.artifacts import ArtifactCreate
from database.schema.jobs import Job, JobCreate
from processing.base import Pipeline
from storage.keys import pipeline_key
from storage.pipe import BlobPipe

# Far past any real session (one chunk per second for a day is 86 400).
_MAX_SEGMENTS = 1_000_000


class SessionStatsPipeline(Pipeline):
    name = "session-stats"

    async def discover(self, limit: int) -> Sequence[JobCreate]:
        async with DatabasePipe() as pipe:
            sessions = await SessionStatsQueries(pipe.connection).discover_unprocessed(
                self.name, limit
            )
        return tuple(
            JobCreate(
                pipeline=self.name,
                session_id=session.id,
                dedup_key=f"{self.name}:session:{session.id}",
            )
            for session in sessions
        )

    async def process(self, job: Job) -> dict[str, Any]:
        assert job.session_id is not None  # every job here comes from discover()
        async with DatabasePipe() as pipe:
            session = await pipe.sessions.get(job.session_id)
            segments = await pipe.segments.list_for_session(
                job.session_id, limit=_MAX_SEGMENTS
            )
        if session is None:
            return {"missing": True}
        if session.metadata.get("boom"):
            # Deterministic failure hook for exercising retry-to-dead.
            raise RuntimeError("boom requested by session metadata")

        stats = {
            "segments": len(segments),
            "total_bytes": sum(segment.byte_size for segment in segments),
            "duration_ms": sum(segment.duration_ms or 0 for segment in segments),
        }
        info = None
        if segments:
            key = pipeline_key(self.name, session.id, "stats.json")
            async with BlobPipe() as blobs:
                info = await blobs.put(
                    key, json.dumps(stats).encode(), "application/json"
                )
        async with DatabasePipe() as pipe:
            await pipe.artifacts.create(
                ArtifactCreate(
                    pipeline=self.name,
                    kind="session-stats",
                    session_id=session.id,
                    bucket=info.bucket if info else None,
                    object_key=info.key if info else None,
                    links={"segments": [str(segment.id) for segment in segments]},
                    metadata=stats,
                )
            )
        return stats
