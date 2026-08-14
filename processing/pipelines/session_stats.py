"""Demo pipeline: per-session aggregates over every segment.

Exists to prove the plumbing — self-discovery, whole-session consumption, the
pipeline blob space, and artifact registration. Real pipelines (transcription,
...) follow this exact shape.
"""

import json
from collections.abc import Sequence
import logging
from typing import Any

from database.pipe import DatabasePipe
from database.repos.pipelines.session_stats import SessionStatsQueries
from database.schema.artifacts import ArtifactCreate
from database.schema.jobs import Job, JobCreate
from processing.base import Pipeline
from storage.keys import pipeline_key
from storage.pipe import BlobPipe


class SessionStatsPipeline(Pipeline):
    name = "session-stats"
    resource = "audio"

    async def discover(self, limit: int) -> Sequence[JobCreate]:
        async with DatabasePipe() as pipe:
            queries = SessionStatsQueries(pipe.connection)
            sessions = await queries.ended_sessions_with_resource_without(
                self.name, self.resource, limit
            )
        logging.debug(f"pipeline {self.name} found {len(sessions)} jobs")
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
            aggregates = await SessionStatsQueries(pipe.connection).aggregate_segments(
                job.session_id
            )
        if session is None:
            return {"missing": True}
        if session.metadata.get("boom"):
            # Deterministic failure hook for exercising retry-to-dead.
            raise RuntimeError("boom requested by session metadata")

        stats = {
            "segments": aggregates.segments,
            "total_bytes": aggregates.total_bytes,
            "duration_ms": aggregates.duration_ms,
        }
        info = None
        if aggregates.segments:
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
                    links={"segments": [str(id) for id in aggregates.segment_ids]},
                    metadata=stats,
                )
            )
        return stats
