"""Discovery queries for the ``diarize`` pipeline."""

from database.repos.artifacts import COLUMNS as _ARTIFACT_COLUMNS
from database.repos.base import BaseRepo
from database.schema.artifacts import PipelineArtifact

_A_COLUMNS = ", ".join(f"a.{column.strip()}" for column in _ARTIFACT_COLUMNS.split(","))


class DiarizeQueries(BaseRepo):
    async def maps_without_jobs(
        self, pipeline: str, source_pipeline: str, limit: int = 100
    ) -> list[PipelineArtifact]:
        """Speech-map artifacts this pipeline has never been enqueued for.

        One map per session marks speech-detect completion, so consuming maps
        (not spans) gives exactly one diarize job per session. Anti-join on
        ``processing_jobs.artifact_id`` (indexed); the dedup unique index is
        the correctness backstop if two passes race.
        """
        return await self._fetch_all(
            PipelineArtifact,
            f"""
                SELECT {_A_COLUMNS} FROM pipeline_artifacts a
                WHERE a.pipeline = %s AND a.kind = 'speech-map'
                  AND NOT EXISTS (
                      SELECT 1 FROM processing_jobs j
                      WHERE j.pipeline = %s AND j.artifact_id = a.id
                  )
                ORDER BY a.created_at, a.id
                LIMIT %s
            """,
            (source_pipeline, pipeline, limit),
        )
