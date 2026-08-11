"""Discovery queries for the ``speaker-cluster`` pipeline."""

from database.repos.artifacts import COLUMNS as _ARTIFACT_COLUMNS
from database.repos.base import BaseRepo
from database.schema.artifacts import PipelineArtifact

_A_COLUMNS = ", ".join(f"a.{column.strip()}" for column in _ARTIFACT_COLUMNS.split(","))


class SpeakerClusterQueries(BaseRepo):
    async def maps_without_jobs(
        self, pipeline: str, source_pipeline: str, limit: int = 100
    ) -> list[PipelineArtifact]:
        """Diarize-map artifacts this pipeline has never been enqueued for.

        The map is diarize's completion marker, committed atomically with the
        embeddings — so a discovered map always has its full embedding set.
        Republication mints a new map (new id), which re-triggers clustering
        for that session.
        """
        return await self._fetch_all(
            PipelineArtifact,
            f"""
                SELECT {_A_COLUMNS} FROM pipeline_artifacts a
                WHERE a.pipeline = %s AND a.kind = 'diarize-map'
                  AND NOT EXISTS (
                      SELECT 1 FROM processing_jobs j
                      WHERE j.pipeline = %s AND j.artifact_id = a.id
                  )
                ORDER BY a.created_at, a.id
                LIMIT %s
            """,
            (source_pipeline, pipeline, limit),
        )
