"""Discovery and input queries for the ``sound-span`` pipeline."""

from uuid import UUID

from pydantic import BaseModel

from database.repos.artifacts import COLUMNS as _ARTIFACT_COLUMNS
from database.repos.base import BaseRepo
from database.schema.artifacts import PipelineArtifact

_A_COLUMNS = ", ".join(f"a.{column.strip()}" for column in _ARTIFACT_COLUMNS.split(","))


class TagWindowRow(BaseModel):
    """Just enough of an ``audio-tag`` window to build spans from."""

    start_ms: int | None
    end_ms: int | None
    metadata: dict


class SoundSpanQueries(BaseRepo):
    async def maps_without_jobs(
        self, pipeline: str, source_pipeline: str, limit: int = 100
    ) -> list[PipelineArtifact]:
        """Audio-tag-map artifacts this pipeline has never been enqueued for.

        The map is audio-tag's completion marker, committed with its windows —
        so a discovered map always has its full window set. Republication mints
        a new map (new id), which re-triggers span building for that session.
        """
        return await self._fetch_all(
            PipelineArtifact,
            f"""
                SELECT {_A_COLUMNS} FROM pipeline_artifacts a
                WHERE a.pipeline = %s AND a.kind = 'audio-tag-map'
                  AND NOT EXISTS (
                      SELECT 1 FROM processing_jobs j
                      WHERE j.pipeline = %s AND j.artifact_id = a.id
                  )
                ORDER BY a.created_at, a.id
                LIMIT %s
            """,
            (source_pipeline, pipeline, limit),
        )

    async def tag_windows(
        self, session_id: UUID, source_pipeline: str, limit: int
    ) -> list[TagWindowRow]:
        """Every classified window of one session, in timeline order.

        Not ``artifacts.list_for_session``: that defaults to 50 rows and
        returns whole artifacts, where span building needs the entire set
        (~2200 windows for a three-hour session) and three columns of it.
        """
        return await self._fetch_all(
            TagWindowRow,
            """
                SELECT start_ms, end_ms, metadata FROM pipeline_artifacts
                WHERE session_id = %s AND pipeline = %s AND kind = 'audio-tag'
                ORDER BY start_ms, id
                LIMIT %s
            """,
            (session_id, source_pipeline, limit),
        )
