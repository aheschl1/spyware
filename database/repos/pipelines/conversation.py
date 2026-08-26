"""Discovery and input queries for the ``conversation`` pipeline."""

from uuid import UUID

from pydantic import BaseModel

from database.repos.artifacts import COLUMNS as _ARTIFACT_COLUMNS
from database.repos.base import BaseRepo
from database.schema.artifacts import PipelineArtifact

_A_COLUMNS = ", ".join(f"a.{column.strip()}" for column in _ARTIFACT_COLUMNS.split(","))


class UtteranceRow(BaseModel):
    """Just enough of an ``utterance`` artifact to group by gap."""

    id: UUID
    start_ms: int | None
    end_ms: int | None
    metadata: dict


class ConversationQueries(BaseRepo):
    async def maps_without_jobs(
        self, pipeline: str, source_pipeline: str, limit: int = 100
    ) -> list[PipelineArtifact]:
        """Diarize-map artifacts this pipeline has never been enqueued for.

        The map is diarize's completion marker, committed with its utterances;
        republication mints a new map, which re-triggers grouping.
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

    async def utterances(
        self, session_id: UUID, source_pipeline: str, limit: int
    ) -> list[UtteranceRow]:
        """Every utterance of one session, in timeline order."""
        return await self._fetch_all(
            UtteranceRow,
            """
                SELECT id, start_ms, end_ms, metadata FROM pipeline_artifacts
                WHERE session_id = %s AND pipeline = %s AND kind = 'utterance'
                ORDER BY start_ms, id
                LIMIT %s
            """,
            (session_id, source_pipeline, limit),
        )
