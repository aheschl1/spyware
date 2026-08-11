"""Discovery queries for the ``transcribe`` pipeline."""

from database.repos.base import BaseRepo
from database.schema.artifacts import PipelineArtifact
from database.repos.artifacts import COLUMNS as _ARTIFACT_COLUMNS

_A_COLUMNS = ", ".join(f"a.{column.strip()}" for column in _ARTIFACT_COLUMNS.split(","))


class TranscribeQueries(BaseRepo):
    async def utterances_without_jobs(
        self, pipeline: str, source_pipeline: str, limit: int = 100
    ) -> list[PipelineArtifact]:
        """Utterance artifacts this pipeline has never been enqueued for.

        Anti-joins on ``processing_jobs.artifact_id`` — a job is scoped to the
        artifact it consumes, and ``processing_jobs_artifact_idx`` serves the
        probe. The NOT EXISTS keeps discovery cheap; the dedup unique index is
        the correctness backstop if two passes race. When diarize republishes
        a session its utterances get new ids, so the anti-join naturally
        re-enqueues them — re-transcription is intended, the content changed.
        """
        return await self._fetch_all(
            PipelineArtifact,
            f"""
                SELECT {_A_COLUMNS} FROM pipeline_artifacts a
                WHERE a.pipeline = %s AND a.kind = 'utterance'
                  AND NOT EXISTS (
                      SELECT 1 FROM processing_jobs j
                      WHERE j.pipeline = %s AND j.artifact_id = a.id
                  )
                ORDER BY a.created_at, a.id
                LIMIT %s
            """,
            (source_pipeline, pipeline, limit),
        )
