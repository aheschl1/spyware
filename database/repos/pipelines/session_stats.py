"""Discovery queries for the ``session-stats`` pipeline."""

from database.repos.base import BaseRepo
from database.schema.sessions import RecordingSession

_COLUMNS = ", ".join(
    f"s.{column}"
    for column in (
        "id, user_id, device, label, started_at, ended_at, "
        "metadata, created_at, updated_at"
    ).split(", ")
)


class SessionStatsQueries(BaseRepo):
    async def discover_unprocessed(
        self, pipeline: str, limit: int = 100
    ) -> list[RecordingSession]:
        """Ended sessions this pipeline has not yet been enqueued for.

        The NOT EXISTS only keeps discovery cheap; the dedup unique index on
        processing_jobs is the correctness backstop if two passes race.
        """
        return await self._fetch_all(
            RecordingSession,
            f"""
                SELECT {_COLUMNS} FROM recording_sessions s
                WHERE s.ended_at IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM processing_jobs j
                      WHERE j.pipeline = %s
                        AND j.dedup_key = %s || ':session:' || s.id
                  )
                ORDER BY s.ended_at
                LIMIT %s
            """,
            (pipeline, pipeline, limit),
        )
