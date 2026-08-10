"""Discovery and aggregation queries for the ``session-stats`` pipeline."""

from uuid import UUID

from pydantic import BaseModel

from database.repos.base import BaseRepo
from database.schema.sessions import RecordingSession


class SessionSegmentAggregates(BaseModel):
    """Per-session totals, computed in SQL rather than by loading every row."""

    segments: int
    total_bytes: int
    duration_ms: int
    segment_ids: list[UUID]

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
        processing_jobs is the correctness backstop if two passes race. It
        anti-joins on ``(pipeline, session_id)`` — indexed on both sides —
        rather than reconstructing the dedup key per row, which no index can
        serve. (Any job of this pipeline for the session suppresses discovery,
        which for a one-job-per-session pipeline is the same set.)
        """
        return await self._fetch_all(
            RecordingSession,
            f"""
                SELECT {_COLUMNS} FROM recording_sessions s
                WHERE s.ended_at IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM processing_jobs j
                      WHERE j.pipeline = %s AND j.session_id = s.id
                  )
                ORDER BY s.ended_at
                LIMIT %s
            """,
            (pipeline, limit),
        )

    async def aggregate_segments(self, session_id: UUID) -> SessionSegmentAggregates:
        """One aggregate row instead of every segment across the wire.

        A day-long chunk-per-second session is 86 400 rows; the stats need
        three sums and the id list, so let Postgres do the arithmetic.
        """
        aggregates = await self._fetch_one(
            SessionSegmentAggregates,
            """
                SELECT COUNT(*) AS segments,
                       COALESCE(SUM(byte_size), 0) AS total_bytes,
                       COALESCE(SUM(duration_ms), 0) AS duration_ms,
                       COALESCE(ARRAY_AGG(id ORDER BY sequence), ARRAY[]::uuid[])
                           AS segment_ids
                FROM audio_segments WHERE session_id = %s
            """,
            (session_id,),
        )
        assert aggregates is not None  # aggregate without GROUP BY always returns a row
        return aggregates
