"""Discovery and aggregation queries for the ``session-stats`` pipeline."""

from uuid import UUID

from pydantic import BaseModel

from database.repos.pipelines.common import PipelineDiscovery


class SessionSegmentAggregates(BaseModel):
    """Per-session totals, computed in SQL rather than by loading every row."""

    segments: int
    total_bytes: int
    duration_ms: int
    segment_ids: list[UUID]

class SessionStatsQueries(PipelineDiscovery):
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
