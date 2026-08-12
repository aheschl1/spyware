"""Cross-session state queries for the transcribe-ab tier."""

from uuid import UUID

from database.repos.base import BaseRepo


class AbQueries(BaseRepo):
    async def session_states(self, user_id: UUID) -> list[dict]:
        """Every enrolled session's latest run state plus live candidate
        counts — the overview's progress source. ``expected`` is 4 per
        utterance (2 models x 2 strategies); a degraded run can finish
        below it, so completion is judged by status, not the ratio."""
        async with self._conn.cursor() as cur:
            await cur.execute(
                """
                    SELECT s.id AS session_id,
                           j.status,
                           coalesce(c.n, 0) AS candidates,
                           coalesce(c.votable, 0) AS votable,
                           coalesce(u.n, 0) * 4 AS expected
                    FROM recording_sessions s
                    JOIN LATERAL (
                        SELECT status FROM processing_jobs
                        WHERE session_id = s.id AND pipeline = 'transcribe-ab'
                        ORDER BY created_at DESC LIMIT 1
                    ) j ON true
                    LEFT JOIN LATERAL (
                        SELECT count(*) AS n,
                               count(DISTINCT links->>'utterance') AS votable
                        FROM pipeline_artifacts
                        WHERE session_id = s.id AND pipeline = 'transcribe-ab'
                          AND kind = 'transcript-candidate'
                    ) c ON true
                    LEFT JOIN LATERAL (
                        SELECT count(*) AS n FROM pipeline_artifacts
                        WHERE session_id = s.id AND pipeline = 'diarize'
                          AND kind = 'utterance'
                    ) u ON true
                    WHERE s.user_id = %s
                """,
                (user_id,),
            )
            return list(await cur.fetchall())
