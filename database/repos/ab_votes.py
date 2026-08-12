"""Raw-SQL repository for ``ab_votes`` — transcription A/B winners."""

from uuid import UUID

from database.repos.base import BaseRepo
from database.schema.ab_votes import AbTallyRow, AbVote

COLUMNS = (
    "id, user_id, session_id, utterance_artifact_id, candidate_artifact_id, "
    "model, strategy, created_at"
)


class AbVotesRepo(BaseRepo):
    async def upsert(
        self,
        user_id: UUID,
        session_id: UUID,
        utterance_artifact_id: UUID,
        candidate_artifact_id: UUID,
        model: str,
        strategy: str,
    ) -> AbVote:
        return await self._fetch_one(
            AbVote,
            f"""
                INSERT INTO ab_votes
                    (user_id, session_id, utterance_artifact_id,
                     candidate_artifact_id, model, strategy)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (utterance_artifact_id) DO UPDATE
                    SET candidate_artifact_id = EXCLUDED.candidate_artifact_id,
                        model = EXCLUDED.model,
                        strategy = EXCLUDED.strategy,
                        created_at = now()
                RETURNING {COLUMNS}
            """,
            (user_id, session_id, utterance_artifact_id,
             candidate_artifact_id, model, strategy),
        )

    async def for_session(self, session_id: UUID) -> list[AbVote]:
        return await self._fetch_all(
            AbVote,
            f"SELECT {COLUMNS} FROM ab_votes WHERE session_id = %s",
            (session_id,),
        )

    async def tally(self, user_id: UUID) -> list[AbTallyRow]:
        return await self._fetch_all(
            AbTallyRow,
            """
                SELECT model, strategy, count(*) AS wins
                FROM ab_votes WHERE user_id = %s
                GROUP BY model, strategy
                ORDER BY wins DESC, model, strategy
            """,
            (user_id,),
        )

    async def counts_by_session(self, user_id: UUID) -> dict[UUID, int]:
        async with self._conn.cursor() as cur:
            await cur.execute(
                "SELECT session_id, count(*) AS n FROM ab_votes"
                " WHERE user_id = %s GROUP BY session_id",
                (user_id,),
            )
            rows = await cur.fetchall()
        return {row["session_id"]: row["n"] for row in rows}
