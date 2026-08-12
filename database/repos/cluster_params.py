"""Raw-SQL repository for ``cluster_params`` — per-user clustering overrides.

A row with only NULL values is meaningless (every field inherits env), so
``set`` deletes the row instead of storing an all-NULL husk.
"""

from uuid import UUID

from database.repos.base import BaseRepo
from database.schema.cluster_params import ClusterParams

COLUMNS = "user_id, cluster_distance, min_talk_ms, updated_at"


class ClusterParamsRepo(BaseRepo):
    async def get(self, user_id: UUID) -> ClusterParams | None:
        return await self._fetch_one(
            ClusterParams,
            f"SELECT {COLUMNS} FROM cluster_params WHERE user_id = %s",
            (user_id,),
        )

    async def set(
        self,
        user_id: UUID,
        cluster_distance: float | None,
        min_talk_ms: int | None,
    ) -> ClusterParams | None:
        """Replace the user's overrides wholesale; all-None clears the row."""
        if cluster_distance is None and min_talk_ms is None:
            await self.clear(user_id)
            return None
        return await self._fetch_one(
            ClusterParams,
            f"""
                INSERT INTO cluster_params (user_id, cluster_distance, min_talk_ms)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                    SET cluster_distance = EXCLUDED.cluster_distance,
                        min_talk_ms = EXCLUDED.min_talk_ms
                RETURNING {COLUMNS}
            """,
            (user_id, cluster_distance, min_talk_ms),
        )

    async def clear(self, user_id: UUID) -> bool:
        return await self._execute(
            "DELETE FROM cluster_params WHERE user_id = %s", (user_id,)
        ) > 0
