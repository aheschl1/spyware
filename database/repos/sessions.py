"""Raw-SQL repository for the ``recording_sessions`` table."""

from datetime import datetime
from uuid import UUID

from psycopg import errors
from psycopg.types.json import Jsonb

from database.exceptions import NotFoundError
from database.repos.base import BaseRepo
from database.schema.sessions import RecordingSession, SessionCreate

COLUMNS = (
    "id, user_id, device, label, started_at, ended_at, metadata, created_at, updated_at"
)


class SessionsRepo(BaseRepo):
    async def create(self, data: SessionCreate) -> RecordingSession:
        sql = f"""
            INSERT INTO recording_sessions (user_id, device, label, started_at, metadata)
            VALUES (%s, %s, %s, COALESCE(%s, now()), %s)
            RETURNING {COLUMNS}
        """
        params = (
            data.user_id,
            data.device,
            data.label,
            data.started_at,
            Jsonb(data.metadata),
        )
        try:
            async with self._conn.transaction():
                session = await self._fetch_one(RecordingSession, sql, params)
        except errors.ForeignKeyViolation as exc:
            raise NotFoundError("user", data.user_id) from exc
        assert session is not None  # INSERT ... RETURNING always yields a row
        return session

    async def get(self, session_id: UUID) -> RecordingSession | None:
        return await self._fetch_one(
            RecordingSession,
            f"SELECT {COLUMNS} FROM recording_sessions WHERE id = %s",
            (session_id,),
        )

    async def list_for_user(
        self,
        user_id: UUID,
        open_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RecordingSession]:
        sql = f"SELECT {COLUMNS} FROM recording_sessions WHERE user_id = %s"
        if open_only:
            sql += " AND ended_at IS NULL"
        sql += " ORDER BY started_at DESC LIMIT %s OFFSET %s"
        return await self._fetch_all(RecordingSession, sql, (user_id, limit, offset))

    async def end(self, session_id: UUID, ended_at: datetime | None = None) -> RecordingSession:
        """Close a session. Re-ending one moves the timestamp."""
        session = await self._fetch_one(
            RecordingSession,
            f"""
                UPDATE recording_sessions SET ended_at = COALESCE(%s, now())
                WHERE id = %s
                RETURNING {COLUMNS}
            """,
            (ended_at, session_id),
        )
        if session is None:
            raise NotFoundError("recording session", session_id)
        return session

    async def delete(self, session_id: UUID) -> bool:
        """Delete a session and, by cascade, its segment rows.

        Blobs are not touched; use ``services.audio.delete_session`` for those.
        """
        return await self._execute(
            "DELETE FROM recording_sessions WHERE id = %s", (session_id,)
        ) > 0
