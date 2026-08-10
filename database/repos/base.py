"""Shared plumbing for repositories.

Repositories run raw SQL against the connection handed to them by
:class:`~database.pipe.DatabasePipe`; every call inside one ``async with`` block
shares that connection's transaction.
"""

from typing import Any, Mapping, Sequence, TypeVar

from psycopg import AsyncConnection
from psycopg.rows import dict_row, tuple_row
from pydantic import BaseModel

Params = Sequence[Any] | Mapping[str, Any] | None


class BaseRepo[M: BaseModel]:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def _fetch_one(self, model: type[M], sql: str, params: Params = None) -> M | None:
        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            row = await cur.fetchone()
        return model.model_validate(row) if row is not None else None

    async def _fetch_all(self, model: type[M], sql: str, params: Params = None) -> list[M]:
        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [model.model_validate(row) for row in rows]

    async def _execute(self, sql: str, params: Params = None) -> int:
        """Run a statement and return the number of rows affected."""
        async with self._conn.cursor() as cur:
            await cur.execute(sql, params)
            return cur.rowcount

    async def _fetch_value(self, sql: str, params: Params = None) -> Any | None:
        """Run a statement and return the first column of the first row."""
        # Explicit tuple rows: the pool sets dict_row as the connection default.
        async with self._conn.cursor(row_factory=tuple_row) as cur:
            await cur.execute(sql, params)
            row = await cur.fetchone()
        return row[0] if row is not None else None
