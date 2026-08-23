"""Raw-SQL repository for the ``processing_jobs`` table.

Generic queue mechanics only — nothing here knows what any pipeline does.
Pipeline-specific discovery queries live in ``database/repos/pipelines/``.
"""

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from database.repos.base import BaseRepo
from database.schema.jobs import Job, JobCreate

# One channel for every pipeline; the payload names the pipeline so each
# worker can filter for its own wake-ups.
NOTIFY_CHANNEL = "processing_jobs"

COLUMNS = (
    "id, pipeline, session_id, artifact_id, priority, status, attempts, max_attempts, "
    "run_at, payload, result, error, dedup_key, claimed_at, claimed_by, finished_at, "
    "created_at, updated_at"
)
_J_COLUMNS = ", ".join(f"j.{column.strip()}" for column in COLUMNS.split(","))


class JobsRepo(BaseRepo):
    async def enqueue(self, data: JobCreate) -> Job | None:
        """Insert a job, or return None when its dedup key is already taken.

        The dedup index spans every status, so work that once succeeded (or
        died) is never re-enqueued under the same key. A fresh insert also
        NOTIFYs the workers' channel — delivered only when this transaction
        commits, so a listener can never see the notification before the row.
        """
        job = await self._fetch_one(
            Job,
            f"""
                INSERT INTO processing_jobs
                    (pipeline, session_id, artifact_id, priority, payload,
                     dedup_key, max_attempts)
                VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s, 5))
                ON CONFLICT (pipeline, dedup_key) WHERE dedup_key IS NOT NULL
                DO NOTHING
                RETURNING {COLUMNS}
            """,
            (
                data.pipeline,
                data.session_id,
                data.artifact_id,
                data.priority,
                Jsonb(data.payload),
                data.dedup_key,
                data.max_attempts,
            ),
        )
        if job is not None:
            await self._execute("SELECT pg_notify(%s, %s)", (NOTIFY_CHANNEL, data.pipeline))
        return job

    async def enqueue_many(self, items: Sequence[JobCreate]) -> int:
        """Insert a batch in one statement, skipping already-taken dedup keys.

        The discovery path: one round trip for a whole pass instead of a
        transaction per item, and one NOTIFY per pipeline that actually gained
        a row. Duplicate keys *within* the batch are dropped up front — ON
        CONFLICT refuses to skip the same row twice in one statement. Returns
        the number inserted.
        """
        seen: set[tuple[str, str]] = set()
        unique: list[JobCreate] = []
        for item in items:
            if item.dedup_key is not None:
                key = (item.pipeline, item.dedup_key)
                if key in seen:
                    continue
                seen.add(key)
            unique.append(item)
        if not unique:
            return 0

        params: list = []
        for item in unique:
            params += [
                item.pipeline,
                item.session_id,
                item.artifact_id,
                item.priority,
                Jsonb(item.payload),
                item.dedup_key,
                item.max_attempts,
            ]
        values = ", ".join(["(%s, %s, %s, %s, %s, %s, COALESCE(%s, 5))"] * len(unique))
        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"""
                    INSERT INTO processing_jobs
                        (pipeline, session_id, artifact_id, priority, payload,
                         dedup_key, max_attempts)
                    VALUES {values}
                    ON CONFLICT (pipeline, dedup_key) WHERE dedup_key IS NOT NULL
                    DO NOTHING
                    RETURNING pipeline
                """,
                params,
            )
            rows = await cur.fetchall()
        for pipeline in sorted({row["pipeline"] for row in rows}):
            await self._execute("SELECT pg_notify(%s, %s)", (NOTIFY_CHANNEL, pipeline))
        return len(rows)

    async def claim(self, pipeline: str, worker_id: str) -> Job | None:
        """Take the next runnable job for one pipeline, or None when idle.

        SKIP LOCKED makes concurrent claimers grab different rows instead of
        blocking. attempts is bumped here, at claim time, so a crash mid-run
        still counts against max_attempts after orphan recovery.
        """
        return await self._fetch_one(
            Job,
            f"""
                WITH next AS (
                    SELECT id FROM processing_jobs
                    WHERE pipeline = %s AND status = 'queued' AND run_at <= now()
                    ORDER BY priority DESC, run_at, id
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE processing_jobs j
                SET status = 'running', attempts = j.attempts + 1,
                    claimed_at = now(), claimed_by = %s
                FROM next WHERE j.id = next.id
                RETURNING {_J_COLUMNS}
            """,
            (pipeline, worker_id),
        )

    async def succeed(self, job_id: UUID, result: dict) -> Job | None:
        """Finish a running job. None means the row vanished mid-run
        (session cascade, test truncation); callers log and move on."""
        return await self._fetch_one(
            Job,
            f"""
                UPDATE processing_jobs
                SET status = 'succeeded', result = %s, error = NULL, finished_at = now()
                WHERE id = %s AND status = 'running'
                RETURNING {COLUMNS}
            """,
            (Jsonb(result), job_id),
        )

    async def retry(self, job_id: UUID, error: str, run_at: datetime) -> Job | None:
        """Return a failed job to the queue for another try at ``run_at``."""
        return await self._fetch_one(
            Job,
            f"""
                UPDATE processing_jobs
                SET status = 'queued', error = %s, run_at = %s,
                    claimed_at = NULL, claimed_by = NULL
                WHERE id = %s AND status = 'running'
                RETURNING {COLUMNS}
            """,
            (error, run_at, job_id),
        )

    async def mark_dead(self, job_id: UUID, error: str) -> Job | None:
        """Terminal failure: attempts are exhausted."""
        return await self._fetch_one(
            Job,
            f"""
                UPDATE processing_jobs
                SET status = 'dead', error = %s, finished_at = now()
                WHERE id = %s AND status = 'running'
                RETURNING {COLUMNS}
            """,
            (error, job_id),
        )

    async def requeue_running(self, pipeline: str) -> int:
        """Recover orphans at worker boot.

        Sound only because exactly one worker process runs per pipeline: any
        'running' row seen at startup belonged to a dead predecessor. 'dead'
        rows come back with a fresh attempt budget: they exhausted it against
        whatever outage this restart is meant to clear.
        """
        return await self._execute(
            """
                UPDATE processing_jobs
                SET status = 'queued', run_at = now(), claimed_at = NULL, claimed_by = NULL,
                    attempts = CASE WHEN status = 'dead' THEN 0 ELSE attempts END
                WHERE pipeline = %s AND ( status = 'running' OR status = 'dead')
            """,
            (pipeline,),
        )

    async def get(self, job_id: UUID) -> Job | None:
        return await self._fetch_one(
            Job, f"SELECT {COLUMNS} FROM processing_jobs WHERE id = %s", (job_id,)
        )

    async def list_for_session(
        self, session_id: UUID, limit: int = 50, offset: int = 0
    ) -> list[Job]:
        return await self._fetch_all(
            Job,
            f"""
                SELECT {COLUMNS} FROM processing_jobs
                WHERE session_id = %s
                ORDER BY created_at, id
                LIMIT %s OFFSET %s
            """,
            (session_id, limit, offset),
        )

    async def delete_for_session(self, session_id: UUID, pipeline: str) -> int:
        """Forget one pipeline's job history for a session so discovery
        re-queues its inputs (the anti-join sees no jobs, and dedup keys
        are freed). The redo-a-tier primitive; a mid-run worker's succeed()
        finding its row gone is harmless."""
        return await self._execute(
            "DELETE FROM processing_jobs WHERE session_id = %s AND pipeline = %s",
            (session_id, pipeline),
        )

    async def count_by_status(self, pipeline: str) -> dict[str, int]:
        async with self._conn.cursor() as cur:
            await cur.execute(
                "SELECT status, count(*) AS n FROM processing_jobs"
                " WHERE pipeline = %s GROUP BY status",
                (pipeline,),
            )
            rows = await cur.fetchall()
        return {row["status"]: row["n"] for row in rows}
