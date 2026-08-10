"""Queue mechanics and the artifacts registry, at the repository level.

No worker process here: these drive JobsRepo/ArtifactsRepo directly against
the migrated database, where ordering and locking can be asserted
deterministically.
"""

from datetime import UTC, datetime, timedelta

from psycopg import AsyncConnection

from database.config import get_settings as db_settings
from database.pipe import DatabasePipe
from database.repos.jobs import NOTIFY_CHANNEL
from database.schema.artifacts import ArtifactCreate
from database.schema.jobs import JobCreate, JobStatus
from tests.e2e.conftest import make_account, make_session


async def test_enqueue_returns_row_and_dedups_forever(clean_state) -> None:
    async with DatabasePipe() as pipe:
        first = await pipe.jobs.enqueue(JobCreate(pipeline="p", dedup_key="k"))
        second = await pipe.jobs.enqueue(JobCreate(pipeline="p", dedup_key="k"))
        assert first is not None and second is None

        # The key stays taken across the whole lifecycle, even after success.
        claimed = await pipe.jobs.claim("p", "w")
        assert claimed is not None
        await pipe.jobs.succeed(claimed.id, {"ok": True})
        assert await pipe.jobs.enqueue(JobCreate(pipeline="p", dedup_key="k")) is None

        # Same key under another pipeline, or no key at all: not deduped.
        assert await pipe.jobs.enqueue(JobCreate(pipeline="q", dedup_key="k")) is not None
        assert await pipe.jobs.enqueue(JobCreate(pipeline="p")) is not None
        assert await pipe.jobs.enqueue(JobCreate(pipeline="p")) is not None


async def test_claim_orders_by_priority_then_run_at(clean_state) -> None:
    async with DatabasePipe() as pipe:
        batch = await pipe.jobs.enqueue(JobCreate(pipeline="p", priority=0))
        live = await pipe.jobs.enqueue(JobCreate(pipeline="p", priority=100))
        assert batch is not None and live is not None

        first = await pipe.jobs.claim("p", "w")
        second = await pipe.jobs.claim("p", "w")
        assert first is not None and first.id == live.id  # higher priority wins
        assert second is not None and second.id == batch.id
        assert first.status is JobStatus.RUNNING
        assert first.attempts == 1
        assert first.claimed_by == "w" and first.claimed_at is not None
        assert await pipe.jobs.claim("p", "w") is None  # queue drained


async def test_claim_skips_jobs_deferred_by_retry(clean_state) -> None:
    async with DatabasePipe() as pipe:
        job = await pipe.jobs.enqueue(JobCreate(pipeline="p"))
        assert job is not None
        claimed = await pipe.jobs.claim("p", "w")
        assert claimed is not None

        future = datetime.now(UTC) + timedelta(hours=1)
        retried = await pipe.jobs.retry(claimed.id, "temporary failure", future)
        assert retried is not None
        assert retried.status is JobStatus.QUEUED
        assert retried.error == "temporary failure"
        assert retried.claimed_by is None

        assert await pipe.jobs.claim("p", "w") is None  # not runnable yet

        # Make it runnable again and observe attempts accumulating.
        await pipe.connection.execute(
            "UPDATE processing_jobs SET run_at = now() WHERE id = %s", (claimed.id,)
        )
        second_claim = await pipe.jobs.claim("p", "w2")
        assert second_claim is not None and second_claim.attempts == 2


async def test_concurrent_claims_take_different_jobs(clean_state) -> None:
    async with DatabasePipe() as pipe:
        a = await pipe.jobs.enqueue(JobCreate(pipeline="p"))
        b = await pipe.jobs.enqueue(JobCreate(pipeline="p"))
        assert a is not None and b is not None

    # Two open transactions: SKIP LOCKED must hand each a different row.
    async with DatabasePipe() as p1, DatabasePipe() as p2:
        j1 = await p1.jobs.claim("p", "w1")
        j2 = await p2.jobs.claim("p", "w2")
        assert j1 is not None and j2 is not None
        assert {j1.id, j2.id} == {a.id, b.id}


async def test_terminal_transitions_and_guards(clean_state) -> None:
    async with DatabasePipe() as pipe:
        job = await pipe.jobs.enqueue(JobCreate(pipeline="p"))
        assert job is not None

        # Only running jobs can finish: a queued one yields None.
        assert await pipe.jobs.succeed(job.id, {}) is None

        claimed = await pipe.jobs.claim("p", "w")
        assert claimed is not None
        done = await pipe.jobs.succeed(claimed.id, {"n": 3})
        assert done is not None
        assert done.status is JobStatus.SUCCEEDED
        assert done.result == {"n": 3} and done.finished_at is not None

        dead_job = await pipe.jobs.enqueue(JobCreate(pipeline="p"))
        assert dead_job is not None
        claimed = await pipe.jobs.claim("p", "w")
        assert claimed is not None
        dead = await pipe.jobs.mark_dead(claimed.id, "gave up")
        assert dead is not None
        assert dead.status is JobStatus.DEAD and dead.error == "gave up"


async def test_requeue_running_touches_only_its_pipeline(clean_state) -> None:
    async with DatabasePipe() as pipe:
        for pipeline in ("a", "b"):
            await pipe.jobs.enqueue(JobCreate(pipeline=pipeline))
            assert await pipe.jobs.claim(pipeline, "w") is not None

        assert await pipe.jobs.requeue_running("a") == 1
        assert (await pipe.jobs.count_by_status("a")) == {"queued": 1}
        assert (await pipe.jobs.count_by_status("b")) == {"running": 1}
        # The requeued job is immediately claimable again.
        assert await pipe.jobs.claim("a", "w") is not None


async def test_enqueue_notifies_on_commit(clean_state) -> None:
    listener = await AsyncConnection.connect(db_settings().conninfo, autocommit=True)
    try:
        await listener.execute(f"LISTEN {NOTIFY_CHANNEL}")
        async with DatabasePipe() as pipe:
            assert await pipe.jobs.enqueue(JobCreate(pipeline="notify-me")) is not None
        payloads = [
            notify.payload
            async for notify in listener.notifies(timeout=5.0, stop_after=1)
        ]
        assert payloads == ["notify-me"]
    finally:
        await listener.close()


async def test_artifacts_crud_and_find(clean_state) -> None:
    account = await make_account()
    session = await make_session(account)
    # Separate transactions, as separate job runs would be: created_at is
    # transaction time, and "newest wins" needs distinct timestamps.
    async with DatabasePipe() as pipe:
        older = await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="transcribe",
                kind="transcript",
                session_id=session.id,
                bucket="test-audio",
                object_key=f"transcribe/sessions/{session.id}/v1.json",
                links={"segments": ["s1"]},
                metadata={"words": 10},
            )
        )
    async with DatabasePipe() as pipe:
        newer = await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="transcribe",
                kind="transcript",
                session_id=session.id,
                object_key=f"transcribe/sessions/{session.id}/v2.json",
            )
        )
    async with DatabasePipe() as pipe:
        other_kind = await pipe.artifacts.create(
            ArtifactCreate(pipeline="transcribe", kind="summary", session_id=session.id)
        )

        found = await pipe.artifacts.find("transcribe", "transcript", session.id)
        assert found is not None and found.id == newer.id  # newest wins

        everything = await pipe.artifacts.list_for_session(session.id)
        assert {a.id for a in everything} == {older.id, newer.id, other_kind.id}
        transcripts = await pipe.artifacts.list_for_session(session.id, kind="transcript")
        assert {a.id for a in transcripts} == {older.id, newer.id}

        fetched = await pipe.artifacts.get(older.id)
        assert fetched is not None
        assert fetched.links == {"segments": ["s1"]} and fetched.metadata == {"words": 10}

        assert await pipe.artifacts.delete(other_kind.id) is True
        assert await pipe.artifacts.get(other_kind.id) is None
