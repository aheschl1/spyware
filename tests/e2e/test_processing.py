"""The processing workers, end to end: a real ``python -m processing``
supervisor discovering, running, chaining, retrying, and deduplicating jobs.
"""

import asyncio
import json

import httpx

from database.pipe import DatabasePipe
from database.schema.jobs import JobStatus
from database.schema.sessions import SessionCreate
from tests.e2e.conftest import (
    TEST_BUCKET,
    Account,
    ingest,
    make_account,
    make_session,
    wait_for_job,
)
from tests.wav import wav_bytes


async def _ended_session_with_segments(account: Account, count: int = 3):
    """A finished session holding `count` one-tenth-second segments."""
    session = await make_session(account)
    payloads = [wav_bytes(seconds=0.1) for _ in range(count)]
    for payload in payloads:
        await ingest(session.id, payload, duration_ms=100)
    return session, payloads


async def test_session_stats_discovers_processes_and_stores(
    worker: None, client: httpx.AsyncClient, account: Account, s3
) -> None:
    session, payloads = await _ended_session_with_segments(account)
    ended = await client.post(f"/v1/sessions/{session.id}/end", headers=account.headers)
    assert ended.status_code == 200

    job = await wait_for_job(session.id, "session-stats", JobStatus.SUCCEEDED)
    assert job.priority == 0
    assert job.dedup_key == f"session-stats:session:{session.id}"
    assert job.result == {
        "segments": len(payloads),
        "total_bytes": sum(len(p) for p in payloads),
        "duration_ms": 100 * len(payloads),
    }

    async with DatabasePipe() as pipe:
        artifact = await pipe.artifacts.find("session-stats", "session-stats", session.id)
    assert artifact is not None
    assert artifact.bucket == TEST_BUCKET
    assert artifact.metadata == job.result
    assert len(artifact.links["segments"]) == len(payloads)

    # The blob landed in the pipeline's own space and matches the result.
    key = f"session-stats/sessions/{session.id}/stats.json"
    assert artifact.object_key == key
    stored = json.loads(s3.get_object(Bucket=TEST_BUCKET, Key=key)["Body"].read())
    assert stored == job.result


async def test_callback_chains_to_artifact_consumer(worker: None, clean_state) -> None:
    account = await make_account()
    session, _ = await _ended_session_with_segments(account, count=1)
    async with DatabasePipe() as pipe:
        await pipe.sessions.end(session.id)

    parent = await wait_for_job(session.id, "session-stats", JobStatus.SUCCEEDED)
    echoed = await wait_for_job(session.id, "stats-echo", JobStatus.SUCCEEDED)

    # The chained job carries the parent's linkage and consumed no segments —
    # its input was the parent's artifact.
    assert echoed.payload["source_job_id"] == str(parent.id)
    assert echoed.payload["source_result"] == parent.result
    assert echoed.result is not None
    assert echoed.result["echoed"]["source_job_id"] == str(parent.id)
    assert echoed.result["artifact_metadata"] == parent.result


async def test_failing_job_retries_then_dies(worker: None, clean_state) -> None:
    account = await make_account()
    async with DatabasePipe() as pipe:
        session = await pipe.sessions.create(
            SessionCreate(
                user_id=account.user.id, device="glasses-01", metadata={"boom": True}
            )
        )
        await pipe.sessions.end(session.id)

    job = await wait_for_job(session.id, "session-stats", JobStatus.DEAD)
    assert job.attempts == 2  # PROCESSING_MAX_ATTEMPTS in conftest
    assert job.error is not None and "boom" in job.error
    assert job.finished_at is not None


async def test_discovery_never_reprocesses_finished_work(
    worker: None, clean_state
) -> None:
    account = await make_account()
    session, _ = await _ended_session_with_segments(account, count=1)
    async with DatabasePipe() as pipe:
        await pipe.sessions.end(session.id)
    await wait_for_job(session.id, "session-stats", JobStatus.SUCCEEDED)

    # Several discovery passes later (poll interval is 0.2s), still one job.
    await asyncio.sleep(1.0)
    async with DatabasePipe() as pipe:
        jobs = await pipe.jobs.list_for_session(session.id)
    stats_jobs = [j for j in jobs if j.pipeline == "session-stats"]
    assert len(stats_jobs) == 1
