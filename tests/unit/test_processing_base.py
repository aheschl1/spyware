"""Pipeline parent-class semantics, pure and Docker-free."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from database.schema.jobs import Job, JobCreate, JobStatus
from processing.base import Pipeline, backoff
from processing.config import ProcessingSettings


class EchoPipeline(Pipeline):
    name = "echo-test"

    async def discover(self, limit: int) -> tuple[JobCreate, ...]:
        return ()

    async def process(self, job: Job) -> dict[str, Any]:
        return dict(job.payload)


def make_job(**overrides) -> Job:
    now = datetime.now(UTC)
    defaults: dict[str, Any] = {
        "id": uuid4(),
        "pipeline": "echo-test",
        "session_id": None,
        "priority": 0,
        "status": JobStatus.RUNNING,
        "attempts": 1,
        "max_attempts": 5,
        "run_at": now,
        "payload": {},
        "created_at": now,
        "updated_at": now,
    }
    return Job(**{**defaults, **overrides})


async def test_maybe_callback_without_callback_is_a_noop() -> None:
    await EchoPipeline().maybe_callback(make_job(), {"x": 1})


async def test_maybe_callback_invokes_with_job_and_result() -> None:
    seen: list[tuple[Job, dict]] = []

    async def record(job: Job, result: dict) -> None:
        seen.append((job, result))

    job = make_job()
    await EchoPipeline(callback=record).maybe_callback(job, {"x": 1})
    assert seen == [(job, {"x": 1})]


async def test_maybe_callback_swallows_and_logs_exceptions(caplog) -> None:
    async def explode(job: Job, result: dict) -> None:
        raise RuntimeError("callback boom")

    job = make_job()
    with caplog.at_level(logging.ERROR, logger="processing.base"):
        # Must not raise: the job is already committed as succeeded.
        await EchoPipeline(callback=explode).maybe_callback(job, {})
    assert "callback failed" in caplog.text
    assert str(job.id) in caplog.text


def test_backoff_doubles_from_base_and_caps() -> None:
    settings = ProcessingSettings(
        retry_backoff_base_seconds=2.0, retry_backoff_cap_seconds=300.0
    )
    assert backoff(1, settings) == 2.0
    assert backoff(2, settings) == 4.0
    assert backoff(3, settings) == 8.0
    assert backoff(9, settings) == 300.0  # 512 capped


def test_job_models_construct_from_plain_dicts() -> None:
    job = make_job(result={"ok": True}, dedup_key="k")
    assert job.status is JobStatus.RUNNING
    assert job.result == {"ok": True}
    create = JobCreate(pipeline="echo-test", dedup_key="k")
    assert create.max_attempts is None  # defers to the table default
