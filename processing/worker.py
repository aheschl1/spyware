"""The per-pipeline worker: one OS process, one pipeline, one loop.

Each pass discovers new work (the pipeline's own query), drains the queue for
this pipeline, then sleeps on the NOTIFY channel with the poll interval as a
timeout. Polling is the source of truth - NOTIFY only shortens the latency of
chained enqueues - so a worker that missed every notification still processes
everything.
"""

import asyncio
import logging
import os
import signal
import socket
import traceback
from contextlib import suppress
from datetime import UTC, datetime, timedelta

from psycopg import AsyncConnection

from database.config import get_settings as get_database_settings
from database.pipe import DatabasePipe, close_pool
from database.repos.jobs import NOTIFY_CHANNEL
from database.schema.jobs import Job
from processing import registry
from processing.base import Pipeline, backoff
from processing.config import ProcessingSettings, get_settings
from storage.pipe import close_blob_client

logger = logging.getLogger(__name__)


def run_worker(name: str, log_level: str = "info") -> None:
    """Child-process entry point. Module-level and string-argumented so the
    supervisor can spawn it under any multiprocessing start method."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_worker_main(name))


async def _worker_main(name: str) -> None:
    settings = get_settings()
    pipeline = registry.build(name)
    worker_id = f"{name}@{socket.gethostname()}:{os.getpid()}"

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    await pipeline.setup()
    listener: AsyncConnection | None = None
    try:
        # Any 'running' row at boot belonged to a dead predecessor — there is
        # exactly one worker per pipeline (see docs/processing-pipelines.md).
        async with DatabasePipe() as pipe:
            requeued = await pipe.jobs.requeue_running(name)
        listener = await _connect_listener()
        logger.info("worker %s ready (requeued %d orphaned job(s))", name, requeued)

        while not stop.is_set():
            try:
                await _discover(pipeline, settings)
                await _drain(pipeline, worker_id, settings, stop)
            except Exception:
                # A DB outage should degrade to retrying next pass, not kill
                # the child into the supervisor's restart backoff.
                logger.exception("worker %s pass failed; retrying", name)
            listener = await _wait(listener, name, settings, stop)
    finally:
        await pipeline.teardown()
        if listener is not None:
            with suppress(Exception):
                await listener.close()
        await close_blob_client()
        await close_pool()
        logger.info("worker %s stopped", name)


async def _discover(pipeline: Pipeline, settings: ProcessingSettings) -> None:
    items = await pipeline.discover(settings.discovery_batch)
    if not items:
        return
    stamped = [
        item
        if item.max_attempts is not None
        else item.model_copy(update={"max_attempts": settings.max_attempts})
        for item in items
    ]
    async with DatabasePipe() as pipe:
        await pipe.jobs.enqueue_many(stamped)  # deduped rows just don't insert


async def _drain(
    pipeline: Pipeline,
    worker_id: str,
    settings: ProcessingSettings,
    stop: asyncio.Event,
) -> None:
    """Claim and run jobs until the queue is empty (or we are stopping).

    The claim transaction never spans process(): the job is committed as
    'running' before the work starts, and finished in a second transaction.
    """
    while not stop.is_set():
        async with DatabasePipe() as pipe:
            job = await pipe.jobs.claim(pipeline.name, worker_id)
        if job is None:
            return
        await _run_one(pipeline, job, settings)


async def _run_one(pipeline: Pipeline, job: Job, settings: ProcessingSettings) -> None:
    try:
        result = await pipeline.process(job)
    except Exception:
        error = traceback.format_exc()
        async with DatabasePipe() as pipe:
            if job.attempts >= job.max_attempts:
                logger.error("%s job %s dead after %d attempts", pipeline.name, job.id, job.attempts)
                await pipe.jobs.mark_dead(job.id, error)
            else:
                delay = backoff(job.attempts, settings)
                logger.warning(
                    "%s job %s failed (attempt %d/%d); retrying in %.1fs",
                    pipeline.name, job.id, job.attempts, job.max_attempts, delay,
                )
                run_at = datetime.now(UTC) + timedelta(seconds=delay)
                await pipe.jobs.retry(job.id, error, run_at)
        return

    async with DatabasePipe() as pipe:
        updated = await pipe.jobs.succeed(job.id, result)
    if updated is None:
        # The row vanished mid-run (session cascade, test truncation).
        logger.warning("%s job %s vanished mid-run; result discarded", pipeline.name, job.id)
        return
    await pipeline.maybe_callback(updated, result)


async def _connect_listener() -> AsyncConnection | None:
    """A dedicated autocommit connection for LISTEN, outside the pool."""
    try:
        conn = await AsyncConnection.connect(
            get_database_settings().conninfo, autocommit=True
        )
        await conn.execute(f"LISTEN {NOTIFY_CHANNEL}")
        return conn
    except Exception:
        logger.exception("LISTEN connection failed; degrading to plain polling")
        return None


async def _wait(
    listener: AsyncConnection | None,
    name: str,
    settings: ProcessingSettings,
    stop: asyncio.Event,
) -> AsyncConnection | None:
    """Sleep until woken: a NOTIFY for this pipeline, or the poll interval.

    Returns the (possibly reconnected, possibly dropped) listener connection.
    """
    if stop.is_set():
        return listener
    if listener is None or listener.closed:
        listener = await _connect_listener()
    if listener is None:
        # No LISTEN available: plain interruptible sleep.
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), settings.poll_interval_seconds)
        return None
    try:
        async for notify in listener.notifies(timeout=settings.poll_interval_seconds):
            # An empty payload is a broadcast; anything else names a pipeline.
            if notify.payload in ("", name) or stop.is_set():
                break
    except Exception:
        logger.exception("LISTEN wait failed; degrading to plain polling")
        with suppress(Exception):
            await listener.close()
        return None
    return listener
