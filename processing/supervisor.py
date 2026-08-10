"""One child process per pipeline, restarted on crash with backoff.

Deliberately synchronous - the supervisor only spawns, watches sentinels, and
forwards signals; all real work happens in the children (processing.worker).
"""

import logging
import multiprocessing
import signal
import time
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from types import FrameType

from processing.config import ProcessingSettings

logger = logging.getLogger(__name__)


class Supervisor:
    def __init__(
        self,
        names: tuple[str, ...],
        settings: ProcessingSettings,
        log_level: str = "info",
    ) -> None:
        self._names = names
        self._settings = settings
        self._log_level = log_level
        self._stop = False

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        logger.info("received %s; shutting down", signal.Signals(signum).name)
        self._stop = True

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        ctx = multiprocessing.get_context()

        # Deferred import: the worker module pulls in the registry, and with
        # forkserver each child re-imports it anyway.
        from processing.worker import run_worker

        children: dict[str, BaseProcess | None] = {name: None for name in self._names}
        started_at: dict[str, float] = {}
        not_before: dict[str, float] = {name: 0.0 for name in self._names}
        backoff_s: dict[str, float] = {
            name: self._settings.restart_backoff_base_seconds for name in self._names
        }

        def spawn(name: str) -> BaseProcess:
            process = ctx.Process(
                target=run_worker,
                args=(name, self._log_level),
                name=f"pipeline-{name}",
            )
            process.start()
            started_at[name] = time.monotonic()
            logger.info("started pipeline %s (pid %s)", name, process.pid)
            return process

        try:
            while not self._stop:
                now = time.monotonic()
                for name in self._names:
                    process = children[name]
                    if process is not None and process.is_alive():
                        continue
                    if process is not None:
                        # A child that ran cleanly for a while earns a reset.
                        if now - started_at[name] >= self._settings.restart_reset_seconds:
                            backoff_s[name] = self._settings.restart_backoff_base_seconds
                        logger.warning(
                            "pipeline %s exited (code %s); restarting in %.1fs",
                            name, process.exitcode, backoff_s[name],
                        )
                        not_before[name] = now + backoff_s[name]
                        backoff_s[name] = min(
                            self._settings.restart_backoff_cap_seconds, backoff_s[name] * 2
                        )
                        children[name] = None
                    if children[name] is None and now >= not_before[name]:
                        children[name] = spawn(name)

                sentinels = [
                    p.sentinel for p in children.values() if p is not None and p.is_alive()
                ]
                if sentinels:
                    # Wakes on any child exit; the timeout bounds restart latency.
                    connection.wait(sentinels, timeout=1.0)
                else:
                    time.sleep(0.2)
        finally:
            self._shutdown(children)

    def _shutdown(self, children: dict[str, BaseProcess | None]) -> None:
        live = [p for p in children.values() if p is not None and p.is_alive()]
        for process in live:
            process.terminate()  # SIGTERM: workers finish the in-flight job
        deadline = time.monotonic() + self._settings.shutdown_grace_seconds
        for process in live:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                logger.error("pipeline %s ignored SIGTERM; killing", process.name)
                process.kill()
                process.join()
        logger.info("supervisor stopped")
