"""The wakeword gate: slices a session's stream into gated windows.

Idle, the gate feeds the detector and keeps a pre-roll ring. On a trigger it
starts one instance of each enabled pipeline behind a bounded drop-oldest
queue, feeds the pre-roll then live frames, and closes the window after
``gate_silence_close_ms`` of trailing non-speech or ``gate_window_ms`` of
audio, whichever comes first. Pipeline finalization runs off the feed path —
a slow finalizer must not stall the connection's read loop — and the gate
refuses to re-trigger until it completes. Pipelines never see the wakeword
machinery.
"""

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Any
from uuid import UUID

from live.base import LiveFrame, LivePipeline, LiveSessionContext
from live.config import LiveSettings
from live.detect import KWS_SAMPLE_RATE, SilenceTracker, WakewordDetector

logger = logging.getLogger(__name__)

EmitEvent = Callable[[str, str, dict[str, Any]], None]  # (effect, event, data)


class _FrameFeed:
    """A bounded frame queue exposed to a pipeline as an async iterator."""

    def __init__(self, limit: int) -> None:
        self._queue: asyncio.Queue[LiveFrame | None] = asyncio.Queue(limit)

    def push(self, item: LiveFrame | None) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._queue.get_nowait()  # drop the oldest; live audio must stay fresh
            self._queue.put_nowait(item)

    async def __aiter__(self) -> AsyncIterator[LiveFrame]:
        while (frame := await self._queue.get()) is not None:
            yield frame


class WakewordGate:
    def __init__(
        self,
        detector: WakewordDetector,
        settings: LiveSettings,
        pipelines: tuple[type[LivePipeline], ...],
        *,
        session_id: UUID,
        user_id: UUID,
        sample_rate_hz: int,
        channels: int,
        emit: EmitEvent,
    ) -> None:
        self._detector = detector
        self._settings = settings
        self._pipelines = pipelines
        self._session_id = session_id
        self._user_id = user_id
        self._sample_rate_hz = sample_rate_hz
        self._channels = channels
        self._emit = emit
        self._bytes_per_ms = sample_rate_hz * channels * 2 / 1000
        self._preroll: deque[LiveFrame] = deque()
        self._preroll_bytes = 0
        self._feeds: list[_FrameFeed] = []
        self._tasks: list[asyncio.Task] = []
        self._window_bytes = 0
        self._silence: SilenceTracker | None = None
        self._finalizing: asyncio.Task | None = None

    @property
    def active(self) -> bool:
        return bool(self._tasks)

    async def feed(self, frame: LiveFrame) -> None:
        self._remember(frame)  # the ring stays warm for the next trigger too
        if self.active:
            for feed in self._feeds:
                feed.push(frame)
            self._window_bytes += len(frame.pcm)
            closing = self._window_bytes >= self._settings.gate_window_ms * self._bytes_per_ms
            if not closing and self._silence is not None:
                closing = (
                    self._silence.feed(frame.pcm)
                    >= self._settings.gate_silence_close_ms
                )
            if closing:
                self._close_window()
            return
        if self._finalizing is not None:
            if not self._finalizing.done():
                return
            self._finalizing = None
        if self._pipelines and self._detector.feed(frame.pcm):
            self._open_window()

    async def close(self) -> None:
        """End of stream: finish any open window."""
        if self.active:
            self._close_window()
        if self._finalizing is not None:
            await self._finalizing
            self._finalizing = None

    def _remember(self, frame: LiveFrame) -> None:
        self._preroll.append(frame)
        self._preroll_bytes += len(frame.pcm)
        limit = self._settings.preroll_ms * self._bytes_per_ms
        while self._preroll and self._preroll_bytes - len(self._preroll[0].pcm) >= limit:
            self._preroll_bytes -= len(self._preroll.popleft().pcm)

    def _open_window(self) -> None:
        logger.info("wakeword trigger on session %s", self._session_id)
        self._window_bytes = 0
        if (
            self._settings.gate_silence_close_ms > 0
            and self._sample_rate_hz == KWS_SAMPLE_RATE
            and self._channels == 1
        ):
            self._silence = SilenceTracker(self._settings.kws_vad_threshold)
        for cls in self._pipelines:
            feed = _FrameFeed(self._settings.pipeline_queue_frames)
            for frame in self._preroll:
                feed.push(frame)
            ctx = LiveSessionContext(
                session_id=self._session_id,
                user_id=self._user_id,
                sample_rate_hz=self._sample_rate_hz,
                channels=self._channels,
                emit=partial(self._emit, cls.name),
            )
            self._feeds.append(feed)
            self._tasks.append(asyncio.create_task(self._run(cls(), ctx, feed)))

    async def _run(self, pipeline: LivePipeline, ctx: LiveSessionContext, feed: _FrameFeed) -> None:
        try:
            await pipeline.run(ctx, aiter(feed))
        except Exception:
            logger.exception("live pipeline %s crashed", pipeline.name)

    def _close_window(self) -> None:
        for feed in self._feeds:
            feed.push(None)
        tasks, self._tasks, self._feeds = self._tasks, [], []
        self._silence = None
        self._detector.reset()
        self._finalizing = asyncio.create_task(self._finalize(tasks))

    async def _finalize(self, tasks: list[asyncio.Task]) -> None:
        await asyncio.gather(*tasks)
