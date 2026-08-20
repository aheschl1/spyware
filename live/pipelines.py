"""Live pipelines: consumers of one gated window each."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from live.base import LiveFrame, LivePipeline, LiveSessionContext
from live.config import get_settings

logger = logging.getLogger(__name__)


class CountingPipeline(LivePipeline):
    """Counts a gated window's frames — exercises the whole live loop."""

    name = "live-counter"

    async def run(self, ctx: LiveSessionContext, frames: AsyncIterator[LiveFrame]) -> None:
        ctx.emit("started", {})
        count = 0
        byte_size = 0
        try:
            async for frame in frames:
                count += 1
                byte_size += len(frame.pcm)
        finally:
            logger.info(
                "live-counter for session %s: %d frames, %d bytes",
                ctx.session_id, count, byte_size,
            )
            ctx.emit("finished", {"frames": count, "bytes": byte_size})


class TranscribePipeline(LivePipeline):
    """Streams a gated window to the ASR sidecar; emits partial transcripts.

    Events: ``started``, ``partial`` / ``final`` with ``{"text": ...}``, and
    ``error`` if the sidecar is unreachable. Audio is downmixed/resampled to
    the sidecar's required 16 kHz mono before sending.
    """

    name = "transcribe"

    async def run(self, ctx: LiveSessionContext, frames: AsyncIterator[LiveFrame]) -> None:
        import websockets

        settings = get_settings()
        ctx.emit("started", {})
        try:
            async with websockets.connect(
                settings.transcribe_url, max_size=None, open_timeout=5
            ) as ws:
                await ws.send(json.dumps({"sample_rate_hz": 16_000, "channels": 1}))
                ready = json.loads(await ws.recv())
                if ready.get("type") != "ready":
                    ctx.emit("error", {"error": ready.get("error", "sidecar not ready")})
                    return
                reader = asyncio.create_task(self._read_events(ctx, ws))
                try:
                    async for frame in frames:
                        await ws.send(_to_16k_mono(frame.pcm, ctx.sample_rate_hz, ctx.channels))
                    await ws.send(json.dumps({"type": "end"}))
                    await asyncio.wait_for(
                        reader, timeout=settings.transcribe_final_timeout_seconds
                    )
                finally:
                    reader.cancel()
        except Exception:
            logger.exception("transcribe pipeline failed for session %s", ctx.session_id)
            ctx.emit("error", {"error": "transcription unavailable"})

    async def _read_events(self, ctx: LiveSessionContext, ws) -> None:
        async for message in ws:
            data = json.loads(message)
            if data.get("type") == "partial":
                ctx.emit("partial", {"text": data["text"]})
            elif data.get("type") == "final":
                ctx.emit("final", {"text": data["text"]})
                return


def _to_16k_mono(pcm: bytes, sample_rate_hz: int, channels: int) -> bytes:
    if sample_rate_hz == 16_000 and channels == 1:
        return pcm
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if sample_rate_hz != 16_000:
        n = int(len(samples) * 16_000 / sample_rate_hz)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples
        )
    return samples.astype(np.int16).tobytes()
