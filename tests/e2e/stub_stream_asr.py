"""Stand-in streaming ASR websocket for the e2e suite.

Speaks the sidecar's /v1/audio/stream contract with canned answers: ready on
hello, one partial after the first audio frame, and a final echoing the
received byte count on ``{"type": "end"}``. Runs as ``python -m
tests.e2e.stub_stream_asr PORT`` next to the real API process.
"""

import asyncio
import json
import sys

import websockets

STUB_PARTIAL = "stub partial"
STUB_FINAL_PREFIX = "stub final"


async def _serve(ws) -> None:
    hello = json.loads(await ws.recv())
    if hello.get("sample_rate_hz") != 16_000 or hello.get("channels", 1) != 1:
        await ws.send(json.dumps({"type": "error", "error": "16 kHz mono s16le required"}))
        await ws.close(code=1003)
        return
    await ws.send(json.dumps({"type": "ready", "model": "stub-streaming"}))
    received = 0
    async for message in ws:
        if isinstance(message, bytes):
            if received == 0:
                await ws.send(json.dumps({"type": "partial", "text": STUB_PARTIAL}))
            received += len(message)
        else:
            await ws.send(
                json.dumps({"type": "final", "text": f"{STUB_FINAL_PREFIX} {received}"})
            )
            await ws.close()
            return


async def _main(port: int) -> None:
    async with websockets.serve(_serve, "127.0.0.1", port):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(_main(int(sys.argv[1])))
