"""Stand-in audio model services for the e2e suite.

Speaks just enough of the transcription, diarization and classification
contracts for the worker tiers, with deterministic canned answers; response
bodies echo the received byte count so tests can assert the clip actually
crossed the wire. Stdlib only — runs as ``python -m
tests.e2e.stub_audio_services PORT`` next to the real worker process.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STUB_TEXT = "hello from the stub transcriber"

# Two turns and two speakers, fixed: the e2e sessions are ~300ms of tone.
STUB_TURNS = [
    {"start_ms": 0, "end_ms": 150, "speaker": "SPEAKER_00"},
    {"start_ms": 150, "end_ms": 300, "speaker": "SPEAKER_01"},
]
STUB_EMBEDDINGS = {
    "SPEAKER_00": [1.0, 0.0, 0.0, 0.0],
    "SPEAKER_01": [0.0, 1.0, 0.0, 0.0],
}
STUB_DIAR_MODEL = "stub-diarizer"

# Two windows, fixed: Music persists across both (qualifies as a session tag
# with the default min_consecutive=2), Speech appears once (window-only).
STUB_WINDOWS = [
    {
        "start_ms": 0,
        "end_ms": 150,
        "labels": [
            {"label": "Music", "score": 0.9},
            {"label": "Speech", "score": 0.35},
        ],
        "embedding": [1.0, 0.0, 0.0, 0.0],
    },
    {
        "start_ms": 150,
        "end_ms": 300,
        "labels": [{"label": "Music", "score": 0.8}],
        "embedding": [0.0, 1.0, 0.0, 0.0],
    },
]
STUB_TAGGER_MODEL = "stub-tagger"
STUB_CLAP_MODEL = "stub-clap"
# Nearest STUB_WINDOWS[0] by cosine distance, so search tests rank window one first.
STUB_TEXT_EMBEDDING = [1.0, 0.0, 0.0, 0.0]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server's naming)
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        if self.path.endswith("/audio/transcriptions"):
            payload = {"text": STUB_TEXT, "received_bytes": len(body)}
        elif self.path.endswith("/audio/diarizations"):
            payload = {
                "turns": STUB_TURNS,
                "embeddings": STUB_EMBEDDINGS,
                "model": STUB_DIAR_MODEL,
                "received_bytes": len(body),
            }
        elif self.path.endswith("/audio/analyze"):
            payload = {
                "windows": STUB_WINDOWS,
                "window_ms": 10_000,
                "hop_ms": 5_000,
                "models": {"tagger": STUB_TAGGER_MODEL, "clap": STUB_CLAP_MODEL},
                "received_bytes": len(body),
            }
        elif self.path.endswith("/text/embeddings"):
            payload = {"embeddings": [STUB_TEXT_EMBEDDING], "model": STUB_CLAP_MODEL}
        else:
            self.send_error(404)
            return
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("content-length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: object) -> None:
        pass  # keep the test output quiet


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), _Handler).serve_forever()
