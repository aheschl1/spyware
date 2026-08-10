"""A stand-in transcription service for the e2e suite.

Speaks just enough of the standard transcriptions API for the transcribe tier:
``POST /v1/audio/transcriptions`` answers a fixed text plus the received body
size, so tests can assert the clip actually crossed the wire. Stdlib only —
it runs as ``python -m tests.e2e.stub_transcriber PORT`` next to the real
worker process.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STUB_TEXT = "hello from the stub transcriber"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server's naming)
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        if not self.path.endswith("/audio/transcriptions"):
            self.send_error(404)
            return
        payload = json.dumps({"text": STUB_TEXT, "received_bytes": len(body)}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("content-length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: object) -> None:
        pass  # keep the test output quiet


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), _Handler).serve_forever()
