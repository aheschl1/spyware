"""The transcription-service client seam.

The transcribe tier depends on this interface, not on any particular model or
container. Two wire protocols cover every practical backend:

- ``openai`` — the standard transcriptions API (``POST {base}/audio/
  transcriptions``, multipart), spoken by our own asr_parakeet container, by
  speaches/faster-whisper, and by hosted providers. This is the default.
  Backends may extend the response with clip-relative ``words`` timestamps
  and a ``language``; both are optional and text-only backends keep working.
- ``cog`` — Replicate cog images run locally (``POST {base}/predictions``,
  base64 JSON), for out-of-the-box community containers.

Swapping backends is environment-only (``PROCESSING_TRANSCRIBER_*``).
"""

import base64
from dataclasses import dataclass
from typing import Any

import httpx

from processing.base import ServiceUnavailable
from processing.config import ProcessingSettings


class TranscriberError(Exception):
    """The service failed or answered something unusable. Retryable."""


class TranscriberUnavailable(TranscriberError, ServiceUnavailable):
    """The service itself is down (unreachable or 503)."""


@dataclass(frozen=True, slots=True)
class Word:
    word: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class Transcription:
    text: str
    raw: dict[str, Any]  # the service's full response, persisted verbatim
    words: tuple[Word, ...] | None = None  # clip-relative ms; None if the backend has none
    language: str | None = None


def _parse_words(body: dict[str, Any]) -> tuple[Word, ...] | None:
    entries = body.get("words")
    if not isinstance(entries, list):
        return None
    words = []
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        word, start, end = entry.get("word"), entry.get("start_ms"), entry.get("end_ms")
        if not isinstance(word, str) or not isinstance(start, int) or not isinstance(end, int):
            return None
        words.append(Word(word=word, start_ms=start, end_ms=end))
    return tuple(words)


class Transcriber:
    def __init__(self, settings: ProcessingSettings) -> None:
        self._base_url = settings.transcriber_base_url.rstrip("/")
        self._model = settings.transcriber_model
        self._protocol = settings.transcriber_protocol
        if self._protocol not in ("openai", "cog"):
            raise ValueError(f"unknown transcriber protocol {self._protocol!r}")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.transcriber_timeout_seconds, connect=10.0)
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def transcribe(
        self, wav: bytes, *, filename: str = "clip.wav", model: str | None = None
    ) -> Transcription:
        try:
            if self._protocol == "openai":
                return await self._transcribe_openai(wav, filename, model)
            return await self._transcribe_cog(wav)
        except httpx.TransportError as exc:
            raise TranscriberUnavailable(f"transcriber unreachable: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TranscriberError(f"transcriber request failed: {exc}") from exc

    async def _transcribe_openai(
        self, wav: bytes, filename: str, model: str | None
    ) -> Transcription:
        response = await self._client.post(
            f"{self._base_url}/audio/transcriptions",
            params={"model": model} if model else None,
            files={"file": (filename, wav, "audio/wav")},
            data={"model": model or self._model, "response_format": "json"},
        )
        body = self._json_or_raise(response)
        text = body.get("text")
        if not isinstance(text, str):
            raise TranscriberError(f"no text in transcriber response: {body!r}")
        language = body.get("language")
        return Transcription(
            text=text.strip(),
            raw=body,
            words=_parse_words(body),
            language=language if isinstance(language, str) else None,
        )

    async def _transcribe_cog(self, wav: bytes) -> Transcription:
        audio = "data:audio/wav;base64," + base64.b64encode(wav).decode()
        response = await self._client.post(
            f"{self._base_url}/predictions", json={"input": {"audio": audio}}
        )
        body = self._json_or_raise(response)
        output = body.get("output")
        # Cog wrappers differ: a bare string, or an object with a text field.
        if isinstance(output, str):
            text = output
        elif isinstance(output, dict) and isinstance(output.get("text"), str):
            text = output["text"]
        else:
            raise TranscriberError(f"no text in cog response: {body!r}")
        return Transcription(text=text.strip(), raw=body)

    def _json_or_raise(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code == 503:
            raise TranscriberUnavailable(
                f"transcriber answered 503: {response.text[:500]}"
            )
        if response.status_code >= 400:
            raise TranscriberError(
                f"transcriber answered {response.status_code}: {response.text[:500]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise TranscriberError(f"non-JSON transcriber response: {exc}") from exc
        if not isinstance(body, dict):
            raise TranscriberError(f"unexpected transcriber response shape: {body!r}")
        return body
