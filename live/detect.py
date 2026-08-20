"""The wakeword detection seam in front of the gate.

Two backends implement the ``WakewordDetector`` protocol: the byte-marker stub
(deterministic, for tests) and a sherpa-onnx keyword spotter with a Silero VAD
pre-gate so silence never reaches the transducer. ``SilenceTracker`` is the
gate's trailing-silence clock for closing windows early.
"""

import logging
import os
import tempfile
from collections import deque
from functools import lru_cache
from glob import glob
from typing import Protocol

import numpy as np
import sherpa_onnx
from pysilero_vad import SileroVoiceActivityDetector

from live.config import LiveSettings

logger = logging.getLogger(__name__)

KWS_SAMPLE_RATE = 16_000
_VAD_FRAME_MS = 32  # Silero's native hop at 16 kHz: 512 samples
_VAD_FRAME_BYTES = 512 * 2


class WakewordDetector(Protocol):
    def reset(self) -> None: ...

    def feed(self, pcm: bytes) -> bool:
        """Consume audio; True means the wakeword just completed."""
        ...


def build_detector(
    settings: LiveSettings, sample_rate_hz: int, channels: int
) -> WakewordDetector:
    if settings.detector == "stub":
        return StubWakewordDetector(settings.wakeword)
    if settings.detector == "sherpa":
        return SherpaKeywordDetector(settings, sample_rate_hz, channels)
    raise ValueError(f"unknown wakeword detector {settings.detector!r}")


class StubWakewordDetector:
    """Triggers when the wakeword's UTF-8 bytes appear in the PCM stream.

    This exists only so tests can fire the gate deterministically (embed the
    bytes in a frame payload) — it is not a placeholder detection algorithm.
    A real detector replaces this class behind the same protocol.
    """

    def __init__(self, wakeword: str) -> None:
        self._needle = wakeword.encode()
        self._tail = b""

    def reset(self) -> None:
        self._tail = b""

    def feed(self, pcm: bytes) -> bool:
        window = self._tail + pcm
        if self._needle in window:
            self._tail = b""
            return True
        self._tail = window[-(len(self._needle) - 1):] if len(self._needle) > 1 else b""
        return False


class SherpaKeywordDetector:
    """sherpa-onnx zipformer keyword spotting behind a Silero VAD pre-gate.

    Accepts arbitrary frame sizes (internally re-framed to the VAD's 32 ms
    hop) and any channel count (downmixed). The VAD pre-gate only runs at
    16 kHz — other rates feed the spotter directly, which resamples itself.
    While the VAD reports silence, audio is held in a short pre-speech ring
    instead of the spotter, so a late speech onset still arrives intact.
    """

    def __init__(
        self, settings: LiveSettings, sample_rate_hz: int, channels: int
    ) -> None:
        self._sample_rate = sample_rate_hz
        self._channels = channels
        self._spotter = _spotter(
            settings.kws_model_dir,
            settings.wakeword,
            settings.kws_score,
            settings.kws_threshold,
            settings.kws_num_threads,
        )
        self._stream = self._spotter.create_stream()
        self._pending = bytearray()
        if settings.kws_vad_pregate and sample_rate_hz == KWS_SAMPLE_RATE:
            self._vad = SileroVoiceActivityDetector()
        else:
            self._vad = None
        self._vad_threshold = settings.kws_vad_threshold
        self._hangover_frames = max(1, settings.kws_vad_hangover_ms // _VAD_FRAME_MS)
        self._quiet_frames = self._hangover_frames + 1
        self._prespeech: deque[bytes] = deque(
            maxlen=max(1, settings.kws_vad_prespeech_ms // _VAD_FRAME_MS)
        )

    def reset(self) -> None:
        self._stream = self._spotter.create_stream()
        self._pending.clear()
        self._prespeech.clear()
        self._quiet_frames = self._hangover_frames + 1
        if self._vad is not None:
            self._vad.reset()

    def feed(self, pcm: bytes) -> bool:
        if self._channels > 1:
            pcm = _downmix(pcm, self._channels)
        if self._vad is None:
            self._accept(pcm)
            return self._decode()
        self._pending += pcm
        hit = False
        while len(self._pending) >= _VAD_FRAME_BYTES:
            chunk = bytes(self._pending[:_VAD_FRAME_BYTES])
            del self._pending[:_VAD_FRAME_BYTES]
            hit = self._feed_gated(chunk) or hit
        return hit

    def _feed_gated(self, chunk: bytes) -> bool:
        if self._vad.process_chunk(chunk) >= self._vad_threshold:
            self._quiet_frames = 0
        else:
            self._quiet_frames += 1
        if self._quiet_frames > self._hangover_frames:
            self._prespeech.append(chunk)
            return False
        while self._prespeech:
            self._accept(self._prespeech.popleft())
        self._accept(chunk)
        return self._decode()

    def _accept(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._stream.accept_waveform(self._sample_rate, samples)

    def _decode(self) -> bool:
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            if self._spotter.get_result(self._stream):
                self._spotter.reset_stream(self._stream)
                return True
        return False


class SilenceTracker:
    """Trailing-silence clock for an open gate window (16 kHz mono only)."""

    def __init__(self, threshold: float) -> None:
        self._vad = SileroVoiceActivityDetector()
        self._threshold = threshold
        self._pending = bytearray()
        self._silence_ms = 0

    def feed(self, pcm: bytes) -> int:
        """Consume audio; return the current trailing silence in ms."""
        self._pending += pcm
        while len(self._pending) >= _VAD_FRAME_BYTES:
            chunk = bytes(self._pending[:_VAD_FRAME_BYTES])
            del self._pending[:_VAD_FRAME_BYTES]
            if self._vad.process_chunk(chunk) >= self._threshold:
                self._silence_ms = 0
            else:
                self._silence_ms += _VAD_FRAME_MS
        return self._silence_ms


def _downmix(pcm: bytes, channels: int) -> bytes:
    samples = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels)
    return samples.mean(axis=1).astype(np.int16).tobytes()


@lru_cache(maxsize=4)
def _spotter(model_dir: str, wakeword: str, score: float, threshold: float, num_threads: int):
    if not model_dir:
        raise ValueError("LIVE_KWS_MODEL_DIR is required for the sherpa detector")
    return sherpa_onnx.KeywordSpotter(
        tokens=os.path.join(model_dir, "tokens.txt"),
        encoder=_model_file(model_dir, "encoder"),
        decoder=_model_file(model_dir, "decoder"),
        joiner=_model_file(model_dir, "joiner"),
        keywords_file=_keywords_file(model_dir, wakeword),
        num_threads=num_threads,
        keywords_score=score,
        keywords_threshold=threshold,
        provider="cpu",
    )


def _model_file(model_dir: str, stem: str) -> str:
    # Prefer the int8 quantization: same accuracy class, a fraction of the CPU.
    for pattern in (f"{stem}*.int8.onnx", f"{stem}*.onnx"):
        if matches := sorted(glob(os.path.join(model_dir, pattern))):
            return matches[0]
    raise FileNotFoundError(f"no {stem}*.onnx under {model_dir}")


def _keywords_file(model_dir: str, wakeword: str) -> str:
    """Write the spotter's keywords file: the wakeword as model tokens.

    Greedy longest-match against tokens.txt stands in for the model's BPE
    segmentation — for common English words the two coincide, and any valid
    token sequence is decodable. A hand-tuned file can be supplied by adding
    entries to this one at deploy time if a phrase segments badly.
    """
    vocab = set()
    with open(os.path.join(model_dir, "tokens.txt"), encoding="utf-8") as f:
        for line in f:
            if parts := line.split():
                vocab.add(parts[0])
    # Case-fold to match the model's vocabulary; ignore <blk>-style specials.
    has_lower = any(
        any(c.islower() for c in token) for token in vocab if not token.startswith("<")
    )
    text = wakeword if has_lower else wakeword.upper()
    tokens = _greedy_tokens(text, vocab)
    if not any(len(t.strip("▁")) for t in tokens):
        raise ValueError(f"wakeword {wakeword!r} has no encodable tokens")
    fd, path = tempfile.mkstemp(prefix="audio-pipeline-kws-", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(" ".join(tokens) + "\n")
    logger.info("wakeword %r spelled as %s", wakeword, tokens)
    return path


def _greedy_tokens(text: str, vocab: set[str]) -> list[str]:
    out: list[str] = []
    for word in text.split():
        piece = "▁" + word  # sentencepiece word-start marker
        while piece:
            for end in range(len(piece), 0, -1):
                if piece[:end] in vocab:
                    out.append(piece[:end])
                    piece = piece[end:]
                    break
            else:
                piece = piece[1:]  # unencodable character: drop it
    return out
