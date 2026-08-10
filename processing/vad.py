"""Voice activity detection: frame scoring backends + pure span assembly.

A backend turns raw 16 kHz mono s16 PCM into per-frame speech probabilities
(32 ms frames). ``spans_from_scores`` is the pure post-processing that turns
those scores into merged, padded, bounded speech spans — all the tunable
behaviour lives there, unit-testable without any model.

Backends:

- ``silero`` (default) — Silero VAD via ``pysilero-vad`` (onnxruntime, CPU,
  ~2 MB). The real classifier.
- ``energy`` — an RMS amplitude threshold. Deterministic and dependency-free:
  the test suite's backend (a generated sine tone *is* activity), and a
  fallback if onnxruntime is ever unavailable.
"""

from dataclasses import dataclass
from typing import Protocol

from processing.config import ProcessingSettings

FRAME_MS = 32  # Silero's native hop at 16 kHz: 512 samples
_FRAME_BYTES = 512 * 2  # 16-bit mono

REQUIRED_SAMPLE_RATE = 16_000
REQUIRED_CHANNELS = 1


class VadBackend(Protocol):
    def scores(self, pcm: bytes) -> list[float]:
        """Speech probability per 32 ms frame. Buffers partial frames across
        calls, so arbitrary window sizes stream through cleanly."""
        ...

    def reset(self) -> None:
        """Forget stream state; call between sessions."""
        ...


class SileroBackend:
    def __init__(self) -> None:
        # Import here: the worker loads this in setup(), never at module import.
        from pysilero_vad import SileroVoiceActivityDetector

        self._detector = SileroVoiceActivityDetector()
        self._pending = bytearray()

    def scores(self, pcm: bytes) -> list[float]:
        self._pending += pcm
        out: list[float] = []
        while len(self._pending) >= _FRAME_BYTES:
            out.append(self._detector.process_chunk(bytes(self._pending[:_FRAME_BYTES])))
            del self._pending[:_FRAME_BYTES]
        return out

    def reset(self) -> None:
        self._detector.reset()
        self._pending.clear()


class EnergyBackend:
    def __init__(self, amplitude_threshold: int) -> None:
        self._threshold = amplitude_threshold
        self._pending = bytearray()

    def scores(self, pcm: bytes) -> list[float]:
        import array

        self._pending += pcm
        out: list[float] = []
        while len(self._pending) >= _FRAME_BYTES:
            samples = array.array("h", bytes(self._pending[:_FRAME_BYTES]))
            del self._pending[:_FRAME_BYTES]
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
            out.append(1.0 if rms >= self._threshold else 0.0)
        return out

    def reset(self) -> None:
        self._pending.clear()


def build_backend(settings: ProcessingSettings) -> VadBackend:
    if settings.vad_backend == "silero":
        return SileroBackend()
    if settings.vad_backend == "energy":
        return EnergyBackend(settings.vad_energy_threshold)
    raise ValueError(f"unknown VAD backend {settings.vad_backend!r}")


@dataclass(frozen=True, slots=True)
class SpeechSpan:
    start_ms: int
    end_ms: int
    confidence: float  # mean frame probability over the pre-pad span


def spans_from_scores(
    scores: list[float],
    *,
    total_ms: int,
    threshold: float,
    min_speech_ms: int,
    merge_gap_ms: int,
    pad_ms: int,
    max_span_ms: int,
    frame_ms: int = FRAME_MS,
) -> list[SpeechSpan]:
    """Assemble speech spans from per-frame probabilities.

    Order matters: merge close spans first (stuttery speech becomes one span
    *before* the minimum-length filter can drop its pieces), then drop the
    too-short, then pad and re-merge any overlap padding created, then split
    anything longer than ``max_span_ms``.

    The cap exists because the transcription model has a bounded input window
    (Canary/Whisper-family models are trained on clips of at most ~40s; longer
    input degrades or truncates), so continuous speech must be cut somewhere.
    Forced cuts land on the quietest frame in the last quarter of the window
    rather than at the hard tick — a lull instead of mid-word.
    """
    raw: list[tuple[int, int, float, int]] = []  # start, end, prob sum, frames
    current: tuple[int, float, int] | None = None  # start frame, prob sum, frames
    for index, probability in enumerate(scores):
        if probability >= threshold:
            if current is None:
                current = (index, probability, 1)
            else:
                current = (current[0], current[1] + probability, current[2] + 1)
        elif current is not None:
            raw.append((current[0] * frame_ms, index * frame_ms, current[1], current[2]))
            current = None
    if current is not None:
        raw.append((current[0] * frame_ms, len(scores) * frame_ms, current[1], current[2]))

    merged: list[tuple[int, int, float, int]] = []
    for span in raw:
        if merged and span[0] - merged[-1][1] < merge_gap_ms:
            last = merged[-1]
            merged[-1] = (last[0], span[1], last[2] + span[2], last[3] + span[3])
        else:
            merged.append(span)

    kept = [s for s in merged if s[1] - s[0] >= min_speech_ms]

    padded: list[tuple[int, int, float, int]] = []
    for start, end, prob, frames in kept:
        start, end = max(0, start - pad_ms), min(total_ms, end + pad_ms)
        if padded and start <= padded[-1][1]:
            last = padded[-1]
            padded[-1] = (last[0], end, last[2] + prob, last[3] + frames)
        else:
            padded.append((start, end, prob, frames))

    spans: list[SpeechSpan] = []
    for start, end, prob, frames in padded:
        confidence = round(prob / frames, 3) if frames else 0.0
        at = start
        while end - at > max_span_ms:
            cut = _quietest_cut(at, scores, frame_ms, max_span_ms)
            spans.append(SpeechSpan(at, cut, confidence))
            at = cut
        spans.append(SpeechSpan(at, end, confidence))
    return spans


def _quietest_cut(at: int, scores: list[float], frame_ms: int, max_span_ms: int) -> int:
    """Where to cut a span that must be split: the lowest-probability frame in
    the last quarter of the allowed window, falling back to the hard cap when
    the window lies beyond the scored audio (end-of-session padding)."""
    lo = (at + max_span_ms * 3 // 4) // frame_ms
    hi = min((at + max_span_ms) // frame_ms, len(scores))
    if lo >= hi:
        return at + max_span_ms
    quietest = min(range(lo, hi), key=scores.__getitem__)
    return quietest * frame_ms
