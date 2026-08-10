"""Span assembly and the energy backend: pure logic, no model."""

from processing.config import ProcessingSettings
from processing.vad import FRAME_MS, EnergyBackend, build_backend, spans_from_scores
from tests.wav import wav_bytes

_DEFAULTS = dict(
    total_ms=100_000,
    threshold=0.5,
    min_speech_ms=250,
    merge_gap_ms=1000,
    pad_ms=200,
    max_span_ms=30_000,
)


def _frames(ms: int) -> int:
    return ms // FRAME_MS


def test_contiguous_speech_becomes_one_padded_span() -> None:
    scores = [0.0] * _frames(1600) + [0.9] * _frames(3200) + [0.0] * _frames(1600)
    (span,) = spans_from_scores(scores, **_DEFAULTS)
    assert span.start_ms == 1600 - 200  # padded
    assert span.end_ms == 1600 + 3200 + 200
    assert span.confidence == 0.9


def test_short_blips_are_dropped() -> None:
    scores = [0.0] * _frames(3200) + [1.0] * _frames(96) + [0.0] * _frames(3200)
    assert spans_from_scores(scores, **_DEFAULTS) == []


def test_stutter_merges_before_the_minimum_length_filter() -> None:
    # Three 96ms bursts separated by 320ms silences: each alone is under
    # min_speech_ms, together (gaps < merge_gap_ms) they clear it.
    burst, gap = [1.0] * _frames(96), [0.0] * _frames(320)
    scores = [0.0] * _frames(3200) + burst + gap + burst + gap + burst
    spans = spans_from_scores(scores, **_DEFAULTS)
    assert len(spans) == 1


def test_distant_spans_stay_separate() -> None:
    speech = [1.0] * _frames(1600)
    scores = speech + [0.0] * _frames(5000) + speech
    spans = spans_from_scores(scores, **_DEFAULTS)
    assert len(spans) == 2
    assert spans[0].end_ms <= spans[1].start_ms


def test_padding_clamps_to_the_session() -> None:
    scores = [1.0] * _frames(1600)
    (span,) = spans_from_scores(scores, **{**_DEFAULTS, "total_ms": 1600})
    assert span.start_ms == 0
    assert span.end_ms == 1600


def test_long_spans_split_at_the_cap() -> None:
    frames = _frames(70_000)
    spans = spans_from_scores([1.0] * frames, **_DEFAULTS)
    assert [span.end_ms - span.start_ms for span in spans][:2] == [30_000, 30_000]
    assert spans[-1].end_ms == frames * FRAME_MS + 200  # padded tail
    assert all(span.end_ms - span.start_ms <= 30_000 for span in spans)


def test_energy_backend_scores_tone_high_and_silence_low() -> None:
    backend = EnergyBackend(amplitude_threshold=500)
    tone = wav_bytes(seconds=0.5)[44:]  # strip the header: raw PCM
    silence = b"\x00" * len(tone)
    assert all(score == 1.0 for score in backend.scores(tone))
    backend.reset()
    assert all(score == 0.0 for score in backend.scores(silence))


def test_energy_backend_buffers_partial_frames_across_calls() -> None:
    backend = EnergyBackend(amplitude_threshold=500)
    pcm = wav_bytes(seconds=0.5)[44:]
    split = 700  # not frame-aligned
    scores = backend.scores(pcm[:split]) + backend.scores(pcm[split:])
    backend.reset()
    assert scores == backend.scores(pcm)


def test_build_backend_selects_energy() -> None:
    settings = ProcessingSettings(vad_backend="energy", vad_energy_threshold=123)
    backend = build_backend(settings)
    assert isinstance(backend, EnergyBackend)
