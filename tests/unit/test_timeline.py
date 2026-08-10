"""Timeline math: ms <-> byte mapping and header synthesis. No stores."""

import struct
from uuid import uuid4

import pytest

from services import stitch
from services.timeline import NotRenderable, SessionTimeline, _parse_wav_fmt, wav_header


def _timeline(data_bytes: int, rate: int = 16_000, channels: int = 1) -> SessionTimeline:
    plan = stitch.StitchPlan(
        pieces=(stitch.Piece(object_key="k", start=stitch.WAV_HEADER_BYTES, data_length=data_bytes),),
        total_size=stitch.WAV_HEADER_BYTES + data_bytes,
    )
    return SessionTimeline(
        session_id=uuid4(), plan=plan, sample_rate_hz=rate, channels=channels
    )


def test_total_ms_matches_pcm_length() -> None:
    line = _timeline(16_000 * 2 * 10)  # 10s of 16k mono s16
    assert line.total_ms == 10_000


def test_byte_range_is_frame_aligned_and_clamped() -> None:
    line = _timeline(16_000 * 2 * 10)
    start, end = line.byte_range(1000, 2000)
    assert start == stitch.WAV_HEADER_BYTES + 16_000 * 2
    assert end == stitch.WAV_HEADER_BYTES + 16_000 * 2 * 2 - 1
    # Past-the-end clamps to the audio that exists.
    _, end = line.byte_range(9000, 99_000)
    assert end == stitch.WAV_HEADER_BYTES + 16_000 * 2 * 10 - 1


def test_byte_range_rejects_empty() -> None:
    line = _timeline(16_000 * 2)
    with pytest.raises(NotRenderable):
        line.byte_range(500, 500)
    with pytest.raises(NotRenderable):
        line.byte_range(5000, 6000)  # entirely past the end


def test_wav_header_round_trips_through_parse() -> None:
    header = wav_header(data_bytes=4242, sample_rate_hz=16_000, channels=1)
    assert len(header) == stitch.WAV_HEADER_BYTES
    assert _parse_wav_fmt(header) == (16_000, 1)
    assert struct.unpack_from("<I", header, 40)[0] == 4242


def test_parse_wav_fmt_rejects_non_wav_and_non_16bit() -> None:
    with pytest.raises(NotRenderable):
        _parse_wav_fmt(b"OggS" + b"\x00" * 40)
    eight_bit = bytearray(wav_header(100, 16_000, 1))
    struct.pack_into("<H", eight_bit, 34, 8)
    with pytest.raises(NotRenderable):
        _parse_wav_fmt(bytes(eight_bit))
