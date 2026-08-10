"""In-memory WAV generation, so the suite carries no binary fixtures."""

import math
import struct

SAMPLE_RATE = 16000


def wav_bytes(seconds: float = 0.2, freq: int = 440, rate: int = SAMPLE_RATE) -> bytes:
    """A mono 16-bit PCM WAV of a sine tone."""
    frames = int(rate * seconds)
    pcm = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(frames)
    )
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
    )
    return header + pcm
