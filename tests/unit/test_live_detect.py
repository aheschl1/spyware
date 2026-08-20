"""The sherpa wakeword detector's pure parts, and the real model if present.

Model-dependent tests need an unpacked KWS zipformer (LIVE_KWS_MODEL_DIR or
~/.cache/audio-pipeline/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01)
and skip cleanly without one.
"""

import os
from pathlib import Path

import pytest

from live.config import LiveSettings
from live.detect import SilenceTracker, _greedy_tokens, _keywords_file, _model_file

KWS_MODEL_DIR = os.environ.get(
    "LIVE_KWS_MODEL_DIR",
    str(Path.home() / ".cache/audio-pipeline/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"),
)
needs_model = pytest.mark.skipif(
    not Path(KWS_MODEL_DIR, "tokens.txt").is_file(), reason="no KWS model downloaded"
)

VOCAB = {"<blk>", "<unk>", "▁", "▁HE", "Y", "▁PI", "PE", "LINE", "▁THE", "S"}


def test_greedy_tokens_prefers_longest_match() -> None:
    assert _greedy_tokens("HEY PIPELINE", VOCAB) == ["▁HE", "Y", "▁PI", "PE", "LINE"]


def test_greedy_tokens_drops_unencodable_characters() -> None:
    assert _greedy_tokens("HEY!", VOCAB) == ["▁HE", "Y"]


def test_keywords_file_uppercases_for_uppercase_vocab(tmp_path: Path) -> None:
    (tmp_path / "tokens.txt").write_text(
        "\n".join(f"{token} {i}" for i, token in enumerate(sorted(VOCAB)))
    )
    path = _keywords_file(str(tmp_path), "hey pipeline")
    assert Path(path).read_text().split() == ["▁HE", "Y", "▁PI", "PE", "LINE"]


def test_keywords_file_rejects_unencodable_wakeword(tmp_path: Path) -> None:
    (tmp_path / "tokens.txt").write_text("<blk> 0\n▁ 1\n")
    with pytest.raises(ValueError):
        _keywords_file(str(tmp_path), "hey pipeline")


def test_model_file_prefers_int8(tmp_path: Path) -> None:
    (tmp_path / "encoder-epoch-1.onnx").touch()
    (tmp_path / "encoder-epoch-1.int8.onnx").touch()
    assert _model_file(str(tmp_path), "encoder").endswith(".int8.onnx")


def test_silence_tracker_accumulates_on_silence() -> None:
    tracker = SilenceTracker(threshold=0.5)
    # 16000 zero samples = 31 full 512-sample frames of silence.
    assert tracker.feed(b"\x00" * 32000) == 31 * 32


def test_silence_tracker_buffers_partial_frames() -> None:
    tracker = SilenceTracker(threshold=0.5)
    tracker.feed(b"\x00" * 1000)  # less than one 1024-byte frame: no full frame yet
    assert tracker.feed(b"\x00" * 48) == 32


@needs_model
def test_sherpa_detector_ignores_silence_and_resets() -> None:
    from live.detect import build_detector

    settings = LiveSettings(
        _env_file=None, detector="sherpa", kws_model_dir=KWS_MODEL_DIR
    )
    detector = build_detector(settings, 16000, 1)
    assert not any(
        detector.feed(b"\x00" * 1600) for _ in range(40)
    )  # 2s of silence never triggers
    detector.reset()


@needs_model
def test_sherpa_detector_downmixes_stereo() -> None:
    from live.detect import build_detector

    settings = LiveSettings(
        _env_file=None, detector="sherpa", kws_model_dir=KWS_MODEL_DIR
    )
    detector = build_detector(settings, 16000, 2)
    assert not detector.feed(b"\x00" * 3200)
