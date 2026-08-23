"""A request in the 'failed' state must reach _ensure_loaded in every sidecar.

The 2026-08-21 ASR wedge: the endpoint 503'd on `_state == "failed"` before
the reload path, so the retry/exit bookkeeping never ran and the container sat
unhealthy for two days. Only the cold "loading" state may short-circuit.
"""

import importlib.util
import sys
import types
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_failed_state_under_test", REPO_ROOT / "sidecars" / name / "app.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _stub_heavy_imports(monkeypatch):
    """The worker functions import torch/pyannote before reaching _ensure_loaded."""
    for name in ("torch", "pyannote", "pyannote.core"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["pyannote.core"].Timeline = object
    sys.modules["pyannote"].core = sys.modules["pyannote.core"]


def _upload() -> UploadFile:
    return UploadFile(BytesIO(b"RIFF"), filename="x.wav")


CASES = [
    ("asr_parakeet", lambda m: m.transcriptions(_upload(), None)),
    ("diar_pyannote", lambda m: m.diarizations(_upload())),
    ("audio_tagger", lambda m: m.analyze(_upload())),
    ("audio_tagger", lambda m: m.text_embeddings(m.TextRequest(texts=["hi"]))),
]


@pytest.mark.parametrize("name,call", CASES)
async def test_failed_state_reaches_ensure_loaded(name, call, monkeypatch) -> None:
    module = _load(name)
    try:
        reached = []

        def fake_ensure(*args, **kwargs):
            reached.append(1)
            raise module.ModelsUnavailable("still broken")

        monkeypatch.setattr(module, "_state", "failed")
        monkeypatch.setattr(module, "_ensure_loaded", fake_ensure)
        with pytest.raises(HTTPException) as info:
            await call(module)
        assert info.value.status_code == 503
        assert reached == [1]
    finally:
        del sys.modules[module.__name__]


@pytest.mark.parametrize("name,call", CASES)
async def test_loading_state_still_short_circuits(name, call, monkeypatch) -> None:
    module = _load(name)
    try:
        monkeypatch.setattr(module, "_state", "loading")
        monkeypatch.setattr(module, "_ensure_loaded", lambda *a, **k: pytest.fail("reached"))
        with pytest.raises(HTTPException) as info:
            await call(module)
        assert info.value.status_code == 503
    finally:
        del sys.modules[module.__name__]
