"""The ASR sidecar's reload-failure handling.

A wedged process cannot reload the checkpoint in place, so a failing reload
must stop reporting healthy and eventually exit for the restart policy.
NeMo is absent here, which makes every load attempt fail for free.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def asr():
    spec = importlib.util.spec_from_file_location(
        "asr_app_under_test", REPO_ROOT / "sidecars" / "asr_parakeet" / "app.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.fixture
def deaths(asr, monkeypatch):
    recorded = []
    monkeypatch.setattr(asr, "_die", lambda reason: recorded.append(reason))
    return recorded


def _reloadable(asr, monkeypatch):
    """A process that loaded once and has since been idle-unloaded."""
    monkeypatch.setattr(asr, "_ever_loaded", True)
    monkeypatch.setattr(asr, "_state", "idle")
    monkeypatch.setattr(asr, "_models", {})


def test_failed_reload_is_reported_as_failed_not_idle(asr, monkeypatch, deaths) -> None:
    _reloadable(asr, monkeypatch)
    with pytest.raises(asr.ModelsUnavailable):
        asr._ensure_loaded("parakeet")
    assert asr._state == "failed"
    assert asr.health().status_code == 503
    assert b"failed" in asr.health().body
    assert not deaths


def test_backoff_suppresses_a_second_attempt(asr, monkeypatch, deaths) -> None:
    _reloadable(asr, monkeypatch)
    with pytest.raises(asr.ModelsUnavailable):
        asr._ensure_loaded("parakeet")
    attempts = []
    monkeypatch.setattr(asr, "_load_models", lambda: attempts.append(1))
    with pytest.raises(asr.ModelsUnavailable):
        asr._ensure_loaded("parakeet")
    assert attempts == []


def test_exits_once_the_retry_limit_is_reached(asr, monkeypatch, deaths) -> None:
    _reloadable(asr, monkeypatch)
    for _ in range(asr.RELOAD_RETRY_LIMIT):
        monkeypatch.setattr(asr, "_retry_after", 0.0)
        with pytest.raises(asr.ModelsUnavailable):
            asr._ensure_loaded("parakeet")
    assert len(deaths) == 1
    assert asr._reload_failures == asr.RELOAD_RETRY_LIMIT


def test_cold_start_failure_never_exits(asr, monkeypatch, deaths) -> None:
    for _ in range(asr.RELOAD_RETRY_LIMIT + 2):
        monkeypatch.setattr(asr, "_retry_after", 0.0)
        asr._load_models()
    assert asr._state == "failed"
    assert deaths == []


def test_a_successful_load_clears_the_failure_count(asr, monkeypatch, deaths) -> None:
    _reloadable(asr, monkeypatch)
    with pytest.raises(asr.ModelsUnavailable):
        asr._ensure_loaded("parakeet")
    assert asr._reload_failures == 1

    real_load = asr._load_models

    def succeeding_load() -> None:
        asr._models["parakeet"] = object()
        real_load()  # skips the import, runs the success bookkeeping

    monkeypatch.setattr(asr, "_retry_after", 0.0)
    monkeypatch.setattr(asr, "_load_models", succeeding_load)
    assert asr._ensure_loaded("parakeet") is asr._models
    assert asr._reload_failures == 0
    assert asr._state == "loaded"


def test_health_is_200_when_genuinely_idle(asr, monkeypatch) -> None:
    monkeypatch.setattr(asr, "_state", "idle")
    response = asr.health()
    assert response.status_code == 200
    assert b"idle-unloaded" in response.body
