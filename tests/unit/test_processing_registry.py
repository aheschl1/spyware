"""The registry's wiring contract."""

import pytest

from processing import registry
from processing.base import Pipeline


def test_names_are_unique_and_nonempty() -> None:
    names = registry.names()
    assert names
    assert len(set(names)) == len(names)
    assert "users" not in names  # reserved by the blob layout (storage/keys.py)


def test_every_pipeline_declares_a_name() -> None:
    for cls in registry.PIPELINES:
        assert issubclass(cls, Pipeline)
        assert isinstance(cls.name, str) and cls.name


def test_build_wires_callbacks() -> None:
    assert registry.build("session-stats").callback is not None  # chains to stats-echo
    assert registry.build("stats-echo").callback is None


def test_build_rejects_unknown_names() -> None:
    with pytest.raises(KeyError):
        registry.build("no-such-pipeline")
