"""Live-layer settings, loaded from the environment / .env file."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class LiveSettings(BaseSettings):
    """Wakeword, socket, and queue tuning, read from LIVE_* variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LIVE_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = True
    # Empty selects a pid-suffixed default, so each API process (and its own
    # worker child) gets a private socket.
    socket_path: str = ""
    wakeword: str = "hey pipeline"

    # Detector backend: "stub" matches the wakeword's UTF-8 bytes in the PCM
    # (deterministic, for tests); "sherpa" is the real spotter and needs
    # kws_model_dir pointing at a sherpa-onnx KWS zipformer model.
    detector: str = "stub"
    kws_model_dir: str = ""
    kws_score: float = 1.5
    kws_threshold: float = 0.25
    kws_num_threads: int = 1
    # Silero pre-gate in front of the spotter (16 kHz streams only): silence
    # never reaches the transducer; a short pre-speech ring keeps late-flagged
    # onsets intact.
    kws_vad_pregate: bool = True
    kws_vad_threshold: float = 0.5
    kws_vad_hangover_ms: int = 480
    kws_vad_prespeech_ms: int = 192

    # Audio handed to pipelines from just before the trigger, and how much
    # audio a gated window spans before the pipelines are finished.
    preroll_ms: int = 2000
    gate_window_ms: int = 30_000
    # Close an open window after this much trailing non-speech (0 disables;
    # the gate_window_ms cap always applies). 16 kHz mono streams only.
    gate_silence_close_ms: int = 0

    # Bounded frame queues, both drop-oldest: the tap's per-connection send
    # queue and the per-pipeline feed inside the worker.
    tap_queue_frames: int = 256
    pipeline_queue_frames: int = 256

    # The transcribe pipeline's sidecar websocket and how long to wait for
    # the final transcript after a window closes.
    transcribe_url: str = "ws://127.0.0.1:8033/v1/audio/stream"
    transcribe_final_timeout_seconds: float = 10.0

    reconnect_backoff_seconds: float = 0.5
    reconnect_backoff_cap_seconds: float = 5.0

    # Supervisor: SIGTERM -> SIGKILL window and child-restart backoff.
    shutdown_grace_seconds: float = 5.0
    restart_backoff_base_seconds: float = 1.0
    restart_backoff_cap_seconds: float = 30.0
    restart_reset_seconds: float = 60.0


def socket_path(settings: "LiveSettings") -> str:
    """The effective UDS path: configured, or the pid-suffixed default."""
    if settings.socket_path:
        return settings.socket_path
    return f"/tmp/audio-pipeline/live-{os.getpid()}.sock"


@lru_cache(maxsize=1)
def get_settings() -> LiveSettings:
    return LiveSettings()
