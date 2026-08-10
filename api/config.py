"""API settings, loaded from the environment / .env file.

Everything here tunes the streaming websocket (docs/streaming-protocol.md);
the HTTP routes need no configuration of their own.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Streaming limits and timeouts, read from API_* variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="API_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Must stay at or below uvicorn's websocket message limit (16 MiB default);
    # frames past that limit tear the connection down instead of erroring.
    stream_max_chunk_bytes: int = 8 * 1024 * 1024
    stream_ack_window_chunks: int = 10
    stream_ack_window_seconds: float = 2.0
    stream_hello_timeout_seconds: float = 10.0
    stream_idle_timeout_seconds: float = 300.0

    # An open session with no activity for this long is ended by the sweeper.
    # Keep it at or above the idle timeout, or a quiet-but-connected client's
    # session can be swept out from under it.
    session_stale_seconds: float = 300.0
    session_sweep_interval_seconds: float = 60.0


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()
