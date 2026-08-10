"""Processing-worker settings, loaded from the environment / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingSettings(BaseSettings):
    """Worker cadence and retry policy, read from PROCESSING_* variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PROCESSING_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # How long a worker waits for a NOTIFY before running another
    # discover-and-drain pass; also the retry/discovery latency ceiling.
    poll_interval_seconds: float = 5.0
    discovery_batch: int = 100

    # Stamped onto jobs at enqueue (the table default is the backstop).
    max_attempts: int = 5
    retry_backoff_base_seconds: float = 2.0
    retry_backoff_cap_seconds: float = 300.0

    # Supervisor: SIGTERM -> SIGKILL window and child-restart backoff.
    shutdown_grace_seconds: float = 20.0
    restart_backoff_base_seconds: float = 1.0
    restart_backoff_cap_seconds: float = 30.0
    restart_reset_seconds: float = 60.0


@lru_cache(maxsize=1)
def get_settings() -> ProcessingSettings:
    return ProcessingSettings()
