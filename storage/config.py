"""Blob store settings, loaded from the environment / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    """S3 credentials and target bucket, read from STORAGE_* variables.

    The defaults point at the shared MinIO service in the docker_deployments
    stack, where each service owns one bucket and holds credentials scoped to
    it; this application owns ``audio``. Moving to another S3-compatible
    provider is a change to these values only.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="STORAGE_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint_url: str = "http://127.0.0.1:9000"
    access_key: str
    secret_key: str
    bucket: str = "audio"
    region: str = "us-east-1"
    presign_expiry_seconds: int = 3600

    @property
    def summary(self) -> str:
        """Human-readable target, safe to print (no secret key)."""
        return f"{self.endpoint_url}/{self.bucket}"


@lru_cache(maxsize=1)
def get_settings() -> StorageSettings:
    return StorageSettings()
