"""Database connection settings, loaded from the environment / .env file."""

from functools import lru_cache
from urllib.parse import quote

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Postgres credentials, read from DATABASE_* environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DATABASE_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "localhost"
    port: int = 5432
    name: str
    user: str = "postgres"
    password: str = ""

    pool_min_size: int = 1
    pool_max_size: int = 10
    connect_timeout: int = Field(default=10, description="Seconds to wait for a connection.")

    @property
    def conninfo(self) -> str:
        """libpq connection string, used by psycopg at runtime."""
        parts = {
            "host": self.host,
            "port": str(self.port),
            "dbname": self.name,
            "user": self.user,
            "password": self.password,
            "connect_timeout": str(self.connect_timeout),
        }
        return " ".join(f"{key}='{value}'" for key, value in parts.items() if value != "")

    @property
    def sqlalchemy_url(self) -> str:
        """SQLAlchemy URL, used only by Alembic's env.py."""
        return (
            f"postgresql+psycopg://{quote(self.user)}:{quote(self.password)}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def summary(self) -> str:
        """Human-readable target, safe to print (no password)."""
        return f"{self.user}@{self.host}:{self.port}/{self.name}"


@lru_cache(maxsize=1)
def get_settings() -> DatabaseSettings:
    return DatabaseSettings()
