"""Database access layer: raw SQL repositories behind a typed pipe."""

from database.config import DatabaseSettings, get_settings
from database.exceptions import (
    DatabaseError,
    DuplicateEmailError,
    InvalidTokenError,
    NotFoundError,
)
from database.pipe import DatabasePipe, close_pool, get_pool

__all__ = [
    "DatabaseError",
    "DatabasePipe",
    "DatabaseSettings",
    "DuplicateEmailError",
    "InvalidTokenError",
    "NotFoundError",
    "close_pool",
    "get_pool",
    "get_settings",
]
