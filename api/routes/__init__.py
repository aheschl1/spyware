"""One module per resource, each exposing a ``router``."""

from api.routes import health, segments, sessions, users

__all__ = ["health", "segments", "sessions", "users"]
