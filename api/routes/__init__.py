"""One module per resource, each exposing a ``router``."""

from api.routes import health, segments, sessions, stream, users

__all__ = ["health", "segments", "sessions", "stream", "users"]
