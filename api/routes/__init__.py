"""One module per resource, each exposing a ``router``."""

from api.routes import auth, health, search, segments, sessions, speakers, stream, users

__all__ = ["auth", "health", "search", "segments", "sessions", "speakers", "stream", "users"]
