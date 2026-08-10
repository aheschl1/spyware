"""Object key layout, within this application's own bucket.

    users/{user_id}/sessions/{session_id}/{sequence:06d}-{segment_id}{suffix}

User first, then session, so a whole user or session can be cleared with one
prefix delete. The segment UUID in the leaf name keeps keys unique even when a
sequence number is reused.
"""

from pathlib import PurePosixPath
from uuid import UUID

SUFFIX_BY_CONTENT_TYPE = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/webm": ".webm",
    "audio/l16": ".pcm",
}


def suffix_for(content_type: str, filename: str | None = None) -> str:
    """Pick a file extension: from the content type, else the source filename."""
    suffix = SUFFIX_BY_CONTENT_TYPE.get(content_type.split(";")[0].strip().lower())
    if suffix is not None:
        return suffix
    if filename:
        return PurePosixPath(filename).suffix
    return ""


def user_prefix(user_id: UUID) -> str:
    return f"users/{user_id}/"


def session_prefix(user_id: UUID, session_id: UUID) -> str:
    return f"{user_prefix(user_id)}sessions/{session_id}/"


def segment_key(
    user_id: UUID,
    session_id: UUID,
    segment_id: UUID,
    sequence: int,
    suffix: str = "",
) -> str:
    return f"{session_prefix(user_id, session_id)}{sequence:06d}-{segment_id}{suffix}"
