"""Ingest and deletion, the two operations that touch both stores.

Postgres and the blob store cannot share a transaction, so the ordering lives
here:

* **ingest** writes the object, then the row; a failed row deletes the object.
* **delete** removes the row, then the object.
"""

import hashlib
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from database.exceptions import (
    DuplicateSequenceError,
    NotFoundError,
    SessionEndedError,
)
from database.pipe import DatabasePipe
from database.schema.segments import AudioSegment, SegmentCreate
from database.schema.sessions import RecordingSession
from storage.keys import segment_key, session_prefix, suffix_for
from storage.pipe import BlobPipe
from storage.s3 import S3BlobStore

DEFAULT_CONTENT_TYPE = "application/octet-stream"


class ChecksumMismatchError(Exception):
    """The caller-declared checksum does not match the received bytes."""


# Concurrent ingests into one session race for the same sequence number; the
# loser retries with a fresh read. N concurrent callers need at most N
# attempts (each round commits at least one), so this bounds contention width.
_SEQUENCE_RETRIES = 5


async def ingest_segment(
    session_id: UUID,
    data: bytes,
    *,
    content_type: str = DEFAULT_CONTENT_TYPE,
    filename: str | None = None,
    sequence: int | None = None,
    captured_at: datetime | None = None,
    offset_ms: int | None = None,
    duration_ms: int | None = None,
    codec: str | None = None,
    sample_rate_hz: int | None = None,
    channels: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> AudioSegment:
    """Store one segment's bytes and register it against its session.

    Nothing is locked across the blob upload: the sequence number is a
    snapshot read committed before the PUT, and the ``(session_id, sequence)``
    unique constraint arbitrates concurrent ingests — a loser deletes its
    object and retries with a fresh number. A caller-supplied ``sequence``
    keeps single-shot semantics: its duplicate is a real error, as on the
    streaming path.
    """
    checksum = hashlib.sha256(data).digest()
    for attempts_left in reversed(range(_SEQUENCE_RETRIES)):
        try:
            return await _ingest_once(
                session_id,
                data,
                checksum=checksum,
                content_type=content_type,
                filename=filename,
                sequence=sequence,
                captured_at=captured_at,
                offset_ms=offset_ms,
                duration_ms=duration_ms,
                codec=codec,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                metadata=metadata,
            )
        except DuplicateSequenceError:
            if sequence is not None or attempts_left == 0:
                raise
    raise AssertionError("unreachable")  # the loop returns or raises


async def _ingest_once(
    session_id: UUID,
    data: bytes,
    *,
    checksum: bytes,
    content_type: str,
    filename: str | None,
    sequence: int | None,
    captured_at: datetime | None,
    offset_ms: int | None,
    duration_ms: int | None,
    codec: str | None,
    sample_rate_hz: int | None,
    channels: int | None,
    metadata: dict[str, Any] | None,
) -> AudioSegment:
    segment_id = uuid4()
    async with DatabasePipe() as pipe:
        session = await pipe.sessions.get(session_id)
        if session is None:
            raise NotFoundError("recording session", session_id)
        if sequence is None:
            sequence = await pipe.segments.next_sequence(session_id)

    key = segment_key(
        session.user_id,
        session_id,
        segment_id,
        sequence,
        suffix_for(content_type, filename),
    )
    async with BlobPipe() as blobs:
        info = await blobs.put(key, data, content_type=content_type)
        try:
            async with DatabasePipe() as pipe:
                await pipe.sessions.touch(session_id)
                return await pipe.segments.create(
                    SegmentCreate(
                        id=segment_id,
                        session_id=session_id,
                        user_id=session.user_id,
                        sequence=sequence,
                        bucket=info.bucket,
                        object_key=info.key,
                        byte_size=info.byte_size,
                        content_type=content_type,
                        checksum_sha256=checksum,
                        captured_at=captured_at,
                        offset_ms=offset_ms,
                        duration_ms=duration_ms,
                        codec=codec,
                        sample_rate_hz=sample_rate_hz,
                        channels=channels,
                        metadata=metadata or {},
                    )
                )
        except BaseException:
            # Without a row, nothing will reference this object again. A
            # failure of the COMMIT itself still orphans it.
            await blobs.delete(key)
            raise


async def stream_segment(
    blobs: S3BlobStore,
    session: RecordingSession,
    sequence: int,
    data: bytes,
    *,
    content_type: str = DEFAULT_CONTENT_TYPE,
    captured_at: datetime | None = None,
    duration_ms: int | None = None,
    codec: str | None = None,
    sample_rate_hz: int | None = None,
    channels: int | None = None,
    metadata: dict[str, Any] | None = None,
    expected_checksum: bytes | None = None,
) -> AudioSegment:
    """Store one streamed chunk against an already-open blob store.

    The websocket ingest path: the caller holds ``blobs`` for its whole
    connection and supplies the sequence number, so unlike :func:`ingest_segment`
    no client is built and no session lock is taken per chunk. The database
    transaction spans only the row insert.

    Raises :class:`ChecksumMismatchError` before anything is written,
    :class:`~database.exceptions.SessionEndedError` when the session was ended
    since the caller loaded it, and
    :class:`~database.exceptions.DuplicateSequenceError` for a retransmit of a
    stored sequence (the original row and object are untouched).
    """
    checksum = hashlib.sha256(data).digest()
    if expected_checksum is not None and expected_checksum != checksum:
        raise ChecksumMismatchError(
            f"declared sha256 {expected_checksum.hex()} != received {checksum.hex()}"
        )

    segment_id = uuid4()
    key = segment_key(
        session.user_id, session.id, segment_id, sequence, suffix_for(content_type)
    )
    info = await blobs.put(key, data, content_type=content_type)
    try:
        async with DatabasePipe() as pipe:
            # Doubles as the liveness check: False means the session ended
            # (REST or sweeper) after this connection attached. Throttled — a
            # chunk-per-second stream must not rewrite the session row (row
            # lock, WAL, trigger) on every chunk.
            if not await pipe.sessions.touch_if_stale(session.id):
                raise SessionEndedError(session.id)
            return await pipe.segments.create(
                SegmentCreate(
                    id=segment_id,
                    session_id=session.id,
                    user_id=session.user_id,
                    sequence=sequence,
                    bucket=info.bucket,
                    object_key=info.key,
                    byte_size=info.byte_size,
                    content_type=content_type,
                    checksum_sha256=checksum,
                    captured_at=captured_at,
                    duration_ms=duration_ms,
                    codec=codec,
                    sample_rate_hz=sample_rate_hz,
                    channels=channels,
                    metadata=metadata or {},
                )
            )
    except BaseException:
        await blobs.delete(key)
        raise


async def read_segment(segment_id: UUID) -> tuple[AudioSegment, bytes]:
    """Fetch a segment's metadata and its bytes."""
    segment = await _require_segment(segment_id)
    async with BlobPipe() as blobs:
        return segment, await blobs.get(segment.object_key)


async def segment_url(segment_id: UUID, expires_in: int | None = None) -> str:
    """A presigned URL that serves the segment's audio directly."""
    segment = await _require_segment(segment_id)
    async with BlobPipe() as blobs:
        return await blobs.presign_get(segment.object_key, expires_in=expires_in)


async def delete_segment(segment_id: UUID) -> bool:
    """Delete one segment: its row, then its object."""
    async with DatabasePipe() as pipe:
        segment = await pipe.segments.get(segment_id)
        if segment is None:
            return False
        await pipe.segments.delete(segment_id)
    async with BlobPipe() as blobs:
        await blobs.delete(segment.object_key)
    return True


async def delete_session(session_id: UUID) -> int:
    """Delete a session, its segment rows (by cascade), and all of its objects.

    Returns the number of objects removed.
    """
    async with DatabasePipe() as pipe:
        session = await pipe.sessions.get(session_id)
        if session is None:
            raise NotFoundError("recording session", session_id)
        await pipe.sessions.delete(session_id)
    async with BlobPipe() as blobs:
        return await blobs.delete_prefix(session_prefix(session.user_id, session_id))


async def _require_segment(segment_id: UUID) -> AudioSegment:
    async with DatabasePipe() as pipe:
        segment = await pipe.segments.get(segment_id)
    if segment is None:
        raise NotFoundError("audio segment", segment_id)
    return segment
