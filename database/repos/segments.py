"""Raw-SQL repository for the ``audio_segments`` table.

Rows point at blobs; the object write is coordinated in :mod:`services.audio`.
"""

from uuid import UUID

from psycopg import errors
from psycopg.types.json import Jsonb

from database.exceptions import DuplicateSequenceError, NotFoundError
from database.repos.base import BaseRepo
from database.schema.segments import (
    AudioSegment,
    SegmentCreate,
    SegmentSetFingerprint,
    UserUsage,
)

COLUMNS = (
    "id, session_id, user_id, sequence, ingested_at, captured_at, offset_ms, duration_ms, "
    "bucket, object_key, byte_size, content_type, checksum_sha256, codec, sample_rate_hz, "
    "channels, metadata"
)


class SegmentsRepo(BaseRepo):
    async def next_sequence(self, session_id: UUID) -> int:
        """Reserve the next sequence number within a session.

        Locks the parent row for the rest of the transaction; without it two
        concurrent ingests read the same MAX(sequence) and one violates the
        (session_id, sequence) unique constraint.
        """
        locked = await self._fetch_value(
            "SELECT id FROM recording_sessions WHERE id = %s FOR UPDATE", (session_id,)
        )
        if locked is None:
            raise NotFoundError("recording session", session_id)
        return await self._fetch_value(
            "SELECT COALESCE(MAX(sequence) + 1, 0) FROM audio_segments WHERE session_id = %s",
            (session_id,),
        )

    async def create(self, data: SegmentCreate) -> AudioSegment:
        """Register a segment whose blob is already written.

        With ``data.sequence`` unset, the next number in the session is reserved
        via :meth:`next_sequence`.
        """
        sequence = data.sequence
        if sequence is None:
            sequence = await self.next_sequence(data.session_id)

        sql = f"""
            INSERT INTO audio_segments (
                id, session_id, user_id, sequence, captured_at, offset_ms, duration_ms,
                bucket, object_key, byte_size, content_type, checksum_sha256,
                codec, sample_rate_hz, channels, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING {COLUMNS}
        """
        params = (
            data.id,
            data.session_id,
            data.user_id,
            sequence,
            data.captured_at,
            data.offset_ms,
            data.duration_ms,
            data.bucket,
            data.object_key,
            data.byte_size,
            data.content_type,
            data.checksum_sha256,
            data.codec,
            data.sample_rate_hz,
            data.channels,
            Jsonb(data.metadata),
        )
        try:
            async with self._conn.transaction():
                segment = await self._fetch_one(AudioSegment, sql, params)
        except errors.ForeignKeyViolation as exc:
            # The composite FK covers both "no such session" and "session owned
            # by another user".
            raise NotFoundError(
                "recording session for user", (data.session_id, data.user_id)
            ) from exc
        except errors.UniqueViolation as exc:
            # object_key embeds a fresh UUID, so of the two unique constraints
            # only (session_id, sequence) can realistically fire: a retransmit.
            if exc.diag.constraint_name == "audio_segments_session_id_sequence_key":
                raise DuplicateSequenceError(data.session_id, sequence) from exc
            raise
        assert segment is not None  # INSERT ... RETURNING always yields a row
        return segment

    async def get(self, segment_id: UUID) -> AudioSegment | None:
        return await self._fetch_one(
            AudioSegment, f"SELECT {COLUMNS} FROM audio_segments WHERE id = %s", (segment_id,)
        )

    async def list_for_session(
        self, session_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[AudioSegment]:
        return await self._fetch_all(
            AudioSegment,
            f"""
                SELECT {COLUMNS} FROM audio_segments WHERE session_id = %s
                ORDER BY sequence LIMIT %s OFFSET %s
            """,
            (session_id, limit, offset),
        )

    async def list_for_user(
        self, user_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[AudioSegment]:
        return await self._fetch_all(
            AudioSegment,
            f"""
                SELECT {COLUMNS} FROM audio_segments WHERE user_id = %s
                ORDER BY ingested_at DESC LIMIT %s OFFSET %s
            """,
            (user_id, limit, offset),
        )

    async def delete(self, segment_id: UUID) -> bool:
        """Delete the row only. The blob is removed by ``services.audio``."""
        return await self._execute("DELETE FROM audio_segments WHERE id = %s", (segment_id,)) > 0

    async def stitch_fingerprint(self, session_id: UUID) -> SegmentSetFingerprint:
        """A cheap change-token for a session's segment set.

        One aggregate row rather than every segment: enough to decide whether a
        cached stitch plan is still valid without loading the rows behind it.
        """
        fingerprint = await self._fetch_one(
            SegmentSetFingerprint,
            """
                SELECT COUNT(*) AS count,
                       COALESCE(MAX(sequence), -1) AS max_sequence,
                       COALESCE(SUM(byte_size), 0) AS total_bytes
                FROM audio_segments WHERE session_id = %s
            """,
            (session_id,),
        )
        assert fingerprint is not None  # aggregate without GROUP BY always returns a row
        return fingerprint

    async def usage_for_user(self, user_id: UUID) -> UserUsage:
        usage = await self._fetch_one(
            UserUsage,
            """
                SELECT COUNT(*) AS segments, COALESCE(SUM(byte_size), 0) AS total_bytes
                FROM audio_segments WHERE user_id = %s
            """,
            (user_id,),
        )
        assert usage is not None  # aggregate without GROUP BY always returns a row
        return usage
