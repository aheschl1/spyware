"""recording_sessions and audio_segments

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-09

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE recording_sessions (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            device     TEXT,
            label      TEXT,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            ended_at   TIMESTAMPTZ,
            metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            -- Target for the composite foreign key from audio_segments.
            UNIQUE (id, user_id)
        );
    """)
    op.execute("""
        CREATE INDEX recording_sessions_user_idx
            ON recording_sessions (user_id, started_at DESC);
    """)
    # set_updated_at() was created by revision 0001.
    op.execute("""
        CREATE TRIGGER recording_sessions_set_updated_at
            BEFORE UPDATE ON recording_sessions
            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)

    # A row is metadata plus the bucket/object_key pointer into the blob store.
    # user_id is denormalized from the parent session so per-user queries and
    # object keys need no join; the composite FK below keeps the two in step.
    op.execute("""
        CREATE TABLE audio_segments (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id      UUID NOT NULL,
            user_id         UUID NOT NULL,
            sequence        INTEGER NOT NULL,
            ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            captured_at     TIMESTAMPTZ,
            offset_ms       BIGINT,
            duration_ms     INTEGER,
            bucket          TEXT   NOT NULL,
            object_key      TEXT   NOT NULL UNIQUE,
            byte_size       BIGINT NOT NULL,
            content_type    TEXT   NOT NULL DEFAULT 'application/octet-stream',
            checksum_sha256 BYTEA,
            codec           TEXT,
            sample_rate_hz  INTEGER,
            channels        SMALLINT,
            metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
            UNIQUE (session_id, sequence),
            FOREIGN KEY (session_id, user_id)
                REFERENCES recording_sessions (id, user_id) ON DELETE CASCADE
        );
    """)
    op.execute("""
        CREATE INDEX audio_segments_user_ingested_idx
            ON audio_segments (user_id, ingested_at DESC);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS audio_segments;")
    op.execute("DROP TRIGGER IF EXISTS recording_sessions_set_updated_at ON recording_sessions;")
    op.execute("DROP TABLE IF EXISTS recording_sessions;")
