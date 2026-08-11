"""artifact time spans

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-10

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Artifacts address a time range on the session's own timeline
    # (milliseconds from session start). NULL spans mean the whole session.
    # This is the long-term shape: labels, transcripts, and embeddings attach
    # to chunks of a session — segments are an implementation detail.
    op.execute("""
        ALTER TABLE pipeline_artifacts
            ADD COLUMN start_ms BIGINT,
            ADD COLUMN end_ms   BIGINT;
    """)
    op.execute("""
        ALTER TABLE pipeline_artifacts
            ADD CONSTRAINT pipeline_artifacts_span_check
            CHECK (
                (start_ms IS NULL) = (end_ms IS NULL)
                AND (start_ms IS NULL OR start_ms >= 0)
                AND (end_ms IS NULL OR end_ms > start_ms)
            );
    """)
    # Overlap lookups: "artifacts of session X touching [t1, t2)".
    op.execute("""
        CREATE INDEX pipeline_artifacts_span_idx
            ON pipeline_artifacts (session_id, start_ms)
            WHERE start_ms IS NOT NULL;
    """)

    # Tiered consumption: a job may be scoped to one upstream artifact (a
    # transcribe job consumes a speech-span). SET NULL, not CASCADE: deleting
    # an artifact must not silently erase job history, and the dedup key still
    # blocks re-enqueue. The index serves tier discovery's anti-join
    # ("spans with no transcribe job yet").
    op.execute("""
        ALTER TABLE processing_jobs
            ADD COLUMN artifact_id UUID
                REFERENCES pipeline_artifacts(id) ON DELETE SET NULL;
    """)
    op.execute("""
        CREATE INDEX processing_jobs_artifact_idx
            ON processing_jobs (pipeline, artifact_id)
            WHERE artifact_id IS NOT NULL;
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS processing_jobs_artifact_idx;")
    op.execute("ALTER TABLE processing_jobs DROP COLUMN IF EXISTS artifact_id;")
    op.execute("DROP INDEX IF EXISTS pipeline_artifacts_span_idx;")
    op.execute("""
        ALTER TABLE pipeline_artifacts
            DROP CONSTRAINT IF EXISTS pipeline_artifacts_span_check,
            DROP COLUMN IF EXISTS end_ms,
            DROP COLUMN IF EXISTS start_ms;
    """)
