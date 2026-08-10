"""processing_jobs and pipeline_artifacts

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-09

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The work queue for background pipelines (see docs/processing-pipelines.md).
    # Lifecycle: queued -> running -> succeeded | queued (retry) | dead.
    op.execute("""
        CREATE TABLE processing_jobs (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pipeline     TEXT NOT NULL,
            session_id   UUID REFERENCES recording_sessions(id) ON DELETE CASCADE,
            priority     INTEGER NOT NULL DEFAULT 0,
            status       TEXT NOT NULL DEFAULT 'queued'
                         CHECK (status IN ('queued', 'running', 'succeeded', 'dead')),
            attempts     INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 5,
            run_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            payload      JSONB NOT NULL DEFAULT '{}'::jsonb,
            result       JSONB,
            error        TEXT,
            dedup_key    TEXT,
            claimed_at   TIMESTAMPTZ,
            claimed_by   TEXT,
            finished_at  TIMESTAMPTZ,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    # Exactly the claim query's shape: queued rows in claim order.
    op.execute("""
        CREATE INDEX processing_jobs_claim_idx
            ON processing_jobs (pipeline, priority DESC, run_at, id)
            WHERE status = 'queued';
    """)
    # Spans every status, so a succeeded or dead job permanently suppresses
    # re-enqueue of its key. That is what makes repeated discovery idempotent.
    op.execute("""
        CREATE UNIQUE INDEX processing_jobs_dedup_key
            ON processing_jobs (pipeline, dedup_key)
            WHERE dedup_key IS NOT NULL;
    """)
    op.execute("""
        CREATE INDEX processing_jobs_session_idx ON processing_jobs (session_id);
    """)
    # set_updated_at() was created by revision 0001.
    op.execute("""
        CREATE TRIGGER processing_jobs_set_updated_at
            BEFORE UPDATE ON processing_jobs
            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)

    # Pipeline outputs: what a pipeline produced and where. kind and links are
    # pipeline-defined; object_key is NULL for artifacts that are pure rows.
    op.execute("""
        CREATE TABLE pipeline_artifacts (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pipeline   TEXT NOT NULL,
            session_id UUID REFERENCES recording_sessions(id) ON DELETE CASCADE,
            kind       TEXT NOT NULL,
            bucket     TEXT,
            object_key TEXT UNIQUE,
            links      JSONB NOT NULL DEFAULT '{}'::jsonb,
            metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute("""
        CREATE INDEX pipeline_artifacts_lookup_idx
            ON pipeline_artifacts (pipeline, kind, session_id);
    """)
    op.execute("""
        CREATE INDEX pipeline_artifacts_session_idx ON pipeline_artifacts (session_id);
    """)
    op.execute("""
        CREATE TRIGGER pipeline_artifacts_set_updated_at
            BEFORE UPDATE ON pipeline_artifacts
            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS pipeline_artifacts;")
    op.execute("DROP TABLE IF EXISTS processing_jobs;")
