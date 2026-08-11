"""speaker embeddings in pgvector; requeue diarize

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-11

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Requires the pgvector extension binaries on the server (image
    # pgvector/pgvector:pg16 or equivalent); the extension itself is
    # per-database, so other databases in a shared cluster are untouched.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # One row per speaker-embedding artifact: the vector itself, queryable
    # with pgvector distance operators (the clustering tier's input). Keyed by
    # the artifact so delete_for_pipeline's delete-then-recreate publication
    # covers the vectors in the same transaction, and session deletion
    # cascades all the way down.
    #
    # The vector column is deliberately dimensionless: the dimension belongs
    # to the embedding model behind the diarizer seam (pyannote 3.1 emits
    # 256). An ANN index needs a typed column — the clustering tier adds one
    # (or a cast expression index) when exact scans stop being enough.
    op.execute("""
        CREATE TABLE speaker_embeddings (
            artifact_id UUID PRIMARY KEY
                REFERENCES pipeline_artifacts(id) ON DELETE CASCADE,
            session_id  UUID NOT NULL,
            speaker     TEXT NOT NULL,
            model       TEXT NOT NULL,
            embedding   vector NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute("""
        CREATE INDEX speaker_embeddings_session_idx
            ON speaker_embeddings (session_id);
    """)

    # Requeue every finished diarize job: embeddings written before this
    # revision live in blobs, and the requeued runs republish them here (the
    # tier's delete-then-recreate makes redoing safe). Terminal states only
    # gain a fresh try; dedup keys keep discovery from double-enqueueing.
    op.execute("""
        UPDATE processing_jobs
        SET status = 'queued', run_at = now(), claimed_at = NULL,
            claimed_by = NULL, finished_at = NULL, result = NULL, error = NULL
        WHERE pipeline = 'diarize'
          AND status IN ('running', 'succeeded', 'dead');
    """)


def downgrade() -> None:
    # The extension stays: dropping it would break any other table that
    # adopted vector columns in the meantime. The job requeue is one-way.
    op.execute("DROP TABLE IF EXISTS speaker_embeddings;")
