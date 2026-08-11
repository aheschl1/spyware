"""audio window embeddings in pgvector

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-11

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # One row per audio-tag window artifact: the window's CLAP embedding,
    # queryable with pgvector distance operators — the text->audio search
    # index. Keyed by the artifact so the tier's delete-then-recreate
    # publication covers the vectors in the same transaction, and session
    # deletion cascades all the way down (same shape as speaker_embeddings,
    # 0006). The extension already exists from 0006.
    #
    # The vector column is deliberately dimensionless: the dimension belongs
    # to the embedding model behind the classifier seam (CLAP projects to
    # 512). An ANN index needs a typed column — add one (or a cast expression
    # index) when exact scans over all windows stop being enough.
    op.execute("""
        CREATE TABLE audio_embeddings (
            artifact_id UUID PRIMARY KEY
                REFERENCES pipeline_artifacts(id) ON DELETE CASCADE,
            session_id  UUID NOT NULL,
            model       TEXT NOT NULL,
            embedding   vector NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    # Serves the per-session variants of search and any future per-session
    # embedding scans, mirroring speaker_embeddings_session_idx.
    op.execute("""
        CREATE INDEX audio_embeddings_session_idx
            ON audio_embeddings (session_id);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS audio_embeddings;")
