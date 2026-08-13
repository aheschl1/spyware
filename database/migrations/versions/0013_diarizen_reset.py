"""reset diarization for the DiariZen switch

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-12

The diarize tier moved from pyannote community-1 to DiariZen. Block
boundaries and local labels are not reproducible across the change, so every
identity keyed on them (clusters, names, pins) is dropped and the corpus is
re-diarized. The embedder (WeSpeaker ResNet34-LM, 256-d) is unchanged, so
voice-prints stay in the same vector space.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The whole identity layer is rebuilt, not migrated.
    op.execute("DELETE FROM speaker_pins;")
    op.execute("DELETE FROM speakers;")

    # Requeue every finished diarize job (same mechanism as 0006); the tier's
    # delete-then-recreate publication replaces downstream rows transactionally.
    op.execute("""
        UPDATE processing_jobs
        SET status = 'queued', run_at = now(), claimed_at = NULL,
            claimed_by = NULL, finished_at = NULL, result = NULL, error = NULL
        WHERE pipeline = 'diarize'
          AND status IN ('running', 'succeeded', 'dead');
    """)


def downgrade() -> None:
    # One-way: the deleted identities and the requeue cannot be reconstructed.
    pass
