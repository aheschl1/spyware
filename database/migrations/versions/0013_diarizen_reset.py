"""reset diarization for the DiariZen switch

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-12

The diarize tier moved from pyannote/speaker-diarization-community-1 to
DiariZen. Block boundaries and local labels are not reproducible across the
change, so every identity keyed on them is stale: clusters, their names, and
the pins that assert a label belongs to a person. All of it is dropped and the
corpus is re-diarized from scratch; hand labelling is being redone.

The embedding model itself is unchanged (WeSpeaker ResNet34-LM, 256-d), so
voice-prints stay in the same vector space and the clustering thresholds
carry over.

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Pins cascade from speakers, but both are cleared explicitly: the whole
    # identity layer is being rebuilt, not migrated.
    op.execute("DELETE FROM speaker_pins;")
    op.execute("DELETE FROM speakers;")

    # Requeue every finished diarize job (same mechanism as 0006). The tier's
    # delete-then-recreate publication drops the old turns, utterances and
    # speaker_embeddings rows and writes the new ones in one transaction, and
    # the fresh diarize-map re-triggers the downstream tiers. Terminal states
    # only gain a fresh try; dedup keys keep discovery from double-enqueueing.
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
