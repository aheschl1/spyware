"""transcription A/B: one winning-candidate vote per utterance

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-13

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # One vote per utterance (UNIQUE): revoting replaces via upsert. model +
    # strategy are DENORMALIZED from the winning candidate on purpose — the
    # tally must survive candidate republication (re-enrolling a session
    # deletes and regenerates transcript-candidate artifacts, hence SET NULL
    # on candidate_artifact_id rather than losing the vote row).
    op.execute("""
        CREATE TABLE ab_votes (
            id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id                UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            session_id             UUID NOT NULL REFERENCES recording_sessions(id) ON DELETE CASCADE,
            utterance_artifact_id  UUID NOT NULL UNIQUE
                REFERENCES pipeline_artifacts(id) ON DELETE CASCADE,
            candidate_artifact_id  UUID REFERENCES pipeline_artifacts(id) ON DELETE SET NULL,
            model                  TEXT NOT NULL,
            strategy               TEXT NOT NULL,
            created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute("CREATE INDEX ab_votes_session_idx ON ab_votes (session_id);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ab_votes;")
