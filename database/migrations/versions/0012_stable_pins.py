"""stable pins: re-key speaker_pins on (session, label, model)

Revision ID: 0012
Revises: 0011
Create Date: 2026-08-12

Pins were keyed by artifact_id and cascaded away whenever diarize republished
a session — destroying the exact curation they exist to protect. The stable
key is (session_id, speaker, model): the block-namespaced label a re-diarize
reproduces. Best-effort across model changes: sub-label numbering (.0/.1)
orders by clean talk and can swap on a near-tie, re-binding a pin to the
other voice; pins are visible in the UI and correctable there.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE speaker_pins_v2 (
            session_id UUID NOT NULL
                REFERENCES recording_sessions(id) ON DELETE CASCADE,
            speaker    TEXT NOT NULL,
            model      TEXT NOT NULL,
            speaker_id UUID NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (session_id, speaker, model)
        );
    """)
    op.execute("""
        INSERT INTO speaker_pins_v2 (session_id, speaker, model, speaker_id, created_at)
        SELECT e.session_id, e.speaker, e.model, p.speaker_id, p.created_at
        FROM speaker_pins p
        JOIN speaker_embeddings e ON e.artifact_id = p.artifact_id;
    """)
    op.execute("DROP TABLE speaker_pins;")
    op.execute("ALTER TABLE speaker_pins_v2 RENAME TO speaker_pins;")
    op.execute("CREATE INDEX speaker_pins_speaker_idx ON speaker_pins (speaker_id);")


def downgrade() -> None:
    op.execute("""
        CREATE TABLE speaker_pins_v1 (
            artifact_id UUID PRIMARY KEY
                REFERENCES pipeline_artifacts(id) ON DELETE CASCADE,
            speaker_id  UUID NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute("""
        INSERT INTO speaker_pins_v1 (artifact_id, speaker_id, created_at)
        SELECT e.artifact_id, p.speaker_id, p.created_at
        FROM speaker_pins p
        JOIN speaker_embeddings e ON e.session_id = p.session_id
            AND e.speaker = p.speaker AND e.model = p.model;
    """)
    op.execute("DROP TABLE speaker_pins;")
    op.execute("ALTER TABLE speaker_pins_v1 RENAME TO speaker_pins;")
    op.execute("CREATE INDEX speaker_pins_speaker_idx ON speaker_pins (speaker_id);")
