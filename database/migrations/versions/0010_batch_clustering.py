"""batch clustering: per-user params, identity pins, utterance clip index

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-12

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Per-user clustering knobs. Columns are NULLABLE on purpose: NULL means
    # "inherit the PROCESSING_* env default", so a later recalibration of the
    # defaults keeps applying to every field the user never explicitly pinned.
    # Params are per-user but thresholds are really per-model calibrations —
    # one embedding model exists today; revisit if a second appears.
    op.execute("""
        CREATE TABLE cluster_params (
            user_id          UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            cluster_distance DOUBLE PRECISION,
            min_talk_ms      BIGINT,
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute("""
        CREATE TRIGGER cluster_params_set_updated_at
            BEFORE UPDATE ON cluster_params
            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)

    # User curation as constraints: a pin asserts "this voice-print IS this
    # identity". The batch clusterer pre-merges each pin-group (must-link)
    # and never merges clusters pinned to different identities (cannot-link),
    # so manual merges/moves survive every rebuild instead of evaporating.
    # CASCADE on artifact_id: diarize republication regenerates embeddings
    # under new artifact ids, making old pins meaningless anyway.
    op.execute("""
        CREATE TABLE speaker_pins (
            artifact_id UUID PRIMARY KEY
                REFERENCES pipeline_artifacts(id) ON DELETE CASCADE,
            speaker_id  UUID NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute("CREATE INDEX speaker_pins_speaker_idx ON speaker_pins (speaker_id);")

    # Serves "a playable snippet per cluster member": join embeddings to
    # utterance artifacts on (session_id, metadata->>'speaker'), the same
    # shape as the transcript index in 0007. Predicate must match the query
    # exactly.
    op.execute("""
        CREATE INDEX pipeline_artifacts_utterance_speaker_idx
            ON pipeline_artifacts (session_id, (metadata->>'speaker'))
            WHERE pipeline = 'diarize' AND kind = 'utterance';
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS pipeline_artifacts_utterance_speaker_idx;")
    op.execute("DROP TABLE IF EXISTS speaker_pins;")
    op.execute("DROP TABLE IF EXISTS cluster_params;")
