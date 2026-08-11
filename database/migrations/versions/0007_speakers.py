"""speakers: global clusters over speaker embeddings

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-11

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # One row per globally-clustered voice, per user. name is the user-given
    # label (NULL = unlabeled; names are deliberately NOT unique — fetch-by-
    # name unions same-named clusters until the merge pass consolidates them).
    # model matters: pgvector's avg() and <=> ERROR on mixed dimensions, and
    # embeddings from different models aren't comparable anyway, so every
    # cluster operation filters by (user_id, model).
    op.execute("""
        CREATE TABLE speakers (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name       TEXT,
            model      TEXT NOT NULL,
            centroid   vector NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute("""
        CREATE TRIGGER speakers_set_updated_at
            BEFORE UPDATE ON speakers
            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)
    # The nearest-centroid scan: a user's clusters for one model (dozens of
    # rows; exact distance scan, no ANN needed at this shape).
    op.execute("CREATE INDEX speakers_user_idx ON speakers (user_id, model);")
    # Fetch-by-name (non-unique: several clusters may share a label).
    op.execute("""
        CREATE INDEX speakers_name_idx ON speakers (user_id, name)
            WHERE name IS NOT NULL;
    """)

    # The assignment: which cluster each embedding belongs to. SET NULL, not
    # CASCADE — deleting a cluster (merge) must never delete voice-prints.
    # The UNIQUE documents and enforces what the transcript join relies on:
    # one embedding per (session, block-local label).
    op.execute("""
        ALTER TABLE speaker_embeddings
            ADD COLUMN speaker_id UUID REFERENCES speakers(id) ON DELETE SET NULL,
            ADD CONSTRAINT speaker_embeddings_session_speaker_key
                UNIQUE (session_id, speaker);
    """)
    # Discovery: "this session's not-yet-clustered embeddings".
    op.execute("""
        CREATE INDEX speaker_embeddings_unassigned_idx
            ON speaker_embeddings (session_id) WHERE speaker_id IS NULL;
    """)
    # Membership scans: centroid recompute and per-speaker transcript joins.
    op.execute("""
        CREATE INDEX speaker_embeddings_speaker_idx
            ON speaker_embeddings (speaker_id) WHERE speaker_id IS NOT NULL;
    """)

    # Serves "transcripts of one speaker": join embeddings to transcript
    # artifacts on (session_id, metadata->>'speaker'). Predicate must match
    # the query exactly.
    op.execute("""
        CREATE INDEX pipeline_artifacts_transcript_speaker_idx
            ON pipeline_artifacts (session_id, (metadata->>'speaker'))
            WHERE pipeline = 'transcribe' AND kind = 'transcript';
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS pipeline_artifacts_transcript_speaker_idx;")
    op.execute("""
        ALTER TABLE speaker_embeddings
            DROP CONSTRAINT IF EXISTS speaker_embeddings_session_speaker_key,
            DROP COLUMN IF EXISTS speaker_id;
    """)
    op.execute("DROP TABLE IF EXISTS speakers;")
