"""lexical transcript search indexes

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-11

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Trigram similarity for the fuzzy fallback (ASR-mangled words, typos).
    # Ships in postgres contrib; per-database like vector (0006).
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    # The inverted index behind exact transcript search: lexeme -> posting
    # list over the stemmed utterance text. Partial, so only transcript rows
    # are indexed and the queries must repeat this exact expression + WHERE
    # to be served by it. English config matches the ASR model (canary-qwen
    # is English-only), so stemming assumptions are safe.
    op.execute("""
        CREATE INDEX transcript_fts_idx ON pipeline_artifacts
            USING gin (to_tsvector('english', metadata->>'text'))
            WHERE pipeline = 'transcribe' AND kind = 'transcript';
    """)

    # Raw-text trigrams for word_similarity(): the fallback when strict FTS
    # finds nothing — a misheard or misspelled word still overlaps most of
    # its trigrams.
    op.execute("""
        CREATE INDEX transcript_trgm_idx ON pipeline_artifacts
            USING gin ((metadata->>'text') gin_trgm_ops)
            WHERE pipeline = 'transcribe' AND kind = 'transcript';
    """)


def downgrade() -> None:
    # The extension stays: dropping it would break anything else that
    # adopted trigram ops in the meantime (0006 precedent with vector).
    op.execute("DROP INDEX IF EXISTS transcript_fts_idx;")
    op.execute("DROP INDEX IF EXISTS transcript_trgm_idx;")
