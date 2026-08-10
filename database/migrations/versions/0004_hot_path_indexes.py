"""hot-path indexes for discovery and the sweeper

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-10

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Pipeline discovery scans ended sessions in ended_at order every worker
    # pass (database/repos/pipelines/); without this it is a growing seq scan
    # plus a sort, every poll interval, forever.
    op.execute("""
        CREATE INDEX recording_sessions_ended_idx
            ON recording_sessions (ended_at)
            WHERE ended_at IS NOT NULL;
    """)
    # The stale-session sweeper (sessions.end_stale) runs every minute in every
    # API worker; exactly its predicate's shape.
    op.execute("""
        CREATE INDEX recording_sessions_open_updated_idx
            ON recording_sessions (updated_at)
            WHERE ended_at IS NULL;
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS recording_sessions_open_updated_idx;")
    op.execute("DROP INDEX IF EXISTS recording_sessions_ended_idx;")
