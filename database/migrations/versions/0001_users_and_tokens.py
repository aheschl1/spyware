"""users and auth_tokens

Revision ID: 0001
Revises:
Create Date: 2026-08-09

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at() RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # gen_random_uuid() is built into Postgres 13+; no pgcrypto needed.
    op.execute("""
        CREATE TABLE users (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email         TEXT        NOT NULL,
            password_hash TEXT        NOT NULL,
            display_name  TEXT,
            is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    # Case-insensitive uniqueness without the citext extension.
    op.execute("CREATE UNIQUE INDEX users_email_lower_key ON users (lower(email));")
    op.execute("""
        CREATE TRIGGER users_set_updated_at
            BEFORE UPDATE ON users
            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)

    # token_hash holds sha256(token); the plaintext is never stored.
    op.execute("""
        CREATE TABLE auth_tokens (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id      UUID  NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token_hash   BYTEA NOT NULL UNIQUE,
            name         TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at   TIMESTAMPTZ,
            last_used_at TIMESTAMPTZ,
            revoked_at   TIMESTAMPTZ
        );
    """)
    op.execute("CREATE INDEX auth_tokens_user_id_idx ON auth_tokens (user_id);")
    op.execute("""
        CREATE INDEX auth_tokens_active_idx ON auth_tokens (user_id)
            WHERE revoked_at IS NULL;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS auth_tokens;")
    op.execute("DROP TRIGGER IF EXISTS users_set_updated_at ON users;")
    op.execute("DROP TABLE IF EXISTS users;")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at();")
