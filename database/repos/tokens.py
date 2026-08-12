"""Raw-SQL repository for the ``auth_tokens`` table.

Tokens are opaque random secrets. Only a SHA-256 digest is stored, so the
plaintext exists exactly once: in the :class:`IssuedToken` returned by
:meth:`TokensRepo.issue`.
"""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from psycopg import errors

from database.exceptions import NotFoundError
from database.repos.base import BaseRepo
from database.repos.users import COLUMNS as USER_COLUMNS
from database.schema.tokens import AuthToken, IssuedToken
from database.schema.users import User
from database.security import generate_token, hash_token

COLUMNS = "id, user_id, name, created_at, expires_at, last_used_at, revoked_at"
LIVE = "revoked_at IS NULL AND (expires_at IS NULL OR expires_at > now())"


class TokensRepo(BaseRepo):
    async def issue(
        self,
        user_id: UUID,
        name: str | None = None,
        ttl: timedelta | None = None,
    ) -> IssuedToken:
        """Mint a token for a user. The plaintext is returned only here."""
        token = generate_token()
        expires_at = datetime.now(UTC) + ttl if ttl is not None else None
        sql = f"""
            INSERT INTO auth_tokens (user_id, token_hash, name, expires_at)
            VALUES (%s, %s, %s, %s)
            RETURNING {COLUMNS}
        """
        try:
            async with self._conn.transaction():
                record = await self._fetch_one(
                    AuthToken, sql, (user_id, hash_token(token), name, expires_at)
                )
        except errors.ForeignKeyViolation as exc:
            raise NotFoundError("user", user_id) from exc
        assert record is not None  # INSERT ... RETURNING always yields a row
        return IssuedToken(token=token, record=record)

    async def resolve(self, token: str) -> AuthToken | None:
        """Look up a token by its secret. Revoked/expired tokens yield ``None``."""
        return await self._fetch_one(
            AuthToken,
            f"SELECT {COLUMNS} FROM auth_tokens WHERE token_hash = %s AND {LIVE}",
            (hash_token(token),),
        )

    async def resolve_user(self, token: str) -> User | None:
        """Resolve a live token to its active owner. Read-only: no row lock held.

        This is the authentication decision on its own; :meth:`touch_last_used`
        records the use separately, so the ``last_used_at`` write's row lock is
        never held for the length of the caller's (possibly long-lived) request.
        """
        sql = f"""
            SELECT {", ".join("u." + column for column in USER_COLUMNS.split(", "))}
            FROM auth_tokens t JOIN users u ON u.id = t.user_id
            WHERE t.token_hash = %s
              AND t.revoked_at IS NULL
              AND (t.expires_at IS NULL OR t.expires_at > now())
              AND u.is_active
        """
        return await self._fetch_one(User, sql, (hash_token(token),))

    async def touch_last_used(self, token: str, min_interval_seconds: float = 60.0) -> bool:
        """Stamp a live token's ``last_used_at``, at most once per interval.

        Deliberately decoupled from :meth:`resolve_user`: the write (and its
        brief row lock) is a statement of its own, and the interval throttle
        stops a busy token from writing -- and taking that lock -- on every
        single request. Returns whether a row was updated.
        """
        affected = await self._execute(
            f"""
                UPDATE auth_tokens SET last_used_at = now()
                WHERE token_hash = %s AND {LIVE}
                  AND (last_used_at IS NULL
                       OR last_used_at < now() - make_interval(secs => %s))
            """,
            (hash_token(token), min_interval_seconds),
        )
        return affected > 0

    async def authenticate(self, token: str, min_interval_seconds: float = 60.0) -> User | None:
        """Resolve a token to its (active) owner, stamping ``last_used_at``.

        The request path's single statement (see ``api.deps.authenticate_token``):
        the throttled stamp rides in a data-modifying CTE — executed even though
        nothing references it — so the common case is one round trip that writes
        nothing and takes no row lock. Callers should still run this on a
        short-lived connection so the rare stamp's lock never rides along in a
        long request transaction.
        """
        digest = hash_token(token)
        sql = f"""
            WITH stamp AS (
                UPDATE auth_tokens SET last_used_at = now()
                WHERE token_hash = %(digest)s AND {LIVE}
                  AND (last_used_at IS NULL
                       OR last_used_at < now() - make_interval(secs => %(interval)s))
            )
            SELECT {", ".join("u." + column for column in USER_COLUMNS.split(", "))}
            FROM auth_tokens t JOIN users u ON u.id = t.user_id
            WHERE t.token_hash = %(digest)s
              AND t.revoked_at IS NULL
              AND (t.expires_at IS NULL OR t.expires_at > now())
              AND u.is_active
        """
        return await self._fetch_one(
            User, sql, {"digest": digest, "interval": min_interval_seconds}
        )

    async def get(self, token_id: UUID) -> AuthToken | None:
        return await self._fetch_one(
            AuthToken, f"SELECT {COLUMNS} FROM auth_tokens WHERE id = %s", (token_id,)
        )

    async def list_for_user(self, user_id: UUID, include_inactive: bool = True) -> list[AuthToken]:
        sql = f"SELECT {COLUMNS} FROM auth_tokens WHERE user_id = %s"
        if not include_inactive:
            sql += f" AND {LIVE}"
        sql += " ORDER BY created_at DESC"
        return await self._fetch_all(AuthToken, sql, (user_id,))

    async def revoke(self, token_id: UUID) -> bool:
        """Revoke a single token. ``False`` if it is unknown or already revoked."""
        affected = await self._execute(
            "UPDATE auth_tokens SET revoked_at = now() WHERE id = %s AND revoked_at IS NULL",
            (token_id,),
        )
        return affected > 0

    async def revoke_all_for_user(self, user_id: UUID) -> int:
        return await self._execute(
            "UPDATE auth_tokens SET revoked_at = now() "
            "WHERE user_id = %s AND revoked_at IS NULL",
            (user_id,),
        )

    async def purge_expired(self) -> int:
        """Delete tokens that are past their expiry. Returns the row count."""
        return await self._execute(
            "DELETE FROM auth_tokens WHERE expires_at IS NOT NULL AND expires_at <= now()"
        )

    async def purge_expired_named(self, user_id: UUID, name: str) -> int:
        """Delete one user's expired tokens carrying ``name``.

        High-churn short-TTL tokens (playback) would otherwise accumulate a
        dead row per mint; the minting route calls this opportunistically.
        """
        return await self._execute(
            "DELETE FROM auth_tokens WHERE user_id = %s AND name = %s "
            "AND expires_at IS NOT NULL AND expires_at <= now()",
            (user_id, name),
        )
