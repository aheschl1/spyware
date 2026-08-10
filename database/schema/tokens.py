"""Pydantic models for the ``auth_tokens`` table."""

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, SecretStr


class AuthToken(BaseModel):
    """Metadata about an issued token. The secret itself is never stored."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    user_id: UUID
    name: str | None = None
    created_at: datetime
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= datetime.now(UTC)

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None and not self.is_expired


class IssuedToken(BaseModel):
    """A freshly minted token: the plaintext secret plus its stored record.

    ``token`` is the only exposure of the secret; only its hash is stored.
    """

    model_config = ConfigDict(frozen=True)

    token: SecretStr
    record: AuthToken
