"""Response models for the authenticated caller."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr

from database.schema.segments import UserUsage
from database.schema.users import User


class UsageRead(BaseModel):
    """How much audio the caller has stored."""

    model_config = ConfigDict(frozen=True)

    segments: int
    total_bytes: int

    @classmethod
    def from_model(cls, usage: UserUsage) -> "UsageRead":
        return cls(segments=usage.segments, total_bytes=usage.total_bytes)


class UserRead(BaseModel):
    """The caller's own account. Never exposes another user."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    email: EmailStr
    display_name: str | None
    created_at: datetime

    @classmethod
    def from_model(cls, user: User) -> "UserRead":
        return cls(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            created_at=user.created_at,
        )


class MeRead(BaseModel):
    """`GET /v1/me`: who the token belongs to, and what they have stored."""

    model_config = ConfigDict(frozen=True)

    user: UserRead
    usage: UsageRead
