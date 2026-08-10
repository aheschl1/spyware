"""Exceptions raised by the database layer."""


class DatabaseError(Exception):
    """Base class for every error raised by this package."""


class NotFoundError(DatabaseError):
    """A row was required but does not exist."""

    def __init__(self, entity: str, identifier: object) -> None:
        super().__init__(f"{entity} not found: {identifier}")
        self.entity = entity
        self.identifier = identifier


class DuplicateEmailError(DatabaseError):
    """A user with this email (case-insensitive) already exists."""

    def __init__(self, email: str) -> None:
        super().__init__(f"a user with email {email!r} already exists")
        self.email = email


class InvalidTokenError(DatabaseError):
    """The supplied token is unknown, expired, or revoked."""
