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


class DuplicateSequenceError(DatabaseError):
    """A segment with this (session, sequence) pair is already stored."""

    def __init__(self, session_id: object, sequence: int) -> None:
        super().__init__(f"session {session_id} already holds sequence {sequence}")
        self.session_id = session_id
        self.sequence = sequence


class SessionEndedError(DatabaseError):
    """The session has ended and accepts no further segments."""

    def __init__(self, session_id: object) -> None:
        super().__init__(f"recording session {session_id} has ended")
        self.session_id = session_id
