"""Every ``Depends`` provider, exported as ``Annotated`` aliases.

Routes declare ``user: CurrentUser`` to require authentication, and
``session: OwnedSession`` / ``segment: OwnedSegment`` to receive a row that is
already loaded and confirmed to belong to the caller.
"""

from typing import Annotated, AsyncIterator
from uuid import UUID

from fastapi import Depends, HTTPException, Path, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.schema.common import PageParams
from database.pipe import DatabasePipe
from database.schema.segments import AudioSegment
from database.schema.sessions import RecordingSession
from database.schema.users import User

# auto_error=False: a missing header reaches get_current_user, which raises 401.
_bearer = HTTPBearer(auto_error=False, description="An API token from `tokens issue`.")

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="a valid bearer token is required",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_pipe() -> AsyncIterator[DatabasePipe]:
    """One connection and one transaction for the whole request.

    FastAPI caches dependencies per request, so the auth dependency and the
    route body share this pipe. The transaction commits when the request ends.
    """
    async with DatabasePipe() as pipe:
        yield pipe


Pipe = Annotated[DatabasePipe, Depends(get_pipe)]


async def authenticate_token(pipe: DatabasePipe, token: str | None) -> User | None:
    """Resolve a bearer token to its owner; None when missing or invalid.

    ``tokens.authenticate`` rejects unknown, revoked and expired tokens and
    inactive users, and stamps ``last_used_at``. Shared with the websocket
    upgrade check, which cannot use ``HTTPBearer``/``HTTPException``.
    """
    if not token:
        return None
    return await pipe.tokens.authenticate(token)


async def get_current_user(
    pipe: Pipe,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    """Resolve the bearer token to its owner, or raise 401."""
    user = await authenticate_token(pipe, credentials.credentials if credentials else None)
    if user is None:
        raise _UNAUTHENTICATED
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_owned_session(
    pipe: Pipe,
    user: CurrentUser,
    session_id: Annotated[UUID, Path(description="A recording session belonging to you.")],
) -> RecordingSession:
    """Load a session the caller owns. Another user's session raises 404."""
    session = await pipe.sessions.get(session_id)
    if session is None or session.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="recording session not found"
        )
    return session


OwnedSession = Annotated[RecordingSession, Depends(get_owned_session)]


async def get_owned_segment(
    pipe: Pipe,
    user: CurrentUser,
    segment_id: Annotated[UUID, Path(description="An audio segment belonging to you.")],
) -> AudioSegment:
    """Load a segment the caller owns. Another user's segment raises 404."""
    segment = await pipe.segments.get(segment_id)
    if segment is None or segment.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="audio segment not found"
        )
    return segment


OwnedSegment = Annotated[AudioSegment, Depends(get_owned_segment)]

Paging = Annotated[PageParams, Depends()]
