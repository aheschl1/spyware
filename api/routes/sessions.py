"""Recording sessions. Read-only."""

from fastapi import APIRouter, Query

from api.deps import CurrentUser, OwnedSession, Paging, Pipe
from api.schema.common import Page
from api.schema.segments import SegmentRead
from api.schema.sessions import SessionRead

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", summary="List your recording sessions")
async def list_sessions(
    user: CurrentUser,
    pipe: Pipe,
    paging: Paging,
    open_only: bool = Query(False, description="Only sessions that have not ended."),
) -> Page[SessionRead]:
    """Newest first, scoped to the caller."""
    rows = await pipe.sessions.list_for_user(
        user.id, open_only=open_only, limit=paging.probe_limit, offset=paging.offset
    )
    return Page.build(rows, paging, SessionRead.from_model)


@router.get("/{session_id}", summary="Fetch one recording session")
async def get_session(session: OwnedSession) -> SessionRead:
    return SessionRead.from_model(session)


@router.get("/{session_id}/segments", summary="List a session's audio segments")
async def list_session_segments(
    session: OwnedSession, pipe: Pipe, paging: Paging
) -> Page[SegmentRead]:
    """In capture order (by `sequence`)."""
    rows = await pipe.segments.list_for_session(
        session.id, limit=paging.probe_limit, offset=paging.offset
    )
    return Page.build(rows, paging, SegmentRead.from_model)
