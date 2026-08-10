"""Recording sessions: create, list, inspect, end.

Audio enters a session through the streaming websocket (see
``api.routes.stream``), not through these routes.
"""

from fastapi import APIRouter, Query, status

from api.deps import CurrentUser, OwnedSession, Paging, Pipe
from api.schema.common import Page
from api.schema.segments import SegmentRead
from api.schema.sessions import SessionCreateRequest, SessionRead
from database.schema.sessions import SessionCreate

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", status_code=status.HTTP_201_CREATED, summary="Start a recording session")
async def create_session(
    user: CurrentUser, pipe: Pipe, body: SessionCreateRequest
) -> SessionRead:
    """Create an open session, then attach to it over the streaming websocket."""
    session = await pipe.sessions.create(
        SessionCreate(
            user_id=user.id,
            device=body.device,
            label=body.label,
            started_at=body.started_at,
            metadata=body.metadata,
        )
    )
    return SessionRead.from_model(session)


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


@router.post("/{session_id}/end", summary="End a recording session")
async def end_session(session: OwnedSession, pipe: Pipe) -> SessionRead:
    """Idempotent in effect; re-ending only moves the end timestamp."""
    return SessionRead.from_model(await pipe.sessions.end(session.id))


@router.get("/{session_id}/segments", summary="List a session's audio segments")
async def list_session_segments(
    session: OwnedSession, pipe: Pipe, paging: Paging
) -> Page[SegmentRead]:
    """In capture order (by `sequence`)."""
    rows = await pipe.segments.list_for_session(
        session.id, limit=paging.probe_limit, offset=paging.offset
    )
    return Page.build(rows, paging, SegmentRead.from_model)
