"""Cross-session resource queries.

The ``/v1/resources/{resource}/...`` namespace queries one resource across
every session the caller owns — the first (and so far only) member is the
wall-clock location query. Session-scoped resource routes live under
``/v1/sessions/{id}/resources/...`` instead.
"""

from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import AwareDatetime

from api.deps import CurrentUser, Paging, Pipe
from api.schema.common import Page
from api.schema.locations import LocationPointRead

router = APIRouter(prefix="/resources", tags=["resources"])


@router.get("/location/points", summary="Query your location by wall-clock time")
async def list_location_points(
    user: CurrentUser,
    pipe: Pipe,
    paging: Paging,
    from_at: AwareDatetime | None = Query(
        None, alias="from", description="Only points at/after this instant."
    ),
    to_at: AwareDatetime | None = Query(
        None, alias="to", description="Only points before this instant."
    ),
    session_id: UUID | None = Query(
        None, description="Only points of this session (foreign ids match nothing)."
    ),
) -> Page[LocationPointRead]:
    """Every stored fix across your sessions, oldest first.

    The window is half-open ``[from, to)`` on the fix's ``captured_at``;
    timestamps must carry a timezone (naive ones are rejected). Like the
    search routes, ``session_id`` narrows within your own data rather than
    404ing on someone else's id.
    """
    rows = await pipe.locations.points_for_user(
        user.id,
        from_at=from_at,
        to_at=to_at,
        session_id=session_id,
        limit=paging.probe_limit,
        offset=paging.offset,
    )
    return Page.build(rows, paging, LocationPointRead.from_model)
