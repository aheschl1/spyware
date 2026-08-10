"""Recording sessions: create, list, inspect, end, and the stitched audio.

Audio enters a session through the streaming websocket (see
``api.routes.stream``), not through these routes.
"""

import hashlib
from collections.abc import AsyncIterator

from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from api import stitch
from api.deps import CurrentUser, OwnedSession, Paging, Pipe
from api.ranges import RangeNotSatisfiable, etag_matches, parse_range
from api.schema.common import ErrorResponse, Page
from api.schema.segments import SegmentRead
from api.schema.sessions import SessionCreateRequest, SessionRead
from database.schema.segments import AudioSegment
from database.schema.sessions import SessionCreate
from storage.pipe import BlobPipe

router = APIRouter(prefix="/sessions", tags=["sessions"])

# Far past any real session (one chunk per second for a day is 86 400).
_MAX_STITCH_SEGMENTS = 1_000_000


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


def _session_etag(segments: list[AudioSegment]) -> str:
    """Strong validator over the exact set of stitched segments."""
    digest = hashlib.sha256()
    for segment in segments:
        digest.update(segment.id.bytes)
        digest.update(segment.checksum_sha256 or segment.byte_size.to_bytes(8, "big"))
    return f'"{digest.hexdigest()}"'


async def _stream_stitched(
    header: bytes, plan: stitch.StitchPlan, start: int, end: int
) -> AsyncIterator[bytes]:
    """Yield the stitched range: header bytes, then blob ranges in order.

    Mirrors ``segments._stream_audio``: the blob store opens inside the
    generator so FastAPI's teardown cannot outrun the body, and the first
    chunk is pulled eagerly so a missing object still surfaces as a 404.
    """

    async def chunks() -> AsyncIterator[bytes]:
        async with BlobPipe() as blobs:
            for piece, piece_start, piece_end in stitch.slices(plan, start, end):
                if piece is None:
                    yield header[piece_start : piece_end + 1]
                    continue
                async for chunk in blobs.stream(
                    piece.object_key, start=piece_start, end=piece_end
                ):
                    yield chunk

    stream = chunks()
    try:
        first = await anext(stream)
    except StopAsyncIteration:
        first = b""

    async def body() -> AsyncIterator[bytes]:
        yield first
        async for chunk in stream:
            yield chunk

    return body()


@router.get(
    "/{session_id}/audio",
    summary="Stream a session's audio as one WAV",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "The whole session, stitched."},
        206: {"content": {"audio/wav": {}}, "description": "The requested byte range."},
        304: {"description": "The client's cached copy is still current."},
        404: {"description": "The session holds no audio."},
        409: {
            "model": ErrorResponse,
            "description": "The session's segments do not form one continuous WAV.",
        },
        416: {"description": "The requested range lies outside the audio."},
    },
)
async def get_session_audio(
    session: OwnedSession, pipe: Pipe, request: Request
) -> Response:
    """Every segment's PCM behind a single WAV header, in sequence order.

    The stitched size is known from the rows alone, so `Range` (seeking) and
    conditional requests work exactly as on the per-segment route. An open
    session serves the audio ingested so far; re-request for more.
    """
    segments = await pipe.segments.list_for_session(session.id, limit=_MAX_STITCH_SEGMENTS)
    if not segments:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(detail="this session holds no audio yet").model_dump(),
        )
    try:
        stitch.check_uniform(segments)
        plan = stitch.plan(segments)
        async with BlobPipe() as blobs:
            template = b"".join(
                [
                    chunk
                    async for chunk in blobs.stream(
                        segments[0].object_key, start=0, end=stitch.WAV_HEADER_BYTES - 1
                    )
                ]
            )
        header = stitch.patch_header(template, plan.data_bytes)
    except stitch.NotStitchable as exc:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=ErrorResponse(detail=str(exc)).model_dump(),
        )

    etag = _session_etag(segments)
    headers = {
        "ETag": etag,
        "Accept-Ranges": "bytes",
        # A closed session's audio can never grow; an open one grows per chunk.
        "Cache-Control": (
            "private, no-cache" if session.is_open else "private, max-age=31536000, immutable"
        ),
        "Content-Disposition": f'inline; filename="session-{session.id}.wav"',
    }

    if etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)

    range_header = request.headers.get("range")
    if_range = request.headers.get("if-range")
    if if_range and not etag_matches(if_range, etag):
        range_header = None

    try:
        byte_range = parse_range(range_header, plan.total_size)
    except RangeNotSatisfiable:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={**headers, "Content-Range": f"bytes */{plan.total_size}"},
        )

    if byte_range is None:
        return StreamingResponse(
            await _stream_stitched(header, plan, 0, plan.total_size - 1),
            media_type="audio/wav",
            headers={**headers, "Content-Length": str(plan.total_size)},
        )

    return StreamingResponse(
        await _stream_stitched(header, plan, byte_range.start, byte_range.end),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type="audio/wav",
        headers={
            **headers,
            "Content-Length": str(byte_range.length),
            "Content-Range": byte_range.content_range(plan.total_size),
        },
    )
