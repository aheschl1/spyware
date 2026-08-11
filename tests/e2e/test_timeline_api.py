"""GET /v1/sessions/{id}/timeline: ordering, paging, windows, and scoping."""

from datetime import datetime

import httpx

from database.pipe import DatabasePipe
from database.schema.artifacts import ArtifactCreate
from tests.e2e.conftest import Account, make_session

DEVICE_TIME = "2026-01-01T00:00:00+00:00"


async def _seed(session_id) -> dict[str, str]:
    """A speech map, two spans, and a transcript on the first span."""
    async with DatabasePipe() as pipe:
        await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="speech-detect", kind="speech-map",
                session_id=session_id, metadata={"spans": 2},
            )
        )
        span_a = await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="speech-detect", kind="speech-span", session_id=session_id,
                start_ms=1_000, end_ms=4_000, metadata={"confidence": 0.9},
            )
        )
        span_b = await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="speech-detect", kind="speech-span", session_id=session_id,
                start_ms=6_000, end_ms=9_000, metadata={"confidence": 0.8},
            )
        )
        transcript = await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="transcribe", kind="transcript", session_id=session_id,
                start_ms=1_000, end_ms=4_000,
                links={"speech_span": str(span_a.id)},
                metadata={"text": "hello there", "chars": 11, "model": "stub"},
            )
        )
    return {"span_a": str(span_a.id), "span_b": str(span_b.id), "transcript": str(transcript.id)}


async def test_timeline_orders_and_pages_events(
    client: httpx.AsyncClient, account: Account
) -> None:
    created = await client.post(
        "/v1/sessions",
        json={"device": "glasses-01", "started_at": DEVICE_TIME},
        headers=account.headers,
    )
    assert created.status_code == 201
    session_id = created.json()["id"]
    ids = await _seed(session_id)

    full = await client.get(f"/v1/sessions/{session_id}/timeline", headers=account.headers)
    assert full.status_code == 200
    items = full.json()["items"]
    assert [(item["type"], item["at_ms"]) for item in items] == [
        ("session-start", 0),
        ("speech-start", 1_000),
        ("transcript", 1_000),
        ("speech-end", 4_000),
        ("speech-start", 6_000),
        ("speech-end", 9_000),
    ]
    # The opening frame carries the device wall-clock time posted at creation.
    assert datetime.fromisoformat(items[0]["started_at"]) == datetime.fromisoformat(DEVICE_TIME)
    assert items[1]["artifact_id"] == ids["span_a"] and items[1]["confidence"] == 0.9
    assert items[2]["text"] == "hello there" and items[2]["truncated"] is False
    assert items[2]["end_ms"] == 4_000 and items[2]["artifact_id"] == ids["transcript"]

    # Offset paging over events: pages of two concatenate to the full list.
    paged: list = []
    offset, has_more = 0, True
    while has_more:
        page = await client.get(
            f"/v1/sessions/{session_id}/timeline",
            params={"limit": 2, "offset": offset},
            headers=account.headers,
        )
        body = page.json()
        paged += body["items"]
        has_more = body["has_more"]
        offset += 2
    assert paged == items

    # Ending the session appends the closing frame.
    ended = await client.post(f"/v1/sessions/{session_id}/end", headers=account.headers)
    after = await client.get(f"/v1/sessions/{session_id}/timeline", headers=account.headers)
    last = after.json()["items"][-1]
    assert last["type"] == "session-end"
    assert datetime.fromisoformat(last["ended_at"]) == datetime.fromisoformat(
        ended.json()["ended_at"]
    )


async def test_timeline_window_clips_by_event_position(
    client: httpx.AsyncClient, account: Account
) -> None:
    session = await make_session(account)
    await _seed(session.id)

    windowed = await client.get(
        f"/v1/sessions/{session.id}/timeline",
        params={"from_ms": 2_000, "to_ms": 9_000},
        headers=account.headers,
    )
    # The straddling span contributes only its end; 9000 is excluded (half-open).
    assert [(item["type"], item["at_ms"]) for item in windowed.json()["items"]] == [
        ("speech-end", 4_000),
        ("speech-start", 6_000),
    ]

    disjoint = await client.get(
        f"/v1/sessions/{session.id}/timeline",
        params={"from_ms": 20_000, "to_ms": 30_000},
        headers=account.headers,
    )
    assert disjoint.json()["items"] == []


async def test_timeline_is_scoped_to_the_owner(
    client: httpx.AsyncClient, account: Account, other_account: Account
) -> None:
    session = await make_session(account)
    await _seed(session.id)
    denied = await client.get(
        f"/v1/sessions/{session.id}/timeline", headers=other_account.headers
    )
    assert denied.status_code == 404


async def test_unprocessed_and_skipped_sessions_serve_only_session_frames(
    client: httpx.AsyncClient, account: Account
) -> None:
    session = await make_session(account)

    bare = await client.get(f"/v1/sessions/{session.id}/timeline", headers=account.headers)
    assert [item["type"] for item in bare.json()["items"]] == ["session-start"]

    async with DatabasePipe() as pipe:
        await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="speech-detect", kind="speech-map", session_id=session.id,
                metadata={"spans": 0, "skipped": "no stitched audio"},
            )
        )
    skipped = await client.get(f"/v1/sessions/{session.id}/timeline", headers=account.headers)
    assert [item["type"] for item in skipped.json()["items"]] == ["session-start"]
