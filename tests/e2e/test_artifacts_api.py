"""GET /v1/sessions/{id}/artifacts: filters, windows, and scoping."""

import httpx

from database.pipe import DatabasePipe
from database.schema.artifacts import ArtifactCreate
from tests.e2e.conftest import Account, make_session


async def _seed(session_id) -> dict[str, str]:
    """One whole-session artifact plus two spans; returns name -> id."""
    async with DatabasePipe() as pipe:
        whole = await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="speech-detect", kind="speech-map",
                session_id=session_id, metadata={"spans": 2},
            )
        )
        early = await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="transcribe", kind="transcript", session_id=session_id,
                start_ms=0, end_ms=5_000, object_key=f"transcribe/sessions/{session_id}/a.json",
                metadata={"text": "early"},
            )
        )
        late = await pipe.artifacts.create(
            ArtifactCreate(
                pipeline="transcribe", kind="transcript", session_id=session_id,
                start_ms=60_000, end_ms=65_000, metadata={"text": "late"},
            )
        )
    return {"whole": str(whole.id), "early": str(early.id), "late": str(late.id)}


async def test_lists_in_timeline_order_with_filters(
    client: httpx.AsyncClient, account: Account
) -> None:
    session = await make_session(account)
    ids = await _seed(session.id)

    listed = await client.get(
        f"/v1/sessions/{session.id}/artifacts", headers=account.headers
    )
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert [item["id"] for item in items] == [ids["whole"], ids["early"], ids["late"]]
    assert items[0]["start_ms"] is None  # whole-session first
    assert items[1]["has_blob"] is True and items[2]["has_blob"] is False

    by_kind = await client.get(
        f"/v1/sessions/{session.id}/artifacts",
        params={"kind": "transcript"},
        headers=account.headers,
    )
    assert [item["id"] for item in by_kind.json()["items"]] == [ids["early"], ids["late"]]

    # A time window keeps only span artifacts overlapping it.
    windowed = await client.get(
        f"/v1/sessions/{session.id}/artifacts",
        params={"from_ms": 4_000, "to_ms": 61_000},
        headers=account.headers,
    )
    assert [item["id"] for item in windowed.json()["items"]] == [ids["early"], ids["late"]]
    disjoint = await client.get(
        f"/v1/sessions/{session.id}/artifacts",
        params={"from_ms": 10_000, "to_ms": 20_000},
        headers=account.headers,
    )
    assert disjoint.json()["items"] == []


async def test_artifacts_are_scoped_to_the_owner(
    client: httpx.AsyncClient, account: Account, other_account: Account
) -> None:
    session = await make_session(account)
    await _seed(session.id)
    denied = await client.get(
        f"/v1/sessions/{session.id}/artifacts", headers=other_account.headers
    )
    assert denied.status_code == 404
