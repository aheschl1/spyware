"""`/v1/me` and the session endpoints."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx

from database.pipe import DatabasePipe
from tests.e2e.conftest import Account, make_session


async def test_me_reports_identity_and_empty_usage(
    client: httpx.AsyncClient, account: Account
) -> None:
    body = (await client.get("/v1/me", headers=account.headers)).json()
    assert body["user"]["id"] == str(account.user.id)
    assert body["user"]["email"] == account.user.email
    assert body["usage"] == {"segments": 0, "total_bytes": 0}


async def test_sessions_are_listed_newest_first(
    client: httpx.AsyncClient, account: Account
) -> None:
    now = datetime.now(UTC)
    from database.schema.sessions import SessionCreate

    async with DatabasePipe() as pipe:
        older = await pipe.sessions.create(
            SessionCreate(user_id=account.user.id, label="older", started_at=now - timedelta(1))
        )
        newer = await pipe.sessions.create(
            SessionCreate(user_id=account.user.id, label="newer", started_at=now)
        )

    body = (await client.get("/v1/sessions", headers=account.headers)).json()
    assert [item["id"] for item in body["items"]] == [str(newer.id), str(older.id)]
    assert body["has_more"] is False


async def test_open_only_filters_ended_sessions(
    client: httpx.AsyncClient, account: Account
) -> None:
    open_session = await make_session(account, label="open")
    ended = await make_session(account, label="ended")
    async with DatabasePipe() as pipe:
        await pipe.sessions.end(ended.id)

    body = (await client.get("/v1/sessions?open_only=true", headers=account.headers)).json()
    assert [item["id"] for item in body["items"]] == [str(open_session.id)]

    everything = (await client.get("/v1/sessions", headers=account.headers)).json()
    assert len(everything["items"]) == 2


async def test_session_detail_shape(client: httpx.AsyncClient, account: Account) -> None:
    created = await make_session(account, device="glasses-01", label="walk")

    body = (await client.get(f"/v1/sessions/{created.id}", headers=account.headers)).json()
    assert body["id"] == str(created.id)
    assert body["device"] == "glasses-01"
    assert body["label"] == "walk"
    assert body["is_open"] is True
    assert body["ended_at"] is None
    assert body["metadata"] == {}
    # The caller's own id is never echoed back.
    assert "user_id" not in body


async def test_ended_session_reports_closed(client: httpx.AsyncClient, account: Account) -> None:
    created = await make_session(account)
    async with DatabasePipe() as pipe:
        await pipe.sessions.end(created.id)

    body = (await client.get(f"/v1/sessions/{created.id}", headers=account.headers)).json()
    assert body["is_open"] is False
    assert body["ended_at"] is not None


async def test_unknown_session_is_404(client: httpx.AsyncClient, account: Account) -> None:
    response = await client.get(f"/v1/sessions/{uuid4()}", headers=account.headers)
    assert response.status_code == 404


async def test_non_uuid_path_is_422(client: httpx.AsyncClient, account: Account) -> None:
    response = await client.get("/v1/sessions/not-a-uuid", headers=account.headers)
    assert response.status_code == 422
