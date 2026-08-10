"""Token authentication: every way in, and every way refused."""

import asyncio
import time
from datetime import timedelta

import httpx

from database.pipe import DatabasePipe
from tests.e2e.conftest import Account

PROTECTED = "/v1/sessions"


async def test_valid_token_is_accepted(client: httpx.AsyncClient, account: Account) -> None:
    assert (await client.get(PROTECTED, headers=account.headers)).status_code == 200


async def test_missing_header_is_refused(client: httpx.AsyncClient, account: Account) -> None:
    response = await client.get(PROTECTED)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"detail": "a valid bearer token is required"}


async def test_unknown_token_is_refused(client: httpx.AsyncClient, account: Account) -> None:
    response = await client.get(PROTECTED, headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


async def test_other_auth_scheme_is_refused(client: httpx.AsyncClient, account: Account) -> None:
    response = await client.get(PROTECTED, headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert response.status_code == 401


async def test_bearer_without_credentials_is_refused(
    client: httpx.AsyncClient, account: Account
) -> None:
    response = await client.get(PROTECTED, headers={"Authorization": "Bearer"})
    assert response.status_code == 401


async def test_revoked_token_is_refused(client: httpx.AsyncClient, account: Account) -> None:
    async with DatabasePipe() as pipe:
        issued = await pipe.tokens.list_for_user(account.user.id)
        assert await pipe.tokens.revoke(issued[0].id) is True

    assert (await client.get(PROTECTED, headers=account.headers)).status_code == 401


async def test_expired_token_is_refused(client: httpx.AsyncClient, account: Account) -> None:
    async with DatabasePipe() as pipe:
        expired = await pipe.tokens.issue(account.user.id, ttl=timedelta(seconds=-1))

    headers = {"Authorization": f"Bearer {expired.token.get_secret_value()}"}
    assert (await client.get(PROTECTED, headers=headers)).status_code == 401


async def test_deactivated_user_cannot_use_a_live_token(
    client: httpx.AsyncClient, account: Account
) -> None:
    async with DatabasePipe() as pipe:
        await pipe.users.set_active(account.user.id, False)

    assert (await client.get(PROTECTED, headers=account.headers)).status_code == 401


async def test_use_stamps_last_used_at(client: httpx.AsyncClient, account: Account) -> None:
    """The stamp lands shortly after the response, not before it.

    The request's transaction commits when FastAPI tears down the `get_pipe`
    dependency, which can happen after the body has been written -- so this
    polls rather than reading once and assuming ordering it does not have.
    """
    async with DatabasePipe() as pipe:
        before = (await pipe.tokens.list_for_user(account.user.id))[0]
    assert before.last_used_at is None

    await client.get(PROTECTED, headers=account.headers)

    deadline = time.monotonic() + 5.0
    while True:
        async with DatabasePipe() as pipe:
            after = (await pipe.tokens.list_for_user(account.user.id))[0]
        if after.last_used_at is not None:
            break
        assert time.monotonic() < deadline, "last_used_at was never stamped"
        await asyncio.sleep(0.05)
