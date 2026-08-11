"""The playback-token route and query-token audio access.

Media elements cannot send an Authorization header, so the session-audio
route accepts ``?token=``; the playback route mints minutes-lived tokens for
that URL. These tests pin both halves plus the ownership and expiry edges.
"""

from datetime import UTC, datetime

import httpx

from database.pipe import DatabasePipe
from tests.e2e.conftest import Account, ingest, make_session
from tests.wav import wav_bytes


async def _session_with_audio(account: Account):
    session = await make_session(account)
    await ingest(session.id, wav_bytes(seconds=0.5), duration_ms=500)
    return session


async def _mint(client: httpx.AsyncClient, account: Account, session_id) -> dict:
    response = await client.post(
        f"/v1/sessions/{session_id}/playback", headers=account.headers
    )
    assert response.status_code == 200
    return response.json()


async def test_playback_token_streams_audio_via_query(
    client: httpx.AsyncClient, account: Account
) -> None:
    session = await _session_with_audio(account)
    body = await _mint(client, account, session.id)
    assert datetime.fromisoformat(body["expires_at"]) > datetime.now(UTC)

    url = f"/v1/sessions/{session.id}/audio"
    full = await client.get(url, params={"token": body["token"]})
    assert full.status_code == 200
    assert full.headers["content-type"] == "audio/wav"

    ranged = await client.get(
        url, params={"token": body["token"]}, headers={"Range": "bytes=0-99"}
    )
    assert ranged.status_code == 206
    assert len(ranged.content) == 100


async def test_audio_still_requires_some_credential(
    client: httpx.AsyncClient, account: Account
) -> None:
    session = await _session_with_audio(account)
    bare = await client.get(f"/v1/sessions/{session.id}/audio")
    assert bare.status_code == 401
    bogus = await client.get(
        f"/v1/sessions/{session.id}/audio", params={"token": "not-a-token"}
    )
    assert bogus.status_code == 401


async def test_header_wins_over_query_token(
    client: httpx.AsyncClient, account: Account
) -> None:
    session = await _session_with_audio(account)
    response = await client.get(
        f"/v1/sessions/{session.id}/audio",
        params={"token": "garbage"},
        headers=account.headers,
    )
    assert response.status_code == 200


async def test_other_users_session_is_404(
    client: httpx.AsyncClient, account: Account, other_account: Account
) -> None:
    session = await _session_with_audio(account)
    body = await _mint(client, other_account, (await _session_with_audio(other_account)).id)
    response = await client.get(
        f"/v1/sessions/{session.id}/audio", params={"token": body["token"]}
    )
    assert response.status_code == 404


async def test_expired_playback_token_is_rejected_and_purged(
    client: httpx.AsyncClient, account: Account
) -> None:
    session = await _session_with_audio(account)
    body = await _mint(client, account, session.id)

    async with DatabasePipe() as pipe:
        await pipe.connection.execute(
            "UPDATE auth_tokens SET expires_at = now() - interval '1 minute' "
            "WHERE name = %s",
            ("playback",),
        )

    rejected = await client.get(
        f"/v1/sessions/{session.id}/audio", params={"token": body["token"]}
    )
    assert rejected.status_code == 401

    # The next mint purges the caller's expired playback rows.
    await _mint(client, account, session.id)
    async with DatabasePipe() as pipe:
        async with pipe.connection.cursor() as cur:
            await cur.execute(
                "SELECT count(*) AS stale FROM auth_tokens WHERE name = %s "
                "AND expires_at IS NOT NULL AND expires_at <= now()",
                ("playback",),
            )
            row = await cur.fetchone()
    assert row["stale"] == 0

    # The login token in the Authorization header is untouched by any of this.
    still_authed = await client.get("/v1/me", headers=account.headers)
    assert still_authed.status_code == 200
