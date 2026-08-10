"""`limit` / `offset` behaviour shared by every listing."""

import httpx
import pytest

from tests.e2e.conftest import Account, ingest, make_session
from tests.wav import wav_bytes


async def _three_segments(account: Account) -> list[str]:
    session = await make_session(account)
    created = [await ingest(session.id, wav_bytes(freq=freq)) for freq in (440, 660, 880)]
    return [str(segment.id) for segment in created]


async def test_has_more_flips_at_the_boundary(
    client: httpx.AsyncClient, account: Account
) -> None:
    await _three_segments(account)

    partial = (await client.get("/v1/segments?limit=2", headers=account.headers)).json()
    assert len(partial["items"]) == 2
    assert partial["has_more"] is True

    exact = (await client.get("/v1/segments?limit=3", headers=account.headers)).json()
    assert len(exact["items"]) == 3
    assert exact["has_more"] is False


async def test_offsets_walk_the_whole_set_without_overlap(
    client: httpx.AsyncClient, account: Account
) -> None:
    expected = set(await _three_segments(account))

    seen: list[str] = []
    for offset in (0, 1, 2):
        page = (
            await client.get(f"/v1/segments?limit=1&offset={offset}", headers=account.headers)
        ).json()
        assert page["limit"] == 1
        assert page["offset"] == offset
        seen.extend(item["id"] for item in page["items"])

    assert len(seen) == len(set(seen)) == 3
    assert set(seen) == expected


async def test_offset_past_the_end_is_an_empty_page(
    client: httpx.AsyncClient, account: Account
) -> None:
    await _three_segments(account)

    page = (await client.get("/v1/segments?offset=99", headers=account.headers)).json()
    assert page["items"] == []
    assert page["has_more"] is False


@pytest.mark.parametrize("query", ["limit=0", "limit=201", "limit=-1", "offset=-1", "limit=abc"])
async def test_invalid_paging_is_rejected(
    client: httpx.AsyncClient, account: Account, query: str
) -> None:
    response = await client.get(f"/v1/segments?{query}", headers=account.headers)
    assert response.status_code == 422


async def test_paging_applies_to_sessions_too(
    client: httpx.AsyncClient, account: Account
) -> None:
    for index in range(3):
        await make_session(account, label=f"s{index}")

    page = (await client.get("/v1/sessions?limit=2", headers=account.headers)).json()
    assert len(page["items"]) == 2
    assert page["has_more"] is True
