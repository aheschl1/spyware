"""Lexical transcript search: FTS stemming/phrases, the trigram fuzzy
fallback, and per-user scoping — against transcripts the real worker chain
produced from the stub transcriber's canned text."""

import asyncio
import time

import httpx

from database.pipe import DatabasePipe
from database.schema.jobs import JobStatus
from tests.e2e.conftest import Account, ingest, make_account, make_session, wait_for_job
from tests.wav import wav_bytes

# The stub transcriber answers this for every utterance (two per session).
STUB_TEXT = "hello from the stub transcriber"


async def _transcribed_session(account: Account):
    session = await make_session(account)
    for _ in range(3):
        await ingest(session.id, wav_bytes(seconds=0.1), duration_ms=100)
    async with DatabasePipe() as pipe:
        await pipe.sessions.end(session.id)
    await wait_for_job(session.id, "transcribe", JobStatus.SUCCEEDED)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        async with DatabasePipe() as pipe:
            transcripts = await pipe.artifacts.list_for_session(
                session.id, kind="transcript"
            )
        if len(transcripts) >= 2:
            return session
        await asyncio.sleep(0.1)
    raise AssertionError("transcripts never materialized")


async def test_stemmed_words_and_highlights(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    session = await _transcribed_session(account)

    # 'stubs' stems to 'stub' — strict FTS finds it despite the plural.
    response = await client.get(
        "/v1/search/transcripts", params={"q": "stubs"}, headers=account.headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["fuzzy"] is False
    assert len(body["items"]) == 2
    hit = body["items"][0]
    assert hit["session_id"] == str(session.id)
    assert hit["speaker"]  # transcripts carry their diarized speaker
    matched = [s["text"] for s in hit["segments"] if s["match"]]
    assert matched == ["stub"]
    joined = "".join(s["text"] for s in hit["segments"])
    assert "[[" not in joined and "]]" not in joined


async def test_phrases_respect_word_order(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    await _transcribed_session(account)

    ordered = await client.get(
        "/v1/search/transcripts",
        params={"q": '"stub transcriber"'},
        headers=account.headers,
    )
    assert ordered.json()["fuzzy"] is False and len(ordered.json()["items"]) == 2

    # Reversed phrase can't match strictly; trigram similarity still finds
    # the close text and the response says so.
    reversed_ = await client.get(
        "/v1/search/transcripts",
        params={"q": '"transcriber stub"'},
        headers=account.headers,
    )
    body = reversed_.json()
    assert body["fuzzy"] is True and len(body["items"]) > 0


async def test_misspelling_falls_back_to_fuzzy(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    await _transcribed_session(account)
    response = await client.get(
        "/v1/search/transcripts", params={"q": "transcryber"}, headers=account.headers
    )
    body = response.json()
    assert body["fuzzy"] is True
    assert len(body["items"]) == 2
    assert all(s["score"] >= 0.35 for s in body["items"])
    # Fuzzy snippets are the plain text, single unmatched segment.
    assert body["items"][0]["segments"] == [{"text": STUB_TEXT, "match": False}]


async def test_scoping_and_auth(
    worker: None, client: httpx.AsyncClient, account: Account, other_account: Account
) -> None:
    await _transcribed_session(other_account)

    # A user with no transcripts gets nothing — not the other user's, and
    # an all-miss query reports fuzzy=false (there was nothing to fall to).
    response = await client.get(
        "/v1/search/transcripts", params={"q": "stub"}, headers=account.headers
    )
    assert response.json() == {"query": "stub", "fuzzy": False, "items": []}

    anonymous = await client.get("/v1/search/transcripts", params={"q": "stub"})
    assert anonymous.status_code == 401
