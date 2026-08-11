"""The audio-tag tier end to end: windows tagged and embedded, the session
map summarised, tags on the timeline, and text->audio search over pgvector."""

import httpx

from database.pipe import DatabasePipe
from database.schema.jobs import JobStatus
from tests.e2e.conftest import Account, ingest, make_account, make_session, wait_for_job
from tests.e2e.stub_audio_services import STUB_CLAP_MODEL, STUB_TAGGER_MODEL, STUB_WINDOWS
from tests.wav import wav_bytes


async def _ended_session(account: Account, count: int = 3):
    session = await make_session(account)
    for _ in range(count):
        await ingest(session.id, wav_bytes(seconds=0.1), duration_ms=100)
    async with DatabasePipe() as pipe:
        await pipe.sessions.end(session.id)
    return session


async def test_session_windows_are_tagged_and_embedded(worker: None, clean_state) -> None:
    """One audio-tag artifact per stub window with its embedding row, plus the
    map: Music persisted across both windows (a session tag), Speech did not
    (window-only)."""
    account = await make_account()
    session = await _ended_session(account)

    job = await wait_for_job(session.id, "audio-tag", JobStatus.SUCCEEDED)
    assert job.result == {"windows": 2, "tags": 1}

    async with DatabasePipe() as pipe:
        windows = await pipe.artifacts.list_for_session(session.id, kind="audio-tag")
        maps = await pipe.artifacts.list_for_session(session.id, kind="audio-tag-map")
        embeddings = await pipe.audio_embeddings.search(
            STUB_WINDOWS[0]["embedding"], user_id=account.user.id, limit=10
        )

    assert [(w.start_ms, w.end_ms) for w in windows] == [(0, 150), (150, 300)]
    first = windows[0]
    assert first.metadata["model"] == STUB_TAGGER_MODEL
    assert first.metadata["labels"] == [
        {"label": "Music", "score": 0.9},
        {"label": "Speech", "score": 0.35},
    ]
    # Postgres-only: the row is the store.
    assert all(w.bucket is None and w.object_key is None for w in windows)

    (tag_map,) = maps
    assert tag_map.metadata["windows"] == 2
    assert tag_map.metadata["tags"] == [{"label": "Music", "score": 0.9}]
    assert tag_map.metadata["models"] == {
        "tagger": STUB_TAGGER_MODEL,
        "embedding": STUB_CLAP_MODEL,
    }

    # One vector per window, keyed by the window's artifact; the query vector
    # equals window one's embedding, so it ranks first at distance ~0.
    assert {e.artifact_id for e in embeddings} == {w.id for w in windows}
    assert embeddings[0].artifact_id == windows[0].id
    assert embeddings[0].distance < 1e-6 < embeddings[1].distance


async def test_tags_appear_on_the_timeline(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    session = await _ended_session(account)
    await wait_for_job(session.id, "audio-tag", JobStatus.SUCCEEDED)

    response = await client.get(
        f"/v1/sessions/{session.id}/timeline", headers=account.headers
    )
    assert response.status_code == 200
    events = [e for e in response.json()["items"] if e["type"] == "audio-tag"]
    assert [(e["start_ms"], e["end_ms"]) for e in events] == [(0, 150), (150, 300)]
    assert events[0]["labels"][0] == {"label": "Music", "score": 0.9}
    assert events[0]["model"] == STUB_TAGGER_MODEL


async def test_search_ranks_matching_windows(
    worker: None, client: httpx.AsyncClient, account: Account, other_account: Account
) -> None:
    """The stub embeds every query as window one's vector, so search returns
    the caller's windows nearest-first — and never another user's."""
    session = await _ended_session(account)
    foreign = await _ended_session(other_account)
    await wait_for_job(session.id, "audio-tag", JobStatus.SUCCEEDED)
    await wait_for_job(foreign.id, "audio-tag", JobStatus.SUCCEEDED)

    response = await client.get(
        "/v1/search/audio", params={"q": "music"}, headers=account.headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "music" and body["model"] == STUB_CLAP_MODEL
    assert {item["session_id"] for item in body["items"]} == {str(session.id)}
    assert [(i["start_ms"], i["end_ms"]) for i in body["items"]] == [(0, 150), (150, 300)]
    assert body["items"][0]["distance"] < 1e-6
    assert body["items"][0]["labels"][0]["label"] == "Music"

    scoped = await client.get(
        "/v1/search/audio",
        params={"q": "music", "session_id": str(foreign.id)},
        headers=account.headers,
    )
    # Another user's session id yields nothing, not their windows.
    assert scoped.status_code == 200 and scoped.json()["items"] == []

    anonymous = await client.get("/v1/search/audio", params={"q": "music"})
    assert anonymous.status_code == 401
