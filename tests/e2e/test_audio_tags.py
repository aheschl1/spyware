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


async def test_tag_filter_search(
    worker: None, client: httpx.AsyncClient, account: Account, other_account: Account
) -> None:
    """The deterministic search: substring class match, calibrated score
    floor, per-user scoping — no model in the loop (works even when the
    embedding service would be down)."""
    session = await _ended_session(account)
    await _ended_session(other_account)
    await wait_for_job(session.id, "audio-tag", JobStatus.SUCCEEDED)

    # Case-insensitive substring: 'mus' finds 'Music' in both windows.
    hits = await client.get(
        "/v1/search/tags", params={"label": "mus"}, headers=account.headers
    )
    assert hits.status_code == 200
    body = hits.json()
    assert body["label"] == "mus" and body["min_score"] == 0.3
    assert [(h["label"], h["score"]) for h in body["items"]] == [
        ("Music", 0.9),
        ("Music", 0.8),
    ]
    assert {h["session_id"] for h in body["items"]} == {str(session.id)}

    # The floor is a real threshold: 0.85 keeps only the first window.
    floored = await client.get(
        "/v1/search/tags",
        params={"label": "music", "min_score": 0.85},
        headers=account.headers,
    )
    assert [h["start_ms"] for h in floored.json()["items"]] == [0]

    # Speech scored 0.35 in window one only.
    speech = await client.get(
        "/v1/search/tags",
        params={"label": "Speech", "min_score": 0.3},
        headers=account.headers,
    )
    assert [(h["start_ms"], h["score"]) for h in speech.json()["items"]] == [(0, 0.35)]

    anonymous = await client.get("/v1/search/tags", params={"label": "music"})
    assert anonymous.status_code == 401


async def test_tag_labels_vocabulary(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    session = await _ended_session(account)
    await wait_for_job(session.id, "audio-tag", JobStatus.SUCCEEDED)

    response = await client.get("/v1/search/tags/labels", headers=account.headers)
    assert response.status_code == 200
    labels = response.json()["labels"]
    assert [(l["label"], l["windows"], l["best"]) for l in labels] == [
        ("Music", 2, 0.9),
        ("Speech", 1, 0.35),
    ]
