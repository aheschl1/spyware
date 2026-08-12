"""Global speaker clustering end to end: the speaker-cluster tier unites
voices across sessions, the API surfaces and labels the clusters, and
fetch-by-name unions everything one person said.

The stub diarizer's embeddings are orthogonal per local speaker and identical
across sessions/blocks, so SPEAKER_00 of every session lands in one cluster
and SPEAKER_01 in another (cosine distance 0 within a voice, 1.0 across —
clean for the 0.65 batch-clustering threshold, and far enough apart that
only a *manual* merge or a loosened threshold can ever unite them).
"""

import asyncio
import subprocess
import sys
import time
from uuid import uuid4

import httpx

from database.pipe import DatabasePipe
from database.schema.jobs import JobStatus
from tests.e2e.conftest import (
    REPO_ROOT,
    Account,
    ingest,
    make_account,
    make_session,
    wait_for_job,
)
from tests.wav import wav_bytes


async def _diarized_session(account: Account):
    """A session run all the way through cluster AND transcribe (2 stub turns)."""
    session = await make_session(account)
    for _ in range(3):
        await ingest(session.id, wav_bytes(seconds=0.1), duration_ms=100)
    async with DatabasePipe() as pipe:
        await pipe.sessions.end(session.id)
    await wait_for_job(session.id, "speaker-cluster", JobStatus.SUCCEEDED)
    # Cluster and transcribe race (both consume diarize output); by-name
    # transcript assertions need both utterances transcribed.
    deadline = time.monotonic() + 15.0
    transcripts: list = []
    while time.monotonic() < deadline and len(transcripts) < 2:
        async with DatabasePipe() as pipe:
            transcripts = await pipe.artifacts.list_for_session(
                session.id, kind="transcript"
            )
        await asyncio.sleep(0.1)
    assert len(transcripts) == 2, "transcripts did not appear in time"
    return session


async def _speakers(client: httpx.AsyncClient, account: Account, **params):
    response = await client.get("/v1/speakers", headers=account.headers, params=params)
    assert response.status_code == 200
    return response.json()["items"]


async def test_clusters_unite_voices_across_sessions(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    account = await make_account()
    first = await _diarized_session(account)
    second = await _diarized_session(account)

    speakers = await _speakers(client, account)
    assert len(speakers) == 2
    for speaker in speakers:
        assert speaker["name"] is None
        assert speaker["model"] == "stub-diarizer"
        assert speaker["sessions"] == 2 and speaker["embeddings"] == 2

    # Both sessions resolve to the same two clusters.
    for session in (first, second):
        listed = await client.get(
            f"/v1/sessions/{session.id}/speakers", headers=account.headers
        )
        rows = listed.json()
        assert {row["speaker_id"] for row in rows} == {s["id"] for s in speakers}
        assert all(row["talk_ms"] == 150 for row in rows)


async def test_labeling_and_fetch_by_name_across_sessions(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    account = await make_account()
    first = await _diarized_session(account)
    second = await _diarized_session(account)

    speakers = await _speakers(client, account)
    target = speakers[0]

    labeled = await client.post(
        f"/v1/speakers/{target['id']}/label",
        headers=account.headers,
        json={"name": "Mom"},
    )
    assert labeled.status_code == 200 and labeled.json()["name"] == "Mom"
    assert (await client.post(
        f"/v1/speakers/{target['id']}/label", headers=account.headers, json={}
    )).status_code == 422  # empty body must not silently clear

    named = await _speakers(client, account, name="Mom")
    assert [s["id"] for s in named] == [target["id"]]

    by_name = (
        await client.get(
            "/v1/speakers/transcripts",
            headers=account.headers,
            params={"name": "Mom"},
        )
    ).json()
    by_id = (
        await client.get(
            f"/v1/speakers/{target['id']}/transcripts", headers=account.headers
        )
    ).json()
    assert by_name["items"] == by_id["items"]
    assert {item["session_id"] for item in by_name["items"]} == {
        str(first.id), str(second.id)
    }
    assert all(
        item["text"] == "hello from the stub transcriber"
        and item["speaker_id"] == target["id"]
        for item in by_name["items"]
    )

    # The timeline carries the resolved identity on that speaker's events.
    timeline = (
        await client.get(f"/v1/sessions/{first.id}/timeline", headers=account.headers)
    ).json()
    transcripts = [e for e in timeline["items"] if e["type"] == "transcript"]
    assert len(transcripts) == 2
    named_events = [e for e in transcripts if e["speaker_name"] == "Mom"]
    assert len(named_events) == 1 and named_events[0]["speaker_id"] == target["id"]
    other_events = [e for e in transcripts if e["speaker_name"] is None]
    assert len(other_events) == 1 and other_events[0]["speaker_id"] is not None

    # Clearing the label works with an explicit null.
    cleared = await client.post(
        f"/v1/speakers/{target['id']}/label",
        headers=account.headers,
        json={"name": None},
    )
    assert cleared.status_code == 200 and cleared.json()["name"] is None


async def test_speakers_are_scoped_to_their_owner(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    account = await make_account()
    other = await make_account()
    await _diarized_session(account)

    assert await _speakers(client, other) == []
    speaker_id = (await _speakers(client, account))[0]["id"]
    for path in (
        f"/v1/speakers/{speaker_id}",
        f"/v1/speakers/{speaker_id}/transcripts",
        f"/v1/speakers/{speaker_id}/similar",
    ):
        response = await client.get(path, headers=other.headers)
        assert response.status_code == 404, path
    merge = await client.post(
        f"/v1/speakers/{speaker_id}/merge",
        headers=other.headers,
        json={"into_speaker_id": speaker_id},
    )
    assert merge.status_code == 404  # foreign path speaker, before any guard


async def test_recluster_rebuild_preserves_named_identity(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    account = await make_account()
    await _diarized_session(account)

    speakers = await _speakers(client, account)
    target = speakers[0]
    await client.post(
        f"/v1/speakers/{target['id']}/label",
        headers=account.headers,
        json={"name": "Anchor"},
    )

    result = subprocess.run(
        [sys.executable, "-m", "cli.main", "speakers", "recluster",
         account.user.email, "--yes"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    after = await _speakers(client, account)
    assert len(after) == 2
    (anchored,) = [s for s in after if s["name"] == "Anchor"]
    assert anchored["id"] == target["id"]  # identity survived the rebuild
    assert anchored["embeddings"] == 1  # and reclaimed its member


async def test_manual_merge_heals_a_split_voice(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    """The stub clusters sit at distance 1.0 — far beyond auto-merge — so
    uniting them exercises exactly the manual path: similar ranks the
    candidate with its real distance, merge folds members and carries the
    loser's name onto an unnamed survivor, and reads resolve to one voice."""
    account = await make_account()
    first = await _diarized_session(account)
    second = await _diarized_session(account)

    loser, survivor = await _speakers(client, account)
    await client.post(
        f"/v1/speakers/{loser['id']}/label",
        headers=account.headers,
        json={"name": "Mom"},
    )

    similar = await client.get(
        f"/v1/speakers/{loser['id']}/similar", headers=account.headers
    )
    assert similar.status_code == 200
    items = similar.json()["items"]
    assert [item["id"] for item in items] == [survivor["id"]]
    assert 0.9 < items[0]["distance"] < 1.1  # orthogonal stub voices

    merged = await client.post(
        f"/v1/speakers/{loser['id']}/merge",
        headers=account.headers,
        json={"into_speaker_id": survivor["id"]},
    )
    assert merged.status_code == 200
    body = merged.json()
    assert body["id"] == survivor["id"]
    assert body["name"] == "Mom"  # unnamed survivor inherits the label
    assert body["embeddings"] == 4 and body["sessions"] == 2

    remaining = await _speakers(client, account)
    assert [speaker["id"] for speaker in remaining] == [survivor["id"]]

    transcripts = (
        await client.get(
            f"/v1/speakers/{survivor['id']}/transcripts", headers=account.headers
        )
    ).json()["items"]
    assert len(transcripts) == 4  # both voices' utterances, both sessions
    assert {item["session_id"] for item in transcripts} == {
        str(first.id), str(second.id)
    }
    assert all(item["speaker_id"] == survivor["id"] for item in transcripts)

    timeline = (
        await client.get(f"/v1/sessions/{first.id}/timeline", headers=account.headers)
    ).json()
    events = [e for e in timeline["items"] if e["type"] == "transcript"]
    assert len(events) == 2
    assert all(
        e["speaker_id"] == survivor["id"] and e["speaker_name"] == "Mom"
        for e in events
    )


async def test_merge_guards_reject_self_and_foreign_targets(
    worker: None, client: httpx.AsyncClient, clean_state
) -> None:
    account = await make_account()
    other = await make_account()
    await _diarized_session(account)
    await _diarized_session(other)
    mine = (await _speakers(client, account))[0]
    foreign = (await _speakers(client, other))[0]

    async def merge_into(target_id: str) -> httpx.Response:
        return await client.post(
            f"/v1/speakers/{mine['id']}/merge",
            headers=account.headers,
            json={"into_speaker_id": target_id},
        )

    assert (await merge_into(mine["id"])).status_code == 422  # self-merge
    assert (await merge_into(foreign["id"])).status_code == 404  # not yours
    assert (await merge_into(str(uuid4()))).status_code == 404  # unknown

    assert len(await _speakers(client, account)) == 2  # nothing merged
    assert len(await _speakers(client, other)) == 2  # and nothing lost
