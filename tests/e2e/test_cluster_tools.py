"""Batch clustering controls end to end: per-user params, the recluster
route, member inspection with playable spans, moves/ejects, and — the point
of the pins design — curation surviving subsequent rebuilds.

The stub diarizer's two voices are orthogonal (cosine distance 1.0), far
beyond the 0.65 default threshold: they only unite under a loosened
threshold or a pin, which makes both mechanisms directly observable.
"""

import subprocess
import sys
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


async def _clustered_session(account: Account):
    session = await make_session(account)
    for _ in range(3):
        await ingest(session.id, wav_bytes(seconds=0.1), duration_ms=100)
    async with DatabasePipe() as pipe:
        await pipe.sessions.end(session.id)
    await wait_for_job(session.id, "speaker-cluster", JobStatus.SUCCEEDED)
    return session


async def _speakers(client: httpx.AsyncClient, account: Account):
    response = await client.get("/v1/speakers", headers=account.headers)
    assert response.status_code == 200
    return response.json()["items"]


async def _params(client: httpx.AsyncClient, account: Account):
    response = await client.get("/v1/speakers/cluster-params", headers=account.headers)
    assert response.status_code == 200
    return response.json()


async def test_params_roundtrip_and_validation(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    from processing.config import get_settings

    body = await _params(client, account)
    assert body["overrides"] == {"cluster_distance": None, "min_talk_ms": None}
    assert body["effective"] == body["defaults"]
    # Defaults mirror the server's env-derived settings, not literals — the
    # e2e env deliberately lowers min_talk to fit the 150ms stub turns.
    assert body["defaults"]["cluster_distance"] == get_settings().cluster_distance
    assert body["defaults"]["min_talk_ms"] == get_settings().cluster_min_talk_ms

    saved = await client.post(
        "/v1/speakers/cluster-params",
        headers=account.headers,
        json={"cluster_distance": 1.5, "min_talk_ms": None},
    )
    assert saved.status_code == 200
    body = saved.json()
    assert body["overrides"]["cluster_distance"] == 1.5
    assert body["overrides"]["min_talk_ms"] is None
    assert body["effective"]["cluster_distance"] == 1.5
    assert body["effective"]["min_talk_ms"] == body["defaults"]["min_talk_ms"]

    for bad in (
        {"cluster_distance": 3.0, "min_talk_ms": None},  # out of range
        {"cluster_distance": 0.0, "min_talk_ms": None},  # zero distance
        {"cluster_distance": None},  # partial body must never silently clear
        {"cluster_distance": None, "min_talk_ms": -1},
    ):
        response = await client.post(
            "/v1/speakers/cluster-params", headers=account.headers, json=bad
        )
        assert response.status_code == 422, bad

    reset = await client.post(
        "/v1/speakers/cluster-params/reset", headers=account.headers
    )
    assert reset.json()["overrides"] == {"cluster_distance": None, "min_talk_ms": None}

    assert (await client.get("/v1/speakers/cluster-params")).status_code == 401


async def test_recluster_threshold_collapses_and_splits(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    await _clustered_session(account)
    assert len(await _speakers(client, account)) == 2

    # Loosen past the stub voices' 1.0 distance: one cluster.
    await client.post(
        "/v1/speakers/cluster-params",
        headers=account.headers,
        json={"cluster_distance": 1.5, "min_talk_ms": None},
    )
    rebuilt = await client.post("/v1/speakers/recluster", headers=account.headers)
    assert rebuilt.status_code == 200
    assert rebuilt.json()["clusters"] == 1 and rebuilt.json()["assigned"] == 2
    assert len(await _speakers(client, account)) == 1

    # Back to defaults: the rebuild re-derives the split.
    await client.post("/v1/speakers/cluster-params/reset", headers=account.headers)
    rebuilt = await client.post("/v1/speakers/recluster", headers=account.headers)
    assert rebuilt.json()["clusters"] == 2
    assert len(await _speakers(client, account)) == 2


async def test_worker_reads_persisted_params(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    # Saved BEFORE any audio exists: the per-session pipeline job must pick
    # the override up on its own.
    await client.post(
        "/v1/speakers/cluster-params",
        headers=account.headers,
        json={"cluster_distance": 1.5, "min_talk_ms": None},
    )
    await _clustered_session(account)
    assert len(await _speakers(client, account)) == 1


async def test_merge_survives_the_next_rebuild(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    """The regression the old incremental design failed: manual curation
    used to evaporate on the next clustering pass."""
    await _clustered_session(account)
    loser, survivor = await _speakers(client, account)
    await client.post(
        f"/v1/speakers/{survivor['id']}/label",
        headers=account.headers,
        json={"name": "Mom"},
    )
    merged = await client.post(
        f"/v1/speakers/{loser['id']}/merge",
        headers=account.headers,
        json={"into_speaker_id": survivor["id"]},
    )
    assert merged.status_code == 200 and merged.json()["embeddings"] == 2

    # A new session triggers a full rebuild at the default threshold, which
    # on geometry alone would split the merged pair apart again.
    await _clustered_session(account)

    after = await _speakers(client, account)
    (mom,) = [s for s in after if s["name"] == "Mom"]
    assert mom["id"] == survivor["id"]
    # The pinned pair stayed together; the average-linkage pull also drew in
    # the new session's matching voice. Without pins Mom would be back to 2.
    assert mom["embeddings"] == 3


async def test_members_move_eject_and_unpin(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    await _clustered_session(account)
    first, second = await _speakers(client, account)

    listed = await client.get(
        f"/v1/speakers/{first['id']}/members", headers=account.headers
    )
    assert listed.status_code == 200
    body = listed.json()
    assert body["has_more"] is False and len(body["items"]) == 1
    member = body["items"][0]
    assert member["pinned"] is False
    assert member["distance"] < 0.01  # sole member ≈ centroid
    assert (member["clip_start_ms"], member["clip_end_ms"]) in ((0, 150), (150, 300))
    assert member["speaker"].startswith("b0:SPEAKER_")

    # Guards: self-move, unknown target, someone else's artifact → 409.
    self_move = await client.post(
        f"/v1/speakers/{first['id']}/members/{member['artifact_id']}/reassign",
        headers=account.headers,
        json={"into_speaker_id": first["id"], "name": None},
    )
    assert self_move.status_code == 422
    unknown = await client.post(
        f"/v1/speakers/{first['id']}/members/{member['artifact_id']}/reassign",
        headers=account.headers,
        json={"into_speaker_id": str(uuid4()), "name": None},
    )
    assert unknown.status_code == 404
    second_members = (
        await client.get(
            f"/v1/speakers/{second['id']}/members", headers=account.headers
        )
    ).json()["items"]
    stale = await client.post(
        f"/v1/speakers/{first['id']}/members/{second_members[0]['artifact_id']}/reassign",
        headers=account.headers,
        json={"into_speaker_id": second["id"], "name": None},
    )
    assert stale.status_code == 409  # that print is not in the path cluster

    # Move: the print lands pinned in the target; the emptied unnamed
    # source dissolves.
    moved = await client.post(
        f"/v1/speakers/{first['id']}/members/{member['artifact_id']}/reassign",
        headers=account.headers,
        json={"into_speaker_id": second["id"], "name": None},
    )
    assert moved.status_code == 200
    body = moved.json()
    assert body["source"] is None
    assert body["target"]["id"] == second["id"] and body["target"]["embeddings"] == 2
    assert len(await _speakers(client, account)) == 1

    members = (
        await client.get(
            f"/v1/speakers/{second['id']}/members", headers=account.headers
        )
    ).json()["items"]
    assert sum(1 for m in members if m["pinned"]) == 1
    pinned = next(m for m in members if m["pinned"])

    unpinned = await client.post(
        f"/v1/speakers/{second['id']}/members/{pinned['artifact_id']}/unpin",
        headers=account.headers,
    )
    assert unpinned.status_code == 200 and unpinned.json()["unpinned"] is True
    again = await client.post(
        f"/v1/speakers/{second['id']}/members/{pinned['artifact_id']}/unpin",
        headers=account.headers,
    )
    assert again.status_code == 404

    # Eject into a fresh named cluster.
    ejected = await client.post(
        f"/v1/speakers/{second['id']}/members/{pinned['artifact_id']}/reassign",
        headers=account.headers,
        json={"into_speaker_id": None, "name": "Guest"},
    )
    assert ejected.status_code == 200
    body = ejected.json()
    assert body["target"]["name"] == "Guest" and body["target"]["embeddings"] == 1
    assert body["source"]["id"] == second["id"] and body["source"]["embeddings"] == 1


async def test_new_routes_are_scoped_to_their_owner(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    other = await make_account()
    await _clustered_session(account)
    speaker = (await _speakers(client, account))[0]
    member = (
        await client.get(
            f"/v1/speakers/{speaker['id']}/members", headers=account.headers
        )
    ).json()["items"][0]

    listed = await client.get(
        f"/v1/speakers/{speaker['id']}/members", headers=other.headers
    )
    assert listed.status_code == 404
    for action, body in (
        ("reassign", {"into_speaker_id": None, "name": None}),
        ("unpin", None),
    ):
        response = await client.post(
            f"/v1/speakers/{speaker['id']}/members/{member['artifact_id']}/{action}",
            headers=other.headers,
            json=body,
        )
        assert response.status_code == 404, action

    # Params are per-user: my override never leaks into another account.
    await client.post(
        "/v1/speakers/cluster-params",
        headers=account.headers,
        json={"cluster_distance": 1.5, "min_talk_ms": None},
    )
    body = await _params(client, other)
    assert body["overrides"] == {"cluster_distance": None, "min_talk_ms": None}


async def test_cli_distance_flag(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    await _clustered_session(account)
    assert len(await _speakers(client, account)) == 2

    result = subprocess.run(
        [sys.executable, "-m", "cli.main", "speakers", "recluster",
         account.user.email, "--distance", "1.5", "--yes"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(await _speakers(client, account)) == 1
