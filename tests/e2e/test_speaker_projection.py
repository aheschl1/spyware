"""The speaker map's projection endpoint.

The contracts worth pinning are the ones a plausible refactor would break:
the basis is fitted on the whole corpus so filters never move a point, the
response is a pure function of its input so the plot never mirrors itself
between polls, and two embedding models are never fitted together.
"""

import httpx

from database.pipe import DatabasePipe
from database.schema.jobs import JobStatus
from tests.e2e.conftest import (
    Account,
    ingest,
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


async def _projection(client: httpx.AsyncClient, account: Account, **params):
    response = await client.get(
        "/v1/speakers/projection", headers=account.headers, params=params
    )
    assert response.status_code == 200
    return response.json()


async def test_requires_auth(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/speakers/projection")).status_code == 401


async def test_empty_corpus_is_not_an_error(
    client: httpx.AsyncClient, account: Account
) -> None:
    body = await _projection(client, account)
    assert body["model"] is None
    assert body["points"] == [] and body["clusters"] == []
    assert body["available_models"] == []
    assert body["explained_variance_ratio"] == [0.0, 0.0, 0.0]
    assert body["fit_points"] == 0 and not body["truncated"]


async def test_shape_matches_the_corpus(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    await _clustered_session(account)
    body = await _projection(client, account)

    async with DatabasePipe() as pipe:
        corpus = await pipe.speakers.projection_corpus(account.user.id, body["model"])

    assert body["fit_points"] == len(corpus) == body["returned"]
    assert len(body["points"]) == len(corpus)
    assert body["available_models"][0]["embeddings"] == len(corpus)
    assert len(body["explained_variance_ratio"]) == 3
    assert all(len(point["coords"]) == 3 for point in body["points"])
    assert len(body["basis_id"]) == 16

    point = body["points"][0]
    assert point["speaker"].startswith("b")  # block-namespaced label
    assert point["start_ms"] is not None and point["end_ms"] is not None
    assert point["started_at"] is not None


async def test_ratios_are_descending_and_bounded(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    await _clustered_session(account)
    ratio = (await _projection(client, account))["explained_variance_ratio"]
    assert ratio == sorted(ratio, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in ratio)
    assert sum(ratio) <= 1.0 + 1e-9


async def test_repeat_requests_are_identical(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    """PCA signs are arbitrary; a viewer that mirrors between polls is broken."""
    await _clustered_session(account)
    first = await _projection(client, account)
    second = await _projection(client, account)
    assert first["basis_id"] == second["basis_id"]
    assert [p["coords"] for p in first["points"]] == [
        p["coords"] for p in second["points"]
    ]


async def test_session_filter_does_not_move_points(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    session = await _clustered_session(account)
    await _clustered_session(account)

    everything = await _projection(client, account)
    filtered = await _projection(client, account, session_id=str(session.id))

    assert filtered["fit_points"] == everything["fit_points"]
    assert filtered["returned"] < everything["returned"]
    assert filtered["basis_id"] == everything["basis_id"]

    placed = {p["artifact_id"]: p["coords"] for p in everything["points"]}
    for point in filtered["points"]:
        assert point["session_id"] == str(session.id)
        assert point["coords"] == placed[point["artifact_id"]]


async def test_include_unassigned_toggle(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    await _clustered_session(account)
    assigned = await _projection(client, account, include_unassigned=False)
    assert all(point["speaker_id"] is not None for point in assigned["points"])
    assert assigned["fit_points"] >= assigned["returned"]


async def test_cluster_markers_are_the_mean_of_their_members(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    await _clustered_session(account)
    body = await _projection(client, account)
    assert body["clusters"]

    for cluster in body["clusters"]:
        members = [
            point["coords"]
            for point in body["points"]
            if point["speaker_id"] == cluster["speaker_id"]
        ]
        assert cluster["embeddings"] == len(members)
        for axis in range(3):
            expected = sum(coords[axis] for coords in members) / len(members)
            assert cluster["coords"][axis] == expected


async def test_never_mixes_embedding_models(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    """Vectors from two models live in different spaces; one basis over both
    would be meaningless."""
    await _clustered_session(account)
    body = await _projection(client, account)
    real = body["model"]

    # A second model's print needs its own artifact: artifact_id is the PK.
    async with DatabasePipe() as pipe:
        await pipe.connection.execute(
            """
                WITH src AS (
                    SELECT e.session_id, e.speaker, e.embedding, a.pipeline, a.kind
                    FROM speaker_embeddings e
                    JOIN pipeline_artifacts a ON a.id = e.artifact_id
                    LIMIT 1
                ), created AS (
                    INSERT INTO pipeline_artifacts (pipeline, kind, session_id, metadata)
                    SELECT pipeline, kind, session_id, '{}'::jsonb FROM src
                    RETURNING id, session_id
                )
                INSERT INTO speaker_embeddings
                    (artifact_id, session_id, speaker, model, embedding)
                SELECT created.id, created.session_id, src.speaker || ':alt',
                       'other/model', src.embedding
                FROM created, src
            """
        )

    default = await _projection(client, account)
    assert default["model"] == real  # majority wins
    assert {m["model"] for m in default["available_models"]} == {real, "other/model"}
    assert all(point["speaker"].endswith(":alt") is False for point in default["points"])

    other = await _projection(client, account, model="other/model")
    assert other["model"] == "other/model"
    assert other["fit_points"] == 1


async def test_limit_truncates_deterministically(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    await _clustered_session(account)
    first = await _projection(client, account, limit=1)
    second = await _projection(client, account, limit=1)
    assert first["returned"] == 1
    assert first["truncated"] is (first["fit_points"] > 1)
    assert first["points"] == second["points"]


async def test_scoped_to_the_owner(
    worker: None, client: httpx.AsyncClient, account: Account, other_account: Account
) -> None:
    await _clustered_session(account)
    body = await _projection(client, other_account)
    assert body["points"] == [] and body["model"] is None


async def test_recluster_moves_assignments_not_geometry(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    """Coordinates come from the vectors; assignment comes from clustering."""
    await _clustered_session(account)
    before = await _projection(client, account)

    response = await client.post("/v1/speakers/recluster", headers=account.headers)
    assert response.status_code == 200

    after = await _projection(client, account)
    assert after["basis_id"] == before["basis_id"]
    assert [p["coords"] for p in after["points"]] == [
        p["coords"] for p in before["points"]
    ]
