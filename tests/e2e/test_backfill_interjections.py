"""The one-off interjection backfill over legacy (unmarked) utterances."""

from database.pipe import DatabasePipe
from database.schema.artifacts import ArtifactCreate
from tests.e2e.conftest import make_account, make_session


def _utterance(session_id, start, end, speaker, **extra):
    return ArtifactCreate(
        pipeline="diarize", kind="utterance", session_id=session_id,
        start_ms=start, end_ms=end, metadata={"speaker": speaker, **extra},
    )


async def test_backfill_marks_legacy_utterances(clean_state) -> None:
    from cli.main import _backfill_session_interjections

    account = await make_account()
    session = await make_session(account)
    async with DatabasePipe() as pipe:
        host, inner, later = await pipe.artifacts.create_many([
            _utterance(session.id, 0, 5_000, "A"),
            _utterance(session.id, 1_000, 2_000, "B"),
            _utterance(session.id, 4_500, 6_000, "B"),  # partial overlap
        ])
        # A stale marker on the partial-overlap row must be cleared.
        await pipe.connection.execute(
            "UPDATE pipeline_artifacts SET links = '{\"host_utterance\": \"x\"}',"
            " metadata = metadata || '{\"interjections\": 9}' WHERE id = %s",
            (later.id,),
        )
        transcript, stale = await pipe.artifacts.create_many([
            ArtifactCreate(
                pipeline="transcribe", kind="transcript", session_id=session.id,
                start_ms=1_000, end_ms=2_000, links={"utterance": str(inner.id)},
                metadata={"text": "hm"},
            ),
            ArtifactCreate(
                pipeline="transcribe", kind="transcript", session_id=session.id,
                start_ms=4_500, end_ms=6_000,
                links={"utterance": str(later.id), "host_utterance": "x"},
                metadata={"text": "so"},
            ),
        ])
        await pipe.artifacts.create(ArtifactCreate(
            pipeline="diarize", kind="diarize-map", session_id=session.id,
            metadata={"utterances": 3},
        ))

    for _ in range(2):  # idempotent
        async with DatabasePipe() as pipe:
            assert await _backfill_session_interjections(pipe, session.id) == 1

    async with DatabasePipe() as pipe:
        rows = {a.id: a for a in await pipe.artifacts.list_for_session(session.id, limit=100)}
        maps = await pipe.artifacts.find("diarize", "diarize-map", session.id)
    assert rows[host.id].metadata["interjections"] == 1
    assert "host_utterance" not in rows[host.id].links
    assert rows[inner.id].links["host_utterance"] == str(host.id)
    assert "interjections" not in rows[inner.id].metadata
    assert rows[later.id].links == {}
    assert "interjections" not in rows[later.id].metadata
    assert rows[transcript.id].links["host_utterance"] == str(host.id)
    assert rows[stale.id].links == {"utterance": str(later.id)}
    assert rows[transcript.id].metadata["text"] == "hm"
    assert maps is not None and maps.metadata["interjections"] == 1
