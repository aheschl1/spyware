"""`sessions rediarize --carry-labels`: names, pins and transcript edits
survive a re-diarize that replaces every voice-print and transcript."""

import asyncio
import subprocess
import sys
import time

import httpx

from database.pipe import DatabasePipe
from database.schema.jobs import JobStatus
from tests.e2e.conftest import REPO_ROOT, Account, wait_for_job
from tests.e2e.test_speakers import _diarized_session


async def _transcripts(session_id, count: int, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    rows: list = []
    while time.monotonic() < deadline and len(rows) < count:
        async with DatabasePipe() as pipe:
            rows = await pipe.artifacts.list_for_session(session_id, kind="transcript")
        await asyncio.sleep(0.1)
    assert len(rows) == count, "transcripts did not appear in time"
    return rows


async def test_curation_survives_rediarize(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    session = await _diarized_session(account)
    speakers = (await client.get("/v1/speakers", headers=account.headers)).json()["items"]
    assert len(speakers) == 2
    pinned, assigned = speakers
    for speaker, name in ((pinned, "Mom"), (assigned, "Dad")):
        await client.post(
            f"/v1/speakers/{speaker['id']}/label", headers=account.headers, json={"name": name}
        )
    async with DatabasePipe() as pipe:
        labels = await pipe.speakers.labels_for_session(session.id)
        member = next(l for l in labels if str(l.speaker_id) == pinned["id"])
        assert member.artifact_id is not None and member.speaker_id is not None
        await pipe.speakers.pin(member.artifact_id, member.speaker_id)
        old_labels = {l.speaker: str(l.speaker_id) for l in labels}

    transcripts = await _transcripts(session.id, 2)
    target = transcripts[0]
    edited = await client.post(
        f"/v1/sessions/{session.id}/transcripts/{target.id}",
        headers=account.headers,
        json={"text": "hand fixed"},
    )
    assert edited.status_code == 200

    result = subprocess.run(
        [sys.executable, "-m", "cli.main", "sessions", "rediarize", str(session.id),
         "--carry-labels", "--yes"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "carrying 2 label(s) and 1 edit(s)" in result.stdout

    await wait_for_job(session.id, "diarize", JobStatus.SUCCEEDED)
    await wait_for_job(session.id, "speaker-cluster", JobStatus.SUCCEEDED)
    new_transcripts = await _transcripts(session.id, 2)

    async with DatabasePipe() as pipe:
        diarize_map = await pipe.artifacts.find("diarize", "diarize-map", session.id)
        labels = await pipe.speakers.labels_for_session(session.id)
    assert diarize_map is not None
    assert diarize_map.metadata["carried"] == {
        "pins": 1, "assignments": 1, "unmapped": 0, "vanished": 0
    }
    # Fresh voice-prints (new artifact ids), same identities under the same labels.
    assert {l.speaker: str(l.speaker_id) for l in labels} == old_labels
    assert {l.name for l in labels} == {"Mom", "Dad"}
    assert all(l.artifact_id != member.artifact_id for l in labels)

    carried = [t for t in new_transcripts if t.metadata.get("edited")]
    assert [t.metadata["text"] for t in carried] == ["hand fixed"]
    assert (carried[0].start_ms, carried[0].end_ms) == (target.start_ms, target.end_ms)
    assert "words" not in carried[0].metadata


async def test_rediarize_all_requires_a_target_or_the_flag() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cli.main", "sessions", "rediarize", "--yes"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SESSION_ID or --all" in result.stderr


async def test_snapshot_never_overwrites_curation_with_nothing(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    """A second rediarize before the worker republishes finds no voice-prints;
    the earlier snapshot must survive it."""
    from services import label_carry

    session = await _diarized_session(account)
    async with DatabasePipe() as pipe:
        first = await label_carry.snapshot(pipe, session.id)
        assert first.metadata["labels"]
        await pipe.artifacts.delete_for_pipeline(session.id, "diarize")
        second = await label_carry.snapshot(pipe, session.id)
    assert second.id == first.id


async def test_pruned_cluster_in_snapshot_is_skipped_not_fatal(
    worker: None, client: httpx.AsyncClient, account: Account
) -> None:
    """A rebuild mid-rerun prunes empty unnamed clusters; a snapshot naming
    one must not kill the diarize job."""
    from uuid import uuid4

    from services import label_carry

    session = await _diarized_session(account)
    async with DatabasePipe() as pipe:
        snap = await label_carry.snapshot(pipe, session.id)
        labels = snap.metadata["labels"]
        labels[0]["speaker_id"] = str(uuid4())
        await pipe.artifacts.merge_metadata(snap.id, {"labels": labels})
    result = subprocess.run(
        [sys.executable, "-m", "cli.main", "sessions", "rediarize", str(session.id), "--yes"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    await wait_for_job(session.id, "diarize", JobStatus.SUCCEEDED)
    async with DatabasePipe() as pipe:
        diarize_map = await pipe.artifacts.find("diarize", "diarize-map", session.id)
    assert diarize_map is not None
    assert diarize_map.metadata["carried"]["vanished"] == 1
    assert diarize_map.metadata["carried"]["assignments"] == 1
