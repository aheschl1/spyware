"""Snapshot a session's speaker curation before a re-diarize, and re-key it
onto the new labels once diarize republishes.

The snapshot is a whole-session ``label-carry/label-snapshot`` artifact:
it survives the diarize/transcribe deletes, diarize applies the label half
in its publish transaction, and transcribe consults the edit half per
utterance. See ``processing/label_carry.py`` for the matching rules.
"""

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from database.pipe import DatabasePipe
from database.schema.artifacts import ArtifactCreate, PipelineArtifact
from processing.label_carry import label_records, map_labels

logger = logging.getLogger(__name__)

PIPELINE = "label-carry"
KIND = "label-snapshot"


async def snapshot(pipe: DatabasePipe, session_id: UUID) -> PipelineArtifact:
    """Record every label's identity and every edited transcript; replaces
    any earlier snapshot for the session."""
    labels = await pipe.label_carry.identities_for_session(session_id)
    edits = await pipe.label_carry.edited_transcripts(session_id)
    previous = await pending(pipe, session_id)
    if previous is not None and previous.metadata.get("applied") is None and not labels:
        # Nothing to snapshot because the tier output is already gone: the
        # earlier snapshot is the curation, so keep it.
        return previous
    await pipe.artifacts.delete_for_pipeline(session_id, PIPELINE)
    return await pipe.artifacts.create(
        ArtifactCreate(
            pipeline=PIPELINE,
            kind=KIND,
            session_id=session_id,
            metadata={
                "labels": [
                    {
                        "speaker": row.speaker,
                        "model": row.model,
                        "speaker_id": str(row.speaker_id),
                        "pinned": row.pinned,
                        "spans": row.spans,
                    }
                    for row in labels
                ],
                "edits": [e.model_dump() for e in edits],
            },
        )
    )


async def pending(pipe: DatabasePipe, session_id: UUID) -> PipelineArtifact | None:
    """The session's snapshot, if one exists."""
    return await pipe.artifacts.find(PIPELINE, KIND, session_id)


async def apply_labels(
    pipe: DatabasePipe,
    snap: PipelineArtifact,
    session_id: UUID,
    utterances: Sequence[ArtifactCreate],
) -> dict[str, Any]:
    """Re-key the snapshot's pins and assignments onto the labels of the
    utterances diarize is publishing. Runs inside diarize's transaction,
    after the new embeddings exist. Idempotent per snapshot."""
    if snap.metadata.get("applied") is not None:
        return snap.metadata["applied"]
    new_spans: dict[str, list[tuple[int, int]]] = {}
    for u in utterances:
        if u.start_ms is not None and u.end_ms is not None:
            new_spans.setdefault(str(u.metadata["speaker"]), []).append((u.start_ms, u.end_ms))
    records = label_records(snap.metadata)
    mapping = map_labels({r.speaker: r.spans for r in records}, new_spans)

    counts = {"pins": 0, "assignments": 0, "unmapped": 0}
    for record in records:
        target = mapping.get(record.speaker)
        if target is None:
            counts["unmapped"] += 1
            continue
        identity = UUID(record.speaker_id)
        if record.pinned:
            if target != record.speaker:
                await pipe.label_carry.unpin_label(session_id, record.speaker, record.model)
            counts["pins"] += await pipe.label_carry.pin_label(session_id, target, identity)
        else:
            counts["assignments"] += await pipe.label_carry.assign_label(
                session_id, target, identity
            )
    await pipe.artifacts.merge_metadata(
        snap.id, {"applied": counts, "mapping": mapping}
    )
    if counts["unmapped"]:
        logger.info(
            "label-carry: %d label(s) of session %s had no overlapping new label",
            counts["unmapped"],
            session_id,
        )
    return counts
