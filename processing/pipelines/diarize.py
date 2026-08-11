"""Tier: split detected speech into per-speaker turns, with embeddings.

Consumes each session's ``speech-map`` (one diarize job per session), but
pyannote never sees the whole session: speech-spans are re-merged into
*blocks* — contiguous speech regions — and one rendered clip per block goes
to the diarization service. Silence between blocks was already excluded by
the VAD tier and is never rendered or uploaded.

Blocks exist because diarization label consistency needs long context: within
one block "SPEAKER_00" is stable; across blocks it is not, so labels are
namespaced per block (``b{start}:SPEAKER_00``) and global identity is the
future clustering tier's job — fed by the per-speaker embeddings this tier
stores as pgvector rows (``speaker_embeddings``), queryable with distance
operators.

Besides turns, the tier publishes ``utterance`` artifacts — same-speaker
turns merged into ASR-sized units — which are what the transcribe tier
consumes. That makes this tier the transcription gate: pyannote's
segmentation hears far-field speakers the VAD misses, so gating ASR on
anything less sensitive silences one side of a conversation.

Publication is atomic: previous diarize output for the session is deleted and
the full new set (turns, utterances, embeddings, the ``diarize-map`` summary)
inserted in one transaction — along with the session's now-stale transcripts,
which this tier owns invalidating: they derive from utterances that no longer
exist. The map's presence is the completion marker consumers wait for, so
partial output is never visible and retries are idempotent.
"""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from database.pipe import DatabasePipe
from database.repos.pipelines.diarize import DiarizeQueries
from database.schema.artifacts import ArtifactCreate
from database.schema.embeddings import SpeakerEmbeddingCreate
from database.schema.jobs import Job, JobCreate
from processing.base import Pipeline
from processing.config import get_settings
from processing.diarizer import Diarizer
from services import stitch, timeline

logger = logging.getLogger(__name__)

_SOURCE_PIPELINE = "speech-detect"


@dataclass(frozen=True, slots=True)
class Block:
    start_ms: int
    end_ms: int


def blocks_from_spans(
    spans: Sequence[tuple[int, int]], *, merge_gap_ms: int, max_block_ms: int
) -> list[Block]:
    """Merge ordered speech spans into bounded contiguous blocks.

    Adjacent spans join when the silence between them is at most
    ``merge_gap_ms``; a block that would exceed ``max_block_ms`` is closed at
    the previous span boundary instead (never mid-span — spans are short, and
    a span boundary is a place the VAD already called quiet).
    """
    blocks: list[Block] = []
    for start, end in spans:
        if (
            blocks
            and start - blocks[-1].end_ms <= merge_gap_ms
            and end - blocks[-1].start_ms <= max_block_ms
        ):
            blocks[-1] = Block(blocks[-1].start_ms, max(blocks[-1].end_ms, end))
        else:
            blocks.append(Block(start, end))
    return blocks


@dataclass(frozen=True, slots=True)
class Utterance:
    start_ms: int
    end_ms: int
    speaker: str  # block-namespaced label
    turns: int  # source turns merged in


def utterances_from_turns(
    turns: Sequence[tuple[int, int, str]], *, merge_gap_ms: int, max_utterance_ms: int
) -> list[Utterance]:
    """Merge same-speaker turns into ASR-sized utterances.

    Merging is per speaker label: block namespacing already prevents
    cross-block joins, and a merge may span another speaker's short
    interruption — a backchannel shouldn't split a sentence. Turns are sorted
    first (service order is not trusted); zero-length turns are dropped;
    overlapping same-speaker turns (pyannote powerset artifacts) collapse. A
    merge that would exceed ``max_utterance_ms`` closes at the previous turn
    boundary instead; a single turn longer than the cap is split into
    near-equal pieces (never a sliver tail).
    """
    by_speaker: dict[str, list[tuple[int, int]]] = {}
    for start, end, speaker in turns:
        if end > start:
            by_speaker.setdefault(speaker, []).append((start, end))

    out: list[Utterance] = []
    for speaker, speaker_turns in by_speaker.items():
        merged: list[list[int]] = []  # [start, end, turn count]
        for start, end in sorted(speaker_turns):
            if (
                merged
                and start - merged[-1][1] <= merge_gap_ms
                and max(end, merged[-1][1]) - merged[-1][0] <= max_utterance_ms
            ):
                merged[-1][1] = max(merged[-1][1], end)
                merged[-1][2] += 1
            else:
                merged.append([start, end, 1])
        for start, end, count in merged:
            duration = end - start
            if duration <= max_utterance_ms:
                out.append(Utterance(start, end, speaker, count))
                continue
            # Only a single over-cap turn reaches here (merges respect the cap).
            pieces = math.ceil(duration / max_utterance_ms)
            for index in range(pieces):
                out.append(
                    Utterance(
                        start + duration * index // pieces,
                        start + duration * (index + 1) // pieces,
                        speaker,
                        count,
                    )
                )
    out.sort(key=lambda u: (u.start_ms, u.end_ms, u.speaker))
    return out


class DiarizePipeline(Pipeline):
    name = "diarize"

    async def setup(self) -> None:
        self._settings = get_settings()
        self._diarizer = Diarizer(self._settings)

    async def teardown(self) -> None:
        await self._diarizer.close()

    async def discover(self, limit: int) -> Sequence[JobCreate]:
        async with DatabasePipe() as pipe:
            maps = await DiarizeQueries(pipe.connection).maps_without_jobs(
                self.name, _SOURCE_PIPELINE, limit
            )
        return tuple(
            JobCreate(
                pipeline=self.name,
                session_id=speech_map.session_id,
                artifact_id=speech_map.id,
                dedup_key=f"{self.name}:artifact:{speech_map.id}",
            )
            for speech_map in maps
        )

    async def process(self, job: Job) -> dict[str, Any]:
        assert job.session_id is not None and job.artifact_id is not None
        settings = self._settings

        async with DatabasePipe() as pipe:
            speech_map = await pipe.artifacts.get(job.artifact_id)
            spans = await pipe.artifacts.list_for_session(
                job.session_id, pipeline=_SOURCE_PIPELINE, kind="speech-span",
                limit=1_000_000,
            )
        if speech_map is None:
            return {"skipped": "speech-map vanished"}
        if not spans or speech_map.metadata.get("skipped"):
            return await self._publish(
                job, turns=[], utterances=[], embeddings=[], blocks=0
            )

        blocks = blocks_from_spans(
            [(span.start_ms, span.end_ms) for span in spans],
            merge_gap_ms=settings.diarize_block_merge_gap_ms,
            max_block_ms=settings.diarize_max_block_ms,
        )
        try:
            line = await timeline.load_timeline(job.session_id)
            if line is None:
                return await self._publish(
                    job,
                    turns=[],
                    utterances=[],
                    embeddings=[],
                    blocks=0,
                    skipped="session audio is gone",
                )
            # DiarizerError raises through: the worker retries with backoff.
            turn_rows, embedding_rows = [], []
            for block in blocks:
                clip = await timeline.render_range(line, block.start_ms, block.end_ms)
                result = await self._diarizer.diarize(
                    clip, filename=f"{job.session_id}-{block.start_ms}.wav"
                )
                turn_rows += self._turn_artifacts(job, block, result)
                embedding_rows += self._embedding_rows(job, block, result)
        except (stitch.NotStitchable, timeline.NotRenderable) as exc:
            # Retrying cannot make the audio renderable.
            return await self._publish(
                job, turns=[], utterances=[], embeddings=[], blocks=0, skipped=str(exc)
            )

        return await self._publish(
            job,
            turns=turn_rows,
            utterances=self._utterance_artifacts(job, turn_rows),
            embeddings=embedding_rows,
            blocks=len(blocks),
        )

    def _turn_artifacts(
        self, job: Job, block: Block, result: Any
    ) -> list[ArtifactCreate]:
        minimum = self._settings.diarize_min_turn_ms
        rows = []
        for turn in result.turns:
            if turn.end_ms - turn.start_ms < minimum:
                continue
            rows.append(
                ArtifactCreate(
                    pipeline=self.name,
                    kind="speaker-turn",
                    session_id=job.session_id,
                    start_ms=block.start_ms + turn.start_ms,
                    end_ms=block.start_ms + turn.end_ms,
                    metadata={
                        "speaker": f"b{block.start_ms}:{turn.speaker}",
                        "block_start_ms": block.start_ms,
                        "block_end_ms": block.end_ms,
                    },
                )
            )
        return rows

    def _utterance_artifacts(
        self, job: Job, turn_rows: Sequence[ArtifactCreate]
    ) -> list[ArtifactCreate]:
        """ASR units from the already-filtered turn rows: what transcribe eats."""
        settings = self._settings
        blocks_by_speaker = {
            row.metadata["speaker"]: (
                row.metadata["block_start_ms"],
                row.metadata["block_end_ms"],
            )
            for row in turn_rows
        }
        utterances = utterances_from_turns(
            [(row.start_ms, row.end_ms, row.metadata["speaker"]) for row in turn_rows],
            merge_gap_ms=settings.diarize_utterance_merge_gap_ms,
            max_utterance_ms=settings.diarize_max_utterance_ms,
        )
        rows = []
        for utterance in utterances:
            block_start, block_end = blocks_by_speaker[utterance.speaker]
            rows.append(
                ArtifactCreate(
                    pipeline=self.name,
                    kind="utterance",
                    session_id=job.session_id,
                    start_ms=utterance.start_ms,
                    end_ms=utterance.end_ms,
                    metadata={
                        "speaker": utterance.speaker,
                        "turns": utterance.turns,
                        "block_start_ms": block_start,
                        "block_end_ms": block_end,
                    },
                )
            )
        return rows

    def _embedding_rows(
        self, job: Job, block: Block, result: Any
    ) -> list[tuple[ArtifactCreate, list[float]]]:
        """One artifact plus its vector per (block, local speaker) embedding.

        The vector lands in ``speaker_embeddings`` (pgvector), cascade-keyed
        by the artifact row, so the delete-then-recreate publication covers
        vectors and rows in the same transaction; the artifact carries the
        addressing.
        """
        talk_ms: dict[str, int] = {}
        for turn in result.turns:
            talk_ms[turn.speaker] = talk_ms.get(turn.speaker, 0) + (
                turn.end_ms - turn.start_ms
            )
        rows = []
        for speaker, vector in sorted(result.embeddings.items()):
            namespaced = f"b{block.start_ms}:{speaker}"
            rows.append(
                (
                    ArtifactCreate(
                        pipeline=self.name,
                        kind="speaker-embedding",
                        session_id=job.session_id,
                        start_ms=block.start_ms,
                        end_ms=block.end_ms,
                        metadata={
                            "speaker": namespaced,
                            "dim": len(vector),
                            "model": result.model,
                            # The clustering tier's quality gate: voice-prints
                            # from very little speech are noise.
                            "talk_ms": talk_ms.get(speaker, 0),
                        },
                    ),
                    vector,
                )
            )
        return rows

    async def _publish(
        self,
        job: Job,
        *,
        turns: list[ArtifactCreate],
        utterances: list[ArtifactCreate],
        embeddings: list[tuple[ArtifactCreate, list[float]]],
        blocks: int,
        skipped: str | None = None,
    ) -> dict[str, Any]:
        """Replace the session's diarize output atomically, map last-in-set."""
        if skipped:
            logger.info("diarize skipping session %s: %s", job.session_id, skipped)
        settings = self._settings
        map_metadata: dict[str, Any] = {
            "blocks": blocks,
            "turns": len(turns),
            "utterances": len(utterances),
            "speakers": len(embeddings),
            "params": {
                "block_merge_gap_ms": settings.diarize_block_merge_gap_ms,
                "max_block_ms": settings.diarize_max_block_ms,
                "min_turn_ms": settings.diarize_min_turn_ms,
                "utterance_merge_gap_ms": settings.diarize_utterance_merge_gap_ms,
                "max_utterance_ms": settings.diarize_max_utterance_ms,
            },
        }
        if skipped:
            map_metadata["skipped"] = skipped
        async with DatabasePipe() as pipe:
            await pipe.artifacts.delete_for_pipeline(job.session_id, self.name)
            # Transcripts derive from utterances this delete just invalidated;
            # leaving them would show stale generations on the timeline, so
            # this tier owns removing them. Transcribe jobs whose utterance
            # vanished skip themselves (artifact_id goes NULL).
            await pipe.artifacts.delete_for_pipeline(job.session_id, "transcribe")
            await pipe.artifacts.create_many(turns)
            await pipe.artifacts.create_many(utterances)
            created = await pipe.artifacts.create_many([row for row, _ in embeddings])
            await pipe.embeddings.create_many(
                [
                    SpeakerEmbeddingCreate(
                        artifact_id=artifact.id,
                        session_id=job.session_id,
                        speaker=artifact.metadata["speaker"],
                        model=artifact.metadata["model"],
                        embedding=vector,
                    )
                    for artifact, (_, vector) in zip(created, embeddings, strict=True)
                ]
            )
            await pipe.artifacts.create(
                ArtifactCreate(
                    pipeline=self.name,
                    kind="diarize-map",
                    session_id=job.session_id,
                    metadata=map_metadata,
                )
            )
        result = {
            "blocks": blocks,
            "turns": len(turns),
            "utterances": len(utterances),
            "speakers": len(embeddings),
        }
        if skipped:
            result["skipped"] = skipped
        return result
