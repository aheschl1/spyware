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
stores as blobs.

Publication is atomic: previous diarize output for the session is deleted and
the full new set (turns, embeddings, the ``diarize-map`` summary) inserted in
one transaction. The map's presence is the completion marker consumers wait
for, so partial output is never visible and retries are idempotent.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from database.pipe import DatabasePipe
from database.repos.pipelines.diarize import DiarizeQueries
from database.schema.artifacts import ArtifactCreate, PipelineArtifact
from database.schema.jobs import Job, JobCreate
from processing.base import Pipeline
from processing.config import get_settings
from processing.diarizer import Diarizer
from services import stitch, timeline
from storage.keys import pipeline_key
from storage.pipe import BlobPipe

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
            return await self._publish(job, turns=[], embeddings=[], blocks=0)

        blocks = blocks_from_spans(
            [(span.start_ms, span.end_ms) for span in spans],
            merge_gap_ms=settings.diarize_block_merge_gap_ms,
            max_block_ms=settings.diarize_max_block_ms,
        )
        try:
            line = await timeline.load_timeline(job.session_id)
            if line is None:
                return await self._publish(
                    job, turns=[], embeddings=[], blocks=0, skipped="session audio is gone"
                )
            # DiarizerError raises through: the worker retries with backoff.
            turn_rows, embedding_rows = [], []
            for block in blocks:
                clip = await timeline.render_range(line, block.start_ms, block.end_ms)
                result = await self._diarizer.diarize(
                    clip, filename=f"{job.session_id}-{block.start_ms}.wav"
                )
                turn_rows += self._turn_artifacts(job, block, result)
                embedding_rows += await self._embedding_artifacts(job, block, result)
        except (stitch.NotStitchable, timeline.NotRenderable) as exc:
            # Retrying cannot make the audio renderable.
            return await self._publish(
                job, turns=[], embeddings=[], blocks=0, skipped=str(exc)
            )

        return await self._publish(
            job, turns=turn_rows, embeddings=embedding_rows, blocks=len(blocks)
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

    async def _embedding_artifacts(
        self, job: Job, block: Block, result: Any
    ) -> list[ArtifactCreate]:
        """One artifact + blob per (block, local speaker) embedding.

        The vector lives in the blob (a retry overwrites the same key, so the
        delete-then-recreate publication stays consistent); the row carries
        only the addressing.
        """
        rows = []
        async with BlobPipe() as blobs:
            for speaker, vector in sorted(result.embeddings.items()):
                namespaced = f"b{block.start_ms}:{speaker}"
                key = pipeline_key(
                    self.name,
                    job.session_id,
                    f"embeddings/{block.start_ms:09d}-{speaker}.json",
                )
                info = await blobs.put(
                    key,
                    json.dumps(
                        {
                            "speaker": namespaced,
                            "embedding": vector,
                            "dim": len(vector),
                            "model": result.model,
                        }
                    ).encode(),
                    "application/json",
                )
                rows.append(
                    ArtifactCreate(
                        pipeline=self.name,
                        kind="speaker-embedding",
                        session_id=job.session_id,
                        start_ms=block.start_ms,
                        end_ms=block.end_ms,
                        bucket=info.bucket,
                        object_key=info.key,
                        metadata={
                            "speaker": namespaced,
                            "dim": len(vector),
                            "model": result.model,
                        },
                    )
                )
        return rows

    async def _publish(
        self,
        job: Job,
        *,
        turns: list[ArtifactCreate],
        embeddings: list[ArtifactCreate],
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
            "speakers": len(embeddings),
            "params": {
                "block_merge_gap_ms": settings.diarize_block_merge_gap_ms,
                "max_block_ms": settings.diarize_max_block_ms,
                "min_turn_ms": settings.diarize_min_turn_ms,
            },
        }
        if skipped:
            map_metadata["skipped"] = skipped
        async with DatabasePipe() as pipe:
            await pipe.artifacts.delete_for_pipeline(job.session_id, self.name)
            await pipe.artifacts.create_many(turns + embeddings)
            await pipe.artifacts.create(
                ArtifactCreate(
                    pipeline=self.name,
                    kind="diarize-map",
                    session_id=job.session_id,
                    metadata=map_metadata,
                )
            )
        result = {"blocks": blocks, "turns": len(turns), "speakers": len(embeddings)}
        if skipped:
            result["skipped"] = skipped
        return result
