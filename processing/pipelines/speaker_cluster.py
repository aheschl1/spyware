"""Tier: global speaker identity — cluster voice embeddings into speakers.

Consumes each session's ``diarize-map`` (one job per session) and attaches
that session's unassigned embeddings to the user's ``speakers`` clusters:
nearest centroid within ``cluster_assign_max_distance`` wins, otherwise a new
(unlabeled) cluster is born. A merge pass then collapses clusters whose
centroids converged. Everything is scoped to (user, embedding model) —
pgvector distance/avg error on mixed dimensions, and cross-model embeddings
are not comparable anyway.

Drift control: embeddings are processed longest-speech-first so strong
voice-prints seed clusters; centroids are always the mean of *current*
members (never an unbounded running walk); and ``cli speakers recluster``
rebuilds a user from scratch — named clusters survive it as identity anchors,
so drift is never permanent.

Assignments live in ``speaker_embeddings.speaker_id`` and cascade away with
diarize republication; the fresh diarize-map re-triggers clustering. Local
labels on transcripts are never rewritten — global identity stays a
resolve-at-read mapping.
"""

import logging
import math
from collections.abc import Sequence
from typing import Any

from database.pipe import DatabasePipe
from database.repos.pipelines.speaker_cluster import SpeakerClusterQueries
from database.schema.jobs import Job, JobCreate
from database.schema.speakers import ClusterCandidate, Speaker, SpeakerCreate
from processing.base import Pipeline
from processing.config import ProcessingSettings, get_settings

logger = logging.getLogger(__name__)

_SOURCE_PIPELINE = "diarize"


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - cosine similarity; 1.0 for degenerate (zero or mismatched) input."""
    if len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0.0:
        return 1.0
    return 1.0 - dot / norm


def pick_survivor(a: Speaker, b: Speaker, members: dict) -> tuple[Speaker, Speaker]:
    """(survivor, loser): named beats unnamed, then more members, then older.

    The survivor keeps its id and name — merging must never delete a
    user-visible labeled identity in favour of an anonymous one.
    """
    def rank(speaker: Speaker) -> tuple:
        return (
            speaker.name is not None,
            members.get(speaker.id, 0),
            -speaker.created_at.timestamp(),
        )

    survivor = max((a, b), key=rank)
    return (survivor, a if survivor is b else b)


async def cluster_batch(
    pipe: DatabasePipe,
    user_id: Any,
    candidates: Sequence[ClusterCandidate],
    settings: ProcessingSettings,
) -> dict[str, int]:
    """Assign a batch of unassigned embeddings, then merge and prune.

    Shared by the per-session pipeline job and the CLI full rebuild; the
    caller supplies candidates already ordered longest-speech-first. Runs
    entirely on the caller's transaction.
    """
    counts = {"assigned": 0, "created": 0, "merged": 0, "skipped_short": 0}
    touched: set = set()
    models: set[str] = set()
    for candidate in candidates:
        if (
            candidate.talk_ms is not None
            and candidate.talk_ms < settings.cluster_min_talk_ms
        ):
            counts["skipped_short"] += 1
            continue
        models.add(candidate.model)
        nearest = await pipe.speakers.nearest(
            user_id, candidate.model, candidate.embedding
        )
        if nearest is not None and nearest[1] <= settings.cluster_assign_max_distance:
            target = nearest[0]
            newly_created = False
        else:
            target = await pipe.speakers.create(
                SpeakerCreate(
                    user_id=user_id, model=candidate.model, centroid=candidate.embedding
                )
            )
            newly_created = True
        # Rowcount-guarded: diarize may republish mid-job and delete the
        # embedding; a stillborn new cluster is pruned below.
        if await pipe.speakers.assign_if_unassigned(candidate.artifact_id, target.id):
            counts["created" if newly_created else "assigned"] += 1
            touched.add(target.id)

    for speaker_id in touched:
        await pipe.speakers.recompute_centroid(speaker_id)

    for model in sorted(models):
        counts["merged"] += await _merge_pass(pipe, user_id, model, settings)
        await pipe.speakers.delete_empty_unlabeled(user_id, model)
    return counts


async def _merge_pass(
    pipe: DatabasePipe, user_id: Any, model: str, settings: ProcessingSettings
) -> int:
    """Collapse converged clusters; one merge per iteration, so it terminates.

    Pairs of two *differently named* clusters are never auto-merged — the
    user said they are different people; skipping them (rather than stopping
    at them) keeps the loop from spinning on an unmergeable closest pair.
    """
    merged = 0
    while True:
        clusters = await pipe.speakers.list_for_user_model(user_id, model)
        members = await pipe.speakers.member_counts(user_id, model)
        best: tuple[float, Speaker, Speaker] | None = None
        for i, a in enumerate(clusters):
            for b in clusters[i + 1 :]:
                if a.name is not None and b.name is not None and a.name != b.name:
                    continue
                distance = cosine_distance(a.centroid, b.centroid)
                if distance <= settings.cluster_merge_max_distance and (
                    best is None or distance < best[0]
                ):
                    best = (distance, a, b)
        if best is None:
            return merged
        survivor, loser = pick_survivor(best[1], best[2], members)
        logger.info(
            "merging speaker %s into %s (distance %.3f)", loser.id, survivor.id, best[0]
        )
        await pipe.speakers.merge(loser.id, survivor.id)
        merged += 1


class SpeakerClusterPipeline(Pipeline):
    name = "speaker-cluster"

    async def setup(self) -> None:
        self._settings = get_settings()

    async def discover(self, limit: int) -> Sequence[JobCreate]:
        async with DatabasePipe() as pipe:
            maps = await SpeakerClusterQueries(pipe.connection).maps_without_jobs(
                self.name, _SOURCE_PIPELINE, limit
            )
        return tuple(
            JobCreate(
                pipeline=self.name,
                session_id=diarize_map.session_id,
                artifact_id=diarize_map.id,
                dedup_key=f"{self.name}:artifact:{diarize_map.id}",
            )
            for diarize_map in maps
        )

    async def process(self, job: Job) -> dict[str, Any]:
        assert job.session_id is not None
        if job.artifact_id is None:
            # Diarize republished; the new map has its own job.
            return {"skipped": "diarize-map republished"}
        async with DatabasePipe() as pipe:
            session = await pipe.sessions.get(job.session_id)
            if session is None:
                return {"skipped": "session vanished"}
            candidates = await pipe.speakers.unassigned_for_session(job.session_id)
            return await cluster_batch(
                pipe, session.user_id, candidates, self._settings
            )
