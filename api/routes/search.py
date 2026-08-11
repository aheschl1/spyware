"""Audio search, two ways.

- ``/search/audio`` — **contrastive**: the query text is embedded by the
  classifier service's CLAP text encoder (same joint space as the stored
  window embeddings), then pgvector ranks every window the caller owns by
  cosine distance. Open vocabulary; relative distances, no absolute cutoff.
- ``/search/tags`` — **filtering**: no model in the loop; windows are
  matched on the tagger's stored per-class sigmoid scores (exact classes,
  calibrated, thresholdable). ``/search/tags/labels`` lists the classes
  actually heard, for the filter UI.

Together they are the retrieval primitives for question-answering over
recordings; consumers feed the hit windows' transcripts and tags to an LLM.
"""

import math
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException, Query, status

from api.config import get_settings
from api.deps import CurrentUser, Pipe
from api.schema.search import (
    AudioSearchRead,
    AudioSearchResponse,
    TagLabelRead,
    TagLabelsResponse,
    TagSearchRead,
    TagSearchResponse,
)

router = APIRouter(prefix="/search", tags=["search"])

_BAD_GATEWAY = HTTPException(
    status_code=status.HTTP_502_BAD_GATEWAY,
    detail="the audio embedding service is unavailable",
)


async def _embed_query(text: str) -> tuple[list[float], str]:
    """The query in CLAP's text space, via the classifier sidecar."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.classifier_timeout_seconds, connect=5.0)
        ) as client:
            response = await client.post(
                f"{settings.classifier_base_url.rstrip('/')}/text/embeddings",
                json={"texts": [text]},
            )
    except httpx.HTTPError as exc:
        raise _BAD_GATEWAY from exc
    if response.status_code >= 400:
        raise _BAD_GATEWAY
    try:
        body = response.json()
        (vector,) = body["embeddings"]
        model = body["model"]
    except (ValueError, KeyError, TypeError):
        raise _BAD_GATEWAY from None
    if (
        not isinstance(vector, list)
        or not vector
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector)
        or not isinstance(model, str)
    ):
        raise _BAD_GATEWAY
    return vector, model


@router.get("/audio", summary="Find audio windows by describing their sound")
async def search_audio(
    user: CurrentUser,
    pipe: Pipe,
    q: str = Query(min_length=1, max_length=500, description="A sound description, e.g. 'keyboard typing' or 'a dog barking'."),
    session_id: UUID | None = Query(None, description="Restrict matches to one session."),
    limit: int = Query(20, ge=1, le=100),
) -> AudioSearchResponse:
    """Ranked ~10s windows from your sessions whose audio best matches the
    description, nearest first. Distances are only comparable within one
    query; there is no absolute 'good match' cutoff — treat the ranking as
    the signal and let downstream consumers (or the tag scores on each hit)
    decide relevance."""
    vector, model = await _embed_query(q)
    hits = await pipe.audio_embeddings.search(
        vector, user_id=user.id, session_id=session_id, limit=limit
    )
    return AudioSearchResponse(
        query=q, model=model, items=[AudioSearchRead.from_model(hit) for hit in hits]
    )


@router.get("/tags/labels", summary="List the sound classes heard in your audio")
async def list_tag_labels(user: CurrentUser, pipe: Pipe) -> TagLabelsResponse:
    """Every class the tagger stored for your windows, most frequent first —
    the vocabulary the ``/search/tags`` filter can act on."""
    rows = await pipe.tags.labels(user_id=user.id)
    return TagLabelsResponse(labels=[TagLabelRead.from_model(row) for row in rows])


@router.get("/tags", summary="Find audio windows by sound-class score")
async def search_tags(
    user: CurrentUser,
    pipe: Pipe,
    label: str = Query(min_length=1, max_length=200, description="Class to match, case-insensitive substring — 'speech' finds 'Male speech, man speaking'."),
    min_score: float = Query(0.3, ge=0.0, le=1.0, description="Keep windows where the class scored at least this."),
    session_id: UUID | None = Query(None, description="Restrict matches to one session."),
    limit: int = Query(20, ge=1, le=100),
) -> TagSearchResponse:
    """The deterministic search: exact classes with calibrated scores, best
    first. Unlike the contrastive route, scores ARE comparable across
    queries and a threshold means the same thing everywhere."""
    hits = await pipe.tags.search(
        user_id=user.id,
        label=label,
        min_score=min_score,
        session_id=session_id,
        limit=limit,
    )
    return TagSearchResponse(
        label=label,
        min_score=min_score,
        items=[TagSearchRead.from_model(hit) for hit in hits],
    )
