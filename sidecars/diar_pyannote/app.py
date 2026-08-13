"""Speaker-diarization server for pyannote/speaker-diarization-community-1.

The owned serving layer for the audio-pipeline's diarize tier: pyannote does
the inference, this file speaks a small JSON contract so the tier (or any
HTTP client) can be pointed at a different diarizer later without code
changes.

    POST /v1/audio/diarizations   multipart file ->
        {"turns": [{"start_ms", "end_ms", "speaker",
                    "overlap_ms": 120,        # time shared with other speakers
                    "clean_ms": 2400,         # non-overlapped time
                    "embedding": [floats] | null}, ...],
         "embeddings": {"SPEAKER_00": [floats], ...},
         "model": "..."}
    GET  /v1/models               the loaded model id
    GET  /health                  503 while the model is still loading

Two kinds of embeddings, deliberately:

- ``embeddings`` (per speaker label) are the pipeline's own aggregates —
  averaged over every frame pyannote assigned to that label, overlap
  included. Kept for backward compatibility and as the caller's fallback
  when a label has no usable per-turn vectors.
- Each turn's ``embedding`` is computed here from that turn's audio alone,
  with overlapping-speech regions excised first (pyannote's per-label
  aggregate cannot reveal when a label wrongly contains several people —
  per-turn vectors let the caller audit label purity and exclude crosstalk
  from voice-prints). Turns with under ``EMBED_MIN_CLEAN_MS`` (env, default
  1000) of clean audio embed unreliably and get ``null``.

``overlap_ms``/``clean_ms`` are pure diarization arithmetic and are always
present; ``embedding`` requires the pipeline's bundled embedding model
(feature-detected — absent support degrades to ``null``). Global clustering
across blocks/sessions is deliberately NOT done here.

The model loads in a background thread so the port binds immediately and the
compose healthcheck gates readiness. Inference is serialized with a lock: one
GPU, one request at a time.
"""

import asyncio
import logging
import math
import os
import tempfile
import threading

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

MODEL_ID = os.environ.get("DIAR_MODEL", "pyannote/speaker-diarization-community-1")
# Turns with less clean (non-overlapped) audio than this get no per-turn
# embedding: the vectors degrade sharply on very short crops.
EMBED_MIN_CLEAN_MS = int(os.environ.get("EMBED_MIN_CLEAN_MS", "1000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diar-pyannote")

app = FastAPI(title="diar-pyannote")

_pipeline = None
_load_error: str | None = None
_gpu_lock = threading.Lock()


def _load_model() -> None:
    global _pipeline, _load_error
    try:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is not set (pyannote weights are gated)")
        logger.info("loading %s ...", MODEL_ID)
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
        if pipeline is None:
            raise RuntimeError(
                "Pipeline.from_pretrained returned None — usually the model "
                "conditions were not accepted on Hugging Face"
            )
        pipeline.to(torch.device("cuda"))
        _pipeline = pipeline
        logger.info(
            "model ready (pipeline=%s, per_turn_embeddings=%s)",
            type(pipeline).__name__,
            getattr(pipeline, "_embedding", None) is not None
            and getattr(pipeline, "_audio", None) is not None,
        )
    except Exception as exc:  # surfaced via /health; the container stays up
        _load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("model load failed")


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_model, daemon=True).start()


@app.get("/health")
def health() -> JSONResponse:
    if _pipeline is not None:
        # per_turn_embeddings=False means the purity audit upstream is
        # silently disabled — surfaced here so it is observable.
        return JSONResponse(
            {
                "status": "ok",
                "model": MODEL_ID,
                "pipeline_class": type(_pipeline).__name__,
                "per_turn_embeddings": (
                    getattr(_pipeline, "_embedding", None) is not None
                    and getattr(_pipeline, "_audio", None) is not None
                ),
            }
        )
    body = {"status": "loading" if _load_error is None else "failed"}
    if _load_error is not None:
        body["error"] = _load_error
    return JSONResponse(body, status_code=503)


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


def _turn_embedding(path: str, clean_parts, duration_s: float):
    """Embed one turn's clean (non-overlapped) audio; None when unusable.

    Crops are clamped to the file duration — pyannote turn boundaries can
    overrun the audio by a frame, and Audio.crop raises past the end.
    """
    import numpy
    import torch
    from pyannote.core import Segment

    embedding_model = getattr(_pipeline, "_embedding", None)
    audio = getattr(_pipeline, "_audio", None)
    if embedding_model is None or audio is None:
        return None
    waveforms = []
    for part in clean_parts:
        clamped = Segment(max(part.start, 0.0), min(part.end, duration_s))
        if clamped.duration <= 0:
            continue
        waveforms.append(audio.crop(path, clamped)[0])
    if not waveforms:
        return None
    vector = embedding_model(torch.cat(waveforms, dim=1).unsqueeze(0))[0]
    if not bool(numpy.isfinite(vector).all()):
        return None
    return [float(value) for value in vector]


def _diarize_file(path: str) -> dict:
    from pyannote.core import Timeline

    with _gpu_lock:
        # pyannote.audio 4.x always computes per-speaker embeddings.
        output = _pipeline(path)
        diarization = output.speaker_diarization
        embeddings = output.speaker_embeddings

        overlap = diarization.get_overlap()
        audio = getattr(_pipeline, "_audio", None)
        duration_s = audio.get_duration(path) if audio is not None else float("inf")

        turns = []
        for segment, _, label in diarization.itertracks(yield_label=True):
            clean_parts = Timeline([segment]).extrude(overlap)
            clean_ms = int(round(sum(part.duration for part in clean_parts) * 1000))
            turn_ms = int(round(segment.duration * 1000))
            embedding = (
                _turn_embedding(path, clean_parts, duration_s)
                if clean_ms >= EMBED_MIN_CLEAN_MS
                else None
            )
            turns.append(
                {
                    "start_ms": int(segment.start * 1000),
                    "end_ms": int(math.ceil(segment.end * 1000)),
                    "speaker": label,
                    "overlap_ms": max(turn_ms - clean_ms, 0),
                    "clean_ms": clean_ms,
                    "embedding": embedding,
                }
            )

    # Embedding matrix rows follow diarization.labels() order.
    by_speaker = (
        {
            label: [float(value) for value in embeddings[index]]
            for index, label in enumerate(diarization.labels())
        }
        if embeddings is not None
        else {}
    )
    return {"turns": turns, "embeddings": by_speaker, "model": MODEL_ID}


@app.post("/v1/audio/diarizations")
async def diarizations(file: UploadFile) -> dict:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="model is not loaded yet")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio file")

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        path = handle.name
    try:
        return await asyncio.to_thread(_diarize_file, path)
    except Exception as exc:
        logger.exception("diarization failed")
        raise HTTPException(status_code=500, detail=f"diarization failed: {exc}") from exc
    finally:
        os.unlink(path)
