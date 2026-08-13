"""Audio tagging + embedding server: CED (AudioSet 527) and LAION-CLAP.

The owned serving layer for the audio-pipeline's audio-tag tier. One container
serves both models because they consume the same audio windows: CED scores 527
sound-event classes per window, CLAP embeds each window into its joint
audio-text space (the retrieval index for text->audio search).

    POST /v1/audio/analyze        multipart WAV ->
        {"windows": [{"start_ms", "end_ms",
                      "labels": [{"label", "score"}, ...],   # top-K sigmoid
                      "embedding": [floats]},                # L2-normalized CLAP
                     ...],
         "window_ms": ..., "hop_ms": ...,
         "models": {"tagger": "...", "clap": "..."}}
    POST /v1/text/embeddings      {"texts": [...]} ->
        {"embeddings": [[floats], ...], "model": "..."}      # CLAP text side
    GET  /v1/models               both loaded model ids
    GET  /health                  503 while the models are still loading

Windowing (10s/5s hop by default) happens here, not in the caller: the tier
uploads ~2-minute clips and gets per-window rows back, so a long session is a
handful of requests instead of hundreds. Window times are relative to the
uploaded clip; the caller adds its span offset.

The models load in a background thread so the port binds immediately and the
compose healthcheck gates readiness. Inference is serialized with a lock: one
GPU, one request at a time; throughput comes from batching windows inside a
request.
"""

import asyncio
import io
import logging
import math
import os
import threading

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

TAGGER_ID = os.environ.get("TAGGER_MODEL", "mispeech/ced-base")
CLAP_ID = os.environ.get("CLAP_MODEL", "laion/larger_clap_general")
WINDOW_MS = int(os.environ.get("WINDOW_MS", "10000"))
HOP_MS = int(os.environ.get("HOP_MS", "5000"))
BATCH = int(os.environ.get("BATCH", "16"))
TOP_K = int(os.environ.get("TOP_K", "20"))
# Windows shorter than this are dropped (except a sole window): a 1s tail is
# mostly the previous window again and scores unreliably.
MIN_WINDOW_MS = int(os.environ.get("MIN_WINDOW_MS", "2000"))

TAGGER_RATE = 16_000  # CED's training rate
CLAP_RATE = 48_000  # CLAP's training rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("audio-tagger")

app = FastAPI(title="audio-tagger")

_models = None  # (tagger_fe, tagger, id2label, clap_processor, clap)
_load_error: str | None = None
_gpu_lock = threading.Lock()


def _load_models() -> None:
    global _models, _load_error
    try:
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForAudioClassification,
            ClapModel,
            ClapProcessor,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 halves VRAM and is accurate enough for sigmoid tag scores; CPU
        # fallback stays fp32 (half-precision CPU inference is slow in torch).
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("loading %s ...", TAGGER_ID)
        tagger_fe = AutoFeatureExtractor.from_pretrained(TAGGER_ID, trust_remote_code=True)
        tagger = (
            AutoModelForAudioClassification.from_pretrained(
                TAGGER_ID, trust_remote_code=True, torch_dtype=dtype
            )
            .to(device)
            .eval()
        )
        logger.info("loading %s ...", CLAP_ID)
        clap_processor = ClapProcessor.from_pretrained(CLAP_ID)
        clap = ClapModel.from_pretrained(CLAP_ID, torch_dtype=dtype).to(device).eval()

        _models = (tagger_fe, tagger, dict(tagger.config.id2label), clap_processor, clap)
        logger.info("models ready on %s (%s)", device, dtype)
    except Exception as exc:  # surfaced via /health; the container stays up
        _load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("model load failed")


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_models, daemon=True).start()


@app.get("/health")
def health() -> JSONResponse:
    if _models is not None:
        return JSONResponse({"status": "ok", "tagger": TAGGER_ID, "clap": CLAP_ID})
    body = {"status": "loading" if _load_error is None else "failed"}
    if _load_error is not None:
        body["error"] = _load_error
    return JSONResponse(body, status_code=503)


@app.get("/v1/models")
def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": TAGGER_ID, "object": "model"}, {"id": CLAP_ID, "object": "model"}],
    }


def _decode(data: bytes) -> tuple[np.ndarray, int]:
    """Any soundfile-readable audio -> (mono float32, rate)."""
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(rate)


def _resample(audio: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target:
        return audio
    import soxr

    return soxr.resample(audio, rate, target)


def _window_bounds(total_ms: int) -> list[tuple[int, int]]:
    """[start, end) clip-relative window times at HOP_MS steps."""
    if total_ms <= 0:
        return []
    bounds = []
    for start in range(0, total_ms, HOP_MS):
        end = min(start + WINDOW_MS, total_ms)
        if end - start < MIN_WINDOW_MS and bounds:
            break
        bounds.append((start, end))
        if end == total_ms:
            break
    return bounds


def _analyze(data: bytes) -> dict:
    import torch

    tagger_fe, tagger, id2label, clap_processor, clap = _models
    audio, rate = _decode(data)
    total_ms = int(len(audio) * 1000 / rate)
    bounds = _window_bounds(total_ms)
    if not bounds:
        return {
            "windows": [],
            "window_ms": WINDOW_MS,
            "hop_ms": HOP_MS,
            "models": {"tagger": TAGGER_ID, "clap": CLAP_ID},
        }

    at_tagger = _resample(audio, rate, TAGGER_RATE)
    at_clap = _resample(audio, rate, CLAP_RATE)
    tagger_clips = [
        at_tagger[start * TAGGER_RATE // 1000 : end * TAGGER_RATE // 1000]
        for start, end in bounds
    ]
    clap_clips = [
        at_clap[start * CLAP_RATE // 1000 : end * CLAP_RATE // 1000] for start, end in bounds
    ]

    device = next(tagger.parameters()).device
    windows = [{"start_ms": start, "end_ms": end} for start, end in bounds]
    with _gpu_lock, torch.inference_mode():
        for offset in range(0, len(bounds), BATCH):
            batch = tagger_clips[offset : offset + BATCH]
            features = tagger_fe(batch, sampling_rate=TAGGER_RATE, return_tensors="pt")
            features = {k: v.to(device=device, dtype=tagger.dtype) for k, v in features.items()}
            scores = torch.sigmoid(tagger(**features).logits.float())
            top = scores.topk(min(TOP_K, scores.shape[1]), dim=1)
            for row, values, indices in zip(
                windows[offset : offset + BATCH], top.values, top.indices
            ):
                row["labels"] = [
                    {"label": id2label[index], "score": round(value, 4)}
                    for value, index in zip(values.tolist(), indices.tolist())
                ]

        for offset in range(0, len(bounds), BATCH):
            batch = clap_clips[offset : offset + BATCH]
            inputs = clap_processor(
                audio=batch, sampling_rate=CLAP_RATE, return_tensors="pt", padding=True
            )
            inputs = {
                k: v.to(device=device, dtype=clap.dtype if v.is_floating_point() else None)
                for k, v in inputs.items()
            }
            out = clap.get_audio_features(**inputs)
            # transformers >=5 returns the full output object; the projected,
            # normalized embedding sits in pooler_output. <=4 returns a tensor.
            embeddings = (out if torch.is_tensor(out) else out.pooler_output).float()
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            for row, vector in zip(windows[offset : offset + BATCH], embeddings):
                row["embedding"] = [round(v, 6) for v in vector.tolist()]

    for row in windows:  # a non-finite score would poison downstream pgvector math
        if not all(math.isfinite(v) for v in row["embedding"]):
            raise RuntimeError("non-finite CLAP embedding")
    return {
        "windows": windows,
        "window_ms": WINDOW_MS,
        "hop_ms": HOP_MS,
        "models": {"tagger": TAGGER_ID, "clap": CLAP_ID},
    }


@app.post("/v1/audio/analyze")
async def analyze(file: UploadFile) -> dict:
    if _models is None:
        raise HTTPException(status_code=503, detail="models are not loaded yet")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio file")
    try:
        return await asyncio.to_thread(_analyze, data)
    except Exception as exc:
        logger.exception("analysis failed")
        raise HTTPException(status_code=500, detail=f"analysis failed: {exc}") from exc


class TextRequest(BaseModel):
    texts: list[str]


@app.post("/v1/text/embeddings")
async def text_embeddings(body: TextRequest) -> dict:
    if _models is None:
        raise HTTPException(status_code=503, detail="models are not loaded yet")
    if not body.texts or not all(t.strip() for t in body.texts):
        raise HTTPException(status_code=400, detail="texts must be non-empty strings")

    def _embed() -> list[list[float]]:
        import torch

        _, _, _, clap_processor, clap = _models
        device = next(clap.parameters()).device
        with _gpu_lock, torch.inference_mode():
            inputs = clap_processor(text=body.texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = clap.get_text_features(**inputs)
            embeddings = (out if torch.is_tensor(out) else out.pooler_output).float()
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return [[round(v, 6) for v in row] for row in embeddings.tolist()]

    try:
        return {"embeddings": await asyncio.to_thread(_embed), "model": CLAP_ID}
    except Exception as exc:
        logger.exception("text embedding failed")
        raise HTTPException(status_code=500, detail=f"text embedding failed: {exc}") from exc
