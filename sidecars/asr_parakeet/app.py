"""Transcription server: parakeet-tdt-0.6b-v3 + faster-whisper large-v3.

A thin, owned serving layer speaking the standard transcriptions API. Both
models stay resident; pick one per request with ``?model=parakeet|whisper``
(full HF ids also accepted; default parakeet). Word/segment timestamps come
back as clip-relative ms from either model — parakeet's are native TDT,
whisper's are faster-whisper word timestamps.

    POST /v1/audio/transcriptions?model=...   multipart file -> {"text", "words", ...}
    GET  /v1/models                           the loaded model ids
    GET  /health                              503 until BOTH models are loaded

Models load in a background thread so the port binds immediately and the
compose healthcheck gates readiness. Inference is serialized with one lock:
one GPU, one request at a time.
"""

import asyncio
import logging
import os
import tempfile
import threading

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

PARAKEET_ID = os.environ.get("ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
# The A/B contender slot. Off by default since the 2026-08 experiment
# concluded (parakeet won — audio-pipeline docs/asr-ab-results.md); flip on
# to re-run an evaluation.
WHISPER_ENABLED = os.environ.get("ASR_WHISPER_ENABLED", "0") == "1"
WHISPER_ID = os.environ.get("ASR_WHISPER_MODEL", "large-v3")
WHISPER_COMPUTE = os.environ.get("ASR_WHISPER_COMPUTE", "int8_float16")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("asr")

app = FastAPI(title="asr-parakeet")

_models: dict[str, object] = {}
_load_errors: dict[str, str] = {}
_gpu_lock = threading.Lock()


def _load_models() -> None:
    try:
        logger.info("loading %s ...", PARAKEET_ID)
        from nemo.collections.asr.models import ASRModel

        _models["parakeet"] = ASRModel.from_pretrained(PARAKEET_ID).eval().to("cuda")
        logger.info("parakeet ready")
    except Exception as exc:
        _load_errors["parakeet"] = f"{type(exc).__name__}: {exc}"
        logger.exception("parakeet load failed")
    if not WHISPER_ENABLED:
        return
    try:
        logger.info("loading whisper %s (%s) ...", WHISPER_ID, WHISPER_COMPUTE)
        from faster_whisper import WhisperModel

        _models["whisper"] = WhisperModel(
            WHISPER_ID, device="cuda", compute_type=WHISPER_COMPUTE
        )
        logger.info("whisper ready")
    except Exception as exc:
        _load_errors["whisper"] = f"{type(exc).__name__}: {exc}"
        logger.exception("whisper load failed")


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_models, daemon=True).start()


def _expected() -> tuple[str, ...]:
    return ("parakeet", "whisper") if WHISPER_ENABLED else ("parakeet",)


@app.get("/health")
def health() -> JSONResponse:
    status = {
        key: "ok" if key in _models else _load_errors.get(key, "loading")
        for key in _expected()
    }
    if not WHISPER_ENABLED:
        status["whisper"] = "disabled"
    if all(key in _models for key in _expected()):
        return JSONResponse({"status": "ok", "models": status})
    return JSONResponse({"status": "loading", "models": status}, status_code=503)


@app.get("/v1/models")
def models() -> dict:
    data = [{"id": PARAKEET_ID, "object": "model", "key": "parakeet"}]
    if WHISPER_ENABLED:
        data.append({"id": WHISPER_ID, "object": "model", "key": "whisper"})
    return {"object": "list", "data": data}


def _resolve(model: str | None) -> str:
    if model in (None, "", "parakeet", PARAKEET_ID):
        return "parakeet"
    if model in ("whisper", WHISPER_ID, f"openai/whisper-{WHISPER_ID}"):
        return "whisper"
    raise HTTPException(status_code=422, detail=f"unknown model {model!r}")


def _transcribe_parakeet(path: str) -> dict:
    import torch

    with torch.inference_mode():
        (hyp,) = _models["parakeet"].transcribe([path], timestamps=True, verbose=False)
    stamps = getattr(hyp, "timestamp", None) or {}
    out = {
        "text": (hyp.text or "").strip(),
        "model": PARAKEET_ID,
        "words": [
            {"word": w["word"], "start_ms": int(w["start"] * 1000), "end_ms": int(w["end"] * 1000)}
            for w in stamps.get("word", [])
        ],
        "segments": [
            {"text": s["segment"], "start_ms": int(s["start"] * 1000), "end_ms": int(s["end"] * 1000)}
            for s in stamps.get("segment", [])
        ],
    }
    lang = getattr(hyp, "langs", None) or getattr(hyp, "language", None)
    if isinstance(lang, list) and lang and all(isinstance(item, str) for item in lang):
        lang = max(set(lang), key=lang.count)
    if isinstance(lang, str):
        out["language"] = lang
    return out


def _transcribe_whisper(path: str) -> dict:
    segments, info = _models["whisper"].transcribe(
        path,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    words, segs, texts = [], [], []
    for seg in segments:
        text = seg.text.strip()
        if text:
            texts.append(text)
        segs.append(
            {"text": text, "start_ms": int(seg.start * 1000), "end_ms": int(seg.end * 1000)}
        )
        for w in seg.words or []:
            words.append(
                {"word": w.word.strip(), "start_ms": int(w.start * 1000), "end_ms": int(w.end * 1000)}
            )
    return {
        "text": " ".join(texts),
        "model": WHISPER_ID,
        "language": info.language,
        "words": words,
        "segments": segs,
    }


def _transcribe_file(path: str, key: str) -> dict:
    with _gpu_lock:
        if key == "whisper":
            return _transcribe_whisper(path)
        return _transcribe_parakeet(path)


@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile, model: str | None = None) -> dict:
    key = _resolve(model)
    if key == "whisper" and not WHISPER_ENABLED:
        raise HTTPException(status_code=422, detail="whisper is disabled (ASR_WHISPER_ENABLED=1)")
    if key not in _models:
        raise HTTPException(status_code=503, detail=f"{key} is not loaded yet")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio file")

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        path = handle.name
    try:
        return await asyncio.to_thread(_transcribe_file, path, key)
    except Exception as exc:
        logger.exception("transcription failed")
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    finally:
        os.unlink(path)
