# diar_pyannote

**pyannote/speaker-diarization-community-1** (best open-source DER, unlimited
speakers) behind a small JSON contract, plus two kinds of embeddings so
downstream clustering never re-runs diarization:

- per-speaker aggregates (pyannote's own, averaged over every frame it gave
  that label — overlap included);
- **per-turn embeddings**, computed here from each turn's clean
  (non-overlapped) audio via the pipeline's bundled embedding model. These
  let the caller audit a label's purity — pyannote sometimes puts several
  people under one label, and the blended aggregate can't reveal that — and
  build voice-prints free of crosstalk. Turns also carry `overlap_ms` /
  `clean_ms` (time shared with / free of other speakers).

Turns with under `EMBED_MIN_CLEAN_MS` (env, default 1000) of clean audio get
`"embedding": null` — short crops embed unreliably. If the loaded pipeline
exposes no embedding model, per-turn `embedding` degrades to `null` and the
caller behaves as before per-turn support existed.

Consumed by the audio-pipeline's `diarize` tier
(`PROCESSING_DIARIZER_BASE_URL=http://127.0.0.1:8034/v1`).

## Setup

1. Hugging Face account: accept the conditions on
   `pyannote/speaker-diarization-community-1`.
2. Set `HF_TOKEN=hf_...` (a read token) in `deploy/.env`.

```bash
make sidecars                     # or, just this one:
docker compose -f deploy/docker-compose.yml up -d --build diar-pyannote

curl -s http://127.0.0.1:8034/health
curl -s -F file=@meeting.wav http://127.0.0.1:8034/v1/audio/diarizations | python3 -m json.tool
```

- `DIAR_MODEL` (default `pyannote/speaker-diarization-community-1`)
- Weights cache in the `diar-pyannote-hf-cache` volume. VRAM: pyannote 4.x
  peaks ~10 GB during reconstruction on a full 30-min block (same for 3.1
  and community-1 — upstream issue #1963); lower `diarize_max_block_ms` in
  the worker if the GPU gets tighter.
- One request at a time (single GPU); the pipeline sends one clip per speech
  *block* (contiguous speech, ≤30 min) — never a whole session.
- Speaker labels are local to one request; identity across blocks/sessions is
  the (future) clustering tier's job, fed by the returned embeddings.
