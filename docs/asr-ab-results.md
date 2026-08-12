# ASR A/B experiment — August 2026

**Question:** should production transcription switch from parakeet to
whisper (reported as more robust on noisy audio), and would block-level
transcription with word-timestamp alignment beat per-utterance clips?

**Method:** the transcribe-ab tier (see `docs/processing-pipelines.md`)
generated four candidates per utterance — parakeet-tdt-0.6b-v3 and
faster-whisper large-v3 (int8_float16), each via `chunk` (the utterance
clip alone, as production does) and `block` (whole diarize block, words
assigned to utterances by midpoint). Voting was **blinded** (candidates
shuffled, no model labels until after each vote), one vote per utterance,
judged by ear against the clip in the web UI.

## Result — 107 blinded votes across 8 sessions (2026-08-13)

| | chunk | block | total |
|---|---|---|---|
| **parakeet** | **37** | 26 | **63** |
| **whisper** | 18 | 26 | 44 |
| total | 55 | 52 | 107 |

- **Model:** parakeet 59% (63/107). Not switching.
- **Strategy:** dead even overall (55/52) — but the interaction is the
  finding. Whisper improves with block context (18 → 26), exactly its
  documented short-clip weakness; even boosted it only ties parakeet-block.
  Parakeet is *better* on chunks than blocks (37 vs 26).
- **Production config (parakeet · chunk) is the single best cell.** The
  planned "phase 2" (block-level transcription with word→turn assignment)
  loses its motivation: block mode only helped the model we aren't picking.
- Caveat: when candidates tie on identical text the vote attributes
  arbitrarily, which dilutes real differences — parakeet's lead survives
  that dilution.

## Decisions taken

- Production stays **parakeet-tdt-0.6b-v3, per-utterance clips**.
- Whisper is **disabled in the sidecar** (`ASR_WHISPER_ENABLED=0`, the
  default; weights removed from the HF cache volume) — reclaims ~3 GB VRAM
  and ~3 GB disk. The `?model=whisper` interface and A/B tiers remain in
  code for the next evaluation.
- `transcript-candidate` artifacts and `transcribe-ab` job history were
  purged; the 107 `ab_votes` rows are the permanent record, and voted
  utterances keep their human-chosen transcript (`ab_source` in metadata).

## Re-running the experiment (e.g. a new model)

1. Set `ASR_WHISPER_ENABLED=1` on the `asr-parakeet` service (or point the
   whisper slot at another model via `ASR_WHISPER_MODEL`) and restart —
   weights re-download on boot.
2. Enroll sessions from the web **ab** tab and vote; `GET /v1/ab/results`
   tallies. Regeneration never erases votes.
