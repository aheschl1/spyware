# Processing pipelines

Background workers that act on recorded audio: transcription, transcript
post-processing, agent triggers - anything derived from a session after (or
beside) capture. This document is the contract; the code lives in
`processing/`.

```
                 ┌───────────────┐
   clients ───▶  │   api (HTTP/  │
                 │   websocket)  │──── writes sessions / segments ───┐
                 └───────────────┘                                   ▼
                                                     ┌────────────────────────┐
                 ┌───────────────┐                   │  Postgres + blob store │
                 │  processing   │  discover/claim   │  processing_jobs       │
                 │  supervisor   │ ◀───────────────▶ │  pipeline_artifacts    │
                 │  ├ session-stats (child process)  │  {pipeline}/... blobs  │
                 │  └ stats-echo    (child process)  └────────────────────────┘
                 └───────────────┘
```

The API and the workers share nothing but the stores. There is **zero
enqueue wiring in the API**: workers find their own work by querying the same
database the API writes.

Run them with `make worker` (`uv run python -m processing`), or a subset with
`--only <name>` (repeatable). `audio-pipeline-worker` is the script entry.

## Topology

- One **supervisor** process (`processing/supervisor.py`) spawns one
  long-lived **child process per registered pipeline** and restarts children
  that die (backoff 1 s doubling to 30 s, reset after 60 s of clean running).
- Each child runs one `Pipeline` instance in its own asyncio loop
  (`processing/worker.py`): discover → drain → wait, forever.
- **Invariant: exactly one worker process per pipeline.** Orphan recovery
  (below) depends on it. Scaling one pipeline to N workers first requires
  claim leases/heartbeats - do not just start the supervisor twice.

## Jobs: lifecycle and the queue

Work is materialized in `processing_jobs` so that retry, backoff,
dead-lettering, priority, and chaining all have one home.

```
queued ──claim──▶ running ──▶ succeeded            (terminal)
   ▲                 │
   │                 ├──▶ queued again (retry: error recorded,
   └─────────────────┘        run_at pushed by base·2^(attempt-1), capped)
                     └──▶ dead   (attempts == max_attempts; terminal)
```

- `claim` takes the next `queued` row with `run_at <= now()`, ordered by
  `priority DESC, run_at, id`, under `FOR UPDATE SKIP LOCKED`; it bumps
  `attempts` at claim time so crashes still count against `max_attempts`.
- The claim transaction never spans `process()`: claiming commits first, the
  outcome lands in a second transaction.
- **Orphan recovery**: at boot a worker requeues every `running` row of its
  pipeline - under the one-worker invariant those can only belong to a dead
  predecessor.
- `pg_notify('processing_jobs', <pipeline>)` fires on every committed
  enqueue. NOTIFY is a latency optimization only - polling (the
  `PROCESSING_POLL_INTERVAL_SECONDS` pass cadence) is the source of truth, so
  jobs enqueued while a worker is down are simply picked up at the next boot
  or pass.

## Self-discovery

Pipelines are responsible for their own querying. Each pipeline implements
`discover(limit) -> Sequence[JobCreate]` - a query it owns, living in
`database/repos/pipelines/<pipeline>.py` (one module per pipeline; classes
extend `BaseRepo`; constructed by the pipeline itself, never hung off
`DatabasePipe`).

Idempotency is the **dedup key**: every discovered item carries one (e.g.
`session-stats:session:{id}`), and the partial unique index on
`(pipeline, dedup_key)` spans *all* statuses - once a key's job exists
(queued, running, succeeded, or dead), re-enqueueing it is a no-op forever.
The discovery query should also exclude already-enqueued work so repeated
passes stay cheap — anti-join on indexed columns (`NOT EXISTS` over
`(pipeline, session_id)`, both sides indexed), never by reconstructing the
dedup key per row, which no index can serve; the unique index is the
correctness backstop either way. Discovered batches are enqueued in one
statement (`pipe.jobs.enqueue_many`), not a transaction per item.

Pipelines fed only by chaining (or manual enqueue) return `()`.

## Blob space and artifacts

- Each pipeline owns the blob prefix **`{name}/`** at the bucket root -
  `storage.keys.pipeline_key(name, session_id, filename)` yields
  `{name}/sessions/{session_id}/{filename}`. (`users/` is reserved by segment
  storage; the registry rejects that name.)
- Outputs are recorded in **`pipeline_artifacts`**: `(pipeline, kind,
  session_id, start_ms, end_ms, bucket, object_key, links, metadata)`.
  `kind`, `links`, and `metadata` mean whatever the pipeline says they mean;
  `object_key` is NULL for artifacts that are pure rows. Consumers locate
  upstream outputs with `pipe.artifacts.find(pipeline, kind, session_id)`
  (newest wins) or `find_overlapping(session_id, from_ms, to_ms, ...)`.
- **Artifacts address session time, not segments.** `start_ms`/`end_ms` place
  an artifact on the session's timeline (ms from session start; both NULL =
  the whole session). `services/timeline.py` is the bridge: it maps the
  uniform-WAV stitched stream to that timeline, renders any `[start_ms,
  end_ms)` as a standalone clip, and walks whole sessions in windows. The
  segment is an implementation detail of capture — labels, transcripts, and
  embeddings all attach to *chunks of a session*. The API surfaces them at
  `GET /v1/sessions/{id}/artifacts` (filters: pipeline, kind, time window).
- **A job can be scoped to the artifact it consumes** (`processing_jobs.
  artifact_id`, FK `ON DELETE SET NULL`). That is how tier N+1 discovers tier
  N's output with an indexed anti-join — see the transcription tiers below.
- Deleting a session cascades its jobs and artifact rows, but blobs under
  `{name}/sessions/{id}/` are currently orphaned - a future cleanup pass owns
  that.

## Writing a pipeline

1. Subclass `processing.base.Pipeline`; set `name` (unique; also the blob
   prefix and the `processing_jobs.pipeline` value).
2. Implement `discover(limit)` with its queries in
   `database/repos/pipelines/<name>.py`, stamping dedup keys.
3. Implement `process(job) -> dict` - query whatever you need (many
   segments, other pipelines' artifacts, nothing at all), write blobs under
   your prefix, record artifacts. The returned dict is stored as
   `jobs.result`. Raise to fail: the worker retries with backoff, then marks
   the job dead.
4. Heavy resources (models, clients) load in `setup()`, once per child
   process - **never at module import**. Module tops import only stdlib,
   `processing.base`, `database.*`, `storage.*`.
5. Register the class in `processing/registry.py:PIPELINES`, and wire a
   callback there if something should follow success.

`session-stats` and `stats-echo` (`processing/pipelines/`) are the living
templates: the first discovers ended sessions, aggregates every segment, and
writes a blob + artifact; the second is chained-only and consumes the
artifact - the transcription → post-processing shape without the
transcription.

## The transcription tiers

Real tiered processing, consuming through artifacts (no chaining callbacks —
each tier discovers its own input):

1. **`speech-detect`** (`processing/pipelines/speech_detect.py`) — discovers
   ended sessions (shared query: `database/repos/pipelines/common.py`), runs
   the session's PCM through a VAD backend (`processing/vad.py`: `silero` via
   pysilero-vad on CPU, or the deterministic `energy` threshold the tests
   use), and records one `speech-span` artifact per merged/padded span plus a
   `speech-map` summary. The threshold is deliberately low (0.15): spans are
   a **coarse, high-recall activity gate** that decides what the diarizer
   sees — they do not gate ASR. (Measured: at 0.5 silero covered 4% of a far
   speaker's talk on a glasses mic; pyannote's own segmentation covers it
   all, so precision here would silence one side of a conversation.)
   Unprocessable sessions (non-WAV, non-16k-mono, no audio) get an empty map
   with a `skipped` reason — never a dead job. Republication replaces the
   previous span set in one transaction.
2. **`diarize`** (`processing/pipelines/diarize.py`) — consumes each
   session's `speech-map` (one job per session), but the diarizer never sees
   a whole session: spans re-merge into *blocks* (contiguous speech, gap ≤
   30 s joins, ≤ 30 min, closed at span boundaries) because label consistency
   needs long context. Emits `speaker-turn` artifacts with
   **block-namespaced** labels (`b{start}:SPEAKER_00` — local identity only;
   global identity is the future clustering tier's job), **`utterance`**
   artifacts — same-speaker turns merged when the gap is ≤ 1.5 s, capped at
   the ASR input window (30 s), the units the transcribe tier consumes — and
   one `speaker-embedding` artifact per (block, speaker) whose vector lives
   in the `speaker_embeddings` table (pgvector, cascade-deleted with its
   artifact row), queryable with distance operators for the clustering tier.
   Publication is atomic: delete-previous + insert-all + `diarize-map` in one
   transaction — vectors included — **and it deletes the session's
   transcripts too**: they derive from utterances that no longer exist, and
   this tier owns invalidating them. The map's presence is the completion
   marker and retries are idempotent. A malformed embedding from the service
   is dropped with a warning (turns are the load-bearing output). Service
   seam: `processing/diarizer.py` → the diar_pyannote container
   (`PROCESSING_DIARIZER_BASE_URL`), pyannote/speaker-diarization-3.1.
3. **`transcribe`** (`processing/pipelines/transcribe.py`) — discovers
   `utterance` artifacts with no transcribe job (anti-join on
   `processing_jobs.artifact_id`), renders each from the timeline, sends it
   to the transcription service, and records a `transcript` artifact on the
   same range — full text, speaker label, and model in metadata; **no blob**
   (utterances are short, the row is the store). When diarize republishes,
   utterances get new ids, so the anti-join re-enqueues them and queued jobs
   whose utterance vanished skip themselves. Transcripts therefore wait on
   the whole session's diarization — the cost of gating ASR on the detector
   that actually hears every speaker.
4. **`speaker-cluster`** (`processing/pipelines/speaker_cluster.py`) —
   discovered per `diarize-map`, but every run **re-clusters the user's whole
   corpus** with constrained agglomerative clustering (average linkage,
   cosine, one threshold: `cluster_distance`, per-user overridable via the
   `cluster_params` table / `POST /v1/speakers/cluster-params`). Batch
   re-clustering is order-independent and self-healing — no centroid drift,
   no permanent splits. User curation persists as **pins** (`speaker_pins`:
   "this voice-print IS this identity", created by merges and member moves):
   pin-groups enter the agglomeration pre-merged, and clusters pinned to
   different identities never merge. After each run, result clusters map
   back to persistent `speakers` rows — pinned identity first, else majority
   previous membership preferring named clusters (named identities keep
   their id), else unnamed ids churn. Assignments live in
   `speaker_embeddings.speaker_id` — a resolve-at-read mapping, so
   re-clustering never rewrites transcripts; local labels stay provenance.
   Embeddings under `cluster_min_talk_ms` of speech are skipped as
   unreliable, unless pinned. The same batch runs on demand via
   `POST /v1/speakers/recluster`, `cli speakers recluster` (one-off
   `--distance`/`--min-talk-ms` flags) and the web UI's clustering-settings
   panel; all three serialize per user on an advisory lock. Manual tools:
   `POST /v1/speakers/{id}/merge` (pins both member sets to the survivor;
   `GET /v1/speakers/{id}/similar` ranks candidates by centroid distance),
   `GET /v1/speakers/{id}/members` (voice-prints farthest-first with
   playable utterance spans), and per-member reassign/eject/unpin routes.

The transcription service is behind a seam (`processing/transcriber.py`):
`PROCESSING_TRANSCRIBER_BASE_URL` speaking the standard transcriptions API
(`openai` protocol — our `asr_canary` container serving nvidia/canary-qwen-2.5b,
or speaches, or a hosted endpoint) or a Replicate cog wrapper (`cog`).
Swapping models/backends is env-only. Canary-qwen is English-only and emits
no word timestamps — timing granularity is the utterance; a forced-alignment
tier can refine it later on the same artifact model.

## The audio-tag tier

`audio-tag` (`processing/pipelines/audio_tag.py`) runs beside the speech
tiers, not downstream of them: it discovers ended sessions directly (same
trigger as `speech-detect`) because non-speech sound — traffic, music, a
keyboard — is exactly what it exists to hear.

1. **audio-tag** — walks the session's audio in ~2-minute rendered spans
   (overlapping by window-minus-hop so the service's 10 s / 5 s-hop window
   grid stays continuous across seams) and sends each span to the
   classification service behind `processing/classifier.py`
   (`PROCESSING_CLASSIFIER_BASE_URL` — our `audio_tagger` container serving
   CED-base for AudioSet's 527 sound classes plus LAION-CLAP for audio
   embeddings; `POST {base}/audio/analyze`). Publishes per ~10 s window one
   `audio-tag` artifact — top tag scores in metadata, ontology ancestors
   suppressed (AudioSet is a DAG: a guitar clip scores Music, Musical
   instrument *and* Guitar; the window keeps the most specific winner) — and
   one `audio_embeddings` pgvector row keyed by that artifact. The
   `audio-tag-map` written last carries the session-level tag list: per-class
   MAX across windows (events are sparse; a mean buries them), gated on
   holding `audio_tag_threshold` for `audio_tag_min_consecutive` windows in a
   row. The ontology parent map is vendored data
   (`processing/data/audioset_parents.json`, generated from
   github.com/audioset/ontology).

The embeddings are the retrieval half of question-answering over recordings:
`GET /v1/search/audio?q=…` embeds the text through the same service's CLAP
text encoder and ranks the caller's windows by cosine distance — consumers
then feed the hit windows' transcripts and tags to an LLM. The vectors are
never fed to a model directly.

Search is a trio: contrastive audio (`/search/audio`, "when did I hear X"),
tag filtering (`/search/tags`, calibrated class scores), and lexical
transcript search (`/search/transcripts`, "when was X said" — Postgres FTS
with a trigram fuzzy fallback; indexes in migration 0009, queries in
`database/repos/transcripts.py`, no pipeline involvement). Semantic
transcript search (sentence embeddings + a `transcript-embed` tier) is
deliberately deferred pending a chunk-overlap design; it would arrive as a
non-breaking `mode=` parameter.

## Callbacks and chaining

`Pipeline.__init__` takes an optional callback; `maybe_callback(job, result)`
invokes it **after the success is committed**. Semantics: at-most-once and
best-effort - a callback exception is logged and swallowed (the job is
already `succeeded`), and a crash between commit and callback drops the
invocation. The registry's `enqueue_follow_up(target)` factory is the
standard chaining callback: it enqueues `target` with the parent's session
linkage, inherited priority, and payload
`{"source_job_id", "source_result"}`, deduped per parent job. If a chain ever
needs to be durable rather than best-effort, enqueue the follow-up inside the
succeed transaction instead - nothing in the schema prevents that.

## Operations

- `make worker`, or `uv run python -m processing --only session-stats`.
- `PROCESSING_*` environment (see `.env.example`): poll interval, discovery
  batch, retry policy, shutdown grace, restart backoff.
- SIGTERM/SIGINT: children finish the in-flight job, the supervisor waits
  `PROCESSING_SHUTDOWN_GRACE_SECONDS`, then kills stragglers. A SIGKILLed
  child leaves a `running` row; the next boot requeues it.
- Priority: plain integer, higher first, default 0. Nothing assigns
  priorities yet; chained jobs inherit their parent's. Constant load at a
  high priority will starve lower ones - there is no aging.

## Future: live pipelines

Live processing (acting on a session *while* it streams) is a different kind,
deliberately not built yet. The decided direction:

- A **live pipeline** runs a *follower per live session*: its worker watches
  for open sessions and runs one follower loop each, consuming new data at
  its own pace via its own queries/cursor until the session ends, then
  finalizing. Priority is inherent - live pipelines are their own processes
  and never contend with the batch queue.
- A **UDS control plane** (one Unix domain socket per live worker) will carry
  API ⇄ worker communication, e.g. pushing live results to the session's
  websocket clients via the protocol's reserved `Hello.effects` /
  `Welcome.effects` negotiation (docs/streaming-protocol.md).
- A live pipeline may share its core code with a paired background pipeline
  (live transcription and batch transcription wrapping one transcriber).

Nothing in the current design blocks this: the jobs/artifacts tables and blob
spaces are shared infrastructure, live followers simply bypass the queue, and
the callback seam stays the background side's extension point.
