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
  session_id, bucket, object_key, links, metadata)`. `kind`, `links`, and
  `metadata` mean whatever the pipeline says they mean; `object_key` is NULL
  for artifacts that are pure rows. Consumers locate upstream outputs with
  `pipe.artifacts.find(pipeline, kind, session_id)` (newest wins) - this is
  how post-processing consumes transcription without touching segments.
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
