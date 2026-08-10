# audio-pipeline

## Database layer

Postgres accessed through raw SQL repositories behind a single entry point:

```python
from database import DatabasePipe
from database.schema.users import UserCreate

async with DatabasePipe() as pipe:
    user = await pipe.users.create(UserCreate(email="me@example.com", password="s3cret"))
    issued = await pipe.tokens.issue(user.id, name="laptop", ttl=timedelta(days=30))
    print(issued.token.get_secret_value())  # shown once; only the hash is stored
```

Everything inside one `async with` block runs on one connection in one transaction:
it commits on a clean exit and rolls back if an exception escapes. Repositories return
Pydantic models from `database/schema/`.

| Path | Purpose |
| --- | --- |
| `database/pipe.py` | `DatabasePipe`, connection pool (`get_pool` / `close_pool`) |
| `database/repos/` | Raw-SQL repositories (`pipe.users`, `pipe.tokens`, `pipe.sessions`, `pipe.segments`) |
| `database/schema/` | Pydantic models returned by the repositories |
| `database/security.py` | argon2 password hashing, token generation/digest |
| `database/migrations/` | Alembic revisions (hand-written SQL, no autogenerate) |
| `storage/` | Blob store: `BlobPipe`, the `BlobStore` seam, S3 implementation, key layout |
| `services/audio.py` | Ingest and deletion — the operations spanning both stores |
| `api/` | FastAPI service: `main.py`, `deps.py`, `routes/`, `schema/` |
| `cli/main.py` | Admin CLI |

## Audio storage

Audio is grouped as `recording_sessions` → `audio_segments`. A segment row is
metadata plus a pointer (`bucket` + `object_key`); the bytes live in the blob
store, never in Postgres.

```python
from services import audio

segment = await audio.ingest_segment(session_id, data, content_type="audio/wav")
url = await audio.segment_url(segment.id)
```

`ingest_segment` writes the object first and then the row, deleting the object
if the row fails — so a failed ingest leaves nothing behind. Deletion goes the
other way (row, then object). Use `services.audio.delete_segment` /
`delete_session` rather than the repositories directly: Postgres cascades remove
rows but cannot touch blobs, so deleting a user through `pipe.users.delete()`
orphans their objects.

### Blob store

The bytes live in the **MinIO** service in `~/docker_deployments` (compose entry
`minio`, data under `~/docker_deployments/minio/`, console at
`https://minio.andrewheschl.ca`). This repo stores no audio.

That store is **shared with other services**. Its root is a flat set of buckets,
one per data domain; this application owns the **`audio`** bucket and holds
credentials scoped to it by policy, so it can neither read nor write anything
else. Object keys are therefore plain paths inside that bucket
(`users/<uuid>/sessions/<uuid>/…`) with no service-name prefix — separation is
by bucket, not by a prefix convention. Provisioning is one command in the
deployment repo:

```bash
~/docker_deployments/builds/minio_add_service.sh audio audiopipeline
```

The application speaks only the S3 API through the `BlobStore` protocol in
`storage/base.py`, so moving to AWS S3, R2, or GCS is a `.env` change; a
non-S3 backend would be one new class implementing that protocol.

## HTTP API

```bash
uv run python -m api                       # docs at /docs
uv run python -m api --reload --port 9000
uv run python -m api --host 0.0.0.0 --workers 4
```

`api/main.py:serve` is the entrypoint, declared as `audio-pipeline-api` in
`[project.scripts]` alongside the CLI's `audio-pipeline`. As with the CLI, no
build backend means the console script is not on `PATH` — use `python -m api`.

Read-only for now — listing and fetching sessions and segments, and downloading
audio. No write endpoints yet.

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health`, `/health/ready` | liveness; readiness checks both stores |
| GET | `/v1/me` | the token's owner and their stored-audio totals |
| GET | `/v1/sessions` | your sessions, newest first (`?open_only=true` to filter) |
| GET | `/v1/sessions/{id}` | one session |
| GET | `/v1/sessions/{id}/segments` | that session's segments, in capture order |
| GET | `/v1/segments` | your segments across all sessions, newest ingested first |
| GET | `/v1/segments/{id}` | one segment's metadata |
| GET | `/v1/segments/{id}/audio` | the audio bytes, streamed |

Authenticate with a token from `tokens issue`:

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/v1/sessions
```

Three things worth knowing about the design:

- **Auth is a dependency that returns the caller**, not middleware. A route asks
  for `user: CurrentUser` (or `session: OwnedSession`, which depends on it), so a
  route is protected because it needs the identity to do its work. Adding an
  unprotected route means forgetting a parameter, not forgetting a decorator.
- **Path parameters resolve to rows.** Handlers receive a loaded
  `RecordingSession` / `AudioSegment` that the dependency already authorized;
  another user's id yields `404`, never `403`, so ids stay unconfirmable.
- **Responses are their own models** in `api/schema/`, built with explicit
  `from_model()` mappings. `bucket` and `object_key` never leave the service.

Lists take `limit` (≤200) and `offset`, and return `{items, limit, offset,
has_more}`. `has_more` costs no extra query — routes fetch one row past the
limit and trim.

### Audio delivery

`GET /v1/segments/{id}/audio` is a proper HTTP media endpoint, not a bare
download:

- **`Range` requests** (RFC 9110) return `206 Partial Content` with
  `Content-Range`. This is what lets a browser's `<audio>` element seek and an
  interrupted transfer resume. The range is passed through to the object store,
  so only the requested slice crosses the network. Multi-range requests and
  malformed `Range` headers fall back to the whole object, as the spec allows;
  a well-formed but out-of-bounds range gets `416`.
- **`ETag` + `If-None-Match`** → `304`. The tag is the segment's stored SHA-256,
  which is a genuinely strong validator, and segments are immutable once
  written, hence `Cache-Control: private, max-age=31536000, immutable`.
- **`If-Range`** serves the full object when the client's copy is stale, so a
  fresh range can never be stitched onto older bytes.

Longer term, playing a whole session as one continuous track is what HLS
(`.m3u8` playlist per session) is for — `recording_sessions → audio_segments` is
already shaped like it. That needs `duration_ms` populated and a settled codec,
so it is not implemented yet.

### Setup

1. Copy `.env.example` to `.env` and fill in the `DATABASE_*` and `STORAGE_*`
   credentials. `STORAGE_ACCESS_KEY`/`STORAGE_SECRET_KEY` must match
   `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` in the deployment stack.
2. `uv sync`
3. Start MinIO: `docker compose -f ~/docker_deployments/builds/docker-compose.yml up -d minio`
4. `uv run alembic upgrade head`, then `uv run python -m cli.main blobs check`

### CLI

```
uv run python -m cli.main check
uv run python -m cli.main users create --email me@example.com
uv run python -m cli.main users list | show EMAIL | set-password EMAIL
uv run python -m cli.main users activate | deactivate | delete EMAIL
uv run python -m cli.main tokens issue EMAIL --name laptop --ttl-days 30
uv run python -m cli.main tokens list EMAIL | revoke TOKEN_ID | revoke-all EMAIL
uv run python -m cli.main tokens purge-expired

uv run python -m cli.main blobs check | ls [PREFIX]
uv run python -m cli.main sessions start EMAIL --device glasses-01 --label walk
uv run python -m cli.main sessions list EMAIL | end SESSION_ID | delete SESSION_ID
uv run python -m cli.main segments ingest SESSION_ID clip.wav
uv run python -m cli.main segments list SESSION_ID | show SEGMENT_ID
uv run python -m cli.main segments download SEGMENT_ID out.wav | url SEGMENT_ID
```

`pyproject.toml` declares the `audio-pipeline` entry point, but with no build backend
the project is not installed onto `PATH` — use `python -m cli.main`.

## Tests

```bash
uv run pytest              # everything
uv run pytest tests/unit   # pure logic, no Docker
```

`tests/e2e/` starts throwaway Postgres and MinIO containers (testcontainers),
applies the migrations, runs the API as a real `python -m api` process on a
loopback port, and drives it with `httpx`. Roughly 13 s for the whole suite.

Two things make it safe and repeatable:

- **A settings guard.** Every `DATABASE_*` / `STORAGE_*` variable is overridden
  to point at the containers and both `@lru_cache`d settings objects are
  cleared; a session fixture then asserts the app really is pointed at them
  before anything destructive runs. The per-test cleanup truncates tables and
  empties a bucket, so one variable falling through to your `.env` would delete
  real data. The test bucket is `test-audio`, never `audio`.
- **Per-test cleanup runs in the test's own event loop.** `database.pipe` caches
  its pool on the loop that created it, so the autouse fixture awaits
  `close_pool()` in teardown.

### Adding a repository

Subclass `BaseRepo` (`database/repos/base.py`), which provides `_fetch_one`,
`_fetch_all`, `_execute`, and `_fetch_value`, then add one `cached_property` to
`DatabasePipe` returning it — nothing else to wire up:

```python
@cached_property
def clips(self) -> ClipsRepo:
    return ClipsRepo(self.connection)
```

Always pass values as `%s` parameters, never f-strings. New tables get a
hand-written Alembic revision under `database/migrations/versions/` — there is
no autogenerate, because there are no ORM models.
