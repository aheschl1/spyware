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

The bytes live in a **MinIO** service, or any other s3 compatible blob storage.

Object keys are plain paths inside the bucket
(`users/<uuid>/sessions/<uuid>/…`) with no service-name prefix.

## HTTP API

```bash
uv run python -m api
```

Authenticate with a token from `tokens issue`:

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/v1/sessions
```

Lists take `limit` (<=200) and `offset`, and return `{items, limit, offset,
has_more}`. `.

### Audio delivery

`GET /v1/segments/{id}/audio` is an HTTP media endpoint, not a bare
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
```

`pyproject.toml` declares the `audio-pipeline` entry point, but with no build backend
the project is not installed onto `PATH` — use `python -m cli.main`.

## Web frontend (`frontend/`)

A React + TypeScript browser UI over the API: session browser, timeline
(transcripts, speakers, sound tags), seekable audio playback, text→audio
search, and speaker curation: labeling, merging split voices (distance-ranked
candidates, same-name prompt), inspecting a cluster's voice-prints by ear,
moving/ejecting wrong voices (pinned so rebuilds honor it), and tuning the
clustering threshold + recomputing from a settings panel. Dev-mode only for now:

```bash
make api    # terminal 1 — the API on 127.0.0.1:8000
make web    # terminal 2 — Vite dev server; proxies /v1 + /health to the API
```

Log in with an account from `cli users create`. Models are **generated, never
hand-written**: `make gen-client` regenerates `frontend/openapi.json` (a pure
function of the route/model declarations — no server needed) and
`frontend/src/api/schema.d.ts` via openapi-typescript; requests go through a
typed openapi-fetch client. Regenerate after any API change — `npm run build`
(tsc) fails on drift. Audio playback uses a minutes-lived token minted by
`POST /v1/sessions/{id}/playback` in the audio URL's `?token=`, because media
elements cannot send an Authorization header.

## Tests

```bash
uv run pytest              # everything
uv run pytest tests/unit   # pure logic, no Docker
```

`tests/e2e/` starts throwaway Postgres and MinIO containers (testcontainers),
applies the migrations, runs the API as a real `python -m api` process on a
loopback port, and drives it with `httpx`.
