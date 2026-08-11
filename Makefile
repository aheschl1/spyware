HOST ?= 127.0.0.1
PORT ?= 8000

.DEFAULT_GOAL := help

.PHONY: help api api-lan worker schema openapi gen-client web migrate test test-unit test-e2e

help:
	@echo "make api        serve the API on $(HOST):$(PORT) (override HOST=/PORT=)"
	@echo "make api-lan    serve on 0.0.0.0 for devices on the LAN (glasses/miniapp dev)"
	@echo "make worker     run the processing pipelines (docs/processing-pipelines.md)"
	@echo "make web        run the web frontend dev server (needs 'make api' running too)"
	@echo "make schema     print the streaming frame schema (served at /stream-schema.json)"
	@echo "make openapi    write the OpenAPI spec to frontend/openapi.json"
	@echo "make gen-client regenerate the frontend's typed API client from the spec"
	@echo "make migrate    apply alembic migrations"
	@echo "make test       run the whole test suite (e2e needs Docker)"
	@echo "make test-unit  fast tests only, no Docker"
	@echo "make test-e2e   containerized end-to-end suite"

api:
	uv run python -m api --host $(HOST) --port $(PORT)

api-lan:
	$(MAKE) api HOST=0.0.0.0

# One supervisor, one child process per registered pipeline.
worker:
	uv run python -m processing

# The running server serves this same document at /stream-schema.json, which
# is what client codegen fetches (e.g. `bun run gen` in the computa miniapp);
# this target prints it without a server.
schema:
	uv run python -m api.schema.stream_export

# The spec is a pure function of the route/model declarations (no server, no
# database), so this is deterministic and the output is committed — the
# frontend's types are generated from it (make gen-client) and reviewed as
# part of any API change.
openapi:
	uv run python -c "import json; from api.main import app; print(json.dumps(app.openapi(), indent=2))" > frontend/openapi.json

gen-client: openapi
	cd frontend && npm run gen

# Dev pairing: `make api` in one terminal, `make web` in another. Vite proxies
# /v1 and /health to API_PROXY (default 127.0.0.1:8000), so the browser talks
# same-origin. Override the bind/port to serve peers, e.g.
#   make web WEB_HOST=10.8.0.1 WEB_PORT=12345 API_PROXY=http://10.8.0.1:8000
WEB_HOST ?= 127.0.0.1
WEB_PORT ?= 5173
web:
	cd frontend && npm install --no-audit --no-fund && npm run dev -- --host $(WEB_HOST) --port $(WEB_PORT)

migrate:
	uv run alembic upgrade head

test:
	uv run pytest tests

test-unit:
	uv run pytest tests/unit

test-e2e:
	uv run pytest tests/e2e
