HOST ?= 127.0.0.1
PORT ?= 8000

.DEFAULT_GOAL := help

.PHONY: help api api-lan worker schema openapi gen-client web migrate test test-unit test-e2e \
        sidecars stack-up stack-down stack-build stack-ps stack-logs stack-migrate stack-cli nginx

help:
	@echo "make api        serve the API on $(HOST):$(PORT) (override HOST=/PORT=)"
	@echo "make api-lan    serve on 0.0.0.0 for devices on the LAN (glasses/miniapp dev)"
	@echo "make worker     run the processing pipelines (docs/processing-pipelines.md)"
	@echo "make web        run the web frontend dev server (needs 'make api' running too)"
	@echo "make sidecars   start just the model containers, for the flow above"
	@echo "make schema     print the streaming frame schema (served at /stream-schema.json)"
	@echo "make openapi    write the OpenAPI spec to frontend/openapi.json"
	@echo "make gen-client regenerate the frontend's typed API client from the spec"
	@echo "make migrate    apply alembic migrations"
	@echo "make test       run the whole test suite (e2e needs Docker)"
	@echo "make test-unit  fast tests only, no Docker"
	@echo "make test-e2e   containerized end-to-end suite"
	@echo ""
	@echo "deployment (deploy/docker-compose.yml, needs deploy/.env):"
	@echo "make stack-up      build and start the whole stack"
	@echo "make stack-down    stop it (weights survive: the caches are external)"
	@echo "make stack-build   build the images without starting anything"
	@echo "make stack-ps      service status"
	@echo "make stack-logs    follow logs; S=worker for one service"
	@echo "make stack-migrate run alembic against the deployment"
	@echo "make stack-cli     admin CLI in the stack, e.g. ARGS=\"users list\""
	@echo "make nginx         install the vhost into the shared front door and reload it"

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

# --- deployment (deploy/docker-compose.yml) ----------------------------------

COMPOSE := docker compose -f deploy/docker-compose.yml

# The bridge between the two worlds. These are the same containers the
# deployment runs, published on 127.0.0.1:8033-8035 — which is where the
# code's own defaults point — so `make api` / `make worker` on the host reach
# them with no configuration. Nothing about the dev flow changes.
sidecars:
	$(COMPOSE) up -d --build asr-parakeet diar-pyannote audio-tagger

stack-up:
	$(COMPOSE) up -d --build

stack-down:
	$(COMPOSE) down

stack-build:
	$(COMPOSE) build

stack-ps:
	$(COMPOSE) ps

# S=api, S=worker, ... or leave unset for everything.
stack-logs:
	$(COMPOSE) logs -f $(S)

stack-migrate:
	$(COMPOSE) run --rm migrate

# `run`, not `exec`: the CLI is a one-shot click command with no long-running
# container of its own. ARGS="users create --email ..."
stack-cli:
	$(COMPOSE) run --rm cli python -m cli.main $(ARGS)

# Installs the vhost into the shared front door and reloads it. server-nginx
# bind-mounts conf.d, so the file has to physically live over there; this repo
# holds the source of truth.
INFRA ?= $(HOME)/docker_deployments
nginx:
	cp deploy/nginx/spyware.conf $(INFRA)/builds/nginx/conf.d/spyware.conf
	docker compose -f $(INFRA)/builds/docker-compose.yml up -d server-nginx
