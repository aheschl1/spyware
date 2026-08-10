HOST ?= 127.0.0.1
PORT ?= 8000

.DEFAULT_GOAL := help

.PHONY: help api api-lan worker schema migrate test test-unit test-e2e

help:
	@echo "make api        serve the API on $(HOST):$(PORT) (override HOST=/PORT=)"
	@echo "make api-lan    serve on 0.0.0.0 for devices on the LAN (glasses/miniapp dev)"
	@echo "make worker     run the processing pipelines (docs/processing-pipelines.md)"
	@echo "make schema     print the streaming frame schema (served at /stream-schema.json)"
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

migrate:
	uv run alembic upgrade head

test:
	uv run pytest tests

test-unit:
	uv run pytest tests/unit

test-e2e:
	uv run pytest tests/e2e
