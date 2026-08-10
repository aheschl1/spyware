"""The HTTP application and its entrypoint.

    uv run python -m api                      # serve
    uv run python -m api --reload --port 9000
    uv run uvicorn api.main:app               # for tooling that wants the app

"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import click
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from api.routes import health, segments, sessions, users
from api.schema.common import ErrorResponse
from database.exceptions import DatabaseError, NotFoundError
from database.pipe import DatabasePipe, close_pool
from storage.base import BlobNotFoundError

logger = logging.getLogger(__name__)

API_PREFIX = "/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Open the pool at boot so a bad DATABASE_* config fails here, not on the
    # first request.
    async with DatabasePipe() as pipe:
        await pipe.ping()
    yield
    await close_pool()


def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ErrorResponse(detail=detail).model_dump())


def create_app() -> FastAPI:
    app = FastAPI(
        title="audio-pipeline",
        version="0.1.0",
        summary="Read access to recording sessions and their audio segments.",
        lifespan=lifespan,
        responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    )

    # Domain errors from the repositories and services map to HTTP here, so no
    # route body needs a try/except.
    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return _error(status.HTTP_404_NOT_FOUND, str(exc))

    @app.exception_handler(BlobNotFoundError)
    async def _blob_missing(request: Request, exc: BlobNotFoundError) -> JSONResponse:
        # A row pointing at a missing object is an inconsistency, not a plain
        # miss: logged even though the caller only sees 404.
        logger.error("segment row references a missing object: %s", exc.key)
        return _error(status.HTTP_404_NOT_FOUND, "the stored audio for this segment is missing")

    @app.exception_handler(DatabaseError)
    async def _database_error(request: Request, exc: DatabaseError) -> JSONResponse:
        logger.exception("database error serving %s", request.url.path)
        return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal error")

    app.include_router(health.router)
    app.include_router(users.router, prefix=API_PREFIX)
    app.include_router(sessions.router, prefix=API_PREFIX)
    app.include_router(segments.router, prefix=API_PREFIX)
    return app


app = create_app()

# An import string, not `app`: uvicorn re-imports the module in each subprocess
# for --reload and --workers.
APP_PATH = "api.main:app"


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind.")
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True, help="Restart on code changes.")
@click.option("--workers", default=1, show_default=True, type=int, help="Worker processes.")
@click.option(
    "--log-level",
    default="info",
    show_default=True,
    type=click.Choice(["critical", "error", "warning", "info", "debug", "trace"]),
)
def serve(host: str, port: int, reload: bool, workers: int, log_level: str) -> None:
    """Run the audio-pipeline HTTP API."""
    if reload and workers > 1:
        raise click.UsageError("--reload cannot be combined with --workers")
    uvicorn.run(
        APP_PATH,
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else None,
        log_level=log_level,
    )


if __name__ == "__main__":
    serve()
