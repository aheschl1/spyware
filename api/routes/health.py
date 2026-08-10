"""Liveness and readiness. The only routes that serve no user data."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from api.deps import Pipe
from storage.pipe import BlobPipe

router = APIRouter(prefix="/health", tags=["health"])


class HealthRead(BaseModel):
    status: str


class ReadinessRead(BaseModel):
    status: str
    database: bool
    storage: bool


@router.get("", summary="Liveness")
async def health() -> HealthRead:
    """The process is up. Takes no dependencies and touches no store."""
    return HealthRead(status="ok")


@router.get("/ready", summary="Readiness")
async def ready(pipe: Pipe) -> ReadinessRead:
    """Both backing stores answer; 503 if either does not."""
    database = await pipe.ping()
    try:
        async with BlobPipe() as blobs:
            await blobs.ensure_bucket()
        storage = True
    except Exception:
        storage = False
    if not (database and storage):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"not ready (database={database}, storage={storage})",
        )
    return ReadinessRead(status="ok", database=database, storage=storage)
