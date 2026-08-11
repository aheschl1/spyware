"""Pydantic models for the ``processing_jobs`` table."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    DEAD = "dead"


class Job(BaseModel):
    """One unit of pipeline work, as materialized in the queue."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    pipeline: str
    session_id: UUID | None = None
    artifact_id: UUID | None = None
    priority: int
    status: JobStatus
    attempts: int
    max_attempts: int
    run_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    dedup_key: str | None = None
    claimed_at: datetime | None = None
    claimed_by: str | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class JobCreate(BaseModel):
    """Input for enqueueing a job.

    A ``dedup_key`` makes the enqueue idempotent forever: the key stays taken
    even after the job succeeds or dies, so discovery can re-emit the same work
    without re-running it. ``max_attempts`` of None defers to the table default.
    """

    model_config = ConfigDict(frozen=True)

    pipeline: str
    session_id: UUID | None = None
    artifact_id: UUID | None = None
    priority: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)
    dedup_key: str | None = None
    max_attempts: int | None = None
