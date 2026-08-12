"""Pydantic model for the ``cluster_params`` table.

Per-user overrides for the batch speaker-clustering knobs. Every field is
nullable: NULL means "inherit the PROCESSING_* env default", so recalibrated
defaults keep applying to fields the user never explicitly pinned.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ClusterParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UUID
    cluster_distance: float | None = None
    min_talk_ms: int | None = None
    updated_at: datetime
