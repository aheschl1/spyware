"""Raw-SQL queries over the audio-tag tier's per-window class scores.

The tagger's output lives in ``pipeline_artifacts.metadata`` as
``{"labels": [{"label", "score"}, ...]}`` — these queries unnest that array
(LATERAL) to filter and rank by class score. This is the deterministic
complement to the CLAP embedding search: exact classes, calibrated sigmoid
scores, no model in the loop.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from database.repos.base import BaseRepo


class TagWindowHit(BaseModel):
    """One window where the class scored at least the floor."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID
    session_id: UUID
    start_ms: int
    end_ms: int
    label: str
    score: float
    metadata: dict[str, Any]
    created_at: datetime


class TagLabelCount(BaseModel):
    """One class that occurs in the caller's windows, with its footprint."""

    model_config = ConfigDict(frozen=True)

    label: str
    windows: int
    best: float


class TagsRepo(BaseRepo):
    async def search(
        self,
        *,
        user_id: UUID,
        label: str,
        min_score: float,
        session_id: UUID | None = None,
        limit: int = 20,
    ) -> list[TagWindowHit]:
        """Windows whose tag list contains a class matching ``label``
        (case-insensitive substring — AudioSet names are long, "speech"
        should find "Male speech, man speaking") at or above ``min_score``,
        best score first. DISTINCT ON keeps one row per window when several
        classes match the substring."""
        conditions = ["s.user_id = %s"]
        params: list = [user_id]
        if session_id is not None:
            conditions.append("a.session_id = %s")
            params.append(session_id)
        params += [f"%{label}%", min_score, limit]
        return await self._fetch_all(
            TagWindowHit,
            f"""
                SELECT * FROM (
                    SELECT DISTINCT ON (a.id)
                        a.id AS artifact_id, a.session_id, a.start_ms, a.end_ms,
                        elem->>'label' AS label,
                        (elem->>'score')::float8 AS score,
                        a.metadata, a.created_at
                    FROM pipeline_artifacts a
                    JOIN recording_sessions s ON s.id = a.session_id
                    CROSS JOIN LATERAL jsonb_array_elements(a.metadata->'labels') elem
                    WHERE a.pipeline = 'audio-tag' AND a.kind = 'audio-tag'
                      AND {" AND ".join(conditions)}
                      AND elem->>'label' ILIKE %s
                      AND (elem->>'score')::float8 >= %s
                    ORDER BY a.id, (elem->>'score')::float8 DESC
                ) hits
                ORDER BY score DESC, session_id, start_ms
                LIMIT %s
            """,
            params,
        )

    async def labels(self, *, user_id: UUID) -> list[TagLabelCount]:
        """Every class present in the caller's windows, most frequent first —
        feeds the filter UI's suggestions."""
        return await self._fetch_all(
            TagLabelCount,
            """
                SELECT elem->>'label' AS label,
                       count(*) AS windows,
                       max((elem->>'score')::float8) AS best
                FROM pipeline_artifacts a
                JOIN recording_sessions s ON s.id = a.session_id
                CROSS JOIN LATERAL jsonb_array_elements(a.metadata->'labels') elem
                WHERE a.pipeline = 'audio-tag' AND a.kind = 'audio-tag'
                  AND s.user_id = %s
                GROUP BY 1
                ORDER BY windows DESC, label
            """,
            (user_id,),
        )
