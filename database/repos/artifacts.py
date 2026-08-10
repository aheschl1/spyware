"""Raw-SQL repository for the ``pipeline_artifacts`` table."""

from uuid import UUID

from psycopg.types.json import Jsonb

from database.repos.base import BaseRepo
from database.schema.artifacts import ArtifactCreate, PipelineArtifact

COLUMNS = (
    "id, pipeline, session_id, kind, bucket, object_key, links, metadata, "
    "created_at, updated_at"
)


class ArtifactsRepo(BaseRepo):
    async def create(self, data: ArtifactCreate) -> PipelineArtifact:
        artifact = await self._fetch_one(
            PipelineArtifact,
            f"""
                INSERT INTO pipeline_artifacts
                    (pipeline, kind, session_id, bucket, object_key, links, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING {COLUMNS}
            """,
            (
                data.pipeline,
                data.kind,
                data.session_id,
                data.bucket,
                data.object_key,
                Jsonb(data.links),
                Jsonb(data.metadata),
            ),
        )
        assert artifact is not None  # INSERT ... RETURNING always yields a row
        return artifact

    async def get(self, artifact_id: UUID) -> PipelineArtifact | None:
        return await self._fetch_one(
            PipelineArtifact,
            f"SELECT {COLUMNS} FROM pipeline_artifacts WHERE id = %s",
            (artifact_id,),
        )

    async def find(
        self, pipeline: str, kind: str, session_id: UUID
    ) -> PipelineArtifact | None:
        """The newest matching artifact — how a consumer locates an upstream
        pipeline's output for a session."""
        return await self._fetch_one(
            PipelineArtifact,
            f"""
                SELECT {COLUMNS} FROM pipeline_artifacts
                WHERE pipeline = %s AND kind = %s AND session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
            """,
            (pipeline, kind, session_id),
        )

    async def list_for_session(
        self,
        session_id: UUID,
        pipeline: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PipelineArtifact]:
        sql = f"SELECT {COLUMNS} FROM pipeline_artifacts WHERE session_id = %s"
        params: list = [session_id]
        if pipeline is not None:
            sql += " AND pipeline = %s"
            params.append(pipeline)
        if kind is not None:
            sql += " AND kind = %s"
            params.append(kind)
        sql += " ORDER BY created_at, id LIMIT %s OFFSET %s"
        params += [limit, offset]
        return await self._fetch_all(PipelineArtifact, sql, params)

    async def delete(self, artifact_id: UUID) -> bool:
        """Delete the row only; any blob it points at is the caller's job."""
        return await self._execute(
            "DELETE FROM pipeline_artifacts WHERE id = %s", (artifact_id,)
        ) > 0
