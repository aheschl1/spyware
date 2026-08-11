"""Raw-SQL repository for the ``speaker_embeddings`` table (pgvector).

Vectors cross the wire in pgvector's text form (``[1.0,2.0]``) with an
explicit ``::vector`` cast, so no driver-side type registration is needed.
"""

from collections.abc import Sequence
from uuid import UUID

from database.repos.base import BaseRepo
from database.schema.embeddings import SpeakerEmbedding, SpeakerEmbeddingCreate

# embedding reads back as text: psycopg has no codec for the vector OID.
COLUMNS = "artifact_id, session_id, speaker, model, embedding::text AS embedding, created_at"


def _literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"


class EmbeddingsRepo(BaseRepo):
    async def create_many(self, items: Sequence[SpeakerEmbeddingCreate]) -> int:
        """Insert a batch in one statement; rows ride the caller's transaction
        alongside the artifact rows they reference."""
        if not items:
            return 0
        params: list = []
        for item in items:
            params += [
                item.artifact_id,
                item.session_id,
                item.speaker,
                item.model,
                _literal(item.embedding),
            ]
        values = ", ".join(["(%s, %s, %s, %s, %s::vector)"] * len(items))
        return await self._execute(
            f"""
                INSERT INTO speaker_embeddings
                    (artifact_id, session_id, speaker, model, embedding)
                VALUES {values}
            """,
            params,
        )

    async def list_for_session(self, session_id: UUID) -> list[SpeakerEmbedding]:
        return await self._fetch_all(
            SpeakerEmbedding,
            f"""
                SELECT {COLUMNS} FROM speaker_embeddings
                WHERE session_id = %s
                ORDER BY speaker, artifact_id
            """,
            (session_id,),
        )

    async def nearest(
        self, vector: Sequence[float], *, limit: int = 5
    ) -> list[SpeakerEmbedding]:
        """Closest stored embeddings by cosine distance — the primitive the
        clustering tier attaches new block-speakers with."""
        return await self._fetch_all(
            SpeakerEmbedding,
            f"""
                SELECT {COLUMNS} FROM speaker_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """,
            (_literal(vector), limit),
        )
