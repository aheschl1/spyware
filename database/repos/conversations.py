"""Reads over ``conversation`` artifacts and the transcripts they group."""

from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from database.repos.artifacts import COLUMNS as _ARTIFACT_COLUMNS
from database.repos.base import BaseRepo
from database.schema.artifacts import PipelineArtifact
from database.schema.conversations import ConversationTranscript, ConversationUtterance

_A_COLUMNS = ", ".join(f"a.{column.strip()}" for column in _ARTIFACT_COLUMNS.split(","))


class ConversationsRepo(BaseRepo):
    async def get_owned(self, conversation_id: UUID, user_id: UUID) -> PipelineArtifact | None:
        """The conversation artifact, only if its session belongs to ``user_id``."""
        return await self._fetch_one(
            PipelineArtifact,
            f"""
                SELECT {_A_COLUMNS} FROM pipeline_artifacts a
                JOIN recording_sessions rs ON rs.id = a.session_id
                WHERE a.id = %s AND a.pipeline = 'conversation'
                  AND a.kind = 'conversation' AND rs.user_id = %s
            """,
            (conversation_id, user_id),
        )

    async def transcripts_for(self, utterance_ids: list[UUID]) -> list[ConversationTranscript]:
        """Transcripts of the given utterances in timeline order, speakers resolved."""
        if not utterance_ids:
            return []
        return await self._fetch_all(
            ConversationTranscript,
            """
                SELECT t.id AS artifact_id, (t.links->>'utterance')::uuid AS utterance_id,
                       t.start_ms, t.end_ms, t.metadata->>'speaker' AS speaker,
                       e.speaker_id, s.name, coalesce(t.metadata->>'text', '') AS text,
                       (t.links->>'host_utterance')::uuid AS interjection_of
                FROM pipeline_artifacts t
                LEFT JOIN speaker_embeddings e ON e.session_id = t.session_id
                    AND e.speaker = t.metadata->>'speaker'
                LEFT JOIN speakers s ON s.id = e.speaker_id
                WHERE t.pipeline = 'transcribe' AND t.kind = 'transcript'
                  AND (t.links->>'utterance')::uuid = ANY(%s)
                ORDER BY t.start_ms, t.id
            """,
            (utterance_ids,),
        )

    async def utterances(self, utterance_ids: list[UUID]) -> list[ConversationUtterance]:
        if not utterance_ids:
            return []
        return await self._fetch_all(
            ConversationUtterance,
            """
                SELECT id, start_ms, end_ms, metadata FROM pipeline_artifacts
                WHERE pipeline = 'diarize' AND kind = 'utterance' AND id = ANY(%s)
                ORDER BY start_ms, id
            """,
            (utterance_ids,),
        )

    async def set_membership(
        self, conversation_id: UUID, *, start_ms: int, end_ms: int, patch: dict[str, Any]
    ) -> PipelineArtifact | None:
        """Rewrite the span and merge ``patch`` into metadata — the curation path."""
        return await self._fetch_one(
            PipelineArtifact,
            f"""
                UPDATE pipeline_artifacts a
                SET start_ms = %s, end_ms = %s, metadata = a.metadata || %s
                WHERE a.id = %s
                RETURNING {_A_COLUMNS}
            """,
            (start_ms, end_ms, Jsonb(patch), conversation_id),
        )
