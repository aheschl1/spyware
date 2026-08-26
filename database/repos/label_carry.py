"""Reads and writes for carrying speaker curation across a re-diarize."""

from uuid import UUID

from pydantic import BaseModel

from database.repos.base import BaseRepo


class LabelIdentity(BaseModel):
    """One block-local label with its identity and utterance spans."""

    speaker: str
    model: str
    speaker_id: UUID
    pinned: bool
    spans: list[list[int]]


class EditedTranscript(BaseModel):
    start_ms: int
    end_ms: int
    text: str


class LabelCarryRepo(BaseRepo):
    async def identities_for_session(self, session_id: UUID) -> list[LabelIdentity]:
        """Every label that resolves to an identity — pinned, or assigned by
        the last clustering run — with the spans of its utterances."""
        return await self._fetch_all(
            LabelIdentity,
            """
                SELECT e.speaker, e.model,
                       coalesce(p.speaker_id, e.speaker_id) AS speaker_id,
                       (p.speaker_id IS NOT NULL) AS pinned,
                       coalesce((
                           SELECT json_agg(json_build_array(u.start_ms, u.end_ms)
                                           ORDER BY u.start_ms)
                           FROM pipeline_artifacts u
                           WHERE u.session_id = e.session_id
                             AND u.pipeline = 'diarize' AND u.kind = 'utterance'
                             AND u.metadata->>'speaker' = e.speaker
                       ), '[]'::json) AS spans
                FROM speaker_embeddings e
                LEFT JOIN speaker_pins p ON p.session_id = e.session_id
                    AND p.speaker = e.speaker AND p.model = e.model
                WHERE e.session_id = %s
                  AND coalesce(p.speaker_id, e.speaker_id) IS NOT NULL
                ORDER BY e.speaker
            """,
            (session_id,),
        )

    async def edited_transcripts(self, session_id: UUID) -> list[EditedTranscript]:
        return await self._fetch_all(
            EditedTranscript,
            """
                SELECT start_ms, end_ms, metadata->>'text' AS text
                FROM pipeline_artifacts
                WHERE session_id = %s AND pipeline = 'transcribe' AND kind = 'transcript'
                  AND metadata->>'edited' = 'true' AND start_ms IS NOT NULL
                ORDER BY start_ms
            """,
            (session_id,),
        )

    async def pin_label(self, session_id: UUID, speaker: str, speaker_id: UUID) -> int:
        """Pin a label under whatever model its current embedding carries."""
        return await self._execute(
            """
                INSERT INTO speaker_pins (session_id, speaker, model, speaker_id)
                SELECT session_id, speaker, model, %s FROM speaker_embeddings
                WHERE session_id = %s AND speaker = %s
                ON CONFLICT (session_id, speaker, model) DO UPDATE
                    SET speaker_id = EXCLUDED.speaker_id
            """,
            (speaker_id, session_id, speaker),
        )

    async def unpin_label(self, session_id: UUID, speaker: str, model: str) -> int:
        return await self._execute(
            "DELETE FROM speaker_pins WHERE session_id = %s AND speaker = %s AND model = %s",
            (session_id, speaker, model),
        )

    async def assign_label(self, session_id: UUID, speaker: str, speaker_id: UUID) -> int:
        """Seed the label's fresh embedding with its previous cluster, so the
        next rebuild's majority match finds the named identity again."""
        return await self._execute(
            """
                UPDATE speaker_embeddings SET speaker_id = %s
                WHERE session_id = %s AND speaker = %s
            """,
            (speaker_id, session_id, speaker),
        )

    async def sessions_with_diarize_map(self) -> list[UUID]:
        rows = await self._fetch_all(
            _SessionId,
            """
                SELECT DISTINCT session_id FROM pipeline_artifacts
                WHERE pipeline = 'diarize' AND kind = 'diarize-map' AND session_id IS NOT NULL
                ORDER BY session_id
            """,
            (),
        )
        return [row.session_id for row in rows]


class _SessionId(BaseModel):
    session_id: UUID
