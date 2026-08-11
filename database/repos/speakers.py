"""Raw-SQL repository for ``speakers`` and the clustering statements.

Every operation that computes a distance or an average filters by
``(user_id, model)``: pgvector's ``<=>`` and ``avg()`` error out on mixed
dimensions, and embeddings from different models are not comparable anyway.
Vector literals cross the wire in text form with ``::vector`` casts, same as
:mod:`database.repos.embeddings`.
"""

from collections.abc import Sequence
from uuid import UUID

from psycopg.rows import dict_row

from database.repos.base import BaseRepo
from database.schema.speakers import (
    ClusterCandidate,
    SessionSpeakerLabel,
    Speaker,
    SpeakerCreate,
    SpeakerSummary,
    SpeakerTranscript,
)

COLUMNS = "id, user_id, name, model, centroid::text AS centroid, created_at, updated_at"

_CANDIDATE_COLUMNS = """
    e.artifact_id, e.session_id, e.speaker, e.model,
    e.embedding::text AS embedding,
    (a.metadata->>'talk_ms')::bigint AS talk_ms
"""

_TRANSCRIPT_COLUMNS = """
    t.id AS artifact_id, t.session_id, e.speaker_id, e.speaker,
    t.start_ms, t.end_ms,
    coalesce(t.metadata->>'text', '') AS text, t.metadata->>'model' AS model
"""


def _literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"


class SpeakersRepo(BaseRepo):
    async def create(self, data: SpeakerCreate) -> Speaker:
        speaker = await self._fetch_one(
            Speaker,
            f"""
                INSERT INTO speakers (user_id, name, model, centroid)
                VALUES (%s, %s, %s, %s::vector)
                RETURNING {COLUMNS}
            """,
            (data.user_id, data.name, data.model, _literal(data.centroid)),
        )
        assert speaker is not None  # INSERT ... RETURNING always yields a row
        return speaker

    async def get(self, speaker_id: UUID) -> Speaker | None:
        return await self._fetch_one(
            Speaker,
            f"SELECT {COLUMNS} FROM speakers WHERE id = %s",
            (speaker_id,),
        )

    async def list_for_user(
        self,
        user_id: UUID,
        name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SpeakerSummary]:
        """The user's clusters with membership counts, named or not.

        LEFT JOIN: a named cluster whose members were cascaded away by a
        diarize republish must stay visible — it anchors the identity that
        reclaims its members on re-cluster.
        """
        sql = f"""
            SELECT s.id, s.user_id, s.name, s.model,
                   s.centroid::text AS centroid, s.created_at, s.updated_at,
                   count(e.artifact_id) AS embeddings,
                   count(DISTINCT e.session_id) AS sessions
            FROM speakers s
            LEFT JOIN speaker_embeddings e ON e.speaker_id = s.id
            WHERE s.user_id = %s
        """
        params: list = [user_id]
        if name is not None:
            sql += " AND s.name = %s"
            params.append(name)
        sql += " GROUP BY s.id ORDER BY s.created_at, s.id LIMIT %s OFFSET %s"
        params += [limit, offset]
        return await self._fetch_all(SpeakerSummary, sql, params)

    async def summarize(self, speaker_id: UUID) -> SpeakerSummary | None:
        """One cluster with its membership counts."""
        return await self._fetch_one(
            SpeakerSummary,
            """
                SELECT s.id, s.user_id, s.name, s.model,
                       s.centroid::text AS centroid, s.created_at, s.updated_at,
                       count(e.artifact_id) AS embeddings,
                       count(DISTINCT e.session_id) AS sessions
                FROM speakers s
                LEFT JOIN speaker_embeddings e ON e.speaker_id = s.id
                WHERE s.id = %s
                GROUP BY s.id
            """,
            (speaker_id,),
        )

    async def set_name(self, speaker_id: UUID, name: str | None) -> Speaker | None:
        return await self._fetch_one(
            Speaker,
            f"UPDATE speakers SET name = %s WHERE id = %s RETURNING {COLUMNS}",
            (name, speaker_id),
        )

    # ----------------------------------------------------------- clustering

    async def nearest(
        self, user_id: UUID, model: str, vector: Sequence[float]
    ) -> tuple[Speaker, float] | None:
        """The user's closest cluster for this model, with cosine distance."""
        literal = _literal(vector)
        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"""
                    SELECT {COLUMNS}, centroid <=> %s::vector AS distance
                    FROM speakers
                    WHERE user_id = %s AND model = %s
                    ORDER BY centroid <=> %s::vector
                    LIMIT 1
                """,
                (literal, user_id, model, literal),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        distance = row.pop("distance")
        return Speaker.model_validate(row), float(distance)

    async def assign_if_unassigned(self, artifact_id: UUID, speaker_id: UUID) -> bool:
        """Attach an embedding to a cluster unless it vanished or was claimed.

        Rowcount 0 means diarize republished mid-job (row deleted) or a
        replayed job already assigned it — either way the caller skips.
        """
        return await self._execute(
            """
                UPDATE speaker_embeddings SET speaker_id = %s
                WHERE artifact_id = %s AND speaker_id IS NULL
            """,
            (speaker_id, artifact_id),
        ) > 0

    async def recompute_centroid(self, speaker_id: UUID) -> None:
        """Centroid = mean of current members; an empty cluster keeps its
        last centroid as the identity anchor (avg over zero rows is NULL)."""
        await self._execute(
            """
                UPDATE speakers s SET centroid = m.c
                FROM (
                    SELECT avg(embedding) AS c FROM speaker_embeddings
                    WHERE speaker_id = %s
                ) m
                WHERE s.id = %s AND m.c IS NOT NULL
            """,
            (speaker_id, speaker_id),
        )

    async def member_counts(self, user_id: UUID, model: str) -> dict[UUID, int]:
        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                    SELECT s.id, count(e.artifact_id) AS members
                    FROM speakers s
                    LEFT JOIN speaker_embeddings e ON e.speaker_id = s.id
                    WHERE s.user_id = %s AND s.model = %s
                    GROUP BY s.id
                """,
                (user_id, model),
            )
            rows = await cur.fetchall()
        return {row["id"]: row["members"] for row in rows}

    async def list_for_user_model(self, user_id: UUID, model: str) -> list[Speaker]:
        return await self._fetch_all(
            Speaker,
            f"""
                SELECT {COLUMNS} FROM speakers
                WHERE user_id = %s AND model = %s
                ORDER BY created_at, id
            """,
            (user_id, model),
        )

    async def merge(self, loser_id: UUID, survivor_id: UUID) -> None:
        """Fold one cluster into another; the survivor keeps id and name."""
        await self._execute(
            "UPDATE speaker_embeddings SET speaker_id = %s WHERE speaker_id = %s",
            (survivor_id, loser_id),
        )
        await self._execute("DELETE FROM speakers WHERE id = %s", (loser_id,))
        await self.recompute_centroid(survivor_id)

    async def delete_empty_unlabeled(self, user_id: UUID, model: str) -> int:
        """Drop unnamed clusters with no members (republication leftovers);
        named empties survive as identity anchors."""
        return await self._execute(
            """
                DELETE FROM speakers s
                WHERE s.user_id = %s AND s.model = %s AND s.name IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM speaker_embeddings e WHERE e.speaker_id = s.id
                  )
            """,
            (user_id, model),
        )

    async def unassigned_for_session(self, session_id: UUID) -> list[ClusterCandidate]:
        """The session's not-yet-clustered embeddings, longest speech first
        (strong voice-prints seed clusters; rows without talk_ms sort first
        and pass the gate — they predate the stamp)."""
        return await self._fetch_all(
            ClusterCandidate,
            f"""
                SELECT {_CANDIDATE_COLUMNS}
                FROM speaker_embeddings e
                LEFT JOIN pipeline_artifacts a ON a.id = e.artifact_id
                WHERE e.session_id = %s AND e.speaker_id IS NULL
                ORDER BY (a.metadata->>'talk_ms')::bigint DESC NULLS FIRST,
                         e.artifact_id
            """,
            (session_id,),
        )

    async def unassigned_for_user(self, user_id: UUID) -> list[ClusterCandidate]:
        """Every unassigned embedding across the user's sessions (rebuild)."""
        return await self._fetch_all(
            ClusterCandidate,
            f"""
                SELECT {_CANDIDATE_COLUMNS}
                FROM speaker_embeddings e
                JOIN recording_sessions rs ON rs.id = e.session_id
                LEFT JOIN pipeline_artifacts a ON a.id = e.artifact_id
                WHERE rs.user_id = %s AND e.speaker_id IS NULL
                ORDER BY (a.metadata->>'talk_ms')::bigint DESC NULLS FIRST,
                         e.artifact_id
            """,
            (user_id,),
        )

    async def clear_assignments_for_user(self, user_id: UUID) -> int:
        """Rebuild prep: detach every embedding of the user from its cluster."""
        return await self._execute(
            """
                UPDATE speaker_embeddings e SET speaker_id = NULL
                FROM recording_sessions rs
                WHERE rs.id = e.session_id AND rs.user_id = %s
                  AND e.speaker_id IS NOT NULL
            """,
            (user_id,),
        )

    async def delete_unlabeled_for_user(self, user_id: UUID) -> int:
        """Rebuild prep: unnamed clusters churn; named ones anchor identity."""
        return await self._execute(
            "DELETE FROM speakers WHERE user_id = %s AND name IS NULL",
            (user_id,),
        )

    # ------------------------------------------------------------- read side

    async def labels_for_session(self, session_id: UUID) -> list[SessionSpeakerLabel]:
        """Every block-local label of a session with its cluster resolution
        (unassigned labels included, speaker_id NULL — nothing is hidden)."""
        return await self._fetch_all(
            SessionSpeakerLabel,
            """
                SELECT e.speaker, e.speaker_id, s.name,
                       (a.metadata->>'talk_ms')::bigint AS talk_ms
                FROM speaker_embeddings e
                LEFT JOIN speakers s ON s.id = e.speaker_id
                LEFT JOIN pipeline_artifacts a ON a.id = e.artifact_id
                WHERE e.session_id = %s
                ORDER BY e.speaker
            """,
            (session_id,),
        )

    async def transcripts_for_speaker(
        self, speaker_id: UUID, limit: int = 50, offset: int = 0
    ) -> list[SpeakerTranscript]:
        return await self._transcripts("s.id = %s", [speaker_id], limit, offset)

    async def transcripts_for_name(
        self, user_id: UUID, name: str, limit: int = 50, offset: int = 0
    ) -> list[SpeakerTranscript]:
        """Everything said by every cluster sharing this name — names are
        non-unique by design, so this is the complete-retrieval path."""
        return await self._transcripts(
            "s.user_id = %s AND s.name = %s", [user_id, name], limit, offset
        )

    async def _transcripts(
        self, where: str, params: list, limit: int, offset: int
    ) -> list[SpeakerTranscript]:
        # One embedding per (session, label) — UNIQUE in the schema — so the
        # join cannot duplicate transcript rows. The predicate matches
        # pipeline_artifacts_transcript_speaker_idx exactly.
        return await self._fetch_all(
            SpeakerTranscript,
            f"""
                SELECT {_TRANSCRIPT_COLUMNS}
                FROM speakers s
                JOIN speaker_embeddings e ON e.speaker_id = s.id
                JOIN recording_sessions rs ON rs.id = e.session_id
                JOIN pipeline_artifacts t ON t.session_id = e.session_id
                    AND t.pipeline = 'transcribe' AND t.kind = 'transcript'
                    AND t.metadata->>'speaker' = e.speaker
                WHERE {where}
                ORDER BY rs.started_at DESC, t.start_ms, t.id
                LIMIT %s OFFSET %s
            """,
            params + [limit, offset],
        )
