"""Raw-SQL lexical search over transcript artifacts.

Two rankings over the same rows (``pipeline='transcribe', kind='transcript'``,
text in ``metadata->>'text'``):

- **strict** — Postgres FTS: ``websearch_to_tsquery`` (Google-style syntax:
  words AND, ``or``, ``-not``, quoted phrases become positional adjacency)
  matched via ``@@`` against the stemmed ``to_tsvector``, served by the
  partial GIN index from migration 0009, ranked by cover density.
- **fuzzy** — trigram ``word_similarity`` over the raw text, the fallback
  when strict finds nothing: an ASR-mangled or misspelled word still shares
  most of its trigrams.

Snippets come from ``ts_headline`` with bracket delimiters the API layer
splits into typed segments — markers never reach a browser as HTML.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from database.repos.base import BaseRepo

# ts_headline trims the matched region to a window; the raw utterance stays
# available in `text` for the fuzzy path and any consumer that wants it all.
_HEADLINE_OPTIONS = "StartSel=[[, StopSel=]], MaxWords=30, MinWords=10"

# Below this trigram word-similarity the "match" is noise, not a close
# spelling. 0.35 keeps one-edit words and drops coincidental overlap.
_FUZZY_FLOOR = 0.35


class TranscriptHit(BaseModel):
    """One utterance matching the query."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID
    session_id: UUID
    start_ms: int
    end_ms: int
    text: str
    snippet: str  # [[..]]-delimited for strict hits; the raw text for fuzzy
    score: float
    metadata: dict[str, Any]
    links: dict[str, Any] = {}
    created_at: datetime


class TranscriptSearchRepo(BaseRepo):
    async def search(
        self,
        *,
        user_id: UUID,
        q: str,
        session_id: UUID | None = None,
        limit: int = 20,
    ) -> list[TranscriptHit]:
        """Strict full-text matches, best cover-density rank first.

        The tsvector expression and partial-index predicate repeat migration
        0009's exactly — that identity is what lets the GIN index serve this.
        ts_rank_cd flag 2 length-normalizes so long utterances can't win on
        repetition alone.
        """
        conditions = ["s.user_id = %s"]
        params: list = [q, user_id]
        if session_id is not None:
            conditions.append("a.session_id = %s")
            params.append(session_id)
        params.append(limit)
        return await self._fetch_all(
            TranscriptHit,
            f"""
                SELECT a.id AS artifact_id, a.session_id, a.start_ms, a.end_ms,
                       a.metadata->>'text' AS text,
                       ts_headline('english', a.metadata->>'text', query,
                                   '{_HEADLINE_OPTIONS}') AS snippet,
                       ts_rank_cd(to_tsvector('english', a.metadata->>'text'),
                                  query, 2) AS score,
                       a.metadata, a.links, a.created_at
                FROM pipeline_artifacts a
                JOIN recording_sessions s ON s.id = a.session_id,
                     websearch_to_tsquery('english', %s) query
                WHERE a.pipeline = 'transcribe' AND a.kind = 'transcript'
                  AND to_tsvector('english', a.metadata->>'text') @@ query
                  AND {" AND ".join(conditions)}
                ORDER BY score DESC, a.session_id, a.start_ms
                LIMIT %s
            """,
            params,
        )

    async def search_fuzzy(
        self,
        *,
        user_id: UUID,
        q: str,
        session_id: UUID | None = None,
        limit: int = 20,
    ) -> list[TranscriptHit]:
        """Trigram close-spelling matches, most similar first.

        word_similarity compares the query against the best-matching *word
        window* of the text (not the whole utterance), which is what makes a
        single misheard word findable inside a long sentence.
        """
        conditions = ["s.user_id = %s"]
        params: list = [q, user_id]
        if session_id is not None:
            conditions.append("a.session_id = %s")
            params.append(session_id)
        params += [q, _FUZZY_FLOOR, limit]
        return await self._fetch_all(
            TranscriptHit,
            f"""
                SELECT a.id AS artifact_id, a.session_id, a.start_ms, a.end_ms,
                       a.metadata->>'text' AS text,
                       a.metadata->>'text' AS snippet,
                       word_similarity(%s, a.metadata->>'text') AS score,
                       a.metadata, a.links, a.created_at
                FROM pipeline_artifacts a
                JOIN recording_sessions s ON s.id = a.session_id
                WHERE a.pipeline = 'transcribe' AND a.kind = 'transcript'
                  AND {" AND ".join(conditions)}
                  AND word_similarity(%s, a.metadata->>'text') >= %s
                ORDER BY score DESC, a.session_id, a.start_ms
                LIMIT %s
            """,
            params,
        )
