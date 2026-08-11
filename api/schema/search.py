"""Response models for the two audio searches: contrastive (CLAP embedding
similarity over free text) and tag filtering (exact class scores from the
tagger)."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api.schema.timeline import AudioTagLabel
from database.repos.embeddings import AudioSearchHit
from database.repos.tags import TagLabelCount, TagWindowHit
from database.repos.transcripts import TranscriptHit


class AudioSearchRead(BaseModel):
    """One audio window matching the query, closest first."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID = Field(description="The window's ``audio-tag`` artifact.")
    session_id: UUID
    start_ms: int = Field(description="Start of the matching window, ms.")
    end_ms: int = Field(description="End (exclusive) of the matching window, ms.")
    distance: float = Field(
        description="Cosine distance between the query and the window's audio "
        "embedding (0 = identical direction; lower is closer)."
    )
    labels: tuple[AudioTagLabel, ...] = Field(
        description="The window's tag scores, for orientation alongside the match."
    )

    @classmethod
    def from_model(cls, hit: AudioSearchHit) -> "AudioSearchRead":
        return cls(
            artifact_id=hit.artifact_id,
            session_id=hit.session_id,
            start_ms=hit.start_ms,
            end_ms=hit.end_ms,
            distance=hit.distance,
            labels=tuple(hit.metadata.get("labels", ())),
        )


class AudioSearchResponse(BaseModel):
    """The ranked matches plus the query's embedding model, for provenance."""

    model_config = ConfigDict(frozen=True)

    query: str
    model: str = Field(description="The embedding model that encoded the query.")
    items: list[AudioSearchRead]


class TagSearchRead(BaseModel):
    """One window where the class scored at or above the floor."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID = Field(description="The window's ``audio-tag`` artifact.")
    session_id: UUID
    start_ms: int
    end_ms: int
    label: str = Field(description="The matched class (canonical AudioSet name).")
    score: float = Field(description="The tagger's sigmoid score for that class.")
    labels: tuple[AudioTagLabel, ...] = Field(
        description="The window's full tag list, for context."
    )

    @classmethod
    def from_model(cls, hit: TagWindowHit) -> "TagSearchRead":
        return cls(
            artifact_id=hit.artifact_id,
            session_id=hit.session_id,
            start_ms=hit.start_ms,
            end_ms=hit.end_ms,
            label=hit.label,
            score=hit.score,
            labels=tuple(hit.metadata.get("labels", ())),
        )


class TagSearchResponse(BaseModel):
    """Windows matching the class filter, best score first."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(description="The filter as given (matched as substring).")
    min_score: float
    items: list[TagSearchRead]


class TagLabelRead(BaseModel):
    """One class present in the caller's windows."""

    model_config = ConfigDict(frozen=True)

    label: str
    windows: int = Field(description="Windows where the class scored above the store floor.")
    best: float = Field(description="The class's highest score anywhere.")

    @classmethod
    def from_model(cls, row: TagLabelCount) -> "TagLabelRead":
        return cls(label=row.label, windows=row.windows, best=row.best)


class TagLabelsResponse(BaseModel):
    """Every class heard in the caller's audio, most frequent first."""

    model_config = ConfigDict(frozen=True)

    labels: list[TagLabelRead]


class SnippetSegment(BaseModel):
    """One run of snippet text; ``match`` marks the highlighted query terms."""

    model_config = ConfigDict(frozen=True)

    text: str
    match: bool


def split_snippet(snippet: str) -> list[SnippetSegment]:
    """ts_headline's ``[[``/``]]`` delimiters -> typed segments.

    Done server-side so no marker syntax (or anything resembling markup)
    crosses the API; the client renders segments, never parses strings.
    """
    segments: list[SnippetSegment] = []
    for i, outer in enumerate(snippet.split("[[")):
        if i == 0:
            if outer:
                segments.append(SnippetSegment(text=outer, match=False))
            continue
        match, _, rest = outer.partition("]]")
        if match:
            segments.append(SnippetSegment(text=match, match=True))
        if rest:
            segments.append(SnippetSegment(text=rest, match=False))
    return segments


class TranscriptSearchRead(BaseModel):
    """One utterance matching the transcript query."""

    model_config = ConfigDict(frozen=True)

    artifact_id: UUID = Field(description="The utterance's ``transcript`` artifact.")
    session_id: UUID
    start_ms: int
    end_ms: int
    speaker: str | None = Field(None, description="Diarized speaker label (block-namespaced).")
    score: float = Field(
        description="Cover-density rank (strict) or trigram word-similarity "
        "(fuzzy); comparable within one response only."
    )
    segments: tuple[SnippetSegment, ...] = Field(
        description="The snippet, split into plain and matched runs."
    )

    @classmethod
    def from_model(cls, hit: TranscriptHit, *, fuzzy: bool) -> "TranscriptSearchRead":
        return cls(
            artifact_id=hit.artifact_id,
            session_id=hit.session_id,
            start_ms=hit.start_ms,
            end_ms=hit.end_ms,
            speaker=hit.metadata.get("speaker"),
            score=hit.score,
            segments=tuple(
                [SnippetSegment(text=hit.snippet, match=False)]
                if fuzzy
                else split_snippet(hit.snippet)
            ),
        )


class TranscriptSearchResponse(BaseModel):
    """Matching utterances, best first."""

    model_config = ConfigDict(frozen=True)

    query: str
    fuzzy: bool = Field(
        description="True when strict full-text matching found nothing and "
        "these are trigram close-spelling matches instead."
    )
    items: list[TranscriptSearchRead]
