"""Response models for the session timeline: what happened, as events.

Every event is one moment on the session timeline (``at_ms``) carrying a
``type`` discriminator; clients must ignore types they do not recognise,
which is how future tiers (diarization, sentiment, summaries, ...) add
events without breaking existing clients.
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TimelineEventBase(BaseModel):
    """Fields every event carries; concrete types add a ``Literal`` ``type``."""

    model_config = ConfigDict(frozen=True)

    at_ms: int = Field(description="Position of this event on the session timeline, ms.")


class ArtifactEventBase(TimelineEventBase):
    """An event derived from one pipeline artifact."""

    artifact_id: UUID = Field(description="The pipeline artifact this event derives from.")


class SessionStartEvent(TimelineEventBase):
    """Opens every timeline at ``at_ms`` 0."""

    type: Literal["session-start"] = "session-start"
    started_at: datetime = Field(
        description="Device-reported wall-clock start; the absolute time behind at_ms 0."
    )


class SessionEndEvent(TimelineEventBase):
    """Closes the timeline of an ended session; an open session has none.

    ``at_ms`` here is the wall-clock duration, which can drift from the
    audio-position time other events use when capture had gaps.
    """

    type: Literal["session-end"] = "session-end"
    ended_at: datetime = Field(description="Wall-clock end of the session.")


class SpeechStartEvent(ArtifactEventBase):
    """Speech was detected starting at this moment."""

    type: Literal["speech-start"] = "speech-start"
    confidence: float | None = Field(
        None, description="Mean VAD frame probability over the span."
    )


class SpeechEndEvent(ArtifactEventBase):
    """The speech span that started last ends here."""

    type: Literal["speech-end"] = "speech-end"
    confidence: float | None = Field(
        None, description="Mean VAD frame probability over the span."
    )


class TranscriptEvent(ArtifactEventBase):
    """What was said over ``[start_ms, end_ms)``; positioned at the span start."""

    type: Literal["transcript"] = "transcript"
    start_ms: int = Field(description="Start of the transcribed span, ms.")
    end_ms: int = Field(description="End (exclusive) of the transcribed span, ms.")
    text: str = Field(description="Transcript text; a preview when ``truncated``.")
    chars: int = Field(description="Full transcript length in characters.")
    truncated: bool = Field(description="True when ``text`` previews a longer transcript.")
    model: str | None = Field(None, description="The transcription model that produced it.")


type TimelineEvent = Annotated[
    SessionStartEvent | SessionEndEvent | SpeechEndEvent | SpeechStartEvent | TranscriptEvent,
    Field(discriminator="type"),
]
