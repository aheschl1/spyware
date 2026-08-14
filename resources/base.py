"""The contract every resource type implements.

This module imports only the standard library and pydantic so the registry is
usable from ``database``, ``services``, ``api`` and ``processing`` without
import cycles.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Literal


class ResourceValidationError(Exception):
    """The chunk's payload or declared attrs violate the resource's contract."""


@dataclass(frozen=True, slots=True)
class ValidatedChunk:
    """What a resource's validator hands back for storage.

    ``payload`` is the parsed JSON document for inline resources and ``None``
    for blob resources (whose bytes are stored verbatim). ``captured_at`` and
    ``duration_ms`` may be derived from the payload when the chunk header
    omitted them.
    """

    attrs: dict[str, Any]
    payload: Any | None
    content_type: str
    captured_at: datetime | None
    duration_ms: int | None


class ResourceType(ABC):
    """One kind of data a session can carry.

    ``name`` is simultaneously the registry key, the wire value in a chunk
    header's ``resource`` field, and the ``resource_segments.resource``
    column value.
    """

    name: ClassVar[str]
    storage: ClassVar[Literal["blob", "inline"]]
    default_content_type: ClassVar[str]

    # Capabilities the API layer interrogates instead of hardcoding "audio".
    renderable: ClassVar[bool] = False  # has a rendered/streamable media form
    stitchable: ClassVar[bool] = False  # whole-session download exists
    wall_clock_queryable: ClassVar[bool] = False  # cross-session captured_at queries
    timeline_events: ClassVar[bool] = False  # raw segments expand to timeline events

    @abstractmethod
    def validate_chunk(
        self,
        payload: bytes,
        *,
        content_type: str | None,
        declared_attrs: Mapping[str, Any],
        captured_at: datetime | None,
        duration_ms: int | None,
    ) -> ValidatedChunk:
        """Check one chunk before anything is written.

        Pure — no I/O. Raises :class:`ResourceValidationError` for a payload
        or attrs the resource cannot accept; the stream reports that as a
        recoverable chunk error.
        """
