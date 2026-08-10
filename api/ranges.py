"""HTTP `Range` header parsing (RFC 9110 §14)."""

from dataclasses import dataclass

UNIT = "bytes"


class RangeNotSatisfiable(Exception):
    """The range is well formed but lies outside the representation."""


@dataclass(frozen=True)
class ByteRange:
    """An inclusive byte range, exactly as HTTP counts them."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def content_range(self, size: int) -> str:
        return f"{UNIT} {self.start}-{self.end}/{size}"


def parse_range(header: str | None, size: int) -> ByteRange | None:
    """Resolve a `Range` header against a representation of ``size`` bytes.

    Returns ``None`` when the whole representation should be sent: no header, an
    unsupported unit, a malformed value (RFC 9110 requires an invalid Range to be
    ignored), or a multi-range request, which is answered with the full body.

    Raises :class:`RangeNotSatisfiable` for a well-formed but unmeetable range,
    which the caller answers with 416.
    """
    if not header:
        return None

    unit, _, spec = header.partition("=")
    if unit.strip().lower() != UNIT or not spec.strip():
        return None
    if "," in spec:  # multi-range: send the full representation
        return None

    first, sep, last = spec.strip().partition("-")
    if not sep:
        return None

    try:
        if not first:  # "-N": the final N bytes
            suffix = int(last)
            if suffix <= 0:
                raise RangeNotSatisfiable(header)
            if size == 0:
                raise RangeNotSatisfiable(header)
            return ByteRange(max(0, size - suffix), size - 1)

        start = int(first)
        # An open-ended "N-" runs to the last byte.
        end = int(last) if last else size - 1
    except ValueError:
        return None

    if start < 0 or end < start or start >= size:
        raise RangeNotSatisfiable(header)
    return ByteRange(start, min(end, size - 1))
