"""`Range` header parsing. Pure logic, so no containers."""

import pytest

from api.ranges import ByteRange, RangeNotSatisfiable, parse_range

SIZE = 1000


@pytest.mark.parametrize(
    "header,expected",
    [
        (None, None),
        ("", None),
        ("bytes=0-99", ByteRange(0, 99)),
        ("bytes=100-", ByteRange(100, 999)),
        ("bytes=-200", ByteRange(800, 999)),
        ("bytes=0-0", ByteRange(0, 0)),
        ("bytes=999-999", ByteRange(999, 999)),
        ("bytes=0-99999", ByteRange(0, 999)),  # end clamped to the last byte
        ("bytes=-5000", ByteRange(0, 999)),  # suffix longer than the object
        ("BYTES=0-9", ByteRange(0, 9)),  # unit is case-insensitive
        ("bytes=abc-def", None),  # malformed: ignored, not rejected
        ("bytes=0-99,200-299", None),  # multi-range: whole representation
        ("items=0-99", None),  # unsupported unit
        ("garbage", None),
        ("bytes=", None),
    ],
)
def test_parse_range(header: str | None, expected: ByteRange | None) -> None:
    assert parse_range(header, SIZE) == expected


@pytest.mark.parametrize("header", ["bytes=1000-", "bytes=1000-1001", "bytes=-0", "bytes=500-400"])
def test_unsatisfiable(header: str) -> None:
    with pytest.raises(RangeNotSatisfiable):
        parse_range(header, SIZE)


def test_empty_representation_cannot_satisfy_a_suffix() -> None:
    with pytest.raises(RangeNotSatisfiable):
        parse_range("bytes=-10", 0)


def test_byte_range_reports_length_and_content_range() -> None:
    byte_range = ByteRange(10, 19)
    assert byte_range.length == 10
    assert byte_range.content_range(SIZE) == "bytes 10-19/1000"
