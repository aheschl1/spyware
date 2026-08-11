"""The ts_headline delimiter split: markers -> typed segments, no HTML."""

from api.schema.search import SnippetSegment, split_snippet


def _flat(segments: list[SnippetSegment]) -> list[tuple[str, bool]]:
    return [(s.text, s.match) for s in segments]


def test_single_match() -> None:
    assert _flat(split_snippet("hello from the [[stub]] transcriber")) == [
        ("hello from the ", False),
        ("stub", True),
        (" transcriber", False),
    ]


def test_multiple_matches() -> None:
    assert _flat(split_snippet("[[hello]] from the [[stub]]")) == [
        ("hello", True),
        (" from the ", False),
        ("stub", True),
    ]


def test_no_matches() -> None:
    assert _flat(split_snippet("plain text")) == [("plain text", False)]


def test_adjacent_matches() -> None:
    assert _flat(split_snippet("[[a]][[b]]")) == [("a", True), ("b", True)]


def test_empty() -> None:
    assert split_snippet("") == []


def test_reassembles_to_original_text() -> None:
    snippet = "say [[invoice]] twice: [[invoice]]!"
    joined = "".join(s.text for s in split_snippet(snippet))
    assert joined == snippet.replace("[[", "").replace("]]", "")
