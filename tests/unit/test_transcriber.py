"""Response parsing for the transcriber seam: word timestamps and language
are optional extensions — a text-only backend must keep working, and a
malformed extension degrades to None rather than failing the transcription."""

from processing.transcriber import Word, _parse_words


def test_well_formed_words_parse():
    body = {
        "text": "hi there",
        "words": [
            {"word": "hi", "start_ms": 0, "end_ms": 200},
            {"word": "there", "start_ms": 250, "end_ms": 600},
        ],
    }
    assert _parse_words(body) == (
        Word(word="hi", start_ms=0, end_ms=200),
        Word(word="there", start_ms=250, end_ms=600),
    )


def test_absent_words_are_none():
    assert _parse_words({"text": "hi"}) is None


def test_empty_words_list_is_empty_tuple():
    assert _parse_words({"words": []}) == ()


def test_malformed_words_degrade_to_none():
    for words in (
        "not a list",
        ["not a dict"],
        [{"word": "hi", "start_ms": 0}],
        [{"word": "hi", "start_ms": 0.5, "end_ms": 200}],
        [{"word": 7, "start_ms": 0, "end_ms": 200}],
    ):
        assert _parse_words({"words": words}) is None, words
