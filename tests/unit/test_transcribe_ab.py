"""Pure helpers of the transcribe-ab tier: block sub-splitting and
word→utterance assignment."""

from types import SimpleNamespace

from processing.pipelines.transcribe_ab import _blocks, assign_words, sub_spans


def _utterance(id, start_ms, end_ms, **metadata):
    return SimpleNamespace(id=id, start_ms=start_ms, end_ms=end_ms, metadata=metadata)


def word(w, s, e):
    return {"w": w, "s": s, "e": e}


def test_sub_spans_splits_at_the_cap():
    assert sub_spans(0, 100, cap_ms=100) == [(0, 100)]
    assert sub_spans(0, 250, cap_ms=100) == [(0, 100), (100, 200), (200, 250)]
    assert sub_spans(500, 600, cap_ms=1000) == [(500, 600)]


def test_assign_words_by_midpoint_half_open():
    utterances = [_utterance("a", 0, 150), _utterance("b", 150, 300)]
    words = [
        word("early", 0, 100),     # mid 50 -> a
        word("edge", 100, 200),    # mid 150 -> b (half-open)
        word("late", 200, 280),    # mid 240 -> b
        word("outside", 400, 500), # nobody
    ]
    assigned = assign_words(words, utterances)
    assert [w["w"] for w in assigned["a"]] == ["early"]
    assert [w["w"] for w in assigned["b"]] == ["edge", "late"]


def test_assign_words_crosstalk_lands_in_every_covering_span():
    utterances = [_utterance("a", 0, 200), _utterance("b", 100, 300)]
    assigned = assign_words([word("both", 140, 160)], utterances)
    assert [w["w"] for w in assigned["a"]] == ["both"]
    assert [w["w"] for w in assigned["b"]] == ["both"]


def test_blocks_group_by_metadata_and_degrade_without_it():
    with_block = _utterance("a", 0, 150, block_start_ms=0, block_end_ms=300)
    sibling = _utterance("b", 150, 300, block_start_ms=0, block_end_ms=300)
    legacy = _utterance("c", 400, 500)
    groups = _blocks([with_block, sibling, legacy])
    assert set(groups) == {(0, 300), (400, 500)}
    assert [u.id for u in groups[(0, 300)]] == ["a", "b"]
    assert [u.id for u in groups[(400, 500)]] == ["c"]
