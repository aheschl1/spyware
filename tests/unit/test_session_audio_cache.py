"""The stitched-audio representation cache: hits, invalidation, and eviction."""

from uuid import uuid4

from services import stitch
from api.session_audio import SessionAudio, _RepresentationCache
from database.schema.segments import SegmentSetFingerprint


def _fingerprint(count: int, total_bytes: int = 0, max_sequence: int = 0) -> SegmentSetFingerprint:
    return SegmentSetFingerprint(
        count=count, max_sequence=max_sequence or count - 1, total_bytes=total_bytes or count
    )


def _audio(pieces: int) -> SessionAudio:
    plan = stitch.StitchPlan(
        pieces=tuple(
            stitch.Piece(object_key=f"k/{i}", start=i, data_length=1) for i in range(pieces)
        ),
        total_size=stitch.WAV_HEADER_BYTES + pieces,
    )
    return SessionAudio(etag=f'"{pieces}"', header=b"RIFF", plan=plan)


def test_hit_returns_cached_representation() -> None:
    cache = _RepresentationCache(piece_budget=100)
    session_id, fingerprint, audio = uuid4(), _fingerprint(3), _audio(3)

    assert cache.get(session_id, fingerprint) is None  # cold
    cache.put(session_id, fingerprint, audio)
    assert cache.get(session_id, fingerprint) is audio  # warm


def test_changed_fingerprint_misses() -> None:
    cache = _RepresentationCache(piece_budget=100)
    session_id = uuid4()
    cache.put(session_id, _fingerprint(3), _audio(3))

    # A new segment moved the fingerprint: the stale entry must not be served.
    assert cache.get(session_id, _fingerprint(4)) is None


def test_budget_evicts_least_recently_used() -> None:
    cache = _RepresentationCache(piece_budget=10)
    a, b, c = uuid4(), uuid4(), uuid4()
    cache.put(a, _fingerprint(4), _audio(4))
    cache.put(b, _fingerprint(4), _audio(4))
    # Touch `a` so `b` becomes the least recently used.
    assert cache.get(a, _fingerprint(4)) is not None
    cache.put(c, _fingerprint(4), _audio(4))  # 12 pieces > budget of 10

    assert cache.get(a, _fingerprint(4)) is not None
    assert cache.get(c, _fingerprint(4)) is not None
    assert cache.get(b, _fingerprint(4)) is None  # evicted


def test_session_larger_than_budget_is_not_cached() -> None:
    cache = _RepresentationCache(piece_budget=10)
    small, huge = uuid4(), uuid4()
    cache.put(small, _fingerprint(5), _audio(5))
    cache.put(huge, _fingerprint(50), _audio(50))  # exceeds the whole budget

    assert cache.get(huge, _fingerprint(50)) is None  # never cached
    assert cache.get(small, _fingerprint(5)) is not None  # and did not evict the small one


def test_reput_replaces_and_reaccounts_pieces() -> None:
    cache = _RepresentationCache(piece_budget=10)
    session_id = uuid4()
    cache.put(session_id, _fingerprint(8), _audio(8))
    # Same session grows; the old 8 pieces must be discounted, not double-counted,
    # or a second small session would be wrongly evicted.
    cache.put(session_id, _fingerprint(9), _audio(9))
    other = uuid4()
    cache.put(other, _fingerprint(1), _audio(1))

    assert cache.get(session_id, _fingerprint(9)) is not None
    assert cache.get(other, _fingerprint(1)) is not None
