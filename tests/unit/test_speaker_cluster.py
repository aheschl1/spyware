"""The constrained agglomerative clusterer as a pure function: geometry,
the threshold boundary, must-link/cannot-link pins, and determinism."""

from processing.pipelines.speaker_cluster import cluster_corpus


def _sets(clusters: list[list[int]]) -> set[frozenset[int]]:
    return {frozenset(cluster) for cluster in clusters}


def test_two_tight_groups_split() -> None:
    vectors = [
        [1.0, 0.0, 0.0],
        [0.99, 0.14, 0.0],
        [0.98, 0.17, 0.1],
        [0.0, 1.0, 0.0],
        [0.1, 0.99, 0.0],
        [0.0, 0.98, 0.17],
    ]
    assert _sets(cluster_corpus(vectors, {}, 0.65)) == {
        frozenset({0, 1, 2}),
        frozenset({3, 4, 5}),
    }


def test_threshold_boundary_is_inclusive() -> None:
    # Orthogonal unit vectors sit at cosine distance exactly 1.0.
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    assert len(cluster_corpus(vectors, {}, 1.0)) == 1
    assert len(cluster_corpus(vectors, {}, 0.999)) == 2


def test_average_linkage_uses_mean_pairwise_distance() -> None:
    # Three points on a line of angles: 0°, 60°, 90°. Distances: d(0,60)=0.5,
    # d(60,90)≈0.134, d(0,90)=1.0. At threshold 0.6 the 60/90 pair merges
    # first; the average distance from {60,90} to {0} is (0.5+1.0)/2 = 0.75
    # > 0.6, so single-linkage-style chaining must NOT pull 0 in.
    import math

    def at(deg: float) -> list[float]:
        return [math.cos(math.radians(deg)), math.sin(math.radians(deg))]

    clusters = _sets(cluster_corpus([at(0), at(60), at(90)], {}, 0.6))
    assert clusters == {frozenset({0}), frozenset({1, 2})}


def test_pins_force_merge_distant_points() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.01]]
    clusters = _sets(cluster_corpus(vectors, {0: "X", 1: "X"}, 0.05))
    # 0 and 1 are orthogonal but share an identity: pre-merged. 2 is near 0,
    # but the average distance to the fused pair exceeds the tight threshold.
    assert frozenset({0, 1}) in clusters


def test_different_pins_never_merge() -> None:
    vectors = [[1.0, 0.0], [1.0, 0.001]]
    clusters = _sets(cluster_corpus(vectors, {0: "X", 1: "Y"}, 2.0))
    assert clusters == {frozenset({0}), frozenset({1})}


def test_cannot_link_survives_absorbing_neighbours() -> None:
    # Two pinned identities, each with an unpinned point nearby. The
    # unpinned points fold into their pinned neighbours, and the resulting
    # enlarged clusters still refuse to merge at any threshold.
    vectors = [
        [1.0, 0.0, 0.0],
        [0.99, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.99, 0.0],
    ]
    clusters = _sets(cluster_corpus(vectors, {0: "A", 2: "B"}, 2.0))
    assert clusters == {frozenset({0, 1}), frozenset({2, 3})}


def test_empty_and_single() -> None:
    assert cluster_corpus([], {}, 0.5) == []
    assert cluster_corpus([[1.0, 0.0]], {}, 0.5) == [[0]]


def test_zero_vector_is_harmless() -> None:
    clusters = cluster_corpus([[0.0, 0.0], [1.0, 0.0]], {}, 0.5)
    assert sorted(len(c) for c in clusters) in ([1, 1], [2])


def test_deterministic() -> None:
    vectors = [
        [1.0, 0.0, 0.0],
        [0.9, 0.3, 0.1],
        [0.0, 1.0, 0.0],
        [0.2, 0.9, 0.0],
        [0.5, 0.5, 0.7],
    ]
    first = cluster_corpus(vectors, {1: "X"}, 0.7)
    for _ in range(5):
        assert cluster_corpus(vectors, {1: "X"}, 0.7) == first
