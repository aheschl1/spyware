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
    # A zero vector sits at distance exactly 1.0 from everything.
    clusters = cluster_corpus([[0.0, 0.0], [1.0, 0.0]], {}, 0.5)
    assert sorted(len(c) for c in clusters) == [1, 1]


def test_nan_vector_stays_singleton() -> None:
    vectors = [
        [1.0, 0.0, 0.0],
        [0.99, 0.14, 0.0],
        [float("nan"), 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.99, 0.0],
    ]
    assert _sets(cluster_corpus(vectors, {}, 0.65)) == {
        frozenset({0, 1}),
        frozenset({2}),
        frozenset({3, 4}),
    }


def test_nan_vector_follows_its_pin() -> None:
    vectors = [[1.0, 0.0], [float("inf"), 0.0], [0.0, 1.0]]
    clusters = _sets(cluster_corpus(vectors, {0: "X", 1: "X"}, 0.1))
    assert frozenset({0, 1}) in clusters


def test_all_nan_corpus() -> None:
    vectors = [[float("nan"), 0.0], [float("nan"), 1.0]]
    assert _sets(cluster_corpus(vectors, {}, 2.0)) == {
        frozenset({0}),
        frozenset({1}),
    }


def _naive_reference(
    vectors: list[list[float]], pins: dict[int, str], threshold: float
) -> list[list[int]]:
    """The pre-NN-chain O(n³) implementation, kept as a parity oracle."""
    import numpy as np

    n = len(vectors)
    if n == 0:
        return []
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = matrix / norms
    dist = np.maximum(1.0 - unit @ unit.T, 0.0)
    np.fill_diagonal(dist, np.inf)
    size = np.ones(n)
    active = np.ones(n, dtype=bool)
    members: list[list[int]] = [[i] for i in range(n)]
    tags: list[str | None] = [None] * n

    def merge(keep: int, drop: int) -> None:
        combined = (size[keep] * dist[keep] + size[drop] * dist[drop]) / (
            size[keep] + size[drop]
        )
        dist[keep, :] = combined
        dist[:, keep] = combined
        dist[keep, keep] = np.inf
        dist[drop, :] = np.inf
        dist[:, drop] = np.inf
        size[keep] += size[drop]
        active[drop] = False
        members[keep].extend(members[drop])
        if tags[keep] is None:
            tags[keep] = tags[drop]

    groups: dict[str, list[int]] = {}
    for index in sorted(pins):
        groups.setdefault(pins[index], []).append(index)
    for indices in sorted(groups.values(), key=lambda ids: ids[0]):
        tags[indices[0]] = pins[indices[0]]
        for other in indices[1:]:
            merge(indices[0], other)
    representatives = [indices[0] for indices in groups.values()]
    for i, a in enumerate(representatives):
        for b in representatives[i + 1 :]:
            dist[a, b] = np.inf
            dist[b, a] = np.inf

    while True:
        flat = int(np.argmin(dist))
        i, j = divmod(flat, n)
        if not dist[i, j] <= threshold:
            break
        merge(i, j)
    return [members[k] for k in range(n) if active[k]]


def test_parity_with_naive_reference() -> None:
    import numpy as np

    rng = np.random.default_rng(7)
    for trial in range(5):
        centers = rng.normal(size=(4, 8))
        vectors = [
            (centers[rng.integers(4)] + rng.normal(scale=0.15, size=8)).tolist()
            for _ in range(60)
        ]
        pins = {0: "A", 7: "A", 13: "B"} if trial % 2 else {}
        for threshold in (0.3, 0.5, 0.65, 0.9):
            ours = _sets(cluster_corpus(vectors, pins, threshold))
            oracle = _sets(_naive_reference(vectors, pins, threshold))
            assert ours == oracle, (trial, threshold)


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
