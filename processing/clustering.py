"""Constrained agglomerative clustering over cosine distance.

Shared by two tiers at two scales: speaker-cluster runs it over the user's
whole corpus of voice-prints (global identity), and diarize runs it over one
label's per-turn embeddings inside a single block (the purity audit that
splits a label pyannote wrongly gave to several people). Pure numpy — no
database, no I/O.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np


def cluster_corpus(
    vectors: Sequence[Sequence[float]],
    pins: dict[int, Any],
    threshold: float,
) -> list[list[int]]:
    """Constrained agglomerative clustering; returns clusters of indices.

    Average linkage over cosine distance via the Lance–Williams update
    (``d(k, i∪j) = (nᵢ·d(k,i) + nⱼ·d(k,j)) / (nᵢ+nⱼ)`` — the exact mean of
    pairwise point distances). ``pins`` maps an index to an identity key:
    same-identity indices are force-merged before agglomeration begins
    (must-link), and pairs of clusters carrying different identities get an
    infinite distance (cannot-link) — infinity survives the weighted-average
    update, so anything later folded into a pinned cluster inherits its bans.
    Deterministic: ties resolve to the lowest flat matrix index.
    """
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
    tags: list[Any] = [None] * n

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
        members[drop] = []
        if tags[keep] is None:
            tags[keep] = tags[drop]

    groups: dict[Any, list[int]] = {}
    for index in sorted(pins):
        groups.setdefault(pins[index], []).append(index)
    for indices in sorted(groups.values(), key=lambda ids: ids[0]):
        head = indices[0]
        tags[head] = pins[head]
        for other in indices[1:]:
            merge(head, other)
    representatives = [indices[0] for indices in groups.values()]
    for i, a in enumerate(representatives):
        for b in representatives[i + 1 :]:
            dist[a, b] = np.inf
            dist[b, a] = np.inf

    while True:
        flat = int(np.argmin(dist))
        i, j = divmod(flat, n)
        if not dist[i, j] <= threshold:  # also catches all-inf (nothing allowed)
            break
        merge(i, j)
    return [members[k] for k in range(n) if active[k]]
