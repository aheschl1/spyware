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
    """Constrained average-linkage clustering; returns clusters of indices.

    NN-chain over cosine distance in O(n²): a cluster is the sum of its
    members' unit vectors, so the mean pairwise distance is exactly
    ``1 − (S_a·S_b)/(n_a·n_b)`` — no distance matrix. ``pins`` maps an index
    to an identity key: same-identity indices are force-merged (must-link)
    and clusters carrying different identities never merge (cannot-link).
    Non-finite vectors carry no geometry: they stay with their pin group if
    pinned and come back as singletons otherwise. Deterministic; the
    threshold cut is inclusive.
    """
    n = len(vectors)
    if n == 0:
        return []
    matrix = np.asarray(vectors, dtype=float)
    finite = np.isfinite(matrix).all(axis=1)
    safe = np.where(finite[:, None], matrix, 0.0)
    norms = np.linalg.norm(safe, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = safe / norms

    groups: dict[Any, list[int]] = {}
    for index in sorted(pins):
        groups.setdefault(pins[index], []).append(index)
    tag_ids = {key: tag for tag, key in enumerate(groups)}
    group_head = {ids[0]: key for key, ids in groups.items()}
    pinned = {index for ids in groups.values() for index in ids}

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for ids in groups.values():
        for other in ids[1:]:
            union(ids[0], other)

    capacity = 2 * n
    dim = matrix.shape[1]
    totals = np.zeros((capacity, dim))
    sizes = np.zeros(capacity)
    tags = np.full(capacity, -1)
    alive = np.zeros(capacity, dtype=bool)
    reps: list[int] = []

    count = 0
    for i in range(n):
        if i in pinned and i not in group_head:
            continue
        if i in group_head:
            ids = groups[group_head[i]]
            fin = [j for j in ids if finite[j]]
            totals[count] = unit[fin].sum(axis=0) if fin else 0.0
            sizes[count] = len(fin)
            tags[count] = tag_ids[group_head[i]]
            alive[count] = bool(fin)
        else:
            totals[count] = unit[i]
            sizes[count] = 1.0
            alive[count] = finite[i]
        reps.append(i)
        count += 1

    chain: list[int] = []
    merges: list[tuple[float, int, int]] = []
    total_clusters = count
    n_active = int(alive[:count].sum())
    while n_active > 1:
        if not chain:
            chain.append(int(np.flatnonzero(alive[:total_clusters])[0]))
        a = chain[-1]
        ids = np.flatnonzero(alive[:total_clusters])
        ids = ids[ids != a]
        dist = np.maximum(
            1.0 - (totals[ids] @ totals[a]) / (sizes[ids] * sizes[a]), 0.0
        )
        if tags[a] != -1:
            dist[(tags[ids] != -1) & (tags[ids] != tags[a])] = np.inf
        low = float(dist.min())
        if not np.isfinite(low):
            break  # every remaining pair is cannot-linked
        b = int(ids[int(np.argmin(dist))])
        if len(chain) >= 2:
            prev = chain[-2]
            # A tie with the previous link must count as reciprocal.
            if dist[int(np.searchsorted(ids, prev))] == low:
                b = prev
        if len(chain) >= 2 and b == chain[-2]:
            chain.pop()
            chain.pop()
            new = total_clusters
            total_clusters += 1
            totals[new] = totals[a] + totals[b]
            sizes[new] = sizes[a] + sizes[b]
            tags[new] = tags[a] if tags[a] != -1 else tags[b]
            alive[a] = False
            alive[b] = False
            alive[new] = True
            reps.append(reps[a])
            merges.append((low, reps[a], reps[b]))
            n_active -= 1
        else:
            chain.append(b)

    # Average linkage is monotone, so the flat cut is every merge at or
    # below the threshold, applied through union-find.
    for height, rep_a, rep_b in merges:
        if height <= threshold:
            union(rep_a, rep_b)

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)
    return list(components.values())
