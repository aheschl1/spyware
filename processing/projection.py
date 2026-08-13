"""PCA of voice-prints down to three dimensions, for the speaker map.

Pure numpy — no database, no I/O, same charter as ``clustering.py``.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

COMPONENTS = 3


@dataclass(frozen=True, slots=True)
class Projection:
    """A fitted basis plus the coordinates of the rows it was fitted on."""

    coords: np.ndarray  # (n, COMPONENTS), row-aligned with the input
    components: np.ndarray  # (COMPONENTS, d), orthonormal, sign-fixed
    mean: np.ndarray  # (d,)
    explained_variance_ratio: np.ndarray  # (COMPONENTS,)
    basis_id: str


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _fix_signs(components: np.ndarray) -> np.ndarray:
    """Pin each component's arbitrary sign to a convention.

    ``sign(Σ vⱼ|vⱼ|)`` is continuous in the loadings, so it survives small
    data changes. sklearn's ``svd_flip`` (sign of the largest-|·| entry) is
    not: two near-equal loadings mean one new row can swap the argmax and
    mirror the whole plot. The argmax is kept only as the exact-zero tiebreak.
    """
    if components.size == 0:
        return components
    skew = (components * np.abs(components)).sum(axis=1)
    rows = np.arange(len(components))
    largest = components[rows, np.argmax(np.abs(components), axis=1)]
    flip = np.where(skew != 0.0, np.sign(skew), np.sign(largest))
    flip[flip == 0.0] = 1.0
    return components * flip[:, None]


def _basis_id(mean: np.ndarray, components: np.ndarray) -> str:
    payload = np.round(np.concatenate([mean, components.ravel()]), 9) + 0.0
    return hashlib.sha256(payload.tobytes()).hexdigest()[:16]


def fit_projection(vectors: Sequence[Sequence[float]], *, dim: int = 0) -> Projection:
    """Fit a PCA basis over ``vectors`` and project them into it.

    ``dim`` is only needed to shape the empty case; otherwise it is inferred.
    """
    matrix = np.asarray(vectors, dtype=float)
    if matrix.size == 0:
        width = dim or 1
        empty = np.zeros((0, COMPONENTS))
        components = np.zeros((COMPONENTS, width))
        mean = np.zeros(width)
        return Projection(
            empty, components, mean, np.zeros(COMPONENTS), _basis_id(mean, components)
        )

    matrix = _unit(matrix)
    mean = matrix.mean(axis=0)
    # Unit vectors sit on a cone, not a ball: uncentered, PC1 is just the mean
    # direction and every point lands on top of every other.
    centered = matrix - mean
    n, width = centered.shape
    denom = max(n - 1, 1)
    # Trace, not the sum of eigenvalues: keeps the ratios exact when only the
    # top components are taken from a covariance whose tail is numerical noise.
    total = float((centered * centered).sum() / denom)

    # eigh on the (d, d) covariance beats SVD of the (n, d) matrix at every
    # n >= d, widening with n (measured: 4.0ms vs 14.7ms at n=404; 19ms vs
    # 605ms at n=50k). There is no crossover to branch on.
    covariance = (centered.T @ centered) / denom
    values, vectors_ = np.linalg.eigh(covariance)
    keep = min(COMPONENTS, width, max(n - 1, 1))
    order = np.argsort(values)[::-1][:keep]
    variance = np.clip(values[order], 0.0, None)  # eigh emits ~-1e-17 on rank loss
    components = _fix_signs(vectors_[:, order].T)

    coords = centered @ components.T
    ratio = variance / total if total > 0.0 else np.zeros(keep)

    pad = COMPONENTS - keep
    if pad:
        coords = np.hstack([coords, np.zeros((n, pad))])
        components = np.vstack([components, np.zeros((pad, width))])
        ratio = np.concatenate([ratio, np.zeros(pad)])

    return Projection(coords, components, mean, ratio, _basis_id(mean, components))


def project(basis: Projection, vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Place out-of-sample rows in an already-fitted basis."""
    matrix = np.asarray(vectors, dtype=float)
    if matrix.size == 0:
        return np.zeros((0, COMPONENTS))
    return (_unit(matrix) - basis.mean) @ basis.components.T
