"""The PCA as a pure function: geometry, the sign convention, determinism,
and the degenerate corpora the endpoint must not 500 on."""

import numpy as np
import pytest

from processing.projection import COMPONENTS, fit_projection, project


def _voices(n: int, groups: int = 4, spread: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((groups, 32))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    idx = rng.integers(0, groups, n)
    matrix = centers[idx] + spread * rng.standard_normal((n, 32))
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_coords_are_centered() -> None:
    basis = fit_projection(_voices(120))
    assert np.allclose(basis.coords.mean(axis=0), 0.0, atol=1e-9)


def test_components_are_orthonormal() -> None:
    basis = fit_projection(_voices(120))
    gram = basis.components @ basis.components.T
    assert np.allclose(gram, np.eye(COMPONENTS), atol=1e-9)


def test_two_groups_separate_on_pc1() -> None:
    left = _voices(60, groups=1, seed=1)
    right = _voices(60, groups=1, seed=2)
    basis = fit_projection(np.vstack([left, right]))
    assert basis.explained_variance_ratio[0] > 0.5
    assert basis.coords[:60, 0].mean() * basis.coords[60:, 0].mean() < 0


def test_matches_an_svd_reference() -> None:
    """The eigh-on-covariance shortcut must agree with textbook PCA.

    This is what makes the optimization safe to keep — and safe to revert.
    """
    matrix = _voices(200)
    basis = fit_projection(matrix)

    centered = matrix - matrix.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2 / (len(matrix) - 1)
    expected = variance[:COMPONENTS] / variance.sum()

    assert np.allclose(basis.explained_variance_ratio, expected, atol=1e-9)
    for i in range(COMPONENTS):
        # Same axis up to the sign convention, which the reference lacks.
        assert abs(float(basis.components[i] @ vt[i])) == pytest.approx(1.0)


def test_repeat_fits_are_bitwise_identical() -> None:
    matrix = _voices(150)
    first, second = fit_projection(matrix), fit_projection(matrix)
    assert np.array_equal(first.coords, second.coords)
    assert first.basis_id == second.basis_id


def test_basis_ignores_row_order() -> None:
    matrix = _voices(150)
    shuffled = matrix[np.random.default_rng(7).permutation(len(matrix))]
    assert fit_projection(matrix).basis_id == fit_projection(shuffled).basis_id


def test_signs_survive_a_small_perturbation() -> None:
    """The reason the convention is sign(Σ v|v|) and not argmax-of-|v|:
    one extra row must not mirror the plot."""
    matrix = _voices(200)
    grown = np.vstack([matrix, _voices(1, seed=99)])
    base, after = fit_projection(matrix), fit_projection(grown)
    assert np.all(np.sign((base.components * after.components).sum(axis=1)) > 0)


def test_ratios_are_descending_and_bounded() -> None:
    ratio = fit_projection(_voices(120)).explained_variance_ratio
    assert np.all(np.diff(ratio) <= 1e-12)
    assert np.all(ratio >= 0.0) and ratio.sum() <= 1.0 + 1e-9


def test_cluster_marker_is_the_mean_of_member_coords() -> None:
    """Why the endpoint pools member coords for a cluster marker.

    Projection is affine, so pooling *is* the projection of the raw member
    mean — exactly, for free. Routing the stored centroid through
    ``project`` instead would re-normalize it and drift.
    """
    matrix = _voices(90)
    basis = fit_projection(matrix)
    centroid = matrix[:30].mean(axis=0)  # what SQL avg(embedding) returns
    pooled = basis.coords[:30].mean(axis=0)

    assert np.allclose((centroid - basis.mean) @ basis.components.T, pooled, atol=1e-12)
    assert not np.allclose(project(basis, [centroid])[0], pooled, atol=1e-3)


def test_project_reproduces_fitted_coords() -> None:
    matrix = _voices(80)
    basis = fit_projection(matrix)
    assert np.allclose(project(basis, matrix), basis.coords, atol=1e-9)


def test_empty_corpus() -> None:
    basis = fit_projection([], dim=32)
    assert basis.coords.shape == (0, COMPONENTS)
    assert basis.explained_variance_ratio.tolist() == [0.0] * COMPONENTS


def test_single_row_lands_at_the_origin() -> None:
    basis = fit_projection([[1.0] + [0.0] * 31])
    assert np.allclose(basis.coords, 0.0)
    assert basis.explained_variance_ratio.tolist() == [0.0] * COMPONENTS


def test_two_rows_have_rank_one() -> None:
    basis = fit_projection([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert basis.explained_variance_ratio[0] > 0.99
    assert basis.explained_variance_ratio[1] == 0.0
    assert basis.coords.shape == (2, COMPONENTS)


def test_identical_rows_give_zeros_not_nan() -> None:
    basis = fit_projection([[1.0, 0.0, 0.0]] * 5)
    assert np.all(np.isfinite(basis.coords))
    assert basis.explained_variance_ratio.tolist() == [0.0] * COMPONENTS


def test_zero_vector_is_harmless() -> None:
    basis = fit_projection([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.all(np.isfinite(basis.coords))


def test_low_dimensional_input_pads_components() -> None:
    basis = fit_projection([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    assert basis.coords.shape == (3, COMPONENTS)
    assert basis.components.shape == (COMPONENTS, 2)
    assert basis.explained_variance_ratio[2] == 0.0


def test_basis_id_moves_with_the_data() -> None:
    assert fit_projection(_voices(50, seed=1)).basis_id != (
        fit_projection(_voices(50, seed=2)).basis_id
    )
