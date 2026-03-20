"""
Basic smoke test for GeomNMF and barplot_Phi.

Generates synthetic data from a known K=3 simplex mixing model,
runs GeomNMF, and plots the estimated source attribution matrix.
"""

import numpy as np
import pytest
from geom_nmf import GeomNMF, barplot_Phi, permute_to_reference

SEED = 123
K, J, N = 3, 8, 500

@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(SEED)
    H_true = rng.dirichlet(np.ones(J), size=K)   # (K, J) row-stochastic endmembers
    W_true = rng.dirichlet(np.ones(K), size=N)    # (N, K) row-stochastic weights
    Y = W_true @ H_true                            # (N, J) observed data
    return Y, H_true, W_true


@pytest.fixture
def geom_nmf_result(synthetic_data):
    Y, _, _ = synthetic_data
    return GeomNMF(Y, K=K, seed=SEED, verbose=False)


def test_result_attributes(geom_nmf_result):
    result = geom_nmf_result
    assert hasattr(result, "H_star")
    assert hasattr(result, "Phi")
    assert hasattr(result, "mu_tilde")
    assert hasattr(result, "logvol")
    assert hasattr(result, "elapsed")


def test_intrinsic_rank(geom_nmf_result):
    result = geom_nmf_result
    assert result.fit_diagnostics["intrinsic_rank"] == K - 1


def test_endmember_shape(geom_nmf_result):
    result = geom_nmf_result
    assert result.H_star.shape == (K, J)


def test_phi_shape(geom_nmf_result):
    result = geom_nmf_result
    assert result.Phi.shape == (K, J)


def test_alignment_error(synthetic_data, geom_nmf_result):
    _, H_true, _ = synthetic_data
    result = geom_nmf_result
    H_perm, _, _, order = permute_to_reference(
        H_true, result.H_star, result.mu_tilde, result.Phi
    )
    assert len(order) == K
    errors = np.linalg.norm(H_true - H_perm, axis=1)
    assert np.all(errors < 0.5), f"Large endmember errors: {errors.round(4)}"


def test_barplot_runs(geom_nmf_result):
    feature_labels = [f"F{j+1}" for j in range(J)]
    barplot_Phi(
        geom_nmf_result.Phi,
        title="Estimated source attribution (GeomNMF)",
        feature_labels=feature_labels,
    )
