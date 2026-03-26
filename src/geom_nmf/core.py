"""
geom_nmf.core
=============
Main GeomNMF pipeline: maximum-volume simplex fitting for geometric NMF.
"""

import time
from typing import NamedTuple
import numpy as np
import pandas as pd

from geom_nmf.endmembers import (
    get_hull_candidates,
    get_random_direction_candidates,
    prune_close_points,
    estimate_H, 
    log_simplex_volume
)
from geom_nmf.weights import estimate_weights, compute_Phi
from geom_nmf.nfindr import nfindr


class GeomNMFResult(NamedTuple):
    """Return type of :func:`GeomNMF`."""
    H_star:             np.ndarray  # (K, J) estimated endmembers (row-normalized)
    W_tilde:            np.ndarray  # (n, K) mixing weights scaled to original row sums
    mu_tilde:           np.ndarray  # (K,)   mean weight per source
    Phi:                np.ndarray  # (K, J) source attribution matrix
    logvol:             float       # log-volume of the selected simplex
    weight_diagnostics: dict        # diagnostics from weight recovery (see estimate_weights)
    fit_diagnostics:    dict        # diagnostics from geometry / candidate steps
    elapsed:            float       # total wall-clock run time in seconds


def GeomNMF(
    Y: np.ndarray | pd.DataFrame,
    K: int,
    seed: int = 123,
    rank_tol: float = 1e-12,
    weight_tol: float = 1e-2,
    candidate_method: str = "exact",
    n_directions: int = 20000,
    n_top: int = 1,
    n_candidates: int | None = None,
    prune: bool = False,
    n_clusters: int | None = None,
    refine_greedy: bool = False,
    verbose: bool = False,
) -> GeomNMFResult:
    """
    Geometric non-negative matrix factorization via maximum-volume simplex fitting (GeomNMF).

    Assumes the observed data matrix Y follows a simplex mixing model::

        Y ≈ W H,  W ≥ 0,  H ≥ 0,  H 1 = 1

    and estimates H (endmembers) by finding the maximum-volume K-simplex
    inscribed in the convex hull of the row-normalized data Y_star.

    Parameters
    ----------
    Y : np.ndarray or pd.DataFrame, shape (n, J)
        Observed data matrix.  Rows are observations; columns are features.
        All entries must be nonnegative (the method uses row-normalization).
    K : int
        Number of sources / endmembers to estimate.
    seed : int, optional
        Random seed (used for random candidate method and greedy refinement).
        Default 123.
    rank_tol : float, optional
        Singular-value threshold for intrinsic rank determination; values below
        this are treated as zero.  Default 1e-12.
    weight_tol : float, optional
        Threshold for flagging simplex constraint violations in
        ``weight_diagnostics``; rows where any weight is below ``-weight_tol``
        or the row sum deviates from 1 by more than ``weight_tol`` are flagged.
        Default 1e-2.
    candidate_method : {"exact", "random"}, optional
        How to generate the pool of endmember candidates:

        * ``"exact"`` – convex-hull vertices via QHull (exact but may be
          slow in high intrinsic dimension).
        * ``"random"`` – random projection extremes (fast approximation;
          controlled by *n_directions*, *n_top*, *n_candidates*). When n >> J, 
          this can be slower than ``"exact"``.

        Default ``"exact"``.
    n_directions : int, optional
        Number of random directions for ``candidate_method="random"``.
        Default 20000.
    n_top : int, optional
        Number of extreme points recorded per random direction (both ends)
        when ``candidate_method="random"``.  Default 1.
    n_candidates : int or None, optional
        Number of candidates to keep after random direction sampling
        (highest-frequency first).  Default None (keep all that appear).
    prune : bool, optional
        If True, cluster the candidates and keep one representative per
        cluster before the exhaustive volume search.  Useful when the
        candidate pool is large.  Default False.
    n_clusters : int or None, optional
        Target minimum number of clusters when *prune=True*.  The actual number
        returned may be lower if empty clusters reduce the count.  
        Defaults to ``10*K``.
    refine_greedy : bool, optional
        If True, run one pass of N-FINDR greedy swaps on the initial
        solution to try to improve the simplex volume.  Only meaningful when
        ``candidate_method="random"`` or ``prune=True`` (with ``"exact"``
        and no pruning the exhaustive search is already globally optimal over
        the hull vertices).  Default False.
    verbose : bool, optional
        Print progress information.  Default False.

    Raises
    ------
    ValueError
        If *Y* contains any NaN or Inf values, any negative values, any
        all-zero rows, or if *candidate_method* is not ``"exact"`` or
        ``"random"``.

    Returns
    -------
    result : GeomNMFResult
        A named tuple with the following fields:

        * ``H_star`` – np.ndarray, shape (K, J): estimated endmembers
          (rows of row-normalized Y_star).
        * ``W_tilde`` – np.ndarray, shape (n, K): estimated mixing weights
          scaled back to the original row-sum magnitudes.
        * ``mu_tilde`` – np.ndarray, shape (K,): mean weight per source.
        * ``Phi`` – np.ndarray, shape (K, J): source attribution matrix
          (see :func:`geom_nmf.weights.compute_Phi`).
        * ``logvol`` – float: log-volume of the selected simplex.
        * ``weight_diagnostics`` – dict: diagnostics from the weight recovery step
          (see :func:`geom_nmf.weights.estimate_weights`), with keys:

          - ``"max_row_sum_dev_H"`` – max absolute deviation of H row sums from 1;
            should be near zero for a well-formed endmember matrix.
          - ``"rank_H_aug"`` – rank of the augmented system ``[H | 1]``; should
            equal K for an identifiable model.
          - ``"cond_G"`` – condition number of ``H_aug @ H_aug.T``; large values
            indicate near-collinear endmembers and amplified weight errors.
          - ``"I_err"`` – ``‖H_aug pinv(H_aug) − I‖_∞``; near zero means the
            pseudoinverse is numerically accurate.
          - ``"aug_resid_inf"`` – ``‖Y_aug − W_raw H_aug‖_∞``; reconstruction
            error of the raw (pre-clipping) weights.
          - ``"large_neg_rows"`` – row indices of observations where the raw
            weight vector has a component below ``−weight_tol``; many such rows
            suggest model misspecification.
          - ``"large_neg_count"`` – number of rows in ``large_neg_rows``.
          - ``"not_sum1_rows"`` – row indices where the raw weight row-sum
            deviates from 1 by more than ``weight_tol``.
          - ``"not_sum1_count"`` – number of rows in ``not_sum1_rows``.
        * ``fit_diagnostics`` – dict: diagnostics from the geometry / candidate
          steps, with keys:

          - ``"intrinsic_rank"`` – int: effective affine dimension of the data
            (number of singular values above *rank_tol*); should equal K-1 for a
            well-posed model.
          - ``"n_cand"`` – int: number of candidate points passed to the
            exhaustive max-volume search (after optional pruning).
          - ``"cand_idx"`` – np.ndarray: indices (into the input *Y*) of the
            candidate extreme points identified in Step 1.
          - ``"direction_hit_counts"`` – np.ndarray or None: per-observation selection
            frequency from random-direction sampling; None when
            ``candidate_method="exact"``.
          - ``"logvol_before_refinement"`` – float or None: log-volume of the exhaustive
            search solution before greedy refinement; None when
            ``refine_greedy=False`` or refinement was skipped.
        * ``elapsed`` – float: total wall-clock run time in seconds.
    """
    t_start = time.time()

    if isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()
    Y = np.asarray(Y, dtype=float)

    # check bad data
    if not np.isfinite(Y).all():
        bad = np.argwhere(~np.isfinite(Y))
        raise ValueError(
            f"Y contains {len(bad)} non-finite value(s) (NaN or Inf). "
            "Remove or impute incomplete rows before calling GeomNMF."
        )
    if (Y < 0).any():
        bad = np.argwhere(Y < 0)
        raise ValueError(
            f"Y contains {len(bad)} negative value(s). "
            "GeomNMF requires a nonnegative data matrix."
        )
    if (Y.sum(axis=1) == 0).any():
        bad_rows = np.flatnonzero(Y.sum(axis=1) == 0)
        raise ValueError(
            f"Y has {len(bad_rows)} all-zero row(s) (indices: {bad_rows.tolist()}). "
            "Row normalization requires each row to have a positive sum."
        )

    # Row sums and normalized data
    n = Y.shape[0]
    r = Y.sum(axis=1, keepdims=True)
    Y_star = Y / r                           # (n, J)

    # --- Step 0: project to intrinsic affine subspace ---
    Y_star_reduced = Y_star[:, :-1].astype(float, copy=False)   # (n, J-1)
    mean = Y_star_reduced.mean(axis=0, keepdims=True)
    Yc = Y_star_reduced - mean
    _, S_svd, Vt = np.linalg.svd(Yc, full_matrices=False)
    mask = S_svd > rank_tol
    basis = Vt[mask].T                       # (J-1, rank)
    Yc_proj = Yc @ basis                     # (n, rank)

    # --- Step 1: build candidate pool ---
    counts = None  # only populated for candidate_method="random"
    if candidate_method == "exact":
        cand_idx = get_hull_candidates(Yc_proj, verbose=verbose)
        cand_proj = Yc_proj[cand_idx]

    elif candidate_method == "random":
        # Whiten (scale to unit covariance -> remove directional stretch and make all directions comparable) 
        # intrinsic coordinates before random projection
        C_cov = (Yc_proj.T @ Yc_proj) / max(n - 1, 1)
        evals, evecs = np.linalg.eigh(C_cov)
        evals = np.maximum(evals, rank_tol)
        inv_sqrt_cov = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        Yc_proj_w = Yc_proj @ inv_sqrt_cov
        cand_idx, counts = get_random_direction_candidates(
            Yc_proj_w, seed=seed, n_directions=n_directions, n_top=n_top, n_candidates=n_candidates, verbose=verbose,
        )
        cand_proj = Yc_proj[cand_idx]        # un-whitened: pruning clusters in the same geometry as the volume objective

    else:
        raise ValueError("candidate_method must be 'exact' or 'random'.")

    cand_ambient = Y_star[cand_idx]          # (m, J)

    # --- Optional: cluster-based pruning ---
    if prune:
        _n_clusters = 10 * K if n_clusters is None else n_clusters
        if _n_clusters >= len(cand_proj):
            if verbose:
                print(
                    f"Pruning skipped: candidate count {len(cand_proj)} ≤ n_clusters {_n_clusters}"
                )
        else:
            _, pruned_idx_local = prune_close_points(cand_proj, n_clusters=_n_clusters, seed=seed)
            cand_ambient = cand_ambient[pruned_idx_local]
            cand_idx = cand_idx[pruned_idx_local]
            if verbose:
                print(f"Number of pruned candidates: {len(pruned_idx_local)}")

    H_candidates = cand_ambient

    # --- Step 2: exhaustive max-volume search over candidates ---
    H_star_hat, logvol_hat = estimate_H(
        H_candidates, K, verbose=verbose
    )

    # --- Optional: N-FINDR greedy refinement ---
    logvol_before = None
    if refine_greedy:
        if candidate_method == "exact" and not prune:
            if verbose:
                print(
                    "Greedy refinement skipped: exact hull without pruning is "
                    "already globally optimal over hull vertices."
                )
        else:
            logvol_before = logvol_hat
            diffs = Y_star[:, np.newaxis, :] - H_star_hat[np.newaxis, :, :] # (n, K, J)
            init_idx = np.argmin((diffs ** 2).sum(axis=2), axis=0) # (K,)

            _, refined_idx = nfindr(
                Y_star, K, normalize=False, init_idx=init_idx, seed=seed,
            )
            H_refined = Y_star[refined_idx]
            logvol_refined, _ = log_simplex_volume(H_refined, rank_tol=rank_tol)

            if logvol_refined > logvol_before:
                H_star_hat = H_refined
                logvol_hat = logvol_refined
                if verbose:
                    vol_ratio = np.exp(logvol_refined - logvol_before)
                    print(
                        f"Greedy refinement accepted; log-vol "
                        f"{logvol_before:.4f} → {logvol_refined:.4f} "
                        f"(volume ratio: {vol_ratio:.3f}x)"
                    )
            else:
                if verbose:
                    print(
                        f"Greedy refinement rejected; refined log-vol "
                        f"{logvol_refined:.4f} did not improve over "
                        f"{logvol_before:.4f}"
                    )

    # --- Step 3: recover mixing weights ---
    W_star_hat, weight_diagnostics = estimate_weights(Y_star, H_star_hat, weight_tol=weight_tol, verbose=verbose)
    W_tilde_hat = W_star_hat * r
    mu_tilde_hat = W_tilde_hat.mean(axis=0)
    Phi_hat = compute_Phi(mu_tilde_hat, H_star_hat)

    elapsed = time.time() - t_start

    fit_diagnostics = {
        "intrinsic_rank": int(mask.sum()),
        "n_cand": len(H_candidates),
        "cand_idx": cand_idx,
        "direction_hit_counts": counts,                # np.ndarray if candidate_method="random", else None
        "logvol_before_refinement": logvol_before,     # float if refine_greedy ran, else None
    }

    return GeomNMFResult(
        H_star=H_star_hat,
        W_tilde=W_tilde_hat,
        mu_tilde=mu_tilde_hat,
        Phi=Phi_hat,
        logvol=logvol_hat,
        weight_diagnostics=weight_diagnostics,
        fit_diagnostics=fit_diagnostics,
        elapsed=elapsed,
    )
