"""
geom_nmf.weights
================
Intensity (mixing weight) recovery and source attribution matrix computation.
"""

import warnings
import numpy as np


def compute_Phi(
    mu: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Compute the source attribution matrix Phi from source means and H.

    Each entry ``Phi[k, j]`` is the fraction of feature j attributable to
    source k, weighted by the source mean abundance::

        Phi[k, j] = mu[k] * H[k, j] / sum_k'(mu[k'] * H[k', j])

    Parameters
    ----------
    mu : np.ndarray, shape (K,)
        Mean intensity of each of the K sources.
    H : np.ndarray, shape (K, J)
        Endmember matrix; rows are source profiles, columns are features.

    Returns
    -------
    Phi : np.ndarray, shape (K, J)
        Source composition matrix; columns sum to 1 over sources.

    Notes
    -----
    If all sources contribute zero to a feature (column of ``mu[:, None] * H``
    sums to zero), the corresponding column of *Phi* will be NaN.
    """
    numerator = mu[:, None] * H          # (K, J)
    denominator = numerator.sum(axis=0)  # (J,)

    Phi = numerator / denominator        # (K, J)
    return Phi


def estimate_weights(
    Y: np.ndarray,
    H: np.ndarray,
    weight_tol: float = 1e-2,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Solve weights (intensity) W such that Y ≈ W H under the affine sum-to-one
    constraint, via the right pseudoinverse of the augmented system [H | 1].

    After computing ``W_raw = Y_aug @ pinv(H*_aug)`` the rows are projected
    onto the probability simplex by clipping negatives and renormalizing
    because rows of Y and rows of H are assumed to sum to 1.  

    Parameters
    ----------
    Y : np.ndarray, shape (n, J)
        Row-normalized observed data matrix; rows are observations, columns are features.
    H : np.ndarray, shape (K, J) 
        Row-stochastic endmember matrix; rows are source profiles.
    weight_tol : float, optional
        Threshold for flagging rows with large simplex violations (negative
        weights or sum-deviation above this value).  Default 1e-2.
    verbose : bool, optional
        Print diagnostic statistics.  Default False.

    Returns
    -------
    W : np.ndarray, shape (n, K)
        Mixing weights; nonnegative, rows sum to 1.
    diag : dict
        Diagnostic information:

        * ``"max_row_sum_dev_H"`` – max deviation of H row sums from 1
        * ``"rank_H_aug"`` – rank of augmented [H | 1]
        * ``"cond_G"`` – condition number of H_aug @ H_aug.T; large values
          mean small errors in H_aug are amplified into large errors in
          H_aug_R and therefore in W_raw
        * ``"I_err"`` – ``||H_aug pinv(H_aug) - I||_inf``
        * ``"aug_resid_inf"`` – ``||Y_aug - W_raw H_aug||_inf``
        * ``"large_neg_rows"`` – row indices where min W_raw < -weight_tol
        * ``"large_neg_count"`` – number of such rows
        * ``"not_sum1_rows"`` – row indices where row sum of W_raw deviates from 1
          by more than weight_tol
        * ``"not_sum1_count"`` – number of such rows

    """
    H_in = np.asarray(H, dtype=float)
    K, J = H_in.shape
    n = Y.shape[0]

    # augment to encode sum-to-one constraint
    H_aug = np.hstack([H_in, np.ones((K, 1))])   # (K, J+1)
    Y_aug = np.hstack([Y,    np.ones((n, 1))])   # (n, J+1)

    # right inverse via SVD pseudoinverse
    H_aug_R = np.linalg.pinv(H_aug)              # (J+1, K)

    # raw intensity 
    W_raw = Y_aug @ H_aug_R                       # (n, K)

    # diagnostics
    neg_mask  = W_raw.min(axis=1) < -weight_tol                   # large negative weights
    sum1_mask = np.abs(W_raw.sum(axis=1) - 1.0) > weight_tol      # row sum deviates from 1

    G = H_aug @ H_aug.T 
    try:
        condG = float(np.linalg.cond(G)) # large cond_G => small errors in H_aug amplified into large erros in H_aug_R
    except np.linalg.LinAlgError:
        condG = float("inf")
    I_err = float(np.linalg.norm(H_aug @ H_aug_R - np.eye(K), ord=np.inf))
    aug_resid = float(np.linalg.norm(Y_aug - W_raw @ H_aug, ord=np.inf))

    if verbose:
        print(f"||H_aug H_aug_R - I||_inf: {I_err:.3e}")

    diag = {
        "max_row_sum_dev_H": float(np.max(np.abs(H_in.sum(axis=1) - 1.0))),
        "rank_H_aug": int(np.linalg.matrix_rank(H_aug)),
        "cond_G": condG,
        "I_err": I_err,
        "aug_resid_inf": aug_resid,
        "large_neg_rows":    np.flatnonzero(neg_mask),
        "large_neg_count":   int(neg_mask.sum()),
        "not_sum1_rows":     np.flatnonzero(sum1_mask),
        "not_sum1_count":    int(sum1_mask.sum()),
    }

    # clip negatives and renormalize.
    # note: if Y and H are both row stochastic and the model is well-specified,
    # W_raw is already row stochastic and this is a no-oeration beyond numerical noise.
    # under misspecification, clipping introduces bias — use diag to assess severity.
    W = np.maximum(W_raw, 0.0)
    s = W.sum(axis=1, keepdims=True)
    # s == 0 only if every weight in a row is negative — degenerate, shouldn't
    # occur when Y and H are row stochastic; guard defensively
    s[s == 0.0] = 1.0
    W /= s

    return W, diag
