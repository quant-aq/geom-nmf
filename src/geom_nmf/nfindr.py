"""
geom_nmf.nfindr
===============
N-FINDR endmember extraction (NumPy-only), based on Winter (1999) with
documented modifications.

Reference
---------
Winter, M. E. (1999). N-FINDR: An algorithm for fast autonomous spectral
end-member determination in hyperspectral data.  *Proc. SPIE 3753*.
"""

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pca_reduce(
    Y: np.ndarray,
    d: int,
) -> np.ndarray:
    """
    Center *Y* and project onto its top *d* principal components.

    Parameters
    ----------
    Y : np.ndarray, shape (n, J)
        Input data matrix.
    d : int
        Number of principal components to retain.

    Returns
    -------
    Z : np.ndarray, shape (n, d)
        Projected coordinates.
    """
    Yc = Y - Y.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Yc, full_matrices=False)
    return U[:, :d] * S[:d]

def _build_simplex_matrix(
    Z: np.ndarray,
    idx: np.ndarray,
    K: int,
) -> tuple[np.ndarray, float]:
    """
    Build the augmented simplex matrix M and compute its volume.

    Parameters
    ----------
    Z : np.ndarray, shape (n, K-1)
        Data in the reduced PCA space.
    idx : np.ndarray, shape (K,)
        Indices of the K simplex vertices into Z.
    K : int
        Number of vertices.

    Returns
    -------
    M : np.ndarray, shape (K, K)
        Augmented matrix [[Z[idx].T], [1...1]].
    vol : float
        Simplex volume |det(M)| / (K-1)!
    """
    M = np.vstack([Z[idx].T, np.ones((1, K))])
    vol = abs(float(np.linalg.det(M))) / math.factorial(K - 1)
    return M, vol

def _init_indices_random(
    n: int,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw *K* distinct indices uniformly at random.

    Parameters
    ----------
    n : int
        Total number of samples.
    K : int
        Number of indices to draw.
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    idx : np.ndarray, shape (K,), dtype int
        Selected indices (no duplicates).
    """
    return rng.choice(n, size=K, replace=False)


def _init_indices_atgp(
    Z: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    ATGP-style greedy initialization in the reduced PCA space.

    Iteratively adds the point whose residual (after projecting out the
    subspace spanned by all current selections) has the largest norm.

    Parameters
    ----------
    Z : np.ndarray, shape (n, d)
        Data in the reduced (K-1)-dimensional PCA space.
    K : int
        Number of endmembers to initialise.

    Returns
    -------
    idx : np.ndarray, shape (K,), dtype int
        Indices of the K initialization points.
    """
    idx: list[int] = []
    norms = np.einsum("ij,ij->i", Z, Z)
    idx.append(int(np.argmax(norms)))

    for _ in range(1, K):
        # Q: (d, m=len(idx)) orthonormal basis for the row space of Z[idx]
        Q, _ = np.linalg.qr(Z[idx].T)
        # project each row of Z onto the subspace spanned by current selections
        proj = Z @ (Q @ Q.T)
        resid_sq = np.einsum("ij,ij->i", Z - proj, Z - proj)
        resid_sq[idx] = -np.inf
        # add the point with the largest orthogonal residual (most unexplained)
        idx.append(int(np.argmax(resid_sq)))

    return np.array(idx, dtype=int)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nfindr(
    Y: np.ndarray | pd.DataFrame,
    K: int,
    max_iter: int = 5,
    seed: int | None = None,
    normalize: bool = True,
    init: str = "atgp",
    init_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    N-FINDR endmember extraction (NumPy-only, no SciPy).

    Finds K endmembers by iteratively replacing each current endmember
    candidate with the pixel that maximizes the simplex volume in the
    (K-1)-dimensional PCA-reduced space.

    Parameters
    ----------
    Y : np.ndarray or pd.DataFrame, shape (n, J) or (rows, cols, J)
        Input spectral data.  Can be a 2-D matrix or a 3-D hyperspectral
        cube; the cube is reshaped to (rows*cols, J) internally.
        All entries should be nonnegative (typical for spectral data).
    K : int
        Number of endmembers to extract.  Must satisfy ``2 ≤ K ≤ J + 1``.
    max_iter : int, optional
        Maximum number of full replacement sweeps (each sweep considers
        swapping every one of the K current endmembers with every pixel).
        Iteration stops early if no improvement is found.  Default 5.
    seed : int or None, optional
        Random seed for reproducibility; used only when ``init="random"``.
        Default None.
    normalize : bool, optional
        If True, L2-normalize each spectrum (row) before PCA projection.
        Default True.
    init : {"atgp", "random"}, optional
        Initialisation strategy when *init_idx* is not provided:

        * ``"atgp"`` – greedy ATGP initialization.
          Recommended; differs from Winter (1999)'s original random init.
        * ``"random"`` – K indices drawn uniformly at random (original
          Winter (1999) behavior).

        Default ``"atgp"``.
    init_idx : array-like of int, shape (K,) or None, optional
        If supplied, use these K indices as the starting endmember set and
        ignore *init*.  All indices must be distinct and in ``[0, n)``.
        Default None.

    Returns
    -------
    E : np.ndarray, shape (K, J)
        Extracted endmember spectra in the original feature space (rows of
        the **unnormalized** *Y*).
    idx : np.ndarray, shape (K,), dtype int
        Indices of the selected pixels in the flattened data array.

    Raises
    ------
    ValueError
        If *Y* is not 2-D or 3-D, if K is out of range, if n < K, or if
        *init_idx* is invalid (wrong shape, duplicates, or out-of-bounds
        indices).

    Notes
    -----
    Deviations from Winter (1999):

    * Defaults to ATGP initialization (Plaza et al., 2002) instead of
      random; pass ``init="random"`` to recover the original behavior.
    * Optional L2 row-normalization before PCA (``normalize=True`` by
      default); Winter (1999) does not normalize.
    * The inner replacement loop is vectorized via the matrix-determinant
      lemma, scoring all n candidate pixels at once per vertex.  This is
      mathematically equivalent to the original but O(N·K²) per vertex
      instead of O(N·K³).  A fallback to explicit volume recomputation
      is used when the current simplex is numerically degenerate.

    Volume formula (for reference)::

        vol = |det(M)| / (K-1)!,   M = [[Z.T], [1...1]]  ∈ R^{K×K}
    """

    # process data to a working matrix
    if isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()

    if Y.ndim == 3:
        rows, cols, J = Y.shape
        Y2 = Y.reshape(rows * cols, J)
    elif Y.ndim == 2:
        Y2 = Y
    else:
        raise ValueError("Y must be 2-D (n, J) or 3-D (rows, cols, bands).")

    n, J = Y2.shape

    if normalize:
        norms = np.linalg.norm(Y2, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        Yn = Y2 / norms
    else:
        Yn = Y2

    if K < 2:
        raise ValueError(f"K must be at least 2 (a single endmember has zero simplex volume); got K={K}.")
    if K > J + 1:
        raise ValueError(f"K must be ≤ J+1={J + 1} (PCA reduces to K-1={K - 1} dims but data has only J={J} features); got K={K}.")
    if n < K:
        raise ValueError(f"Need at least K={K} observations to form a simplex; got n={n}.")

    Z = _pca_reduce(Yn, d=K - 1)    # (n, K-1)

    # initialize K indices
    rng = np.random.default_rng(seed)
    if init_idx is not None:
        idx = np.asarray(init_idx, dtype=int)
        if idx.shape != (K,):
            raise ValueError(
                f"init_idx must have exactly K={K} entries; got shape {idx.shape}."
            )
        if len(set(idx.tolist())) != K:
            raise ValueError("init_idx must not contain duplicate indices.")
        if idx.min() < 0 or idx.max() >= n:
            raise ValueError(f"init_idx entries must lie in [0, n={n}).")
        idx = idx.copy()
    elif init == "atgp":
        idx = _init_indices_atgp(Z, K)
    elif init == "random":
        idx = _init_indices_random(n, K, rng)
    else:
        raise ValueError("init must be 'atgp' or 'random'.")

    M, best_vol = _build_simplex_matrix(Z, idx, K)

    for _ in range(max_iter):
        improved = False
        for j in range(K):
            locked = set(idx.tolist()) - {idx[j]}
            det_M = np.linalg.det(M)
            if abs(det_M) < 1e-300:
                # Degenerate — fall back to explicit volume recomputation
                best_j = idx[j]
                trial = idx.copy()
                for cand in range(n):
                    if cand in locked:
                        continue
                    trial[j] = cand
                    _, v = _build_simplex_matrix(Z, trial, K)
                    if v > best_vol:
                        best_vol = v
                        best_j = cand
                if best_j != idx[j]:
                    idx[j] = best_j
                    M, best_vol = _build_simplex_matrix(Z, idx, K) # safer to recompute
                    improved = True
                continue

            # Vectorized scoring via matrix-determinant lemma:
            #   det(M_new) = det(M) + (c_n - M[:,j]) · adj(M)[:,j]
            # where adj(M)[:,j] = det(M) * (M^{-1})[j,:]
            M_inv = np.linalg.inv(M)
            adj_col_j = det_M * M_inv[j, :]           # (K,)
            M_j_dot = float(M[:, j] @ adj_col_j)
            scores = Z @ adj_col_j[:-1] + adj_col_j[-1]   # (n,)
            new_dets = det_M + (scores - M_j_dot)
            new_vols = np.abs(new_dets) / math.factorial(K - 1)

            new_vols[np.array(list(locked), dtype=int)] = -np.inf

            best_j = int(np.argmax(new_vols))
            if new_vols[best_j] > best_vol:
                idx[j] = best_j
                M, best_vol = _build_simplex_matrix(Z, idx, K) # safer to recompute
                improved = True

        if not improved:
            break

    return Y2[idx, :], idx
