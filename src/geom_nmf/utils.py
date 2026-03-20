"""
geom_nmf.utils
==============
Algorithmic utilities: endmember permutation matching 
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

def permute_to_reference(
    H_ref: np.ndarray,
    H: np.ndarray,
    mu: np.ndarray,
    Phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Permute estimated endmembers to best match a reference ordering.

    Uses the Hungarian algorithm to find the permutation of rows of *H*
    that minimizes total L2 distance to *H_ref*, then applies the same
    permutation to *mu* and *Phi*.

    Parameters
    ----------
    H_ref : np.ndarray, shape (K, J)
        Reference endmember matrix; rows are sources.
    H : np.ndarray, shape (K, J)
        Estimated endmember matrix whose rows are to be permuted.
    mu : np.ndarray, shape (K,)
        Estimated mean weight per source; permuted alongside H.
    Phi : np.ndarray, shape (K, J)
        Source attribution matrix (rows are sources, columns are features);
        permuted alongside H.  

    Returns
    -------
    H_perm : np.ndarray, shape (K, J)
        Permuted endmember matrix aligned to H_ref.
    mu_perm : np.ndarray, shape (K,)
        Permuted mean weights.
    Phi_perm : np.ndarray, shape (K, J)
        Permuted source attribution matrix.
    order : np.ndarray, shape (K,), dtype int
        Permutation indices such that ``H_perm = H[order]``.
    """
    H_ref = np.asarray(H_ref)
    H     = np.asarray(H)
    mu    = np.asarray(mu)
    Phi   = np.asarray(Phi)

    # pairwise L2 cost: cost[i, j] = ||H_ref[i] - H[j]||
    cost = np.linalg.norm(H_ref[:, None, :] - H[None, :, :], axis=2)  # (K_ref, K)
    row_ind, col_ind = linear_sum_assignment(cost)

    # reorder so that estimated rows align with H_ref row order
    order = col_ind[np.argsort(row_ind)]

    return H[order], mu[order], Phi[order], order