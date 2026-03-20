"""
geom_nmf.endmembers
===================
Candidate endmember generation: convex-hull vertices (exact) or random
projection extremes (approximate), plus clustering-based pruning.
Volume scoring and exhaustive maximum-volume simplex search over a candidate set of endmember points.
"""

import time
import numpy as np
from itertools import combinations
from scipy.special import gammaln, comb

def get_hull_candidates(
    Yc_proj: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """
    Return convex-hull vertex indices of *Yc_proj* via QHull.

    Parameters
    ----------
    Yc_proj : np.ndarray, shape (n, r)
        Centred, intrinsic-space projections of the data.
    verbose : bool, optional
        Print timing and candidate count.  Default False.

    Returns
    -------
    cand_idx : np.ndarray, shape (m,)
        Indices (into *Yc_proj*) of the convex-hull vertices.
    """
    from scipy.spatial import ConvexHull

    if verbose:
        print("Computing convex hull...", end="", flush=True)

    start = time.time()
    hull = ConvexHull(Yc_proj, qhull_options="Qx Qt Q12 Pp")

    if verbose:
        print(f" done in {time.time() - start:.2f}s; #cands={len(hull.vertices)}")

    return hull.vertices


def get_random_direction_candidates(
    Yc_proj_w: np.ndarray,
    seed: int = 123,
    n_directions: int = 20000,
    n_top: int = 1,
    n_candidates: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find extreme points by projecting onto many random directions (vectorized).

    For each random unit vector u the *n_top* largest **and** smallest dot
    products ``Yc_proj_w @ u`` are recorded.  Points that appear as extremes
    most often are the most likely simplex vertices.

    Parameters
    ----------
    Yc_proj_w : np.ndarray, shape (n, r)
        Whitened, centered projections of the data in intrinsic space.
    seed : int, optional
        Random seed for reproducibility.  Default 123.
    n_directions : int, optional
        Number of random directions to sample.  Default 20000.
    n_top : int, optional
        How many extreme points to record per direction (both ends).
        Default 1.
    n_candidates : int or None, optional
        If given, return at most this many candidate indices (highest counts
        first).  Default None (return all that were ever selected).
    verbose : bool, optional
        Print timing and candidate count.  Default False.

    Returns
    -------
    chosen_sorted : np.ndarray, shape (m,)
        Indices (into *Yc_proj_w*) of candidate extreme points, sorted by
        descending selection frequency.
    counts : np.ndarray, shape (n,)
        Full frequency array for all n points.
    """
    X = np.asarray(Yc_proj_w, dtype=float)
    n, r = X.shape
    rng = np.random.default_rng(seed)
    counts = np.zeros(n, dtype=np.intp)

    if verbose:
        print(
            f"Sampling {n_top} extreme points in each of {n_directions} random directions...",
            end="",
            flush=True,
        )
    start = time.time()

    batch = min(n_directions, 2048)
    remaining = n_directions
    while remaining > 0:
        b = min(batch, remaining)
        remaining -= b

        U = rng.standard_normal((r, b))
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-12

        S = X @ U   # (n, b)

        if n_top == 1:
            np.add.at(counts, S.argmax(axis=0), 1)
            np.add.at(counts, S.argmin(axis=0), 1)
        else:
            St = S.T  # (b, n)
            hi_idx = np.argpartition(St, -n_top, axis=1)[:, -n_top:]
            lo_idx = np.argpartition(St,  n_top, axis=1)[:,  :n_top]
            np.add.at(counts, hi_idx.ravel(), 1)
            np.add.at(counts, lo_idx.ravel(), 1)

    chosen = np.flatnonzero(counts > 0)
    chosen_sorted = chosen[np.argsort(counts[chosen])[::-1]] # min(n, 2n_directions x n_top) candidates, sorted by count

    if n_candidates is not None:
        chosen_sorted = chosen_sorted[:n_candidates] # min(n, 2n_directions x n_top, n_candidates) candidates

    if verbose:
        print(f" done in {time.time() - start:.2f}s; #cands={len(chosen_sorted)}")

    return chosen_sorted, counts

def prune_close_points(
    points: np.ndarray,
    n_clusters: int = 25,
    seed: int = 123,
) -> tuple[np.ndarray, list[int]]:
    """
    Cluster *points* and return one representative per cluster.
 
    The representative is the cluster member closest (in Euclidean distance)
    to the cluster centroid, so the returned points are always actual data
    points (not interpolated centroids).
 
    The number of clusters is chosen automatically by maximizing the
    silhouette score over the range
    ``[max(2, n_clusters), min(2 * n_clusters, m - 1)]``,
    with *n_clusters* as the lower bound.
 
    Parameters
    ----------
    points : np.ndarray, shape (m, J)
        Points to prune.  Must be finite (no NaN or Inf).
    n_clusters : int, optional
        Minimum number of representatives (clusters) to retain after pruning.  The
        silhouette search explores a range around this value, so the actual
        number returned may differ slightly.  Default 25.
    seed : int, optional
        Random seed passed to MiniBatchKMeans.  Default 123.
 
    Returns
    -------
    representatives : np.ndarray, shape (n_clusters_out, J)
        Selected representative points.
    selected_indices : list of int
        Row indices into *points* for each representative.
    """
    X = np.asarray(points, dtype=float)
    m = X.shape[0]
 
    if m == 0:
        return X, []
    if m == 1:
        return X.copy(), [0]
 
    # try to import sklearn lazily; if missing, skip pruning safely.
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        return X, list(range(m))
 
    lower = max(2, n_clusters)
    upper = min(2 * n_clusters, m - 1)
 
    if lower > upper:
        return X, list(range(m))
 
    best_k, best_s = lower, -np.inf
    for k in range(lower, upper + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=4096)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(
            X, labels,
            metric="euclidean",
            sample_size=min(10000, m),
            random_state=seed,
        )
        if s > best_s:
            best_s, best_k = s, k
 
    km = MiniBatchKMeans(n_clusters=best_k, random_state=seed, n_init=10, batch_size=4096)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
 
    selected = []
    for k in range(best_k):
        members = np.flatnonzero(labels == k)
        if len(members) == 0:
            continue
        diffs = X[members] - centers[k]
        nearest = members[np.einsum("ij,ij->i", diffs, diffs).argmin()]
        selected.append(int(nearest))
 
    selected = sorted(set(selected))
    return X[selected], selected


def log_simplex_volume(
    subset: np.ndarray,
    rank_tol: float = 1e-12,
) -> tuple[float, int]:
    """
    Compute the log-volume of the (K-1)-simplex spanned by *subset*.

    The volume is defined as::

        vol = (∏ sᵢ) / r!

    where sᵢ are the nonzero singular values of the edge matrix
    ``subset[1:] - subset[0]`` and r is the affine rank.

    Parameters
    ----------
    subset : np.ndarray, shape (K, J)
        K points in J-dimensional ambient space.
    rank_tol : float, optional
        Singular-value threshold below which a dimension is considered zero.
        Default 1e-12.

    Returns
    -------
    log_vol : float
        Natural log of the simplex volume, or ``-np.inf`` if degenerate.
    r : int
        Affine rank of *subset* (number of non-negligible singular values).
    """
    base = subset[0]
    A = subset[1:] - base              # edge matrix, shape (K-1, J)
    _, S, _ = np.linalg.svd(A, full_matrices=False)

    mask = S > rank_tol
    r = int(mask.sum())

    if r < A.shape[0]:                 # degenerate simplex
        return -np.inf, r

    log_vol = float(np.log(S[mask]).sum() - gammaln(r + 1))
    return log_vol, r


def estimate_H(
    hull_pts: np.ndarray,
    K: int,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Find the K-point subset of *hull_pts* that maximizes simplex log-volume.

    Exhaustively iterates over all C(m, K) combinations.  Feasible when
    the number of hull / candidate points is small (≲ a few hundred).

    Parameters
    ----------
    hull_pts : np.ndarray, shape (m, J)
        Candidate points (convex-hull vertices or random-direction extremes)
        in the ambient endmember space.
    K : int
        Number of endmembers (simplex vertices) to select.
    verbose : bool, optional
        If True, prints the number of combinations and shows a tqdm progress
        bar (requires tqdm to be installed).  Default False.

    Returns
    -------
    H_hat_best : np.ndarray, shape (K, J)
        The K rows of *hull_pts* that achieve the maximum log-volume.
    best_logvol : float
        Corresponding log-volume value.

    Raises
    ------
    RuntimeError
        If every K-subset of *hull_pts* forms a degenerate simplex (zero
        volume), e.g. when all candidates are collinear.
    """
    m = int(hull_pts.shape[0])
    total = int(comb(m, K, exact=True))
    if verbose:
        print(f"H candidates={m}, K={K}, combinations={total:,}")

    it = combinations(range(m), K)
    if verbose:
        try:
            from tqdm import tqdm as _tqdm
            it = _tqdm(it, total=total, desc=f"Searching K={K} subsets for max volume")
        except ImportError:
            pass

    best_logvol = -np.inf
    best_inds: tuple | None = None

    for inds in it:
        logvol, _ = log_simplex_volume(hull_pts[list(inds)])
        if logvol > best_logvol:
            best_logvol = logvol
            best_inds = inds

    if best_inds is None:
        raise RuntimeError(
            f"No non-degenerate K={K}-simplex found among the {m} candidate points. "
            "The candidate set may be collinear or too small."
        )

    H_hat_best = hull_pts[list(best_inds)]
    return H_hat_best, float(best_logvol)
