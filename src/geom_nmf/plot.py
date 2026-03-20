"""
geom_nmf.plot
=============
Plotting helpers for GeomNMF results.
"""

import numpy as np
import matplotlib.pyplot as plt


def barplot_Phi(
    Phi: np.ndarray,
    title: str = "Estimated source attribution (GeomNMF)",
    feature_labels: list[str] | None = None,
    feature_colors = None,
    show_err: bool = True,
    savepath: str | None = None, 
) -> None:
    """
    Grouped bar plot of the source attribution matrix Phi.

    Each group of bars corresponds to one source (k); within each group
    there is one bar per feature (j), with height ``Phi[k, j] * 100`` (%).
    When a bootstrap stack is provided, bars show the mean across replicates
    and optional error bars show ±1 standard deviation.

    Parameters
    ----------
    Phi : array-like
        Either:

        * shape ``(K, J)``: single Phi matrix.
        * shape ``(n_reps, K, J)``: bootstrap stack; mean and std are
          computed across the first axis.

        Values are assumed to be in [0, 1] and are multiplied by 100 for
        display.
    title : str, optional
        Figure title.  Default ``"Estimated source attribution (GeomNMF)"``.
    feature_labels : list[str] or None, optional
        Feature labels of length J used in the legend.
        If None, defaults to ``["Feature 1", ..., "Feature J"]``.
    feature_colors : None, list, or dict, optional
        Bar colors per feature.  None uses matplotlib defaults; a list of
        length J assigns colors by position; a dict maps feature label to
        color.  Default None.
    show_err : bool, optional
        If True and *Phi* is 3D, draw error bars (±1 std across bootstrap
        reps).  Default True.
    savepath : str or None, optional
        File path to save the figure (200 dpi, tight bbox).  Default None.
    

    Returns
    -------
    None
        Displays the figure (and saves it if *savepath* is given).

    Raises
    ------
    ValueError
        If *Phi* is not 2D or 3D, or if *feature_labels* has the wrong
        length.
    """
    A = np.asarray(Phi)
    if A.ndim == 2:
        K, J = A.shape                                          # (K, J)
        Phi_mean = A * 100                                      # in %
        Phi_std = None
    elif A.ndim == 3:
        n_reps, K, J = A.shape                                  # (n_reps, K, J)
        Phi_mean = np.nanmean(A, axis=0) * 100                  # (K, J), in %
        Phi_std  = np.nanstd(A, axis=0, ddof=1 if n_reps > 1 else 0) * 100 if show_err else None
    else:
        raise ValueError(f"Expected 2D (K, J) or 3D (n_reps, K, J); got shape {A.shape}")

    # feature labels
    if feature_labels is None:
        labels = [f"Feature {j+1}" for j in range(J)]
    else:
        if len(feature_labels) != J:
            raise ValueError(f"feature_labels must have length {J} (got {len(feature_labels)})")
        labels = list(feature_labels)

    x = np.arange(K)                    # one tick per source
    width = 0.8 / max(J, 1)             # bar width within each source group

    fig, ax = plt.subplots(figsize=(10, 5))

    for j in range(J):
        # build error bar kwargs only if we have usable stds
        kwargs = {}
        if Phi_std is not None:
            yerr = Phi_std[:, j]        # (K,) std for feature j across sources
            if np.any(np.isfinite(yerr)):
                kwargs["yerr"] = yerr
                kwargs["error_kw"] = {"elinewidth": 1.2, "capsize": 3}

        # choose color for this feature
        if feature_colors is None:
            c = None
        elif isinstance(feature_colors, dict):
            c = feature_colors.get(labels[j], None)
        else:
            c = feature_colors[j]              

        ax.bar(
            x + j * width,
            Phi_mean[:, j],            # (K,) attribution of feature j per source
            width,
            color=c,
            label=labels[j],
            **kwargs,
        )

    ax.set_xticks(x + (J - 1) * width / 2)
    ax.set_xticklabels([f"Source {k+1}" for k in range(K)])
    ax.set_ylabel("Expected % of pollutants\nattributable to each source")
    ax.set_title(title)
    ax.set_ylim(0, 100)

    # Legend outside on the right
    ax.legend(title="", loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()
