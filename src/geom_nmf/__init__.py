"""
geom_nmf
========
Geometric NMF: geometric non-negative matrix factorization via maximum-volume simplex fitting.

Public API
----------
GeomNMF                      Main pipeline.
nfindr                       N-FINDR endmember extraction (standalone).
log_simplex_volume           Log-volume of a simplex spanned by K points.
compute_Phi                  Source attribution matrix from source means and H.
estimate_weights             Recover W from Y and H.
permute_to_reference         Permute estimated H, W, Phi to best match a reference ordering.
barplot_Phi                  Grouped bar plot of Phi (% attribution per source and feature).
"""

from importlib.metadata import version
from .core import GeomNMF, GeomNMFResult
from .nfindr import nfindr
from .endmembers import log_simplex_volume
from .weights import compute_Phi, estimate_weights
from .utils import permute_to_reference
from .plot import barplot_Phi

__version__ = version("geom-nmf")
__all__ = [
    "GeomNMF",
    "GeomNMFResult",
    "nfindr",
    "log_simplex_volume",
    "compute_Phi",
    "estimate_weights",
    "permute_to_reference",
    "barplot_Phi"
]
