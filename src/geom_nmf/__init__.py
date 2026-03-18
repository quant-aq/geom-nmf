from importlib.metadata import version

from ._base import GeomNMF
from . import _viz as viz

__version__ = version("geom-nmf")
__all__ = ["GeomNMF", "viz"]
