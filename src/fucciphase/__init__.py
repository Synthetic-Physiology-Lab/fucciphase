"""
FUCCIphase: Analysis tools for cell-cycle estimation from FUCCI imaging.

This module exposes the main public API of the package, including the
core processing functions and the package version.

"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fucciphase")
except PackageNotFoundError:
    __version__ = "uninstalled"
__all__ = ["__version__", "logistic", "process_dataframe", "process_trackmate"]

from .fucci_phase import process_dataframe, process_trackmate
from .sensor import logistic
