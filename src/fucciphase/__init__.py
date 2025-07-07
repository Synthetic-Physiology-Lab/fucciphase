"""Cell cycle analysis plugin."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fucciphase")
except PackageNotFoundError:
    __version__ = "uninstalled"
__all__ = ["__version__", "logistic", "process_dataframe", "process_trackmate"]

from .fucci_phase import process_dataframe, process_trackmate
from .sensor import logistic
