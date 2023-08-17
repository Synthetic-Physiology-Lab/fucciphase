"""Cell cycle analysis plugin."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fucciphase")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Joran Deschamps"
__email__ = "first.last@example.com"
