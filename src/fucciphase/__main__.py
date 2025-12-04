"""
Module entry point for ``python -m fucciphase``.

This thin wrapper forwards execution to :func:`fucciphase.main_cli.main_cli`,
which implements the command-line interface used by the ``fucciphase``
console script.
"""

from .main_cli import main_cli

if __name__ == "__main__":
    main_cli()
