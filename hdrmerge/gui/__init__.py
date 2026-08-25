"""Desktop GUI for hdrmerge.

Importing this package does not require PySide6 -- only the modules that build
widgets do -- so the command-line tool is unaffected when the GUI extra is not
installed.

    hdrmerge-gui
"""

from __future__ import annotations


def main(argv=None) -> int:
    """Launch the desktop app. See :mod:`hdrmerge.gui.app`."""
    from .app import main as _main

    return _main(argv)


__all__ = ["main"]
