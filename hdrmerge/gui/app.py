"""Entry point for the desktop app: ``hdrmerge-gui``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence

from .. import __version__


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hdrmerge-gui",
        description="Merge bracketed exposures into an HDR photograph.",
    )
    parser.add_argument("inputs", nargs="*", metavar="IMAGE",
                        help="optionally open these frames or folders at startup")
    parser.add_argument("--version", action="version", version=f"hdrmerge {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
        force=True,
    )

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        sys.stderr.write(
            "The GUI needs PySide6, which is not installed.\n"
            '  pip install "hdrmerge[gui]"\n'
            "The command-line tool (hdrmerge) works without it.\n"
        )
        return 1

    from . import theme
    from .window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("hdrmerge")
    app.setApplicationDisplayName("hdrmerge")
    theme.apply(app)

    window = MainWindow(args.inputs)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
