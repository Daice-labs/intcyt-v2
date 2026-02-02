"""Compatibility wrapper.

The installed command-line entry point is ``intcyt`` (see ``pyproject.toml``).
This wrapper keeps the historical convenience command working:

  python cli.py ...

"""

from __future__ import annotations

import sys

from intcyt.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
