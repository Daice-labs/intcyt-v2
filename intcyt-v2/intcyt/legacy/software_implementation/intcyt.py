"""Local shim so legacy scripts can `import intcyt` as before."""

from importlib import import_module
from pathlib import Path
import sys

try:  # When executed as part of the package.
    from ..intcyt import *  # type: ignore  # noqa: F401,F403
    from ..intcyt import __all__  # type: ignore
except Exception:  # pragma: no cover - fall back to runtime path injection.
    package_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(package_root))
    _legacy_intcyt = import_module("intcyt.legacy.intcyt")

    globals().update({name: getattr(_legacy_intcyt, name) for name in _legacy_intcyt.__all__})
    __all__ = _legacy_intcyt.__all__
    del _legacy_intcyt
