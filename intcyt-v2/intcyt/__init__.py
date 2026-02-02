"""IntCyt v2: operadic self-organization with thermodynamic bookkeeping.

This package bundles:

- ``intcyt.legacy``: the original research prototype (cell/operad runtime and
  stand-alone scripts) with minimal modernization for Python 3.
- ``intcyt.core``: lightweight programmatic helpers around the legacy runtime.
- ``intcyt.thermo`` + ``intcyt.metrics``: opt-in thermodynamic ledger and
  analysis utilities.

For backwards compatibility with existing notebooks/scripts, the top-level
package re-exports the core legacy API:

- ``Cell``, ``SuperCell``, ``Operad``
- ``intcyt`` (single update step)
- ``usf`` (utility functions used by the legacy scripts)
- ``debug_time``

"""

from __future__ import annotations

from importlib import import_module

# Import the legacy compatibility layer as a *module* (not via
# ``intcyt.legacy`` package attributes) to avoid any name collisions with the
# ``intcyt`` function.
_legacy = import_module("intcyt.legacy.intcyt")

# Re-export legacy public API.
Cell = _legacy.Cell
SuperCell = _legacy.SuperCell
Operad = _legacy.Operad
intcyt = _legacy.intcyt
usf = _legacy.usf
debug_time = _legacy.debug_time

__all__ = [
    "Cell",
    "SuperCell",
    "Operad",
    "intcyt",
    "usf",
    "debug_time",
]

# Optional convenience: expose subpackages for discoverability.
from . import core, legacy, metrics, thermo, viz  # noqa: E402,F401
