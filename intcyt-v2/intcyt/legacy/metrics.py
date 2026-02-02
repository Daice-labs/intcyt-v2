"""Legacy shim that proxies to the modern `metrics` package.

Older code used ``import legacy.metrics`` expecting helper modules such as
``entropy`` or ``lnd`` to live under that namespace.  We lazily forward attribute
access so the new implementation (now located in ``metrics``) can evolve
independently while keeping the old import path valid.
"""

from importlib import import_module
from types import ModuleType
from typing import Dict

_SUBMODULES = {"entropy", "lnd", "doe", "tur"}
_CACHE: Dict[str, ModuleType] = {}


def __getattr__(name: str) -> ModuleType:
  if name in _SUBMODULES:
    module = import_module(f"intcyt.metrics.{name}")
    _CACHE[name] = module
    globals()[name] = module
    return module
  raise AttributeError(f"module 'legacy.metrics' has no attribute {name!r}")


def __dir__():
  return sorted(list(globals().keys()) + list(_SUBMODULES))


__all__ = sorted(_SUBMODULES)
