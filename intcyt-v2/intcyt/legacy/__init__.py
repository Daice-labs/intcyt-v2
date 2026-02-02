"""Legacy package bootstrapper."""

from . import intcyt as _legacy_intcyt
from .intcyt import *  # noqa: F401,F403

__all__ = _legacy_intcyt.__all__

del _legacy_intcyt
