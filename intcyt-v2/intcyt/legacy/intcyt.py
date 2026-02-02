"""Compatibility layer that exposes the public legacy API in one place.

Historically these modules relied on manipulating ``sys.path`` at runtime.
Instead we re-export the relevant classes/functions so modern callers can
simply ``import legacy.intcyt`` (or ``from legacy import *``) without any of
the previous side effects.
"""

from .celloperad.cl_cel import Cell
from .celloperad.cl_sup import SuperCell
from .celloperad.cl_ope import Operad
from .cellint.ict import intcyt
from .useful.cl_usf import usf
from .useful.cl_dbg import debug_time

__all__ = [
    "Cell",
    "SuperCell",
    "Operad",
    "intcyt",
    "usf",
    "debug_time",
]
