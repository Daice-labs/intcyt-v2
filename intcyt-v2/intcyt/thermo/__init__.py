"""Thermodynamic bookkeeping helpers for IntCyt.

This module stays outside the legacy package so we can track irreversible operations
without mutating the original implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Event:
  """Single irreversible event (e.g., cytosol zeroing, merge discard)."""

  t: int
  kind: str
  count: int = 1
  bits: Optional[float] = None
  meta: Optional[Dict[str, object]] = None


@dataclass
class Ledger:
  """Simple in-memory event ledger."""

  events: List[Event] = field(default_factory=list)

  def add(
    self,
    *,
    t: int,
    kind: str,
    count: int = 1,
    bits: Optional[float] = None,
    meta: Optional[Dict[str, object]] = None,
  ) -> None:
    """Record an event; defaults `bits` to `count` when omitted."""

    payload = Event(t=t, kind=kind, count=count, bits=bits, meta=meta)
    self.events.append(payload)

  def clear(self) -> None:
    self.events.clear()

  def __len__(self) -> int:  # pragma: no cover - trivial
    return len(self.events)

  def to_timeseries(self, *, total_steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (event_counts, bit_lower_bounds) arrays for downstream metrics."""

    if not self.events:
      if total_steps is None:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
      return np.zeros(total_steps, dtype=int), np.zeros(total_steps, dtype=float)

    max_t = max(event.t for event in self.events)
    if total_steps is None:
      total_steps = max_t + 1

    counts = np.zeros(total_steps, dtype=int)
    bits = np.zeros(total_steps, dtype=float)

    for event in self.events:
      if 0 <= event.t < total_steps:
        counts[event.t] += event.count
        bits[event.t] += event.bits if event.bits is not None else event.count

    return counts, bits


from .runtime import Snapshot, snapshot, step_with_ledger  # noqa: E402  (re-export)

__all__ = ["Event", "Ledger", "Snapshot", "snapshot", "step_with_ledger"]
