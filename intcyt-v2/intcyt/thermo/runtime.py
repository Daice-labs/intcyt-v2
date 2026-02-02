"""Runtime helpers that observe legacy SuperCells without mutating them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from ..legacy.cellint.ict import intcyt as legacy_intcyt

from . import Ledger


@dataclass(frozen=True)
class Snapshot:
  """Summary of the current memory footprint."""

  cyt_nonzero: int
  total_organelles: int


def snapshot(supercell) -> Snapshot:
  """Compute aggregate counters over the entire super cell tree."""

  cyt = 0
  organelles = 0
  stack = [supercell]

  while stack:
    node = stack.pop()
    cell = getattr(node, "cell", None)
    if cell is None:
      continue

    cytosol = getattr(cell, "cytosol", [])
    organelle_list = getattr(cell, "organelles", [])
    cyt += sum(1 for value in cytosol if value != 0)
    organelles += len(organelle_list)

    if getattr(node, "is_leaf", True) is False:
      stack.extend(getattr(node, "innercells", []))

  return Snapshot(cyt_nonzero=cyt, total_organelles=organelles)


def _log_if_positive(
  ledger: Ledger,
  *,
  t: int,
  kind: str,
  delta: int,
  meta: Optional[dict] = None,
) -> None:
  if delta > 0:
    ledger.add(t=t, kind=kind, bits=float(delta), meta=meta)


def step_with_ledger(
  operad,
  supercell,
  index: int,
  events: Sequence[Sequence[int]],
  vector: Sequence[float],
  gamma,
  filtering: Sequence[float],
  ledger: Optional[Ledger] = None,
) -> None:
  """Call the legacy intcyt step while observing irreversible operations."""

  before = snapshot(supercell) if ledger is not None else None

  legacy_intcyt(
    operad,
    supercell,
    index,
    events,
    vector,
    gamma,
    filtering,
  )

  if ledger is None or before is None:
    return

  after = snapshot(supercell)
  _log_if_positive(
    ledger,
    t=index,
    kind="Z",
    delta=before.cyt_nonzero - after.cyt_nonzero,
    meta={"note": "cytosol reset"},
  )
  _log_if_positive(
    ledger,
    t=index,
    kind="M",
    delta=before.total_organelles - after.total_organelles,
    meta={"note": "organelles discarded"},
  )
