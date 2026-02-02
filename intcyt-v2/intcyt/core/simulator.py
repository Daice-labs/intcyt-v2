"""High-level Simulator facade around the legacy IntCyt runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..legacy.cellint.ict import intcyt as legacy_intcyt
from ..legacy.celloperad.cl_cel import Cell
from ..legacy.celloperad.cl_ope import Operad
from ..legacy.celloperad.cl_sup import SuperCell
from ..thermo import Ledger, step_with_ledger


def create_supercell(dimension: int, arity: int) -> SuperCell:
  """Build a trivial height-0 super cell with zeroed organelles."""

  organelles = [[0] * dimension for _ in range(arity)]
  root = Cell(dimension, 0, [0] * dimension, organelles)
  return SuperCell(root)


@dataclass
class Simulator:
  """Wrap a ``SuperCell`` with optional ledger logging."""

  dimension: int
  events: Sequence[Sequence[int]]
  arity: int = 2
  filtering: Sequence[float] = (1.5, 1.5, 10)
  ledger: Optional[Ledger] = None

  operad: Operad = field(init=False)
  supercell: SuperCell = field(init=False)

  def __post_init__(self) -> None:
    self.operad = Operad(self.dimension)
    self.supercell = create_supercell(self.dimension, self.arity)

  def step(
    self,
    index: int,
    vector: Sequence[float],
    gamma,
    filtering: Optional[Sequence[float]] = None,
  ) -> None:
    filt = filtering if filtering is not None else self.filtering
    if self.ledger is None:
      legacy_intcyt(self.operad, self.supercell, index, self.events, vector, gamma, filt)
    else:
      step_with_ledger(
        self.operad,
        self.supercell,
        index=index,
        events=self.events,
        vector=vector,
        gamma=gamma,
        filtering=filt,
        ledger=self.ledger,
      )


def run_stream(simulator: Simulator, stream: Iterable[dict]) -> None:
  """Iterate over a sequence of payload dictionaries."""

  for idx, payload in enumerate(stream):
    simulator.step(
      index=idx,
      vector=payload["vector"],
      gamma=payload["gamma"],
      filtering=payload.get("filtering"),
    )
