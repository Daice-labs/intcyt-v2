"""High-level helpers that wrap the legacy IntCyt scripts.

This module is intentionally lightweight: it forwards to the modernized
`legacy/software_implementation/*.py` entry points so notebooks or services can
invoke the legacy routines without spawning subprocesses.

It also exposes a helper for the thermodynamic ledger demo so callers do not
have to reimplement the JSON plumbing used by ``cli.py ledger-run``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ..thermo import Ledger, step_with_ledger


def _call_legacy(module: str, argv: Sequence[str]) -> int:
  mod = __import__(module, fromlist=["main"])
  return mod.main(list(argv))


@dataclass
class ChallengeConfig:
  dataset: str
  samples: int
  option: Optional[str] = None
  option_args: Sequence[str] = field(default_factory=list)


def run_challenge(cfg: ChallengeConfig) -> int:
  args: List[str] = [cfg.dataset, str(cfg.samples)]
  if cfg.option:
    args.append(cfg.option)
    args.extend(str(a) for a in cfg.option_args)
  return _call_legacy("intcyt.legacy.software_implementation.challenge", args)


@dataclass
class TrainingConfig:
  dataset: str
  iterations: int = 30000
  data_dir: str = "data"
  result_load: Optional[str] = None
  dream3_training: Optional[str] = None
  load_mode: Optional[str] = None  # "-load", "-load-selfsup-right", ...
  load_index: Optional[int] = None
  seed: Optional[int] = None
  extra_args: Sequence[str] = field(default_factory=list)


def run_training(cfg: TrainingConfig) -> int:
  args: List[str] = [cfg.dataset, "-iterations", str(cfg.iterations)]
  if cfg.seed is not None:
    args.extend(["-seed", str(cfg.seed)])
  if cfg.load_mode:
    args.append(cfg.load_mode)
    if cfg.load_index is not None:
      args.append(str(cfg.load_index))
  if cfg.result_load:
    args.extend(["--result-load", cfg.result_load])
  if cfg.dream3_training:
    args.extend(["--dream3-training", cfg.dream3_training])
  args.extend(["--data-dir", cfg.data_dir])
  args.extend(cfg.extra_args)
  return _call_legacy("intcyt.legacy.software_implementation.main", args)


@dataclass
class DataCommand:
  command: str  # e.g., "training", "figure1", "method", "result"
  command_args: Sequence[str] = field(default_factory=list)


def run_data_processing(cfg: DataCommand) -> int:
  args = [cfg.command, *map(str, cfg.command_args)]
  return _call_legacy("intcyt.legacy.software_implementation.data_processing", args)


@dataclass
class LedgerRunConfig:
  config_path: str
  stream_path: str
  output_path: Optional[str] = None


def run_ledger_stream(cfg: LedgerRunConfig) -> dict:
  """Programmatic equivalent of ``cli.py ledger-run``."""

  from ..legacy.celloperad.cl_ope import Operad
  from ..legacy.celloperad.cl_sup import SuperCell
  from ..legacy.celloperad.cl_cel import Cell

  payload = json.loads(Path(cfg.config_path).read_text())
  dimension = int(payload["dimension"])
  arity = int(payload.get("arity", 2))
  iterations = int(payload.get("iterations", 0))
  events = payload["events"]

  root = Cell(dimension, 0, [0] * dimension, [[0] * dimension for _ in range(arity)])
  supercell = SuperCell(root)
  operad = Operad(dimension)
  ledger = Ledger()

  lines = Path(cfg.stream_path).read_text().splitlines()
  if iterations and len(lines) < iterations:
    raise ValueError("stream file has fewer entries than requested iterations")

  limit = iterations or len(lines)
  for idx in range(limit):
    row = json.loads(lines[idx])
    vector = row["vector"]
    gamma = row["gamma"]
    filtering = row.get("filtering", [1.5, 1.5, 10])
    step_with_ledger(
      operad,
      supercell,
      index=idx,
      events=events,
      vector=vector,
      gamma=gamma,
      filtering=filtering,
      ledger=ledger,
    )

  counts, bit_bounds = ledger.to_timeseries()
  result = {
    "counts": counts.tolist(),
    "bit_lower_bounds": bit_bounds.tolist(),
    "events": [event.__dict__ for event in ledger.events],
  }
  if cfg.output_path:
    Path(cfg.output_path).write_text(json.dumps(result, indent=2))
  return result
