"""Convenience CLI for the IntCyt workspace.

This wrapper keeps the legacy scripts runnable without requiring separate entry
points, and also exposes a couple of helper utilities for the thermodynamic
ledger.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .thermo import Ledger, step_with_ledger


def _forward(module_path: str, func: str, args: Sequence[str]) -> int:
  module = __import__(module_path, fromlist=[func])
  runner = getattr(module, func)
  return runner(list(args))


def _cmd_legacy_main(args: argparse.Namespace) -> int:
  return _forward("intcyt.legacy.software_implementation.main", "main", args.legacy_args)


def _cmd_legacy_challenge(args: argparse.Namespace) -> int:
  return _forward("intcyt.legacy.software_implementation.challenge", "main", args.chal_args)


def _cmd_legacy_data(args: argparse.Namespace) -> int:
  return _forward("intcyt.legacy.software_implementation.data_processing", "main", args.data_args)


def _cmd_ledger_info(_: argparse.Namespace) -> int:
  example = Ledger()
  example.add(t=0, kind="Z", bits=5, meta={"note": "cytosol reset"})
  example.add(t=1, kind="M", bits=2, meta={"note": "merging discard"})
  counts, bits = example.to_timeseries()
  print("Ledger quickstart:")
  print("  events:", example.events)
  print("  counts:", counts.tolist())
  print("  bit lower bounds:", bits.tolist())
  print("\nSee thermo/README.md for full instrumentation instructions.")
  return 0


def _cmd_ledger_run(args: argparse.Namespace) -> int:
  """Minimal runner that wraps the legacy intcyt call with step_with_ledger.

  This is deliberately simple: it expects a JSONL file describing vectors/
  gammas/filtering per iteration.  Each line must contain a dict with keys
  'vector', 'gamma', and optional 'filtering'.  Events schedule is provided via
  a JSON array.
  """

  from .legacy.celloperad.cl_ope import Operad
  from .legacy.celloperad.cl_sup import SuperCell
  from .legacy.celloperad.cl_cel import Cell
  from .legacy.cellint.ict import intcyt  # noqa: F401 (ensures module available)

  config = json.loads(Path(args.config).read_text())
  dimension = config["dimension"]
  events = config["events"]
  arity = config.get("arity", 2)
  iterations = config.get("iterations", 1)

  operad = Operad(dimension)
  # start from a simple zeroed cell
  root = Cell(dimension, 0, [0] * dimension, [[0] * dimension for _ in range(arity)])
  supercell = SuperCell(root)

  ledger = Ledger()
  lines = Path(args.stream).read_text().splitlines()
  if len(lines) < iterations:
    raise ValueError("Not enough stream entries for requested iterations")

  for idx in range(iterations):
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
  print("ledger_run summary")
  print("  iterations:", len(counts))
  print("  total events:", int(np.sum(counts)))
  print("  total bits (lower bound):", float(np.sum(bit_bounds)))
  if args.output:
    Path(args.output).write_text(
      json.dumps(
        {
          "counts": counts.tolist(),
          "bit_lower_bounds": bit_bounds.tolist(),
          "events": [event.__dict__ for event in ledger.events],
        },
        indent=2,
      )
    )
    print(f"  detailed ledger written to {args.output}")
  return 0


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Utility entry point for legacy IntCyt scripts.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  legacy_main = subparsers.add_parser(
    "legacy-main",
    help="Run legacy/software_implementation/main.py with forwarded args.",
  )
  legacy_main.add_argument(
    "legacy_args",
    nargs=argparse.REMAINDER,
    help="Arguments forwarded verbatim to the original script.",
  )
  legacy_main.set_defaults(func=_cmd_legacy_main)

  legacy_challenge = subparsers.add_parser(
    "legacy-challenge",
    help="Run legacy/software_implementation/challenge.py with forwarded args.",
  )
  legacy_challenge.add_argument(
    "chal_args",
    nargs=argparse.REMAINDER,
    help="Arguments forwarded verbatim to the original script.",
  )
  legacy_challenge.set_defaults(func=_cmd_legacy_challenge)

  legacy_data = subparsers.add_parser(
    "legacy-data",
    help="Run legacy/software_implementation/data_processing.py with forwarded args.",
  )
  legacy_data.add_argument(
    "data_args",
    nargs=argparse.REMAINDER,
    help="Arguments forwarded verbatim to the original script.",
  )
  legacy_data.set_defaults(func=_cmd_legacy_data)

  ledger = subparsers.add_parser(
    "ledger-info",
    help="Show a minimal example of the thermodynamic ledger API.",
  )
  ledger.set_defaults(func=_cmd_ledger_info)

  ledger_run = subparsers.add_parser(
    "ledger-run",
    help="Wrap a simple training stream with step_with_ledger for demonstration.",
  )
  ledger_run.add_argument(
    "--config",
    required=True,
    help="Path to JSON config describing dimension, events, iterations, arity.",
  )
  ledger_run.add_argument(
    "--stream",
    required=True,
    help="JSONL file providing per-iteration vectors/gamma/filtering.",
  )
  ledger_run.add_argument(
    "--output",
    help="Optional path to write the ledger dump as JSON.",
  )
  ledger_run.set_defaults(func=_cmd_ledger_run)

  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":
  sys.exit(main())
