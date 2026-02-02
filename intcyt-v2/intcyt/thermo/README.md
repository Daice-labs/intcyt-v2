# Thermodynamic Ledger

The helpers under `thermo/` instrument the legacy IntCyt runtime without
modifying its source.  They provide a lightweight event ledger that records
irreversible operations (cytosol resets, organelle discards) so you can feed the
result into the `metrics/` modules (`lnd`, `doe`, `tur`, …).

## 1. Components

- `thermo.Ledger` — in-memory list of `(time, kind, bits)` events with a
  `to_timeseries()` helper that returns per-step operation counts and bit
  lower bounds.
- `thermo.snapshot(supercell)` — walks a legacy `SuperCell` tree and counts:
  - non-zero cytosol coordinates,
  - total organelles across the structure.
- `thermo.step_with_ledger(...)` — wraps `legacy.cellint.ict.intcyt`:
  it snapshots the tree before/after the call and logs `Z` events when cytosol
  entries are reset and `M` events when organelles are discarded.

## 2. Quick Start

```python
from intcyt.thermo import Ledger, step_with_ledger
from intcyt.legacy.celloperad.cl_ope import Operad

operad = Operad(dimension)
supercell = ...  # build the initial tree via the legacy API
ledger = Ledger()

for t, payload in enumerate(stream_of_vectors):
    step_with_ledger(
        operad,
        supercell,
        index=t,
        events=events,
        vector=payload,
        gamma=gamma_fn(supercell),
        filtering=[1.5, 1.5, 10],
        ledger=ledger,
    )

counts, bit_bounds = ledger.to_timeseries()
```

Feed `counts`/`bit_bounds` into the metric modules, e.g.
`intcyt.metrics.lnd.trajectory_dissipation(bit_bounds)`.

## 3. CLI Helper

The repository root exposes a tiny CLI so you can run the legacy scripts and
inspect the ledger API without hunting for module paths:

```bash
python cli.py legacy-main MNIST -iterations 10000
python cli.py legacy-challenge fashion-MNIST 20 -right
python cli.py ledger-info
```

`ledger-info` simply prints a minimal example.  To collect real events you still
need to wrap your training loop with `step_with_ledger` as shown above.

### Demo config/stream format (`cli.py ledger-run`)

The `ledger-run` subcommand expects:

1. `--config`: JSON file describing the problem setup, e.g.

   ```json
   {
     "dimension": 4,
     "arity": 2,
     "iterations": 3,
     "events": [
       [0],  # start index
       [1],  # epoch length
       [0],  # fission events (indices modulo epoch)
       [0],  # fusion events
       [0]   # compose events
     ]
   }
   ```

2. `--stream`: JSON Lines file where each line provides the per-iteration
   payload.  Minimum fields: `vector` (list of floats) and `gamma` (whatever the
   legacy ``gamma`` callable expects).  Optional `filtering` overrides the
   default `[1.5, 1.5, 10]`.

   ```jsonl
   {"vector": [0.1, 0.2, 0.3, 0.4], "gamma": [[1, 0, 0, 0]], "filtering": [1.5, 1.5, 5]}
   {"vector": [0.2, 0.1, 0.0, 0.7], "gamma": [[0, 1, 0, 0]]}
   {"vector": [0.4, 0.4, 0.1, 0.1], "gamma": [[0, 0, 1, 0]]}
   ```

We plan to bundle sample files under `demo/` so you can run:

```bash
python cli.py ledger-run --config demo/config.json --stream demo/stream.jsonl --output demo/ledger.json
```

Until then, use the snippets above as templates for your own experiments.
