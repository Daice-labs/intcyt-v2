# IntCyt v2 — Compact Documentation

This document condenses the original IntCyt manual for the current public
repository.  It covers setup, legacy workflows, thermodynamic extensions, and
where to look for deeper background.

---

## 1. Setup & Dependencies

1. Use **Python 3.10+**.
2. Install the repo (editable mode exposes the `intcyt` CLI):

   ```bash
   pip install -e .
   ```

3. Prepare data directories:

   ```bash
   mkdir -p data result-load
   ```

4. Download datasets:

   | Dataset | Files |
   | ------- | ----- |
   | SVHN | `data/train_32x32.mat` |
   | MNIST | `data/train-images-idx3-ubyte.gz`, `data/train-labels-idx1-ubyte.gz`, `data/t10k-images-idx3-ubyte.gz`, `data/t10k-labels-idx1-ubyte.gz` |
   | Fashion-MNIST | `data/train-images-idx3-ubyte.gz`, `data/train-labels-idx1-ubyte.gz`, `data/t10k-images-idx3-ubyte.gz`, `data/t10k-labels-idx1-ubyte.gz` |
   | DREAM3 | `data/dream3_training.gz` (generated via `intcyt/legacy/software_implementation/data_processing.py training`) |
   | Challenge docs | `data/expression_challenge/ExpressionData_UPDATED.txt`, `TargetList.txt`, `ExpressionChallenge.txt` |

---

## 2. Legacy Workflow (proof-of-concept)

The historical implementation lives under `intcyt/legacy/`.  Scripts can be run
directly or through the CLI:

### Challenge generation

```bash
python cli.py legacy-challenge MNIST 20 -right
python cli.py legacy-challenge fashion-MNIST 20 -left-noisy 2 5
```

Outputs land in `result-load/load_initial.gz`.

### Training loop (self-organized learning)

```bash
python cli.py legacy-main MNIST -iterations 10000 \
    --data-dir data --result-load result-load/load_initial.gz
```

Artifacts:
`save_roo.gz`, `save_org.gz`, `save_tre.gz`, `c_info.txt`.

### DREAM3 analysis / figures

```bash
python cli.py legacy-data training
python cli.py legacy-data method
python cli.py legacy-data result 42000 --interval 500
```

See `intcyt/legacy/README.md` for the full command catalogue.

---

## 3. Thermodynamic Ledger & Metrics

Modern helpers sit outside `legacy/` so the historical code remains untouched.

### Instrumentation

```python
from intcyt.thermo import Ledger, step_with_ledger
from intcyt.legacy.celloperad.cl_ope import Operad

operad = Operad(dim)
supercell = ...  # build via legacy API
ledger = Ledger()

for t, payload in enumerate(stream):
    step_with_ledger(
        operad,
        supercell,
        index=t,
        events=events,
        vector=payload["vector"],
        gamma=payload["gamma"],
        filtering=payload.get("filtering", [1.5, 1.5, 10]),
        ledger=ledger,
    )

counts, bit_bounds = ledger.to_timeseries()
```

Feed `counts` / `bit_bounds` into the metric modules in `intcyt/metrics/` (e.g., `intcyt.metrics.entropy`, `intcyt.metrics.lnd`, `intcyt.metrics.doe`, `intcyt.metrics.tur`) to reproduce LND/DOE/TUR diagnostics.

### CLI helper (`ledger-run`)

Use JSON inputs (`intcyt/thermo/README.md` shows the format):

```bash
python cli.py ledger-run --config demo/config.json --stream demo/stream.jsonl \
    --output demo/ledger.json
```

This is a proof-of-concept runner that wraps the legacy `intcyt` call with
`step_with_ledger` using synthetic streams.

---

## 4. Mathematical & Research References

- `README-math.md` — compact summary of the operadic framework (Box 1).
- `README-paper.md` — paper overview + citation placeholder.
- `documentation.md` — full tutorial / DREAM3 methodology (converted from the
  legacy PDF).

Scope note: this public repo demonstrates thermodynamically informed /
adaptive ML on the legacy IntCyt code.  A hybrid architecture combining these
principles with modern DL/LLM systems is under active development in-house.

---

## 5. Quick Links

- Legacy usage guide: [`intcyt/legacy/README.md`](intcyt/legacy/README.md)
- Thermodynamic helper guide: [`intcyt/thermo/README.md`](intcyt/thermo/README.md)
- CLI entry point: `cli.py` (installed as `intcyt`)
- Metrics modules: [`intcyt/metrics/`](intcyt/metrics/)
- Visualization stub: [`intcyt/viz/thermo_fig.py`](intcyt/viz/thermo_fig.py) — extend to
  plot DOE/LND/TUR panels once ledger outputs are available.
