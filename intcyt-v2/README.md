# IntCyt v2

This repository contains a research implementation of **IntCyt** (a self-organizing, compartment-based learning system) and supporting utilities used in the manuscript.

It includes:

- **Legacy IntCyt** (`intcyt/legacy/`) — the original research prototype (cell/operad runtime + stand-alone scripts), lightly modernized for Python 3.
- **Modern helpers** (`intcyt/core/`, `intcyt/thermo/`, `intcyt/metrics/`) — thin wrappers and analysis tooling (thermodynamic ledgers + derived metrics).
- **Paper evaluation scripts** (`examples/`) — test-recall evaluation and ablations.

---

## Repository layout

| Path | Purpose |
| ---- | ------- |
| `intcyt/` | Installable Python package (top-level API + CLI implementation). |
| `intcyt/legacy/` | Original IntCyt modules (`celloperad`, `cellint`, `useful`) and legacy scripts (`software_implementation/`). |
| `intcyt/core/` | Programmatic drivers / simulator wrapper around the legacy runtime. |
| `intcyt/thermo/` | Opt-in snapshot & ledger utilities that observe the runtime without modifying it. |
| `intcyt/metrics/` | Entropy / LND / DOE / TUR utilities for analyzing ledger traces. |
| `intcyt/viz/` | Plotting helpers. |
| `examples/` | Scripts to reproduce Table S2/Table 2 (test recall) + ablations. |
| `tests/` | Small smoke tests for the evaluation pipeline. |
| `cli.py` | Convenience wrapper so `python cli.py ...` still works (the installed entry point is `intcyt`). |
| `new-documentation.md` | Compact documentation/tutorial (legacy workflow + thermodynamic extensions + notes). |

---

## Installation

### Recommended (reproduce evaluation + ablations)

Use Python **3.10+**.

```bash
pip install -r requirements.txt
pip install -e .
```

### Minimal install (no torch)

```bash
pip install -e .
```

Optional extras (defined in `pyproject.toml`):

```bash
pip install -e ".[eval]"   # torch/torchvision/scikit-learn (evaluation scripts)
pip install -e ".[dev]"    # pytest (tests)
```

---

## Data directories

The legacy scripts expect these directories:

```bash
mkdir -p data result-load
```

### Dataset files (IDX gzip)

If you use `--source idx` for MNIST/Fashion-MNIST, place the standard IDX gzip files under `data/`:

| Dataset | Files |
| ------- | ----- |
| **SVHN** | `data/train_32x32.mat` |
| **MNIST** | `data/train-images-idx3-ubyte.gz`, `data/train-labels-idx1-ubyte.gz`, `data/t10k-images-idx3-ubyte.gz`, `data/t10k-labels-idx1-ubyte.gz` |
| **Fashion-MNIST** | `data/train-images-idx3-ubyte.gz`, `data/train-labels-idx1-ubyte.gz`, `data/t10k-images-idx3-ubyte.gz`, `data/t10k-labels-idx1-ubyte.gz` |

By default, the evaluation scripts use `torchvision` (`--source auto`) and will download the datasets automatically.

---

## test recall

The paper report **test recall** for MNIST and Fashion-MNIST.

For the manuscript configuration with **K = 10** initial prototypes (arity) and a flat tree (**levels = 0**):

```bash
python examples/evaluate_test_recall.py --dataset mnist --runs 5 --seed 0 --iterations 30000 --levels 0 --arity 10
python examples/evaluate_test_recall.py --dataset fashion-mnist --runs 5 --seed 0 --iterations 30000 --levels 0 --arity 10
```

If you want to force IDX gzip loading (no `torchvision`):

```bash
python examples/evaluate_test_recall.py --dataset mnist --source idx --data-dir data --runs 5 --seed 0 --levels 0 --arity 10
```

### Residual ablation

```bash
python examples/ablation_residuals.py --dataset both --runs 5 --seed 0 --iterations 30000 --levels 0 --arity 10
```

### Structural/fluctuation ablation (+ `no_struct_ops`)

```bash
python examples/ablation_fluctuations.py --dataset both --runs 5 --seed 0 --iterations 30000 --levels 0 --arity 10
```

You can also run the strong ablation (`no_struct_ops`) inside the standard eval script:

```bash
python examples/evaluate_test_recall.py --dataset mnist --runs 5 --seed 0 --iterations 30000 --levels 0 --arity 10 --disable-struct-ops
```

---

## CLI shortcuts

Installing the package provides the `intcyt` command (defined in `pyproject.toml`).

Examples:

```bash
# Legacy training loop
intcyt legacy-main MNIST -iterations 1000 --data-dir data

# Challenge generator
intcyt legacy-challenge fashion-MNIST 20 -right

# DREAM3 helper commands
intcyt legacy-data training

# Inspect ledger API
intcyt ledger-info
```

For convenience, the same subcommands work via the repo-root wrapper:

```bash
python cli.py legacy-main MNIST -iterations 1000
```

---

## Thermodynamic ledger & metrics

See:

- `intcyt/thermo/README.md` — instrumentation guide + examples.
- `intcyt/metrics/` — entropy/LND/DOE/TUR utilities.

---

## License & citation

- License: see `LICENSE` (Apache-2.0)
- Citation metadata: `CITATION.cff`
