# Legacy IntCyt

This directory contains the original IntCyt research prototype described in
the accompanying documentation PDF.  The codebase predates the modern
`core/` API, so it exposes a collection of standalone scripts that expect the
classic datasets (SVHN, MNIST, Fashion-MNIST, DREAM3).

The legacy implementation is **not** installed as a package.  Instead you run
the scripts directly from the repository root after preparing the expected
files.

## 1. Requirements

1. Use Python 3.10+.
2. Install the scientific stack:

   ```bash
   python -m pip install -r legacy/requirements.txt
   ```

3. Download the datasets and place them under `data/`:

   | Dataset        | Expected files                                                     |
   | -------------- | ------------------------------------------------------------------ |
   | SVHN           | `data/train_32x32.mat`                                             |
   | MNIST          | `data/train-images-idx3-ubyte.gz`, `data/train-labels-idx1-ubyte.gz` |
   | Fashion-MNIST  | `data/t10k-images-idx3-ubyte.gz`, `data/t10k-labels-idx1-ubyte.gz` |
   | DREAM3         | `data/dream3_training.gz` (generated via `data_processing.py training`) |
   | Challenge docs | `data/expression_challenge/ExpressionData_UPDATED.txt`, etc.       |

Create the helper directories used by the scripts:

```bash
mkdir -p data result-load
```

## 2. Challenge Generator (`legacy/software_implementation/challenge.py`)

Creates self-supervision challenges by sampling the public datasets and
optionally masking regions with noise.

```bash
python legacy/software_implementation/challenge.py MNIST 20 -right
python legacy/software_implementation/challenge.py fashion-MNIST 20 -left-noisy 2 5
python legacy/software_implementation/challenge.py SVHN 50
```

Options (after the dataset and sample count):

- `-right`, `-left` — hide half of each image with white noise.
- `-right-noisy a b`, `-left-noisy a b` — keep both halves but sprinkle white
  noise with likelihood `a/b`.

Outputs are written to `result-load/load_initial.gz`.

## 3. Training Loop (`legacy/software_implementation/main.py`)

Runs the original IntCyt learner.  The modernized script uses argparse:

```bash
python legacy/software_implementation/main.py MNIST \
    --data-dir data \
    --result-load result-load/load_initial.gz \
    -iterations 10000
```

Key options:

| Option                | Description |
| --------------------- | ----------- |
| positional dataset    | One of `MNIST`, `fashion-MNIST`, `SVHN`, `DREAM3`. |
| `-iterations N`       | Number of learning iterations (default 30000). |
| `-seed N`             | Optional RNG seed. |
| `-load [idx]`         | Initialize from `result-load/load_initial.gz`. |
| `-load-selfsup-left/right [idx]` | Same as `-load` but marks the guidance region. |
| `--data-dir PATH`     | Directory containing the dataset files. |
| `--save-root PATH`    | Output directory for `save_roo.gz`, `save_org.gz`, `save_tre.gz`. |
| `--dream3-training PATH` | Location of `dream3_training.gz` when using `DREAM3`. |

The script captures snapshots of the super cell at every cycle (`save_*.gz`)
and writes textual metadata to `c_info.txt`.

## 4. Data Analysis & Visualization

`legacy/software_implementation/data_processing.py` exposes subcommands that
recreate the figures shown in the documentation and evaluate the DREAM3
challenge:

```bash
# Build DREAM3 training files and initialization snapshots
python legacy/software_implementation/data_processing.py training

# Plot one of the example figures
python legacy/software_implementation/data_processing.py figure1

# Re-run the DREAM3 analysis pipeline
python legacy/software_implementation/data_processing.py method

# Score saved organelles against the golden standard
python legacy/software_implementation/data_processing.py result 42000 --interval 500
```

The script internally uses the `data/expression_challenge/*.txt` files, so
make sure they are available before running the visualization commands.

For tree-structured visualizations you can also call
`legacy/software_implementation/visualize.py`, which reads the `save_*.gz`
artifacts emitted by `main.py` and renders tabular or tree-based summaries.

## 5. Outputs and Next Steps

Legacy scripts write gzip-compressed dumps in the current working directory.
Typical files:

- `result-load/load_initial.gz` — challenge initializations.
- `save_roo.gz`, `save_org.gz`, `save_tre.gz` — snapshots of the root, organelles,
  and tree at each iteration.
- `c_info.txt` — textual log of key hyperparameters.

Modern tooling (e.g., thermodynamic ledgers or new CLIs) can consume these
artifacts without modifying the legacy modules.  The next phase of the project
documents how to instrument the training loop via the opt-in `thermo`
wrappers.
