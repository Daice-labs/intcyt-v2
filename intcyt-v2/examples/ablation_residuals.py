"""Residual ablation experiments for Intcyt on MNIST / Fashion-MNIST.

This script runs the same test-recall evaluation pipeline as
``examples/evaluate_test_recall.py``, but compares two training conditions:

1) **Full Intcyt** (default): residuals accumulate as in the legacy code.
2) **No-residuals ablation**: residual accumulation is disabled during training.

Implementation detail
---------------------
Residuals are accumulated in the legacy
``legacy.celloperad.cl_cel.Cell.spontaneous_reaction`` method. For the ablation
condition we temporarily monkey-patch that method during training so that:
- cytosol is still cleared and bookkeeping variables are updated
- the residual variable is held at 0

The readout is a simple majority-vote mapping from learned compartments
(prototypes) to labels, constructed on the training set, and evaluated on the
held-out test set.

Usage
-----
  python examples/ablation_residuals.py --dataset mnist --runs 5
  python examples/ablation_residuals.py --dataset fashion-mnist --runs 5
  python examples/ablation_residuals.py --dataset both --runs 5

Notes
-----
* For single-label multiclass classification, **micro recall == accuracy**.
* This script prints mean ± sd across independent seeds.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Import eval module for dataset loading + training/eval helpers.
import evaluate_test_recall as evalmod


def _datasets_arg_to_list(dataset: str) -> List[str]:
    key = dataset.strip().lower().replace("_", "-")
    if key in {"both", "all"}:
        return ["mnist", "fashion-mnist"]
    return [dataset]


@contextlib.contextmanager
def _residuals_disabled(enable: bool):
    """Context manager that disables residual accumulation in the legacy Cell."""

    if not enable:
        yield
        return

    # Import inside the context so simply importing this script does not mutate anything.
    from intcyt.legacy.celloperad.cl_cel import Cell as LegacyCell

    orig = LegacyCell.spontaneous_reaction

    def patched(self):  # noqa: ANN001
        # Copy the legacy behaviour *except* for the residual accumulation.
        new_residual = sum(self.cytosol)

        # Update fast bookkeeping (legacy code paths rely on these).
        for u in range(self.dimension):
            self.K[u] -= self.cytosol[u]
        self.SK = self.SK - new_residual

        # Clear cytosol, but keep residual fixed at 0.
        self.cytosol = [0] * self.dimension
        self.residual = 0.0
        return self

    LegacyCell.spontaneous_reaction = patched
    try:
        yield
    finally:
        LegacyCell.spontaneous_reaction = orig


def _make_jsonable_runs(runs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for m in runs:
        m2 = dict(m)
        # confusion_matrix is a numpy array in evalmod.evaluate
        if hasattr(m2.get("confusion_matrix"), "tolist"):
            m2["confusion_matrix"] = m2["confusion_matrix"].tolist()  # type: ignore[assignment]
        out.append(m2)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="mnist | fashion-mnist | both")
    parser.add_argument(
        "--data-dir", default=str(evalmod.REPO_ROOT / ".data"), help="Where to download/store datasets"
    )
    parser.add_argument(
        "--source",
        default="auto",
        help="Dataset source: 'auto' (default), 'torchvision', or 'idx'",
    )

    parser.add_argument("--arity", type=int, default=20, help="Initial number of organelles (prototypes)")
    parser.add_argument("--levels", type=int, default=0, help="Initial tree height (legacy default: 0)")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training samples to process")
    parser.add_argument("--epoch", type=int, default=4, help="Number of Intcyt cycles per sample")
    parser.add_argument("--selfsup", default=None, help="Optional: 'left' or 'right' to mask half the image")

    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs per condition")

    parser.add_argument(
        "--train-limit", type=int, default=None, help="Optional cap on samples for majority-vote mapping"
    )
    parser.add_argument("--test-limit", type=int, default=None, help="Optional cap on test samples")

    parser.add_argument("--verbose", action="store_true", help="Do not silence legacy per-step prints")
    parser.add_argument("--save-json", default=None, help="Optional path to write a JSON summary")

    args = parser.parse_args(list(argv) if argv is not None else None)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    datasets = _datasets_arg_to_list(args.dataset)
    conditions = [
        ("full", False),
        ("no_residuals", True),
    ]

    all_results: Dict[str, object] = {
        "args": vars(args),
        "datasets": {},
    }

    t0 = time.time()

    for ds in datasets:
        print("\n==============================")
        print(f"Dataset: {ds}")
        print("==============================")

        train_ds, test_ds = evalmod.load_dataset(ds, data_dir=data_dir, source=args.source)

        ds_block: Dict[str, object] = {}

        for cond_name, disable_residuals in conditions:
            micro_vals: List[float] = []
            macro_vals: List[float] = []
            runs: List[Dict[str, object]] = []

            print(f"\n--- Condition: {cond_name} (disable_residuals={disable_residuals}) ---")

            for run in range(int(args.runs)):
                seed = int(args.seed) + run
                cfg = evalmod.TrainConfig(
                    dataset=ds,
                    arity=int(args.arity),
                    levels=int(args.levels),
                    iterations=int(args.iterations),
                    epoch=int(args.epoch),
                    seed=seed,
                    verbose=bool(args.verbose),
                    selfsup=args.selfsup,
                )

                print(f"Run {run + 1}/{args.runs} (seed={seed})")

                with _residuals_disabled(bool(disable_residuals)):
                    sc = evalmod.train_intcyt_model(cfg, train_ds)

                protos = evalmod.PrototypeIndex.from_supercell(sc)
                proto_to_label = evalmod.build_majority_vote_readout(
                    protos=protos, sc=sc, train_ds=train_ds, limit=args.train_limit
                )
                metrics = evalmod.evaluate(
                    protos=protos,
                    proto_to_label=proto_to_label,
                    test_ds=test_ds,
                    limit=args.test_limit,
                )
                metrics["seed"] = seed
                metrics["n_prototypes"] = len(protos.ids)
                metrics["disable_residuals"] = bool(disable_residuals)
                runs.append(metrics)

                micro_vals.append(float(metrics["micro_recall"]))
                macro_vals.append(float(metrics["macro_recall"]))

                print(f"  micro recall (== accuracy): {metrics['micro_recall'] * 100:.2f}%")
                print(f"  macro recall:              {metrics['macro_recall'] * 100:.2f}%")

            micro_mean, micro_sd = evalmod._mean_std(micro_vals)
            macro_mean, macro_sd = evalmod._mean_std(macro_vals)

            print("\n  Summary for condition:")
            print(f"    micro recall mean ± sd: {micro_mean * 100:.2f}% ± {micro_sd * 100:.2f}%")
            print(f"    macro recall mean ± sd: {macro_mean * 100:.2f}% ± {macro_sd * 100:.2f}%")

            ds_block[cond_name] = {
                "disable_residuals": bool(disable_residuals),
                "runs": _make_jsonable_runs(runs),
                "summary": {
                    "micro_recall_mean": micro_mean,
                    "micro_recall_sd": micro_sd,
                    "macro_recall_mean": macro_mean,
                    "macro_recall_sd": macro_sd,
                },
            }

        all_results["datasets"][ds] = ds_block

        # Convenience: print a one-line comparison table for this dataset.
        full = ds_block["full"]["summary"]
        ablt = ds_block["no_residuals"]["summary"]

        print("\n=== Comparison (mean ± sd) ===")
        print(
            f"micro recall (full):         {float(full['micro_recall_mean']) * 100:.2f}% ± {float(full['micro_recall_sd']) * 100:.2f}%"
        )
        print(
            f"micro recall (no_residuals): {float(ablt['micro_recall_mean']) * 100:.2f}% ± {float(ablt['micro_recall_sd']) * 100:.2f}%"
        )
        print(
            f"macro recall (full):         {float(full['macro_recall_mean']) * 100:.2f}% ± {float(full['macro_recall_sd']) * 100:.2f}%"
        )
        print(
            f"macro recall (no_residuals): {float(ablt['macro_recall_mean']) * 100:.2f}% ± {float(ablt['macro_recall_sd']) * 100:.2f}%"
        )

    elapsed = time.time() - t0
    all_results["elapsed_s"] = float(elapsed)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(evalmod.json_dumps(all_results), encoding="utf-8")
        print(f"\nWrote: {out_path}")

    print(f"\nDone. Elapsed (s): {elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
