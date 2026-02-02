"""Evaluate Intcyt test recall on MNIST / FashionMNIST.

This script implements the evaluation protocol discussed in the manuscript:

1) Train Intcyt *without labels*.
2) Create a simple supervised readout by mapping each learned compartment
   (prototype) to a class label via majority vote over the training set.
3) Report test recall.

Notes
-----
* For single-label multiclass classification, **micro-averaged recall equals
  overall accuracy**.
* The table in the paper reports a single “test recall” number; by default we
  print both micro and macro recall so you can choose what to report.

Usage
-----
  python examples/evaluate_test_recall.py --dataset mnist --runs 5
  python examples/evaluate_test_recall.py --dataset fashion-mnist --seed 123


    And if you want to run the single-condition “no_struct_ops” through the standard eval script:
python examples/evaluate_test_recall.py --dataset mnist --runs 5 --disable-struct-ops

The training loop here mirrors the legacy implementation but avoids requiring
IDX gzip files by using torchvision datasets.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import gzip
import struct

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)


# Ensure `import intcyt` works when the script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import intcyt  # noqa: E402  (import after sys.path fix)


@contextlib.contextmanager
def _residuals_disabled(enabled: bool):
    """Temporarily disable residual accumulation in the legacy runtime.

    The legacy ``Cell.spontaneous_reaction`` transfers all cytosol content into
    ``cell.residual`` and then clears the cytosol.  For the ablation we want to
    keep the cytosol clearing (so downstream code sees the same state shape)
    but force ``residual`` to remain zero.
    """

    if not enabled:
        yield
        return

    # Patch the legacy Cell class (this is what the shim re-exports).
    from intcyt.legacy.celloperad.cl_cel import Cell as LegacyCell  # noqa: WPS433

    orig = LegacyCell.spontaneous_reaction

    def _no_residual_spontaneous_reaction(self):
        new_residual = sum(self.cytosol)

        # Keep the fast updates from the legacy code.
        for u in range(self.dimension):
            self.K[u] -= self.cytosol[u]
        self.SK = self.SK - new_residual
        self.cytosol = [0] * self.dimension

        # Disable residual bookkeeping.
        self.residual = 0.0
        return self

    LegacyCell.spontaneous_reaction = _no_residual_spontaneous_reaction
    try:
        yield
    finally:
        LegacyCell.spontaneous_reaction = orig


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dim: int
    image_shape: Tuple[int, int, int]


def _dataset_spec(dataset: str) -> DatasetSpec:
    key = dataset.strip().lower().replace("_", "-")
    if key in {"mnist"}:
        return DatasetSpec(name="MNIST", dim=28 * 28 * 1, image_shape=(28, 28, 1))
    if key in {"fashion", "fashion-mnist", "fashionmnist"}:
        return DatasetSpec(name="FashionMNIST", dim=28 * 28 * 1, image_shape=(28, 28, 1))
    raise ValueError(f"Unsupported dataset: {dataset!r} (expected 'mnist' or 'fashion-mnist')")


class _ArrayDataset(torch.utils.data.Dataset):
    """Minimal Dataset wrapper around pre-loaded numpy arrays."""

    def __init__(self, images_uint8: np.ndarray, labels_int: np.ndarray):
        if images_uint8.ndim != 3:
            raise ValueError(f"images must have shape (N,H,W); got {images_uint8.shape}")
        if labels_int.ndim != 1:
            raise ValueError(f"labels must have shape (N,); got {labels_int.shape}")
        if images_uint8.shape[0] != labels_int.shape[0]:
            raise ValueError("images/labels length mismatch")
        self._images = images_uint8
        self._labels = labels_int

    def __len__(self) -> int:  # type: ignore[override]
        return int(self._labels.shape[0])

    def __getitem__(self, idx: int):  # type: ignore[override]
        img = torch.from_numpy(self._images[idx]).unsqueeze(0).float() / 255.0  # (1,H,W) in [0,1]
        label = int(self._labels[idx])
        return img, label


def _read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError(f"Bad IDX image file header: {path}")
        magic, num, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Unexpected IDX image magic {magic} in {path}")
        buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    if data.size != num * rows * cols:
        # Some files may be truncated; reshape will raise anyway, but keep a clearer message.
        raise ValueError(f"IDX image file has {data.size} bytes, expected {num * rows * cols} ({path})")
    return data.reshape(num, rows, cols)


def _read_idx_labels_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Bad IDX label file header: {path}")
        magic, num = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Unexpected IDX label magic {magic} in {path}")
        buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    if data.size != num:
        raise ValueError(f"IDX label file has {data.size} labels, expected {num} ({path})")
    return data.astype(np.int64)


def _load_from_idx(dataset: str, data_dir: Path) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load MNIST/Fashion-MNIST from raw IDX gzip files (no torchvision required)."""
    # By convention we expect the four standard names inside `data_dir`.
    train_images = data_dir / "train-images-idx3-ubyte.gz"
    train_labels = data_dir / "train-labels-idx1-ubyte.gz"
    test_images = data_dir / "t10k-images-idx3-ubyte.gz"
    test_labels = data_dir / "t10k-labels-idx1-ubyte.gz"

    missing = [p for p in (train_images, train_labels, test_images, test_labels) if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "IDX dataset files not found. Expected the following files inside --data-dir: "
            f"{missing_str}"
        )

    x_train = _read_idx_images_gz(train_images)
    y_train = _read_idx_labels_gz(train_labels)
    x_test = _read_idx_images_gz(test_images)
    y_test = _read_idx_labels_gz(test_labels)
    return _ArrayDataset(x_train, y_train), _ArrayDataset(x_test, y_test)


def _load_torchvision(dataset: str, data_dir: Path) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load datasets via torchvision (downloads automatically).

    We import torchvision lazily because some environments have mismatched
    torch/torchvision builds (which raises at import time).
    """

    try:
        from torchvision import datasets as tv_datasets  # type: ignore
        from torchvision import transforms as tv_transforms  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torchvision import failed: {exc}") from exc

    spec = _dataset_spec(dataset)
    tfm = tv_transforms.ToTensor()  # -> float32 in [0, 1]
    if spec.name == "MNIST":
        train = tv_datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tfm)
        test = tv_datasets.MNIST(root=str(data_dir), train=False, download=True, transform=tfm)
        return train, test
    if spec.name == "FashionMNIST":
        train = tv_datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=tfm)
        test = tv_datasets.FashionMNIST(root=str(data_dir), train=False, download=True, transform=tfm)
        return train, test
    raise AssertionError("unreachable")


def load_dataset(
    dataset: str,
    *,
    data_dir: Path,
    source: str = "auto",
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return (train_ds, test_ds).

    Parameters
    ----------
    source:
        "auto" (default): try torchvision first, then fall back to IDX.
        "torchvision": force torchvision.
        "idx": force raw IDX gzip loading.
    """

    src = source.strip().lower()
    if src not in {"auto", "torchvision", "idx"}:
        raise ValueError(f"Unsupported source: {source!r}")

    if src in {"auto", "torchvision"}:
        try:
            return _load_torchvision(dataset, data_dir)
        except Exception as exc:
            if src == "torchvision":
                raise
            print(f"[INFO] torchvision loader unavailable ({exc}); falling back to IDX files.")

    # IDX fallback / forced path.
    return _load_from_idx(dataset, data_dir)


def _to_int_label(label: torch.Tensor | int) -> int:
    if isinstance(label, torch.Tensor):
        return int(label.item())
    return int(label)


def _preprocess_vector(img: torch.Tensor) -> np.ndarray:
    """Match the legacy normalization: x := 1e-4 * x / max(x)."""
    flat = img.view(-1).detach().cpu().numpy().astype(np.float32)
    maxv = float(flat.max())
    if maxv > 0:
        flat = (1e-4 * flat) / maxv
    return flat


def _iter_leaf_supercells(sc: "intcyt.SuperCell") -> Iterator["intcyt.SuperCell"]:
    """Yield leaf SuperCells in deterministic left-to-right DFS order."""
    if sc.is_leaf:
        yield sc
    else:
        for child in sc.innercells:
            yield from _iter_leaf_supercells(child)


@dataclass
class PrototypeIndex:
    """Flat view of learned compartments (leaf organelles)."""

    ids: List[str]
    protos: np.ndarray  # shape (n, dim)
    proto_norms: np.ndarray  # shape (n,)

    @classmethod
    def from_supercell(cls, sc: "intcyt.SuperCell") -> "PrototypeIndex":
        ids: List[str] = []
        protos: List[np.ndarray] = []

        for leaf_i, leaf in enumerate(_iter_leaf_supercells(sc)):
            for org_i, org in enumerate(leaf.cell.organelles):
                # Stable identifier for majority-vote mapping / debugging.
                ids.append(f"leaf{leaf_i}:org{org_i}")
                protos.append(np.asarray(org, dtype=np.float32))

        if not protos:
            raise RuntimeError("No prototypes found: the trained SuperCell has no leaf organelles")

        proto_mat = np.stack(protos, axis=0)
        norms = np.linalg.norm(proto_mat, axis=1)
        return cls(ids=ids, protos=proto_mat, proto_norms=norms)

    def predict_id(self, x: np.ndarray) -> str:
        """Return the prototype id with maximum cosine similarity."""
        x_norm = float(np.linalg.norm(x))
        if x_norm == 0:
            # Degenerate input -> just pick the first prototype.
            return self.ids[0]

        dots = self.protos @ x  # (n,)
        denom = self.proto_norms * x_norm
        # Avoid divide-by-zero (prototypes can be all-zero early in training).
        sims = np.divide(dots, denom, out=np.full_like(dots, -np.inf), where=denom != 0)
        best = int(np.argmax(sims))
        return self.ids[best]


def _default_gamma_params(dataset: str) -> Tuple[float, float, List[float], List[List[Tuple[float, float]]], List[Tuple[float, float]]]:
    """Return (E, F, brightness, profiles, scores) defaults from legacy scripts."""
    key = dataset.strip().lower().replace("_", "-")
    brightness = [0.1, 0.25, 0.5, 0.75, 0.9]

    if key == "mnist":
        # Defaults used in legacy `main.py` for MNIST (self-supervised setting).
        profiles = [
            [(0, 0.45), (0, 0.25), (0, 0.1), (0, 0.05), (0, 0.01)],
            [(0, 0.3), (0, 0.3), (0, 0.1), (0, 0.05), (0, 0.01)],
        ]
        scores = [(0.7, 1.0), (0.8, 1.0)]
        E = 14.0
        F = 25.0
        return E, F, brightness, profiles, scores

    if key in {"fashion", "fashion-mnist", "fashionmnist"}:
        profiles = [
            [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
            [(0, 0.7), (0, 0.6), (0, 0.5), (0, 0.4), (0, 0.3)],
        ]
        scores = [(0.85, 1.0), (0.95, 1.0)]
        E = 13.5
        F = 22.5
        return E, F, brightness, profiles, scores

    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _make_silent_gamma(
    *,
    E: float,
    F: float,
    brightness: Sequence[float],
    profiles: Sequence[Sequence[Tuple[float, float]]],
    scores: Sequence[Tuple[float, float]],
    challenge: Optional[Tuple[int, int, int]] = None,
):
    """Re-implementation of `usf.gamma` without per-step printing."""

    def _contrast_tests(vector: Sequence[float]) -> List[bool]:
        # Match legacy behaviour: compute percent of pixels above each brightness threshold.
        maximum = 0.0
        for u, val in enumerate(vector):
            if challenge and (challenge[0] <= (u % challenge[2]) <= challenge[1]):
                continue
            maximum = max(maximum, float(val))

        if maximum == 0:
            percents = [0.0 for _ in brightness]
        else:
            counts = [0.0 for _ in brightness]
            total = 0.0
            for u, val in enumerate(vector):
                if challenge and (challenge[0] <= (u % challenge[2]) <= challenge[1]):
                    continue
                total += 1.0
                pixel_u = float(val) / maximum
                for j, thr in enumerate(brightness):
                    if pixel_u > thr:
                        counts[j] += 1.0
            percents = [c / total if total else 0.0 for c in counts]

        tests: List[bool] = []
        for prof in profiles:
            ok = True
            for (lo, hi), pct in zip(prof, percents):
                ok = ok and (lo <= pct <= hi)
            tests.append(ok)
        return tests

    def gamma(cell: "intcyt.Cell", pre_action: Sequence[Sequence[float]]):
        parameters: List[List[float]] = []

        maximum = 0.0
        for i in range(len(cell.organelles)):
            contrast = _contrast_tests(cell.organelles[i])
            agreement = cell.agreement(i, pre_action[i], *([challenge] if challenge else []))

            good_scenario = True
            for j in range(len(profiles)):
                bad_agreement = not (scores[j][0] <= agreement <= scores[j][1])
                good_scenario = good_scenario and not (contrast[j] and bad_agreement)

            good_agreement = agreement if good_scenario else 0.0
            maximum = max(maximum, good_agreement)

            row: List[float] = []
            for u in range(len(cell.organelles[i])):
                if challenge and (challenge[0] <= (u % challenge[2]) <= challenge[1]):
                    row.append(0.0)
                else:
                    row.append(good_agreement)
            parameters.append(row)

        maximum = 1.0 if maximum == 0.0 else maximum
        scale = 10.0 ** E
        for i in range(len(parameters)):
            for u in range(len(parameters[i])):
                parameters[i][u] = scale * ((parameters[i][u] / maximum) ** F)
        return parameters

    return gamma


@dataclass
class TrainConfig:
    dataset: str
    arity: int
    levels: int
    iterations: int
    epoch: int
    seed: int
    verbose: bool
    # Ablations
    disable_residuals: bool = False
    disable_fluctuations: bool = False
    # Stronger ablation: disable all structural operations (fission/fusion/compose)
    # while keeping numerical allostasis/homeostasis updates intact.
    disable_struct_ops: bool = False
    # Structural events (legacy defaults)
    start_cycle: int = 4
    fission_events: Tuple[int, ...] = (0,)
    fusion_events: Tuple[int, ...] = (2,)
    compose_events: Tuple[int, ...] = (1, 3)
    filtering: Tuple[float, float, float] = (1.5, 1.5, 10.0)
    selfsup: Optional[str] = None  # 'left' | 'right' | None


def _challenge_region(selfsup: Optional[str], *, width: int) -> Optional[Tuple[int, int, int]]:
    if not selfsup:
        return None
    key = selfsup.strip().lower()
    if key == "right":
        # Mask the right half -> ignore updates for columns [width/2, width-1].
        return (width // 2, width - 1, width)
    if key == "left":
        # Mask the left half -> ignore updates for columns [0, width/2-1].
        return (0, width // 2 - 1, width)
    raise ValueError(f"Unsupported --selfsup value: {selfsup!r} (expected 'left', 'right', or omitted)")


def train_intcyt_model(cfg: TrainConfig, train_ds: torch.utils.data.Dataset) -> "intcyt.SuperCell":
    """Train and return a SuperCell."""
    spec = _dataset_spec(cfg.dataset)

    # Reproducibility.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    operad = intcyt.Operad(spec.dim)
    sc = operad.generate(levels=cfg.levels, arity=cfg.arity, key=lambda: random.randint(0, 30))

    E, F, brightness, profiles, scores = _default_gamma_params(cfg.dataset)
    challenge = _challenge_region(cfg.selfsup, width=spec.image_shape[1] * spec.image_shape[2])
    gamma = _make_silent_gamma(E=E, F=F, brightness=brightness, profiles=profiles, scores=scores, challenge=challenge)

    events = [cfg.start_cycle, cfg.epoch, list(cfg.fission_events), list(cfg.fusion_events), list(cfg.compose_events)]
    if cfg.disable_struct_ops:
        # Strong ablation: remove *all* structural operations.
        # This disables the internal oscillatory fluctuation windows (fission/fusion)
        # and the compositional structural step (compose), while preserving the
        # numerical allostatic/homeostatic updates.
        events[2] = []  # no fission
        events[3] = []  # no fusion
        events[4] = []  # no compose
    elif cfg.disable_fluctuations:
        # Ablation: remove the internal compartment *dynamics* (fission/fusion)
        # that act as oscillatory fluctuation windows in the baseline.
        events[2] = []  # no fission
        events[3] = []  # no fusion
    filtering = list(cfg.filtering)

    # We iterate over the dataset sequentially (like the legacy gzip reader).
    # If iterations > len(train_ds), we cycle.
    idx = 0

    # Optionally silence extremely verbose legacy prints.
    devnull = None
    stdout_cm: contextlib.AbstractContextManager
    if cfg.verbose:
        stdout_cm = contextlib.nullcontext()
    else:
        devnull = open(os.devnull, "w")
        stdout_cm = contextlib.redirect_stdout(devnull)

    try:
        with stdout_cm:
            with _residuals_disabled(cfg.disable_residuals):
                while idx < cfg.iterations:
                    img, _ = train_ds[idx % len(train_ds)]
                    vector = _preprocess_vector(img).tolist()

                    # Run `epoch` cycles per sample (legacy default is 4).
                    for k in range(cfg.epoch):
                        intcyt.intcyt(operad, sc, cfg.epoch * idx + k, events, vector, gamma, filtering)

                    idx += 1
    finally:
        if devnull is not None:
            devnull.close()

    return sc


def build_majority_vote_readout(
    *,
    protos: PrototypeIndex,
    sc: "intcyt.SuperCell",
    train_ds: torch.utils.data.Dataset,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    """Map each prototype id to a class label using majority vote."""
    counts: Dict[str, Counter[int]] = defaultdict(Counter)
    n = len(train_ds) if limit is None else min(limit, len(train_ds))
    for i in range(n):
        img, label = train_ds[i]
        x = _preprocess_vector(img)
        proto_id = protos.predict_id(x)
        counts[proto_id][_to_int_label(label)] += 1

    proto_to_label: Dict[str, int] = {}
    for proto_id, counter in counts.items():
        proto_to_label[proto_id] = int(counter.most_common(1)[0][0])

    return proto_to_label


def evaluate(
    *,
    protos: PrototypeIndex,
    proto_to_label: Dict[str, int],
    test_ds: torch.utils.data.Dataset,
    limit: Optional[int] = None,
    fallback_label: Optional[int] = None,
) -> Dict[str, object]:
    """Return a dict of evaluation metrics."""
    # Default fallback: global majority label across mapped prototypes.
    if fallback_label is None:
        all_labels = list(proto_to_label.values())
        fallback_label = int(Counter(all_labels).most_common(1)[0][0]) if all_labels else 0

    y_true: List[int] = []
    y_pred: List[int] = []

    n = len(test_ds) if limit is None else min(limit, len(test_ds))
    for i in range(n):
        img, label = test_ds[i]
        x = _preprocess_vector(img)
        proto_id = protos.predict_id(x)
        pred = proto_to_label.get(proto_id, fallback_label)
        y_true.append(_to_int_label(label))
        y_pred.append(int(pred))

    micro_recall = float(recall_score(y_true, y_pred, average="micro"))
    macro_recall = float(recall_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    return {
        "n_test": n,
        "micro_recall": micro_recall,
        "macro_recall": macro_recall,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="mnist | fashion-mnist")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / ".data"), help="Where to download/store datasets")
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
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs (seed, seed+1, ...)")

    parser.add_argument(
        "--disable-residuals",
        action="store_true",
        help="Ablation: disable residual accumulation during training.",
    )
    parser.add_argument(
        "--disable-fluctuations",
        action="store_true",
        help="Ablation: disable fission/fusion (oscillatory fluctuation windows) during training.",
    )
    parser.add_argument(
        "--disable-struct-ops",
        action="store_true",
        help="Ablation: disable all structural operations (fission/fusion/compose) during training.",
    )

    parser.add_argument("--train-limit", type=int, default=None, help="Optional cap on samples for majority-vote mapping")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional cap on test samples")

    parser.add_argument("--verbose", action="store_true", help="Do not silence legacy per-step prints")
    parser.add_argument("--save-json", default=None, help="Optional path to write a JSON summary")

    args = parser.parse_args(list(argv) if argv is not None else None)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = load_dataset(args.dataset, data_dir=data_dir, source=args.source)

    run_metrics: List[Dict[str, object]] = []
    micro_vals: List[float] = []
    macro_vals: List[float] = []

    t0 = time.time()
    for run in range(args.runs):
        seed = int(args.seed) + run
        cfg = TrainConfig(
            dataset=args.dataset,
            arity=int(args.arity),
            levels=int(args.levels),
            iterations=int(args.iterations),
            epoch=int(args.epoch),
            seed=seed,
            verbose=bool(args.verbose),
            disable_residuals=bool(args.disable_residuals),
            disable_fluctuations=bool(args.disable_fluctuations),
            disable_struct_ops=bool(args.disable_struct_ops),
            selfsup=args.selfsup,
        )

        print(f"\n=== Run {run + 1}/{args.runs} (seed={seed}) ===")
        sc = train_intcyt_model(cfg, train_ds)
        protos = PrototypeIndex.from_supercell(sc)
        proto_to_label = build_majority_vote_readout(
            protos=protos, sc=sc, train_ds=train_ds, limit=args.train_limit
        )
        metrics = evaluate(
            protos=protos, proto_to_label=proto_to_label, test_ds=test_ds, limit=args.test_limit
        )
        metrics["seed"] = seed
        metrics["n_prototypes"] = len(protos.ids)
        run_metrics.append(metrics)

        micro_vals.append(float(metrics["micro_recall"]))
        macro_vals.append(float(metrics["macro_recall"]))

        print(f"micro recall (== accuracy): {metrics['micro_recall'] * 100:.2f}%")
        print(f"macro recall:              {metrics['macro_recall'] * 100:.2f}%")
        print(f"accuracy:                 {metrics['accuracy'] * 100:.2f}%")

    micro_mean, micro_sd = _mean_std(micro_vals)
    macro_mean, macro_sd = _mean_std(macro_vals)
    elapsed = time.time() - t0

    print("\n=== Summary ===")
    print(f"dataset: {args.dataset}")
    print(f"runs: {args.runs}")
    print(f"micro recall mean ± sd: {micro_mean * 100:.2f}% ± {micro_sd * 100:.2f}%")
    print(f"macro recall mean ± sd: {macro_mean * 100:.2f}% ± {macro_sd * 100:.2f}%")
    print(f"elapsed (s): {elapsed:.1f}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Make JSON-serialisable (confusion matrices -> lists).
        serialisable = []
        for m in run_metrics:
            m2 = dict(m)
            if isinstance(m2.get("confusion_matrix"), np.ndarray):
                m2["confusion_matrix"] = m2["confusion_matrix"].tolist()
            serialisable.append(m2)
        out = {
            "dataset": args.dataset,
            "args": vars(args),
            "runs": serialisable,
            "summary": {
                "micro_recall_mean": micro_mean,
                "micro_recall_sd": micro_sd,
                "macro_recall_mean": macro_mean,
                "macro_recall_sd": macro_sd,
            },
        }
        out_path.write_text(json_dumps(out), encoding="utf-8")
        print(f"Wrote: {out_path}")

    return 0


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
