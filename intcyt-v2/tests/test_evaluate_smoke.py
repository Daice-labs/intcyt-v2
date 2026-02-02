from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

evalmod = importlib.import_module("evaluate_test_recall")
import intcyt  # noqa: F401


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 2, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self._images = torch.rand((n, 1, 28, 28), generator=gen)
        self._labels = torch.randint(0, 10, (n,), generator=gen)

    def __len__(self) -> int:  # type: ignore[override]
        return int(self._labels.shape[0])

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self._images[idx], int(self._labels[idx])


def test_eval_pipeline_smoke() -> None:
    cfg = evalmod.TrainConfig(
        dataset="mnist",
        arity=4,
        levels=0,
        iterations=1,
        epoch=1,
        seed=0,
        verbose=False,
        disable_residuals=True,
    )
    train_ds = _TinyDataset()
    sc = evalmod.train_intcyt_model(cfg, train_ds)
    protos = evalmod.PrototypeIndex.from_supercell(sc)
    assert len(protos.ids) > 0
