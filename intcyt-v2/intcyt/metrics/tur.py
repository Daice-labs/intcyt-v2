"""Thermodynamic Uncertainty Relation helpers."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple


def current_statistics(currents: Iterable[float]) -> Tuple[float, float]:
    """Return (mean, variance) for the supplied current time-series."""

    samples = list(currents)
    if not samples:
        return 0.0, 0.0
    mean = sum(samples) / len(samples)
    variance = sum((value - mean) ** 2 for value in samples) / max(len(samples) - 1, 1)
    return mean, variance


def uncertainty_bound(
    mean: float,
    variance: float,
    entropy_production: float,
    temperature: float = 1.0,
) -> float:
    """Return the TUR bound variance/mean^2 ≥ 2/(Σ/ (k_B T))."""

    if mean == 0:
        return math.inf
    lhs = variance / (mean ** 2)
    rhs = 2.0 * temperature / max(entropy_production, 1e-12)
    return lhs - rhs
