"""Learning Non-equilibrium Dissipation (LND) helpers.

These utilities follow the qualitative definition used throughout the IntCyt notes:
successive probability distributions over compartments form a trajectory, and the
cumulative KL divergence between consecutive steps serves as a proxy for dissipated work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import math

from .entropy import kl_divergence, normalize


@dataclass
class DissipationSummary:
    """Container with the aggregate dissipation properties of a trajectory."""

    total_dissipation: float
    average_step: float
    temperature: float

    @property
    def efficiency(self) -> float:
        """Convert dissipation into a unitless efficiency metric."""

        if self.total_dissipation == 0:
            return math.inf
        return 1.0 / self.total_dissipation


def trajectory_dissipation(
    distributions: Sequence[Sequence[float]],
    temperature: float = 1.0,
    base: float = math.e,
) -> DissipationSummary:
    """Estimate the dissipation of a learning trajectory.

    The value is the sum of KL divergences between successive distributions, scaled by the
    effective temperature (default 1) so it can be compared to the work term used in the
    documentation.
    """

    if len(distributions) < 2:
        return DissipationSummary(0.0, 0.0, temperature)

    normed = [normalize(dist) for dist in distributions]
    divergences = [
        kl_divergence(normed[i + 1], normed[i], base) for i in range(len(normed) - 1)
    ]
    total = sum(divergences) * max(temperature, 1e-12)
    average = total / len(divergences) if divergences else 0.0
    return DissipationSummary(total, average, temperature)


def dissipation_vs_gain(
    dissipation: Sequence[float],
    gains: Sequence[float],
) -> List[float]:
    """Compute the ratio between dissipation and performance gain per iteration."""

    if len(dissipation) != len(gains):
        raise ValueError("dissipation and gains must have the same length")
    ratios = []
    for d, g in zip(dissipation, gains):
        if g == 0:
            ratios.append(math.inf)
        else:
            ratios.append(d / g)
    return ratios
