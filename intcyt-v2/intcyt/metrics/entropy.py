"""Information-theoretic helpers for the thermodynamic metrics package.

These utilities are thin wrappers around the definitions used throughout the
supplementary ``math.pdf``: distributions are represented as non-negative vectors, and
all divergences are computed with a configurable logarithmic base so they can be
interpreted either as nats or bits.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import math


def normalize(values: Iterable[float], minimum: float = 0.0) -> List[float]:
    """Return a probability distribution derived from ``values``.

    Negative entries are clipped to ``minimum`` before normalization to keep the
    calculations numerically stable.
    """

    sanitized = [max(minimum, float(v)) for v in values]
    total = sum(sanitized)
    if total == 0:
        if not sanitized:
            return []
        return [1.0 / len(sanitized)] * len(sanitized)
    return [v / total for v in sanitized]


def shannon(probabilities: Sequence[float], base: float = math.e) -> float:
    """Shannon entropy H(p) = -Σ p log p with the requested logarithmic base."""

    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log(p, base)
    return entropy


def renyi(probabilities: Sequence[float], order: float, base: float = math.e) -> float:
    """Rényi entropy of arbitrary order (order ≠ 1)."""

    if order == 1:
        raise ValueError("Use shannon() when order==1.")
    total = sum(p ** order for p in probabilities if p > 0)
    if total == 0:
        return 0.0
    return math.log(total, base) / (1 - order)


def cross_entropy(
    p: Sequence[float],
    q: Sequence[float],
    base: float = math.e,
    epsilon: float = 1e-12,
) -> float:
    """Cross-entropy H(p, q) used for work/heat bookkeeping."""

    if len(p) != len(q):
        raise ValueError("Distributions must have the same length")
    value = 0.0
    for p_i, q_i in zip(p, q):
        if p_i <= 0:
            continue
        value -= p_i * math.log(max(q_i, epsilon), base)
    return value


def kl_divergence(
    p: Sequence[float],
    q: Sequence[float],
    base: float = math.e,
    epsilon: float = 1e-12,
) -> float:
    """Kullback–Leibler divergence D_KL(p || q)."""

    if len(p) != len(q):
        raise ValueError("Distributions must have the same length")
    divergence = 0.0
    for p_i, q_i in zip(p, q):
        if p_i <= 0:
            continue
        divergence += p_i * math.log(p_i / max(q_i, epsilon), base)
    return divergence


def jensen_shannon(
    p: Sequence[float],
    q: Sequence[float],
    base: float = math.e,
) -> float:
    """Jensen-Shannon divergence, the symmetrized and smoothed KL divergence."""

    if len(p) != len(q):
        raise ValueError("Distributions must have the same length")
    m = [(p_i + q_i) / 2.0 for p_i, q_i in zip(p, q)]
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)


def sliding_window_entropy(
    series: Sequence[Sequence[float]],
    window: int,
    base: float = math.e,
) -> List[Tuple[int, float]]:
    """Compute the average entropy over a sliding window of distributions."""

    if window <= 0:
        raise ValueError("window must be positive")
    entropies: List[Tuple[int, float]] = []
    for start in range(0, max(0, len(series) - window + 1)):
        window_slice = series[start : start + window]
        avg = sum(shannon(normalize(dist), base) for dist in window_slice) / window
        entropies.append((start + window - 1, avg))
    return entropies
