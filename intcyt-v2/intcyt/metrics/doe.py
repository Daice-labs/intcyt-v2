"""Degree Of Exploration (DoE) metrics."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence, Tuple


def state_occupancy(states: Iterable[int]) -> Counter:
    """Return a histogram of visited states."""

    return Counter(states)


def degree_of_exploration(states: Iterable[int]) -> float:
    """Fraction of unique states relative to the total number of visits."""

    sequence = list(states)
    if not sequence:
        return 0.0
    unique = len(set(sequence))
    return unique / len(sequence)


def exploration_profile(
    states: Sequence[int],
    window: int,
) -> List[Tuple[int, float]]:
    """Sliding-window DoE profile."""

    if window <= 0:
        raise ValueError("window must be positive")
    profile: List[Tuple[int, float]] = []
    for start in range(0, max(0, len(states) - window + 1)):
        window_slice = states[start : start + window]
        profile.append((start + window - 1, degree_of_exploration(window_slice)))
    return profile
