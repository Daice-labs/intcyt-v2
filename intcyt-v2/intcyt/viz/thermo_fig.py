"""Plotting helpers for thermodynamic diagnostics."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_counts_and_bits(
  counts: Sequence[float],
  bit_bounds: Sequence[float],
  ax: Optional[plt.Axes] = None,
) -> plt.Axes:
  """Plot per-iteration irreversible events alongside the bit lower bound."""

  if ax is None:
    _, ax = plt.subplots()

  counts_arr = np.asarray(counts)
  bits_arr = np.asarray(bit_bounds)
  steps = np.arange(len(counts_arr))

  ax.bar(steps, counts_arr, color="#4b8bbe", alpha=0.6, label="Events (Z/M)")
  ax.set_xlabel("Iteration")
  ax.set_ylabel("Irreversible ops")

  ax2 = ax.twinx()
  ax2.plot(steps, bits_arr, color="#e07a5f", label="Bits (LND lower bound)")
  ax2.set_ylabel("Bits")

  ax.legend(loc="upper left")
  ax2.legend(loc="upper right")
  return ax


def plot_efficiency_scatter(
  bits: Sequence[float],
  organization_metric: Sequence[float],
  ax: Optional[plt.Axes] = None,
) -> plt.Axes:
  """Scatter plot for DOE-style efficiency (organization gain vs bits)."""

  if ax is None:
    _, ax = plt.subplots()

  bits_arr = np.asarray(bits)
  org_arr = np.asarray(organization_metric)
  ax.scatter(bits_arr, org_arr, c="#264653", alpha=0.7)
  ax.set_xlabel("Bits (ΔS_res)")
  ax.set_ylabel("Organization gain (−ΔS_comp)")
  ax.set_title("Drive-to-Organization Efficiency (DOE proxy)")
  return ax


def plot_tur_ratio(
  precision: Sequence[float],
  variance: Sequence[float],
  bits: Sequence[float],
  ax: Optional[plt.Axes] = None,
) -> plt.Axes:
  """Visualize a TUR-style diagnostic ρ = variance / (precision^2 * bits)."""

  if ax is None:
    _, ax = plt.subplots()

  precision_arr = np.asarray(precision)
  variance_arr = np.asarray(variance)
  bits_arr = np.asarray(bits)
  rho = variance_arr / np.maximum(precision_arr**2 * np.maximum(bits_arr, 1e-12), 1e-12)

  ax.plot(rho, color="#2a9d8f")
  ax.set_xlabel("Window index")
  ax.set_ylabel("ρ (heuristic)")
  ax.set_title("TUR-style diagnostic")
  return ax


def plot_summary_panels(
  counts: Sequence[float],
  bit_bounds: Sequence[float],
  organization_metric: Sequence[float],
  precision: Sequence[float],
  variance: Sequence[float],
) -> Tuple[plt.Axes, plt.Axes, plt.Axes]:
  """Convenience helper to create the three standard panels."""

  fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
  plot_counts_and_bits(counts, bit_bounds, ax=axes[0])
  plot_efficiency_scatter(bit_bounds, organization_metric, ax=axes[1])
  plot_tur_ratio(precision, variance, bit_bounds, ax=axes[2])
  return axes
