# Box 1 — Summary of the Mathematical Framework

*(For full details see Appendix A in the documentation/paper.)*

## Motivation

We seek a substrate-independent formulation of **adaptive specialization**. The
language must:

- Encode **hierarchical nesting** (compartments within compartments).
- Allow **reversible reorganization** (structures can assemble/dissolve).
- Track the **information-processing cost** of irreversible updates (dissipation).

## Operadic cells (hierarchical tree)

Systems are modeled as trees of *operadic cells* — algebraic compartments that
support multi-level nesting. Each cell has:

- **Cytosol**: undifferentiated resource pool shared by the whole cell.
- **Organelles**: specialized sub-compartments with specific functions.
- **Residual ledger**: accumulates the cost (bits) of logically irreversible
  operations.

The tree structure allows recursive operations at any level, capturing
multi-scale organization.

## Core operations

Two complementary processes drive structural evolution:

1. **Composition (assembly)** — combine components into specialized compartments.
2. **Factorization (disassembly)** — break compartments into simpler parts.

Additional processes:

- **Fission/Fusion** — tune compartment granularity across levels.
- **Content exchange** — move material between cytosol and organelles to maintain
  balance (homeostasis).

## Drives and system-level action

External inputs modulate organelle operations. Outputs are computed via
aggregate weighted sums across the compartment tree, propagating signals from
child to parent nodes until a system-level response emerges.

## Optimization objective $(U^2)$

The objective $U^2$ measures the squared difference between:

1. The response under the current specialized structure.
2. The response if all compartments were homogenized (no specialization).

- **Large $U^2$** ⇒ specialization is beneficial; compartments should be refined/preserved.
- **Small $U^2$** ⇒ compartments are redundant; structure can be simplified/dissolved.

Iteratively enforcing this objective aligns the architecture with observed inputs.

## Information-processing accounting

We partition the total entropy change:

$$
\Delta S_{\text{total}} = \Delta S_{\text{comp}} + \Delta S_{\text{res}}
$$

- $\Delta S_{\text{comp}}$ ↓ when compartmental organization sharpens (local order ↑).
- $\Delta S_{\text{res}}$ ↑ with every logically irreversible operation
  (information erasure / discarded content).

Costs are reported in **Landauer-normalized units (LND)** — bits-equivalent
dissipation that serves as a conservative lower bound on energy requirements.

## Learning dynamics

Each input cycle proceeds through:

1. Evaluate $U^2$.
2. Adjust compartment content via gradient-like steps if $U^2$ changes.
3. Reorganize structure via composition/factorization when needed.
4. Restore homeostasis (content/cytosolic exchanges).
5. Clear cytosolic byproducts and update the residual ledger (erasure cost).

During productive learning, structural fluctuations grow — akin to the
oscillatory dynamics observed in biological cells.

## Key insight

The operadic nonequilibrium optimization paradigm unifies:

- **Structure discovery** — specialization emerges via oscillatory drives that
  temporarily lower barriers for irreversible updates.
- **Thermodynamic accounting** — residual ledgers capture the bit-cost of those
  changes.

The result is an adaptive system that rectifies fluctuations into enduring,
functionally beneficial structure while tracking the physical budget of learning.
