"""Canonical 6D phase-space coordinate convention.

The entire codebase uses a single, fixed state-vector layout. It matches the
Xsuite/MAD-X external ordering so reference cross-checks are direct. This choice
is documented authoritatively in ``docs/CONVENTIONS.md`` — do not deviate.

State vector ``(x, px, y, py, zeta, delta)``:

==========  =====  ======================================================
index       name   meaning
==========  =====  ======================================================
0           x      horizontal position [m]
1           px     horizontal momentum Px / P0 (normalised, dimensionless)
2           y      vertical position [m]
3           py     vertical momentum Py / P0 (normalised, dimensionless)
4           zeta   longitudinal position s - beta0*c*t [m]; reference = 0
5           delta  relative MOMENTUM deviation (P - P0) / P0 (dimensionless)
==========  =====  ======================================================

``zeta > 0`` means the particle is ahead of the synchronous (reference)
particle. ``delta`` is a *momentum* deviation, not an energy deviation — this
distinction changes the longitudinal transfer-matrix coefficients (see the
drift derivation in ``docs/CONVENTIONS.md``).
"""

from __future__ import annotations

X: int = 0
PX: int = 1
Y: int = 2
PY: int = 3
ZETA: int = 4
DELTA: int = 5

DIM: int = 6

COORD_NAMES: tuple[str, ...] = ("x", "px", "y", "py", "zeta", "delta")
