"""Symplectic-structure helpers.

A linear map ``M`` is symplectic iff ``M^T J M = J``, where ``J`` is the
canonical unit-symplectic matrix for the coordinate pairs
``(x, px), (y, py), (zeta, delta)``. Symplecticity is the structural guarantee
that long-term tracking neither artificially damps nor blows up — see §5/§7 of
the handoff. Every linear element matrix should pass :func:`is_symplectic`.

Caveat (documented in CONVENTIONS.md): ``(zeta, delta)`` is canonically conjugate
only in the ultrarelativistic / constant-velocity approximation used by the
linear maps here; ``delta`` (momentum) rather than the strictly-canonical
``ptau`` (energy) is the longitudinal coordinate. For the drift this distinction
does not break the ``M^T J M = J`` check, but it is flagged for the longitudinal
stages.
"""

from __future__ import annotations

import numpy as np

from .coords import DIM


def unit_symplectic_matrix() -> np.ndarray:
    """Return the 6x6 canonical symplectic form J for ``(x,px,y,py,zeta,delta)``.

    Block-diagonal with ``[[0, 1], [-1, 0]]`` on each conjugate pair.
    """
    J = np.zeros((DIM, DIM))
    for i in range(0, DIM, 2):
        J[i, i + 1] = 1.0
        J[i + 1, i] = -1.0
    return J


J6 = unit_symplectic_matrix()


def is_symplectic(matrix: np.ndarray, atol: float = 1e-12) -> bool:
    """True if ``matrix^T J matrix == J`` to absolute tolerance ``atol``."""
    M = np.asarray(matrix, dtype=float)
    if M.shape != (DIM, DIM):
        raise ValueError(f"expected a {DIM}x{DIM} matrix, got shape {M.shape}")
    return bool(np.allclose(M.T @ J6 @ M, J6, atol=atol, rtol=0.0))
