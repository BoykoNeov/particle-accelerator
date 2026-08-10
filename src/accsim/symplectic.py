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

from collections.abc import Callable, Sequence

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


def jacobian(
    map_fn: Callable[[np.ndarray], np.ndarray],
    state: Sequence[float] | np.ndarray,
    step: float = 1e-6,
) -> np.ndarray:
    r"""Central-difference Jacobian ``J[i, j] = d(map_fn(state)_i) / d(state_j)``.

    The linearisation of a (possibly nonlinear) 6D map *about a given state* — for
    a linear element it returns that element's ``matrix`` to round-off, and for a
    nonlinear one it returns the local map that Twiss-style linear optics would
    see there. Central differences so the error is ``O(step^2)``, not ``O(step)``.

    ``step`` is absolute, and the default ``1e-6`` is a compromise: too small and
    round-off in the ``2*step`` division dominates; too large and the quadratic
    truncation error does. For a sextupole kick, whose second derivative is exactly
    constant, the truncation error vanishes identically and only round-off remains.
    """
    x0 = np.asarray(state, dtype=float)
    if x0.shape != (DIM,):
        raise ValueError(f"expected a length-{DIM} state vector, got shape {x0.shape}")
    if step <= 0.0:
        raise ValueError(f"step must be > 0, got {step}")
    out = np.empty((DIM, DIM))
    for j in range(DIM):
        plus, minus = x0.copy(), x0.copy()
        plus[j] += step
        minus[j] -= step
        out[:, j] = (np.asarray(map_fn(plus)) - np.asarray(map_fn(minus))) / (2.0 * step)
    return out


def is_symplectic_map(
    map_fn: Callable[[np.ndarray], np.ndarray],
    state: Sequence[float] | np.ndarray,
    step: float = 1e-6,
    atol: float = 1e-9,
) -> bool:
    """True if ``map_fn`` is symplectic *at* ``state``, by its numerical Jacobian.

    Symplecticity of a nonlinear map is a statement about its Jacobian at **every**
    point, not about a single matrix: ``M(x)^T J M(x) = J`` for all ``x``. This
    checks one point, so a test must sample several — in particular a nonzero
    amplitude, since every thin kick is trivially symplectic at the origin where
    its Jacobian is the identity.

    ``atol`` is looser than :func:`is_symplectic`'s ``1e-12`` because the Jacobian
    is a finite difference: with ``step = 1e-6`` the entries carry ``~1e-10``
    round-off, which squares into the product only as a first-order perturbation.
    """
    return is_symplectic(jacobian(map_fn, state, step), atol=atol)
