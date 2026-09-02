"""Symplectic-structure helpers.

A linear map ``M`` is symplectic iff ``M^T J M = J``, where ``J`` is the
canonical unit-symplectic matrix for the coordinate pairs
``(x, px), (y, py), (zeta, delta)``. Symplecticity is the structural guarantee
that long-term tracking neither artificially damps nor blows up — see §5/§7 of
the handoff. Every linear element matrix should pass :func:`is_symplectic`.

Caveat (documented in CONVENTIONS.md): ``(zeta, delta)`` is canonically conjugate
only in the ultrarelativistic / constant-velocity approximation used by the
linear maps here; ``delta`` (momentum) rather than the strictly-canonical
``p_zeta`` (energy) is the longitudinal coordinate.

**That caveat has teeth, and this module now carries the check that has them.**
It was written expecting the distinction never to bite; it does. Every *linear*
element matrix passes :func:`is_symplectic` in ``(zeta, delta)`` — the drift's
matrix is three independent shear blocks, and a shear is symplectic in whatever
pair it acts on. An **exact** (nonlinear) map is different: the exact drift is
symplectic by construction, being the flow of a Hamiltonian, yet its
``(zeta, delta)`` Jacobian misses ``M^T J M = J`` in exactly the two
``(px, delta)`` / ``(py, delta)`` entries, by an amount second order in the
amplitude (measured on ``Drift(2.0)`` at ``gamma0 = 5``: ``8.0e-14`` at
amplitude ``1e-6``, ``7.7e-8`` at ``1e-3``, ``7.7e-6`` at ``1e-2`` — a clean
square). The *same map* rewritten in ``(zeta, p_zeta)`` gives **exactly zero**.

The same is true of every later exact map, and the thick
:class:`~accsim.elements.quadrupole.Quadrupole` (L2) is a sharper case than the
drift: its ``(zeta, delta)`` residual is driven by ``x`` and ``y`` rather than by
the angles, so a *small-angle* state — the kind a test naturally picks — can sit
just inside a loose ``atol`` and pass for the wrong reason. Measured on
``Quadrupole(0.4, 1.7, roll=0.02)`` at a generic ``2e-3`` state: ``1.7e-9``, which
clears an ``atol`` of ``1e-8`` by a factor of six while the canonical check returns
a clean pass. A margin like that is not a result.

So in ``(zeta, delta)`` the more faithful map fails the check the cruder one
passes, and :func:`is_symplectic_map` cannot be used to judge an exact map. Use
:func:`is_symplectic_map_canonical`, which changes to ``p_zeta`` first. It is
also the gate that catches the tempting half-fix — making the transverse motion
exact while leaving ``zeta`` linear — which is wrong at **first** order in the
amplitude (``2.0e-4`` where the correct map is ``0``) and which the
``(zeta, delta)`` check misses because it rejects both.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .coords import DELTA, DIM
from .reference import ReferenceParticle


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

    **Only valid for a map whose longitudinal part is linear in ``delta``.** This
    tests the condition in accsim's ``(zeta, delta)`` coordinates, which are not
    canonically conjugate; an exact map is rejected here despite being symplectic.
    Use :func:`is_symplectic_map_canonical` for those — see the module docstring.
    """
    return is_symplectic(jacobian(map_fn, state, step), atol=atol)


def pzeta_from_delta(delta: float | np.ndarray, ref: ReferenceParticle) -> float | np.ndarray:
    r"""The canonical partner of ``zeta``, from the momentum deviation ``delta``.

    ``p_zeta = (E - E0) / (beta0^2 E0)`` — xtrack's ``pzeta``, equal to its
    ``ptau / beta0``. This is the variable canonically conjugate to
    ``zeta = s - beta0 c t``; ``delta = Delta p / p0`` is **not**, which is the
    whole reason this function exists (module docstring).

    Evaluated as ``(P^2 - P0^2) / ((E + E0) beta0^2 E0)`` rather than by
    subtracting two nearly-equal energies, so small ``delta`` keeps full relative
    precision instead of losing ``~log10(delta)`` digits to cancellation.

    Elementwise on an array of ``delta`` (P1 expands maps on a bunch of samples); a
    scalar in gives a plain ``float`` out, as before.
    """
    P0, m = ref.momentum_eV, ref.mass_eV
    E0 = ref.total_energy_eV
    d = np.asarray(delta, dtype=float)
    P = P0 * (1.0 + d)
    E = np.hypot(P, m)
    # E - E0 = (P^2 - P0^2)/(E + E0), and P^2 - P0^2 = P0^2 delta (2 + delta).
    out = P0 * P0 * d * (2.0 + d) / ((E + E0) * ref.beta0**2 * E0)
    return float(out) if out.ndim == 0 else out


def delta_from_pzeta(pzeta: float | np.ndarray, ref: ReferenceParticle) -> float | np.ndarray:
    """The momentum deviation ``delta``, from ``zeta``'s canonical partner.

    Exact inverse of :func:`pzeta_from_delta`. The energy *difference*
    ``dE = E - E0 = beta0^2 E0 p_zeta`` is carried directly and ``E`` itself is
    never formed: writing ``E = E0 (1 + beta0^2 p_zeta)`` and subtracting ``E0``
    again would round the small part away, losing ``delta`` to a relative
    ``1e-4`` by ``p_zeta ~ 1e-12`` (measured — the round-trip gate below caught
    exactly that). With ``dE`` in hand,

        P^2 = E^2 - m^2 = P0^2 + dE (2 E0 + dE)
        delta = (P - P0)/P0 = dE (2 E0 + dE) / (P0 (P + P0))

    and no step subtracts two nearly-equal numbers. Elementwise on an array, as
    :func:`pzeta_from_delta` is.
    """
    P0 = ref.momentum_eV
    E0 = ref.total_energy_eV
    dE = ref.beta0**2 * E0 * np.asarray(pzeta, dtype=float)
    numerator = dE * (2.0 * E0 + dE)
    P = np.sqrt(P0 * P0 + numerator)
    out = numerator / (P0 * (P + P0))
    return float(out) if out.ndim == 0 else out


def to_canonical(state: Sequence[float] | np.ndarray, ref: ReferenceParticle) -> np.ndarray:
    """``(x, px, y, py, zeta, delta)`` -> ``(x, px, y, py, zeta, p_zeta)``.

    A ``(6,)`` state or a ``(6, n)`` bunch; the conversion is elementwise in ``delta``.
    """
    out = np.array(state, dtype=float)
    out[DELTA] = pzeta_from_delta(out[DELTA], ref)
    return out


def from_canonical(state: Sequence[float] | np.ndarray, ref: ReferenceParticle) -> np.ndarray:
    """``(x, px, y, py, zeta, p_zeta)`` -> ``(x, px, y, py, zeta, delta)``, elementwise."""
    out = np.array(state, dtype=float)
    out[DELTA] = delta_from_pzeta(out[DELTA], ref)
    return out


def is_symplectic_map_canonical(
    map_fn: Callable[[np.ndarray], np.ndarray],
    state: Sequence[float] | np.ndarray,
    ref: ReferenceParticle,
    step: float = 1e-6,
    atol: float = 1e-9,
) -> bool:
    r"""True if ``map_fn`` is symplectic *at* ``state``, tested in ``(zeta, p_zeta)``.

    :func:`is_symplectic_map` with the longitudinal pair changed to the
    canonically conjugate one first: the map is conjugated by the coordinate
    change, ``g = to_canonical . map_fn . from_canonical``, and ``g``'s Jacobian
    is tested. ``map_fn`` still takes and returns accsim's usual
    ``(x, px, y, py, zeta, delta)`` — the change of variables happens here, so an
    element's ``track`` can be passed directly.

    This is the check an **exact** (nonlinear) map must pass, and the only one
    that separates a correct exact map from the half-fix that leaves ``zeta``
    linear. ``state`` is given in accsim coordinates, ``delta`` and all.
    """
    x0 = np.asarray(state, dtype=float)
    if x0.shape != (DIM,):
        raise ValueError(f"expected a length-{DIM} state vector, got shape {x0.shape}")

    def conjugated(canonical: np.ndarray) -> np.ndarray:
        return to_canonical(map_fn(from_canonical(canonical, ref)), ref)

    return is_symplectic(jacobian(conjugated, to_canonical(x0, ref), step), atol=atol)
