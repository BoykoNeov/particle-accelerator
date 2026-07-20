"""Nonlinear longitudinal dynamics: the RF bucket, its Hamiltonian and separatrix.

Small-amplitude synchrotron motion (the tune :func:`accsim.synchrotron_tune`) is
linear, but the RF *bucket* is intrinsically nonlinear: the cavity kick keeps its
full ``sin``, and the one-turn longitudinal map

    zeta_{n+1}  = zeta_n - eta C delta_n            (arc slip, from the slip factor)
    delta_{n+1} = delta_n + (q V / beta0^2 E0) [ sin(phi_s - k_rf zeta) - sin phi_s ]

is a pendulum (the "standard map" kick-drift pair). In the smooth (per-turn)
approximation it conserves the synchrotron Hamiltonian

    H(zeta, delta) = -1/2 eta C delta^2 + U(zeta),
    U(zeta) = -(q V / beta0^2 E0) [ (1/k_rf) cos(phi_s - k_rf zeta) - zeta sin phi_s ],

with ``dzeta/dn = dH/d(delta)`` and ``ddelta/dn = -dH/d(zeta)``. The **separatrix**
is the level set through the unstable fixed point; the region it encloses is the
bucket, and particles launched inside stay bounded (verified by ≥1e4-turn
tracking). See ``docs/CONVENTIONS.md`` -> *RF cavity / synchrotron tune*.

**Moving buckets are supported.** For ``sin phi_s != 0`` the ``-zeta sin phi_s``
tilt in ``U`` makes the bucket *asymmetric* about ``zeta = 0`` and shrinks it; the
height obeys the closed form

    delta_max(phi_s)^2 / delta_max(stationary)^2 = cos(psi) - (pi/2 - psi) sin(psi),
    psi = asin |sin phi_s|,

on **all four** branches (proton/electron x below/above transition) — derived
symbolically, see ``docs/CONVENTIONS.md`` -> *Moving-bucket acceptance*. The
bucket *area* is a non-elementary integral and is **not** provided.

A single RF harmonic is assumed: multiple cavities are allowed only if they share
``frequency`` and ``phi_s`` (their voltages add) — double-RF / multi-harmonic
buckets are out of scope.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.optimize import brentq

from .lattice import Lattice
from .twiss import slip_factor


class _EffectiveCavity:
    """The single effective RF harmonic of a lattice (summed voltage)."""

    __slots__ = ("amplitude", "k_rf", "phi_s")

    def __init__(self, amplitude: float, k_rf: float, phi_s: float) -> None:
        self.amplitude = amplitude  # A = q V_tot / (beta0^2 E0)  [dimensionless]
        self.k_rf = k_rf  # [1/m]
        self.phi_s = phi_s  # [rad]


def _effective_cavity(lattice: Lattice) -> _EffectiveCavity:
    """Collapse the lattice's RF cavities into one effective harmonic.

    Raises ``ValueError`` if there is no cavity, and ``NotImplementedError`` if the
    cavities do not share a single ``(frequency, phi_s)`` harmonic (Stage-3 scope).
    """
    from .elements.rfcavity import RFCavity

    cavities = [elem for elem in lattice.elements if isinstance(elem, RFCavity)]
    if not cavities:
        raise ValueError("no RFCavity in the lattice; there is no RF bucket.")
    freq0, phi0 = cavities[0].frequency, cavities[0].phi_s
    for cav in cavities[1:]:
        if not (math.isclose(cav.frequency, freq0) and math.isclose(cav.phi_s, phi0)):
            raise NotImplementedError(
                "rf_bucket_height/separatrix assume a single RF harmonic; the "
                "cavities differ in frequency or phi_s (double-RF is out of scope)."
            )
    ref = lattice.ref
    voltage = sum(cav.voltage for cav in cavities)
    amplitude = ref.charge * voltage / (ref.beta0**2 * ref.total_energy_eV)
    return _EffectiveCavity(amplitude, cavities[0].k_rf(ref), phi0)


def _adjacent_unstable_zetas(cav: _EffectiveCavity) -> tuple[float, float]:
    """The two unstable fixed points straddling the bucket centre ``zeta = 0``.

    ``dU/dzeta = A [sin phi_s - sin(phi_s - k_rf zeta)]`` vanishes on two families:
    ``k_rf zeta = 2 pi n`` (stable, the bucket centre is ``n = 0``) and
    ``k_rf zeta = 2 phi_s + pi + 2 pi n`` (unstable). Reducing the latter mod
    ``2 pi`` picks the two that bracket ``zeta = 0``, with no assumption about the
    sign or the range of ``phi_s``.
    """
    hi = (2.0 * cav.phi_s + math.pi) % (2.0 * math.pi)
    return (hi - 2.0 * math.pi) / cav.k_rf, hi / cav.k_rf


def _bucket_bounds(
    cav: _EffectiveCavity, eta: float, circumference: float
) -> tuple[float, float, float]:
    """``(zeta_u, zeta_other, delta_max^2)`` for the bucket around ``zeta = 0``.

    Both adjacent unstable points give a candidate separatrix; the bucket is the
    **inner** one, i.e. the smaller positive ``delta_max^2``. Which one that is
    depends on ``sign(eta q V)`` and on ``sign(sin phi_s)``, so it cannot be
    hardcoded: for ``q V < 0`` (an electron ring driven by a positive voltage) it
    is *not* the ``k_rf zeta_u = 2 phi_s - pi`` member that the stationary-only
    Stage-3 code assumed. ``zeta_other`` is the outer one, used to bracket the far
    turning point in :func:`separatrix`.
    """

    def potential(zeta: float) -> float:
        return -cav.amplitude * (
            math.cos(cav.phi_s - cav.k_rf * zeta) / cav.k_rf - zeta * math.sin(cav.phi_s)
        )

    u0 = potential(0.0)
    best: tuple[float, float, float] | None = None
    for zeta_u in _adjacent_unstable_zetas(cav):
        delta2 = 2.0 * (u0 - potential(zeta_u)) / (eta * circumference)
        if delta2 > 0.0 and (best is None or delta2 < best[2]):
            other = [z for z in _adjacent_unstable_zetas(cav) if z != zeta_u][0]
            best = (zeta_u, other, delta2)
    if best is None:
        raise ValueError(
            "no stable RF bucket (delta_max^2 <= 0 at both unstable fixed points): "
            f"phi_s = {cav.phi_s} is on the unstable branch for this lattice. "
            "Stability needs sign(cos phi_s) = -sign(eta q V); see "
            "accsim.synchronous_phase."
        )
    return best


def longitudinal_hamiltonian(lattice: Lattice) -> Callable[[float, float], float]:
    """Return the synchrotron Hamiltonian ``H(zeta, delta)`` (conserved per turn).

    ``H = -1/2 eta C delta^2 + U(zeta)`` (see module docstring). It is the
    approximate invariant of the nonlinear one-turn map; ``H`` constant along a
    trajectory (up to the small kick-drift discretisation ripple) is the
    longitudinal symplecticity check.
    """
    eta = slip_factor(lattice)
    circumference = lattice.length
    cav = _effective_cavity(lattice)

    def hamiltonian(zeta: float, delta: float) -> float:
        potential = -cav.amplitude * (
            math.cos(cav.phi_s - cav.k_rf * zeta) / cav.k_rf - zeta * math.sin(cav.phi_s)
        )
        return -0.5 * eta * circumference * delta**2 + potential

    return hamiltonian


def rf_bucket_height(lattice: Lattice) -> float:
    """Half-height of the RF bucket, ``delta_max`` (max ``|delta|``).

    Evaluated at the bucket centre (``zeta = 0``, where ``dU/dzeta = 0`` puts the
    maximum of the separatrix for a moving bucket too) from the Hamiltonian level
    of the separatrix: ``delta_max^2 = 2 [U(0) - U(zeta_u)] / (eta C)``.

    For a **stationary** bucket this reduces to ``delta_max = 2 Qs / (h |eta|)``;
    for a **moving** one it is smaller by the factor ``sqrt(cos psi - (pi/2 - psi)
    sin psi)``, ``psi = asin |sin phi_s|`` — both pinned symbolically in the tests.
    Raises ``ValueError`` if ``phi_s`` is on the unstable branch (no bucket).
    """
    eta = slip_factor(lattice)
    cav = _effective_cavity(lattice)
    _, _, delta2 = _bucket_bounds(cav, eta, lattice.length)
    return math.sqrt(delta2)


def separatrix(lattice: Lattice, n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Sample the RF-bucket separatrix as ``(zeta, delta)`` closed-curve arrays.

    Returns the upper (``delta >= 0``) branch followed by the lower branch reversed,
    so plotting the pair traces the closed separatrix.

    ``zeta`` spans the bucket from the bounding unstable fixed point ``zeta_u`` to
    the **far turning point** — the other root of ``U(zeta) = U(zeta_u)``, on the
    opposite side of ``zeta = 0``. For a stationary bucket that root is ``-zeta_u``
    and the curve is symmetric, one full RF wavelength wide; for a **moving** bucket
    it is transcendental (the potential is periodic-plus-tilt) and the bucket is
    asymmetric and narrower. It is bracketed between ``zeta = 0`` and the *other*
    adjacent unstable point — ``U`` is monotonic there, so the sign change is unique
    — and located with Brent's method rather than assumed.
    """
    eta = slip_factor(lattice)
    circumference = lattice.length
    hamiltonian = longitudinal_hamiltonian(lattice)
    cav = _effective_cavity(lattice)
    zeta_u, zeta_other, delta2 = _bucket_bounds(cav, eta, circumference)
    h_u = hamiltonian(zeta_u, 0.0)

    def barrier(zeta: float) -> float:
        return hamiltonian(zeta, 0.0) - h_u

    # For a stationary bucket the two unstable points are degenerate (equal barrier)
    # and the far tip is exactly -zeta_u. Compare against the bucket *depth* rather
    # than to 0.0: the residual is rounding, and near that double root the level set
    # is quadratic, so brentq would only reach ~sqrt(eps) precision.
    depth = abs(hamiltonian(0.0, 0.0) - h_u)
    if abs(barrier(zeta_other)) <= 1e-12 * depth:
        zeta_far = -zeta_u
    else:
        zeta_far = float(brentq(barrier, 0.0, zeta_other, xtol=1e-15, rtol=8.9e-16))
    lo, hi = sorted((zeta_u, zeta_far))
    zetas = np.linspace(lo, hi, n_points)
    # delta^2 = 2 (U(zeta) - U(zeta_u)) / (eta C), clipped to >= 0 at the tips.
    u = np.array([hamiltonian(z, 0.0) for z in zetas])
    delta2 = np.clip(2.0 * (u - h_u) / (eta * circumference), 0.0, None)
    delta = np.sqrt(delta2)
    zeta_loop = np.concatenate([zetas, zetas[::-1]])
    delta_loop = np.concatenate([delta, -delta[::-1]])
    return zeta_loop, delta_loop
