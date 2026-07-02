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

**Stationary bucket only (Stage 3).** ``phi_s = 0`` below transition, ``phi_s = pi``
above; the accelerating (``sin phi_s != 0``) moving bucket is Stage 5. A single RF
harmonic is assumed: multiple cavities are allowed only if they share ``frequency``
and ``phi_s`` (their voltages add) — double-RF / multi-harmonic buckets are out of
scope.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

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


def _unstable_fixed_point_zeta(cav: _EffectiveCavity) -> float:
    """``zeta`` of the unstable fixed point: k_rf zeta_u = 2 phi_s - pi."""
    return (2.0 * cav.phi_s - math.pi) / cav.k_rf


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
    """Half-height of the stationary RF bucket, ``delta_max`` (max ``|delta|``).

    Evaluated at the bucket centre (``zeta = 0``) from the Hamiltonian level of the
    separatrix: ``delta_max^2 = 2 [U(0) - U(zeta_u)] / (eta C)``. For a stationary
    bucket this reduces to the closed form ``delta_max = 2 Qs / (h |eta|)`` (pinned
    symbolically in the tests).
    """
    eta = slip_factor(lattice)
    circumference = lattice.length
    hamiltonian = longitudinal_hamiltonian(lattice)
    cav = _effective_cavity(lattice)
    zeta_u = _unstable_fixed_point_zeta(cav)
    # U(zeta) = H(zeta, 0); delta_max^2 = 2 (U(0) - U(zeta_u)) / (eta C).
    delta2 = 2.0 * (hamiltonian(0.0, 0.0) - hamiltonian(zeta_u, 0.0)) / (eta * circumference)
    if delta2 <= 0.0:
        raise ValueError("no stable RF bucket (delta_max^2 <= 0): check phi_s vs transition.")
    return math.sqrt(delta2)


def separatrix(lattice: Lattice, n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Sample the RF-bucket separatrix as ``(zeta, delta)`` closed-curve arrays.

    Returns the upper (``delta >= 0``) branch followed by the lower branch reversed,
    so plotting the pair traces the closed separatrix. ``zeta`` spans the bucket
    between its two unstable fixed points (for a stationary bucket, symmetric about
    ``zeta = 0``, one full RF wavelength wide).
    """
    eta = slip_factor(lattice)
    circumference = lattice.length
    hamiltonian = longitudinal_hamiltonian(lattice)
    cav = _effective_cavity(lattice)
    zeta_u = _unstable_fixed_point_zeta(cav)
    # The bucket is bounded by zeta_u and its mirror across the stable point zeta=0.
    lo, hi = sorted((zeta_u, -zeta_u))
    zetas = np.linspace(lo, hi, n_points)
    h_sep = hamiltonian(zeta_u, 0.0)
    # delta^2 = 2 (U(zeta) - h_sep) / (eta C), clipped to >= 0 at the tips.
    u = np.array([hamiltonian(z, 0.0) for z in zetas])
    delta2 = np.clip(2.0 * (u - h_sep) / (eta * circumference), 0.0, None)
    delta = np.sqrt(delta2)
    zeta_loop = np.concatenate([zetas, zetas[::-1]])
    delta_loop = np.concatenate([delta, -delta[::-1]])
    return zeta_loop, delta_loop
