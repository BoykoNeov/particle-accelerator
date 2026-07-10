r"""Analytic gates — synchrotron radiation & radiation damping (Stage 7, commit 1).

Radiation integrals ``I1..I4`` and the damping quantities built on them, on a
periodic **isomagnetic** ring (all bends the same ``|rho|``, total bend ``2*pi``).
The gates, ordered so a wrong integrand can't hide:

  1. **Robinson's theorem** ``J_x + J_y + J_z = 4`` — an exact algebraic identity for
     *any* lattice (the ``I4/I2`` cancels), machine precision. The structural gate.
  2. **Isomagnetic closed forms** ``I2 = 2*pi/rho`` and ``I3 = 2*pi/rho^2`` (a
     restatement of total-bend ``= 2*pi`` + isomagnetic) — pins the ``h^2`` / ``|h|^3``
     integrands and the arc-length weighting.
  3. ``I1 == alpha_c * C`` — an **independent within-baseline** check of the
     dispersion transport (``momentum_compaction`` computes the same ``∮ D_x h ds`` by a
     path this reuses, but it is separately validated vs the symplecticity identity
     and xtrack in ``test_momentum_compaction.py``).
  4. **Energy loss** ``U0 = C_gamma * E^4 / rho`` (isomagnetic) — the ``88.5 keV``
     electron formula; pins ``C_gamma`` and the ``I2`` assembly together.
  5. **Constants** ``C_gamma``, ``C_q`` — the exact rational coefficients (``4*pi/3``,
     ``55/(32*sqrt3)``) pinned symbolically (sympy, tight), and the assembled constants
     pinned against the textbook electron values ``8.846e-5 m/GeV^3`` / ``3.832e-13 m``
     (loose — those are rounded). Symbolic pins the factor, numeric pins the units.
  6. **Damping times** ``tau_i = 2 E T0 / (J_i U0)`` — the common-factor invariant
     ``tau_x J_x = tau_y J_y = tau_z J_z`` and ``tau_y = 2 E T0 / U0`` (``J_y = 1``),
     pinning the ``2`` and the revolution period.

There is **no** clean absolute closed form for the equilibrium emittance (``I5``); it
is gated separately by energy-scaling + xtrack in commit 2.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from accsim import (
    Dipole,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    momentum_compaction,
)
from accsim.radiation import (
    HBAR_C_EV_M,
    damping_partition_numbers,
    damping_times,
    energy_loss_per_turn,
    quantum_constant_cq,
    radiation_constant_cgamma,
    radiation_integrals,
)
from accsim.reference import CLIGHT

ELECTRON_MASS_EV = 0.51099895069e6
N_CELLS = 20
F_FOCAL = 2.5
L_BEND = 1.0
ANGLE = 2.0 * math.pi / (2 * N_CELLS)  # two bends per cell → total bend = 2*pi
RHO = L_BEND / ANGLE  # isomagnetic bending radius [m]


def _electron_ring(total_energy_eV: float = 1.0e9) -> Lattice:
    """Isomagnetic FODO ring of ``N_CELLS`` symmetric arc cells (total bend 2*pi)."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, total_energy_eV)
    cell = [
        ThinQuadrupole(0.5 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(-1.0 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]
    return Lattice(cell * N_CELLS, ref=ref)


# ---------------------------------------------------------------------------
# Gate 1 — Robinson's theorem (any lattice, exact).
# ---------------------------------------------------------------------------
def test_robinson_partition_sum() -> None:
    jx, jy, jz = damping_partition_numbers(_electron_ring())
    assert jx + jy + jz == pytest.approx(4.0, abs=1e-12)
    assert jy == 1.0  # flat lattice, no vertical bending/gradient


# ---------------------------------------------------------------------------
# Gate 2 — isomagnetic closed forms for I2, I3.
# ---------------------------------------------------------------------------
def test_isomagnetic_I2_I3() -> None:
    ri = radiation_integrals(_electron_ring())
    assert ri.i2 == pytest.approx(2.0 * math.pi / RHO, rel=1e-12)
    assert ri.i3 == pytest.approx(2.0 * math.pi / RHO**2, rel=1e-12)


# ---------------------------------------------------------------------------
# Gate 3 — I1 equals alpha_c * C (independent dispersion-transport path).
# ---------------------------------------------------------------------------
def test_I1_matches_momentum_compaction() -> None:
    ring = _electron_ring()
    ri = radiation_integrals(ring)
    assert ri.i1 == pytest.approx(momentum_compaction(ring) * ring.length, rel=1e-10)


# ---------------------------------------------------------------------------
# Gate 4 — energy loss per turn, isomagnetic closed form U0 = C_gamma E^4 / rho.
# ---------------------------------------------------------------------------
def test_energy_loss_isomagnetic() -> None:
    ring = _electron_ring()
    u0 = energy_loss_per_turn(ring)
    cg = radiation_constant_cgamma(ring.ref)
    e = ring.ref.total_energy_eV
    assert u0 == pytest.approx(cg * e**4 / RHO, rel=1e-10)
    # Sanity: a 1 GeV electron at rho ~ 6.4 m loses ~ 14 keV/turn (88.5 keV/GeV^4/m).
    assert u0 == pytest.approx(88.46e3 * (e / 1e9) ** 4 / RHO, rel=2e-3)


# ---------------------------------------------------------------------------
# Gate 5 — the radiation constants: symbolic rational + numeric units.
# ---------------------------------------------------------------------------
def test_cgamma_symbolic_and_numeric() -> None:
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 1.0e9)
    # Symbolic closed form C_gamma = 4*pi*r0 / (3 (m c^2)^3); same formula the code
    # uses, so it must match to full float precision (pins the 4*pi/3 factor).
    r0, mc2 = sp.symbols("r0 mc2", positive=True)
    expr = 4 * sp.pi * r0 / (3 * mc2**3)
    val = float(expr.subs({r0: ref.classical_radius_m, mc2: ref.mass_eV}))
    assert radiation_constant_cgamma(ref) == pytest.approx(val, rel=1e-14)
    # Numeric units: reproduces the textbook 8.846e-5 m/GeV^3 (rounded → loose).
    cgamma_m_per_gev3 = radiation_constant_cgamma(ref) * 1e27
    assert cgamma_m_per_gev3 == pytest.approx(8.846e-5, rel=1e-3)


def test_cq_symbolic_and_numeric() -> None:
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 1.0e9)
    # C_q = 55/(32 sqrt3) * hbar c / (m c^2); pin the exact rational symbolically.
    hbarc, mc2 = sp.symbols("hbarc mc2", positive=True)
    expr = sp.Rational(55, 32) / sp.sqrt(3) * hbarc / mc2
    val = float(expr.subs({hbarc: HBAR_C_EV_M, mc2: ref.mass_eV}))
    assert quantum_constant_cq(ref) == pytest.approx(val, rel=1e-14)
    # Numeric: reproduces the textbook C_q = 3.832e-13 m for the electron.
    assert quantum_constant_cq(ref) == pytest.approx(3.832e-13, rel=1e-3)


# ---------------------------------------------------------------------------
# Gate 6 — damping times: partition-weighting invariant + the 2 E T0 / U0 factor.
# ---------------------------------------------------------------------------
def test_damping_times_invariant() -> None:
    ring = _electron_ring()
    tx, ty, tz = damping_times(ring)
    jx, jy, jz = damping_partition_numbers(ring)
    # tau_i J_i is the common factor 2 E T0 / U0 for all three planes.
    assert tx * jx == pytest.approx(ty * jy, rel=1e-12)
    assert tz * jz == pytest.approx(ty * jy, rel=1e-12)
    # tau_y = 2 E T0 / U0 exactly (J_y = 1) — pins the 2 and the revolution period.
    e = ring.ref.total_energy_eV
    t0 = ring.length / (ring.ref.beta0 * CLIGHT)
    u0 = energy_loss_per_turn(ring)
    assert ty == pytest.approx(2.0 * e * t0 / u0, rel=1e-12)
    # All damping times positive and physically ~ms for a 1 GeV ring.
    assert tx > 0 and ty > 0 and tz > 0


def test_damping_time_is_amplitude_convention() -> None:
    # tau_i here is the AMPLITUDE damping time (betatron amplitude ~ e^{-t/tau});
    # emittance/action damp at tau/2. This matches quantum_lifetime's
    # amplitude_damping_time input (Stage 4), so the two compose without a stray 2.
    from accsim.lifetime import quantum_lifetime

    ring = _electron_ring()
    tau_x = damping_times(ring)[0]
    # A 6-sigma circular aperture with the ring's damping time gives a long but
    # finite quantum lifetime — just a smoke test that the convention plugs in.
    tau_q = quantum_lifetime(aperture=6.0, sigma=1.0, amplitude_damping_time=tau_x)
    assert tau_q > tau_x  # quantum lifetime >> damping time at 6 sigma
