r"""Analytic gates — synchrotron radiation & radiation damping (Stage 7).

Radiation integrals ``I1..I5``, the damping quantities, and the equilibrium
emittance/energy spread, on a periodic **isomagnetic** ring (all bends the same
``|rho|``, total bend ``2*pi``). The gates, ordered so a wrong integrand can't hide:

  1. **Robinson's theorem** ``J_x + J_y + J_z = 4`` — an exact algebraic identity for
     *any* lattice (the ``I4/I2`` cancels), machine precision. The structural gate.
  2. **Isomagnetic closed forms** ``I2 = 2*pi/rho`` and ``I3 = 2*pi/rho^2`` (a
     restatement of total-bend ``= 2*pi`` + isomagnetic) — pins the ``h^2`` / ``|h|^3``
     integrands and the arc-length weighting.
  3. ``I1 == alpha_c * C`` and (isomagnetic) ``I4 == h^2 * alpha_c * C`` — pin ``I1``
     and ``I4`` within the baseline against the validated ``momentum_compaction``.
  4. ``I5`` **vs an independent ``propagate_twiss`` integration** — the one integral on
     the new beta/alpha co-transport, so it is anchored (not just gamma-scaled) by
     re-integrating curly-H over a pre-sliced ring with the xtrack-validated Twiss.
  5. **Energy loss** ``U0 = C_gamma * E^4 / rho`` (isomagnetic) — the ``88.5 keV``
     electron formula; pins ``C_gamma`` and the ``I2`` assembly together.
  6. **Constants** ``C_gamma``, ``C_q`` — the exact rational coefficients (``4*pi/3``,
     ``55/(32*sqrt3)``) pinned symbolically (sympy, tight) + the textbook electron
     values (loose — rounded). Symbolic pins the factor, numeric pins the units.
  7. **Damping times** ``tau_i = 2 E T0 / (J_i U0)`` — the common-factor invariant
     ``tau_x J_x = tau_y J_y = tau_z J_z`` and ``tau_y = 2 E T0 / U0`` (``J_y = 1``).
  8. **Equilibrium** — the integrals are energy-independent geometry, so ``eps_x ∝
     gamma^2`` and ``sigma_delta ∝ gamma`` to machine precision (the scaling gate; there
     is no clean absolute closed form for ``eps_x`` — the absolute is anchored by gate 4
     within-baseline and by xtrack in ``tests/reference/``).
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
    closed_twiss,
    momentum_compaction,
    propagate_twiss,
)
from accsim.radiation import (
    HBAR_C_EV_M,
    _curly_h,
    damping_partition_numbers,
    damping_times,
    energy_loss_per_turn,
    equilibrium_emittance,
    equilibrium_energy_spread,
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
# Gate 3 — I1 and I4 pinned to alpha_c * C within the baseline.
# ---------------------------------------------------------------------------
def test_I1_matches_momentum_compaction() -> None:
    ring = _electron_ring()
    ri = radiation_integrals(ring)
    # I1 = ∮ D_x h ds is the *same* trapezoid alpha_c's quadrature route runs, so the
    # two agree to round-off -- that is what pins the dispersion transport here.
    assert ri.i1 == pytest.approx(
        momentum_compaction(ring, method="quadrature") * ring.length, rel=1e-10
    )
    # And against alpha_c's exact (D4 default) route, to the trapezoid's own error:
    # this is the physics check, and it is the arm that would catch a shared-machinery
    # bug the round-off comparison above cannot see.
    assert ri.i1 == pytest.approx(momentum_compaction(ring) * ring.length, rel=1e-5)


def test_I4_isomagnetic_equals_h2_alpha_c_C() -> None:
    # Isomagnetic (single |h|): I4 = ∮ D_x h^3 ds = h^2 ∮ D_x h ds = h^2 * alpha_c * C.
    # Pins I4 within the baseline via the validated I1 = alpha_c * C (so the partition
    # numbers rest on a validated integral, not just Robinson's structural identity).
    ring = _electron_ring()
    ri = radiation_integrals(ring)
    h = ANGLE / L_BEND
    assert ri.i4 == pytest.approx(
        h**2 * momentum_compaction(ring, method="quadrature") * ring.length, rel=1e-10
    )


def test_I5_matches_independent_propagate_twiss_integration() -> None:
    # I5 is the one integral riding on the NEW beta/alpha co-transport through the
    # dipole body, and curly-H varies through the bend so it does NOT reduce to a
    # validated factor times I1 (unlike I4). Independent within-baseline check: build
    # the ring with each dipole pre-sliced into K REAL sub-dipoles, get (beta, alpha,
    # D, D') at every sub-boundary from propagate_twiss (validated vs xtrack Twiss to
    # 1e-14), trapezoid-integrate curly_H |h|^3 over the dipoles, and compare to
    # radiation_integrals().i5. A match proves the co-transported optics equal
    # propagate_twiss's, so I5's absolute value is anchored (not only its gamma scaling).
    k = 64
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 1.0e9)
    sub_cell = [
        ThinQuadrupole(0.5 / F_FOCAL),
        *[Dipole(L_BEND / k, ANGLE / k) for _ in range(k)],
        ThinQuadrupole(-1.0 / F_FOCAL),
        *[Dipole(L_BEND / k, ANGLE / k) for _ in range(k)],
        ThinQuadrupole(0.5 / F_FOCAL),
    ]
    sliced = Lattice(sub_cell * N_CELLS, ref=ref)
    pts = propagate_twiss(sliced, closed_twiss(sliced))  # len(elements)+1 boundaries
    h = ANGLE / L_BEND
    ds = L_BEND / k
    i5_indep = 0.0
    for i, elem in enumerate(sliced.elements):
        if isinstance(elem, Dipole) and elem.angle != 0.0:
            h0 = _curly_h(pts[i].beta_x, pts[i].alpha_x, pts[i].disp_x, pts[i].disp_px)
            h1 = _curly_h(
                pts[i + 1].beta_x, pts[i + 1].alpha_x, pts[i + 1].disp_x, pts[i + 1].disp_px
            )
            i5_indep += abs(h) ** 3 * 0.5 * (h0 + h1) * ds  # composite trapezoid
    i5_code = radiation_integrals(_electron_ring(1.0e9)).i5
    assert i5_indep == pytest.approx(i5_code, rel=1e-6)


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


# ---------------------------------------------------------------------------
# Commit 2 — I5 curly-H, equilibrium emittance & energy spread.
#
# There is no clean absolute closed form for eps_x (the curly-H integral), so the
# analytic gate is the ENERGY SCALING: I1..I5 are pure geometry (energy-independent
# optics), so eps_x ∝ gamma^2 and sigma_delta ∝ gamma to machine precision. The
# absolute magnitude is pinned against xtrack in tests/reference/.
# ---------------------------------------------------------------------------
def test_integrals_are_energy_independent_geometry() -> None:
    # The transverse optics (beta, dispersion) don't depend on the reference energy,
    # so every radiation integral is identical at two energies — the premise of the
    # scaling gates below (and an independent check the I5 beta co-transport is
    # geometry, not accidentally energy-coupled).
    lo = radiation_integrals(_electron_ring(0.7e9))
    hi = radiation_integrals(_electron_ring(2.1e9))
    for a, b in zip(
        (lo.i1, lo.i2, lo.i3, lo.i4, lo.i5), (hi.i1, hi.i2, hi.i3, hi.i4, hi.i5), strict=True
    ):
        assert a == pytest.approx(b, rel=1e-12)


def test_equilibrium_emittance_scales_as_gamma_squared() -> None:
    e1, e2 = 0.8e9, 2.4e9  # gamma ratio 3
    ring1, ring2 = _electron_ring(e1), _electron_ring(e2)
    g_ratio = ring2.ref.gamma0 / ring1.ref.gamma0
    ratio = equilibrium_emittance(ring2) / equilibrium_emittance(ring1)
    assert ratio == pytest.approx(g_ratio**2, rel=1e-10)
    assert equilibrium_emittance(ring1) > 0.0  # geometric emittance, positive


def test_equilibrium_energy_spread_scales_as_gamma() -> None:
    e1, e2 = 0.8e9, 2.4e9
    ring1, ring2 = _electron_ring(e1), _electron_ring(e2)
    g_ratio = ring2.ref.gamma0 / ring1.ref.gamma0
    ratio = equilibrium_energy_spread(ring2) / equilibrium_energy_spread(ring1)
    assert ratio == pytest.approx(g_ratio, rel=1e-10)
    # Sanity: a 1 GeV electron ring has a per-mille-ish relative energy spread.
    sd = equilibrium_energy_spread(_electron_ring(1.0e9))
    assert 1e-4 < sd < 1e-2
