r"""Analytic gates — classical ("mean") synchrotron radiation *in tracking* (B2).

Axis B's existing gates are all **design-route**: the radiation integrals ride the Twiss
functions and ``damping_times`` / ``equilibrium_*`` are closed forms evaluated on them.
Nothing there touches a tracked particle, which until B2 circulated forever without
losing an eV. These gates check the *map* that makes the loss real, and then that the
damping it produces is the damping the closed forms predict — two routes that share no
arithmetic, one of them written a year before the other existed.

The gates, ordered so a wrong kick cannot hide behind a right one:

  1. **Direction preservation, exactly.** A photon leaves along the direction of motion,
     so ``(px, py, 1+delta)`` all scale by ONE factor ``f``. Then ``pz`` scales by ``f``
     too and the angles ``x' = px/pz``, ``y' = py/pz`` are invariant to the last bit —
     not to leading order, *exactly*. It is the structural gate.
  2. **The factor from the radiated energy.** On-shell:
     ``f = sqrt(1 - U(2E-U)/(E^2-m^2))``, which is ``1 - U/(beta^2 E)`` to first order
     and **exactly** ``1 - U/E`` as ``m -> 0``. Derived symbolically (sympy).
  3. **The energy loss itself**, against the design route's ``U0``. The residual is not
     a tolerance: a turn's loss is ``U0 (1 - c U0/E)`` because the particle radiates at a
     progressively *lower* energy as it goes round, and ``c`` is asserted to be the same
     number across a factor 64 in ``U0/E``.
  4. **The discriminating gate: the vertical damping time**, measured by tracking the
     action for 1500 turns. It is the only gate that can see the ``px, py`` scaling —
     drop it (gate 6) and the longitudinal answer is unchanged and still right.
  5. **All three damping times**, from the eigenvalues of the tracked one-turn map at the
     radiation-shifted fixed point (the damped-map eigenanalysis, which is what xtrack's
     own ``radiation_analysis`` uses). Gate 4 checks this route against explicit tracking
     first, so the cheap measurement is earned rather than assumed.
  6. **The wrong map, asserted.** A delta-only kick anti-damps the *angle* inside the
     element (first order, positive) yet produces **exactly zero** net transverse damping
     per turn, because ``py`` is never touched and the RF restores ``delta``.
  7. **Robinson from the measured rates** — ``J_x + J_y + J_z = 4`` recovered from the
     tracked map, not from the formula that has it by construction, and converging to it
     as the lattice is sliced.
  8. **Where the two routes genuinely part company.** The damped-map eigenanalysis and
     the integral method disagree about the *split* ``J_x`` / ``J_z`` by an amount that
     grows with ``I4/I2`` — 0.4% on a normal arc, 11% on a very strong one — and **one**
     number explains both planes at once. Stage 7 already recorded this method difference
     against xtrack; here it is measured from inside.
  9. **The scalings**, ``U0 ∝ E^4`` with its predicted ``U0/E`` correction and
     ``tau ∝ 1/E^3``, which no absolute tolerance can fake.
 10. **Radiation is not symplectic**, deliberately: the map must *fail* both symplecticity
     checks, and that rejection is asserted rather than worked around.
 11. **The lumping order.** One kick per element evaluates the loss at the element's
     *entry* energy, so slicing converges it as ``(N-1)/N``; asserted as that law.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Dipole,
    Lattice,
    Particle,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    Tracker,
    closed_twiss,
)
from accsim.radiation import (
    damping_times,
    energy_loss_per_turn,
    radiation_constant_cgamma,
    radiation_integrals,
)
from accsim.radiation_kick import mean_radiation_kick
from accsim.reference import CLIGHT
from accsim.symplectic import is_symplectic_map, is_symplectic_map_canonical

ELECTRON_MASS_EV = 0.51099895069e6
L_BEND = 1.0

# Two rings, because the milestone needs two different things from one.
#
# FAST: 8 FODO cells, 3 GeV. rho = 2.55 m makes tau_y ~ 2100 turns, which is what
# makes an explicit damping measurement affordable at all (on axis B's own 1 GeV
# ring it is 144,000 turns). Its I4/I2 = 0.71 is extreme, which is exactly why it
# is *not* used for the damping-partition gates -- see gate 8.
#
# WEAK: 20 cells, 5 GeV. A normal arc (I4/I2 = 0.38) where the damped-map
# eigenanalysis and the integral method agree to 0.2%, so the partition numbers can
# be gated sharply. Above transition in both cases, so phi_s = pi.
FAST = {"cells": 8, "focal": 2.0, "energy": 3.0e9, "voltage": 10.0e6, "harmonic": 20}
WEAK = {"cells": 20, "focal": 2.5, "energy": 5.0e9, "voltage": 30.0e6, "harmonic": 20}


def _ring(
    cells: int,
    focal: float,
    energy: float,
    voltage: float,
    harmonic: int,
    rf: bool = True,
    slices: int = 1,
) -> Lattice:
    """Isomagnetic FODO ring, total bend ``2 pi``, with the RF that replaces ``U0``."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, energy)
    angle = 2.0 * math.pi / (2 * cells)
    bend = [Dipole(L_BEND / slices, angle / slices) for _ in range(slices)]
    cell = [
        ThinQuadrupole(0.5 / focal),
        *bend,
        ThinQuadrupole(-1.0 / focal),
        *bend,
        ThinQuadrupole(0.5 / focal),
    ]
    elements = list(cell) * cells
    lat = Lattice(elements, ref=ref)
    if not rf:
        return lat
    # Above transition, so the stable synchronous phase is pi. The cavity's kick is
    # zero at zeta = 0, so with radiation on the beam settles at the zeta where it
    # replaces exactly U0 -- which needs V > U0, and is why the voltages are large.
    cavity = RFCavity.from_harmonic(voltage, harmonic, lat.length, ref, phi_s=math.pi)
    return Lattice([*elements, cavity], ref=ref)


def _tau_turns(lat: Lattice) -> tuple[float, float, float]:
    """``damping_times`` converted from seconds to turns (amplitude convention)."""
    t0 = lat.length / (lat.ref.beta0 * CLIGHT)
    tx, ty, tz = damping_times(lat)
    return tx / t0, ty / t0, tz / t0


# ---------------------------------------------------------------------------
# Measurement machinery: the radiation-shifted fixed point, and the damping rates
# of the tracked map about it. Not physics -- the physics is what it is applied to.
# ---------------------------------------------------------------------------
def _one_turn_jacobian(tracker: Tracker, state: np.ndarray, step: float = 1e-7) -> np.ndarray:
    jac = np.empty((6, 6))
    for i in range(6):
        plus, minus = state.copy(), state.copy()
        plus[i] += step
        minus[i] -= step
        jac[:, i] = (
            tracker.track_once(plus, radiation="mean") - tracker.track_once(minus, radiation="mean")
        ) / (2.0 * step)
    return jac


def _equilibrium_orbit(tracker: Tracker) -> np.ndarray:
    """Newton on ``track_once(s) = s`` with radiation on.

    Damping to the fixed point by brute-force tracking is not good enough: the
    horizontal damping time is thousands of turns, so a "converged" orbit is still
    drifting and the drift contaminates every rate measured against it. Newton takes
    it to round-off in a handful of turns.
    """
    state = np.zeros(6)
    for _ in range(50):
        residual = tracker.track_once(state, radiation="mean") - state
        if np.max(np.abs(residual)) < 1e-14:
            break
        state = state - np.linalg.solve(_one_turn_jacobian(tracker, state) - np.eye(6), residual)
    return state


def _eigen_damping_times(lat: Lattice) -> tuple[float, float, float]:
    """``(tau_z, tau_y, tau_x)`` in turns from the tracked map's own eigenvalues.

    ``|lambda| = exp(-1/tau)`` for each conjugate pair, the amplitude convention
    ``damping_times`` uses. This is the damped-one-turn-map eigenanalysis — the method
    xtrack's ``radiation_analysis`` uses, and the one Stage 7's write-up already named
    as differing from the integral method on a strong ring.
    """
    tracker = Tracker(lat)
    jac = _one_turn_jacobian(tracker, _equilibrium_orbit(tracker))
    rates = -np.log(np.abs(np.linalg.eigvals(jac)))
    taus = sorted(1.0 / rates[rates > 1e-14])
    return taus[0], taus[2], taus[4]  # one of each conjugate pair, ascending


def _fitted_damping_time(actions: np.ndarray) -> float:
    """Turns per e-fold of the AMPLITUDE from an action history (``J ~ exp(-2n/tau)``)."""
    turns = np.arange(actions.size)
    good = actions > 0
    slope = np.polyfit(turns[good], np.log(actions[good]), 1)[0]
    return -2.0 / slope


# ---------------------------------------------------------------------------
# Gate 1 — the kick preserves the direction of motion EXACTLY.
# ---------------------------------------------------------------------------
def test_one_common_factor_leaves_the_angles_exactly_invariant() -> None:
    """Symbolic: scaling ``(px, py, 1+delta)`` by one ``f`` scales ``pz`` by ``f`` too."""
    px, py, d, f = sp.symbols("px py delta f", real=True, positive=True)
    pz = sp.sqrt((1 + d) ** 2 - px**2 - py**2)
    pz_new = sp.sqrt((f * (1 + d)) ** 2 - (f * px) ** 2 - (f * py) ** 2)
    assert sp.simplify(pz_new - f * pz) == 0
    assert sp.simplify(f * px / pz_new - px / pz) == 0
    assert sp.simplify(f * py / pz_new - py / pz) == 0


def test_the_tracked_kick_preserves_the_angles_to_machine_precision() -> None:
    """The same statement through the code, on a bend at a real amplitude."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 20.0e9)
    elem = Dipole(L_BEND, 2.0 * math.pi / 16, k1=0.6)
    st = np.array([1e-3, 5e-4, 1e-3, 5e-4, 0.0, 1e-3])
    plain = elem.track(st.copy(), ref)
    rad = elem.track(st.copy(), ref, radiation="mean")
    assert rad[5] < plain[5]  # it really did lose momentum
    pz_p = math.sqrt((1 + plain[5]) ** 2 - plain[1] ** 2 - plain[3] ** 2)
    pz_r = math.sqrt((1 + rad[5]) ** 2 - rad[1] ** 2 - rad[3] ** 2)
    for mom, pos in ((1, 0), (3, 2)):  # (px, x) and (py, y)
        assert rad[mom] / pz_r == pytest.approx(plain[mom] / pz_p, rel=1e-14)
        assert rad[pos] == plain[pos]  # positions are untouched by a momentum kick


# ---------------------------------------------------------------------------
# Gate 2 — the factor in terms of the radiated energy (symbolic).
# ---------------------------------------------------------------------------
def test_the_scale_factor_is_the_on_shell_momentum_ratio() -> None:
    E, m, U = sp.symbols("E m U", positive=True)
    f_exact = sp.sqrt((E - U) ** 2 - m**2) / sp.sqrt(E**2 - m**2)
    # the rationalised form the code uses -- it never subtracts two numbers of size E
    f_code = sp.sqrt(1 - U * (2 * E - U) / (E**2 - m**2))
    # compared squared: sympy will not move a symbolic sqrt across a quotient without
    # being told the sign, and both roots are the positive one (a momentum ratio)
    assert sp.simplify(f_exact**2 - f_code**2) == 0
    # first order in U: 1 - U/(beta^2 E), with beta = P/E
    beta = sp.sqrt(E**2 - m**2) / E
    assert sp.simplify(sp.series(f_exact, U, 0, 2).removeO() - (1 - U / (beta**2 * E))) == 0
    # and EXACTLY 1 - U/E in the massless limit (sqrt((1-u)^2) = 1-u for U < E), so
    # there is no second-order term to argue about at ultra-relativistic energy
    assert sp.simplify(f_code.subs(m, 0) ** 2 - (1 - U / E) ** 2) == 0
    assert float(f_code.subs({m: 0, E: 1, U: sp.Rational(1, 100)})) == pytest.approx(0.99)


def test_the_kick_constant_is_the_radiation_integrals_own_c_gamma() -> None:
    """One constant, two call sites: the kick must not carry a second copy."""
    from accsim.radiation_kick import radiation_constant_cgamma as cg_kick

    for energy in (1.0e9, 3.0e9, 20.0e9):
        ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, energy)
        assert cg_kick(ref) == radiation_constant_cgamma(ref)


# ---------------------------------------------------------------------------
# Gate 3 — one tracked turn against the design route's U0.
# ---------------------------------------------------------------------------
def test_a_tracked_turn_loses_U0_less_a_term_of_a_measured_and_constant_order() -> None:
    r"""``loss = U0 (1 - c U0/E)``, and ``c`` is the same number over a factor 64 in
    ``U0/E``.

    The tracked route radiates at the energy the particle *has*, which falls through the
    turn; the closed form evaluates everything at ``E0``. That is a real difference, not
    an error, so the gate is that its **order** is right and its coefficient stable —
    never a tolerance wide enough to swallow it.
    """
    coefficients = []
    for energy in (1.0e9, 2.0e9, 3.0e9, 4.0e9):
        lat = _ring(**{**FAST, "energy": energy}, rf=False)
        u0 = energy_loss_per_turn(lat)
        e0 = lat.ref.total_energy_eV
        out = Tracker(lat).track(Particle(0, 0, 0, 0, 0, 0), nonlinear=True, radiation="mean")
        loss = -out.delta * e0  # ultra-relativistic: dE = E0 * ddelta
        assert loss < u0  # radiating at a falling energy loses less, not more
        coefficients.append((1.0 - loss / u0) / (u0 / e0))
    assert min(coefficients) == pytest.approx(max(coefficients), rel=0.01)
    assert 0.9 < coefficients[0] < 2.0  # order unity, dominated by (N-1)/N over 16 bends


def test_a_ring_with_no_bends_radiates_nothing() -> None:
    """Only curvature radiates, and a thin quadrupole has no length to radiate over."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 3.0e9)
    lat = Lattice([ThinQuadrupole(0.25), Dipole(1.0, 0.0), ThinQuadrupole(-0.25)], ref=ref)
    out = Tracker(lat).track(Particle(0, 0, 0, 0, 0, 0), nonlinear=True, radiation="mean")
    assert out.delta == 0.0


# ---------------------------------------------------------------------------
# Gate 4 — the discriminating gate: vertical damping, measured by tracking.
# ---------------------------------------------------------------------------
def test_the_tracked_vertical_damping_time_is_the_closed_form() -> None:
    """1500 turns of a vertically displaced particle, fitted against ``damping_times``.

    Vertical is the clean plane: no dispersion, no coupling, and ``J_y = 1`` exactly, so
    this is the ``px, py`` scaling and nothing else. Drop that scaling and the measured
    time is infinite (the next gate); halve it and this misses by a factor of two.
    """
    lat = _ring(**FAST)
    tracker = Tracker(lat)
    tw = closed_twiss(lat)
    fixed = _equilibrium_orbit(tracker)
    state = fixed.copy()
    state[2] += 1e-4
    n_turns = 1500
    dev = np.empty((n_turns + 1, 2))
    dev[0] = state[[2, 3]] - fixed[[2, 3]]
    for turn in range(1, n_turns + 1):
        state = tracker.track_once(state, radiation="mean")
        dev[turn] = state[[2, 3]] - fixed[[2, 3]]
    gamma_y = (1.0 + tw.alpha_y**2) / tw.beta_y
    action = 0.5 * (
        gamma_y * dev[:, 0] ** 2
        + 2.0 * tw.alpha_y * dev[:, 0] * dev[:, 1]
        + tw.beta_y * dev[:, 1] ** 2
    )
    measured = _fitted_damping_time(action)
    assert measured == pytest.approx(_tau_turns(lat)[1], rel=1e-3)
    # ... and the cheap eigen route agrees with the expensive tracked one, which is what
    # earns the right to use it for the other two planes.
    assert measured == pytest.approx(_eigen_damping_times(lat)[1], rel=1e-3)


# ---------------------------------------------------------------------------
# Gate 5 — all three damping times, on a ring where both routes are valid.
# ---------------------------------------------------------------------------
def test_all_three_damping_times_match_the_closed_forms_on_a_normal_arc() -> None:
    lat = _ring(**WEAK)
    tau_z, tau_y, tau_x = _eigen_damping_times(lat)
    expected_x, expected_y, expected_z = _tau_turns(lat)
    assert tau_y == pytest.approx(expected_y, rel=1e-4)
    assert tau_x == pytest.approx(expected_x, rel=5e-3)
    assert tau_z == pytest.approx(expected_z, rel=5e-3)


def test_robinson_from_the_MEASURED_rates_and_its_convergence_under_slicing() -> None:
    """``J_i = tau_y/tau_i`` from the tracked map must sum to 4 — and does, better and
    better as the bends are sliced, because the one-kick-per-element lumping is what
    breaks it."""
    sums = []
    for slices in (1, 2, 4):
        tau_z, tau_y, tau_x = _eigen_damping_times(_ring(**FAST, slices=slices))
        sums.append(tau_y / tau_x + 1.0 + tau_y / tau_z)
    assert all(abs(s - 4.0) < 0.03 for s in sums)
    assert abs(sums[-1] - 4.0) < abs(sums[0] - 4.0) / 5.0
    assert sums[-1] == pytest.approx(4.0, abs=2e-3)


def test_the_synchrotron_period_fits_inside_the_longitudinal_damping_time() -> None:
    """The three-sided squeeze on the test ring, checked rather than assumed."""
    for params in (FAST, WEAK):
        lat = _ring(**params)
        M, _ = lat.one_turn_map()
        block = M[np.ix_([4, 5], [4, 5])]
        cos_mu = 0.5 * (block[0, 0] + block[1, 1])
        assert abs(cos_mu) < 1.0  # there is a bucket at all (phi_s = pi, above transition)
        qs = math.acos(cos_mu) / (2.0 * math.pi)
        tau_z = _tau_turns(lat)[2]
        assert 1.0 / qs < tau_z / 10.0  # ten synchrotron periods per damping time, at least
        assert tau_z < 1500.0  # ... and the damping itself is affordable to track
        assert energy_loss_per_turn(lat) < params["voltage"]  # the RF can replace it


# ---------------------------------------------------------------------------
# Gate 6 — the wrong map, asserted from both sides.
# ---------------------------------------------------------------------------
def test_a_delta_only_kick_anti_damps_the_angle_inside_the_element() -> None:
    """Symbolic: dropping ``px`` from the scaling raises ``x'`` at FIRST order."""
    px, py, d, eps = sp.symbols("px py delta epsilon", real=True, positive=True)
    pz = sp.sqrt((1 + d) ** 2 - px**2 - py**2)
    pz_wrong = sp.sqrt(((1 - eps) * (1 + d)) ** 2 - px**2 - py**2)
    lead = sp.simplify(sp.series(px / pz_wrong - px / pz, eps, 0, 2).removeO())
    assert sp.simplify(lead - eps * px * (1 + d) ** 2 / pz**3) == 0
    assert lead.subs({px: sp.Rational(1, 1000), py: 0, d: 0, eps: sp.Rational(1, 100000)}) > 0


def test_a_delta_only_kick_leaves_the_transverse_momenta_bit_for_bit_untouched() -> None:
    """The structural half: through one element ``px`` and ``py`` do not move at all."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 20.0e9)
    elem = Dipole(L_BEND, 2.0 * math.pi / 16, k1=0.4)
    st = np.array([1e-3, 5e-4, 1e-3, 5e-4, 0.0, 1e-3])
    plain = elem.track(st.copy(), ref)
    wrong = elem.track(st.copy(), ref, radiation="mean_delta_only")
    assert wrong[1] == plain[1] and wrong[3] == plain[3]
    assert wrong[5] < plain[5]  # the longitudinal half of the kick is there, and right


def test_a_delta_only_kick_produces_no_vertical_damping_at_all() -> None:
    """And per turn it is invisible: the fitted damping time is five orders of magnitude
    too long, on the very ring where the correct map hits the closed form to 3e-5."""
    lat = _ring(**FAST)
    tracker = Tracker(lat)
    tw = closed_twiss(lat)
    state = np.zeros(6)
    state[2] = 1e-4
    n_turns = 800
    dev = np.empty((n_turns + 1, 2))
    dev[0] = state[[2, 3]]
    for turn in range(1, n_turns + 1):
        state = tracker.track_once(state, radiation="mean_delta_only")
        dev[turn] = state[[2, 3]]
    gamma_y = (1.0 + tw.alpha_y**2) / tw.beta_y
    action = 0.5 * (
        gamma_y * dev[:, 0] ** 2
        + 2.0 * tw.alpha_y * dev[:, 0] * dev[:, 1]
        + tw.beta_y * dev[:, 1] ** 2
    )
    assert abs(_fitted_damping_time(action)) > 1000.0 * _tau_turns(lat)[1]


# ---------------------------------------------------------------------------
# Gate 8 — where the two routes genuinely part company, and by how much.
# ---------------------------------------------------------------------------
def test_the_partition_split_departs_from_the_integral_method_as_I4_over_I2_grows() -> None:
    r"""The damped-map eigenanalysis and the radiation integrals are different methods.

    They agree in the perturbative limit and part company on a strong ring, exactly as
    Stage 7 recorded against xtrack (~1% at ``I4/I2 = 0.38``). Measured from inside here,
    and the important half is that **one** number accounts for both planes: whatever
    ``I4/I2`` the tracked map implies, it reproduces ``J_x = 1 - I4/I2`` *and*
    ``J_z = 2 + I4/I2`` together — so this is a method difference, not a broken plane.
    """
    departures = []
    for cells, focal in ((20, 2.5), (12, 2.5), (8, 2.0)):
        lat = _ring(**{**FAST, "cells": cells, "focal": focal}, slices=2)
        ratio_integral = radiation_integrals(lat).i4 / radiation_integrals(lat).i2
        tau_z, tau_y, tau_x = _eigen_damping_times(lat)
        jx, jz = tau_y / tau_x, tau_y / tau_z
        ratio_measured = 1.0 - jx
        # one number, both planes: J_z = 2 + I4/I2 with the SAME measured I4/I2
        assert jz == pytest.approx(2.0 + ratio_measured, rel=3e-3)
        departures.append(abs(ratio_measured / ratio_integral - 1.0))
    assert departures[0] < 0.01  # a normal arc: the two routes agree
    assert departures[-1] > 0.03  # a very strong one: they do not
    assert departures[0] < departures[1] < departures[-1]


# ---------------------------------------------------------------------------
# Gate 9 — the scalings.
# ---------------------------------------------------------------------------
def test_the_tracked_energy_loss_scales_as_E_to_the_fourth_with_its_own_correction() -> None:
    r"""``U ∝ E^4``, corrected by the same ``c U0/E`` gate 3 measures.

    ``c`` is measured here from these two energies rather than copied from gate 3: a
    number pasted between tests is a number that survives a change to the test ring in
    one place and not the other.
    """
    losses, ratios, coefficients = [], [], []
    for energy in (2.0e9, 4.0e9):
        lat = _ring(**{**FAST, "energy": energy}, rf=False)
        u0 = energy_loss_per_turn(lat)
        e0 = lat.ref.total_energy_eV
        out = Tracker(lat).track(Particle(0, 0, 0, 0, 0, 0), nonlinear=True, radiation="mean")
        losses.append(-out.delta * e0)
        ratios.append(u0 / e0)
        coefficients.append((1.0 - losses[-1] / u0) / (u0 / e0))
    coefficient = sum(coefficients) / len(coefficients)
    predicted = 16.0 * (1.0 - coefficient * (ratios[1] - ratios[0]))
    assert losses[1] / losses[0] == pytest.approx(predicted, rel=1e-4)
    assert losses[1] / losses[0] == pytest.approx(16.0, rel=3e-3)  # ... and it IS E^4


def test_the_closed_form_damping_time_scales_as_one_over_energy_cubed() -> None:
    """``tau [turns] = 2 rho/(C_gamma E^3)`` — pure geometry, machine precision."""
    taus = [_tau_turns(_ring(**{**FAST, "energy": e}, rf=False))[1] for e in (2.0e9, 4.0e9)]
    assert taus[0] / taus[1] == pytest.approx(8.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Gate 10 — radiation is deliberately not symplectic.
# ---------------------------------------------------------------------------
def test_radiation_makes_the_map_non_symplectic_and_that_is_asserted() -> None:
    """Both checks: the linear one and L1's canonical ``(zeta, pzeta)`` one."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 20.0e9)
    elem = Dipole(L_BEND, 2.0 * math.pi / 16)
    st = np.array([1e-4, 1e-4, 1e-4, 1e-4, 0.0, 1e-4])
    assert is_symplectic_map(lambda s: elem.track(s, ref), st)
    assert is_symplectic_map_canonical(lambda s: elem.track(s, ref), st, ref)
    assert not is_symplectic_map(lambda s: elem.track(s, ref, radiation="mean"), st)
    assert not is_symplectic_map_canonical(lambda s: elem.track(s, ref, radiation="mean"), st, ref)


# ---------------------------------------------------------------------------
# Gate 11 — the lumping order, asserted as a law.
# ---------------------------------------------------------------------------
def test_slicing_converges_the_lumped_loss_as_one_over_n() -> None:
    r"""``dE(N) = U (1 - (N-1)/N * U/E)``: one kick per element evaluates the loss at the
    entry energy, so the excess over the converged answer falls as ``1/N``. Solving each
    slicing for that coefficient must give 1 to the ``O(U/E)`` the law itself drops."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 20.0e9)
    angle = 2.0 * math.pi / 16
    losses = {}
    for n in (1, 2, 4, 8, 16):
        lat = Lattice([Dipole(L_BEND / n, angle / n) for _ in range(n)], ref=ref)
        out = Tracker(lat).track(Particle(0, 0, 0, 0, 0, 0), nonlinear=True, radiation="mean")
        losses[n] = -out.delta
    u = losses[1]
    for n in (2, 4, 8, 16):
        coefficient = (1.0 - losses[n] / u) / (u * (n - 1) / n)
        assert coefficient == pytest.approx(1.0, abs=2.0 * u)
    assert losses[16] < losses[8] < losses[4] < losses[2] < losses[1]


# ---------------------------------------------------------------------------
# The seams: one kick, one constant, every entry point.
# ---------------------------------------------------------------------------
def test_the_kick_is_the_same_whether_the_state_is_one_particle_or_a_bunch() -> None:
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 20.0e9)
    elem = Dipole(L_BEND, 2.0 * math.pi / 16, k1=0.4)
    states = np.array(
        [
            [1e-3, 0.0, 5e-4],
            [2e-4, 0.0, -1e-4],
            [0.0, 1e-3, 5e-4],
            [0.0, 3e-4, 1e-4],
            [0.0, 0.0, 0.0],
            [0.0, 1e-3, -1e-3],
        ]
    )
    bunch = elem.track(states.copy(), ref, radiation="mean")
    for i in range(states.shape[1]):
        one = elem.track(states[:, i].copy(), ref, radiation="mean")
        assert np.allclose(bunch[:, i], one, rtol=0.0, atol=1e-16)


def test_the_kick_helper_and_the_element_path_agree() -> None:
    """``mean_radiation_kick`` is the seam; the element must not carry a second copy."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 20.0e9)
    elem = Dipole(L_BEND, 2.0 * math.pi / 16, k1=0.4)
    st = np.array([1e-3, 5e-4, 1e-3, 5e-4, 0.0, 1e-3])
    plain = elem.track(st.copy(), ref)
    assert np.array_equal(
        elem.track(st.copy(), ref, radiation="mean"),
        mean_radiation_kick(elem, st, plain, ref),
    )


def test_a_misaligned_magnet_radiates_according_to_where_it_really_is() -> None:
    """The kick is applied in the body frame, so a shifted quadrupole's field is the
    field at the particle's position *relative to the magnet*."""
    from accsim import Quadrupole

    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 20.0e9)
    shifted = Quadrupole(0.4, 0.7, dx=1e-3)
    st = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    # the particle sits exactly on the shifted magnet's axis: no field, no radiation
    assert shifted.track(st.copy(), ref, radiation="mean")[5] == 0.0
    aligned = Quadrupole(0.4, 0.7)
    assert aligned.track(st.copy(), ref, radiation="mean")[5] < 0.0


def test_a_skew_quadrupole_radiates_exactly_like_the_normal_one_it_is() -> None:
    """``|B|^2 = k1^2 (x^2 + y^2)`` is roll-invariant, so a 45 deg roll cannot change it."""
    from accsim import Quadrupole, SkewQuadrupole

    k1, radius = 0.7, 2e-3
    normal, skew = Quadrupole(0.4, k1), SkewQuadrupole(0.4, k1)
    for phi in (0.0, 0.3, 1.1):
        x, y = radius * math.cos(phi), radius * math.sin(phi)
        bx_n, by_n = normal.normalized_field(x, y)
        bx_s, by_s = skew.normalized_field(x, y)
        assert bx_s * bx_s + by_s * by_s == pytest.approx(bx_n * bx_n + by_n * by_n, rel=1e-14)


def test_radiation_without_the_nonlinear_path_raises_instead_of_being_ignored() -> None:
    """The linear path has no element to radiate in; silently returning an undamped
    answer would be the worst outcome, so every entry point refuses."""
    lat = _ring(**FAST)
    tracker = Tracker(lat)
    particle = Particle(0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="nonlinear=True"):
        tracker.track(particle, radiation="mean")
    with pytest.raises(ValueError, match="nonlinear=True"):
        tracker.track_turns(particle, 1, radiation="mean")
    with pytest.raises(ValueError, match="must be one of"):
        tracker.track(particle, nonlinear=True, radiation="quantum")


def test_track_bunch_and_track_bunch_losses_radiate_like_the_single_particle_path() -> None:
    """The bunch entry points are not a separate implementation, and are gated as such.

    ``track_bunch(nonlinear=True)`` and the aperture-aware ``track_bunch_losses`` each
    walk the lattice by their own loop; a radiation flag that reached one and not the
    other, or reached neither, is exactly the ungated composition L3 found the hard way.
    """
    from accsim import Bunch

    lat = _ring(**FAST)
    tracker = Tracker(lat)
    states = np.array(
        [
            [1e-4, 0.0, -2e-4],
            [0.0, 1e-5, 0.0],
            [0.0, 1e-4, 1e-4],
            [0.0, 0.0, 2e-5],
            [0.0, 0.0, 0.0],
            [0.0, 1e-4, -1e-4],
        ]
    )
    one_turn = tracker.track_bunch(Bunch(states.copy()), nonlinear=True, radiation="mean")
    expected = tracker.track_once(states.copy(), radiation="mean")
    assert np.array_equal(one_turn.states, expected)
    assert np.all(one_turn.states[5] < states[5])  # every particle really lost momentum

    two_turns = tracker.track_bunch_losses(
        Bunch(states.copy()), n_turns=2, nonlinear=True, radiation="mean"
    )
    twice = tracker.track_once(tracker.track_once(states.copy(), radiation="mean"), "mean")
    assert np.allclose(two_turns.states, twice, rtol=0.0, atol=1e-16)
    assert two_turns.n_survived == states.shape[1]  # no apertures in this ring
