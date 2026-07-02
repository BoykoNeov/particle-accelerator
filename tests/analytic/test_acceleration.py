"""Analytic checks for the acceleration ramp (Stage 5).

The Stage-3 cavity kick ``A[sin(phi_s - k_rf zeta) - sin(phi_s)]`` already carries
the accelerating physics: the ``- sin(phi_s)`` term is the energy the reference
absorbs, so a synchronous particle (``zeta = 0``) gets zero net kick. Stage 5 turns
the ramp on -- the reference energy climbs by ``Delta E_s = q V sin(phi_s)`` per turn
-- and adds the accompanying adiabatic damping. The acceptance gates:

1. **Energy gain per turn == q V sin(phi_s)** (summed over cavities), both as a
   closed-form quantity and as the actual per-turn reference-energy increment.
2. **The synchronous particle stays synchronous**: launched at the origin it stays
   at the origin while the reference energy ramps -- asserted *together* (the ramp
   is real, and the particle rides it), so it is not a hollow ``delta == 0`` check.
3. **Consistent with the Stage-3 model**: with ``sin(phi_s) = 0`` the ramp is a
   no-op and :func:`accelerate` reproduces Stage-3 nonlinear tracking bit-for-bit.

Two physics nets beyond the bare gates:

- **Adiabatic damping, derived** (not a remembered factor): for a drift+cavity ring
  the normalised momentum telescopes to the exact closed form
  ``px[n] = px0 * P0(0)/P0(n)``.
- **Adiabatic invariant, not raw amplitude**: an off-momentum neighbour executes a
  *damped* synchrotron oscillation whose amplitude shrinks while the action
  ``delta_max^2 / Qs`` (area ~ amplitude^2 / frequency) is conserved. The geometric
  amplitude shrinking is physics, not a symplecticity leak.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    DELTA,
    PX,
    PY,
    ZETA,
    Dipole,
    Drift,
    Lattice,
    Particle,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    Tracker,
    accelerate,
    energy_gain_per_turn,
    longitudinal_hamiltonian,
    rf_bucket_height,
    separatrix,
    slip_factor,
    synchronous_phase,
)
from accsim.reference import CLIGHT

MASS0 = 938.27208816e6
GAMMA0 = 5.0
HARMONIC = 10
VOLTAGE = 2.0e6


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _bend_free_arc() -> list:
    """Drift + thin-quad cell: alpha_c = 0 => eta = -1/gamma0^2 < 0 (below transition)."""
    return [Drift(1.0), ThinQuadrupole(0.3), Drift(1.0), ThinQuadrupole(-0.3)]


def _above_transition_arc() -> list:
    """Dispersive FODO with bends: at gamma0 = 5 this has eta > 0 (above transition)."""
    return [
        ThinQuadrupole(0.5 / 2.5),
        Dipole(1.0, 0.15),
        ThinQuadrupole(-1.0 / 2.5),
        Dipole(1.0, 0.15),
        ThinQuadrupole(0.5 / 2.5),
    ]


def _accelerating_lattice(
    ref: ReferenceParticle,
    arc: list,
    energy_gain: float,
    voltage: float = VOLTAGE,
) -> tuple[Lattice, float]:
    """Build (lattice, phi_s) for a target ``energy_gain`` per turn on ``arc``."""
    circumference = sum(e.length for e in arc)
    above = slip_factor(Lattice(arc, ref)) > 0.0
    phi_s = synchronous_phase(voltage, energy_gain, above_transition=above)
    cav = RFCavity.from_harmonic(voltage, HARMONIC, circumference, ref, phi_s)
    return Lattice([*arc, cav], ref), phi_s


# --- gate 1: energy gain per turn == q V sin(phi_s) -----------------------------
def test_energy_gain_per_turn_equals_qV_sin_phis(ref: ReferenceParticle) -> None:
    phi_s = 0.25
    C = 2.0
    lat = Lattice([Drift(C), RFCavity.from_harmonic(VOLTAGE, HARMONIC, C, ref, phi_s)], ref)
    expected = ref.charge * VOLTAGE * math.sin(phi_s)
    assert energy_gain_per_turn(lat) == pytest.approx(expected, rel=1e-15)


def test_energy_gain_sums_over_cavities(ref: ReferenceParticle) -> None:
    # Multi-cavity support: two cavities (different V and phi_s) add their gains.
    C = 3.0
    c1 = RFCavity.from_harmonic(1.5e6, HARMONIC, C, ref, 0.20)
    c2 = RFCavity.from_harmonic(1.0e6, HARMONIC, C, ref, 0.35)
    lat = Lattice([Drift(C), c1, c2], ref)
    expected = ref.charge * (1.5e6 * math.sin(0.20) + 1.0e6 * math.sin(0.35))
    assert energy_gain_per_turn(lat) == pytest.approx(expected, rel=1e-15)


def test_stationary_bucket_has_zero_gain(ref: ReferenceParticle) -> None:
    C = 2.0
    below = Lattice([Drift(C), RFCavity.from_harmonic(VOLTAGE, HARMONIC, C, ref, 0.0)], ref)
    above = Lattice([Drift(C), RFCavity.from_harmonic(VOLTAGE, HARMONIC, C, ref, math.pi)], ref)
    assert energy_gain_per_turn(below) == 0.0
    assert energy_gain_per_turn(above) == pytest.approx(0.0, abs=1e-6)


def test_no_cavity_raises(ref: ReferenceParticle) -> None:
    with pytest.raises(ValueError, match="requires at least one RFCavity"):
        energy_gain_per_turn(Lattice(_bend_free_arc(), ref))


# --- gate 1 (dynamic): the reference energy ramps by exactly the gain ------------
def test_reference_energy_ramps_linearly(ref: ReferenceParticle) -> None:
    dE = 3.0e5
    lat, _ = _accelerating_lattice(ref, _bend_free_arc(), dE)
    res = accelerate(lat, Particle(), 100)
    # E0(n) = E0(0) + n * Delta E_s, exactly.
    n = np.arange(res.n_turns + 1)
    expected = ref.total_energy_eV + n * energy_gain_per_turn(lat)
    assert np.allclose(res.energy_eV, expected, rtol=1e-13, atol=0)
    # The per-turn increment is constant and equals the gain.
    assert np.allclose(np.diff(res.energy_eV), dE, rtol=1e-6)


# --- gate 2: the synchronous particle stays synchronous (and the ramp is real) ---
def test_synchronous_particle_stays_synchronous(ref: ReferenceParticle) -> None:
    dE = 3.0e5
    lat, _ = _accelerating_lattice(ref, _bend_free_arc(), dE)
    res = accelerate(lat, Particle(), 500)  # launched exactly on the reference
    # It never leaves the origin ...
    assert np.max(np.abs(res.states)) == 0.0
    # ... while the reference energy genuinely climbs (both, or the check is hollow).
    assert res.energy_eV[-1] > res.energy_eV[0]
    assert res.energy_eV[-1] - res.energy_eV[0] == pytest.approx(500 * dE, rel=1e-6)


def test_synchronous_stays_synchronous_above_transition(ref: ReferenceParticle) -> None:
    lat, phi_s = _accelerating_lattice(ref, _above_transition_arc(), 2.0e5)
    assert math.pi / 2 < phi_s < math.pi  # above-transition branch
    res = accelerate(lat, Particle(), 300)
    assert np.max(np.abs(res.states)) == 0.0
    assert res.energy_eV[-1] - res.energy_eV[0] == pytest.approx(300 * energy_gain_per_turn(lat))


# --- adiabatic damping: derived exact closed form -------------------------------
def test_transverse_adiabatic_damping_is_p0_ratio(ref: ReferenceParticle) -> None:
    # Drift + cavity ring, particle on-axis longitudinally (zeta=0 => no delta kick).
    # The drift leaves px, py untouched; the only change is the per-turn re-referencing
    # px, py *= P0(n)/P0(n+1), which telescopes to the exact closed form
    #     px[n] = px0 * P0(0)/P0(n).
    C = 2.0
    # Built directly: a pure drift is transverse-unstable, so slip_factor (used by
    # the helper) is undefined -- but it is bend-free, hence below transition.
    phi_s = synchronous_phase(VOLTAGE, 3.0e5, above_transition=False)
    lat = Lattice([Drift(C), RFCavity.from_harmonic(VOLTAGE, HARMONIC, C, ref, phi_s)], ref)
    px0, py0 = 1.0e-4, -2.0e-4
    res = accelerate(lat, Particle(px=px0, py=py0), 200)
    ratio = res.momentum_eV[0] / res.momentum_eV
    assert np.allclose(res.states[:, PX], px0 * ratio, rtol=1e-12, atol=0)
    assert np.allclose(res.states[:, PY], py0 * ratio, rtol=1e-12, atol=0)
    # delta was never kicked (zeta stayed 0) and is not spuriously excited.
    assert np.max(np.abs(res.states[:, DELTA])) == 0.0
    # Damping is real: |px| strictly decreases as the energy rises.
    assert np.all(np.diff(np.abs(res.states[:, PX])) < 0.0)


# --- adiabatic damping: the invariant, not the raw amplitude ---------------------
def test_neighbor_damped_oscillation_conserves_action(ref: ReferenceParticle) -> None:
    # An off-momentum neighbour of the synchronous particle librates (bounded, does
    # not run away) and its delta oscillation *damps* as the energy ramps -- while
    # the adiabatic invariant (synchrotron action ~ delta_max^2 / Qs) is conserved.
    dE = 3.75e5  # ~40% energy gain over the run
    arc = _bend_free_arc()
    lat, phi_s = _accelerating_lattice(ref, arc, dE)
    N = 5000
    res = accelerate(lat, Particle(zeta=0.0, delta=2.0e-3), N)
    zeta, delta = res.states[:, ZETA], res.states[:, DELTA]

    # Bounded libration: well inside the bucket (unstable fixed point at k zeta_u).
    zeta_u = abs((2.0 * phi_s - math.pi) / lat.elements[-1].k_rf(ref))
    assert np.max(np.abs(zeta)) < zeta_u

    # Envelope of |delta| in sliding windows; it must shrink (adiabatic damping).
    w = 200
    env = np.array([np.max(np.abs(delta[i : i + w])) for i in range(0, N - w, w)])
    idx = np.arange(len(env)) * w
    assert env[-1] < 0.97 * env[0]  # clearly damped, not merely noisy

    # Qs(n) from the small-amplitude formula (eta, cos phi_s constant on a bend-free
    # ring; only beta0^2 E0 ramps): the action delta_max^2 / Qs is the invariant.
    E = res.energy_eV[idx]
    beta2E = (1.0 - (MASS0 / E) ** 2) * E
    eta = slip_factor(lat)
    qs = np.sqrt(
        HARMONIC * abs(eta) * ref.charge * VOLTAGE * math.cos(phi_s) / (2.0 * math.pi * beta2E)
    )
    action = env**2 / qs
    # Raw amplitude damped ~9%; the action is conserved to a few % (window ripple).
    assert action.max() / action.min() - 1.0 < 0.05


# --- gate 3: reduces to Stage-3 nonlinear tracking when there is no gain ---------
def test_no_acceleration_matches_stage3_bit_for_bit(ref: ReferenceParticle) -> None:
    # phi_s = 0 (below transition) => sin phi_s = 0 => Delta E_s = 0 => r = 1 exactly,
    # so accelerate() is Stage-3 nonlinear track_turns to the bit.
    arc = _bend_free_arc()
    C = sum(e.length for e in arc)
    lat = Lattice([*arc, RFCavity.from_harmonic(VOLTAGE, HARMONIC, C, ref, 0.0)], ref)
    p = Particle(zeta=0.02, delta=1.0e-3)
    res = accelerate(lat, p, 300)
    stage3 = Tracker(lat).track_turns(p, 300, nonlinear=True)
    assert np.array_equal(res.states, stage3)
    assert np.all(res.energy_eV == ref.total_energy_eV)  # reference never moved


# --- synchronous_phase helper: stable branch, transition, round trip -------------
def test_synchronous_phase_selects_stable_branch(ref: ReferenceParticle) -> None:
    dE, V = 3.0e5, VOLTAGE
    s = dE / (ref.charge * V)

    below = synchronous_phase(V, dE, above_transition=False)
    above = synchronous_phase(V, dE, above_transition=True)
    # Both accelerate (sin = s) but sit on opposite sides of pi/2.
    assert math.sin(below) == pytest.approx(s) and 0.0 < below < math.pi / 2
    assert math.sin(above) == pytest.approx(s) and math.pi / 2 < above < math.pi

    # Stability: eta cos(phi_s) < 0 with the plane's own eta.
    eta_below = slip_factor(Lattice(_bend_free_arc(), ref))
    eta_above = slip_factor(Lattice(_above_transition_arc(), ref))
    assert eta_below < 0.0 and eta_below * math.cos(below) < 0.0
    assert eta_above > 0.0 and eta_above * math.cos(above) < 0.0


def test_synchronous_phase_zero_gain_is_stage3_phase() -> None:
    # No acceleration => the Stage-3 stationary phases: 0 below, pi above.
    assert synchronous_phase(VOLTAGE, 0.0, above_transition=False) == 0.0
    assert synchronous_phase(VOLTAGE, 0.0, above_transition=True) == pytest.approx(math.pi)


def test_synchronous_phase_round_trips_energy_gain(ref: ReferenceParticle) -> None:
    # A lattice built via synchronous_phase gains exactly the requested energy/turn.
    dE = 2.5e5
    lat, _ = _accelerating_lattice(ref, _above_transition_arc(), dE)
    assert energy_gain_per_turn(lat) == pytest.approx(dE, rel=1e-12)


def test_synchronous_phase_rejects_impossible_gain(ref: ReferenceParticle) -> None:
    with pytest.raises(ValueError, match="no synchronous phase"):
        synchronous_phase(1.0e5, 2.0e5, above_transition=False)  # need sin > 1


# --- harmonic-number interface --------------------------------------------------
def test_from_harmonic_sets_frequency_and_round_trips(ref: ReferenceParticle) -> None:
    C, h = 26.0, 12
    cav = RFCavity.from_harmonic(VOLTAGE, h, C, ref, 0.1)
    assert cav.frequency == pytest.approx(h * ref.beta0 * CLIGHT / C, rel=1e-15)
    assert cav.harmonic_number(ref, C) == pytest.approx(h, rel=1e-12)
    # k_rf C = 2 pi h exactly (the defining relation).
    assert cav.k_rf(ref) * C == pytest.approx(2.0 * math.pi * h, rel=1e-12)


def test_from_harmonic_validates_inputs(ref: ReferenceParticle) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RFCavity.from_harmonic(VOLTAGE, 0, 10.0, ref)
    with pytest.raises(ValueError, match="circumference"):
        RFCavity.from_harmonic(VOLTAGE, 5, 0.0, ref)


# --- moving-bucket guard: Stage-3 bucket funcs refuse phi_s != 0/pi -------------
def test_moving_bucket_functions_raise(ref: ReferenceParticle) -> None:
    lat, _ = _accelerating_lattice(ref, _bend_free_arc(), 3.0e5)  # phi_s in (0, pi/2)
    for fn in (rf_bucket_height, separatrix, longitudinal_hamiltonian):
        with pytest.raises(NotImplementedError, match="stationary"):
            fn(lat)


def test_stationary_bucket_still_works(ref: ReferenceParticle) -> None:
    # The guard keys on sin(phi_s); phi_s = 0 and pi (sin ~ 0) remain valid Stage-3.
    arc = _bend_free_arc()
    C = sum(e.length for e in arc)
    lat = Lattice([*arc, RFCavity.from_harmonic(VOLTAGE, HARMONIC, C, ref, 0.0)], ref)
    assert rf_bucket_height(lat) > 0.0
