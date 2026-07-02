"""Analytic checks for the RF bucket: height, separatrix, bounded tracking (Stage 3).

The nonlinear one-turn longitudinal map (linear arc slip + full ``sin`` cavity
kick) is a pendulum. Its acceptance gates:

1. **Bucket height** ``delta_max`` from the synchrotron Hamiltonian, derived
   symbolically (sympy) and shown to equal the closed form ``2 Qs / (h |eta|)`` --
   no remembered coefficient (flag D);
2. **Separatrix** is the Hamiltonian level set through the unstable fixed point;
3. **Bounded motion**: a particle launched *inside* the separatrix stays bounded
   (``zeta`` librates within the bucket) over >=1e4 turns, while one *outside*
   rotates -- ``zeta`` runs away. This is the longitudinal analogue of the
   transverse 1e4-turn symplecticity smoke test; the Hamiltonian is conserved
   along the bounded trajectory (the kick-drift map is symplectic).

The tracking uses a **bend-free** periodic lattice so ``alpha_c = 0`` and
``eta = -1/gamma0^2`` exactly (below transition, ``phi_s = 0``): the reduced
Hamiltonian is then exact (no dispersion coupling), giving a crisp separatrix for
the bounded/unbounded discriminator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    CLIGHT,
    DELTA,
    ZETA,
    Dipole,
    Drift,
    Lattice,
    Particle,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    Tracker,
    longitudinal_hamiltonian,
    rf_bucket_height,
    separatrix,
    slip_factor,
    synchrotron_tune,
)

MASS0 = 938.27208816e6
GAMMA0 = 5.0
HARMONIC = 10
VOLTAGE = 2.0e6


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _bend_free_arc() -> list:
    """Drift + thin-quad cell: alpha_c = 0 => eta = -1/gamma0^2 (below transition)."""
    return [Drift(1.0), ThinQuadrupole(0.3), Drift(1.0), ThinQuadrupole(-0.3)]


def _frequency(ref: ReferenceParticle, circumference: float) -> float:
    return HARMONIC * ref.beta0 * CLIGHT / circumference  # k_rf C = 2 pi h


def _bucket_lattice(ref: ReferenceParticle, voltage: float = VOLTAGE) -> Lattice:
    arc = _bend_free_arc()
    circumference = sum(e.length for e in arc)
    freq = _frequency(ref, circumference)
    return Lattice([*arc, RFCavity(voltage, freq, 0.0)], ref)  # phi_s = 0 below transition


# --- above transition (eta > 0, phi_s = pi): the sign-mirrored branch -----------
ABOVE_HARMONIC = 8
ABOVE_VOLTAGE = 1.0e6


def _above_transition_lattice(ref: ReferenceParticle) -> Lattice:
    """Dispersive arc (bends) at gamma0=5 => alpha_c > 1/gamma0^2 => eta > 0.

    Above transition the stable phase is phi_s = pi, which flips the RF slope sign
    (cos pi = -1); eta > 0 flips the arc drift sign too. delta_max^2 = 2(U(0) -
    U(zeta_u))/(eta C) must stay positive through both flips.
    """
    arc = [
        ThinQuadrupole(0.5 / 2.5),
        Dipole(1.0, 0.15),
        ThinQuadrupole(-1.0 / 2.5),
        Dipole(1.0, 0.15),
        ThinQuadrupole(0.5 / 2.5),
    ]
    circumference = sum(e.length for e in arc)
    freq = ABOVE_HARMONIC * ref.beta0 * CLIGHT / circumference
    return Lattice([*arc, RFCavity(ABOVE_VOLTAGE, freq, math.pi)], ref)


def test_bucket_height_matches_symbolic_hamiltonian(ref: ReferenceParticle) -> None:
    sp = pytest.importorskip("sympy")

    # --- symbolic: delta_max from H, and the closed form 2 Qs/(h|eta|) ----------
    A, k, etaC, h, eta_s = sp.symbols("A k etaC h eta", positive=True)
    # Stationary bucket phi_s = 0: U(zeta) = -A/k cos(k zeta); UFP at k zeta_u = pi.
    zeta = sp.Symbol("zeta", real=True)
    U = -A * (sp.cos(-k * zeta) / k)  # phi_s = 0 term of the potential
    zeta_u = sp.pi / k
    # eta C < 0 below transition; write etaC_signed = -etaC (etaC>0 magnitude).
    etaC_signed = -etaC
    delta_max2 = sp.simplify(2 * (U.subs(zeta, 0) - U.subs(zeta, zeta_u)) / etaC_signed)
    # Qs^2 = -(h eta A)/(2 pi) with eta < 0 => = h |eta| A/(2 pi); and etaC = |eta| C.
    # Substitute A via Qs^2 and k C = 2 pi h to recover 2 Qs/(h|eta|).
    Qs = sp.Symbol("Qs", positive=True)
    C = etaC / eta_s  # circumference from |eta| C / |eta|
    A_from_qs = 2 * sp.pi * Qs**2 / (h * eta_s)  # A = 2 pi Qs^2 / (h |eta|)
    delta_max2_qs = sp.simplify(delta_max2.subs(A, A_from_qs).subs(k, 2 * sp.pi * h / C))
    assert sp.simplify(delta_max2_qs - (2 * Qs / (h * eta_s)) ** 2) == 0

    # --- numeric: rf_bucket_height matches the tune-based form -------------------
    lat = _bucket_lattice(ref)
    eta = slip_factor(lat)
    qs = synchrotron_tune(lat)
    dmax_tune = 2.0 * qs / (HARMONIC * abs(eta))
    # Both use the exact arccos Qs / Hamiltonian; agree to the O((2 pi Qs)^2) that
    # separates the small-amplitude tune from the arccos value.
    assert rf_bucket_height(lat) == pytest.approx(dmax_tune, rel=1e-3)


def test_bucket_height_scales_as_sqrt_voltage(ref: ReferenceParticle) -> None:
    # delta_max^2 = 2 q V /(pi h |eta| beta0^2 E0)  => delta_max ∝ sqrt(V).
    d1 = rf_bucket_height(_bucket_lattice(ref, voltage=1.0e6))
    d4 = rf_bucket_height(_bucket_lattice(ref, voltage=4.0e6))
    assert d4 / d1 == pytest.approx(2.0, rel=1e-12)


def test_bucket_height_closed_form_in_voltage(ref: ReferenceParticle) -> None:
    # Independent voltage-form check: delta_max^2 = 2 q V /(pi h |eta| beta0^2 E0).
    lat = _bucket_lattice(ref)
    eta = slip_factor(lat)
    expected = math.sqrt(
        2.0
        * ref.charge
        * VOLTAGE
        / (math.pi * HARMONIC * abs(eta) * ref.beta0**2 * ref.total_energy_eV)
    )
    assert rf_bucket_height(lat) == pytest.approx(expected, rel=1e-12)


def test_separatrix_is_a_hamiltonian_level_set(ref: ReferenceParticle) -> None:
    lat = _bucket_lattice(ref)
    ham = longitudinal_hamiltonian(lat)
    zeta, delta = separatrix(lat, n_points=200)
    # Every sampled point sits on H = H(unstable fixed point) = H at the tips.
    h_vals = np.array([ham(z, d) for z, d in zip(zeta, delta, strict=True)])
    h_ref = ham(float(zeta[0]), 0.0)  # a tip (delta ~ 0)
    assert np.allclose(h_vals, h_ref, rtol=0, atol=1e-9 * abs(h_ref))
    # The bucket half-height (max |delta| on the curve) equals rf_bucket_height.
    assert np.max(np.abs(delta)) == pytest.approx(rf_bucket_height(lat), rel=1e-4)


def test_nonlinear_tracking_reduces_to_linear_for_tiny_amplitude(ref: ReferenceParticle) -> None:
    # At small amplitude the sin kick ~ its R65 linearisation, so element-by-element
    # nonlinear tracking matches the linear one-turn matrix to leading order.
    lat = _bucket_lattice(ref)
    dmax = rf_bucket_height(lat)
    p = Particle(zeta=1e-6, delta=1e-9 * dmax)
    lin = Tracker(lat).track_turns(p, 50, nonlinear=False)
    non = Tracker(lat).track_turns(p, 50, nonlinear=True)
    assert np.allclose(lin, non, rtol=1e-6, atol=1e-18)


def test_no_cavity_has_no_bucket(ref: ReferenceParticle) -> None:
    with pytest.raises(ValueError, match="no RFCavity"):
        rf_bucket_height(Lattice(_bend_free_arc(), ref))


def test_multi_harmonic_is_out_of_scope(ref: ReferenceParticle) -> None:
    arc = _bend_free_arc()
    circumference = sum(e.length for e in arc)
    f0 = _frequency(ref, circumference)
    lat = Lattice([*arc, RFCavity(1e6, f0, 0.0), RFCavity(1e6, 2 * f0, 0.0)], ref)
    with pytest.raises(NotImplementedError, match="single RF harmonic"):
        rf_bucket_height(lat)


def test_bucket_height_above_transition(ref: ReferenceParticle) -> None:
    # Sign-mirrored branch (eta > 0, phi_s = pi): delta_max stays positive and
    # matches BOTH closed forms, confirming the double sign flip cancels.
    lat = _above_transition_lattice(ref)
    eta = slip_factor(lat)
    assert eta > 0.0  # genuinely above transition
    qs = synchrotron_tune(lat)
    dmax = rf_bucket_height(lat)
    assert dmax == pytest.approx(2.0 * qs / (ABOVE_HARMONIC * abs(eta)), rel=1e-3)
    expected_v = math.sqrt(
        2.0
        * ref.charge
        * ABOVE_VOLTAGE
        / (math.pi * ABOVE_HARMONIC * abs(eta) * ref.beta0**2 * ref.total_energy_eV)
    )
    assert dmax == pytest.approx(expected_v, rel=1e-12)


@pytest.mark.slow
def test_above_transition_bucket_bounds_and_rotates(ref: ReferenceParticle) -> None:
    # The φs=π / η>0 map must also confine an inside particle and let an outside one
    # run away, over 1e4 turns. The ring is dispersive, so the reduced separatrix is
    # approximate at the sub-percent coupling level -- use generous margins.
    lat = _above_transition_lattice(ref)
    dmax = rf_bucket_height(lat)
    zeta_u = math.pi / lat.elements[-1].k_rf(ref)

    inside = Tracker(lat).track_turns(Particle(zeta=0.0, delta=0.7 * dmax), 10_000, nonlinear=True)
    assert np.max(np.abs(inside[:, ZETA])) < zeta_u  # librates, bounded in phase

    outside = Tracker(lat).track_turns(Particle(zeta=0.0, delta=1.3 * dmax), 10_000, nonlinear=True)
    assert np.max(np.abs(outside[:, ZETA])) > 20.0 * zeta_u  # rotates, runs away


@pytest.mark.slow
def test_inside_separatrix_bounded_over_1e4_turns(ref: ReferenceParticle) -> None:
    lat = _bucket_lattice(ref)
    dmax = rf_bucket_height(lat)
    k_rf = lat.elements[-1].k_rf(ref)
    zeta_u = math.pi / k_rf  # bucket half-width in zeta
    ham = longitudinal_hamiltonian(lat)

    # Launched inside (90% of the bucket height at the centre): librates, bounded.
    p = Particle(zeta=0.0, delta=0.9 * dmax)
    traj = Tracker(lat).track_turns(p, 10_000, nonlinear=True)
    zeta, delta = traj[:, ZETA], traj[:, DELTA]
    assert np.max(np.abs(zeta)) < zeta_u  # stays within the bucket in phase
    assert np.max(np.abs(delta)) < dmax  # never exceeds the separatrix height

    # The Hamiltonian is conserved along the bounded orbit (symplectic map): a
    # bounded kick-drift ripple (small relative to the bucket depth), with NO
    # secular drift over 1e4 turns. Normalise by the bucket depth
    # |H(zeta_u, 0) - H(0, 0)| = the well from centre to separatrix.
    h_vals = np.array([ham(z, d) for z, d in zip(zeta, delta, strict=True)])
    depth = abs(ham(zeta_u, 0.0) - ham(0.0, 0.0))
    ripple = np.max(np.abs(h_vals - h_vals[0])) / depth
    assert ripple < 0.05  # bounded band (near-separatrix first-order integrator)
    # No secular drift: the ripple in the second half is no larger than the first.
    half = len(h_vals) // 2
    first = np.ptp(h_vals[:half])
    second = np.ptp(h_vals[half:])
    assert second <= 1.5 * first


@pytest.mark.slow
def test_outside_separatrix_runs_away_over_1e4_turns(ref: ReferenceParticle) -> None:
    lat = _bucket_lattice(ref)
    dmax = rf_bucket_height(lat)
    k_rf = lat.elements[-1].k_rf(ref)
    zeta_u = math.pi / k_rf

    # Launched just outside: rotation, not libration -- zeta slips without bound.
    p = Particle(zeta=0.0, delta=1.1 * dmax)
    traj = Tracker(lat).track_turns(p, 10_000, nonlinear=True)
    zeta = traj[:, ZETA]
    assert np.max(np.abs(zeta)) > 20.0 * zeta_u  # far beyond one bucket => unbounded
