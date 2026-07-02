"""Analytic checks for the small-amplitude synchrotron tune ``Qs`` (Stage 3).

Longitudinal motion is a rotation in ``(zeta, delta)`` from the arc slip
``Delta zeta = -eta C delta`` and the RF focusing ``Delta delta = R65 zeta``.
The reduced one-turn 2x2 gives ``cos(2 pi Qs) = 1 - R65_tot eta C / 2``, whose
small-amplitude limit is the closed form

    Qs^2 = -(h eta q V cos phi_s) / (2 pi beta0^2 E0),     k_rf C = 2 pi h.

The independent nets here:

1. a **sympy re-derivation** of that closed form straight from the reduced
   matrix trace (proves the coefficient *and* sign, incl. the ``beta0^2 E0``
   energy factor and the ``k_rf C = 2 pi h`` identity), then accsim's
   :func:`synchrotron_tune` matched to it in the ``V -> 0`` limit where the exact
   ``arccos`` and the small-amplitude form agree to ``O((2 pi Qs)^2)``;
2. a **flag-A discriminator**: the test ring is dispersive, so its bare one-turn
   ``R56`` entry is nowhere near ``-eta C`` (here it is even the *wrong sign*).
   ``Qs`` is asserted to follow the ``eta``-based value, so an implementation that
   reduced the raw ``(zeta, delta)`` block instead of consuming ``eta`` fails;
3. **transition / stability**: below transition the stable phase is ``phi_s = 0``,
   above it ``phi_s = pi``; the wrong side has no bucket and raises.

Tested at ``gamma0 = 5`` (``beta0 ~ 0.98``) so a dropped ``beta0^2`` would be a
~4% gap, not rounding.
"""

from __future__ import annotations

import math

import pytest

from accsim import (
    DELTA,
    ZETA,
    Dipole,
    Drift,
    Lattice,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    UnstableLatticeError,
    slip_factor,
    synchrotron_tune,
)
from accsim.reference import CLIGHT

MASS0 = 938.27208816e6
GAMMA0 = 5.0
F_FOCAL = 2.5
L_BEND = 1.0
ANGLE = 0.15
HARMONIC = 8


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _arc(ref: ReferenceParticle) -> list:
    """Dispersive arc FODO (thin quads + thick bends). eta > 0 => above transition."""
    return [
        ThinQuadrupole(0.5 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(-1.0 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]


def _rf_frequency(ref: ReferenceParticle, circumference: float) -> float:
    """Frequency of harmonic ``HARMONIC``: f = h * beta0 c / C (so k_rf C = 2 pi h)."""
    return HARMONIC * ref.beta0 * CLIGHT / circumference


def _closed_form_qs(ref: ReferenceParticle, eta: float, voltage: float, phi_s: float) -> float:
    """Small-amplitude Qs^2 = -(h eta q V cos phi_s)/(2 pi beta0^2 E0), then sqrt."""
    qs2 = -(HARMONIC * eta * ref.charge * voltage * math.cos(phi_s)) / (
        2.0 * math.pi * ref.beta0**2 * ref.total_energy_eV
    )
    return math.sqrt(qs2)


def test_arc_is_above_transition(ref: ReferenceParticle) -> None:
    # Sanity: this arc has eta > 0, so phi_s = pi is the stable phase below.
    assert slip_factor(Lattice(_arc(ref), ref)) > 0.0


def test_matches_symbolic_closed_form(ref: ReferenceParticle) -> None:
    sp = pytest.importorskip("sympy")

    # --- symbolic derivation of the closed form from the reduced 2x2 trace ------
    q, V, k, b0, E0, phi, eta_s, C, h, zeta = sp.symbols("q V k b0 E0 phi eta C h zeta", real=True)
    # R65 = d/dzeta [ (qV/(b0^2 E0))(sin(phi - k zeta) - sin phi) ] |_0
    r65 = sp.diff((q * V / (b0**2 * E0)) * (sp.sin(phi - k * zeta) - sp.sin(phi)), zeta).subs(
        zeta, 0
    )
    m_arc = sp.Matrix([[1, -eta_s * C], [0, 1]])
    m_cav = sp.Matrix([[1, 0], [r65, 1]])
    trace = sp.trace(m_cav * m_arc)
    # cos(2 pi Qs) = trace/2  =>  (2 pi Qs)^2 = 2 - trace  to leading order.
    qs2 = sp.simplify((2 - trace) / (4 * sp.pi**2)).subs(k, 2 * sp.pi * h / C)
    target = -(h * eta_s * q * V * sp.cos(phi)) / (2 * sp.pi * b0**2 * E0)
    assert sp.simplify(qs2 - target) == 0

    # --- accsim matches it in the small-amplitude (V -> 0) limit -----------------
    arc = _arc(ref)
    circumference = sum(e.length for e in arc)
    eta = slip_factor(Lattice(arc, ref))
    freq = _rf_frequency(ref, circumference)
    voltage, phi_s = 1.0e2, math.pi  # tiny V => Qs ~ 1e-4, arccos ~ small-amp to O(Qs^2)
    lat = Lattice([*arc, RFCavity(voltage, freq, phi_s)], ref)
    got = synchrotron_tune(lat)
    expected = _closed_form_qs(ref, eta, voltage, phi_s)
    assert got == pytest.approx(expected, rel=1e-6)


def test_small_amplitude_is_the_v_to_zero_limit(ref: ReferenceParticle) -> None:
    # The exact arccos exceeds the small-amplitude form by O((2 pi Qs)^2); dropping
    # V by 100x (Qs by 10x) must shrink that relative gap by ~100x.
    arc = _arc(ref)
    circumference = sum(e.length for e in arc)
    eta = slip_factor(Lattice(arc, ref))
    freq = _rf_frequency(ref, circumference)

    def rel_gap(voltage: float) -> float:
        lat = Lattice([*arc, RFCavity(voltage, freq, math.pi)], ref)
        got = synchrotron_tune(lat)
        exp = _closed_form_qs(ref, eta, voltage, math.pi)
        return abs(got - exp) / exp

    gap_hi = rel_gap(1.0e6)
    gap_lo = rel_gap(1.0e4)
    assert gap_lo < gap_hi
    assert gap_hi / gap_lo == pytest.approx(100.0, rel=0.1)


def test_uses_slip_factor_not_bare_r56(ref: ReferenceParticle) -> None:
    # Flag A: on a dispersive ring the bare one-turn R56 is nowhere near -eta C, so
    # a Qs built from the raw (zeta, delta) block would be wrong. Prove the ring is
    # discriminating, then that Qs follows the eta-based value.
    arc = _arc(ref)
    circumference = sum(e.length for e in arc)
    eta = slip_factor(Lattice(arc, ref))
    freq = _rf_frequency(ref, circumference)
    voltage, phi_s = 1.0e5, math.pi
    lat = Lattice([*arc, RFCavity(voltage, freq, phi_s)], ref)

    bare_r56 = lat.one_turn_matrix()[ZETA, DELTA]
    # The dispersion coupling dominates: the true arc off-diagonal -eta C is even the
    # OPPOSITE SIGN of the bare R56 entry, so the two are qualitatively different.
    assert (-eta * circumference) * bare_r56 < 0.0

    # accsim uses the eta arc drift [[1, -eta C], [0, 1]] and gets a stable bucket;
    # a bare-R56 implementation would use [[1, R56], [0, 1]] and, with the flipped
    # sign, is UNSTABLE here (no real Qs at all).
    r65 = lat.elements[-1].slope(ref)
    half_trace_eta = 1.0 - 0.5 * r65 * eta * circumference
    half_trace_bare = 1.0 + 0.5 * r65 * bare_r56
    assert abs(half_trace_eta) < 1.0  # accsim: stable
    assert abs(half_trace_bare) > 1.0  # bare-R56 surrogate: unstable
    assert synchrotron_tune(lat) == pytest.approx(
        math.acos(half_trace_eta) / (2 * math.pi), rel=1e-12
    )


def test_stable_phase_flips_across_transition() -> None:
    # Below transition (eta < 0): phi_s = 0 stable, phi_s = pi has no bucket.
    below = ReferenceParticle.from_gamma(MASS0, 1.05)  # gamma0 ~ 1 => 1/gamma0^2 large => eta < 0
    arc = _arc(below)
    circumference = sum(e.length for e in arc)
    assert slip_factor(Lattice(arc, below)) < 0.0
    freq = _rf_frequency(below, circumference)
    Qs = synchrotron_tune(Lattice([*arc, RFCavity(1.0e5, freq, 0.0)], below))
    assert Qs > 0.0
    with pytest.raises(UnstableLatticeError):
        synchrotron_tune(Lattice([*arc, RFCavity(1.0e5, freq, math.pi)], below))


def test_above_transition_needs_phi_s_pi(ref: ReferenceParticle) -> None:
    arc = _arc(ref)  # gamma0 = 5 => eta > 0 => above transition
    circumference = sum(e.length for e in arc)
    freq = _rf_frequency(ref, circumference)
    assert synchrotron_tune(Lattice([*arc, RFCavity(1.0e5, freq, math.pi)], ref)) > 0.0
    with pytest.raises(UnstableLatticeError):
        synchrotron_tune(Lattice([*arc, RFCavity(1.0e5, freq, 0.0)], ref))


def test_no_cavity_raises(ref: ReferenceParticle) -> None:
    with pytest.raises(ValueError, match="requires at least one RFCavity"):
        synchrotron_tune(Lattice([Drift(1.0), ThinQuadrupole(0.3)], ref))


def test_qs_scales_as_sqrt_voltage(ref: ReferenceParticle) -> None:
    # Qs ∝ sqrt(V) in the small-amplitude regime: 4x voltage -> 2x tune.
    arc = _arc(ref)
    circumference = sum(e.length for e in arc)
    freq = _rf_frequency(ref, circumference)
    q1 = synchrotron_tune(Lattice([*arc, RFCavity(1.0e4, freq, math.pi)], ref))
    q4 = synchrotron_tune(Lattice([*arc, RFCavity(4.0e4, freq, math.pi)], ref))
    assert q4 / q1 == pytest.approx(2.0, rel=1e-3)
