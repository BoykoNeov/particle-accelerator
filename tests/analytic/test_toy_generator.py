r"""Phase 2 acceptance — the toy ``e+ e- -> mu+ mu-`` Monte-Carlo generator.

Three analytic gates, ordered so a sampler bug and a matrix-element bug cannot
cancel into a right-looking cross-section (the advisor's warning):

  1. **Phase-space volume** — RAMBO's flat 2-body LIPS volume equals the
     symbolically-derived ``1/(8 pi)`` (and the general formula matches the
     independently-derived 3-body ``s/(256 pi^3)``); the sampler conserves
     four-momentum, stays massless, and fills phase space *isotropically*.
  2. **dsigma/dOmega shape** — the tree-level ``<|M|^2>`` reproduces the
     ``1 + cos^2 theta`` law, and its solid-angle integral is ``4 pi alpha^2/(3 s)``.
  3. **Total cross-section** — the MC estimate matches the analytic value within
     its own Monte-Carlo error (the roadmap's Phase 2 acceptance clause).

Coefficients are derived with sympy, not remembered (project working agreement).
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp

from accsim.events import (
    ALPHA_EM,
    EEtoMuMu,
    ee_to_mumu_cross_section,
    ee_to_mumu_events,
    gev2_to_barn,
    invariant_mass_squared,
    massless_phase_space_volume,
    minkowski_dot,
    rambo,
)

# ---------------------------------------------------------------------------
# Gate 1 — flat two-body phase space (RAMBO), validated before any |M|^2.
# ---------------------------------------------------------------------------


def test_massless_two_body_volume_is_one_over_eight_pi_symbolic() -> None:
    """``Phi_2 = 1/(8 pi)`` — derived from the beta factor, not remembered."""
    s, m = sp.symbols("s m", positive=True)
    # Two-body LIPS closed form Phi_2 = (1/8pi) * 2|p*|/sqrt(s), |p*|=sqrt(s/4-m^2).
    p_star = sp.sqrt(s / 4 - m**2)
    phi2 = (2 * p_star / sp.sqrt(s)) / (8 * sp.pi)
    phi2_massless = sp.simplify(phi2.subs(m, 0))
    assert phi2_massless == 1 / (8 * sp.pi)
    for sqrt_s in (1.0, 10.0, 91.19):
        assert math.isclose(
            massless_phase_space_volume(2, sqrt_s), 1.0 / (8.0 * math.pi), rel_tol=1e-12
        )


def test_general_volume_formula_matches_independent_three_body() -> None:
    """RAMBO's n-body volume formula reproduces the recursively-derived ``Phi_3``.

    Independent derivation via the phase-space convolution
    ``Phi_3(s) = int_0^s (dmu2/2pi) Phi_2(s;0,mu2) Phi_2(mu2;0,0)`` with massless
    two-body factors — yields ``s/(256 pi^3)``.
    """
    s, mu2 = sp.symbols("s mu2", positive=True)
    phi2_split = (s - mu2) / s / (8 * sp.pi)  # Phi_2(s; 0, mu2), massless-limit lambda^1/2=s-mu2
    phi2_final = 1 / (8 * sp.pi)  # Phi_2(mu2; 0, 0)
    phi3 = sp.integrate(phi2_split * phi2_final / (2 * sp.pi), (mu2, 0, s))
    assert sp.simplify(phi3 - s / (256 * sp.pi**3)) == 0
    for sqrt_s in (2.0, 7.0):
        analytic = float((sqrt_s**2) / (256 * math.pi**3))
        assert math.isclose(massless_phase_space_volume(3, sqrt_s), analytic, rel_tol=1e-12)


def test_rambo_conserves_momentum_and_is_massless() -> None:
    """Every RAMBO event sums to ``(sqrt_s, 0, 0, 0)`` with on-shell massless legs."""
    rng = np.random.default_rng(20260709)
    sqrt_s = 10.0
    batch = rambo(3, sqrt_s, 5000, rng)
    total = batch.momenta.sum(axis=1)  # (n_events, 4)
    assert np.allclose(total[:, 0], sqrt_s, atol=1e-9)
    assert np.allclose(total[:, 1:], 0.0, atol=1e-9)
    # Masslessness: p_i.p_i ~ 0 relative to s.
    m2 = invariant_mass_squared(batch.momenta)  # (n_events, n_particles)
    assert np.max(np.abs(m2)) < 1e-6 * sqrt_s**2


def test_rambo_two_body_is_isotropic() -> None:
    """Flat 2-body phase space is isotropic: ``cos theta`` uniform on [-1, 1].

    This is the sampler's physical signature, independent of the volume constant —
    together they are "sampled volume vs 1/8pi".
    """
    rng = np.random.default_rng(11)
    batch = rambo(2, 10.0, 200_000, rng)
    p = batch.momenta[:, 0, :]
    cos_theta = p[:, 3] / np.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2 + p[:, 3] ** 2)
    # Uniform[-1,1] has mean 0, variance 1/3.
    assert abs(cos_theta.mean()) < 5e-3
    assert abs(cos_theta.var() - 1.0 / 3.0) < 5e-3
    # Flatness across bins: no bin deviates from uniform by more than ~4%.
    counts, _ = np.histogram(cos_theta, bins=10, range=(-1.0, 1.0))
    frac = counts / counts.sum()
    assert np.max(np.abs(frac - 0.1)) < 0.01


# ---------------------------------------------------------------------------
# Gate 2 — matrix element: 1 + cos^2 theta shape and its angular integral.
# ---------------------------------------------------------------------------


def _two_body_momenta(sqrt_s: float, cos_theta: float) -> tuple[np.ndarray, ...]:
    """Explicit CM momenta for e-(+z), e+(-z) -> mu-(theta), mu+(back-to-back)."""
    half = 0.5 * sqrt_s
    sin_theta = math.sqrt(1.0 - cos_theta**2)
    p1 = np.array([half, 0.0, 0.0, half])
    p2 = np.array([half, 0.0, 0.0, -half])
    p3 = np.array([half, half * sin_theta, 0.0, half * cos_theta])  # mu-
    p4 = np.array([half, -half * sin_theta, 0.0, -half * cos_theta])  # mu+
    return p1, p2, p3, p4


def test_squared_amplitude_is_one_plus_cos2_law() -> None:
    """``<|M|^2> = 16 pi^2 alpha^2 (1 + cos^2 theta)`` at explicit angles."""
    me = EEtoMuMu()
    sqrt_s = 10.0
    for cos_theta in (-0.9, -0.3, 0.0, 0.25, 0.8):
        p1, p2, p3, p4 = _two_body_momenta(sqrt_s, cos_theta)
        got = float(me.squared_amplitude(p1, p2, p3, p4))
        expected = 16.0 * math.pi**2 * ALPHA_EM**2 * (1.0 + cos_theta**2)
        assert math.isclose(got, expected, rel_tol=1e-12)


def test_dsigma_domega_integrates_to_total_symbolic() -> None:
    """``int (dsigma/dOmega) dOmega = 4 pi alpha^2/(3 s)`` — sympy, not remembered."""
    alpha, s, theta, phi = sp.symbols("alpha s theta phi", positive=True)
    dsigma = alpha**2 * (1 + sp.cos(theta) ** 2) / (4 * s)
    total = sp.integrate(
        sp.integrate(dsigma * sp.sin(theta), (theta, 0, sp.pi)), (phi, 0, 2 * sp.pi)
    )
    assert sp.simplify(total - 4 * sp.pi * alpha**2 / (3 * s)) == 0
    # The Python closed form agrees numerically.
    me = EEtoMuMu()
    s_num = 100.0
    assert math.isclose(
        me.total_cross_section(s_num), 4 * math.pi * ALPHA_EM**2 / (3 * s_num), rel_tol=1e-12
    )


def test_incoming_flux_gives_s() -> None:
    """Mandelstam ``s = (p1+p2)^2 = sqrt_s^2`` for the collinear massless beams."""
    p1, p2, _, _ = _two_body_momenta(10.0, 0.0)
    assert math.isclose(float(invariant_mass_squared(p1 + p2)), 100.0, rel_tol=1e-12)
    # Beams are massless and back-to-back.
    assert math.isclose(float(minkowski_dot(p1, p1)), 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Gate 3 — total cross-section: MC matches analytic within Monte-Carlo error.
# ---------------------------------------------------------------------------


def test_mc_cross_section_matches_analytic_within_error() -> None:
    """The Phase 2 acceptance clause: toy MC sigma == analytic within MC error."""
    rng = np.random.default_rng(2026)
    sqrt_s = 10.0
    est = ee_to_mumu_cross_section(sqrt_s, 200_000, rng)
    analytic = EEtoMuMu().total_cross_section(sqrt_s**2)
    # Within 4 sigma of the analytic value, and the error itself is sub-percent.
    assert abs(est.value - analytic) < 4.0 * est.error
    assert est.error / est.value < 0.01
    # The famous numeric anchor: ~0.87 nb at 10 GeV.
    assert math.isclose(est.value_nb, 0.87, abs_tol=0.03)
    assert math.isclose(gev2_to_barn(analytic) / 1e-9, 0.87, abs_tol=0.02)


def test_cross_section_scales_as_one_over_s() -> None:
    """``sigma ~ 1/s``: doubling sqrt_s quarters the cross-section (analytic + MC)."""
    me = EEtoMuMu()
    assert math.isclose(
        me.total_cross_section(10.0**2) / me.total_cross_section(20.0**2), 4.0, rel_tol=1e-12
    )
    rng = np.random.default_rng(7)
    lo = ee_to_mumu_cross_section(10.0, 100_000, rng)
    hi = ee_to_mumu_cross_section(20.0, 100_000, rng)
    assert math.isclose(lo.value / hi.value, 4.0, rel_tol=0.02)


def test_generator_determinism_and_labelled_distribution() -> None:
    """Same seed -> same events; the angular histogram favours |cos theta| -> 1."""
    cos_a, dist = ee_to_mumu_events(10.0, 50_000, np.random.default_rng(3))
    cos_b, _ = ee_to_mumu_events(10.0, 50_000, np.random.default_rng(3))
    assert np.array_equal(cos_a, cos_b)
    assert dist.counts.sum() == cos_a.size
    assert len(dist.bin_edges) == len(dist.counts) + 1
    assert "mu" in dist.label
    # 1 + cos^2 theta: outer bins (|cos| near 1) exceed central bins.
    outer = dist.counts[0] + dist.counts[-1]
    n = len(dist.counts)
    central = dist.counts[n // 2 - 1] + dist.counts[n // 2]
    assert outer > central
