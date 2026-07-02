"""Stage 6 low-β insertion: the interaction-point waist evolution.

The collider "low-β insertion" squeezes β to a minimum β* at the IP so the beam
size σ = √(εβ*) — and hence the luminosity — is maximal. That optics needs **no
new code**: a waist is a zero-α point, and the Stage-1 drift Twiss propagation
(``B → C B Cᵀ``) already gives the textbook hyperbolic growth away from it,

    β(s) = β* + s²/β*,     α(s) = -s/β*,

for a drift starting at the waist. This test turns that claim into a validated
deliverable using only the Stage-1-proven machinery.
"""

from __future__ import annotations

import math

from accsim import Drift, Lattice, ReferenceParticle, Twiss, propagate_twiss

REF = ReferenceParticle.from_total_energy(938.272e6, 7.0e12)


def test_waist_beta_and_alpha_evolution() -> None:
    """β(s)=β*+s²/β*, α(s)=-s/β* downstream of a zero-α waist (both planes)."""
    beta_star = 0.55  # m (LHC-like)
    waist = Twiss(0.0, beta_star, 0.0, 0.0, beta_star, 0.0, 0.0)
    for s_len in (0.1, 0.55, 1.3, 3.0):
        end = propagate_twiss(Lattice([Drift(s_len)], ref=REF), waist)[-1]
        assert math.isclose(end.beta_x, beta_star + s_len**2 / beta_star, rel_tol=1e-12)
        assert math.isclose(end.beta_y, beta_star + s_len**2 / beta_star, rel_tol=1e-12)
        assert math.isclose(end.alpha_x, -s_len / beta_star, rel_tol=1e-12)
        assert math.isclose(end.alpha_y, -s_len / beta_star, rel_tol=1e-12)


def test_waist_is_the_beta_minimum_and_symmetric() -> None:
    """β grows symmetrically either side of the waist (β(-s) = β(+s))."""
    beta_star = 0.55
    s_len = 2.0
    # Start upstream (alpha = +s/beta*, converging) and drift *to* the waist and beyond.
    upstream = Twiss(
        0.0,
        beta_star + s_len**2 / beta_star,
        s_len / beta_star,
        0.0,
        beta_star + s_len**2 / beta_star,
        s_len / beta_star,
        0.0,
    )
    pts = propagate_twiss(Lattice([Drift(s_len), Drift(s_len)], ref=REF), upstream)
    beta_start, beta_mid, beta_end = pts[0].beta_x, pts[1].beta_x, pts[2].beta_x
    assert math.isclose(beta_mid, beta_star, rel_tol=1e-12)  # waist reached
    assert math.isclose(pts[1].alpha_x, 0.0, abs_tol=1e-12)  # zero-alpha at the waist
    assert math.isclose(beta_start, beta_end, rel_tol=1e-12)  # symmetric about the waist
    assert beta_mid < beta_start  # the waist is the minimum
