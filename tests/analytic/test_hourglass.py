"""C2 acceptance: the hourglass luminosity reduction ``H(sigma_z / beta*)``.

The gate is layered so that a wrong integrand and a wrong closed form cannot
cancel:

1. **The integrand is *derived*, not asserted (sympy).** Doing the ``x``, ``y``
   and ``t`` Gaussian integrals of ``rho1 * rho2`` makes both load-bearing pieces
   fall out on their own: the longitudinal weight ``exp(-s^2/sigma_z^2)`` (the
   collision points have rms ``sigma_z/sqrt(2)``, *not* ``sigma_z`` — the classic
   trap) and the ``1/sqrt((1+s^2/beta_x*^2)(1+s^2/beta_y*^2))`` waist factor. The
   same derivation reproduces the ``1/(4 pi sigma_x sigma_y)`` overlap that
   Stage 6's ``luminosity`` is built on, tying the new code to the old.
2. **Closed form vs quadrature of that derived integrand** — this only checks the
   ``erfc`` algebra, which is why (1) has to come first.
3. **An independent 2D numeric overlap** integrated over ``(s, t)`` directly,
   never using the ``sigma_z/sqrt(2)`` collapse, so a wrong collision-point width
   would *not* cancel between the two sides.
4. Limits, monotonicity, the unequal-``beta*`` bracket, and an LHC worked example.
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp
from scipy import integrate

from accsim.collider import hourglass_reduction


def test_overlap_integrand_is_derived_from_rho1_rho2() -> None:
    """The exp(-s^2/sigma_z^2) weight and the waist factor fall out of the overlap."""
    x, y, s, t = sp.symbols("x y s t", real=True)
    c = sp.Symbol("c", positive=True)
    sz, sx0, sy0, bx, by = sp.symbols("sigma_z sigma_x0 sigma_y0 beta_x beta_y", positive=True)

    def overlap(sig_x: sp.Expr, sig_y: sp.Expr) -> sp.Expr:
        """int rho1 rho2 dx dy dt for counter-propagating Gaussian bunches."""

        def rho(sign: int) -> sp.Expr:
            norm = 1 / ((2 * sp.pi) ** sp.Rational(3, 2) * sig_x * sig_y * sz)
            return norm * sp.exp(
                -(x**2) / (2 * sig_x**2)
                - y**2 / (2 * sig_y**2)
                - (s + sign * c * t) ** 2 / (2 * sz**2)
            )

        out = sp.integrate(rho(-1) * rho(+1), (x, -sp.oo, sp.oo))
        out = sp.integrate(out, (y, -sp.oo, sp.oo))
        return sp.simplify(sp.integrate(out, (t, -sp.oo, sp.oo)))

    waist_x = sx0 * sp.sqrt(1 + s**2 / bx**2)
    waist_y = sy0 * sp.sqrt(1 + s**2 / by**2)
    with_hourglass = overlap(waist_x, waist_y)
    constant_sigma = overlap(sx0, sy0)

    # (a) The longitudinal weight is exp(-s^2/sigma_z^2): collision points have rms
    #     sigma_z/sqrt(2), because *both* bunches have to be there.
    expected_flat = sp.exp(-(s**2) / sz**2) / (8 * sp.pi ** sp.Rational(3, 2) * c * sx0 * sy0 * sz)
    assert sp.simplify(constant_sigma - expected_flat) == 0

    # (b) The waist factor is the ratio, with no leftover sigma_z dependence.
    ratio = sp.simplify(with_hourglass / constant_sigma)
    assert sp.simplify(ratio - 1 / (waist_x * waist_y / (sx0 * sy0))) == 0

    # (c) Integrating the constant-sigma case over s (and folding in the 2c flux
    #     factor) gives back Stage 6's 1/(4 pi sigma_x sigma_y) -- same physics,
    #     same coefficient, so the hourglass rides on validated ground.
    flat_total = 2 * c * sp.integrate(constant_sigma, (s, -sp.oo, sp.oo))
    assert sp.simplify(flat_total - 1 / (4 * sp.pi * sx0 * sy0)) == 0

    # (d) The reduction factor is therefore the exp(-s^2/sigma_z^2)-weighted mean of
    #     the waist factor -- which is what hourglass_reduction computes. Check it
    #     numerically against the symbolic weight for an unequal-beta* case.
    sz_v, bx_v, by_v = 0.06, 0.11, 0.30
    weight = sp.exp(-(s**2) / sz**2) / (sz * sp.sqrt(sp.pi))
    expr = (weight * ratio).subs({sz: sz_v, bx: bx_v, by: by_v})
    expected = float(sp.N(sp.Integral(expr, (s, -sp.oo, sp.oo)).evalf(20)))
    assert math.isclose(hourglass_reduction(sz_v, bx_v, by_v), expected, rel_tol=1e-10)


def test_round_closed_form_matches_quadrature() -> None:
    """sqrt(pi) a erfcx(a) == the derived integral, over five decades of a."""
    for sigma_z, beta in [(1e-3, 0.5), (0.05, 0.2), (0.1, 0.1), (0.5, 0.05), (2.0, 0.01)]:
        a = beta / sigma_z
        quad, err = integrate.quad(
            lambda u, a=a: math.exp(-(u**2)) / (1 + (u / a) ** 2) / math.sqrt(math.pi),
            -np.inf,
            np.inf,
        )
        assert err < 1e-8  # quad's own estimate; the a = 500 case is sharply peaked
        assert math.isclose(hourglass_reduction(sigma_z, beta), quad, rel_tol=1e-9)


def test_independent_two_dimensional_overlap_closure() -> None:
    """Integrate over (s, t) directly -- never using the sigma_z/sqrt(2) collapse."""
    sigma_z, beta = 0.08, 0.10  # a = 1.25: deep in the hourglass regime, H ~ 0.7

    def integrand(t: float, s: float, waist: bool) -> float:
        # c = 1; the transverse overlap at fixed s is 1/(4 pi sigma_x(s) sigma_y(s)),
        # and sigma_x(s) sigma_y(s) scales as (1 + s^2/beta^2) for a round waist.
        shape = (1.0 + (s / beta) ** 2) if waist else 1.0
        arg = ((s - t) ** 2 + (s + t) ** 2) / (2.0 * sigma_z**2)
        return math.exp(-arg) / shape

    span = 12.0 * sigma_z
    num, _ = integrate.dblquad(integrand, -span, span, -span, span, args=(True,))
    den, _ = integrate.dblquad(integrand, -span, span, -span, span, args=(False,))
    assert math.isclose(num / den, hourglass_reduction(sigma_z, beta), rel_tol=1e-8)


def test_limits_and_monotonicity() -> None:
    """H -> 1 for a point bunch, H -> 0 for a long one, monotone in between."""
    beta = 0.5
    assert hourglass_reduction(0.0, beta) == 1.0
    # Short bunch: H -> 1 with the leading 1 - sigma_z^2/(2 beta*^2) correction.
    for ratio in (1e-2, 1e-3):
        sigma_z = ratio * beta
        approx = 1.0 - 0.5 * ratio**2
        assert math.isclose(hourglass_reduction(sigma_z, beta), approx, rel_tol=5 * ratio**4)
    # Long bunch: H -> sqrt(pi) beta*/sigma_z -> 0, and never overflows.
    assert math.isclose(
        hourglass_reduction(1e6, beta), math.sqrt(math.pi) * beta / 1e6, rel_tol=1e-5
    )
    assert 0.0 < hourglass_reduction(1e-30, beta) <= 1.0  # a ~ 1e29: erfcx, not exp(a^2)

    values = [hourglass_reduction(sz, beta) for sz in np.geomspace(1e-3, 1e3, 60)]
    assert all(hi < lo for lo, hi in zip(values, values[1:], strict=False))
    assert all(0.0 < v <= 1.0 for v in values)


def test_unequal_beta_star_is_bracketed_by_the_round_cases() -> None:
    """H(bx, by) sits between H(round bx) and H(round by), and reduces when equal."""
    sigma_z, bx, by = 0.07, 0.15, 0.60
    mixed = hourglass_reduction(sigma_z, bx, by)
    assert hourglass_reduction(sigma_z, bx) < mixed < hourglass_reduction(sigma_z, by)
    # Explicit beta_y_star == beta_x_star must reproduce the closed form exactly.
    assert math.isclose(
        hourglass_reduction(sigma_z, bx, bx), hourglass_reduction(sigma_z, bx), rel_tol=1e-11
    )
    # Symmetric in the two planes.
    assert math.isclose(mixed, hourglass_reduction(sigma_z, by, bx), rel_tol=1e-12)


def test_lhc_nominal_hourglass_is_a_one_percent_effect() -> None:
    """LHC nominal (beta* = 0.55 m, sigma_z = 7.55 cm) loses ~1% to the hourglass."""
    h = hourglass_reduction(0.0755, 0.55)
    assert math.isclose(h, 0.9907, rel_tol=1e-3)

    # Squeezing beta* to 0.15 m (HL-LHC-like) at the same bunch length turns that
    # sub-percent loss into ~10%: the reason a beta* squeeze alone does not buy the
    # luminosity it appears to. The *scaling* is the check, not a remembered number
    # -- the loss grows as beta*^-2 (from 1 - sigma_z^2/2beta*^2), but sub-linearly
    # in that estimate once sigma_z ~ beta* and the expansion stops being small.
    loss_ratio = (1 - hourglass_reduction(0.0755, 0.15)) / (1 - h)
    assert 1.0 < loss_ratio < (0.55 / 0.15) ** 2
