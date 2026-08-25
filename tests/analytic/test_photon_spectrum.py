r"""Analytic gates — the synchrotron photon spectrum and its sampler (B5, part 1).

B3 shipped the graininess of the radiation loss as a Gaussian of the right mean and the
right variance, and said plainly what that leaves out: the tail. This file gates the
thing the Gaussian stands in for — the distribution a *single* photon's energy is drawn
from — with no lattice, no tracking and no reference code involved. The kick that uses
it is the next file; the sampler stands alone.

The gates, ordered so that a sampler which is wrong in the tail cannot hide behind
moments that are right:

  1. **Every constant is derived, none quoted.** sympy integrates ``K_{5/3}`` outright:
     ``int_0^inf K_{5/3}(t) t dt = 5 pi / 3`` is what normalises the spectrum, and the
     same integral at three more powers gives ``<x> = 8/(15 sqrt3)``, ``<x^2> = 11/27``
     and ``<x^3> = 224/(135 sqrt3)`` as exact rationals-times-surds. B3 obtained the
     first two by quadrature; here they are closed forms, and the shipped
     :func:`photon_spectrum_moment` is checked against them.
  2. **Which spectrum, said out loud.** The textbook figure is the *power* spectrum
     ``F(x) = x int_x^inf K_{5/3}``; photons are counted off the *number* spectrum
     ``F(x)/x``. Sampling the wrong one makes every photon exactly ``<x^2>/<x>^2 =
     4.297`` times too energetic -- the same constant B3 already uses in the other
     direction to count photons -- so it is asserted as that identity, not as a comment.
  3. **The two quadrature routes are independent and must sum to one.** The cumulative
     distribution is computed from ``int_0^x K t dt + x int_x^inf K dt`` and the
     exceedance from the single collapsed ``int_x^inf K(t)(t-x) dt``; neither is derived
     from the other, and ``F + Q = 1`` to round-off is what says both are right.
  4. **The sampler is gated deterministically, not statistically.** It is an inverse
     transform, so one uniform in is one photon energy out with a closed-form answer:
     feed a chosen quantile and the quadrature must return it. That works *in the tail*,
     at exceedances of ``1e-16``, where no amount of sampling would ever show a mistake.
  5. **The two limiting laws.** ``F ~ 1.2316 x^(1/3)`` at the bottom (the coefficient
     derived symbolically from the leading behaviour of ``K_{5/3}``) and
     ``Q ~ (3/5) e^-x / sqrt(2 pi x)`` at the top, both with their approach measured.
  6. **The headline: the tail is exponential, and a Gaussian's is not.** ``-log Q`` is
     asymptotically *linear* in ``x`` with slope ``1 + 1/2x``, measured across a decade.
     A Gaussian carrying this distribution's own mean and variance is 61 orders of
     magnitude low at ``x = 10`` and 261 at ``x = 20``. That gap is the entire physics
     content of the milestone, and it is why B3's model was named a stand-in.
  7. **Then, and only then, statistics.** The sampled moments reproduce the closed forms
     within a pre-stated sampling budget, and the compound-Poisson sum has the mean, the
     variance and the *skewness* the spectrum's three moments predict — the skewness
     being the one that inverts to a photon count, and the one B3's Gaussian sets to
     zero by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim.photon_spectrum import (
    SMALL_X_COEFFICIENT,
    SPECTRUM_NORMALISATION,
    SPECTRUM_ORDER,
    photon_energy_quantile,
    photon_log_survival,
    photon_number_cdf,
    photon_number_pdf,
    photon_spectrum_moment,
    sample_photon_sum,
)

SQRT3 = math.sqrt(3.0)


# ---------------------------------------------------------------------------
# Gate 1 — every constant derived, none quoted.
# ---------------------------------------------------------------------------
def _bessel_moment(power: int) -> sp.Expr:
    """``int_0^inf K_{5/3}(t) t^power dt``, integrated by sympy rather than looked up."""
    t = sp.symbols("t", positive=True)
    return sp.simplify(sp.integrate(sp.besselk(sp.Rational(5, 3), t) * t**power, (t, 0, sp.oo)))


def test_the_normalisation_is_the_bessel_integral_sympy_returns() -> None:
    r"""``int_0^inf K_{5/3}(t) t dt = 5 pi / 3`` exactly, so the norm is ``3 / 5 pi``.

    The photon *number* spectrum is ``p(x) = N int_x^inf K_{5/3}``; one swap of the
    order of integration turns ``int_0^inf p`` into ``N int_0^inf K_{5/3}(t) t dt``, so
    this single integral is the whole normalisation. Quoting it would leave the sampler
    free to be uniformly mis-scaled — which every *ratio* in this file would forgive.
    """
    assert _bessel_moment(1) == 5 * sp.pi / 3
    assert SPECTRUM_NORMALISATION == pytest.approx(float(3 / (5 * sp.pi)), rel=1e-15)


@pytest.mark.parametrize(
    ("order", "closed"),
    [(1, 8 / (15 * SQRT3)), (2, 11 / 27), (3, 224 / (135 * SQRT3))],
)
def test_the_moments_are_exact_and_the_shipped_helper_returns_them(
    order: int, closed: float
) -> None:
    r"""``<x^m> = N / (m+1) * int_0^inf K_{5/3}(t) t^(m+1) dt``, symbolically.

    The first two are the numbers B2's mean loss and B3's variance are built on, which
    B3 obtained by quadrature; here sympy returns them as exact surds. The third is new,
    and it is the one B5 needs: the skewness of a compound-Poisson sum is
    ``<x^3> / (sqrt(n) <x^2>^(3/2))``, which is how a photon *count* is recovered from
    the shape of a loss distribution that never counted anything.
    """
    exact = sp.Rational(3, 5) / sp.pi / (order + 1) * _bessel_moment(order + 1)
    assert sp.simplify(exact - sp.nsimplify(closed, [sp.sqrt(3)])) == 0
    assert photon_spectrum_moment(order) == pytest.approx(float(exact), rel=1e-14)


def test_the_moments_agree_with_the_distribution_the_sampler_actually_inverts() -> None:
    """The closed forms describe *this* ``pdf``, not merely some spectrum.

    :func:`photon_spectrum_moment` is Mellin algebra and never touches
    :func:`photon_number_pdf`; if the shipped pdf carried a different normalisation the
    two would part company here and nowhere else in this file.
    """
    from scipy.integrate import quad

    for order in (0, 1, 2):
        integrated = quad(lambda x, m=order: x**m * photon_number_pdf(x), 0.0, np.inf, limit=200)[0]
        expected = 1.0 if order == 0 else photon_spectrum_moment(order)
        assert integrated == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# Gate 2 — number spectrum or power spectrum: a factor of 4.3, said out loud.
# ---------------------------------------------------------------------------
def test_sampling_the_power_spectrum_instead_would_be_wrong_by_this_much() -> None:
    r"""``<x>_power / <x>_number = <x^2>/<x>^2 = 4.297``: photons 4.3x too energetic.

    The distribution plotted in every textbook is ``F(x) = x int_x^inf K_{5/3}`` — the
    *power* radiated per unit ``x``. Photons are counted off ``F(x)/x``. Normalising
    ``F`` as a density and drawing energies from it is the natural mistake, and nothing
    dimensional would catch it: it is a pure number, and the number is **4.297**, not
    the 1.3 that the ratio of the two means looks like at a glance.

    That constant is not a new one. ``<x^2>/<x>^2`` is exactly the factor B3's suite
    already uses in the other direction, to recover a photon *count* from a relative
    fluctuation it never counted — so this gate and B3's ``4.297`` are one number, and
    it is asserted here as that identity rather than as a decimal.
    """
    first, second = photon_spectrum_moment(1), photon_spectrum_moment(2)
    power_mean = second / first  # <x^2>/<x>: the mean of the x-weighted distribution
    assert power_mean / first == pytest.approx(second / first**2, rel=1e-14)
    assert second / first**2 == pytest.approx(4.297, rel=1e-3)  # B3's own number
    assert power_mean == pytest.approx(55 / (24 * SQRT3), rel=1e-13)
    # and the shipped sampler is the number one: its mean is 8/(15 sqrt3)
    assert first == pytest.approx(8 / (15 * SQRT3), rel=1e-14)


# ---------------------------------------------------------------------------
# Gate 3 — two independent quadratures, and the one thing they must add up to.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x", [0.01, 0.05, 0.0783781, 0.3, 1.0, 3.0, 8.0, 20.0, 30.0])
def test_the_cumulative_and_the_exceedance_add_to_one(x: float) -> None:
    """``F(x) + P(x' > x) = 1`` to round-off, from two integrals sharing no algebra.

    The cumulative route integrates ``K t`` up to ``x`` and adds ``x`` times the tail;
    the exceedance route integrates ``K(t)(t - x)`` from ``x`` in exponentially-scaled
    form. A wrong normalisation, a wrong Bessel order or a wrong order-swap would break
    the sum; nothing that is merely *slow* would.
    """
    assert photon_number_cdf(x) + math.exp(photon_log_survival(x)) == pytest.approx(1.0, abs=1e-12)


def test_the_one_piece_quadrature_that_silently_lies_is_not_the_one_shipped() -> None:
    r"""``quad(K_{5/3}, X, inf)`` in one piece returns **nothing**, and warns instead of raising.

    Not a hypothetical: it is the bug that was walked into while building this module.
    At ``X = 1e-6`` the one-piece integral comes back at ``-1e-4`` times the true tail —
    numerically zero — with an ``IntegrationWarning`` and no exception. The damage is
    quiet rather than loud: a cumulative distribution built on it loses the ``x int_x^inf
    K`` half and so returns exactly **2/3** of the truth, because that half is exactly
    1/2 of the half that survives (the 2:1 split the small-``x`` law gates below). A
    third of the smallest photons, missing, with a plausible-looking power law intact.
    """
    from scipy.integrate import quad
    from scipy.special import kv

    x = 1.0e-6
    with np.errstate(all="ignore"):
        naive = quad(lambda t: kv(SPECTRUM_ORDER, t), x, np.inf, limit=400)[0]
    honest = photon_number_pdf(x) / SPECTRUM_NORMALISATION
    assert abs(naive / honest) < 1.0e-3  # the lie: it simply returns zero

    inner_half = quad(lambda t: kv(SPECTRUM_ORDER, t) * t, 0.0, x, limit=400)[0]
    naive_cdf = SPECTRUM_NORMALISATION * (inner_half + x * naive)
    assert naive_cdf / photon_number_cdf(x) == pytest.approx(2.0 / 3.0, rel=1e-4)
    # and the shipped cumulative is the honest one, on its own x^(1/3) law
    assert photon_number_cdf(x) == pytest.approx(SMALL_X_COEFFICIENT * x ** (1 / 3), rel=1e-4)


# ---------------------------------------------------------------------------
# Gate 4 — the sampler is an inverse, so it is gated deterministically.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", [1e-4, 1e-3, 0.01, 0.1, 0.3, 0.5])
def test_a_chosen_quantile_below_the_median_inverts_to_the_quadratures_own_answer(
    q: float,
) -> None:
    """``F(quantile(q)) == q``. One uniform in, one closed-form photon energy out.

    Inverse-transform sampling is what makes this milestone testable without statistics:
    the draw is a deterministic function of the uniform, so the shipped fast table can
    be held against the slow quadrature point by point.
    """
    assert photon_number_cdf(photon_energy_quantile(q)) == pytest.approx(q, rel=1e-8)


@pytest.mark.parametrize("exceedance", [0.4, 0.1, 1e-3, 1e-8, 1e-12, 1e-16])
def test_a_chosen_quantile_in_the_tail_inverts_there_too(exceedance: float) -> None:
    r"""``P(x' > quantile(1 - e)) == e``, down to ``e = 1e-16``.

    The point of the whole exercise. At ``e = 1e-16`` the photon energy is ``33 u_c``;
    a sampling test would need ``1e18`` draws to place one particle there, and this
    places it exactly. ``1 - q`` is exact in floating point for ``q >= 1/2``, so the
    tail costs no cancellation to look up.
    """
    x = photon_energy_quantile(1.0 - exceedance)
    assert math.exp(photon_log_survival(x)) == pytest.approx(exceedance, rel=1e-7)


def test_the_inverse_is_monotone_and_spans_the_range_the_uniforms_can_reach() -> None:
    """A quantile function must not go backwards, and must cover ``[0, 1)``.

    ``rng.random()`` returns multiples of ``2^-53``, so ``1 - q >= 1.1e-16``: the
    largest photon this sampler can ever draw is ~33 ``u_c``, and the smallest is zero.
    Both edges are asserted because both are reachable in a long tracking run.
    """
    # np.unique, because 1 - e collides onto the same double once e drops below an ulp:
    # distinct exceedances are not distinct quantiles down there, and the sampler cannot
    # be asked to separate two uniforms it will never be handed separately.
    q = np.unique(
        np.concatenate([np.geomspace(1e-17, 0.5, 4000), 1.0 - np.geomspace(1e-16, 0.5, 4000)])
    )
    x = np.asarray(photon_energy_quantile(q))
    assert np.all(np.diff(x) > 0.0)
    assert photon_energy_quantile(0.0) == 0.0
    assert 30.0 < photon_energy_quantile(1.0 - 2.0**-53) < 40.0
    for bad in (-1e-9, 1.0, 1.5):
        with pytest.raises(ValueError, match="q in"):
            photon_energy_quantile(bad)


# ---------------------------------------------------------------------------
# Gate 5 — the two limiting laws, both with derived coefficients.
# ---------------------------------------------------------------------------
def test_the_small_x_law_has_the_coefficient_the_leading_bessel_term_gives() -> None:
    r"""``F(x) -> (27 / 10 pi) 2^(2/3) Gamma(5/3) x^(1/3)``, derived not fitted.

    ``K_nu(t) -> (1/2) Gamma(nu) (2/t)^nu`` as ``t -> 0``; putting that into both halves
    of ``F = N [int_0^x K t dt + x int_x^inf K dt]`` gives ``3 x^(1/3)`` from the first
    and ``(3/2) x^(1/3)`` from the second — the two halves are *not* equal, which is
    what the ``9/2`` in the coefficient records, and dropping either would be a 33% or
    67% error in the smallest photons.
    """
    t, x = sp.symbols("t x", positive=True)
    nu = sp.Rational(5, 3)
    leading = sp.Rational(1, 2) * sp.gamma(nu) * (2 / t) ** nu
    inner = sp.integrate(leading * t, (t, 0, x))
    outer = x * sp.integrate(leading, (t, x, sp.oo))
    coefficient = sp.simplify(
        (sp.Rational(3, 5) / sp.pi * (inner + outer)) / x ** sp.Rational(1, 3)
    )
    assert sp.simplify(inner / outer) == 2  # the two halves are 2:1, not 1:1
    assert SMALL_X_COEFFICIENT == pytest.approx(float(coefficient), rel=1e-14)
    assert photon_number_cdf(1e-12) == pytest.approx(SMALL_X_COEFFICIENT * 1e-4, rel=1e-8)


@pytest.mark.parametrize("x", [10.0, 20.0, 40.0])
def test_the_tail_law_is_three_fifths_e_to_the_minus_x_over_root_two_pi_x(x: float) -> None:
    r"""``Q(x) -> (3/5) e^-x / sqrt(2 pi x)``, approached as ``1 + 0.264/x``.

    From ``K_nu(z) -> sqrt(pi/2z) e^-z`` inside the collapsed exceedance integral. The
    approach is asserted as a *law in* ``1/x`` rather than a tolerance, so a coefficient
    that was right at one energy and wrong at another could not pass.
    """
    leading = 0.6 * math.exp(-x) / math.sqrt(2.0 * math.pi * x)
    ratio = math.exp(photon_log_survival(x)) / leading
    assert ratio == pytest.approx(1.0, abs=0.5 / x)
    assert ratio > 1.0  # the 0.264/x correction is positive


# ---------------------------------------------------------------------------
# Gate 6 — the headline. The tail is exponential; a Gaussian's is not.
# ---------------------------------------------------------------------------
def test_minus_log_of_the_exceedance_is_linear_in_x_and_not_quadratic() -> None:
    r"""``d(-log Q)/dx -> 1 + 1/(2x)``: a straight line, measured across a decade.

    This is the one structural statement that separates the photon spectrum from any
    Gaussian: for a Gaussian ``-log Q`` grows as ``x^2``, so its slope would *double*
    between ``x`` and ``2x``. Here the slope is 1.017 on ``[20, 40]`` and 1.03 on
    ``[10, 20]`` — flat to the ``1/2x`` correction, which is itself asserted.
    """
    for lo, hi in ((5.0, 10.0), (10.0, 20.0), (20.0, 40.0)):
        slope = -(photon_log_survival(hi) - photon_log_survival(lo)) / (hi - lo)
        predicted = 1.0 + 0.5 * math.log(hi / lo) / (hi - lo)  # the 1/(2x) term, integrated
        assert slope == pytest.approx(predicted, abs=0.01)
        assert slope < 1.1  # emphatically not the 1.5x-per-doubling a Gaussian gives


@pytest.mark.parametrize(("x", "orders"), [(5.0, 13.5), (10.0, 61.4), (20.0, 261.4)])
def test_a_gaussian_of_the_same_mean_and_variance_is_this_many_orders_low(
    x: float, orders: float
) -> None:
    r"""The size of what B3's model leaves out, in decades.

    A Gaussian carrying this distribution's own mean ``8/(15 sqrt3)`` and variance
    ``11/27 - <x>^2`` puts ``x = 10`` at 17.3 standard deviations, and so predicts a
    photon that far out ``1e61`` times more rarely than the spectrum does. That is not a
    quantitative correction to the tail; it is the absence of one.

    The gap is *measured*, not predicted — there is no closed form for it — so what this
    gate buys is a pin: the numbers grow like ``x^2/2sigma^2 - x``, and any future change
    that flattened or steepened the tail would have to move them. Held to a tenth of a
    decade against gaps of hundreds, which no tolerance-loosening could survive.
    """
    from scipy.stats import norm

    mean = photon_spectrum_moment(1)
    sigma = math.sqrt(photon_spectrum_moment(2) - mean**2)
    gaussian = norm.logsf((x - mean) / sigma)
    gap = (photon_log_survival(x) - gaussian) / math.log(10.0)
    assert gap == pytest.approx(orders, abs=0.1)


def test_that_gap_quadruples_on_every_doubling_which_is_what_makes_it_a_gaussians() -> None:
    r"""The gap grows as ``x^2``, because the *Gaussian* does and the spectrum does not.

    A ratio test, so it needs none of the three measured numbers above to be right in
    absolute terms: doubling ``x`` multiplies a Gaussian's ``-log Q`` by four and the
    spectrum's by two, so their difference must quadruple. It does, twice over
    (4.55x then 4.26x, approaching 4 from above as the linear term thins out).
    """
    from scipy.stats import norm

    mean = photon_spectrum_moment(1)
    sigma = math.sqrt(photon_spectrum_moment(2) - mean**2)

    def gap(x: float) -> float:
        return float(photon_log_survival(x)) - float(norm.logsf((x - mean) / sigma))

    assert gap(10.0) / gap(5.0) == pytest.approx(4.55, abs=0.1)
    assert gap(20.0) / gap(10.0) == pytest.approx(4.26, abs=0.1)
    assert gap(20.0) / gap(10.0) < gap(10.0) / gap(5.0)  # descending towards 4


# ---------------------------------------------------------------------------
# Gate 7 — and only now, statistics.
# ---------------------------------------------------------------------------
def test_the_sampled_moments_are_the_closed_forms_within_a_stated_budget() -> None:
    r"""Three moments from ``4e6`` draws, each against ``sqrt(Var(x^m)/N)``.

    Budgets computed from the spectrum's own higher moments rather than tuned: the
    ``m``-th sample moment has standard error ``sqrt((<x^2m> - <x^m>^2)/N)``, so the
    gate tightens as ``1/sqrt(N)`` and would fail a sampler that was biased at the
    ``1e-3`` level while passing one that is merely noisy.
    """
    draws = np.asarray(photon_energy_quantile(np.random.default_rng(20260825).random(4_000_000)))
    for order in (1, 2, 3):
        expected = photon_spectrum_moment(order)
        spread = math.sqrt((photon_spectrum_moment(2 * order) - expected**2) / draws.size)
        assert float(np.mean(draws**order)) == pytest.approx(expected, abs=4.0 * spread)
    assert draws.min() > 0.0  # a photon carries energy; none of them is a gain
    assert 8.0 < draws.max() < 25.0  # 4e6 draws reach ~1e-7 exceedance, i.e. x ~ 13


def test_the_compound_sum_has_the_mean_variance_and_skew_the_moments_predict() -> None:
    r"""``mean = n <x>``, ``var = n <x^2>``, ``skew = <x^3> / (sqrt(n) <x^2>^(3/2))``.

    The three statements that separate this model from B3's. The variance uses
    ``<x^2>``, *not* ``<x>^2`` — a compound Poisson sum's variance is the rate times the
    mean **square**, which is the single algebraic fact B3's ``photon_energy_variance``
    rests on and which no mean-loss gate can see. The skewness is the new one: it is
    positive here and exactly zero for a Gaussian, and it inverts to the photon count,
    which is how B3's reference arm counted xtrack's photons from the outside.
    """
    rate, n_samples = 16.0, 400_000
    rng = np.random.default_rng(11)
    total = sample_photon_sum(np.full(n_samples, rate), rng)

    mean, second, third = (photon_spectrum_moment(m) for m in (1, 2, 3))
    assert float(total.mean()) == pytest.approx(rate * mean, rel=3e-3)
    assert float(total.var()) == pytest.approx(rate * second, rel=6e-3)
    skew = float(np.mean((total - total.mean()) ** 3)) / float(total.std()) ** 3
    assert skew == pytest.approx(third / (math.sqrt(rate) * second**1.5), rel=0.02)
    assert skew > 0.9  # B3's Gaussian is 0, and xtrack's own loss skew is -0.91
    assert np.all(total >= 0.0)  # and this model can never hand energy back

    # the skewness inverts to the photon count, with nothing else put in
    implied = (third / (skew * second**1.5)) ** 2
    assert implied == pytest.approx(rate, rel=0.05)


def test_the_sum_is_zero_exactly_when_no_photon_is_emitted() -> None:
    """A rate of zero radiates nothing at all, and consumes no uniforms.

    The degenerate case a lattice reaches at every drift and every thin element. It has
    to return zeros of the right shape rather than an empty ``bincount``.
    """
    rng = np.random.default_rng(3)
    assert np.array_equal(sample_photon_sum(np.zeros(5), rng), np.zeros(5))
    assert sample_photon_sum(np.zeros(()), rng).shape == (1,)
    drawn = sample_photon_sum(np.full(2000, 0.01), rng)
    assert drawn.shape == (2000,)
    assert np.count_nonzero(drawn) < 60  # Poisson(0.01): ~20 of 2000 emit at all


def test_the_same_seed_reproduces_bit_for_bit_and_a_different_seed_does_not() -> None:
    """A stochastic sampler is still an experiment that has to be repeatable."""
    rate = np.full(500, 12.0)
    first = sample_photon_sum(rate, np.random.default_rng(19))
    again = sample_photon_sum(rate, np.random.default_rng(19))
    other = sample_photon_sum(rate, np.random.default_rng(20))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
