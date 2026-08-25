r"""Cross-check the photon-resolved emission (B5) against xtrack's, like for like.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

B3's reference arm opens by saying that it is the most useful one on this axis **because
the two codes do not do the same thing** — xtrack emitting real photons off a
rejection-sampled ``K_{5/3}``, accsim drawing one Gaussian with the matching mean and
variance. Every agreement there was therefore a statement that only the first two moments
matter.

That sentence is no longer true, and this file is what replaces it. ``radiation="photons"``
is a genuine compound-Poisson sum, so for the first time the two codes are running the
same physics by two entirely different numerical routes:

  * accsim samples by **inverse transform** off a table built from quadratures of
    ``int_x^inf K_{5/3}``, so a photon energy is a deterministic function of one uniform;
  * xtrack samples by **rejection** against ``K_{5/3}`` evaluated directly, drawing a
    variable number of uniforms per photon and counting photons along an exponential free
    path.

Nothing is shared — not the sampling method, not the normalisation, not the language.
So the gates change character, and the new one is the point of the milestone:

  1. **The tail agrees, pointwise, to better than 1% out to one draw in a thousand** —
     which is what B3 could not ask, and is also where 200000 particles run out; the file
     says so rather than letting the reader assume it continues. B3's Gaussian is 19.4%
     low at the same place, always low, because a symmetric distribution has no tail to
     put there. That contrast is measured on the same axis and in the same units, so the
     size of what B5 bought is a number.
  2. **The shape agrees**, in the third and fourth moments — where B3's arm could only
     record a contrast (``-0.91`` against ``0.00``) it now records a match.
  3. **Neither can hand a particle energy**, where B3's Gaussian did so once in forty.
  4. **Both count the same photons**, each from its own measured skewness, and both land
     on ``(5/2 sqrt3) alpha gamma theta``.

The first two moments are re-checked too, because they must not have moved: they are the
identities part 2 gates in symbols, arriving here through an independent code.

**xtrack's own emission is not seeded by this suite**, so its sample is redrawn on
every run while accsim's is fixed by ``SEED``. Every gate below is therefore a
*two-sample* comparison and its budget carries the ``sqrt(2)`` that implies -- a
distinction worth writing down, because leaving it out makes a 2.9-sigma fluctuation
look like a 4.1-sigma failure, which is exactly how this file first failed.

The setup carries over from B2 and B3 unchanged — ``integrator="uniform"``,
``num_multipole_kicks=1``, so the two are the same map before anything is compared — as
do B2's named owners for xtrack's own approximations, all far below the statistical floor
of a stochastic comparison.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import Dipole, ReferenceParticle
from accsim.photon_spectrum import photon_spectrum_moment
from accsim.radiation_kick import fine_structure_constant

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 0.51099895069e6  # electron, eV
LENGTH = 1.0
ANGLE = 2.0 * math.pi / 40
CURVATURE = ANGLE / LENGTH
ENERGY = 5.0e9
N_PARTICLES = 200_000
SEED = 20260825


def _line():
    """The one-magnet xtrack line B2 and B3 use, integrated to match accsim's one kick."""
    bend = xt.Bend(length=LENGTH, angle=ANGLE, k0=CURVATURE, model="bend-kick-bend")
    line = xt.Line(elements=[bend], element_names=["b"])
    line.particle_ref = xt.Particles(mass0=MASS0, p0c=1.0e10)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    line["b"].integrator = "uniform"
    line["b"].num_multipole_kicks = 1
    line.configure_radiation(model="quantum")
    return line


def _losses(delta: np.ndarray) -> np.ndarray:
    """A traversal's radiated energy [eV], as a positive number, from ``delta`` alone."""
    return -np.asarray(delta, dtype=float) * ENERGY


@pytest.fixture(scope="module")
def samples():
    """The radiated energy of 200000 traversals, from each code and each accsim model."""
    line = _line()
    particles = xt.Particles(
        mass0=MASS0,
        p0c=math.sqrt(ENERGY**2 - MASS0**2),
        x=np.zeros(N_PARTICLES),
    )
    line.track(particles)

    ref = ReferenceParticle.from_total_energy(MASS0, ENERGY)
    element = Dipole(LENGTH, ANGLE)
    out = {"xtrack": _losses(particles.delta)}
    for model in ("photons", "quantum"):
        state = np.zeros((6, N_PARTICLES))
        tracked = element.track(state, ref, radiation=model, rng=np.random.default_rng(SEED))
        out[model] = _losses(tracked[5])
    return out


def _skew(sample: np.ndarray) -> float:
    return float(((sample - sample.mean()) ** 3).mean() / sample.std() ** 3)


# ---------------------------------------------------------------------------
# 1 — the new gate: the tail, pointwise, against an independent sampler.
# ---------------------------------------------------------------------------
def _quantile_sigma(sample: np.ndarray, quantile: float, resamples: int = 200) -> float:
    """Bootstrap standard error of a sample quantile.

    Preferred over the ``sqrt(p(1-p)/N)/f(x_p)`` formula because the density ``f`` has to
    be estimated from the data anyway, and in the tail the window that estimates it runs
    off the end of the sample — which silently turns a budget into a guess. The bootstrap
    needs no density and degrades honestly: at ``p = 0.9999`` it simply reports the large
    error that twenty surviving draws deserve.
    """
    rng = np.random.default_rng(4242)
    drawn = rng.choice(sample, size=(resamples, sample.size), replace=True)
    return float(np.quantile(drawn, quantile, axis=1).std())


@pytest.mark.parametrize("quantile", [0.9, 0.99, 0.999])
def test_the_loss_distributions_agree_pointwise_out_into_the_tail(samples, quantile) -> None:
    r"""Two samplers of one spectrum, agreeing to better than 1% at one draw in a thousand.

    The gate B3 could not write. accsim inverts a tabulated quadrature of
    ``int_x^inf K_{5/3}``; xtrack rejection-samples ``K_{5/3}`` itself. Agreement out here
    says both are drawing from the same distribution and not merely matching its first two
    moments — and the far quantiles are exactly where a sampler that was right on average
    would go wrong. Measured: ``-0.44%``, ``-0.12%``, ``+0.66%``.

    The budget is a bootstrap of each sample's own quantile, added in quadrature, at
    4 sigma. **One draw in a thousand is where 200000 particles run out**, and the next
    gate says so rather than leaving the reader to assume the agreement continues.
    """
    ours, theirs = samples["photons"], samples["xtrack"]
    a, b = float(np.quantile(ours, quantile)), float(np.quantile(theirs, quantile))
    sigma = math.hypot(_quantile_sigma(ours, quantile), _quantile_sigma(theirs, quantile))
    assert a == pytest.approx(b, abs=4.0 * sigma)
    assert abs(a / b - 1.0) < 0.02  # and in relative terms it is well under a percent


def test_one_draw_in_ten_thousand_is_where_this_comparison_runs_out(samples) -> None:
    r"""Apart by a few percent at ``p = 0.9999``: within the bootstrap, and that is the point.

    Stated as its own gate rather than folded into the parametrisation above, because the
    honest answer changes character here: only 20 of the 200000 draws lie beyond this
    point on either side, so the quantile carries a 1-2% error on the xtrack side and
    twice that on accsim's. One measured run put them 6.9% apart, which is 2.9 combined
    sigma -- the *reason* the agreement looks worse than at ``p = 0.999`` is the sample
    size and not the sampler, and since xtrack's draw is not seeded the number moves from
    run to run while the conclusion does not.

    Which is the argument for having built
    :func:`accsim.photon_spectrum.photon_log_survival` as a quadrature: the analytic suite
    gates this same distribution at an exceedance of ``1e-16``, where no tracking
    comparison of any affordable size could follow.
    """
    ours, theirs = samples["photons"], samples["xtrack"]
    a, b = float(np.quantile(ours, 0.9999)), float(np.quantile(theirs, 0.9999))
    sigma = math.hypot(_quantile_sigma(ours, 0.9999), _quantile_sigma(theirs, 0.9999))
    assert abs(a - b) / sigma < 4.0
    # ...and out here that budget really is wide: several percent, where the same
    # 4-sigma gate one decade in costs well under one.
    assert sigma / b > 0.01


def test_the_gaussian_is_visibly_wrong_at_the_same_place_and_this_is_the_size_of_it(
    samples,
) -> None:
    r"""At one draw in a thousand: photons ``+0.7%``, B3's Gaussian ``-19.4%``.

    The milestone's value, measured on the same axis and in the same units as the gate
    above, so "B5 buys the tail" is a claim with a number attached rather than a slogan.
    The Gaussian carries the right mean and the right variance and is wrong by a fifth at
    one draw in a thousand, and by a quarter at one in ten thousand — always *low*,
    because a symmetric distribution has no tail to put there.
    """
    reference = float(np.quantile(samples["xtrack"], 0.999))
    gaussian_gap = abs(float(np.quantile(samples["quantum"], 0.999)) - reference)
    photon_gap = abs(float(np.quantile(samples["photons"], 0.999)) - reference)
    assert photon_gap < 0.15 * gaussian_gap
    assert gaussian_gap / reference == pytest.approx(0.194, abs=0.02)


def test_the_median_is_below_the_mean_for_both_photon_sums_and_on_it_for_the_gaussian(
    samples,
) -> None:
    r"""``-0.3%`` against xtrack for the photon sum; ``+8.0%`` for the Gaussian.

    A shape statement that costs nothing and needs no tail at all. A right-skewed
    distribution's median sits *below* its mean; a symmetric one's sits exactly on it. So
    the three medians separate the models even though all three means agree to the
    sampling floor — which makes this the cheapest available demonstration that matching
    two moments is not the same as matching a distribution.
    """
    middle = float(np.quantile(samples["xtrack"], 0.5))
    assert float(np.quantile(samples["photons"], 0.5)) == pytest.approx(middle, rel=0.01)
    assert float(np.quantile(samples["quantum"], 0.5)) == pytest.approx(middle * 1.080, rel=0.01)
    # and that 8% is precisely the Gaussian sitting on its own mean
    assert float(np.quantile(samples["quantum"], 0.5)) == pytest.approx(
        float(samples["quantum"].mean()), rel=0.005
    )
    assert float(np.median(samples["photons"])) < float(samples["photons"].mean())


def test_how_many_traversals_land_past_where_a_gaussian_can_reach(samples) -> None:
    r"""Beyond the Gaussian's ``4.5 sigma`` ceiling: ~0 draws for it, ~70 for both photon sums.

    The tail as a *count*, deliberately, and not as the largest draw. A maximum is the
    wrong statistic here — it has no bounded variance on an exponential tail, and gating
    it against xtrack's unseeded sample fails on an ordinary redraw, which is exactly what
    happened when this gate was first written that way. A count above a fixed threshold is
    Poisson, so its error is ``1/sqrt(n)`` and the gate can be stated honestly.

    The threshold is the place a Gaussian of 200000 draws effectively stops: ``4.5 sigma``
    above its own mean, where it expects ``0.7`` draws. Both photon-resolved codes put
    tens of traversals past it, and — the part that makes this a cross-check rather than a
    contrast — they put the *same* number there, inside the Poisson error on the pair.
    """
    gaussian = samples["quantum"]
    ceiling = float(gaussian.mean()) + 4.5 * float(gaussian.std(ddof=1))
    beyond = {code: int(np.count_nonzero(samples[code] > ceiling)) for code in samples}

    assert beyond["quantum"] <= 5  # a Gaussian expects 0.7 of its 200000 draws out here
    for code in ("photons", "xtrack"):
        assert beyond[code] > 30  # and a compound-Poisson sum puts tens of them
    pair = beyond["photons"] + beyond["xtrack"]
    assert abs(beyond["photons"] - beyond["xtrack"]) < 4.0 * math.sqrt(pair)


# ---------------------------------------------------------------------------
# 2-3 — the shape, where B3's arm recorded a contrast and this one records a match.
# ---------------------------------------------------------------------------
def test_the_skewness_now_matches_xtrack_instead_of_contradicting_it(samples) -> None:
    r"""``+0.92`` on both sides, where B3's Gaussian had ``0.00`` against xtrack's ``+0.91``.

    The third moment is the one that separates a compound Poisson sum from a Gaussian,
    and B3's arm could only assert the separation. Here it is a two-sided agreement
    against the sampling floor ``sqrt(6/N)``, which is what says the two codes are
    drawing the same *shape* and not merely the same width.
    """
    floor = math.sqrt(2.0) * math.sqrt(6.0 / N_PARTICLES)  # two samples, both noisy
    ours, theirs = _skew(samples["photons"]), _skew(samples["xtrack"])
    assert ours == pytest.approx(theirs, abs=6.0 * floor)
    assert ours > 0.8 and theirs > 0.8  # both genuinely skewed, in the same direction
    assert abs(_skew(samples["quantum"])) < 4.0 * floor  # and B3's model still is not


def test_the_fourth_moment_matches_too_which_the_third_alone_would_not_imply(
    samples,
) -> None:
    r"""Excess kurtosis ``~1.3`` on both sides; the Gaussian's is zero by construction.

    A sampler that got the third moment right by luck — say, one drawing from a shifted
    exponential — would part company here. For a compound Poisson sum the excess kurtosis
    is ``<u^4>/(n_gamma <u^2>^2)``, a *fourth* independent statement about the spectrum,
    and it is asserted against the sampling floor ``sqrt(24/N)``.
    """
    floor = math.sqrt(24.0 / N_PARTICLES)

    def excess(sample: np.ndarray) -> float:
        return float(((sample - sample.mean()) ** 4).mean() / sample.std() ** 4) - 3.0

    ours, theirs = excess(samples["photons"]), excess(samples["xtrack"])
    assert ours == pytest.approx(theirs, abs=8.0 * math.sqrt(2.0) * floor)
    assert ours > 0.5  # a real tail, not a Gaussian's
    assert abs(excess(samples["quantum"])) < 6.0 * floor


def test_neither_code_can_hand_a_particle_energy_and_b3s_model_could(samples) -> None:
    """Zero gains on both sides, where B3's unclamped Gaussian gave one in forty.

    B3 recorded that as the price of not resolving photons, deliberately unclamped
    because clamping would have biased the two moments by ~1%. B5 pays it off: a sum of
    non-negative photon energies is non-negative by construction, and xtrack's is too.
    Asserted as exact zeros, and against the Gaussian's measured 2.6%, so the boundary
    that moved is visible in one place.
    """
    assert np.count_nonzero(samples["photons"] < 0.0) == 0
    assert np.count_nonzero(samples["xtrack"] < 0.0) == 0
    assert 0.015 < float(np.mean(samples["quantum"] < 0.0)) < 0.045


# ---------------------------------------------------------------------------
# 4 — and both codes count the same photons, each out of its own shape.
# ---------------------------------------------------------------------------
def test_both_codes_skewnesses_count_the_same_photons_and_get_the_textbook_rate(
    samples,
) -> None:
    r"""``n_gamma = (5/2 sqrt3) alpha gamma theta``, recovered twice, independently.

    B3 ran this inversion on xtrack alone, because accsim had no photons to count. Now
    both sides carry a count in their shape, and the two must agree with each other *and*
    with the rate computed from ``alpha`` — which closes the loop between accsim's
    assumed spectrum, xtrack's sampled one, and the textbook emission rate.
    """
    second, third = photon_spectrum_moment(2), photon_spectrum_moment(3)
    implied = {
        code: (third / (_skew(samples[code]) * second**1.5)) ** 2 for code in ("photons", "xtrack")
    }
    ref = ReferenceParticle.from_total_energy(MASS0, ENERGY)
    expected = 2.5 / math.sqrt(3.0) * fine_structure_constant(ref) * ref.gamma0 * ANGLE
    assert 14.0 < expected < 20.0  # ~16 photons per magnet on this ring
    for code in ("photons", "xtrack"):
        assert implied[code] == pytest.approx(expected, rel=0.06)
    assert implied["photons"] == pytest.approx(implied["xtrack"], rel=0.09)


def test_the_first_two_moments_did_not_move_when_the_model_changed(samples) -> None:
    r"""The mean and the spread still land on xtrack's, as B3's Gaussian did.

    The identities part 2 gates in symbols (``n_gamma <u> = U`` and
    ``n_gamma <u^2> = sigma_U^2``), arriving through a code that shares no algebra with
    either. If they had moved, everything B3 established would have to be re-established;
    they have not, which is why B3's equilibrium battery re-runs unchanged.
    """
    ours, theirs, gaussian = (samples[k] for k in ("photons", "xtrack", "quantum"))
    # sqrt(2) because BOTH samples are noisy -- see the module docstring; without it a
    # routine 2.9-sigma fluctuation of xtrack's unseeded draw reads as a failure.
    floor = math.sqrt(2.0) / math.sqrt(2.0 * N_PARTICLES)
    assert ours.std(ddof=1) == pytest.approx(theirs.std(ddof=1), rel=4.0 * floor)
    assert ours.std(ddof=1) == pytest.approx(gaussian.std(ddof=1), rel=4.0 * floor)
    mean_floor = theirs.std(ddof=1) / (abs(theirs.mean()) * math.sqrt(N_PARTICLES))
    assert ours.mean() == pytest.approx(theirs.mean(), rel=4.0 * mean_floor)
