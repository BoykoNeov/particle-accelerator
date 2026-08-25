r"""The synchrotron-radiation photon spectrum, and drawing single photons from it (B5).

B3 gave the tracked radiation loss its *graininess* -- light comes in photons, so what a
particle loses crossing a magnet is a random variable, not a number -- and modelled that
randomness as a Gaussian of the right mean and the right variance. This module is the
thing the Gaussian stands in for: the actual distribution a single photon's energy is
drawn from, and a sampler that draws from it.

**The spectrum.** In units of the critical energy ``u_c = (3/2) hbar c gamma^3 kappa``,
the number of photons a particle emits per unit ``x = u/u_c`` is

    ``p(x) = (3 / 5 pi) * int_x^inf K_{5/3}(t) dt``,

the *number* spectrum. It is worth being explicit about which spectrum this is, because
the textbook figure is usually the **power** spectrum ``F(x) = x int_x^inf K_{5/3}``, and
the two differ by a factor of ``x`` -- using one where the other belongs makes every
photon ``<x^2>/<x>^2 = 4.297`` times too energetic, a pure number no dimensional
analysis would catch. Photons are counted here, so the number spectrum is the one to
sample.

**The normalisation is exact, not fitted.** ``int_0^inf t^(mu-1) K_nu(t) dt =
2^(mu-2) Gamma((mu-nu)/2) Gamma((mu+nu)/2)``, so swapping the order of integration once
gives ``int_0^inf p = (3/5 pi) int_0^inf K_{5/3}(t) t dt = (3/5 pi) Gamma(1/6)
Gamma(11/6) = (3/5 pi)(5 pi / 3) = 1`` by the reflection formula. The same Mellin
transform hands over every moment in closed form:

    ``<x>   = 8 / (15 sqrt3)``   (the mean photon energy B2's ``U`` is built on),
    ``<x^2> = 11 / 27``          (the mean square B3's variance is built on),
    ``<x^3> = 224 / (135 sqrt3)``,

so this module's sampler is gated against **rationals**, not against a quadrature that
could share its own mistake. The analytic suite derives all three.

**Two shapes, and where each one is computed.** The distribution is a power law at the
bottom and an exponential at the top, and no single expression is accurate at both ends:

    ``F(x) -> (27 / 10 pi) 2^(2/3) Gamma(5/3) x^(1/3)``   as ``x -> 0``,
    ``P(x > X) -> (3/5) e^-X / sqrt(2 pi X)``             as ``X -> inf``.

The far tail is the entire reason this module exists -- it is where a Gaussian of the
same mean and variance is wrong by orders of magnitude, predicting ``e^-X^2`` where the
truth is ``e^-X``.

**The quadratures, and the trap in them.** ``int_X^inf K_{5/3}(t) dt`` handed to
``scipy.integrate.quad`` in one piece **silently returns a wrong answer** for small
``X`` -- it warns "probably divergent" and returns essentially zero, which costs a
cumulative distribution exactly one third of its value -- so
every integral here is split at ``t = 1``: below it in ``log t``, which turns the
``t^(-5/3)`` singularity into a decaying exponential, and above it in the
exponentially-scaled ``kve`` so nothing underflows. The exceedance also collapses to a
*single* quadrature by swapping the order of integration, exactly as the moments do:

    ``P(x > X) = (3 / 5 pi) int_X^inf K_{5/3}(t) (t - X) dt``,

which in scaled form is ``e^-X`` times a well-conditioned integral -- so
:func:`photon_log_survival` is accurate at ``X = 40``, where the exceedance itself is
``1.6e-19`` and would be lost in any ``1 - F(X)``.

**The sampler is an inverse, which is what makes it gate-able.** Drawing by inverse
transform makes each photon energy an exactly predictable function of one uniform, so a
test can feed a chosen quantile and check the answer against the quadratures above --
including far out in the tail, where no amount of sampling would show a mistake. The
inverse is tabulated once per process (about a second of quadrature) in the two
variables the two shapes are straight lines in: ``log x`` against ``log q`` below the
median, where ``x ~ q^3``, and ``x`` against ``log P(x > X)`` above it, where ``x`` is
nearly ``-log q``. That is accurate to ~1e-9 relative across the whole range, which the
analytic suite measures rather than assumes.

What this module is **not**: it says nothing about *how many* photons are emitted or how
big ``u_c`` is. Those are the two dimensional numbers that couple the spectrum to a
ring, and they live with the kick that uses them, in :mod:`accsim.radiation_kick`.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import kv, kve

__all__ = [
    "SMALL_X_COEFFICIENT",
    "SPECTRUM_NORMALISATION",
    "SPECTRUM_ORDER",
    "photon_energy_quantile",
    "photon_log_survival",
    "photon_number_cdf",
    "photon_number_pdf",
    "photon_spectrum_moment",
    "sample_photon_sum",
]

#: The order of the modified Bessel function the spectrum is built on.
SPECTRUM_ORDER: Final[float] = 5.0 / 3.0

#: ``3 / (5 pi)`` -- the reciprocal of ``int_0^inf K_{5/3}(t) t dt = 5 pi / 3``, which is
#: what normalises the photon *number* spectrum to one photon.
SPECTRUM_NORMALISATION: Final[float] = 3.0 / (5.0 * math.pi)

#: ``(27 / 10 pi) 2^(2/3) Gamma(5/3) = 1.2316`` -- the coefficient of the ``x^(1/3)``
#: law the cumulative distribution obeys as ``x -> 0``. Used below the tabulated range,
#: where it is exact to better than 1e-11.
SMALL_X_COEFFICIENT: Final[float] = (
    27.0 / (10.0 * math.pi) * 2.0 ** (2.0 / 3.0) * math.gamma(SPECTRUM_ORDER)
)

# The smallest tabulated x. Below it the x^(1/3) law above takes over; it is reached by
# about one draw in 175000, and carries a photon energy of 1e-16 u_c.
_X_MIN: Final[float] = 1.0e-16
# The deepest exceedance the inverse must reach: numpy's uniforms are multiples of
# 2^-53, so 1 - q is never smaller than 1.1e-16 and log(1 - q) never below -36.74.
_LOG_Q_MIN: Final[float] = -37.0
# Points in each dense lookup table. 2^18 keeps the linear interpolation error ~1e-9.
_TABLE_POINTS: Final[int] = 1 << 18


# ---------------------------------------------------------------------------
# The exact spectrum, by quadrature. Slow and accurate: the reference the fast
# sampler below is gated against, never the thing tracking calls.
# ---------------------------------------------------------------------------
def _k53_tail_above_one() -> float:
    """``int_1^inf K_{5/3}(t) dt``, in scaled form so nothing underflows."""
    scaled = quad(lambda s: kve(SPECTRUM_ORDER, 1.0 + s) * math.exp(-s), 0.0, np.inf, limit=400)[0]
    return scaled * math.exp(-1.0)


_K53_TAIL_AT_ONE: Final[float] = _k53_tail_above_one()


def _log_k53_tail(x: float) -> float:
    """``log int_x^inf K_{5/3}(t) dt``, split at ``t = 1`` -- see the module docstring."""
    if x >= 1.0:
        scaled = quad(lambda s: kve(SPECTRUM_ORDER, x + s) * math.exp(-s), 0.0, np.inf, limit=400)[
            0
        ]
        return -x + math.log(scaled)
    below = quad(
        lambda u: kv(SPECTRUM_ORDER, math.exp(u)) * math.exp(u), math.log(x), 0.0, limit=400
    )[0]
    return math.log(below + _K53_TAIL_AT_ONE)


def _cdf_scalar(x: float) -> float:
    """``F(x)``, from ``int_0^x K t dt + x int_x^inf K dt`` -- accurate as ``x -> 0``."""
    if x <= 0.0:
        return 0.0
    inner = quad(lambda t: kv(SPECTRUM_ORDER, t) * t, 0.0, x, limit=400)[0]
    return SPECTRUM_NORMALISATION * (inner + x * math.exp(_log_k53_tail(x)))


def _log_survival_scalar(x: float) -> float:
    """``log P(x' > x)``, from the single collapsed quadrature -- accurate in the tail."""
    if x <= 0.0:
        return 0.0
    scaled = quad(lambda s: kve(SPECTRUM_ORDER, x + s) * s * math.exp(-s), 0.0, np.inf, limit=400)[
        0
    ]
    return math.log(SPECTRUM_NORMALISATION) - x + math.log(scaled)


def _elementwise(fn, x: np.ndarray | float) -> np.ndarray | float:
    """Apply a scalar quadrature over an array, preserving scalar-in/scalar-out."""
    values = np.array([fn(float(v)) for v in np.atleast_1d(np.asarray(x, dtype=float))])
    if np.ndim(x) == 0:
        return float(values[0])
    return values.reshape(np.shape(x))


def photon_number_pdf(x: np.ndarray | float) -> np.ndarray | float:
    r"""``p(x) = (3 / 5 pi) int_x^inf K_{5/3}(t) dt`` -- photons per unit ``x = u/u_c``.

    The **number** spectrum, normalised so ``int_0^inf p = 1``; multiply by the photon
    rate for photons per unit ``x`` per traversal. Underflows to zero past ``x ~ 700``,
    which is 300 orders of magnitude beyond anything a ring produces.
    """
    return _elementwise(
        lambda v: math.inf if v <= 0.0 else SPECTRUM_NORMALISATION * math.exp(_log_k53_tail(v)),
        x,
    )


def photon_number_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """``P(x' <= x)`` by quadrature. Relatively accurate down to ``x ~ 1e-20``; slow.

    The floor is quadrature, not physics: below it the ``log t`` integral spans more
    dynamic range than ``quad``'s relative tolerance can hold, and the answer drifts (3%
    at ``x = 5e-25``). It is far below anything that carries energy -- ``x = 1e-20`` is
    one draw in ``4e6`` -- and the region below it is where
    :func:`photon_energy_quantile` uses the exact ``x^(1/3)`` law instead of the table.
    """
    return _elementwise(_cdf_scalar, x)


def photon_log_survival(x: np.ndarray | float) -> np.ndarray | float:
    """``log P(x' > x)`` by quadrature. Relatively accurate as ``x -> inf``; slow.

    The logarithm is the point: at ``x = 40`` the exceedance is ``1.6e-19``, which no
    ``1 - cdf`` could resolve, and which is exactly where the Gaussian B3 ships is wrong
    by hundreds of orders of magnitude.
    """
    return _elementwise(_log_survival_scalar, x)


def photon_spectrum_moment(order: int) -> float:
    r"""``<x^m>`` in closed form, from the Mellin transform of ``K_{5/3}``.

    ``int_0^inf t^(mu-1) K_nu(t) dt = 2^(mu-2) Gamma((mu-nu)/2) Gamma((mu+nu)/2)``, and
    one swap of the order of integration turns ``<x^m>`` into that at ``mu = m + 2``,
    divided by ``m + 1`` and by the normalisation. ``m = 1, 2, 3`` give
    ``8/(15 sqrt3)``, ``11/27`` and ``224/(135 sqrt3)`` exactly; the analytic suite
    checks this against both the rationals and the quadrature.
    """
    mu = order + 2.0
    mellin = (
        2.0 ** (mu - 2.0)
        * math.gamma((mu - SPECTRUM_ORDER) / 2.0)
        * math.gamma((mu + SPECTRUM_ORDER) / 2.0)
    )
    return SPECTRUM_NORMALISATION * mellin / (order + 1.0)


# ---------------------------------------------------------------------------
# The fast inverse. Built once per process; this is what tracking calls.
# ---------------------------------------------------------------------------
class _InverseTable:
    """Two dense uniform lookup tables, one per end of the distribution.

    ``lo`` maps ``log q -> log x`` for ``q <= 1/2`` (a straight line of slope 3, near the
    bottom); ``hi`` maps ``log(1 - q) -> x`` for ``q > 1/2`` (a straight line of slope
    -1, in the tail). Splitting at the median is what lets both ends be *relatively*
    accurate: ``1 - q`` is exact in floating point for ``q >= 1/2`` (Sterbenz), so the
    tail never pays a cancellation to be looked up.
    """

    def __init__(self) -> None:
        knots = np.concatenate([np.geomspace(_X_MIN, 1.0, 2400), np.arange(1.0, 50.0, 0.02)[1:]])
        log_q = np.array([_log_survival_scalar(float(x)) for x in knots])
        below = knots <= 0.2  # comfortably past the median (0.0784), so it stays interior
        log_f = np.log([_cdf_scalar(float(x)) for x in knots[below]])

        # log x <- log q, below the median.
        spline_lo = CubicSpline(log_f, np.log(knots[below]))
        self.lo_min = float(log_f[0])
        self.lo_step = (math.log(0.5) - self.lo_min) / (_TABLE_POINTS - 1)
        self.lo_values = spline_lo(self.lo_min + self.lo_step * np.arange(_TABLE_POINTS))

        # x <- log(1 - q), above the median. log_q descends, so both arrays are reversed.
        above = knots >= 0.02
        spline_hi = CubicSpline(log_q[above][::-1], knots[above][::-1])
        self.hi_min = _LOG_Q_MIN
        self.hi_step = (math.log(0.5) - _LOG_Q_MIN) / (_TABLE_POINTS - 1)
        self.hi_values = spline_hi(self.hi_min + self.hi_step * np.arange(_TABLE_POINTS))

        # A cumulative distribution's inverse is monotone; a cubic spline is under no
        # obligation to be. Cheap to assert, and it would be a silent physics bug.
        # (``hi`` is indexed by log(1 - q), which *rises* towards the median, so the
        # energy it returns must fall.)
        if not (np.all(np.diff(self.lo_values) > 0.0) and np.all(np.diff(self.hi_values) < 0.0)):
            raise RuntimeError("the tabulated photon-energy inverse came out non-monotone")

    @staticmethod
    def _lookup(s: np.ndarray, start: float, step: float, values: np.ndarray) -> np.ndarray:
        t = (s - start) * (1.0 / step)
        # Clip the *position* into the table, then the *index* to the last interval --
        # not the position to the last index, which would silently drop the fractional
        # part at the top edge and put a 4e-5 kink exactly at the median.
        np.clip(t, 0.0, _TABLE_POINTS - 1.0, out=t)
        index = np.minimum(t.astype(np.intp), _TABLE_POINTS - 2)
        low = values[index]
        return low + (t - index) * (values[index + 1] - low)

    def __call__(self, q: np.ndarray) -> np.ndarray:
        out = np.empty_like(q)
        tail = q > 0.5
        head = ~tail
        if np.any(head):
            with np.errstate(divide="ignore"):  # q == 0 is legal and lands at x = 0
                log_q = np.log(q[head])
            tabulated = np.exp(self._lookup(log_q, self.lo_min, self.lo_step, self.lo_values))
            # Below the table the x^(1/3) law takes over; it is exact there to ~1e-11.
            out[head] = np.where(
                log_q < self.lo_min, (q[head] / SMALL_X_COEFFICIENT) ** 3, tabulated
            )
        if np.any(tail):
            out[tail] = self._lookup(np.log1p(-q[tail]), self.hi_min, self.hi_step, self.hi_values)
        return out


_TABLE: _InverseTable | None = None


def _inverse_table() -> _InverseTable:
    """The tabulated inverse, built on first use (about a second of quadrature)."""
    global _TABLE
    if _TABLE is None:
        _TABLE = _InverseTable()
    return _TABLE


def photon_energy_quantile(q: np.ndarray | float) -> np.ndarray | float:
    r"""``x`` such that ``P(x' <= x) = q`` -- the photon energy at quantile ``q``.

    In units of the critical energy. This is the sampler's whole content: one uniform
    in, one photon energy out, deterministically -- which is what lets a test feed a
    chosen quantile and compare against :func:`photon_number_cdf` /
    :func:`photon_log_survival` rather than estimate anything from samples.
    """
    values = np.array(np.asarray(q, dtype=float), copy=True, ndmin=1)
    if np.any(values < 0.0) or np.any(values >= 1.0):
        raise ValueError(f"photon-energy quantiles need q in [0, 1), got {q!r}")
    out = _inverse_table()(values.ravel())
    out[values.ravel() == 0.0] = 0.0
    if np.ndim(q) == 0:
        return float(out[0])
    return out.reshape(np.shape(q))


def sample_photon_sum(rate: np.ndarray | float, rng: np.random.Generator) -> np.ndarray:
    r"""``sum_i x_i`` over ``N ~ Poisson(rate)`` photons drawn from the spectrum.

    In units of the critical energy, one entry per entry of ``rate``. This is the
    compound-Poisson sum B3's Gaussian approximates: its mean is ``rate <x>`` and its
    variance ``rate <x^2>`` -- *not* ``rate <x>^2``, which is why a model that got the
    mean photon energy right and the mean square wrong would still pass every one of
    B2's gates. It can never be negative and it is positively skewed; the Gaussian is
    neither, and the analytic suite gates on exactly that pair.

    Two draws come off ``rng``, always in this order: the photon counts for the whole
    input at once, then one uniform per photon. So a bunch and a single particle consume
    the generator differently -- the same statement B3's Gaussian already carries.
    """
    rates = np.atleast_1d(np.asarray(rate, dtype=float))
    counts = rng.poisson(rates)
    total = int(np.sum(counts))
    if total == 0:
        return np.zeros(rates.shape)
    energies = photon_energy_quantile(rng.random(total))
    return np.bincount(
        np.repeat(np.arange(rates.size), counts.ravel()),
        weights=np.asarray(energies).ravel(),
        minlength=rates.size,
    ).reshape(rates.shape)
