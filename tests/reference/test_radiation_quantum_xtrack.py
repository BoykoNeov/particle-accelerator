r"""Cross-check quantum excitation (B3) against xtrack's photon-resolved emission.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**This is the most useful reference arm on axis B, because the two codes do not do the
same thing.** ``configure_radiation(model="quantum")`` in xtrack is a genuine compound
Poisson process: it draws the number of photons from an exponential free path, samples
each one's energy from the true synchrotron spectrum by rejection against ``K_{5/3}``,
and subtracts them one at a time. accsim draws a single Gaussian with the right mean and
the right variance and never counts a photon. Every number below that agrees is
therefore a statement that **the first two moments are all that matter** — which is the
entire justification for the Gaussian, checked against the thing it approximates rather
than asserted.

Three kinds of gate:

- **What must agree.** The mean loss (B2 already pins this to 6.5e-9 with
  ``model="mean"``; here it is only as sharp as xtrack's own sampling allows) and, the
  new content, the **standard deviation** — 0.2% on 200,000 particles, which is the
  statistical floor. A wrong ``C_q``, a dropped ``11/27``, or ``kappa^2`` instead of
  ``kappa^3`` would all move it far outside that.
- **What must differ, and by how much.** The distribution *shape*. xtrack's loss is
  skewed (a few hard photons in a long tail) and can never be negative; accsim's is
  symmetric and gains energy ~2.6% of the time. Both are asserted, because "the Gaussian
  is an approximation" is a claim with a size, and this is the size.
- **The photon count, recovered from the shape.** A compound Poisson sum's skewness is
  ``<u^3> / (sqrt(n_gamma) <u^2>^(3/2))``, so xtrack's measured skewness *counts its own
  photons* — and the count lands on ``(5/(2 sqrt3)) alpha gamma theta``, the textbook
  rate. That is the number the Gaussian throws away, measured from the tracking that
  proves the Gaussian is throwing it away.

The setup carries over from B2 unchanged: ``integrator="uniform"``,
``num_multipole_kicks=1``, so the two are the same map before any of this is compared.
xtrack's remaining approximations are B2's named owners (its pre-2019 elementary charge,
and ``gamma = gamma0(1+delta)`` in place of the on-shell forms); for the *variance* it
also uses ``beta0 gamma0`` in the photon rate and ``gamma^2 gamma0`` in the critical
energy, so its diffusion carries ``(1+delta)^4`` where the exact result carries
``(1+delta)^7``. All of these are far below the statistical floor of a stochastic
comparison, which is why the gates below are stated in units of that floor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

from accsim import Dipole, ReferenceParticle
from accsim.radiation_kick import HBAR_C_EV_M

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 0.51099895069e6  # electron, eV
LENGTH = 1.0
ANGLE = 2.0 * math.pi / 40
CURVATURE = ANGLE / LENGTH
N_PARTICLES = 200_000
SEED = 20260819


def _spectrum_moment(mu: float) -> float:
    """``int_0^inf x^mu F(x)/x dx`` — the same integral the analytic suite uses."""
    return quad(lambda t: kv(5.0 / 3.0, t) * t ** (mu + 1.0) / (mu + 1.0), 0.0, np.inf, limit=200)[
        0
    ]


def _line(k1: float = 0.0):
    """A one-magnet xtrack line whose integration matches accsim's one lumped kick."""
    model = "mat-kick-mat" if k1 else "bend-kick-bend"
    bend = xt.Bend(length=LENGTH, angle=ANGLE, k0=CURVATURE, k1=k1, model=model)
    line = xt.Line(elements=[bend], element_names=["b"])
    line.particle_ref = xt.Particles(mass0=MASS0, p0c=1.0e10)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    line["b"].integrator = "uniform"
    line["b"].num_multipole_kicks = 1  # one kick, like accsim -- see B2's module docstring
    line.configure_radiation(model="quantum")
    return line


def _xtrack_deltas(line, energy: float, n: int = N_PARTICLES, **coords) -> np.ndarray:
    """``delta`` after one magnet, for ``n`` particles started at the same point."""
    start = {key: np.full(n, value) for key, value in coords.items()}
    p = xt.Particles(
        mass0=MASS0,
        p0c=math.sqrt(energy**2 - MASS0**2),
        x=start.pop("x", np.zeros(n)),
        **start,
    )
    line.track(p)
    return np.asarray(p.delta, dtype=float)


def _accsim_deltas(energy: float, n: int = N_PARTICLES, k1: float = 0.0, **coords) -> np.ndarray:
    ref = ReferenceParticle.from_total_energy(MASS0, energy)
    elem = Dipole(LENGTH, ANGLE, k1=k1)
    state = np.zeros((6, n))
    for index, key in enumerate(("x", "px", "y", "py", "zeta", "delta")):
        if key in coords:
            state[index] = coords[key]
    rng = np.random.default_rng(SEED)
    return elem.track(state, ref, radiation="quantum", rng=rng)[5]


@pytest.fixture(scope="module")
def pure_bend_line():
    return _line()


@pytest.fixture(scope="module")
def combined_line():
    return _line(k1=0.6)


# ---------------------------------------------------------------------------
# What must agree: the first two moments, to the statistical floor.
# ---------------------------------------------------------------------------
def test_the_gaussian_and_the_photon_sum_have_the_same_spread(pure_bend_line) -> None:
    r"""The standard deviation of the loss agrees with xtrack's to the sampling floor.

    The headline cross-check of the milestone. accsim's spread comes from
    ``sigma_U^2 = 2 C_q E gamma^2 kappa U`` in closed form; xtrack's comes from actually
    emitting ~16 photons off a rejection-sampled ``K_{5/3}`` spectrum. Nothing is shared
    but the physics. With 200,000 particles the floor on a width comparison is
    ``1/sqrt(2N) = 0.16%``; the gate is three times that, and the observed disagreement
    is inside one.
    """
    energy = 5.0e9
    theirs = _xtrack_deltas(pure_bend_line, energy)
    ours = _accsim_deltas(energy)
    floor = 1.0 / math.sqrt(2.0 * N_PARTICLES)
    assert ours.std(ddof=1) == pytest.approx(theirs.std(ddof=1), rel=3.0 * floor)


def test_the_mean_loss_still_agrees_now_that_it_is_a_random_variable(pure_bend_line) -> None:
    """And the mean is undisturbed by the graininess, on both sides.

    B2 pins the mean to 6.5e-9 against ``model="mean"``; that gate is untouched and this
    one cannot be sharper than xtrack's own sampling error, which for the mean is
    ``sigma / (|mean| sqrt(N))`` ~ 0.11% here. What it *can* catch is a variance
    accidentally added to the mean — the natural implementation slip — which would show
    up as a bias of order ``sigma_U`` rather than of order ``sigma_U/sqrt(N)``.
    """
    energy = 5.0e9
    theirs = _xtrack_deltas(pure_bend_line, energy)
    ours = _accsim_deltas(energy)
    floor = theirs.std(ddof=1) / (abs(theirs.mean()) * math.sqrt(N_PARTICLES))
    assert ours.mean() == pytest.approx(theirs.mean(), rel=4.0 * floor)


def test_the_spread_tracks_xtrack_across_a_factor_sixteen_in_energy(pure_bend_line) -> None:
    r"""``sigma_U ∝ E^3.5`` on both sides — the power, not just one number.

    ``sigma_U^2 = 2 C_q E gamma^2 kappa U ∝ gamma^7 kappa^3 l``. A coefficient error is
    energy-independent and would survive this; a wrong *power* of ``gamma`` (the easy
    slip, since ``U`` itself carries ``gamma^4``) could not. Checked as the ratio of the
    two codes staying put across the range, and as the ratio within each code being the
    same power.
    """
    ratios, ours_by_energy, theirs_by_energy = [], [], []
    for energy in (2.5e9, 10.0e9):
        theirs = _xtrack_deltas(pure_bend_line, energy).std(ddof=1)
        ours = _accsim_deltas(energy).std(ddof=1)
        ratios.append(ours / theirs)
        ours_by_energy.append(ours)
        theirs_by_energy.append(theirs)
    floor = 1.0 / math.sqrt(2.0 * N_PARTICLES)
    assert ratios[0] == pytest.approx(ratios[1], rel=4.0 * floor)
    # delta = -U/(beta^2 E) so the spread in delta goes as gamma^3.5 / gamma = gamma^2.5
    assert math.log2(ours_by_energy[1] / ours_by_energy[0]) == pytest.approx(2.5 * 2.0, rel=2e-3)
    assert math.log2(theirs_by_energy[1] / theirs_by_energy[0]) == pytest.approx(5.0, rel=1e-2)


def test_the_spread_agrees_off_axis_in_a_combined_function_magnet(combined_line) -> None:
    r"""And with the field sampled at the particle's own ``x``, where ``kappa`` varies.

    The gradient makes the local curvature a function of position, so this checks that
    the ``kappa^3`` in the variance is evaluated on the same field, at the same point in
    the traversal, as the ``kappa^2`` in the mean. B2 established the mean here; the
    variance is a *different* power of the same quantity, so it is not implied.
    """
    energy = 5.0e9
    theirs = _xtrack_deltas(combined_line, energy, x=2.0e-3, px=1.0e-4)
    ours = _accsim_deltas(energy, k1=0.6, x=2.0e-3, px=1.0e-4)
    floor = 1.0 / math.sqrt(2.0 * N_PARTICLES)
    assert ours.std(ddof=1) == pytest.approx(theirs.std(ddof=1), rel=4.0 * floor)


# ---------------------------------------------------------------------------
# What must differ, and by how much: the model boundary, gated from both sides.
# ---------------------------------------------------------------------------
def test_xtrack_never_gains_energy_and_accsim_does_about_one_in_forty(pure_bend_line) -> None:
    r"""The clamping question, settled against the process being approximated.

    A real photon process cannot give a particle energy: xtrack's loss is a sum of
    positive draws and its ``delta`` never rises. accsim's unclamped Gaussian goes the
    wrong way ~2.6% of the time, because ``n_gamma ~ 16`` puts zero at 1.9 sigma. That is
    the price of not resolving photons, and it is deliberate: clamping would bias the
    mean and the variance by ~1%, which is five times the agreement the gates above
    achieve. Recorded as a measured boundary rather than a caveat in prose.
    """
    energy = 5.0e9
    theirs = _xtrack_deltas(pure_bend_line, energy)
    ours = _accsim_deltas(energy)
    assert float((theirs > 0.0).mean()) == 0.0
    assert 0.015 < float((ours > 0.0).mean()) < 0.045


def test_the_photon_sum_is_skewed_and_the_gaussian_is_not(pure_bend_line) -> None:
    r"""The third moment is where the two models part company, and it is not subtle.

    A handful of hard photons gives the compound Poisson a long tail; the Gaussian has
    none. This is the only moment that separates them, which is the point: the
    equilibrium is a second-moment quantity, so it cannot see this at all. What *can* is
    beam lifetime — a particle knocked out of the RF bucket by a single hard photon —
    which is exactly Stage 4's ``quantum_lifetime`` and a separate axis from this one.
    """
    energy = 5.0e9
    theirs = _xtrack_deltas(pure_bend_line, energy)
    ours = _accsim_deltas(energy)

    def skew(sample: np.ndarray) -> float:
        centred = sample - sample.mean()
        return float((centred**3).mean() / sample.std() ** 3)

    floor = math.sqrt(6.0 / N_PARTICLES)  # standard error of a sample skewness
    assert abs(skew(ours)) < 4.0 * floor  # symmetric, to the sampling floor
    assert skew(theirs) < -0.5  # and theirs is emphatically not


def test_xtracks_own_skewness_counts_its_photons_and_gets_the_textbook_rate(
    pure_bend_line,
) -> None:
    r"""``n_gamma`` recovered from the shape, landing on ``(5/(2 sqrt3)) alpha gamma theta``.

    For a compound Poisson sum the third central moment is ``n_gamma <u^3>`` and the
    second is ``n_gamma <u^2>``, so the skewness is
    ``<u^3> / (sqrt(n_gamma) <u^2>^(3/2))`` — it *falls* as the photon count rises, and
    inverting it counts the photons. The spectrum moments come from the analytic suite's
    own ``K_{5/3}`` integrals, so this closes the loop: accsim's assumed spectrum,
    applied to xtrack's measured shape, reproduces the emission rate xtrack computes
    from ``alpha`` independently. It is also the direct measurement of the number that
    decides whether a Gaussian is legitimate at all — ~16 photons per magnet here.
    """
    energy = 5.0e9
    theirs = _xtrack_deltas(pure_bend_line, energy)
    centred = theirs - theirs.mean()
    skewness = abs(float((centred**3).mean() / theirs.std() ** 3))

    number = _spectrum_moment(0.0)
    second = _spectrum_moment(2.0) / number  # <u^2>/u_c^2 = 11/27
    third = _spectrum_moment(3.0) / number  # <u^3>/u_c^3
    implied = (third / (skewness * second**1.5)) ** 2

    gamma = energy / MASS0
    alpha = ReferenceParticle.from_total_energy(MASS0, energy).classical_radius_m
    alpha = alpha * MASS0 / HBAR_C_EV_M  # r_e m c^2 = alpha hbar c
    expected = 2.5 / math.sqrt(3.0) * alpha * gamma * ANGLE
    assert implied == pytest.approx(expected, rel=0.05)
    assert 12.0 < implied < 22.0
