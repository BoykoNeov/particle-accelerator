r"""Analytic gates — the photon-resolved radiation kick (B5, part 2).

Part 1 gated the spectrum and its sampler with no ring attached. This file attaches it:
``radiation="photons"`` draws a Poisson count of photons per element and an energy for
each off the synchrotron spectrum, where ``radiation="quantum"`` drew one Gaussian of
the same mean and the same variance.

**The claim the milestone is built on is that almost nothing changes.** A model that
alters every single draw and no aggregate is the strongest available statement that both
models are right, so the gates are arranged as two lists: what must be *identical*, and
what must be *different*, both written down before they are measured.

Identical, and not approximately — as algebraic identities:

  1. ``n_gamma <u> = U``, the classical loss B2 ships. B3 gated this symbolically; here
     it is gated **through the shipped code path**, on an off-axis trajectory with a
     non-zero ``zeta``, where a ``kappa`` or an ``l_path`` computed differently between
     the mean route and the photon route would show up and a symbolic identity could
     not. Both dimensional numbers are *captured out of* the shipped kick by a stand-in
     generator rather than recomputed in the test.
  2. ``n_gamma <u^2> =`` :func:`photon_energy_variance`, the variance B3 ships, by the
     same route. This is the one that matters: the equilibrium beam is the fixed point
     of diffusion against damping, and the diffusion coefficient *is* this variance, so
     these two identities are the whole reason B3's battery re-runs unchanged.

Different, in exactly three ways, all pre-committed:

  3. **The loss can never be negative.** B3's Gaussian hands a particle *energy* in ~2.6%
     of draws — deliberately, because clamping it would bias the two moments above. A sum
     of photons cannot, and the gate asserts the pair: 2.6% against exactly zero.
  4. **The loss is skewed**, and the skewness *counts the photons*:
     ``<u^3> / (sqrt(n_gamma) <u^2>^(3/2))``. Run on ``delta`` it comes out at ``-0.92``,
     which is where xtrack's genuine photon sum sits (``-0.91``) and where B3's Gaussian
     cannot go (``0``). The count it implies is checked against :func:`photon_rate`.
  5. ...**and yet none of that survives a turn**, which is the central-limit theorem and
     is measured here as a law: crossing ``N`` magnets suppresses the skewness as
     ``1/sqrt(N)``, checked over a factor of 100 in ``N``. One turn of the real ring is
     already down to ``-0.129`` with a Gaussian kurtosis, and the escape B4 measures
     takes a thousand turns. This is the reason B5 predicts B4's lifetime will not move,
     and gating it here gates the *argument* rather than waiting for an expensive null
     result to confirm it.

The deterministic gate throughout is a stand-in generator that returns a chosen photon
count and chosen quantiles, so one element's loss is a closed-form number and the whole
kick — mid-point curvature, path length, the on-shell factor, the one common scaling of
``(px, py, 1+delta)`` — is checked with no statistics at all.

Cost: the whole file is ~25 s, of which ~3 s is the one-off build of the sampler's
inverse table and the rest is the two 200000-particle shape measurements.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import Bunch, Dipole, Lattice, ReferenceParticle, RFCavity, ThinQuadrupole, Tracker
from accsim.coords import DELTA
from accsim.photon_spectrum import photon_energy_quantile, photon_spectrum_moment
from accsim.radiation_kick import (
    HBAR_C_EV_M,
    RADIATION_MODELS,
    STOCHASTIC_MODELS,
    critical_photon_energy,
    fine_structure_constant,
    photon_energy_variance,
    photon_rate,
    radiation_kick,
)
from accsim.symplectic import is_symplectic_map

ELECTRON_MASS_EV = 0.51099895069e6
L_BEND = 1.0
# The same isomagnetic 20-cell 5 GeV ring B3 gates its equilibrium on, so the two
# milestones are measured on one machine and their numbers are directly comparable.
COLD = {"cells": 20, "focal": 2.5, "energy": 5.0e9, "voltage": 9.0e6, "harmonic": 20}


def _ring(cells: int, focal: float, energy: float, voltage: float, harmonic: int) -> Lattice:
    """B3's ring: isomagnetic FODO, total bend ``2 pi``, with the RF that replaces ``U0``."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, energy)
    angle = 2.0 * math.pi / (2 * cells)
    cell = [
        ThinQuadrupole(0.5 / focal),
        Dipole(L_BEND, angle),
        ThinQuadrupole(-1.0 / focal),
        Dipole(L_BEND, angle),
        ThinQuadrupole(0.5 / focal),
    ]
    elements = list(cell) * cells
    lat = Lattice(elements, ref=ref)
    cavity = RFCavity.from_harmonic(voltage, harmonic, lat.length, ref, phi_s=math.pi)
    return Lattice([*elements, cavity], ref=ref)


# ---------------------------------------------------------------------------
# Measurement machinery. Not physics -- the physics is what it is applied to.
#
# _ChosenPhotons is the piece worth reading: a stand-in for numpy's Generator that
# hands back a chosen photon count and chosen quantiles, and *records the emission rate
# the shipped code asked it for*. That makes the loss a closed-form number and hands
# the test n_gamma without recomputing it, which is the difference between checking the
# kick and checking a copy of the kick.
# ---------------------------------------------------------------------------
class _ChosenPhotons:
    """``rng`` double: ``count`` photons each, at ``quantiles``, cycling if need be."""

    def __init__(self, count: int, quantiles: float | list[float]) -> None:
        self.count = count
        self.quantiles = np.atleast_1d(np.asarray(quantiles, dtype=float))
        self.rates: np.ndarray | None = None  # captured out of the shipped call

    def poisson(self, rates: np.ndarray) -> np.ndarray:
        self.rates = np.array(rates, dtype=float, copy=True)
        return np.full(np.shape(rates), self.count, dtype=np.int64)

    def random(self, size: int) -> np.ndarray:
        return np.resize(self.quantiles, size)


def _radiated(before: np.ndarray, after: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
    """The energy [eV] a traversal actually took out, recovered from ``delta`` alone.

    Inverting the on-shell factor rather than differencing two total energies: the naive
    ``E(delta_in) - E(delta_out)`` cancels two numbers of size ``E`` and keeps four
    digits of a loss that is ``1e-5`` of it. Written as ``u = L / (E + sqrt(E^2 - L))``,
    the rationalised root of ``u(2E - u) = L``, which cancels nothing.

    ``1 - f`` is formed as ``(delta_in - delta_out)/(1 + delta_in)`` and never as
    ``1 - (1+delta_out)/(1+delta_in)``: the second adds one to a number of size ``1e-7``
    and then takes it away again, which costs nine digits of the answer and was
    measured doing so. What is left is the honest floor of the measurement — the loss is
    only knowable through ``delta`` to ``1e-16 * delta_in / (delta_in - delta_out)`` —
    so the tight gates below start a particle at ``delta = 0``, where that is exact.
    """
    momentum = ref.momentum_eV * (1.0 + before)
    energy = np.sqrt(momentum**2 + ref.mass_eV**2)
    one_minus_f = (before - after) / (1.0 + before)
    one_plus_f = 1.0 + (1.0 + after) / (1.0 + before)
    lhs = one_minus_f * one_plus_f * momentum**2
    return lhs / (energy + np.sqrt(energy * energy - lhs))


def _probe(element, state: np.ndarray, ref: ReferenceParticle, quantile: float = 0.5):
    """Run one element under ``"photons"`` with exactly one photon of a chosen quantile.

    Returns ``(u_c, n_gamma)`` as the shipped kick computes them: the rate is captured
    from the generator call, and the critical energy falls out of the single photon's
    energy divided by its (independently known) dimensionless draw.
    """
    after = element.track(state.copy(), ref)
    rng = _ChosenPhotons(1, quantile)
    out = radiation_kick(element, state, after, ref, model="photons", rng=rng)
    loss = _radiated(after[DELTA], out[DELTA], ref)
    assert rng.rates is not None
    return float(loss) / photon_energy_quantile(quantile), float(np.ravel(rng.rates)[0])


# ---------------------------------------------------------------------------
# Gate 1 — the two identities, through the shipped code, off the design orbit.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("state", "tolerance"),
    [
        (np.zeros(6), 1e-13),
        (np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 3.0e-3, 0.0]), 1e-13),
        (np.array([-5.0e-3, -3.0e-4, 1.0e-3, -2.0e-4, -8.0e-3, 0.0]), 1e-13),
        # and one off-momentum, where reading the loss back out of delta costs digits
        (np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 3.0e-3, 4.0e-4]), 1e-10),
    ],
)
def test_the_photon_rate_times_the_mean_photon_energy_is_the_classical_loss(
    state: np.ndarray, tolerance: float
) -> None:
    r"""``n_gamma <u> == U`` to 1e-13, on trajectories that are not the design orbit.

    B3 proved this identity in symbols, where ``kappa`` and ``l`` are letters. Here they
    are numbers the shipped kick computed for itself — off-axis, at an angle, at a
    non-zero ``zeta`` (so the path length is not the element length) and at a non-zero
    ``delta`` (so the energy is not the reference one). If the photon route sampled the
    field at a different point along the traversal from the mean route, or used the
    element length where the mean route uses the trajectory length, the two would part
    company here by parts in a thousand and nowhere else in the suite.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    bend = Dipole(L_BEND, 2.0 * math.pi / (2 * COLD["cells"]))
    u_c, rate = _probe(bend, state, ref)

    after = bend.track(state.copy(), ref)
    classical = float(_radiated(after[DELTA], radiation_kick(bend, state, after, ref)[DELTA], ref))
    assert rate * u_c * photon_spectrum_moment(1) == pytest.approx(classical, rel=tolerance)


@pytest.mark.parametrize(
    ("state", "tolerance"),
    [
        (np.zeros(6), 1e-13),
        (np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 3.0e-3, 0.0]), 1e-13),
        (np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 3.0e-3, 4.0e-4]), 1e-10),
    ],
)
def test_the_photon_rate_times_the_mean_square_is_b3s_shipped_variance(
    state: np.ndarray, tolerance: float
) -> None:
    r"""``n_gamma <u^2> ==`` :func:`photon_energy_variance` to 1e-13, same route.

    The identity the equilibrium rests on. ``kappa`` is not returned by the kick, so it
    is recovered from the critical energy the kick itself used
    (``u_c = (3/2) hbar c gamma^3 kappa``) and handed to B3's shipped variance helper —
    which means a wrong ``kappa`` cannot cancel between the two sides.

    A model that got ``<u>`` right and ``<u^2>`` wrong would pass every one of B2's
    gates and every mean-loss gate here; this is the only place it dies.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    bend = Dipole(L_BEND, 2.0 * math.pi / (2 * COLD["cells"]))
    u_c, rate = _probe(bend, state, ref)

    after = bend.track(state.copy(), ref)
    classical = float(_radiated(after[DELTA], radiation_kick(bend, state, after, ref)[DELTA], ref))
    momentum = ref.momentum_eV * (1.0 + after[DELTA])
    energy = math.sqrt(momentum**2 + ref.mass_eV**2)
    gamma = energy / ref.mass_eV
    kappa = u_c / (1.5 * HBAR_C_EV_M * gamma**3)

    assert rate * u_c**2 * photon_spectrum_moment(2) == pytest.approx(
        float(photon_energy_variance(classical, energy, kappa, ref)), rel=tolerance
    )


def test_the_two_dimensional_numbers_are_the_textbook_ones_on_the_design_orbit() -> None:
    r"""On axis the kick's own ``u_c`` and ``n_gamma`` are computable by hand, and are.

    Gate 1 is a *consistency* statement: it would survive if both routes shared one
    wrong curvature. This is the absolute one — on the design orbit ``kappa`` is the
    bend's ``angle/length`` and the path is its length, so ``u_c = (3/2) hbar c gamma^3
    kappa`` and ``n_gamma = (5/2 sqrt3) alpha gamma theta`` are numbers, and the shipped
    kick has to produce them. 16.2 photons per magnet at ``u_c/E = 8.7e-6``.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    theta = 2.0 * math.pi / (2 * COLD["cells"])
    bend = Dipole(L_BEND, theta)
    u_c, rate = _probe(bend, np.zeros(6), ref)

    gamma, energy = ref.gamma0, ref.total_energy_eV
    assert u_c == pytest.approx(1.5 * HBAR_C_EV_M * gamma**3 * theta / L_BEND, rel=1e-13)
    assert rate == pytest.approx(
        2.5 / math.sqrt(3.0) * fine_structure_constant(ref) * gamma * theta, rel=1e-13
    )
    assert u_c / energy == pytest.approx(8.71e-6, rel=1e-2)
    assert rate == pytest.approx(16.19, rel=1e-2)
    # and the shipped helpers agree with what the kick did
    assert critical_photon_energy(energy, theta / L_BEND, ref) == pytest.approx(u_c, rel=1e-13)
    assert photon_rate(energy, theta / L_BEND, L_BEND, ref) == pytest.approx(rate, rel=1e-13)


# ---------------------------------------------------------------------------
# Gate 2 — the whole map, deterministically, with no statistics anywhere.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("quantiles", [[0.5], [0.1, 0.9], [0.2, 0.4, 0.6, 0.8, 0.999999]])
def test_chosen_photons_give_a_closed_form_state_out(quantiles: list[float]) -> None:
    r"""``u = u_c sum_i x(q_i)``, then the on-shell factor, then one common scaling.

    The sharp gate for the map. Each photon energy is a closed form (part 1 gates the
    inverse against the spectrum's own quadrature), ``u_c`` is the textbook number on the
    design orbit, and everything after that is arithmetic — so the expected output state
    is written out in full and compared to the shipped one. The last case includes a
    ``q = 1 - 1e-6`` photon at ``9.7 u_c``, i.e. a genuine tail event placed on demand.

    It also gates the *linearity*: the loss is the sum of the drawn energies, so a model
    that (say) averaged them instead of adding them would agree at one photon and fail
    at two.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    theta = 2.0 * math.pi / (2 * COLD["cells"])
    bend = Dipole(L_BEND, theta)
    state = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4, 0.0, 0.0])
    after = bend.track(state.copy(), ref)

    rng = _ChosenPhotons(len(quantiles), quantiles)
    out = radiation_kick(bend, state, after, ref, model="photons", rng=rng)

    u_c = 1.5 * HBAR_C_EV_M * ref.gamma0**3 * theta / L_BEND
    loss = u_c * sum(float(photon_energy_quantile(q)) for q in quantiles)
    momentum = ref.momentum_eV * (1.0 + after[DELTA])
    energy = math.sqrt(momentum**2 + ref.mass_eV**2)
    factor = math.sqrt(1.0 - loss * (2.0 * energy - loss) / (energy**2 - ref.mass_eV**2))

    assert out[DELTA] == pytest.approx(factor * (1.0 + after[DELTA]) - 1.0, rel=1e-11)
    assert out[1] == pytest.approx(after[1] * factor, rel=1e-11)
    assert out[3] == pytest.approx(after[3] * factor, rel=1e-11)
    assert out[0] == after[0] and out[2] == after[2] and out[4] == after[4]


def test_no_photons_means_no_loss_at_all() -> None:
    """A traversal that emits nothing leaves the state exactly as the element mapped it.

    Not a rounding statement — bit-for-bit. It is the degenerate case a real ring hits
    constantly (``Poisson(16)`` returns zero once in ten million, but ``Poisson(0.01)``
    at a weak corrector returns it almost always), and an implementation that applied an
    on-shell factor of ``sqrt(1 - 0)`` would differ in the last bit.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    bend = Dipole(L_BEND, 2.0 * math.pi / (2 * COLD["cells"]))
    state = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4, 1.0e-3, 2.0e-4])
    after = bend.track(state.copy(), ref)
    out = radiation_kick(bend, state, after, ref, model="photons", rng=_ChosenPhotons(0, 0.5))
    # delta round-trips through f*(1+delta)-1 with f exactly 1.0, which is a one-ulp
    # identity rather than a bitwise one; px, py and the three positions are untouched.
    assert out[DELTA] == pytest.approx(after[DELTA], rel=1e-12)
    assert np.array_equal(out[[0, 1, 2, 3, 4]], after[[0, 1, 2, 3, 4]])


# ---------------------------------------------------------------------------
# Gate 3-4 — what must be different: the sign of the tail, and the shape.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def one_magnet_losses() -> dict[str, np.ndarray]:
    """200000 traversals of one bend under each model, as radiated energies [eV]."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    bend = Dipole(L_BEND, 2.0 * math.pi / (2 * COLD["cells"]))
    state = np.zeros((6, 200_000))
    after = bend.track(state.copy(), ref)
    out = {
        model: _radiated(
            after[DELTA],
            radiation_kick(bend, state, after, ref, model=model, rng=np.random.default_rng(4))[
                DELTA
            ],
            ref,
        )
        for model in ("mean", "quantum", "photons")
    }
    out["ref"] = ref  # type: ignore[assignment]
    return out


def test_the_gaussian_hands_energy_back_and_the_photon_sum_never_does(
    one_magnet_losses: dict[str, np.ndarray],
) -> None:
    r"""2.6% of Gaussian draws are an energy *gain*; exactly 0 of the photon draws are.

    B3 kept its Gaussian unclamped on purpose — clamping would bias both moments by the
    percent the equilibrium is gated to — and recorded the negative losses as the price.
    This is that price being paid off. The photon sum is a sum of non-negative energies,
    so the statement is not "small" but "impossible", and it is asserted as the pair so
    that the contrast is what fails if either model changes.
    """
    assert float(np.mean(one_magnet_losses["quantum"] < 0.0)) == pytest.approx(0.026, abs=0.004)
    assert float(np.min(one_magnet_losses["photons"])) >= 0.0
    assert np.count_nonzero(one_magnet_losses["photons"] < 0.0) == 0


def test_the_two_models_agree_on_the_mean_and_the_spread_and_only_on_those(
    one_magnet_losses: dict[str, np.ndarray],
) -> None:
    r"""Same ``U``, same ``sigma_U``, to the sampling floor — the identities, sampled.

    Gate 1 proved these as algebra; this is the same statement arriving through 200000
    actual draws, which is what catches a sampler that is correct on paper and wired up
    wrongly. Budgets are the standard errors of a mean and of a standard deviation at
    this ``N``, not tuned numbers: ``sigma/sqrt(N)`` is 0.11% of the mean here.
    """
    mean_loss = float(one_magnet_losses["mean"][0])
    for model in ("quantum", "photons"):
        sample = one_magnet_losses[model]
        budget = float(sample.std()) / math.sqrt(sample.size)
        assert float(sample.mean()) == pytest.approx(mean_loss, abs=4.0 * budget)
    spread = one_magnet_losses["quantum"].std() / one_magnet_losses["photons"].std()
    assert spread == pytest.approx(1.0, abs=4.0 / math.sqrt(2.0 * 200_000))


def test_the_photon_sums_skewness_counts_the_photons_and_the_gaussians_does_not(
    one_magnet_losses: dict[str, np.ndarray],
) -> None:
    r"""``skew = <u^3> / (sqrt(n_gamma) <u^2>^(3/2))``, inverted back to ``n_gamma``.

    The signature that separates the two models in *shape* rather than in scale. B3's
    reference arm used exactly this identity in reverse, to count xtrack's photons from
    a distribution that never reported a count; run on accsim's own photon sum it has to
    return the rate :func:`photon_rate` was asked for. The Gaussian's skewness is zero
    by construction and is asserted to be, at the sampling floor ``sqrt(6/N)``.
    """
    floor = math.sqrt(6.0 / 200_000)  # standard error of a sample skewness

    def skew(sample: np.ndarray) -> float:
        return float(np.mean((sample - sample.mean()) ** 3) / sample.std() ** 3)

    assert abs(skew(one_magnet_losses["quantum"])) < 4.0 * floor
    ours = skew(one_magnet_losses["photons"])
    assert ours > 0.8  # positive: a few hard photons make a long tail towards more loss

    second, third = photon_spectrum_moment(2), photon_spectrum_moment(3)
    implied = (third / (ours * second**1.5)) ** 2
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    theta = 2.0 * math.pi / (2 * COLD["cells"])
    assert implied == pytest.approx(
        float(photon_rate(ref.total_energy_eV, theta / L_BEND, L_BEND, ref)), rel=0.03
    )


def test_the_skew_in_delta_is_the_negative_of_it_and_lands_on_xtracks_number() -> None:
    r"""``skew(delta) = -0.92``, against xtrack's measured ``-0.91`` and B3's ``0.00``.

    A loss distribution with a long tail towards *more* loss is a ``delta`` distribution
    with a long tail towards *less* momentum, so the sign flips. It is worth pinning in
    ``delta`` rather than only in the loss, because ``delta`` is the coordinate the
    reference arm compares and the number it reported for xtrack — the only genuine
    photon sum this project can measure from the outside — is ``-0.91``.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    bend = Dipole(L_BEND, 2.0 * math.pi / (2 * COLD["cells"]))
    state = np.zeros((6, 200_000))
    after = bend.track(state.copy(), ref)
    out = radiation_kick(bend, state, after, ref, model="photons", rng=np.random.default_rng(9))
    d = out[DELTA]
    assert float(np.mean((d - d.mean()) ** 3) / d.std() ** 3) == pytest.approx(-0.92, abs=0.05)


# ---------------------------------------------------------------------------
# Gate 5 — the central-limit theorem, measured. The reason B4's lifetime cannot move.
# ---------------------------------------------------------------------------
def test_the_skew_falls_as_one_over_the_root_of_the_number_of_magnets_crossed() -> None:
    r"""``skew(N magnets) = <u^3> / (sqrt(N n_gamma) <u^2>^(3/2))`` — a law, over 100x in N.

    B5's most exposed prediction is that switching B4's ring from ``"quantum"`` to
    ``"photons"`` leaves the lifetime alone, and this is that prediction's *argument*,
    gated directly rather than inferred from an expensive null result. Escape from the
    bucket is a many-step random walk; the individual photon spectrum survives that
    summation only through its first two moments, which gate 1 pins as identities, and
    everything that distinguishes the two models is suppressed as ``1/sqrt(N)``.

    Measured on a bare bend applied ``N`` times to the same state, so the only thing in
    play is the sum of losses — no dispersion, no cavity, no damping to weight one
    magnet against another. ``skew * sqrt(N)`` is then a constant, and it is the
    compound-Poisson prediction for a *single* traversal: 0.916 at ``n_gamma = 16.19``.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    bend = Dipole(L_BEND, 2.0 * math.pi / (2 * COLD["cells"]))
    n = 120_000
    state = np.zeros((6, n))
    after = bend.track(state.copy(), ref)
    rng = np.random.default_rng(23)

    second, third = photon_spectrum_moment(2), photon_spectrum_moment(3)
    rate = float(photon_rate(ref.total_energy_eV, 2.0 * math.pi / (2 * COLD["cells"]), L_BEND, ref))
    single = third / (math.sqrt(rate) * second**1.5)

    total = np.zeros(n)
    crossed = 0
    for target in (1, 10, 100):
        while crossed < target:
            out = radiation_kick(bend, state, after, ref, model="photons", rng=rng)
            total += _radiated(after[DELTA], out[DELTA], ref)
            crossed += 1
        skew = float(np.mean((total - total.mean()) ** 3) / total.std() ** 3)
        assert skew * math.sqrt(target) == pytest.approx(single, rel=0.06)
    assert single == pytest.approx(0.916, rel=0.01)


def test_one_turn_of_the_ring_is_already_nearly_gaussian_and_a_thousand_are_entirely() -> None:
    r"""One turn: skew ``-0.129``, kurtosis ``3.00``. B4 walks for a thousand of them.

    The same statement on the real ring, where it is not quite the clean ``1/sqrt(N)``
    above: a turn's 40 magnets do not contribute equally, because the loss each one
    takes is fed through the remaining ``R56`` into ``zeta`` and then through the cavity,
    so a turn is a *weighted* sum of its photons rather than a plain one. That weighting
    is why ``-0.129`` is 11% short of ``-0.92/sqrt(40) = -0.146``, and it is named here
    rather than left as a discrepancy.

    What matters for B4 is the size, not the last digit: the shape a photon spectrum has
    is already suppressed sevenfold after **one** turn, and the escape B4 measures takes
    a thousand. The kurtosis is Gaussian to the sampling floor after one turn already.
    """
    tracker = Tracker(_ring(**COLD))
    n = 60_000
    start = np.zeros((6, n))
    steps = {
        model: tracker.track_once(start.copy(), model, np.random.default_rng(17))[DELTA]
        for model in ("quantum", "photons")
    }
    skews = {
        model: float(np.mean((s - s.mean()) ** 3) / s.std() ** 3) for model, s in steps.items()
    }
    floor = math.sqrt(6.0 / n)
    assert abs(skews["quantum"]) < 4.0 * floor  # the Gaussian model has none, still
    assert skews["photons"] == pytest.approx(-0.129, abs=0.01)
    assert abs(skews["photons"]) < 0.2  # and it is already a seventh of one magnet's

    for sample in steps.values():
        centred = sample - sample.mean()
        assert float(np.mean(centred**4) / sample.std() ** 4) == pytest.approx(
            3.0, abs=4.0 * math.sqrt(24.0 / n)
        )

    # ...and the two moments that *do* survive the summation are the same ones
    budget = float(steps["quantum"].std()) / math.sqrt(n)
    assert float(steps["photons"].mean()) == pytest.approx(
        float(steps["quantum"].mean()), abs=4.0 * budget
    )
    assert float(steps["photons"].std()) / float(steps["quantum"].std()) == pytest.approx(
        1.0, abs=4.0 / math.sqrt(2.0 * n)
    )


# ---------------------------------------------------------------------------
# Gate 6 — the API, the degenerate cases, and the property radiation must not have.
# ---------------------------------------------------------------------------
def test_photons_is_advertised_stochastic_and_refuses_to_run_unseeded() -> None:
    """It joins the offered models, and it will not draw from a generator it invented."""
    assert "photons" in RADIATION_MODELS
    assert "photons" in STOCHASTIC_MODELS
    tracker = Tracker(_ring(**COLD))
    with pytest.raises(ValueError, match="needs an explicit rng"):
        tracker.track_once(np.zeros(6), "photons")
    with pytest.raises(ValueError, match="needs an explicit rng"):
        tracker.track_bunch(Bunch(np.zeros((6, 4))), nonlinear=True, radiation="photons")
    # and the kick refuses on its own, not only through the tracker
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    bend = Dipole(L_BEND, 0.1)
    with pytest.raises(ValueError, match="needs an explicit rng"):
        radiation_kick(bend, np.zeros(6), np.zeros(6), ref, model="photons")


def test_the_same_seed_reproduces_bit_for_bit_and_a_different_seed_does_not() -> None:
    """A stochastic track is an experiment, and has to be repeatable."""
    tracker = Tracker(_ring(**COLD))
    state = np.array([1e-4, 0.0, 1e-4, 0.0, 0.0, 0.0])
    first = tracker.track_once(state.copy(), "photons", np.random.default_rng(7))
    again = tracker.track_once(state.copy(), "photons", np.random.default_rng(7))
    other = tracker.track_once(state.copy(), "photons", np.random.default_rng(8))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_a_bunch_and_a_single_particle_agree_distributionally_and_not_draw_by_draw() -> None:
    r"""One particle at a time and a whole bunch consume the generator differently.

    A photon-resolved model draws a *random number* of uniforms per particle, so the
    bunch path cannot be the single-particle path repeated — the streams diverge after
    the first traversal. What must agree is the distribution, and the gate says exactly
    that, with the same budgets the identities carry.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    bend = Dipole(L_BEND, 2.0 * math.pi / (2 * COLD["cells"]))
    n = 20_000
    rng = np.random.default_rng(21)
    state = np.zeros((6, n))
    after = bend.track(state.copy(), ref)
    bunched = _radiated(
        after[DELTA],
        radiation_kick(bend, state, after, ref, model="photons", rng=rng)[DELTA],
        ref,
    )
    single = np.array(
        [
            float(
                _radiated(
                    after[DELTA][0],
                    radiation_kick(
                        bend, np.zeros(6), after[:, 0].copy(), ref, model="photons", rng=rng
                    )[DELTA],
                    ref,
                )
            )
            for _ in range(n)
        ]
    )
    assert single.mean() == pytest.approx(
        bunched.mean(), rel=4.0 * bunched.std() / bunched.mean() / math.sqrt(n)
    )
    assert single.std() / bunched.std() == pytest.approx(1.0, abs=4.0 / math.sqrt(2.0 * n))


def test_nothing_radiates_where_there_is_no_field_and_no_length() -> None:
    """Drifts, thin elements and a bend switched off emit exactly zero photons.

    A scope statement, not an approximation: a real short magnet radiates, and modelling
    it means giving it a length. Gated because the photon path has an extra way to get
    this wrong — ``Poisson(0)`` is fine, but a rate computed from a zero-length path
    could still draw from a spectrum whose critical energy is finite.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, COLD["energy"])
    state = np.array([1e-3, 2e-4, 1e-3, 2e-4, 0.0, 1e-4])
    for element in (Dipole(L_BEND, 0.0), ThinQuadrupole(0.4)):
        after = element.track(state.copy(), ref)
        out = radiation_kick(
            element, state, after, ref, model="photons", rng=np.random.default_rng(2)
        )
        assert np.array_equal(out, after)


def test_radiation_with_photons_is_still_not_symplectic() -> None:
    """Radiation is dissipative, and the suite asserts the rejection rather than ducking it."""
    tracker = Tracker(_ring(**COLD))
    rng = np.random.default_rng(6)
    assert not is_symplectic_map(
        lambda s: tracker.track_once(s, "photons", rng), np.zeros(6), atol=1e-9
    )
