"""Stage 4 acceptance: aperture loss accounting (location, transmission, loss map).

Three gates, per the roadmap:

1. A lost particle is flagged at the **correct longitudinal location** — a
   hand-placed particle that clears the first aperture and dies at the second;
   the recorded ``loss_s`` is the second aperture's geometric ``s`` (not the
   particle's ``zeta``).
2. Transmission through a known aperture matches a **hand calculation** — a round
   Gaussian bunch through a circular aperture has survival ``1 − exp(−R²/2σ²)``
   (Rayleigh radial CDF, derived symbolically below); the empirical survival is
   compared with a *binomial* tolerance, not a tuned number.
3. The **loss map** reproduces a simple analytic case — a hand-placed bunch split
   across two apertures of different radius gives the expected per-location counts.

The transmission formula holds **only** for a round beam (σ_x = σ_y) through a
*circular* aperture. An independent, equally-clean case with a different shape is
the separable rectangular one: ``T = erf(a_x/√2σ_x)·erf(a_y/√2σ_y)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Aperture,
    Bunch,
    Drift,
    Lattice,
    ReferenceParticle,
    Tracker,
    X,
    Y,
)


# =========================================================================
# Gate 1 — a lost particle is flagged at the correct longitudinal location
# =========================================================================
def test_loss_flagged_at_correct_location(proton_gamma5: ReferenceParticle) -> None:
    R = 1.0e-2
    # Two same-size circular apertures, 1 m apart, each preceded by a 1 m drift.
    # Element indices:  0:Drift  1:ApA(s=1)  2:Drift  3:ApB(s=2).
    lattice = Lattice(
        [Drift(1.0), Aperture("circular", R), Drift(1.0), Aperture("circular", R)],
        proton_gamma5,
    )
    # x grows as x0 + s*px. With px = 0.6 R / m: at ApA (s=1) x = 0.6 R (survives);
    # at ApB (s=2) x = 1.2 R (lost). Off the knife-edge either way.
    px = 0.6 * R
    states = np.zeros((6, 1))
    states[1, 0] = px  # PX
    result = Tracker(lattice).track_bunch_losses(Bunch(states), n_turns=1)

    assert result.n_lost == 1
    assert result.loss_turn[0] == 0
    assert result.loss_element[0] == 3  # the SECOND aperture
    assert result.loss_s[0] == pytest.approx(2.0)  # geometric s of ApB, not zeta
    assert not result.alive[0]


def test_multi_turn_loss_turn_recorded(proton_gamma5: ReferenceParticle) -> None:
    # One aperture after a 1 m drift; a particle with a small angle grows past it
    # only after several turns. Confirms loss_turn (and skip-after-loss) work.
    R = 1.0e-2
    lattice = Lattice([Drift(1.0), Aperture("circular", R)], proton_gamma5)
    # x after the k-th turn's drift = 0.3 k R (px = 0.3 R / m): 0.9 R at turn 2
    # (survives), 1.2 R at turn 3 (lost) — clean margins, no knife-edge.
    px = 0.3 * R
    states = np.zeros((6, 1))
    states[1, 0] = px
    result = Tracker(lattice).track_bunch_losses(Bunch(states), n_turns=20)

    assert result.loss_turn[0] == 3
    assert result.loss_s[0] == pytest.approx(1.0)


# =========================================================================
# Gate 2 — transmission through a known aperture matches a hand calculation
# =========================================================================
def test_circular_transmission_formula_symbolic() -> None:
    """Derive the round-beam circular survival ``1 − exp(−R²/2σ²)`` in sympy."""
    sp = pytest.importorskip("sympy")
    r, R, sigma = sp.symbols("r R sigma", positive=True)
    # Radial density of a round 2D Gaussian: (r/sigma^2) exp(-r^2 / 2 sigma^2).
    radial_pdf = (r / sigma**2) * sp.exp(-(r**2) / (2 * sigma**2))
    survival = sp.integrate(radial_pdf, (r, 0, R))
    expected = 1 - sp.exp(-(R**2) / (2 * sigma**2))
    assert sp.simplify(survival - expected) == 0


def test_circular_transmission_empirical(proton_gamma5: ReferenceParticle) -> None:
    rng = np.random.default_rng(20260702)
    n = 200_000
    sigma = 1.0e-3
    R = 1.5 * sigma  # T = 1 - exp(-1.125) ≈ 0.675
    analytic = 1.0 - math.exp(-(R**2) / (2 * sigma**2))

    states = np.zeros((6, n))
    states[X] = rng.normal(0.0, sigma, n)
    states[Y] = rng.normal(0.0, sigma, n)
    lattice = Lattice([Aperture("circular", R)], proton_gamma5)
    result = Tracker(lattice).track_bunch_losses(Bunch(states), n_turns=1)

    # Binomial standard error of the survival fraction; allow 5 sigma.
    se = math.sqrt(analytic * (1.0 - analytic) / n)
    assert result.transmission == pytest.approx(analytic, abs=5.0 * se)


def test_rectangular_transmission_separable(proton_gamma5: ReferenceParticle) -> None:
    """Independent shape: rectangular acceptance is separable in x and y."""
    rng = np.random.default_rng(19730401)
    n = 200_000
    sx, sy = 1.0e-3, 2.0e-3
    ax, ay = 1.2e-3, 3.0e-3
    # P(|x|<=ax) P(|y|<=ay) = erf(ax/sqrt2 sx) erf(ay/sqrt2 sy).
    analytic = math.erf(ax / (math.sqrt(2) * sx)) * math.erf(ay / (math.sqrt(2) * sy))

    states = np.zeros((6, n))
    states[X] = rng.normal(0.0, sx, n)
    states[Y] = rng.normal(0.0, sy, n)
    lattice = Lattice([Aperture("rectangular", ax, ay)], proton_gamma5)
    result = Tracker(lattice).track_bunch_losses(Bunch(states), n_turns=1)

    se = math.sqrt(analytic * (1.0 - analytic) / n)
    assert result.transmission == pytest.approx(analytic, abs=5.0 * se)


# =========================================================================
# Gate 3 — the loss map reproduces a simple analytic case
# =========================================================================
def test_loss_map_two_apertures(proton_gamma5: ReferenceParticle) -> None:
    Ra, Rb = 2.0e-2, 1.0e-2  # first aperture wider than the second
    lattice = Lattice(
        [Drift(1.0), Aperture("circular", Ra), Drift(1.0), Aperture("circular", Rb)],
        proton_gamma5,
    )
    # Static particles (px = 0): x is constant, so each dies at the first aperture
    # whose radius it exceeds.  3 fail at ApA (s=1), 2 fail at ApB (s=2), 4 survive.
    xs = [3.0e-2] * 3 + [1.5e-2] * 2 + [0.5e-2] * 4
    states = np.zeros((6, len(xs)))
    states[X] = xs
    result = Tracker(lattice).track_bunch_losses(Bunch(states), n_turns=1)

    assert result.n_lost == 5
    assert result.n_survived == 4
    assert result.transmission == pytest.approx(4 / 9)

    s_locs, counts = result.loss_map()
    np.testing.assert_allclose(s_locs, [1.0, 2.0])
    np.testing.assert_array_equal(counts, [3, 2])


def test_survivors_have_no_loss_record(proton_gamma5: ReferenceParticle) -> None:
    lattice = Lattice([Aperture("circular", 1.0e-2)], proton_gamma5)
    states = np.zeros((6, 3))  # all on-axis -> all survive
    result = Tracker(lattice).track_bunch_losses(Bunch(states), n_turns=5)
    assert result.n_survived == 3
    np.testing.assert_array_equal(result.loss_turn, [-1, -1, -1])
    assert np.all(np.isnan(result.loss_s))
    s_locs, counts = result.loss_map()
    assert s_locs.size == 0 and counts.size == 0
