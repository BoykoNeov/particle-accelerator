"""Analytic checks for Courant-Snyder Twiss: matching, propagation, tunes.

Expected values are closed-form (waist optics, synthetic CS rotations, linear
phase accumulation), never read back from the routine under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinQuadrupole,
    Twiss,
    UnstableLatticeError,
    closed_twiss,
    is_stable,
    match_periodic,
    propagate_twiss,
    tunes,
)


def _cs_block(beta: float, alpha: float, mu: float) -> np.ndarray:
    """Courant-Snyder one-turn 2x2 for given (beta, alpha, mu)."""
    gamma = (1.0 + alpha**2) / beta
    c, s = math.cos(mu), math.sin(mu)
    return np.array([[c + alpha * s, beta * s], [-gamma * s, c - alpha * s]])


def _embed(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Embed two transverse 2x2 blocks into a 6x6 identity map."""
    M = np.eye(6)
    M[np.ix_([0, 1], [0, 1])] = cx
    M[np.ix_([2, 3], [2, 3])] = cy
    return M


def _thin_fodo(f: float, ll: float) -> list:
    """Symmetric thin-lens FODO cell, split focusing quads at the ends."""
    return [
        ThinQuadrupole(0.5 / f, name="qf/2"),
        Drift(ll),
        ThinQuadrupole(-1.0 / f, name="qd"),
        Drift(ll),
        ThinQuadrupole(0.5 / f, name="qf/2"),
    ]


def test_match_periodic_recovers_courant_snyder() -> None:
    # A synthetic one-turn map built from known (beta, alpha, mu) must be
    # inverted exactly by match_periodic.
    bx, ax, mux = 7.3, -1.2, 1.1
    by, ay, muy = 3.1, 0.8, 2.4
    M = _embed(_cs_block(bx, ax, mux), _cs_block(by, ay, muy))
    tw = match_periodic(M)
    assert tw.beta_x == pytest.approx(bx)
    assert tw.alpha_x == pytest.approx(ax)
    assert tw.beta_y == pytest.approx(by)
    assert tw.alpha_y == pytest.approx(ay)
    # gamma is the derived invariant.
    assert tw.gamma_x == pytest.approx((1 + ax**2) / bx)


def test_matched_beta_is_positive_with_negative_m12() -> None:
    # mu in (pi, 2pi) makes sin mu < 0 so M12 < 0; beta must still come out > 0.
    bx, ax, mux = 5.0, 0.3, 4.0  # 4.0 rad > pi
    M = _embed(_cs_block(bx, ax, mux), _cs_block(2.0, 0.0, 1.0))
    tw = match_periodic(M)
    assert M[0, 1] < 0.0
    assert tw.beta_x > 0.0
    assert tw.beta_x == pytest.approx(bx)


def test_drift_waist_propagation(proton_gamma5: ReferenceParticle) -> None:
    # From a waist (alpha = 0), a drift gives beta(s) = beta* + s^2/beta*,
    # alpha(s) = -s/beta*, mu(s) = arctan(s/beta*).
    beta_star, ll = 4.0, 3.0
    tw0 = Twiss(0.0, beta_star, 0.0, 0.0, beta_star, 0.0, 0.0)
    end = propagate_twiss(Lattice([Drift(ll)], proton_gamma5), tw0)[-1]
    assert end.beta_x == pytest.approx(beta_star + ll**2 / beta_star)
    assert end.alpha_x == pytest.approx(-ll / beta_star)
    assert end.mu_x == pytest.approx(math.atan2(ll, beta_star))
    # Vertical plane is identical here (same start, same drift).
    assert end.beta_y == pytest.approx(end.beta_x)


def test_thin_lens_advances_no_phase(proton_gamma5: ReferenceParticle) -> None:
    # A thin quad has C12 = 0, so it advances no betatron phase by itself.
    tw0 = Twiss(0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 0.0)
    end = propagate_twiss(Lattice([ThinQuadrupole(0.3)], proton_gamma5), tw0)[-1]
    assert end.mu_x == pytest.approx(0.0)
    assert end.mu_y == pytest.approx(0.0)


def test_twiss_invariant_preserved_through_lattice(proton_gamma5: ReferenceParticle) -> None:
    # gamma*beta - alpha^2 = 1 must hold at every boundary (symplectic propagation).
    lat = Lattice(_thin_fodo(2.0, 1.0) + [Quadrupole(0.4, 1.1), Drift(0.7)], proton_gamma5)
    tw0 = Twiss(0.0, 5.0, 0.5, 0.0, 3.0, -0.3, 0.0)
    for tw in propagate_twiss(lat, tw0):
        assert tw.gamma_x * tw.beta_x - tw.alpha_x**2 == pytest.approx(1.0)
        assert tw.gamma_y * tw.beta_y - tw.alpha_y**2 == pytest.approx(1.0)


def test_tune_accumulates_past_one(proton_gamma5: ReferenceParticle) -> None:
    # The central correctness check for continuous phase accumulation: N identical
    # cells must have N times the single-cell tune, even when that exceeds 1 — a
    # naive acos of the one-turn matrix would alias it back into (0, 0.5).
    cell = _thin_fodo(2.0, 1.0)
    one = Lattice(cell, proton_gamma5)
    assert is_stable(one.one_turn_matrix())
    qx1, qy1 = tunes(one)

    assert qx1 < 0.5  # each cell aliases into acos's (0, 0.5) range
    n = 16
    many = Lattice(cell * n, proton_gamma5)
    qxn, qyn = tunes(many)
    assert qxn == pytest.approx(n * qx1)
    assert qyn == pytest.approx(n * qy1)
    assert qxn > 1.0  # total genuinely past the aliasing point


def test_matched_beta_is_cell_periodic(proton_gamma5: ReferenceParticle) -> None:
    # Matching one cell vs. the N-cell ring gives the same beta at the start
    # (periodicity), and propagating a matched cell returns to itself.
    cell = _thin_fodo(2.0, 1.0)
    one = Lattice(cell, proton_gamma5)
    tw0 = closed_twiss(one)
    end = propagate_twiss(one, tw0)[-1]
    assert end.beta_x == pytest.approx(tw0.beta_x)
    assert end.alpha_x == pytest.approx(tw0.alpha_x)
    assert end.beta_y == pytest.approx(tw0.beta_y)
    assert end.alpha_y == pytest.approx(tw0.alpha_y)


def test_unstable_lattice_raises(proton_gamma5: ReferenceParticle) -> None:
    # Two strong focusing thin lenses with a short drift: over-focused, |Tr| > 2.
    lat = Lattice([ThinQuadrupole(5.0), Drift(0.2), ThinQuadrupole(5.0)], proton_gamma5)
    assert not is_stable(lat.one_turn_matrix())
    with pytest.raises(UnstableLatticeError):
        closed_twiss(lat)


def test_is_stable_true_for_fodo(proton_gamma5: ReferenceParticle) -> None:
    assert is_stable(Lattice(_thin_fodo(2.0, 1.0), proton_gamma5).one_turn_matrix())
