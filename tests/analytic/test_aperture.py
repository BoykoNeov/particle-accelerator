"""Analytic checks for the Aperture element (Stage 4 — geometric predicate).

These pin the *geometry* of the survival predicate with hand-placed particles,
deliberately kept off the knife-edge so the (inclusive ``≤``) boundary
convention never decides a test. The aperture must also be optics-transparent:
its 6x6 is the identity, so inserting one perturbs no linear optics. Loss
*accounting* (loss location, transmission fraction) is exercised separately once
the loss-aware tracker lands.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    DELTA,
    ZETA,
    AcceptanceElement,
    Aperture,
    Bunch,
    Collimator,
    Drift,
    Lattice,
    MomentumAperture,
    Particle,
    ReferenceParticle,
    Tracker,
    X,
    Y,
)


def _state(x: float = 0.0, y: float = 0.0) -> np.ndarray:
    p = Particle(x=x, y=y)
    return p.state


# --- optics-transparent: the 6x6 is the identity regardless of shape/size ------
@pytest.mark.parametrize(
    "ap",
    [
        Aperture("circular", 1e-2),
        Aperture("elliptical", 1e-2, 2e-2),
        Aperture("rectangular", 1e-2, 2e-2),
        Collimator("circular", 5e-3, length=0.1),
    ],
)
def test_aperture_matrix_is_identity(ap: Aperture, proton_gamma5: ReferenceParticle) -> None:
    np.testing.assert_array_equal(ap.matrix(proton_gamma5), np.eye(6))


def test_aperture_does_not_perturb_optics(proton_gamma5: ReferenceParticle) -> None:
    # A drift with an aperture spliced in has the same transfer matrix as the bare drift.
    bare = Lattice([Drift(1.0)], proton_gamma5).transfer_matrix()
    with_ap = Lattice(
        [Drift(0.4), Aperture("elliptical", 1e-2, 3e-2), Drift(0.6)], proton_gamma5
    ).transfer_matrix()
    np.testing.assert_allclose(with_ap, bare, rtol=1e-14, atol=1e-16)


# --- circular predicate: survives iff x^2 + y^2 <= R^2 -------------------------
def test_circular_predicate() -> None:
    R = 1.0e-2
    ap = Aperture("circular", R)
    # Well inside on each axis and diagonally.
    assert ap.survives(_state(x=0.9 * R))
    assert ap.survives(_state(y=0.9 * R))
    assert ap.survives(_state(x=0.6 * R, y=0.6 * R))  # r = 0.85 R < R
    # Outside: just past the radius on axis, and a diagonal point with r > R.
    assert not ap.survives(_state(x=1.1 * R))
    assert not ap.survives(_state(x=0.8 * R, y=0.8 * R))  # r = 1.13 R > R


# --- elliptical predicate: the axes are independent ---------------------------
def test_elliptical_predicate() -> None:
    ax, ay = 1.0e-2, 3.0e-2
    ap = Aperture("elliptical", ax, ay)
    # A point outside the x-radius but inside the (larger) y-radius is still lost:
    # it is inside the *circle* of radius ay but outside the *ellipse*.
    assert not ap.survives(_state(x=1.2 * ax, y=0.0))
    assert ap.survives(_state(x=0.0, y=0.9 * ay))
    # (x/ax)^2 + (y/ay)^2 = 0.5^2 + 0.5^2 = 0.5 <= 1 -> survives.
    assert ap.survives(_state(x=0.5 * ax, y=0.5 * ay))
    # 0.9^2 + 0.9^2 = 1.62 > 1 -> lost.
    assert not ap.survives(_state(x=0.9 * ax, y=0.9 * ay))


# --- rectangular predicate: a corner outside the circle can still survive ------
def test_rectangular_predicate() -> None:
    ax, ay = 1.0e-2, 2.0e-2
    ap = Aperture("rectangular", ax, ay)
    assert ap.survives(_state(x=0.99 * ax, y=0.99 * ay))  # near the corner, inside
    assert not ap.survives(_state(x=1.01 * ax, y=0.0))  # past x half-width
    assert not ap.survives(_state(x=0.0, y=1.01 * ay))  # past y half-width
    # Sign-symmetric: all four quadrants behave identically.
    assert ap.survives(_state(x=-0.5 * ax, y=-0.5 * ay))
    assert not ap.survives(_state(x=-1.5 * ax, y=0.5 * ay))


# --- vectorised over a bunch: (6, N) -> (N,) bool ------------------------------
def test_survives_vectorised() -> None:
    R = 1.0e-2
    ap = Aperture("circular", R)
    states = np.zeros((6, 4))
    states[X] = [0.0, 0.5 * R, 1.5 * R, 0.0]
    states[Y] = [0.0, 0.0, 0.0, 2.0 * R]
    mask = ap.survives(states)
    assert mask.shape == (4,)
    np.testing.assert_array_equal(mask, [True, True, False, False])


# --- construction guards ------------------------------------------------------
def test_construction_guards() -> None:
    with pytest.raises(ValueError):
        Aperture("triangular", 1e-2)  # unknown shape
    with pytest.raises(ValueError):
        Aperture("circular", -1e-2)  # non-positive half-width
    with pytest.raises(ValueError):
        Aperture("elliptical", 1e-2)  # elliptical needs half_y
    with pytest.raises(ValueError):
        Aperture("circular", 1e-2, 2e-2)  # circular takes a single radius


def test_collimator_has_length() -> None:
    c = Collimator("rectangular", 1e-2, 2e-2, length=0.25)
    assert c.length == 0.25
    assert Collimator("circular", 5e-3).length > 0.0  # default jaw length is finite


# --- MomentumAperture (B4) — the longitudinal acceptance ----------------------
#
# Same discipline as above: hand-placed particles off the knife-edge, and the
# element must be optics-transparent. The one thing this class has that Aperture
# does not is ``center``, and the tests that matter are the ones that would pass
# for a class that silently ignored it.
def test_momentum_aperture_matrix_is_identity(proton_gamma5: ReferenceParticle) -> None:
    assert np.allclose(MomentumAperture(1.0e-3).matrix(proton_gamma5), np.eye(6))


def test_momentum_aperture_does_not_perturb_optics(proton_gamma5: ReferenceParticle) -> None:
    """Inserting one leaves the one-turn map byte-identical — it is a predicate."""
    bare = Lattice([Drift(1.0), Drift(1.0)], ref=proton_gamma5)
    with_cut = Lattice(
        [Drift(1.0), MomentumAperture(1.0e-3, center=5.0e-4), Drift(1.0)], ref=proton_gamma5
    )
    assert np.array_equal(bare.one_turn_matrix(), with_cut.one_turn_matrix())
    assert with_cut.length == bare.length  # thin by default


def test_momentum_predicate_is_a_window_on_delta_alone() -> None:
    """Only ``delta`` is consulted: a huge ``x``/``y``/``zeta`` does not kill a particle."""
    cut = MomentumAperture(2.0e-3)
    inside = np.zeros(6)
    inside[X], inside[Y], inside[ZETA] = 10.0, 10.0, 10.0
    inside[DELTA] = 1.0e-3
    assert bool(cut.survives(inside))
    outside = inside.copy()
    outside[DELTA] = 3.0e-3
    assert not bool(cut.survives(outside))


def test_the_window_is_centred_on_center_and_not_on_zero() -> None:
    """``|delta − center| ≤ half_delta`` — asymmetric about zero when centred.

    The gate a class that ignored ``center`` would fail: with the window shifted to
    ``[+1, +9] × 1e-3``, ``delta = 0`` is *lost* and ``delta = 9e-3`` survives, which
    is the opposite of what an uncentred cut says about both.
    """
    cut = MomentumAperture(4.0e-3, center=5.0e-3)
    states = np.zeros((6, 6))
    states[DELTA] = [0.0, 0.9e-3, 1.1e-3, 5.0e-3, 8.9e-3, 9.1e-3]
    assert list(cut.survives(states)) == [False, False, True, True, True, False]
    # a negative centre is the mirror image, exactly
    mirrored = MomentumAperture(4.0e-3, center=-5.0e-3)
    assert list(mirrored.survives(-states)) == list(cut.survives(states))


def test_momentum_survives_is_scalar_for_one_and_vector_for_many() -> None:
    cut = MomentumAperture(1.0e-3)
    one = np.zeros(6)
    one[DELTA] = 5.0e-4
    assert cut.survives(one).shape == ()
    many = np.zeros((6, 3))
    many[DELTA] = [0.0, 5.0e-4, 5.0e-3]
    got = cut.survives(many)
    assert got.shape == (3,)
    assert list(got) == [True, True, False]


def test_momentum_boundary_is_inclusive_like_the_geometric_one() -> None:
    """On the boundary survives, matching :class:`Aperture` and xtrack's limits."""
    cut = MomentumAperture(2.0e-3, center=1.0e-3)
    on = np.zeros(6)
    on[DELTA] = 3.0e-3  # exactly center + half_delta
    assert bool(cut.survives(on))
    on[DELTA] = -1.0e-3  # exactly center - half_delta
    assert bool(cut.survives(on))


def test_momentum_construction_guards() -> None:
    for bad in (0.0, -1.0e-3):
        with pytest.raises(ValueError, match="half_delta must be > 0"):
            MomentumAperture(bad)
    assert MomentumAperture(1.0e-3, center=-2.0, length=0.5).length == 0.5  # centre is free
    assert "center=-2.0" in repr(MomentumAperture(1.0e-3, center=-2.0))


def test_both_kinds_are_acceptance_elements_and_nothing_else_is() -> None:
    """The loss pass dispatches on the base class, so membership is the contract."""
    assert isinstance(MomentumAperture(1.0e-3), AcceptanceElement)
    assert isinstance(Aperture("circular", 1.0e-2), AcceptanceElement)
    assert isinstance(Collimator("rectangular", 1.0e-2, 1.0e-2), AcceptanceElement)
    assert not isinstance(Drift(1.0), AcceptanceElement)


def test_the_loss_pass_consults_a_momentum_aperture(proton_gamma5: ReferenceParticle) -> None:
    """End to end: an off-momentum particle is recorded lost at the cut's own ``s``.

    Both kinds in one lattice, each killing a different particle, so a pass that
    had kept dispatching on :class:`Aperture` alone would leave particle 1 alive.
    """
    lattice = Lattice(
        [
            Drift(1.0),
            Aperture("circular", 5.0e-3),
            Drift(2.0),
            MomentumAperture(1.0e-3, center=0.0),
            Drift(1.0),
        ],
        ref=proton_gamma5,
    )
    states = np.zeros((6, 3))
    states[X] = [0.0, 0.0, 1.0e-2]  # particle 2 is outside the geometric aperture
    states[DELTA] = [0.0, 5.0e-3, 0.0]  # particle 1 is outside the momentum one
    result = Tracker(lattice).track_bunch_losses(Bunch(states), n_turns=1)

    assert list(result.alive) == [True, False, False]
    assert result.loss_element[1] == 3  # the MomentumAperture
    assert result.loss_element[2] == 1  # the Aperture
    assert result.loss_s[1] == pytest.approx(3.0)  # 1 m + 2 m of drift
    assert result.loss_s[2] == pytest.approx(1.0)
    assert list(result.loss_turn[[1, 2]]) == [0, 0]
