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
    Aperture,
    Collimator,
    Drift,
    Lattice,
    Particle,
    ReferenceParticle,
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
