"""Analytic checks for the Aperture element (Stage 4 — geometric predicate).

These pin the *geometry* of the survival predicate with hand-placed particles,
deliberately kept off the knife-edge so the (inclusive ``≤``) boundary
convention never decides a test. A thin aperture must also be optics-transparent:
its 6x6 is the identity, so inserting one perturbs no linear optics. One with a
length is a **drift** of that length — the jaws are a gap in a block of metal, and
the beam crosses that gap field-free — which is what MAD-X (``COLLIMATOR``,
``RCOLLIMATOR``, ``ECOLLIMATOR``) and xtrack's MAD-X loader both do. Loss
*accounting* (loss location, transmission fraction) is exercised separately once
the loss-aware tracker lands; the two-face check a thick boundary needs is gated at
the end of this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    DELTA,
    PX,
    PY,
    ZETA,
    AcceptanceElement,
    Aperture,
    Bunch,
    Collimator,
    Drift,
    Lattice,
    MomentumAperture,
    Particle,
    Quadrupole,
    ReferenceParticle,
    Tracker,
    X,
    Y,
)


def _state(x: float = 0.0, y: float = 0.0) -> np.ndarray:
    p = Particle(x=x, y=y)
    return p.state


# --- optics-transparent when thin: the 6x6 is the identity regardless of shape --
@pytest.mark.parametrize(
    "ap",
    [
        Aperture("circular", 1e-2),
        Aperture("elliptical", 1e-2, 2e-2),
        Aperture("rectangular", 1e-2, 2e-2),
        MomentumAperture(1.0e-3),
    ],
)
def test_thin_acceptance_matrix_is_identity(
    ap: AcceptanceElement, proton_gamma5: ReferenceParticle
) -> None:
    np.testing.assert_array_equal(ap.matrix(proton_gamma5), np.eye(6))


def _off_orbit_bunch() -> np.ndarray:
    """A bunch with angles and momentum spread — where the exact drift's ``1/pz`` acts."""
    states = np.zeros((6, 4))
    states[X] = [0.0, 1.0e-3, -2.0e-3, 5.0e-4]
    states[PX] = [0.0, 2.0e-3, -1.5e-3, 4.0e-3]
    states[Y] = [0.0, -1.0e-3, 3.0e-4, 1.0e-3]
    states[PY] = [0.0, 1.0e-3, 2.5e-3, -3.0e-3]
    states[ZETA] = [0.0, 1.0e-2, -2.0e-2, 0.0]
    states[DELTA] = [0.0, 1.0e-3, -2.0e-3, 5.0e-3]
    return states


# --- with a length it is a drift of that length --------------------------------
#
# MAD-X's three collimators give a transfer matrix equal to a DRIFT's to the last
# digit, and xtrack's MAD-X loader converts all three with ``convert_drift_like``
# (checked 2026-10-08; ``tests/reference/test_collimator_madx.py`` re-measures the
# MAD-X half). Before this, accsim returned the identity while ``s`` and the
# circumference still counted the length: the ring was L longer than its own map.
_THICK = [
    Collimator("circular", 5e-3, length=0.1),
    Collimator("rectangular", 1e-2, 2e-2, length=1.3),
    Aperture("elliptical", 1e-2, 2e-2, length=0.7),
    MomentumAperture(1.0e-3, center=2.0e-4, length=0.5),
]


@pytest.mark.parametrize("ap", _THICK)
def test_thick_acceptance_matrix_is_a_drift(
    ap: AcceptanceElement, proton_gamma5: ReferenceParticle
) -> None:
    drift = Drift(ap.length).matrix(proton_gamma5)
    np.testing.assert_array_equal(ap.matrix(proton_gamma5), drift)


@pytest.mark.parametrize("ap", _THICK)
def test_thick_acceptance_tracks_as_the_exact_drift(
    ap: AcceptanceElement, proton_gamma5: ReferenceParticle
) -> None:
    """Bit-for-bit the drift's exact map, ``1/pz`` and all — not its linear matrix."""
    states = _off_orbit_bunch()
    got = ap.track(states, proton_gamma5)
    np.testing.assert_array_equal(got, Drift(ap.length).track(states, proton_gamma5))
    # and that differs from the matrix off the orbit, so the test can tell them apart
    assert not np.allclose(got, ap.matrix(proton_gamma5) @ states, rtol=0.0, atol=1e-12)


def test_thin_acceptance_tracks_as_the_identity(proton_gamma5: ReferenceParticle) -> None:
    states = _off_orbit_bunch()
    for ap in (Aperture("circular", 1e-2), MomentumAperture(1.0e-3)):
        np.testing.assert_array_equal(ap.track(states, proton_gamma5), states)


def test_a_collimator_is_a_drift_in_a_ring(proton_gamma5: ReferenceParticle) -> None:
    """Swap a drift for a collimator of the same length: nothing about the ring moves.

    The contradiction this gates: the ring's length (and so an RF harmonic's
    frequency) always counted the jaw, while its one-turn map did not.
    """

    def ring(middle: Drift | Collimator) -> Lattice:
        return Lattice(
            [
                Quadrupole(0.3, 1.2),
                Drift(1.0),
                middle,
                Drift(1.0),
                Quadrupole(0.3, -1.2),
                Drift(2.0),
            ],
            ref=proton_gamma5,
        )

    with_drift = ring(Drift(0.8))
    with_jaw = ring(Collimator("rectangular", 2e-2, 2e-2, length=0.8))
    assert with_jaw.length == with_drift.length
    np.testing.assert_array_equal(with_jaw.one_turn_matrix(), with_drift.one_turn_matrix())
    states = 0.1 * _off_orbit_bunch()
    jaw = Tracker(with_jaw).track_bunch_losses(Bunch(states), n_turns=3, nonlinear=True)
    assert jaw.alive.all()
    drift = Tracker(with_drift).track_bunch_losses(Bunch(states), n_turns=3, nonlinear=True)
    np.testing.assert_array_equal(jaw.states, drift.states)


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


# --- a thick boundary is checked at BOTH faces ----------------------------------
#
# Inside a field-free jaw x(s) and y(s) are straight lines — the exact drift's too,
# x = x0 + s px/pz with px, pz constant — and every shape here is convex, so a
# trajectory inside at both faces is inside all along. Checking the entry face and
# the exit face is therefore EXACT: no particle can peak inside the jaw unseen.
# Exit-only would miss a particle converging through the entry face; entry-only one
# diverging out of the exit face. Each face logs its own ``s``.
@pytest.mark.parametrize("nonlinear", [False, True])
def test_a_thick_jaw_loses_at_the_face_the_particle_crosses(
    nonlinear: bool, proton_gamma5: ReferenceParticle
) -> None:
    a, L = 5.0e-3, 2.0
    lattice = Lattice(
        [Drift(1.0), Collimator("rectangular", a, a, length=L), Drift(1.0)], ref=proton_gamma5
    )
    states = np.zeros((6, 4))
    # every face position is >= 0.1a from the edge, so the knife-edge never decides
    # 0: converging — outside at the entry face (1.2a), inside at the exit (0.8a)
    states[X, 0], states[PX, 0] = 1.2 * a, -0.4 * a / L
    # 1: diverging — inside at the entry face (0.8a), outside at the exit (1.2a)
    states[X, 1], states[PX, 1] = 0.8 * a, 0.4 * a / L
    # 2: the same in y, on the negative side
    states[Y, 2], states[PY, 2] = -0.8 * a, -0.4 * a / L
    # 3: inside at both faces (0.9a, -0.9a) while crossing the axis — no clipping
    states[X, 3], states[PX, 3] = 0.9 * a, -1.8 * a / L
    # the first drift moves them all by 1 m of slope before the jaw; undo that
    states[X] -= states[PX] * 1.0
    states[Y] -= states[PY] * 1.0
    result = Tracker(lattice).track_bunch_losses(Bunch(states), nonlinear=nonlinear)

    assert list(result.alive) == [False, False, False, True]
    assert list(result.loss_element[:3]) == [1, 1, 1]
    assert result.loss_s[0] == 1.0  # the entry face
    assert result.loss_s[1] == 1.0 + L  # the exit face
    assert result.loss_s[2] == 1.0 + L
    # a particle lost at the entry face is frozen there, before the jaw moves it
    at_entry = lattice.elements[0].track(states[:, 0], proton_gamma5)
    if not nonlinear:
        at_entry = lattice.elements[0].matrix(proton_gamma5) @ states[:, 0]
    np.testing.assert_array_equal(result.states[:, 0], at_entry)


def test_a_thin_boundary_is_checked_once(proton_gamma5: ReferenceParticle) -> None:
    """A zero-length aperture has one face; its loss ``s`` is its position, as before."""
    lattice = Lattice([Drift(1.0), Aperture("circular", 5e-3), Drift(1.0)], ref=proton_gamma5)
    states = np.zeros((6, 2))
    states[X] = [6e-3, 4e-3]
    result = Tracker(lattice).track_bunch_losses(Bunch(states))
    assert list(result.alive) == [False, True]
    assert result.loss_s[0] == 1.0
