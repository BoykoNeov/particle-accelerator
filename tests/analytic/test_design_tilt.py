r"""V1 — the design tilt: a bending magnet that bends the machine out of the plane.

Until this milestone every bend in ``accsim`` turned the beam sideways, so every machine it
could describe was flat — ``geometry.py`` said so as a documented refusal. A **design tilt**
(MAD-X ``TILT``, xtrack's plain ``rot_s_rad``) turns a magnet about the beam axis *and takes
the reference frame with it*: a bend tilted by ``pi/2`` is a vertical bend, and a ring can
leave the horizontal plane.

**The map is a conjugation, for a bend as for a straight element.** Because the frame turns
with the magnet, the particle enters the magnet's own frame by ``R(+tilt)``, crosses the
untilted body, and leaves by ``R(-tilt)`` — the *same* rotation undone, with no exit-face
correction. That is the whole difference from K2's ``roll`` (a misalignment), whose exit
face is pitched, yawed and displaced because the frame did **not** follow the magnet. A
tilted bend therefore has exactly zero kick on the design orbit, and a rolled one does not.

**Which gates can catch a flipped sign, and which cannot — said up front.**

- *Sign-blind* (they hold for either sense of the tilt, so they are consistency checks
  only): the ring rolled as a whole closes in a tilted plane, keeps its tunes, and has the
  one-turn map ``R(-t) M R(+t)``; a tilted gradient magnet equals a rolled one; a tilted
  bend radiates what a flat one does.
- *Sign-sharp inside this file*: a bend's **dispersion column points away from the way the
  survey says it turns**, for every tilt. The map and the survey are written separately,
  so this is the gate that ties the two senses together.
- *Sign-sharp against the outside*: ``+pi/2`` on a positive bend turns the machine
  **down**, and the signed ``D_y`` of the dogleg ring — both in
  ``tests/reference/test_design_tilt_{xtrack,madx}.py``.

**Everything that reads a bend as horizontal was listed, and each one handles the tilt or
refuses it here** (the S1 pattern): the survey and the momentum-compaction quadrature handle
it; the natural-chromaticity sum, the radiation integrals, the polarisation quadrature, spin
through a tilted bend and the scenario file refuse it, each with a test that fails the day the
refusal is lifted.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from accsim.coords import DELTA, PY, X, Y
from accsim.elements.alignment import s_rotation
from accsim.elements.dipole import Dipole
from accsim.elements.drift import Drift
from accsim.elements.quadrupole import Quadrupole
from accsim.elements.rfcavity import RFCavity
from accsim.geometry import survey
from accsim.lattice import Lattice
from accsim.orbit import closed_orbit
from accsim.radiation import polarization_integrals, radiation_integrals
from accsim.reference import ELECTRON_ANOMALOUS_MOMENT, ReferenceParticle
from accsim.scenario import ScenarioError, element_from_dict, element_to_dict
from accsim.tapering import taper, taper_profile
from accsim.twiss import (
    chromaticity,
    chromaticity_on_orbit,
    closed_twiss,
    momentum_compaction,
    natural_chromaticity,
    natural_chromaticity_on_orbit,
    propagate_twiss,
    tunes,
)

ELECTRON_MASS_EV = 0.51099895069e6

#: The filter run's 4-cell ring (2026-10-06): 3 GeV electrons, eight 45-degree sector bends.
NCELL = 4
ANG = 2.0 * math.pi / (2 * NCELL)
#: The dogleg's two vertical bends, opposite in sign, a metre apart.
VA = 0.02


def _ref() -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(
        ELECTRON_MASS_EV, 3.0e9, charge=-1.0, anomalous_moment=ELECTRON_ANOMALOUS_MOMENT
    )


def _cell(tilt: float = 0.0) -> list:
    """One FODO cell. ``tilt`` rolls the whole cell with its frame — the quadrupoles by
    ``roll`` (for a straight element a roll *is* a design tilt: same map, same survey)."""
    return [
        Quadrupole(0.5, 0.30, roll=tilt),
        Drift(0.6),
        Dipole(1.5, ANG, tilt=tilt),
        Drift(0.6),
        Quadrupole(0.5, -0.30, roll=tilt),
        Drift(0.6),
        Dipole(1.5, ANG, tilt=tilt),
        Drift(0.6),
    ]


def flat_ring() -> Lattice:
    return Lattice([e for _ in range(NCELL) for e in _cell()], ref=_ref())


def rolled_ring(t: float) -> Lattice:
    """F2: every element of the flat ring turned by ``t`` with its frame."""
    return Lattice([e for _ in range(NCELL) for e in _cell(t)], ref=_ref())


def dogleg_ring() -> Lattice:
    """F1: the flat ring with a vertical dogleg (two ``tilt = pi/2`` bends) in one straight.

    Horizontal and vertical bends mix, so the frame rotations do **not** commute — the
    fixture a walker composing in the wrong order fails on.
    """
    dogleg = [
        Dipole(0.3, VA, tilt=math.pi / 2, name="vb1"),
        Drift(1.0, name="dd"),
        Dipole(0.3, -VA, tilt=math.pi / 2, name="vb2"),
    ]
    return Lattice(_cell() + dogleg + [e for _ in range(NCELL - 1) for e in _cell()], ref=_ref())


#: F3: a line whose four bends are horizontal, tilted by 0.7, vertical and horizontal again.
F3 = [(1.0, 0.3, 0.0), (1.0, 0.4, 0.7), (1.0, 0.25, math.pi / 2), (1.0, -0.2, 0.0)]


def f3_line() -> Lattice:
    elements: list = []
    for length, angle, tilt in F3:
        elements += [Dipole(length, angle, tilt=tilt), Drift(0.8)]
    return Lattice(elements, ref=_ref())


# ---------------------------------------------------------------------------
# Gate 1 — zero tilt changes nothing, to the last bit.


@pytest.mark.parametrize("k1", [0.0, 0.2])
def test_zero_tilt_is_the_flat_bend_bit_for_bit(k1: float) -> None:
    """``tilt=0.0`` must not route a design bend through any rotation at all."""
    ref = _ref()
    plain, zero = Dipole(1.5, ANG, k1=k1, e1=0.1), Dipole(1.5, ANG, k1=k1, e1=0.1, tilt=0.0)
    assert np.array_equal(plain.matrix(ref), zero.matrix(ref))
    assert np.array_equal(plain.kick(ref), zero.kick(ref))
    state = np.array([1e-3, -2e-4, 5e-4, 3e-4, 1e-3, 2e-3])
    assert np.array_equal(plain.track(state, ref), zero.track(state, ref))
    assert repr(plain) == repr(zero)


def test_zero_tilt_leaves_the_survey_exactly_planar() -> None:
    """The flat ring's survey is the R1 survey: ``Y``, ``phi`` and ``psi`` exactly zero."""
    table = survey(flat_ring())
    assert not np.any(table.Y)
    assert not np.any(table.phi)
    assert not np.any(table.psi)


# ---------------------------------------------------------------------------
# Gate 2 — the map is the conjugation, and the design orbit is untouched.


@pytest.mark.parametrize("tilt", [0.7, math.pi / 2, -1.1])
def test_the_matrix_is_the_flat_bend_conjugated(tilt: float) -> None:
    ref = _ref()
    flat = Dipole(1.5, ANG, k1=0.1, e1=0.05, e2=-0.03).matrix(ref)
    tilted = Dipole(1.5, ANG, k1=0.1, e1=0.05, e2=-0.03, tilt=tilt).matrix(ref)
    expected = s_rotation(-tilt) @ flat @ s_rotation(tilt)
    assert np.allclose(tilted, expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("fringe", [False, True])
def test_tracking_is_the_flat_bend_conjugated(fringe: bool) -> None:
    """The exact map too — including the nonlinear faces — and for a bunch."""
    ref, tilt = _ref(), 0.7
    flat = Dipole(1.5, ANG, e1=0.05, fringe=fringe)
    tilted = Dipole(1.5, ANG, e1=0.05, fringe=fringe, tilt=tilt)
    rng = np.random.default_rng(1)
    bunch = rng.normal(scale=[[1e-3], [1e-4], [1e-3], [1e-4], [1e-3], [1e-3]], size=(6, 7))
    expected = s_rotation(-tilt) @ flat.track(s_rotation(tilt) @ bunch, ref)
    assert np.allclose(tilted.track(bunch, ref), expected, rtol=0.0, atol=1e-17)


def test_a_tilted_bend_has_no_kick_where_a_rolled_one_does() -> None:
    """The design orbit is rolled with the magnet, so it is still a fixed point.

    The control is K2's roll by the same angle: same magnet, same rotation, but the frame
    stays — and the exit face is then somewhere else, which *is* a kick.
    """
    ref = _ref()
    tilted, rolled = Dipole(1.5, ANG, tilt=0.3), Dipole(1.5, ANG, roll=0.3)
    assert not np.any(tilted.kick(ref))
    assert not np.any(tilted.track(np.zeros(6), ref))
    assert abs(rolled.kick(ref)[PY]) > 1e-2


def test_a_tilted_straight_dipole_is_a_rolled_one() -> None:
    """For a straight element the two rotations coincide — **sign-blind**: both use
    ``s_rotation``, so this pins the wiring, not the sense."""
    ref = _ref()
    tilted = Dipole(0.5, 0.0, k1=0.3, tilt=0.4).matrix(ref)
    rolled = Dipole(0.5, 0.0, k1=0.3, roll=0.4).matrix(ref)
    assert np.array_equal(tilted, rolled)


def test_the_tilt_is_symplectic() -> None:
    from accsim.symplectic import is_symplectic

    assert is_symplectic(Dipole(1.5, ANG, k1=0.1, tilt=0.7).matrix(_ref()))


# ---------------------------------------------------------------------------
# Gate 3 — the map and the survey agree on which way the bend turns (sign-sharp).


@pytest.mark.parametrize("tilt", [0.0, 0.7, math.pi / 2, 2.5, -1.1])
@pytest.mark.parametrize("angle", [0.4, -0.3])
def test_dispersion_points_away_from_the_way_the_survey_turns(angle: float, tilt: float) -> None:
    """A faster particle is bent less, so it lands on the **outside** of the turn.

    The single-bend dispersion column ``(R16, R36)`` must therefore be anti-parallel to the
    transverse part of the chord the survey walks — the direction the bend turns towards,
    expressed in the frame the bend is entered in. The map (``s_rotation``) and the survey
    (``R_z``) are written separately, so a flipped sense in either one fails this for every
    tilt that is not a multiple of ``pi``.
    """
    bend = Dipole(1.0, angle, tilt=tilt)
    M = bend.matrix(_ref())
    table = survey(Lattice([bend], ref=_ref()))
    chord_local = table.W[0].T @ (table.positions[1] - table.positions[0])
    turn = chord_local[:2]  # (x, y) of the chord: where the bend turns towards
    disp = np.array([M[X, DELTA], M[Y, DELTA]])
    # anti-parallel, and the magnitudes are the same sagitta rho (1 - cos a)
    assert np.allclose(disp, -turn, rtol=0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Gate 4 — the survey leaves the plane, and the walk is pinned independently.


def test_a_vertical_bend_turns_the_machine_down() -> None:
    """``tilt = +pi/2`` on a positive bend: ``Y = -rho (1 - cos a)``, ``phi = -a``.

    The sense is both reference codes' (measured 2026-10-06, MAD-X and xtrack agreeing to
    ``1e-15``); here it is written as the closed form it implies.
    """
    a, length = 0.25, 1.0
    rho = length / a
    table = survey(Lattice([Dipole(length, a, tilt=math.pi / 2)], ref=_ref()))
    assert table.X[-1] == pytest.approx(0.0, abs=1e-15)
    assert table.Y[-1] == pytest.approx(-rho * (1.0 - math.cos(a)), rel=1e-14)
    assert table.Z[-1] == pytest.approx(rho * math.sin(a), rel=1e-14)
    assert table.phi[-1] == pytest.approx(-a, rel=1e-14)
    assert table.theta[-1] == pytest.approx(0.0, abs=1e-15)
    assert table.psi[-1] == pytest.approx(0.0, abs=1e-15)


def _rotvec_walk(spec) -> tuple[np.ndarray, list[np.ndarray]]:
    """The independent leg: each bend is a rotation by ``-a`` about its **tilted** ``y``
    axis, composed as scipy rotations from a rotation vector — no ``R_y``, no ``R_z``, no
    conjugation anywhere. The arc's chord is the circle through the entry point with its
    centre ``rho`` away along the tilted ``-x`` axis."""
    W = Rotation.identity()
    pos = [np.zeros(3)]
    frames = [W.as_matrix()]
    for length, angle, tilt in spec:
        u = np.array([math.cos(tilt), math.sin(tilt), 0.0])  # the bend's own x, in local axes
        n = np.array([-math.sin(tilt), math.cos(tilt), 0.0])  # the bend's own y
        if angle == 0.0:
            step = np.array([0.0, 0.0, length])
        else:
            rho = length / angle
            step = rho * (math.cos(angle) - 1.0) * u + rho * math.sin(angle) * np.array([0, 0, 1.0])
        pos.append(pos[-1] + W.apply(step))
        W = W * Rotation.from_rotvec(-angle * n)
        frames.append(W.as_matrix())
    return np.asarray(pos), frames


def test_the_walk_matches_an_independent_rotation_vector_walk() -> None:
    spec: list = []
    for length, angle, tilt in F3:
        spec += [(length, angle, tilt), (0.8, 0.0, 0.0)]
    pos, frames = _rotvec_walk(spec)
    table = survey(f3_line())
    assert np.allclose(table.positions, pos, rtol=0.0, atol=1e-14)
    assert np.allclose(table.W, np.asarray(frames), rtol=0.0, atol=1e-14)


def test_the_composition_order_is_gated() -> None:
    """Control: swap the tilted and the vertical bend and the line ends half a metre away —
    so the comparison above could not pass with the rotations composed in either order."""
    spec: list = []
    for i in (0, 2, 1, 3):
        spec += [F3[i], (0.8, 0.0, 0.0)]
    swapped, _ = _rotvec_walk(spec)
    assert np.linalg.norm(swapped[-1] - survey(f3_line()).positions[-1]) > 0.3


def test_the_angles_reconstruct_the_frame() -> None:
    """``theta, phi, psi`` are read off ``W`` in MAD-X's order, ``W = R_y(theta) R_x(-phi)
    R_z(psi)``; rebuilding ``W`` from them must give it back."""
    table = survey(f3_line())
    for i in range(len(table)):
        t, p, s = table.theta[i], table.phi[i], table.psi[i]
        rebuilt = Rotation.from_euler("YXZ", [t, -p, s]).as_matrix()
        assert np.allclose(rebuilt, table.W[i], rtol=0.0, atol=1e-14)


def test_a_ring_rolled_as_a_whole_is_the_flat_ring_rolled() -> None:
    """**Sign-blind**: rolling every frame by ``t`` rotates the whole machine by
    ``R_z(+t)`` about the start, and it still closes — in a tilted plane."""
    t = 0.1
    flat, rolled = survey(flat_ring()), survey(rolled_ring(t))
    c, s = math.cos(t), math.sin(t)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(rolled.positions, flat.positions @ Rz.T, rtol=0.0, atol=1e-13)
    assert rolled.closure_gap < 1e-13
    assert rolled.closure_angle < 1e-13
    assert np.max(np.abs(rolled.Y)) > 0.5  # it really has left the plane


def test_the_dogleg_ring_misses_closure_by_exactly_the_dogleg() -> None:
    """The dogleg is 1.6 m of extra straight in a closed ring, so the ring cannot close in
    *position* — but its two bends cancel, so it closes in *direction*, and then the rest of
    the ring is a closed polygon translated by the dogleg. The gap is therefore exactly the
    dogleg's own displacement, and the machine is out of plane only from the dogleg on."""
    table = survey(dogleg_ring())
    assert table.closure_angle < 1e-14
    names = list(table.names)
    i1, i2 = names.index("vb1"), names.index("vb2")
    across = table.positions[i2 + 1] - table.positions[i1]
    gap = table.positions[-1] - table.positions[0]
    assert np.allclose(gap, across, rtol=0.0, atol=1e-13)
    rise = table.Y[i2 + 1]
    assert rise == pytest.approx(table.Y[-1], abs=1e-15)  # level again after the dogleg
    assert rise < 0.0  # a positive vertical bend goes down first
    assert not np.any(table.Y[: i1 + 1])


# ---------------------------------------------------------------------------
# Gate 5 — optics: what the matrices give for free, checked by identity.


def test_the_rolled_ring_keeps_its_tunes_and_conjugates_its_one_turn_map() -> None:
    """**Sign-blind**, and exact by telescoping: every ``R(+t) R(-t)`` between elements
    cancels, leaving ``R(-t) M_flat R(+t)``."""
    t = 0.1
    M_flat, M_rolled = flat_ring().one_turn_matrix(), rolled_ring(t).one_turn_matrix()
    assert np.allclose(M_rolled, s_rotation(-t) @ M_flat @ s_rotation(t), rtol=0.0, atol=1e-12)
    from accsim.twiss import normal_mode_tunes

    qa = normal_mode_tunes(flat_ring())
    qb = normal_mode_tunes(rolled_ring(t))
    assert qb == pytest.approx(qa, abs=1e-12)


def test_the_dogleg_makes_vertical_dispersion_and_no_orbit() -> None:
    """The dogleg's bends are a *design*: they leave the closed orbit at zero (a roll
    would not) and make ``D_y`` that is zero before the dogleg's first bend."""
    lat = dogleg_ring()
    assert np.max(np.abs(closed_orbit(lat))) < 1e-15
    pts = propagate_twiss(lat, closed_twiss(lat))
    dy = np.array([p.disp_y for p in pts])
    assert np.max(np.abs(dy)) > 1e-2
    assert tunes(lat)[1] != pytest.approx(tunes(flat_ring())[1], abs=1e-6)


def test_momentum_compaction_quadrature_follows_the_tilt() -> None:
    """The quadrature reads ``h`` against the dispersion **in the bend's own plane**, so
    a vertical bend's ``h D_y`` is counted. It is the second route, sharing no matrix entry
    with the identity route, and the two must meet on the dogleg ring."""
    lat = dogleg_ring()
    ident = momentum_compaction(lat, method="identity")
    quad = momentum_compaction(lat, method="quadrature", slices=256)
    flat = momentum_compaction(flat_ring(), method="identity")
    assert quad == pytest.approx(ident, rel=1e-7)
    assert ident != pytest.approx(flat, rel=1e-6)  # the dogleg is visible to both


# ---------------------------------------------------------------------------
# Gate 6 — radiation in tracking is rotation-invariant (sign-blind), and so is the taper.


def test_a_tilted_bend_radiates_what_a_flat_one_does() -> None:
    ref = _ref()
    state = np.array([1e-4, 2e-5, -3e-4, 1e-5, 0.0, 1e-3])
    flat = Dipole(1.5, ANG).track(state, ref, radiation="mean")
    t = 0.7
    tilted = Dipole(1.5, ANG, tilt=t).track(s_rotation(-t) @ state, ref, radiation="mean")
    assert np.allclose(s_rotation(t) @ tilted, flat, rtol=0.0, atol=1e-18)


def test_the_taper_of_a_rolled_ring_is_the_flat_rings() -> None:
    a, b = taper_profile(flat_ring()), taper_profile(rolled_ring(0.1))
    assert np.allclose(b.delta, a.delta, rtol=0.0, atol=1e-16)


def test_taper_scales_a_tilted_ring_like_the_flat_one_and_keeps_the_tilt() -> None:
    """``taper()`` — not only the profile — goes through tracking, never the radiation
    integrals, so it accepts a tilted ring: the same field factors, the tilt carried over."""

    def with_rf(lat: Lattice) -> Lattice:
        # taper() needs a 6D fixed point, so a cavity paying the ~3.7 MeV turn loss.
        cav = RFCavity.from_harmonic(20.0e6, 20, lat.length, lat.ref, phi_s=math.pi)
        return Lattice([*lat.elements, cav], ref=lat.ref)

    flat, rolled = taper(with_rf(flat_ring())), taper(with_rf(rolled_ring(0.1)))
    # not vacuous: the taper really moved the fields it is compared on
    assert any(isinstance(e, Dipole) and abs(e.taper) > 1e-4 for e in flat.elements)
    for a, b in zip(flat.elements, rolled.elements, strict=True):
        if isinstance(a, Dipole):
            assert b.k0 == pytest.approx(a.k0, rel=1e-14)
            assert b.tilt == 0.1


@pytest.mark.parametrize(
    "entry", [chromaticity, chromaticity_on_orbit, natural_chromaticity_on_orbit]
)
def test_every_chromaticity_entry_point_refuses_a_tilted_bend(entry) -> None:
    """The refusal sits in ``natural_chromaticity``; these reach it — the on-orbit pair
    through ``linearised_lattice``, which must therefore keep the tilted ``Dipole``."""
    with pytest.raises(NotImplementedError, match="tilt"):
        entry(dogleg_ring())


def test_the_compaction_quadrature_refuses_a_rolled_bend() -> None:
    """K2's rolled bend moves its exit face, which the quadrature's frame change (a
    rotation) does not describe. The ring is coupled anyway, so the identity route refuses
    it too — through the coupling guard, not this one."""
    lat = Lattice(_cell() + [Dipole(1.5, ANG, roll=0.01)], ref=_ref())
    with pytest.raises(NotImplementedError, match="roll"):
        momentum_compaction(lat, method="quadrature")


# ---------------------------------------------------------------------------
# Gate 7 — every consumer that reads a bend as horizontal refuses a tilted one.


def test_natural_chromaticity_refuses_a_tilted_bend() -> None:
    with pytest.raises(NotImplementedError, match="tilt"):
        natural_chromaticity(dogleg_ring())


def test_natural_chromaticity_refuses_a_tilted_gradient_magnet() -> None:
    """A gradient magnet tilted by ``pi/2`` is uncoupled (a quadrupole of the opposite
    sign), so nothing upstream catches it — and the sum would read ``+k1``."""
    lat = Lattice(_cell() + [Dipole(0.4, 0.0, k1=0.1, tilt=math.pi / 2)], ref=_ref())
    with pytest.raises(NotImplementedError, match="tilt"):
        natural_chromaticity(lat)


def test_the_radiation_integrals_accept_a_tilted_bend_since_v2() -> None:
    """V1 refused this; V2 lifted it (``test_vertical_emittance.py`` holds the physics). What
    is left here is the flip itself: the dogleg ring now has a vertical half."""
    ri = radiation_integrals(dogleg_ring())
    assert ri.i5y > 0.0 and ri.i4y != 0.0


def test_the_polarization_integrals_refuse_a_tilted_bend() -> None:
    with pytest.raises(NotImplementedError, match="tilt"):
        polarization_integrals(dogleg_ring())


def test_spin_through_a_tilted_bend_is_refused() -> None:
    bend = Dipole(1.5, ANG, tilt=0.3)
    with pytest.raises(NotImplementedError, match="tilt"):
        bend.track_with_spin(np.zeros(6), np.array([0.0, 1.0, 0.0]), _ref())


def test_spin_through_a_tilted_straight_dipole_is_the_rolled_one() -> None:
    ref = _ref()
    spin = np.array([0.0, 1.0, 0.0])
    state = np.array([1e-3, 0.0, 2e-3, 0.0, 0.0, 0.0])
    _, a = Dipole(0.5, 0.0, k1=0.3, tilt=0.4).track_with_spin(state, spin, ref)
    _, b = Dipole(0.5, 0.0, k1=0.3, roll=0.4).track_with_spin(state, spin, ref)
    assert np.allclose(a, b, rtol=0.0, atol=1e-16)


@pytest.mark.parametrize("kwargs", [{"roll": 0.01}, {"dx": 1e-4}], ids=["with-roll", "with-offset"])
def test_a_tilt_and_a_misalignment_together_are_refused(kwargs: dict) -> None:
    """Which rotation acts first is a convention, and it is not chosen by argument."""
    angle = 0.0 if "dx" in kwargs else ANG
    with pytest.raises(NotImplementedError, match="tilt"):
        Dipole(1.5, angle, k1=0.1, tilt=0.3, **kwargs)


def test_a_misalignment_set_after_construction_is_still_refused() -> None:
    bend = Dipole(1.5, ANG, tilt=0.3)
    bend.roll = 0.01
    with pytest.raises(NotImplementedError, match="tilt"):
        bend.matrix(_ref())


def test_the_scenario_file_refuses_a_tilt_both_ways() -> None:
    """The editor's optics is a JavaScript port held to ``1e-9``; a tilt it cannot draw must
    not load — and a tilted lattice must not save as a flat one."""
    with pytest.raises(ScenarioError, match="tilt"):
        element_to_dict(Dipole(1.5, ANG, tilt=0.3))
    with pytest.raises(ScenarioError, match="tilt"):
        element_from_dict({"type": "Dipole", "length": 1.5, "angle": ANG, "tilt": 0.3})
    # a zero tilt is a flat bend, and round-trips as one
    assert "tilt" not in element_to_dict(Dipole(1.5, ANG, tilt=0.0))


def test_a_tilt_is_not_hidden_by_repr_or_lost_by_copy() -> None:
    bend = Dipole(1.5, ANG, tilt=0.3)
    assert "tilt=0.3" in repr(bend)
    assert copy.copy(bend).tilt == 0.3


def test_the_tilt_is_a_dipole_attribute_only() -> None:
    """For a straight element a design tilt **is** a roll — same map, same survey — so
    ``roll`` covers it and no second spelling is offered."""
    with pytest.raises(TypeError):
        Quadrupole(0.5, 0.3, tilt=0.1)  # type: ignore[call-arg]
    assert Quadrupole(0.5, 0.3).tilt == 0.0


def test_the_dogleg_dispersion_sign_under_tilt_reversal() -> None:
    """Reversing the dogleg's tilt (``-pi/2``) flips the sign of ``D_y`` everywhere: the
    sign is carried by the tilt, not by anything symmetric about it."""
    lat = dogleg_ring()
    rev = Lattice(
        [
            Dipole(e.length, e.angle, tilt=-e.tilt, name=e.name) if e.tilt else e
            for e in lat.elements
        ],
        ref=_ref(),
    )
    a = np.array([p.disp_y for p in propagate_twiss(lat, closed_twiss(lat))])
    b = np.array([p.disp_y for p in propagate_twiss(rev, closed_twiss(rev))])
    assert np.allclose(b, -a, rtol=0.0, atol=1e-15)
    assert np.max(np.abs(a)) > 1e-2
    # the horizontal plane does not know which way the dogleg went
    ax = np.array([p.disp_x for p in propagate_twiss(lat, closed_twiss(lat))])
    bx = np.array([p.disp_x for p in propagate_twiss(rev, closed_twiss(rev))])
    assert np.allclose(ax, bx, rtol=0.0, atol=1e-14)
