r"""V3 — spin through a tilted bend: the machine that turns a spin out of the vertical.

V1 gave a bend a design ``tilt`` (MAD-X ``TILT``, xtrack's plain ``rot_s_rad``) and refused
to carry a spin through one. The map needed no new physics: in the bend's own frame the
precession is N1's, and the tilt is a conjugation, ``R(-t) . bend . R(+t)`` — the spin rides
the same one as the coordinates. Lifting the refusal was one deleted guard. Everything in
this file is about whether that is *right*, and in particular whether its **sense** is.

**Which gates can catch a flipped sense, and which cannot — said up front.**

- *Sign-blind* (controls only): at ``G = 0`` on the design orbit every bend is the identity
  on the spin, and a conjugated identity is still the identity; the spin tune of a ring
  with a rotator pair is ``G gamma`` for either sense, because the pair is a conjugation of
  the arc.
- *Sign-sharp, outside every code's spin machinery*: through any bend, on the design orbit,
  the spin turns **relative to the frame** about the axis the *survey* turns the frame about,
  by ``G gamma`` times the angle. That is Thomas-BMT and nothing else — for a transverse
  field the spin precesses ``(1 + G gamma)`` times as fast as the momentum about the same
  axis, and the frame follows the momentum — and the survey (``geometry.py``) is written
  separately from the spin map and pinned against MAD-X and xtrack to ``1e-13`` (V1). So
  the spin map must be the survey's turn raised to the power ``G gamma``; the wrong sense
  misses by order one.
- *Sign-sharp on a ring*: with a vertical bend each side of an interaction point, the closed
  spin solution there leans **backwards** out of the vertical by ``G gamma b`` — a spin
  rotator, the reason vertical bends exist in a polarised collider. The sign of its ``s``
  component is physical once ``n_0 . y > 0`` is fixed, and the wrong sense flips it.
- *Sign-sharp against the outside*: the tilted bend against xtrack, element and ring, in
  ``tests/reference/test_spin_tilt_xtrack.py``.

**What stays refused.** The polarisation quadrature (Sokolov-Ternov, depolarisation,
Derbenev-Kondratenko): it rebuilds each bend from untilted sub-slices and reads the guide
field as vertical, which is V1's trap. Each entry point is asserted to refuse below.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp
from scipy.spatial.transform import Rotation
from test_design_tilt import ANG, NCELL, _cell

import accsim.spin
from accsim.coords import PX, PY, X, Y
from accsim.elements.alignment import s_rotation
from accsim.elements.dipole import Dipole
from accsim.elements.drift import Drift
from accsim.geometry import survey
from accsim.lattice import Lattice
from accsim.orbit import closed_orbit_nonlinear
from accsim.radiation import (
    depolarization_integrals,
    derbenev_kondratenko_polarization,
    polarization_buildup_time,
    polarization_integrals,
    polarization_time,
    sokolov_ternov_polarization,
)
from accsim.reference import (
    ELECTRON_ANOMALOUS_MOMENT,
    ELECTRON_MASS_EV,
    PROTON_ANOMALOUS_MOMENT,
    PROTON_MASS_EV,
    ReferenceParticle,
)
from accsim.spin import (
    along_direction_of_motion,
    closed_spin_solution,
    propagate_spin_solution,
    rotate_about_s,
    spin_orbit_coupling,
)

#: Tilts away from the multiples of ``pi/2`` as well as on them: a gate that only ever saw
#: ``pi/2`` could not tell a rotation from a reflection of the two transverse axes.
TILTS = [0.3, math.pi / 2, 1.1, -0.7, math.pi, 2.5]


def electron(g: float = ELECTRON_ANOMALOUS_MOMENT) -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(
        ELECTRON_MASS_EV, 3.0e9, charge=-1.0, anomalous_moment=g
    )


def proton() -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(
        PROTON_MASS_EV, 20.0e9, charge=1.0, anomalous_moment=PROTON_ANOMALOUS_MOMENT
    )


def g_gamma(ref: ReferenceParticle) -> float:
    return ref.anomalous_moment * ref.gamma0


def spin_matrix(element, ref: ReferenceParticle, state: np.ndarray | None = None) -> np.ndarray:
    """The 3x3 rotation ``element`` applies to a spin — exact, since it is linear in the spin."""
    state = np.zeros(6) if state is None else state
    columns = [element.track_with_spin(state.copy(), e, ref)[1] for e in np.eye(3)]
    return np.array(columns).T


def survey_turn(element, ref: ReferenceParticle) -> np.ndarray:
    """The frame rotation the survey walks through ``element``, in its entrance frame."""
    W = survey(Lattice([element], ref=ref)).W
    return W[0].T @ W[1]


def power(rotation: np.ndarray, exponent: float) -> np.ndarray:
    """``rotation ** exponent``: the same axis, the angle multiplied by ``exponent``."""
    return Rotation.from_rotvec(exponent * Rotation.from_matrix(rotation).as_rotvec()).as_matrix()


def rotator_ring(g_gamma_b: float, ref: ReferenceParticle | None = None) -> Lattice:
    """V1's flat ring with a vertical bend each side of an interaction point at ``s = 0``.

    ``rot1`` bends down by ``b`` (V1: ``tilt = +pi/2`` on a positive angle turns the machine
    down), the arc is flat in the frame it leaves, and ``rot2`` undoes ``rot1``. ``b`` is set
    through ``G gamma b``, the spin-rotation angle each vertical bend applies.

    The ring closes in **direction** — the arc is a full turn about an axis in the pitched
    frame, and a full turn about any axis is the identity — but not in position: the arc
    closes on itself, so the rotator section is an excursion inserted into a closed ring,
    the same construction as V1's dogleg ring. Nothing on this axis reads the laboratory
    position; asserted below so it is a stated property, not an accident.
    """
    ref = electron() if ref is None else ref
    b = g_gamma_b / g_gamma(ref)
    elements = [Drift(0.5, name="ip_a"), Dipole(1.0, b, tilt=math.pi / 2, name="rot1")]
    elements += [e for _ in range(NCELL) for e in _cell()]
    elements += [Dipole(1.0, -b, tilt=math.pi / 2, name="rot2"), Drift(0.5, name="ip_b")]
    return Lattice(elements, ref=ref)


def _wrong_sense(spin: np.ndarray, phi: float) -> np.ndarray:
    return rotate_about_s(spin, -phi)


# --- the sense the spin is turned in ---------------------------------------------------


def test_the_spin_and_the_coordinates_are_turned_the_same_way() -> None:
    """``rotate_about_s`` on a spin is ``s_rotation`` on ``(x, y)``, the same sense.

    The only thing V3's conjugation adds over V1's, and the one the existing *tilted straight
    dipole equals a rolled one* test cannot see: both of its sides go through the same
    ``rotate_about_s``. ``s_rotation`` is byte-identical to xtrack's ``SRotation``, and
    xtrack's ``SRotation`` turns ``spin_x``/``spin_y`` with exactly the coordinate formula
    (``track_srotation.h``) — so this is the link that makes the reference comparison mean
    the same tilt on both sides.
    """
    for phi in TILTS:
        for vec in np.eye(3)[:2]:
            coords = s_rotation(phi) @ np.array([vec[0], 0.0, vec[1], 0.0, 0.0, 0.0])
            turned = rotate_about_s(vec, phi)
            assert np.array_equal(turned[:2], coords[[X, Y]])
            assert turned[2] == 0.0


# --- one bend ---------------------------------------------------------------------------


@pytest.mark.parametrize("particle", ["electron", "proton"])
@pytest.mark.parametrize("k1", [0.0, 0.2])
@pytest.mark.parametrize("tilt", [0.0, *TILTS])
def test_a_bend_turns_the_spin_about_the_axis_the_survey_turns_about(
    tilt: float, k1: float, particle: str
) -> None:
    r"""Spin map ``= (survey turn) ** (G gamma)``, on the design orbit, for every tilt.

    **Why this is the sharp gate, and why it is not a mirror.** Thomas-BMT for a transverse
    field: the momentum turns by ``Theta`` about ``w`` and the spin by ``(1 + G gamma) Theta``
    about the same ``w``. The local frame follows the momentum, so in that frame the spin
    turns by ``G gamma Theta`` about ``w`` expressed there — and the frame's own turn is what
    the survey composes, ``W_in^T W_out``. Raising one to the power ``G gamma`` gives the
    other. The survey and the spin map share no code; the survey's sense is pinned against
    both reference codes (V1).

    Exact here, not second order: on the design orbit the field is constant and the
    midpoint rule is the integral. ``k1`` is a control — a gradient has no field on axis.
    The **charge** drops out (the spin and the momentum both turn as ``q B``), so an electron
    and a proton must each satisfy the same statement with their own ``G gamma``.
    """
    ref = electron() if particle == "electron" else proton()
    bend = Dipole(1.5, ANG, k1=k1, tilt=tilt)
    turn = survey_turn(bend, ref)

    measured = spin_matrix(bend, ref)
    # Round-off scales with the spin angle: a 20 GeV proton turns its spin ~30 rad through
    # this bend (an electron ~5), and every trig evaluation of that angle loses ~angle*eps.
    floor = 1e-15 * max(1.0, abs(g_gamma(ref)) * ANG)
    assert np.abs(measured - power(turn, g_gamma(ref))).max() < floor

    # The opposite sense of the tilt is a different axis, missed by order one — except at
    # tilts that are their own opposite, where the gate cannot see the sense and says so.
    opposite = power(survey_turn(Dipole(1.5, ANG, k1=k1, tilt=-tilt), ref), g_gamma(ref))
    if math.sin(tilt) != pytest.approx(0.0, abs=1e-12):
        assert np.abs(measured - opposite).max() > 0.1


def test_on_the_design_orbit_g_zero_is_a_control() -> None:
    """``G = 0`` leaves a spin unchanged through any bend on axis — for either sense.

    Labelled a control because it cannot fail on the sense: the body map is the identity,
    and ``R(-t) I R(+t) = I`` whichever way ``R`` turns.
    """
    ref = electron(g=0.0)
    for tilt in TILTS:
        assert np.abs(spin_matrix(Dipole(1.5, ANG, tilt=tilt), ref) - np.eye(3)).max() < 1e-15


def test_off_axis_at_g_zero_the_spin_rides_the_momentum(monkeypatch) -> None:
    """N1's identity through a tilted bend, off axis — where it *can* see the sense.

    At ``G = 0`` the BMT rotation is the cyclotron rotation, so a spin launched along the
    direction of motion stays along it, whatever the field. Off axis the transverse
    momentum enters the body frame turned by the tilt; a spin turned the *other* way enters
    out of step with it and leaves off the momentum. Exact for a sector bend (constant field),
    so the right sense sits at round-off and the wrong one 10 orders above it.
    """
    ref = electron(g=0.0)
    state = np.array([2e-3, 1.5e-3, -1e-3, 2.5e-3, 0.0, 1e-3])

    def residual(tilt: float) -> float:
        out, spin = Dipole(1.5, ANG, tilt=tilt).track_with_spin(
            state.copy(), along_direction_of_motion(state), ref
        )
        return float(np.linalg.norm(spin - along_direction_of_motion(out)))

    right = [residual(t) for t in TILTS if t != math.pi]
    monkeypatch.setattr(accsim.spin, "rotate_about_s", _wrong_sense)
    wrong = [residual(t) for t in TILTS if t != math.pi]

    assert max(right) < 1e-15
    assert min(wrong) > 1e-6


# --- a ring: the spin rotator ----------------------------------------------------------


def _rotator_closed_form():
    r"""``n_0`` at the interaction point of :func:`rotator_ring`, derived with sympy.

    Each vertical bend turns the spin by ``G gamma`` times the survey's turn (the gate
    above): ``rot1`` turns the frame by ``+b`` about ``+x`` (down), so the spin by ``phi =
    G gamma b`` about ``+x``; ``rot2`` by ``-phi``. The arc is N1's ``-2 pi G gamma`` about
    ``+y``. The one-turn rotation from the IP is ``R_x(-phi) R_y(-2 pi nu) R_x(+phi)``.
    Returns the unit vector it leaves fixed, with ``n_0 . y > 0``.
    """
    phi, nu = sp.symbols("phi nu", real=True)

    def rot_x(a: sp.Expr) -> sp.Matrix:
        c, s = sp.cos(a), sp.sin(a)
        return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_y(a: sp.Expr) -> sp.Matrix:
        c, s = sp.cos(a), sp.sin(a)
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    one_turn = rot_x(-phi) * rot_y(-2 * sp.pi * nu) * rot_x(phi)
    # Candidate: the arc's own axis carried back through rot1. Verified, not assumed.
    n0 = rot_x(-phi) * sp.Matrix([0, 1, 0])
    assert sp.simplify(one_turn * n0 - n0) == sp.zeros(3, 1)
    assert sp.simplify(n0.dot(n0)) == 1
    return lambda value: np.array(n0.subs(phi, value).evalf(), dtype=float).ravel()


@pytest.mark.parametrize("frac", [1 / 12, 1 / 6, 1 / 3, 0.4])
def test_the_rotator_leans_n0_back_out_of_the_vertical(frac: float) -> None:
    """``n_0(IP) = (0, cos phi, -sin phi)``, ``phi = G gamma b`` — on the design orbit.

    The first ring in the package whose closed spin solution leaves the vertical **with no
    closed orbit at all**: every one before (N2-N5) needed a steered orbit to tilt ``n_0``.
    The sign of the ``s`` component is physical (``n_0 . y > 0`` is fixed), and it is
    backwards: a spin must start leaning back for the downward bend, which turns it
    ``G gamma`` times faster than the beam, to bring it up to vertical.
    """
    phi = frac * math.pi
    lattice = rotator_ring(phi)
    predicted = _rotator_closed_form()(phi)

    solution = closed_spin_solution(lattice)
    assert np.abs(solution.orbit).max() < 1e-15  # the design orbit: nothing is steered
    assert np.abs(solution.n0 - predicted).max() < 1e-14
    assert solution.n0[2] < 0.0


def test_the_arc_between_the_rotators_is_polarised_vertically() -> None:
    """Past ``rot1`` the closed solution is ``y`` — what a rotator pair is for.

    The arc sees an ordinary flat-ring spin; only the straight between ``rot2`` and ``rot1``
    sees the leaning one. Propagated, not re-solved, so it checks the element-by-element
    walk on a ring where ``n_0`` actually moves.
    """
    lattice = rotator_ring(math.pi / 3)
    along = np.array(propagate_spin_solution(lattice))
    arc = slice(2, len(lattice) - 2)  # after ip_a and rot1, before rot2 and ip_b
    assert np.abs(along[arc] - np.array([0.0, 1.0, 0.0])).max() < 1e-14
    assert np.abs(along[-1] - along[0]).max() < 1e-14  # and it closes


def test_the_rotator_sign_flips_with_the_sense_of_the_tilt(monkeypatch) -> None:
    """The ring-level gate is sharp: the wrong sense gives the mirror-image lean."""
    lattice = rotator_ring(math.pi / 3)
    right = closed_spin_solution(lattice).n0
    monkeypatch.setattr(accsim.spin, "rotate_about_s", _wrong_sense)
    wrong = closed_spin_solution(lattice).n0
    assert right[2] == pytest.approx(-math.sqrt(3) / 2, abs=1e-14)
    assert wrong[2] == pytest.approx(+math.sqrt(3) / 2, abs=1e-14)


def test_the_spin_tune_is_g_gamma_for_either_sense_a_control(monkeypatch) -> None:
    """The rotator pair is a conjugation of the arc, so ``nu_0 = G gamma`` — either way.

    A control, labelled: a conjugation does not change a rotation's angle, so this holds
    for a flipped sense too (asserted) and gates only that the pair cancels.
    """
    lattice = rotator_ring(math.pi / 3)
    expected = g_gamma(lattice.ref) % 1.0
    assert closed_spin_solution(lattice).spin_tune == pytest.approx(expected, abs=1e-13)
    monkeypatch.setattr(accsim.spin, "rotate_about_s", _wrong_sense)
    assert closed_spin_solution(lattice).spin_tune == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize("frac", [0.49, 0.5, 0.51])
def test_at_ninety_degrees_the_reported_pair_flips_and_the_rotation_does_not(frac: float) -> None:
    """The textbook rotator, ``G gamma b = pi/2`` (longitudinal spin at the IP), and either side.

    Before V3 an ``n_0`` with no vertical component could not occur on the design orbit; the
    rotator reaches it at exactly the setting a real one is built for. ``spin_axis_and_tune``
    orients ``n_0 . y > 0`` (xtrack's convention), so past ``pi/2`` — where the closed form
    ``(0, cos phi, -sin phi)`` points *down* — it reports the opposite vector and the spin tune
    as ``1 - frac(G gamma)``. At exactly ``pi/2`` the vertical part is round-off, the fallback
    picks ``+z``, and lands on the same side. ``(n_0, nu)`` and ``(-n_0, 1 - nu)`` are the same
    rotation: asserted, so the jump is known to be in the *report*, not in the physics.
    """
    phi = frac * math.pi
    lattice = rotator_ring(phi)
    solution = closed_spin_solution(lattice)
    closed_form = _rotator_closed_form()(phi)
    gg = g_gamma(lattice.ref)

    if frac < 0.5:
        assert np.abs(solution.n0 - closed_form).max() < 1e-14
        assert solution.spin_tune == pytest.approx(gg % 1.0, abs=1e-13)
    else:
        assert np.abs(solution.n0 + closed_form).max() < 1e-14
        assert solution.spin_tune == pytest.approx(1.0 - gg % 1.0, abs=1e-13)

    # The rotation itself is continuous: -2 pi G gamma about the closed-form axis, every side.
    rotation = Rotation.from_rotvec(-2.0 * math.pi * gg * closed_form).as_matrix()
    assert np.abs(solution.one_turn_matrix - rotation).max() < 1e-13


def test_the_rotator_ring_closes_in_direction_but_not_in_position() -> None:
    """The stated construction of :func:`rotator_ring`, asserted rather than assumed."""
    table = survey(rotator_ring(math.pi / 3))
    assert table.closure_angle < 1e-14
    assert table.closure_gap > 1.0


def test_the_spin_orbit_coupling_reproduces_the_off_momentum_solution() -> None:
    """N4's arbiter-free identity, on a ring where it is not trivially zero.

    On a flat ring ``dn/ddelta`` vanishes identically (N4); on the rotator ring the spin at
    the IP leans by ``G gamma b``, and ``gamma`` moves with ``delta``, so it does not. The
    coupling matrix ``N`` from the Sylvester solve must reproduce the derivative of the
    closed solution re-closed at ``+-ddelta`` along the dispersion — no shared machinery.
    """
    lattice = rotator_ring(math.pi / 3)
    coupling = spin_orbit_coupling(lattice)

    step = 1e-6
    dispersion = (
        closed_orbit_nonlinear(lattice, delta=+step) - closed_orbit_nonlinear(lattice, delta=-step)
    ) / (2.0 * step)
    measured = (
        closed_spin_solution(lattice, delta=+step).n0
        - closed_spin_solution(lattice, delta=-step).n0
    ) / (2.0 * step)
    predicted = coupling.matrix[:, [X, PX, Y, PY]] @ dispersion + coupling.dn_ddelta

    assert np.linalg.norm(measured) > 0.05
    assert np.linalg.norm(measured - predicted) < 5e-8 * np.linalg.norm(measured)


# --- what stays refused ------------------------------------------------------------------


@pytest.mark.parametrize(
    "consumer",
    [
        polarization_integrals,
        sokolov_ternov_polarization,
        polarization_buildup_time,
        depolarization_integrals,
        derbenev_kondratenko_polarization,
        polarization_time,
    ],
    ids=lambda f: f.__name__,
)
def test_the_polarization_quadrature_still_refuses_a_tilted_bend(consumer) -> None:
    """Every polarisation entry point ends at the refused quadrature walk.

    It rebuilds each bend as an untilted ``Dipole`` sub-slice and reads the guide field as
    vertical — V1's trap, and a milestone of its own. The spin tracking it would consume is
    now right; the quadrature is not yet.
    """
    with pytest.raises(NotImplementedError, match="tilt"):
        consumer(rotator_ring(math.pi / 3))
