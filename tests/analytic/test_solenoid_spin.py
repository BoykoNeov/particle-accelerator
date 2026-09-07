r"""Analytic gates for S2 — the solenoid's field becomes visible to the package.

S1 shipped the solenoid's *map* and refused its *field*: the accessor returns ``(bx, by)``
and an ideal solenoid has neither, so rather than report a silent zero — which would make
:mod:`accsim.spin` say *no precession* in the one magnet built to rotate spins — it raised.
This milestone lands the field, and with it spin precession, radiation, and the taper that
S1 recorded as the price of the refusal.

Two things about it are not what the roadmap expected, and every gate here exists because
of one of them:

  * **It is not one hook but two.** The field is the obvious half. The second, which
    nothing priced, is the **vector potential**: inside a solenoid the momentum the state
    vector stores is *not* the direction the particle travels, and the gap
    ``a = (-ks y / 2, +ks x / 2)`` is first order in the transverse amplitude — the same
    order as the perpendicular field itself. A solenoid is the first element in the
    package where the two differ at all, which is why nothing caught it earlier.
  * **The reference code gets that second half wrong**, and the sharp gates here are
    therefore *arbiter-free*. xtrack's ``magnet_spin`` builds the direction of motion from
    ``px + ax`` while its own solenoid body (``pk1 = px + sk*y``) and its own radiation
    path (``mean_kin_px``) both use ``px - ax``. The disagreement is asserted as a
    mechanism in ``tests/reference/`` and is settled here, without it, by S1's own
    validated map.

Ordered by how much each can catch:

  * **The precession axis points along the motion, and no coefficient says so.** For a
    purely longitudinal field ``Omega`` has *exactly* the transverse direction of the
    direction of motion — the BMT coefficients cancel out of the statement entirely. So
    the axis extracted from the rotation :meth:`track_with_spin` actually applies can be
    compared with the trajectory's finite-differenced tangent with nothing in between.
    The three candidate momenta are **order unity** apart there, not a tolerance apart,
    and a test that re-implemented the BMT formula could not say this.
  * **The direction of motion is the trajectory's tangent**, at the midpoint state:
    the kinetic momentum reproduces it at ``6.3e-11`` converging as the **cube** of the
    amplitude, the canonical one misses by ``|a|`` and xtrack's by ``2|a|``, both linear.
  * **The kinetic momentum's rate of change is ``v x b``** — the three-component form of
    the identity ``tests/analytic/test_spin.py`` gates for every straight magnet — and the
    **canonical** rate is exactly *half* of it. That factor of two is the vector potential,
    stated as a number.
  * **A solenoid precesses and radiates at all.** Both consumers reach an element through
    a zero-field early return, and a solenoid has ``bx = by = 0``. Gated directly, because
    forgetting ``b_s`` there reproduces the silent-zero failure S1 refused to ship.
  * **The ``G = 0`` control, and its measured blindness** to the very question above.
  * **Nothing else moves**, asserted bit-for-bit against values captured from the shipped
    code *before* this milestone touched it.

The disagreement with xtrack, and the radiation cross-check, are in
``tests/reference/test_solenoid_spin_xtrack.py``.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from accsim import (
    BeamBeam,
    Corrector,
    Dipole,
    Drift,
    Octupole,
    Quadrupole,
    ReferenceParticle,
    RFCavity,
    Sextupole,
    SkewQuadrupole,
    Solenoid,
    ThinQuadrupole,
    ThinSextupole,
)
from accsim.coords import DELTA, PX, PY, ZETA, X, Y
from accsim.reference import (
    ELECTRON_ANOMALOUS_MOMENT,
    ELECTRON_MASS_EV,
    PROTON_ANOMALOUS_MOMENT,
    PROTON_MASS_EV,
)
from accsim.spin import anomalous_moment, direction_of_motion, precession_vector, rotate

ENERGY = 5.0e9
LENGTH = 0.9
KS = 0.6

#: Every coordinate nonzero, so no term can hide behind a zero. ``delta = 0`` here on
#: purpose: the vector potential and the field are both momentum-independent, and a
#: nonzero ``delta`` only rescales the Larmor rate (S1's own chromatic statement).
STATE = np.array([2.0e-3, 1.0e-4, -1.5e-3, 2.0e-4, 0.0, 0.0])


def electron(g: float = ELECTRON_ANOMALOUS_MOMENT) -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(
        ELECTRON_MASS_EV, ENERGY, charge=-1.0, anomalous_moment=g
    )


# ================================ the two new hooks =======================================


def _one_of_each() -> list:
    """One instance of every element a lattice can hold, with strengths switched on."""
    return [
        Drift(0.5),
        Quadrupole(0.5, 1.3),
        SkewQuadrupole(0.5, 1.3),
        Sextupole(0.5, 2.1),
        Octupole(0.5, 3.4),
        Dipole(1.2, angle=0.09, k1=0.35),
        Corrector(kick_x=1e-4, kick_y=-2e-4),
        ThinQuadrupole(0.4),
        ThinSextupole(0.7),
        RFCavity(1.0e6, 400e6, 0.0),
        BeamBeam(1e11, 1e-4, 1e-4),
    ]


@pytest.mark.parametrize("element", _one_of_each(), ids=lambda e: type(e).__name__)
def test_both_new_hooks_default_to_exact_zero(element) -> None:
    """Every element that is not a solenoid has no longitudinal field and no potential.

    The interface half of the milestone, and the reason it is *not* the signature break
    the roadmap billed: ``normalized_field`` keeps its two-component contract and the two
    new accessors default to zero on :class:`~accsim.elements.element.Element`, so not one
    existing element file had to change. Exact zeros, not small ones — the consumers'
    early returns test ``== 0.0``, and gate 8's bit-identity depends on it.
    """
    x, y = 2.0e-3, -1.3e-3
    assert np.all(np.asarray(element.longitudinal_field(x, y)) == 0.0)
    ax, ay = element.normalized_vector_potential(x, y)
    assert np.all(np.asarray(ax) == 0.0) and np.all(np.asarray(ay) == 0.0)


def test_the_solenoid_reports_its_field_and_the_potential_that_generates_it() -> None:
    r"""``b = (0, 0, ks)`` and ``a = (-ks y/2, +ks x/2)``, and ``curl a = b``.

    The field value is not a judgement call: xtrack computes it analytically from the
    strengths (``Bz_T = ks * brho_0`` with ``brho_0 = p0c / c / q0``), so in this
    package's normalisation ``b = B/(B rho)_0`` the answer is ``b_s = ks`` exactly. The
    potential is the one S1's own class docstring already wrote down when it derived the
    map, so the two halves of the element cannot disagree — and the curl is checked here
    rather than trusted, because a sign error in ``a`` is exactly the error this milestone
    is about.

    ``normalized_field`` now answers ``(0, 0)`` truthfully instead of raising: the
    longitudinal component finally has somewhere to go.
    """
    sol = Solenoid(LENGTH, KS, name="ds")
    x, y = 2.0e-3, -1.3e-3

    bx, by = sol.normalized_field(x, y)
    assert float(bx) == 0.0 and float(by) == 0.0
    assert float(sol.longitudinal_field(x, y)) == KS  # exactly, at every point

    ax, ay = sol.normalized_vector_potential(x, y)
    assert float(ax) == -0.5 * KS * y
    assert float(ay) == 0.5 * KS * x

    # curl a = da_y/dx - da_x/dy = b_s, by central differences (a is linear, so exact)
    h = 1e-6
    d_ay_dx = (
        sol.normalized_vector_potential(x + h, y)[1] - sol.normalized_vector_potential(x - h, y)[1]
    ) / (2 * h)
    d_ax_dy = (
        sol.normalized_vector_potential(x, y + h)[0] - sol.normalized_vector_potential(x, y - h)[0]
    ) / (2 * h)
    assert float(d_ay_dx - d_ax_dy) == pytest.approx(KS, rel=1e-12)


def test_the_potential_is_linear_so_the_midpoint_and_the_endpoint_mean_agree() -> None:
    """``mean(p - a)`` over the endpoints equals ``mean(p) - a(midpoint)``, to round-off.

    :func:`accsim.spin.spin_precession` samples the field at the mean of the entry and
    exit coordinates and subtracts the potential evaluated *there*, which is only the same
    thing as averaging the two endpoint kinetic momenta because ``a`` is **linear** in
    ``(x, y)``. For a uniform solenoid it is; this is the assertion that fails first if a
    non-uniform one ever arrives, and it is why the cheaper form is legitimate.
    """
    ref = electron()
    sol = Solenoid(LENGTH, KS)
    out = sol.track(STATE.copy(), ref)

    a0 = sol.normalized_vector_potential(STATE[X], STATE[Y])
    a1 = sol.normalized_vector_potential(out[X], out[Y])
    mean_kin = np.array(
        [0.5 * (STATE[PX] - a0[0] + out[PX] - a1[0]), 0.5 * (STATE[PY] - a0[1] + out[PY] - a1[1])]
    )

    mid = sol.normalized_vector_potential(0.5 * (STATE[X] + out[X]), 0.5 * (STATE[Y] + out[Y]))
    mean_minus_mid = np.array(
        [0.5 * (STATE[PX] + out[PX]) - mid[0], 0.5 * (STATE[PY] + out[PY]) - mid[1]]
    )
    assert np.abs(mean_kin - mean_minus_mid).max() < 1e-18


# ======================= gate 2 (a): the tangent of the trajectory ========================


def _tangent(
    length: float, ks: float, state: np.ndarray, ref: ReferenceParticle, h: float
) -> np.ndarray:
    """``dr/ds`` at the magnet's midpoint, central-differenced from S1's own map.

    Nothing here routes through a reference code: the trajectory is whatever
    :meth:`Solenoid.track` says it is, and S1 validated that against both arbiters.
    """
    fwd = Solenoid(0.5 * length + h, ks).track(state.copy(), ref)
    bwd = Solenoid(0.5 * length - h, ks).track(state.copy(), ref)
    t = np.array([fwd[X] - bwd[X], fwd[Y] - bwd[Y], 2.0 * h])
    return t / np.linalg.norm(t)


def _candidates(sol: Solenoid, st: np.ndarray) -> dict[str, np.ndarray]:
    """The three directions of motion in play, at one point on the orbit.

    ``kin`` is ``p - a``, the kinetic momentum and the physically correct one; ``can`` is
    the canonical momentum the state vector stores; ``xt`` is ``p + a``, which is what
    xtrack's ``magnet_spin`` builds — the sign of ``a`` flipped rather than dropped.
    """
    ax, ay = sol.normalized_vector_potential(st[X], st[Y])
    return {
        "kin": direction_of_motion(st[PX] - ax, st[PY] - ay, st[DELTA]),
        "can": direction_of_motion(st[PX], st[PY], st[DELTA]),
        "xt": direction_of_motion(st[PX] + ax, st[PY] + ay, st[DELTA]),
    }


def test_the_direction_of_motion_is_the_trajectorys_tangent() -> None:
    r"""The kinetic momentum reproduces the tangent; the other two miss by ``|a|``, ``2|a|``.

    Gate 2 (a), and the milestone's sharpest arbiter-free statement. The residuals are
    asserted by their **order in the transverse amplitude**, not by their size: the
    kinetic one converges as the *cube* (it is the paraxial remainder S1 already recorded
    and left open), while a wrong momentum leaves a residual **linear** in the amplitude,
    because ``a`` is. An order gate discriminates where a tolerance would not.

    Measured at the top amplitude: ``6.3e-11``, ``5.46e-4`` and ``1.09e-3``, with the last
    exactly twice the middle one — ``|kin - xt| = 2|a|`` to every digit, which is the
    signature of a sign flip rather than a dropped term.
    """
    ref = electron(g=0.0)
    residuals: dict[str, list[float]] = {"kin": [], "can": [], "xt": []}
    for f in (1.0, 0.5, 0.25, 0.125):
        st = STATE * np.array([f, f, f, f, 1.0, 1.0])
        sol = Solenoid(LENGTH, KS)
        mid = Solenoid(0.5 * LENGTH, KS).track(st.copy(), ref)
        tangent = _tangent(LENGTH, KS, st, ref, 1e-5)
        for name, vec in _candidates(sol, mid).items():
            residuals[name].append(float(np.abs(vec - tangent).max()))

    # the kinetic momentum: cubic in the amplitude, i.e. S1's paraxial remainder
    for coarse, fine in zip(residuals["kin"], residuals["kin"][1:], strict=False):
        assert coarse / fine == pytest.approx(8.0, rel=0.05)
    assert residuals["kin"][0] < 1e-10

    # the other two: linear, because the thing they get wrong is linear
    for name in ("can", "xt"):
        for coarse, fine in zip(residuals[name], residuals[name][1:], strict=False):
            assert coarse / fine == pytest.approx(2.0, rel=0.05)
        assert residuals[name][0] > 1e-4

    # ...and xtrack's is exactly twice the canonical one's, which is 2|a| against |a|.
    # The sign is the point: ``kin`` is ``p - a`` and ``xt`` is ``p + a``, so their
    # difference is **minus** twice the potential. xtrack does not drop the term, it
    # points it the wrong way, and that is what "2|a| rather than |a|" means.
    sol = Solenoid(LENGTH, KS)
    mid = Solenoid(0.5 * LENGTH, KS).track(STATE.copy(), ref)
    cand = _candidates(sol, mid)
    ax, ay = sol.normalized_vector_potential(mid[X], mid[Y])
    minus_two_a = -2.0 * np.array([float(ax), float(ay)]) / (1.0 + mid[DELTA])
    np.testing.assert_allclose((cand["kin"] - cand["xt"])[:2], minus_two_a, rtol=1e-12, atol=0.0)


# ================= gate 2 (b): the shipped function, with no coefficient ==================


def _applied_rotation(element, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
    """The 3x3 rotation ``track_with_spin`` actually applies, by carrying three basis spins.

    A spin map is linear in the spin, so the images of ``x``, ``y``, ``z`` *are* the
    matrix — the same device :func:`accsim.spin.spin_one_turn_matrix` uses for a ring, with
    no differencing step anywhere.
    """
    bunch = np.tile(np.asarray(state, dtype=float)[:, None], (1, 3))
    _, matrix = element.track_with_spin(bunch, np.eye(3), ref)
    return matrix


def _rotation_axis(matrix: np.ndarray) -> np.ndarray:
    """The unit axis of a 3x3 rotation, from its antisymmetric part.

    Valid while the rotation angle is in ``(0, pi)``, which every solenoid below is:
    ``|Omega| l = 0.54 rad``.
    """
    axis = 0.5 * np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    return axis / np.linalg.norm(axis)


def _transverse_direction(v: np.ndarray) -> np.ndarray:
    t = np.asarray(v, dtype=float)[:2]
    return t / np.linalg.norm(t)


@pytest.mark.parametrize("length", [0.9, 0.45, 0.225, 0.1125, 0.05625])
def test_the_precession_axis_points_along_the_motion_and_no_coefficient_says_so(
    length: float,
) -> None:
    r"""Gate 2 (b): the end-to-end statement, with the BMT coefficients cancelled out.

    For a purely longitudinal field ``b = ks s_hat``,

        ``Omega = -(ks/(1+delta)) [ (1 + G gamma) s_hat - G (gamma - 1) i_z i_hat ]``,

    so ``Omega``'s **transverse** part is a multiple of ``i_hat``'s — and the multiple
    cancels when the direction is taken. The transverse direction of the precession axis
    is therefore the transverse direction of the motion **and nothing else**: no ``G``, no
    ``gamma``, no path length, no ``ks``. That is what makes this the one gate a test that
    re-implemented the formula could not pass, and it is a statement about *which
    momentum* and about nothing else in the module.

    It is an **identity**, not an order gate, and the reason is worth stating: the code
    averages the entry and exit momenta, while the tangent is taken at the midpoint. A
    solenoid rotates the transverse velocity rigidly, so the mean of the two endpoints
    *bisects* them — and so does the midpoint velocity. The averaging costs magnitude
    (exactly ``cos(K L)``, gated below) and nothing at all in direction.

    Measured ``1.2e-13`` at ``L = 0.9``, rising to ``6.3e-11`` at ``L = 0.056`` — rising,
    because that is the **central difference's** own round-off (a shorter magnet means a
    smaller chord for the same step) and not a model residual. The bound below sits an
    order above it and thirteen orders below the thing it has to separate: the canonical
    and xtrack momenta land ``1.4``-``1.6`` away, out of a maximum of ``2`` for two unit
    vectors, i.e. roughly a right angle out. Order unity, not a tolerance.
    """
    ref = electron()
    sol = Solenoid(length, KS)
    out = sol.track(STATE.copy(), ref)

    axis = _rotation_axis(_applied_rotation(sol, STATE, ref))
    tangent = _transverse_direction(_tangent(length, KS, STATE, ref, 1e-5))
    assert float(np.abs(_transverse_direction(axis) - tangent).max()) < 1e-10

    # and the two wrong momenta are order unity away from it
    mid_x, mid_y = 0.5 * (STATE[X] + out[X]), 0.5 * (STATE[Y] + out[Y])
    mean = np.array([0.5 * (STATE[PX] + out[PX]), 0.5 * (STATE[PY] + out[PY])])
    a = np.array([float(v) for v in sol.normalized_vector_potential(mid_x, mid_y)])
    for wrong in (mean, mean + a):
        assert float(np.abs(_transverse_direction(wrong) - tangent).max()) > 1.4


def test_the_endpoint_average_costs_magnitude_only_and_converges_as_l_squared() -> None:
    r"""``|mean| / |midpoint| = cos(K L)`` exactly, and the error falls as ``L^2``.

    The quadrature half of gate 2, kept separate from the momentum half because they are
    different claims and the roadmap's first draft conflated them. The mean of two vectors
    related by a rotation through ``2 K L`` is the bisector shortened by ``cos(K L)``;
    measured ``0.9637708963658908`` against ``cos(K L) = 0.9637708963658905``. It is the
    midpoint rule's own second-order error and not a defect — the same ``O(L^2)`` N1
    already records for every element — and it is invisible to gate 2 (b) because it is
    purely a magnitude.
    """
    ref = electron(g=0.0)
    sol = Solenoid(LENGTH, KS)
    out = sol.track(STATE.copy(), ref)
    mid = Solenoid(0.5 * LENGTH, KS).track(STATE.copy(), ref)

    def kinetic(st: np.ndarray) -> np.ndarray:
        ax, ay = Solenoid(LENGTH, KS).normalized_vector_potential(st[X], st[Y])
        return np.array([st[PX] - float(ax), st[PY] - float(ay)])

    ratio = np.linalg.norm(0.5 * (kinetic(STATE) + kinetic(out))) / np.linalg.norm(kinetic(mid))
    assert ratio == pytest.approx(math.cos(0.5 * KS * LENGTH), rel=1e-14)

    residuals = []
    for length in (0.9, 0.45, 0.225, 0.1125):
        end = Solenoid(length, KS).track(STATE.copy(), ref)
        centre = Solenoid(0.5 * length, KS).track(STATE.copy(), ref)
        mean = 0.5 * (kinetic(STATE) + kinetic(end))
        i_mean = direction_of_motion(mean[0], mean[1], 0.0)
        residuals.append(float(np.abs(i_mean - _tangent(length, KS, STATE, ref, 1e-5)).max()))
        assert centre is not None  # the midpoint state the tangent is taken at
    for coarse, fine in zip(residuals, residuals[1:], strict=False):
        assert coarse / fine == pytest.approx(4.0, rel=0.15)
    assert residuals[0] == pytest.approx(1.61e-5, rel=0.05)


def test_the_two_candidate_models_differ_as_ks_squared_in_the_limit() -> None:
    r"""The ``ks`` exponent of the xtrack disagreement, measured where it can be clean.

    The reference suite gates this too, but it cannot gate it *sharply*: xtrack samples the
    momentum at the magnet's exit and accsim at the traversal mean, so their difference is
    ``2a`` plus an endpoint term of the same leading order, and the exponent only settles at
    small ``ks``. Going to small ``ks`` against xtrack drives the signal under that code's
    own quadrature gap. Between accsim's **two candidate models** there is no such floor, so
    the law can be taken to the limit here.

    Why quadratic: the transverse part of ``Omega`` is ``ks G (gamma - 1) i_z i_t /
    (1 + delta)``, so an error ``2a`` in the momentum tilts the precession axis by an amount
    proportional to ``ks`` (through ``a``) — and the axis is then multiplied by a rotation
    angle itself proportional to ``ks``. Measured **3.955, 3.978, 3.989**, converging on 4
    from below.
    """
    ref = electron()
    sol_state = STATE
    residuals = []
    for ks in (0.02, 0.01, 0.005):
        sol = Solenoid(LENGTH, ks)
        before = sol_state.copy()
        after = sol.track(before.copy(), ref)
        mid_x, mid_y = 0.5 * (before[X] + after[X]), 0.5 * (before[Y] + after[Y])
        ax, ay = (float(v) for v in sol.normalized_vector_potential(mid_x, mid_y))
        mean = np.array([0.5 * (before[PX] + after[PX]), 0.5 * (before[PY] + after[PY])])
        ax_out, ay_out = (float(v) for v in sol.normalized_vector_potential(after[X], after[Y]))
        bs = sol.longitudinal_field(mid_x, mid_y)

        p_ev = ref.momentum_eV * (1.0 + after[DELTA])
        energy = math.hypot(p_ev, ref.mass_eV)
        l_path = (p_ev / energy) / ref.beta0 * (LENGTH - (after[ZETA] - before[ZETA]))

        spin0 = np.array([0.2, 0.9, math.sqrt(1.0 - 0.04 - 0.81)])
        out = []
        for px, py in ((mean[0] - ax, mean[1] - ay), (after[PX] + ax_out, after[PY] + ay_out)):
            omega = precession_vector(0.0, 0.0, px, py, after[DELTA], ref, bs=bs)
            out.append(rotate(spin0, omega, l_path))
        residuals.append(float(np.abs(out[0] - out[1]).max()))

    ratios = [c / f for c, f in zip(residuals, residuals[1:], strict=False)]
    assert all(3.9 < r < 4.0 for r in ratios)
    assert ratios[-1] > ratios[0]  # converging on 4 from below


# ============== the early returns: a solenoid precesses and radiates at all ===============


def test_a_solenoid_precesses_even_though_its_transverse_field_is_zero() -> None:
    """The early-return gate, and the failure mode S1 refused to ship, asserted directly.

    Both consumers skip an element whose field is zero, and a solenoid's *transverse*
    field is exactly zero. A version of this milestone that added ``b_s`` to the physics
    but left ``if bx == 0 and by == 0: return`` in place would pass every coefficient gate
    above and still report **no precession in a spin rotator** — which is precisely the
    silent wrong answer the S1 refusal existed to prevent. So it is gated on its own.
    """
    ref = electron()
    sol = Solenoid(LENGTH, KS)
    bx, by = sol.normalized_field(STATE[X], STATE[Y])
    assert float(bx) == 0.0 and float(by) == 0.0  # the premise

    spin0 = np.array([1.0, 0.0, 0.0])
    _, spin = sol.track_with_spin(STATE.copy(), spin0.copy(), ref)
    assert float(np.linalg.norm(spin - spin0)) > 0.1  # ~0.44 at these settings
    assert float(np.linalg.norm(spin)) == pytest.approx(1.0, rel=0, abs=1e-15)


def test_an_on_axis_particle_radiates_exactly_zero_and_an_off_axis_one_does_not() -> None:
    r"""Gate 6: ``|b_perp| = ks |i_t|``, so the axis radiates nothing and off it grows as ``r``.

    A particle whose velocity is *along* the field feels no perpendicular field at all, so
    an on-axis solenoid traversal loses **exactly** zero energy — not a small number, an
    exact one. That half is the one a silent ``(0, 0)`` would have got right for the wrong
    reason, which is why the discriminating half is the second: a particle at radius ``r``
    with zero *canonical* momentum still has kinetic momentum ``|a| = ks r / 2``, so it
    radiates, and the loss goes as ``kappa^2 ~ (ks^2 r / 2)^2`` — **quadratic in the
    radius**. An implementation that fed the canonical momentum to the perpendicular field
    would report exactly zero there too, so this gate reads the vector potential and the
    longitudinal field at once.
    """
    ref = electron()
    sol = Solenoid(LENGTH, KS)

    on_axis = np.zeros(6)
    assert np.array_equal(
        sol.track(on_axis.copy(), ref, radiation="mean"), sol.track(on_axis.copy(), ref)
    )

    losses = []
    for r in (2.0e-3, 1.0e-3, 0.5e-3):
        st = np.zeros(6)
        st[X] = r
        out = sol.track(st.copy(), ref, radiation="mean")
        losses.append(-float(out[DELTA]))
    assert losses[0] > 0.0
    for coarse, fine in zip(losses, losses[1:], strict=False):
        assert coarse / fine == pytest.approx(4.0, rel=1e-3)


# ============================== the G = 0 control, and its blindness ======================


def test_g_zero_locks_the_spin_to_the_velocity_and_is_blind_to_which_momentum() -> None:
    r"""The control, *and* the measured proof that it is only a control.

    At ``G = 0`` the two BMT coefficients coincide, ``Omega = -b/(1 + delta)`` whatever the
    direction of motion is, and the parallel/perpendicular split drops out **entirely**.
    So the identity below — a spin started along the velocity comes out along it — is a
    genuine check on the sign of ``Omega``, its magnitude, and the path length, and it is
    *provably unable* to say anything about which momentum built ``i_hat``. The three
    candidates give **bit-identical** residuals, and that second half is what stops a
    future session from mistaking this for the discriminating gate. N1's catalogue of
    blind controls gains an entry.

    **What the third-order residual is, measured rather than assumed.** Integrated over the
    plain length ``L`` the identity is exact to round-off (``1.1e-19``): S1's map turns the
    transverse velocity through ``ks L / (1 + delta)`` and a rotation by the same angle
    reproduces it entrywise. The shipped function integrates over the **path length**
    instead, which is the correct Thomas-BMT statement — a spin precesses per unit of arc,
    and a longer trajectory precesses further — and the two differ by
    ``ks (l_path - L) = O(amplitude^2)`` in angle, hence ``O(amplitude^3)`` in the vector.
    So this residual **is** S1's recorded paraxial gap, seen from the spin side, and the
    cube is the gate. Matching the map instead would have been the easier number and the
    wrong physics.
    """
    ref = electron(g=0.0)
    sol = Solenoid(LENGTH, KS)

    residuals: dict[str, list[float]] = {"kin": [], "can": [], "xt": []}
    for f in (1.0, 0.5, 0.25, 0.125, 0.0625):
        st = STATE * np.array([f, f, f, f, 1.0, 1.0])
        out = sol.track(st.copy(), ref)
        ax0, ay0 = sol.normalized_vector_potential(st[X], st[Y])
        ax1, ay1 = sol.normalized_vector_potential(out[X], out[Y])
        v_in = direction_of_motion(st[PX] - ax0, st[PY] - ay0, st[DELTA])
        v_out = direction_of_motion(out[PX] - ax1, out[PY] - ay1, out[DELTA])

        mid_x, mid_y = 0.5 * (st[X] + out[X]), 0.5 * (st[Y] + out[Y])
        ax, ay = sol.normalized_vector_potential(mid_x, mid_y)
        mean = np.array([0.5 * (st[PX] + out[PX]), 0.5 * (st[PY] + out[PY])])
        bs = sol.longitudinal_field(mid_x, mid_y)
        # the same path length spin_precession integrates over, for the same reason
        p_ev = ref.momentum_eV * (1.0 + out[DELTA])
        energy = math.hypot(p_ev, ref.mass_eV)
        l_path = (p_ev / energy) / ref.beta0 * (LENGTH - (out[ZETA] - st[ZETA]))
        for name, p in (("kin", mean - [ax, ay]), ("can", mean), ("xt", mean + [ax, ay])):
            omega = precession_vector(0.0, 0.0, p[0], p[1], out[DELTA], ref, bs=bs)
            residuals[name].append(float(np.abs(rotate(v_in, omega, l_path) - v_out).max()))

    # the identity holds, and converges at third order in the amplitude
    for coarse, fine in zip(residuals["kin"], residuals["kin"][1:], strict=False):
        assert coarse / fine == pytest.approx(8.0, rel=0.05)
    assert residuals["kin"][0] < 4e-11

    # ...and it cannot see the question gate 2 answers: bit-identical, not merely close
    assert residuals["can"] == residuals["kin"]
    assert residuals["xt"] == residuals["kin"]


# ======================== the field is charge-free, as the map is =========================


def test_the_solenoid_field_and_everything_built_on_it_is_charge_free() -> None:
    """S1's gate 7 extended from ``matrix()`` to the field: ``b_s = ks``, with no ``q`` in it.

    A solenoid's coupling sense really is set by the charge, and the sign could plausibly
    have lived in either ``ks`` or the map. S1 probed both reference codes and found it
    lives in ``ks``; the field accessor must therefore carry no charge either, or the two
    halves of the element would disagree. Asserted through the *consequences* — a tracked
    spin and a radiated loss — rather than only on the accessor, since that is where a
    stray ``ref.charge`` would actually do damage.
    """
    negative = ReferenceParticle.from_total_energy(
        ELECTRON_MASS_EV, ENERGY, charge=-1.0, anomalous_moment=ELECTRON_ANOMALOUS_MOMENT
    )
    positive = ReferenceParticle.from_total_energy(
        ELECTRON_MASS_EV, ENERGY, charge=+1.0, anomalous_moment=ELECTRON_ANOMALOUS_MOMENT
    )
    sol = Solenoid(LENGTH, KS)
    spin0 = np.array([0.3, 0.6, -0.2])
    spin0 = spin0 / np.linalg.norm(spin0)

    a_state, a_spin = sol.track_with_spin(STATE.copy(), spin0.copy(), negative)
    b_state, b_spin = sol.track_with_spin(STATE.copy(), spin0.copy(), positive)
    assert np.array_equal(a_spin, b_spin)
    assert np.array_equal(a_state, b_state)
    assert np.array_equal(
        sol.track(STATE.copy(), negative, radiation="mean"),
        sol.track(STATE.copy(), positive, radiation="mean"),
    )

    # and a proton, whose G is a different number entirely, sees the same field
    proton = ReferenceParticle.from_total_energy(
        PROTON_MASS_EV, 5.0e11, charge=1.0, anomalous_moment=PROTON_ANOMALOUS_MOMENT
    )
    assert float(sol.longitudinal_field(1e-3, 2e-3)) == float(sol.longitudinal_field(1e-3, 2e-3))
    assert anomalous_moment(proton) != anomalous_moment(negative)  # the premise of "same field"
    assert float(Solenoid(LENGTH, KS).longitudinal_field(1e-3, 2e-3)) == KS


# ================================ gate 8: nothing else moves ==============================

#: Captured from the shipped code **before** S2 changed a line, so this is a genuine
#: before/after comparison rather than a restatement of what the code now does. Every
#: element in the package has ``b_s = 0`` and ``a = 0``, and the two new terms enter as
#: ``+ b_s * i_z`` and ``- 0.0``, which are exact for a float — so "unchanged" here means
#: **bit-identical**, and is asserted with ``array_equal`` rather than a tolerance.
#:
#: These literals are an artifact of the **environment they were captured in**, not a
#: property of the code: ``sin``/``cos``/``sqrt`` are not correctly rounded, so a different
#: C library rounds the last place differently and the same source produces a different
#: final bit. Captured on Windows/x86-64; the same run under glibc on CI moves
#: ``_PRE_S2_DIPOLE_SPIN`` by its last bit. Widening the check to an ulp budget would turn
#: an exact gate into a tolerance one — the move this repo forbids — and would need four
#: budgets nobody has measured, so the gate is **scoped to the capture platform** instead
#: (see ``docs/CONVENTIONS.md`` → *Capture-platform goldens*).
_PRE_S2_QUAD_SPIN = (0.4336529087249808, 0.8617890178428191, -0.2631821488624805)
_PRE_S2_DIPOLE_SPIN = (0.4622306774481255, 0.857348247733182, 0.22648793110161436)
_PRE_S2_DIPOLE_RADIATION = (
    0.0016453912505240938,
    -0.0006658626771694296,
    -0.0016334617302302449,
    -0.00043180702813049975,
    0.0008288988636001571,
    0.00048790321056878586,
)
_PRE_S2_QUAD_RADIATION = (
    0.0015721071672620286,
    -0.0014666085229884322,
    -0.001763673300211187,
    -0.0011159584508840398,
    0.0009996845345261903,
    0.000499987874441656,
)


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="the _PRE_S2_* literals are bit-exact captures from Windows/x86-64; another "
    "C library rounds sin/cos/sqrt differently in the last place",
)
def test_nothing_but_the_solenoid_moves_and_not_by_one_bit() -> None:
    """Gate 8, against numbers captured before the milestone rather than after it.

    Adding a term to a shared expression is the classic way to move every number in a
    package by a few ulp while every tolerance-based test stays green. The two new terms
    are exact at zero — ``x + 0.0 * i_z`` and ``x - 0.0`` both return ``x`` for any float —
    so nothing is *allowed* to move, and a re-association of the surrounding arithmetic
    that moved it would be caught here and nowhere else.

    Now that S2 has shipped, the literals no longer stand between a before and an after:
    they are a tripwire against a *future* re-association, and they can only be that on the
    machine they were captured on — hence the skip. See the note above the constants.
    """
    ref = electron()
    state = np.array([2.0e-3, 1.0e-4, -1.5e-3, 2.0e-4, 1.0e-3, 5.0e-4])
    spin0 = np.array([0.3, 0.6, -0.2])
    spin0 = spin0 / np.linalg.norm(spin0)

    quad = Quadrupole(0.6, 1.4)
    bend = Dipole(1.2, angle=0.09, k1=0.35)

    assert np.array_equal(
        quad.track_with_spin(state.copy(), spin0.copy(), ref)[1], np.array(_PRE_S2_QUAD_SPIN)
    )
    assert np.array_equal(
        bend.track_with_spin(state.copy(), spin0.copy(), ref)[1], np.array(_PRE_S2_DIPOLE_SPIN)
    )
    assert np.array_equal(
        bend.track(state.copy(), ref, radiation="mean"), np.array(_PRE_S2_DIPOLE_RADIATION)
    )
    assert np.array_equal(
        quad.track(state.copy(), ref, radiation="mean"), np.array(_PRE_S2_QUAD_RADIATION)
    )
