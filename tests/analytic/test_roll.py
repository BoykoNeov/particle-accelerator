r"""K2 — the rolled magnet, and the first *source* of vertical dispersion.

A **roll** turns a magnet about the beam axis while the machine stays where it is
(MAD-X ``EALIGN``'s ``DPSI``, xtrack's ``rot_s_rad_no_frame``). That is not the same
thing as a *design* tilt (MAD-X ``TILT``, xtrack's plain ``rot_s_rad``), which rolls
the reference frame along with the magnet; the frame-following version has **exactly
zero kick**, because the design orbit was rolled too. accsim implements the error, not
the design choice.

Two regimes, and the whole milestone is the second:

- **Straight elements.** The two rolls coincide and the map is the conjugation
  ``R(-phi) . body . R(+phi)``. Nothing here is new physics; it is checked against
  machinery that predates axis K, since a normal ``2(n+1)``-pole rolled by
  ``-pi/(2(n+1))`` *is* the skew one (J3's angle rule) — a quadrupole at ``-45 deg``
  is G1's skew quadrupole, a sextupole at ``-30 deg`` is J3's skew sextupole, both to
  machine precision.
- **A bending dipole.** A bend carries the reference frame around with it, so the
  rolled exit face is somewhere the lattice does not expect: displaced, pitched,
  yawed, with only part of the entrance rotation left to undo. This is exactly the
  curved-body geometry K1 declined for offsets, and it is what K2 implements.

The exact kick, derived in sympy here rather than recalled, is

    k_py = -sin(phi) sin(theta)                                  [EXACT]
    k_y  = -rho (1 - cos theta) sin(phi) / (sin^2 theta cos phi + cos^2 theta)
    k_px = +(1 - cos phi) sin(theta) cos(theta)                  [EXACT]

so the **vertical** effect is first order in the roll and the horizontal loss only
second — and the vertical effect is an *angle* **and** an *offset*. The opening
roadmap entry claimed ``theta sin(phi)`` for the angle and no offset at all; both are
falsified below, and the offset is not a refinement: dropping it gets the vertical
dispersion **wrong in sign**.

What is genuinely new is the **source vector**. A rolled bend is the first element in
this package whose *matrix* carries a vertical ``delta`` column: G1's skew quadrupole
only *rotates* dispersion the horizontal bends already made, and a
:class:`Corrector`'s matrix is the identity. So a rolled bend produces ``D_y`` in a
ring with no coupling element anywhere, which nothing before it could.

⚠️ **That is a narrower claim than "the first source of vertical dispersion", and
deliberately so.** In a real machine — and in xtrack — *any* vertical closed-orbit
angle makes vertical dispersion, because the exact map is ``y += L py / pz`` where
accsim's linear one is ``y += L py``. accsim is blind to that route: a vertically
steered ring gives ``D_y = 0`` here and ``2.1e-4`` in xtrack, with the two closed
orbits agreeing to eight digits. On the rolled test ring that route is the *larger*
of the two. It predates axis K (it belongs to the drift), it is measured and fully
accounted for in ``tests/reference/test_roll_xtrack.py``, and it is recorded at
CONVENTIONS.md -> *Orbit-driven vertical dispersion*.

The signs are pinned against xtrack in ``tests/reference/test_roll_xtrack.py``; this
file owns the closed forms and the consequences.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Corrector,
    CoupledLatticeError,
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    ThinQuadrupole,
    ThinSextupole,
    ThinSkewQuadrupole,
    ThinSkewSextupole,
    closed_orbit,
    closest_tune_approach,
    coupled_twiss,
    is_symplectic_map,
    is_symplectic_map_canonical,
    jacobian,
    linearised_lattice,
    normal_mode_tunes,
)
from accsim.coords import DELTA, DIM, PX, PY, ZETA, X, Y
from accsim.elements.alignment import arc_motion, frame_change, roll_motion, s_rotation
from accsim.symplectic import J6, from_canonical, to_canonical

# The bend the whole file measures on: big enough that sin(theta) and theta differ
# in the third digit, which is what separates the right kick formula from the wrong one.
L_BEND = 1.0
ANGLE = 0.3
RHO = L_BEND / ANGLE
ROLL = 0.02  # 20 mrad — far larger than any real error, so O(phi^2) terms are visible

# A dispersion-generating arc, the shape of tests/analytic/test_dispersion.py.
F_FOCAL = 2.5
N_CELLS = 8
SMALL_ROLL = 1.0e-3

# A generic probe state: every coordinate nonzero, so no term can hide.
STATE = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 1.0e-3, 2.0e-4])


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(938.27208816e6, 5.0)


def _arc(ref: ReferenceParticle, roll: float = 0.0, steer: float = 0.0) -> Lattice:
    """An ``N_CELLS`` arc FODO; the first bend carries ``roll``, or a steerer is added."""
    elems: list = []
    for i in range(N_CELLS):
        elems += [
            ThinQuadrupole(0.5 / F_FOCAL, name=f"qf_a{i}"),
            Dipole(L_BEND, ANGLE, roll=roll if i == 0 else 0.0, name=f"b1_{i}"),
            ThinQuadrupole(-1.0 / F_FOCAL, name=f"qd{i}"),
            Dipole(L_BEND, ANGLE, name=f"b2_{i}"),
            ThinQuadrupole(0.5 / F_FOCAL, name=f"qf_b{i}"),
        ]
    if steer != 0.0:
        elems.insert(1, Corrector(kick_y=steer, name="vsteer"))
    return Lattice(elems, ref)


def _delta_column(lattice: Lattice) -> np.ndarray:
    """The 4D dispersion **source** vector: the one-turn map's ``delta`` column."""
    return lattice.one_turn_matrix()[[X, PX, Y, PY], DELTA]


def _dispersion_from(m4: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Solve ``D = (I - M4)^-1 d`` — the same solve ``_matched_dispersion`` does."""
    return np.linalg.solve(np.eye(4) - m4, source)


# ---------------------------------------------------------------------------
# A. The geometry: a frame change, derived symbolically
# ---------------------------------------------------------------------------


def test_the_frame_change_linearisation_is_the_sympy_one() -> None:
    r"""``frame_change`` is the exact rigid-motion map, linearised — checked in sympy.

    The exact map is: move the point and the momentum into the new frame, drift back
    to its ``s = 0`` plane, and charge the *time* for that drift without charging the
    *design length* for it. That last step is the sign trap of the construction, and
    the reason this is derived rather than assembled from remembered pieces.

    Compared on a rigid motion with **all three** rotations and a translation, so no
    entry of the Jacobian is left untested by a degenerate choice.
    """
    sp = pytest.importorskip("sympy")
    x, px, y, py, z, d = sp.symbols("x px y py zeta delta", real=True)
    b0 = sp.symbols("beta0", positive=True)
    R = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"r{i}{j}", real=True))
    t = sp.Matrix(3, 1, lambda i, _: sp.Symbol(f"t{i}", real=True))

    # rvv = beta/beta0 at momentum p0 (1 + delta).
    g0 = 1 / sp.sqrt(1 - b0**2)
    pmc = (1 + d) * b0 * g0
    rvv = (pmc / sp.sqrt(pmc**2 + 1)) / b0

    pz = sp.sqrt((1 + d) ** 2 - px**2 - py**2)
    rr = R * sp.Matrix([x, y, 0]) + t
    pp = R * sp.Matrix([px, py, pz])
    ds = -rr[2]  # drift back to the old frame's transverse plane
    out = sp.Matrix(
        [
            rr[0] + ds * pp[0] / pp[2],
            pp[0],
            rr[1] + ds * pp[1] / pp[2],
            pp[1],
            z - ds * (1 + d) / (rvv * pp[2]),  # time only: no design length here
            d,
        ]
    )
    origin = {x: 0, px: 0, y: 0, py: 0, z: 0, d: 0}
    k_sym = out.subs(origin)
    M_sym = out.jacobian(sp.Matrix([x, px, y, py, z, d])).subs(origin)

    # A motion with yaw, pitch, roll and a translation, all distinct.
    ref = ReferenceParticle.from_gamma(938.27208816e6, 7.5)
    yaw, pitch, roll = 0.13, -0.07, 0.21
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, spitch = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rot = (
        np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[1, 0, 0], [0, cp, -spitch], [0, spitch, cp]])
        @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    )
    tvec = np.array([0.31, -0.17, 0.44])
    motion = np.eye(4)
    motion[:3, :3], motion[:3, 3] = Rot, tvec

    subs = {b0: sp.Float(ref.beta0, 30)}
    subs |= {R[i, j]: sp.Float(Rot[i, j], 30) for i in range(3) for j in range(3)}
    subs |= {t[i]: sp.Float(tvec[i], 30) for i in range(3)}
    want_k = np.array([float(v) for v in k_sym.subs(subs)])
    want_M = np.array([[float(M_sym[i, j].subs(subs)) for j in range(DIM)] for i in range(DIM)])

    got_M, got_k = frame_change(motion, ref)
    np.testing.assert_allclose(got_k, want_k, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(got_M, want_M, rtol=1e-14, atol=1e-15)


def test_the_frame_change_of_no_motion_is_exactly_the_identity(ref: ReferenceParticle) -> None:
    """The design case, asserted bit-for-bit: an aligned element pays nothing."""
    M, k = frame_change(np.eye(4), ref)
    assert np.array_equal(M, np.eye(DIM))
    assert np.array_equal(k, np.zeros(DIM))


def test_the_frame_change_of_a_pure_longitudinal_shift_is_a_drift(
    ref: ReferenceParticle,
) -> None:
    """Sliding the frame forward by ``ds`` is a ``Drift(-ds)`` plus a ``zeta`` refund.

    The independent check on step 3 of the construction. A real drift advances the
    design length *and* the particle; a frame change advances only the particle, so
    the two differ by exactly the constant ``ds`` — and that constant is the whole
    difference between ``zeta`` bookkeeping that closes and bookkeeping that drifts.
    """
    ds = 0.37
    motion = np.eye(4)
    motion[2, 3] = ds
    M, k = frame_change(motion, ref)
    want_M = np.eye(DIM)
    want_M[X, PX] = want_M[Y, PY] = -ds  # a Drift of length -ds; the class forbids one
    want_M[ZETA, DELTA] = -ds / ref.gamma0**2
    np.testing.assert_allclose(M, want_M, atol=1e-15)
    want_k = np.zeros(DIM)
    want_k[ZETA] = ds
    np.testing.assert_allclose(k, want_k, atol=1e-18)


def test_the_rolled_bends_exit_is_not_a_rotation_about_s() -> None:
    r"""``A^-1 R_s(phi) A`` is a rotation about a **tilted** axis — the whole of K2.

    If the exit were ``R_s(-phi)`` the roll would be a conjugation and the kick would
    be exactly zero, which is what the frame-following *design* tilt does. It is not:
    conjugating by the arc turns the rotation axis by the bend angle, so the exit
    carries a pitch ``sin(phi) sin(theta)`` and only ``arctan(cos(theta) tan(phi))``
    of roll, plus a translation. Derived in sympy from the two motions.
    """
    sp = pytest.importorskip("sympy")
    th, ph, rho = sp.symbols("theta phi rho", real=True)
    c, s, cp, spp = sp.cos(th), sp.sin(th), sp.cos(ph), sp.sin(ph)
    A = sp.Matrix([[c, 0, -s, rho * (c - 1)], [0, 1, 0, 0], [s, 0, c, rho * s], [0, 0, 0, 1]])
    Rr = sp.Matrix([[cp, -spp, 0, 0], [spp, cp, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T = sp.simplify(A.inv() * Rr * A)

    # The pitch component: what a pure s-rotation can never have.
    assert sp.simplify(T[1, 2] + spp * s) == 0
    # The roll that is left: cos(theta) of it, not all of it.
    assert sp.simplify(T[1, 0] - spp * c) == 0
    assert sp.simplify(T[1, 1] - cp) == 0
    # And it is a translation as well as a rotation.
    assert sp.simplify(T[1, 3] - rho * (c - 1) * spp) == 0

    # The package builds exactly this motion.
    vals = {th: sp.Float(ANGLE, 30), ph: sp.Float(ROLL, 30), rho: sp.Float(RHO, 30)}
    want = np.array([[float(T[i, j].subs(vals)) for j in range(4)] for i in range(4)])
    arc = arc_motion(ANGLE, RHO)
    got = np.linalg.solve(arc, roll_motion(ROLL) @ arc)
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-15)

    # ...and it is *not* the conjugation model, by a mile.
    naive = roll_motion(ROLL)
    assert np.max(np.abs(got - naive)) > 1e-3


# ---------------------------------------------------------------------------
# B. The kick a rolled bend produces — exact closed forms
# ---------------------------------------------------------------------------


def _kick_closed_form(theta: float, phi: float, rho: float) -> np.ndarray:
    """The K2 kick, from the sympy derivation in the test below."""
    den = math.sin(theta) ** 2 * math.cos(phi) + math.cos(theta) ** 2
    k = np.zeros(DIM)
    k[X] = rho * (1 - math.cos(phi)) * (1 - math.cos(theta)) * math.cos(theta) / den
    k[PX] = (1 - math.cos(phi)) * math.sin(theta) * math.cos(theta)
    k[Y] = -rho * (1 - math.cos(theta)) * math.sin(phi) / den
    k[PY] = -math.sin(phi) * math.sin(theta)
    k[ZETA] = -rho * (1 - math.cos(phi)) * (1 - math.cos(theta)) * math.sin(theta) / den
    return k


def test_the_rolled_bend_kick_is_the_derived_closed_form(ref: ReferenceParticle) -> None:
    r"""The whole kick, derived in sympy from the composed rigid motions.

    ``k_py = -sin(phi) sin(theta)`` comes out with **no** ``rho`` and **no**
    denominator — it is exact, and it is the entry that the opening roadmap claim
    (``theta sin(phi)``) got wrong.
    """
    sp = pytest.importorskip("sympy")
    th, ph, rho = sp.symbols("theta phi rho", real=True)
    c, s, cp, spp = sp.cos(th), sp.sin(th), sp.cos(ph), sp.sin(ph)
    A = sp.Matrix([[c, 0, -s, rho * (c - 1)], [0, 1, 0, 0], [s, 0, c, rho * s], [0, 0, 0, 1]])
    Rr = sp.Matrix([[cp, -spp, 0, 0], [spp, cp, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T = sp.simplify(A.inv() * Rr * A)
    Rot, t = T[:3, :3], T[:3, 3]
    a = t[2] / Rot[2, 2]
    den = s**2 * cp + c**2

    assert sp.simplify(Rot[1, 2] - (-spp * s)) == 0  # k_py
    assert sp.simplify((t[1] - a * Rot[1, 2]) - (-rho * (1 - c) * spp / den)) == 0  # k_y
    assert sp.simplify(Rot[0, 2] - (1 - cp) * s * c) == 0  # k_px
    assert sp.simplify((t[0] - a * Rot[0, 2]) - rho * (1 - cp) * (1 - c) * c / den) == 0  # k_x
    assert sp.simplify(a - (-rho * (1 - cp) * (1 - c) * s / den)) == 0  # k_zeta

    # The O(phi^2) entries lose four digits to cancellation against the O(1) geometry
    # they are built from, so 1e-10 is a round-off floor, not a physics allowance.
    got = Dipole(L_BEND, ANGLE, roll=ROLL).kick(ref)
    np.testing.assert_allclose(got, _kick_closed_form(ANGLE, ROLL, RHO), rtol=1e-10, atol=1e-20)


def test_the_vertical_kick_is_phi_sin_theta_and_not_theta_sin_phi(
    ref: ReferenceParticle,
) -> None:
    r"""The falsified roadmap claim, measured.

    ``sin(phi) sin(theta)`` is exact and symmetric in the two angles; ``theta sin(phi)``
    is the small-``theta`` limit of it. At ``theta = 0.3`` they differ by
    ``sin(theta)/theta - 1 = 1.5%``, which is 10 000 times the exactness of the closed
    form — so this is not a tolerance question.
    """
    got = Dipole(L_BEND, ANGLE, roll=ROLL).kick(ref)[PY]
    right = -math.sin(ROLL) * math.sin(ANGLE)
    wrong = -ANGLE * math.sin(ROLL)
    assert got == pytest.approx(right, rel=1e-13)
    assert abs(got - wrong) / abs(right) > 1.4e-2
    # ...and the discrepancy is exactly the ratio the two forms differ by.
    assert wrong / right == pytest.approx(ANGLE / math.sin(ANGLE), rel=1e-12)


def test_the_vertical_effect_is_an_offset_as_well_as_an_angle(ref: ReferenceParticle) -> None:
    """ "A small roll is a pure vertical bend" is false: there is a ``y`` term too.

    The arc's sagitta ``rho (1 - cos theta)``, tipped out of the plane by the roll.
    It is not a small correction to the angle — the two are different coordinates —
    and :func:`test_the_offset_term_flips_the_sign_of_the_vertical_dispersion` shows
    that dropping it gets the dispersion wrong in sign.
    """
    k = Dipole(L_BEND, ANGLE, roll=ROLL).kick(ref)
    assert k[Y] == pytest.approx(-RHO * (1 - math.cos(ANGLE)) * math.sin(ROLL), rel=1e-4)
    assert abs(k[Y]) > 1e-4  # not a rounding artefact: half a millimetre at 20 mrad


def test_the_vertical_effect_is_first_order_and_the_horizontal_loss_second(
    ref: ReferenceParticle,
) -> None:
    """Halving the roll halves ``(y, py)`` and quarters ``(x, px)`` — the orders, not
    a coefficient. This is what makes a roll a *vertical* error at leading order."""
    small, half = 1.0e-4, 5.0e-5
    ka = Dipole(L_BEND, ANGLE, roll=small).kick(ref)
    kb = Dipole(L_BEND, ANGLE, roll=half).kick(ref)
    for i in (Y, PY):
        assert ka[i] / kb[i] == pytest.approx(2.0, rel=1e-7)
    for i in (X, PX, ZETA):
        assert ka[i] / kb[i] == pytest.approx(4.0, rel=1e-7)


def test_a_rolled_bend_that_does_not_bend_is_just_a_rolled_drift(
    ref: ReferenceParticle,
) -> None:
    """At ``angle = 0`` the curved branch must collapse onto the straight one exactly.

    A zero-angle dipole is a drift, and a drift is rotationally symmetric about ``s``,
    so rolling it does nothing at all. Asserted bit-for-bit in both directions: the
    curved geometry must not leak into the straight case, and the straight case must
    not quietly acquire a kick.
    """
    rolled = Dipole(2.0, 0.0, roll=ROLL)
    assert np.array_equal(rolled.kick(ref), np.zeros(DIM))
    np.testing.assert_allclose(rolled.matrix(ref), Drift(2.0).matrix(ref), atol=1e-16)


def test_rolling_by_zero_leaves_a_bend_bit_for_bit_alone(ref: ReferenceParticle) -> None:
    """A design lattice must not pay a single bit for the machinery (K1's rule)."""
    a, b = Dipole(L_BEND, ANGLE), Dipole(L_BEND, ANGLE, roll=0.0)
    assert np.array_equal(a.matrix(ref), b.matrix(ref))
    assert np.array_equal(a.kick(ref), b.kick(ref))
    assert np.array_equal(a.track(STATE, ref), b.track(STATE, ref))


def test_a_bending_dipole_still_refuses_to_be_displaced_but_not_to_be_rolled(
    ref: ReferenceParticle,
) -> None:
    """K1's refusal was about the *translation* of a curved body, and it stands.

    K2 implements the curved geometry for the **roll**, which is what that refusal
    said was missing — but a translated curved body is a different rigid motion again
    and nothing needs it, so the offset is still refused. The two must not be confused
    into each other.
    """
    Dipole(L_BEND, ANGLE, roll=ROLL).kick(ref)  # allowed
    with pytest.raises(NotImplementedError, match="cannot displace the bending Dipole"):
        Dipole(L_BEND, ANGLE, dx=1e-4).kick(ref)
    with pytest.raises(NotImplementedError, match="cannot displace the bending Dipole"):
        Dipole(L_BEND, ANGLE, dy=1e-4, roll=ROLL).kick(ref)


# ---------------------------------------------------------------------------
# C. Straight elements: the roll is a conjugation, and it is already known physics
# ---------------------------------------------------------------------------


def test_a_quadrupole_rolled_by_45_degrees_is_g1s_skew_quadrupole(
    ref: ReferenceParticle,
) -> None:
    r"""J3's angle rule ``pi/(2(n+1))`` at ``n = 1``, checked against G1's element.

    The **sign** is the content: with accsim's conventions a roll of ``-45 deg``
    gives ``+k1sl``, so a ``+45 deg`` roll gives the *negative*-strength skew
    quadrupole. That is exactly the kind of relative sign this package has had to
    pin by measurement rather than argument, and here G1's element — validated long
    before axis K — is the measurement.
    """
    k1l = 0.37
    np.testing.assert_allclose(
        ThinQuadrupole(k1l, roll=-math.pi / 4).matrix(ref),
        ThinSkewQuadrupole(k1l).matrix(ref),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        ThinQuadrupole(k1l, roll=+math.pi / 4).matrix(ref),
        ThinSkewQuadrupole(-k1l).matrix(ref),
        atol=1e-15,
    )


def test_a_sextupole_rolled_by_30_degrees_is_j3s_skew_sextupole(
    ref: ReferenceParticle,
) -> None:
    """The same rule at ``n = 2``, against the element J3 shipped — tracked, because a
    thin sextupole's *matrix* is the identity and would agree for the wrong reason."""
    k2l = 7.0
    for amp in (STATE, -0.5 * STATE):
        np.testing.assert_allclose(
            ThinSextupole(k2l, roll=-math.pi / 6).track(amp, ref),
            ThinSkewSextupole(k2l).track(amp, ref),
            atol=1e-17,
        )


def test_a_roll_is_exactly_a_conjugation_for_every_straight_element(
    ref: ReferenceParticle,
) -> None:
    """``M = R(-phi) M_body R(+phi)`` — and the two rotations are exact inverses.

    This is the statement that stops being true for a bend, so it is asserted for the
    straight elements rather than assumed for all of them.
    """
    np.testing.assert_allclose(s_rotation(ROLL) @ s_rotation(-ROLL), np.eye(DIM), atol=1e-16)
    for aligned, rolled in [
        (Quadrupole(0.4, 1.7), Quadrupole(0.4, 1.7, roll=ROLL)),
        (ThinQuadrupole(0.5), ThinQuadrupole(0.5, roll=ROLL)),
        (Dipole(1.0, 0.0, k1=0.9), Dipole(1.0, 0.0, k1=0.9, roll=ROLL)),
        (Corrector(kick_x=1e-4, kick_y=-2e-4), Corrector(kick_x=1e-4, kick_y=-2e-4, roll=ROLL)),
    ]:
        want = s_rotation(-ROLL) @ aligned.matrix(ref) @ s_rotation(ROLL)
        np.testing.assert_allclose(rolled.matrix(ref), want, atol=1e-15)


def test_a_drift_is_exactly_roll_invariant(ref: ReferenceParticle) -> None:
    """Nothing to be off-axis of, and nothing to be turned: a drift is symmetric."""
    np.testing.assert_allclose(Drift(2.0, roll=0.7).matrix(ref), Drift(2.0).matrix(ref), atol=1e-16)
    assert np.array_equal(Drift(2.0, roll=0.7).kick(ref), np.zeros(DIM))


def test_a_rolled_corrector_rotates_its_kick(ref: ReferenceParticle) -> None:
    """The one element that is *only* a kick: rolling it must turn the kick and
    nothing else. A horizontal steerer rolled by 90 degrees is a vertical one, and
    which way up is a convention — pinned against xtrack in the reference suite, not
    argued here."""
    c = Corrector(kick_x=3e-4, roll=math.pi / 2)
    k = c.kick(ref)
    assert k[PY] == pytest.approx(+3e-4, rel=1e-12)
    assert abs(k[PX]) < 1e-19
    np.testing.assert_allclose(c.matrix(ref), np.eye(DIM), atol=1e-16)


def test_the_rolled_map_is_still_symplectic(ref: ReferenceParticle) -> None:
    """A rotation is a symplectomorphism and so is a rigid frame change, so this
    passes by construction — which is exactly why it is worth running: it is the
    check that would catch the entry and exit halves being applied inconsistently,
    which no amount of dispersion agreement would reveal.

    The elements whose ``track`` is still linear in ``delta`` are checked in accsim's
    ``(zeta, delta)``; those whose map is exact in ``delta`` need the **canonical**
    check. Which is which has now moved three times: L2 put the thick :class:`Quadrupole`
    in the second group, L3 put the **pure sector** :class:`Dipole` there, and L4 has just
    moved the **combined-function** bend across as well. That last one is a live change to
    this test rather than a note — before L4 a gradient bend's ``track`` was its
    ``matrix``, so it sat in the first group and passed there; its map is now L4's
    expanded one, which is exact in ``delta``, and the plain check rejects it.

    What is left in the first group are the genuinely ``delta``-linear maps: the thin
    multipoles, whose kicks carry no rigidity factor at all.

    The distinction is not a formality, and the two rolled elements below show the two
    different ways it bites. In ``(zeta, delta)`` the rolled quadrupole's residual is
    ``1.7e-9``, which clears the ``1e-8`` used here by a factor of six — so it would go
    on *passing*, for a reason unconnected to symplecticity, until an unlucky ``STATE``
    made it fail. The rolled exact bend's residual is ``1.9e-6``, so the same check
    **rejects a correct map** outright. Neither verdict from it means anything on its
    own. See ``accsim/symplectic.py``'s module docstring.
    """
    for elem in (
        ThinSextupole(7.0, roll=ROLL),
        ThinOctupole(40.0, roll=ROLL),
    ):
        assert is_symplectic_map(lambda s, e=elem: e.track(s, ref), STATE, atol=1e-8)

    rolled_quad = Quadrupole(0.4, 1.7, roll=ROLL)
    assert is_symplectic_map_canonical(lambda s: rolled_quad.track(s, ref), STATE, ref)
    assert is_symplectic_map_canonical(lambda s: Dipole(L_BEND, ANGLE).track(s, ref), STATE, ref)

    # The combined-function bend, in its new group — and rejected by its old one, which
    # is the assertion that says the move was necessary rather than tidy.
    cf = Dipole(L_BEND, ANGLE, k1=0.6, e1=0.1, e2=0.1)
    assert is_symplectic_map_canonical(lambda s: cf.track(s, ref), STATE, ref)
    assert not is_symplectic_map(lambda s: cf.track(s, ref), STATE, atol=1e-8)

    # ...and the wrong check's two failure modes, pinned, so the switch is not folklore.
    assert is_symplectic_map(lambda s: rolled_quad.track(s, ref), STATE, atol=1e-8)
    assert not is_symplectic_map(lambda s: rolled_quad.track(s, ref), STATE, atol=1e-10)
    rolled_bend = Dipole(L_BEND, ANGLE, roll=ROLL)
    assert not is_symplectic_map(lambda s: rolled_bend.track(s, ref), STATE, atol=1e-8)


def test_a_rolled_bend_is_symplectic_only_to_first_order_in_the_roll(
    ref: ReferenceParticle,
) -> None:
    r"""A cost of L3 that K2 could not have seen, measured rather than assumed.

    :func:`~accsim.elements.alignment.frame_change` returns the **affine linearisation
    about the origin** of the true frame change — its own docstring says so, and adds
    that it "is exact for accsim's linear elements". It was, and now it is not: L3 made
    the pure sector bend's body an *exact* map, and conjugating an exact map by a
    linearised frame change is no longer exactly symplectic.

    Every part of that sentence is checked below, because it would otherwise be a story:

    - the **aligned** exact bend is symplectic to ``3.7e-13`` — the body is not the
      problem;
    - the two frame-change matrices are each symplectic in their own right to
      ``3.3e-16`` — the linearisation is not *wrong*, it is only a linearisation;
    - a rolled **straight** dipole, whose alignment is a plain rotation rather than the
      curved rigid motion, is symplectic to ``2e-13`` — so it is the curved frame change
      and nothing else;
    - the rolled bending dipole's residual is ``4.7e-8`` and **first order in the roll**,
      halving with it over four halvings. A second-order residual would point at the
      body; first order points at the frame change, which is where the linearisation is.

    ``matrix()`` and ``kick()`` are unaffected — they are linear by construction, so the
    linearised frame change is exact *for them*, and every K2 number stands. What this
    bounds is tracking a rolled bend for many turns, where a non-symplectic map at
    ``5e-8`` per element is a slow leak rather than a wrong answer. Making the frame
    change nonlinear in ``track`` would close it and is not this milestone.

    **L4 extends the cost to the combined-function bend, and by the same mechanism.**
    Before L4 a rolled gradient bend was exactly linear in ``track`` and so exactly
    symplectic; its body is now L4's expanded map, and the curved frame change is still
    the affine linearisation, so it degrades the same way. Every arm below is run for both
    bodies, which is what says the cause is shared rather than assumed to be.
    """
    aligned = Dipole(L_BEND, ANGLE)
    assert is_symplectic_map_canonical(lambda s: aligned.track(s, ref), STATE, ref)
    assert is_symplectic_map_canonical(
        lambda s: Dipole(L_BEND, ANGLE, k1=0.6).track(s, ref), STATE, ref
    )

    straight_rolled = Dipole(L_BEND, 0.0, roll=ROLL)
    assert is_symplectic_map_canonical(lambda s: straight_rolled.track(s, ref), STATE, ref)
    assert is_symplectic_map_canonical(
        lambda s: Dipole(L_BEND, 0.0, k1=0.6, roll=ROLL).track(s, ref), STATE, ref
    )

    bend = Dipole(L_BEND, ANGLE, roll=ROLL)
    for M, _k in (bend._alignment_entry(ref), bend._alignment_exit(ref)):
        assert np.abs(M.T @ J6 @ M - J6).max() < 1e-14

    def residual(roll: float, k1: float = 0.0) -> float:
        elem = Dipole(L_BEND, ANGLE, k1=k1, roll=roll)

        def canonical(c: np.ndarray) -> np.ndarray:
            return to_canonical(elem.track(from_canonical(np.asarray(c), ref), ref), ref)

        J = jacobian(canonical, to_canonical(STATE, ref))
        return float(np.abs(J.T @ J6 @ J - J6).max())

    residuals = [residual(ROLL / 2**k) for k in range(4)]
    assert residuals[0] == pytest.approx(4.73e-8, rel=2e-2)
    for big, small in zip(residuals[:-1], residuals[1:], strict=True):
        assert big / small == pytest.approx(2.0, rel=1e-2)  # first order, not second

    combined = [residual(ROLL / 2**k, k1=0.6) for k in range(4)]
    assert combined[0] == pytest.approx(6.22e-8, rel=2e-2)  # same size, same mechanism
    for big, small in zip(combined[:-1], combined[1:], strict=True):
        assert big / small == pytest.approx(2.0, rel=1e-2)


def test_a_roll_broadcasts_over_a_bunch(ref: ReferenceParticle) -> None:
    """Tracking a ``(6, n)`` bunch must equal tracking each particle on its own."""
    bunch = np.stack([STATE, -0.5 * STATE, np.zeros(DIM)], axis=1)
    elem = Dipole(L_BEND, ANGLE, roll=ROLL)
    got = elem.track(bunch, ref)
    for j in range(bunch.shape[1]):
        np.testing.assert_allclose(got[:, j], elem.track(bunch[:, j], ref), atol=1e-18)


def test_repr_reports_the_roll() -> None:
    """A printed lattice must never look aligned while its orbit says otherwise."""
    assert "roll=0.02" in repr(Dipole(L_BEND, ANGLE, roll=ROLL))
    assert "roll" not in repr(Dipole(L_BEND, ANGLE))
    assert "roll=0.02" in repr(ThinQuadrupole(0.5, roll=ROLL))


# ---------------------------------------------------------------------------
# D. The milestone: a rolled bend is a *source* of vertical dispersion
# ---------------------------------------------------------------------------


def test_a_rolled_bend_makes_vertical_dispersion_with_no_coupling_element_anywhere(
    ref: ReferenceParticle,
) -> None:
    """The discriminating statement, and the one the opening roadmap control missed.

    "Remove the bends to kill the horizontal dispersion" removes the *rolled* bend
    too, so it kills both routes and proves nothing. What does discriminate: this
    ring contains no skew quadrupole and no other coupling element, so before K2
    there was no route to ``D_y`` at all — and the aligned ring gives exactly zero.
    """
    aligned, rolled = _arc(ref), _arc(ref, roll=SMALL_ROLL)
    assert not any(type(e).__name__.startswith("ThinSkew") for e in rolled.elements)
    assert _delta_column(aligned)[2] == 0.0  # exactly, not to tolerance
    assert _delta_column(aligned)[3] == 0.0
    assert coupled_twiss(aligned).disp_y == 0.0

    tw = coupled_twiss(rolled)
    assert abs(tw.disp_y) > 1e-5
    assert abs(tw.disp_x) > 1.0  # the horizontal dispersion is unharmed


def test_the_rolled_bend_adds_to_the_source_vector_where_a_skew_quad_does_not(
    ref: ReferenceParticle,
) -> None:
    """K2's actual claim, at exact zero.

    A skew quadrupole *rotates* dispersion the bends already made — its own
    contribution to the ``delta`` column is identically nothing, because its matrix
    has no ``delta`` column at all. Put one in a ring with no bends and the source is
    exactly zero; put a rolled bend in and it is not.
    """
    skew_only = Lattice([ThinSkewQuadrupole(0.05), Drift(1.0), ThinQuadrupole(0.3)], ref)
    assert np.array_equal(_delta_column(skew_only), np.zeros(4))

    source = _delta_column(_arc(ref, roll=SMALL_ROLL))
    assert source[2] != 0.0 and source[3] != 0.0


def test_the_offset_term_flips_the_sign_of_the_vertical_dispersion(
    ref: ReferenceParticle,
) -> None:
    r"""Why "a roll is a pure vertical bend" is not a harmless simplification.

    Solve ``D = (I - M4)^-1 d`` from the ring's own source, then again with the
    **position** half of the vertical source deleted — which is precisely the
    pure-vertical-bend model. The answer does not merely shrink: it changes sign.
    Deleting the *angle* half instead costs 5%.
    """
    lat = _arc(ref, roll=SMALL_ROLL)
    m4 = lat.one_turn_matrix()[np.ix_([X, PX, Y, PY], [X, PX, Y, PY])]
    source = _delta_column(lat)

    full = _dispersion_from(m4, source)
    no_offset = _dispersion_from(m4, np.where(np.arange(4) == 2, 0.0, source))
    no_angle = _dispersion_from(m4, np.where(np.arange(4) == 3, 0.0, source))

    assert full[2] == pytest.approx(coupled_twiss(lat).disp_y, rel=1e-12)
    assert np.sign(no_offset[2]) != np.sign(full[2])  # the angle-only model, wrong sign
    assert abs(no_angle[2] / full[2] - 1.0) < 0.10  # the offset alone is within 10%


def test_the_vertical_dispersion_is_first_order_in_the_roll(ref: ReferenceParticle) -> None:
    """Doubling the roll doubles ``D_y`` — the order, which no prefactor can fake."""
    d1 = coupled_twiss(_arc(ref, roll=SMALL_ROLL)).disp_y
    d2 = coupled_twiss(_arc(ref, roll=2 * SMALL_ROLL)).disp_y
    d4 = coupled_twiss(_arc(ref, roll=4 * SMALL_ROLL)).disp_y
    assert d2 / d1 == pytest.approx(2.0, rel=1e-5)
    assert d4 / d1 == pytest.approx(4.0, rel=1e-5)


def test_in_accsims_linear_model_a_vertical_steerer_adds_nothing_to_the_source(
    ref: ReferenceParticle,
) -> None:
    r"""A statement about **accsim's linear model**, not about the physics.

    A :class:`Corrector`'s matrix is the identity and its kick carries no
    ``1/(1+delta)``, so it contributes *identically nothing* to the ``delta`` column:
    accsim returns ``D_y = 0`` exactly for a vertically steered ring. That is what
    makes it useless as the calibration the opening roadmap entry wanted — a steerer
    cannot be tuned to any ``D_y`` at all.

    ⚠️ **It is not a physical control.** Measured against xtrack the same ring gives
    ``dy = 2.1e-4``: the exact map's ``y += L py / pz`` turns any vertical orbit
    *angle* into vertical dispersion, and accsim's ``y += L py`` cannot. Both closed
    orbits agree to eight digits, so the difference is entirely the momentum
    dependence of the maps. That blind spot predates axis K (it is the drift's, and
    K2 only makes it consequential); it is measured in
    ``tests/reference/test_roll_xtrack.py`` and recorded at CONVENTIONS.md ->
    *Orbit-driven vertical dispersion*, and it is why K2's claim is the narrow one:
    a rolled bend is the first element whose **matrix** has a vertical ``delta``
    column, not the only way a real machine gets ``D_y``.
    """
    steered = _arc(ref, steer=-2.2e-4)
    assert abs(closed_orbit(steered)[Y]) > 1e-6  # there *is* a vertical orbit
    assert np.array_equal(_delta_column(steered), _delta_column(_arc(ref)))
    assert coupled_twiss(steered).disp_y == 0.0

    rolled = _arc(ref, roll=SMALL_ROLL)
    assert abs(closed_orbit(rolled)[Y]) > 1e-6  # a rolled bend moves the source itself
    assert coupled_twiss(rolled).disp_y != 0.0


def test_a_rolled_element_still_responds_to_a_displacement_as_one_minus_m_times_d(
    ref: ReferenceParticle,
) -> None:
    r"""K1's response formula survives a roll, which is not obvious and is used.

    With a roll the kick is ``M_out (M_body k_in + k_body) + k_out``, and substituting
    ``k_in = -M_in d``, ``k_out = d`` collapses it back to ``(I - matrix) d`` — with
    ``matrix`` the *rolled* one. :func:`accsim.orbit.misalignment_response` and
    ``_default_sources`` both rely on that, since they read ``elem.matrix(ref)``
    without knowing whether it is rolled. Checked against the orbit the package
    actually solves for, not against the algebra that produced it.
    """
    d = 3.0e-4
    lat = Lattice(
        [ThinQuadrupole(0.4, roll=0.3, name="q"), Drift(1.0), ThinQuadrupole(-0.35), Drift(1.0)],
        ref,
    )
    elems = list(lat.elements)
    rolled = ThinQuadrupole(0.4, roll=0.3, dx=d, name="q")
    displaced = Lattice([rolled] + elems[1:], ref)

    want = (np.eye(DIM) - rolled.matrix(ref)) @ rolled.offset()
    np.testing.assert_allclose(rolled.kick(ref), want, atol=1e-18)
    # The response is genuinely there in both planes: a rolled quad couples, so a
    # horizontal displacement moves the *vertical* orbit too — which an unrolled one
    # cannot do (CONVENTIONS -> offsets cannot couple the planes).
    co = closed_orbit(displaced)
    assert abs(co[X]) > 1e-6
    assert abs(co[Y]) > 1e-6
    assert np.array_equal(closed_orbit(lat), np.zeros(4))


def test_the_new_dispersion_reaches_the_beam_size(ref: ReferenceParticle) -> None:
    """``coupled_beam_sigma`` already reads ``D_y``, so a new source needs no plumbing —
    asserted rather than assumed, because "it should just work" is how a term goes
    missing."""
    from accsim import coupled_beam_sigma, propagate_coupled_twiss

    def sigma_y(lat: Lattice) -> float:
        tw = propagate_coupled_twiss(lat)
        return max(coupled_beam_sigma(tw, emit_1=1e-8, emit_2=1e-12, sigma_delta=1e-3)[1])

    assert sigma_y(_arc(ref, roll=50 * SMALL_ROLL)) > 1.5 * sigma_y(_arc(ref))


# ---------------------------------------------------------------------------
# E. The wrong model, measured — and the blindness the roll creates
# ---------------------------------------------------------------------------


def test_the_conjugation_model_predicts_no_vertical_orbit_at_all(
    ref: ReferenceParticle,
) -> None:
    """The failure mode this milestone exists to avoid, built and measured.

    Treating a rolled bend as ``R(-phi) M R(+phi)`` — the *design tilt*, which is
    what "a simple rotation" means — gives a map with **exactly zero** kick, because
    an aligned bend has none and a rotation cannot create one. So it predicts no
    vertical closed orbit whatever, where the real thing gives one. The gap is not a
    tolerance: it is the entire effect.
    """
    rolled = Dipole(L_BEND, ANGLE, roll=ROLL)
    conjugated = s_rotation(-ROLL) @ Dipole(L_BEND, ANGLE).matrix(ref) @ s_rotation(ROLL)

    assert np.max(np.abs(rolled.kick(ref))) > 5e-3  # the real kick
    assert np.max(np.abs(rolled.matrix(ref) - conjugated)) > 5e-3  # and a different matrix
    # The conjugation model's own kick is identically zero — there is nothing to compare.
    lat = Lattice([Dipole(L_BEND, ANGLE, roll=ROLL)], ref)
    assert np.max(np.abs(lat.transfer_map()[1])) > 5e-3


def test_a_rolled_bend_couples_the_planes_and_the_type_walk_says_so(
    ref: ReferenceParticle,
) -> None:
    """The residual roll is real coupling, and ``closest_tune_approach`` refuses it.

    Entry rolls by ``phi``; the exit gives back only ``arctan(cos(theta) tan(phi))``,
    so ``phi (1 - cos theta)`` of roll is left over and the transverse blocks mix.
    ``closest_tune_approach`` sums over skew-quadrupole *elements*, so it cannot see
    this at all — and returning ``0.0`` for a demonstrably coupled ring is the kind
    of reassuring wrong answer this package refuses. :func:`normal_mode_tunes`
    diagonalises the map and does see it.
    """
    aligned, rolled = _arc(ref), _arc(ref, roll=SMALL_ROLL)
    off = rolled.one_turn_matrix()[np.ix_([X, PX], [Y, PY])]
    assert np.max(np.abs(off)) > 1e-6
    assert np.array_equal(aligned.one_turn_matrix()[np.ix_([X, PX], [Y, PY])], np.zeros((2, 2)))

    with pytest.raises(CoupledLatticeError, match="without being a skew quadrupole"):
        closest_tune_approach(rolled)
    assert closest_tune_approach(aligned) == 0.0

    # The eigenvalue path is not blind: the normal modes move.
    assert normal_mode_tunes(rolled) != normal_mode_tunes(aligned)


def test_the_coupling_is_first_order_in_the_roll_and_dies_with_the_bend_angle(
    ref: ReferenceParticle,
) -> None:
    r"""It is ``phi (1 - cos theta)`` of leftover roll, and the largest off-block entry
    comes out at exactly twice that — measured across two decades of bend angle, which
    is what pins the ``1 - cos theta`` rather than merely "it grows". The ``theta^2``
    is why nobody notices the coupling on a weak arc, and why it is asserted here
    instead of being discovered later as a mystery ``DeltaQ_min``."""

    def off_block(roll: float, angle: float) -> float:
        M = Dipole(L_BEND, angle, roll=roll).matrix(ref)
        return float(np.max(np.abs(M[np.ix_([X, PX], [Y, PY])])))

    assert off_block(2e-3, 0.05) / off_block(1e-3, 0.05) == pytest.approx(2.0, rel=1e-5)
    for angle in (0.02, 0.04, 0.08):
        want = 2.0 * 1e-3 * (1.0 - math.cos(angle))
        assert off_block(1e-3, angle) == pytest.approx(want, rel=1e-3)


def test_linearised_lattice_refuses_a_rolled_multipole(ref: ReferenceParticle) -> None:
    """Rolled higher multipoles are out of scope for K2, and the feed-down walk says
    so rather than silently emitting the unrolled split — K1's rule for a source a
    type-walking helper cannot see."""
    for elem in (ThinSextupole(3.0, roll=0.1), ThinOctupole(40.0, roll=0.1)):
        lat = Lattice([ThinQuadrupole(0.4), Drift(1.0), elem, Drift(1.0)], ref)
        with pytest.raises(NotImplementedError, match="cannot linearise the rolled"):
            linearised_lattice(lat)
