r"""K2 cross-check: the rolled magnet, against xtrack's ``rot_s_rad_no_frame``.

xtrack carries **two** rolls, and telling them apart is the milestone:

- ``rot_s_rad`` — a *design* tilt (MAD-X ``TILT``): the reference frame is rolled with
  the magnet. This one is the plain conjugation ``R(-phi) M R(+phi)`` and has
  **exactly zero** kick, because the design orbit was rolled too.
- ``rot_s_rad_no_frame`` — a roll **misalignment** (MAD-X ``EALIGN``'s ``DPSI``): the
  magnet turns and the machine does not. This is what accsim's ``roll`` is, and for a
  *bending* magnet it is not a conjugation at all.

On a straight element the two coincide, which is exactly why the sign cannot be pinned
on a quadrupole: a probe there would pass against either attribute and say nothing
about the bend. Both halves are therefore probed here, and the *contrast* between them
is asserted as a measured number rather than described.

The comparison is between **affine linearisations** ``(M, k)``: accsim's bend is a
linear element by construction, so what is being checked is xtrack's Jacobian and
constant part at the origin. The bar is set by the *aligned* pair — whatever accsim's
linear sector bend already differs from xtrack's exact bend by, the roll must not add
to. It does not: measured 2026-08-17, the rolled pair agrees to the same ``1e-9`` as
the aligned one, while the conjugation model misses by ``5.9e-3``.

Costs **two** ``xt.Line`` builds: every element under test lives in one line and is
tracked through by index, plus one ring for the ``twiss`` dispersion cross-check.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Corrector,
    Dipole,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    closed_orbit,
    coupled_twiss,
    propagate_coupled_twiss,
)
from accsim.coords import DELTA, DIM, PX, PY, X, Y
from accsim.elements.alignment import s_rotation
from accsim.twiss import coupled_twiss_on_orbit

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6
GAMMA0 = 5.0
L_BEND = 1.0
ANGLE = 0.3  # big enough that sin(theta) and theta differ in the third digit
RHO = L_BEND / ANGLE
ROLL = 0.02
K1L = 0.5
F_FOCAL = 2.5
N_CELLS = 8
RING_ANGLE = 0.15
RING_ROLL = 1.0e-3
STEER = 1.0e-4  # a vertical steerer, for the blind-spot control

STEP = 1.0e-7  # central-difference step for the Jacobian


def _bend(**kwargs) -> xt.Bend:
    """A pure sector bend: xtrack's edges off, to match accsim's default."""
    b = xt.Bend(length=L_BEND, angle=ANGLE, k1=0.0, **kwargs)
    b.edge_entry_active = 0
    b.edge_exit_active = 0
    return b


# The one line every map-level probe is taken from, laid out as (label, slice).
SEGMENTS: dict[str, tuple[int, int]] = {
    "aligned": (0, 1),
    "roll": (1, 2),
    "roll_flipped": (2, 3),
    "tilt": (3, 4),
    "conjugation": (4, 7),
    "thin_quad_roll": (7, 8),
}


def _elements() -> list:
    return [
        _bend(),
        _bend(rot_s_rad_no_frame=ROLL),
        _bend(rot_s_rad_no_frame=-ROLL),
        _bend(rot_s_rad=ROLL),
        xt.Rotation(rot_s_rad=ROLL),
        _bend(),
        xt.Rotation(rot_s_rad=-ROLL),
        xt.Multipole(knl=[0.0, K1L], length=0.0, rot_s_rad_no_frame=ROLL),
    ]


@pytest.fixture(scope="module")
def maps() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{label: (M, k)}`` for every segment, from a single JIT build."""
    line = xt.Line(elements=_elements())
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")

    # 13 states: the origin, then +/- one step along each coordinate.
    states = [np.zeros(DIM)]
    for i in range(DIM):
        for sign in (+1.0, -1.0):
            v = np.zeros(DIM)
            v[i] = sign * STEP
            states.append(v)
    st = np.array(states).T

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, (start, stop) in SEGMENTS.items():
        p = xt.Particles(
            mass0=MASS0,
            q0=1,
            gamma0=GAMMA0,
            x=st[X].copy(),
            px=st[PX].copy(),
            y=st[Y].copy(),
            py=st[PY].copy(),
            zeta=st[4].copy(),
            delta=st[DELTA].copy(),
        )
        line.track(p, ele_start=start, ele_stop=stop)
        order = np.argsort(p.particle_id)
        got = np.array(
            [p.x[order], p.px[order], p.y[order], p.py[order], p.zeta[order], p.delta[order]]
        )
        M = np.empty((DIM, DIM))
        for i in range(DIM):
            M[:, i] = (got[:, 1 + 2 * i] - got[:, 2 + 2 * i]) / (2.0 * STEP)
        out[label] = (M, got[:, 0])
    return out


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _accsim(elem, ref: ReferenceParticle) -> tuple[np.ndarray, np.ndarray]:
    return elem.matrix(ref), elem.kick(ref)


# ---------------------------------------------------------------------------
# A. The rolled bend: the curved-body geometry, against xtrack's own
# ---------------------------------------------------------------------------


def test_the_rolled_bend_agrees_with_xtrack_as_well_as_the_aligned_one(
    maps, ref: ReferenceParticle
) -> None:
    """The bar is the *aligned* pair — the roll must add no error of its own.

    accsim's sector bend is a linear map and xtrack's is exact, so the two already
    differ at the level of the finite-difference Jacobian. Whatever that residual is,
    a correct roll model must sit inside it rather than adding a term. It does.
    """
    Ma, ka = _accsim(Dipole(L_BEND, ANGLE), ref)
    Mr, kr = _accsim(Dipole(L_BEND, ANGLE, roll=ROLL), ref)
    Mxa, kxa = maps["aligned"]
    Mxr, kxr = maps["roll"]

    aligned_gap = max(np.max(np.abs(Ma - Mxa)), np.max(np.abs(ka - kxa)))
    rolled_gap = max(np.max(np.abs(Mr - Mxr)), np.max(np.abs(kr - kxr)))
    # Measured 2026-08-17: both 3.3063e-9 — the *same* number to five figures, which
    # is a far stronger statement than a bound would be. The residual is entirely the
    # pre-existing linear-map-vs-exact-map difference of the aligned bend; the roll
    # contributes nothing detectable to it. For contrast, the conjugation model misses
    # by 5.9e-3, six orders larger.
    assert aligned_gap < 1e-8
    assert rolled_gap == pytest.approx(aligned_gap, rel=1e-3)


def test_the_roll_sign_is_xtracks_and_flipping_it_is_visible(maps, ref: ReferenceParticle) -> None:
    """The sign has no analytic gate — the vertical kick is odd in the roll, so the
    wrong sense misses by exactly twice it. Pinned against ``rot_s_rad_no_frame``,
    which for a bend is the only attribute that could pin it at all."""
    _, kr = _accsim(Dipole(L_BEND, ANGLE, roll=ROLL), ref)
    _, kxr = maps["roll"]
    _, kx_flip = maps["roll_flipped"]

    assert kr[PY] == pytest.approx(kxr[PY], rel=1e-9)
    assert kr[PY] == pytest.approx(-kx_flip[PY], rel=1e-9)
    assert abs(kr[PY] - kx_flip[PY]) == pytest.approx(2.0 * abs(kxr[PY]), rel=1e-9)
    assert abs(kxr[PY]) > 5e-3  # ...and it is a large number, not a rounding artefact


def test_the_vertical_kick_is_sin_phi_sin_theta_in_xtrack_too(maps) -> None:
    """The closed form accsim derives, read straight off xtrack's own map.

    This is the number the opening roadmap entry got wrong: ``theta sin(phi)`` is
    1.5% away at this bend angle, which xtrack rejects by five orders.
    """
    _, kxr = maps["roll"]
    right = -math.sin(ROLL) * math.sin(ANGLE)
    wrong = -ANGLE * math.sin(ROLL)
    assert kxr[PY] == pytest.approx(right, rel=1e-9)
    assert abs(kxr[PY] - wrong) > 1e-4 * abs(right) * 100


def test_the_vertical_offset_term_is_xtracks_too(maps) -> None:
    """xtrack puts the sagitta term there as well — so "a pure vertical bend" is not
    a modelling choice accsim made differently, it is simply wrong."""
    _, kxr = maps["roll"]
    den = math.sin(ANGLE) ** 2 * math.cos(ROLL) + math.cos(ANGLE) ** 2
    want = -RHO * (1.0 - math.cos(ANGLE)) * math.sin(ROLL) / den
    assert kxr[Y] == pytest.approx(want, rel=1e-9)
    assert abs(kxr[Y]) > 1e-3


# ---------------------------------------------------------------------------
# B. The other roll — the one accsim deliberately does not implement
# ---------------------------------------------------------------------------


def test_the_design_tilt_is_the_conjugation_and_has_no_kick(maps) -> None:
    """``rot_s_rad`` is a hand-built ``SRotation . Bend . SRotation`` to a few ulp.

    Which is what makes it the *wrong* model for a misalignment: it has exactly zero
    kick, so it predicts no vertical closed orbit at all. Measured, so that the choice
    accsim made is a fact in the suite rather than a claim in a docstring.

    The two agree to a few ulp rather than exactly, because the sandwich rotates in
    two separate elements where ``rot_s_rad`` rotates inside one. That is round-off,
    not a model difference — the comparison that matters is the one below, where the
    gap is ``5.9e-3``.
    """
    Mt, kt = maps["tilt"]
    Mc, kc = maps["conjugation"]
    _, ka = maps["aligned"]
    np.testing.assert_allclose(kt, kc, rtol=1e-14, atol=1e-20)
    np.testing.assert_allclose(Mt, Mc, atol=1e-17)
    # No kick: what is left is xtrack's own round-off on the aligned bend (3e-11),
    # eight orders below the misalignment roll's 5.9e-3.
    _, kr = maps["roll"]
    assert np.max(np.abs(kt)) < 1e-9
    assert np.max(np.abs(ka)) < 1e-9
    assert np.max(np.abs(kr)) > 1e6 * np.max(np.abs(kt))


def test_the_conjugation_model_misses_the_misalignment_by_six_orders(
    maps, ref: ReferenceParticle
) -> None:
    """The K1-refusal shape: the wrong model is built and measured, not argued.

    A conjugation and a rigid motion differ by ``5.9e-3`` in the kick and ``6.2e-3``
    in the matrix, where the aligned maps agree to ``1e-9``. No tolerance could hide
    the difference, which is the point of measuring it.
    """
    Mc, kc = maps["conjugation"]
    Mr, kr = maps["roll"]
    Ma, ka = maps["aligned"]

    aligned_gap = np.max(np.abs(Ma - Dipole(L_BEND, ANGLE).matrix(ref)))
    assert np.max(np.abs(kc - kr)) > 5e-3
    assert np.max(np.abs(Mc - Mr)) > 5e-3
    assert np.max(np.abs(kc - kr)) > 1e6 * aligned_gap

    # accsim implements the rigid motion, and the difference from the conjugation is
    # the same number on its side of the comparison.
    accsim_conj = s_rotation(-ROLL) @ Dipole(L_BEND, ANGLE).matrix(ref) @ s_rotation(ROLL)
    assert np.max(np.abs(accsim_conj - Mr)) == pytest.approx(np.max(np.abs(Mc - Mr)), rel=1e-5)


# ---------------------------------------------------------------------------
# C. Straight elements: the conjugation is right there, and that is why it is
#    useless as a probe of the bend
# ---------------------------------------------------------------------------


def test_a_rolled_thin_quadrupole_matches_xtrack(maps, ref: ReferenceParticle) -> None:
    """The straight case, where the two rolls coincide — pinned to a few ulp.

    Both codes rotate, apply the same polynomial kick and rotate back, so there is
    almost nothing left to differ in. The point of the test is not that it is tight
    but that it is the *same* attribute: the sign convention it pins is shared with
    the bend, where it is the only thing that could pin it.
    """
    M, k = _accsim(ThinQuadrupole(K1L, roll=ROLL), ref)
    Mx, kx = maps["thin_quad_roll"]
    np.testing.assert_allclose(M, Mx, atol=1e-9)
    np.testing.assert_allclose(k, kx, atol=1e-12)
    # A flipped roll would show up in the off-diagonal block at first order.
    Mflip, _ = _accsim(ThinQuadrupole(K1L, roll=-ROLL), ref)
    assert np.max(np.abs(Mflip - Mx)) > 1e-2 * K1L


# ---------------------------------------------------------------------------
# D. The borrowed verdict: xtrack's own vertical dispersion
# ---------------------------------------------------------------------------


def _accsim_ring(ref: ReferenceParticle, roll: float) -> Lattice:
    elems: list = []
    for i in range(N_CELLS):
        elems += [
            ThinQuadrupole(0.5 / F_FOCAL),
            Dipole(L_BEND, RING_ANGLE, roll=roll if i == 0 else 0.0),
            ThinQuadrupole(-1.0 / F_FOCAL),
            Dipole(L_BEND, RING_ANGLE),
            ThinQuadrupole(0.5 / F_FOCAL),
        ]
    return Lattice(elems, ref)


def _xtrack_ring(roll: float, steer: float = 0.0):
    def bend(r: float):
        b = xt.Bend(length=L_BEND, angle=RING_ANGLE, k1=0.0, rot_s_rad_no_frame=r)
        b.edge_entry_active = 0
        b.edge_exit_active = 0
        return b

    elements: list = []
    for i in range(N_CELLS):
        elements += [
            xt.Multipole(knl=[0.0, 0.5 / F_FOCAL], length=0.0),
            bend(roll if i == 0 else 0.0),
            xt.Multipole(knl=[0.0, -1.0 / F_FOCAL], length=0.0),
            bend(0.0),
            xt.Multipole(knl=[0.0, 0.5 / F_FOCAL], length=0.0),
        ]
    if steer != 0.0:
        elements.insert(1, xt.Multipole(ksl=[steer], length=0.0))
    return elements


@pytest.fixture(scope="module")
def ring_twiss():
    """xtrack twiss for the rolled ring, the steered ring and the aligned one."""
    out = {}
    for label, kwargs in (
        ("rolled", {"roll": RING_ROLL}),
        ("steered", {"roll": 0.0, "steer": STEER}),
        ("aligned", {"roll": 0.0}),
    ):
        line = xt.Line(elements=_xtrack_ring(**kwargs))
        line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
        try:
            line.build_tracker()
            out[label] = line.twiss(method="4d")
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return out


def test_xtrack_confirms_a_rolled_bend_is_a_vertical_dispersion_source(
    ring_twiss, ref: ReferenceParticle
) -> None:
    """The independent verdict on the *claim*: rolling one bend turns ``D_y`` on.

    xtrack's aligned arc gives ``dy = 0`` identically and the rolled one does not,
    on a ring with no skew quadrupole and no other coupling element. That is K2's
    statement, checked by a dispersion solve accsim did not write. The *value* is a
    different matter — see the model-gap test below — and the closed orbit, which is
    the thing both codes model the same way, agrees to eight digits.
    """
    tw_a, tw_r = ring_twiss["aligned"], ring_twiss["rolled"]
    assert np.max(np.abs(np.array(tw_a.dy))) == 0.0
    assert np.max(np.abs(np.array(tw_r.dy))) > 1e-4
    assert coupled_twiss(_accsim_ring(ref, 0.0)).disp_y == 0.0
    assert abs(coupled_twiss(_accsim_ring(ref, RING_ROLL)).disp_y) > 1e-5

    # Both codes place the closed orbit the roll produces in the same place.
    co = closed_orbit(_accsim_ring(ref, RING_ROLL))
    assert co[Y] == pytest.approx(tw_r.y[0], rel=1e-5)
    assert co[PY] == pytest.approx(tw_r.py[0], rel=1e-5)
    # ...and the horizontal dispersion is untouched by the roll, as both codes agree.
    pts = propagate_coupled_twiss(_accsim_ring(ref, RING_ROLL))
    np.testing.assert_allclose(
        np.array([p.disp_x for p in pts]), np.array(tw_r.dx), rtol=1e-6, atol=1e-9
    )


def test_a_steered_ring_isolates_the_design_optics_blind_spot(
    ring_twiss, ref: ReferenceParticle
) -> None:
    r"""Why the ring-level ``D_y`` **value** was not a gate, measured in isolation.

    **Read this together with the model-gap test below, which now closes it.** What is
    isolated here is a property of ``matrix()``, and that property is permanent: the
    design optics is built on 6x6 matrices, the missing terms are bilinear, and no
    amount of exact tracking changes what a matrix can hold. What L1-L3 changed is that
    the *on-orbit* route now supplies the physical answer, so the sentence "accsim
    returns exactly 0" below is still true and no longer the whole story.

    accsim's linear elements drop the ``1/(1 + delta)`` on angles — a drift is
    ``y += L py``, where the exact map is ``y += L py / pz``. That missing
    ``-L py delta`` is already on record as a drift-model difference worth ``1e-8``
    (CONVENTIONS -> *the thick sextupole is compared by difference*). What K2 makes
    visible is that it is **not** always small: wherever the closed orbit has a
    vertical *angle*, the exact map turns that angle into vertical dispersion and
    accsim's linear one cannot.

    Isolated here by a **vertical steerer in an otherwise perfect ring** — no roll,
    no coupling, nothing K2 touched. accsim returns exactly ``0``; xtrack returns
    ``2.1e-4``. The two closed orbits agree to eight digits, so this is a statement
    about the momentum dependence of the maps and nothing else.

    Consequence for K2, stated rather than absorbed: on the rolled ring accsim's
    ``D_y`` is the **source-vector** part alone, and xtrack's is an order of
    magnitude larger because the orbit-driven part dominates there. The roadmap's
    "a vertical steerer produces the orbit without the dispersion" is a true
    statement about accsim's linear model and a **false** one about the physics.
    """
    lat = _accsim_ring(ref, 0.0)
    elems = list(lat.elements)
    elems.insert(1, Corrector(kick_y=STEER))
    steered = Lattice(elems, ref)
    tw_s = ring_twiss["steered"]

    # Same machine, same orbit, to eight digits.
    assert closed_orbit(steered)[Y] == pytest.approx(tw_s.y[0], rel=1e-7)
    # ...and completely different vertical dispersion.
    assert coupled_twiss(steered).disp_y == 0.0
    assert abs(tw_s.dy[0]) > 1e-4

    # On the rolled ring the same term is what the two codes differ by, and it is
    # bigger than accsim's whole answer — so this is reported, not tolerated.
    acc_dy = coupled_twiss(_accsim_ring(ref, RING_ROLL)).disp_y
    xt_dy = ring_twiss["rolled"].dy[0]
    assert abs(xt_dy) > 5.0 * abs(acc_dy)


def _missing_source_dispersion(lat: Lattice, ref: ReferenceParticle) -> np.ndarray:
    r"""``D`` re-solved with the two terms accsim's linear elements drop put back.

    Per element of length ``L``, the exact vertical motion is ``dy/ds = py (1 + h x)/pz``
    against accsim's ``dy/ds = py``. Differentiating in ``delta`` at the closed orbit:

        extra source = py L (h <D_x> - 1)

    — the ``-1`` from ``1/pz`` (the drift term already on record from J1), the
    ``+h <D_x>`` from the extra arc a dispersed particle travels on the outside of a
    bend. The horizontal plane gets the same with ``px``. Each element's extra source
    is transported to the end of the ring and added to the ``delta`` column before the
    usual ``D = (I - M4)^-1 d`` solve.
    """
    from accsim import propagate_orbit

    idx = [X, PX, Y, PY]
    orbit = propagate_orbit(lat)
    tw = propagate_coupled_twiss(lat)
    one_turn = lat.one_turn_matrix()
    m4, source = one_turn[np.ix_(idx, idx)], one_turn[idx, DELTA].copy()

    mats = [e.matrix(ref)[np.ix_(idx, idx)] for e in lat.elements]
    after = [np.eye(4)]
    for m in reversed(mats):
        after.append(after[-1] @ m)
    after = after[::-1]  # after[i] transports from element i's entrance to the end

    for i, elem in enumerate(lat.elements):
        if elem.length == 0.0:
            continue
        scale = getattr(elem, "curvature", 0.0) * 0.5 * (tw[i].disp_x + tw[i + 1].disp_x) - 1.0
        extra = np.array(
            [
                elem.length * float(orbit[i][PX]) * scale,
                0.0,
                elem.length * float(orbit[i][PY]) * scale,
                0.0,
            ]
        )
        source = source + after[i + 1] @ extra
    return np.linalg.solve(np.eye(4) - m4, source)


def test_the_model_gap_is_fully_accounted_for_and_not_a_mystery(
    ring_twiss, ref: ReferenceParticle
) -> None:
    r"""**The gap is closed** — the package's own answer, not a hand reconstruction.

    This test was written by K2 as a *specification*. It could then only put the two
    dropped terms back by hand and show that doing so reproduced xtrack to 0.2 %, with
    a note that representing them for real meant exact nonlinear maps for ``Drift``,
    ``Quadrupole`` and ``Dipole`` and would re-baseline every gate in the suite. L1, L2
    and L3 are those three maps. So the assertion has moved from the reconstruction to
    :func:`~accsim.twiss.coupled_twiss_on_orbit`, and the number moved with it: from
    ``2e-3`` relative to **``1.7e-8``** on the rolled ring and ``3.5e-9`` on the steered
    one, on both ``dy`` and ``dpy``.

    The design optics still reports the old answers and is still *right* to: the terms
    are bilinear in ``(p, delta)`` and cannot live in a 6x6 at all. On the rolled ring
    that is an order of magnitude out, and on the steered ring it is exactly zero. The
    two routes are not in conflict — they are the linear map and the exact one.

    The hand reconstruction is kept, demoted to what it always was: an *independent*
    first-order account, built from the closed form rather than by differencing
    ``track()``. It agrees to 0.2 %, and the fact that the package now does five orders
    better than its own specification is the measure of what the exact maps added — the
    reconstruction has only the ``delta`` column, and the exact bend also couples the
    planes (``tests/analytic/test_exact_dipole.py``).
    """
    for label, lat in (
        ("rolled", _accsim_ring(ref, RING_ROLL)),
        ("steered", None),
    ):
        if lat is None:
            elems = list(_accsim_ring(ref, 0.0).elements)
            elems.insert(1, Corrector(kick_y=STEER))
            lat = Lattice(elems, ref)
        tw = ring_twiss[label]

        # The package, against xtrack. This is the milestone.
        on_orbit = coupled_twiss_on_orbit(lat)
        assert on_orbit.disp_y == pytest.approx(tw.dy[0], rel=1e-6), label
        assert on_orbit.disp_py == pytest.approx(tw.dpy[0], rel=1e-5, abs=1e-12), label

        # The design optics, which cannot carry the terms and says so plainly.
        design = coupled_twiss(lat).disp_y
        assert abs(design - tw.dy[0]) > 0.5 * abs(tw.dy[0]), label

        # K2's own first-order account, still good to 0.2 % and no better.
        got = _missing_source_dispersion(lat, ref)
        assert got[2] == pytest.approx(tw.dy[0], rel=3e-3), label
        assert got[3] == pytest.approx(tw.dpy[0], rel=5e-3, abs=1e-12), label
