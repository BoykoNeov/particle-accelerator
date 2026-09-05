r"""P3 (b) against xtrack: the gradient face, tracked, on both elements.

The analytic file derives the map's coefficient from Maxwell and the Lorentz force, which
is the gate with teeth (nothing structural can see the ``1/12``). This file is the
independent implementation: MAD-NG's ``MultFringe`` kernel, ported into xtrack, reached
through ``xt.Quadrupole``'s ``edge_entry_active``/``edge_exit_active`` and
``xt.Bend``'s ``edge_*_model="full"``.

**Two things about the comparison were not obvious and are the reason it is written this
way.**

- **The faces are switched independently on the quadrupole**, and that is used. An
  exit-face sign error is *partially self-cancelling* — with both faces on it changes the
  magnitude and leaves the structure — so it is the error a whole-element comparison is
  least likely to catch. All four combinations are compared, and the two single-face ones
  are where the sign actually gets pinned.
- **The bend's body model has to be named before any of this is measurable.** accsim's
  combined-function body is xtrack's ``mat-kick-mat`` with one uniform kick, exactly
  (``test_dipole_combined_xtrack.py`` pins that to ``1e-15``); it is *not*
  ``bend-kick-bend``, which keeps the curvilinear ``(1 + h x)`` metric factor and differs
  by ``1.3e-5`` — **half the size of the whole fringe effect** measured here. Against that
  family the comparison would have to be a difference of differences and would gate at
  ``5e-8`` instead of ``1e-16``. Naming the right family is what buys the absolute
  comparison, and it is worth five orders of magnitude.

What this leg cannot see: the ``E/E0`` factor in the arrival-time term. The whole file
runs at ``gamma0 = 20``, where it differs from ``1`` below every tolerance here — and
unlike P2 (i), symplecticity does not recover it either, because a cubic map's ``zeta``
share is too small to move a finite-difference Poisson bracket. It is derived instead,
from the generating function, in ``tests/analytic/test_multipole_fringe.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import Dipole, Quadrupole, ReferenceParticle
from accsim.elements.fringe import multipole_fringe_map
from accsim.elements.quadrupole import thick_quadrupole_map

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0, GAMMA0 = 938.27208816e6, 20.0
L_Q, K1_Q = 0.7, 1.3
L_B, ANGLE_B, K1_B = 1.0, 0.3, 0.4

#: Every coordinate live at once. The fringe's entries are products of three coordinates,
#: so a state with a zero in it cannot separate one from another, and ``delta`` is out at a
#: few percent because the map carries ``1/(1+delta)`` in four places.
STATES = [
    np.array([3.0e-3, 8.0e-3, -2.0e-3, 5.0e-3, 1.0e-3, 2.0e-3]),
    np.array([2.0e-3, -5.0e-3, 1.5e-3, 4.0e-3, 0.0, 5.0e-2]),
    np.array([1.0e-2, 4.0e-3, -8.0e-3, 2.0e-3, 5.0e-4, -1.0e-2]),
    np.array([-6.0e-3, 3.0e-3, 9.0e-3, -7.0e-3, 2.0e-3, 3.0e-2]),
]


def _ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _build(element) -> object:
    """One tracker, built once. Every xt.Line build JIT-compiles a fresh C kernel."""
    line = xt.Line(elements=[element])
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


def _track(line) -> list[np.ndarray]:
    out = []
    for st in STATES:
        p = xt.Particles(
            mass0=MASS0,
            q0=1,
            gamma0=GAMMA0,
            x=st[0],
            px=st[1],
            y=st[2],
            py=st[3],
            zeta=st[4],
            delta=st[5],
        )
        line.track(p)
        out.append(np.array([p.x[0], p.px[0], p.y[0], p.py[0], p.zeta[0], p.delta[0]]))
    return out


# ---------------------------------------------------------------------------
# The quadrupole — the leg where an absolute comparison exists
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quad_line():
    """``xt.Quadrupole`` in accsim's own model family, with the faces switchable.

    ``edge_entry_active``/``edge_exit_active`` are plain data fields, so the four
    combinations below cost **one** kernel compile between them rather than four.
    """
    return _build(xt.Quadrupole(length=L_Q, k1=K1_Q, model="mat-kick-mat"))


@pytest.mark.parametrize(
    ("entry", "exit_"), [(False, False), (True, True), (True, False), (False, True)]
)
def test_the_gradient_fringe_is_xtracks_multfringe_face_by_face(
    quad_line, entry: bool, exit_: bool
) -> None:
    r"""Every face combination, absolutely, to ``1e-14``.

    ``xt.Quadrupole``'s edge is an **on/off switch only** — no model choice, no face angle
    — and with ``k0 = 0`` its ``DipoleFringe`` returns immediately and its wedge and
    ``y``-rotation are skipped, so what is left is the bare ``MultFringe`` kernel at
    ``min_order = 1``. That is exactly :func:`~accsim.elements.fringe.multipole_fringe_map`,
    and the two agree to the last bit including ``zeta``.

    The two **single-face** rows are the ones that matter. Composing both faces around a
    body makes an exit-sign error shrink rather than vanish, so a two-face comparison alone
    would not pin the sign; one face at a time does.
    """
    ref = _ref()
    quad_line.elements[0].edge_entry_active = entry
    quad_line.elements[0].edge_exit_active = exit_
    want = _track(quad_line)

    for st, w in zip(STATES, want, strict=True):
        got = st.copy()
        if entry:
            got = multipole_fringe_map(got, K1_Q, ref, exit_face=False)
        got = thick_quadrupole_map(got, L_Q, K1_Q, ref)
        if exit_:
            got = multipole_fringe_map(got, K1_Q, ref, exit_face=True)
        np.testing.assert_allclose(got, w, rtol=0.0, atol=1e-14)


def test_the_quadrupole_element_carries_the_faces_and_they_are_not_a_rounding_correction(
    quad_line,
) -> None:
    r"""``Quadrupole(fringe=True)`` is that composition, and the effect is ``1e-8``, not ``0``.

    The control matters more here than usual: every structural gate in the analytic file
    passes for a map that is uniformly mis-scaled, and *all* of them pass for a map that
    does nothing. So the size of the difference is asserted — an accsim that quietly
    ignored ``fringe=True`` would sit six orders below this line.
    """
    ref = _ref()
    quad_line.elements[0].edge_entry_active = True
    quad_line.elements[0].edge_exit_active = True
    want = _track(quad_line)

    plain, fringed = Quadrupole(L_Q, K1_Q), Quadrupole(L_Q, K1_Q, fringe=True)
    worst_off = 0.0
    for st, w in zip(STATES, want, strict=True):
        np.testing.assert_allclose(fringed.track(st, ref), w, rtol=0.0, atol=1e-14)
        worst_off = max(worst_off, float(np.abs(plain.track(st, ref) - w).max()))
    assert worst_off > 1e-9, "the faces must move the tracked state by far more than the gate"


# ---------------------------------------------------------------------------
# The combined-function bend — the whole face, all five maps of it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bend_line():
    """``xt.Bend`` in accsim's model family with the full nonlinear edge.

    ``mat-kick-mat`` + one ``uniform`` kick **is** accsim's combined-function body (pinned
    to ``1e-15`` in ``test_dipole_combined_xtrack.py``), which is what makes an absolute
    comparison of the *faces* possible at all. ``edge_*_angle`` are data fields, so the
    sector and rotated cases share one compile.
    """
    bend = xt.Bend(
        length=L_B,
        angle=ANGLE_B,
        k1=K1_B,
        model="mat-kick-mat",
        num_multipole_kicks=1,
        integrator="uniform",
    )
    bend.edge_entry_model = "full"
    bend.edge_exit_model = "full"
    # Hard edge: a non-zero fringe integral would add the soft-edge terms accsim does not
    # model, and would do it silently.
    assert bend.edge_entry_fint == 0.0 and bend.edge_exit_fint == 0.0
    assert bend.edge_entry_hgap == 0.0 and bend.edge_exit_hgap == 0.0
    return _build(bend)


@pytest.mark.parametrize(("e1", "e2"), [(0.0, 0.0), (0.08, 0.05)])
def test_the_whole_gradient_face_of_a_combined_bend_is_xtracks(
    bend_line, e1: float, e2: float
) -> None:
    r"""The composed face — rotation, dipole fringe, gradient fringe, both wedges — to ``1e-14``.

    This is what ``Dipole(fringe=True, k1=...)`` refused to do until P3 (b), and the two
    rows are the two halves of the refusal:

    - **the sector face** (``e = 0``) exercises the gradient fringe alone: the wedges
      collapse to the identity and only the *cubic* map is added;
    - **the rotated face** adds the quadrupole wedge, which is quadratic and is therefore
      the one piece of this milestone that a second-order arbiter could have seen.

    Both land at machine precision against an effect of ``2e-5``, ten orders above the
    gate — with the ordering asserted implicitly, since the five maps do not commute.
    """
    ref = _ref()
    bend_line.elements[0].edge_entry_angle = e1
    bend_line.elements[0].edge_exit_angle = e2
    want = _track(bend_line)

    fringed = Dipole(L_B, ANGLE_B, k1=K1_B, e1=e1, e2=e2, fringe=True)
    plain = Dipole(L_B, ANGLE_B, k1=K1_B, e1=e1, e2=e2)
    worst_off = 0.0
    for st, w in zip(STATES, want, strict=True):
        np.testing.assert_allclose(fringed.track(st, ref), w, rtol=0.0, atol=1e-14)
        worst_off = max(worst_off, float(np.abs(plain.track(st, ref) - w).max()))
    assert worst_off > 1e-6, "the nonlinear face must be far larger than the gate"


def test_the_gradient_maps_are_both_needed_and_neither_is_the_other(bend_line) -> None:
    r"""Drop either ``k1`` map from the rotated face and the agreement is gone.

    Two maps were added to this face, not one, and they are different physics — a *cubic*
    fringe present at every face and a *quadratic* wedge present only at a rotated one.
    That is the kind of pair that gets silently merged, so each is removed in turn and the
    damage measured: ``2.5e-6`` without the fringe and ``4.5e-6`` without the wedge, both
    eight orders above the gate.

    They are **comparable in size**, which was not the guess — the wedge is quadratic and
    the fringe cubic, so a factor of a hundred was expected and is not there. The reason is
    that ``k1 e`` is small (``0.032``) where ``k1`` is not, and the ``1e-2`` amplitudes here
    put ``k1 x^3`` and ``k1 e x^2`` within a factor of two. So *size* does not separate
    these two maps; only the **order in the amplitude** does, and that separation is made
    on the accsim side, in ``tests/analytic/test_multipole_fringe.py``, where the fringe is
    shown to leave the second-order map untouched and the wedge to move it.
    """
    ref = _ref()
    e1, e2 = 0.08, 0.05
    bend_line.elements[0].edge_entry_angle = e1
    bend_line.elements[0].edge_exit_angle = e2
    want = _track(bend_line)

    full = Dipole(L_B, ANGLE_B, k1=K1_B, e1=e1, e2=e2, fringe=True)

    misses_fringe, misses_wedge = 0.0, 0.0
    for st, w in zip(STATES, want, strict=True):
        np.testing.assert_allclose(full.track(st, ref), w, rtol=0.0, atol=1e-14)
        # the hand-built face with *both* maps reproduces the element, which is what makes
        # the two variants below a fair statement about the missing map and nothing else
        np.testing.assert_allclose(
            _face_variants(st, ref, fringe=True, wedge=True), w, rtol=0.0, atol=1e-14
        )
        misses_fringe = max(misses_fringe, float(np.abs(_without_fringe(st, ref) - w).max()))
        misses_wedge = max(misses_wedge, float(np.abs(_without_wedge(st, ref) - w).max()))

    assert misses_fringe > 1e-8, "the gradient fringe must be doing something"
    assert misses_wedge > 1e-8, "the quadrupole wedge must be doing something"
    # neither is the other: dropping one is not the same damage as dropping the other
    assert abs(misses_wedge - misses_fringe) > 1e-7


def _face_variants(state: np.ndarray, ref: ReferenceParticle, *, fringe: bool, wedge: bool):
    """The combined bend with either gradient map switched out of the face."""
    from accsim.elements.dipole import (
        curvature_sextupole_kick,
        expanded_cfd_map,
        hard_edge_fringe_map,
        wedge_map,
    )
    from accsim.elements.fringe import quad_wedge_map

    h = ANGLE_B / L_B
    e1, e2 = 0.08, 0.05
    st = wedge_map(state, e1, 0.0, ref)
    st = hard_edge_fringe_map(st, h, ref)
    if fringe:
        st = multipole_fringe_map(st, K1_B, ref, exit_face=False)
    if wedge:
        st = quad_wedge_map(st, -e1, K1_B)
    st = wedge_map(st, -e1, h, ref)

    half = 0.5 * L_B
    st = expanded_cfd_map(st, half, h, K1_B, ref)
    st = curvature_sextupole_kick(st, h * K1_B * L_B)
    st = expanded_cfd_map(st, half, h, K1_B, ref)

    st = wedge_map(st, -e2, h, ref)
    if wedge:
        st = quad_wedge_map(st, -e2, K1_B)
    if fringe:
        st = multipole_fringe_map(st, K1_B, ref, exit_face=True)
    st = hard_edge_fringe_map(st, -h, ref)
    return wedge_map(st, e2, 0.0, ref)


def _without_fringe(state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
    return _face_variants(state, ref, fringe=False, wedge=True)


def _without_wedge(state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
    return _face_variants(state, ref, fringe=True, wedge=False)
