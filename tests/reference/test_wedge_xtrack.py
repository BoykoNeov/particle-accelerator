r"""P3 (a) against xtrack: the **rotated** pole face, tracked particle by particle.

``xt.Bend`` with ``edge_*_model="full"`` and a nonzero ``edge_*_angle`` applies, in order,
a ``y``-rotation into the face plane, the MAD-NG dipole fringe, and a wedge back to the
sector plane. accsim now applies the same three, with two differences that make this a
cross-check rather than a transcription test:

- **the rotation is not a separate map.** accsim's :func:`~accsim.elements.dipole.wedge_map`
  has no branch on the curvature at all, so the "rotation" is the same function with the
  field switched off. xtrack has two kernels and a ``fabs(b1) < 1e-10`` branch between
  them; that the two agree at ``1e-14`` says the branch and the limit are the same map.
- **the wedge is derived here, not ported.** ``tests/analytic/test_wedge.py`` derives it
  from the uniform-field circle and checks the plane-crossing collapse symbolically; the
  closed form xtrack evaluates is the *consequence*. So agreement is evidence about the
  physics, not about the copying — the circularity P2 (iv) had to rule out for PTC.

The second-order leg lives in ``test_second_order_map_madx.py``: ``sectormap`` sees the
wedge's quadratic content and is the arbiter for it. What only *this* leg reaches is
``zeta`` (the flight time across the sliver) and the ``1/(1+delta)``-shaped factors,
exactly as in P2 (i) one map down.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import Dipole, ReferenceParticle
from accsim.coords import ZETA

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0, GAMMA0 = 938.27208816e6, 20.0
LENGTH, ANGLE = 1.0, 0.3
E1, E2 = 0.12, -0.07

#: Every coordinate live at once, ``delta`` out to 5%. The wedge's ``x`` map is quadratic
#: in ``x`` and its ``y``/``zeta`` shares are proportional to ``x``, so a state with a zero
#: in it cannot separate the wedge from the fringe it sits next to.
STATES = [
    np.array([3.0e-3, 8.0e-3, -2.0e-3, 5.0e-3, 1.0e-3, 2.0e-3]),
    np.array([2.0e-3, -5.0e-3, 1.5e-3, 4.0e-3, 0.0, 5.0e-2]),
    np.array([1.0e-2, 4.0e-3, -8.0e-3, 2.0e-3, 5.0e-4, -1.0e-2]),
    np.array([-6.0e-3, 6.0e-3, 9.0e-3, -7.0e-3, 2.0e-3, 3.0e-2]),
]


def _xtrack_tracked(edge_model: str, e1: float, e2: float) -> list[np.ndarray]:
    """Every state in :data:`STATES` through one ``xt.Bend`` with the given face model."""
    bend = xt.Bend(length=LENGTH, angle=ANGLE, k1=0.0, model="bend-kick-bend")
    bend.edge_entry_angle = e1
    bend.edge_exit_angle = e2
    bend.edge_entry_model = edge_model
    bend.edge_exit_model = edge_model
    # Hard edge: the fringe integrals must be zero, or the kernel adds the soft-edge
    # ``fint``/``hgap`` terms accsim does not model.
    assert bend.edge_entry_fint == 0.0 and bend.edge_exit_fint == 0.0
    assert bend.edge_entry_hgap == 0.0 and bend.edge_exit_hgap == 0.0
    line = xt.Line(elements=[bend])
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
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


def _accsim_tracked(*, fringe: bool, e1: float = E1, e2: float = E2) -> list[np.ndarray]:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    bend = Dipole(LENGTH, ANGLE, e1=e1, e2=e2, fringe=fringe)
    return [bend.track(st, ref) for st in STATES]


def test_the_rotated_face_is_xtracks_full_edge_model_to_the_last_bit() -> None:
    """``Dipole(e1, e2, fringe=True).track`` **is** ``edge_*_model="full"`` at a rotated face.

    The whole composition at once: rotation, fringe, wedge, the exact L3 body, and the
    three again in reverse at the exit. Both codes at ``1e-14`` on every coordinate of
    every state — the residual is the wedge's own arithmetic (accsim forms the turned
    angle without ever dividing by the curvature; xtrack divides), not a disagreement
    about the map.
    """
    theirs = _xtrack_tracked("full", E1, E2)
    ours = _accsim_tracked(fringe=True)
    for st, a, b in zip(STATES, ours, theirs, strict=True):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-14, err_msg=str(st))


def test_the_wedge_is_not_a_rounding_correction_and_the_linear_edge_is_not_it() -> None:
    r"""Three controls, so the agreement above cannot be an agreement about nothing.

    - **The face moves the answer** by ``1e-4``, ten orders above the gate, so a
      ``fringe=True`` that quietly kept the linear edge could not pass.
    - **The linear edge is not the full one.** accsim's ``fringe=False`` bend is the
      ``h tan(e)`` thin kick, and xtrack's ``linear`` model is the same thing; those two
      agree, and both differ from ``full`` by the wedge and the fringe together. So the
      difference being measured is a *model* difference, not a missing edge.
    - **And the displacement itself matches**, per state and per coordinate: what the
      wedge and fringe together do to a trajectory is the same vector in both codes.
      A max-over-states of a max-over-coordinates would be two argmaxes agreeing by luck.
    """
    full, linear = _xtrack_tracked("full", E1, E2), _xtrack_tracked("linear", E1, E2)
    ours, bare = _accsim_tracked(fringe=True), _accsim_tracked(fringe=False)

    moved = max(np.max(np.abs(a - c)) for a, c in zip(ours, bare, strict=True))
    assert moved > 1e-5, moved
    for st, a, b in zip(STATES, bare, linear, strict=True):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=2e-6, err_msg=str(st))
    for st, a, c, b, d in zip(STATES, ours, bare, full, linear, strict=True):
        np.testing.assert_allclose(a - c, b - d, rtol=0.0, atol=2e-6, err_msg=str(st))


def test_zeta_is_gated_on_its_own_because_the_second_order_leg_cannot_see_it() -> None:
    r"""The flight time across the sliver, alone.

    The wedge advances the *reference* particle by nothing and the particle itself by an
    extra path, so ``zeta`` picks up ``-(path) beta0/beta``. MAD-X's ``sectormap`` reports
    the second-order map in its own frame and cannot arbitrate the ``beta0/beta``
    conversion; another code's arrival time can. It is asserted on its own line because
    the wedge moves ``zeta`` by ``1e-6`` against a transverse effect of ``1e-4``, so a
    ``max`` over six coordinates would hide it.
    """
    theirs = _xtrack_tracked("full", E1, E2)
    ours = _accsim_tracked(fringe=True)
    bare = _accsim_tracked(fringe=False)
    for st, a, b, c in zip(STATES, ours, theirs, bare, strict=True):
        assert abs(a[ZETA] - b[ZETA]) < 1e-14, str(st)
        assert abs(c[ZETA] - b[ZETA]) > 1e-8, str(st)


def test_a_rectangular_bend_agrees_too_and_it_is_the_case_that_matters() -> None:
    r"""``e1 = e2 = angle/2`` — the rbend, where the faces are as rotated as they get.

    The sector face is the easy case and the one P2 (i) already covered; the reason P3
    exists is that real lattices are full of rectangular bends, whose faces carry half
    the bend angle each. At ``angle = 0.3`` that is a ``0.15`` rad face, twice the
    entrance angle used above, and the wedge's content grows with it.
    """
    half = ANGLE / 2.0
    theirs = _xtrack_tracked("full", half, half)
    ours = _accsim_tracked(fringe=True, e1=half, e2=half)
    for st, a, b in zip(STATES, ours, theirs, strict=True):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-14, err_msg=str(st))


def test_the_faces_are_not_symmetric_and_swapping_them_is_visible() -> None:
    """``e1`` and ``e2`` are different maps, and the test would pass if they were not.

    A face is applied in the opposite order at the two ends and the *fringe* alone flips
    sign there, so the exit face is not the entrance face. Swapping ``e1`` and ``e2``
    therefore has to change the answer — otherwise the agreement above would survive an
    accsim that used one face twice.
    """
    swapped = _accsim_tracked(fringe=True, e1=E2, e2=E1)
    ours = _accsim_tracked(fringe=True)
    moved = max(np.max(np.abs(a - b)) for a, b in zip(ours, swapped, strict=True))
    assert moved > 1e-4, moved
