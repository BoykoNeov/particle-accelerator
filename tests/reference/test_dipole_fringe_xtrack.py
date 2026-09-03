r"""P2 (i) against xtrack: the hard-edge fringe as a **tracked** map, not a Taylor entry.

The MAD-X leg (``test_second_order_map_madx.py``) compares the fringe through the
second-order map, which is where the milestone's five closed forms live. It therefore
gates only what ``T`` carries: the ``zeta`` term is cubic and every ``1/(1 + delta)``
inside the generator is one order higher again, so a map that got the whole rigidity
dependence wrong would pass it entry for entry.

This leg closes that. ``xt.Bend`` with ``edge_entry_model="full"`` is the MAD-NG/PTC
fringe at ``fint = hgap = 0``, and with a sector face (``edge_*_angle = 0``) its wedge
and its ``y``-rotation drop out, leaving the bare ``DipoleFringe`` kernel — the same
object accsim generates from ``Phi = h px pz / ((1+delta)^2 - px^2)``. So the comparison
is *tracking*, particle by particle, at ``delta`` up to 5%, and it holds to ``1e-15``.

Two things are asserted separately on purpose:

- **``zeta`` on its own line.** It is four to six orders smaller than ``x`` here, so a
  ``max`` over the six coordinates would hide any error in it behind the transverse
  agreement — and ``zeta`` is the only coordinate the MAD-X leg cannot see at all. What
  this leg does **not** reach is the ``beta0/beta`` factor *inside* that ``zeta`` term:
  the whole file runs at ``gamma0 = 20``, which is precisely where dropping the factor
  costs ``5.1e-10`` — below every tolerance here. That one is gated only by the
  low-energy leg in ``tests/analytic/test_dipole_fringe.py``, and it is the piece a
  future edit is most likely to break.
- **The fringe is not a rounding correction.** The same states through a fringe-off bend
  differ from xtrack's fringe-on answer by ``1.7e-5``, ten orders above the gate, so an
  accsim that silently ignored ``fringe=True`` could not pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import Dipole, ReferenceParticle
from accsim.coords import DIM, ZETA

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0, GAMMA0 = 938.27208816e6, 20.0
LENGTH, ANGLE = 1.0, 0.3

#: Every coordinate live at once, and ``delta`` out to 5%: the fringe's rigidity
#: dependence sits in ``1/(1+delta)``-shaped factors that a ``delta = 0`` state cannot
#: separate from the rest, and its ``y`` shift is quadratic so a ``y = 0`` state is blind
#: to the whole effect.
STATES = [
    np.array([3.0e-3, 8.0e-3, -2.0e-3, 5.0e-3, 1.0e-3, 2.0e-3]),
    np.array([2.0e-3, -5.0e-3, 1.5e-3, 4.0e-3, 0.0, 5.0e-2]),
    np.array([1.0e-2, 4.0e-3, -8.0e-3, 2.0e-3, 5.0e-4, -1.0e-2]),
    np.array([0.0, 6.0e-3, 9.0e-3, -7.0e-3, 2.0e-3, 3.0e-2]),
]


def _xtrack_tracked(edge_model: str) -> list[np.ndarray]:
    """Every state in :data:`STATES` through one ``xt.Bend`` with the given edge model."""
    bend = xt.Bend(length=LENGTH, angle=ANGLE, k1=0.0, model="bend-kick-bend")
    bend.edge_entry_angle = 0.0  # a sector face: no wedge, no y-rotation, fringe only
    bend.edge_exit_angle = 0.0
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


def _accsim_tracked(*, fringe: bool) -> list[np.ndarray]:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    bend = Dipole(LENGTH, ANGLE, fringe=fringe)
    return [bend.track(st, ref) for st in STATES]


def test_the_fringed_bend_is_xtracks_full_edge_model_to_the_last_bit() -> None:
    """``Dipole(fringe=True).track`` **is** ``edge_*_model="full"`` at a sector face.

    Two independent implementations of one map — accsim's from the generating function
    ``W = -Phi ybar^2 / 2`` with the three gradients derived symbolically, xtrack's from
    MAD-NG's chain rule through ``atan``/``tan`` — agreeing to ``1e-15`` on every
    coordinate. The exact body (``bend-kick-bend``) is the L3 circle already pinned at
    ``1.9e-16``, so what is left over in the difference is the two faces.
    """
    theirs = _xtrack_tracked("full")
    ours = _accsim_tracked(fringe=True)
    for st, a, b in zip(STATES, ours, theirs, strict=True):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-15, err_msg=str(st))


def test_zeta_is_gated_on_its_own_because_nothing_else_gates_it() -> None:
    r"""The longitudinal term, alone — the one piece ``T`` and the MAD-X leg cannot see.

    ``zeta`` picks up ``(1/2) (beta0/beta) (dPhi/ddelta) ybar^2``: a product of three
    coordinates, so it never reaches the second-order map, and the ``beta0/beta``
    conversion inside it is a ``7e-5`` correction at ``gamma0 = 20`` that even
    symplecticity cannot resolve there (``tests/analytic/test_dipole_fringe.py``). What
    *can* resolve it is another code's arrival time, so it is asserted here on its own
    line rather than inside a ``max`` over six coordinates where the transverse agreement
    would swamp it.

    The fringe moves ``zeta`` by ``6e-8`` to ``4e-6`` on these states, against a ``zeta``
    of ``1e-3``: asserted to be **present** (the fringe-off bend misses xtrack's arrival
    time by that much) and **right** (the fringe-on bend agrees to ``1.4e-16``, which is
    round-off on a ``1e-3`` coordinate).
    """
    theirs = _xtrack_tracked("full")
    ours = _accsim_tracked(fringe=True)
    bare = _accsim_tracked(fringe=False)
    for st, a, b, c in zip(STATES, ours, theirs, bare, strict=True):
        assert abs(a[ZETA] - b[ZETA]) < 1e-15, str(st)
        assert abs(c[ZETA] - b[ZETA]) > 1e-8, str(st)


def test_the_fringe_is_not_a_rounding_correction_and_the_linear_edge_is_not_it() -> None:
    r"""Two controls, so the agreement above cannot be an agreement about nothing.

    The fringe moves the tracked coordinates by ``1.7e-5`` — ten orders above the ``1e-15``
    gate — so a ``fringe=True`` that quietly did nothing would fail. And xtrack's
    *default* ``linear`` edge model at a sector face is the **identity** (``e = 0``, so
    ``h tan e = 0``), which is where accsim's fringe-off bend already agrees with it: the
    two models differ by the fringe alone, not by an edge focusing that neither applies.
    """
    full, linear = _xtrack_tracked("full"), _xtrack_tracked("linear")
    ours, bare = _accsim_tracked(fringe=True), _accsim_tracked(fringe=False)

    moved = max(np.max(np.abs(a - c)) for a, c in zip(ours, bare, strict=True))
    assert moved > 1e-7, moved
    for a, b in zip(bare, linear, strict=True):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-15)
    # ...and what the fringe *does* is the same displacement in both codes, compared
    # per state and per coordinate. A max-over-states of a max-over-coordinates would be
    # two argmaxes agreeing by luck, which is not a statement about the map.
    for st, a, c, b, d in zip(STATES, ours, bare, full, linear, strict=True):
        np.testing.assert_allclose(a - c, b - d, rtol=0.0, atol=1e-15, err_msg=str(st))


def test_the_map_is_a_bunch_map_and_the_planes_it_couples_are_named() -> None:
    r"""Vectorised over a bunch, and the effect is where the physics says: ``y`` and ``px``.

    A ``(6, n)`` bunch takes the same path as a ``(6,)`` state — asserted rather than
    assumed, since the fringe is the first thing on this element with a scalar-looking
    quadratic solve in it. And the *shape* of the effect is checked against the mechanism:
    the kick is ``B_s = y dB_y/ds``, so a particle that stays in the median plane is
    untouched however large its other coordinates.

    ``y = 0`` at the *entrance* is not enough for that, which is worth stating because it
    is the obvious wrong version of this check: a sector bend drifts vertically, so a
    particle entering on the median plane with ``py != 0`` leaves it and the **exit** face
    kicks it (measured: ``1.2e-5`` in ``py``). The median-plane particle is the one with
    ``y = py = 0``, and a horizontal bend keeps it there exactly.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    bend = Dipole(LENGTH, ANGLE, fringe=True)
    bunch = np.stack(STATES, axis=1)
    assert bunch.shape == (DIM, len(STATES))
    together = bend.track(bunch, ref)
    for i, st in enumerate(STATES):
        np.testing.assert_allclose(together[:, i], bend.track(st, ref), rtol=0.0, atol=0.0)

    plain = Dipole(LENGTH, ANGLE)
    median = STATES[0].copy()
    median[2] = median[3] = 0.0  # y = py = 0: the bend cannot take it out of the plane
    np.testing.assert_allclose(
        bend.track(median, ref), plain.track(median, ref), rtol=0.0, atol=0.0
    )

    entering_flat = STATES[0].copy()
    entering_flat[2] = 0.0  # y = 0 but py != 0: the *exit* face still sees it
    moved = bend.track(entering_flat, ref) - plain.track(entering_flat, ref)
    assert np.max(np.abs(moved)) > 1e-6
