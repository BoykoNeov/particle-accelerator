"""M1 cross-check: the chromatic functions and ``Q''`` against xtrack.

What this suite establishes, in order:

1. **The convention**, by probe rather than by recall — that ``bx_chrom`` is
   ``(dbetx/ddelta)/betx`` and ``ax_chrom`` is ``dalfx/ddelta - dbetx alfx/betx``,
   confirmed by finite-differencing xtrack's *own* ``betx`` and ``alfx``. This is
   where a stray ``alpha``, a missing division or a ``pzeta``-vs-``delta`` swap
   would live.
2. **The chromatic functions agree element by element** on a ring with bends. This
   is the milestone's validated deliverable: not a single scalar that could be
   right by cancellation, but the whole ``dbeta/ddelta`` curve around the ring.
3. **``Q''`` agrees on a bend-free ring** to seven digits — the control that proves
   the second-difference machinery itself.
4. **``Q''`` disagrees on a bendy ring, and it is not accsim's maps.** The Dipole
   Jacobian is shown equal to ``xt.Bend``'s to ``5e-9`` entry by entry *on the
   off-momentum closed orbit*, and the closed orbits equal to ``1e-9``. Identical
   maps and identical orbits cannot give different tunes, so the disagreement is
   pinned here as a named boundary and left to M2.

The edge model matters and is asserted, not assumed: accsim's
:class:`~accsim.Dipole` uses the **linear hard-edge** kick, which is the identity
at ``e1 = e2 = 0``, while xtrack's ``edge='full'`` applies a *nonlinear* wedge and
fringe that focuses vertically even for a sector bend. The apples-to-apples
comparison is therefore ``edge='suppressed'``, and the test below shows the
difference between the two settings is confined to the vertical block.

Marked ``reference``: skips (not fails) when xtrack or its JIT compiler is
unavailable — see ``docs/CONVENTIONS.md``.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from accsim import (
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    chromatic_functions,
    second_order_chromaticity,
)
from accsim.orbit import closed_orbit_nonlinear
from accsim.symplectic import jacobian

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
LQ, K1, LD, LB, ANG, N_CELLS = 0.3, 1.2, 0.5, 1.0, 0.12, 3

# xtrack's own second difference is noise-limited below ~5e-4 on these rings (its
# closed-orbit tolerance enters a second difference as 1/delta^2), and truncation
# limited above ~2e-3. 1e-3 sits in the flat middle for both codes.
DELTA = 1e-3


def _ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _accsim_lattice(*, bends: bool) -> Lattice:
    els: list = []
    for _ in range(N_CELLS):
        straight = [Drift(LB), Drift(LB)]
        bend = [Dipole(LB, ANG), Dipole(LB, ANG)]
        first, second = bend if bends else straight
        els += [
            Quadrupole(LQ, K1),
            Drift(LD),
            Drift(LD),
            first,
            Quadrupole(LQ, -K1),
            second,
            Drift(LD),
        ]
    return Lattice(els, _ref())


def _xtrack_elements(*, bends: bool) -> list:
    els: list = []
    for _ in range(N_CELLS):
        first = xt.Bend(length=LB, angle=ANG, k0=ANG / LB) if bends else xt.Drift(length=LB)
        second = xt.Bend(length=LB, angle=ANG, k0=ANG / LB) if bends else xt.Drift(length=LB)
        els += [
            xt.Quadrupole(length=LQ, k1=K1),
            xt.Drift(length=LD),
            xt.Drift(length=LD),
            first,
            xt.Quadrupole(length=LQ, k1=-K1),
            second,
            xt.Drift(length=LD),
        ]
    return els


@functools.cache
def _line(bends: bool, edge: str = "suppressed"):
    """A built xtrack line, cached: every ``xt.Line`` build JIT-compiles a fresh kernel."""
    line = xt.Line(elements=_xtrack_elements(bends=bends))
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    line.configure_bend_model(core="bend-kick-bend", edge=edge)
    try:
        line.build_tracker()
        line.twiss(method="4d")  # probe: raises here if the JIT cannot build
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


# ---------------------------------------------------------------------------
# 1. The convention, pinned by probe
# ---------------------------------------------------------------------------


def test_chromatic_function_convention_is_ddelta_and_mad8_normalised() -> None:
    r"""``bx_chrom`` and ``ax_chrom`` are the MAD8 combinations of ``d/ddelta``.

    Finite-differences xtrack's *own* ``betx``/``alfx`` at ``delta = +/- h`` and
    rebuilds its reported ``bx_chrom``/``ax_chrom`` from them. This pins three
    things at once that a remembered formula gets wrong: that the derivative is
    with respect to ``delta`` (not ``pzeta``, not ``ptau``), that ``b`` is
    normalised by ``beta`` while ``a`` is **not**, and that ``a`` carries the
    ``- dbeta alpha / beta`` correction rather than being a bare ``dalpha``.
    """
    line = _line(True)
    tw0 = line.twiss(method="4d")
    twp = line.twiss(method="4d", delta0=+DELTA)
    twm = line.twiss(method="4d", delta0=-DELTA)

    dbetx = (twp.betx[0] - twm.betx[0]) / (2.0 * DELTA)
    dalfx = (twp.alfx[0] - twm.alfx[0]) / (2.0 * DELTA)
    betx = 0.5 * (twp.betx[0] + twm.betx[0])
    alfx = 0.5 * (twp.alfx[0] + twm.alfx[0])

    assert dbetx / betx == pytest.approx(tw0.bx_chrom[0], rel=1e-4)
    assert dalfx - dbetx * alfx / betx == pytest.approx(tw0.ax_chrom[0], rel=1e-4)
    assert np.hypot(tw0.ax_chrom[0], tw0.bx_chrom[0]) == pytest.approx(tw0.wx_chrom[0], rel=1e-9)


# ---------------------------------------------------------------------------
# 2. The validated deliverable: the chromatic functions, element by element
# ---------------------------------------------------------------------------


def test_chromatic_functions_match_xtrack_around_a_bendy_ring() -> None:
    r"""``dbeta/ddelta`` agrees with xtrack at **every** element boundary.

    The milestone's headline cross-check. A one-turn scalar can be right by
    cancellation around the ring; a curve cannot, so this is what makes the
    agreement mean the optics are right rather than merely the total.

    The tolerance is set by the two codes' finite steps (both differentiate a
    twiss, neither is exact), not by a physics difference — the residual is flat
    around the ring at the ``1e-5`` relative level rather than growing at any
    particular element.
    """
    lattice = _accsim_lattice(bends=True)
    line = _line(True)
    tw = line.twiss(method="4d")
    chrom = chromatic_functions(lattice, delta=DELTA)

    assert len(chrom) == len(tw.betx)
    for i, ch in enumerate(chrom):
        assert ch.b_x == pytest.approx(tw.bx_chrom[i], rel=2e-4, abs=1e-5)
        assert ch.b_y == pytest.approx(tw.by_chrom[i], rel=2e-4, abs=1e-5)
        assert ch.a_x == pytest.approx(tw.ax_chrom[i], rel=2e-4, abs=1e-5)
        assert ch.a_y == pytest.approx(tw.ay_chrom[i], rel=2e-4, abs=1e-5)
        assert ch.w_x == pytest.approx(tw.wx_chrom[i], rel=2e-4, abs=1e-5)

    # The functions genuinely vary around the ring, so the agreement above is not
    # the trivial one of two nearly-constant curves.
    b_values = [c.b_x for c in chrom]
    assert max(b_values) - min(b_values) > 0.5


# ---------------------------------------------------------------------------
# 3. The control: Q'' where there are no bends
# ---------------------------------------------------------------------------


def test_second_order_chromaticity_matches_xtrack_without_bends() -> None:
    """``Q''`` agrees with xtrack to seven digits on a bend-free ring.

    The control for the disagreement below: it proves the second-difference
    machinery, the thick-quadrupole map, the exact drift and the phase
    accumulation are all right at second order in ``delta``. Whatever separates
    the two codes on a bendy ring, it is none of these.
    """
    lattice = _accsim_lattice(bends=False)
    line = _line(False)

    ours = second_order_chromaticity(lattice, delta=DELTA)
    twp = line.twiss(method="4d", delta0=+DELTA)
    tw0 = line.twiss(method="4d")
    twm = line.twiss(method="4d", delta0=-DELTA)
    theirs = (
        (twp.qx - 2.0 * tw0.qx + twm.qx) / DELTA**2,
        (twp.qy - 2.0 * tw0.qy + twm.qy) / DELTA**2,
    )
    assert ours[0] == pytest.approx(theirs[0], rel=1e-6)
    assert ours[1] == pytest.approx(theirs[1], rel=1e-6)


# ---------------------------------------------------------------------------
# 4. The named boundary, and the evidence that it is not accsim's maps
# ---------------------------------------------------------------------------


def test_the_dipole_jacobian_equals_xtracks_on_the_off_momentum_orbit() -> None:
    r"""accsim's ``Dipole`` and ``xt.Bend`` linearise **identically** off-momentum.

    This is the load-bearing evidence for the boundary named below. The Jacobian —
    not the tracked point — is what sets the tune, so it is the Jacobian that is
    compared, at the place the bendy ring actually samples: on the dispersion orbit
    (``x != 0``) at a non-zero ``delta``.

    With xtrack's nonlinear fringe suppressed the two agree to ``5e-9``, which is
    the finite-difference floor of the comparison itself. So every momentum-
    dependent entry of accsim's bend — its weak focusing, its dispersion generation
    and its path lengthening — is xtrack's, to the precision this test can see.
    """
    ref = _ref()
    lattice = _accsim_lattice(bends=True)
    orbit = closed_orbit_nonlinear(lattice, delta=DELTA)
    state = np.array([orbit[0], orbit[1], 0.0, 0.0, 0.0, DELTA])
    step = 1e-7

    ours = jacobian(lambda s: Dipole(LB, ANG).track(s, ref), state, step=step)

    line = xt.Line(elements=[xt.Bend(length=LB, angle=ANG, k0=ANG / LB)])
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    line.configure_bend_model(core="bend-kick-bend", edge="suppressed")
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")

    coords: list[np.ndarray] = []
    for j in range(6):
        for sign in (+1.0, -1.0):
            shifted = state.copy()
            shifted[j] += sign * step
            coords.append(shifted)
    stacked = np.array(coords)
    particles = line.build_particles(
        x=stacked[:, 0],
        px=stacked[:, 1],
        y=stacked[:, 2],
        py=stacked[:, 3],
        zeta=stacked[:, 4],
        delta=stacked[:, 5],
    )
    line.track(particles)
    order = np.argsort(particles.particle_id)
    out = np.stack(
        [
            particles.x[order],
            particles.px[order],
            particles.y[order],
            particles.py[order],
            particles.zeta[order],
            particles.delta[order],
        ]
    )
    theirs = np.zeros((6, 6))
    for j in range(6):
        theirs[:, j] = (out[:, 2 * j] - out[:, 2 * j + 1]) / (2.0 * step)

    assert np.max(np.abs(np.asarray(ours) - theirs)) < 5e-9


def test_the_off_momentum_closed_orbits_agree() -> None:
    """The two codes put the off-momentum beam in the same place, to ``1e-9``.

    The other half of the evidence: identical maps would still permit different
    tunes if the two codes linearised about different orbits. They do not — this
    includes the second-order dispersion that separates ``x_co(+delta)`` from
    ``-x_co(-delta)``.
    """
    lattice = _accsim_lattice(bends=True)
    line = _line(True)
    for delta in (+DELTA, -DELTA):
        ours = closed_orbit_nonlinear(lattice, delta=delta)
        tw = line.twiss(method="4d", delta0=delta)
        assert ours[0] == pytest.approx(tw.x[0], abs=1e-9)
        assert ours[1] == pytest.approx(tw.px[0], abs=1e-9)


def test_the_edge_model_difference_is_confined_to_the_vertical_block() -> None:
    r"""xtrack's ``edge='full'`` adds vertical focusing a sector bend has no edge angle for.

    accsim's :func:`~accsim.elements.dipole._edge_matrix` is the *linear hard-edge*
    kick and is the identity at ``e1 = e2 = 0``; xtrack's ``edge='full'`` applies a
    nonlinear wedge/fringe that focuses vertically regardless.

    The fringe turns out to be **invisible on-momentum** — it moves neither tune at
    ``delta = 0``, to thirteen digits — and to act only at *second* order in
    ``delta``, in the vertical plane alone. That is a sharper statement than "the
    edge model matters", and it is why the choice is not a free parameter for this
    milestone: ``Q''_x``, the 5% disagreement named below, is identical under both
    settings, so the edge model cannot be what explains it.
    """
    line_full = _line(True, "full")
    line_supp = _line(True)  # the default; reusing the key avoids a third JIT build

    # On-momentum, the nonlinear fringe changes nothing at all in either plane.
    assert line_full.twiss(method="4d").qx == pytest.approx(
        line_supp.twiss(method="4d").qx, rel=1e-11
    )
    assert line_full.twiss(method="4d").qy == pytest.approx(
        line_supp.twiss(method="4d").qy, rel=1e-11
    )

    def second_difference(line, plane: str) -> float:
        twp = line.twiss(method="4d", delta0=+DELTA)
        tw0 = line.twiss(method="4d")
        twm = line.twiss(method="4d", delta0=-DELTA)
        get = (lambda t: t.qx) if plane == "x" else (lambda t: t.qy)
        return (get(twp) - 2.0 * get(tw0) + get(twm)) / DELTA**2

    # Off-momentum it acts, and only in the vertical plane.
    assert second_difference(line_full, "x") == pytest.approx(
        second_difference(line_supp, "x"), rel=1e-4
    )
    assert second_difference(line_full, "y") != pytest.approx(
        second_difference(line_supp, "y"), rel=1e-3
    )


def test_second_order_chromaticity_disagrees_with_bends_and_it_is_not_the_maps() -> None:
    r"""The named boundary: with bends, the two codes' ``Q''`` differ by ~5%.

    Pinned, not blessed. Two tests above establish that accsim's bend Jacobian and
    the off-momentum closed orbit both match xtrack, and the analytic suite
    establishes that accsim's two independent tune routes agree with each other, so
    this cannot be located in accsim's model of the machine. MAD-X gives a *third*
    answer (see ``test_chromatic_optics_madx.py``), which is why no reference is
    treated as the arbiter here.

    The assertion is deliberately two-sided: the gap is real (so the milestone
    cannot be quietly declared validated) **and** bounded (so a future change that
    made it much worse would fail).

    Both sides are computed live, so nothing here can go stale against an xtrack
    version bump. The ``0.055`` window is a property of **this ring**, not of the
    two codes: the gap scales as the square of the bending angle, so it runs from
    ~1% at ``angle = 0.03`` to ~11% at ``0.12`` (measured against MAD-X). The ring
    is pinned in this module's constants — if they are ever changed, this window
    must be re-measured rather than widened.
    """
    lattice = _accsim_lattice(bends=True)
    line = _line(True)

    ours = second_order_chromaticity(lattice, delta=DELTA)[0]
    twp = line.twiss(method="4d", delta0=+DELTA)
    tw0 = line.twiss(method="4d")
    twm = line.twiss(method="4d", delta0=-DELTA)
    theirs = (twp.qx - 2.0 * tw0.qx + twm.qx) / DELTA**2

    # ...while Q and Q' agree, which is what makes the second-order gap specific.
    assert ours != pytest.approx(theirs, rel=1e-3)
    assert abs(ours / theirs - 1.0) == pytest.approx(0.055, abs=0.02)
