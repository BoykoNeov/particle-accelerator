r"""P1 against xtrack's ``get_T_matrix`` — the weakest leg, and written down as such.

xtrack obtains ``T`` by finite differences of its own tracking, which is accsim's method
too. So this leg **shares the method** and can only gate what the method cannot get wrong
on its own: the storage convention (symmetric — ``T[3,0,2] = T[3,2,0] = k2l/2`` on a thin
sextupole, measured 2026-08-31 before the milestone was written), the index bookkeeping
of a ``6x6x6`` object, the coordinate frame, and — what earns the leg its place — the
**misalignment half**: it is the only arbiter that consumes ``shift_x`` / ``shift_y``, so
a displaced sextupole's map about a steered closed orbit has this file as its one
external check. The physics is gated next door, by MAD-X's analytic ``sectormap`` and
PTC's differential algebra.

**xtrack's ``T`` is in ``(zeta, p_zeta)``, not ``(zeta, delta)`` — measured, and it is the
frame this leg is compared in.** Its drift lands on ``T[zeta, delta, delta] =
-3L/(2 gamma0^2)`` where the ``(zeta, delta)`` value is ``-L (2 + beta0^2)/(2 gamma0^2)``,
and its bend on ``T[px, delta, delta] = -R26/(2 gamma0^2)`` where the ``(zeta, delta)``
value is ``0``. Both are exactly the canonical conversion ``delta = p_zeta -
p_zeta^2/(2 gamma0^2) + ...``, i.e. ``T_can[i, 5, 5] = T[i, 5, 5] - R[i, 5]/(2 gamma0^2)``
for anything that leaves ``delta`` alone; O1 recorded "nothing to correct at linear order"
and this is the order at which there is. accsim's maps are conjugated into the canonical
pair with :func:`accsim.canonical_map` before every comparison here, and the two closed
forms are asserted so the choice of frame is a gate and not a fit.

**Two other things about the API that were measured rather than read.** ``start``/``end``
are start-inclusive and end-*exclusive* (``start == end`` returns a full turn). And the
default differencing steps put xtrack's second differences at ``eps/h^2 ~ 1e-4`` —
``3.9e-4`` on a thick quadrupole, no gate at all — so the steps are passed explicitly at
``3e-4``, where its second-order stencils sit near ``1e-7`` on thick elements
(``4e-8`` on the drift, ``1e-6`` on the bend) and thin kicks are exact.

**The drift model is matched first**, M2's finding applied a fourth time: xtrack's
default drift is the expanded one and accsim's is exact. The bend is ``bend-kick-bend``
(accsim's exact sector bend, per ``test_dipole_xtrack``) with xtrack's default *linear*
edge model, which — like accsim and unlike MAD-X's default — carries no second-order
fringe; see the MAD-X leg. And the closed orbit is asked for **at fixed ``delta = 0``**:
without that xtrack solves in 6D on an RF-free ring and closes the arrival time by going
off-momentum (``delta_co = -1.04e-3`` here), which is a different, equally valid fixed
point that accsim's 4D solve does not look for.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    Corrector,
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    TaylorMap,
    ThinOctupole,
    ThinSextupole,
    canonical_map,
    closed_orbit_nonlinear,
    second_order_element_maps,
    taylor_expand,
)
from accsim.coords import DELTA, DIM, PX, PY, ZETA, X, Y
from accsim.symplectic import to_canonical

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0, GAMMA0 = 938.27208816e6, 20.0
KF, KD = 1.2, -1.2
LQ, LB, ANGLE = 0.3, 1.0, 0.1
K2L, K3L = 2.0, 300.0
KICK = 3.0e-4
SHIFT_X, SHIFT_Y = 1.0e-3, -0.5e-3

NAMES = ["qf", "d1", "ms", "d2", "mb", "qd", "d3", "mo", "d4"]

#: xtrack's differencing steps (module docstring): ``3e-4`` in every coordinate.
STEPS = dict.fromkeys(("dx", "dpx", "dy", "dpy", "dzeta", "ddelta"), 0.0003)

#: Measured floors, 2026-09-02, with those steps and the canonical frame: thin kicks
#: ``1e-10``, the drift ``4e-8``, the bend ``1e-6``; the gates carry ~10x headroom.
THIN_ATOL = 1e-8
THICK_ATOL = 5e-6


def _accsim_cell(shift: bool = False) -> list:
    dx, dy = (SHIFT_X, SHIFT_Y) if shift else (0.0, 0.0)
    return [
        Quadrupole(LQ, KF),
        Drift(0.5),
        ThinSextupole(K2L, dx=dx, dy=dy),
        Drift(0.5),
        Dipole(LB, ANGLE),
        Quadrupole(LQ, KD),
        Drift(0.4),
        ThinOctupole(K3L),
        Drift(0.6),
    ]


def _xtrack_cell(shift: bool = False) -> list:
    sext = xt.Multipole(knl=[0.0, 0.0, K2L])
    if shift:
        sext.shift_x, sext.shift_y = SHIFT_X, SHIFT_Y
    return [
        xt.Quadrupole(length=LQ, k1=KF),
        xt.Drift(length=0.5, model="exact"),
        sext,
        xt.Drift(length=0.5, model="exact"),
        xt.Bend(length=LB, angle=ANGLE, k0=ANGLE / LB, model="bend-kick-bend"),
        xt.Quadrupole(length=LQ, k1=KD),
        xt.Drift(length=0.4, model="exact"),
        xt.Multipole(knl=[0.0, 0.0, 0.0, K3L]),
        xt.Drift(length=0.6, model="exact"),
    ]


def _ring(*, cells: int, kick: float = 0.0, shift: bool = False):
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    acc: list = [Corrector(kick_x=kick)] if kick else []
    xtr: list = [xt.Multipole(knl=[-kick])] if kick else []
    names: list[str] = ["kck"] if kick else []
    for c in range(cells):
        acc += _accsim_cell(shift)
        xtr += _xtrack_cell(shift)
        names += [f"{n}{c}" for n in NAMES]
    line = xt.Line(elements=xtr, element_names=names)
    line.particle_ref = xt.Particles(mass0=MASS0, gamma0=GAMMA0)
    line.build_tracker()
    return Lattice(acc, ref), line, names


def _canonical_element_maps(lat: Lattice, z0: np.ndarray) -> list[TaylorMap]:
    """accsim's element maps in ``(zeta, p_zeta)``, expanded about the tracked orbit."""
    ref = lat.ref
    maps = []
    state = np.asarray(z0, dtype=float)
    for elem in lat.elements:
        fn = canonical_map(lambda s, e=elem: e.track(s, ref), ref)
        maps.append(taylor_expand(fn, to_canonical(state, ref)))
        state = elem.track(state, ref)
    return maps


def _canonical_turn(lat: Lattice, z0: np.ndarray) -> TaylorMap:
    maps = _canonical_element_maps(lat, z0)
    turn = TaylorMap.identity(maps[0].origin)
    for m in maps:
        turn = turn.then(m)
    return turn


@pytest.fixture(scope="module")
def design():
    return _ring(cells=1)


def _element_t(line, names: list[str], index: int, particle) -> np.ndarray:
    """xtrack's ``T`` of one element: from its start to the start of the next."""
    return np.asarray(
        line.get_T_matrix(
            start=names[index], end=names[index + 1], particle_on_co=particle, steps=STEPS
        )
    )


def test_xtrack_is_in_the_canonical_pair_and_the_conversion_is_closed_form(design) -> None:
    """The drift's and the bend's momentum-diagonal entries land on the ``(zeta, p_zeta)``
    values, off the ``(zeta, delta)`` ones by exactly ``-R[i, delta]/(2 gamma0^2)``."""
    lat, line, names = design
    p = line.build_particles(x=0, px=0, y=0, py=0, zeta=0, delta=0)
    plain = second_order_element_maps(lat)
    canon = _canonical_element_maps(lat, np.zeros(DIM))
    g2 = 2 * GAMMA0**2
    # the drift (index 1) and the bend (index 4)
    for index, row, tol in ((1, ZETA, 2e-7), (4, PX, 2e-7)):
        T = _element_t(line, names, index, p)
        conversion = plain[index].T[row, DELTA, DELTA] - plain[index].R[row, DELTA] / g2
        assert abs(canon[index].T[row, DELTA, DELTA] - conversion) < 1e-12  # the rule
        assert abs(T[row, DELTA, DELTA] - canon[index].T[row, DELTA, DELTA]) < tol  # xtrack
        assert abs(T[row, DELTA, DELTA] - plain[index].T[row, DELTA, DELTA]) > 1e-6  # not delta
    L = 0.5
    assert abs(plain[1].T[ZETA, DELTA, DELTA] + L * (2 + lat.ref.beta0**2) / g2) < 1e-11
    assert abs(canon[1].T[ZETA, DELTA, DELTA] + 3 * L / g2) < 1e-11


def test_thin_sextupole_convention_and_frame(design) -> None:
    lat, line, names = design
    p = line.build_particles(x=0, px=0, y=0, py=0, zeta=0, delta=0)
    T_s = _element_t(line, names, 2, p)
    assert abs(T_s[PX, X, X] + K2L / 2) < THIN_ATOL
    assert abs(T_s[PY, X, Y] - K2L / 2) < THIN_ATOL and abs(T_s[PY, Y, X] - K2L / 2) < THIN_ATOL
    assert np.max(np.abs(T_s - second_order_element_maps(lat)[2].T)) < THIN_ATOL


@pytest.mark.parametrize("index", range(len(NAMES) - 1))
def test_every_element_map_agrees_element_by_element(design, index: int) -> None:
    """Each element's ``T`` about the design orbit, all 216 entries, in the canonical pair.

    The last drift has no successor to name as ``end`` and is the same element as the
    three drifts before it; it is covered by the one-turn gate.
    """
    lat, line, names = design
    p = line.build_particles(x=0, px=0, y=0, py=0, zeta=0, delta=0)
    ours = _canonical_element_maps(lat, np.zeros(DIM))[index]
    theirs = _element_t(line, names, index, p)
    tol = THIN_ATOL if names[index].startswith(("ms", "mo")) else THICK_ATOL
    miss = np.max(np.abs(theirs - ours.T))
    assert miss < tol, (names[index], miss)


def test_element_maps_about_a_steered_orbit_with_a_displaced_sextupole() -> None:
    """The misalignment half: ``shift_x``/``shift_y`` on the sextupole, a steerer in front,
    and both codes' element maps about the closed orbit at ``delta = 0``.

    The orbits agree first (I2's gate, re-run), then each element's ``T`` about its own
    point on that orbit. On this orbit the sextupole's ``R`` carries the feed-down of its
    *offset plus orbit* and its ``T`` is unchanged — the convention ``z_0 = z_co - d`` that
    O6 could only gate internally is here checked against the one arbiter that consumes
    the shift, and it holds to ``4e-11`` on all 216 entries.
    """
    lat, line, names = _ring(cells=4, kick=KICK, shift=True)
    co = line.find_closed_orbit(delta0=0.0)
    ours_orbit = closed_orbit_nonlinear(lat)
    theirs_orbit = np.array([co.x[0], co.px[0], co.y[0], co.py[0]])
    assert abs(co.delta[0]) < 1e-15 and abs(co.zeta[0]) < 1e-12
    assert np.max(np.abs(theirs_orbit - ours_orbit)) < 1e-11
    assert abs(ours_orbit[0]) > 1e-4 and abs(ours_orbit[2]) > 1e-7  # steered in both planes

    z0 = np.zeros(DIM)
    z0[[X, PX, Y, PY]] = ours_orbit
    maps = _canonical_element_maps(lat, z0)
    plain = second_order_element_maps(lat, ours_orbit)
    state = z0.copy()
    checked = 0
    for index, (elem, name) in enumerate(zip(lat.elements, names, strict=True)):
        if name in ("qf0", "ms0", "d20", "mb0", "qd0"):
            p = line.build_particles(
                x=state[X],
                px=state[PX],
                y=state[Y],
                py=state[PY],
                zeta=state[ZETA],
                delta=state[DELTA],
            )
            theirs = _element_t(line, names, index, p)
            tol = THIN_ATOL if name == "ms0" else THICK_ATOL
            assert np.max(np.abs(theirs - maps[index].T)) < tol, name
            checked += 1
        if isinstance(elem, ThinSextupole) and name == "ms0":
            m = plain[index]
            x_rel, y_rel = m.origin[X] - SHIFT_X, m.origin[Y] - SHIFT_Y
            assert abs(m.R[PX, X] + K2L * x_rel) < 1e-10
            assert abs(m.R[PX, Y] - K2L * y_rel) < 1e-10
            assert abs(m.T[PX, X, X] + K2L / 2) < 1e-11
        state = elem.track(state, lat.ref)
    assert checked == 5


def test_one_turn_map_converges_onto_the_composed_one_as_xtrack_steps_shrink() -> None:
    """xtrack's whole-turn ``T`` about the steered orbit approaches accsim's composed map as
    the **square** of its step, and the composed map is what it converges to.

    The point is a floor, stated: a one-turn map with entries near ``700`` has fourth
    derivatives that make a second-order stencil's truncation ``0.3`` at ``1e-4`` and
    ``3e-4`` at ``3e-6`` (measured: ``0.30, 0.027, 0.0030, 0.00030`` over the four
    steps), ten-fold per ``sqrt(10)`` in the step. So the whole-turn comparison cannot be
    sharp, and the element-by-element one above is the gate that is.
    """
    lat, line, names = _ring(cells=4, kick=KICK, shift=True)
    co = line.find_closed_orbit(delta0=0.0)
    z0 = np.zeros(DIM)
    z0[[X, PX, Y, PY]] = closed_orbit_nonlinear(lat)
    turn = _canonical_turn(lat, z0)
    R = np.asarray(line.compute_one_turn_matrix_finite_differences(co)["R_matrix"])
    assert np.max(np.abs(R - turn.R)) < 1e-8
    misses = []
    for h in (3e-5, 1e-5, 3e-6):
        steps = dict.fromkeys(("dx", "dpx", "dy", "dpy", "dzeta", "ddelta"), h)
        T = np.asarray(line.get_T_matrix(particle_on_co=co, steps=steps))
        misses.append(np.max(np.abs(T - turn.T)))
    assert misses[-1] < 2e-3
    assert misses[0] / misses[1] > 5.0 and misses[1] / misses[2] > 5.0
    assert np.max(np.abs(turn.T)) > 100.0


def test_default_drift_model_would_have_shown_up_here() -> None:
    """Control: the same steered ring with xtrack's default (expanded) drift disagrees with
    accsim's exact one on ``R`` about the orbit, by more than the gate — so the model
    matching above is load-bearing and not decoration."""
    lat, line, names = _ring(cells=4, kick=KICK)
    line.configure_drift_model(model="expanded")
    co = line.find_closed_orbit(delta0=0.0)
    z0 = np.zeros(DIM)
    z0[[X, PX, Y, PY]] = closed_orbit_nonlinear(lat)
    turn = _canonical_turn(lat, z0)
    R = np.asarray(line.compute_one_turn_matrix_finite_differences(co)["R_matrix"])
    assert np.max(np.abs(R - turn.R)) > 1e-8
