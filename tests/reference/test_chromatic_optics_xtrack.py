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
4. **``Q''`` disagrees on a bendy ring, and M2 found why: the drift model.** The
   Dipole Jacobian is equal to ``xt.Bend``'s to ``1.2e-9`` entry by entry *on the
   off-momentum closed orbit*, and the closed orbits are equal � but M1 inferred
   "identical maps" from that and never checked the **drift** off-momentum, where
   the two codes differ by ``1e-7``. accsim's ``Drift`` is exact
   (``x += L px/pz``); xtrack's default is paraxial (``x += L px/(1+delta)``).
   Setting ``xt.Drift(model="exact")`` collapses the whole disagreement to the two
   codes' own truncation error, and the per-element sweep below shows the drift is
   the only element that ever differed.

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


def _xtrack_elements(*, bends: bool, drift_model: str | None = None) -> list:
    """``drift_model=None`` is xtrack's default (paraxial); ``"exact"`` matches accsim.

    Which of the two is used is the whole subject of M2, so it is a parameter here
    rather than a default left implicit.
    """

    def _drift(length: float):
        return xt.Drift(length=length, model=drift_model)

    els: list = []
    for _ in range(N_CELLS):
        first = xt.Bend(length=LB, angle=ANG, k0=ANG / LB) if bends else _drift(LB)
        second = xt.Bend(length=LB, angle=ANG, k0=ANG / LB) if bends else _drift(LB)
        els += [
            xt.Quadrupole(length=LQ, k1=K1),
            _drift(LD),
            _drift(LD),
            first,
            xt.Quadrupole(length=LQ, k1=-K1),
            second,
            _drift(LD),
        ]
    return els


@functools.cache
def _line(bends: bool, edge: str = "suppressed", drift_model: str | None = None):
    """A built xtrack line, cached: every ``xt.Line`` build JIT-compiles a fresh kernel."""
    line = xt.Line(elements=_xtrack_elements(bends=bends, drift_model=drift_model))
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

    The Jacobian — not the tracked point — is what sets the tune, so it is the
    Jacobian that is compared, at the place the bendy ring actually samples: on the
    dispersion orbit (``x != 0``) at a non-zero ``delta``. With xtrack's nonlinear
    fringe suppressed the two agree to ``1.2e-9``: every momentum-dependent entry of
    accsim's bend — its weak focusing, its dispersion generation and its path
    lengthening — is xtrack's.

    **What this test does not establish, and was once read as establishing.** M1
    called ``5e-9`` "the finite-difference floor of the comparison itself" and
    concluded from this test that the two codes' maps were identical, hence that
    their ``Q''`` disagreement had to live somewhere else. Both halves were wrong.
    The threshold was not a floor — M2's per-element sweep puts every dipole in this
    ring between ``6.7e-10`` and ``1.2e-9`` — and one element was never compared
    off-momentum at all. The **drift** differs by ``1e-7``, a hundred times larger,
    and that is the whole of the gap. This test remains true and useful; it is the
    *inference* drawn from it that M2 retired. See
    ``test_the_drift_is_the_element_the_two_codes_disagreed_about`` below.
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

    assert np.max(np.abs(np.asarray(ours) - theirs)) < 2e-9


def test_the_off_momentum_closed_orbits_agree() -> None:
    """The two codes put the off-momentum beam in nearly the same place — and why not exactly.

    On xtrack's default paraxial drift the two orbits agree to ``1e-9``, and that
    residual is **not** solver tolerance: it is the drift-model difference itself,
    displacing the orbit by ``~L px^3 / 2``. Switch xtrack to ``model="exact"`` and
    the same comparison tightens to ``2e-15`` — asserted below, because a residual
    that collapses by six orders of magnitude under a model change is evidence,
    where the same residual quoted alone reads as agreement.

    M1 read the ``1e-9`` as "the two codes linearise about the same orbit" and, taken
    with the dipole Jacobian above, as "identical maps about identical orbits". The
    orbit half of that was sound. The map half was not.
    """
    lattice = _accsim_lattice(bends=True)
    line = _line(True)
    for delta in (+DELTA, -DELTA):
        ours = closed_orbit_nonlinear(lattice, delta=delta)
        tw = line.twiss(method="4d", delta0=delta)
        assert ours[0] == pytest.approx(tw.x[0], abs=1e-9)
        assert ours[1] == pytest.approx(tw.px[0], abs=1e-9)

    exact_line = _line(True, "suppressed", "exact")
    for delta in (+DELTA, -DELTA):
        ours = closed_orbit_nonlinear(lattice, delta=delta)
        tw = exact_line.twiss(method="4d", delta0=delta)
        assert ours[0] == pytest.approx(tw.x[0], abs=2e-15)
        assert ours[1] == pytest.approx(tw.px[0], abs=2e-15)


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


def test_the_drift_is_the_element_the_two_codes_disagreed_about() -> None:
    r"""M2's localisation: sweep **every** element's Jacobian, on and off momentum.

    M1 compared one element — the dipole — and generalised. Walking the closed orbit
    element by element and differencing accsim's tracked Jacobian against xtrack's,
    at ``delta = 0`` and at ``delta = 1e-3``, gives a table with one outlier:

        Quadrupole   6.2e-11 on-momentum   5.3e-10 off
        Dipole       6.0e-10 on-momentum   6.7e-10 .. 1.1e-9 off
        Drift        1.0e-10 on-momentum   6.4e-08 .. 1.0e-07 off

    The drift is a hundred times the others and **only** off-momentum, which is the
    signature of a model difference rather than of finite-difference noise: accsim's
    ``Drift`` is exact, xtrack's default is paraxial, and the two coincide exactly
    when the orbit is straight.

    The whole sweep is condensed here into the one assertion that carries it: the
    worst drift exceeds the worst non-drift by more than fifty times off-momentum,
    while on-momentum every element agrees at the same ``1e-9`` level.
    """
    ref = _ref()
    lattice = _accsim_lattice(bends=True)
    orbit = closed_orbit_nonlinear(lattice, delta=DELTA)
    step = 1e-7

    # one xtrack line per element, built once each; the cell repeats, so seven suffice
    cell = _xtrack_elements(bends=True)[:7]
    lines = []
    for element in cell:
        ln = xt.Line(elements=[element])
        ln.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
        ln.configure_bend_model(core="bend-kick-bend", edge="suppressed")
        try:
            ln.build_tracker()
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
        lines.append(ln)

    def _their_jacobian(line, state: np.ndarray) -> np.ndarray:
        coords = []
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
        return np.stack([(out[:, 2 * j] - out[:, 2 * j + 1]) / (2.0 * step) for j in range(6)], 1)

    def _our_jacobian(element, state: np.ndarray) -> np.ndarray:
        return np.asarray(jacobian(lambda s: element.track(s, ref), state, step=step))

    off = np.array([orbit[0], orbit[1], 0.0, 0.0, 0.0, DELTA])
    on = np.zeros(6)
    drift_gaps, other_gaps, on_gaps = [], [], []
    for i, element in enumerate(lattice.elements):
        line = lines[i % 7]
        gap_off = float(np.max(np.abs(_our_jacobian(element, off) - _their_jacobian(line, off))))
        on_gaps.append(
            float(np.max(np.abs(_our_jacobian(element, on) - _their_jacobian(line, on))))
        )
        (drift_gaps if isinstance(element, Drift) else other_gaps).append(gap_off)
        off = element.track(off, ref)
        on = element.track(on, ref)

    assert min(drift_gaps) > 50.0 * max(other_gaps)  # the drift is the outlier, every time
    assert max(other_gaps) < 2e-9  # and nothing else in the ring differs off-momentum
    assert max(on_gaps) < 2e-9  # on-momentum the drift is not an outlier at all


def test_second_order_chromaticity_agrees_with_bends_once_the_drift_models_match() -> None:
    r"""M2's headline against xtrack: the 5% gap M1 could not place was the drift model.

    The same bendy ring, the same second difference, the same xtrack — with
    ``xt.Drift(model="exact")`` instead of the default paraxial drift:

        Q''_x   accsim 0.793072   xtrack default 0.752050   xtrack exact 0.793087
        Q''_y   accsim 0.768303   xtrack default 0.754138   xtrack exact 0.768303

    Two-sided on purpose. The **default** must still disagree by ~5%, because that
    disagreement is a real difference between two documented models and a future
    change that quietly removed it would mean accsim had stopped being exact.

    **The exact model agrees to nine digits in ``y`` and only to ``2e-5`` relative in
    ``x``, and the two planes differ for a measured reason, not a supposed one.**
    Checking each code's residual against the minimal ring's sixty-digit arbiter as
    ``delta`` halves:

        y   9.79e-6  2.45e-6  6.12e-7  1.53e-7      ratios 4.00, 4.00, 4.00
        x   1.02e-5  3.00e-6  2.96e-6  5.18e-6      ratios 3.39, 1.01, 0.57

    The vertical closed orbit is identically zero, so there is nothing to solve and
    xtrack truncates cleanly at ``delta^2`` all the way down — and accsim's vertical
    residual is ``6.12e-7`` at the same step, the *same number*, so the truncation
    cancels between the two codes and what is left is nine digits. The horizontal
    orbit must be solved, and xtrack's residual stops falling below
    ``delta ~ 2.5e-3`` and then grows: that is closed-orbit noise entering as
    ``1/delta^2``, and ``~3e-6`` is xtrack's floor on this ring. accsim is still
    converging there (``7.1e-7``, and ``1.1e-7`` one step further), so the ``2e-5``
    relative gap in ``x`` is xtrack's noise floor rather than a difference between the
    two models. The analytic suite gates accsim against the arbiter directly, which
    is why nothing here has to rest on that inference.
    """
    lattice = _accsim_lattice(bends=True)
    ours = second_order_chromaticity(lattice, delta=DELTA)

    def _their_qpp(line) -> tuple[float, float]:
        twp = line.twiss(method="4d", delta0=+DELTA)
        tw0 = line.twiss(method="4d")
        twm = line.twiss(method="4d", delta0=-DELTA)
        return tuple(
            (getattr(twp, q) - 2.0 * getattr(tw0, q) + getattr(twm, q)) / DELTA**2
            for q in ("qx", "qy")
        )

    default = _their_qpp(_line(True))
    exact = _their_qpp(_line(True, "suppressed", "exact"))

    # the default drift: still ~5% away, and that gap is a model difference, not a bug
    assert abs(ours[0] / default[0] - 1.0) == pytest.approx(0.055, abs=0.02)

    # the exact drift: the disagreement is gone in both planes
    assert ours[0] == pytest.approx(exact[0], rel=5e-5)
    assert ours[1] == pytest.approx(exact[1], rel=1e-8)


# ---------------------------------------------------------------------------
# 5. the minimal ring: each drift model against the number it should produce
# ---------------------------------------------------------------------------


@functools.cache
def _minimal_line(drift_model: str | None):
    """M2's five-element arbiter ring, built in xtrack."""
    from _m2_minimal_ring import ANG as M_ANG
    from _m2_minimal_ring import KF
    from _m2_minimal_ring import LB as M_LB
    from _m2_minimal_ring import LD as M_LD

    line = xt.Line(
        elements=[
            xt.Multipole(knl=[0.0, KF]),
            xt.Drift(length=M_LD, model=drift_model),
            xt.Bend(length=M_LB, angle=M_ANG, k0=M_ANG / M_LB),
            xt.Multipole(knl=[0.0, -KF]),
            xt.Drift(length=M_LD, model=drift_model),
        ]
    )
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    line.configure_bend_model(core="bend-kick-bend", edge="suppressed")
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


def test_each_xtrack_drift_model_reproduces_its_own_arbiter_on_the_minimal_ring() -> None:
    r"""The sharpest form of M2's result: both codes land where the geometry says they must.

    ``tests/_m2_minimal_ring.py`` derives this ring's ``Q''`` from lab-frame geometry
    at sixty digits, twice — once over the exact drift and once over the paraxial one.
    Those two numbers are properties of the *models*, computed with neither code in
    the room:

        exact drift     Q''_x = 0.3073788909    Q''_y = 0.2985909737
        paraxial drift  Q''_x = 0.2932235794    Q''_y = 0.2938154492

    xtrack on its default drift reproduces the **paraxial** number to ``4e-6``; on
    ``model="exact"`` it reproduces the **exact** one to ``3e-6``. Both residuals are
    xtrack's own second-difference truncation at ``delta = 1.25e-3``, and both are
    four thousand times smaller than the ``1.4e-2`` that separates the two models.

    This is what closes the milestone. A disagreement between two codes can always be
    argued about; a code landing on an independently derived number cannot.
    """
    from _m2_minimal_ring import second_order_chromaticity as arbiter

    step = 1.25e-3

    def _qpp(line) -> dict[str, float]:
        twp = line.twiss(method="4d", delta0=+step)
        tw0 = line.twiss(method="4d")
        twm = line.twiss(method="4d", delta0=-step)
        return {
            p: (getattr(twp, "q" + p) - 2.0 * getattr(tw0, "q" + p) + getattr(twm, "q" + p))
            / step**2
            for p in ("x", "y")
        }

    for drift_model, exact_drift in ((None, False), ("exact", True)):
        theirs = _qpp(_minimal_line(drift_model))
        wanted = arbiter(exact_drift=exact_drift)
        other = arbiter(exact_drift=not exact_drift)
        for plane in ("x", "y"):
            assert theirs[plane] == pytest.approx(wanted[plane], abs=5e-6)
            # ...and is nowhere near the other model's number
            assert abs(theirs[plane] - other[plane]) > 1000.0 * abs(theirs[plane] - wanted[plane])
