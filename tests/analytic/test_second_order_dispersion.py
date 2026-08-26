r"""M3: second-order dispersion, gated against answers derived without accsim.

``ddisp_x = d^2 x_co/ddelta^2`` is where the off-momentum closed orbit sits once the
straight-line term is used up. Two independent derivations gate it here, and neither
of them is a reference code:

1. **M2's minimal ring, at sixty digits.** ``tests/_m2_minimal_ring.py`` already
   builds that ring's closed orbit from lab-frame geometry for M2's ``Q''``;
   differentiating the same fixed point twice costs three extra solves and gives a
   twenty-digit answer. accsim converges onto it at second order in the step.
2. **A ring with no bend in it at all, in exact rational arithmetic.** A thin
   corrector plus thin quadrupoles plus *paraxial* drifts has a closed orbit that is
   a rational function of ``delta``, because momentum enters only as
   ``L -> L/(1+delta)``. sympy solves it in closed form here — no floating point, no
   iteration — and accsim's exact drift departs from it at the **third** power of the
   kick angle, which is measured as an exponent rather than asserted as a tolerance.

**The milestone's finding, and it reverses what was written down in advance.** The
roadmap pre-committed that any ``ddx`` cross-check would have to force
``xt.Drift(model="exact")`` or reproduce M1's 5% disagreement. It does not: the exact
and paraxial drifts put the closed orbit in different places only at ``O(delta^3)``,
so a symmetric second difference — of an odd function — cannot see the difference at
all. ``Q''`` is split by the same term because ``Q''`` differentiates the *Jacobian*
about the orbit, which brings the ``O(px^3)`` displacement down one order. Both halves
of that statement are asserted below, out of the same arbiter, with no reference code
in the room.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Corrector,
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinQuadrupole,
    ThinSextupole,
    ThinSkewQuadrupole,
    chromatic_functions,
    closed_twiss,
    second_order_dispersion,
)
from accsim.twiss import CoupledLatticeError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _m2_minimal_ring as arbiter  # noqa: E402

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0


def _ref(gamma: float = GAMMA0) -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, gamma)


# ---------------------------------------------------------------------------
# 1. the value, against the sixty-digit arbiter
# ---------------------------------------------------------------------------


def test_the_minimal_rings_second_order_dispersion_is_the_independently_derived_one() -> None:
    r"""Both orders, against M2's arbiter — and the first one against the linear solve.

    The second-order coefficients are the milestone's deliverable. The first-order
    ones are a free by-product of the same three orbits, and they are checked twice:
    against the arbiter, and against :func:`~accsim.closed_twiss`'s matched
    ``(I - M4)^-1 d``. That second comparison is the one that would catch a sign or a
    factor shared between the two second-order terms, because the linear dispersion
    was pinned against xtrack back in Stage 1 and has not moved since.
    """
    exact = arbiter.dispersion_orders(exact_drift=True)
    lattice = arbiter.lattice()
    point = second_order_dispersion(lattice, delta=1e-3)[0]

    assert point.ddisp_x == pytest.approx(exact["dd_x"], abs=5e-8)
    assert point.ddisp_px == pytest.approx(exact["dd_px"], abs=5e-8)
    assert point.disp_x == pytest.approx(exact["D_x"], abs=5e-8)
    assert point.disp_px == pytest.approx(exact["D_px"], abs=5e-8)

    twiss0 = closed_twiss(lattice)
    assert point.disp_x == pytest.approx(twiss0.disp_x, abs=1e-7)
    assert point.disp_px == pytest.approx(twiss0.disp_px, abs=1e-7)

    # The ring is flat: the vertical plane has no dispersion at either order.
    assert point.ddisp_y == 0.0 and point.ddisp_py == 0.0
    assert point.disp_y == 0.0 and point.disp_py == 0.0


def test_the_residual_against_the_arbiter_is_second_order_in_the_momentum_step() -> None:
    r"""Halving ``delta`` quarters the error — the gate is the order, not a tolerance.

    B4's argument applied to a derivative: a value at one step can be made to pass by
    choosing the step, whereas a convergence *rate* cannot. The truncation term of a
    symmetric second difference is the fourth derivative over twelve, so the residual
    must fall as ``delta^2`` — and it does, over two halvings, before the closed-orbit
    solve's own noise (which grows as ``1/delta^2``) takes over far below here.
    """
    exact = arbiter.dispersion_orders(exact_drift=True)["dd_x"]
    lattice = arbiter.lattice()
    residuals = [
        abs(second_order_dispersion(lattice, delta=d)[0].ddisp_x - exact)
        for d in (1e-2, 5e-3, 2.5e-3)
    ]
    for coarse, fine in zip(residuals, residuals[1:], strict=False):
        assert coarse / fine == pytest.approx(4.0, rel=0.02)


# ---------------------------------------------------------------------------
# 2. the finding: the drift model is invisible here, and M2 says why it should not be
# ---------------------------------------------------------------------------


def test_the_two_drift_models_give_the_same_second_order_dispersion() -> None:
    r"""The pre-committed warning was wrong, and the arbiter settles it on its own.

    M2's whole subject is that accsim's exact drift and xtrack's/MAD-X's paraxial one
    give ``Q''`` values 5% apart on this very ring. Asserted alongside it here, so the
    two are not separable by argument: the *same* ring, the *same* two drift models,
    and the second-order dispersion is identical to the arbiter's last digit while the
    second-order chromaticity is not.
    """
    exact = arbiter.dispersion_orders(exact_drift=True)
    paraxial = arbiter.dispersion_orders(exact_drift=False)
    for key in ("D_x", "D_px", "dd_x", "dd_px"):
        assert abs(exact[key] - paraxial[key]) < 1e-15

    q_exact = arbiter.second_order_chromaticity(exact_drift=True)
    q_paraxial = arbiter.second_order_chromaticity(exact_drift=False)
    assert abs(q_exact["x"] - q_paraxial["x"]) > 1e-2  # the M2 split, on the same ring


def test_the_drift_model_splits_the_closed_orbit_at_the_third_power_of_delta() -> None:
    r"""Why: the split is odd in ``delta``, so a symmetric second difference kills it.

    The exact drift exceeds the paraxial one by the relative factor
    ``(px^2 + py^2)/(2 (1+delta)^2)``, and on the closed orbit ``px ~ D_px delta``, so
    the displacement difference goes as ``delta^3``. Measured as a ratio held fixed
    over three decades — a pure cubic — rather than as a fitted exponent, which at this
    dynamic range is the sharper statement.
    """
    splits = [arbiter.drift_model_orbit_split(e) for e in (-2, -3, -4)]
    ratios = [s["dx_over_delta3"] for s in splits]
    for coarse, fine in zip(ratios, ratios[1:], strict=False):
        assert coarse / fine == pytest.approx(1.0, rel=1e-2)
    assert abs(ratios[-1]) > 1e-3  # the split is real, not a zero being divided


def test_the_arbiters_exact_drift_reproduces_accsims_drift() -> None:
    r"""The arbiter's drift is hand-written; this is what licenses trusting its value.

    M2 pinned the arbiter's *bend* against ``exact_sector_bend_map`` for exactly this
    reason and left the drift unpinned, which was safe while the bend carried the
    answer. It no longer is: the drift is now the element the milestone's finding is
    about, so the two implementations of it are compared directly, over random states
    and at an amplitude where the exact and paraxial maps are far apart.
    """
    ref = _ref()
    rng = np.random.default_rng(3)
    with arbiter.mp.workdps(50):
        worst = 0.0
        for _ in range(120):
            state = np.array(
                [
                    rng.normal(0.0, 2e-3),
                    rng.normal(0.0, 1e-3),
                    rng.normal(0.0, 2e-3),
                    rng.normal(0.0, 1e-3),
                    0.0,
                    rng.normal(0.0, 2e-3),
                ]
            )
            ours = Drift(arbiter.LD).track(state.copy(), ref)
            theirs = arbiter.drift_map(
                [arbiter.mpf(v) for v in state[:4]] + [arbiter.mpf(state[5])],
                arbiter.mpf(arbiter.LD),
                exact=True,
            )
            worst = max(worst, max(abs(float(theirs[i]) - ours[i]) for i in range(4)))
    assert worst < 1e-15


# ---------------------------------------------------------------------------
# 3. a closed form in exact arithmetic, with no bend anywhere in the ring
# ---------------------------------------------------------------------------

# A thin corrector, thin quadrupoles, and drifts. Deliberately asymmetric: the
# symmetric arrangement is degenerate (see the last test in this section).
CORRECTOR_CELL = (("d", 0.5), ("q", 0.9), ("d", 0.7), ("q", -1.3), ("d", 0.3), ("q", 0.4))


def _paraxial_orbit_symbolically(cell: tuple, kick: sp.Symbol, d: sp.Symbol) -> tuple:
    r"""The ``(x, px)`` closed orbit of a thin-lens corrector ring, in exact arithmetic.

    Momentum enters a **paraxial** drift only as ``x += L px/(1+delta)``, i.e. as
    ``L -> L/(1+delta)`` and nothing else, because a thin quadrupole's kick and a
    corrector's kick are both momentum-independent (a fixed field changes every
    particle's *momentum* equally — see :class:`~accsim.ThinQuadrupole`). So the
    one-turn map is a matrix with rational entries in ``delta``, the fixed point
    ``(I - M)^-1 M k`` is a rational function, and its second derivative at ``delta=0``
    is an exact rational number. Nothing here is differentiated numerically.
    """

    def drift(length: sp.Expr) -> sp.Matrix:
        return sp.Matrix([[1, length / (1 + d)], [0, 1]])

    def quad(k1l: sp.Expr) -> sp.Matrix:
        return sp.Matrix([[1, 0], [-k1l, 1]])

    one_turn = sp.eye(2)
    for kind, value in cell:
        v = sp.nsimplify(value, rational=True)
        one_turn = (drift(v) if kind == "d" else quad(v)) * one_turn
    orbit = (sp.eye(2) - one_turn).inv() * (one_turn * sp.Matrix([0, kick]))
    return sp.simplify(orbit[0]), sp.simplify(orbit[1])


def _corrector_lattice(cell: tuple, theta: float) -> Lattice:
    elements: list = [Corrector(kick_x=theta)]
    for kind, value in cell:
        elements.append(Drift(value) if kind == "d" else ThinQuadrupole(value))
    return Lattice(elements, _ref())


def _corrector_closed_form(cell: tuple) -> tuple[float, float]:
    """``(d^2x/ddelta^2, d^2px/ddelta^2)`` per unit kick angle, as exact rationals."""
    d, theta = sp.symbols("d theta", real=True)
    x0, p0 = _paraxial_orbit_symbolically(cell, theta, d)
    return (
        float(sp.diff(x0, d, 2).subs(d, 0) / theta),
        float(sp.diff(p0, d, 2).subs(d, 0) / theta),
    )


def test_a_corrector_ring_lands_on_its_exact_rational_second_order_dispersion() -> None:
    r"""A bend-free ring whose answer is a rational number, not a converged one.

    This isolates the mechanism the minimal ring mixes with bend geometry: with no
    dipole anywhere, every bit of second-order dispersion here comes from the drift,
    and the drift's paraxial part is exactly solvable. accsim tracks the *exact*
    drift, so it must land on this number plus a correction that vanishes as the kick
    angle does — quantified in the next test.
    """
    ddx, ddpx = _corrector_closed_form(CORRECTOR_CELL)
    assert abs(ddx) > 0.1  # a real number, not a degenerate zero

    theta = 1e-2
    point = second_order_dispersion(_corrector_lattice(CORRECTOR_CELL, theta), delta=1e-3)[0]
    assert point.ddisp_x == pytest.approx(ddx * theta, rel=1e-3)
    assert point.ddisp_px == pytest.approx(ddpx * theta, rel=1e-3)


def test_the_exact_drifts_correction_is_third_order_in_the_kick_angle() -> None:
    r"""What separates accsim from the closed form scales as ``theta^3``, and only that.

    The exact drift exceeds the paraxial one by ``L px^3/(2 pz^3) + ...``, and on this
    ring ``px`` on the closed orbit is proportional to the kick angle. Tripling the
    angle must therefore multiply the residual by 27 — an exponent, which discriminates
    where a tolerance would not: a *uniformly* mis-scaled second-order dispersion would
    show up as a residual growing like ``theta`` and would sail through any tolerance
    chosen on one ring.
    """
    ddx, _ = _corrector_closed_form(CORRECTOR_CELL)
    residuals = []
    for theta in (0.06, 0.02):
        point = second_order_dispersion(_corrector_lattice(CORRECTOR_CELL, theta), delta=1e-3)[0]
        residuals.append(abs(point.ddisp_x - ddx * theta))
    assert residuals[0] / residuals[1] == pytest.approx(27.0, rel=0.05)


def test_a_symmetric_corrector_ring_has_exactly_zero_second_order_dispersion() -> None:
    r"""The degenerate control: a real ring whose answer is zero, and accsim returns zero.

    A symmetric thin FODO with the kick at the entrance has a closed orbit that is
    **linear** in ``delta`` to all orders of the paraxial map — sympy returns a
    numerator of degree one — so its second-order dispersion is not small, it is
    identically zero. Any spurious additive term in the implementation (a stray
    on-momentum orbit, a mis-centred difference) would have nothing to hide behind here.
    """
    symmetric = (("d", 0.6), ("q", 1.1), ("d", 0.6), ("q", -1.1))
    ddx, ddpx = _corrector_closed_form(symmetric)
    assert ddx == 0.0 and ddpx == 0.0

    point = second_order_dispersion(_corrector_lattice(symmetric, 1e-2), delta=1e-3)[0]
    assert abs(point.ddisp_x) < 1e-8
    assert abs(point.ddisp_px) < 1e-8
    # ...while the *first* order is emphatically not zero, so the ring is not trivial.
    assert abs(point.disp_x) > 0.01


# ---------------------------------------------------------------------------
# 4. what drives it: the exponent in the sextupole strength
# ---------------------------------------------------------------------------


def _arc(*, bends: bool = True, k2l: float = 0.0, skew: float = 0.0) -> Lattice:
    elements: list = []
    for _ in range(3):
        elements += [
            Quadrupole(0.3, 1.2),
            Drift(0.5),
            Drift(0.5),
            Dipole(1.0, 0.12) if bends else Drift(1.0),
            Quadrupole(0.3, -1.2),
            Dipole(1.0, 0.12) if bends else Drift(1.0),
            Drift(0.5),
        ]
        if k2l:
            elements.append(ThinSextupole(k2l))
        if skew:
            elements.append(ThinSkewQuadrupole(skew))
    return Lattice(elements, _ref())


def test_a_sextupole_moves_it_linearly_in_the_sextupole_strength() -> None:
    r"""Exponent **one** — and M1's ``Q''`` took the same element at exponent two.

    A sextupole at dispersion ``D`` sees an orbit ``D delta`` and gives back a dipole
    kick ``-1/2 k2l (D delta)^2``. That is second order in ``delta`` and first order in
    ``k2l``, so it lands on this quantity *linearly* and exactly once. ``Q''`` cannot be
    reached that way — the same feed-down enters the tune as a gradient, which is first
    order in ``delta``, so the tune's curvature needs the perturbation twice and goes as
    ``k2l^2`` (M1 measured ``2.02``). One element, two quantities, two different powers:
    a uniformly mis-scaled sextupole kick would be invisible to a tolerance on either
    and is caught by the pair.
    """
    base = second_order_dispersion(_arc(), delta=1e-3)[0].ddisp_x
    shifts = []
    for k2l in (0.04, 0.08, 0.16):
        shifts.append(second_order_dispersion(_arc(k2l=k2l), delta=1e-3)[0].ddisp_x - base)
    assert abs(shifts[0]) > 0.1
    for coarse, fine in zip(shifts[1:], shifts, strict=False):
        assert coarse / fine == pytest.approx(2.0, rel=1e-3)


def test_a_ring_without_bends_has_no_dispersion_at_either_order() -> None:
    """The control: no bending, no off-momentum orbit, nothing to curve."""
    for point in second_order_dispersion(_arc(bends=False), delta=1e-3):
        assert abs(point.disp_x) < 1e-12
        assert abs(point.ddisp_x) < 1e-9
        assert abs(point.ddisp_px) < 1e-9


# ---------------------------------------------------------------------------
# 5. it is a property of the orbit, so it survives where the optics do not
# ---------------------------------------------------------------------------


def test_it_is_defined_on_a_coupled_ring_where_the_chromatic_functions_are_not() -> None:
    r"""The capability difference that makes this a different object from M1's.

    :func:`~accsim.chromatic_functions` differentiates a Courant-Snyder ``beta``, which
    an x-y coupled lattice does not have, so it refuses. A closed orbit exists all the
    same, and a skew quadrupole standing at horizontal dispersion tilts it into the
    vertical plane at **both** orders. Routing this function through the tracked orbit
    rather than through the on-orbit Twiss is what buys that, and it is asserted here
    rather than left as an accident of the implementation.
    """
    coupled = _arc(skew=0.05)
    with pytest.raises(CoupledLatticeError):
        chromatic_functions(coupled)

    point = second_order_dispersion(coupled, delta=1e-3)[0]
    assert abs(point.disp_y) > 0.1
    assert abs(point.ddisp_y) > 0.1
    # The same ring without the skew quadrupole has none of it, so the vertical
    # second-order dispersion is the coupling's doing and not the ring's.
    assert second_order_dispersion(_arc(), delta=1e-3)[0].ddisp_y == 0.0


# ---------------------------------------------------------------------------
# 6. the API contract
# ---------------------------------------------------------------------------


def test_the_points_align_with_the_twiss_grid_and_close_on_themselves() -> None:
    """One point per boundary, ``s`` from the element lengths, periodic in both orders."""
    lattice = _arc()
    points = second_order_dispersion(lattice, delta=1e-3)
    assert len(points) == len(lattice.elements) + 1
    assert points[0].s == 0.0
    assert points[-1].s == pytest.approx(sum(e.length for e in lattice.elements))
    assert points[-1].ddisp_x == pytest.approx(points[0].ddisp_x, rel=1e-9)
    assert points[-1].disp_x == pytest.approx(points[0].disp_x, rel=1e-9)
    assert all(math.isfinite(p.ddisp_x) for p in points)


def test_the_momentum_step_must_be_positive() -> None:
    """``delta`` is a step size, not a tolerance; zero or negative is a caller error."""
    for bad in (0.0, -1e-3):
        with pytest.raises(ValueError, match="delta must be > 0"):
            second_order_dispersion(_arc(), delta=bad)
