r"""Analytic gates for the wiggler (T1) — the magnet built to radiate.

The element under test is the first in the package whose field varies **along** ``s``, and
the physics content is not that it radiates (that is T2) but that its focusing lands in the
plane nobody expects. Every bend-shaped intuition puts a wiggler's focusing in the
horizontal plane, where it wiggles; Maxwell puts it in the **vertical** one, and leaves the
horizontal a plain drift.

Neither reference code has the element at all — ``xtrack`` has no wiggler among its 110
public classes, and MAD-X takes the *process* down on the keyword — so the arbiter here is
a direct integration of the Lorentz force through the real ``cos``/``sinh`` field
(:func:`_integrate`), and ``tests/reference/test_wiggler_xtrack.py`` records, as a
mechanism rather than a tolerance, the two ways xtrack can be made to spell "wiggler" and
the opposite planes in which each of them fails.

Ordered by how much each can catch:

  * **The vertical coefficient, against the integrated field.** The only discriminating
    gate there is. ``k_y = h0^2``, ``h0^2/2`` and ``h0^2/4`` are indistinguishable to
    Maxwell and to symplecticity — both of which are asserted here *together with their
    blindness*, so that no later session mistakes a structural gate for a discriminating
    one. Against the field, ``h0^2/2`` lands at ``1.26e-05`` and the other two at
    ``9.6e-02`` and ``4.9e-02``.
  * **The plane.** A drift misses the vertical block by ``9.94e-02`` — the same size as the
    wrong coefficients — while the *horizontal* block is a drift to ``5e-10``. The
    inversion is order-unity, not a tolerance.
  * **The momentum dependence is the SECOND power**, ``h0^2/(2(1+delta)^2)``, where every
    other magnet in the package carries the first. Nothing the roadmap pre-committed could
    see it: all of those gates are at ``delta = 0``.
  * **The path lengthening is a constant**, ``L theta^2/4``, exact to ``3e-12`` relative
    against the integrated trajectory. It makes this the only aligned, on-design element in
    the package with a nonzero ``kick()``.
  * **The reference trajectory closes**, gated on the integration rather than on the
    shipped map — where it would be true by construction and would test nothing.
  * **The refusals.** The field accessor raises rather than quietly answering "no field",
    and the radiation kick and the spin track refuse through it. T2 has since lifted two of
    them — the radiation *integrals* now see the wiggler
    (``tests/analytic/test_wiggler_radiation.py``), and ``_scaled`` knows a wiggler is a
    powered magnet — so the two tests at the foot of this file are the record of what T1
    refused and what it cost, not the refusal itself. ``taper()`` on a whole ring still
    refuses, through the accessor, and that one is the tracking gap.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    Wiggler,
    radiation_integrals,
    taper,
)
from accsim.coords import DELTA, PX, PY, ZETA, X, Y
from accsim.symplectic import is_symplectic_map_canonical

# --- the probe machine, the one the roadmap entry measured on ---------------------------
# B0 = 1.5 T, lambda_w = 0.1 m, N = 10 periods, E = 1 GeV electron:
#   rho0 = 2.223761 m, h0 = 0.449689 /m, k = 62.8319 /m, L = 1.0 m, theta = 7.157018e-03.
PERIOD = 0.1
PERIODS = 10
PEAK_FIELD_T = 1.5


@pytest.fixture
def ref() -> ReferenceParticle:
    """1 GeV electrons — the probe beam of the roadmap's axis-T entry."""
    return ReferenceParticle.from_total_energy(0.51099895069e6, 1.0e9, charge=-1.0)


@pytest.fixture
def wig(ref: ReferenceParticle) -> Wiggler:
    """The probe wiggler itself."""
    return Wiggler.from_peak_field(PERIOD, PEAK_FIELD_T, PERIODS, ref, name="w")


# --- the arbiter: the Lorentz force through the real field ------------------------------
def _rhs(s: float, u: np.ndarray, h0: float, k: float, delta: float, paraxial: bool) -> list[float]:
    r"""``d/ds (x, x', y, y')`` in the wiggler's own field.

    ``b_y = h0 cos(ks) cosh(ky)``, ``b_s = -h0 sin(ks) sinh(ky)``, ``b_x = 0`` — the exact
    field, not a truncation of it, so the ``cosh``/``sinh`` that carry the vertical
    focusing are present at every order.

    ``paraxial=True`` (the default everywhere below) integrates
    ``x'' = -b_y + y' b_s``, ``y'' = -x' b_s``, dropping only the
    ``sqrt(1 + x'^2 + y'^2)`` factors. That is the *same* approximation class the shipped
    map is in — every thick element here is paraxial in the angles — so it isolates the
    **averaging** error, which is what T1 is about, from the kinematic one, which is P2
    (iv)'s subject and is gated separately in
    :func:`test_the_non_paraxial_remainder_is_three_quarters_theta_squared_in_r12`.
    """
    x, xp, y, yp = u
    by = h0 * np.cos(k * s) * np.cosh(k * y)
    bs = -h0 * np.sin(k * s) * np.sinh(k * y)
    f = (1.0 if paraxial else np.sqrt(1.0 + xp * xp + yp * yp)) / (1.0 + delta)
    if paraxial:
        return [xp, f * (-by + yp * bs), yp, f * (-xp * bs)]
    return [xp, f * (-(1.0 + xp * xp) * by + yp * bs), yp, f * (-xp * yp * by - xp * bs)]


def _integrate(
    wiggler: Wiggler,
    u0: np.ndarray,
    delta: float = 0.0,
    *,
    paraxial: bool = True,
    path: bool = False,
) -> np.ndarray:
    """Integrate ``u0 = (x, x', y, y')`` through ``wiggler``'s field, exit state out.

    ``max_step`` is a twentieth of a period: an adaptive integrator handed a whole
    ten-period magnet can step over the oscillation entirely and return a very precise
    answer to the wrong problem. With ``path=True`` a fifth component accumulates
    ``int (x'^2 + y'^2)/2 ds``, the path lengthening.
    """
    h0, k, length = wiggler.h0, wiggler.wavenumber, wiggler.length

    def rhs(s: float, u: np.ndarray) -> list[float]:
        d = _rhs(s, u[:4], h0, k, delta, paraxial)
        return d + [0.5 * (u[1] ** 2 + u[3] ** 2)] if path else d

    state = np.concatenate([u0, [0.0]]) if path else np.asarray(u0, dtype=float)
    sol = solve_ivp(
        rhs,
        (0.0, length),
        state,
        rtol=1e-12,
        atol=1e-16,
        method="DOP853",
        max_step=wiggler.period / 20.0,
    )
    return sol.y[:, -1]


def _integrated_block(
    wiggler: Wiggler, delta: float = 0.0, *, paraxial: bool = True, eps: float = 1e-6
) -> np.ndarray:
    """The 4x4 transverse Jacobian of the integrated field, by central differences.

    ``eps = 1e-6`` against ``rtol = 1e-12`` is where the differencing noise (``~1e-10``, and
    it *is* noise — the value swings sign between step sizes) and the truncation of the
    vertical nonlinearity (``sinh(ky)``) are both smaller than everything gated on it. The
    convergence was measured over ``eps`` from ``1e-4`` to ``1e-8`` before this number was
    chosen; the vertical residual moves in its fourth digit across that whole range.
    """
    J = np.zeros((4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = eps
        plus = _integrate(wiggler, +e, delta, paraxial=paraxial)
        minus = _integrate(wiggler, -e, delta, paraxial=paraxial)
        J[:, j] = (plus - minus) / (2.0 * eps)
    return J


def _shipped_block(wiggler: Wiggler, ref: ReferenceParticle) -> np.ndarray:
    """The shipped 6x6's transverse 4x4, in the ``(x, x', y, y')`` basis the ODE uses.

    At ``delta = 0`` the canonical momenta *are* the geometric angles, so no conversion is
    needed — but saying so here rather than silently indexing is the difference between a
    comparison and a coincidence.
    """
    return wiggler.matrix(ref)[np.ix_([X, PX, Y, PY], [X, PX, Y, PY])]


def _drift_block(length: float) -> np.ndarray:
    return np.array([[1.0, length], [0.0, 1.0]])


# --- Maxwell, and its blindness ---------------------------------------------------------
def test_the_shipped_field_model_satisfies_maxwell_and_is_blind_to_the_coefficient() -> None:
    r"""``div b = 0`` and ``curl b = 0``, symbolically — and the gate cannot see ``k_y``.

    A planar pole face cannot have a purely vertical field: forcing ``b_y`` to fall off as
    ``cosh(ky)`` in the gap forces the longitudinal ``b_s``, and that ``b_s`` is the entire
    source of the vertical focusing. Maxwell is therefore the *reason* for the milestone's
    physics.

    It is not a *gate* on it. The field expression contains no ``k_y`` at all — the
    focusing is derived from it, not put into it — so this test passes identically for a
    class that ships ``h0^2``, ``h0^2/2``, ``h0^2/4`` or zero. That is recorded here in the
    same breath as the gate, the way J1 recorded that a symplecticity check cannot see a
    kick coefficient.
    """
    s, y, x, h0, k = sp.symbols("s y x h0 k", real=True, positive=True)
    bx = sp.Integer(0)
    by = h0 * sp.cos(k * s) * sp.cosh(k * y)
    bs = -h0 * sp.sin(k * s) * sp.sinh(k * y)

    assert sp.simplify(sp.diff(bx, x) + sp.diff(by, y) + sp.diff(bs, s)) == 0
    # curl b = 0 too: the gap is current-free, so b is a gradient as well as divergence-free
    assert sp.simplify(sp.diff(bs, y) - sp.diff(by, s)) == 0
    assert sp.simplify(sp.diff(bx, s) - sp.diff(bs, x)) == 0
    assert sp.simplify(sp.diff(by, x) - sp.diff(bx, y)) == 0

    # ...and the blindness, stated as an assertion rather than a comment: k_y is simply not
    # a symbol of the field, so no manipulation of the field can constrain it.
    assert not (by.free_symbols | bs.free_symbols) & {sp.Symbol("k_y")}


def test_the_averaged_equation_of_motion_is_derived_not_recalled(wig: Wiggler) -> None:
    r"""``y'' = -h0^2 y sin^2(ks)``, and its period average is ``-(h0^2/2) y``.

    The chain the whole milestone rests on, done symbolically end to end: the horizontal
    equation gives the reference trajectory's angle ``x' = -(h0/k) sin(ks)``; that angle
    crossed with ``b_s`` gives the vertical equation; and averaging *that* over one period
    gives the coefficient the class ships. The horizontal average, by the same integral, is
    exactly zero — which is why the element is a drift in that plane and not a bend.
    """
    s, y, h0, k = sp.symbols("s y h0 k", real=True, positive=True)

    xp = sp.integrate(-h0 * sp.cos(k * s), (s, 0, s))  # x'(0) = 0: entering on axis
    assert sp.simplify(xp + h0 * sp.sin(k * s) / k) == 0

    bs = -h0 * sp.sin(k * s) * sp.sinh(k * y)
    ypp = sp.simplify(sp.series(-xp * bs, y, 0, 2).removeO())  # linearised in y
    assert sp.simplify(ypp + h0**2 * y * sp.sin(k * s) ** 2) == 0

    period = 2 * sp.pi / k
    averaged = sp.simplify(sp.integrate(ypp, (s, 0, period)) / period)
    assert sp.simplify(averaged + h0**2 * y / 2) == 0

    # The same average in the horizontal plane is exactly zero -- no net bend, and no weak
    # focusing to go with it.
    assert sp.simplify(sp.integrate(-h0 * sp.cos(k * s), (s, 0, period))) == 0

    # ...and that symbolic h0^2/2 is the number the element ships.
    assert wig.focusing == pytest.approx(0.5 * wig.h0**2, rel=1e-15)


# --- the discriminating gate ------------------------------------------------------------
def test_the_vertical_focusing_matches_the_integrated_field(
    wig: Wiggler, ref: ReferenceParticle
) -> None:
    """The only gate that can tell ``h0^2/2`` from ``h0^2`` or ``h0^2/4``.

    The shipped vertical block reproduces the integrated field's to ``1.26e-05``; the two
    neighbouring coefficients miss by ``9.6e-02`` and ``4.9e-02``, four orders larger. Both
    of those pass the Maxwell and symplecticity gates above and below, which is exactly why
    this test is the one that matters.
    """
    exact = _integrated_block(wig)[2:, 2:]
    shipped = _shipped_block(wig, ref)[2:, 2:]

    assert np.max(np.abs(shipped - exact)) < 2e-5

    from accsim.elements.quadrupole import _focusing_block

    for wrong in (wig.h0**2, 0.25 * wig.h0**2):
        assert np.max(np.abs(_focusing_block(wrong, wig.length) - exact)) > 1e-2


def test_the_focusing_lands_in_the_vertical_plane_and_not_the_horizontal(
    wig: Wiggler, ref: ReferenceParticle
) -> None:
    """The headline, stated as the order-unity separation it is.

    A drift misses the *vertical* block by ``9.94e-02`` — a focal length of ``1/f = 0.0994``
    over a 1 m magnet, which is what a bend-based model of a wiggler gets — while the
    *horizontal* block is a drift to ``5e-10``. Every intuition carried over from a dipole
    has these two the other way round.
    """
    exact = _integrated_block(wig)
    drift = _drift_block(wig.length)

    assert np.max(np.abs(drift - exact[2:, 2:])) > 9e-2  # vertical: NOT a drift
    assert np.max(np.abs(drift - exact[:2, :2])) < 1e-9  # horizontal: a drift

    shipped = _shipped_block(wig, ref)
    assert shipped[:2, :2] == pytest.approx(drift, abs=0.0)  # bit-identical, by construction
    assert abs(shipped[1, 0]) == 0.0  # R21: no weak focusing at all
    assert abs(exact[1, 0]) < 1e-9  # ...and the field agrees


def test_the_reference_trajectory_closes(wig: Wiggler) -> None:
    """An on-axis particle leaves on axis, travelling as it arrived — the field's own answer.

    Gated on the **integration**, deliberately. The shipped map is linear with a ``zeta``
    kick and no transverse one, so it sends ``(x, px) = 0`` to ``0`` identically: asserting
    the closure there would be a statement about the code's shape, not about the magnet.
    Integrating the real field is what can actually fail — a non-integer number of periods
    leaves the beam deflected by up to ``theta = 7.16e-03``, five orders above the *tenth*
    of a period tested here as the control.
    """
    out = _integrate(wig, np.zeros(4))
    assert abs(out[0]) < 1e-14  # x
    assert abs(out[1]) < 1e-14  # x'

    # ...and the control: a magnet cut off at a quarter period is a bend, not a wiggler.
    stub = Wiggler(wig.period / 4.0, wig.h0, 1)
    assert abs(_integrate(stub, np.zeros(4))[1]) > 0.5 * wig.deflection

    # The constructor refuses the shape that would break it silently.
    with pytest.raises(ValueError, match="positive integer"):
        Wiggler(PERIOD, 0.45, 0)
    with pytest.raises(ValueError, match="positive integer"):
        Wiggler(PERIOD, 0.45, 3.5)  # type: ignore[arg-type]
    assert Wiggler(PERIOD, 0.45, PERIODS).length == pytest.approx(1.0, rel=1e-15)


# --- the momentum dependence the roadmap's gates were all blind to ----------------------
@pytest.mark.parametrize("delta", [0.01, 0.05])
def test_the_momentum_dependence_is_the_second_power(
    wig: Wiggler, ref: ReferenceParticle, delta: float
) -> None:
    r"""``k_y = h0^2/(2(1+delta)^2)`` — **squared**, where every other magnet is linear.

    ``h0`` is normalised to the reference rigidity, so an off-momentum particle is deflected
    by ``theta/(1+delta)``; the focusing is the *square* of that deflection, so it picks up
    the square of the rigidity factor. A :class:`~accsim.elements.quadrupole.Quadrupole`
    carries ``k1/(1+delta)`` and a :class:`~accsim.elements.solenoid.Solenoid` carries
    ``ks/(1+delta)``, so the first power is the plausible thing to write here and it is
    wrong.

    Every gate the roadmap pre-committed for T1 is at ``delta = 0``, where all three powers
    agree exactly. At ``delta = 0.05`` the first power misses the integrated field by
    ``4.4e-03`` and the second by ``1.1e-05`` — the averaging remainder, unchanged. A
    factor of ~390.
    """
    from accsim.elements.quadrupole import _focusing_block

    exact = _integrated_block(wig, delta)[2:, 2:]
    L = wig.length

    err = {
        p: np.max(np.abs(_focusing_block(wig.focusing / (1.0 + delta) ** p, L) - exact))
        for p in (0, 1, 2)
    }
    assert err[2] < 2e-5  # the averaging remainder, and nothing else
    assert err[1] > 40 * err[2]
    assert err[0] > 80 * err[2]

    # ...and the shipped track() is the winning power, not just the winning block. The two
    # are in different bases and the conversion is the whole reason this is worth asserting:
    # the ODE integrates the geometric angle y', the state vector carries the canonical
    # momentum py = (1 + delta) y'. So the tracked Jacobian is the field's own block
    # conjugated by diag(1, 1+delta) -- and at delta = 0.01 that conjugation is itself a 1%
    # effect, the same size as the thing being tested.
    state = np.zeros(6)
    state[DELTA] = delta
    eps = 1e-7
    col = []
    for j in (Y, PY):
        plus, minus = state.copy(), state.copy()
        plus[j] += eps
        minus[j] -= eps
        col.append((wig.track(plus, ref) - wig.track(minus, ref))[[Y, PY]] / (2 * eps))
    D = np.diag([1.0, 1.0 + delta])
    tracked = np.linalg.inv(D) @ np.column_stack(col) @ D
    assert np.max(np.abs(tracked - exact)) < 2e-5


def test_the_matrix_is_the_tracked_jacobian_at_the_origin(
    wig: Wiggler, ref: ReferenceParticle
) -> None:
    """``matrix()`` is the origin Jacobian of ``track()``, entry for entry.

    The package-wide contract, and here it also pins that the ``(1+delta)^2`` of the tracked
    map collapses to the matrix's plain ``h0^2/2`` at ``delta = 0`` rather than to something
    that merely looks like it.
    """
    from accsim.symplectic import jacobian

    J = jacobian(lambda st: wig.track(st, ref), np.zeros(6))
    assert J == pytest.approx(wig.matrix(ref), abs=1e-9)


# --- the path lengthening, which the roadmap entry did not name -------------------------
def test_the_path_lengthening_is_a_constant_and_is_l_theta_squared_over_four(
    wig: Wiggler, ref: ReferenceParticle
) -> None:
    r"""The wiggle is a longer road: ``Delta s = L theta^2 / 4``, exactly.

    The closed form reproduces the integrated trajectory to ``3e-12`` relative — it is not
    a fit. Over an integer number of periods the cross term between the particle's own
    angle and the wiggle's ``sin`` integrates to zero, which is what makes the extra path
    *separate* into the ordinary coordinate-dependent piece and this constant, and what
    makes the constant a :meth:`~accsim.elements.element.Element.kick` rather than a term
    of the matrix.

    It makes this the only aligned, on-design element in the package with a nonzero kick —
    everything else there is a corrector or a misalignment. ``1.28e-05 m`` on the probe
    magnet: a real shift of the synchronous phase, not a rounding term.
    """
    integrated = _integrate(wig, np.zeros(4), path=True)[4]
    closed_form = 0.25 * wig.length * wig.deflection**2
    assert integrated == pytest.approx(closed_form, rel=1e-10)
    assert closed_form > 1e-5  # the bound above is not vacuous: the quantity is not zero

    # The element's own constant term is that path, with the sign zeta takes: a longer road
    # means arriving later, and zeta = s - beta0 c t.
    kick = wig.kick(ref)
    assert kick[ZETA] == pytest.approx(-closed_form, rel=1e-15)
    assert np.all(kick[[X, PX, Y, PY, DELTA]] == 0.0)

    # ...and track() delivers it: an on-axis, on-momentum particle loses exactly that much
    # zeta and nothing else moves.
    out = wig.track(np.zeros(6), ref)
    assert out[ZETA] == pytest.approx(-closed_form, rel=1e-12)
    assert np.all(out[[X, PX, Y, PY, DELTA]] == 0.0)

    # A drift of the same length does not: this is the wiggler's own, not geometry.
    assert Drift(wig.length).track(np.zeros(6), ref)[ZETA] == 0.0


@pytest.mark.parametrize("delta", [0.05, -0.05])
def test_the_wiggle_path_carries_the_second_power_of_the_rigidity_too(
    wig: Wiggler, delta: float
) -> None:
    r"""``Delta s = L theta^2 / (4 (1+delta)^2)`` — and **nothing structural can see it**.

    This gate exists because of what it is the *only* witness to. The wiggle path is a
    function of ``delta`` alone, so its transverse derivatives are all zero — and every
    symplectic condition involving ``zeta`` pairs a transverse derivative of ``zeta`` against
    the transverse map. A term with none therefore drops out of all of them:
    :func:`~accsim.symplectic.is_symplectic_map_canonical` **cannot see this exponent**, in
    contrast to the ``(k_y/2) y^2`` term beside it, which it pins exactly.

    Nor can the ``R56`` gate: that compares the closed form against ``_matrix_body`` and
    against the tracked slope, both of which evaluate the same expression. Both legs are
    circular with respect to the exponent. The integrated trajectory is not, and since it
    reproduces the ``delta = 0`` closed form to ``3e-12`` relative, the discrimination here
    is about **six orders**: the first power misses by ~5% of the quantity.

    That matters because this term is the whole source of the ``R56`` finding — get the
    exponent wrong and the wiggler's contribution to momentum compaction is wrong by 5% at
    ``delta = 0.05``, with every other gate in the file still green.
    """
    integrated = _integrate(wig, np.zeros(4), delta, path=True)[4]
    L, theta = wig.length, wig.deflection

    shipped = 0.25 * L * (theta / (1.0 + delta)) ** 2
    assert integrated == pytest.approx(shipped, rel=1e-8)

    # ...and the plausible alternatives are nowhere near, on an arbiter this sharp.
    for power in (0, 1):
        wrong = 0.25 * L * theta**2 / (1.0 + delta) ** power
        assert abs(wrong - integrated) > 1e5 * abs(shipped - integrated)


def test_the_design_kick_reaches_the_closed_orbit_and_moves_the_momentum(
    ref: ReferenceParticle,
) -> None:
    r"""End to end for the constant kick: a wiggler shifts the ring's **momentum**, not its
    synchronous phase.

    This is the only test in the suite that puts a wiggler in front of the machinery its new
    constant term actually feeds. Everything else reaches optics through
    ``one_turn_matrix``, which cannot see a ``kick()`` at all: the tune tests, the radiation
    integrals and the taper refusal all stop short of a closed-orbit solve. ``Wiggler`` is
    the package's first *aligned, on-design* element with a nonzero ``kick()``, and I1 made
    the element map affine precisely so a solver could consume one.

    **What it does, measured rather than assumed.** With the RF frequency fixed to the
    design circumference, a wiggler makes the closed orbit longer, and the ring answers by
    running **off momentum** — ``delta_co`` moves to roughly ``-k_zeta/R56``, and dispersion
    carries that into a transverse orbit distortion. ``zeta`` does **not** move: the cavity's
    zero-crossing is still where a non-radiating ring's synchronous particle sits. An earlier
    draft of the class docstring called this "a real shift of the synchronous phase"; it is
    not, and this test is what corrected it.

    Without RF the 6D solve **raises**, cleanly and with its reason — nothing reads ``zeta``,
    so a constant ``zeta`` kick has no restoring force to balance it. That is N5's guard
    doing its job on a case it was not written for, and it is asserted here rather than
    left to be discovered.
    """
    from accsim import Dipole, RFCavity, closed_orbit, closed_orbit_6d
    from accsim.orbit import ClosedOrbitError

    n_cell = 8
    angle = np.pi / n_cell  # two bends per cell, 2 pi over the ring

    def ring(insert: object, cavity: object | None = None) -> Lattice:
        els: list[object] = []
        for _ in range(n_cell):
            els += [
                Quadrupole(0.3, 1.5, "qf"),
                Dipole(1.0, angle),
                Quadrupole(0.3, -1.5, "qd"),
                Dipole(1.0, angle),
                insert() if callable(insert) else insert,
            ]
        if cavity is not None:
            els.append(cavity)
        return Lattice(els, ref)  # type: ignore[arg-type]

    def a_wiggler() -> Wiggler:
        return Wiggler(PERIOD, 0.449689, PERIODS)

    circumference = sum(e.length for e in ring(a_wiggler).elements)
    cavity = RFCavity(2.0e6, 200 * ref.beta0 * 299792458.0 / circumference, name="rf")

    # Without RF: the guard fires, and says why.
    with pytest.raises(ClosedOrbitError, match="no RF cavity"):
        closed_orbit_6d(ring(a_wiggler))
    # ...while the purely transverse solve is untroubled: the kick is longitudinal only.
    assert np.array_equal(closed_orbit(ring(a_wiggler)), np.zeros(4))

    # With RF: it converges, and the design ring it is compared against is exactly zero.
    bare = np.asarray(closed_orbit_6d(ring(lambda: Drift(a_wiggler().length), cavity)))
    assert np.array_equal(bare, np.zeros(6))

    orbit = np.asarray(closed_orbit_6d(ring(a_wiggler, cavity)))
    lattice = ring(a_wiggler, cavity)
    R56 = lattice.transfer_matrix()[ZETA, DELTA]
    k_zeta = lattice.transfer_map()[1][ZETA]

    # The ring's whole constant term is n_cell wigglers' worth, and nothing else's.
    assert k_zeta == pytest.approx(n_cell * a_wiggler().kick(ref)[ZETA], rel=1e-12)

    # zeta stays at the cavity's zero-crossing; delta is what moves.
    assert abs(orbit[ZETA]) < 1e-15
    assert orbit[DELTA] == pytest.approx(-k_zeta / R56, rel=0.1)
    assert abs(orbit[DELTA]) > 1e-6  # not a rounding term

    # ...and dispersion turns that momentum shift into a real transverse orbit.
    assert abs(orbit[X]) > 1e-5


def test_the_averaging_remainder_and_the_path_lengthening_are_the_same_quantity(
    wig: Wiggler, ref: ReferenceParticle
) -> None:
    r"""``|M_y(shipped) - M_y(field)| -> L theta^2/4`` — the residual is physics, not noise.

    The gate above bounds the vertical residual at ``2e-05``; this says *what it is*, which
    is the sharper statement and the one that distinguishes an averaging remainder from a
    bug. Halving the field quarters the residual, and the ratio to ``L theta^2/4`` tends to
    1 as the wiggle weakens (measured ``0.9992``, ``0.9968``, ``0.9874`` at a quarter, a
    half and the full probe field). A coefficient error would not scale this way, and
    numerical noise would not scale at all.
    """
    ratios = []
    for scale in (0.25, 0.5, 1.0):
        w = Wiggler(wig.period, scale * wig.h0, wig.periods)
        residual = np.max(np.abs(_shipped_block(w, ref)[2:, 2:] - _integrated_block(w)[2:, 2:]))
        ratios.append(residual / (0.25 * w.length * w.deflection**2))

    assert ratios[0] == pytest.approx(1.0, rel=2e-3)  # the weakest wiggle: closest to 1
    assert ratios[0] > ratios[1] > ratios[2]  # ...and it approaches 1 from above
    assert ratios[2] == pytest.approx(1.0, rel=2e-2)


def test_the_non_paraxial_remainder_is_three_quarters_theta_squared_in_r12(
    wig: Wiggler,
) -> None:
    r"""What the shipped map is *not* exact in: ``R12 = L (1 + 3 theta^2/4)``.

    Restoring the ``sqrt(1 + x'^2 + y'^2)`` factors the paraxial equations drop moves the
    horizontal ``R12`` off ``L`` by ``3 theta^2 L / 4`` — the wiggle orbit's own ``1/p_s``
    averaged along the sine — while ``R21`` stays zero and the plane stays unfocused. The
    coefficient is measured at ``0.750001`` for a quarter-strength magnet and holds to
    ``0.7501`` at twice the probe field, so it is identified by its **scaling**, the way L2
    identified the quadrupole's own kinematic remainder rather than papering over it.

    This is P2 (iv)'s subject, not T1's, and it is deliberately not shipped: building it in
    would make the horizontal block something other than a drift, and there is no reference
    code that could arbitrate the choice.
    """
    for scale in (0.5, 1.0, 2.0):
        w = Wiggler(wig.period, scale * wig.h0, wig.periods)
        r12 = _integrated_block(w, paraxial=False)[0, 1]
        assert (r12 - w.length) / (w.length * w.deflection**2) == pytest.approx(0.75, rel=1e-3)
        # ...and the plane is still unfocused, non-paraxially too.
        assert abs(_integrated_block(w, paraxial=False)[1, 0]) < 1e-8

    # The paraxial integration -- the arbiter every other gate uses -- has none of it.
    assert _integrated_block(wig)[0, 1] - wig.length == pytest.approx(0.0, abs=1e-9)


# --- the structural gates, and their blindness ------------------------------------------
def test_symplecticity_holds_and_is_blind_to_the_coefficient(
    wig: Wiggler, ref: ReferenceParticle
) -> None:
    """Symplectic to machine precision — and so is every wrong coefficient.

    ``is_symplectic_map_canonical`` rather than the plain version, for the reason it rejects
    :class:`~accsim.elements.quadrupole.Quadrupole` and
    :class:`~accsim.elements.solenoid.Solenoid` too: ``(zeta, delta)`` is not a canonical
    pair.

    The blindness is the point of the second half. *Any* focusing block is symplectic, so a
    wiggler shipping ``h0^2`` or ``h0^2/4`` — both four orders away from the field's own
    answer — passes this identically. So does one with no path lengthening at all, since a
    constant translation has unit Jacobian.
    """
    amp = np.array([1e-4, 1e-5, 1e-4, 1e-5, 1e-3, 1e-4])
    assert is_symplectic_map_canonical(lambda st: wig.track(st, ref), amp, ref)

    for wrong_h0 in (wig.h0 * np.sqrt(2.0), wig.h0 / np.sqrt(2.0)):
        wrong = Wiggler(wig.period, wrong_h0, wig.periods)
        assert is_symplectic_map_canonical(lambda st, w=wrong: w.track(st, ref), amp, ref)


def test_the_map_is_even_in_the_field_and_free_of_the_charge(ref: ReferenceParticle) -> None:
    """Flipping the polarity gives a **bit-identical** map, and so does changing species.

    ``k_y = h0^2/2`` is even in ``h0``, so a wiggler does not care which way its first pole
    points — physically obvious and a genuine constraint on the code, since a map built from
    ``h0`` rather than ``h0^2`` anywhere would break it. The same evenness is what makes the
    element charge-free: an electron and a proton in the same physical magnet get opposite
    normalised ``h0`` from :meth:`~accsim.elements.wiggler.Wiggler.from_peak_field`, and the
    same map out of it. S1's charge-free gate for ``ks``, in the form this element takes it.
    """
    plus = Wiggler(PERIOD, 0.449689, PERIODS)
    minus = Wiggler(PERIOD, -0.449689, PERIODS)
    assert np.array_equal(plus.matrix(ref), minus.matrix(ref))
    assert np.array_equal(plus.kick(ref), minus.kick(ref))

    state = np.array([1e-4, 2e-5, -3e-4, 1e-5, 1e-3, 1e-4])
    assert np.array_equal(plus.track(state, ref), minus.track(state, ref))

    proton = ReferenceParticle.from_total_energy(938.27208816e6, 1.0e9, charge=1.0)
    electron_h0 = Wiggler.from_peak_field(PERIOD, PEAK_FIELD_T, PERIODS, ref).h0
    proton_h0 = Wiggler.from_peak_field(PERIOD, PEAK_FIELD_T, PERIODS, proton).h0
    assert electron_h0 * proton_h0 < 0.0  # opposite sign, from the charge
    assert Wiggler(PERIOD, electron_h0, PERIODS).focusing == pytest.approx(
        Wiggler(PERIOD, -electron_h0, PERIODS).focusing, rel=0.0
    )


def test_r56_is_not_a_drift_s_and_the_wiggle_dominates_it(
    wig: Wiggler, ref: ReferenceParticle
) -> None:
    r"""``R56 = L/gamma0^2 + (L theta^2/4)(2 + 1/gamma0^2)`` — a *geometric* momentum term.

    A straight element's ``R56`` is supposed to be pure kinematics: ``L/gamma0^2``, the
    slower particle falling behind. A wiggler breaks that. Its deflection is
    ``theta/(1+delta)``, so a stiffer particle wiggles *less* and travels a shorter road —
    a path-length dependence on momentum with no dispersion and no bend anywhere in sight.

    **The ratio of the two terms is the wiggler parameter, squared and halved.** Dividing
    them out, ``wiggle/drift = (theta^2/4) gamma0^2 (2 + 1/gamma0^2) = K^2/2 + theta^2/4``
    where ``K = gamma0 theta`` is the conventional wiggler parameter — the quantity the
    roadmap entry deliberately set aside in favour of ``theta``, arriving here on its own.
    Since ``K`` depends on the field and the period but **not on the energy**, that ratio is
    the *same at every energy* for a given magnet: ``98.08`` here, and still ``98.08`` at
    200 MeV and at 5 GeV, which this test asserts rather than describes. It is what corrects
    the first version of this docstring, which said the term "grows with energy".

    The term was found by the package's contract that ``matrix()`` be the origin Jacobian of
    ``track()`` — the first draft of the element shipped a drift's ``R56`` and that test is
    what refused it. The *exponent* inside it is gated separately and elsewhere, in
    :func:`test_the_wiggle_path_carries_the_second_power_of_the_rigidity_too`, because
    nothing here can see it: this test compares the closed form against ``_matrix_body`` and
    against the tracked slope, and all three are the same expression.
    """
    M = wig.matrix(ref)
    drift_term = wig.length / ref.gamma0**2
    wiggle_term = 0.25 * wig.length * wig.deflection**2 * (2.0 + 1.0 / ref.gamma0**2)

    assert M[ZETA, DELTA] == pytest.approx(drift_term + wiggle_term, rel=1e-15)

    # The ratio is K^2/2 + theta^2/4, and K is an energy-free property of the magnet -- so
    # the same physical wiggler gives the same ratio across a 25x range of beam energy.
    for energy_eV in (2.0e8, 1.0e9, 5.0e9):
        beam = ReferenceParticle.from_total_energy(0.51099895069e6, energy_eV, charge=-1.0)
        w = Wiggler.from_peak_field(PERIOD, PEAK_FIELD_T, PERIODS, beam)
        K = beam.gamma0 * w.deflection
        drift = w.length / beam.gamma0**2
        assert (w.matrix(beam)[ZETA, DELTA] - drift) / drift == pytest.approx(
            0.5 * K * K + 0.25 * w.deflection**2, rel=1e-12
        )
        assert abs(K) == pytest.approx(14.006, rel=1e-4)  # ...and K itself barely moves
    assert wiggle_term > 90 * drift_term  # the wiggle dominates, and by two orders

    # ...and it is the tracked map's own derivative, not an independent formula.
    eps = 1e-6
    plus, minus = np.zeros(6), np.zeros(6)
    plus[DELTA], minus[DELTA] = eps, -eps
    slope = (wig.track(plus, ref)[ZETA] - wig.track(minus, ref)[ZETA]) / (2 * eps)
    assert slope == pytest.approx(M[ZETA, DELTA], rel=1e-8)

    # What *is* a straight element's: no dispersion, and no zeta from a transverse offset.
    assert M[ZETA, X] == 0.0 and M[ZETA, PX] == 0.0
    assert M[X, DELTA] == 0.0 and M[PX, DELTA] == 0.0


def test_a_wiggler_focuses_one_plane_without_defocusing_the_other(
    ref: ReferenceParticle,
) -> None:
    """End to end: a wiggler moves the **vertical** tune and leaves the horizontal one
    bit-for-bit alone — which no quadrupole in the package can do.

    The milestone's consequence, at the level someone designing a machine would meet it.
    This test was first written asserting that a wiggler *is* ``Quadrupole(L, -h0^2/2)``,
    and half of that is true: the vertical tunes agree to ``1e-12``. The horizontal ones do
    not, and cannot — a quadrupole's gradient focuses one plane by exactly as much as it
    defocuses the other, so the "equivalent" quadrupole moves ``Qx`` by ``0.033`` while the
    wiggler moves it by nothing at all.

    That is not a violation of the theorem behind a quadrupole's antisymmetry. A wiggler's
    focusing is not a transverse gradient: it is the period average of the *square* of a
    field that integrates to zero, so it enters the equations of motion where a gradient
    cannot and is free to act in one plane only.
    """
    from accsim import tunes

    def ring(insert: object) -> Lattice:
        return Lattice(
            [
                Quadrupole(0.5, 1.2, "qf"),
                Drift(1.0),
                Quadrupole(0.5, -1.2, "qd"),
                Drift(0.5),
                insert,  # type: ignore[list-item]
                Drift(0.5),
            ],
            ref,
        )

    wig = Wiggler(PERIOD, 0.449689, PERIODS)
    bare_qx, bare_qy = tunes(ring(Drift(wig.length)))
    wig_qx, wig_qy = tunes(ring(wig))
    eq_qx, eq_qy = tunes(ring(Quadrupole(wig.length, -wig.focusing)))

    assert wig_qy != pytest.approx(bare_qy, abs=1e-6)  # the vertical tune moves...
    assert wig_qx == pytest.approx(bare_qx, abs=1e-14)  # ...and the horizontal does not
    assert wig_qy == pytest.approx(eq_qy, abs=1e-12)  # vertically it IS that quadrupole
    assert abs(eq_qx - bare_qx) > 0.03  # ...and horizontally it is not, by a wide margin


# --- the refusals, each written to fail the day T2 lands --------------------------------
def test_the_field_accessor_refuses_rather_than_reporting_no_field(wig: Wiggler) -> None:
    """The accessor has no ``s``, so it raises — it does not quietly answer ``(0, 0)``.

    ``b_y`` reverses sign twenty times inside the probe magnet and averages to exactly zero
    over a period, while ``radiation_kick`` samples the field **once**, at the mid-point.
    The inherited zero would therefore report no radiation from the magnet whose only
    purpose is to radiate — and radiation goes as the field *squared*, so no sample point
    repairs it.

    This is louder than the roadmap's T1 asked for (it specified "invisible to
    ``radiation_kick``") and it is deliberate: a silent zero is the exact hazard the axis-T
    entry identifies. T2 lifts it by giving the period-averaged ``<h^2>`` and ``<|h|^3>`` a
    place to go, at which point this test is expected to fail and be rewritten.
    """
    with pytest.raises(NotImplementedError, match="no s-independent field"):
        wig.normalized_field(0.0, 0.0)

    # The two siblings are honestly zero: an ideal planar wiggler has neither a net
    # longitudinal field on axis nor a transverse vector potential the state vector hides.
    assert wig.longitudinal_field(0.0, 0.0) == 0.0
    assert wig.normalized_vector_potential(0.0, 0.0) == (0.0, 0.0)


def test_radiation_tracking_through_a_wiggler_refuses(wig: Wiggler, ref: ReferenceParticle) -> None:
    """A wiggler tracks with ``radiation="off"`` and refuses every other model.

    The refusal reaches the caller through the field accessor, which is the right place for
    it: the tracking gap is not that the physics is unknown but that the interface cannot
    express an ``s``-dependent field. T2 gate 6 pre-commits this raise; it arrives here
    because shipping the silent alternative for one milestone was not worth it.
    """
    state = np.array([1e-4, 0.0, 1e-4, 0.0, 0.0, 0.0])
    wig.track(state, ref)  # radiation="off": fine

    for model in ("mean", "quantum", "photons"):
        with pytest.raises(NotImplementedError, match="magnet built to radiate"):
            wig.track(state, ref, radiation=model, rng=np.random.default_rng(0))


def test_the_radiation_integrals_now_see_the_wiggler_and_t1_did_not(
    ref: ReferenceParticle,
) -> None:
    """T1's refusal, **lifted** by T2 — and kept here as the record of what it cost.

    This test used to assert an exact zero. ``radiation_integrals`` keyed on
    ``isinstance(elem, Dipole)``, so a wiggler was not merely approximated there, it was
    *absent*: a ring with a 1 m wiggler in it returned integrals bit-identical to the same
    ring with a 1 m drift, when the true ``i2 = h0^2 L/2 = 0.101`` is more than twice the
    dipole's. T1 shipped that as a loud refusal (the field accessor raises) rather than a
    silent zero, and said this test was written to fail the day T2 landed.

    T2 landed. What is asserted now is the opposite — that the zero is gone, and that what
    replaced it is the closed form — with the full gate list in
    ``tests/analytic/test_wiggler_radiation.py``. This one stays because the *size* of what
    was being silently dropped is the reason the refusal was worth making loud.
    """
    from accsim import Dipole

    def ring(insert: object) -> Lattice:
        """A stable FODO arc cell with a bend in it — ``radiation_integrals`` needs both."""
        return Lattice(
            [
                Quadrupole(0.5, 1.2, "qf"),
                Dipole(2.0, 0.3),
                Quadrupole(0.5, -1.2, "qd"),
                Drift(0.5),
                insert,  # type: ignore[list-item]
                Drift(0.5),
            ],
            ref,
        )

    wig = Wiggler(PERIOD, 0.449689, PERIODS)
    with_wig = radiation_integrals(ring(wig))
    without = radiation_integrals(ring(Drift(wig.length)))

    assert with_wig.i2 - without.i2 == pytest.approx(0.5 * wig.h0**2 * wig.length, rel=1e-15)
    assert with_wig.i3 > without.i3
    assert with_wig.i5 > without.i5
    # ...and what T1 was refusing was never small: the wiggler's own i2 is a factor of two
    # more than the dipole's, so the silent zero would have been a 3x error in energy loss.
    assert 0.5 * wig.h0**2 * wig.length > 2.0 * without.i2


def test_tapering_a_ring_with_a_wiggler_is_refused_for_one_remaining_reason(
    ref: ReferenceParticle,
) -> None:
    """``taper()`` still refuses a wiggler ring — but for **one** reason now, not two.

    A taper needs the ring's *radiating* closed orbit, and it also needs to know how to
    scale every powered magnet. T1 failed both, and **which one fired first was measured
    rather than predicted**: this test was first written expecting ``taper``'s own
    ``_STRENGTHS`` guard, and the field accessor got there first, because the radiating
    orbit is built before any magnet is scaled.

    T2 answered the second question — a wiggler *is* a powered magnet and ``h0`` is its
    strength, so ``_scaled`` no longer raises — which leaves only the first. The remaining
    refusal is the honest one and it is the tracking gap: ``radiation_kick`` samples the
    field once per traversal and a wiggler's reverses twenty times inside itself.

    Written to fail the day per-period tracking through the real field lands.
    """
    from accsim import Dipole
    from accsim.tapering import _scaled

    lattice = Lattice([Dipole(2.0, 0.3), Wiggler(PERIOD, 0.449689, PERIODS, "w")], ref)
    with pytest.raises(NotImplementedError, match="no s-independent field"):
        taper(lattice)

    # the half that lifted, asserted as lifted (T2 gate 7)
    scaled = _scaled(Wiggler(PERIOD, 0.449689, PERIODS, "w"), 1.001)
    assert scaled.h0 == pytest.approx(1.001 * 0.449689, rel=1e-15)
    assert scaled.focusing == pytest.approx(1.001**2 * 0.5 * 0.449689**2, rel=1e-14)


def test_nothing_else_moves(ref: ReferenceParticle) -> None:
    """The element is purely additive: at zero field it *is* a drift, in every entry.

    Not a claim about the rest of the suite (the suite itself is that claim) but the
    strongest local form of it — the new code path collapses onto the oldest one in the
    package when its own strength is switched off, matrix, kick and track alike. Note the
    track comparison is against a **paraxial** drift, not
    :class:`~accsim.elements.drift.Drift`'s exact map, for the reason
    :class:`~accsim.elements.quadrupole.Quadrupole` records: this element is paraxial in
    the angles and a zero-strength one does not become exact.
    """
    off = Wiggler(PERIOD, 0.0, PERIODS)
    drift = Drift(off.length)

    assert np.array_equal(off.matrix(ref), drift.matrix(ref))
    assert np.array_equal(off.kick(ref), np.zeros(6))
    assert off.focusing == 0.0 and off.deflection == 0.0

    state = np.array([1e-4, 3e-5, -2e-4, 1e-5, 1e-3, 0.0])
    assert off.track(state, ref) == pytest.approx(
        Quadrupole(off.length, 0.0).track(state, ref), abs=0.0
    )

    assert "Wiggler(period=0.1, h0=0.0, periods=10)" == repr(off)
