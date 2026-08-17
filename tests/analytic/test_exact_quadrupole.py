r"""L2 — the quadrupole's momentum-dependent map, and the chromaticity it closes.

``k1`` is normalised to the **reference** rigidity, so a particle of momentum
``(1 + delta)`` is stiffer and is focused by ``k1/(1 + delta)``. accsim's thick
quadrupole did not know that: its ``track()`` was its ``matrix()`` at *every*
momentum, which is the definition of a chromatically ideal magnet and is not what a
magnet is. L1 gave the drift its exact map and left tracking seeing 45% of the
natural chromaticity — the drifts' share; this milestone gives the quadrupole its
own and closes the rest.

**The gate shape the ROADMAP prescribed does not transfer, and this file uses a
different one.** L1's discriminating axis was *large angles*, because the exact
drift and the expanded one differ at ``O(angle^3)``. Large angles are exactly where
this map is deliberately wrong: it is the flow of the **paraxial** Hamiltonian, and
no closed form exists for the exact one. The axis that discriminates here is **large
delta**, and there is an exact identity along it rather than a tolerance —

    tracked tunes at momentum ``delta``
        == design tunes of the same ring with every ``k1 -> k1/(1 + delta)``

which holds to ``1e-15`` at ``delta`` up to ``0.05``. Substituting ``px = (1+delta) p``
turns the exact drift's ``L/(1+delta)`` back into ``L`` and the quadrupole's block
into a design quadrupole of the rescaled strength, so the ring at momentum ``delta``
*is* the design ring reweighted. That pins all orders in ``delta`` at once: a wrong
power (``1/(1+delta)^2``), or the factor applied in one plane only, fails at
``O(delta)`` and no tolerance absorbs it.

Everything the map is exact in, and everything it is not, is recorded on
:class:`~accsim.elements.quadrupole.Quadrupole`. The reference cross-check against
``xt.Quadrupole`` lives in ``tests/reference/test_quadrupole_xtrack.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    SkewQuadrupole,
    ThinQuadrupole,
    is_symplectic_map,
    is_symplectic_map_canonical,
    jacobian,
    natural_chromaticity,
    tunes,
    tunes_on_orbit,
)
from accsim.coords import DELTA, DIM, PX, PY, X, Y
from accsim.elements.quadrupole import (
    _focusing_block,
    _focusing_functions,
    _path_lengthening,
)
from accsim.symplectic import J6

L_Q = 0.7
K1 = 1.2

# Deliberately generic: every coordinate nonzero, so no term can hide behind a zero.
STATE = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 1.0e-3, 2.0e-3])


@pytest.fixture
def ring_ref() -> ReferenceParticle:
    """gamma0 = 20 — relativistic enough to be a real machine, not so much that the
    ``1/gamma0^2`` longitudinal term vanishes into the round-off."""
    from accsim import PROTON_MASS_EV

    return ReferenceParticle.from_gamma(PROTON_MASS_EV, 20.0)


def _straight_ring(ref: ReferenceParticle, cells: int = 6, k1: float = K1) -> Lattice:
    """A FODO ring of **thick** quadrupoles and drifts, and no bend anywhere.

    Bend-free on purpose. The :class:`~accsim.elements.dipole.Dipole` map is still
    linear (L3), and a dipole is the one element that would put an untracked term
    into the chromaticity comparison below. With ``h = 0`` every element in the ring
    has its exact map, so the identity is exact rather than approximate.
    """
    els: list = []
    for _ in range(cells):
        els += [Quadrupole(0.3, k1), Drift(0.5), Quadrupole(0.3, -k1), Drift(0.5)]
    return Lattice(els, ref)


def _tracked_chromaticity(lattice: Lattice, h: float = 1.0e-5) -> tuple[float, float]:
    """``dQ/ddelta`` by tracking alone: Newton for the orbit, then the tune of the
    finite-difference Jacobian there. Shares no code with any Twiss integral."""
    qx_p, qy_p = tunes_on_orbit(lattice, delta=+h)
    qx_m, qy_m = tunes_on_orbit(lattice, delta=-h)
    return (qx_p - qx_m) / (2.0 * h), (qy_p - qy_m) / (2.0 * h)


# --------------------------------------------------------------------------
# 1. The map, derived rather than recalled
# --------------------------------------------------------------------------


def test_the_transverse_map_solves_the_off_momentum_equation_of_motion() -> None:
    r"""``x(s) = x0 C + (px0/(1+delta)) S`` solves ``x'' + [k1/(1+delta)] x = 0``.

    Derived in sympy from the equation of motion rather than transcribed from a
    reference implementation, per the project's rule about coefficients. The content
    is the **placement of** ``(1 + delta)``: it multiplies ``k1`` inside the
    trigonometric argument *and* divides ``px`` where the angle is formed, and those
    are two separate appearances of the same physical fact (canonical ``px`` is not
    an angle). Getting one and not the other gives a map that is neither symplectic
    nor chromatically right, and no on-axis test would notice.
    """
    s, k1, d, x0, px0 = sp.symbols("s k1 delta x0 px0", real=True)
    K = k1 / (1 + d)
    w = sp.sqrt(K)
    x = x0 * sp.cos(w * s) + (px0 / (1 + d)) * sp.sin(w * s) / w

    assert sp.simplify(sp.diff(x, s, 2) + K * x) == 0
    assert sp.simplify(x.subs(s, 0) - x0) == 0
    assert sp.simplify(sp.diff(x, s).subs(s, 0) - px0 / (1 + d)) == 0

    # The canonical momentum is the angle times (1 + delta), which is what makes
    # px -> (-K x S + x' C)(1 + delta) rather than the bare bracket.
    A, B = -K * x0, px0 / (1 + d)
    assert sp.simplify(sp.diff(x, s) - (A * sp.sin(w * s) / w + B * sp.cos(w * s))) == 0


def test_the_path_lengthening_integral_is_exact_and_has_no_pole_at_zero_gradient() -> None:
    r"""``I = int (u'^2/2) ds`` in closed form, and the ``1/K`` that cancels.

    The longitudinal half of the map is the extra distance an angled particle
    travels. Written the textbook way — and the way MAD-X and xtrack's
    ``track_thick_cfd`` write it — every term carries a ``1/K``, which needs a
    ``K == 0`` branch and makes a weak quadrupole numerically awkward. Substituting
    ``A = -K u0`` and ``B = u0'`` back in cancels all of them: the result is
    **entire** in ``K``, so the drift limit is continuous and there is no branch to
    get wrong.

    Two independent statements: the closed form equals the integral (the algebra), and
    its **limit** at ``K = 0`` exists and is the drift's own ``u0'^2 L / 2``. The
    integral is taken at ``K > 0``, where sympy does not split into cases; the
    expression is entire, so the defocusing branch follows by continuation and the
    ``K = 0`` seam is the limit rather than a third case.
    """
    s, L, u0, up0 = sp.symbols("s L u0 up0", real=True)
    K = sp.Symbol("K", positive=True)
    w = sp.sqrt(K)
    C, S = sp.cos(w * s), sp.sin(w * s) / w
    angle = -K * u0 * S + up0 * C  # u'(s)

    integral = sp.integrate(angle**2 / 2, (s, 0, L))
    CL, SL = sp.cos(w * L), sp.sin(w * L) / w
    T = (L - CL * SL) / 2
    closed = (K * u0**2 * T - K * u0 * up0 * SL**2 + up0**2 * (L - T)) / 2
    assert sp.simplify(sp.expand_trig(sp.simplify(integral - closed))) == 0

    assert sp.simplify(sp.limit(closed, K, 0) - up0**2 * L / 2) == 0

    # ...and the implementation evaluates that same closed form.
    got = _path_lengthening(
        np.float64(0.0), np.float64(3e-3), np.float64(2e-4), 0.7, *_focusing_functions(0.0, 0.7)
    )
    assert float(got) == pytest.approx(0.5 * 2e-4**2 * 0.7, rel=1e-14)


def test_the_longitudinal_slip_is_rationalised_not_differenced() -> None:
    r"""``L(1 - 1/rvv)`` without subtracting two numbers of size ``L``.

    L1's parting warning, and it applies here unchanged: MAD-X and xtrack evaluate
    ``dzeta = L - path/rvv``, differencing two quantities of size ``L`` to get a
    small one. That is invisible in the value and **not** in its derivative, which is
    what :func:`~accsim.orbit.linearised_element_maps` takes. Splitting the momentum
    term off and rationalising it through ``(1+delta) + E/E0`` — using
    ``(E/E0)^2 = 1 + beta0^2 delta (2+delta)``, the drift's own identity — removes
    the cancellation exactly. Proved here symbolically, so it is algebra and not an
    approximation.
    """
    L, d, m, P0 = sp.symbols("L delta m P0", positive=True)
    E0, E = sp.sqrt(P0**2 + m**2), sp.sqrt(P0**2 * (1 + d) ** 2 + m**2)
    rvv = E0 * (1 + d) / E  # beta / beta0
    gamma0_sq = E0**2 / m**2

    naive = L * (1 - 1 / rvv)
    used = L * d * (2 + d) / gamma0_sq / ((1 + d) * ((1 + d) + E / E0))
    assert sp.simplify(naive - used) == 0


def test_the_vectorised_focusing_functions_are_the_scalar_blocks_entries() -> None:
    """``_focusing_block(g, L) == [[C, S], [-g S, C]]`` — one analytic family, two forms.

    ``matrix()`` keeps the scalar version (it is only ever evaluated at ``delta = 0``);
    the map needs a per-particle one, because a bunch with a momentum spread has a
    different ``g = k1/(1+delta)`` for every particle. Two implementations of one
    family is a place for them to drift apart, so they are pinned against each other
    across all three branches — focusing, defocusing and the ``g = 0`` seam.
    """
    for g in (1.2, -1.2, 0.0, 1.0e-14, 25.0):
        C, S = _focusing_functions(g, L_Q)
        want = _focusing_block(g, L_Q)
        np.testing.assert_allclose(np.array([[C, S], [-g * S, C]]), want, atol=1e-15, rtol=1e-14)

    # Vectorised: n different gradients in one call, matching n scalar calls.
    gs = np.array([1.2, -1.2, 0.0, 0.3])
    C, S = _focusing_functions(gs, L_Q)
    for j, g in enumerate(gs):
        c1, s1 = _focusing_functions(float(g), L_Q)
        assert C[j] == c1 and S[j] == s1


# --------------------------------------------------------------------------
# 2. What is unchanged — the bound on the whole milestone
# --------------------------------------------------------------------------


def test_at_zero_momentum_the_transverse_map_is_the_linear_matrix_to_the_last_bit(
    ring_ref: ReferenceParticle,
) -> None:
    r"""``delta = 0`` ⇒ ``K = k1`` and ``x' = px``: the map *is* the matrix, analytically.

    This is what bounds L2, and it is a stronger statement than L1 could make. The
    drift's exact map differs from its matrix at any nonzero *angle*; this one differs
    only at nonzero *momentum*. So an on-momentum particle of any amplitude tracks
    through a quadrupole exactly as it did before this milestone, in all four
    transverse coordinates.

    The difference is asserted at **one unit in the last place**, not at zero, and the
    distinction is worth keeping honest: the two arithmetics are not the same
    expression. The map forms ``(-K x S + x' C)(1 + delta)``, the matrix forms
    ``-w sin(wL) x + cos(wL) px``, and at ``delta = 0`` those are the same number
    differently associated — so ``px`` can land one bit apart while ``x``, ``y`` and
    ``py`` are exactly equal. Claiming bit equality here would be claiming something
    about floating-point association rather than about the physics.

    ``zeta`` is the one coordinate that genuinely moves, and it must: a quadrupole now
    lengthens an off-axis particle's path, exactly as a drift has since L1. It is
    checked here with its sign, because "the change is confined to zeta" is only
    worth asserting alongside "and zeta really did change".
    """
    q = Quadrupole(L_Q, K1)
    for amp in (1.0, -0.5, 3.0):
        st = amp * STATE.copy()
        st[DELTA] = 0.0
        exact, linear = q.track(st, ring_ref), q.matrix(ring_ref) @ st
        transverse = [X, PX, Y, PY]
        assert np.all(
            np.abs(exact[transverse] - linear[transverse]) <= np.spacing(np.abs(linear[transverse]))
        )
        # zeta: the path lengthening, which is always a *delay* (zeta decreases).
        assert exact[DELTA] == linear[DELTA]
        assert exact[4] < linear[4]


def test_the_linear_matrix_is_the_exact_maps_jacobian_at_the_origin(
    ring_ref: ReferenceParticle,
) -> None:
    r"""The invariant every design-optics gate in the package rests on.

    ``matrix()`` must be the Jacobian of ``track()`` at the reference particle, or the
    design optics and the tracked machine describe different rings. It is why this
    milestone takes the *closed-form* paraxial map and not a symplectic splitting: a
    drift-kick-drift quadrupole is symplectic and more accurate in the angles, but its
    Jacobian at the origin is the sliced approximation to the cos/sin block, not the
    block, and every "tracking agrees with the matrix on the design orbit" gate would
    have moved.

    The residual is the central difference's own ``O(step^2)`` truncation, sitting in
    ``(zeta, delta)`` where the map is genuinely curved in momentum. It is asserted as
    that — quartering with the step — rather than bounded by a number that could be
    hiding a real term.
    """
    for elem in (Quadrupole(L_Q, K1), Quadrupole(L_Q, -K1), Quadrupole(L_Q, 0.0)):
        M = elem.matrix(ring_ref)
        for step, bound in ((1.0e-6, 1.0e-13), (1.0e-5, 1.0e-11)):
            D = jacobian(lambda s, e=elem: e.track(s, ring_ref), np.zeros(DIM), step=step) - M
            assert np.max(np.abs(D)) < bound

    q = Quadrupole(L_Q, K1)

    def floor(step: float) -> float:
        J = jacobian(lambda s: q.track(s, ring_ref), np.zeros(DIM), step=step)
        return float(np.max(np.abs(J - q.matrix(ring_ref))))

    # It is the differencing, not the map: ten times the step, a hundred times the
    # residual — and it lives in the (zeta, delta) entry, nowhere else.
    assert floor(1.0e-5) / floor(1.0e-6) == pytest.approx(100.0, rel=0.05)
    J = jacobian(lambda s: q.track(s, ring_ref), np.zeros(DIM), step=1e-6)
    D = np.abs(J - q.matrix(ring_ref))
    assert np.unravel_index(int(np.argmax(D)), D.shape) == (4, DELTA)


def test_a_thin_quadrupole_was_already_exact_and_is_untouched(
    ring_ref: ReferenceParticle,
) -> None:
    r"""The element L2 deliberately does **not** change, and why that is not an omission.

    A thin kick is ``Delta px = -(q/P0)(dB_y/dx) x L = -k1l x`` — the change in
    *canonical momentum* is the same for every particle, with no ``1/(1+delta)``
    anywhere. The chromatic effect appears when that momentum is turned into an
    angle, which happens in the drifts either side. So a thin quadrupole is already
    exact in ``delta``, and its ``track`` is its ``matrix`` for the right reason
    rather than by omission. Asserted at bit equality across a range of momenta.
    """
    tq = ThinQuadrupole(0.4)
    M = tq.matrix(ring_ref)
    for delta in (0.0, 1.0e-3, 5.0e-2, -3.0e-2):
        st = STATE.copy()
        st[DELTA] = delta
        np.testing.assert_array_equal(tq.track(st, ring_ref), M @ st)


# --------------------------------------------------------------------------
# 3. Symplecticity — and the check that must not be used
# --------------------------------------------------------------------------


def test_the_map_is_symplectic_in_the_canonical_variables_and_rejected_in_the_others(
    ring_ref: ReferenceParticle,
) -> None:
    r"""L1's trap, one element later: the correct map fails ``is_symplectic_map``.

    The map is the *exact* flow of an *approximate* Hamiltonian, so it is exactly
    symplectic — which is the property worth having, since a truncated-but-symplectic
    map is safe to iterate for a million turns and a more accurate non-symplectic one
    is not. :func:`~accsim.symplectic.is_symplectic_map_canonical` confirms it across
    three decades of amplitude, and the ``(zeta, delta)`` check rejects it, exactly as
    it rejects the exact drift.

    Asserting the rejection matters as much as asserting the acceptance: it is what
    stops someone "repairing" a correct map until the wrong gate goes green.

    ⚠️ **And the rejection is not reliable, which is the sharper warning.** The
    ``(zeta, delta)`` residual is second order in the amplitude *and* suppressed by
    ``1/gamma0^2`` — the two variables differ by ``delta^2/(2 gamma0^2)``. On this
    ``gamma0 = 20`` ring at amplitude ``1e-3`` it is ``8.4e-10``, which slips **under**
    ``is_symplectic_map``'s default ``1e-9`` and passes. So the wrong check does not
    merely reject a correct map; at a plausible amplitude and a realistic energy it
    *accepts* one, for no reason connected to symplecticity. Both behaviours are
    pinned below, with the order that explains them.
    """
    q = Quadrupole(L_Q, K1)
    for amp in (1.0e-3, 1.0e-2, 5.0e-2):
        st = np.array([amp, amp, -amp, 0.7 * amp, amp, amp])
        assert is_symplectic_map_canonical(lambda s: q.track(s, ring_ref), st, ring_ref)

    # The (zeta, delta) check rejects it once the amplitude clears its tolerance...
    for amp in (1.0e-2, 5.0e-2):
        st = np.array([amp, amp, -amp, 0.7 * amp, amp, amp])
        assert not is_symplectic_map(lambda s: q.track(s, ring_ref), st)

    # ...and its residual is second order in the amplitude, which is what says the
    # residual is the coordinates and not the map — and why it can hide below a
    # tolerance at small amplitude instead of failing honestly.
    def zeta_delta_residual(amp: float) -> float:
        st = np.array([amp, amp, -amp, 0.7 * amp, amp, amp])
        M = jacobian(lambda s: q.track(s, ring_ref), st, step=1e-6)
        return float(np.max(np.abs(M.T @ J6 @ M - J6)))

    assert zeta_delta_residual(1.0e-2) / zeta_delta_residual(1.0e-3) == pytest.approx(
        100.0, rel=0.05
    )
    assert zeta_delta_residual(1.0e-3) < 1.0e-9  # under the default atol: a false pass

    # Both planes, both signs of the gradient, and the skew form: symplecticity is a
    # property of the whole 6x6 and a one-plane slip would survive a single case.
    for elem in (
        Quadrupole(L_Q, -K1),
        Quadrupole(L_Q, 0.0),
        Quadrupole(L_Q, K1, roll=0.3),
        SkewQuadrupole(L_Q, K1),
    ):
        assert is_symplectic_map_canonical(
            lambda s, e=elem: e.track(s, ring_ref), 5.0 * STATE, ring_ref
        )


def test_dropping_the_momentum_factor_in_one_place_breaks_symplecticity(
    ring_ref: ReferenceParticle,
) -> None:
    r"""The gate has teeth: the plausible half-implementation is caught.

    ``(1 + delta)`` appears twice — inside the trigonometric argument, and where the
    canonical momentum is turned into an angle. Scaling ``k1`` but forgetting the
    second (so ``x' = px``, the old behaviour) still produces a map that focuses less
    off-momentum and would move the chromaticity in the right direction. It is not
    symplectic, and the canonical check says so at every amplitude, so "the
    chromaticity came out about right" cannot rescue it.
    """
    L, k1 = L_Q, K1

    def half_fixed(state: np.ndarray) -> np.ndarray:
        out = np.asarray(state, dtype=float).copy()
        K = k1 / (1.0 + out[DELTA])
        C, S = _focusing_functions(K, L)
        out[X] = state[X] * C + state[PX] * S  # px used as an angle: the omission
        out[PX] = -K * state[X] * S + state[PX] * C
        return out

    for amp in (1.0e-4, 1.0e-3, 1.0e-2):
        st = np.array([amp, amp, -amp, 0.7 * amp, amp, amp])
        assert not is_symplectic_map_canonical(half_fixed, st, ring_ref)


# --------------------------------------------------------------------------
# 4. The headline: an exact identity in delta, and the chromaticity it closes
# --------------------------------------------------------------------------


def test_the_ring_at_momentum_delta_is_the_design_ring_with_k1_rescaled(
    ring_ref: ReferenceParticle,
) -> None:
    r"""**The discriminating gate**, and it is an identity rather than a tolerance.

    Substituting ``px = (1 + delta) p`` in the exact maps of a bend-free ring turns
    the drift's ``L/(1+delta)`` back into ``L`` and the quadrupole's block into a
    design quadrupole of strength ``k1/(1 + delta)``. So the machine a particle of
    momentum ``delta`` traverses **is** the design machine with every gradient
    rescaled — not approximately, and not to first order.

    That makes the tracked tunes off momentum equal to the *design* tunes of a
    rescaled lattice, and the agreement is ``1e-15`` out to ``delta = 0.05``. It pins
    every order in ``delta`` at once. A map with ``1/(1+delta)^2``, or with the factor
    in the trigonometric argument only, or applied to the horizontal plane alone,
    all agree at ``delta = 0`` and all fail here at ``O(delta)``, where no tolerance
    can absorb them — which is what the first-order chromaticity number below, on its
    own, would not distinguish.
    """
    lat = _straight_ring(ring_ref)
    for delta in (1.0e-3, 1.0e-2, 5.0e-2, -3.0e-2):
        rescaled = Lattice(
            [
                Quadrupole(e.length, e.k1 / (1.0 + delta))
                if isinstance(e, Quadrupole)
                else Drift(e.length)
                for e in lat.elements
            ],
            ring_ref,
        )
        got, want = tunes_on_orbit(lat, delta=delta), tunes(rescaled)
        assert got[0] % 1.0 == pytest.approx(want[0] % 1.0, abs=1e-13)
        assert got[1] % 1.0 == pytest.approx(want[1] % 1.0, abs=1e-13)

    # Non-vacuous: the tune really does move with momentum, by far more than that.
    assert abs(tunes_on_orbit(lat, delta=0.05)[0] - tunes(lat)[0]) > 1.0e-2


def test_tracking_now_recovers_the_whole_natural_chromaticity_of_a_straight_ring(
    ring_ref: ReferenceParticle,
) -> None:
    r"""**What the milestone is for.** 45% becomes 100%, on a ring with no bend.

    L1 left the tracked route seeing the drifts' share of the natural chromaticity and
    not the quadrupoles' — ``-0.1289`` against ``-0.2893`` on the analytic suite's arc.
    With the thick quadrupole carrying its own ``k1/(1+delta)``, tracking and the
    ``-(1/4pi) oint beta k1 ds`` integral are measuring the same thing on a bend-free
    machine, by two routes that share no arithmetic: Newton plus a finite-difference
    Jacobian on one side, a beta-weighted quadrature over ``matrix()`` on the other.

    **The residual is named rather than tolerated.** It does not go to zero, and it
    should not: :func:`~accsim.twiss.natural_chromaticity` sub-slices a thick magnet
    and trapezoids the integrand, so what is left is that quadrature's own error. It
    therefore falls by **four per doubling** of ``slices`` — asserted as that order,
    across four doublings, which a wrong map could not imitate: a mis-scaled ``k1``
    factor would leave a *fixed* offset that no number of slices removes.
    """
    lat = _straight_ring(ring_ref)
    tracked = _tracked_chromaticity(lat)

    # The ring plainly has a natural chromaticity, and tracking lands on it.
    converged = natural_chromaticity(lat, slices=4096)
    assert converged[0] < -0.2 and converged[1] < -0.2
    assert tracked[0] == pytest.approx(converged[0], rel=1e-7)
    assert tracked[1] == pytest.approx(converged[1], rel=1e-7)

    # The residual is the trapezoid's, not the map's: four per doubling of slices.
    residuals = [
        abs(tracked[0] - natural_chromaticity(lat, slices=n)[0]) for n in (64, 128, 256, 512, 1024)
    ]
    for coarse, fine in zip(residuals[:-1], residuals[1:], strict=True):
        assert coarse / fine == pytest.approx(4.0, rel=0.05)
    assert residuals[0] == pytest.approx(9.765e-6, rel=1e-2)


def test_a_thin_quadrupole_ring_already_had_all_of_it(ring_ref: ReferenceParticle) -> None:
    r"""The control that separates "the map is wrong" from "the integral is wrong".

    A *thin* quadrupole needed no fixing (above), so a ring of thin quads and exact
    drifts has had 100% of its natural chromaticity since L1 — the drifts alone carry
    it, because the thin kick is momentum-independent and the drift turns momentum
    into angle. This ring shares the chromaticity integral with the thick-quad ring
    above and shares none of the new code, so if both gates moved together the
    integral would be the suspect and not the map.
    """
    els: list = []
    for _ in range(6):
        els += [ThinQuadrupole(0.4), Drift(0.8), ThinQuadrupole(-0.4), Drift(0.8)]
    lat = Lattice(els, ring_ref)

    tracked = _tracked_chromaticity(lat)
    analytic = natural_chromaticity(lat)  # thin kicks are exact points: no slicing
    assert analytic[0] < -0.1
    assert tracked[0] == pytest.approx(analytic[0], rel=1e-6)
    assert tracked[1] == pytest.approx(analytic[1], rel=1e-6)


def test_the_bend_is_the_only_thing_tracking_is_still_blind_to(
    ring_ref: ReferenceParticle,
) -> None:
    r"""What L2 leaves for L3, in a controlled experiment rather than an estimate.

    A zero-angle :class:`~accsim.elements.dipole.Dipole` and a
    :class:`~accsim.elements.drift.Drift` of the same length have the **same matrix**,
    so a ring built with one has exactly the same design optics — and the same
    analytic natural chromaticity — as the ring built with the other. They differ only
    in ``track``: the drift's is exact, the dipole's is still its matrix.

    Swapping one for the other therefore changes nothing except how much of the
    chromaticity tracking can see, and it moves from **48% to 100%**. That is the
    sharpest available statement of what remains: the residual blindness is the
    dipole's map and nothing else, and it is L3's to close.
    """

    def ring(bend) -> Lattice:
        els: list = []
        for _ in range(4):
            els += [Quadrupole(0.3, K1), bend(), Quadrupole(0.3, -K1), Drift(0.5)]
        return Lattice(els, ring_ref)

    from accsim import Dipole

    straight_bend, real_drift = ring(lambda: Dipole(1.0, 0.0)), ring(lambda: Drift(1.0))

    # Identical design optics — that is what makes this a controlled experiment.
    np.testing.assert_allclose(
        straight_bend.one_turn_matrix(), real_drift.one_turn_matrix(), atol=1e-14
    )
    blind = natural_chromaticity(straight_bend, slices=1024)
    seeing = natural_chromaticity(real_drift, slices=1024)
    assert blind[0] == pytest.approx(seeing[0], rel=1e-9)

    assert _tracked_chromaticity(real_drift)[0] == pytest.approx(seeing[0], rel=1e-5)
    share = _tracked_chromaticity(straight_bend)[0] / blind[0]
    assert 0.3 < share < 0.7, "the dipole's map carries none of it — L3's business"


# --------------------------------------------------------------------------
# 5. The residual inconsistency, asserted rather than hidden
# --------------------------------------------------------------------------


def test_a_zero_strength_thick_quadrupole_is_the_expanded_drift_not_the_exact_one(
    ring_ref: ReferenceParticle,
) -> None:
    r"""**The ROADMAP predicted this milestone would close a gap. It narrows it.**

    L2 was written up as removing L1's inconsistency that a zero-strength thick magnet
    is documented as identical to a :class:`~accsim.elements.drift.Drift` and no
    longer tracks like one. It does not. At ``k1 = 0`` this map is the *expanded*
    drift ``x += L px/(1+delta)``, because the paraxial Hamiltonian it solves exactly
    is the one whose kinetic term has been expanded in the angles; ``Drift`` is the
    *exact* ``x += L px/pz``.

    What L2 does is take the gap from **first** order to **third**: the old linear map
    differed from the exact drift at ``O(px delta)``, this one at ``O(px^3)`` — the
    same gap that separates xtrack's own two drift models, and the price of a closed
    form. Asserted in the shape that identifies it: cubic in the angle (a factor of
    five in ``px`` is a factor of ``125``) and **independent of** ``k1`` as ``k1 -> 0``,
    which is what says it is the angle expansion rather than a gradient bug.

    Short-circuiting ``k1 == 0`` to the exact drift would close it only by making the
    map discontinuous in ``k1``, so the residual is documented instead.
    """
    L = 2.0

    def gap(k1: float, px: float) -> float:
        st = np.array([0.0, px, 0.0, 0.0, 0.0, 1.0e-3])
        return float(
            np.max(np.abs(Quadrupole(L, k1).track(st, ring_ref) - Drift(L).track(st, ring_ref)))
        )

    # Cubic in the angle.
    assert gap(0.0, 5.0e-2) / gap(0.0, 1.0e-2) == pytest.approx(125.0, rel=0.02)
    # Independent of k1 in the limit — the angle expansion, not the gradient.
    assert gap(1.0e-6, 1.0e-2) == pytest.approx(gap(0.0, 1.0e-2), rel=0.02)
    # And it *is* third order and not first: the old linear map was ~1e3 worse here.
    st = np.array([0.0, 1.0e-2, 0.0, 0.0, 0.0, 1.0e-3])
    linear = Quadrupole(L, 0.0).matrix(ring_ref) @ st
    assert gap(0.0, 1.0e-2) < 1.0e-5
    assert float(np.max(np.abs(linear - Drift(L).track(st, ring_ref)))) > 1.0e-5

    # The matrices, meanwhile, are identical — the documented statement stands there.
    np.testing.assert_array_equal(Quadrupole(L, 0.0).matrix(ring_ref), Drift(L).matrix(ring_ref))


# --------------------------------------------------------------------------
# 6. The same magnet, however it is spelled
# --------------------------------------------------------------------------


def test_a_skew_quadrupole_tracks_as_the_rolled_normal_one(
    ring_ref: ReferenceParticle,
) -> None:
    r"""``SkewQuadrupole(k1s)`` is ``Quadrupole(k1s, roll=-45 deg)`` — now in ``track``.

    The identity was already asserted on ``matrix`` (``tests/analytic/test_roll.py``).
    It would have quietly stopped being true in ``track`` when the normal quadrupole
    became momentum-dependent and the skew one did not, so the skew element carries
    the same map through the same conjugation, and the identity is re-asserted where
    it now has content: at nonzero ``delta``, where a linear skew quad would differ.

    :class:`~accsim.elements.skew_quadrupole.ThinSkewQuadrupole` needs no such
    treatment, for the same reason :class:`ThinQuadrupole` did not.
    """
    for k1s in (K1, -0.6):
        skew = SkewQuadrupole(L_Q, k1s)
        rolled = Quadrupole(L_Q, k1s, roll=-math.pi / 4.0)
        for amp in (1.0, 4.0):
            np.testing.assert_allclose(
                skew.track(amp * STATE, ring_ref), rolled.track(amp * STATE, ring_ref), atol=1e-17
            )

    # ...and its Jacobian at the origin is still its own matrix, so G1/G2's coupled
    # optics — all built on matrix() — are untouched.
    skew = SkewQuadrupole(L_Q, K1)
    J = jacobian(lambda s: skew.track(s, ring_ref), np.zeros(DIM), step=1e-6)
    assert np.max(np.abs(J - skew.matrix(ring_ref))) < 1.0e-13


def test_the_map_broadcasts_over_a_bunch_with_a_momentum_spread(
    ring_ref: ReferenceParticle,
) -> None:
    r"""A ``(6, n)`` bunch equals ``n`` single particles — the point being the spread.

    This is not the usual broadcasting check. The focusing strength ``k1/(1+delta)``
    is now a **per-particle** number, so a bunch whose particles have different
    momenta needs ``n`` different cos/sin blocks in one call. The scalar
    :func:`_focusing_block` that ``matrix()`` uses could not do it, and a
    implementation that reached for the reference momentum instead would pass every
    single-particle test in this file.
    """
    bunch = np.stack([STATE, -0.5 * STATE, np.zeros(DIM), 3.0 * STATE], axis=1)
    bunch[DELTA] = np.array([0.0, 1.0e-2, -2.0e-2, 5.0e-2])  # genuinely spread
    for elem in (Quadrupole(L_Q, K1), SkewQuadrupole(L_Q, K1), Quadrupole(L_Q, K1, roll=0.02)):
        got = elem.track(bunch, ring_ref)
        for j in range(bunch.shape[1]):
            np.testing.assert_allclose(got[:, j], elem.track(bunch[:, j], ring_ref), atol=1e-18)

    # Non-vacuous: the spread really does make the particles see different focusing.
    q = Quadrupole(L_Q, K1)
    same = np.stack([STATE, STATE], axis=1)
    same[DELTA] = np.array([0.0, 5.0e-2])
    out = q.track(same, ring_ref)
    assert abs(out[X, 0] - out[X, 1]) > 1.0e-6
