"""M1 acceptance: the optics off-momentum — chromatic functions and ``Q''``.

Every other optics quantity in this package describes the machine at one momentum.
M1 adds the *derivative*: how ``beta`` and ``alpha`` move with ``delta``
(:func:`~accsim.chromatic_functions`) and the curvature of the tune-versus-momentum
curve (:func:`~accsim.second_order_chromaticity`), where
:func:`~accsim.chromaticity` is its slope.

The gates, and why each one is here
-----------------------------------

**The bend-free ring is the control, not a warm-up.** A thin-lens ring has an
*exact* symbolic answer, because a thin quadrupole carries no ``1/(1+delta)`` at
all (its kick changes every particle's momentum by the same amount) and the exact
:class:`~accsim.Drift`, linearised at the origin, is simply a drift of length
``L/(1+delta)``. So the whole momentum dependence of a thin-lens ring is one
substitution, and sympy can differentiate the resulting tune twice in closed form.
That gate proves the finite-difference machinery, the quadrupole map, the drift map
and the phase accumulation are all correct **at second order**, which is what makes
the bendy-ring comparison below attributable to something else.

**The order is gated, not a tolerance.** A second difference divides by
``delta^2``, so closed-orbit noise enters as ``1/delta^2`` and the error curve is
U-shaped: too large a step is truncation, too small and the Newton residual
dominates. A pass/fail at one step size would hide both ends, so the gate is that
halving ``delta`` **quarters** the residual against the symbolic answer — B4's
argument, applied to a derivative instead of a lifetime.

**The bendy ring's ``Q''`` is pinned as a named model boundary, not as a
validated number.** accsim, xtrack and MAD-X give three different answers there
while agreeing on ``Q`` to ten digits and ``Q'`` to seven. That is *not* an accsim
map error — the reference suite proves the Dipole Jacobian and the closed orbit
both match xtrack — so the value is pinned here to stop it drifting silently, and
the milestone that closes it is M2. See
``docs/CONVENTIONS.md`` -> *Second-order chromaticity is not arbitrated on a bendy
ring*.
"""

from __future__ import annotations

import math

import pytest

from accsim import (
    ChromaticTwiss,
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinQuadrupole,
    ThinSextupole,
    chromatic_functions,
    chromaticity,
    propagate_twiss_on_orbit,
    second_order_chromaticity,
)

# An *asymmetric* thin FODO: the F and D strengths differ, so the two planes carry
# genuinely different numbers. A symmetric cell makes Q''_x == Q''_y identically and
# a plane swap would pass unnoticed.
KF = 1.0 / 1.5  # integrated strength of the (split) focusing quad [m^-1]
KD = 1.0 / 1.1  # integrated strength of the defocusing quad [m^-1]
L_HALF = 1.0  # half-cell drift [m]
N_CELLS = 4

# The dispersive arc the bendy-ring gates use.
LQ, K1, LD, LB, ANG, N_ARC = 0.3, 1.2, 0.5, 1.0, 0.12, 3


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _thin_ring(ref: ReferenceParticle) -> Lattice:
    cell = [
        ThinQuadrupole(0.5 * KF),
        Drift(L_HALF),
        ThinQuadrupole(-KD),
        Drift(L_HALF),
        ThinQuadrupole(0.5 * KF),
    ]
    return Lattice(cell * N_CELLS, ref)


def _arc(ref: ReferenceParticle, k2l: float = 0.0) -> Lattice:
    els: list = []
    for _ in range(N_ARC):
        els += [
            Quadrupole(LQ, K1),
            Drift(LD),
            ThinSextupole(k2l),
            Drift(LD),
            Dipole(LB, ANG),
            Quadrupole(LQ, -K1),
            Dipole(LB, ANG),
            Drift(LD),
        ]
    return Lattice(els, ref)


# ---------------------------------------------------------------------------
# 1. The mechanism the closed form rests on
# ---------------------------------------------------------------------------


def test_the_exact_drift_off_momentum_is_a_shorter_drift(ref: ReferenceParticle) -> None:
    r"""Linearised at the origin, ``Drift(L)`` at momentum ``delta`` is ``Drift(L/(1+delta))``.

    This is the fact the whole thin-lens closed form below rests on, so it is
    asserted rather than assumed. The exact map moves ``x`` by ``L px / p_s`` with
    ``p_s = sqrt((1+delta)^2 - px^2 - py^2)``; on axis ``p_s = 1 + delta``, so the
    Jacobian's ``M[x, px]`` entry is ``L/(1+delta)`` and nothing else in the
    transverse block moves.

    A thin quadrupole contributes **no** ``delta`` dependence of its own (see
    :class:`~accsim.ThinQuadrupole`), so on a thin-lens ring this drift term is the
    entire source of chromaticity — first order and second alike.
    """
    from accsim.symplectic import jacobian

    length, d = 2.5, 3e-3
    state = [0.0, 0.0, 0.0, 0.0, 0.0, d]
    j = jacobian(lambda s: Drift(length).track(s, ref), state, step=1e-7)
    assert j[0][1] == pytest.approx(length / (1.0 + d), rel=1e-9)
    assert j[2][3] == pytest.approx(length / (1.0 + d), rel=1e-9)


# ---------------------------------------------------------------------------
# 2. The closed form, derived symbolically — the bend-free control
# ---------------------------------------------------------------------------


def _symbolic() -> dict[str, float]:
    """``Q'``, ``Q''`` and ``dbeta/ddelta`` of the asymmetric thin FODO, in sympy.

    The one-turn map is built as a function of ``delta`` by the single substitution
    ``L -> L/(1+delta)``, then the tune comes from the trace and beta from ``M12 /
    sin mu``. Nothing here reuses accsim's formulas — the derivatives are taken of
    the *map*, so agreement is a real cross-check rather than the same sum twice.
    """
    sp = pytest.importorskip("sympy")
    d = sp.symbols("delta")
    le = sp.nsimplify(L_HALF) / (1 + d)
    out: dict[str, float] = {}
    for plane, sign in (("x", 1), ("y", -1)):
        qfh = sp.Matrix([[1, 0], [-sign * sp.nsimplify(0.5 * KF), 1]])
        qd = sp.Matrix([[1, 0], [sign * sp.nsimplify(KD), 1]])
        dr = sp.Matrix([[1, le], [0, 1]])
        m = qfh * dr * qd * dr * qfh
        cos_mu = (m[0, 0] + m[1, 1]) / 2
        mu = sp.acos(cos_mu)
        q = N_CELLS * mu / (2 * sp.pi)
        out[f"dq_{plane}"] = float(sp.diff(q, d).subs(d, 0))
        out[f"ddq_{plane}"] = float(sp.diff(q, d, 2).subs(d, 0))
        beta = m[0, 1] / sp.sin(mu)
        out[f"beta_{plane}"] = float(beta.subs(d, 0))
        out[f"dbeta_{plane}"] = float(sp.diff(beta, d).subs(d, 0))
    return out


def test_second_order_chromaticity_matches_the_symbolic_second_derivative(
    ref: ReferenceParticle,
) -> None:
    """``Q''`` on the bend-free ring reproduces ``d^2Q/ddelta^2`` from the map.

    The acceptance gate. Both planes are checked and they carry different numbers,
    so a plane swap fails here.
    """
    sym = _symbolic()
    ddq_x, ddq_y = second_order_chromaticity(_thin_ring(ref), delta=5e-4)
    assert ddq_x == pytest.approx(sym["ddq_x"], rel=2e-5)
    assert ddq_y == pytest.approx(sym["ddq_y"], rel=2e-5)
    # The planes really are distinguishable on this ring.
    assert abs(sym["ddq_x"] - sym["ddq_y"]) > 0.05 * abs(sym["ddq_x"])


def test_the_second_difference_is_q_double_prime_not_half_of_it(
    ref: ReferenceParticle,
) -> None:
    r"""The factor-of-two convention, pinned.

    ``Q(delta) = Q0 + Q' delta + Q'' delta^2 / 2`` — so a code that returns the
    *coefficient* of ``delta^2`` returns half of what this function does. MAD-X's
    and xtrack's columns do not agree with each other about such factors in
    general, which is exactly why this is asserted against sympy rather than
    against a reference column.
    """
    sym = _symbolic()
    ddq_x, _ = second_order_chromaticity(_thin_ring(ref), delta=5e-4)
    assert ddq_x == pytest.approx(sym["ddq_x"], rel=2e-5)
    assert ddq_x != pytest.approx(0.5 * sym["ddq_x"], rel=1e-2)


def test_the_residual_falls_as_the_square_of_the_step(ref: ReferenceParticle) -> None:
    r"""Halving ``delta`` quarters the error — the convergence *order*, not a tolerance.

    A second difference has a U-shaped error curve (truncation at large ``delta``,
    closed-orbit noise amplified by ``1/delta^2`` at small), so agreement at one
    step size proves very little. Gating the order instead says the machinery is
    converging on the closed form rather than landing near it by luck.
    """
    lat, exact = _thin_ring(ref), _symbolic()["ddq_x"]
    residuals = [
        abs(second_order_chromaticity(lat, delta=d)[0] - exact) for d in (2e-3, 1e-3, 5e-4)
    ]
    assert residuals[0] > residuals[1] > residuals[2]
    for coarse, fine in zip(residuals, residuals[1:], strict=False):
        assert coarse / fine == pytest.approx(4.0, rel=0.25)


def test_chromatic_beta_derivative_matches_the_symbolic_one(ref: ReferenceParticle) -> None:
    """``dbeta/ddelta`` at the cell start, against the symbolic matched beta.

    The per-element readout is what makes this gate discriminating: a one-turn
    scalar can come out right by cancellation around the ring, a beta derivative at
    a named point cannot.
    """
    sym = _symbolic()
    ch = chromatic_functions(_thin_ring(ref), delta=5e-4)[0]
    tw = propagate_twiss_on_orbit(_thin_ring(ref))[0]
    assert tw.beta_x == pytest.approx(sym["beta_x"], rel=1e-9)
    assert tw.beta_y == pytest.approx(sym["beta_y"], rel=1e-9)
    assert ch.dbeta_x == pytest.approx(sym["dbeta_x"], rel=1e-5)
    assert ch.dbeta_y == pytest.approx(sym["dbeta_y"], rel=1e-5)


# ---------------------------------------------------------------------------
# 3. The MAD8 combinations are definitions, and are applied as such
# ---------------------------------------------------------------------------


def test_the_mad8_combinations_are_what_they_claim(ref: ReferenceParticle) -> None:
    r"""``b = dbeta/beta``, ``a = dalpha - dbeta alpha/beta``, ``w = hypot(a, b)``.

    These three are a *normalisation*, not new physics, so the gate is that the
    reported combination really is that combination of the reported raw
    derivatives and the on-momentum optics — the place a stray ``alpha`` or a
    missing division would sit.
    """
    lat = _arc(ref)
    chrom = chromatic_functions(lat, delta=1e-3)
    centre = propagate_twiss_on_orbit(lat)
    assert len(chrom) == len(centre)
    for ch, tw in zip(chrom, centre, strict=True):
        assert ch.b_x == pytest.approx(ch.dbeta_x / tw.beta_x, rel=1e-12)
        assert ch.b_y == pytest.approx(ch.dbeta_y / tw.beta_y, rel=1e-12)
        assert ch.a_x == pytest.approx(
            ch.dalpha_x - ch.dbeta_x * tw.alpha_x / tw.beta_x, rel=1e-12, abs=1e-14
        )
        assert ch.a_y == pytest.approx(
            ch.dalpha_y - ch.dbeta_y * tw.alpha_y / tw.beta_y, rel=1e-12, abs=1e-14
        )
        assert ch.w_x == pytest.approx(math.hypot(ch.a_x, ch.b_x), rel=1e-12)
        assert ch.w_y == pytest.approx(math.hypot(ch.a_y, ch.b_y), rel=1e-12)
        assert ch.s == tw.s


def test_chromatic_functions_are_reported_at_every_boundary(ref: ReferenceParticle) -> None:
    """One :class:`ChromaticTwiss` per element boundary, in ``s`` order."""
    lat = _arc(ref)
    chrom = chromatic_functions(lat, delta=1e-3)
    assert all(isinstance(c, ChromaticTwiss) for c in chrom)
    assert chrom[0].s == pytest.approx(0.0)
    assert chrom[-1].s == pytest.approx(lat.length)
    assert all(b.s >= a.s for a, b in zip(chrom, chrom[1:], strict=False))


# ---------------------------------------------------------------------------
# 4. The sextupole reaches Q'' — the feed-down share, gated on its order
# ---------------------------------------------------------------------------


def test_the_sextupole_reaches_q_prime_linearly_and_q_double_prime_quadratically(
    ref: ReferenceParticle,
) -> None:
    r"""The two chromaticities take the sextupole at **different powers**, and that is the gate.

    A sextupole at dispersion sits at ``D_x delta`` and feeds down a gradient
    ``k2l D_x delta``. That gradient is first order in both ``k2l`` and ``delta``,
    so it lands on ``Q'`` **linearly** — and it cannot reach ``Q''`` by the same
    route, because a term linear in ``delta`` contributes nothing to a second
    derivative. ``Q''`` is reached only at second order in the perturbation, where
    the feed-down gradient beats ``beta`` and dispersion and those in turn shift the
    tune, which is **quadratic** in ``k2l``.

    So the same element arrives at two quantities at two different powers, and the
    gate is the pair of exponents rather than either value. This is J2's lesson —
    gate on the order — and it discriminates in a way no tolerance does: a uniformly
    mis-scaled feed-down keeps both exponents, but a term that reached ``Q''`` by
    the wrong mechanism would carry the wrong power of ``k2l``.

    The linear half is exact to round-off, which is what makes it worth asserting so
    tightly; the quadratic half converges on its exponent from above as ``k2l``
    shrinks, because a cubic term is also present.
    """
    strengths = (0.5, 1.0, 2.0)
    bare_q1 = chromaticity(_arc(ref))[0]
    bare_q2 = second_order_chromaticity(_arc(ref), delta=1e-3)[0]
    dq1 = [chromaticity(_arc(ref, k2l=k))[0] - bare_q1 for k in strengths]
    dq2 = [second_order_chromaticity(_arc(ref, k2l=k), delta=1e-3)[0] - bare_q2 for k in strengths]

    # Q' takes it exactly linearly: dQ'/k2l is one number, to round-off.
    per_unit = [d / k for d, k in zip(dq1, strengths, strict=True)]
    for value in per_unit[1:]:
        assert value == pytest.approx(per_unit[0], rel=1e-9)

    # Q'' takes it quadratically: the measured exponent is 2, not 1.
    assert abs(dq2[0]) > 1.0  # the sextupole genuinely reaches Q'' at all
    exponents = [
        math.log(fine / coarse) / math.log(2.0) for coarse, fine in zip(dq2, dq2[1:], strict=False)
    ]
    for p in exponents:
        assert p == pytest.approx(2.0, abs=0.06)
    assert min(exponents) > 1.5  # decisively not the linear power Q' takes


# ---------------------------------------------------------------------------
# 5. The named model boundary — pinned so it cannot drift quietly
# ---------------------------------------------------------------------------


def test_the_bendy_ring_second_order_chromaticity_is_pinned(ref: ReferenceParticle) -> None:
    r"""``Q''`` on the dispersive arc, pinned as a **boundary**, not as a validated number.

    accsim gives ``(+0.79307, +0.76830)`` here; xtrack gives ``+0.75202`` and MAD-X
    ``+0.70441`` in ``x``, all three agreeing on ``Q`` to ten digits and on ``Q'``
    to seven. The reference suite establishes that this is not an accsim map error
    (the Dipole Jacobian matches ``xt.Bend`` to ``5.3e-10`` on the off-momentum
    orbit, and the closed orbits agree to ``2.9e-11``), and the two independent
    accsim tune routes agree with each other to seven digits.

    So this test exists to make the number *stable*, not to bless it: if a future
    change moves it, that is a real change in the package's off-momentum optics and
    should be noticed. M2 is the milestone that resolves which of the three is
    right.
    """
    ddq_x, ddq_y = second_order_chromaticity(_arc(ref), delta=1e-3)
    assert ddq_x == pytest.approx(0.793072, rel=1e-4)
    assert ddq_y == pytest.approx(0.768303, rel=1e-4)


def test_the_two_tune_routes_agree_on_the_bendy_ring(ref: ReferenceParticle) -> None:
    r"""Accumulated Twiss phase and the one-turn trace give the same ``Q''``.

    ``second_order_chromaticity`` differences :func:`~accsim.tunes_on_orbit`, which
    accumulates phase element by element. The one-turn map's trace is a wholly
    separate route to the same tune (fractional only, which is why it is not the
    shipped one). They share the element maps but nothing else, so agreement rules
    out a bias in the phase accumulation — the remaining place a second-order tune
    error could hide once the maps are known to be right.
    """
    import numpy as np

    from accsim.orbit import linearised_one_turn_map

    lat = _arc(ref)

    def fractional(d: float) -> float:
        m = linearised_one_turn_map(lat, delta=d)
        block = m[np.ix_([0, 1], [0, 1])]
        half_trace = 0.5 * (block[0, 0] + block[1, 1])
        q = math.acos(max(-1.0, min(1.0, half_trace))) / (2.0 * math.pi)
        return q if block[0, 1] >= 0.0 else 1.0 - q

    h = 1e-3
    by_trace = (fractional(h) - 2.0 * fractional(0.0) + fractional(-h)) / (h * h)
    by_phase = second_order_chromaticity(lat, delta=h)[0]
    assert by_trace == pytest.approx(by_phase, rel=1e-4)


# ---------------------------------------------------------------------------
# 6. Argument handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1e-3])
def test_a_non_positive_step_is_rejected(ref: ReferenceParticle, bad: float) -> None:
    """``delta`` is a step size, so zero or negative is an error, not a silent nan."""
    lat = _thin_ring(ref)
    with pytest.raises(ValueError, match="delta"):
        second_order_chromaticity(lat, delta=bad)
    with pytest.raises(ValueError, match="delta"):
        chromatic_functions(lat, delta=bad)
