r"""H2 acceptance: insertion matching — local optics at a point, N knobs -> M targets.

H1 matched two *global* scalars (the tunes, the chromaticities) with two knobs.
H2 asks the harder and more useful question: put a **waist of a given ``beta*``
at this point**, which is a constraint on the local Twiss functions rather than
on a one-turn trace.

The physics content is the waist condition itself, and the strong test derives it
symbolically rather than trusting the matcher's own propagation. For a line that
starts at a waist ``beta0``, drifts ``d1``, meets a thin lens ``u = 1/f`` and
drifts ``d2``, demanding ``alpha = 0`` at the exit gives, in ``u``,

    (d1^2 d2 + d2 beta0^2) u^2 - (d1^2 + 2 d1 d2 + beta0^2) u + (d1 + d2) = 0,

derived here with sympy from ``B -> M B M^T`` and never hard-coded as a remembered
formula. Two consequences drive the whole gate:

* the emergent ``beta*`` is **determined, not chosen** — with one knob you may ask
  for a waist *or* for a value of ``beta*``, not both, so a one-knob two-target
  problem is over-determined by construction and must be *reported* as such;
* the waist is **not unique** — a quadratic has two roots, with two different
  ``beta*``. Newton lands on whichever the starting strength is nearest, so the
  gate matches from two different starts and asserts each root separately. A test
  that asserted "the" focal length would flake on the branch.

The two-knob exercise builds a target with a *known exact* solution the same way:
fix the second lens, solve the (still quadratic) waist condition exactly for the
first, and read off the emergent ``beta*``. Matching ``(beta*, alpha*=0)`` from
that pair then has an analytic answer to assert the recovered **strengths**
against, not merely a small residual.

The finite-difference Jacobian is pinned the way H1 pinned its perturbation
integral: against a symbolic ``d(beta)/dv`` differentiated from the closed
solution of a thin FODO, which knows nothing about finite differences.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Drift,
    Knob,
    Lattice,
    MatchingError,
    Quadrupole,
    ReferenceParticle,
    Sextupole,
    Target,
    ThinQuadrupole,
    Twiss,
    UnstableLatticeError,
    closed_twiss,
    insertion_response_matrix,
    match_insertion,
    matching,
    propagate_twiss,
    tunes,
)

# Thin waist-to-waist line: waist(BETA0) -> Drift(D1) -> ThinQuadrupole -> Drift(D2).
BETA0 = 5.0
D1 = 3.0
D2 = 2.0

# Thin FODO used for the periodic branch and the Jacobian pin.
VF_NOMINAL = 1.0 / 1.5
VD_NOMINAL = 1.0 / 1.6
L_HALF = 1.0


@pytest.fixture
def ref() -> ReferenceParticle:
    # Thin quads + drifts are energy-independent; any ref works.
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _waist(beta0: float = BETA0) -> Twiss:
    """Entrance Twiss of a round waist: alpha = 0 in both planes."""
    return Twiss(0.0, beta0, 0.0, 0.0, beta0, 0.0, 0.0)


def _waist_line(ref: ReferenceParticle, u: float) -> tuple[Lattice, ThinQuadrupole]:
    """``Drift(D1) -> ThinQuadrupole(u) -> Drift(D2)``; the exit is boundary 3."""
    q = ThinQuadrupole(u)
    return Lattice([Drift(D1), q, Drift(D2)], ref=ref), q


def _thin_fodo(ref: ReferenceParticle, vf: float = VF_NOMINAL, vd: float = VD_NOMINAL):
    """Thin FODO cell built from the F centre: QF/2 D QD D QF/2."""
    qf1, qf2 = ThinQuadrupole(0.5 * vf), ThinQuadrupole(0.5 * vf)
    qd = ThinQuadrupole(-vd)
    lat = Lattice([qf1, Drift(L_HALF), qd, Drift(L_HALF), qf2], ref=ref)
    return lat, (qf1, qf2), qd


# --------------------------------------------------------------------------
# The waist condition, derived symbolically
# --------------------------------------------------------------------------


def _waist_roots(beta0: float = BETA0, d1: float = D1, d2: float = D2):
    """Both symbolic roots ``u = 1/f`` of the waist condition, and their ``beta*``.

    Derives ``alpha`` at the exit from ``B -> M B M^T`` with sympy — no formula is
    recalled, and in particular the quadratic's coefficients are read off the
    derivation rather than written down.
    """
    sp = pytest.importorskip("sympy")
    u, b0, dd1, dd2 = sp.symbols("u beta0 d1 d2", real=True)

    def drift(length):
        return sp.Matrix([[1, length], [0, 1]])

    # ThinQuadrupole(k1l): px -> px - k1l*x, so the x block is [[1, 0], [-k1l, 1]].
    m = drift(dd2) * sp.Matrix([[1, 0], [-u, 1]]) * drift(dd1)
    b_mat = m * sp.Matrix([[b0, 0], [0, 1 / b0]]) * m.T
    alpha, beta = -b_mat[0, 1], b_mat[0, 0]

    subs = {b0: sp.nsimplify(beta0), dd1: sp.nsimplify(d1), dd2: sp.nsimplify(d2)}
    poly = sp.Poly(sp.expand(sp.numer(sp.together(alpha.subs(subs)))), u)
    roots = sorted(float(r) for r in sp.solve(sp.Eq(poly.as_expr(), 0), u) if sp.im(r) == 0)
    betas = [float(beta.subs(subs).subs(u, sp.nsimplify(r, rational=False))) for r in roots]
    return poly, roots, betas


def test_waist_condition_is_a_quadratic_with_two_distinct_roots() -> None:
    """The symbolic alpha=0 condition has the stated coefficients, and two roots."""
    sp = pytest.importorskip("sympy")
    poly, roots, betas = _waist_roots()

    # Coefficients, up to the overall sign a numerator is defined to within.
    u = poly.gens[0]
    stated = (D1**2 * D2 + D2 * BETA0**2) * u**2 - (D1**2 + 2 * D1 * D2 + BETA0**2) * u + (D1 + D2)
    ratio = sp.simplify(poly.as_expr() / sp.expand(stated))
    assert ratio.is_number and abs(abs(float(ratio)) - 1.0) < 1e-12

    assert len(roots) == 2
    assert abs(roots[0] - roots[1]) > 1e-3
    # Both roots are waists, and they are *different* waists: beta* is determined
    # by the root, which is exactly why one knob cannot also choose beta*.
    assert abs(betas[0] - betas[1]) > 1e-3


# --------------------------------------------------------------------------
# One knob -> one target: recover the focal length, on both branches
# --------------------------------------------------------------------------


@pytest.mark.parametrize("branch", [0, 1])
def test_one_knob_recovers_the_symbolic_waist_root(ref: ReferenceParticle, branch: int) -> None:
    """Matching alpha_x = 0 at the exit lands on the nearer symbolic root, exactly."""
    _, roots, betas = _waist_roots()
    root, beta_star = roots[branch], betas[branch]

    # Start off the answer, but nearer this root than the other one.
    start = root + 0.1 * (roots[1] - roots[0]) * (1 if branch == 0 else -1)
    lattice, quad = _waist_line(ref, start)
    knob = Knob([quad], name="f")

    result = match_insertion(lattice, [Target("alpha_x", at=3, value=0.0)], [knob], twiss0=_waist())

    assert quad.k1l == pytest.approx(root, abs=1e-12)
    assert result.values[0] == pytest.approx(root, abs=1e-12)
    assert result.residual <= 1e-10
    # beta* was never asked for; it is whatever this root delivers.
    exit_twiss = propagate_twiss(lattice, _waist())[3]
    assert exit_twiss.beta_x == pytest.approx(beta_star, rel=1e-12)
    assert exit_twiss.alpha_x == pytest.approx(0.0, abs=1e-12)


def test_matched_waist_reproduces_the_drift_beta_growth(ref: ReferenceParticle) -> None:
    """Downstream of the matched waist, beta(s) = beta* + s^2/beta* — the Stage-6 relation."""
    _, roots, betas = _waist_roots()
    lattice, quad = _waist_line(ref, roots[0] + 0.02)
    match_insertion(lattice, [Target("alpha_x", at=3, value=0.0)], [Knob([quad])], twiss0=_waist())

    beta_star = propagate_twiss(lattice, _waist())[3].beta_x
    assert beta_star == pytest.approx(betas[0], rel=1e-12)
    for extra in (0.5, 1.0, 3.0):
        longer = Lattice([*lattice.elements, Drift(extra)], ref=ref)
        end = propagate_twiss(longer, _waist())[-1]
        assert end.beta_x == pytest.approx(beta_star + extra**2 / beta_star, rel=1e-12)
        assert end.alpha_x == pytest.approx(-extra / beta_star, rel=1e-12)


# --------------------------------------------------------------------------
# Two knobs -> two targets: (beta*, alpha* = 0), against a known exact solution
# --------------------------------------------------------------------------

# Geometry of the doublet insertion: waist -> d1 -> Q1 -> d2 -> Q2 -> d3 -> IP.
DD1, DD2, DD3 = 2.0, 1.0, 3.0
U2_FIXED = 0.25


def _doublet_solution(beta0: float = BETA0, u2_fixed: float = U2_FIXED, *, big: bool = False):
    """An exactly-solvable ``(beta*, alpha*=0)`` target for the doublet.

    Fix ``u2``; the waist condition is then still a quadratic in ``u1``, so the
    pair ``(u1, u2)`` and the emergent ``beta*`` are all known in closed form.
    Matching *back* to that ``beta*`` therefore has a known answer to assert the
    recovered strengths against. ``big`` selects the root with the larger emergent
    ``beta*`` (the two roots differ by orders of magnitude).
    """
    sp = pytest.importorskip("sympy")
    u1, u2 = sp.symbols("u1 u2", real=True)

    def drift(length):
        return sp.Matrix([[1, sp.nsimplify(length)], [0, 1]])

    def thinq(u):
        return sp.Matrix([[1, 0], [-u, 1]])

    m = drift(DD3) * thinq(u2) * drift(DD2) * thinq(u1) * drift(DD1)
    b0 = sp.nsimplify(beta0)
    b_mat = m * sp.Matrix([[b0, 0], [0, 1 / b0]]) * m.T
    alpha, beta = -b_mat[0, 1], b_mat[0, 0]

    at_u2 = {u2: sp.nsimplify(u2_fixed)}
    roots = [r for r in sp.solve(sp.Eq(sp.expand(alpha.subs(at_u2)), 0), u1) if sp.im(r) == 0]
    assert roots, "the fixed u2 must admit a real waist"
    key = (lambda r: float(beta.subs(at_u2).subs(u1, r))) if big else (lambda r: -abs(float(r)))
    u1_true = max(roots, key=key)
    beta_star = float(beta.subs(at_u2).subs(u1, u1_true))
    return float(u1_true), float(u2_fixed), beta_star


def test_two_knobs_recover_the_known_doublet_strengths(ref: ReferenceParticle) -> None:
    """N=M=2: (beta*, alpha*=0) at the IP recovers the exact (u1, u2) it was built from."""
    u1_true, u2_true, beta_star = _doublet_solution()
    q1, q2 = ThinQuadrupole(u1_true * 0.8), ThinQuadrupole(u2_true * 1.2)
    lattice = Lattice(
        [Drift(DD1), q1, Drift(DD2), q2, Drift(DD3)],
        ref=ref,
    )
    ip = len(lattice)  # the exit boundary

    result = match_insertion(
        lattice,
        [Target("beta_x", at=ip, value=beta_star), Target("alpha_x", at=ip, value=0.0)],
        [Knob([q1], name="q1"), Knob([q2], name="q2")],
        twiss0=_waist(),
    )

    assert q1.k1l == pytest.approx(u1_true, abs=1e-10)
    assert q2.k1l == pytest.approx(u2_true, abs=1e-10)
    assert result.residual <= 1e-10
    assert len(result.residuals) == 2
    assert max(abs(r) for r in result.residuals) < 1e-9
    assert result.iterations > 0


# --------------------------------------------------------------------------
# The finite-difference Jacobian, pinned against a symbolic derivative
# --------------------------------------------------------------------------


def _symbolic_dbeta_dv(vf: float, vd: float) -> np.ndarray:
    """``d(beta_x, beta_y)/d(vf, vd)`` of the *periodic* thin FODO at its start.

    Builds the one-turn map symbolically, takes ``beta = M12 / sin mu`` with
    ``cos mu = 1/2 Tr M`` — the closed solution, differentiated directly. It knows
    nothing about finite differences, so agreement pins the step choice, the
    central-difference formula, and the sign of every column at once.
    """
    sp = pytest.importorskip("sympy")
    f, d, ll = sp.symbols("v_f v_d L", positive=True)

    def beta_of(sign: int):
        qfh = sp.Matrix([[1, 0], [-sign * f / 2, 1]])
        qd = sp.Matrix([[1, 0], [+sign * d, 1]])
        drift = sp.Matrix([[1, ll], [0, 1]])
        m = qfh * drift * qd * drift * qfh
        cos_mu = (m[0, 0] + m[1, 1]) / 2
        return m[0, 1] / sp.sqrt(1 - cos_mu**2)

    subs = {f: vf, d: vd, ll: L_HALF}
    return np.array(
        [
            [float(sp.diff(beta_of(+1), v).subs(subs)) for v in (f, d)],
            [float(sp.diff(beta_of(-1), v).subs(subs)) for v in (f, d)],
        ]
    )


def test_response_matrix_matches_the_symbolic_derivative(ref: ReferenceParticle) -> None:
    """The FD Jacobian equals d(beta)/dv from the symbolic closed solution."""
    lattice, qfs, qd = _thin_fodo(ref)
    knobs = [Knob(list(qfs), [0.5, 0.5], name="kqf"), Knob([qd], [-1.0], name="kqd")]
    targets = [Target("beta_x", at=0, value=0.0), Target("beta_y", at=0, value=0.0)]

    jac = insertion_response_matrix(lattice, targets, knobs)
    expected = _symbolic_dbeta_dv(VF_NOMINAL, VD_NOMINAL)
    # Measured 7.9e-11 relative — the central-difference truncation floor at
    # h ~ 1e-6, not a loose gate hiding a sign or a factor.
    assert jac == pytest.approx(expected, rel=1e-9)


def test_finite_difference_survives_a_knob_at_zero(ref: ReferenceParticle) -> None:
    """A knob starting at v=0 still gets a real column, not a zero one."""
    q = ThinQuadrupole(0.0)
    lattice = Lattice([Drift(D1), q, Drift(D2)], ref=ref)
    jac = insertion_response_matrix(
        lattice, [Target("alpha_x", at=3, value=0.0)], [Knob([q])], twiss0=_waist()
    )
    # d(alpha)/du at u = 0 is -(beta0 + d1*d2*(...)) — nonzero and finite is the point.
    assert abs(float(jac[0, 0])) > 1.0
    assert math.isfinite(float(jac[0, 0]))


# --------------------------------------------------------------------------
# The periodic branch: the closed solution is re-solved, not frozen
# --------------------------------------------------------------------------


def test_periodic_match_moves_the_closed_solution(ref: ReferenceParticle) -> None:
    """Matching beta_x at the cell start re-solves the ring optics each step."""
    lattice, qfs, qd = _thin_fodo(ref)
    knobs = [Knob(list(qfs), [0.5, 0.5], name="kqf"), Knob([qd], [-1.0], name="kqd")]
    before = closed_twiss(lattice)
    target_bx, target_by = before.beta_x * 1.15, before.beta_y * 0.9

    result = match_insertion(
        lattice,
        [Target("beta_x", at=0, value=target_bx), Target("beta_y", at=0, value=target_by)],
        knobs,
    )

    after = closed_twiss(lattice)
    assert after.beta_x == pytest.approx(target_bx, rel=1e-11)
    assert after.beta_y == pytest.approx(target_by, rel=1e-11)
    assert result.residual <= 1e-10
    # A quadrupole that moves beta moves the tunes too — the closed solution really
    # was re-solved rather than propagated from a frozen entrance Twiss.
    assert tunes(lattice) != pytest.approx((0.0, 0.0), abs=0.0)
    assert abs(closed_twiss(lattice).beta_x - before.beta_x) > 1e-3


def test_periodic_target_at_an_interior_boundary(ref: ReferenceParticle) -> None:
    """A target at the D-quad boundary is matched, and the point really is interior."""
    lattice, qfs, qd = _thin_fodo(ref)
    knobs = [Knob(list(qfs), [0.5, 0.5]), Knob([qd], [-1.0])]
    at = 2  # exit of QD
    before = propagate_twiss(lattice, closed_twiss(lattice))[at]

    match_insertion(
        lattice,
        [
            Target("beta_x", at=at, value=before.beta_x * 0.92),
            Target("beta_y", at=at, value=before.beta_y * 1.08),
        ],
        knobs,
    )
    after = propagate_twiss(lattice, closed_twiss(lattice))[at]
    assert after.beta_x == pytest.approx(before.beta_x * 0.92, rel=1e-11)
    assert after.beta_y == pytest.approx(before.beta_y * 1.08, rel=1e-11)


# --------------------------------------------------------------------------
# Dispersion targets
# --------------------------------------------------------------------------


def test_dispersion_target_in_the_periodic_branch(ref: ReferenceParticle) -> None:
    """D_x at a ring boundary comes from the *re-solved* matched dispersion.

    Different code from the line case: there the dispersion is transported
    affinely from a fixed entrance, here it is the periodic solution of the
    one-turn map, re-found every evaluation as the quadrupole moves.
    """
    from accsim import Dipole

    qf = Quadrupole(0.3, 1.2)
    qd = Quadrupole(0.3, -1.2)
    cell = [qf, Drift(0.5), Dipole(1.0, 0.12), qd, Drift(0.5), Dipole(1.0, 0.12)]
    lattice = Lattice(cell * 3, ref)

    before = closed_twiss(lattice)
    assert abs(before.disp_x) > 0.1  # the ring really is dispersive
    want = before.disp_x * 0.9

    result = match_insertion(
        lattice, [Target("disp_x", at=0, value=want)], [Knob([qf], name="kqf")]
    )
    assert closed_twiss(lattice).disp_x == pytest.approx(want, rel=1e-11)
    assert result.residual <= 1e-12
    # Not vacuous: the strength moved and so did the dispersion.
    assert qf.k1 != pytest.approx(1.2, rel=1e-4)
    assert abs(closed_twiss(lattice).disp_x - before.disp_x) > 1e-3


def test_response_matrix_leaves_the_lattice_untouched(ref: ReferenceParticle) -> None:
    """Differencing perturbs the knobs, so it must restore them — even when it raises.

    This is the only response matrix in the package that mutates the lattice at
    all, and it is public: a standalone caller has no outer rollback to fall back
    on. The failing case is an exception that is *not* ``UnstableLatticeError``.
    """
    lattice, qfs, qd = _thin_fodo(ref)
    knobs = [Knob(list(qfs), [0.5, 0.5]), Knob([qd], [-1.0])]
    targets = [Target("beta_x", at=0, value=1.0), Target("beta_y", at=0, value=1.0)]
    strengths = [q.k1l for q in (*qfs, qd)]

    insertion_response_matrix(lattice, targets, knobs)
    assert [q.k1l for q in (*qfs, qd)] == strengths  # exact, not approximate

    # Now fail the observation *after* the baseline, i.e. with a knob displaced.
    # UnstableLatticeError is handled and would not exercise the finally.
    class _BoomError(RuntimeError):
        pass

    real = matching._observe
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise _BoomError("the optics blew up mid-difference")
        return real(*args, **kwargs)

    matching._observe = flaky
    try:
        with pytest.raises(_BoomError):
            insertion_response_matrix(lattice, targets, knobs)
    finally:
        matching._observe = real
    assert calls["n"] > 1  # the failure really happened while a knob was displaced
    assert [q.k1l for q in (*qfs, qd)] == strengths


def test_dispersion_target_in_a_line(ref: ReferenceParticle) -> None:
    """A quadrupole knob can zero D_px downstream of a dispersive entrance."""
    from accsim import Dipole

    q = Quadrupole(0.4, 0.3)
    lattice = Lattice([Dipole(1.0, 0.1), Drift(0.5), q, Drift(0.5)], ref=ref)
    start = Twiss(0.0, 6.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    at = len(lattice)

    result = match_insertion(
        lattice, [Target("disp_px", at=at, value=0.0)], [Knob([q])], twiss0=start
    )
    end = propagate_twiss(lattice, start)[at]
    # The Jacobian is first-order but the residual is exact, so the fixed point is
    # exact: this lands at machine precision, not at the finite-difference error.
    assert end.disp_px == pytest.approx(0.0, abs=1e-13)
    assert result.residual <= 1e-12


# --------------------------------------------------------------------------
# N != M
# --------------------------------------------------------------------------


def test_underdetermined_takes_the_minimum_norm_step(ref: ReferenceParticle) -> None:
    """Three knobs for one target converge, moving the strengths as little as possible."""
    q1, q2, q3 = ThinQuadrupole(0.1), ThinQuadrupole(0.1), ThinQuadrupole(0.1)
    lattice = Lattice([Drift(1.0), q1, Drift(1.0), q2, Drift(1.0), q3, Drift(1.0)], ref=ref)
    start = _waist()
    at = len(lattice)
    knobs = [Knob([q1]), Knob([q2]), Knob([q3])]
    initial = np.array([k.value for k in knobs])

    result = match_insertion(lattice, [Target("alpha_x", at=at, value=0.0)], knobs, twiss0=start)
    assert propagate_twiss(lattice, start)[at].alpha_x == pytest.approx(0.0, abs=1e-11)
    assert len(result.values) == 3

    # Minimum-norm: no *smaller* move along the same solution surface. Compare
    # against the largest single-knob move that also lands on alpha = 0.
    moved = np.linalg.norm(np.array(result.values) - initial)
    assert moved > 0.0
    for j in range(3):
        alone = [Knob([q1]), Knob([q2]), Knob([q3])]
        for k, v0 in zip(alone, initial, strict=True):
            k.apply(float(v0))
        try:
            single = match_insertion(
                lattice, [Target("alpha_x", at=at, value=0.0)], [alone[j]], twiss0=start
            )
        except MatchingError:
            continue
        assert moved <= abs(single.values[0] - initial[j]) + 1e-12
    for k, v0 in zip(knobs, initial, strict=True):
        k.apply(float(v0))


def test_overdetermined_reports_the_least_squares_floor(ref: ReferenceParticle) -> None:
    """One knob cannot pick both a waist and an arbitrary beta*: that must raise.

    The quadratic has exactly two roots, so exactly two ``beta*`` are compatible
    with ``alpha* = 0``. Asking for anything between them is unreachable however
    the knob is set, and the matcher must say so rather than stop on the
    least-squares floor and call it a match.
    """
    _, roots, betas = _waist_roots()
    unreachable = 0.5 * (betas[0] + betas[1])
    assert min(betas) < unreachable < max(betas)
    lattice, quad = _waist_line(ref, roots[0] + 0.05)
    at = 3
    with pytest.raises(MatchingError) as exc:
        match_insertion(
            lattice,
            [Target("alpha_x", at=at, value=0.0), Target("beta_x", at=at, value=unreachable)],
            [Knob([quad])],
            twiss0=_waist(),
        )
    message = str(exc.value)
    assert "least-squares floor" in message or "did not converge" in message
    assert "beta_x@3" in message and "alpha_x@3" in message
    # Rollback: a failed match leaves the strength exactly as it found it.
    assert quad.k1l == pytest.approx(roots[0] + 0.05, abs=0.0)


def test_overdetermined_but_consistent_still_converges(ref: ReferenceParticle) -> None:
    """More targets than knobs is not itself a failure — only an inconsistent set is.

    ``(alpha* = 0, beta* = betas[1])`` is two targets for one knob, but the second
    waist root satisfies both exactly, so the matcher must find it rather than
    refuse on the count.
    """
    _, roots, betas = _waist_roots()
    lattice, quad = _waist_line(ref, roots[1] - 0.05)
    at = 3
    result = match_insertion(
        lattice,
        [Target("alpha_x", at=at, value=0.0), Target("beta_x", at=at, value=betas[1])],
        [Knob([quad])],
        twiss0=_waist(),
    )
    assert quad.k1l == pytest.approx(roots[1], abs=1e-10)
    assert result.residual <= 1e-12


# --------------------------------------------------------------------------
# Refusals, validation and rollback
# --------------------------------------------------------------------------


def test_sextupole_knob_is_refused_with_the_reason(ref: ReferenceParticle) -> None:
    """A sextupole's linear map is a drift, so it can never move a local beta."""
    sext = Sextupole(0.5, 1.0)
    lattice = Lattice([Drift(1.0), sext, Drift(1.0)], ref=ref)
    with pytest.raises(MatchingError, match="linear.*map is a drift"):
        match_insertion(
            lattice, [Target("beta_x", at=3, value=5.0)], [Knob([sext])], twiss0=_waist()
        )


def test_unknown_quantity_is_refused() -> None:
    with pytest.raises(MatchingError, match="unknown target quantity"):
        Target("beta_z", at=0, value=1.0)


def test_negative_or_out_of_range_boundary_is_refused(ref: ReferenceParticle) -> None:
    with pytest.raises(MatchingError, match="non-negative boundary index"):
        Target("beta_x", at=-1, value=1.0)
    lattice, quad = _waist_line(ref, 0.2)
    with pytest.raises(MatchingError, match="boundary points"):
        match_insertion(
            lattice, [Target("beta_x", at=4, value=5.0)], [Knob([quad])], twiss0=_waist()
        )


def test_non_positive_weight_is_refused() -> None:
    with pytest.raises(MatchingError, match="weight must be finite and positive"):
        Target("beta_x", at=0, value=1.0, weight=0.0)


def test_default_weight_is_scale_free_for_beta_and_finite_for_alpha() -> None:
    assert Target("beta_x", at=0, value=200.0).scale == pytest.approx(1.0 / 200.0)
    assert Target("alpha_x", at=0, value=0.0).scale == pytest.approx(1.0)
    assert Target("beta_x", at=0, value=200.0, weight=3.0).scale == pytest.approx(3.0)


def test_mixed_scale_targets_are_both_satisfied(ref: ReferenceParticle) -> None:
    """A beta ~ 80 m target must not swamp an alpha = 0 target in the 2-norm.

    Unweighted, ``|r|`` would be ~80 from the beta alone and the alpha constraint
    would be invisible until the very end; the default ``1/max(|value|, 1)``
    weights make both residuals order one.
    """
    beta0 = 120.0
    u1_true, u2_true, beta_star = _doublet_solution(beta0, -0.25, big=True)
    assert beta_star > 50.0  # the point of the test is a large-beta target
    q1, q2 = ThinQuadrupole(u1_true * 0.9), ThinQuadrupole(u2_true * 1.1)
    lattice = Lattice([Drift(DD1), q1, Drift(DD2), q2, Drift(DD3)], ref=ref)
    start = Twiss(0.0, beta0, 0.0, 0.0, beta0, 0.0, 0.0)
    at = len(lattice)

    result = match_insertion(
        lattice,
        [Target("beta_x", at=at, value=beta_star), Target("alpha_x", at=at, value=0.0)],
        [Knob([q1]), Knob([q2])],
        twiss0=start,
    )
    end = propagate_twiss(lattice, start)[at]
    assert end.beta_x == pytest.approx(beta_star, rel=1e-11)
    assert end.alpha_x == pytest.approx(0.0, abs=1e-10)
    assert q1.k1l == pytest.approx(u1_true, abs=1e-10)
    assert q2.k1l == pytest.approx(u2_true, abs=1e-10)
    assert result.residual <= 1e-12


def test_degenerate_knobs_are_refused(ref: ReferenceParticle) -> None:
    """Two knobs at identical optics move the observation point identically."""
    q1 = ThinQuadrupole(0.2)
    q2 = ThinQuadrupole(0.2)
    # q1 and q2 sit at the same place in the line (adjacent, zero length between),
    # so their columns are proportional.
    lattice = Lattice([Drift(D1), q1, q2, Drift(D2)], ref=ref)
    with pytest.raises(MatchingError, match="singular or ill-conditioned"):
        match_insertion(
            lattice,
            [Target("alpha_x", at=4, value=0.0), Target("beta_x", at=4, value=3.0)],
            [Knob([q1]), Knob([q2])],
            twiss0=_waist(),
        )


def test_no_knobs_or_no_targets_is_refused(ref: ReferenceParticle) -> None:
    lattice, quad = _waist_line(ref, 0.2)
    with pytest.raises(MatchingError, match="at least one knob"):
        match_insertion(lattice, [Target("alpha_x", at=3, value=0.0)], [], twiss0=_waist())
    with pytest.raises(MatchingError, match="at least one target"):
        match_insertion(lattice, [], [Knob([quad])], twiss0=_waist())


def test_unreachable_target_rolls_back(ref: ReferenceParticle) -> None:
    """A failed match restores every strength it touched."""
    lattice, quad = _waist_line(ref, 0.2)
    knob = Knob([quad])
    with pytest.raises(MatchingError):
        # beta at the exit has a positive minimum over u; ask for far less than it.
        match_insertion(lattice, [Target("beta_x", at=3, value=1e-6)], [knob], twiss0=_waist())
    assert quad.k1l == pytest.approx(0.2, abs=0.0)


def test_desynced_knob_is_caught_at_match_time(ref: ReferenceParticle) -> None:
    """A family silently un-ganged after construction must be refused, as in H1."""
    lattice, qfs, qd = _thin_fodo(ref)
    knob_f = Knob(list(qfs), [0.5, 0.5])
    qfs[1].k1l *= 1.5  # desync behind the knob's back
    with pytest.raises(MatchingError, match="not consistent with one shared value"):
        match_insertion(lattice, [Target("beta_x", at=0, value=5.0)], [knob_f, Knob([qd], [-1.0])])


def test_overlapping_knobs_are_refused(ref: ReferenceParticle) -> None:
    lattice, qfs, qd = _thin_fodo(ref)
    with pytest.raises(MatchingError, match="belongs to two knobs"):
        match_insertion(
            lattice,
            [Target("beta_x", at=0, value=5.0)],
            [Knob([qfs[0], qfs[1]], [0.5, 0.5]), Knob([qfs[0]])],
        )


def test_unstable_start_is_reported_as_such(ref: ReferenceParticle) -> None:
    """The periodic branch refuses a lattice with no closed solution to start from."""
    lattice, qfs, qd = _thin_fodo(ref, vf=8.0, vd=8.0)
    knobs = [Knob(list(qfs), [0.5, 0.5]), Knob([qd], [-1.0])]
    with pytest.raises(MatchingError, match="no matched optics"):
        match_insertion(lattice, [Target("beta_x", at=0, value=5.0)], knobs)


def test_backtracking_crosses_the_stability_boundary(ref: ReferenceParticle) -> None:
    """Count the unstable excursions a long first step takes, then assert one happened.

    Measured rather than assumed: most starts never trip it, so a gate that only
    asserted convergence would pass without ever exercising the backtracking.
    """
    lattice, qfs, qd = _thin_fodo(ref)
    knobs = [Knob(list(qfs), [0.5, 0.5]), Knob([qd], [-1.0])]
    want = 12.0  # nominal is ~4.0 / ~2.5, so the first step is a long one
    target = [Target("beta_x", at=0, value=want), Target("beta_y", at=0, value=want)]

    # Probe: walk the first Newton direction and count how many trial points along
    # it have no closed solution at all.
    jac = insertion_response_matrix(lattice, target, knobs)
    weight = np.array([t.scale for t in target])
    start = propagate_twiss(lattice, closed_twiss(lattice))[0]
    resid = weight * np.array([start.beta_x - want, start.beta_y - want])
    step = np.linalg.lstsq(weight[:, None] * jac, -resid, rcond=None)[0]

    snapshots = [k.snapshot() for k in knobs]
    v0 = np.array([k.value for k in knobs])
    unstable, lam = 0, 1.0
    for _ in range(20):
        for k, value in zip(knobs, v0 + lam * step, strict=True):
            k.apply(float(value))
        try:
            closed_twiss(lattice)
        except UnstableLatticeError:
            unstable += 1
        lam *= 0.5
    for k, snap in zip(knobs, snapshots, strict=True):
        k.restore(snap)
    assert unstable > 0, "this start never leaves the stable region; pick a harder one"

    # And with the backtracking in place the match still succeeds from that start.
    result = match_insertion(lattice, target, knobs)
    assert result.residual <= 1e-12
    assert result.iterations > 0
    after = closed_twiss(lattice)
    assert after.beta_x == pytest.approx(want, rel=1e-11)
    assert after.beta_y == pytest.approx(want, rel=1e-11)


# --------------------------------------------------------------------------
# The finite-difference fallbacks at the stability boundary
# --------------------------------------------------------------------------
#
# These three branches are invisible to every other gate in this file, and for a
# structural reason worth stating. H2 uses the H1 pattern -- approximate Jacobian,
# exact residual -- so the converged fixed point is exact no matter how wrong the
# Jacobian is; a halved column would only cost iterations. So the convergence
# gates cannot see a denominator bug, and the one gate that pins the Jacobian
# numerically runs on a comfortably stable lattice where both trial points
# succeed. Hence a dedicated gate that drives a trial point across the boundary.

# The thin FODO's horizontal stability limit, trace = 2, sits at vd = 1 exactly;
# the test asserts that rather than trusting it, so the constant is not magic.
VD_CRIT = 1.0
# Sit inside the boundary by less than one FD step, so v +- h straddles it.
VD_MARGIN = 5e-7


def _beta_x_at_entrance(ref: ReferenceParticle, vd: float) -> float:
    """``beta_x`` at boundary 0 of a fresh thin FODO with this defocusing strength."""
    lattice, _, _ = _thin_fodo(ref, vd=vd)
    return propagate_twiss(lattice, closed_twiss(lattice))[0].beta_x


def test_the_thin_fodo_stability_limit_is_where_the_gate_places_it(
    ref: ReferenceParticle,
) -> None:
    """``VD_CRIT`` is the real boundary: trace = 2 exactly, stable below, not above."""
    lattice, _, _ = _thin_fodo(ref, vd=VD_CRIT)
    assert np.trace(lattice.one_turn_matrix()[:2, :2]) == 2.0

    _beta_x_at_entrance(ref, VD_CRIT - VD_MARGIN)  # stable side: must not raise
    with pytest.raises(UnstableLatticeError):
        _beta_x_at_entrance(ref, VD_CRIT + VD_MARGIN)


@pytest.mark.parametrize("weight", [-1.0, +1.0])
def test_one_sided_difference_when_a_trial_point_is_unstable(
    ref: ReferenceParticle, weight: float
) -> None:
    """A trial point off the stability edge falls back to a quotient over ``h``.

    The knob weight flips which side dies -- ``vd = -weight * v``, so ``weight =
    -1`` loses the ``+h`` point and ``weight = +1`` loses the ``-h`` point. Both
    one-sided branches are covered by the parametrisation.

    The column is asserted against the exact one-sided quotient rebuilt from the
    public API, *not* against a finer central difference: right at the boundary
    ``beta`` diverges, and the one-sided truncation error was measured at 58-86%
    -- larger than the factor of two a ``2h`` denominator would introduce, so a
    comparison against a finer difference could not tell the bug from the
    truncation. The exact quotient agrees bit for bit and leaves no such room.
    """
    fd_step = 1e-6
    vd0 = VD_CRIT - VD_MARGIN
    lattice, _, qd = _thin_fodo(ref, vd=vd0)
    knobs = [Knob([qd], [weight])]
    targets = [Target("beta_x", at=0, value=1.0)]

    v = knobs[0].value
    h = fd_step * max(abs(v), 1.0)
    assert h > VD_MARGIN, "the step must reach past the boundary or nothing is tested"

    # Non-vacuity: one trial point really is unstable and the other really is not.
    # Both weights lose the *same* physical point -- vd0 + h is past the limit --
    # but it is reached from opposite signs of v, which is what makes this cover
    # both branches rather than the same one twice.
    with pytest.raises(UnstableLatticeError):
        _beta_x_at_entrance(ref, vd0 + h)
    _beta_x_at_entrance(ref, vd0 - h)

    jac = insertion_response_matrix(lattice, targets, knobs, fd_step=fd_step)

    base = _beta_x_at_entrance(ref, vd0)
    other = _beta_x_at_entrance(ref, vd0 - h)
    expected = (base - other) / h if weight < 0 else (other - base) / h
    assert jac[0, 0] == pytest.approx(expected, rel=1e-14)
    # And explicitly: a 2h denominator would read half of this.
    assert jac[0, 0] != pytest.approx(0.5 * expected, rel=1e-3)

    assert qd.k1l == -vd0, "the knob must come back exactly, boundary or not"


def test_both_trial_points_unstable_is_reported_not_guessed(
    ref: ReferenceParticle,
) -> None:
    """When the stable window is narrower than the step, say so and touch nothing.

    Reached here with a deliberately huge ``fd_step`` rather than a knife-edge
    knob, which is also why the message does not claim the knob sits *on* the
    boundary -- that conclusion does not follow from what the code can see.
    """
    vd0 = VD_CRIT - VD_MARGIN
    lattice, _, qd = _thin_fodo(ref, vd=vd0)
    knobs = [Knob([qd], [-1.0])]
    targets = [Target("beta_x", at=0, value=1.0)]

    with pytest.raises(MatchingError, match="unstable on both sides"):
        insertion_response_matrix(lattice, targets, knobs, fd_step=1.0)
    assert qd.k1l == -vd0
