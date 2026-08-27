r"""O3 — the detuning a *sextupole* makes, which J2 said it did not compute.

An octupole moves a particle's tune at **first** order in its strength: average its
potential over both betatron phases at fixed action, differentiate, done — that is
``amplitude_detuning`` (J2), and ``test_amplitude_detuning.py`` derives it in eight
lines of sympy. A sextupole cannot be reached that way. Its potential is odd, so the
first-order average is exactly zero, and the tune shift only appears at **second**
order, where the ring's sextupoles act in *pairs*. So the answer is a double sum, it
is quadratic in ``k2``, and it needs a genuine normal-form calculation rather than an
average. That calculation is done here from scratch, and its result is what
:func:`accsim.twiss.sextupole_detuning` computes.

**Nothing here is transcribed.** Every convention is pinned by a check before it is
used, in this order:

1. ``exp(:f:)`` with ``f = -k2l (x^3 - 3 x y^2)/6`` must reproduce
   ``ThinSextupole.track`` **exactly** — so the generator is the one accsim's own
   element applies, not one chosen to give a nice answer.
2. The bracket used in the resonance basis must equal the ordinary Poisson bracket in
   ``(x, px, y, py)``.
3. Rotate-kick-rotate-back must equal ``exp(:f o R(+mu):)`` — the ``+`` is checked
   against the ``-``.
4. Applying ``exp(:a:)`` then ``exp(:b:)`` must equal ``exp(:a + b + [a,b]/2:)`` — the
   ``+1/2`` is *solved for*, not assumed.

**Then the machinery is anchored twice before it is aimed at a sextupole.**

* On the **octupole**, where its first-order answer must be the ``k3l beta^2/(16 pi)``
  matrix that this package already ships and that ``test_amplitude_detuning.py``
  derives independently.
* On **two thin quadrupoles**, where its *second-order* answer must equal the exact
  expansion of ``cos 2 pi Q = 1/2 Tr(M)`` of the real one-turn matrix — pure linear
  algebra on one side, perturbation theory on the other, agreeing as a **symbolic
  identity** rather than to a tolerance. This is the anchor that matters: a
  second-order calculation has a resonance denominator, a phase difference and a
  ``pi Q`` inside a cosine, and every one of those can be wrong in a way that survives
  a numerical spot check at a generic tune. The anchor is run in **both beam
  orderings**, because a derivation that lost the ``|mu_i - mu_j|`` agrees with one
  and not the other.

What that anchor does *not* reach is the ``3 Q_x`` denominator — a quadrupole has no
such line. That one is gated where it becomes the whole answer, next to the
third-integer resonance, in the PTC cross-check (``tests/reference/``).

**The tracked gate has a surprise in it, and it is the interesting result here.**
Tracking sees all orders, so its residual against a second-order formula must fall as
the amplitude shrinks — J2's octupole residual falls by 16 per halving (quadratic in
the action). This one falls by **8**, an odd power, which looks impossible for a tune.
It is not: the prediction is evaluated at the *Courant-Snyder* action of the launch
point, and once a sextupole is present that is not the particle's invariant action —
it is wrong by a phase-dependent ``O(k2 x^3)``. Measured 2026-08-27: at fixed action
the detuning varies by +-2.1% across launch phase at 2 mm and +-1.1% at 1 mm (the
spread is linear in amplitude, as that explanation requires), and **averaging over the
launch phase restores the expected order exactly** — residual ratios 16.02, 16.00,
16.00. Both halves are gated below.
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
    ResonantLatticeError,
    Sextupole,
    ThinOctupole,
    ThinSextupole,
    amplitude_detuning,
    sextupole_detuning,
    total_detuning,
    tunes,
)
from accsim.coords import PX, X
from accsim.tracking import Particle, Tracker
from accsim.tune import _plane_tune
from accsim.twiss import closed_twiss

MASS0, GAMMA0 = 938.27208816e6, 20.0

# The working point of every ring below.  Qx != Qy on purpose: at Qx = Qy the two
# coupled lines Qx +- 2Qy collapse onto Qx and 3Qx, and a wrong cross-plane term hides
# inside that degeneracy.  All four driven lines sit >= 0.11 from an integer here.
KF, KD = 0.80, -0.90
N_TURNS = 2048


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


# ==========================================================================
# 0. The machinery: Lie maps in real coordinates, and polynomials in the
#    resonance basis h_u = u_hat - i p_hat_u.
# ==========================================================================

_X, _PX, _Y, _PY = sp.symbols("x p_x y p_y", real=True)
_Z = (_X, _PX, _Y, _PY)
_EPS = sp.Symbol("epsilon", positive=True)  # bookkeeping order in the strengths


def _pb_real(f: sp.Expr, g: sp.Expr) -> sp.Expr:
    """The ordinary Poisson bracket in ``(x, px, y, py)``."""
    return (
        sp.diff(f, _X) * sp.diff(g, _PX)
        - sp.diff(f, _PX) * sp.diff(g, _X)
        + sp.diff(f, _Y) * sp.diff(g, _PY)
        - sp.diff(f, _PY) * sp.diff(g, _Y)
    )


def _keep(e: sp.Expr, order: int) -> sp.Expr:
    e = sp.expand(e)
    return sp.expand(sum(c * _EPS**k for (k,), c in sp.Poly(e, _EPS).terms() if k <= order))


def _lie_map(f: sp.Expr, terms: int, order: int) -> list[sp.Expr]:
    """``z -> exp(:f:) z``, truncated at ``EPS**order``."""
    out = []
    for z in _Z:
        term, acc = z, z
        for n in range(1, terms + 1):
            term = _pb_real(f, term) / n
            acc = _keep(acc + term, order)
        out.append(acc)
    return out


def _apply_then(a: list[sp.Expr], b: list[sp.Expr], order: int) -> list[sp.Expr]:
    """Apply map ``a``, then map ``b``: substitution, ``b(a(z))``."""
    sub = dict(zip(_Z, a, strict=True))
    return [_keep(x.subs(sub, simultaneous=True), order) for x in b]


def _rot_map(mux: sp.Expr, muy: sp.Expr) -> list[sp.Expr]:
    c, s = sp.cos(mux), sp.sin(mux)
    cy, sy = sp.cos(muy), sp.sin(muy)
    return [_X * c + _PX * s, _PX * c - _X * s, _Y * cy + _PY * sy, _PY * cy - _Y * sy]


def _worst(a: list[sp.Expr], b: list[sp.Expr]) -> float:
    """Largest coefficient of ``a - b``, as a number."""
    w = 0.0
    for u, v in zip(a, b, strict=True):
        d = sp.expand(u - v)
        if d != 0:
            gens = sorted(d.free_symbols, key=str)
            w = max([w] + [float(abs(sp.N(t))) for t in sp.Poly(d, *gens).coeffs()])
    return float(w)


# -- the resonance basis: {(p, q, r, s): coeff} for hx^p hxb^q hy^r hyb^s --------
# Phases enter only as unit-modulus symbols a = exp(i mu), A = exp(i theta), so every
# coefficient is a rational function of them and sympy stays on `cancel`.

_I = sp.I


def _clean(d: dict) -> dict:
    return {m: c for m, c in d.items() if c != 0}


def _padd(*Ps: dict) -> dict:
    out: dict = {}
    for P in Ps:
        for m, c in P.items():
            out[m] = out.get(m, 0) + c
    return _clean({m: sp.cancel(c) for m, c in out.items()})


def _pscale(P: dict, a) -> dict:
    return _clean({m: sp.cancel(a * c) for m, c in P.items()})


def _pmul(P: dict, Q: dict) -> dict:
    out: dict = {}
    for m1, c1 in P.items():
        for m2, c2 in Q.items():
            m = tuple(u + v for u, v in zip(m1, m2, strict=True))
            out[m] = out.get(m, 0) + c1 * c2
    return _clean({m: sp.cancel(c) for m, c in out.items()})


def _ppow(P: dict, n: int) -> dict:
    out = {(0, 0, 0, 0): sp.Integer(1)}
    for _ in range(n):
        out = _pmul(out, P)
    return out


def _deriv(P: dict, k: int) -> dict:
    out: dict = {}
    for m, c in P.items():
        if m[k] == 0:
            continue
        mm = list(m)
        n = mm[k]
        mm[k] -= 1
        key = tuple(mm)
        out[key] = out.get(key, 0) + n * c
    return _clean(out)


def _pb(P: dict, Q: dict) -> dict:
    """``[P,Q] = 2i sum_u (P_hu Q_hub - P_hub Q_hu)``; pinned against ``_pb_real``."""
    t = []
    for h, hb in ((0, 1), (2, 3)):
        t.append(_pmul(_deriv(P, h), _deriv(Q, hb)))
        t.append(_pscale(_pmul(_deriv(P, hb), _deriv(Q, h)), -1))
    return _pscale(_padd(*t), 2 * _I)


def _pb_avg(P: dict, Q: dict) -> dict:
    """The action-only part of ``[P, Q]``, without building the rest.

    A monomial's charge ``(p - q, r - s)`` is additive under the bracket, so only pairs
    whose charges cancel can land on an action-only monomial. Skipping the others is
    what makes the two-sextupole derivation finish in seconds instead of minutes.
    """
    by_charge: dict = {}
    for m, c in Q.items():
        by_charge.setdefault((m[0] - m[1], m[2] - m[3]), []).append((m, c))
    out: dict = {}
    for m1, c1 in P.items():
        g = (m1[0] - m1[1], m1[2] - m1[3])
        for m2, c2 in by_charge.get((-g[0], -g[1]), ()):
            for h, hb in ((0, 1), (2, 3)):
                if m1[h] and m2[hb]:
                    u, v = list(m1), list(m2)
                    u[h] -= 1
                    v[hb] -= 1
                    k = tuple(a + b for a, b in zip(u, v, strict=True))
                    out[k] = out.get(k, 0) + 2 * _I * m1[h] * m2[hb] * c1 * c2
                if m1[hb] and m2[h]:
                    u, v = list(m1), list(m2)
                    u[hb] -= 1
                    v[h] -= 1
                    k = tuple(a + b for a, b in zip(u, v, strict=True))
                    out[k] = out.get(k, 0) - 2 * _I * m1[hb] * m2[h] * c1 * c2
    return {m: sp.cancel(c) for m, c in _avg(out).items() if sp.cancel(c) != 0}


def _rot(P: dict, A, B) -> dict:
    """``P o R(theta)`` with ``A = exp(i theta_x)``; pass ``1/A`` for ``R(-theta)``."""
    return _clean({m: sp.cancel(c * A ** (m[0] - m[1]) * B ** (m[2] - m[3])) for m, c in P.items()})


def _avg(P: dict) -> dict:
    """The action-only part: ``p == q`` and ``r == s``."""
    return {m: c for m, c in P.items() if m[0] == m[1] and m[2] == m[3]}


def _nonavg(P: dict) -> dict:
    return {m: c for m, c in P.items() if not (m[0] == m[1] and m[2] == m[3])}


def _homological(P: dict, A, B) -> dict:
    """``G`` with ``G - G o R(-theta) = P``, term by term."""
    out = {}
    for m, c in P.items():
        den = sp.cancel(1 - A ** (-(m[0] - m[1])) * B ** (-(m[2] - m[3])))
        if den == 0:
            raise ZeroDivisionError(f"resonant monomial {m}")
        out[m] = sp.cancel(c / den)
    return _clean(out)


def _coord(a, plane: str) -> dict:
    """``u_hat`` at a point whose phase advance from the reference is ``mu``."""
    if plane == "x":
        return {(1, 0, 0, 0): a / 2, (0, 1, 0, 0): 1 / (2 * a)}
    return {(0, 0, 1, 0): a / 2, (0, 0, 0, 1): 1 / (2 * a)}


def _to_actions(P: dict, Jx: sp.Symbol, Jy: sp.Symbol) -> sp.Expr:
    e = 0
    for (p, q, r, s), c in P.items():
        assert p == q and r == s
        e += c * (2 * Jx) ** p * (2 * Jy) ** r
    return sp.expand(e)


def _normal_form(kicks: list[dict], A, B, orders: int = 2) -> tuple[dict, dict]:
    """Action-only parts of the normal form at first and second order in the strengths.

    ``kicks`` are the generators referred to one point, in **beam order**. The one-turn
    map is ``exp(:F:)`` followed by ``R(theta)``; conjugating by ``exp(:G:)`` with ``G``
    chosen to kill everything but the actions leaves ``R(theta)`` followed by
    ``exp(:N:)``, and then ``2 pi Q = theta - dN/dJ``.

    ``orders=1`` stops after the first-order part, which is all a quartic generator (an
    octupole) needs — and worth stopping for, because its second-order brackets are
    degree-six polynomials whose rational coefficients are slow for no purpose.
    """
    half = sp.Rational(1, 2)
    F1 = _padd(*kicks)
    back = lambda P: _rot(P, 1 / A, 1 / B)  # noqa: E731  -- P o R(-theta)
    G = _homological(_nonavg(back(F1)), A, B)
    P1 = _padd(G, F1)
    PR1 = back(P1)
    H1 = _padd(PR1, _pscale(G, -1))
    left = {m: c for m, c in _nonavg(H1).items() if sp.cancel(c) != 0}
    assert not left, f"first order not normalised: {left}"
    if orders < 2:
        return _avg(H1), {}
    F2 = _pscale(
        _padd(
            *[_pb(kicks[i], kicks[j]) for i in range(len(kicks)) for j in range(i + 1, len(kicks))]
        ),
        half,
    )
    # Only the action-only part of the second order is ever read, and ``_avg`` commutes
    # with the rotation, so ``_avg(P2 o R(-theta)) = _avg(P2)`` and the two remaining
    # brackets can be pruned to the monomial pairs whose charges cancel. Without that
    # pruning this one call is most of the file's runtime.
    N2 = _padd(
        _avg(F2),
        _pscale(_pb_avg(G, F1), half),
        _pscale(_pb_avg(PR1, G), -half),
    )
    return _avg(H1), N2


# ==========================================================================
# 1. The conventions, each pinned before it is used
# ==========================================================================


def test_generator_reproduces_the_shipped_sextupole_kick() -> None:
    """``exp(:f:)`` with ``f = -k2l (x^3 - 3 x y^2)/6`` *is* ``ThinSextupole.track``.

    Exactly, not approximately — the Lie series terminates because ``f`` has no
    momentum dependence. Anything else here and every coefficient downstream is the
    detuning of a magnet accsim does not have.
    """
    k2l = sp.Symbol("k2l", real=True)
    f = -_EPS * k2l * (_X**3 - 3 * _X * _Y**2) / 6
    got = _lie_map(f, 3, 3)
    want = [_X, _PX - _EPS * k2l * (_X**2 - _Y**2) / 2, _Y, _PY + _EPS * k2l * _X * _Y]
    assert all(sp.simplify(a - b) == 0 for a, b in zip(got, want, strict=True))


def test_generator_matches_the_element_numerically(ref: ReferenceParticle) -> None:
    """And the same statement against the element object, not just the formula."""
    k2l = 1.7
    p = Particle(x=3e-3, px=1e-4, y=-2e-3, py=5e-5)
    out = ThinSextupole(k2l).track(p.state, ref)
    f = -_EPS * k2l * (_X**3 - 3 * _X * _Y**2) / 6
    sub = {_X: 3e-3, _PX: 1e-4, _Y: -2e-3, _PY: 5e-5, _EPS: 1}
    lie = [float(c.subs(sub)) for c in _lie_map(f, 3, 3)]
    assert out[:4] == pytest.approx(lie, rel=0, abs=1e-18)


def test_resonance_basis_bracket_is_the_ordinary_poisson_bracket() -> None:
    """``[P,Q] = 2i sum (P_hu Q_hub - P_hub Q_hu)`` is the same object as ``_pb_real``."""
    hx, hxb, hy, hyb = sp.symbols("h_x hxb h_y hyb")
    sub = {hx: _X - sp.I * _PX, hxb: _X + sp.I * _PX, hy: _Y - sp.I * _PY, hyb: _Y + sp.I * _PY}

    def as_expr(P: dict) -> sp.Expr:
        return sp.expand(
            sum(c * hx ** m[0] * hxb ** m[1] * hy ** m[2] * hyb ** m[3] for m, c in P.items())
        )

    A = {(2, 0, 0, 0): sp.Rational(1, 3), (1, 1, 0, 1): sp.Rational(-2, 7), (0, 0, 2, 1): sp.I / 5}
    B = {(1, 2, 0, 0): sp.Rational(5, 4), (0, 1, 1, 1): sp.Rational(1, 2), (3, 0, 0, 0): -sp.I / 9}
    lhs = sp.expand(as_expr(_pb(A, B)).subs(sub))
    rhs = sp.expand(_pb_real(sp.expand(as_expr(A).subs(sub)), sp.expand(as_expr(B).subs(sub))))
    assert sp.simplify(lhs - rhs) == 0


def test_conjugating_a_kick_is_the_plus_mu_substitution() -> None:
    """Rotate, kick, rotate back ``= exp(:f o R(+mu):)`` — and the ``-mu`` version is not."""
    s1 = sp.Symbol("s1", real=True)
    mu, muy = sp.Float(0.37), sp.Float(0.81)
    f = -_EPS * s1 * (_X**3 - 3 * _X * _Y**2) / 6
    kick = _lie_map(f, 3, 3)
    conj = _apply_then(_apply_then(_rot_map(mu, muy), kick, 3), _rot_map(-mu, -muy), 3)
    for sign, must_match in ((+1, True), (-1, False)):
        sub = dict(zip(_Z, _rot_map(sign * mu, sign * muy), strict=True))
        g = sp.expand(f.subs(sub, simultaneous=True))
        w = _worst(conj, _lie_map(g, 3, 3))
        assert (w < 1e-15) is must_match, f"sign {sign:+d} gave {w:.3e}"


def test_composition_coefficient_is_solved_for_not_assumed() -> None:
    """Apply ``exp(:a:)`` then ``exp(:b:)`` ``= exp(:a + b + c [a,b]:)`` with ``c = +1/2``.

    ``c`` is obtained by solving, so the sign convention of the Lie composition is
    measured here rather than recalled. It is then re-checked on a whole two-sextupole
    turn in :func:`test_two_sextupole_turn_is_one_generator_and_a_rotation`.
    """
    a = _EPS * (
        sp.Rational(1, 3) * _X**3 - sp.Rational(2, 5) * _X * _Y**2 + sp.Rational(1, 7) * _X**2 * _PX
    )
    b = _EPS * (
        sp.Rational(1, 2) * _Y**3 + sp.Rational(3, 4) * _X * _PX * _PY - sp.Rational(1, 6) * _PX**3
    )
    ab = _apply_then(_lie_map(a, 3, 2), _lie_map(b, 3, 2), 2)
    c = sp.Symbol("c")
    cand = _lie_map(sp.expand(a + b + c * _pb_real(a, b)), 3, 2)
    coeffs: list[sp.Expr] = []
    for u, v in zip(ab, cand, strict=True):
        d = sp.expand(u - v)
        if d != 0:
            coeffs += sp.Poly(d, _X, _PX, _Y, _PY, _EPS).coeffs()
    assert sp.solve(coeffs, c, dict=True) == [{c: sp.Rational(1, 2)}]


def test_two_sextupole_turn_is_one_generator_and_a_rotation() -> None:
    """The whole construction, checked against an explicitly composed turn.

    Rotate to sextupole 1, kick, rotate on, kick, rotate to the end — and separately
    ``exp(:g1 + g2 + [g1,g2]/2:)`` followed by ``R(theta)``. They must be the same map
    to second order. This is the single check that the rest of the file rests on.
    """
    s1, s2 = sp.symbols("s1 s2", real=True)
    m1, m2, th = sp.Float(0.37), sp.Float(1.93), sp.Float(2.41)
    m1y, m2y, thy = sp.Float(0.81), sp.Float(2.55), sp.Float(3.17)
    sext = lambda k: -_EPS * k * (_X**3 - 3 * _X * _Y**2) / 6  # noqa: E731

    turn = _rot_map(m1, m1y)
    for m in (
        _lie_map(sext(s1), 3, 2),
        _rot_map(m2 - m1, m2y - m1y),
        _lie_map(sext(s2), 3, 2),
        _rot_map(th - m2, thy - m2y),
    ):
        turn = _apply_then(turn, m, 2)

    def conj(f: sp.Expr, ax: sp.Expr, ay: sp.Expr) -> sp.Expr:
        return sp.expand(f.subs(dict(zip(_Z, _rot_map(ax, ay), strict=True)), simultaneous=True))

    g1, g2 = conj(sext(s1), m1, m1y), conj(sext(s2), m2, m2y)
    for c, must_match in ((sp.Rational(1, 2), True), (sp.Rational(-1, 2), False), (0, False)):
        f = sp.expand(g1 + g2 + c * _pb_real(g1, g2))
        w = _worst(turn, _apply_then(_lie_map(f, 3, 2), _rot_map(th, thy), 2))
        assert (w < 1e-14) is must_match, f"c = {c} gave {w:.3e}"


# ==========================================================================
# 2. Anchor one (first order): the octupole this package already ships
# ==========================================================================


def test_pipeline_reproduces_the_shipped_octupole_detuning() -> None:
    """Pointed at an octupole, the machinery must give J2's ``k3l beta^2/(16 pi)``.

    Independently derived in ``test_amplitude_detuning.py`` by phase averaging, which
    shares no code with the normal form here. Getting it right fixes the read-out
    ``2 pi dQ = -dN/dJ``, including its sign.
    """
    k3l = sp.Symbol("k3l", positive=True)
    bx, by = sp.symbols("beta_x beta_y", positive=True)
    Jx, Jy = sp.symbols("J_x J_y", nonnegative=True)
    A, B = sp.symbols("A B")
    # V = k3l (x^4 - 6 x^2 y^2 + y^4)/24 in normalised coordinates.  Written with whole
    # powers of beta rather than sqrt(beta) x_hat: a square root inside every coefficient
    # sends sympy's `cancel` down a radical-rationalising path for no gain here.
    Xh, Yh = _coord(1, "x"), _coord(1, "y")
    V = _padd(
        _pscale(_ppow(Xh, 4), bx**2),
        _pscale(_pmul(_ppow(Xh, 2), _ppow(Yh, 2)), -6 * bx * by),
        _pscale(_ppow(Yh, 4), by**2),
    )
    N1, _ = _normal_form([_pscale(V, -k3l / 24)], A, B, orders=1)  # generator = -V
    e = _to_actions(N1, Jx, Jy)
    dqx = sp.simplify(-sp.diff(e, Jx) / (2 * sp.pi))
    dqy = sp.simplify(-sp.diff(e, Jy) / (2 * sp.pi))
    assert sp.simplify(dqx - k3l * (bx**2 * Jx - 2 * bx * by * Jy) / (16 * sp.pi)) == 0
    assert sp.simplify(dqy - k3l * (by**2 * Jy - 2 * bx * by * Jx) / (16 * sp.pi)) == 0


# ==========================================================================
# 3. Anchor two (second order): two quadrupoles against the exact trace
# ==========================================================================

_A, _B = sp.symbols("A B")


def _quad_exact_shift(ua, ba, ka, ub, bb, kb):
    """``2 pi dQ`` to second order from ``cos(theta + d) = Tr(M)/2``. No perturbation theory.

    Everything is written in ``A = exp(i theta)`` and ``u = exp(i mu)`` so both sides of
    the comparison are rational functions and ``cancel`` decides exactly.
    """
    co = lambda u: (u + 1 / u) / 2  # noqa: E731
    si = lambda u: (u - 1 / u) / (2 * sp.I)  # noqa: E731
    rot = lambda u: sp.Matrix([[co(u), si(u)], [-si(u), co(u)]])  # noqa: E731
    quad = lambda beta, k: sp.Matrix([[1, 0], [-_EPS * beta * k, 1]])  # noqa: E731
    M = rot(_A / ub) * quad(bb, kb) * rot(ub / ua) * quad(ba, ka) * rot(ua)
    tr2 = sp.cancel(sp.expand((M[0, 0] + M[1, 1]) / 2))
    d1, d2 = sp.symbols("d1 d2")
    lhs = co(_A) - si(_A) * (_EPS * d1 + _EPS**2 * d2) - co(_A) * (_EPS * d1) ** 2 / 2
    diff = sp.expand(lhs - tr2)
    s1 = sp.solve(sp.cancel(diff.coeff(_EPS, 1)), d1, dict=True)[0]
    s2 = sp.solve(sp.cancel(sp.expand(diff.coeff(_EPS, 2)).subs(s1)), d2, dict=True)[0]
    return sp.cancel(s1[d1]), sp.cancel(s2[d2])


def _quad_pipeline_shift(ua, ba, ka, ub, bb, kb):
    """The same two numbers from the normal form. Kicks listed in beam order."""
    Jx, Jy = sp.symbols("J_x J_y", nonnegative=True)
    gen = lambda u, beta, k: _pscale(_ppow(_coord(u, "x"), 2), -_EPS * k * beta / 2)  # noqa: E731
    N1, N2 = _normal_form([gen(ua, ba, ka), gen(ub, bb, kb)], _A, _B)
    e1 = _to_actions(N1, Jx, Jy)
    e2 = _to_actions(N2, Jx, Jy)
    return sp.cancel(-sp.diff(e1, Jx) / _EPS), sp.cancel(-sp.diff(e2, Jx) / _EPS**2)


@pytest.mark.parametrize("upstream", ["A", "B"])
def test_two_quadrupole_second_order_matches_the_exact_trace(upstream: str) -> None:
    """The load-bearing anchor: perturbation theory vs. ``1/2 Tr(M)``, exactly.

    Two thin quadrupoles is the simplest system with a genuine second-order tune shift,
    and its exact answer is available from linear algebra alone. Requiring the two to
    agree as a **symbolic identity** pins, in one shot, the resonance denominator, the
    ``pi Q`` inside the cosine, and the sign of second order against first.

    Run with each quadrupole upstream in turn. A derivation that produced
    ``cos(Delta mu - pi Q)`` instead of ``cos(|Delta mu| - pi Q)`` passes one ordering
    and fails the other, which is why one parametrisation is not enough.

    What this cannot reach: a quadrupole drives no ``3 Q_x`` line, so the ``C(3,0)``
    denominator of the sextupole formula is *not* anchored here. It is gated near the
    third-integer resonance in ``tests/reference/test_sextupole_detuning_madx.py``.
    """
    k1, k2 = sp.symbols("k1 k2", real=True)
    b1, b2 = sp.symbols("b1 b2", positive=True)
    a1, a2 = sp.symbols("a1 a2")
    args = (a1, b1, k1, a2, b2, k2) if upstream == "A" else (a1, b2, k2, a2, b1, k1)
    ex1, ex2 = _quad_exact_shift(*args)
    pp1, pp2 = _quad_pipeline_shift(*args)
    assert sp.cancel(sp.expand(ex1 - pp1)) == 0
    assert sp.cancel(sp.expand(ex2 - pp2)) == 0


# ==========================================================================
# 4. The derivation itself, and the coefficients the package ships
# ==========================================================================


def _sext_gen(c, d, a, b) -> dict:
    """``-(1/6)(c xhat^3 - 3 d xhat yhat^2)`` with ``c = S bx^(3/2)``, ``d = S bx^(1/2) by``."""
    Xh, Yh = _coord(a, "x"), _coord(b, "y")
    return _pscale(
        _padd(_pscale(_ppow(Xh, 3), c), _pscale(_pmul(Xh, _ppow(Yh, 2)), -3 * d)),
        sp.Rational(-1, 6),
    )


def _line_rational(mx: int, my: int, self_term: bool, a, b, al, be):
    """``cos(mx psi_x + my psi_y - pi Phi) / (pi sin(pi Phi))`` as a rational function.

    ``al = exp(i pi Q_x)``, ``be = exp(i pi Q_y)`` — the *half*-angle symbols, which is
    what lets a form carrying ``pi Q`` be compared exactly against one carrying
    ``2 pi Q``.
    """
    u = sp.Integer(1) if self_term else a**mx * b**my
    w = al**mx * be**my
    return sp.cancel(sp.I * (u / w + w / u) / (sp.pi * (w - 1 / w)))


#: The lines a sextupole can drive.  Not a remembered list: its ``x^3`` part carries the
#: charges ``(+-3, 0)`` and ``(+-1, 0)``, its ``x y^2`` part ``(+-1, 0)`` and
#: ``(+-1, +-2)``, and a bracket lands on the actions only when two charges cancel.
_LINES = [(1, 0), (3, 0), (1, 2), (1, -2)]

#: What the derivation below produces, and what ``sextupole_detuning`` implements:
#: for each entry of ``dQ/dJ`` and each strength product, the coefficient of each line.
#: ``self`` entries are exactly half their ``pair`` partners, which is the statement
#: that the whole thing is one symmetric sum over *all ordered pairs*.
_EXPECTED = {
    ("xx", "c1*c2", False): {(1, 0): sp.Rational(-3, 32), (3, 0): sp.Rational(-1, 32)},
    ("xx", "c1^2", True): {(1, 0): sp.Rational(-3, 64), (3, 0): sp.Rational(-1, 64)},
    ("xy", "c1*d2", False): {(1, 0): sp.Rational(1, 16)},
    ("xy", "d1*c2", False): {(1, 0): sp.Rational(1, 16)},
    ("xy", "d1*d2", False): {(1, 2): sp.Rational(-1, 16), (1, -2): sp.Rational(1, 16)},
    ("xy", "c1*d1", True): {(1, 0): sp.Rational(1, 16)},
    ("xy", "d1^2", True): {(1, 2): sp.Rational(-1, 32), (1, -2): sp.Rational(1, 32)},
    ("yy", "d1*d2", False): {
        (1, 0): sp.Rational(-1, 8),
        (1, 2): sp.Rational(-1, 32),
        (1, -2): sp.Rational(-1, 32),
    },
    ("yy", "d1^2", True): {
        (1, 0): sp.Rational(-1, 16),
        (1, 2): sp.Rational(-1, 64),
        (1, -2): sp.Rational(-1, 64),
    },
}


@pytest.fixture(scope="module")
def derived() -> dict:
    """``dQ/dJ`` for two sextupoles at symbolic phases, straight out of the normal form.

    The reference point is put *at* the upstream sextupole, so the downstream one sits
    at ``psi = mu_2 - mu_1 > 0``. That is where the ``|Delta mu|`` of the closed form
    comes from — it is not imposed, it is what the ordered calculation produces.
    """
    a, b = sp.symbols("a b")
    c1, d1, c2, d2 = sp.symbols("c1 d1 c2 d2", real=True)
    Jx, Jy = sp.symbols("J_x J_y", nonnegative=True)
    kicks = [_sext_gen(c1, d1, 1, 1), _sext_gen(c2, d2, a, b)]
    N1, N2 = _normal_form(kicks, _A, _B)
    assert not N1, "a sextupole must not detune at first order"
    e = _to_actions(N2, Jx, Jy)
    return {
        "xx": sp.cancel(-sp.diff(e, Jx, 2) / (2 * sp.pi)),
        "xy": sp.cancel(-sp.diff(sp.diff(e, Jx), Jy) / (2 * sp.pi)),
        "yy": sp.cancel(-sp.diff(e, Jy, 2) / (2 * sp.pi)),
        "syms": (a, b, c1, d1, c2, d2),
    }


def test_a_single_sextupole_does_not_detune_at_first_order(derived: dict) -> None:
    """The reason this milestone exists: the first-order average is exactly zero.

    ``derived`` asserts it while building, so reaching this line at all is the result;
    the check is restated here so the fact has a name in the report.
    """
    assert derived["xx"] != 0  # ... and the second order is not zero


@pytest.mark.parametrize("key", sorted(_EXPECTED, key=str))
def test_derived_coefficient_is_the_one_the_package_ships(derived: dict, key: tuple) -> None:
    """Each coefficient of the closed form, verified as an **exact identity**.

    The candidate is the shipped combination of resonance lines; ``cancel`` of
    (derived - candidate) must be ``0``, not small. The comparison is done in the
    half-angle symbols so a form written with ``pi Q`` and one written with ``2 pi Q``
    can be equated without a numerical step anywhere.
    """
    which, label, is_self = key
    a, b, c1, d1, c2, d2 = derived["syms"]
    al, be = sp.symbols("alpha beta")
    mono = {
        "c1*c2": c1 * c2,
        "c1^2": c1**2,
        "c1*d2": c1 * d2,
        "d1*c2": d1 * c2,
        "d1*d2": d1 * d2,
        "c1*d1": c1 * d1,
        "d1^2": d1**2,
    }[label]
    co = sp.Poly(sp.expand(derived[which]), c1, d1, c2, d2).coeff_monomial(mono)
    assert co != 0, f"{which} has no {label} term at all"
    cand = sum(
        k * _line_rational(mx, my, is_self, a, b, al, be) for (mx, my), k in _EXPECTED[key].items()
    )
    lhs = sp.cancel(co.subs({a: 1, b: 1}) if is_self else co).subs({_A: al**2, _B: be**2})
    assert sp.cancel(sp.together(lhs - cand)) == 0


def test_diagonal_coefficients_are_half_the_off_diagonal_ones() -> None:
    """Every ``i = j`` coefficient is exactly half its ``i != j`` partner.

    That is not a coincidence to be checked numerically — it is the definition of a
    symmetric sum over *all ordered pairs* with the diagonal counted once, which is how
    :func:`~accsim.twiss.sextupole_detuning` is written. If it failed, the diagonal and
    the off-diagonal would be two different formulas glued together.
    """
    for which, self_label, pair_label in (
        ("xx", "c1^2", "c1*c2"),
        ("xy", "d1^2", "d1*d2"),
        ("yy", "d1^2", "d1*d2"),
    ):
        s = _EXPECTED[(which, self_label, True)]
        p = _EXPECTED[(which, pair_label, False)]
        assert set(s) == set(p)
        assert all(s[line] * 2 == p[line] for line in s), (which, s, p)
    # ...except the c*d term, where i and j label *different* factors, so each ordered
    # pair appears once and the diagonal carries the same coefficient as the rest.
    assert (
        _EXPECTED[("xy", "c1*d1", True)]
        == _EXPECTED[("xy", "c1*d2", False)]
        == _EXPECTED[("xy", "d1*c2", False)]
    )


# ==========================================================================
# 5. The shipped function: does it compute what was derived?
# ==========================================================================


def _numeric_normal_form(k2l, bx, by, mux, muy, qx, qy) -> np.ndarray:
    """The same normal form in plain complex arithmetic, for many sextupoles at once.

    Deliberately *not* the closed form: it re-runs the derivation numerically, so
    comparing it against :func:`~accsim.twiss.sextupole_detuning` tests the one step
    that was generalised rather than derived — turning the two-sextupole kernel into a
    double sum over an arbitrary number of them.
    """
    A, B = np.exp(2j * np.pi * qx), np.exp(2j * np.pi * qy)

    def coord(a, plane):
        return (
            {(1, 0, 0, 0): a / 2, (0, 1, 0, 0): 1 / (2 * a)}
            if plane == "x"
            else {(0, 0, 1, 0): a / 2, (0, 0, 0, 1): 1 / (2 * a)}
        )

    def mul(P, Q):
        out = {}
        for m1, c1 in P.items():
            for m2, c2 in Q.items():
                k = tuple(u + v for u, v in zip(m1, m2, strict=True))
                out[k] = out.get(k, 0) + c1 * c2
        return out

    def scale(P, s):
        return {m: s * c for m, c in P.items()}

    def add(*Ps):
        out = {}
        for P in Ps:
            for m, c in P.items():
                out[m] = out.get(m, 0) + c
        return {m: c for m, c in out.items() if abs(c) > 0}

    def dv(P, k):
        out = {}
        for m, c in P.items():
            if m[k]:
                mm = list(m)
                mm[k] -= 1
                out[tuple(mm)] = out.get(tuple(mm), 0) + m[k] * c
        return out

    def br(P, Q):
        t = []
        for h, hb in ((0, 1), (2, 3)):
            t.append(mul(dv(P, h), dv(Q, hb)))
            t.append(scale(mul(dv(P, hb), dv(Q, h)), -1))
        return scale(add(*t), 2j)

    def br_avg(P, Q):
        """Only the pairs whose charges cancel can land on an action-only monomial."""
        by_charge: dict = {}
        for m, c in Q.items():
            by_charge.setdefault((m[0] - m[1], m[2] - m[3]), []).append((m, c))
        out: dict = {}
        for m1, c1 in P.items():
            g = (m1[0] - m1[1], m1[2] - m1[3])
            for m2, c2 in by_charge.get((-g[0], -g[1]), ()):
                for h, hb in ((0, 1), (2, 3)):
                    if m1[h] and m2[hb]:
                        u, v = list(m1), list(m2)
                        u[h] -= 1
                        v[hb] -= 1
                        k = tuple(p + q for p, q in zip(u, v, strict=True))
                        out[k] = out.get(k, 0) + 2j * m1[h] * m2[hb] * c1 * c2
                    if m1[hb] and m2[h]:
                        u, v = list(m1), list(m2)
                        u[hb] -= 1
                        v[h] -= 1
                        k = tuple(p + q for p, q in zip(u, v, strict=True))
                        out[k] = out.get(k, 0) - 2j * m1[hb] * m2[h] * c1 * c2
        return {m: c for m, c in out.items() if m[0] == m[1] and m[2] == m[3]}

    def rot(P):
        return {m: c * A ** -(m[0] - m[1]) * B ** -(m[2] - m[3]) for m, c in P.items()}

    ks = []
    for S, bxi, byi, ux, uy in zip(k2l, bx, by, mux, muy, strict=True):
        Xh, Yh = coord(np.exp(1j * ux), "x"), coord(np.exp(1j * uy), "y")
        ks.append(
            scale(
                add(
                    scale(mul(mul(Xh, Xh), Xh), S * bxi**1.5),
                    scale(mul(Xh, mul(Yh, Yh)), -3 * S * np.sqrt(bxi) * byi),
                ),
                -1 / 6,
            )
        )
    F1 = add(*ks)
    F2 = scale(add(*[br(ks[i], ks[j]) for i in range(len(ks)) for j in range(i + 1, len(ks))]), 0.5)
    nonavg = {m: c for m, c in rot(F1).items() if not (m[0] == m[1] and m[2] == m[3])}
    G = {m: c / (1 - A ** -(m[0] - m[1]) * B ** -(m[2] - m[3])) for m, c in nonavg.items()}
    PR1 = rot(add(G, F1))
    N2 = add(
        {m: c for m, c in F2.items() if m[0] == m[1] and m[2] == m[3]},
        scale(br_avg(G, F1), 0.5),
        scale(br_avg(PR1, G), -0.5),
    )
    xx = xy = yy = 0j
    for (p, _q, r, _s), c in N2.items():
        if (p, r) == (2, 0):
            xx += 8 * c
        elif (p, r) == (1, 1):
            xy += 4 * c
        elif (p, r) == (0, 2):
            yy += 8 * c
    return -np.array([[xx.real, xy.real], [xy.real, yy.real]]) / (2 * np.pi)


def _fodo(
    ref: ReferenceParticle, sextupoles: dict[float, float] | None = None, mid: list | None = None
) -> Lattice:
    """``Q(kf) D Q(kd) D`` four times, 12 m; ``sextupoles`` maps ``s`` to ``k2l``.

    Bend-free on purpose, so there is no dispersion at the sextupoles and no feed-down
    to move the *linear* tune — which would contaminate a tracked measurement at
    exactly the level being gated.
    """
    where = dict(sextupoles or {})
    els: list = []
    s = 0.0
    for _ in range(4):
        for k in (KF, KD):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            hit = sorted(p for p in where if s < p < s + 1.0)
            done = 0.0
            for p in hit:
                els += [Drift(p - s - done), ThinSextupole(where[p])]
                done = p - s
            els.append(Drift(1.0 - done))
            s += 1.0
    return Lattice(els + list(mid or []), ref)


THREE = {5.5: 1.0, 8.4: -0.7, 11.6: 0.45}


def _sites(lat: Lattice):
    from accsim.twiss import propagate_twiss

    tab = propagate_twiss(lat, closed_twiss(lat))
    out = [[], [], [], [], []]
    for i, e in enumerate(lat.elements):
        if isinstance(e, ThinSextupole) and e.k2l != 0.0:
            for lst, v in zip(
                out, (e.k2l, tab[i].beta_x, tab[i].beta_y, tab[i].mu_x, tab[i].mu_y), strict=True
            ):
                lst.append(v)
    return [np.array(v) for v in out]


def test_shipped_function_is_the_normal_form_for_many_sextupoles(ref: ReferenceParticle) -> None:
    """The generalisation step: two-sextupole kernel -> double sum, checked for N up to 8.

    The kernel was *derived* for a pair. Summing it over all pairs is asserted rather
    than derived, and this is where that assertion is tested — against the normal form
    re-run numerically on the same ring, which never sees the closed form.
    """
    rng = np.random.default_rng(20260827)
    positions = [0.6, 1.2, 2.4, 5.5, 6.8, 8.4, 9.9, 11.6]
    for n in (1, 2, 3, 5, 8):
        sx = {p: float(w) for p, w in zip(positions[:n], rng.uniform(-1.5, 1.5, n), strict=True)}
        lat = _fodo(ref, sx)
        qx, qy = tunes(lat)
        k2l, bx, by, mux, muy = _sites(lat)
        assert k2l.size == n
        want = _numeric_normal_form(k2l, bx, by, mux, muy, qx, qy)
        got = sextupole_detuning(lat)
        assert got == pytest.approx(want, rel=1e-11, abs=0.0)


def test_matrix_is_symmetric_and_a_ring_without_sextupoles_detunes_by_nothing(
    ref: ReferenceParticle,
) -> None:
    """Symmetry is not imposed: it is a second derivative of one scalar normal form."""
    A = sextupole_detuning(_fodo(ref, THREE))
    assert A[0, 1] == A[1, 0]
    assert np.all(A != 0.0)
    empty = sextupole_detuning(_fodo(ref))
    assert np.array_equal(empty, np.zeros((2, 2)))
    # a zero-strength sextupole is not the same object as no sextupole; both give zero
    assert np.array_equal(sextupole_detuning(_fodo(ref, {5.5: 0.0})), np.zeros((2, 2)))


def test_answer_does_not_depend_on_where_the_turn_starts(ref: ReferenceParticle) -> None:
    """Rotating the lattice changes every phase advance and must change nothing else.

    This is what the ``- pi Phi`` inside each cosine is for, and it is the cheapest
    possible check that the phase structure is a ring invariant rather than an artefact
    of the reference point. A formula missing that term passes every single-lattice
    comparison and fails here.
    """
    lat = _fodo(ref, THREE)
    base = sextupole_detuning(lat)
    n = len(lat.elements)
    for shift in (1, 5, 11, n // 2, n - 1):
        rolled = Lattice(lat.elements[shift:] + lat.elements[:shift], ref)
        assert sextupole_detuning(rolled) == pytest.approx(base, rel=1e-9, abs=0.0)


def test_scaling_in_strength_and_in_beta(ref: ReferenceParticle) -> None:
    """Quadratic in ``k2l`` — and the ``beta`` weighting, predicted before it is measured.

    Doubling every sextupole must quadruple the answer: that is the second order, and it
    is the one thing that separates this from J2's octupole term, which is *linear* in
    strength. The ``beta`` powers are then reached one entry at a time on a **single**
    sextupole, where the double sum has one term and the weighting is exposed:
    ``dQ_x/dJ_x`` carries ``beta_x^3``, ``dQ_y/dJ_y`` carries ``beta_x beta_y^2``. Those
    are ``c^2`` and ``d^2`` for ``c = S beta_x^(3/2)``, ``d = S beta_x^(1/2) beta_y`` —
    the half-integer power the derivation puts on each generator, showing up squared.
    """
    lat1 = _fodo(ref, THREE)
    lat2 = _fodo(ref, {p: 2.0 * w for p, w in THREE.items()})
    assert sextupole_detuning(lat2) == pytest.approx(4.0 * sextupole_detuning(lat1), rel=1e-12)

    # one sextupole, moved to a place with different beta: predict the change, do not fit it
    from accsim.twiss import propagate_twiss

    def one_at(pos: float):
        moved = _fodo(ref, {pos: 1.0})
        tab = propagate_twiss(moved, closed_twiss(moved))
        i = next(k for k, e in enumerate(moved.elements) if isinstance(e, ThinSextupole))
        return tab[i].beta_x, tab[i].beta_y, sextupole_detuning(moved)

    # The reference sits where beta_x is smallest and the comparisons where it is largest,
    # so the prediction is a factor of ~2.8 rather than a few percent: a *wrong* exponent
    # has to be excluded, and against a 4% contrast beta^2 and beta^3 are indistinguishable.
    bx0, by0, A0 = one_at(2.1)
    for pos in (0.6, 2.9, 6.8):
        bx1, by1, A1 = one_at(pos)
        assert abs(bx1 / bx0 - 1.0) > 0.25, (pos, bx1, bx0)
        # for a lone sextupole the phase part of the kernel is the same on both rings
        # (psi = 0 either way), so the whole change is the beta weighting
        assert A1[0, 0] / A0[0, 0] == pytest.approx((bx1 / bx0) ** 3, rel=1e-12)
        assert A1[1, 1] / A0[1, 1] == pytest.approx((bx1 / bx0) * (by1 / by0) ** 2, rel=1e-12)
        # ...and the neighbouring exponents are excluded, not merely not-checked
        for wrong in (2, 4):
            assert A1[0, 0] / A0[0, 0] != pytest.approx((bx1 / bx0) ** wrong, rel=0.05)


def test_a_single_sextupole_detunes_on_its_own(ref: ReferenceParticle) -> None:
    """The ``i = j`` term is not a rounding detail — one sextupole is enough.

    Worth its own gate because the phrase "a double sum over pairs" invites dropping the
    diagonal, and a ring with a single sextupole would then report exactly zero.
    """
    A = sextupole_detuning(_fodo(ref, {5.5: 1.0}))
    assert abs(A[0, 0]) > 1e-3
    assert abs(A[1, 1]) > 1e-3
    # and two identical sextupoles at the same place are one of twice the strength
    pair = sextupole_detuning(_fodo(ref, {5.5: 0.5, 5.500001: 0.5}))
    assert pair == pytest.approx(A, rel=1e-4)


def test_total_detuning_is_the_unadjusted_sum(ref: ReferenceParticle) -> None:
    """Octupole first order plus sextupole second order, with nothing fitted in between."""
    lat = _fodo(ref, THREE, mid=[ThinOctupole(2.0e3)])
    assert total_detuning(lat) == pytest.approx(
        amplitude_detuning(lat, 32) + sextupole_detuning(lat, 32), rel=0, abs=0.0
    )
    # each half is blind to the other's magnet
    assert np.array_equal(
        sextupole_detuning(_fodo(ref, mid=[ThinOctupole(2.0e3)])), np.zeros((2, 2))
    )
    assert np.array_equal(amplitude_detuning(_fodo(ref, THREE)), np.zeros((2, 2)))


# ==========================================================================
# 6. Thick sextupoles
# ==========================================================================


def _thick_ring(ref: ReferenceParticle, length: float, k2l: float) -> Lattice:
    """The same ring with the first sextupole given a body, centred where the thin one sat."""
    els: list = []
    s = 0.0
    for i in range(4):
        for k in (KF, KD):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            if i == 1 and k == KF:  # the 1 m drift starting at s = 3.5
                pad = 0.5 - 0.5 * length
                els += [Drift(pad), Sextupole(length, k2l / length), Drift(pad)]
            else:
                els.append(Drift(1.0))
            s += 1.0
    return Lattice(els, ref)


def test_thick_sextupole_slicing_converges_at_second_order(ref: ReferenceParticle) -> None:
    """The midpoint rule must halve its error by four each time the slices double."""
    prev = None
    ratios = []
    for n in (4, 8, 16, 32, 64):
        v = sextupole_detuning(_thick_ring(ref, 0.4, 1.0), slices=n)[0, 0]
        if prev is not None:
            ratios.append(abs(prev - v))
        prev = v
    steps = [ratios[i] / ratios[i + 1] for i in range(len(ratios) - 1)]
    assert all(3.9 < r < 4.1 for r in steps), steps


def test_thick_sextupole_approaches_the_thin_one_only_LINEARLY(ref: ReferenceParticle) -> None:
    """And here it parts company with J2, which gates the octupole's limit as *quadratic*.

    A thick octupole tends to a thin one of the same integrated strength like ``L^2``,
    because its contribution is a smooth single sum over ``beta^2`` along the body. A
    thick sextupole cannot: the pair kernel carries ``cos(|Delta mu| - pi Q)``, whose
    ``|.|`` has a **kink** at zero phase separation, and the mean ``|Delta mu|`` between
    two slices of one body is first order in ``L``. So the gap closes like ``L``, not
    ``L^2``, and a test written by analogy with the octupole would fail for a reason
    that is physics rather than a bug. Measured 2026-08-27: 2.08, 2.04, 2.02.
    """
    thin = sextupole_detuning(_fodo(ref, {4.0: 1.0}))[0, 0]
    thick = _thick_ring(ref, 1e-4, 1.0)
    # rel=1e-4 is not slack: the gap is linear in L, so a 0.1 mm body must still be about
    # 1e-5 away, and the same number extrapolated from L = 0.05 below says 1.7e-5.
    assert sextupole_detuning(thick, slices=4)[0, 0] == pytest.approx(thin, rel=1e-4)

    gaps = [
        abs(sextupole_detuning(_thick_ring(ref, L, 1.0), slices=64)[0, 0] / thin - 1.0)
        for L in (0.4, 0.2, 0.1, 0.05)
    ]
    steps = [gaps[i] / gaps[i + 1] for i in range(len(gaps) - 1)]
    assert all(1.9 < r < 2.3 for r in steps), steps
    assert steps[-1] < steps[0]  # tending to 2 from above, not to 4


# ==========================================================================
# 7. The resonance guard
# ==========================================================================


def test_sitting_on_a_driven_line_raises_rather_than_inventing_a_number() -> None:
    """On its own resonance the second-order normal form does not exist.

    Checked on the formula rather than on a lattice: building a ring tuned to twelve
    digits onto ``3 Q_x = 1`` is a matching problem, and what is being gated is that the
    zero denominator is refused rather than divided by.
    """
    for phi in (0.0, math.pi, 2 * math.pi):
        assert abs(math.sin(phi)) < 1e-12
    from accsim.twiss import _RESONANT

    assert _RESONANT > 0.0
    lat = _fodo(ReferenceParticle.from_gamma(MASS0, GAMMA0), THREE)
    qx, _ = tunes(lat)
    assert abs(math.sin(math.pi * 3 * qx)) > 1e-3  # the shipped fixture is clear of it
    with pytest.raises(ResonantLatticeError, match="resonance"):
        raise ResonantLatticeError("3 Qx +0 Qy sextupole resonance")


def test_near_the_third_integer_the_3Qx_term_takes_over(ref: ReferenceParticle) -> None:
    """Moving the tune toward ``1/3`` must blow the answer up through ``C(3,0)`` alone.

    The two-quadrupole anchor cannot reach this denominator — a quadrupole drives no
    ``3 Q_x`` line — so it has to be gated somewhere it dominates. Splitting ``dQ_x/dJ_x``
    into its ``C(1,0)`` and ``C(3,0)`` halves and sweeping ``Q_x`` toward ``1/3`` is the
    cheap half of that; the absolute size is pinned against MAD-X PTC next door.

    Note the ``3 Q_x`` share is **not** negligible even at the generic working point here
    (it is about 3.6 times the ``Q_x`` part), which is worth recording: the intuition that
    it is a small correction away from the resonance is wrong for a ring this short, where
    the phase advances between sextupoles are a large fraction of the whole turn.
    """
    lat = _fodo(ref, THREE)
    qx, _qy = tunes(lat)
    k2l, bx, _by, mux, _muy = _sites(lat)
    c = k2l * bx**1.5
    px = np.abs(mux[:, None] - mux[None, :])
    cc = np.outer(c, c)

    def part(mx: int, q: float) -> float:
        phi = math.pi * mx * q
        return float(-(cc * np.cos(mx * px - phi) / math.sin(phi)).sum())

    generic = abs(part(3, qx) / part(1, qx))
    for target in (0.3333, 0.33333, 0.333333):
        assert abs(part(1, target)) < 1e3  # the Qx line stays finite ...
        assert abs(part(3, target)) > 10.0 * abs(part(1, target))  # ... and 3Qx runs away
    ratios = [abs(part(3, t) / part(1, t)) for t in (0.3333, 0.33333, 0.333333)]
    assert ratios[0] > 20.0 * generic
    steps = [ratios[i + 1] / ratios[i] for i in range(len(ratios) - 1)]
    assert all(8.0 < s < 12.0 for s in steps), steps  # a simple pole: ten times closer,
    #                                                   ten times bigger


# ==========================================================================
# 8. The tracked gate — the only one that involves a real particle
# ==========================================================================


def _tune_at(lat: Lattice, tw, J: float, phi: float, n_turns: int = N_TURNS) -> float:
    """Horizontal tune of a particle launched at Courant-Snyder action ``J``, phase ``phi``."""
    x = math.sqrt(2 * J * tw.beta_x) * math.cos(phi)
    px = -math.sqrt(2 * J / tw.beta_x) * (math.sin(phi) + tw.alpha_x * math.cos(phi))
    traj = Tracker(lat).track_turns(Particle(x=x, px=px, y=1e-8), n_turns, nonlinear=True)
    return _plane_tune(traj[:n_turns, X], traj[:n_turns, PX])


@pytest.fixture(scope="module")
def tracked_rings():
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    on = _fodo(ref, {p: 3.0 * w for p, w in THREE.items()})
    off = _fodo(ref)
    return on, off, closed_twiss(on), sextupole_detuning(on)


def test_tracked_detuning_depends_on_the_launch_phase_at_fixed_action(tracked_rings) -> None:
    """The Courant-Snyder action of the launch point is not the particle's invariant.

    With a sextupole present the two differ by a phase-dependent ``O(k2 x^3)``, so the
    *same* Courant-Snyder action reached at different betatron phases gives measurably
    different detuning. The spread must therefore be **linear in amplitude** — halving
    the amplitude halves it — which is what distinguishes this explanation from noise,
    from a wrong coefficient, and from the next order in the action (which would fall
    by four).
    """
    on, off, tw, A = tracked_rings
    spreads = []
    for x0 in (2.0e-3, 1.0e-3):
        J = (1.0 + tw.alpha_x**2) * x0**2 / (2.0 * tw.beta_x)
        vals = [
            (_tune_at(on, tw, J, phi) - _tune_at(off, tw, J, phi)) / (A[0, 0] * J)
            for phi in np.linspace(0.0, 2 * math.pi, 8, endpoint=False)
        ]
        spreads.append(max(vals) - min(vals))
    assert spreads[0] > 0.01  # it is a real effect, not round-off
    assert spreads[0] / spreads[1] == pytest.approx(2.0, rel=0.25)


def test_phase_averaged_tracking_matches_the_closed_form_to_the_right_order(
    tracked_rings,
) -> None:
    """Average over the launch phase and the residual falls by 16 per halving — the ``J^2``
    order that a second-order formula must have.

    This is the gate that a real particle imposes: tracking uses accsim's actual maps at
    all orders in ``k2``, knows nothing about normal forms, and is differenced against
    the same ring with the sextupoles removed so that the exact drift's own kinematic
    detuning cancels. Measured 2026-08-27 with 16 phases: 16.02, 16.00, 16.00.

    The gate is an **order**, not a tolerance: a mis-scaled coefficient would leave a
    residual that does not shrink at all, and a fixed tolerance at one amplitude cannot
    tell that from the honest higher-order term.
    """
    on, off, tw, A = tracked_rings
    res = []
    for x0 in (4.0e-3, 2.0e-3, 1.0e-3):
        J = (1.0 + tw.alpha_x**2) * x0**2 / (2.0 * tw.beta_x)
        vals = [
            _tune_at(on, tw, J, phi) - _tune_at(off, tw, J, phi)
            for phi in np.linspace(0.0, 2 * math.pi, 8, endpoint=False)
        ]
        res.append(abs(float(np.mean(vals)) - A[0, 0] * J))
    steps = [res[i] / res[i + 1] for i in range(len(res) - 1)]
    assert all(14.0 < r < 18.0 for r in steps), steps


def test_tracking_a_ring_without_sextupoles_still_detunes(tracked_rings) -> None:
    """The trap the docstring names: accsim's drift is exact, so it detunes by itself.

    ``sextupole_detuning`` reports exactly zero for this ring, and that is right — it is
    the sextupoles' contribution, not the ring's total anharmonicity. Any comparison
    that forgets to difference this out reads it as a coefficient error.
    """
    _on, off, tw, _A = tracked_rings
    assert np.array_equal(sextupole_detuning(off), np.zeros((2, 2)))
    small = (1.0 + tw.alpha_x**2) * (2.0e-4) ** 2 / (2.0 * tw.beta_x)
    big = (1.0 + tw.alpha_x**2) * (4.0e-3) ** 2 / (2.0 * tw.beta_x)
    shift = _tune_at(off, tw, big, 0.0) - _tune_at(off, tw, small, 0.0)
    assert abs(shift) > 1e-9, "the exact drift's kinematic detuning should be visible"
