r"""P1 acceptance: the transfer map to second order — ``T`` per element, ``T`` for the turn.

Everything this package returned as a *matrix* was first order, and everything beyond
first order was expressed per effect. This file gates the object those effects are
projections of, the ``6x6x6`` second-order map, on four fronts:

1. **Closed forms, derived here rather than recalled.** The thin sextupole is exact and
   pins the symmetric storage convention (``T_413 = T_431 = k2l/2``, ratio ``1`` not
   ``2``). The drift's ``T`` is compared entry by entry with the sympy expansion of its
   own exact map, written down from the geometry rather than transcribed; the thick
   quadrupole's momentum column with the derivative of its ``k1/(1 + delta)`` block; the
   cavity's ``T[delta, zeta, zeta]`` with the curvature of its ``sin``.
2. **The composition rule** — exact on polynomial maps (sympy composes two quadratic maps
   and truncates), and on a real ring against a direct expansion of the whole turn, where
   the two converge as the *fourth* power of the direct step and a deliberately wrong
   ``1/2`` breaks the agreement by orders of magnitude.
3. **Symplecticity as exact identities.** ``R^T J T_k + T_k^T J R = 0`` is derived
   symbolically from ``M^T J M = J``, holds to the differencing floor for every thin kick
   and every on-axis thick map — and the sector bend **fails** it in ``(zeta, delta)`` by
   a closed-form amount while passing it in ``(zeta, p_zeta)``, which is
   :mod:`accsim.symplectic`'s first-order caveat met one order up.
4. **Three shipped quantities as projections of ``T``**, O6's "two anchors, never one"
   rule with a third for good measure: the chromaticity (F2/M1, via ``dR/ddelta =
   2 T[:, :, delta]`` and, with bends, the dispersion direction), the second-order
   dispersion (M3, via the fixed point of the quadratic map) and all five first-order
   sextupole driving terms (O4, via the cubic generator recovered from ``T`` and
   re-expanded in the normal form).

**What is measured, not assumed:** the differencing floor. Every gate below sits above a
number this file measures — a step sweep on the drift, the composed/direct agreement on a
ring — and states the headroom, per M2's rule that a tolerance widened around a miss is a
bug filed under another name.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Corrector,
    Dipole,
    Drift,
    Lattice,
    Octupole,
    Quadrupole,
    ReferenceParticle,
    RFCavity,
    TaylorMap,
    ThinOctupole,
    ThinQuadrupole,
    ThinSextupole,
    ThinSkewSextupole,
    canonical_map,
    chromaticity,
    closed_orbit_nonlinear,
    closed_twiss,
    compose,
    linearised_element_maps,
    linearised_one_turn_map,
    natural_chromaticity,
    normal_form,
    resonance_driving_terms,
    second_order_dispersion,
    second_order_element_maps,
    second_order_one_turn_map,
    second_order_symplectic_residual,
    taylor_expand,
    tunes,
)
from accsim.coords import DELTA, DIM, PX, PY, ZETA, X, Y
from accsim.symplectic import J6

MASS0 = 938.27208816e6
GAMMA0 = 20.0


@pytest.fixture(scope="module")
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _expand(elem, ref: ReferenceParticle, state=None, **kw) -> TaylorMap:
    z0 = np.zeros(DIM) if state is None else np.asarray(state, dtype=float)
    return taylor_expand(lambda s: elem.track(s, ref), z0, **kw)


def _whole_turn(lattice: Lattice):
    def turn(state: np.ndarray) -> np.ndarray:
        out = np.asarray(state, dtype=float)
        for elem in lattice.elements:
            out = elem.track(out, lattice.ref)
        return out

    return turn


def _fodo_ring(ref: ReferenceParticle, k2l: float = 2.0, cells: int = 4) -> Lattice:
    """Thick FODO with a bend and one thin sextupole per cell — stable at (0.394, 0.377)."""
    cell = [
        Quadrupole(0.3, 1.2),
        Drift(0.5),
        ThinSextupole(k2l),
        Drift(0.5),
        Dipole(1.0, 0.1),
        Quadrupole(0.3, -1.2),
        Drift(1.0),
    ]
    return Lattice(cell * cells, ref)


# ---------------------------------------------------------------------------
# 1. closed forms, derived
# ---------------------------------------------------------------------------


def test_thin_sextupole_is_exact_and_pins_the_symmetric_convention(ref) -> None:
    """``T[px,x,x] = -k2l/2``, ``T[px,y,y] = T[py,x,y] = T[py,y,x] = +k2l/2``, all else zero.

    The stencils are exact on a quadratic, so this is round-off and nothing else. The
    ratio ``|T[py,x,y]| / |T[px,x,x]|`` is **1**: the symmetric convention, in which the
    cross term of ``sum_{jk} T_ijk u_j u_k`` is counted twice by the sum and not by ``T``.
    A triangular convention would put ``k2l`` there and the ratio at ``2``.
    """
    for k2l in (12.0, -9.0):
        m = _expand(ThinSextupole(k2l), ref)
        expected = np.zeros((DIM,) * 3)
        expected[PX, X, X] = -0.5 * k2l
        expected[PX, Y, Y] = +0.5 * k2l
        expected[PY, X, Y] = expected[PY, Y, X] = +0.5 * k2l
        assert np.max(np.abs(m.T - expected)) < 1e-12
        assert np.max(np.abs(m.R - np.eye(DIM))) < 1e-13
        assert np.max(np.abs(m.k)) == 0.0
        assert abs(abs(m.T[PY, X, Y]) / abs(m.T[PX, X, X]) - 1.0) < 1e-12


def test_thin_skew_sextupole_is_the_normal_one_rotated(ref) -> None:
    """``Delta px = +k2sl x y``, ``Delta py = +k2sl (x^2 - y^2)/2`` — J3's element, exact."""
    k2sl = 7.0
    m = _expand(ThinSkewSextupole(k2sl), ref)
    expected = np.zeros((DIM,) * 3)
    expected[PX, X, Y] = expected[PX, Y, X] = 0.5 * k2sl
    expected[PY, X, X] = +0.5 * k2sl
    expected[PY, Y, Y] = -0.5 * k2sl
    assert np.max(np.abs(m.T - expected)) < 1e-12


def test_an_octupole_does_not_appear_in_t_at_all(ref) -> None:
    """A cubic kick has no second-order part: ``T`` is exactly zero and ``R`` the identity.

    The refusal in the roadmap made concrete: anyone reading ``T`` as "the nonlinear map"
    is wrong about octupoles, whose whole content is the third-order term.
    """
    m = _expand(ThinOctupole(400.0), ref)
    assert np.max(np.abs(m.T)) < 1e-12
    assert np.max(np.abs(m.R - np.eye(DIM))) < 1e-13


def test_the_sliced_thick_bodies_carry_the_linear_drift_and_t_says_so(ref) -> None:
    """A thick :class:`Octupole`'s ``T`` is exactly zero; an exact drift of its length is not.

    **A finding, recorded rather than fixed here.** The thick sextupole and octupole
    bodies are drift-kick-drift integrators whose drift halves are the *linear* drift
    matrix, so at second order they lack the drift's own ``-L px delta`` chromatic term
    and ``-L px^2/2`` path lengthening — ``T[x, px, delta] = -L/2 = -0.1`` on this 0.2 m
    body, where the sliced map has ``0``. MAD-X's thick sextupole carries those entries
    (``t126 = -L/(2 beta0)``, measured), so the reference leg pins the gap as exactly the
    drift's ``T``. It is an approximation that predates this milestone (L1 made the
    standalone drift exact and left the sliced bodies alone) and it is named in
    ``docs/ROADMAP.md`` as P1's follow-up; a thin element has no such term to miss.
    """
    thick = _expand(Octupole(0.2, 2000.0), ref)
    drift = _expand(Drift(0.2), ref)
    assert np.max(np.abs(thick.T)) < 1e-12
    assert abs(drift.T[X, PX, DELTA] + 0.1) < 1e-11
    assert np.max(np.abs(thick.T - drift.T)) > 0.09  # the whole drift T is the gap
    assert np.max(np.abs(thick.R - drift.R)) < 1e-12  # at first order they agree


def _symbolic_drift(L: sp.Symbol, beta0: sp.Symbol, gamma0: sp.Symbol):
    """The exact field-free map from the geometry, not from :mod:`accsim.elements.drift`.

    A drift is a straight line: ``x`` advances by ``L`` times the direction cosine
    ``px/pz``. ``zeta = s - beta0 c t`` advances by ``L`` less ``beta0 c`` times the
    time of flight; the path is ``L |p|/pz`` at speed ``beta c``, and
    ``(beta0/beta)(|p|/p0) = E/E0``, so ``zeta += L (1 - E/(E0 pz))`` with
    ``E/E0 = sqrt(beta0^2 (1+delta)^2 + 1/gamma0^2)`` and ``1/gamma0^2 = 1 - beta0^2``.
    """
    x, px, y, py, zeta, delta = sp.symbols("x px y py zeta delta", real=True)
    pz = sp.sqrt((1 + delta) ** 2 - px**2 - py**2)
    e_over_e0 = sp.sqrt(beta0**2 * (1 + delta) ** 2 + 1 - beta0**2)
    out = [x + L * px / pz, px, y + L * py / pz, py, zeta + L * (1 - e_over_e0 / pz), delta]
    return (x, px, y, py, zeta, delta), out


def _symbolic_second_order(coords, outputs, subs) -> np.ndarray:
    """``T_ijk = (1/2) d^2 f_i / dz_j dz_k`` at the origin — the symmetric convention."""
    T = np.zeros((DIM,) * 3)
    at_zero = dict.fromkeys(coords, 0)
    for i, f in enumerate(outputs):
        for j in range(DIM):
            for k in range(j, DIM):
                val = sp.diff(f, coords[j], coords[k]).subs(at_zero).subs(subs)
                T[i, j, k] = T[i, k, j] = 0.5 * float(val)
    return T


def test_drift_matches_its_own_symbolic_expansion(ref) -> None:
    """All 216 entries against sympy, and the three closed forms named.

    ``T[x, px, delta] = -L/2`` (the chromatic term of the focusing-free drift),
    ``T[zeta, px, px] = -L/2`` (path lengthening by the angle) and
    ``T[zeta, delta, delta] = -L (2 + beta0^2) / (2 gamma0^2)`` — the last derived, not
    remembered, and the one MAD-X's ``t566 = -3L/(2 beta0^3 gamma0^2)`` becomes once
    ``PT``'s quadratic term is carried through (``tests/reference``).
    """
    L = 1.7
    Ls, b0, g0 = sp.symbols("L beta0 gamma0", positive=True)
    coords, outputs = _symbolic_drift(Ls, b0, g0)
    subs = {Ls: L, b0: ref.beta0, g0: ref.gamma0}
    expected = _symbolic_second_order(coords, outputs, subs)

    m = _expand(Drift(L), ref, step=2.5e-4)
    assert np.max(np.abs(m.T - expected)) < 1e-12
    assert np.max(np.abs(m.R - Drift(L).matrix(ref))) < 1e-12

    assert abs(expected[X, PX, DELTA] + L / 2) < 1e-15
    assert abs(expected[ZETA, PX, PX] + L / 2) < 1e-15
    assert abs(expected[ZETA, DELTA, DELTA] + L * (2 + ref.beta0**2) / (2 * ref.gamma0**2)) < 1e-15
    # The symbolic closed form, as a symbol, so the test says what it claims.
    _, sym_out = _symbolic_drift(Ls, b0, g0)
    zeta_dd = sp.diff(sym_out[ZETA], coords[DELTA], 2).subs(dict.fromkeys(coords, 0)) / 2
    assert sp.simplify(zeta_dd + Ls * (2 + b0**2) / (2 * (1 / (1 - b0**2)))) == 0


def test_drift_step_sweep_measures_the_differencing_floor(ref) -> None:
    """Truncation falls as ``step^4`` until round-off takes over below ``1e-13``.

    Recorded so the default step is a measured choice: on a 1 m drift the error of
    ``T[x, px, delta]`` is ``4e-13`` at ``5e-4`` (the default), ``6e-12`` at ``1e-3``,
    ``3e-14`` at ``2.5e-4`` and ``1.5e-9`` at ``4e-3`` — sixteen-fold per doubling, the
    fourth-order stencil's signature. A map with no fourth derivative (any thin kick) has no truncation term at
    all, which the sextupole gate above already showed.
    """
    errs = {}
    for step in (5e-4, 1e-3, 2e-3, 4e-3):
        m = _expand(Drift(1.0), ref, step=step)
        errs[step] = abs(m.T[X, PX, DELTA] + 0.5)
    assert errs[1e-3] < 2e-11
    assert errs[5e-4] < 2e-12
    for a, b in ((1e-3, 2e-3), (2e-3, 4e-3)):
        ratio = errs[b] / errs[a]
        assert 12.0 < ratio < 20.0, (a, b, ratio)


def test_thick_quadrupole_momentum_column_is_the_derivative_of_its_focusing(ref) -> None:
    """``T[:, :, delta]`` on the transverse block is ``(1/2) dM/ddelta`` of ``k1/(1+delta)``.

    Derived: with ``w = sqrt(k1/(1+delta))`` the focusing block is
    ``[[cos wL, sin(wL)/(w (1+delta))], [-(1+delta) w sin wL, cos wL]]`` and its
    defocusing partner the same with ``k1 -> -k1``; ``T`` holds half the
    ``delta``-derivative. The zeta row (path
    lengthening) is not derived here — the MAD-X leg gates it.
    """
    L, k1 = 0.5, 1.2
    d = sp.symbols("delta", real=True)
    T = _expand(Quadrupole(L, k1), ref).T

    def block(k):
        # Canonical px, not the slope: H = px^2/(2(1+delta)) + k x^2/2, so the sin
        # entries carry (1+delta) — the difference between px and x' that a remembered
        # "M(k/(1+delta))" gets wrong in both off-diagonal entries (tried, and it did).
        w = sp.sqrt(k / (1 + d))
        return sp.Matrix(
            [
                [sp.cos(w * L), sp.sin(w * L) / (w * (1 + d))],
                [-(1 + d) * w * sp.sin(w * L), sp.cos(w * L)],
            ]
        )

    for (a, b), k in (((X, PX), k1), ((Y, PY), -k1)):
        dM = (block(k).diff(d) / 2).subs(d, 0)
        got = T[[a, a, b, b], [a, b, a, b], DELTA]
        want = np.array([complex(dM[i, j]).real for i in range(2) for j in range(2)])
        assert np.max(np.abs(got - want)) < 1e-11, (a, got, want)
    # No transverse-transverse content: a quadrupole is linear in the transverse plane.
    assert np.max(np.abs(T[:4, :4, :4])) < 1e-11


def test_rf_cavity_curvature_is_the_second_derivative_of_its_sin(ref) -> None:
    """``T[delta, zeta, zeta] = -(qV/(beta0^2 E0)) k_rf^2 sin(phi_s) / 2``.

    The one element that reads ``zeta`` — the reason PTC's ``icase=6`` leg needs a cavity
    in the ring. Its floor is the cavity's own ``sin(phi_s - k zeta) - sin(phi_s)``
    cancellation, which is why a *larger* zeta step is the accurate one here: measured
    ``4e-10`` relative at ``1e-2``, ``9e-8`` at ``1e-3``, ``4e-7`` at the default ``5e-4``,
    ``4e-6`` at ``1e-4``.
    """
    rf = RFCavity(1e6, 3e6, 0.3)
    amp = rf.voltage / (ref.beta0**2 * ref.total_energy_eV)
    k = rf.k_rf(ref)
    want = -amp * k * k * math.sin(0.3) / 2
    step = np.array([1e-3] * 4 + [1e-2, 1e-3])
    m = _expand(rf, ref, step=step)
    assert abs(m.T[DELTA, ZETA, ZETA] / want - 1) < 5e-9
    assert abs(m.R[DELTA, ZETA] - rf.matrix(ref)[DELTA, ZETA]) < 1e-15
    others = m.T.copy()
    others[DELTA, ZETA, ZETA] = 0.0
    assert np.max(np.abs(others)) < 1e-12  # round-off on entries that are exactly zero


# ---------------------------------------------------------------------------
# 2. the composition rule
# ---------------------------------------------------------------------------


def _random_map(rng: np.random.Generator, origin: np.ndarray) -> TaylorMap:
    """A sparse rational quadratic map, exactly representable so sympy can compose it."""
    R = rng.integers(-3, 4, size=(DIM, DIM)) / 4.0
    T = rng.integers(-3, 4, size=(DIM, DIM, DIM)) / 8.0
    T = np.where(rng.random(T.shape) < 0.7, 0.0, T)
    T = 0.5 * (T + np.swapaxes(T, 1, 2))
    k = rng.integers(-2, 3, size=DIM) / 8.0
    return TaylorMap(origin, k, R, T)


def test_composition_rule_is_exact_on_polynomial_maps() -> None:
    """``B . A`` truncated to second order, by sympy, equals :func:`compose` to round-off.

    Two quadratic maps compose to a quartic; the second-order truncation of that quartic is
    what the rule claims to be, and sympy computes it with no rule at all — substitute,
    expand, drop degree three and above. Every ``1/2`` in the convention is exercised:
    a symmetric ``T`` read back from the coefficient of ``u_j u_k`` is *half* that
    coefficient.
    """
    rng = np.random.default_rng(7)
    A = _random_map(rng, origin=np.array([0.1, 0.0, -0.2, 0.05, 0.0, 0.01]))
    B = _random_map(rng, origin=A.k)
    BA = compose(B, A)

    u = sp.symbols("u0:6")
    uv = sp.Matrix(u)

    def poly(m: TaylorMap):
        R = sp.Matrix(6, 6, lambda i, j: sp.Rational(m.R[i, j]))
        expr = sp.Matrix(6, 1, lambda i, _: sp.Rational(m.k[i]))
        expr += R * uv
        for i in range(6):
            expr[i] += sum(
                sp.Rational(m.T[i, j, k]) * u[j] * u[k] for j in range(6) for k in range(6)
            )
        return expr

    inner = poly(A) - sp.Matrix(6, 1, lambda i, _: sp.Rational(B.origin[i]))
    outer = poly(B).subs(dict(zip(u, inner, strict=True)), simultaneous=True)
    for i in range(6):
        P = sp.Poly(sp.expand(outer[i]), *u)
        for monom, coeff in P.terms():
            deg = sum(monom)
            if deg == 0:
                assert coeff == sp.Rational(BA.k[i])
            elif deg == 1:
                j = monom.index(1)
                assert coeff == sp.Rational(BA.R[i, j]), (i, j)
            elif deg == 2:
                idx = [n for n, e in enumerate(monom) for _ in range(e)]
                j, k = idx
                factor = 1 if j == k else 2
                assert coeff == factor * sp.Rational(BA.T[i, j, k]), (i, j, k)
        for j in range(6):
            for k in range(6):
                if sp.Poly(sp.expand(outer[i]), *u).coeff_monomial(u[j] * u[k]) == 0:
                    assert BA.T[i, j, k] == 0.0


def test_ring_composition_reproduces_the_directly_expanded_turn(ref) -> None:
    """The composed one-turn ``T`` equals the whole turn differenced at once — and how.

    The direct expansion converges onto the composed map as the **fourth power** of its
    step (measured on this ring: ``6e-4`` at ``1e-3``, ``6e-8`` at ``1e-4``, ``6e-10`` at
    ``3e-5``, on entries up to ``620``), and bottoms at the round-off floor. So the
    composition rule is checked at ``1e-8`` absolute — ``2e-11`` relative to the largest
    entry — with the shape of the convergence asserted so the agreement cannot be a
    coincidence of one step.
    """
    lat = _fodo_ring(ref, k2l=2.0)
    qx, qy = tunes(lat)
    assert 0.3 < qx < 0.45 and 0.3 < qy < 0.45  # stable, away from any low-order line
    composed = second_order_one_turn_map(lat)
    assert composed.symmetry_defect() < 1e-12
    assert np.max(np.abs(composed.T)) > 500.0  # a real ring, not a small-T fixture

    turn = _whole_turn(lat)
    errs = {}
    for step in (1e-3, 1e-4, 3e-5):
        direct = taylor_expand(turn, np.zeros(DIM), step=step)
        errs[step] = np.max(np.abs(direct.T - composed.T))
        assert np.max(np.abs(direct.R - composed.R)) < 1e-4
    assert errs[3e-5] < 1e-8
    assert errs[1e-3] / errs[1e-4] > 1e3  # fourth-order convergence, not luck
    assert errs[1e-4] / errs[3e-5] > 30.0


def test_composition_is_blind_to_no_factor(ref) -> None:
    """Halving or doubling one element's ``T`` breaks the composed map by orders of magnitude.

    The negative control for the gate above: the direct expansion agrees with the
    composed map at ``1e-8``; the same composition with the sextupole's ``T`` scaled by
    ``2`` misses by more than ``1``. A convention error is not a small number.
    """
    lat = _fodo_ring(ref, k2l=2.0)
    maps = second_order_element_maps(lat)
    direct = taylor_expand(_whole_turn(lat), np.zeros(DIM), step=3e-5)
    for factor in (0.5, 2.0):
        wrong = TaylorMap.identity()
        for elem, m in zip(lat.elements, maps, strict=True):
            if isinstance(elem, ThinSextupole):
                m = TaylorMap(m.origin, m.k, m.R, factor * m.T)
            wrong = wrong.then(m)
        assert np.max(np.abs(wrong.T - direct.T)) > 1.0


def test_element_maps_on_the_design_orbit_are_the_matrices_plus_t(ref) -> None:
    lat = _fodo_ring(ref, k2l=2.0)
    for elem, m in zip(lat.elements, second_order_element_maps(lat), strict=True):
        assert np.max(np.abs(m.R - elem.matrix(ref))) < 1e-11, elem
        assert np.max(np.abs(m.k)) == 0.0 and np.max(np.abs(m.origin)) == 0.0
        assert m.symmetry_defect() == 0.0


# ---------------------------------------------------------------------------
# 3. symplecticity as identities
# ---------------------------------------------------------------------------


def test_second_order_symplectic_identity_is_derived() -> None:
    """``M^T J M - J`` at first order in ``u`` is ``2 sum_k u_k (R^T J T_k + T_k^T J R)``.

    Done in two planes with a fully symbolic ``R`` and symmetric ``T``, since the
    statement is dimension-blind: the Jacobian of ``k + R u + T u u`` is
    ``R + 2 sum_k T_k u_k``, and the ``O(u)`` term of the symplectic condition is the
    identity :func:`second_order_symplectic_residual` returns.
    """
    n = 4
    R = sp.Matrix(n, n, sp.symbols(f"r0:{n * n}"))
    u = sp.symbols(f"u0:{n}")
    Tsyms = {}
    for i in range(n):
        for j in range(n):
            for k in range(j, n):
                Tsyms[i, j, k] = Tsyms[i, k, j] = sp.Symbol(f"t{i}{j}{k}")
    J = sp.zeros(n, n)
    for p in range(0, n, 2):
        J[p, p + 1], J[p + 1, p] = 1, -1
    M = R + 2 * sp.Matrix(n, n, lambda i, j: sum(Tsyms[i, j, k] * u[k] for k in range(n)))
    cond = sp.expand(M.T * J * M - J)
    for k in range(n):
        Tk = sp.Matrix(n, n, lambda i, j, k=k: Tsyms[i, j, k])
        S = R.T * J * Tk + Tk.T * J * R
        first_order = cond.applyfunc(lambda e, k=k: e.coeff(u[k], 1))
        # Only the pure-u_k coefficient: strip the other u's, which belong to other k.
        first_order = first_order.subs({uu: 0 for uu in u if uu != u[k]})
        assert sp.simplify(first_order - 2 * S) == sp.zeros(n, n)


def test_identity_holds_for_thin_kicks_and_on_axis_thick_maps(ref) -> None:
    """Exact for a gradient kick, and to the floor for the exact drift and quadrupole.

    The drift and the quadrupole pass **in** ``(zeta, delta)``: on the design orbit
    neither couples anything transverse into ``zeta`` at first order, which is the
    condition under which the non-canonical pair does no harm at this order (the bend
    test below is the case where it does).
    """
    elements = [
        ThinSextupole(12.0),
        ThinSkewSextupole(-7.0),
        ThinQuadrupole(0.9),
        Drift(1.3),
        Quadrupole(0.5, 1.2),
        Quadrupole(0.4, -0.8),
        RFCavity(1e6, 3e6, 0.3),
    ]
    for elem in elements:
        m = _expand(elem, ref)
        assert np.max(np.abs(second_order_symplectic_residual(m.R, m.T))) < 2e-11, elem
    m = _expand(ThinSextupole(12.0), ref)
    assert np.max(np.abs(second_order_symplectic_residual(m.R, m.T))) < 1e-12


def test_the_bend_fails_the_identity_in_delta_by_a_closed_form_and_passes_canonically(
    ref,
) -> None:
    r"""In ``(zeta, delta)`` the sector bend's residual is exactly the non-canonical term.

    ``p_zeta = delta + delta^2/(2 gamma0^2) + ...``, so the symplectic form in
    ``(zeta, delta)`` is ``J + (delta/gamma0^2) J_long`` to first order. For a map that
    leaves ``delta`` alone the identity then acquires, in its ``k = delta`` slice only,

        S_delta = -(1/(2 gamma0^2)) (v e_delta^T - e_delta v^T),   v = R[zeta, transverse],

    which vanishes for anything with no transverse-to-zeta coupling and does not for a
    bend (``R51 = -sin theta``, ``R52 = -rho (1 - cos theta)``). Measured on a 1 m, 0.12 rad
    bend: ``1.5e-4`` where the floor is ``1e-12`` — the residual is physics, and it goes
    away in ``(zeta, p_zeta)``.
    """
    bend = Dipole(1.0, 0.12)
    m = _expand(bend, ref)
    S = second_order_symplectic_residual(m.R, m.T)
    assert np.max(np.abs(S)) > 1e-4  # decisively not the floor
    assert np.max(np.abs(S[:, :, :DELTA])) < 1e-11  # every other slice is clean

    v = m.R[ZETA].copy()
    v[ZETA] = v[DELTA] = 0.0
    e6 = np.eye(DIM)[DELTA]
    predicted = -(np.outer(v, e6) - np.outer(e6, v)) / (2 * ref.gamma0**2)
    assert np.max(np.abs(S[:, :, DELTA] - predicted)) < 1e-11

    mc = taylor_expand(canonical_map(lambda s: bend.track(s, ref), ref), np.zeros(DIM))
    assert np.max(np.abs(second_order_symplectic_residual(mc.R, mc.T))) < 1e-11
    # And the linear part, for orientation: R itself is symplectic either way.
    assert np.max(np.abs(m.R.T @ J6 @ m.R - J6)) < 1e-12


# ---------------------------------------------------------------------------
# the object: evaluation, operators, refusals
# ---------------------------------------------------------------------------


def test_evaluation_on_a_bunch_matches_per_particle_and_the_tracked_map(ref) -> None:
    """Evaluating the quadratic map on a ``(6, n)`` bunch equals the per-particle loop, and
    departs from ``track()`` at third order in the amplitude."""
    m = _expand(Drift(2.0), ref)
    rng = np.random.default_rng(3)
    bunch = 1e-3 * rng.standard_normal((DIM, 40))
    vec = m(bunch)
    for n in range(bunch.shape[1]):
        assert np.allclose(vec[:, n], m(bunch[:, n]), rtol=0, atol=1e-16)
    tracked = Drift(2.0).track(bunch, ref)
    miss = np.max(np.abs(tracked - vec))
    assert miss < 1e-7  # L px (px^2 + py^2)/2 at a few 1e-3: cubic
    tracked2 = Drift(2.0).track(2 * bunch, ref)
    miss2 = np.max(np.abs(tracked2 - m(2 * bunch)))
    assert 6.0 < miss2 / miss < 10.0  # eight-fold for a doubling: the leftover is cubic


def test_identity_and_operator_order(ref) -> None:
    sext = _expand(ThinSextupole(3.0), ref)
    drift = _expand(Drift(1.0), ref)
    ident = TaylorMap.identity()
    assert np.array_equal(ident.then(sext).T, sext.T)
    beam_order = sext.then(drift)  # sextupole first, then the drift
    matrix_order = drift @ sext
    assert np.array_equal(beam_order.T, matrix_order.T)
    assert np.array_equal(beam_order.R, drift.R @ sext.R)
    # Order matters: the drift's R carries the kick's T forward into x.
    assert abs(beam_order.T[X, X, X] - (-1.5)) < 1e-11
    assert abs(sext.T[X, X, X]) < 1e-12


def test_compose_refuses_mismatched_expansion_points(ref) -> None:
    a = TaylorMap.identity(np.array([1e-3, 0, 0, 0, 0, 0]))
    b = TaylorMap.identity()
    with pytest.raises(ValueError, match="cannot compose"):
        compose(b, a)
    with pytest.raises(ValueError, match="shape"):
        TaylorMap(np.zeros(6), np.zeros(6), np.eye(6), np.zeros((6, 6)))
    with pytest.raises(ValueError, match="vectorised=False"):
        taylor_expand(lambda s: np.asarray(s)[:, 0], np.zeros(6))
    with pytest.raises(ValueError, match="step"):
        taylor_expand(lambda s: s, np.zeros(6), step=0.0)


# ---------------------------------------------------------------------------
# on a steered orbit
# ---------------------------------------------------------------------------


def test_on_a_steered_orbit_r_feeds_down_and_t_does_not(ref) -> None:
    """``R`` picks up ``-k2l x_co`` exactly as I2/I3 say; the sextupole's ``T`` is unchanged.

    The sharpest single statement of feed-down: a thin sextupole's second derivative is a
    constant, so its ``T`` is orbit-independent to round-off while its Jacobian is not.
    Every element's ``R`` agrees with :func:`linearised_element_maps` — the first-order
    sibling, differenced with its own stencil and step — to the shared floor.
    """
    k2l = 2.0
    lat = Lattice([Corrector(kick_x=2e-4, kick_y=-1e-4)] + _fodo_ring(ref, k2l).elements, ref)
    orbit = closed_orbit_nonlinear(lat)
    maps = second_order_element_maps(lat)
    first = linearised_element_maps(lat)
    assert max(np.max(np.abs(a.R - b)) for a, b in zip(maps, first, strict=True)) < 1e-10

    state = np.array([*orbit, 0.0, 0.0])
    for elem, m in zip(lat.elements, maps, strict=True):
        assert np.allclose(m.origin, state, rtol=0, atol=0)
        if isinstance(elem, ThinSextupole):
            x_co, y_co = state[X], state[Y]
            assert abs(m.R[PX, X] + k2l * x_co) < 1e-11
            assert abs(m.R[PX, Y] - k2l * y_co) < 1e-11
            assert abs(m.R[PY, X] - k2l * y_co) < 1e-11
            assert abs(m.T[PX, X, X] + k2l / 2) < 1e-12
            assert abs(m.T[PY, X, Y] - k2l / 2) < 1e-12
        state = elem.track(state, ref)
        assert np.array_equal(m.k, state)

    turn = second_order_one_turn_map(lat)
    closes = [X, PX, Y, PY, DELTA]
    assert np.max(np.abs(turn.k[closes] - turn.origin[closes])) < 1e-12  # closes on the orbit
    # zeta does not close: the 4D orbit is longer than the design one and nothing (no
    # RF) pulls the arrival time back — N5/I4's point, visible here as k != origin.
    assert abs(turn.k[ZETA] - turn.origin[ZETA]) > 1e-5
    assert np.max(np.abs(turn.R - linearised_one_turn_map(lat))) < 1e-9


# ---------------------------------------------------------------------------
# 4. shipped quantities as projections of T
# ---------------------------------------------------------------------------


def _chromaticity_from(R: np.ndarray, dR: np.ndarray) -> tuple[float, float]:
    """``dQ/ddelta = -tr(dR_plane/ddelta) / (4 pi sin mu)`` from ``cos mu = tr/2``."""
    out = []
    for a, b in ((X, PX), (Y, PY)):
        mu = math.acos(0.5 * (R[a, a] + R[b, b]))
        if R[a, b] < 0:
            mu = 2 * math.pi - mu
        out.append(-(dR[a, a] + dR[b, b]) / (4 * math.pi * math.sin(mu)))
    return out[0], out[1]


def test_chromaticity_is_the_momentum_column_of_t_on_a_dispersion_free_ring(ref) -> None:
    """``dR/ddelta = 2 T[:, :, delta]`` exactly in the symmetric convention, so ``Q'`` follows.

    Three routes on a bend-free thick FODO: ``T``; I3's finite difference of
    :func:`linearised_one_turn_map` at ``+-delta``; and F2's integral
    :func:`natural_chromaticity`, which converges onto the other two as its trapezoid is
    refined (``6e-6`` off at 64 slices, ``1.4e-9`` at 4096 — the integral's floor, not
    ``T``'s).
    """
    lat = Lattice([Quadrupole(0.3, 1.2), Drift(1.0), Quadrupole(0.3, -1.2), Drift(1.0)] * 4, ref)
    turn = second_order_one_turn_map(lat)
    from_t = _chromaticity_from(turn.R, 2 * turn.T[:, :, DELTA])
    d = 1e-4
    dR = (linearised_one_turn_map(lat, delta=+d) - linearised_one_turn_map(lat, delta=-d)) / (2 * d)
    from_fd = _chromaticity_from(turn.R, dR)
    coarse = natural_chromaticity(lat)
    fine = natural_chromaticity(lat, slices=4096)
    for plane in (0, 1):
        assert abs(from_t[plane] - from_fd[plane]) < 1e-8
        assert abs(from_t[plane] - fine[plane]) < 1e-8
        assert abs(from_t[plane] - coarse[plane]) > 1e-6  # the coarse integral is the outlier
    assert abs(from_t[0] + 0.2806) < 1e-3  # orientation: the natural value, negative


def test_chromaticity_with_bends_runs_along_the_dispersion(ref) -> None:
    """``dR_co/ddelta = 2 sum_m T[:, :, m] D_m`` with ``D = (D_x, D_px, D_y, D_py, 0, 1)``.

    With bends the off-momentum orbit moves, so the momentum derivative of the map *about
    the closed orbit* is a directional derivative along ``(D, 1)``, not the plain
    ``delta`` column. Both against I3's finite difference and F2/M1's
    :func:`chromaticity`; and the plain column alone is shown to miss by ``1e-2``.
    """
    cell = [
        Quadrupole(0.3, 1.2),
        Drift(0.5),
        Dipole(1.0, 0.1),
        Drift(0.5),
        Quadrupole(0.3, -1.2),
        Drift(1.0),
    ]
    lat = Lattice(cell * 4, ref)
    turn = second_order_one_turn_map(lat)
    tw = closed_twiss(lat)
    direction = np.zeros(DIM)
    direction[X], direction[PX], direction[DELTA] = tw.disp_x, tw.disp_px, 1.0
    from_t = _chromaticity_from(turn.R, 2 * np.einsum("ijm,m->ij", turn.T, direction))
    d = 1e-4
    dR = (linearised_one_turn_map(lat, delta=+d) - linearised_one_turn_map(lat, delta=-d)) / (2 * d)
    from_fd = _chromaticity_from(turn.R, dR)
    integral = chromaticity(lat, slices=1024)
    column_only = _chromaticity_from(turn.R, 2 * turn.T[:, :, DELTA])
    for plane in (0, 1):
        assert abs(from_t[plane] - from_fd[plane]) < 1e-7
        assert abs(from_t[plane] - integral[plane]) < 1e-7
        assert abs(from_t[plane] - column_only[plane]) > 2e-3


def test_second_order_dispersion_is_the_fixed_point_of_t(ref) -> None:
    r"""``x_co(delta) = D delta + (1/2) D2 delta^2`` follows from ``R`` and ``T`` alone.

    Order ``delta``: ``(I - R4) D = R[:4, delta]`` (Stage 1's dispersion). Order
    ``delta^2``: ``(I - R4) D2/2 = [T(Dhat, Dhat)]_transverse`` with ``Dhat = (D, 0, 1)``.
    Compared with M3's :func:`second_order_dispersion`, which twice-differences the
    tracked orbit and is MAD-X-validated: agreement ``1e-8`` on ``ddisp_x = 1.65``.
    """
    cell = [
        Quadrupole(0.3, 1.2),
        Drift(0.5),
        Dipole(1.0, 0.1),
        Drift(0.5),
        Quadrupole(0.3, -1.2),
        Drift(1.0),
    ]
    lat = Lattice(cell * 4, ref)
    turn = second_order_one_turn_map(lat)
    R4 = turn.R[:4, :4]
    D = np.linalg.solve(np.eye(4) - R4, turn.R[:4, DELTA])
    dhat = np.zeros(DIM)
    dhat[:4], dhat[DELTA] = D, 1.0
    D2 = 2 * np.linalg.solve(np.eye(4) - R4, np.einsum("ijk,j,k->i", turn.T, dhat, dhat)[:4])
    m3 = second_order_dispersion(lat)[0]
    assert np.allclose(D, [m3.disp_x, m3.disp_px, m3.disp_y, m3.disp_py], rtol=0, atol=1e-6)
    assert np.allclose(D2, [m3.ddisp_x, m3.ddisp_px, m3.ddisp_y, m3.ddisp_py], rtol=0, atol=1e-7)
    assert abs(D2[0]) > 1.0  # a real second-order dispersion, not a zero matched to a zero


def test_first_order_driving_terms_are_read_off_t(ref) -> None:
    r"""O4's five sextupole terms from ``T``, through the generator and the normal form.

    For a thin kick at the entrance of an otherwise linear ring the transverse block obeys
    ``T = R G`` with ``G`` the kick's own quadratic part, and a gradient kick is
    ``Delta p = -dV/dx``, i.e. ``G u u = J grad V``. So ``grad V = -J R^-1 T u u``, ``V``
    is recovered by Euler's identity ``V = (u . grad V) / 3``, and it must come back
    **curl-free and equal to** ``k2l (x^3/6 - x y^2/2)`` — both asserted. Re-expanded in
    O1's normalised coordinates ``u = W w`` and the resonance basis
    ``h = u_hat + i p_hat``, its coefficients divided by ``exp(-2 pi i [(j-k) Q_x +
    (l-m) Q_y]) - 1`` are O4's ``f_jklm`` with O4's generator being ``-V`` (the sign the
    Lie machinery attaches to a potential): the ratio is ``-1`` to ``1e-10`` on all five.
    """
    k2l = 3.0
    cell = [ThinQuadrupole(1.0), Drift(1.0), ThinQuadrupole(-0.8), Drift(1.2)]
    lat = Lattice([ThinSextupole(k2l)] + cell * 4, ref)
    turn = second_order_one_turn_map(lat)
    nf = normal_form(turn.R, method="4d")

    R4, T4 = turn.R[:4, :4], turn.T[:4, :4, :4]
    J4 = J6[:4, :4]
    G = np.einsum("ai,ijk->ajk", np.linalg.inv(R4), T4)
    grad = -np.einsum("ab,bjk->ajk", J4, G)
    assert np.max(np.abs(grad - np.swapaxes(grad, 0, 1))) < 1e-10  # a gradient: curl-free

    z = sp.symbols("x px y py")
    V_recovered = sp.expand(
        sum(
            z[a] * sum(grad[a, j, k] * z[j] * z[k] for j in range(4) for k in range(4))
            for a in range(4)
        )
        / 3
    )
    V = k2l * (z[0] ** 3 / 6 - z[0] * z[2] ** 2 / 2)
    assert max(abs(c) for c in sp.Poly(V_recovered - V, *z).coeffs()) < 1e-10

    w = sp.symbols("uh ph vh qh")
    lab = sp.Matrix(nf.w[:4, :4]) * sp.Matrix(w)
    F = sp.expand((-V_recovered).subs(dict(zip(z, lab, strict=True)), simultaneous=True))
    hx, hxb, hy, hyb = sp.symbols("hx hxb hy hyb")
    F = sp.expand(
        F.subs(
            {
                w[0]: (hx + hxb) / 2,
                w[1]: (hx - hxb) / (2 * sp.I),
                w[2]: (hy + hyb) / 2,
                w[3]: (hy - hyb) / (2 * sp.I),
            },
            simultaneous=True,
        )
    )
    P = sp.Poly(F, hx, hxb, hy, hyb)
    qx, qy = nf.tunes
    shipped = resonance_driving_terms(lat)
    for key in ("f3000", "f2100", "f1020", "f1011", "f1002"):
        j, k, l, m = (int(c) for c in key[1:])
        coeff = complex(P.coeff_monomial(hx**j * hxb**k * hy**l * hyb**m))
        f = coeff / (cmath.exp(-2j * math.pi * ((j - k) * qx + (l - m) * qy)) - 1)
        assert abs(f / shipped[key] - 1) < 1e-9, (key, f, shipped[key])
        assert abs(shipped[key]) > 0.1  # every term is far from zero on this fixture
