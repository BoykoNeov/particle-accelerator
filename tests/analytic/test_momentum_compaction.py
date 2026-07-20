"""Analytic checks for the momentum-compaction factor ``alpha_c`` (Stage 3).

``alpha_c = (1/C) ∮ D_x h ds`` is the fractional circumference change per unit
momentum deviation — purely geometric (no ``gamma0``).

Since D4 the *default* implementation is the exact **symplecticity identity**
``alpha_c = 1/gamma0^2 - (R51 Dx + R52 Dpx + R56)/C``, which reads the one-turn
*longitudinal row*; the physical integral survives as ``method="quadrature"``.
That makes the arms here:

1. **quadrature vs identity** — the integral touches only the
   dispersion-generating matrix entries, the identity only the longitudinal row,
   so the two are genuinely disjoint routes to the same number and a sign error
   in either fails the comparison; and
2. an exact **sympy re-derivation** of both paths on a thick-dipole arc cell.

**Every assertion below that compares against** :func:`_identity_alpha_c` **must
pass** ``method="quadrature"`` — comparing the default against the identity is a
tautology (it is literally the same code) and would silently test nothing. Only
:func:`test_default_is_the_exact_identity` compares default-to-identity, and it
is a regression guard on the D4 switch, *not* one of the two routes.

The drift/no-bend limit (``D=0``, ``R56=C/gamma0^2`` ⇒ ``alpha_c=0``) is pinned
too, though it cannot test *sign* (both sides are zero there) — the bending cases
do that.
"""

from __future__ import annotations

import pytest

from accsim import (
    DELTA,
    PX,
    ZETA,
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinQuadrupole,
    X,
    closed_twiss,
    momentum_compaction,
    slip_factor,
)

F_FOCAL = 2.5
L_BEND = 1.0
ANGLE = 0.15


def _arc_cell() -> list:
    """Symmetric arc FODO with **thick** bends (nontrivial D_x(s) integral)."""
    return [
        ThinQuadrupole(0.5 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(-1.0 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(938.27208816e6, 5.0)


def _identity_alpha_c(lat: Lattice) -> float:
    """alpha_c via the one-turn longitudinal row on the matched dispersion orbit.

    Over one turn on the dispersion orbit ``(x, px) = (D_x, D_px) delta``,
    ``zeta`` slips by ``(R51 D_x + R52 D_px + R56) delta``; that slip equals
    ``(1/gamma0^2 - alpha_c) C delta`` (the drift limit ``D=0`` gives the
    ``1/gamma0^2`` piece). Uses matrix entries disjoint from the D-generating ones.
    """
    M = lat.one_turn_matrix()
    tw = closed_twiss(lat)
    r = M[ZETA, X] * tw.disp_x + M[ZETA, PX] * tw.disp_px + M[ZETA, DELTA]
    return 1.0 / lat.ref.gamma0**2 - r / lat.length


def test_default_is_the_exact_identity(ref: ReferenceParticle) -> None:
    # D4 regression guard: the default route IS the identity, to machine precision
    # (no quadrature error left). Not an independent check -- see the module docstring.
    lat = Lattice(_arc_cell(), ref)
    assert momentum_compaction(lat) == pytest.approx(_identity_alpha_c(lat), rel=1e-14)
    # slices is inert on the identity path: refining it changes nothing at all.
    assert momentum_compaction(lat, slices=4) == momentum_compaction(lat, slices=4096)


def test_rejects_unknown_method(ref: ReferenceParticle) -> None:
    with pytest.raises(ValueError, match="identity"):
        momentum_compaction(Lattice(_arc_cell(), ref), method="simpson")


def test_matches_symplecticity_identity(ref: ReferenceParticle) -> None:
    # Independent path: longitudinal row + dispersion, no curvature integral.
    lat = Lattice(_arc_cell(), ref)
    got = momentum_compaction(lat, method="quadrature")
    expected = _identity_alpha_c(lat)
    # Trapezoid quadrature of the smooth D_x(s) converges as O((h ds)^2); at 64
    # slices this is ~1e-6, while any sign/coefficient error would be O(1).
    assert got == pytest.approx(expected, rel=1e-5)


def test_identity_is_exact_at_high_slice_count(ref: ReferenceParticle) -> None:
    # Refining the quadrature drives the integral onto the exact identity value.
    lat = Lattice(_arc_cell(), ref)
    assert momentum_compaction(lat, slices=4096, method="quadrature") == pytest.approx(
        _identity_alpha_c(lat), rel=1e-8
    )


def test_matches_symbolic_derivation(ref: ReferenceParticle) -> None:
    sp = pytest.importorskip("sympy")
    # Exact rational re-derivation of BOTH paths from hand-built symbolic element
    # matrices; assert they are algebraically identical, then that accsim matches.
    f, lb, ang, g0 = sp.Rational(5, 2), sp.Integer(1), sp.Rational(15, 100), sp.Integer(5)
    h = ang / lb
    s = sp.Symbol("s", nonnegative=True)

    def thinq2(k1l: sp.Expr) -> sp.Matrix:
        return sp.Matrix([[1, 0, 0], [-k1l, 1, 0], [0, 0, 1]])

    def bend2(length: sp.Expr) -> sp.Matrix:
        a = h * length
        c, si = sp.cos(a), sp.sin(a)
        return sp.Matrix([[c, si / h, (1 - c) / h], [-h * si, c, si], [0, 0, 1]])

    # (x, px, delta) one-turn map and matched dispersion.
    seq = [
        thinq2(sp.Rational(1, 2) / f),
        bend2(lb),
        thinq2(-1 / f),
        bend2(lb),
        thinq2(sp.Rational(1, 2) / f),
    ]
    m3 = sp.eye(3)
    for e in seq:
        m3 = e * m3
    m2, d = m3[0:2, 0:2], m3[0:2, 2]
    disp = (sp.eye(2) - m2).inv() * d
    dx0, dpx0 = disp[0], disp[1]

    # Path A: (1/C) ∮ D_x h ds through the two thick dipole bodies (quads thin).
    def dx_after(pre: list) -> tuple:
        v = sp.Matrix([dx0, dpx0, 1])
        for e in pre:
            v = e * v
        return v[0], v[1]

    total = sp.Integer(0)
    for pre in (
        [thinq2(sp.Rational(1, 2) / f)],
        [thinq2(sp.Rational(1, 2) / f), bend2(lb), thinq2(-1 / f)],
    ):
        dxin, dpxin = dx_after(pre)
        dx_s = (bend2(s) * sp.Matrix([dxin, dpxin, 1]))[0]
        total += sp.integrate(dx_s, (s, 0, lb))
    alpha_A = sp.nsimplify(h * total / (2 * lb))

    # Path B: matrix identity via the one-turn zeta row.
    def bend4(length: sp.Expr) -> sp.Matrix:
        a = h * length
        c, si = sp.cos(a), sp.sin(a)
        m = sp.eye(4)
        m[0, 0], m[0, 1], m[0, 2] = c, si / h, (1 - c) / h
        m[1, 0], m[1, 1], m[1, 2] = -h * si, c, si
        m[3, 0], m[3, 1] = -si, (c - 1) / h
        m[3, 2] = si / h - length + length / g0**2
        return m

    def thinq4(k1l: sp.Expr) -> sp.Matrix:
        m = sp.eye(4)
        m[1, 0] = -k1l
        return m

    seq4 = [
        thinq4(sp.Rational(1, 2) / f),
        bend4(lb),
        thinq4(-1 / f),
        bend4(lb),
        thinq4(sp.Rational(1, 2) / f),
    ]
    m4 = sp.eye(4)
    for e in seq4:
        m4 = e * m4
    r51, r52, r56 = m4[3, 0], m4[3, 1], m4[3, 2]
    alpha_B = 1 / g0**2 - (r51 * dx0 + r52 * dpx0 + r56) / (2 * lb)

    # The two symbolic paths are algebraically identical (proves signs + coeffs).
    assert sp.simplify(alpha_A - alpha_B) == 0
    expected = float(alpha_A)
    lat = Lattice(_arc_cell(), ref)
    # Both routes against the symbolic truth: the exact default to near machine
    # precision, the trapezoid to its O((h ds)^2) quadrature error at 64 slices.
    assert momentum_compaction(lat) == pytest.approx(expected, rel=1e-12)
    assert momentum_compaction(lat, method="quadrature") == pytest.approx(expected, rel=1e-5)


def test_no_bend_lattice_has_zero_alpha_c(ref: ReferenceParticle) -> None:
    # No curvature anywhere => no path lengthening => alpha_c == 0 exactly.
    lat = Lattice([ThinQuadrupole(0.4), Drift(1.0), ThinQuadrupole(-0.4), Drift(1.0)], ref)
    assert momentum_compaction(lat) == pytest.approx(0.0, abs=1e-15)
    # Drift-limit anchor: slip factor is pure velocity term -1/gamma0^2 there.
    assert slip_factor(lat) == pytest.approx(-1.0 / ref.gamma0**2, abs=1e-15)


def test_alpha_c_positive_for_normal_arc(ref: ReferenceParticle) -> None:
    # Outward dispersion in a focusing arc => longer high-momentum orbit => alpha_c > 0.
    assert momentum_compaction(Lattice(_arc_cell(), ref)) > 0.0


def test_slip_factor_definition(ref: ReferenceParticle) -> None:
    # eta = alpha_c - 1/gamma0^2, single-sourced from the reference gamma0.
    lat = Lattice(_arc_cell(), ref)
    assert slip_factor(lat) == pytest.approx(
        momentum_compaction(lat) - 1.0 / ref.gamma0**2, rel=1e-14
    )


def test_identity_holds_for_thick_quad_arc(ref: ReferenceParticle) -> None:
    # The integral<->identity consistency (the CI sign net) must survive finite-
    # length quads, which transport beta AND dispersion through their bodies. This
    # is NOT a thin-vs-thick alpha_c comparison: the thick variant has a different
    # circumference (2.6 m vs 2.0 m) and focusing profile, so its alpha_c genuinely
    # differs -- we only check each lattice reproduces its own matrix identity.
    thin = Lattice(_arc_cell(), ref)
    assert momentum_compaction(thin, method="quadrature") == pytest.approx(
        _identity_alpha_c(thin), rel=1e-5
    )
    kq = 1.0 / (F_FOCAL * 0.2)
    thick = Lattice(
        [
            Quadrupole(0.2, 0.5 * kq),
            Dipole(L_BEND, ANGLE),
            Quadrupole(0.2, -kq),
            Dipole(L_BEND, ANGLE),
            Quadrupole(0.2, 0.5 * kq),
        ],
        ref,
    )
    assert momentum_compaction(thick, method="quadrature") == pytest.approx(
        _identity_alpha_c(thick), rel=1e-5
    )
