"""Analytic checks for linear dispersion (matched + propagated).

Dispersion is the first-order off-momentum closed orbit, ``D = dx/ddelta``. The
matched value solves the closure condition ``D = M4 @ D + d`` exactly; that
defining equation (independent of how the code computes it) is the primary check,
backed by a sympy exact re-derivation and the physical invariants of a symmetric
arc cell.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    DELTA,
    PX,
    Dipole,
    Drift,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    X,
    closed_twiss,
    propagate_twiss,
)
from accsim.coords import PY, Y

F_FOCAL = 2.5
L_BEND = 1.0
ANGLE = 0.15


def _arc_cell() -> list:
    """Symmetric arc FODO: half-F, bend, D, bend, half-F (dispersion-generating)."""
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


def test_matched_dispersion_satisfies_closure(ref: ReferenceParticle) -> None:
    # The defining property: the matched 4D dispersion reproduces itself after one
    # turn, D = M4 @ D + d. This is independent of the (I - M4)^-1 solve.
    lat = Lattice(_arc_cell(), ref)
    M = lat.one_turn_matrix()
    tw = closed_twiss(lat)
    D = np.array([tw.disp_x, tw.disp_px, tw.disp_y, tw.disp_py])
    idx = [X, PX, Y, PY]
    m4 = M[np.ix_(idx, idx)]
    d = M[idx, DELTA]
    np.testing.assert_allclose(m4 @ D + d, D, rtol=1e-12, atol=1e-14)


def test_matched_dispersion_matches_symbolic(ref: ReferenceParticle) -> None:
    sp = pytest.importorskip("sympy")
    # Re-derive D = (I - M4)^-1 d with exact rational arithmetic from hand-built
    # symbolic element matrices — independent of numpy's float solve.
    f, lb, ang = sp.Rational(5, 2), sp.Integer(1), sp.Rational(15, 100)
    h = ang / lb
    c, s = sp.cos(ang), sp.sin(ang)

    def thinq(k1l: sp.Expr) -> sp.Matrix:
        m = sp.eye(4)
        m[1, 0] = -k1l  # px -= k1l x
        m[3, 2] = k1l  # py += k1l y
        return m

    bend4 = sp.Matrix([[c, s / h, 0, 0], [-h * s, c, 0, 0], [0, 0, 1, lb], [0, 0, 0, 1]])
    kick = sp.Matrix([(1 - c) / h, s, 0, 0])  # [R16, R26, R36, R46]
    # Build the one-turn 4x4 and accumulated dispersive kick (right-to-left).
    m4 = sp.eye(4)
    dvec = sp.zeros(4, 1)
    for blk, k in [
        (thinq(sp.Rational(1, 2) / f), sp.zeros(4, 1)),
        (bend4, kick),
        (thinq(-1 / f), sp.zeros(4, 1)),
        (bend4, kick),
        (thinq(sp.Rational(1, 2) / f), sp.zeros(4, 1)),
    ]:
        m4 = blk * m4
        dvec = blk * dvec + k
    D_sym = (sp.eye(4) - m4).inv() * dvec
    expected = np.array(D_sym.evalf(), dtype=float).ravel()

    tw = closed_twiss(Lattice(_arc_cell(), ref))
    got = np.array([tw.disp_x, tw.disp_px, tw.disp_y, tw.disp_py])
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_dispersion_physical_invariants(ref: ReferenceParticle) -> None:
    tw = closed_twiss(Lattice(_arc_cell(), ref))
    assert tw.disp_x > 0.0  # outward dispersion in an arc
    assert tw.disp_px == pytest.approx(0.0, abs=1e-12)  # parallel at symmetry point
    assert tw.disp_y == pytest.approx(0.0, abs=1e-14)  # no vertical bending
    assert tw.disp_py == pytest.approx(0.0, abs=1e-14)


def test_dispersion_is_periodic_through_cell(ref: ReferenceParticle) -> None:
    lat = Lattice(_arc_cell(), ref)
    pts = propagate_twiss(lat, closed_twiss(lat))
    start, end = pts[0], pts[-1]
    assert end.disp_x == pytest.approx(start.disp_x)
    assert end.disp_px == pytest.approx(start.disp_px, abs=1e-12)
    # Dispersion peaks at the F quad and troughs at the D quad, like beta_x.
    dx = [p.disp_x for p in pts]
    assert dx[0] == pytest.approx(max(dx))
    assert min(dx) < dx[0]


def test_quad_only_lattice_has_zero_dispersion(ref: ReferenceParticle) -> None:
    # No bending magnet => no dispersion anywhere.
    lat = Lattice([ThinQuadrupole(0.4), Drift(1.0), ThinQuadrupole(-0.4), Drift(1.0)], ref)
    for tw in propagate_twiss(lat, closed_twiss(lat)):
        assert tw.disp_x == pytest.approx(0.0, abs=1e-14)
        assert tw.disp_px == pytest.approx(0.0, abs=1e-14)
