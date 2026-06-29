"""Analytic checks for the Quadrupole (thick) and ThinQuadrupole elements.

Expected values are hand-computed or re-derived symbolically as ``exp(L*A)`` from
the Hamiltonian generator ``A`` — never produced by re-running the matrix under
test. The generator route also guarantees symplecticity structurally.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    DELTA,
    PROTON_MASS_EV,
    PX,
    PY,
    ZETA,
    Drift,
    Lattice,
    Particle,
    Quadrupole,
    ReferenceParticle,
    ThinQuadrupole,
    Tracker,
    X,
    Y,
)
from accsim.symplectic import is_symplectic


# --- gold pin: the runtime matrix equals exp(L*A) of the Hamiltonian generator ---
def test_quadrupole_matrix_matches_symbolic_exponential(proton_gamma5: ReferenceParticle) -> None:
    sp = pytest.importorskip("sympy")

    L_val, k1_val = 0.45, 1.7
    g0_val = proton_gamma5.gamma0

    L, k1, g0 = sp.symbols("L k1 gamma0", positive=True)
    # Generator A of state' = A @ state for (x, px, y, py, zeta, delta):
    #   x'' + k1 x = 0,  y'' - k1 y = 0,  zeta' = delta / gamma0^2.
    A = sp.Matrix(
        [
            [0, 1, 0, 0, 0, 0],
            [-k1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, k1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1 / g0**2],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    M_sym = (A * L).exp()
    subs = {L: L_val, k1: k1_val, g0: g0_val}
    expected = np.array(sp.simplify(M_sym).subs(subs).evalf(), dtype=float)

    got = Quadrupole(L_val, k1_val).matrix(proton_gamma5)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_quadrupole_thick_blocks_hand_computed(proton_gamma5: ReferenceParticle) -> None:
    L, k1 = 0.6, 2.5
    w = math.sqrt(k1)
    M = Quadrupole(L, k1).matrix(proton_gamma5)

    # Focusing plane (x): trig block.
    assert M[X, X] == pytest.approx(math.cos(w * L))
    assert M[X, PX] == pytest.approx(math.sin(w * L) / w)
    assert M[PX, X] == pytest.approx(-w * math.sin(w * L))
    assert M[PX, PX] == pytest.approx(math.cos(w * L))
    # Defocusing plane (y): hyperbolic block, same w.
    assert M[Y, Y] == pytest.approx(math.cosh(w * L))
    assert M[Y, PY] == pytest.approx(math.sinh(w * L) / w)
    assert M[PY, Y] == pytest.approx(w * math.sinh(w * L))
    assert M[PY, PY] == pytest.approx(math.cosh(w * L))
    # Longitudinal slip identical to a drift of the same length.
    assert M[ZETA, DELTA] == pytest.approx(L / proton_gamma5.gamma0**2)


def test_quadrupole_sign_flip_swaps_planes(proton_gamma5: ReferenceParticle) -> None:
    # k1 < 0 must focus y and defocus x: the (x,px) block of Q(-k1) equals the
    # (y,py) block of Q(+k1) and vice versa.
    L, k1 = 0.6, 2.5
    Mp = Quadrupole(L, k1).matrix(proton_gamma5)
    Mm = Quadrupole(L, -k1).matrix(proton_gamma5)
    np.testing.assert_allclose(Mm[np.ix_([X, PX], [X, PX])], Mp[np.ix_([Y, PY], [Y, PY])])
    np.testing.assert_allclose(Mm[np.ix_([Y, PY], [Y, PY])], Mp[np.ix_([X, PX], [X, PX])])


def test_quadrupole_zero_strength_is_drift(proton_gamma5: ReferenceParticle) -> None:
    L = 1.3
    np.testing.assert_allclose(
        Quadrupole(L, 0.0).matrix(proton_gamma5), Drift(L).matrix(proton_gamma5)
    )


def test_quadrupole_thick_is_symplectic(proton_gamma5: ReferenceParticle) -> None:
    for k1 in (2.5, -2.5, 0.0):
        assert is_symplectic(Quadrupole(0.7, k1).matrix(proton_gamma5))


# --- thin lens -----------------------------------------------------------------
def test_thin_quadrupole_is_pure_kick(proton_gamma5: ReferenceParticle) -> None:
    k1l = 0.8
    M = ThinQuadrupole(k1l).matrix(proton_gamma5)
    expected = np.eye(6)
    expected[PX, X] = -k1l  # focusing in x
    expected[PY, Y] = k1l  # defocusing in y
    np.testing.assert_allclose(M, expected)
    assert M[ZETA, DELTA] == 0.0  # zero length -> no longitudinal slip


def test_thin_quadrupole_is_symplectic(proton_gamma5: ReferenceParticle) -> None:
    assert is_symplectic(ThinQuadrupole(0.8).matrix(proton_gamma5))
    assert is_symplectic(ThinQuadrupole(-0.8).matrix(proton_gamma5))


def test_thin_lens_focuses_parallel_ray(proton_gamma5: ReferenceParticle) -> None:
    # Textbook lens law: a ray parallel to the axis (px = 0) crosses the axis a
    # focal length f downstream of a focusing thin lens.
    f = 2.0
    quad = ThinQuadrupole(1.0 / f)  # k1l = 1/f
    x0 = 1.0e-3
    p = Particle(x=x0, px=0.0)
    out = Tracker(Lattice([quad, Drift(f)], proton_gamma5)).track(p)
    assert out.x == pytest.approx(0.0, abs=1e-15)  # focused to the axis
    assert out.px == pytest.approx(-x0 / f)  # angular kick = -x0/f
    # Same lens defocuses the other plane: y grows instead of focusing.
    py = Particle(y=x0, py=0.0)
    outy = Tracker(Lattice([quad, Drift(f)], proton_gamma5)).track(py)
    assert outy.y == pytest.approx(x0 * (1.0 + 1.0))  # y0 + f*(+x0/f) = 2 x0


def test_thick_quad_approaches_thin_lens(proton_gamma5: ReferenceParticle) -> None:
    # As L -> 0 at fixed integrated strength k1l = k1*L, the thick quad's kick
    # converges to the thin-lens kick. Expanding R21 = -w sin(wL) with
    # w = sqrt(k1l/L) gives -k1l + k1l^2 L/6 - k1l^3 L^2/120 + ..., so the
    # residual is the thin-lens correction series (leading term O(L)).
    k1l = 0.5
    thin_R21 = -k1l
    for L in (0.1, 0.01, 0.001):
        thick_R21 = Quadrupole(L, k1l / L).matrix(proton_gamma5)[PX, X]
        residual = thick_R21 - thin_R21
        expected = k1l**2 * L / 6.0 - k1l**3 * L**2 / 120.0
        assert residual == pytest.approx(expected, rel=1e-4)


# --- composition order (roadmap: distinguish M_last @ ... @ M_first) -----------
def test_quad_drift_composition_is_order_sensitive(proton_gamma5: ReferenceParticle) -> None:
    # Unlike two drifts, a quad and a drift do NOT commute. This pins the
    # right-to-left product order: for [A, B] (A entered first) the lattice must
    # return M_B @ M_A, which differs from M_A @ M_B.
    drift = Drift(1.5)
    quad = ThinQuadrupole(0.7)
    Md = drift.matrix(proton_gamma5)
    Mq = quad.matrix(proton_gamma5)

    drift_then_quad = Lattice([drift, quad], proton_gamma5).transfer_matrix()
    quad_then_drift = Lattice([quad, drift], proton_gamma5).transfer_matrix()

    np.testing.assert_allclose(drift_then_quad, Mq @ Md)  # element-first => rightmost
    np.testing.assert_allclose(quad_then_drift, Md @ Mq)
    # The two orders genuinely differ (guards against a commuting-only test).
    assert not np.allclose(drift_then_quad, quad_then_drift)


def test_defocusing_thin_quad_focuses_vertical(proton_gamma5: ReferenceParticle) -> None:
    # A negative k1l focuses y (and defocuses x) — the D quad of a FODO cell.
    f = 2.0
    quad = ThinQuadrupole(-1.0 / f)
    y0 = 1.0e-3
    out = Tracker(Lattice([quad, Drift(f)], proton_gamma5)).track(Particle(y=y0, py=0.0))
    assert out.y == pytest.approx(0.0, abs=1e-15)


def test_ultrarelativistic_quad_has_no_longitudinal_slip() -> None:
    ultra = ReferenceParticle.from_total_energy(PROTON_MASS_EV, 1.0e15)
    M = Quadrupole(2.0, 1.0).matrix(ultra)
    assert M[ZETA, DELTA] == pytest.approx(0.0, abs=1e-9)
