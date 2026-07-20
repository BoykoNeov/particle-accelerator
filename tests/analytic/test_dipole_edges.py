"""Analytic checks for Dipole pole-face (edge) focusing -- hard-edge model.

The edge kick is a thin quadrupole-like map at each pole face,
``px -> px + h tan(e) x`` / ``py -> py - h tan(e) y``, sandwiching the sector
body: ``M = Edge(e2) @ Body @ Edge(e1)``. The gates here, in order of teeth:

1. **Regression** -- ``e1 = e2 = 0`` is byte-identical to the pure sector map, so
   nothing about the Stage-1 dipole moved.
2. **Symbolic** -- the runtime map equals the symbolic edge-body-edge product,
   with the body taken as ``exp(L*A)`` (the same generator the sector test uses).
3. **Rectangular bend** (``e1 = e2 = theta/2``) -- the strongest remember-free
   check: the two edges *exactly* cancel the body's horizontal weak focusing, so
   the horizontal block collapses to a drift ``[[1, rho sin theta], [0, 1]]``
   (``R21 = 0`` to machine precision, proven symbolically), while all vertical
   focusing comes from the edges.
4. **Sign / plane** -- a positive edge angle defocuses ``x`` and focuses ``y``.
5. **Symplecticity** -- across a range of ``(e1, e2)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import PX, ReferenceParticle, X
from accsim.coords import PY, Y
from accsim.elements.dipole import Dipole
from accsim.symplectic import is_symplectic

L_VAL = 2.0
THETA_VAL = 0.3


def test_zero_edges_are_pure_sector(proton_gamma5: ReferenceParticle) -> None:
    """Default edges leave the Stage-1 sector map untouched, bit-for-bit."""
    got = Dipole(L_VAL, THETA_VAL, e1=0.0, e2=0.0).matrix(proton_gamma5)
    expected = Dipole(L_VAL, THETA_VAL).matrix(proton_gamma5)
    assert np.array_equal(got, expected)


def test_edge_matrix_matches_symbolic(proton_gamma5: ReferenceParticle) -> None:
    """Runtime map == Edge(e2) @ exp(L A) @ Edge(e1), all symbolic."""
    sp = pytest.importorskip("sympy")

    e1_val, e2_val = 0.12, 0.20
    g0_val = proton_gamma5.gamma0
    h_val = THETA_VAL / L_VAL

    h, g0, L = sp.symbols("h gamma0 L", positive=True)
    e1, e2 = sp.symbols("e1 e2")
    A = sp.Matrix(
        [
            [0, 1, 0, 0, 0, 0],
            [-(h**2), 0, 0, 0, 0, h],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [-h, 0, 0, 0, 0, 1 / g0**2],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    body = (A * L).exp()

    def edge(e):
        E = sp.eye(6)
        E[1, 0] = h * sp.tan(e)  # R21
        E[3, 2] = -h * sp.tan(e)  # R43
        return E

    M_sym = edge(e2) * body * edge(e1)
    subs = {h: h_val, g0: g0_val, L: L_VAL, e1: e1_val, e2: e2_val}
    expected = np.array(sp.simplify(M_sym).subs(subs).evalf(), dtype=float)

    got = Dipole(L_VAL, THETA_VAL, e1=e1_val, e2=e2_val).matrix(proton_gamma5)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-13)


def test_rectangular_bend_horizontal_is_a_drift(proton_gamma5: ReferenceParticle) -> None:
    """e1 = e2 = theta/2: horizontal block == drift of length rho sin theta.

    The two edge defocus kicks exactly cancel the body's horizontal weak
    focusing (an exact identity in this linear hard-edge model, proven
    symbolically in ``test_rectangular_bend_R21_is_symbolically_zero``): the
    horizontal 2x2 becomes ``[[1, rho sin theta], [0, 1]]``.
    """
    theta = THETA_VAL
    rho = L_VAL / theta
    M = Dipole(L_VAL, theta, e1=theta / 2, e2=theta / 2).matrix(proton_gamma5)

    assert M[X, X] == pytest.approx(1.0, abs=1e-13)
    assert M[PX, PX] == pytest.approx(1.0, abs=1e-13)
    assert M[PX, X] == pytest.approx(0.0, abs=1e-13)  # focusing cancels exactly
    assert M[X, PX] == pytest.approx(rho * math.sin(theta))


def test_rectangular_bend_R21_is_symbolically_zero() -> None:
    """The horizontal focusing cancellation is exact, not merely small."""
    sp = pytest.importorskip("sympy")
    u = sp.symbols("u", positive=True)  # u = theta/2
    theta = 2 * u
    # R21 of Edge(theta/2) @ Body_horiz @ Edge(theta/2), factored by h:
    r21 = sp.sin(theta) * sp.tan(u) ** 2 - sp.sin(theta) + 2 * sp.cos(theta) * sp.tan(u)
    assert sp.simplify(r21.rewrite(sp.sin).rewrite(sp.cos)) == 0


def test_rectangular_bend_vertical_focusing(proton_gamma5: ReferenceParticle) -> None:
    """All vertical focusing comes from the edges: R43 ~ -2 h tan(theta/2) < 0."""
    theta = THETA_VAL
    h = theta / L_VAL
    t = h * math.tan(theta / 2)
    M = Dipole(L_VAL, theta, e1=theta / 2, e2=theta / 2).matrix(proton_gamma5)

    # Exact composed value R43 = -2t + t^2 L (edges + their overlap through the body drift).
    assert M[PY, Y] == pytest.approx(-2 * t + t * t * L_VAL)
    assert M[PY, Y] < 0.0  # net vertical focusing
    # A pure sector bend has no vertical focusing at all -- the edges made it.
    assert Dipole(L_VAL, theta).matrix(proton_gamma5)[PY, Y] == pytest.approx(0.0, abs=1e-15)


def test_edge_sign_and_plane(proton_gamma5: ReferenceParticle) -> None:
    """A single positive entrance edge defocuses x (R21 > 0) and focuses y (R43 < 0)."""
    e = 0.25
    h = THETA_VAL / L_VAL
    M = Dipole(L_VAL, THETA_VAL, e1=e, e2=0.0).matrix(proton_gamma5)
    body = Dipole(L_VAL, THETA_VAL).matrix(proton_gamma5)

    # The entrance edge adds +h tan(e) to R21 and -h tan(e) to R43 of the body...
    # ...but R21 is read after composition, so pin the *change* is in the right direction.
    assert M[PX, X] > body[PX, X]  # more defocusing in x
    assert M[PY, Y] < body[PY, Y]  # more focusing in y
    # And the vertical block, which the body leaves as a pure drift, now focuses.
    assert M[PY, Y] == pytest.approx(-h * math.tan(e))  # body R43 = 0, drift adds nothing to R43


def test_edges_are_symplectic(proton_gamma5: ReferenceParticle) -> None:
    for e1, e2 in ((0.1, 0.1), (0.15, -0.05), (THETA_VAL / 2, THETA_VAL / 2), (0.3, 0.0)):
        M = Dipole(L_VAL, THETA_VAL, e1=e1, e2=e2).matrix(proton_gamma5)
        assert is_symplectic(M), f"non-symplectic at e1={e1}, e2={e2}"
