"""Analytic checks for the SkewQuadrupole (thick) and ThinSkewQuadrupole elements.

A skew quadrupole is a normal quad rolled 45 deg about the s-axis -- the canonical
betatron (x-y) coupling source. The expected maps are re-derived symbolically as
``exp(L*A)`` of the *rolled* Hamiltonian generator ``A = R A_quad R^-1`` (never by
re-running the matrix under test), and the roll identity is checked with teeth
(the element is built directly from a closed form, not by rolling the quad, so the
comparison can fail). The sign of ``k1s`` versus xtrack lives in the reference suite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    DELTA,
    PX,
    PY,
    ZETA,
    Drift,
    Quadrupole,
    ReferenceParticle,
    SkewQuadrupole,
    ThinQuadrupole,
    ThinSkewQuadrupole,
    X,
    Y,
)
from accsim.symplectic import is_symplectic

_T = [X, PX, Y, PY]  # transverse subspace


def _roll_6d(phi: float) -> np.ndarray:
    """6x6 roll by ``phi`` about the s-axis: rotates (x,y) and (px,py), leaves (zeta,delta)."""
    c, s = math.cos(phi), math.sin(phi)
    R = np.eye(6)
    R[X, X] = c
    R[X, Y] = s
    R[Y, X] = -s
    R[Y, Y] = c
    R[PX, PX] = c
    R[PX, PY] = s
    R[PY, PX] = -s
    R[PY, PY] = c
    return R


# --- gold pin: matrix == exp(L*A) of the rolled Hamiltonian generator ---
def test_skew_matrix_matches_symbolic_exponential(proton_gamma5: ReferenceParticle) -> None:
    sp = pytest.importorskip("sympy")

    L_val, k1s_val = 0.37, 1.9
    g0_val = proton_gamma5.gamma0

    L, k1s = sp.symbols("L k1s", positive=True)
    # normal-quad transverse generator: x'' + k1 x = 0, y'' - k1 y = 0
    A_quad = sp.Matrix([[0, 1, 0, 0], [-k1s, 0, 0, 0], [0, 0, 0, 1], [0, 0, k1s, 0]])
    phi = sp.pi / 4
    c, s = sp.cos(phi), sp.sin(phi)
    R4 = sp.Matrix([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])
    A_skew = R4 * A_quad * R4.inv()
    expLA = sp.simplify((L * A_skew).exp())
    expected4 = np.array(expLA.subs({L: L_val, k1s: k1s_val}).evalf(), dtype=float)

    M = SkewQuadrupole(L_val, k1s_val).matrix(proton_gamma5)
    assert np.allclose(M[np.ix_(_T, _T)], expected4, atol=1e-12)
    # longitudinal: like a drift/quad, R56 = L/gamma0^2, no transverse-long coupling
    assert M[ZETA, DELTA] == pytest.approx(L_val / g0_val**2, rel=1e-14)
    assert is_symplectic(M)


# --- the roll identity, with teeth: direct closed form == R(45) Q R(-45) ---
def test_skew_is_rolled_normal_quad(proton_gamma5: ReferenceParticle) -> None:
    L_val, k1s_val = 0.52, -0.8  # negative gradient too
    R = _roll_6d(math.pi / 4)
    rolled = R @ Quadrupole(L_val, k1s_val).matrix(proton_gamma5) @ R.T  # R^-1 = R^T
    direct = SkewQuadrupole(L_val, k1s_val).matrix(proton_gamma5)
    assert np.allclose(direct, rolled, atol=1e-13)


def test_thin_skew_is_rolled_thin_quad(proton_gamma5: ReferenceParticle) -> None:
    k1sl = 0.11
    R = _roll_6d(math.pi / 4)
    rolled = R @ ThinQuadrupole(k1sl).matrix(proton_gamma5) @ R.T
    direct = ThinSkewQuadrupole(k1sl).matrix(proton_gamma5)
    assert np.allclose(direct, rolled, atol=1e-14)


# --- explicit thin kick entries: dpx = k1sl y, dpy = k1sl x ---
def test_thin_skew_entries(proton_gamma5: ReferenceParticle) -> None:
    k1sl = 0.23
    M = ThinSkewQuadrupole(k1sl).matrix(proton_gamma5)
    expected = np.eye(6)
    expected[PX, Y] = k1sl
    expected[PY, X] = k1sl
    assert np.allclose(M, expected, atol=1e-15)
    assert is_symplectic(M)


# --- k1s = 0 is a plain drift ---
def test_skew_zero_gradient_is_drift(proton_gamma5: ReferenceParticle) -> None:
    L_val = 0.9
    M = SkewQuadrupole(L_val, 0.0).matrix(proton_gamma5)
    assert np.allclose(M, Drift(L_val).matrix(proton_gamma5), atol=1e-15)


# --- thin limit: SkewQuadrupole(L, k1sl/L) -> ThinSkewQuadrupole(k1sl) as L -> 0 ---
def test_skew_thin_limit(proton_gamma5: ReferenceParticle) -> None:
    k1sl = 0.15
    thin = ThinSkewQuadrupole(k1sl).matrix(proton_gamma5)
    for L_val in (1e-2, 1e-3, 1e-4):
        thick = SkewQuadrupole(L_val, k1sl / L_val).matrix(proton_gamma5)
        # transverse block converges to the thin kick (longitudinal R56 -> 0)
        assert np.allclose(thick[np.ix_(_T, _T)], thin[np.ix_(_T, _T)], atol=5.0 * L_val)


# --- negative skew gradient reverses the coupling ---
def test_skew_sign_flips_coupling(proton_gamma5: ReferenceParticle) -> None:
    L_val, k1s_val = 0.4, 1.3
    Mp = SkewQuadrupole(L_val, k1s_val).matrix(proton_gamma5)
    Mm = SkewQuadrupole(L_val, -k1s_val).matrix(proton_gamma5)
    # diagonal (own-plane) blocks are even in k1s; off-diagonal coupling blocks odd
    assert np.allclose(Mp[np.ix_([X, PX], [X, PX])], Mm[np.ix_([X, PX], [X, PX])], atol=1e-14)
    assert np.allclose(Mp[np.ix_([X, PX], [Y, PY])], -Mm[np.ix_([X, PX], [Y, PY])], atol=1e-14)


# --- physical coupling: a purely horizontal offset acquires vertical motion ---
def test_skew_couples_x_into_y(proton_gamma5: ReferenceParticle) -> None:
    M = SkewQuadrupole(0.5, 1.0).matrix(proton_gamma5)
    state = np.zeros(6)
    state[X] = 1e-3  # x only, no y
    out = M @ state
    assert abs(out[PY]) > 1e-6  # vertical momentum kicked by the horizontal offset
    assert abs(out[Y]) > 0.0  # and vertical position follows
