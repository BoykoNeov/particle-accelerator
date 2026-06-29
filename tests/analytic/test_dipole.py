"""Analytic checks for the Dipole (pure sector bend).

The 6x6 is re-derived symbolically as ``exp(L*A)`` of the sector-bend Hamiltonian
generator (which is symplectic by construction), and compared entrywise to the
runtime matrix. Hand-computed values and the theta -> 0 drift limit are checked
independently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    DELTA,
    PX,
    ZETA,
    Dipole,
    Drift,
    ReferenceParticle,
    X,
)
from accsim.coords import PY, Y
from accsim.symplectic import is_symplectic


def test_dipole_matrix_matches_symbolic_exponential(proton_gamma5: ReferenceParticle) -> None:
    sp = pytest.importorskip("sympy")

    L_val, theta_val = 2.0, 0.2
    g0_val = proton_gamma5.gamma0
    h_val = theta_val / L_val

    h, g0, L = sp.symbols("h gamma0 L", positive=True)
    # Generator of the linearised sector-bend Hamiltonian
    #   H2 = px^2/2 + py^2/2 + h^2 x^2/2 - h x delta + delta^2/(2 gamma0^2):
    #   x'=px, px'=-h^2 x + h delta, y'=py, py'=0,
    #   zeta'=-h x + delta/gamma0^2, delta'=0.
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
    M_sym = (A * L).exp()
    subs = {h: h_val, g0: g0_val, L: L_val}
    expected = np.array(sp.simplify(M_sym).subs(subs).evalf(), dtype=float)

    got = Dipole(L_val, theta_val).matrix(proton_gamma5)
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-13)


def test_dipole_entries_hand_computed(proton_gamma5: ReferenceParticle) -> None:
    L, theta = 2.0, 0.2
    h = theta / L
    rho = L / theta
    c, s = math.cos(theta), math.sin(theta)
    M = Dipole(L, theta).matrix(proton_gamma5)

    # Horizontal + dispersion.
    assert M[X, X] == pytest.approx(c)
    assert M[X, PX] == pytest.approx(rho * s)
    assert M[X, DELTA] == pytest.approx(rho * (1 - c))  # outward shift, > 0
    assert M[PX, X] == pytest.approx(-h * s)
    assert M[PX, PX] == pytest.approx(c)
    assert M[PX, DELTA] == pytest.approx(s)
    # Vertical: pure drift.
    assert M[Y, PY] == pytest.approx(L)
    assert M[Y, Y] == pytest.approx(1.0)
    # Longitudinal: symplectic partners of dispersion + reduced drift slip.
    assert M[ZETA, X] == pytest.approx(-s)
    assert M[ZETA, PX] == pytest.approx(-rho * (1 - c))
    assert M[ZETA, DELTA] == pytest.approx(rho * s - L + L / proton_gamma5.gamma0**2)


def test_dipole_R51_R52_are_symplectic_partners(proton_gamma5: ReferenceParticle) -> None:
    # The longitudinal coupling is forced by symplecticity:
    #   R51 = R21*R16 - R11*R26,  R52 = R22*R16 - R12*R26.
    M = Dipole(2.0, 0.2).matrix(proton_gamma5)
    assert M[ZETA, X] == pytest.approx(M[PX, X] * M[X, DELTA] - M[X, X] * M[PX, DELTA])
    assert M[ZETA, PX] == pytest.approx(M[PX, PX] * M[X, DELTA] - M[X, PX] * M[PX, DELTA])


def test_dipole_is_symplectic(proton_gamma5: ReferenceParticle) -> None:
    for theta in (0.2, -0.35, 1.0):
        assert is_symplectic(Dipole(2.0, theta).matrix(proton_gamma5))


def test_dipole_zero_angle_is_drift(proton_gamma5: ReferenceParticle) -> None:
    L = 1.7
    np.testing.assert_allclose(Dipole(L, 0.0).matrix(proton_gamma5), Drift(L).matrix(proton_gamma5))


def test_dipole_approaches_drift_as_angle_shrinks(proton_gamma5: ReferenceParticle) -> None:
    L = 1.7
    drift = Drift(L).matrix(proton_gamma5)
    prev = math.inf
    for theta in (0.1, 0.01, 0.001):
        err = np.max(np.abs(Dipole(L, theta).matrix(proton_gamma5) - drift))
        assert err < prev  # monotone convergence to the straight limit
        prev = err
    assert prev < 1e-3


def test_dipole_dispersion_sign(proton_gamma5: ReferenceParticle) -> None:
    # A higher-momentum particle (delta > 0) is bent less and ends up displaced
    # outward: x increases with delta, so R16 > 0.
    M = Dipole(2.0, 0.3).matrix(proton_gamma5)
    assert M[X, DELTA] > 0.0


def test_dipole_negative_length_rejected() -> None:
    with pytest.raises(ValueError):
        Dipole(-1.0, 0.1)


def test_dipole_angle_without_length_rejected() -> None:
    with pytest.raises(ValueError):
        Dipole(0.0, 0.1)
