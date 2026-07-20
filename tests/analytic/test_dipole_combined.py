"""Analytic checks for the combined-function Dipole (bend + quadrupole gradient).

The body obeys ``x'' + (h^2 + k1) x = h*delta``, ``y'' - k1 y = 0``. The 6x6 is
re-derived symbolically as ``exp(L*A)`` of that Hamiltonian generator and compared
entrywise to the runtime map, across the sign regimes that matter (``K_x = h^2+k1``
positive, negative, and the singular ``K_x = 0``). The two reduction limits --
pure sector at ``k1 = 0`` and pure quadrupole at ``h = 0`` -- are free regressions
against already-validated elements, so a bug in the combined map cannot hide by
agreeing with a wrong generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import DELTA, PX, PY, Dipole, Quadrupole, ReferenceParticle, X, Y
from accsim.symplectic import is_symplectic

L_VAL = 2.0
THETA_VAL = 0.2


def _generator(h_val: float, k1_val: float, g0_val: float, L_val: float):
    """``exp(L*A)`` of the combined-function bend generator, as a real float array.

    The generator ``A`` is filled with exact rationals *before* exponentiating, so
    sympy computes the matrix exp from the (possibly defective, at ``K_x = 0``)
    Jordan form -- no ``1/K_x`` closed-form singularity to go ``nan`` on the
    ``h^2 = -k1`` tune. The result is real; any ~1e-21 imaginary residue from the
    mixed-sign eigenvalues (cos vs cosh) is numerical, so we keep the real part.
    """
    sp = pytest.importorskip("sympy")
    h, k1, g0, L = (sp.nsimplify(v, rational=True) for v in (h_val, k1_val, g0_val, L_val))
    A = sp.Matrix(
        [
            [0, 1, 0, 0, 0, 0],
            [-(h**2 + k1), 0, 0, 0, 0, h],
            [0, 0, 0, 1, 0, 0],
            [0, 0, k1, 0, 0, 0],
            [-h, 0, 0, 0, 0, 1 / g0**2],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    M = (A * L).exp()
    return np.array(M.evalf(), dtype=complex).real


@pytest.mark.parametrize(
    "k1",
    [
        0.5,  # K_x = h^2 + k1 > 0  (strong horizontal focus)
        -0.02,  # K_x < 0           (horizontal defocus overall)
        -(0.1**2),  # K_x = 0        (singular tune h^2 = -k1, removable limit)
        0.15,  # k1 > 0            (vertical defocus)
    ],
)
def test_combined_body_matches_symbolic(k1: float, proton_gamma5: ReferenceParticle) -> None:
    h = THETA_VAL / L_VAL
    expected = _generator(h, k1, proton_gamma5.gamma0, L_VAL)
    got = Dipole(L_VAL, THETA_VAL, k1=k1).matrix(proton_gamma5)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-13)


def test_reduces_to_pure_sector_at_zero_gradient(proton_gamma5: ReferenceParticle) -> None:
    """k1 = 0 is byte-identical to the pure sector dipole (no regression)."""
    got = Dipole(L_VAL, THETA_VAL, k1=0.0).matrix(proton_gamma5)
    expected = Dipole(L_VAL, THETA_VAL).matrix(proton_gamma5)
    assert np.array_equal(got, expected)


def test_reduces_to_quadrupole_at_zero_curvature(proton_gamma5: ReferenceParticle) -> None:
    """h = 0 (angle = 0) is a pure quadrupole; dispersion vanishes with the bend."""
    k1 = 0.7
    got = Dipole(L_VAL, 0.0, k1=k1).matrix(proton_gamma5)
    expected = Quadrupole(L_VAL, k1).matrix(proton_gamma5)
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-15)
    assert got[X, DELTA] == pytest.approx(0.0, abs=1e-15)  # no dispersion without curvature
    assert got[PX, DELTA] == pytest.approx(0.0, abs=1e-15)


def test_gradient_sign_and_planes(proton_gamma5: ReferenceParticle) -> None:
    """k1 > 0 strengthens horizontal focus (K_x > h^2) and defocuses vertically."""
    sector = Dipole(L_VAL, THETA_VAL).matrix(proton_gamma5)
    combined = Dipole(L_VAL, THETA_VAL, k1=0.5).matrix(proton_gamma5)

    # Horizontal R21 = -K_x * sin(...)/... becomes more negative (stronger focus).
    assert combined[PX, X] < sector[PX, X] < 0.0
    # Vertical: the sector is a drift (R43 = 0); k1 > 0 makes it defocus (R43 > 0).
    assert sector[PY, Y] == pytest.approx(0.0, abs=1e-15)
    assert combined[PY, Y] > 0.0
    # The horizontal focusing constant really is h^2 + k1, not just k1: a k1 = 0
    # sector already focuses horizontally (weak) -- R21 < 0 even at zero gradient.
    assert sector[PX, X] < 0.0


def test_combined_is_symplectic(proton_gamma5: ReferenceParticle) -> None:
    for k1 in (0.5, -0.02, -(0.1**2), 0.15, -0.3):
        M = Dipole(L_VAL, THETA_VAL, k1=k1).matrix(proton_gamma5)
        assert is_symplectic(M), f"non-symplectic at k1={k1}"


def test_combined_function_with_edges_matches_symbolic(proton_gamma5: ReferenceParticle) -> None:
    """Gradient and pole-face edges compose: M = Edge(e2) @ exp(L A) @ Edge(e1)."""
    h_val = THETA_VAL / L_VAL
    k1_val, e1_val, e2_val = 0.3, 0.12, 0.20
    body = _generator(h_val, k1_val, proton_gamma5.gamma0, L_VAL)  # skips if sympy absent

    def edge(e):
        E = np.eye(6)
        E[PX, X] = h_val * np.tan(e)
        E[PY, Y] = -h_val * np.tan(e)
        return E

    expected = edge(e2_val) @ body @ edge(e1_val)
    got = Dipole(L_VAL, THETA_VAL, k1=k1_val, e1=e1_val, e2=e2_val).matrix(proton_gamma5)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-13)
