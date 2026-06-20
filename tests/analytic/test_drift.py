"""Analytic checks for the Drift element (Stage 0 acceptance + full 6x6 map).

The checks here are deliberately *independent* of the implementation: expected
values are hand-computed or re-derived symbolically from the exact drift map,
never produced by re-running the matrix under test.
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
    ReferenceParticle,
    Tracker,
    X,
    Y,
)
from accsim.symplectic import is_symplectic


# --- Stage 0 acceptance: a Drift propagates a particle to the expected place ---
def test_drift_transverse_propagation_hand_computed(proton_gamma5: ReferenceParticle) -> None:
    L = 2.5
    p = Particle(x=1.0e-3, px=2.0e-4, y=-5.0e-4, py=1.0e-4)
    out = Tracker(Lattice([Drift(L)], proton_gamma5)).track(p)

    # Hand-computed independently: x_f = x + L*px, y_f = y + L*py; momenta unchanged.
    assert out.x == pytest.approx(1.0e-3 + 2.5 * 2.0e-4)  # = 1.5e-3
    assert out.y == pytest.approx(-5.0e-4 + 2.5 * 1.0e-4)  # = -2.5e-4
    assert out.px == pytest.approx(2.0e-4)
    assert out.py == pytest.approx(1.0e-4)
    # No momentum spread -> no longitudinal slip.
    assert out.zeta == pytest.approx(0.0, abs=1e-18)
    assert out.delta == pytest.approx(0.0, abs=1e-18)


# --- longitudinal coupling checked against the EXACT time-of-flight, not L/gamma^2 ---
def test_drift_longitudinal_matches_time_of_flight(proton_gamma5: ReferenceParticle) -> None:
    L = 2.0
    delta = 1.0e-4  # small, so the linear matrix should match the exact map closely
    p = Particle(delta=delta)
    out = Tracker(Lattice([Drift(L)], proton_gamma5)).track(p)

    # Exact, first-principles: on-axis the geometric path is L; the particle
    # travels it at speed beta_p, so zeta changes by L*(1 - beta0/beta_p).
    P0c = proton_gamma5.momentum_eV
    m = proton_gamma5.mass_eV
    beta0 = proton_gamma5.beta0
    Pc = P0c * (1.0 + delta)
    Ec = math.hypot(Pc, m)
    beta_p = Pc / Ec
    dzeta_exact = L * (1.0 - beta0 / beta_p)

    # Linear matrix and exact map agree to within the O(delta^2) truncation.
    assert out.zeta == pytest.approx(dzeta_exact, rel=1e-3)
    assert out.zeta > 0.0  # delta>0 -> faster -> arrives earlier -> zeta increases


def test_drift_matrix_matches_symbolic_derivation(proton_gamma5: ReferenceParticle) -> None:
    """Re-derive the full 6x6 from the exact map symbolically; compare entrywise."""
    sp = pytest.importorskip("sympy")

    L_val = 1.37
    m_val = proton_gamma5.mass_eV
    P0_val = proton_gamma5.momentum_eV

    L, m, P0 = sp.symbols("L m P0", positive=True)
    px, py, delta = sp.symbols("px py delta", real=True)

    pz = sp.sqrt((1 + delta) ** 2 - px**2 - py**2)  # Ps / P0
    E0 = sp.sqrt(P0**2 + m**2)
    beta0 = P0 / E0
    P = P0 * (1 + delta)
    E = sp.sqrt(P**2 + m**2)
    beta_p = P / E
    path = L * (1 + delta) / pz  # geometric path length through the drift
    dt = path / beta_p  # c = 1
    dzeta = L - beta0 * dt  # change in zeta = s - beta0*c*t

    xp = L * px / pz  # change in x
    yp = L * py / pz  # change in y

    origin = {px: 0, py: 0, delta: 0}

    def d(expr, var):
        return sp.diff(expr, var).subs(origin)

    R = sp.eye(6)
    R[X, PX] = d(xp, px)
    R[Y, PY] = d(yp, py)
    R[ZETA, DELTA] = d(dzeta, delta)

    subs = {L: L_val, m: m_val, P0: P0_val}
    expected = np.array(R.subs(subs).evalf(), dtype=float)

    got = Drift(L_val).matrix(proton_gamma5)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-14)
    # And the symbolically-derived R56 is exactly L/gamma0^2.
    gamma0 = proton_gamma5.gamma0
    assert got[ZETA, DELTA] == pytest.approx(L_val / gamma0**2)


def test_drift_is_symplectic(proton_gamma5: ReferenceParticle) -> None:
    assert is_symplectic(Drift(3.3).matrix(proton_gamma5))


def test_zero_length_drift_is_identity(proton_gamma5: ReferenceParticle) -> None:
    np.testing.assert_array_equal(Drift(0.0).matrix(proton_gamma5), np.eye(6))


def test_drifts_compose_additively(proton_gamma5: ReferenceParticle) -> None:
    # Two consecutive drifts equal one drift of the summed length.
    combined = Lattice([Drift(1.0), Drift(2.0)], proton_gamma5).transfer_matrix()
    single = Drift(3.0).matrix(proton_gamma5)
    np.testing.assert_allclose(combined, single, rtol=1e-14, atol=1e-16)


def test_negative_length_rejected() -> None:
    with pytest.raises(ValueError):
        Drift(-1.0)


def test_longitudinal_coupling_vanishes_ultrarelativistically() -> None:
    # gamma0 -> inf  =>  R56 = L/gamma0^2 -> 0.
    ultra = ReferenceParticle.from_total_energy(PROTON_MASS_EV, 1.0e15)
    R56 = Drift(10.0).matrix(ultra)[ZETA, DELTA]
    assert R56 == pytest.approx(0.0, abs=1e-9)
