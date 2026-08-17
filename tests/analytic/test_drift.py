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
from accsim.symplectic import is_symplectic, is_symplectic_map, is_symplectic_map_canonical


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


def test_the_exact_map_is_symplectic_but_only_in_canonical_variables(
    proton_gamma5: ReferenceParticle,
) -> None:
    r"""The exact map's symplecticity, and the check that is allowed to judge it.

    Read this next to the assertion above, which passes and says nothing about the
    exact map. ``Drift.matrix`` is three independent shear blocks, and a shear is
    symplectic in whatever pair it acts on — including the ``(zeta, delta)`` pair that
    is **not** canonically conjugate. The exact map is the flow of a Hamiltonian, so it
    is symplectic; but tested in ``(zeta, delta)`` it is *rejected*, because there the
    coordinates are wrong rather than the map.

    So the more faithful map fails the check the cruder one passes. Both facts are
    asserted here, because the trap is to "repair" the exact map until
    :func:`~accsim.symplectic.is_symplectic_map` goes green — which lands on a map that
    is wrong at first order (see ``test_symplectic_canonical.py``).
    """
    ref = proton_gamma5
    drift = Drift(2.0)
    for amp in (1.0e-3, 1.0e-2, 5.0e-2):
        st = np.array([amp, amp, -amp, 0.7 * amp, amp, amp])
        assert is_symplectic_map_canonical(lambda s: drift.track(s, ref), st, ref)
        assert not is_symplectic_map(lambda s: drift.track(s, ref), st)


def test_the_exact_map_matches_a_symbolic_derivation_at_large_angles(
    proton_gamma5: ReferenceParticle,
) -> None:
    r"""The exact map itself, re-derived in sympy and evaluated — not just its slope.

    ``test_drift_matrix_matches_symbolic_derivation`` above builds the same exact
    expressions but only differentiates them at the origin, so it pins the linear
    matrix and is blind to which nonlinear map that matrix came from. This evaluates
    the map, at angles large enough that the candidates are far apart.

    **The angles are the whole point.** At ``px = py = 0`` the exact map, xtrack's
    "expanded" model and the linear matrix all coincide, so an on-axis check would
    pass whatever was implemented. The control below is the *expanded* map
    ``x += L px / (1 + delta)`` — xtrack's default, and the plausible wrong choice —
    which differs from the exact one by ``(px^2 + py^2) / 2`` relatively: measured
    ``1.5e-6`` at ``px = 1e-2`` and ``1.7e-4`` at ``px = 5e-2``, against a ``4e-16``
    agreement for the right map. Orders apart, so the choice cannot hide in a
    tolerance.
    """
    sp = pytest.importorskip("sympy")
    ref = proton_gamma5

    L, m, P0 = sp.symbols("L m P0", positive=True)
    x_s, px_s, y_s, py_s, zeta_s, delta_s = sp.symbols("x px y py zeta delta", real=True)

    pz = sp.sqrt((1 + delta_s) ** 2 - px_s**2 - py_s**2)
    E0 = sp.sqrt(P0**2 + m**2)
    beta0 = P0 / E0
    P = P0 * (1 + delta_s)
    E = sp.sqrt(P**2 + m**2)
    path = L * (1 + delta_s) / pz  # geometric path length through the drift
    dt = path / (P / E)  # c = 1
    exact = sp.Matrix(
        [x_s + L * px_s / pz, px_s, y_s + L * py_s / pz, py_s, zeta_s + L - beta0 * dt, delta_s]
    )
    # The control: xtrack's default "expanded" model, exact only to second order in
    # the angles. Same longitudinal physics, wrong transverse denominator.
    xp, yp = px_s / (1 + delta_s), py_s / (1 + delta_s)
    expanded = sp.Matrix(
        [
            x_s + L * xp,
            px_s,
            y_s + L * yp,
            py_s,
            zeta_s + L * (1 - (E / (P / P0 * E0)) * (1 + (xp**2 + yp**2) / 2)),
            delta_s,
        ]
    )

    L_val = 2.0
    subs0 = {L: L_val, m: ref.mass_eV, P0: ref.momentum_eV}
    for st in (
        np.array([1.0e-3, 1.0e-2, -5.0e-4, 7.0e-3, 2.0e-3, 1.0e-3]),
        np.array([0.0, 5.0e-2, 0.0, -3.0e-2, 0.0, 1.0e-2]),
    ):
        pt = dict(zip([x_s, px_s, y_s, py_s, zeta_s, delta_s], (float(v) for v in st), strict=True))
        want = np.array(exact.subs({**subs0, **pt}).evalf(40), dtype=float).ravel()
        control = np.array(expanded.subs({**subs0, **pt}).evalf(40), dtype=float).ravel()

        got = Drift(L_val).track(st, ref)
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-15)

        # Non-vacuous: the wrong model, and the linear matrix, are far away.
        assert np.max(np.abs(control - want)) > 1.0e-6
        assert np.max(np.abs(Drift(L_val).matrix(ref) @ st - want)) > 1.0e-4


def test_on_the_design_orbit_the_exact_map_is_the_linear_matrix_bit_for_bit(
    proton_gamma5: ReferenceParticle,
) -> None:
    r"""At zero transverse angle the exact map's Jacobian *is* ``matrix()``, exactly.

    This is what bounds the change: ``d(L px / pz)/d(delta) = -L px (1+delta)/pz^3``
    vanishes identically at ``px = 0``, and so does the conjugate ``d(zeta)/d(px)``.
    So an aligned lattice on its design orbit has bit-for-bit unchanged beta, tunes,
    chromaticity and dispersion, and every design-optics cross-check in the suite is
    untouched by the exact map. Asserted at **exact equality** on the entries that
    could have moved, not to a tolerance — a finite-difference Jacobian would only
    reach round-off, so the derivative is taken symbolically instead.

    ``delta != 0`` on purpose: it is the *angles* that switch the new terms on, not
    the momentum, so an off-momentum particle on axis must still see the linear map.
    """
    sp = pytest.importorskip("sympy")
    ref = proton_gamma5

    L, m, P0 = sp.symbols("L m P0", positive=True)
    px_s, py_s, delta_s = sp.symbols("px py delta", real=True)
    pz = sp.sqrt((1 + delta_s) ** 2 - px_s**2 - py_s**2)
    E = sp.sqrt((P0 * (1 + delta_s)) ** 2 + m**2)
    E0 = sp.sqrt(P0**2 + m**2)
    dx, dy = L * px_s / pz, L * py_s / pz
    dzeta = L * (1 - E / (E0 * pz))

    L_val = 1.9
    at_axis = {px_s: 0, py_s: 0, delta_s: 0.0}
    subs0 = {L: L_val, m: ref.mass_eV, P0: ref.momentum_eV}

    for delta_val in (0.0, 1.0e-3, -2.0e-3):
        pt = {**at_axis, delta_s: delta_val}
        # The four entries that carry the new physics are all exactly zero on axis.
        assert sp.diff(dx, delta_s).subs(pt).subs(subs0) == 0
        assert sp.diff(dy, delta_s).subs(pt).subs(subs0) == 0
        assert sp.diff(dzeta, px_s).subs(pt).subs(subs0) == 0
        assert sp.diff(dzeta, py_s).subs(pt).subs(subs0) == 0

    # On an on-axis, off-momentum particle the transverse coordinates come out
    # bit-for-bit equal to the matrix's, because px = py = 0 kills every new term.
    st = np.array([1.0e-3, 0.0, -5.0e-4, 0.0, 2.0e-3, 1.0e-3])
    linear = Drift(L_val).matrix(ref) @ st
    exact = Drift(L_val).track(st, ref)
    np.testing.assert_array_equal(exact[[X, PX, PY, DELTA]], linear[[X, PX, PY, DELTA]])

    # ``zeta`` is the one coordinate that still differs, and by the *matrix's* own
    # truncation rather than by anything new: R56 delta is the first term of an exact
    # time of flight. Expanding the exact form on axis (pz = 1 + delta) gives
    #     dzeta / L = (delta / gamma0^2) [1 - delta (1 + beta0^2 / 2)]
    # so the increment's relative error is not merely "first order in delta" but
    # delta (1 + beta0^2 / 2) — 1.48 delta at gamma0 = 5. Pinning the coefficient makes
    # this a check on the longitudinal map rather than a tolerance it has to fit under.
    def zeta_error(delta: float) -> float:
        s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, delta])
        got = Drift(L_val).track(s, ref)[ZETA]
        want = Drift(L_val).matrix(ref)[ZETA, DELTA] * delta
        return abs(got - want) / abs(want)

    coefficient = 1.0 + 0.5 * ref.beta0**2
    for delta_val in (1.0e-4, 1.0e-3):
        assert zeta_error(delta_val) / delta_val == pytest.approx(coefficient, rel=2e-3)


def test_the_exact_map_tracks_a_bunch_the_same_as_one_particle(
    proton_gamma5: ReferenceParticle,
) -> None:
    """A ``(6, n)`` bunch is the per-particle map applied columnwise, to the last bit.

    The exact map divides by a per-particle ``pz``, which is the kind of thing that
    broadcasts wrongly in silence — a scalar ``pz`` would give every particle the
    leading one's momentum and still return a plausible array.
    """
    ref = proton_gamma5
    drift = Drift(2.4)
    bunch = np.array(
        [
            [1.0e-3, -2.0e-3, 0.0, 5.0e-3],
            [1.0e-2, 3.0e-3, 0.0, -1.0e-2],
            [-5.0e-4, 1.0e-3, 0.0, 2.0e-3],
            [7.0e-3, -4.0e-3, 0.0, 6.0e-3],
            [2.0e-3, 1.0e-3, 0.0, -3.0e-3],
            [1.0e-3, -5.0e-3, 0.0, 8.0e-3],
        ]
    )
    out = drift.track(bunch, ref)
    assert out.shape == bunch.shape
    for j in range(bunch.shape[1]):
        np.testing.assert_array_equal(out[:, j], drift.track(bunch[:, j].copy(), ref))
    # The all-zero column is the reference particle and must come out untouched.
    np.testing.assert_array_equal(out[:, 2], np.zeros(6))


def test_a_zero_length_drift_is_the_identity_for_the_exact_map_too(
    proton_gamma5: ReferenceParticle,
) -> None:
    """A marker-length drift moves nothing, without evaluating ``pz`` at all.

    Asserted for a state whose angles are large, so that a ``0 * NaN`` or a spurious
    ``0 * something`` would show up rather than being masked by small numbers.
    """
    st = np.array([1.0e-3, 0.4, -5.0e-4, 0.3, 2.0e-3, 1.0e-3])
    np.testing.assert_array_equal(Drift(0.0).track(st, proton_gamma5), st)


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
