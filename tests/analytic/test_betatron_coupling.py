"""Analytic gates for linear betatron (x-y) coupling: normal-mode tunes + DeltaQ_min.

The milestone's closed-form check is the **closest tune approach**: with a single
skew quad, as the ring is tuned toward the difference resonance ``Q_x = Q_y`` the
two normal-mode tunes repel, and their minimum gap is ``|C^-|``. It is pinned two
ways that cannot share an error:

  * the **exact** eigenvalue gap of the coupled 4x4 (:func:`normal_mode_tunes`),
  * the **closed form** ``|C^-| = (1/2pi) sqrt(beta_x beta_y) |k1s l|``
    (:func:`closest_tune_approach`),

which converge with an ``O((k1s l)^2)`` residual as the coupling -> 0. The ``1/2pi``
prefactor is re-derived symbolically **inside** this file (the eigen-tune split of a
single-kick model), never recalled. A guard test confirms the uncoupled 2x2 path
refuses a coupled lattice rather than returning a wrong decoupled answer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    CoupledLatticeError,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    SkewQuadrupole,
    ThinQuadrupole,
    ThinSkewQuadrupole,
    UnstableLatticeError,
    closed_twiss,
    closest_tune_approach,
    normal_mode_tunes,
    tunes,
)


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(0.938272e9, 5.0)


def _fodo(kq: float, cell_len: float = 2.0, nq_len: float = 0.3) -> list:
    d = (cell_len - 2 * nq_len) / 2.0
    return [Quadrupole(nq_len, kq), Drift(d), Quadrupole(nq_len, -kq), Drift(d)]


# ============================ normal-mode tunes ============================
def test_normal_mode_tunes_reduce_to_uncoupled(ref: ReferenceParticle) -> None:
    """With no coupling, the eigen mode tunes equal the CS tunes mod 1 exactly."""
    lat = Lattice(_fodo(1.2) * 4, ref)
    qx, qy = tunes(lat)
    q1, q2 = normal_mode_tunes(lat)
    assert np.allclose([qx % 1.0, qy % 1.0], [q1, q2], atol=1e-10)


def test_normal_mode_tunes_reduce_when_split(ref: ReferenceParticle) -> None:
    """Same, on a lattice whose planes are tune-split (Qx != Qy)."""
    base = _fodo(1.2) * 4
    base = base[:4] + [ThinQuadrupole(0.06)] + base[4:]
    lat = Lattice(base, ref)
    qx, qy = tunes(lat)
    q1, q2 = normal_mode_tunes(lat)
    # q1 is the x-like mode, q2 the y-like one (labelled by dominant plane)
    assert q1 == pytest.approx(qx % 1.0, abs=1e-10)
    assert q2 == pytest.approx(qy % 1.0, abs=1e-10)


def test_normal_mode_tunes_raise_on_coupled_instability(ref: ReferenceParticle) -> None:
    """A skew kick strong enough to drive eigenvalues off the unit circle raises."""
    # symmetric FODO sits on the difference resonance; a large skew opens the
    # sum-resonance-adjacent stop band -> eigenvalues leave |lambda| = 1.
    base = _fodo(1.2) * 4
    lat = Lattice(base[:8] + [ThinSkewQuadrupole(5.0)] + base[8:], ref)
    with pytest.raises(UnstableLatticeError):
        normal_mode_tunes(lat)


# ============================ the coupling guard ============================
def test_uncoupled_path_refuses_coupled_lattice(ref: ReferenceParticle) -> None:
    """closed_twiss / tunes raise rather than silently decoupling a skew lattice."""
    base = _fodo(1.2) * 4
    lat = Lattice(base[:8] + [ThinSkewQuadrupole(0.02)] + base[8:], ref)
    with pytest.raises(CoupledLatticeError):
        closed_twiss(lat)
    with pytest.raises(CoupledLatticeError):
        tunes(lat)


def test_guard_is_a_noop_without_coupling(ref: ReferenceParticle) -> None:
    """A plain FODO (exactly block-diagonal) is unaffected by the guard."""
    lat = Lattice(_fodo(1.2) * 4, ref)
    closed_twiss(lat)  # must not raise
    tunes(lat)


# ==================== closest tune approach (DeltaQ_min) ====================
def test_no_skew_gives_zero_coupling(ref: ReferenceParticle) -> None:
    assert closest_tune_approach(Lattice(_fodo(1.2) * 4, ref)) == 0.0


def test_closest_tune_approach_equals_eigen_gap_on_resonance(ref: ReferenceParticle) -> None:
    """On the symmetric FODO (Qx = Qy exactly), the eigen-gap == the closed form |C^-|."""
    base = _fodo(1.2) * 4
    lat = Lattice(base[:8] + [ThinSkewQuadrupole(0.005)] + base[8:], ref)
    cmin = closest_tune_approach(lat)
    q1, q2 = normal_mode_tunes(lat)
    assert abs(q1 - q2) == pytest.approx(cmin, rel=1e-3)


def test_closest_tune_approach_quadratic_convergence(ref: ReferenceParticle) -> None:
    """gap / |C^-| -> 1 with a residual that falls ~4x each time k1sl halves (O(k1sl^2))."""
    base = _fodo(1.2) * 4
    resid = []
    for k1sl in (0.02, 0.01, 0.005):
        lat = Lattice(base[:8] + [ThinSkewQuadrupole(k1sl)] + base[8:], ref)
        cmin = closest_tune_approach(lat)
        q1, q2 = normal_mode_tunes(lat)
        resid.append(abs(abs(q1 - q2) / cmin - 1.0))
    # each halving of k1sl shrinks the relative residual by ~4 (quadratic)
    assert resid[0] / resid[1] == pytest.approx(4.0, rel=0.15)
    assert resid[1] / resid[2] == pytest.approx(4.0, rel=0.15)


def test_off_resonance_gap_is_the_hyperbola(ref: ReferenceParticle) -> None:
    """A tune distance D away, the eigen-gap opens as sqrt(D^2 + |C^-|^2)."""
    base = _fodo(1.2) * 4
    base = base[:4] + [ThinQuadrupole(0.06)] + base[4:]  # split the planes
    lat0 = Lattice(base, ref)
    qx, qy = tunes(lat0)
    D = abs(qx % 1.0 - qy % 1.0)
    assert D > 0.01  # genuinely off resonance
    lat = Lattice(base[:9] + [ThinSkewQuadrupole(0.01)] + base[9:], ref)
    cmin = closest_tune_approach(lat)
    q1, q2 = normal_mode_tunes(lat)
    assert abs(q1 - q2) == pytest.approx(math.hypot(D, cmin), rel=2e-3)


def test_closest_tune_approach_closed_form(ref: ReferenceParticle) -> None:
    """|C^-| == (1/2pi) sqrt(beta_x beta_y) |k1sl| at the skew quad's own location.

    beta is read from the *uncoupled* optics (a thin skew leaves them unchanged),
    an independent path from the phasor sum inside closest_tune_approach.
    """
    k1sl = 0.008
    cell = _fodo(1.2) * 4
    # skew quad at the ring start -> beta there is the matched entrance beta
    tw = closed_twiss(Lattice(cell, ref))
    expected = math.sqrt(tw.beta_x * tw.beta_y) * abs(k1sl) / (2.0 * math.pi)
    lat = Lattice([ThinSkewQuadrupole(k1sl), *cell], ref)
    assert closest_tune_approach(lat) == pytest.approx(expected, rel=1e-12)


def test_thick_skew_matches_thin_equivalent(ref: ReferenceParticle) -> None:
    """A thick skew quad's sliced |C^-| ~ that of the thin quad with the same k1s*L."""
    L, k1s = 0.2, 0.05
    cell = _fodo(1.2) * 4
    thick = Lattice([SkewQuadrupole(L, k1s), *cell], ref)
    thin = Lattice([ThinSkewQuadrupole(k1s * L), *cell], ref)
    # short magnet: the two agree to O(L^2) (beta varies little across the body)
    assert closest_tune_approach(thick) == pytest.approx(closest_tune_approach(thin), rel=5e-3)


def test_thick_skew_closest_approach_equals_eigen_gap(ref: ReferenceParticle) -> None:
    """Independent CI check of the THICK trapezoidal phasor integral.

    The thick body's own ``(F+D)/2`` focusing is equal in both planes, so it shifts
    Qx and Qy together and leaves the symmetric FODO *on* the difference resonance;
    there the eigen-gap still equals ``|C^-|``. This gates the sliced integral
    against the exact eigenvalue route -- a different code path from the closed form,
    unlike the thick-vs-thin check above (which shares the phasor machinery).
    """
    base = _fodo(1.2) * 4
    lat = Lattice(base[:8] + [SkewQuadrupole(0.2, 0.03)] + base[8:], ref)
    cmin = closest_tune_approach(lat)
    q1, q2 = normal_mode_tunes(lat)
    assert abs(q1 - q2) == pytest.approx(cmin, rel=1e-2)


# ---- the 1/2pi prefactor is DERIVED here, not recalled ----
def test_cminus_prefactor_derived_symbolically() -> None:
    """Re-derive |C^-| = (1/2pi) sqrt(beta_x beta_y) k from a single-kick eigen-gap.

    Uncoupled one-turn R(mu_x) (+) R(mu_y) in physical coords at (beta, alpha=0),
    plus a thin skew kick dpx=k y, dpy=k x. At the resonance mu_x=mu_y=mu the two
    cos(mode) roots split by +- sqrt(bx by) k sin(mu)/2, giving a tune gap
    sqrt(bx by) k / (2pi). The coefficient 1/(2pi) is what the code uses.
    """
    sp = pytest.importorskip("sympy")
    bx, by, mu, k, lam = sp.symbols("beta_x beta_y mu k lam", positive=True)

    def rphys(beta: sp.Expr) -> sp.Matrix:
        N = sp.Matrix([[sp.sqrt(beta), 0], [0, 1 / sp.sqrt(beta)]])
        R = sp.Matrix([[sp.cos(mu), sp.sin(mu)], [-sp.sin(mu), sp.cos(mu)]])
        return N * R * N.inv()

    M0 = sp.zeros(4, 4)
    M0[0:2, 0:2] = rphys(bx)
    M0[2:4, 2:4] = rphys(by)
    K = sp.eye(4)
    K[1, 2] = k  # px <- y
    K[3, 0] = k  # py <- x
    M = K * M0

    cp = sp.expand((M - lam * sp.eye(4)).det())
    a1 = sp.simplify(-cp.coeff(lam, 3))
    a2 = sp.simplify(cp.coeff(lam, 2))
    # cos(mode) roots c solve 4c^2 - 2 a1 c + (a2 - 2) = 0
    c = sp.symbols("c")
    roots = sp.solve(4 * c**2 - 2 * a1 * c + (a2 - 2), c)
    # the two cos(mode) roots are cos(mu) +- sqrt(bx by) k sin(mu) / 2, so their
    # difference / sin(mu) is +- sqrt(bx by) k (the phase gap, sign = root order).
    phase_gap = sp.simplify((roots[0] - roots[1]) / sp.sin(mu))
    # matched beta > 0 fixes the resonance phase to 0 < mu < pi, i.e. sin(mu) > 0
    phase_gap = phase_gap.subs(sp.Abs(sp.sin(mu)), sp.sin(mu))
    dq = sp.Abs(phase_gap) / (2 * sp.pi)  # all of bx, by, k positive
    assert sp.simplify(dq - sp.sqrt(bx * by) * k / (2 * sp.pi)) == 0
