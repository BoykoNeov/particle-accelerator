"""Stage 6 acceptance (gate 3): the head-on weak-strong beam-beam kick conserves
the expected invariants.

For a round Gaussian strong bunch the kick is radial and derives from a potential.
The concrete invariants (the vague "conserves the expected invariants" acceptance
line):

1. **Matches the independent closed form.** The code (regularised ``g(u)`` form) is
   compared to a separately written ``(q2/q1)(2 N r0/gamma)(1/r)(1-e^{-r^2/2s^2})``
   evaluated with a bare ``1/r`` — same physics, different arithmetic.
2. **Radial ⇒ angular momentum ``L_z = x py - y px`` conserved** (exactly, since a
   thin kick leaves the positions fixed).
3. **Curl-free force** ``d Delta px/dy == d Delta py/dx`` (finite-difference on the
   code) — the potential/symplecticity property, verified symbolically upstream.
4. **Sign from the Lorentz force**: like charges defocus, opposite charges focus.
5. **Linear limit**: the small-amplitude kick and ``matrix()`` reproduce the
   analytic strength ``K``, and ``matrix`` is the Jacobian of ``track`` at the axis.
"""

from __future__ import annotations

import math

import numpy as np

from accsim import PX, PY, BeamBeam, ReferenceParticle, X, Y

REF = ReferenceParticle.from_total_energy(938.272e6, 7.0e12)  # LHC-like proton
N = 1.15e11
SIGMA = 16.63e-6


def _closed_form_kick(x: float, y: float, q_ratio: float) -> tuple[float, float]:
    """Independent Delta p_perp = (q2/q1)(2 N r0/gamma)(r/r^2)(1 - e^{-r^2/2s^2})."""
    r2 = x * x + y * y
    coeff = q_ratio * 2.0 * N * REF.classical_radius_m / REF.gamma0
    radial = coeff * (1.0 - math.exp(-r2 / (2.0 * SIGMA**2))) / r2
    return radial * x, radial * y


def test_kick_matches_independent_closed_form() -> None:
    """Code kick == the bare-1/r closed form at several off-axis points."""
    bb = BeamBeam(N, SIGMA, strong_charge=1.0)
    for x, y in [(3e-5, 0.0), (0.0, 2e-5), (1e-5, -4e-5), (5e-5, 5e-5), (2e-6, 1e-6)]:
        state = np.array([x, 0.0, y, 0.0, 0.0, 0.0])
        out = bb.track(state, REF)
        dpx, dpy = _closed_form_kick(x, y, q_ratio=1.0)
        assert math.isclose(out[PX], dpx, rel_tol=1e-12, abs_tol=1e-18)
        assert math.isclose(out[PY], dpy, rel_tol=1e-12, abs_tol=1e-18)


def test_angular_momentum_conserved_round_beam() -> None:
    """L_z = x py - y px is exactly conserved by the radial kick (many particles)."""
    rng = np.random.default_rng(0)
    n = 500
    states = np.zeros((6, n))
    states[X] = rng.normal(0, 2 * SIGMA, n)
    states[Y] = rng.normal(0, 2 * SIGMA, n)
    states[PX] = rng.normal(0, 1e-6, n)
    states[PY] = rng.normal(0, 1e-6, n)
    bb = BeamBeam(N, SIGMA)
    out = bb.track(states, REF)
    lz_before = states[X] * states[PY] - states[Y] * states[PX]
    lz_after = out[X] * out[PY] - out[Y] * out[PX]
    assert np.allclose(lz_after, lz_before, rtol=0, atol=1e-20)
    # And the kick really did something (guard against a no-op passing trivially).
    assert not np.allclose(out[PX], states[PX])


def test_force_is_curl_free() -> None:
    """d Delta px/dy == d Delta py/dx via central differences on the code."""
    bb = BeamBeam(N, SIGMA)
    x0, y0, h = 2.5e-5, 1.5e-5, 1e-9

    def dp(xx: float, yy: float) -> tuple[float, float]:
        out = bb.track(np.array([xx, 0.0, yy, 0.0, 0.0, 0.0]), REF)
        return out[PX], out[PY]

    dpx_dy = (dp(x0, y0 + h)[0] - dp(x0, y0 - h)[0]) / (2 * h)
    dpy_dx = (dp(x0 + h, y0)[1] - dp(x0 - h, y0)[1]) / (2 * h)
    assert math.isclose(dpx_dy, dpy_dx, rel_tol=1e-6)


def test_sign_like_charges_defocus_opposite_focus() -> None:
    """Lorentz-force sign: q1 q2 > 0 defocuses (K>0), q1 q2 < 0 focuses (K<0)."""
    x = 3e-5
    same = BeamBeam(N, SIGMA, strong_charge=+1.0)  # pp: like charges
    opp = BeamBeam(N, SIGMA, strong_charge=-1.0)  # p-pbar / e+e-: opposite
    dpx_same = same.track(np.array([x, 0, 0, 0, 0, 0.0]), REF)[PX]
    dpx_opp = opp.track(np.array([x, 0, 0, 0, 0, 0.0]), REF)[PX]
    assert dpx_same > 0  # pushed outward -> defocusing
    assert dpx_opp < 0  # pulled inward -> focusing
    assert same.strength(REF) > 0 and opp.strength(REF) < 0


def test_linear_limit_matrix_and_strength() -> None:
    """Small-amplitude kick, matrix R21, and the Jacobian at the axis all == K."""
    bb = BeamBeam(N, SIGMA)
    K = bb.strength(REF)
    # Analytic K = N r0 / (gamma sigma^2) for q2/q1 = 1.
    assert math.isclose(K, N * REF.classical_radius_m / (REF.gamma0 * SIGMA**2), rel_tol=1e-14)

    M = bb.matrix(REF)
    assert math.isclose(M[PX, X], K, rel_tol=1e-14)
    assert math.isclose(M[PY, Y], K, rel_tol=1e-14)
    # Round beam focuses BOTH planes with the same sign (unlike a quadrupole).
    assert M[PX, X] == M[PY, Y]

    # matrix() is the Jacobian of the nonlinear kick at the axis: tiny x -> Delta px = K x.
    tiny = 1e-9
    out = bb.track(np.array([tiny, 0, tiny, 0, 0, 0.0]), REF)
    assert math.isclose(out[PX], K * tiny, rel_tol=1e-6)
    assert math.isclose(out[PY], K * tiny, rel_tol=1e-6)


def test_axis_is_regular() -> None:
    """On-axis kick is exactly zero and no NaN appears for tiny r (g(0)=1 limit)."""
    bb = BeamBeam(N, SIGMA)
    on_axis = bb.track(np.array([0.0, 0, 0.0, 0, 0, 0.0]), REF)
    assert on_axis[PX] == 0.0 and on_axis[PY] == 0.0
    near = bb.track(np.array([1e-15, 0, 0.0, 0, 0, 0.0]), REF)
    assert np.isfinite(near[PX]) and np.isfinite(near[PY])
