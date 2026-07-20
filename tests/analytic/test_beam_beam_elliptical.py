r"""C1 — the elliptical (Bassetti-Erskine) weak-strong beam-beam kick.

The Stage-6 kick assumes a **round** strong bunch. This is the ``sigma_x != sigma_y``
generalisation. The stated milestone gate is "reduces to the round ``g(u)`` in the
``sigma_x -> sigma_y`` limit", but that gate alone is **not sufficient**: it is a
*singular* limit (the closed form carries ``1/sqrt(2(sigma_x^2 - sigma_y^2))``, which
is exactly where it blows up), and a formula can be right on-axis and in the round
limit while wrong in its off-axis *angular* structure. The correct complex assignment is
``E_y + i E_x``; writing ``E_x + i E_y`` instead is the classic Bassetti-Erskine error.

So the suite is layered so that a wrong transcription and a wrong reference cannot
cancel — the same discipline used for the hourglass factor (C2):

1. **The field is derived from Coulomb's law, symbolically** — no Bassetti-Erskine
   formula is transcribed as an input. Writing ``1/r^2 = int_0^inf e^{-r^2 t} dt``
   turns the convolution of the 2D point field with the Gaussian charge into an
   elementary Gaussian integral, and sympy returns the ``q``-integral

       S_x = (1/2) int_0^inf dq  x exp(-x^2/(2A) - y^2/(2B)) / (A^{3/2} B^{1/2}),
       S_y = (1/2) int_0^inf dq  y exp(-x^2/(2A) - y^2/(2B)) / (A^{1/2} B^{3/2}),
       A = q + sigma_x^2,   B = q + sigma_y^2,

   with the exponent *falling out* rather than being asserted. ``S = 2 pi eps0 E`` is
   the normalised field shape; the physical kick multiplies it by the Stage-6-validated
   prefactor ``(q2/q1)(2 N r0/gamma)``.
2. **The closed form matches that derived integral** off-axis, to ~1e-12.
3. **An independent 2D Coulomb sum** (polar grid centred on the *field* point, so the
   ``1/rho`` singularity is cancelled by the Jacobian) matches both. It shares no code
   with ``wofz`` — this is what pins the ``E_y + i E_x`` assignment *empirically*.
4. **The round limit** reproduces the validated Stage-6 ``g(u)`` kick, and the linear
   limit reproduces Stage-6's ``K``.
5. **Gauss's law** on the two (now unequal) linear gradients: they must sum to the
   central charge density — an independent check on the overall normalisation.
6. **The honest cost is asserted, not hidden**: the elliptical field is *not* radial,
   so ``L_z`` is no longer conserved (the round beam's invariant). Curl-free survives,
   and that is the property that matters for symplectic tracking.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp
from scipy import integrate

from accsim import PX, PY, BeamBeam, ReferenceParticle, X, Y, beam_beam_tune_shift

REF = ReferenceParticle.from_total_energy(938.272e6, 7.0e12)  # LHC-like proton
N = 1.15e11
SIGMA_X = 2.0e-5
SIGMA_Y = 8.0e-6

# Off-axis probe points (in units of SIGMA_X), deliberately asymmetric in x/y so a
# swapped-component bug cannot hide.
PROBES = [(0.75, 0.30), (-1.00, 0.60), (0.15, -0.45), (2.00, 1.50), (0.40, 2.20)]


def _amplitude(q_ratio: float = 1.0) -> float:
    """Stage-6-validated prefactor ``(q2/q1)(2 N r0/gamma)`` [m]."""
    return q_ratio * 2.0 * N * REF.classical_radius_m / REF.gamma0


# --------------------------------------------------------------------------------
# 1. the derivation
# --------------------------------------------------------------------------------


def test_field_integral_is_derived_from_coulomb_symbolically() -> None:
    """Derive the q-integrand from Coulomb + Gaussian; nothing is remembered.

    ``1/r^2 = int_0^inf e^{-r^2 t} dt`` makes the source integral elementary. After
    the substitution ``t = 1/(2q)`` the standard form must appear *exactly*.
    """
    x, y, xp, yp, t = sp.symbols("x y x_p y_p t", real=True)
    sx, sy = sp.symbols("sigma_x sigma_y", positive=True)
    q = sp.Symbol("q", positive=True)

    rho = 1 / (2 * sp.pi * sx * sy) * sp.exp(-(xp**2) / (2 * sx**2) - yp**2 / (2 * sy**2))

    # E_x: integrate the source out at fixed t (both integrals are Gaussian).
    inner = sp.integrate(
        rho * (x - xp) * sp.exp(-((x - xp) ** 2) * t), (xp, -sp.oo, sp.oo), conds="none"
    )
    ex_t = sp.integrate(inner * sp.exp(-((y - yp) ** 2) * t), (yp, -sp.oo, sp.oo), conds="none")

    # t = 1/(2q) maps t: 0..inf to q: inf..0; |dt/dq| carries the orientation flip.
    derived = (ex_t.subs(t, 1 / (2 * q)) * sp.Abs(sp.diff(1 / (2 * q), q))).subs(sp.exp_polar(0), 1)
    derived = sp.powdenest(sp.expand_power_base(derived, force=True), force=True)

    A, B = q + sx**2, q + sy**2
    expected = (
        x * sp.exp(-(x**2) / (2 * A) - y**2 / (2 * B)) / (2 * A ** sp.Rational(3, 2) * sp.sqrt(B))
    )

    assert sp.simplify(derived - expected) == 0


def test_derived_integral_reduces_to_stage6_round_shape() -> None:
    """The q-integral collapses to the validated round shape when sigma_x == sigma_y.

    Substituting ``w = 1/(q + sigma^2)`` turns it into an elementary exponential
    integral giving ``x (1 - e^{-r^2/2 sigma^2}) / r^2`` — Stage 6's ``g(u)`` form.
    """
    sigma = 1.3e-5
    for fx, fy in PROBES:
        x, y = fx * SIGMA_X, fy * SIGMA_X
        sx_num, _ = _shape_q_integral(x, y, sigma, sigma)
        r2 = x * x + y * y
        expected = x * -math.expm1(-r2 / (2 * sigma**2)) / r2
        assert math.isclose(sx_num, expected, rel_tol=1e-11)


# --------------------------------------------------------------------------------
# independent reference implementations (no shared code with the module under test)
# --------------------------------------------------------------------------------


def _shape_q_integral(x: float, y: float, sx: float, sy: float) -> tuple[float, float]:
    """The derived integral, quadratured. ``q`` is mapped to a finite interval:
    ``quad`` on the raw half-line mis-samples the sharp exponential near ``q = 0``."""
    c = max(sx, sy) ** 2 + x * x + y * y

    def parts(v: float) -> tuple[float, float, float, float]:
        q = c * v / (1.0 - v)
        jac = c / (1.0 - v) ** 2
        a, b = q + sx**2, q + sy**2
        e = math.exp(-(x**2) / (2 * a) - y**2 / (2 * b)) / math.sqrt(a * b)
        return e, a, b, jac

    def fx(v: float) -> float:
        e, a, _, jac = parts(v)
        return x * e / a * jac

    def fy(v: float) -> float:
        e, _, b, jac = parts(v)
        return y * e / b * jac

    vx, _ = integrate.quad(fx, 0.0, 1.0, limit=400)
    vy, _ = integrate.quad(fy, 0.0, 1.0, limit=400)
    return 0.5 * vx, 0.5 * vy


def _shape_direct_coulomb(
    x: float, y: float, sx: float, sy: float, n: int = 400
) -> tuple[float, float]:
    """Brute-force ``S = int rho(r') (r - r')/|r - r'|^2 d^2r'``, no special functions.

    Integrating in polar coordinates *about the field point* cancels the ``1/rho``
    singularity against the Jacobian, leaving a smooth integrand:
    ``S = -int du dp rho(r + u e_p) e_p``. Gauss-Legendre in ``u``; the trapezoid in
    ``p`` is spectrally accurate because that integrand is periodic.
    """
    umax = 9.0 * max(sx, sy) + 3.0 * math.hypot(x, y)
    nodes, weights = np.polynomial.legendre.leggauss(n)
    us = 0.5 * umax * (nodes + 1.0)
    wu = 0.5 * umax * weights
    ps = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    dp = 2 * np.pi / n
    u_grid, p_grid = np.meshgrid(us, ps, indexing="ij")
    xp = x + u_grid * np.cos(p_grid)
    yp = y + u_grid * np.sin(p_grid)
    rho = np.exp(-(xp**2) / (2 * sx**2) - yp**2 / (2 * sy**2)) / (2 * np.pi * sx * sy)
    s_x = -float(np.sum(wu[:, None] * rho * np.cos(p_grid)) * dp)
    s_y = -float(np.sum(wu[:, None] * rho * np.sin(p_grid)) * dp)
    return s_x, s_y


def _code_shape(x: float, y: float, sx: float, sy: float) -> tuple[float, float]:
    """The module's kick, divided back down to the bare field shape."""
    bb = BeamBeam(N, sx, sy)
    out = bb.track(np.array([x, 0.0, y, 0.0, 0.0, 0.0]), REF)
    amp = _amplitude()
    return out[PX] / amp, out[PY] / amp


# --------------------------------------------------------------------------------
# 2 + 3. the discriminating checks
# --------------------------------------------------------------------------------


def test_closed_form_matches_derived_integral() -> None:
    """The shipped closed form == the symbolically derived integral, off-axis."""
    for fx, fy in PROBES:
        x, y = fx * SIGMA_X, fy * SIGMA_X
        got = _code_shape(x, y, SIGMA_X, SIGMA_Y)
        ref = _shape_q_integral(x, y, SIGMA_X, SIGMA_Y)
        assert math.isclose(got[0], ref[0], rel_tol=1e-10)
        assert math.isclose(got[1], ref[1], rel_tol=1e-10)


def test_matches_independent_2d_coulomb_sum() -> None:
    """The kick == a brute-force 2D Coulomb integral that never calls ``wofz``.

    This is the test that pins the ``E_y + i E_x`` component assignment: swapping the
    real/imaginary parts leaves the round limit and the on-axis values intact but
    breaks the off-axis angular structure, which this catches.
    """
    for fx, fy in PROBES:
        x, y = fx * SIGMA_X, fy * SIGMA_X
        got = _code_shape(x, y, SIGMA_X, SIGMA_Y)
        ref = _shape_direct_coulomb(x, y, SIGMA_X, SIGMA_Y)
        assert math.isclose(got[0], ref[0], rel_tol=1e-9)
        assert math.isclose(got[1], ref[1], rel_tol=1e-9)


def test_swapped_components_would_fail_the_coulomb_check() -> None:
    """Guard on the guard: the previous test is *able* to see the classic error.

    If it could not, it would be passing vacuously.
    """
    x, y = 0.75 * SIGMA_X, 0.30 * SIGMA_X
    ref = _shape_direct_coulomb(x, y, SIGMA_X, SIGMA_Y)
    got = _code_shape(x, y, SIGMA_X, SIGMA_Y)
    # The correct values differ from each other, so the swap is detectable...
    assert not math.isclose(ref[0], ref[1], rel_tol=1e-3)
    # ...and the swapped assignment is genuinely wrong.
    assert not math.isclose(got[1], ref[0], rel_tol=1e-3)


# --------------------------------------------------------------------------------
# 4. the stated milestone gate: the round limit
# --------------------------------------------------------------------------------


def test_round_limit_reproduces_stage6_kick() -> None:
    """sigma_y -> sigma_x reproduces the validated round-beam kick element."""
    sigma = 1.5e-5
    round_bb = BeamBeam(N, sigma)
    for eps in (1e-3, 1e-4, 1e-5, 1e-6):
        ell = BeamBeam(N, sigma * (1 + eps), sigma * (1 - eps))
        for fx, fy in PROBES:
            state = np.array([fx * sigma, 0.0, fy * sigma, 0.0, 0.0, 0.0])
            a = round_bb.track(state, REF)
            b = ell.track(state, REF)
            # The elliptical result approaches the round one linearly in eps.
            assert math.isclose(a[PX], b[PX], rel_tol=20 * eps, abs_tol=1e-30)
            assert math.isclose(a[PY], b[PY], rel_tol=20 * eps, abs_tol=1e-30)


def test_exactly_equal_sigmas_use_the_round_path() -> None:
    """sigma_x == sigma_y must not divide by zero; it takes the round branch exactly."""
    sigma = 1.5e-5
    round_bb = BeamBeam(N, sigma)
    ell = BeamBeam(N, sigma, sigma)
    for fx, fy in PROBES:
        state = np.array([fx * sigma, 0.0, fy * sigma, 0.0, 0.0, 0.0])
        a, b = round_bb.track(state, REF), ell.track(state, REF)
        assert np.isfinite(b).all()
        assert a[PX] == pytest.approx(b[PX], rel=1e-15, abs=1e-30)
        assert a[PY] == pytest.approx(b[PY], rel=1e-15, abs=1e-30)


def test_near_round_fallback_is_continuous() -> None:
    """Crossing the round-fallback threshold must not step the answer.

    The threshold is set where the round approximation is already accurate to better
    than the closed form's own achievable precision, so the seam must be invisible.
    """
    sigma = 1.5e-5
    x, y = 0.75 * sigma, 0.30 * sigma
    state = np.array([x, 0.0, y, 0.0, 0.0, 0.0])
    prev = None
    for eps in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 0.0):
        bb = BeamBeam(N, sigma * (1 + eps), sigma * (1 - eps))
        out = bb.track(state, REF)
        assert np.isfinite(out).all()
        if prev is not None:
            assert math.isclose(out[PX], prev[PX], rel_tol=1e-5)
            assert math.isclose(out[PY], prev[PY], rel_tol=1e-5)
        prev = out


# --------------------------------------------------------------------------------
# 5. the linear limit and Gauss's law
# --------------------------------------------------------------------------------


def test_linear_gradients_match_analytic_limit() -> None:
    """K_x = amp/(sx(sx+sy)), K_y = amp/(sy(sx+sy)) — the q-integral at the origin."""
    bb = BeamBeam(N, SIGMA_X, SIGMA_Y)
    kx, ky = bb.strengths(REF)
    amp = _amplitude()
    assert math.isclose(kx, amp / (SIGMA_X * (SIGMA_X + SIGMA_Y)), rel_tol=1e-13)
    assert math.isclose(ky, amp / (SIGMA_Y * (SIGMA_X + SIGMA_Y)), rel_tol=1e-13)
    # The flat plane is focused harder — that is the physical content.
    assert ky > kx

    # matrix() is the Jacobian of track() at the axis.
    M = bb.matrix(REF)
    assert math.isclose(M[PX, X], kx, rel_tol=1e-13)
    assert math.isclose(M[PY, Y], ky, rel_tol=1e-13)

    # ...and track() really does approach it near the axis.
    tiny = 1e-4 * SIGMA_Y
    out = bb.track(np.array([tiny, 0.0, tiny, 0.0, 0.0, 0.0]), REF)
    assert math.isclose(out[PX], kx * tiny, rel_tol=1e-6)
    assert math.isclose(out[PY], ky * tiny, rel_tol=1e-6)


def test_gauss_law_fixes_the_normalisation() -> None:
    """div E = rho/eps0 at the origin: the two gradients must sum to 2 pi rho(0).

    In shape units (S = 2 pi eps0 E) that reads
        dS_x/dx + dS_y/dy = 1/(sigma_x sigma_y),
    an independent constraint on the overall coefficient — it would catch a stray
    factor of 2 or pi that the round limit alone might absorb.
    """
    for sx, sy in [(SIGMA_X, SIGMA_Y), (3e-5, 3e-5), (5e-6, 4e-5)]:
        bb = BeamBeam(N, sx, sy)
        kx, ky = bb.strengths(REF)
        amp = _amplitude()
        assert math.isclose((kx + ky) / amp, 1.0 / (sx * sy), rel_tol=1e-13)


def test_round_strength_matches_stage6() -> None:
    """The round element's scalar K is unchanged, and strengths() agrees with it."""
    sigma = 1.5e-5
    bb = BeamBeam(N, sigma)
    k = bb.strength(REF)
    assert math.isclose(k, _amplitude() / (2 * sigma**2), rel_tol=1e-14)
    assert bb.strengths(REF) == pytest.approx((k, k), rel=1e-14)


def test_scalar_strength_rejected_for_elliptical_beam() -> None:
    """There is no single K for an elliptical beam — asking must fail loudly."""
    bb = BeamBeam(N, SIGMA_X, SIGMA_Y)
    with pytest.raises(ValueError, match="strengths"):
        bb.strength(REF)


# --------------------------------------------------------------------------------
# 6. invariants: what survives and what does not
# --------------------------------------------------------------------------------


def test_force_is_curl_free() -> None:
    """d(Delta px)/dy == d(Delta py)/dx — the kick still derives from a potential.

    This is the property that keeps long-term tracking symplectic, and it survives
    the loss of radial symmetry.
    """
    bb = BeamBeam(N, SIGMA_X, SIGMA_Y)
    h = 1e-4 * SIGMA_Y

    def kick(x: float, y: float) -> tuple[float, float]:
        out = bb.track(np.array([x, 0.0, y, 0.0, 0.0, 0.0]), REF)
        return out[PX], out[PY]

    for fx, fy in PROBES:
        x, y = fx * SIGMA_X, fy * SIGMA_X
        d_px_dy = (kick(x, y + h)[0] - kick(x, y - h)[0]) / (2 * h)
        d_py_dx = (kick(x + h, y)[1] - kick(x - h, y)[1]) / (2 * h)
        assert math.isclose(d_px_dy, d_py_dx, rel_tol=1e-6, abs_tol=1e-12)


def test_angular_momentum_is_not_conserved_and_that_is_expected() -> None:
    """The honest cost of ellipticity: the field is no longer radial, so L_z moves.

    The round beam conserves ``L_z`` exactly (Stage 6). Asserting the *breakage* here
    keeps the round-beam invariant test from being silently over-claimed.
    """
    rng = np.random.default_rng(0)
    n = 400
    states = np.zeros((6, n))
    states[X] = rng.normal(0, 2 * SIGMA_X, n)
    states[Y] = rng.normal(0, 2 * SIGMA_Y, n)
    states[PX] = rng.normal(0, 1e-6, n)
    states[PY] = rng.normal(0, 1e-6, n)
    lz_before = states[X] * states[PY] - states[Y] * states[PX]

    def lz_drift(bb: BeamBeam) -> float:
        out = bb.track(states, REF)
        return float(np.max(np.abs((out[X] * out[PY] - out[Y] * out[PX]) - lz_before)))

    drift_elliptical = lz_drift(BeamBeam(N, SIGMA_X, SIGMA_Y))
    drift_round = lz_drift(BeamBeam(N, SIGMA_X))

    # The round beam is radial -> exactly conserving, down to float noise.
    assert drift_round < 1e-15 * float(np.max(np.abs(lz_before)))
    # The elliptical beam exerts a torque; the effect is physical, not noise.
    assert drift_elliptical > 1e6 * max(drift_round, 1e-300)


# --------------------------------------------------------------------------------
# geometry, signs, validation
# --------------------------------------------------------------------------------


def test_tall_beam_is_the_mirror_of_a_flat_one() -> None:
    """sigma_y > sigma_x must work: swapping both the sizes and the coordinates
    swaps the kick components. (The closed form assumes sigma_x > sigma_y, so this
    exercises the internal axis swap.)"""
    flat = BeamBeam(N, SIGMA_X, SIGMA_Y)
    tall = BeamBeam(N, SIGMA_Y, SIGMA_X)
    for fx, fy in PROBES:
        x, y = fx * SIGMA_X, fy * SIGMA_X
        a = flat.track(np.array([x, 0.0, y, 0.0, 0.0, 0.0]), REF)
        b = tall.track(np.array([y, 0.0, x, 0.0, 0.0, 0.0]), REF)
        assert math.isclose(a[PX], b[PY], rel_tol=1e-12)
        assert math.isclose(a[PY], b[PX], rel_tol=1e-12)


def test_tall_beam_matches_the_coulomb_reference() -> None:
    """The swap is not merely self-consistent — it is right."""
    for fx, fy in PROBES:
        x, y = fx * SIGMA_X, fy * SIGMA_X
        got = _code_shape(x, y, SIGMA_Y, SIGMA_X)
        ref = _shape_direct_coulomb(x, y, SIGMA_Y, SIGMA_X)
        assert math.isclose(got[0], ref[0], rel_tol=1e-9)
        assert math.isclose(got[1], ref[1], rel_tol=1e-9)


def test_sign_like_charges_defocus_opposite_focus() -> None:
    """Unchanged from the round case: the signed q2/q1 sets focus vs defocus."""
    x, y = 0.75 * SIGMA_X, 0.30 * SIGMA_X
    state = np.array([x, 0.0, y, 0.0, 0.0, 0.0])
    same = BeamBeam(N, SIGMA_X, SIGMA_Y, strong_charge=+1.0).track(state, REF)
    opp = BeamBeam(N, SIGMA_X, SIGMA_Y, strong_charge=-1.0).track(state, REF)
    assert same[PX] > 0 and same[PY] > 0  # like charges repel -> defocus
    assert opp[PX] < 0 and opp[PY] < 0  # opposite charges attract -> focus
    assert math.isclose(same[PX], -opp[PX], rel_tol=1e-13)


def test_tune_shift_is_per_plane_for_an_elliptical_beam() -> None:
    """The beam-beam tune shift splits between the planes, and reduces to Stage 6."""
    beta_x, beta_y = 0.55, 0.55
    bb = BeamBeam(N, SIGMA_X, SIGMA_Y)
    dqx, dqy = beam_beam_tune_shift(bb, REF, beta_x, beta_y)
    kx, ky = bb.strengths(REF)
    assert math.isclose(dqx, -beta_x * kx / (4 * math.pi), rel_tol=1e-13)
    assert math.isclose(dqy, -beta_y * ky / (4 * math.pi), rel_tol=1e-13)
    # Flat beam: the vertical plane takes the larger shift.
    assert abs(dqy) > abs(dqx)

    # Round beam: both planes agree, matching the Stage-6 scalar result.
    sigma = 1.5e-5
    rb = BeamBeam(N, sigma)
    rqx, rqy = beam_beam_tune_shift(rb, REF, beta_x, beta_y)
    assert math.isclose(rqx, rqy, rel_tol=1e-14)


def test_invalid_sigma_y_rejected() -> None:
    for bad in (0.0, -1e-6):
        with pytest.raises(ValueError, match="sigma_y"):
            BeamBeam(N, SIGMA_X, bad)
