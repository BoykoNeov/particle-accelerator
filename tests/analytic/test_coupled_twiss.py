"""Analytic gates for the Edwards-Teng coupled Twiss (G2).

The decomposition under test factorises the transverse one-turn map as

    M4 = V U V^-1,   V = [[gamma_c I, C], [-adj(C), gamma_c I]],   U = diag(A, B),

so the two betatron **normal modes** get ordinary Courant-Snyder parameters
``(beta_1, alpha_1)`` and ``(beta_2, alpha_2)`` even though no plane does.

Gates, ordered by how much they can catch:

  * **Exactness** — the reconstruction residual ``||M4 - V U V^-1||`` and the
    symplecticity constraint ``gamma_c^2 + det C = 1`` must hold to machine
    precision. Together these leave no room for a wrong prefactor: they pin the
    ``lambda`` normalisation the implementation derives.
  * **The Riccati root is re-derived symbolically here** (sympy), not recalled —
    ``lambda = -sgn(Delta)/(|Delta| + R)`` is *solved for* from
    ``n + m X - X q - X p X = 0`` under the ansatz ``X = lambda H``, and compared
    against the shipped value on a real lattice.
  * **Reduction** — with the coupling off, ``gamma_c = 1``, ``C = 0`` and
    ``beta_1/beta_2`` equal the uncoupled Courant-Snyder ``beta_x/beta_y`` exactly.
  * **Consistency with the independent eigenvalue route** — the mode tunes implied
    by ``tr A`` / ``tr B`` must equal :func:`normal_mode_tunes`, which shares no code
    with the Edwards-Teng path.
  * **Physics of the mixing** — ``gamma_c`` decreases monotonically with coupling
    strength, saturates at ``1/sqrt(2)`` (45 degrees) only on the difference
    resonance, and ``det C`` follows the difference-resonance geometry
    ``sin^2 phi = (1 - Delta/G)/2`` with ``G = sqrt(Delta^2 + |C^-|^2)`` — tying the
    G2 tilt to the G1 ``|C^-|`` with no new free coefficient.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    CoupledTwiss,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    SkewQuadrupole,
    ThinQuadrupole,
    ThinSkewQuadrupole,
    beam_sigma,
    closed_twiss,
    closest_tune_approach,
    coupled_beam_sigma,
    coupled_twiss,
    normal_mode_tunes,
    propagate_coupled_twiss,
    propagate_twiss,
    tunes,
)
from accsim.twiss import _adj2, _edwards_teng, _transverse_4d

J4 = np.zeros((4, 4))
J4[0, 1] = J4[2, 3] = 1.0
J4[1, 0] = J4[3, 2] = -1.0


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(0.938272e9, 5.0)


def _fodo(kq: float, cell_len: float = 2.0, nq_len: float = 0.3) -> list:
    d = (cell_len - 2 * nq_len) / 2.0
    return [Quadrupole(nq_len, kq), Drift(d), Quadrupole(nq_len, -kq), Drift(d)]


def _coupled_ring(k1sl: float, ref: ReferenceParticle, split: float = 0.0) -> Lattice:
    """FODO ring with one thin skew kick, and an optional thin quad to split the tunes."""
    base = _fodo(1.2) * 4
    elems = base[:4] + [ThinSkewQuadrupole(k1sl)] + base[4:]
    if split != 0.0:
        elems = elems + [ThinQuadrupole(split)]
    return Lattice(elems, ref)


def _v_matrix(ct: CoupledTwiss) -> np.ndarray:
    return ct.v_matrix


# ============================ exactness of the factorisation ============================
@pytest.mark.parametrize("k1sl", [0.0, 1e-4, 1e-2, 0.05, -0.05, 0.2])
def test_decomposition_reconstructs_the_map(k1sl: float, ref: ReferenceParticle) -> None:
    """``V U V^-1`` returns the original 4x4 to machine precision, for either sign."""
    lat = _coupled_ring(k1sl, ref, split=0.05)
    m4 = _transverse_4d(lat.one_turn_matrix())
    gamma_c, C, A, B = _edwards_teng(lat.one_turn_matrix())
    V = np.block([[gamma_c * np.eye(2), C], [-_adj2(C), gamma_c * np.eye(2)]])
    V_inv = np.block([[gamma_c * np.eye(2), -C], [_adj2(C), gamma_c * np.eye(2)]])
    U = np.block([[A, np.zeros((2, 2))], [np.zeros((2, 2)), B]])
    assert np.abs(V @ U @ V_inv - m4).max() < 1e-13
    # V^-1 really is the inverse (this is the gamma_c^2 + det C = 1 constraint at work)
    assert np.abs(V @ V_inv - np.eye(4)).max() < 1e-14


@pytest.mark.parametrize("k1sl", [0.0, 1e-4, 1e-2, 0.05, -0.05, 0.2])
def test_symplectic_constraint_and_blocks(k1sl: float, ref: ReferenceParticle) -> None:
    """``gamma_c^2 + det C = 1`` exactly; ``V`` symplectic; ``det A = det B = 1``.

    This is the gate that a wrong normalisation of ``C`` cannot survive: an
    off-by-a-factor in ``lambda`` breaks ``gamma_c^2 + det C = 1`` immediately (the
    error is O(1), not O(tolerance)).
    """
    lat = _coupled_ring(k1sl, ref, split=0.05)
    gamma_c, C, A, B = _edwards_teng(lat.one_turn_matrix())
    assert gamma_c**2 + np.linalg.det(C) == pytest.approx(1.0, abs=1e-14)
    V = np.block([[gamma_c * np.eye(2), C], [-_adj2(C), gamma_c * np.eye(2)]])
    assert np.abs(V.T @ J4 @ V - J4).max() < 1e-14
    assert np.linalg.det(A) == pytest.approx(1.0, abs=1e-13)
    assert np.linalg.det(B) == pytest.approx(1.0, abs=1e-13)
    # gamma_c is the *smaller* rotation branch by construction
    assert 1.0 / math.sqrt(2.0) - 1e-14 <= gamma_c <= 1.0 + 1e-14


def test_riccati_root_derived_symbolically(ref: ReferenceParticle) -> None:
    r"""Re-derive ``lambda`` from the Riccati equation instead of trusting the code.

    The decoupling condition is the vanishing of the off-diagonal block of
    ``V^-1 M4 V``, i.e. ``n + m X - X q - X p X = 0`` with ``X = C/gamma_c``. Under
    the (numerically verified) ansatz ``X = lambda H``, ``H = n + adj(p)``, sympy
    solves the resulting scalar equation; the physical root must equal the shipped
    ``lambda = -sgn(Delta)/(|Delta| + sqrt(Delta^2 + det H))``.
    """
    lat = _coupled_ring(0.05, ref, split=0.05)
    m4 = _transverse_4d(lat.one_turn_matrix())
    m, n, p, q = m4[:2, :2], m4[:2, 2:], m4[2:, :2], m4[2:, 2:]
    H = n + _adj2(p)

    lam = sp.symbols("lam", real=True)
    S = sp.Matrix  # exact rationals from the float entries: no re-rounding
    m_s, n_s, q_s, p_s, H_s = (S(a.tolist()).applyfunc(sp.nsimplify) for a in (m, n, q, p, H))
    X = lam * H_s
    resid = sp.expand(n_s + m_s * X - X * q_s - X * p_s * X)
    # every entry must vanish for the same lambda; solve entry (0,0) and check the rest
    roots = sp.solve(sp.Eq(resid[0, 0], 0), lam)
    assert roots, "the ansatz X = lambda*H admits no root"
    delta = 0.5 * (float(np.trace(m)) - float(np.trace(q)))
    root_r = math.sqrt(delta**2 + float(np.linalg.det(H)))
    expected = -math.copysign(1.0, delta) / (abs(delta) + root_r)
    got = min((float(r) for r in roots), key=lambda r: abs(r - expected))
    assert got == pytest.approx(expected, rel=1e-10)
    # and that same root kills the whole 2x2 residual, not just one entry
    assert max(abs(float(e)) for e in resid.subs(lam, sp.Float(got, 30))) < 1e-10
    # finally: it is the root the shipped code actually used
    gamma_c, C, _A, _B = _edwards_teng(lat.one_turn_matrix())
    assert np.abs(C / gamma_c - expected * H).max() < 1e-12


# ============================ reduction to the uncoupled case ============================
def test_uncoupled_reduces_to_courant_snyder(ref: ReferenceParticle) -> None:
    """No coupling: ``gamma_c = 1``, ``C = 0``, and mode betas equal plane betas exactly."""
    lat = Lattice(_fodo(1.2) * 4, ref)
    ct = coupled_twiss(lat)
    tw = closed_twiss(lat)
    assert ct.gamma_c == 1.0
    assert np.abs(ct.c_matrix).max() == 0.0
    assert ct.coupling_angle == 0.0
    assert ct.beta_1 == pytest.approx(tw.beta_x, rel=1e-15)
    assert ct.alpha_1 == pytest.approx(tw.alpha_x, abs=1e-15)
    assert ct.beta_2 == pytest.approx(tw.beta_y, rel=1e-15)
    assert ct.alpha_2 == pytest.approx(tw.alpha_y, abs=1e-15)


def test_uncoupled_reduces_with_split_tunes(ref: ReferenceParticle) -> None:
    """Same reduction on a tune-split lattice (Delta != 0, so a different code branch)."""
    lat = Lattice(_fodo(1.2) * 4 + [ThinQuadrupole(0.05)], ref)
    ct, tw = coupled_twiss(lat), closed_twiss(lat)
    assert ct.gamma_c == pytest.approx(1.0, abs=1e-15)
    assert ct.beta_1 == pytest.approx(tw.beta_x, rel=1e-14)
    assert ct.beta_2 == pytest.approx(tw.beta_y, rel=1e-14)


def test_weak_coupling_perturbs_betas_quadratically(ref: ReferenceParticle) -> None:
    """As ``k1s l -> 0`` the mode betas approach the plane betas as O((k1s l)^2).

    A linear-in-``k1sl`` error in the decomposition (the classic sign/factor slip)
    would show up here as a first-order residual.
    """
    lat0 = _coupled_ring(0.0, ref, split=0.05)
    tw = closed_twiss(lat0)
    resid = []
    for k in (0.02, 0.01, 0.005):
        ct = coupled_twiss(_coupled_ring(k, ref, split=0.05))
        resid.append(abs(ct.beta_1 - tw.beta_x) / tw.beta_x)
    # halving the coupling must quarter the residual (quadratic), not halve it
    assert resid[0] / resid[1] == pytest.approx(4.0, rel=0.15)
    assert resid[1] / resid[2] == pytest.approx(4.0, rel=0.15)


# ============================ agreement with the eigenvalue route ============================
@pytest.mark.parametrize("k1sl", [1e-3, 0.02, 0.05, -0.05, 0.2])
def test_mode_tunes_match_the_eigenvalue_route(k1sl: float, ref: ReferenceParticle) -> None:
    """``tr A``/``tr B`` give the same mode tunes as the independent eigen-decomposition.

    :func:`normal_mode_tunes` diagonalises ``M4`` directly and shares no code with the
    Edwards-Teng factorisation, so this is a genuine cross-check of both.
    """
    lat = _coupled_ring(k1sl, ref, split=0.05)
    _g, _C, A, B = _edwards_teng(lat.one_turn_matrix())
    q1, q2 = normal_mode_tunes(lat)
    for block, q in ((A, q1), (B, q2)):
        cos_mu = 0.5 * (block[0, 0] + block[1, 1])
        sin_mu = math.copysign(math.sqrt(1.0 - cos_mu**2), block[0, 1])
        q_et = math.atan2(sin_mu, cos_mu) / (2.0 * math.pi) % 1.0
        assert q_et == pytest.approx(q, abs=1e-12)


def test_mode_labelling_follows_the_x_like_mode(ref: ReferenceParticle) -> None:
    """Mode 1 stays the x-like one: beta_1 tracks beta_x off resonance, both signs of Delta."""
    for split in (0.05, -0.05):
        lat_u = Lattice(_fodo(1.2) * 4 + [ThinQuadrupole(split)], ref)
        tw = closed_twiss(lat_u)
        ct = coupled_twiss(_coupled_ring(0.01, ref, split=split))
        assert abs(ct.beta_1 - tw.beta_x) < abs(ct.beta_1 - tw.beta_y)
        assert abs(ct.beta_2 - tw.beta_y) < abs(ct.beta_2 - tw.beta_x)


# ============================ physics of the mixing angle ============================
def test_mixing_angle_saturates_at_45_degrees_on_resonance(ref: ReferenceParticle) -> None:
    """On the difference resonance the modes are 50/50: ``gamma_c = 1/sqrt(2)`` exactly.

    The symmetric FODO has ``Q_x = Q_y``, so ``Delta = 0`` identically and the mixing
    is maximal *regardless* of how weak the skew is — the hallmark of a resonance.
    """
    for k1sl in (1e-4, 1e-2, 0.1):
        ct = coupled_twiss(_coupled_ring(k1sl, ref))
        assert ct.gamma_c == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-12)
        assert ct.coupling_angle == pytest.approx(math.pi / 4.0, abs=1e-9)


def test_mixing_angle_grows_monotonically_with_coupling(ref: ReferenceParticle) -> None:
    """Off resonance, stronger skew -> smaller gamma_c (more mixing), monotonically."""
    angles = [
        coupled_twiss(_coupled_ring(k, ref, split=0.05)).coupling_angle
        for k in (0.0, 0.01, 0.03, 0.06, 0.12)
    ]
    assert all(b > a for a, b in zip(angles, angles[1:], strict=False))
    assert angles[0] == 0.0
    assert angles[-1] < math.pi / 4.0


def test_tilt_follows_the_difference_resonance_geometry(ref: ReferenceParticle) -> None:
    r"""``sin^2 phi`` matches the G1 geometry ``(1 - Delta/G)/2``, ``G = sqrt(Delta^2+|C^-|^2)``.

    This ties the G2 mixing angle to the G1 coupling coefficient with **no new free
    coefficient**: ``det C = sin^2 phi`` must equal the same ``sin^2`` of the
    difference-resonance mixing that sets the eigen-emittance sharing. ``Delta`` here
    is the decoupled tune distance to the resonance and ``|C^-|`` is
    :func:`closest_tune_approach` — both computed by unrelated code paths.
    """
    for split, k1sl in ((0.05, 0.02), (0.05, 0.05), (-0.05, 0.03), (0.02, 0.02)):
        lat = _coupled_ring(k1sl, ref, split=split)
        lat_u = Lattice(_fodo(1.2) * 4 + [ThinQuadrupole(split)], ref)
        qx, qy = tunes(lat_u)
        delta = qx - qy
        delta -= round(delta)  # distance to the nearest difference resonance
        c_minus = closest_tune_approach(lat)
        g = math.hypot(delta, c_minus)
        expected = 0.5 * (1.0 - abs(delta) / g)  # sin^2 phi
        ct = coupled_twiss(lat)
        assert np.linalg.det(ct.c_matrix) == pytest.approx(expected, rel=0.06)
        assert 1.0 - ct.gamma_c**2 == pytest.approx(expected, rel=0.06)


# ============================ coupled dispersion ============================
def test_skew_at_dispersion_generates_vertical_dispersion(ref: ReferenceParticle) -> None:
    """A skew quad at nonzero ``D_x`` makes ``D_y != 0`` — solved from the coupled 4x4.

    The uncoupled path cannot see this (it would refuse the lattice); here the matched
    dispersion comes from the full 4x4, so vertical dispersion appears for free. Its
    size must scale linearly with the skew strength.
    """
    from accsim import Dipole

    cell = [Dipole(1.0, 0.05), Quadrupole(0.3, 1.2), Drift(0.7), Quadrupole(0.3, -1.2), Drift(0.7)]
    dy = []
    for k1sl in (0.0, 0.005, 0.01):
        elems = cell * 6
        elems = elems[:3] + [ThinSkewQuadrupole(k1sl)] + elems[3:]
        ct = coupled_twiss(Lattice(elems, ref))
        dy.append(abs(ct.disp_y))
    assert dy[0] == pytest.approx(0.0, abs=1e-14)
    assert dy[1] > 1e-6
    assert dy[2] / dy[1] == pytest.approx(2.0, rel=0.05)  # linear in k1s


# ============================ thick skew + guards ============================
def test_thick_skew_quadrupole_is_handled(ref: ReferenceParticle) -> None:
    """The decomposition is a property of the map, so a thick skew works identically."""
    base = _fodo(1.2) * 4
    lat = Lattice(base[:4] + [SkewQuadrupole(0.2, 0.25)] + base[4:] + [ThinQuadrupole(0.05)], ref)
    ct = coupled_twiss(lat)
    gamma_c, C, A, B = _edwards_teng(lat.one_turn_matrix())
    assert gamma_c**2 + np.linalg.det(C) == pytest.approx(1.0, abs=1e-14)
    assert ct.beta_1 > 0.0 and ct.beta_2 > 0.0
    q1, q2 = normal_mode_tunes(lat)
    cos_mu = 0.5 * (A[0, 0] + A[1, 1])
    sin_mu = math.copysign(math.sqrt(1.0 - cos_mu**2), A[0, 1])
    assert math.atan2(sin_mu, cos_mu) / (2.0 * math.pi) % 1.0 == pytest.approx(q1, abs=1e-12)


def test_coupling_angle_bounded_and_v_matrix_consistent(ref: ReferenceParticle) -> None:
    """``v_matrix`` rebuilds ``V`` from the stored scalars, and phi never exceeds 45 deg."""
    lat = _coupled_ring(0.05, ref, split=0.05)
    ct = coupled_twiss(lat)
    gamma_c, C, _A, _B = _edwards_teng(lat.one_turn_matrix())
    V = _v_matrix(ct)
    assert np.abs(V[:2, 2:] - C).max() < 1e-15
    assert V[0, 0] == pytest.approx(gamma_c, abs=1e-15)
    assert 0.0 <= ct.coupling_angle <= math.pi / 4.0 + 1e-12


# ============================ propagation around the ring ============================
def test_propagation_reduces_to_uncoupled_twiss(ref: ReferenceParticle) -> None:
    """No coupling: beta_1(s)/beta_2(s) equal the Courant-Snyder betas at every boundary.

    The local-rematch route (this function) and forward Twiss transport
    (:func:`propagate_twiss`) share no code, so agreeing at every point is a real gate.
    """
    lat = Lattice(_fodo(1.2) * 4 + [ThinQuadrupole(0.05)], ref)
    pts = propagate_coupled_twiss(lat)
    ref_pts = propagate_twiss(lat, closed_twiss(lat))
    assert len(pts) == len(ref_pts)
    for c, u in zip(pts, ref_pts, strict=True):
        assert c.s == pytest.approx(u.s, abs=1e-15)
        assert c.beta_1 == pytest.approx(u.beta_x, rel=1e-11)
        assert c.alpha_1 == pytest.approx(u.alpha_x, abs=1e-11)
        assert c.beta_2 == pytest.approx(u.beta_y, rel=1e-11)
        assert c.alpha_2 == pytest.approx(u.alpha_y, abs=1e-11)
        assert c.gamma_c == pytest.approx(1.0, abs=1e-12)


def test_propagation_is_periodic_and_continuous(ref: ReferenceParticle) -> None:
    """Optics close on themselves after one turn, with no mode-label swap in between.

    The no-swap check is done at *weak* coupling and as a bound on the largest
    deviation over the whole ring: mode 1 must stay within a fraction of a percent of
    the uncoupled horizontal plane everywhere. A label swap at any point where the two
    plane betas differ (they range over 6.1-9.3 m here) would show up as a ~40%
    deviation. Comparing "is beta_1 nearer beta_x than beta_y" point-by-point would
    *not* work: the two plane betas cross several times per cell, and near a crossing
    the comparison is meaningless.
    """
    lat = _coupled_ring(0.03, ref, split=0.05)
    pts = propagate_coupled_twiss(lat)
    assert pts[-1].beta_1 == pytest.approx(pts[0].beta_1, rel=1e-10)
    assert pts[-1].beta_2 == pytest.approx(pts[0].beta_2, rel=1e-10)
    assert pts[-1].gamma_c == pytest.approx(pts[0].gamma_c, abs=1e-10)

    lat0 = _coupled_ring(0.0, ref, split=0.05)
    uncoupled = propagate_twiss(lat0, closed_twiss(lat0))
    weak = propagate_coupled_twiss(_coupled_ring(0.005, ref, split=0.05))
    dev_1 = max(abs(c.beta_1 - u.beta_x) / u.beta_x for c, u in zip(weak, uncoupled, strict=True))
    dev_2 = max(abs(c.beta_2 - u.beta_y) / u.beta_y for c, u in zip(weak, uncoupled, strict=True))
    assert dev_1 < 5e-3
    assert dev_2 < 5e-3


def test_propagated_points_stay_exact(ref: ReferenceParticle) -> None:
    """The decomposition invariants hold at *every* propagated point, not just the start."""
    lat = _coupled_ring(0.05, ref, split=0.05)
    for ct in propagate_coupled_twiss(lat):
        assert ct.gamma_c**2 + np.linalg.det(ct.c_matrix) == pytest.approx(1.0, abs=1e-12)
        assert ct.beta_1 > 0.0
        assert ct.beta_2 > 0.0


# ============================ projected beam sizes ============================
def test_coupled_beam_sigma_reduces_to_uncoupled(ref: ReferenceParticle) -> None:
    """No coupling: projected sizes equal the plain ``beam_sigma`` and the tilt is zero."""
    lat = Lattice(_fodo(1.2) * 4 + [ThinQuadrupole(0.05)], ref)
    e1, e2 = 2e-9, 5e-10
    sx, sy, tilt = coupled_beam_sigma(propagate_coupled_twiss(lat), e1, e2)
    ux, uy = beam_sigma(propagate_twiss(lat, closed_twiss(lat)), e1, e2)
    assert np.allclose(sx, ux, rtol=1e-11)
    assert np.allclose(sy, uy, rtol=1e-11)
    assert np.allclose(tilt, 0.0, atol=1e-14)


def test_four_dimensional_emittance_is_invariant(ref: ReferenceParticle) -> None:
    r"""``det Sigma = (emit_1 emit_2)^2`` at every point, coupled or not.

    ``V`` is symplectic (``det V = 1``) and ``det B_i = beta gamma - alpha^2 = 1``, so
    the 4D emittance is untouched by the coupling — a strong check that the sigma
    matrix is built with the right transformation (``V ... V^T``, not ``V^-1``).
    """
    e1, e2 = 3e-9, 4e-10
    for k1sl in (0.0, 0.02, 0.08):
        for ct in propagate_coupled_twiss(_coupled_ring(k1sl, ref, split=0.05)):
            b1 = np.array([[ct.beta_1, -ct.alpha_1], [-ct.alpha_1, ct.gamma_1]])
            b2 = np.array([[ct.beta_2, -ct.alpha_2], [-ct.alpha_2, ct.gamma_2]])
            mode = np.block([[e1 * b1, np.zeros((2, 2))], [np.zeros((2, 2)), e2 * b2]])
            sigma = ct.v_matrix @ mode @ ct.v_matrix.T
            assert np.linalg.det(sigma) == pytest.approx((e1 * e2) ** 2, rel=1e-9)


def test_single_mode_leaks_into_the_vertical(ref: ReferenceParticle) -> None:
    r"""Excite mode 1 only (``emit_2 = 0``): the projected vertical size grows as ``k1s l``.

    With ``emit_2 = 0`` the vertical block of ``Sigma`` is ``emit_1 adj(C) B_1 adj(C)^T``,
    i.e. ``O(det C) = O(sin^2 phi)``, so ``sigma_y^2`` is quadratic in the coupling and
    ``sigma_y`` itself is linear. A flat beam on a coupled machine is never flat: this
    is the *projected* effect that
    :func:`~accsim.radiation.equilibrium_emittances_coupled` deliberately does not
    describe (it returns eigen-emittances).
    """
    e1 = 2e-9

    def leaked(k1sl: float, split: float) -> float:
        pts = propagate_coupled_twiss(_coupled_ring(k1sl, ref, split=split))
        _sx, sy, _t = coupled_beam_sigma(pts, e1, 0.0)
        return max(sy)

    # Well away from the resonance (|C^-|/Delta <= 0.11 here) the leak is linear in k1s.
    weak = {k: leaked(k, 0.2) for k in (0.005, 0.01, 0.02)}
    assert weak[0.005] > 0.0
    assert weak[0.01] / weak[0.005] == pytest.approx(2.0, rel=0.01)
    assert weak[0.02] / weak[0.01] == pytest.approx(2.0, rel=0.01)

    # Closer in, the mixing angle saturates (sin^2 phi -> 1/2) and the growth falls
    # *below* linear. This is the physics, not a tolerance: asserting it here keeps the
    # linear gate above honest about the regime it holds in.
    strong = {k: leaked(k, 0.05) for k in (0.02, 0.04)}
    assert strong[0.04] / strong[0.02] < 1.85


def test_projected_vertical_size_exceeds_the_mode_size(ref: ReferenceParticle) -> None:
    """Coupled: ``sigma_y > sqrt(emit_2 beta_2)`` — mode 1 leaks into the vertical plane.

    This is the projected-vs-eigen distinction the G1 emittance work flagged, made
    explicit and gated here.
    """
    e1, e2 = 4e-9, 2e-10
    pts = propagate_coupled_twiss(_coupled_ring(0.05, ref, split=0.05))
    _sx, sy, _t = coupled_beam_sigma(pts, e1, e2)
    mode_only = [math.sqrt(e2 * ct.beta_2) for ct in pts]
    assert all(s > m for s, m in zip(sy, mode_only, strict=True))
    # and the excess vanishes with the coupling
    pts0 = propagate_coupled_twiss(_coupled_ring(0.0, ref, split=0.05))
    _sx0, sy0, _t0 = coupled_beam_sigma(pts0, e1, e2)
    assert np.allclose(sy0, [math.sqrt(e2 * ct.beta_2) for ct in pts0], rtol=1e-12)


def test_beam_tilt_flips_sign_with_the_skew(ref: ReferenceParticle) -> None:
    """The x-y tilt of the projected ellipse reverses when the skew gradient reverses."""
    e1, e2 = 4e-9, 2e-10
    tilts = {}
    for k1sl in (0.05, -0.05):
        pts = propagate_coupled_twiss(_coupled_ring(k1sl, ref, split=0.05))
        _sx, _sy, tilt = coupled_beam_sigma(pts, e1, e2)
        tilts[k1sl] = tilt[len(tilt) // 2]
    assert tilts[0.05] != pytest.approx(0.0, abs=1e-6)
    assert tilts[0.05] == pytest.approx(-tilts[-0.05], rel=1e-6)


def test_dispersive_contribution_adds_in_quadrature(ref: ReferenceParticle) -> None:
    """``sigma_delta`` enters exactly as ``(D sigma_delta)^2`` on top of the betatron size."""
    from accsim import Dipole

    cell = [Dipole(1.0, 0.05), Quadrupole(0.3, 1.2), Drift(0.7), Quadrupole(0.3, -1.2), Drift(0.7)]
    elems = cell * 6
    elems = elems[:3] + [ThinSkewQuadrupole(0.01)] + elems[3:]
    pts = propagate_coupled_twiss(Lattice(elems, ref))
    e1, e2, sd = 3e-9, 3e-10, 1e-3
    sx0, sy0, _t0 = coupled_beam_sigma(pts, e1, e2, sigma_delta=0.0)
    sx1, sy1, _t1 = coupled_beam_sigma(pts, e1, e2, sigma_delta=sd)
    for a, b, ct in zip(sx0, sx1, pts, strict=True):
        assert b * b == pytest.approx(a * a + (ct.disp_x * sd) ** 2, rel=1e-12)
    for a, b, ct in zip(sy0, sy1, pts, strict=True):
        assert b * b == pytest.approx(a * a + (ct.disp_y * sd) ** 2, rel=1e-12)
    assert max(abs(ct.disp_y) for ct in pts) > 1e-6  # the skew really did make D_y
