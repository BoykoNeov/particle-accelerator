r"""Analytic gates for the solenoid (S1) — the magnet whose field points along the beam.

The element under test is the first in the package whose field is **longitudinal**, and
almost every gate here exists because that one fact breaks an assumption something else was
built on.

Ordered by how much each can catch:

  * **The ``1/2`` is derived, not recalled.** ``sympy`` solves the paraxial Hamiltonian's
    equations of motion with the solenoid's vector potential and the Larmor rate ``ks/2``
    falls out of the solve; the resulting 4x4 is compared entrywise with
    :meth:`~accsim.elements.solenoid.Solenoid._matrix_body`. Half the cyclotron rate is the
    single most plausible thing to get wrong here, and this project's own record (G1's
    ``eps_y``, G2's coupling matrix, P2 (iii)'s amplitude) is that a remembered coefficient
    is a coefficient that is wrong.
  * **The Larmor factorisation, each factor alone.** ``M4 = Rot(KL) . blockdiag(F, F)`` with
    ``F`` the *quadrupole's* focusing block at ``K^2``, equal in both planes. A half-strength
    body, or one that treats the two planes differently, survives the composed matrix's
    symplecticity and its determinant; it does not survive this.
  * **The hard edge is a factor, and dropping it is visible.** In geometric angles the map's
    velocity rows contain no position at all — a solenoid *rotates* the transverse velocity
    rather than focusing it — and the shear that says so is the fringe field. A map built on
    ``px`` as if it were the angle is wrong at first order in ``ks``.
  * **The momentum dependence**, which ``matrix()`` is blind to by construction, gated as an
    exact identity rather than a tolerance.
  * **Axial symmetry** — a roll does *nothing* to a solenoid, bit-for-bit. No other element
    in the package can say that.
  * **The refusals**, each one a test: the field accessor, the two perturbative coupling
    sums, and the uncoupled Courant-Snyder path.

The absolute conventions (the sign of the coupling, and that ``ks`` is charge-free) are
pinned against xtrack and MAD-X in ``tests/reference/``; what is here is everything that can
be settled without them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    CoupledLatticeError,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    Solenoid,
    closest_tune_approach,
    coupled_twiss,
    normal_mode_tunes,
    resonance_driving_terms,
    survey,
    taper,
    tunes,
)
from accsim.coords import DELTA, DIM, PX, PY, ZETA, X, Y
from accsim.elements.quadrupole import _focusing_block
from accsim.symplectic import is_symplectic, is_symplectic_map_canonical
from accsim.taylor import (
    canonical_map,
    second_order_symplectic_residual,
    taylor_expand,
)

MASS_E = 0.51099895069e6
GAMMA0 = 20.0
LENGTH = 0.9
KS = 0.6

TRANSVERSE = ([X, PX, Y, PY],)

#: A generic probe state: every coordinate nonzero, so no term can hide behind a zero.
STATE = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 1.0e-3, 2.0e-4])


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS_E, GAMMA0, charge=-1.0)


def _t4(M: np.ndarray) -> np.ndarray:
    """The transverse 4x4 of a 6x6, in ``(x, px, y, py)`` order."""
    return M[np.ix_([X, PX, Y, PY], [X, PX, Y, PY])]


def _rotation(theta: float) -> np.ndarray:
    """Rigid rotation of ``(x, y)`` and ``(px, py)`` together, in the 4x4 ordering."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s, 0.0], [0.0, c, 0.0, s], [-s, 0.0, c, 0.0], [0.0, -s, 0.0, c]])


def _edge_shear(K: float) -> np.ndarray:
    """``J``: canonical momentum -> geometric angle inside the magnet, ``x' = px + K y``."""
    J = np.eye(4)
    J[1, 2] = K
    J[3, 0] = -K
    return J


# ================================ the coefficient, derived ================================
def test_map_is_derived_from_the_hamiltonian(ref: ReferenceParticle) -> None:
    r"""Solve the solenoid's equations of motion in sympy; the ``ks/2`` falls out.

    Nothing about the Larmor rate is asserted *into* this derivation. The input is the
    normalised vector potential ``a = (-ks y/2, +ks x/2)`` — which is just "the curl is
    ``ks`` along ``s``", i.e. the definition of the field — and the paraxial kinetic term.
    ``dsolve`` returns the flow; its Jacobian is the shipped matrix.

    The gate is entrywise against :meth:`Solenoid._matrix_body`, so a factor of two anywhere
    in the implementation (the classic solenoid error: using the cyclotron rate where the
    orbit rotates at half of it) fails here at ``O(1)``.
    """
    s, L, ks = sp.symbols("s L k_s", real=True, positive=True)
    x, px, y, py = sp.symbols("x p_x y p_y", real=True)
    Xf, Yf = sp.Function("X")(s), sp.Function("Y")(s)
    PXf, PYf = sp.Function("PX")(s), sp.Function("PY")(s)

    # a_x = -ks y / 2, a_y = +ks x / 2  =>  (curl a)_s = ks. The paraxial Hamiltonian at
    # delta = 0 is the mechanical kinetic energy, (p - a)^2 / 2.
    H = ((PXf + ks * Yf / 2) ** 2 + (PYf - ks * Xf / 2) ** 2) / 2
    eqs = [
        sp.Eq(Xf.diff(s), sp.diff(H, PXf)),
        sp.Eq(Yf.diff(s), sp.diff(H, PYf)),
        sp.Eq(PXf.diff(s), -sp.diff(H, Xf)),
        sp.Eq(PYf.diff(s), -sp.diff(H, Yf)),
    ]
    sol = sp.dsolve(
        eqs,
        [Xf, Yf, PXf, PYf],
        ics={Xf.subs(s, 0): x, Yf.subs(s, 0): y, PXf.subs(s, 0): px, PYf.subs(s, 0): py},
    )
    flow = {eq.lhs: eq.rhs for eq in sol}
    variables = [x, px, y, py]
    derived = sp.Matrix(
        [
            [sp.simplify(sp.diff(flow[f].subs(s, L), v)) for v in variables]
            for f in (Xf, PXf, Yf, PYf)
        ]
    )

    # The Larmor rate is *read off* the solution rather than substituted into it, by asking
    # for the solution's **period**: the orbit closes on itself after a full turn, so the map
    # is the identity at L = 2 pi / (ks/2)... but the length that matters is the one the
    # rotation angle KL = pi asks for. Rot(pi) = -I and F(K^2) at KL = pi is -I too, so the
    # map is +I at K L = pi, i.e. L = 2 pi / ks. A solenoid built at the *cyclotron* rate
    # would have half that period, so the second assert is what makes the first sharp.
    assert sp.simplify(derived.subs(L, 2 * sp.pi / ks)) == sp.eye(4), (
        "the orbit does not have period 2 pi / ks, so it is not rotating at ks/2"
    )
    assert sp.simplify(derived.subs(L, sp.pi / ks)) != sp.eye(4), (
        "the map is periodic at pi/ks, which is the cyclotron rate, not the Larmor rate"
    )

    numeric = np.array(sp.lambdify((ks, L), derived, "numpy")(KS, LENGTH), dtype=float).reshape(
        4, 4
    )
    np.testing.assert_allclose(_t4(Solenoid(LENGTH, KS).matrix(ref)), numeric, rtol=0, atol=1e-15)


def test_larmor_factorisation_each_factor_alone(ref: ReferenceParticle) -> None:
    r"""``M4 = Rot(KL) . blockdiag(F(K^2), F(K^2))``, and both factors are checked on their own.

    This is the gate a composed matrix cannot give you. ``Rot`` must be a *rigid* rotation
    (orthogonal, unit determinant, the same angle in both planes) and ``F`` must be the
    **quadrupole's** focusing block at ``K^2`` — the same function, not a re-derivation — and
    identical in the two planes, which is what makes a solenoid focus both planes equally
    where a quadrupole focuses one and defocuses the other.
    """
    K = 0.5 * KS
    rot = _rotation(K * LENGTH)
    F = _focusing_block(K * K, LENGTH)
    block = np.block([[F, np.zeros((2, 2))], [np.zeros((2, 2)), F]])

    # each factor, alone
    assert np.abs(rot.T @ rot - np.eye(4)).max() < 1e-15  # rigid: orthogonal
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-14)
    assert np.array_equal(block[:2, :2], block[2:, 2:])  # both planes identical, bit-for-bit
    assert F[0, 0] == pytest.approx(math.cos(K * LENGTH), abs=1e-15)  # F is at K^2, not K

    np.testing.assert_allclose(
        _t4(Solenoid(LENGTH, KS).matrix(ref)), rot @ block, rtol=0, atol=1e-15
    )
    # ... and the two factors commute, which is what "equal in both planes" buys.
    assert np.abs(rot @ block - block @ rot).max() < 1e-15


def test_a_half_strength_body_fails_the_factorisation(ref: ReferenceParticle) -> None:
    """The control for the gate above: the two most plausible wrong bodies are rejected.

    A body at the cyclotron rate ``ks`` rather than the Larmor rate ``ks/2``, and a body that
    focuses one plane and defocuses the other (a quadrupole's habit). Both are ``O(1)`` away.
    """
    M4 = _t4(Solenoid(LENGTH, KS).matrix(ref))
    K = 0.5 * KS
    F, F_wrong = _focusing_block(K * K, LENGTH), _focusing_block(KS * KS, LENGTH)
    D = _focusing_block(-K * K, LENGTH)
    zero = np.zeros((2, 2))
    cyclotron = _rotation(KS * LENGTH) @ np.block([[F_wrong, zero], [zero, F_wrong]])
    split_planes = _rotation(K * LENGTH) @ np.block([[F, zero], [zero, D]])
    assert np.abs(M4 - cyclotron).max() > 0.1
    assert np.abs(M4 - split_planes).max() > 0.01


# ================================= the hard edge ==========================================
def test_in_geometric_angles_the_velocity_is_only_rotated(ref: ReferenceParticle) -> None:
    r"""``J M J^-1`` has **no position** in its angle rows, and rotates them by ``2KL``.

    The sentence that explains the element. Conjugating by the hard-edge shear
    ``x' = px + K y`` (the geometric angle *inside* the magnet) turns the map into

        [[1, *, 0, *], [0, cos 2KL, 0, sin 2KL], [0, *, 1, *], [0, -sin 2KL, 0, cos 2KL]]

    — the transverse velocity is rigidly rotated through twice the Larmor angle and its
    magnitude is untouched, while the position rows carry the whole of the focusing. It is
    also why the path lengthening below is a drift's.
    """
    K = 0.5 * KS
    J = _edge_shear(K)
    angles = J @ _t4(Solenoid(LENGTH, KS).matrix(ref)) @ np.linalg.inv(J)

    # the angle rows do not depend on position, at all
    assert abs(angles[1, 0]) < 1e-15 and abs(angles[1, 2]) < 1e-15
    assert abs(angles[3, 0]) < 1e-15 and abs(angles[3, 2]) < 1e-15
    # ... and what they do to the velocity is a rotation by 2KL
    np.testing.assert_allclose(
        angles[np.ix_([1, 3], [1, 3])],
        [
            [math.cos(2 * K * LENGTH), math.sin(2 * K * LENGTH)],
            [-math.sin(2 * K * LENGTH), math.cos(2 * K * LENGTH)],
        ],
        rtol=0,
        atol=1e-15,
    )
    # the position rows, by contrast, are not identity: that is the focusing
    assert abs(angles[0, 1]) > 0.1


def test_dropping_the_fringe_changes_the_map_at_first_order(ref: ReferenceParticle) -> None:
    """A map that treats ``px`` as the angle has silently dropped both faces.

    The edge-free map is the angle-frame one used as if it were the canonical one. It is
    still symplectic and still reduces to a drift at ``ks = 0``, so neither of those gates
    can see the error — but it differs from the true map at **first order in ``ks``**, which
    this asserts by measuring the difference at two strengths and checking it halves.
    """

    def gap(ks: float) -> float:
        K = 0.5 * ks
        M4 = _t4(Solenoid(LENGTH, ks).matrix(ref))
        edgeless = _edge_shear(K) @ M4 @ np.linalg.inv(_edge_shear(K))
        return float(np.abs(M4 - edgeless).max())

    big, small = gap(KS), gap(0.5 * KS)
    assert big > 1e-2, "the fringe must not be a rounding-level correction"
    assert big / small == pytest.approx(2.0, rel=0.05), "the fringe is first order in ks"


# ================================ limits and structure ====================================
def test_zero_strength_is_a_drift_bit_for_bit(ref: ReferenceParticle) -> None:
    """``ks = 0`` is a :class:`Drift`'s matrix exactly, and the approach to it is smooth.

    The closed form is written so that no ``1/K`` is ever formed (every one of them appears
    as ``L sinc(KL)`` times a ``cos`` or a ``sin``), so this needs no branch — and the
    continuity check at ``ks = 1e-12`` is what says the implementation really is entire in
    ``K`` rather than special-casing zero.
    """
    assert np.array_equal(Solenoid(LENGTH, 0.0).matrix(ref), Drift(LENGTH).matrix(ref))
    weak = Solenoid(LENGTH, 1e-12).matrix(ref)
    assert np.abs(weak - Drift(LENGTH).matrix(ref)).max() < 1e-12
    # tracking too, where the drift's own map is exact and the solenoid's is paraxial:
    # they agree to the kinematic term the class docstring names.
    got = Solenoid(LENGTH, 0.0).track(STATE, ref)
    want = Drift(LENGTH).track(STATE, ref)
    assert np.abs(got - want).max() < 1e-11


def test_matrix_is_symplectic_and_is_the_origin_jacobian(ref: ReferenceParticle) -> None:
    """``matrix()`` is symplectic, and it is what ``track`` linearises to at the origin."""
    sol = Solenoid(LENGTH, KS)
    M = sol.matrix(ref)
    assert is_symplectic(M)
    h = 1e-7
    jac = np.zeros((DIM, DIM))
    for j in range(DIM):
        e = np.zeros(DIM)
        e[j] = h
        jac[:, j] = (sol.track(e, ref) - sol.track(-e, ref)) / (2 * h)
    np.testing.assert_allclose(jac, M, rtol=0, atol=1e-9)


@pytest.mark.parametrize("delta", [0.0, 1e-3, -2e-2])
@pytest.mark.parametrize("ks", [KS, -KS])
def test_tracked_map_is_symplectic_off_axis(
    delta: float, ks: float, ref: ReferenceParticle
) -> None:
    """The exact map is symplectic at a real amplitude, in the *canonical* longitudinal pair.

    ``(zeta, delta)`` are not conjugate, so the plain check would reject a correct exact map;
    this is the one that separates a right longitudinal half from the half-fix that leaves
    ``zeta`` linear (the path-lengthening term is exactly what is at stake).
    """
    state = np.array([3.0e-3, 4.0e-4, -2.0e-3, -1.0e-4, 1.0e-3, delta])
    sol = Solenoid(LENGTH, ks)
    assert is_symplectic_map_canonical(lambda st: sol.track(st, ref), state, ref, atol=1e-8)


def test_the_coupling_is_odd_in_ks(ref: ReferenceParticle) -> None:
    """Reversing the field reverses the sense of the coupling and leaves the focusing.

    ``M(-ks)`` is ``M(ks)`` with the two planes exchanged — the coupling blocks change sign,
    the diagonal blocks do not. A map that got the coupling's sign from somewhere other than
    ``ks`` (the charge, say) would not satisfy this.
    """
    plus = _t4(Solenoid(LENGTH, KS).matrix(ref))
    minus = _t4(Solenoid(LENGTH, -KS).matrix(ref))
    np.testing.assert_allclose(minus[:2, :2], plus[:2, :2], rtol=0, atol=1e-15)
    np.testing.assert_allclose(minus[2:, 2:], plus[2:, 2:], rtol=0, atol=1e-15)
    np.testing.assert_allclose(minus[:2, 2:], -plus[:2, 2:], rtol=0, atol=1e-15)
    np.testing.assert_allclose(minus[2:, :2], -plus[2:, :2], rtol=0, atol=1e-15)


def test_ks_is_charge_free() -> None:
    """An electron and a proton at the same ``ks`` get the **same** map, bit-for-bit.

    Measured convention of both reference codes (``q0 = +1`` and ``q0 = -1`` give identical
    maps in xtrack, identical ``re<ij>`` in MAD-X), and the reason
    :meth:`Solenoid._matrix_body` never reads ``ref.charge``. The physical charge sign lives
    in ``ks`` itself, exactly as it does in every ``k`` strength in the package.
    """
    electron = ReferenceParticle.from_gamma(MASS_E, GAMMA0, charge=-1.0)
    proton = ReferenceParticle.from_gamma(MASS_E, GAMMA0, charge=+1.0)
    sol = Solenoid(LENGTH, KS)
    assert np.array_equal(sol.matrix(electron), sol.matrix(proton))
    assert np.array_equal(sol.track(STATE, electron), sol.track(STATE, proton))


# ================================ the momentum dependence =================================
@pytest.mark.parametrize("delta", [1e-4, 1e-3, 1e-2, -5e-3])
def test_the_map_is_exactly_the_matrix_at_the_particles_own_momentum(
    delta: float, ref: ReferenceParticle
) -> None:
    r"""The chromatic law, as an identity rather than a tolerance.

    ``ks`` is normalised to the *reference* rigidity, so a particle of momentum
    ``(1 + delta)`` is rotated at ``ks/(1 + delta)`` — half of that in its orbit. The
    transverse map must therefore be the matrix of a solenoid of strength ``ks/(1+delta)``,
    conjugated by ``diag(1, 1+delta, 1, 1+delta)`` (which converts angles to momenta):

        out = D M4(ks/(1+delta)) D^-1 in.

    This is exact, so it is asserted at ``1e-15`` and not at a physics tolerance. A
    ``delta``-independent solenoid fails it by the ratio the reference suite measures against
    xtrack (``5.6e3`` at ``delta = 1e-3``).
    """
    state = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 0.0, delta])
    got = Solenoid(LENGTH, KS).track(state, ref)
    D = np.diag([1.0, 1.0 + delta, 1.0, 1.0 + delta])
    want = D @ _t4(Solenoid(LENGTH, KS / (1.0 + delta)).matrix(ref)) @ np.linalg.inv(D)
    np.testing.assert_allclose(got[TRANSVERSE], want @ state[TRANSVERSE], rtol=0, atol=1e-15)
    # and the frozen-K map — the one this milestone exists to reject — is far away
    frozen = _t4(Solenoid(LENGTH, KS).matrix(ref)) @ state[TRANSVERSE]
    assert np.abs(got[TRANSVERSE] - frozen).max() > 100.0 * abs(delta) * 1e-6


def test_at_fixed_momentum_the_transverse_map_is_exactly_linear(ref: ReferenceParticle) -> None:
    """Paraxial in the angles means the transverse half is linear at fixed ``delta``.

    Doubling the transverse state doubles the transverse output exactly. This is what makes
    the residual against xtrack (which is not paraxial) a *cubic* one, and it is asserted
    here so that the reference file's cubic-scaling gate is measuring xtrack's extra term
    rather than an accidental nonlinearity of accsim's own.
    """
    sol = Solenoid(LENGTH, KS)
    a = np.array([1.0e-3, 2.0e-4, -3.0e-3, 1.0e-4, 0.0, 1e-3])
    b = a.copy()
    b[TRANSVERSE] *= 2.0
    np.testing.assert_allclose(
        sol.track(b, ref)[TRANSVERSE], 2.0 * sol.track(a, ref)[TRANSVERSE], rtol=0, atol=1e-18
    )


def test_path_lengthening_uses_the_inside_angles(ref: ReferenceParticle) -> None:
    r"""``zeta``'s transverse term is ``L (x'^2 + y'^2)/2`` with the angles **inside** the face.

    The solenoid rotates the transverse velocity without changing its magnitude, so
    ``x'^2 + y'^2`` is constant along the body and the path lengthening is a drift's — but
    evaluated at the angles the particle has *inside* the magnet, ``x' = px/(1+delta) + K y``.
    Using the outside angles instead (i.e. forgetting the fringe) is the error this catches,
    and on this state it is a factor of ``1.5`` on the term, not a rounding difference.
    """
    delta = STATE[DELTA]
    one = 1.0 + delta
    K = 0.5 * KS / one
    xp, yp = STATE[PX] / one, STATE[PY] / one
    inside = 0.5 * LENGTH * ((xp + K * STATE[Y]) ** 2 + (yp - K * STATE[X]) ** 2)
    outside = 0.5 * LENGTH * (xp**2 + yp**2)
    assert inside / outside > 1.2  # the two really are different on this state

    drift_zeta = Drift(LENGTH).track(STATE, ref)[ZETA]
    slip_only = Solenoid(LENGTH, 0.0).track(STATE, ref)[ZETA]
    got = Solenoid(LENGTH, KS).track(STATE, ref)[ZETA]
    E_over_E0 = math.hypot(ref.momentum_eV * one, ref.mass_eV) / ref.total_energy_eV
    predicted = slip_only + (outside - inside) * E_over_E0 / one
    assert got == pytest.approx(predicted, abs=1e-18)
    assert abs(got - drift_zeta) > 1e-9  # a solenoid is not a drift longitudinally either


def test_tracks_a_bunch(ref: ReferenceParticle) -> None:
    """A ``(6, n)`` bunch tracks to the same thing as its columns, one at a time."""
    bunch = np.stack([STATE, -STATE, np.zeros(DIM), 2.0 * STATE], axis=1)
    sol = Solenoid(LENGTH, KS)
    out = sol.track(bunch, ref)
    for j in range(bunch.shape[1]):
        np.testing.assert_allclose(out[:, j], sol.track(bunch[:, j], ref), rtol=0, atol=1e-18)


# ================================== axial symmetry ========================================
@pytest.mark.parametrize("roll", [0.3, math.pi / 4, -1.1, math.pi])
def test_a_roll_does_nothing_to_a_solenoid(roll: float, ref: ReferenceParticle) -> None:
    """A solenoid is invariant about its own axis, so ``roll`` is a no-op — bit-for-bit.

    This element's own sharp gate, and nothing else in the package can be tested this way: a
    rolled quadrupole is a skew quadrupole, a rolled sextupole is a skew sextupole, a rolled
    bend is K2's whole milestone. A rolled solenoid is the same solenoid. It gates the
    alignment conjugation at the same time — if ``_alignment_entry``/``_exit`` were not exact
    inverses, or the rotation acted on only one of position and momentum, this would fail.
    """
    plain = Solenoid(LENGTH, KS)
    rolled = Solenoid(LENGTH, KS, roll=roll)
    np.testing.assert_allclose(rolled.matrix(ref), plain.matrix(ref), rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        rolled.track(STATE, ref), plain.track(STATE, ref), rtol=0, atol=1e-15
    )


def test_a_displacement_leaves_the_matrix_alone_and_lands_in_the_kick(
    ref: ReferenceParticle,
) -> None:
    """K1, restated for the first element that couples: the matrix is untouched.

    A transverse offset moves the closed orbit and nothing else, so ``matrix`` is returned
    unchanged and the whole effect is the constant ``(I - M) d``. For a solenoid that kick
    has **both** transverse planes in it from a purely horizontal offset, which is the
    coupling showing up where it should.
    """
    d = 1.5e-3
    plain, moved = Solenoid(LENGTH, KS), Solenoid(LENGTH, KS, dx=d)
    assert np.array_equal(moved.matrix(ref), plain.matrix(ref))
    offset = np.zeros(DIM)
    offset[X] = d
    np.testing.assert_allclose(
        moved.kick(ref), (np.eye(DIM) - plain.matrix(ref)) @ offset, rtol=0, atol=1e-16
    )
    assert abs(moved.kick(ref)[Y]) > 1e-6  # a horizontal offset kicks vertically too


# ==================================== the refusals ========================================
def test_normalized_field_refuses(ref: ReferenceParticle) -> None:
    """The field accessor raises rather than answering ``(0, 0)``.

    ``(bx, by)`` has no ``s`` component, and a solenoid's field is entirely along ``s``. A
    silent zero would make :mod:`accsim.spin` report no precession through a spin *rotator*
    and :mod:`accsim.radiation_kick` report no radiation for an off-axis particle. Both
    consumers reach an element only through this method, so this one raise covers both — and
    the two asserts below are what would fail if a future S2 gave the accessor an ``s``
    component and forgot to revisit the refusal.
    """
    sol = Solenoid(LENGTH, KS, name="ds")
    with pytest.raises(NotImplementedError, match="longitudinal"):
        sol.normalized_field(1e-3, 1e-3)
    with pytest.raises(NotImplementedError):
        sol.track(STATE, ref, radiation="mean")
    with pytest.raises(NotImplementedError):
        sol.track_with_spin(STATE, np.array([0.0, 1.0, 0.0]), ref)


def _solenoid_ring(ks: float, ref: ReferenceParticle) -> Lattice:
    """A FODO ring with one solenoid in it — the smallest coupled machine this element makes.

    The trailing weak quadrupole splits the tunes, and it is not cosmetic: the bare 4-cell
    FODO has ``Q_x = Q_y`` exactly, i.e. it sits *on* the difference resonance, where any
    coupling at all mixes the modes fully (``gamma_c -> 1/sqrt(2)``) and the mixing then
    *decreases* as the solenoid's own focusing pushes the tunes apart. Measured, and it is
    why the strength-dependence gate below needs an off-resonance ring to say anything.
    """
    cell = [Quadrupole(0.3, 1.2), Drift(0.7), Quadrupole(0.3, -1.2), Drift(0.7)]
    elems = cell * 4
    return Lattice(elems[:4] + [Solenoid(0.4, ks)] + elems[4:] + [Quadrupole(0.01, 5.0)], ref)


def test_the_perturbative_coupling_sums_refuse_a_solenoid(ref: ReferenceParticle) -> None:
    """``closest_tune_approach`` and ``resonance_driving_terms`` raise, and that is right.

    Both walk element *types* and sum skew quadrupoles; a solenoid couples without being one,
    so they would report ``DeltaQ_min = 0`` and ``f1001 = 0`` for a ring that is
    demonstrably coupled. Their measured guard catches it — the solenoid is the first element
    other than a rolled magnet to trip it, and the guard's existence is the reason this
    milestone needed no change in :mod:`accsim.twiss` at all.
    """
    lat = _solenoid_ring(0.5, ref)
    for fn in (closest_tune_approach, resonance_driving_terms):
        with pytest.raises(CoupledLatticeError, match="couples x and y"):
            fn(lat)
    # ... and with the solenoid switched off the same ring is fine
    assert closest_tune_approach(_solenoid_ring(0.0, ref)) == pytest.approx(0.0, abs=1e-15)


def test_the_uncoupled_path_refuses_and_the_normal_mode_path_works(ref: ReferenceParticle) -> None:
    """Courant-Snyder raises; the eigen and Edwards-Teng routes see the solenoid exactly.

    The measured guard in :func:`~accsim.twiss.tunes` is on the one-turn map's off-block, so
    it needs no knowledge of this element. ``normal_mode_tunes`` and ``coupled_twiss`` are
    the whole of what G1/G2 built, and a solenoid arrives at them for free.
    """
    lat = _solenoid_ring(0.5, ref)
    with pytest.raises(CoupledLatticeError):
        tunes(lat)
    q1, q2 = normal_mode_tunes(lat)
    ct = coupled_twiss(lat)
    assert 0.0 < ct.gamma_c < 1.0  # genuinely mixed
    assert ct.gamma_c**2 + np.linalg.det(ct.c_matrix) == pytest.approx(1.0, abs=1e-13)
    assert ct.beta_1 > 0.0 and ct.beta_2 > 0.0
    assert q1 != q2


def test_the_solenoid_mixes_the_modes_quadratically_in_ks(ref: ReferenceParticle) -> None:
    r"""The ring-level effect, and it has a law rather than only a direction.

    With ``ks = 0`` the ring is uncoupled: ``gamma_c`` is exactly 1 and ``C`` is exactly zero.
    Switching the solenoid on mixes the modes, and **``1 - gamma_c`` grows as ``ks^2``** in
    the weak regime — because the coupling matrix ``C`` is first order in ``ks`` and
    ``gamma_c^2 + det C = 1`` makes the mixing angle's cosine second order in it. Doubling
    ``ks`` therefore quadruples the mixing, twice over, and that is asserted rather than mere
    monotonicity: a coefficient error in the solenoid's off-diagonal blocks would still be
    monotone.

    This element has no closed-form ``|C^-|`` in this package (the perturbative sum refuses
    it, above), so the ``ks^2`` law plus the exact ``ks = 0`` reduction is the whole of what
    can be asserted without a reference code. The absolute value is xtrack's business, in
    ``tests/reference/test_solenoid_xtrack.py``.
    """
    off = coupled_twiss(_solenoid_ring(0.0, ref))
    assert off.gamma_c == pytest.approx(1.0, abs=1e-14)
    assert np.abs(off.c_matrix).max() < 1e-14

    strengths = (0.0125, 0.025, 0.05, 0.1)
    mixing = [1.0 - coupled_twiss(_solenoid_ring(ks, ref)).gamma_c for ks in strengths]
    assert mixing[0] > 1e-5, "the weak-regime mixing must be measurable at all"
    ratios = [b / a for a, b in zip(mixing[:-1], mixing[1:], strict=True)]
    assert ratios[0] == pytest.approx(4.0, rel=0.01)
    # ... and it is a *limit*, not a coincidence at one amplitude: the ratio approaches 4
    # monotonically from below as the solenoid weakens (3.93, 3.98, 4.00 as measured).
    assert ratios[0] > ratios[1] > ratios[2]
    assert all(3.9 < r < 4.0 for r in ratios)

    q_off = normal_mode_tunes(_solenoid_ring(0.0, ref))
    q_on = normal_mode_tunes(_solenoid_ring(0.6, ref))
    assert max(abs(a - b) for a, b in zip(q_off, q_on, strict=True)) > 1e-3


# ============================ the two shipped modules it meets ============================
def test_survey_is_blind_to_the_solenoid(ref: ReferenceParticle) -> None:
    """A solenoid is straight, so the ring's geometry does not know it is there.

    R1's blind-to-the-taper gate in a new costume: the walk reads ``length`` and, for a
    bending element, ``angle``. A solenoid has no ``angle``, and a survey that reached for a
    strength would move the machine. Bit-identical tables.
    """
    a = survey(_solenoid_ring(0.0, ref))
    b = survey(_solenoid_ring(0.6, ref))
    assert np.array_equal(a.X, b.X) and np.array_equal(a.Z, b.Z)
    assert np.array_equal(a.theta, b.theta)
    assert np.array_equal(b.Y, np.zeros_like(b.Y))


def test_taper_scales_ks_and_is_q2s_fixed_point(ref: ReferenceParticle) -> None:
    r"""A solenoid is a powered magnet, so a taper scales it — and the choice is forced.

    :func:`accsim.tapering._scaled` raises for an element on neither of its lists, so
    shipping this element *required* the decision rather than allowing a default. It is
    powered: ``ks`` is normalised to the reference rigidity exactly as ``k1`` is, so a beam
    that has sagged to ``1 + delta`` is over-rotated by a design-strength solenoid in the
    same way it is over-focused by a design-strength quadrupole.

    The gate is Q2's fixed point: a solenoid whose ``ks`` has been scaled by ``1 + delta``
    has, *at that momentum*, exactly the design magnet's map.
    """
    delta = -1.7e-3
    design = Solenoid(LENGTH, KS)
    tapered = Solenoid(LENGTH, KS * (1.0 + delta))
    state = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 0.0, delta])
    D = np.diag([1.0, 1.0 + delta, 1.0, 1.0 + delta])
    on_momentum = D @ _t4(design.matrix(ref)) @ np.linalg.inv(D) @ state[TRANSVERSE]
    np.testing.assert_allclose(
        tapered.track(state, ref)[TRANSVERSE], on_momentum, rtol=0, atol=1e-16
    )
    # and the un-tapered magnet is *not* that: the taper is doing something
    assert np.abs(design.track(state, ref)[TRANSVERSE] - on_momentum).max() > 1e-9


def test_the_taper_classifies_a_solenoid_as_powered(ref: ReferenceParticle) -> None:
    """``_scaled`` scales ``ks`` and leaves the geometry alone — Q2's split, one element on.

    The classification is asserted directly rather than through :func:`accsim.taper`, because
    the end-to-end call cannot reach it yet (next test). It is still the right classification
    and the right place for it: the moment S2 gives the field accessor an ``s`` component,
    :func:`accsim.taper` works on a solenoid ring with no further change.
    """
    from accsim.tapering import _scaled

    scaled = _scaled(Solenoid(0.4, 0.6, name="ds"), 1.0 - 1.7e-3)
    assert isinstance(scaled, Solenoid)
    assert scaled.ks == pytest.approx(0.6 * (1.0 - 1.7e-3), rel=1e-15)
    assert scaled.length == 0.4  # geometry untouched
    assert scaled.name == "ds"


def test_tapering_a_ring_with_a_solenoid_is_refused_for_now(ref: ReferenceParticle) -> None:
    """And the honest consequence: you cannot taper a ring that contains a solenoid.

    A taper needs the ring's **radiating** closed orbit, radiation reaches an element through
    :meth:`Solenoid.normalized_field`, and that refuses. So the refusal above is not free —
    it costs the whole of axis Q on any machine with a solenoid in it, and saying so here is
    better than discovering it. It is the right trade all the same: the alternative is a
    silent zero, which on the *spin* side would be a wrong answer rather than a missing one.

    This test is what fails, loudly, when S2 lands — at which point it becomes the end-to-end
    taper gate instead.
    """
    with pytest.raises(NotImplementedError, match="longitudinal"):
        taper(_solenoid_ring(0.6, ref))


# ============================ the second-order map it gets for free =======================
def test_the_second_order_map_is_the_chromatic_term_and_the_path_lengthening(
    ref: ReferenceParticle,
) -> None:
    r"""P1's machinery takes the new element with no change, and says what it is made of.

    :func:`accsim.taylor.taylor_expand` differences ``track`` and needs to know nothing about
    element types, so a solenoid arrives with a second-order map for free. What that map
    *contains* is the milestone restated as structure: **every** non-negligible entry either
    carries a ``delta`` index — the chromatic term, ``K = ks/(2(1+delta))`` — or lies in the
    ``zeta`` row — the path lengthening. Nothing else is second order, because the transverse
    map is exactly linear at fixed momentum (asserted separately above).

    The largest entry outside those two sets is ``5.4e-14``, i.e. the differencing floor; the
    largest inside is ``L/2 = 0.45``, the path lengthening's own coefficient.
    """
    sol = Solenoid(LENGTH, KS)
    m = taylor_expand(lambda st: sol.track(st, ref), np.zeros(DIM))

    np.testing.assert_allclose(m.R, sol.matrix(ref), rtol=0, atol=1e-14)

    outside = np.ones_like(m.T, dtype=bool)
    outside[ZETA] = False  # the path lengthening
    outside[:, DELTA, :] = False  # the chromatic term, either way round
    outside[:, :, DELTA] = False
    assert np.abs(m.T[outside]).max() < 1e-12
    assert np.abs(m.T[ZETA]).max() == pytest.approx(0.5 * LENGTH, rel=1e-9)
    assert np.abs(m.T[:, :, DELTA]).max() > 0.1  # the chromatic term is not small


def test_the_second_order_map_is_symplectic(ref: ReferenceParticle) -> None:
    """And it satisfies P1's second-order symplectic identity, in both longitudinal pairs.

    A solenoid has no transverse-to-``zeta`` coupling in its *linear* map (``R51..R54 = 0``:
    the path lengthening is quadratic, not linear), so unlike the sector bend it passes the
    identity in ``(zeta, delta)`` as well as in the canonical ``(zeta, p_zeta)``. Both are
    asserted, because the equality of the two is the statement that the non-canonical term
    P1 isolated really is absent here rather than merely small.
    """
    sol = Solenoid(LENGTH, KS)
    m = taylor_expand(lambda st: sol.track(st, ref), np.zeros(DIM))
    mc = taylor_expand(canonical_map(lambda st: sol.track(st, ref), ref), np.zeros(DIM))
    assert np.abs(second_order_symplectic_residual(m.R, m.T)).max() < 1e-11
    assert np.abs(second_order_symplectic_residual(mc.R, mc.T)).max() < 1e-11
    assert np.abs(m.R[ZETA, [X, PX, Y, PY]]).max() < 1e-15  # why the two agree
