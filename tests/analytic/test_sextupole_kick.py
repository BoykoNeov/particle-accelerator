r"""J1 — the sextupole's nonlinear kick as a real tracking map.

Until now ``Sextupole``/``ThinSextupole`` carried ``k2`` but no map: the linear
matrix is a drift (thin: the identity), and the only place ``k2`` was felt was the
first-order feed-down chromaticity. This suite gates the actual kick

    Delta px = -1/2 k2l (x^2 - y^2),     Delta py = +k2l (x y).

**What gates the ``1/2`` — and what conspicuously does not.**

The tempting gates are structural and *all* of them are blind to the coefficient:

- symplecticity at any amplitude — true for **any** gradient kick, whatever its
  overall strength, so a kick scaled by ``1/2``, ``1``, ``2`` or ``-1/2`` passes
  identically: a mis-scaled sextupole is still a sextupole;
- ``d(Delta px)/dy == d(Delta py)/dx`` (curl-free, i.e. Maxwell) — the *same*
  statement as symplecticity for a thin kick, not an independent one. It does bite
  on a kick whose two components are scaled *differently* (that is no longer a
  field), which is a different class of bug from the strength being wrong;
- Jacobian at the origin is the identity, hence tunes/beta unmoved — true for any
  purely quadratic kick;
- a sympy derivation from a potential ``V`` chosen to reproduce the kick — that
  re-proves the algebra rather than the physics.

Two gates do discriminate, and they are the point of this file:

1. :func:`test_thin_kick_matches_multipole_field_expansion` derives the kick from
   the **field**, ``B_y + i B_x = (B rho) sum_n k_n (x + i y)^n / n!``, and anchors
   that expansion at ``n = 1`` against the :class:`Quadrupole` element already
   validated against xtrack and MAD-X. The sextupole's ``1/2`` is then the ``1/n!``
   of the expansion whose ``n = 1`` term is independently right.
2. :func:`test_tracked_feeddown_chromaticity_matches_analytic` linearises the new
   nonlinear map about the **off-momentum closed orbit** and reads the resulting
   chromaticity straight off the tracked one-turn Jacobian, comparing it with
   ``chromaticity - natural_chromaticity`` — the feed-down term pinned symbolically
   in ``test_sextupole.py`` and cross-checked against xtrack's real tracking at
   ``rel ~ 5e-4`` in ``tests/reference/test_sextupole_xtrack.py``. A wrong
   coefficient rescales the recovered gradient by exactly that factor;
   :func:`test_wrong_kick_coefficient_is_decisively_rejected` proves the gate has
   teeth by feeding it a kick with ``1`` in place of ``1/2``.

The structural checks are still here — they catch a *different* class of bug (a map
that is not a magnet at all) — but they are labelled for what they are.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    DELTA,
    DIM,
    PX,
    PY,
    ZETA,
    Dipole,
    Drift,
    Lattice,
    Particle,
    Quadrupole,
    ReferenceParticle,
    Sextupole,
    ThinSextupole,
    Tracker,
    X,
    Y,
    chromaticity,
    is_symplectic_map,
    jacobian,
    natural_chromaticity,
    tracked_tunes,
    tunes,
)

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


# --------------------------------------------------------------------------
# 1. The coefficient, from the field expansion (anchored on the quadrupole)
# --------------------------------------------------------------------------


def _symbolic_multipole_kick(order: int) -> tuple[sp.Expr, sp.Expr]:
    r"""Thin-lens ``(Delta px, Delta py)`` of the pure ``n = order`` normal multipole.

    From the MAD-X / Xsuite field expansion

        B_y + i B_x = (B rho) * sum_n k_n (x + i y)^n / n!,

    and the thin-lens kick of a transverse field over length ``L``,

        Delta px = -(1/B rho) B_y L,      Delta py = +(1/B rho) B_x L.

    Returned in terms of the symbols ``x, y, kn, L``. Nothing here knows what the
    sextupole code does — the expansion is written once and evaluated at two
    different ``n``.
    """
    x, y, kn, L = sp.symbols("x y k_n L", real=True)
    z = x + sp.I * y
    field_over_brho = kn * z**order / sp.factorial(order)
    By = sp.re(sp.expand(field_over_brho))
    Bx = sp.im(sp.expand(field_over_brho))
    return sp.simplify(-By * L), sp.simplify(Bx * L)


def test_field_expansion_reproduces_the_quadrupole_element(ref: ReferenceParticle) -> None:
    """``n = 1`` of the expansion must be the Quadrupole this package already validates.

    This is the anchor that makes the ``n = 2`` coefficient non-circular: the same
    ``1/n!`` series that will supply the sextupole's ``1/2`` is checked here against
    an element cross-validated against both xtrack and MAD-X. ``Delta px = -k1 L x``
    and ``Delta py = +k1 L y`` is the thin-lens limit of ``Quadrupole``.
    """
    x, y, kn, L = sp.symbols("x y k_n L", real=True)
    dpx, dpy = _symbolic_multipole_kick(1)
    assert sp.simplify(dpx - (-kn * L * x)) == 0
    assert sp.simplify(dpy - (+kn * L * y)) == 0

    # ...and the real element agrees to first order in L (thin-lens limit).
    k1, length = 1.7, 1e-4
    M = Quadrupole(length, k1).matrix(ref)
    assert M[PX, X] == pytest.approx(-k1 * length, rel=1e-7)
    assert M[PY, Y] == pytest.approx(+k1 * length, rel=1e-7)


def test_thin_kick_matches_multipole_field_expansion(ref: ReferenceParticle) -> None:
    """``n = 2`` of the *same* expansion is exactly ``ThinSextupole.track``.

    The ``1/2`` is ``1/2!``. Evaluated at generic amplitudes, not at a special point.
    """
    x, y, kn, L = sp.symbols("x y k_n L", real=True)
    dpx_sym, dpy_sym = _symbolic_multipole_kick(2)

    # The symbolic form is the documented kick, with k2l = k_n * L.
    assert sp.simplify(dpx_sym - (-kn * L * (x**2 - y**2) / 2)) == 0
    assert sp.simplify(dpy_sym - (+kn * L * x * y)) == 0

    k2l = 3.5
    elem = ThinSextupole(k2l)
    for x0, y0 in [(1e-3, 4e-4), (-2e-3, 1e-3), (5e-4, -3e-3), (0.0, 2e-3), (2e-3, 0.0)]:
        state = np.array([x0, 0.1, y0, -0.2, 0.0, 0.0])
        out = elem.track(state, ref)
        # k2l = kn * L, realised as kn = k2l with unit length.
        subs = {x: x0, y: y0, kn: k2l, L: 1.0}
        assert out[PX] - state[PX] == pytest.approx(float(dpx_sym.subs(subs)), abs=1e-15)
        assert out[PY] - state[PY] == pytest.approx(float(dpy_sym.subs(subs)), abs=1e-15)


# --------------------------------------------------------------------------
# 2. The coefficient, from feed-down: tracked chromaticity vs. the analytic term
# --------------------------------------------------------------------------


def _dispersive_lattice(ref: ReferenceParticle, sextupole_factory) -> Lattice:
    """A FODO-with-dipoles cell (nonzero D_x) carrying one sextupole at dispersion."""
    cell = [
        Quadrupole(0.3, 1.2),
        Drift(0.5),
        sextupole_factory(),
        Drift(0.5),
        Dipole(1.0, 0.12),
        Quadrupole(0.3, -1.2),
        Dipole(1.0, 0.12),
        Drift(0.5),
    ]
    return Lattice(cell * 3, ref)


def _transverse_tunes_of(M: np.ndarray) -> tuple[float, float]:
    """Fractional tunes from the 2x2 blocks of an uncoupled transverse map."""
    out = []
    for a, b in ((X, PX), (Y, PY)):
        half_trace = 0.5 * (M[a, a] + M[b, b])
        if abs(half_trace) >= 1.0:
            raise AssertionError(f"unstable tracked map: |tr/2| = {abs(half_trace)}")
        q = math.acos(half_trace) / (2.0 * math.pi)
        # sin(2 pi Q) has the sign of M[a, b] (= beta sin mu, beta > 0).
        out.append(q if M[a, b] >= 0.0 else 1.0 - q)
    return out[0], out[1]


def _tracked_tunes_at_delta(lattice: Lattice, delta: float) -> tuple[float, float]:
    """Tunes of the *nonlinear* one-turn map, linearised about its closed orbit at ``delta``.

    ``delta`` is invariant here (no RF), and ``zeta`` only slips, so the transverse
    closed orbit is the fixed point of the one-turn map with ``zeta`` and ``delta``
    pinned. Newton on that map, then the Jacobian at the fixed point *is* the linear
    optics an off-momentum particle sees — including whatever gradient the sextupole
    feeds down at its orbit offset. accsim's linear element matrices carry no
    ``delta`` dependence of their own, so the entire ``delta``-dependence of these
    tunes is the sextupole feed-down.
    """
    tracker = Tracker(lattice)

    def turn(state: np.ndarray) -> np.ndarray:
        out = tracker._track_once(np.asarray(state, dtype=float).copy())
        out[ZETA] = 0.0
        out[DELTA] = delta
        return out

    co = np.zeros(DIM)
    co[DELTA] = delta
    for _ in range(40):
        residual = turn(co) - co
        if np.max(np.abs(residual)) < 1e-15:
            break
        Jm = jacobian(turn, co, step=1e-8)
        co = co - np.linalg.solve(Jm - np.eye(DIM), residual)
    else:  # pragma: no cover - the lattice is well inside the stable region
        raise AssertionError("closed-orbit Newton did not converge")

    return _transverse_tunes_of(jacobian(turn, co, step=1e-8))


def _tracked_feeddown_chromaticity(lattice: Lattice, delta: float = 1e-4) -> tuple[float, float]:
    """``dQ/ddelta`` measured by central difference of the tracked, linearised map."""
    qx_p, qy_p = _tracked_tunes_at_delta(lattice, +delta)
    qx_m, qy_m = _tracked_tunes_at_delta(lattice, -delta)
    return (qx_p - qx_m) / (2.0 * delta), (qy_p - qy_m) / (2.0 * delta)


def test_tracked_feeddown_chromaticity_matches_analytic(ref: ReferenceParticle) -> None:
    r"""**The gate on the ``1/2``.** Tracked feed-down == the analytic feed-down term.

    ``chromaticity - natural_chromaticity`` is accsim's first-order sextupole term
    ``+/-(1/4pi) oint beta k2 D_x ds``, pinned symbolically and cross-checked against
    xtrack's real nonlinear tracking. Here the *same* number is re-measured by
    linearising the new nonlinear map about the off-momentum closed orbit — a route
    that shares no code with the analytic formula and that scales linearly with the
    kick coefficient. The two agreeing pins the coefficient, and the opposite signs
    in the two planes pin the ``x^2 - y^2`` structure.
    """
    k2l = 2.0
    lat = _dispersive_lattice(ref, lambda: ThinSextupole(k2l))

    fx = chromaticity(lat)[0] - natural_chromaticity(lat)[0]
    fy = chromaticity(lat)[1] - natural_chromaticity(lat)[1]
    tx, ty = _tracked_feeddown_chromaticity(lat)

    assert fx > 0.0 and fy < 0.0  # the x^2 - y^2 structure, plane by plane
    assert tx == pytest.approx(fx, rel=1e-5)
    assert ty == pytest.approx(fy, rel=1e-5)


class _MisscaledSextupole(ThinSextupole):
    """A *consistently* mis-scaled sextupole: ``1`` where the expansion says ``1/2``.

    Both components are scaled together, so this is still the gradient of a potential
    — it is exactly a sextupole of strength ``2 k2l`` wearing the label ``k2l``. Every
    structural property of the real map survives; only the strength is wrong.
    """

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        out = np.array(state, dtype=float, copy=True)
        out[PX] -= 1.0 * self.k2l * (out[X] ** 2 - out[Y] ** 2)
        out[PY] += 2.0 * self.k2l * out[X] * out[Y]
        return out


class _NonGradientSextupole(ThinSextupole):
    """The ``px`` coefficient wrong and the ``py`` coefficient right — not a magnet.

    Scaling one component alone breaks ``d(Delta px)/dy == d(Delta py)/dx``, so the
    kick is no longer the gradient of any potential and no longer describes a
    curl-free transverse field.
    """

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        out = np.array(state, dtype=float, copy=True)
        out[PX] -= 1.0 * self.k2l * (out[X] ** 2 - out[Y] ** 2)
        out[PY] += self.k2l * out[X] * out[Y]
        return out


def test_wrong_kick_coefficient_is_decisively_rejected(ref: ReferenceParticle) -> None:
    """**The feed-down gate has teeth**: doubling the coefficient doubles the answer.

    Every structural gate in this file passes for :class:`_MisscaledSextupole` — it
    is a gradient kick, symplectic at every amplitude, the identity at the origin,
    curl-free. It has to be: it *is* a sextupole, just twice as strong as it claims.
    No amount of structural checking can see that, because the structure is right and
    only the strength is wrong. The feed-down comparison sees it as a clean factor of
    two, not a marginal tolerance miss.
    """
    k2l = 2.0
    lat = _dispersive_lattice(ref, lambda: _MisscaledSextupole(k2l))

    fx = chromaticity(lat)[0] - natural_chromaticity(lat)[0]
    tx, _ = _tracked_feeddown_chromaticity(lat)

    assert tx == pytest.approx(2.0 * fx, rel=1e-4)  # exactly the wrong factor
    assert tx != pytest.approx(fx, rel=0.5)  # and nowhere near right

    # The blind gates, on the same wrong map: all green.
    bad = _MisscaledSextupole(k2l)
    state = np.array([2e-3, 1e-4, -1e-3, 5e-5, 0.0, 0.0])
    assert is_symplectic_map(lambda s: bad.track(s, ref), state)
    assert np.allclose(jacobian(lambda s: bad.track(s, ref), np.zeros(DIM)), np.eye(DIM))
    Jm = jacobian(lambda s: bad.track(s, ref), state)
    assert Jm[PX, Y] == pytest.approx(Jm[PY, X], abs=1e-9)


def test_symplecticity_rejects_a_kick_that_is_not_a_gradient(ref: ReferenceParticle) -> None:
    """What the structural gates *are* for: a kick that is not a magnetic field at all.

    Scaling ``Delta px`` without ``Delta py`` gives a field with curl — unphysical,
    and non-symplectic, so long-term tracking through it would gain or lose
    emittance from nothing. This is the class of bug symplecticity catches, and it
    is a different class from getting the strength wrong.
    """
    bad = _NonGradientSextupole(2.0)
    state = np.array([2e-3, 1e-4, -1e-3, 5e-5, 0.0, 0.0])
    assert not is_symplectic_map(lambda s: bad.track(s, ref), state)
    Jm = jacobian(lambda s: bad.track(s, ref), state)
    assert Jm[PX, Y] != pytest.approx(Jm[PY, X], abs=1e-9)


# --------------------------------------------------------------------------
# 3. Structural gates (blind to the coefficient — they catch a different bug)
# --------------------------------------------------------------------------


def test_thin_kick_is_symplectic_at_finite_amplitude(ref: ReferenceParticle) -> None:
    """Symplectic *away from the origin*, where a thin kick is not trivially so.

    At ``(x, y) = 0`` every quadratic kick has the identity Jacobian and passes
    vacuously; the amplitudes here are large enough that the off-diagonal
    ``-k2l x``, ``+k2l y`` entries are of order ``1e-2``.
    """
    elem = ThinSextupole(5.0)
    for x0, y0 in [(3e-3, 2e-3), (-4e-3, 1e-3), (1e-2, -8e-3)]:
        state = np.array([x0, 1e-4, y0, -2e-4, 1e-3, 1e-4])
        Jm = jacobian(lambda s: elem.track(s, ref), state)
        assert abs(Jm[PX, X]) > 1e-3  # non-vacuous: the Jacobian really moved
        assert is_symplectic_map(lambda s: elem.track(s, ref), state)


def test_thin_kick_jacobian_at_origin_is_the_identity(ref: ReferenceParticle) -> None:
    """The nonlinear map has *no* linear part — so ``matrix`` staying the identity is right.

    This is why adding a real kick cannot move beta, dispersion or the tunes of the
    linear lattice: the linearisation about the design orbit is unchanged.
    """
    elem = ThinSextupole(7.0)
    Jm = jacobian(lambda s: elem.track(s, ref), np.zeros(DIM))
    assert np.allclose(Jm, np.eye(DIM), atol=1e-12)
    assert np.allclose(Jm, elem.matrix(ref), atol=1e-12)


def test_kick_is_curl_free_maxwell(ref: ReferenceParticle) -> None:
    """``d(Delta px)/dy == d(Delta py)/dx``: the transverse field has no curl.

    Equivalent to symplecticity for a thin kick (both say the kick is a gradient),
    and stated separately only because it is the *Maxwell* reading of it — the same
    condition that forced the curvature-sextupole term in the combined-function
    dipole (F2).
    """
    elem = ThinSextupole(4.0)
    state = np.array([2e-3, 0.0, -1.5e-3, 0.0, 0.0, 0.0])
    Jm = jacobian(lambda s: elem.track(s, ref), state)
    assert Jm[PX, Y] == pytest.approx(Jm[PY, X], abs=1e-9)


def test_kick_vanishes_on_axis_and_flips_sign_between_planes(ref: ReferenceParticle) -> None:
    """A particle on the magnetic axis is untouched; the two planes see opposite signs."""
    elem = ThinSextupole(3.0)
    on_axis = np.array([0.0, 1e-4, 0.0, 2e-4, 1e-3, 1e-4])
    assert np.allclose(elem.track(on_axis, ref), on_axis, atol=0.0)

    a = 2e-3
    horizontal = elem.track(np.array([a, 0.0, 0.0, 0.0, 0.0, 0.0]), ref)
    vertical = elem.track(np.array([0.0, 0.0, a, 0.0, 0.0, 0.0]), ref)
    assert horizontal[PX] == pytest.approx(-0.5 * elem.k2l * a**2)
    assert vertical[PX] == pytest.approx(+0.5 * elem.k2l * a**2)
    assert horizontal[PY] == 0.0 and vertical[PY] == 0.0


def test_kick_only_touches_transverse_momenta(ref: ReferenceParticle) -> None:
    """Positions and the longitudinal pair pass through a thin kick untouched."""
    elem = ThinSextupole(6.0)
    state = np.array([1e-3, 2e-4, -5e-4, 3e-4, 7e-3, 2e-4])
    out = elem.track(state, ref)
    for i in (X, Y, ZETA, DELTA):
        assert out[i] == state[i]


def test_kick_tracks_a_bunch_columnwise(ref: ReferenceParticle) -> None:
    """A ``(6, n)`` bunch gets the same map as n single particles (the loss-track path)."""
    elem = ThinSextupole(2.5)
    rng = np.random.default_rng(7)
    states = rng.normal(scale=1e-3, size=(DIM, 12))
    bunched = elem.track(states, ref)
    for j in range(states.shape[1]):
        assert np.allclose(bunched[:, j], elem.track(states[:, j], ref), atol=0.0)


def test_reversed_kick_undoes_it(ref: ReferenceParticle) -> None:
    """``k2l`` then ``-k2l`` is the identity — a thin kick has no length to disagree over."""
    state = np.array([3e-3, 1e-4, -2e-3, 5e-5, 1e-3, 1e-4])
    there = ThinSextupole(9.0).track(state, ref)
    back = ThinSextupole(-9.0).track(there, ref)
    assert np.allclose(back, state, atol=1e-18)


# --------------------------------------------------------------------------
# 4. The thick element: drift-kick-drift
# --------------------------------------------------------------------------


def test_thick_sextupole_with_zero_k2_is_exactly_a_drift(ref: ReferenceParticle) -> None:
    """At ``k2 = 0`` the integrator must collapse onto the linear drift map identically."""
    state = np.array([1e-3, 2e-4, -5e-4, 3e-4, 1e-3, 2e-4])
    for n in (1, 3, 8):
        elem = Sextupole(0.4, 0.0, n_slices=n)
        assert np.allclose(elem.track(state, ref), elem.matrix(ref) @ state, atol=0.0)
        assert np.allclose(elem.matrix(ref), Drift(0.4).matrix(ref), atol=0.0)


class _KickingSextupole(Sextupole):
    """A sextupole with a (physically fictitious) constant kick, to test the contract.

    Overrides ``_kick_body``, the constant-part hook — K1 moved the extension point
    there when :meth:`Element.kick` grew the misalignment term ``(I - M) d``, and
    this class overriding the *public* ``kick`` instead is precisely how the two
    would drift apart unnoticed.
    """

    def _kick_body(self, ref: ReferenceParticle) -> np.ndarray:
        k = np.zeros(DIM)
        k[PX] = 1e-5
        return k


def test_thick_track_respects_the_affine_contract_at_zero_strength(
    ref: ReferenceParticle,
) -> None:
    """A ``_track_body()`` override must not drop ``_kick_body()`` — I1's contract, gated.

    ``Sextupole._kick_body()`` is the inherited zero, so a dropped constant part would
    be invisible in every physical lattice and would sit there until some later element
    inherited the shortcut. Subclassing in a nonzero kick makes the omission
    observable.

    It is also the gate that caught K1's own version of the trap: the zero-strength
    shortcut used to read ``super().track(...)``, which after K1 re-enters the
    misalignment wrapper and would shift the state **twice**.
    """
    state = np.array([1e-3, 2e-4, -5e-4, 3e-4, 1e-3, 2e-4])
    elem = _KickingSextupole(0.4, 0.0)
    expected = elem.matrix(ref) @ state + elem.kick(ref)
    assert np.allclose(elem.track(state, ref), expected, atol=0.0)


def test_thick_sextupole_is_symplectic_at_every_slicing(ref: ReferenceParticle) -> None:
    """Drift-kick-drift is symplectic *exactly* — that is the reason to use it."""
    state = np.array([4e-3, 1e-3, -3e-3, 5e-4, 1e-3, 1e-4])
    for n in (1, 2, 5):
        elem = Sextupole(0.5, 12.0, n_slices=n)
        assert is_symplectic_map(lambda s, e=elem: e.track(s, ref), state)


def test_single_slice_thick_is_exactly_drift_kick_drift(ref: ReferenceParticle) -> None:
    """The definition, asserted rather than assumed: one slice *is* the split map.

    ``Sextupole(L, k2)`` at ``n_slices = 1`` is bit-for-bit ``Drift(L/2)`` then
    ``ThinSextupole(k2 L)`` then ``Drift(L/2)``. Worth pinning because it is what
    makes the accuracy statements below meaningful — and because a test that
    compared the thick element against *this* construction would be measuring
    nothing at all.
    """
    length, k2 = 0.5, 12.0
    state = np.array([3e-3, 1e-4, -2e-3, 2e-4, 1e-3, 2e-4])
    half = Drift(length / 2).matrix(ref)
    split = half @ ThinSextupole(k2 * length).track(half @ state, ref)
    assert np.allclose(Sextupole(length, k2).track(state, ref), split, atol=0.0)


def test_thick_sextupole_remainder_is_third_order_in_length(ref: ReferenceParticle) -> None:
    """At fixed ``k2``, the splitting remainder falls as ``L^3`` — a second-order integrator.

    Halving the length must cut the discrepancy from the converged answer by ~8.
    This is the statement that "drift-kick-drift is second order": one step of size
    ``L`` carries an ``O(L^3)`` Baker-Campbell-Hausdorff remainder, so ``n`` steps of
    size ``L/n`` carry ``O(L^3/n^2)`` in total.

    Note what is *not* claimed: at fixed **integrated** strength ``k2l`` (so
    ``k2 ~ 1/L``) the same remainder falls only as ``L``, because the commutator term
    quadratic in the strength picks up ``k2^2 L^3 = k2l^2 L``. The thin-lens limit is
    therefore approached more slowly than the naive reading of "second order"
    suggests.
    """
    k2 = 15.0
    state = np.array([3e-3, 1e-4, -2e-3, 2e-4, 0.0, 0.0])

    errors = []
    for length in (0.4, 0.2, 0.1):
        one = Sextupole(length, k2, n_slices=1).track(state, ref)
        converged = Sextupole(length, k2, n_slices=512).track(state, ref)
        errors.append(float(np.max(np.abs(one - converged))))

    assert errors[0] > 0.0
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert fine / coarse == pytest.approx(0.125, rel=0.05)


def test_thick_sextupole_converges_in_slices_as_one_over_n_squared(
    ref: ReferenceParticle,
) -> None:
    """The same second-order statement in the slice count: error ~ 1/n_slices^2.

    This is what makes ``n_slices`` a knob rather than decoration — and the reason
    a single-slice thick sextupole differing from xtrack's own slicing is an
    integration remainder, not a bug.
    """
    state = np.array([4e-3, 1e-4, -3e-3, 2e-4, 0.0, 0.0])
    exact = Sextupole(0.6, 15.0, n_slices=256).track(state, ref)

    errors = [
        float(np.max(np.abs(Sextupole(0.6, 15.0, n_slices=n).track(state, ref) - exact)))
        for n in (2, 4, 8)
    ]
    assert errors[0] > 0.0
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert fine / coarse == pytest.approx(0.25, rel=0.1)


def test_thick_sextupole_rejects_a_non_positive_slice_count() -> None:
    with pytest.raises(ValueError, match="n_slices"):
        Sextupole(0.3, 1.0, n_slices=0)


# --------------------------------------------------------------------------
# 5. What the new map does (and does not) do to the lattice
# --------------------------------------------------------------------------


def test_linear_optics_are_untouched_by_the_new_map(ref: ReferenceParticle) -> None:
    """Turning the kick on must not move the *linear* tunes — it has no linear part."""
    with_k2 = _dispersive_lattice(ref, lambda: ThinSextupole(20.0))
    without = _dispersive_lattice(ref, lambda: ThinSextupole(0.0))
    assert tunes(with_k2) == pytest.approx(tunes(without), abs=1e-14)


def test_tracked_tunes_see_the_kick_only_at_amplitude(ref: ReferenceParticle) -> None:
    """Small-amplitude tracked tunes match the linear ones; large amplitude detunes.

    Amplitude-dependent detuning from a sextupole is *second* order in ``k2`` (the
    first-order term averages away), so this gates its existence and its sign-free
    growth with amplitude, not a coefficient — the closed form for it is not claimed
    anywhere in the package.
    """
    lat = _dispersive_lattice(ref, lambda: ThinSextupole(20.0))
    qx_lin, qy_lin = tunes(lat)

    small = tracked_tunes(lat, n_turns=1024, x0=1e-8, y0=1e-8, nonlinear=True)
    assert small[0] == pytest.approx(qx_lin % 1.0, abs=1e-6)
    assert small[1] == pytest.approx(qy_lin % 1.0, abs=1e-6)

    large = tracked_tunes(lat, n_turns=1024, x0=1e-3, y0=1e-3, nonlinear=True)
    assert abs(large[0] - small[0]) > 1e-5  # the map is live, not decoration


def test_linear_tracking_silently_ignores_the_kick(ref: ReferenceParticle) -> None:
    """``nonlinear=False`` drops the sextupole kick — documented, and gated as documented.

    A lattice with a sextupole is no longer a lattice whose two tracking paths agree;
    this test exists so that fact is asserted rather than discovered.
    """
    lat = _dispersive_lattice(ref, lambda: ThinSextupole(20.0))
    p = Particle(x=2e-3, y=1e-3)
    linear = Tracker(lat).track(p, nonlinear=False)
    nonlinear = Tracker(lat).track(p, nonlinear=True)
    assert abs(nonlinear.px - linear.px) > 1e-6

    # ...and with k2l = 0 they agree again, so the difference is the kick and not
    # some unrelated divergence between the two code paths.
    flat = _dispersive_lattice(ref, lambda: ThinSextupole(0.0))
    a = Tracker(flat).track(p, nonlinear=False)
    b = Tracker(flat).track(p, nonlinear=True)
    assert np.allclose(a.state, b.state, atol=1e-15)
