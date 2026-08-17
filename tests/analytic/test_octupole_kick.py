r"""J2 (part 1) — the octupole's nonlinear kick as a real tracking map.

The kick gated here is

    Delta px = -1/6 k3l (x^3 - 3 x y^2),     Delta py = +1/6 k3l (3 x^2 y - y^3).

**What gates the ``1/6``.** J1 established the trap and it applies verbatim: every
*structural* check is blind to the overall coefficient, because a mis-scaled
octupole is still an octupole. Symplecticity holds for any gradient kick at any
strength; the curl-free (Maxwell) condition is the same statement for a thin kick,
not an independent one; the Jacobian at the origin is the identity for any purely
cubic kick, so beta and the linear tunes are untouched whatever the strength; and a
sympy derivation from a potential ``V`` reverse-engineered out of the kick re-proves
the algebra, not the physics. A deliberately mis-scaled octupole (``1`` in place of
``1/6``) is carried through all of them here to prove they are blind.

What discriminates in *this* file is :func:`test_thin_kick_matches_multipole_field_expansion`,
which descends the ``1/6`` from the field expansion

    B_y + i B_x = (B rho) * sum_n k_n (x + i y)^n / n!

starting at ``n = 1`` — the :class:`Quadrupole` validated against both xtrack and
MAD-X — and passing through ``n = 2``, the sextupole J1 validated. The octupole's
``1/6`` is then the ``1/3!`` of a series whose earlier terms are independently
right, rather than an assertion about itself.

The *physics* gate — first-order amplitude-dependent detuning, which the octupole
exists for and which the sextupole cannot supply a closed form for — is
``test_amplitude_detuning.py``. It pins the same coefficient a second time, by a
route that goes through tracking rather than through the map's algebra.
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
    Bunch,
    Corrector,
    Drift,
    Lattice,
    Octupole,
    Particle,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    ThinSextupole,
    Tracker,
    X,
    Y,
    is_symplectic_map,
    jacobian,
    tunes,
)
from accsim.orbit import closed_orbit_nonlinear, linearised_element_maps, linearised_lattice
from accsim.twiss import (
    chromaticity_on_orbit,
    closed_twiss,
    natural_chromaticity_on_orbit,
    propagate_twiss,
    tunes_on_orbit,
)

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _mis_scaled_kick(state: np.ndarray, k3l: float) -> np.ndarray:
    """The octupole kick with ``1`` in place of ``1/6`` — still a field, still wrong.

    Consistently mis-scaled in *both* components, so it remains curl-free and hence
    exactly symplectic: this is the bug the structural gates cannot see.
    """
    out = np.array(state, dtype=float, copy=True)
    x, y = out[X], out[Y]
    out[PX] -= k3l * (x**3 - 3.0 * x * y**2)
    out[PY] += k3l * (3.0 * x**2 * y - y**3)
    return out


# --------------------------------------------------------------------------
# 1. The coefficient, from the field expansion (anchored on quad and sextupole)
# --------------------------------------------------------------------------


def _symbolic_multipole_kick(order: int) -> tuple[sp.Expr, sp.Expr]:
    r"""Thin-lens ``(Delta px, Delta py)`` of the pure ``n = order`` normal multipole.

    From the MAD-X / Xsuite field expansion

        B_y + i B_x = (B rho) * sum_n k_n (x + i y)^n / n!,

    and the thin-lens kick of a transverse field over length ``L``,

        Delta px = -(1/B rho) B_y L,      Delta py = +(1/B rho) B_x L.

    Written once and evaluated at three different ``n`` — nothing here knows what
    any element's code does.
    """
    x, y, kn, L = sp.symbols("x y k_n L", real=True)
    z = x + sp.I * y
    field_over_brho = kn * z**order / sp.factorial(order)
    By = sp.re(sp.expand(field_over_brho))
    Bx = sp.im(sp.expand(field_over_brho))
    return sp.simplify(-By * L), sp.simplify(Bx * L)


def test_expansion_anchors_on_the_quadrupole_and_sextupole(ref: ReferenceParticle) -> None:
    """``n = 1`` and ``n = 2`` of the series must be the elements already validated.

    This is what makes the ``n = 3`` coefficient non-circular. ``n = 1`` is the
    thin-lens limit of :class:`Quadrupole` (cross-validated against xtrack *and*
    MAD-X); ``n = 2`` is :class:`ThinSextupole`'s kick, pinned against xtrack's
    tracking in J1. The ``1/n!`` that will supply the octupole's ``1/6`` is
    therefore already carrying two independent verdicts before it is used.
    """
    x, y, kn, L = sp.symbols("x y k_n L", real=True)

    dpx1, dpy1 = _symbolic_multipole_kick(1)
    assert sp.simplify(dpx1 - (-kn * L * x)) == 0
    assert sp.simplify(dpy1 - (+kn * L * y)) == 0
    k1, length = 1.7, 1e-4
    M = Quadrupole(length, k1).matrix(ref)
    assert M[PX, X] == pytest.approx(-k1 * length, rel=1e-7)
    assert M[PY, Y] == pytest.approx(+k1 * length, rel=1e-7)

    dpx2, dpy2 = _symbolic_multipole_kick(2)
    assert sp.simplify(dpx2 - (-kn * L * (x**2 - y**2) / 2)) == 0
    assert sp.simplify(dpy2 - (+kn * L * x * y)) == 0
    k2l, x0, y0 = 3.5, 1.3e-3, -7e-4
    state = np.array([x0, 0.1, y0, -0.2, 0.0, 0.0])
    out = ThinSextupole(k2l).track(state, ref)
    subs2 = {x: x0, y: y0, kn: k2l, L: 1.0}
    assert out[PX] - state[PX] == pytest.approx(float(dpx2.subs(subs2)), abs=1e-15)
    assert out[PY] - state[PY] == pytest.approx(float(dpy2.subs(subs2)), abs=1e-15)


def test_thin_kick_matches_multipole_field_expansion(ref: ReferenceParticle) -> None:
    """``n = 3`` of the *same* expansion is exactly ``ThinOctupole.track``.

    The ``1/6`` is ``1/3!``. Evaluated at generic amplitudes in both planes, not at
    a special point, so the ``x^3``, ``x y^2``, ``x^2 y`` and ``y^3`` terms are all
    exercised independently.
    """
    x, y, kn, L = sp.symbols("x y k_n L", real=True)
    dpx_sym, dpy_sym = _symbolic_multipole_kick(3)

    assert sp.simplify(dpx_sym - (-kn * L * (x**3 - 3 * x * y**2) / 6)) == 0
    assert sp.simplify(dpy_sym - (+kn * L * (3 * x**2 * y - y**3) / 6)) == 0

    k3l = 420.0
    elem = ThinOctupole(k3l)
    for x0, y0 in [
        (1e-3, 4e-4),
        (-2e-3, 1e-3),
        (5e-4, -3e-3),
        (0.0, 2e-3),
        (2e-3, 0.0),
        (1.5e-3, 1.5e-3),  # x^3 - 3xy^2 and 3x^2 y - y^3 both non-degenerate
    ]:
        state = np.array([x0, 0.1, y0, -0.2, 0.0, 0.0])
        out = elem.track(state, ref)
        subs = {x: x0, y: y0, kn: k3l, L: 1.0}  # k3l = kn * L, realised at unit length
        assert out[PX] - state[PX] == pytest.approx(float(dpx_sym.subs(subs)), abs=1e-16)
        assert out[PY] - state[PY] == pytest.approx(float(dpy_sym.subs(subs)), abs=1e-16)


def test_mis_scaled_kick_fails_the_field_expansion(ref: ReferenceParticle) -> None:
    """The gate above has teeth: ``1`` in place of ``1/6`` misses by exactly 6.

    Every structural check below passes on this same map. This is the only place in
    *this* file that catches it — and the detuning suite catches it again, through a
    completely different route.
    """
    x, y, kn, L = sp.symbols("x y k_n L", real=True)
    dpx_sym, _ = _symbolic_multipole_kick(3)
    k3l, x0, y0 = 420.0, 1.1e-3, 3e-4
    state = np.array([x0, 0.0, y0, 0.0, 0.0, 0.0])

    expected = float(dpx_sym.subs({x: x0, y: y0, kn: k3l, L: 1.0}))
    wrong = _mis_scaled_kick(state, k3l)[PX] - state[PX]
    assert wrong == pytest.approx(6.0 * expected, rel=1e-12)
    assert abs(wrong - expected) > 4.0 * abs(expected)


# --------------------------------------------------------------------------
# 2. Structural properties — all of them blind to the coefficient, and labelled
# --------------------------------------------------------------------------


def test_kick_is_minus_gradient_of_a_potential(ref: ReferenceParticle) -> None:
    """``(Delta px, Delta py) = -grad V`` for ``V = k3l (x^4 - 6 x^2 y^2 + y^4)/24``.

    BLIND to the coefficient by construction — ``V`` is reverse-engineered from the
    kick, so this re-proves the algebra. It is here because that ``V`` is the object
    the detuning derivation averages over betatron phase, so it must be written down
    and checked once; and because being a gradient is *why* the map is symplectic.
    """
    x, y, k3l = sp.symbols("x y k3l", real=True)
    V = k3l * (x**4 - 6 * x**2 * y**2 + y**4) / 24
    assert sp.simplify(-sp.diff(V, x) - (-k3l * (x**3 - 3 * x * y**2) / 6)) == 0
    assert sp.simplify(-sp.diff(V, y) - (+k3l * (3 * x**2 * y - y**3) / 6)) == 0

    k3l_val, x0, y0 = 300.0, 2e-3, -1.1e-3
    state = np.array([x0, 0.0, y0, 0.0, 0.0, 0.0])
    out = ThinOctupole(k3l_val).track(state, ref)
    subs = {x: x0, y: y0, k3l: k3l_val}
    assert out[PX] - state[PX] == pytest.approx(float((-sp.diff(V, x)).subs(subs)), abs=1e-16)
    assert out[PY] - state[PY] == pytest.approx(float((-sp.diff(V, y)).subs(subs)), abs=1e-16)


def test_thin_kick_is_symplectic_at_amplitude(ref: ReferenceParticle) -> None:
    """Exactly symplectic everywhere — and BLIND: the mis-scaled kick passes too."""
    elem = ThinOctupole(500.0)
    for x0, y0 in [(0.0, 0.0), (2e-3, 1e-3), (-4e-3, 3e-3), (1e-2, -8e-3)]:
        state = np.array([x0, 1e-4, y0, -2e-4, 0.0, 1e-4])
        assert is_symplectic_map(lambda s: elem.track(s, ref), state, atol=1e-9)
        assert is_symplectic_map(lambda s: _mis_scaled_kick(s, 500.0), state, atol=1e-9)


def test_thick_octupole_is_symplectic_at_amplitude(ref: ReferenceParticle) -> None:
    """Drift-kick-drift is symplectic exactly, for any number of slices."""
    for n_slices in (1, 3, 8):
        elem = Octupole(0.4, 900.0, n_slices=n_slices)
        state = np.array([3e-3, 1e-4, -2e-3, 2e-4, 1e-3, 5e-4])
        assert is_symplectic_map(lambda s, e=elem: e.track(s, ref), state, atol=1e-9)


def test_kick_is_curl_free(ref: ReferenceParticle) -> None:
    """``d(Delta px)/dy == d(Delta py)/dx`` — Maxwell for a thin kick.

    BLIND to the overall scale (same statement as symplecticity here), but it does
    bite on a kick whose two components are scaled *differently*, which is no longer
    a magnetic field at all. Both facts are asserted.
    """
    k3l = 700.0
    x0, y0 = 1.7e-3, -9e-4
    # d(Delta px)/dy = +k3l x y ; d(Delta py)/dx = +k3l x y
    assert pytest.approx(k3l * x0 * y0, rel=1e-12) == k3l * x0 * y0

    def kick_only(state: np.ndarray, scale_py: float) -> np.ndarray:
        out = np.array(state, dtype=float, copy=True)
        x, y = out[X], out[Y]
        out[PX] -= k3l * (x**3 - 3 * x * y**2) / 6.0
        out[PY] += scale_py * k3l * (3 * x**2 * y - y**3) / 6.0
        return out

    state = np.array([x0, 0.0, y0, 0.0, 0.0, 0.0])
    J_ok = jacobian(lambda s: kick_only(s, 1.0), state)
    assert J_ok[PX, Y] == pytest.approx(J_ok[PY, X], abs=1e-6)
    J_bad = jacobian(lambda s: kick_only(s, 2.0), state)
    assert abs(J_bad[PX, Y] - J_bad[PY, X]) > 1e-3 * abs(J_bad[PY, X])
    # ...and the mis-scaled-but-consistent kick is curl-free, i.e. undetectable here.
    J_mis = jacobian(lambda s: _mis_scaled_kick(s, k3l), state)
    assert J_mis[PX, Y] == pytest.approx(J_mis[PY, X], abs=1e-4)


def test_jacobian_at_the_origin_is_the_identity(ref: ReferenceParticle) -> None:
    """A cubic kick has no linear part at the closed orbit — BLIND to the scale.

    The thin element's Jacobian is the identity; the thick one's is its drift map.
    What is left over is *not* a linear part: a central difference of ``x^3`` leaves
    ``k3l h^2/6``, so the residual is asserted to fall by 4 per halving of the step
    — the finite-difference truncation, vanishing in the limit, rather than physics.
    """
    thin = ThinOctupole(1e4)
    residuals = []
    for step in (4e-6, 2e-6, 1e-6):
        J = jacobian(lambda s: thin.track(s, ref), np.zeros(DIM), step=step)
        residuals.append(float(np.abs(J - np.eye(DIM)).max()))
    assert residuals[-1] < 1e-8
    ratios = [residuals[i] / residuals[i + 1] for i in range(len(residuals) - 1)]
    for r in ratios:
        assert r == pytest.approx(4.0, rel=0.05), f"ratios {ratios} — not O(step^2)"

    thick = Octupole(0.4, 1e4)
    J = jacobian(lambda s: thick.track(s, ref), np.zeros(DIM), step=1e-6)
    assert np.allclose(J, Drift(0.4).matrix(ref), atol=1e-8)


def test_linear_matrix_is_exactly_a_drift(ref: ReferenceParticle) -> None:
    """``matrix()`` is the drift map — including ``R56 = L/gamma0^2`` — at any ``k3``."""
    L = 0.37
    M_drift = Drift(L).matrix(ref)
    assert np.array_equal(Octupole(L, 0.0).matrix(ref), M_drift)
    assert np.array_equal(Octupole(L, 5e4).matrix(ref), M_drift)
    assert M_drift[ZETA, DELTA] == pytest.approx(L / ref.gamma0**2, rel=1e-14)
    assert np.array_equal(ThinOctupole(5e4).matrix(ref), np.eye(DIM))


def test_zero_strength_collapses_onto_the_linear_map(ref: ReferenceParticle) -> None:
    """At ``k3 = 0`` tracking is the drift map identically, for any ``n_slices``."""
    state = np.array([1e-3, 2e-4, -5e-4, 1e-4, 3e-3, 7e-4])
    M = Drift(0.6).matrix(ref)
    for n_slices in (1, 5):
        out = Octupole(0.6, 0.0, n_slices=n_slices).track(state, ref)
        assert np.allclose(out, M @ state, atol=0.0, rtol=0.0)
    assert np.array_equal(ThinOctupole(0.0).track(state, ref), state)


def test_linear_optics_are_untouched_by_the_octupole(ref: ReferenceParticle) -> None:
    """Beta and the linear tunes do not depend on ``k3l`` — BLIND to the scale.

    This is the physics (an octupole has no linear part), and it is exactly why
    amplitude detuning needs *tracking* to be seen at all: no matrix-based optics
    function in the package can respond to ``k3l``.
    """
    cell = [Quadrupole(0.3, 1.6), Drift(1.0), Quadrupole(0.3, -1.6), Drift(1.0)]
    plain = Lattice(cell * 4, ref)
    with_oct = Lattice([*cell, ThinOctupole(3e4), *(cell * 3)], ref)
    assert tunes(with_oct) == tunes(plain)  # bit for bit, not to tolerance
    t_plain, t_oct = closed_twiss(plain), closed_twiss(with_oct)
    for field in ("beta_x", "alpha_x", "beta_y", "alpha_y", "disp_x", "disp_px"):
        assert getattr(t_oct, field) == getattr(t_plain, field), field
    # ...and along the ring, not only at s = 0: the zero-length octupole inserts one
    # duplicate row, and every other row must be untouched.
    tab_plain = propagate_twiss(plain, t_plain)
    tab_oct = propagate_twiss(with_oct, t_oct)
    assert len(tab_oct) == len(tab_plain) + 1
    betas_plain = [(t.beta_x, t.beta_y) for t in tab_plain]
    betas_oct = [(t.beta_x, t.beta_y) for i, t in enumerate(tab_oct) if i != 5]
    assert betas_oct == betas_plain


# --------------------------------------------------------------------------
# 3. The thick element: integrator order, and the thin limit
# --------------------------------------------------------------------------


def test_thick_slicing_converges_second_order(ref: ReferenceParticle) -> None:
    """Drift-kick-drift error falls as ``1/n_slices^2`` — a second-order integrator.

    Measured against a heavily sliced reference rather than a closed form (there is
    none for a thick octupole): the ratio of successive errors is the claim.
    """
    L, k3 = 0.5, 4e4
    state = np.array([2e-3, 1e-4, -1.5e-3, 2e-4, 0.0, 0.0])
    exact = Octupole(L, k3, n_slices=2048).track(state, ref)
    errs = [
        float(np.linalg.norm(Octupole(L, k3, n_slices=n).track(state, ref) - exact))
        for n in (4, 8, 16, 32)
    ]
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    for r in ratios:
        assert r == pytest.approx(4.0, rel=0.05), f"ratios {ratios} — not O(1/n^2)"


def test_thick_error_is_third_order_in_length(ref: ReferenceParticle) -> None:
    """One-slice error vs. the exact-in-the-limit map falls as ``L^3`` at fixed ``k3``."""
    k3 = 4e4
    state = np.array([2e-3, 1e-4, -1.5e-3, 2e-4, 0.0, 0.0])
    errs = []
    for L in (0.4, 0.2, 0.1, 0.05):
        exact = Octupole(L, k3, n_slices=4096).track(state, ref)
        errs.append(float(np.linalg.norm(Octupole(L, k3).track(state, ref) - exact)))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    for r in ratios:
        assert r == pytest.approx(8.0, rel=0.05), f"ratios {ratios} — not O(L^3)"


def test_short_thick_octupole_approaches_the_thin_one(ref: ReferenceParticle) -> None:
    """At fixed ``k3l`` a shrinking body tends to ``ThinOctupole`` — but only linearly.

    The leading remainder at fixed integrated strength carries a factor ``L``, so
    halving the length halves the gap: a short thick octupole is not a thin one, the
    same caveat J1 recorded for the sextupole.
    """
    k3l = 2e3
    state = np.array([2e-3, 1e-4, -1.5e-3, 2e-4, 0.0, 0.0])
    thin = ThinOctupole(k3l).track(state, ref)
    gaps = []
    for L in (0.4, 0.2, 0.1):
        thick = Octupole(L, k3l / L).track(state, ref)
        # Strip the drift both sides so only the kick's placement is compared.
        undrift = np.linalg.inv(Drift(L).matrix(ref))
        gaps.append(float(np.linalg.norm(undrift @ thick - thin)))
    ratios = [gaps[i] / gaps[i + 1] for i in range(len(gaps) - 1)]
    for r in ratios:
        assert r == pytest.approx(2.0, rel=0.1), f"ratios {ratios} — not O(L)"


# --------------------------------------------------------------------------
# 4. Scope lines, enforced rather than documented
# --------------------------------------------------------------------------


def test_linearised_lattice_handles_a_thin_octupole_and_refuses_a_thick_one(
    ref: ReferenceParticle,
) -> None:
    """The scope line J2 drew, as **J3** left it. This is the boundary, not the physics.

    J2 refused every octupole here, because passing one through would report a drift —
    i.e. claim the beam on a distorted orbit sees no gradient from it. J3 derived the
    six-way split, so a *thin* octupole is now expanded rather than refused, and the
    line has moved to where the thick sextupole's already was: a thick body's offset
    varies across it, so a single entrance-orbit split would carry an ``O(L^2)`` error.

    The physics of the split is gated in ``tests/analytic/test_octupole_feeddown.py``;
    what is asserted here is only which inputs are accepted.
    """
    cell = [Quadrupole(0.3, 1.6), Drift(1.0), Quadrupole(0.3, -1.6), Drift(1.0)]

    thin = Lattice([*cell, ThinOctupole(3e4, name="oct"), *cell], ref)
    expanded = linearised_lattice(thin)
    # One octupole in, five elements out: quad + skew + sextupole + skew sextupole,
    # then the octupole itself (all zero-strength here — the ring is not steered).
    assert len(expanded.elements) == len(thin.elements) + 4

    thick = Lattice([*cell, Octupole(0.2, 1e5, name="oct"), *cell], ref)
    with pytest.raises(NotImplementedError, match="thick Octupole"):
        linearised_lattice(thick)

    # A zero-strength thick octupole is a drift and is allowed through unchanged.
    lat0 = Lattice([*cell, Octupole(0.2, 0.0), *cell], ref)
    assert len(linearised_lattice(lat0).elements) == len(lat0.elements)


def test_linearised_element_maps_does_see_the_octupole(ref: ReferenceParticle) -> None:
    """The *numerical* linearisation handles it, because it differentiates ``track()``.

    At an orbit offset the cubic kick's Jacobian is a gradient
    ``k1l_eff = -d(Delta px)/dx = k3l (x^2 - y^2)/2``. This is not a scope claim about
    feed-down optics — it is the statement that the two linearising helpers differ,
    and which of them is safe to use here.
    """
    k3l, x_off = 3e4, 2e-3
    elem = ThinOctupole(k3l)
    lat = Lattice([elem], ref)
    maps = linearised_element_maps(lat, np.array([x_off, 0.0, 0.0, 0.0]))
    assert maps[0][PX, X] == pytest.approx(-k3l * x_off**2 / 2.0, rel=1e-5)
    assert maps[0][PY, Y] == pytest.approx(+k3l * x_off**2 / 2.0, rel=1e-5)


def test_the_whole_on_orbit_family_now_works_around_a_live_thin_octupole(
    ref: ReferenceParticle,
) -> None:
    """J2 split this family in half; J3 closed the half that refused.

    The two routes to on-orbit optics are still different — :func:`tunes_on_orbit`
    and its siblings differentiate the real ``track()``, while
    :func:`~accsim.twiss.chromaticity_on_orbit` walks element **types** through
    :func:`~accsim.orbit.linearised_lattice` — but with the split derived, both now
    answer for a thin octupole and *agree*, which is J3's own gate. What remains here
    is the J2 statement they share: an octupole on a distorted orbit changes the
    optics, and the Newton path converges on a cubic kick.

    The refusal that used to be asserted here now applies only to a **thick**
    octupole, and is checked in the test above.
    """
    cell = [Quadrupole(0.3, 1.6), Drift(1.0), Quadrupole(0.3, -1.6), Drift(1.0)]
    steered = Lattice(
        [Corrector(kick_x=2e-4), *cell, ThinOctupole(3e4, name="oct"), *(cell * 3)], ref
    )
    design = Lattice([*cell, ThinOctupole(3e4), *(cell * 3)], ref)

    # The orbit really is displaced, and the Newton solve converges with the cubic kick.
    orbit = closed_orbit_nonlinear(steered)
    assert abs(orbit[X]) > 1e-5

    q_steered = tunes_on_orbit(steered)
    q_design = tunes_on_orbit(design)
    assert all(math.isfinite(q) for q in q_steered)
    assert abs(q_steered[0] - q_design[0]) > 1e-9  # the octupole really has fed down

    # No longer raises: the split is derived, so the type-walking route answers too.
    for value in (*chromaticity_on_orbit(steered), *natural_chromaticity_on_orbit(steered)):
        assert math.isfinite(value)
    # ...and it is a *different* number from the design-orbit one, which is the point.
    assert chromaticity_on_orbit(steered) != chromaticity_on_orbit(design)


def test_nonlinear_bunch_tracking_applies_the_kick(ref: ReferenceParticle) -> None:
    """The kick works on a ``(6, n)`` bunch, not only on a single state.

    ``_apply_kick`` is written to broadcast; this is the assertion that makes the
    claim honest, through the loss-aware bunch path that J1 had to fix for the
    sextupole. Each particle must receive its *own* amplitude's kick.
    """
    lat = Lattice([Drift(0.5), ThinOctupole(5e4), Drift(0.5)], ref)
    xs = np.array([1e-3, 2e-3, 3e-3])
    states = np.zeros((DIM, xs.size))
    states[X] = xs
    result = Tracker(lat).track_bunch_losses(Bunch(states), nonlinear=True)
    assert result.transmission == 1.0
    for i, x0 in enumerate(xs):
        expected = -5e4 * x0**3 / 6.0
        assert result.states[PX, i] == pytest.approx(expected, rel=1e-12)


def test_linear_tracking_silently_drops_the_kick(ref: ReferenceParticle) -> None:
    """``Tracker.track(nonlinear=False)`` linearises the octupole away — asserted, not discovered.

    The same documented hazard J1 recorded for the sextupole: the default tracking
    path uses ``matrix()``, and an octupole's matrix is a drift.
    """
    lat = Lattice([Drift(0.5), ThinOctupole(5e4), Drift(0.5)], ref)
    p = Particle(x=3e-3, y=1e-3)
    linear = Tracker(lat).track(p, nonlinear=False)
    nonlinear = Tracker(lat).track(p, nonlinear=True)
    assert linear.px == 0.0  # exactly: the matrix path never touches px
    assert abs(nonlinear.px) > 1e-7


def test_chromaticity_is_unaffected_at_first_order(ref: ReferenceParticle) -> None:
    """An octupole at dispersion does not shift ``Q'`` — so ``chromaticity()`` ignoring it is right.

    Expanding the cubic kick in ``x = x_beta + D_x delta`` gives a *sextupole* term
    linear in ``delta`` and a *gradient* term of order ``delta^2``. There is no
    ``delta * x_beta`` gradient, so first-order chromaticity is untouched and the
    blind spot is ``Q''`` — the honest scope line, derived rather than asserted.
    """
    x_beta, y_beta, D, delta, k3l = sp.symbols("x_beta y_beta D delta k3l", real=True)
    dpx = -k3l * ((x_beta + D * delta) ** 3 - 3 * (x_beta + D * delta) * y_beta**2) / 6
    # The focusing gradient an off-momentum particle sees, as a series in delta.
    grad = sp.series(-sp.diff(dpx, x_beta).subs({x_beta: 0, y_beta: 0}), delta, 0, 3).removeO()
    assert sp.simplify(grad.coeff(delta, 0)) == 0  # no on-momentum gradient
    assert sp.simplify(grad.coeff(delta, 1)) == 0  # ...and none linear in delta: no Q'
    assert sp.simplify(grad.coeff(delta, 2) - k3l * D**2 / 2) == 0  # Q'' lives here
