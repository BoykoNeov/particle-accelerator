r"""J3 (part 1) — the skew sextupole, the one feed-down term with no element yet.

J3 expands the octupole's cubic kick about an orbit offset ``(x_co, y_co)``. Five of
the six terms that come out are elements accsim already validates; the sixth is a
**skew sextupole**, and this file is the element that has to exist before the
expansion can be written down without dropping a term:

    Delta px = +k2sl (x y),      Delta py = +1/2 k2sl (x^2 - y^2).

**What can and cannot pin this element here.** Nothing accsim computes reads a skew
sextupole: :func:`~accsim.twiss.chromaticity` takes ``k2l`` off *normal* sextupoles
and only at ``D_x``, and :func:`~accsim.twiss.amplitude_detuning` walks octupoles.
So there is no closed-form quantity to check the sign against, and every gate in this
file is either **structural** (symplecticity, the curl-free condition, the identity
:meth:`matrix`) — blind to the coefficient, exactly as J1 and J2 kept re-establishing
— or **shape** (the roll identity below), which is blind to the overall sign. The
sign is fixed by probe against xtrack in ``tests/reference/test_skew_sextupole_xtrack.py``
and nowhere else, which is the J1/J2 rule for a convention accsim also derives.

Two things here do carry weight:

1. **The coefficient descends from a series whose neighbours are validated.** The
   ``k_n + i k_ns`` expansion is written once and evaluated at three places: its
   ``n = 1`` *skew* term must be :class:`ThinSkewQuadrupole` (pinned against xtrack in
   G1) and its ``n = 2`` *normal* term must be :class:`ThinSextupole` (pinned against
   xtrack in J1). The skew sextupole sits at the intersection of two independently
   validated directions rather than being asserted about itself.
2. **The roll angle is derived, not recalled.** A skew sextupole is a normal one
   rolled about the beam axis, and the angle is *solved for* symbolically rather than
   remembered as "30 degrees" — including which sign of roll, since ``+pi/6`` gives
   exactly the opposite kick, and including the fact that the angle is not unique
   (``-pi/6 + 2pi/3`` is the same magnet).
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from accsim import (
    DIM,
    PX,
    PY,
    ReferenceParticle,
    ThinSextupole,
    ThinSkewQuadrupole,
    ThinSkewSextupole,
    X,
    Y,
    is_symplectic_map,
    jacobian,
)

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0

K2SL = 7.0  # integrated skew strength [m^-2]
# A generic probe state: every coordinate nonzero, so no term can hide.
STATE = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 1.0e-3, 2.0e-4])


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _mis_scaled_kick(state: np.ndarray, k2sl: float) -> np.ndarray:
    """The skew kick with ``1`` in place of ``1/2`` in ``Delta py`` — and still a field.

    Scaled consistently across *both* components (``2`` for ``1`` in ``Delta px``), so
    it stays curl-free and therefore exactly symplectic. This is the bug none of the
    structural gates in this file can see.
    """
    out = np.array(state, dtype=float, copy=True)
    x, y = out[X], out[Y]
    out[PX] += 2.0 * k2sl * x * y
    out[PY] += k2sl * (x * x - y * y)
    return out


# --------------------------------------------------------------------------
# 1. The coefficient, from the field expansion (anchored on two validated terms)
# --------------------------------------------------------------------------


def _symbolic_multipole_kick(order: int, skew: bool) -> tuple[sp.Expr, sp.Expr]:
    r"""Thin-lens ``(Delta px, Delta py)`` of the pure ``n = order`` normal/skew multipole.

    From the MAD-X / Xsuite field expansion with both families present,

        B_y + i B_x = (B rho) * sum_n (k_n + i k_ns) (x + i y)^n / n!,

    and the thin-lens kick of a transverse field over length ``L``,

        Delta px = -(1/B rho) B_y L,      Delta py = +(1/B rho) B_x L.

    Written once and evaluated at three ``(order, skew)`` combinations below; nothing
    here knows what any element's code does. The **only** difference between the
    normal and skew families is the ``i`` multiplying the strength — that is what
    "skew" means in this convention, and it is why a roll by a fixed angle turns one
    into the other.
    """
    x, y, kn, L = sp.symbols("x y k_n L", real=True)
    z = x + sp.I * y
    strength = sp.I * kn if skew else kn
    field_over_brho = sp.expand(strength * z**order / sp.factorial(order))
    By = sp.re(field_over_brho)
    Bx = sp.im(field_over_brho)
    return sp.simplify(-By * L), sp.simplify(Bx * L)


def test_expansion_anchors_on_the_skew_quad_and_the_normal_sextupole(
    ref: ReferenceParticle,
) -> None:
    r"""The two neighbours of this term in the series are already xtrack-validated.

    ``n = 1`` skew is :class:`ThinSkewQuadrupole` (``px += k1sl y``, ``py += k1sl x``),
    whose sign G1 pinned against xtrack; ``n = 2`` normal is :class:`ThinSextupole`
    (``px -= 1/2 k2l (x^2 - y^2)``, ``py += k2l x y``), whose sign J1 pinned against
    xtrack bit-for-bit. Both are reproduced *by the same symbolic expression* that
    will then be evaluated at ``n = 2`` skew, so the ``1/2`` and the relative sign of
    the two skew components inherit two independent verdicts instead of standing on
    their own algebra.
    """
    x, y, kn, L = sp.symbols("x y k_n L", real=True)

    # n = 1, skew: the coupling kick G1 validated.
    dpx, dpy = _symbolic_multipole_kick(1, skew=True)
    assert sp.simplify(dpx - (+kn * L * y)) == 0
    assert sp.simplify(dpy - (+kn * L * x)) == 0
    k1sl = 1.7
    M = ThinSkewQuadrupole(k1sl).matrix(ref)
    assert M[PX, Y] == pytest.approx(+k1sl, rel=1e-14)
    assert M[PY, X] == pytest.approx(+k1sl, rel=1e-14)

    # n = 2, normal: the sextupole kick J1 validated.
    dpx, dpy = _symbolic_multipole_kick(2, skew=False)
    assert sp.simplify(dpx - (-kn * L * (x**2 - y**2) / 2)) == 0
    assert sp.simplify(dpy - (+kn * L * x * y)) == 0
    k2l = 3.1
    got = ThinSextupole(k2l).track(STATE, ref) - STATE
    assert got[PX] == pytest.approx(-0.5 * k2l * (STATE[X] ** 2 - STATE[Y] ** 2), rel=1e-14)
    assert got[PY] == pytest.approx(+k2l * STATE[X] * STATE[Y], rel=1e-14)


def test_thin_skew_sextupole_is_the_n2_skew_term_of_that_expansion(
    ref: ReferenceParticle,
) -> None:
    r"""``n = 2`` skew: ``Delta px = +k2sl x y``, ``Delta py = +1/2 k2sl (x^2 - y^2)``.

    The element evaluated against the symbolic series, at the generic probe state so
    that no term can vanish accidentally. This fixes the ``1/2`` and the *relative*
    sign of the two components. It does **not** fix the overall sign — flipping
    ``k2sl`` flips both the series and the element together — which is why the xtrack
    probe exists.
    """
    x_s, y_s, kn, L = sp.symbols("x y k_n L", real=True)
    dpx, dpy = _symbolic_multipole_kick(2, skew=True)
    assert sp.simplify(dpx - (+kn * L * x_s * y_s)) == 0
    assert sp.simplify(dpy - (+kn * L * (x_s**2 - y_s**2) / 2)) == 0

    # k_n L is the integrated strength k2sl: the series is per unit length, the
    # element carries the product, and there is no length left to disagree about.
    at = {x_s: STATE[X], y_s: STATE[Y], kn: K2SL, L: 1.0}
    got = ThinSkewSextupole(K2SL).track(STATE, ref) - STATE
    assert got[PX] == pytest.approx(float(dpx.subs(at)), rel=1e-14)
    assert got[PY] == pytest.approx(float(dpy.subs(at)), rel=1e-14)

    # Only the momenta move, and both really do — non-vacuous.
    assert np.array_equal(got[[X, Y]], np.zeros(2))
    assert abs(got[PX]) > 1e-9 and abs(got[PY]) > 1e-9

    # The mis-scaled variant differs, which is what makes the comparison above a gate.
    bad = _mis_scaled_kick(STATE, K2SL) - STATE
    assert bad[PX] / got[PX] == pytest.approx(2.0, rel=1e-12)
    assert bad[PY] / got[PY] == pytest.approx(2.0, rel=1e-12)


# --------------------------------------------------------------------------
# 2. The roll identity — shape, derived rather than recalled
# --------------------------------------------------------------------------


def test_the_roll_angle_is_solved_for_not_remembered() -> None:
    r"""A skew sextupole is a normal one **rolled by -30 degrees**, and that is derived.

    Rolling a magnet by ``phi`` means: rotate the particle into the magnet frame,
    apply the normal kick there, rotate the resulting kick back. With
    ``R(phi) = [[cos, sin], [-sin, cos]]`` that is ``R(phi)^T k_normal(R(phi) r)``.
    The angle is obtained by **solving** for it, not by asserting a remembered 30
    degrees, and three separate facts come out:

    - ``phi = -pi/6`` reproduces the skew kick identically, at every ``(x, y)``;
    - ``phi = +pi/6`` gives exactly *minus* it — so the roll pins the shape of the
      element and says nothing about its overall sign (the reason for the xtrack
      probe);
    - the angle is **not unique**: ``-pi/6 + 2pi/3`` is the same magnet, because a
      sextupole field is unchanged by a third of a turn. Asserted so that "the" roll
      angle is not read as a claim it cannot support.
    """
    x, y, k, phi = sp.symbols("x y k phi", real=True)
    R = sp.Matrix([[sp.cos(phi), sp.sin(phi)], [-sp.sin(phi), sp.cos(phi)]])
    xr, yr = R * sp.Matrix([x, y])
    normal = sp.Matrix([-sp.Rational(1, 2) * k * (xr**2 - yr**2), k * xr * yr])
    rolled = sp.simplify(R.T * normal)
    skew = sp.Matrix([k * x * y, sp.Rational(1, 2) * k * (x**2 - y**2)])

    # Solve rather than assert: which rolls turn the normal kick into the skew one?
    # Every coefficient of the residual, in x and y, has to vanish at once.
    residual = sp.expand(sp.expand_trig(sp.expand((rolled - skew).subs(k, 1))))
    conditions = [c for comp in residual for c in sp.Poly(comp, x, y).coeffs()]
    solved = sp.solve(conditions, phi, dict=True)
    assert solved, "the roll identity has to have a solution at all"
    # Whatever branch sympy hands back is congruent to -pi/6 modulo **2pi/3**: the
    # sextupole's three-fold field symmetry, which is where both the 30 degrees and
    # the non-uniqueness come from.
    for s in solved:
        assert sp.simplify(sp.Mod(s[phi] + sp.pi / 6, 2 * sp.pi / 3)) == 0

    assert sp.simplify(rolled.subs(phi, -sp.pi / 6) - skew) == sp.zeros(2, 1)
    assert sp.simplify(rolled.subs(phi, +sp.pi / 6) + skew) == sp.zeros(2, 1)
    assert sp.simplify(rolled.subs(phi, -sp.pi / 6 + 2 * sp.pi / 3) - skew) == sp.zeros(2, 1)


def test_the_roll_identity_holds_for_the_elements_themselves(ref: ReferenceParticle) -> None:
    """The same identity, between the two *elements*, numerically.

    :class:`ThinSkewSextupole` is not implemented as a rotated
    :class:`ThinSextupole`, so this is two independent pieces of code agreeing, not a
    tautology.
    """
    phi = -np.pi / 6.0
    c, s = np.cos(phi), np.sin(phi)

    into = STATE.copy()
    into[X] = c * STATE[X] + s * STATE[Y]
    into[Y] = -s * STATE[X] + c * STATE[Y]
    kick = ThinSextupole(K2SL).track(into, ref) - into

    got = ThinSkewSextupole(K2SL).track(STATE, ref) - STATE
    assert got[PX] == pytest.approx(c * kick[PX] - s * kick[PY], rel=1e-12)
    assert got[PY] == pytest.approx(s * kick[PX] + c * kick[PY], rel=1e-12)


# --------------------------------------------------------------------------
# 3. Structural properties — all true of the mis-scaled element too
# --------------------------------------------------------------------------


def test_the_kick_is_a_gradient_of_a_potential_and_hence_curl_free() -> None:
    r"""``(Delta px, Delta py) = -grad V`` for ``V = -k2sl (3 x^2 y - y^3) / 6``.

    The Maxwell/curl-free condition ``d(Delta px)/dy = d(Delta py)/dx`` for a thin
    kick is the *same* statement, not an independent one — J1's lesson, repeated here
    because it is the check most likely to be mistaken for a coefficient gate. The
    mis-scaled kick satisfies both, and that is asserted rather than assumed.
    """
    x, y, k = sp.symbols("x y k", real=True)
    V = -k * (3 * x**2 * y - y**3) / 6
    assert sp.simplify(-sp.diff(V, x) - k * x * y) == 0
    assert sp.simplify(-sp.diff(V, y) - k * (x**2 - y**2) / 2) == 0

    dpx, dpy = k * x * y, k * (x**2 - y**2) / 2
    assert sp.simplify(sp.diff(dpx, y) - sp.diff(dpy, x)) == 0

    # ...and the blind spot, made explicit: twice the kick is still curl-free.
    assert sp.simplify(sp.diff(2 * dpx, y) - sp.diff(2 * dpy, x)) == 0


def test_the_map_is_exactly_symplectic_at_every_amplitude(ref: ReferenceParticle) -> None:
    """Symplectic at 1 mm and at 5 cm alike — and so is the mis-scaled element."""
    sx = ThinSkewSextupole(K2SL)
    for amp in (1e-3, 5e-2):
        state = np.array([amp, 1e-4, -0.6 * amp, 5e-5, 1e-3, 2e-4])
        assert is_symplectic_map(lambda s: sx.track(s, ref), state, step=1e-7, atol=1e-9)
        assert is_symplectic_map(lambda s: _mis_scaled_kick(s, K2SL), state, step=1e-7, atol=1e-9)


def test_the_linear_map_is_the_identity_and_the_optics_do_not_move(
    ref: ReferenceParticle,
) -> None:
    """A thin quadratic kick has zero linear part **at the origin**, exactly.

    So beta, dispersion, the tunes and the coupling of the *linear* lattice do not
    depend on ``k2sl`` at all — a skew sextupole is invisible to matrix optics, which
    is precisely why it needed an xtrack probe to pin its sign.
    """
    sx = ThinSkewSextupole(K2SL)
    assert np.array_equal(sx.matrix(ref), np.eye(DIM))
    assert sx.length == 0.0

    at_origin = np.zeros(DIM)
    assert np.array_equal(sx.track(at_origin, ref), at_origin)
    assert np.allclose(jacobian(lambda s: sx.track(s, ref), at_origin, step=1e-7), np.eye(DIM))

    # Away from the origin it is *not* the identity — that difference is J3's subject.
    off = np.array([2e-3, 0.0, 1e-3, 0.0, 0.0, 0.0])
    J = jacobian(lambda s: sx.track(s, ref), off, step=1e-7)
    assert not np.allclose(J, np.eye(DIM), atol=1e-9)


def test_zero_strength_is_the_identity_and_the_kick_is_odd_in_k2sl(
    ref: ReferenceParticle,
) -> None:
    """``k2sl = 0`` changes nothing at all, and reversing it reverses the kick exactly."""
    assert np.array_equal(ThinSkewSextupole(0.0).track(STATE, ref), STATE)
    plus = ThinSkewSextupole(K2SL).track(STATE, ref) - STATE
    minus = ThinSkewSextupole(-K2SL).track(STATE, ref) - STATE
    assert np.array_equal(plus, -minus)


def test_it_tracks_a_bunch_and_leaves_the_caller_s_array_alone(
    ref: ReferenceParticle,
) -> None:
    """Vectorised over a ``(6, n)`` bunch, column by column, without mutating input."""
    sx = ThinSkewSextupole(K2SL)
    bunch = np.column_stack([STATE, 2.0 * STATE, -STATE])
    original = bunch.copy()

    out = sx.track(bunch, ref)
    assert out.shape == bunch.shape
    assert np.array_equal(bunch, original)  # the caller's array is untouched
    for j in range(bunch.shape[1]):
        assert np.allclose(out[:, j], sx.track(bunch[:, j], ref), atol=0.0, rtol=0.0)


def test_repr_round_trips_the_strength_and_the_name() -> None:
    assert repr(ThinSkewSextupole(K2SL)) == f"ThinSkewSextupole(k2sl={K2SL})"
    assert repr(ThinSkewSextupole(K2SL, name="sxs")) == (
        f"ThinSkewSextupole(k2sl={K2SL}, name='sxs')"
    )


# --------------------------------------------------------------------------
# 4. What this element is invisible to — the blind spots, asserted
# --------------------------------------------------------------------------


def test_no_optics_function_in_the_package_reads_it(ref: ReferenceParticle) -> None:
    r"""The honest scope statement: accsim computes nothing that a skew sextupole moves.

    :func:`~accsim.twiss.chromaticity` sums ``k2l`` over *normal* sextupoles at
    ``D_x``; :func:`~accsim.twiss.amplitude_detuning` walks octupoles. A skew
    sextupole at dispersion feeds down a ``delta``-dependent **skew quadrupole**
    ``k1sl = k2sl D_x delta``, i.e. chromatic *coupling* — real physics, and a
    quantity this package does not model at all. Both non-responses are pinned here
    so the blind spot is documented rather than discovered later.
    """
    from accsim import Drift, Lattice, Quadrupole
    from accsim.twiss import amplitude_detuning, chromaticity

    base = [
        Quadrupole(0.5, 1.2, name="qf"),
        Drift(1.0),
        Quadrupole(0.5, -1.2, name="qd"),
        Drift(1.0),
    ]
    plain = Lattice(list(base), ref)
    with_skew = Lattice([ThinSkewSextupole(K2SL, name="sxs"), *base], ref)

    assert chromaticity(with_skew) == chromaticity(plain)
    assert np.array_equal(amplitude_detuning(with_skew), amplitude_detuning(plain))

    # The feed-down it *does* produce, measured off the tracked map: at a horizontal
    # offset the Jacobian is a pure skew quadrupole of the derived strength.
    x0 = 1.3e-3
    at = np.array([x0, 0.0, 0.0, 0.0, 0.0, 0.0])
    J = jacobian(lambda s: ThinSkewSextupole(K2SL).track(s, ref), at, step=1e-7)
    assert J[PX, Y] == pytest.approx(K2SL * x0, rel=1e-6)
    assert J[PY, X] == pytest.approx(K2SL * x0, rel=1e-6)
    assert J[PX, X] == pytest.approx(0.0, abs=1e-9)  # no *normal* gradient from x alone
    assert J[PY, Y] == pytest.approx(0.0, abs=1e-9)
