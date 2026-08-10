r"""I2 acceptance: sextupole feed-down on a distorted closed orbit.

I1 closed the orbit with a *linear* solve, ``(I - M4) x_co = k4``, and qualified
"correctors do not move the optics" as a linear-order, on-axis-sextupole claim.
J1 then gave the sextupole a real kick. This file is what those two together
imply: expanding ``Delta px = -1/2 k2l (x^2 - y^2)``, ``Delta py = +k2l x y``
about an orbit offset ``(x_co, y_co)`` splits one sextupole into four elements,

    dipole       theta_x  = -1/2 k2l (x_co^2 - y_co^2),  theta_y = +k2l x_co y_co
    normal quad  k1l_eff  = +k2l x_co
    skew quad    k1sl_eff = +k2l y_co
    sextupole    unchanged

every coefficient of which is *derived* in :func:`test_feeddown_expansion_is_derived`
rather than recalled.

**What is deliberately not tested here.** J1 already measured ``k1l_eff`` by
finite-differencing ``track()`` about a dispersive offset. Re-measuring it about a
corrector-induced offset is the same measurement, so the gradient term is used
here only as an ingredient, never as the headline gate. Three things have no J1
analogue and carry the file:

1. **The dipole term moves the orbit that creates it.** ``theta_x`` depends on
   ``x_co``, so the closed orbit is the fixed point of a *nonlinear* map. The lead
   gate (:func:`test_departure_is_the_linear_response_to_the_derived_dipole`) is
   that the departure from the linear orbit equals I1's linear response to that
   derived kick, with a residual one order higher — a convergence-order check, not
   a magnitude comparison, so a wrong coefficient breaks the *order*.
2. **A vertical orbit makes a normal sextupole a skew quadrupole**, i.e. a source
   of betatron coupling. J1 only ever saw the horizontal, dispersive plane; this
   reaches G1/G2's coupling machinery from a completely new direction. A pure
   *vertical* steerer even moves the *horizontal* orbit, through
   ``theta_x = +1/2 k2l y_co^2`` — impossible in the linear theory at any kick.
3. **Orbit correction stops being one solve.** With a live off-axis sextupole the
   response matrix is affine only to first order, so one application leaves an
   ``O(k2l x_co^2)`` residual and the correction has to iterate. That is the
   operational content of feed-down, and it is why real machines run a loop.

The lattice is a **palindromic** thin FODO ring, so ``alpha = 0`` at the entrance;
the single sextupole sits there. That is not decoration — it is what makes the
perturbed beta function at the sextupole an exactly-solvable closed form
(:func:`test_beta_at_the_source_matches_the_exact_gradient_error_form`) instead of
a recalled beta-beat formula.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    ClosedOrbitError,
    Corrector,
    Drift,
    Lattice,
    OrbitConvergenceError,
    ReferenceParticle,
    Sextupole,
    ThinQuadrupole,
    ThinSextupole,
    ThinSkewQuadrupole,
    chromaticity,
    closed_orbit,
    closed_orbit_nonlinear,
    closed_twiss,
    correct_orbit,
    linearised_element_maps,
    linearised_one_turn_map,
    match_periodic,
    match_periodic_coupled,
    propagate_orbit,
    propagate_orbit_nonlinear,
    propagate_twiss,
)
from accsim.coords import DIM, PX, PY, X, Y

# Thin FODO cell, as in the I1 suite. The cell is a *palindrome*
# (halfF | drift | D | drift | halfF), so alpha = 0 at every cell boundary — the
# symmetry the exact beta closed form below relies on.
VF = 1.0 / 1.5  # full-quad inverse focal length, F family [m^-1]
VD = 1.0 / 1.6  # ditto, D family [m^-1]
L_HALF = 1.0  # half-cell drift [m]

N_CELLS = 6
K2L = 20.0  # integrated sextupole strength [m^-2]
KICK = 2e-4  # steerer angle [rad] -> a sub-mm orbit


@pytest.fixture
def ref() -> ReferenceParticle:
    # Thin quads + drifts are energy-independent; any reference works.
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _cell(tag: str = "") -> list:
    return [
        ThinQuadrupole(0.5 * VF, name=f"qf_a{tag}"),
        Drift(L_HALF, name=f"d1{tag}"),
        ThinQuadrupole(-VD, name=f"qd{tag}"),
        Drift(L_HALF, name=f"d2{tag}"),
        ThinQuadrupole(0.5 * VF, name=f"qf_b{tag}"),
    ]


def _ring(n_cells: int = N_CELLS) -> list:
    return [e for i in range(n_cells) for e in _cell(tag=f"_{i}")]


def _lattice(
    ref: ReferenceParticle,
    k2l: float = 0.0,
    kick_x: float = 0.0,
    kick_y: float = 0.0,
    n_cells: int = N_CELLS,
) -> Lattice:
    """Steerer + one sextupole at the (``alpha = 0``) entrance, then the ring.

    The steerer is thin and touches only ``px``/``py``, so the *position* the
    sextupole sees is the closed orbit reported at the lattice entrance.
    """
    return Lattice(
        [
            Corrector(kick_x=kick_x, kick_y=kick_y, name="steerer"),
            ThinSextupole(k2l, name="sx"),
            *_ring(n_cells),
        ],
        ref,
    )


def _fractional_tunes(M: np.ndarray) -> tuple[float, float]:
    """Fractional tunes from the 2x2 diagonal blocks of an uncoupled 6x6 map."""
    out = []
    for a, b in ((X, PX), (Y, PY)):
        half_trace = 0.5 * (M[a, a] + M[b, b])
        assert abs(half_trace) < 1.0, f"unstable linearised map: |tr/2| = {abs(half_trace)}"
        q = math.acos(half_trace) / (2.0 * math.pi)
        out.append(q if M[a, b] >= 0.0 else 1.0 - q)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# The expansion itself — derived, not recalled
# ---------------------------------------------------------------------------


def test_feeddown_expansion_is_derived(ref: ReferenceParticle) -> None:
    r"""Every feed-down coefficient, from the kick, in sympy.

    The four terms are read straight off the Taylor expansion of the J1 kick about
    ``(x_co, y_co)`` and matched to the *existing, separately validated* elements
    they equal: :class:`Corrector`, :class:`ThinQuadrupole` and
    :class:`ThinSkewQuadrupole`. Nothing here is a remembered formula, and in
    particular the ``theta / k1l_eff = -x_co / 2`` ratio at the end is pure
    geometry — independent of ``k2l`` — which is a discriminator J1 structurally
    could not have had, since J1 only ever saw the gradient term.
    """
    sp = pytest.importorskip("sympy")
    xb, yb, k2l, xco, yco = sp.symbols("X Y k2l x_co y_co", real=True)

    dpx = sp.expand(-sp.Rational(1, 2) * k2l * ((xco + xb) ** 2 - (yco + yb) ** 2))
    dpy = sp.expand(k2l * (xco + xb) * (yco + yb))

    def order(expr, n):
        return sum(t for t in expr.as_ordered_terms() if sp.Poly(t, xb, yb).total_degree() == n)

    # Constant term: a Corrector, whose kick() adds directly to (px, py).
    assert sp.simplify(order(dpx, 0) + sp.Rational(1, 2) * k2l * (xco**2 - yco**2)) == 0
    assert sp.simplify(order(dpy, 0) - k2l * xco * yco) == 0

    # Linear term: ThinQuadrupole is  px -= k1l x, py += k1l y;
    #              ThinSkewQuadrupole is  px += k1sl y, py += k1sl x.
    k1l_eff, k1sl_eff = k2l * xco, k2l * yco
    assert sp.simplify(order(dpx, 1) - (-k1l_eff * xb + k1sl_eff * yb)) == 0
    assert sp.simplify(order(dpy, 1) - (+k1l_eff * yb + k1sl_eff * xb)) == 0

    # Quadratic term: the sextupole is unchanged by the shift.
    assert sp.simplify(order(dpx, 2) + sp.Rational(1, 2) * k2l * (xb**2 - yb**2)) == 0
    assert sp.simplify(order(dpy, 2) - k2l * xb * yb) == 0

    # The dipole/gradient ratio is pure geometry: no k2l survives.
    ratio = sp.simplify(order(dpx, 0).subs(yco, 0) / k1l_eff)
    assert sp.simplify(ratio + xco / 2) == 0

    # ...and the whole kick stays curl-free, i.e. it is still a real magnet.
    assert sp.simplify(sp.diff(dpx, yb) - sp.diff(dpy, xb)) == 0


def test_linearised_sextupole_is_a_thin_normal_plus_skew_quadrupole(
    ref: ReferenceParticle,
) -> None:
    """Element level: the Jacobian at an offset **is** those two thin quads' product.

    This is the derivation above made numerical against elements the package
    validates independently (both against xtrack), so the feed-down decomposition
    borrows their credibility rather than re-deriving its own.
    """
    from accsim.symplectic import jacobian

    k2l, x_co, y_co = 7.0, 1.3e-3, -0.9e-3
    sx = ThinSextupole(k2l)
    state = np.array([x_co, 0.0, y_co, 0.0, 0.0, 0.0])

    got = jacobian(lambda s: sx.track(s, ref), state, step=1e-7)
    want = ThinQuadrupole(k2l * x_co).matrix(ref) @ ThinSkewQuadrupole(k2l * y_co).matrix(ref)

    assert np.allclose(got, want, atol=1e-9, rtol=0.0)
    # The two thin kicks commute exactly (their nilpotent parts annihilate), so the
    # order this product is written in is not a hidden assumption.
    other = ThinSkewQuadrupole(k2l * y_co).matrix(ref) @ ThinQuadrupole(k2l * x_co).matrix(ref)
    assert np.array_equal(want, other)


# ---------------------------------------------------------------------------
# The fixed point: it exists, it is a fixed point, and it degenerates correctly
# ---------------------------------------------------------------------------


def test_without_a_sextupole_the_nonlinear_orbit_is_the_linear_one(
    ref: ReferenceParticle,
) -> None:
    """Newton reproduces I1's solve to round-off when nothing is nonlinear.

    The free consistency gate that comes from seeding Newton at
    :func:`closed_orbit`: if the residual is already zero the iteration returns
    immediately, so this asserts the *tracked* map and the *affine* map agree —
    two independent code paths for the same lattice.
    """
    lat = _lattice(ref, k2l=0.0, kick_x=KICK, kick_y=-0.7 * KICK)
    assert np.allclose(closed_orbit_nonlinear(lat), closed_orbit(lat), atol=1e-16, rtol=0.0)

    perfect = _lattice(ref, k2l=K2L)  # sextupole present, but no kick -> on axis
    assert np.array_equal(closed_orbit_nonlinear(perfect), np.zeros(4))


def test_the_nonlinear_orbit_is_a_fixed_point_of_the_tracked_map(
    ref: ReferenceParticle,
) -> None:
    """Track it once, element by element, and it comes back to itself."""
    lat = _lattice(ref, k2l=K2L, kick_x=KICK)
    co = closed_orbit_nonlinear(lat)

    state = np.zeros(DIM)
    state[:4] = co
    for elem in lat.elements:
        state = elem.track(state, lat.ref)
    assert np.allclose(state[:4], co, atol=1e-14, rtol=0.0)

    # ...and the propagated table closes, which is what "closed" means.
    table = propagate_orbit_nonlinear(lat)
    assert np.allclose(table[0], table[-1], atol=1e-14, rtol=0.0)
    assert np.allclose(table[0], co, atol=0.0, rtol=0.0)


def test_newton_finds_the_same_orbit_from_a_deliberately_wrong_guess(
    ref: ReferenceParticle,
) -> None:
    """The default seed is a convenience, not part of the answer."""
    lat = _lattice(ref, k2l=K2L, kick_x=KICK)
    want = closed_orbit_nonlinear(lat)
    got = closed_orbit_nonlinear(lat, guess=np.array([5e-3, 1e-3, -2e-3, 4e-4]))
    assert np.allclose(got, want, atol=1e-14, rtol=0.0)


def test_failure_to_converge_is_reported_not_returned(ref: ReferenceParticle) -> None:
    """Non-convergence raises, and the two failure modes stay distinguishable.

    :class:`OrbitConvergenceError` is a :class:`ClosedOrbitError` so I1-era callers
    that already roll back on "no orbit" keep working, but it says something
    different: the iteration ran out of budget, not that the map has an eigenvalue
    1. The "no orbit at all" case still comes through unchanged, raised by the
    default seed on a ring with no focusing (zero tune).
    """
    assert issubclass(OrbitConvergenceError, ClosedOrbitError)

    tight = _lattice(ref, k2l=K2L, kick_x=KICK)
    with pytest.raises(OrbitConvergenceError, match="did not converge"):
        closed_orbit_nonlinear(tight, guess=np.array([2e-2, 0.0, 0.0, 0.0]), max_iter=1)

    unfocused = Lattice([Drift(2.0), ThinSextupole(K2L), Corrector(kick_x=KICK)], ref)
    with pytest.raises(ClosedOrbitError, match="integer"):
        closed_orbit_nonlinear(unfocused)


def test_feeddown_is_self_limiting_and_a_far_guess_finds_another_orbit(
    ref: ReferenceParticle,
) -> None:
    """Two honest limits on what :func:`closed_orbit_nonlinear` promises.

    **Feed-down does not run away.** Raising ``k2l`` by five orders of magnitude
    *shrinks* the orbit rather than destroying it: the same gradient the sextupole
    feeds down also stiffens the ``(I - M4)`` it is inverted against, so the fixed
    point is self-limiting. Convergence is therefore not evidence of a healthy
    machine — the linearised lattice here is wildly unstable and an orbit still
    closes through it, because closure needs ``(I - M4)`` invertible, not stable.

    **And the fixed point is not unique.** Started 50 m out, Newton converges onto
    a completely different orbit — one of the outer fixed points of the nonlinear
    map, the unstable ones outside the dynamic aperture. The docstring makes no
    claim about which one a far guess lands on; this is that non-claim, asserted.
    """
    weak = closed_orbit_nonlinear(_lattice(ref, k2l=K2L, kick_x=5e-3))
    strong = closed_orbit_nonlinear(_lattice(ref, k2l=1e6, kick_x=5e-3))
    assert abs(strong[X]) < 0.1 * abs(weak[X])

    lat = _lattice(ref, k2l=K2L, kick_x=KICK)
    near = closed_orbit_nonlinear(lat)
    far = closed_orbit_nonlinear(lat, guess=np.array([50.0, 0.0, 0.0, 0.0]))
    assert abs(far[X]) > 100.0 * abs(near[X])
    # Both really are fixed points — the map has more than one, and this is not a
    # convergence failure dressed up as an answer.
    for co in (near, far):
        state = np.zeros(DIM)
        state[:4] = co
        for elem in lat.elements:
            state = elem.track(state, lat.ref)
        assert np.allclose(state[:4], co, atol=1e-13, rtol=0.0)


def test_bad_arguments_are_rejected(ref: ReferenceParticle) -> None:
    """Shape and budget errors are errors, not silently-wrong orbits."""
    lat = _lattice(ref, k2l=K2L, kick_x=KICK)
    with pytest.raises(ValueError, match="length-4"):
        closed_orbit_nonlinear(lat, guess=np.zeros(6))
    with pytest.raises(ValueError, match="tol"):
        closed_orbit_nonlinear(lat, tol=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        closed_orbit_nonlinear(lat, max_iter=0)
    with pytest.raises(ValueError, match="step"):
        closed_orbit_nonlinear(lat, step=-1.0)
    with pytest.raises(ValueError, match="length-4"):
        propagate_orbit_nonlinear(lat, orbit0=np.zeros(3))
    with pytest.raises(ValueError, match="length-4"):
        linearised_element_maps(lat, orbit0=np.zeros(5))


# ---------------------------------------------------------------------------
# THE LEAD GATE: the dipole term, and the order at which it is right
# ---------------------------------------------------------------------------


def _predicted_departure(ref: ReferenceParticle, kick_x: float, kick_y: float, k2l: float):
    """I1's *linear* response to the derived feed-down dipole, and the true departure.

    The prediction never runs a nonlinear map: it evaluates ``theta`` from the
    **linear** orbit and puts it in the lattice as an ordinary
    :class:`Corrector`, then solves ``(I - M4) x = k4`` exactly as I1 does.
    """
    linear = _lattice(ref, k2l=0.0, kick_x=kick_x, kick_y=kick_y)
    x_lin = closed_orbit(linear)
    x_nl = closed_orbit_nonlinear(_lattice(ref, k2l=k2l, kick_x=kick_x, kick_y=kick_y))

    x_co, y_co = x_lin[X], x_lin[Y]
    probe = Lattice(
        [
            Corrector(kick_x=kick_x, kick_y=kick_y),
            Corrector(
                kick_x=-0.5 * k2l * (x_co**2 - y_co**2),
                kick_y=k2l * x_co * y_co,
            ),
            *_ring(),
        ],
        ref,
    )
    return x_nl - x_lin, closed_orbit(probe) - x_lin


def test_departure_is_the_linear_response_to_the_derived_dipole(
    ref: ReferenceParticle,
) -> None:
    r"""**The lead gate.** Nonlinear orbit - linear orbit == I1's response to ``theta``.

    Two routes that share no code: Newton on the tracked sextupole map, versus a
    :class:`Corrector` of the derived strength ``-1/2 k2l (x_co^2 - y_co^2)`` put
    through the I1 linear solve. They must agree to first order, and the *residual*
    must be one order higher — the quadratic-convergence pattern the beam-beam
    ``Delta Q`` gate uses.

    The order is the content, not the magnitude. Halving the steerer halves
    ``x_co``, so the departure falls by 4 (it is ``O(x_co^2)``) while the residual
    falls by 8 (``O(x_co^3)``: the dipole should have been evaluated at the
    *nonlinear* orbit, and the feed-down gradient perturbs the response matrix
    that transports it). A mis-scaled kick coefficient cannot fix both ratios at
    once.
    """
    dep, pred = _predicted_departure(ref, KICK, 0.0, K2L)
    assert np.max(np.abs(dep)) > 1e-7  # non-vacuous: there *is* a departure
    assert np.max(np.abs(dep - pred)) < 0.02 * np.max(np.abs(dep))

    residuals, departures = [], []
    for kick in (4e-4, 2e-4, 1e-4, 5e-5):
        d, p = _predicted_departure(ref, kick, 0.0, K2L)
        departures.append(np.max(np.abs(d)))
        residuals.append(np.max(np.abs(d - p)))

    for a, b in zip(departures[:-1], departures[1:], strict=True):
        assert a / b == pytest.approx(4.0, rel=0.02)  # O(x_co^2)
    for a, b in zip(residuals[:-1], residuals[1:], strict=True):
        assert a / b == pytest.approx(8.0, rel=0.05)  # O(x_co^3) — one order better


def test_a_mis_scaled_kick_breaks_the_dipole_gate(ref: ReferenceParticle) -> None:
    """The gate has teeth: a consistently doubled sextupole misses by a factor 2.

    :class:`_Misscaled` is still a gradient kick, still symplectic at every
    amplitude, still curl-free — every structural property of a sextupole survives
    (that is J1's lesson). It is simply twice as strong as it claims, and the
    dipole prediction, which is built from the *declared* ``k2l``, sees exactly
    that factor.
    """

    class _Misscaled(ThinSextupole):
        def track(self, state: np.ndarray, ref_: ReferenceParticle) -> np.ndarray:
            out = np.array(state, dtype=float, copy=True)
            out[PX] -= 1.0 * self.k2l * (out[X] ** 2 - out[Y] ** 2)
            out[PY] += 2.0 * self.k2l * out[X] * out[Y]
            return out

    linear = _lattice(ref, k2l=0.0, kick_x=KICK)
    x_lin = closed_orbit(linear)
    bad = Lattice(
        [Corrector(kick_x=KICK, name="steerer"), _Misscaled(K2L, name="sx"), *_ring()], ref
    )
    dep_bad = closed_orbit_nonlinear(bad) - x_lin

    _, pred = _predicted_departure(ref, KICK, 0.0, K2L)
    assert np.max(np.abs(dep_bad)) / np.max(np.abs(pred)) == pytest.approx(2.0, rel=0.02)


def test_the_dipole_to_gradient_ratio_is_pure_geometry(ref: ReferenceParticle) -> None:
    r"""``theta_x / k1l_eff = -x_co / 2``, whatever ``k2l`` is.

    Both terms are *measured* off the tracked sextupole map at the closed orbit —
    the constant part of the kick and the slope of the kick — and their ratio has
    no ``k2l`` in it. That independence is the discriminator J1 could not have
    built: J1 saw only the gradient, and a gradient alone fixes the product
    ``k2l x_co``, never the split between the two terms.
    """
    lat = _lattice(ref, k2l=K2L, kick_x=KICK)
    co = closed_orbit_nonlinear(lat)
    x_co = co[X]

    sx = ThinSextupole(K2L)
    at = np.array([x_co, 0.0, 0.0, 0.0, 0.0, 0.0])
    theta_x = sx.track(at, ref)[PX] - at[PX]  # constant part of the kick, at the orbit

    h = 1e-7
    plus, minus = at.copy(), at.copy()
    plus[X] += h
    minus[X] -= h
    k1l_eff = -(sx.track(plus, ref)[PX] - sx.track(minus, ref)[PX]) / (2.0 * h)

    assert k1l_eff == pytest.approx(K2L * x_co, rel=1e-9)
    assert theta_x / k1l_eff == pytest.approx(-x_co / 2.0, rel=1e-9)

    # ...and the ratio is unchanged at a tenth of the strength, which is the point.
    weak = ThinSextupole(0.1 * K2L)
    t2 = weak.track(at, ref)[PX] - at[PX]
    g2 = -(weak.track(plus, ref)[PX] - weak.track(minus, ref)[PX]) / (2.0 * h)
    assert t2 / g2 == pytest.approx(-x_co / 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# The equivalent lattice: feed-down rebuilt out of three validated elements
# ---------------------------------------------------------------------------


def _equivalent_lattice(
    ref: ReferenceParticle, k2l: float, orbit: np.ndarray, kick_x: float, kick_y: float
) -> Lattice:
    r"""The sextupole replaced by its linearisation about ``orbit``, as real elements.

    The corrector strength is **not** the ``theta`` of the expansion. The expansion
    is in the betatron deviation ``X = x - x_co``, whereas a
    :class:`ThinQuadrupole` in a lattice acts on the laboratory ``x`` and so
    already delivers ``-k1l_eff x_co`` at the orbit. Cancelling that back out
    leaves ``+1/2 k2l (x_co^2 - y_co^2)`` — the same coefficient with the opposite
    sign. Getting this wrong is the obvious way to build a plausible equivalent
    lattice that is silently wrong, so it is spelled out rather than inlined.
    """
    x_co, y_co = orbit[X], orbit[Y]
    return Lattice(
        [
            Corrector(kick_x=kick_x, kick_y=kick_y, name="steerer"),
            Corrector(
                kick_x=+0.5 * k2l * (x_co**2 - y_co**2),
                kick_y=-k2l * x_co * y_co,
                name="feeddown_dipole",
            ),
            ThinQuadrupole(k2l * x_co, name="feeddown_quad"),
            ThinSkewQuadrupole(k2l * y_co, name="feeddown_skew"),
            *_ring(),
        ],
        ref,
    )


def test_the_equivalent_linear_lattice_reproduces_orbit_and_optics(
    ref: ReferenceParticle,
) -> None:
    """A wholly *linear* lattice built from the feed-down terms has the same orbit and map.

    This is the strongest statement the decomposition can make: with the sextupole
    replaced by a corrector + a thin quad + a thin skew quad — three elements the
    package validates independently, none of which can track a nonlinear map — I1's
    linear solve lands on the *nonlinear* orbit, and the linear one-turn matrix
    equals the tracked map's Jacobian. Both planes are steered so the skew term is
    live, which a horizontal-only test could not see.
    """
    lat = _lattice(ref, k2l=K2L, kick_x=KICK, kick_y=0.6 * KICK)
    co = closed_orbit_nonlinear(lat)

    equiv = _equivalent_lattice(ref, K2L, co, KICK, 0.6 * KICK)
    assert np.allclose(closed_orbit(equiv), co, atol=1e-15, rtol=0.0)

    M_lin, _ = equiv.one_turn_map()
    assert np.allclose(linearised_one_turn_map(lat, co), M_lin, atol=1e-9, rtol=0.0)


def test_element_maps_multiply_to_the_one_turn_jacobian(ref: ReferenceParticle) -> None:
    """Chain rule, checked: the per-element product *is* the whole-turn linearisation.

    ``linearised_element_maps`` exists because Twiss propagation needs a matrix per
    element; this asserts that using them piecewise cannot disagree with a single
    finite difference of the whole turn.
    """
    from accsim.symplectic import jacobian

    lat = _lattice(ref, k2l=K2L, kick_x=KICK, kick_y=0.6 * KICK)
    co = closed_orbit_nonlinear(lat)

    product = np.eye(DIM)
    for m in linearised_element_maps(lat, co):
        product = m @ product
    assert np.allclose(product, linearised_one_turn_map(lat, co), atol=0.0, rtol=0.0)

    state = np.zeros(DIM)
    state[:4] = co

    def turn(s: np.ndarray) -> np.ndarray:
        for elem in lat.elements:
            s = elem.track(s, lat.ref)
        return s

    assert np.allclose(product, jacobian(turn, state, step=1e-7), atol=1e-8, rtol=0.0)

    # Every linear element is returned as its own matrix, so the feed-down is the
    # *only* place the linearisation differs from the on-axis optics.
    maps = linearised_element_maps(lat, co)
    for elem, m in zip(lat.elements, maps, strict=True):
        if isinstance(elem, ThinSextupole):
            assert not np.allclose(m, elem.matrix(ref), atol=1e-9)
        else:
            assert np.allclose(m, elem.matrix(ref), atol=1e-9, rtol=0.0)


# ---------------------------------------------------------------------------
# The optics move: tunes and the beta function at the source
# ---------------------------------------------------------------------------


def test_correctors_do_move_the_optics_once_a_sextupole_is_off_axis(
    ref: ReferenceParticle,
) -> None:
    """I1's headline claim, and the exact condition under which it fails.

    With the sextupole on axis (or absent) steering leaves beta and the tunes
    untouched to machine precision — I1 is right. Turn on ``k2l`` *and* the
    steerer together and both move. Either alone is not enough, which is what makes
    the qualification "linear-order, on-axis-sextupole" precise rather than
    defensive.
    """
    on_axis = _lattice(ref, k2l=K2L)
    steered_no_sext = _lattice(ref, k2l=0.0, kick_x=KICK)
    base = _fractional_tunes(_lattice(ref).one_turn_map()[0])

    for lat in (on_axis, steered_no_sext):
        co = closed_orbit_nonlinear(lat)
        q = _fractional_tunes(linearised_one_turn_map(lat, co))
        assert q == pytest.approx(base, abs=1e-12)

    both = _lattice(ref, k2l=K2L, kick_x=KICK)
    q = _fractional_tunes(linearised_one_turn_map(both, closed_orbit_nonlinear(both)))
    assert abs(q[0] - base[0]) > 1e-6
    assert abs(q[1] - base[1]) > 1e-6


def test_tune_shift_matches_the_beta_weighted_feeddown_sum(ref: ReferenceParticle) -> None:
    r"""``Delta Q_x = +sum beta_x k2l x_co / (4 pi)``, and its sign-flipped mirror in y.

    Made non-vacuous the way a single sextupole cannot be: **four** sextupoles, in
    different cells, at two different ``beta_x`` and with ``k2l`` of alternating
    sign, so the sum has real cancellation in it. A single source would let a wrong
    ``beta`` weighting or a dropped sign hide inside one number.

    ``beta`` and ``x_co`` are taken from the *unperturbed* lattice, so this is a
    genuine first-order prediction; the residual is checked to fall quadratically
    with ``k2l``, which is the statement that the first-order coefficient is right
    rather than merely close.
    """
    strengths = [+K2L, -0.6 * K2L, +0.35 * K2L, -0.8 * K2L]
    # Two per cell: at the cell boundary (beta_x max, alpha = 0) and at the D quad.
    slots = [(0, 0), (1, 2), (3, 0), (4, 2)]

    def build(scale: float) -> tuple[Lattice, list[int]]:
        elems: list = [Corrector(kick_x=KICK, name="steerer")]
        idx: list[int] = []
        for cell_i in range(N_CELLS):
            cell = _cell(tag=f"_{cell_i}")
            for j, (c, pos) in enumerate(slots):
                if c == cell_i:
                    cell.insert(pos, ThinSextupole(scale * strengths[j], name=f"sx{j}"))
            elems.extend(cell)
        for i, e in enumerate(elems):
            if isinstance(e, ThinSextupole):
                idx.append(i)
        return Lattice(elems, ref), idx

    def predicted(scale: float, lat: Lattice, idx: list[int]) -> tuple[float, float]:
        """First-order sum, from the *linear* lattice's beta and orbit."""
        flat = Lattice(
            [
                ThinSextupole(0.0, name=e.name) if isinstance(e, ThinSextupole) else e
                for e in lat.elements
            ],
            ref,
        )
        tw = propagate_twiss(flat, closed_twiss(flat))
        orb = propagate_orbit(flat)
        dqx = dqy = 0.0
        for i in idx:
            k2l = lat.elements[i].k2l
            dqx += tw[i].beta_x * k2l * orb[i][X] / (4.0 * math.pi)
            dqy -= tw[i].beta_y * k2l * orb[i][X] / (4.0 * math.pi)
        return dqx, dqy

    base_lat, base_idx = build(0.0)
    q0 = _fractional_tunes(base_lat.one_turn_map()[0])

    resid = []
    scales = (1.0, 0.5, 0.25)
    for scale in scales:
        lat, idx = build(scale)
        co = closed_orbit_nonlinear(lat)
        qx, qy = _fractional_tunes(linearised_one_turn_map(lat, co))
        px_, py_ = predicted(scale, lat, idx)
        if scale == 1.0:
            # Non-vacuous: real cancellation in the sum, and a measurable shift.
            terms = [
                propagate_twiss(base_lat, closed_twiss(base_lat))[i].beta_x
                * lat.elements[i].k2l
                * propagate_orbit(base_lat)[i][X]
                / (4.0 * math.pi)
                for i in idx
            ]
            assert abs(sum(terms)) < 0.6 * sum(abs(t) for t in terms)
            assert abs(px_) > 1e-5 and abs(py_) > 1e-5
            assert (qx - q0[0]) == pytest.approx(px_, rel=0.05)
            assert (qy - q0[1]) == pytest.approx(py_, rel=0.05)
        resid.append((abs((qx - q0[0]) - px_), abs((qy - q0[1]) - py_)))

    # The residual is second order in the strength: halving k2l quarters it.
    for a, b in zip(resid[:-1], resid[1:], strict=True):
        assert a[0] / b[0] == pytest.approx(4.0, rel=0.15)
        assert a[1] / b[1] == pytest.approx(4.0, rel=0.15)


def test_beta_at_the_source_matches_the_exact_gradient_error_form(
    ref: ReferenceParticle,
) -> None:
    r"""Beta at the sextupole, against an **exactly solved** thin-gradient closed form.

    No recalled beta-beat formula (the G2 trap). The ring is a palindrome, so
    ``alpha = 0`` where the sextupole sits, and the perturbed one-turn map there is
    just ``M0 Q`` with ``Q = [[1, 0], [-k1l, 1]]``. That 2x2 is solved in sympy for
    ``beta' = M12 / sin mu'`` with no expansion at all, so the comparison is exact
    rather than first-order:

        beta' = beta sin mu / sqrt(1 - (cos mu - k1l beta sin mu / 2)^2).

    The gate's content is *localisation*: the perturbation belongs to the
    sextupole's own ``beta_k``, and feeding the formula a different location's beta
    is decisively rejected.

    The comparison is made on ``(beta'/beta)^2`` so that no branch of ``sin mu``
    has to be chosen. ``M12 = beta sin mu`` is *negative* in this ring (the tune's
    fractional part exceeds 1/2), and ``beta = M12 / sin mu`` is positive only
    because both factors are; squaring keeps the statement exact and sidesteps a
    sign convention that would otherwise have to be asserted rather than derived.
    """
    sp = pytest.importorskip("sympy")
    beta_s, c_s, s_s, k1l_s = sp.symbols("beta c s k1l", real=True)
    M0 = sp.Matrix([[c_s, beta_s * s_s], [-s_s / beta_s, c_s]])  # alpha = 0, c/s = cos/sin mu
    M = M0 * sp.Matrix([[1, 0], [-k1l_s, 1]])

    # M12 is untouched by a thin gradient, so beta'/beta is purely a phase effect.
    assert sp.simplify(M[0, 1] - beta_s * s_s) == 0
    half_tr = sp.simplify(sp.trace(M) / 2)
    assert sp.simplify(half_tr - (c_s - k1l_s * beta_s * s_s / 2)) == 0

    # beta' = M12 / sin mu',  sin^2 mu' = 1 - (tr/2)^2  =>  (beta'/beta)^2 exactly:
    ratio_sq = sp.simplify(s_s**2 / (1 - half_tr**2))
    unperturbed = sp.simplify(ratio_sq.subs(k1l_s, 0).subs(c_s, sp.sqrt(1 - s_s**2)))
    assert sp.simplify(unperturbed - 1) == 0

    flat = _lattice(ref, k2l=0.0, kick_x=KICK)
    tw0 = closed_twiss(flat)
    assert abs(tw0.alpha_x) < 1e-12, "the palindrome symmetry this gate rests on"
    mu0 = 2.0 * math.pi * _fractional_tunes(flat.one_turn_map()[0])[0]

    lat = _lattice(ref, k2l=K2L, kick_x=KICK)
    co = closed_orbit_nonlinear(lat)
    got = match_periodic(linearised_one_turn_map(lat, co)).beta_x

    def predicted(beta_k: float) -> float:
        r = ratio_sq.subs(
            {beta_s: beta_k, c_s: math.cos(mu0), s_s: math.sin(mu0), k1l_s: K2L * co[X]}
        )
        return beta_k * math.sqrt(float(r))

    assert got != pytest.approx(tw0.beta_x, rel=1e-6)  # non-vacuous: beta really moved
    assert got == pytest.approx(predicted(tw0.beta_x), rel=1e-9)

    # Localisation: the *other* plane's beta is the wrong beta_k, and is rejected.
    assert got != pytest.approx(predicted(tw0.beta_y), rel=1e-3)


# ---------------------------------------------------------------------------
# The vertical orbit: a normal sextupole becomes a skew quadrupole
# ---------------------------------------------------------------------------


def test_a_horizontal_bump_leaves_the_orbit_exactly_planar(ref: ReferenceParticle) -> None:
    """``y = py = 0`` is an *exact* invariant subspace of the sextupole kick.

    At ``y = 0`` the kick is ``Delta px = -1/2 k2l x^2``, ``Delta py = 0``: nothing
    ever leaves the plane. This is exact arithmetic, not a small number, so it is
    asserted at zero — and it is what makes the vertical results below attributable
    to the vertical bump rather than to leakage.
    """
    lat = _lattice(ref, k2l=K2L, kick_x=KICK)
    co = closed_orbit_nonlinear(lat)
    assert np.array_equal(co[2:], np.zeros(2))
    for point in propagate_orbit_nonlinear(lat, co):
        assert np.array_equal(point[2:], np.zeros(2))

    # Uncoupled, therefore: the linearised map has no skew part at all.
    assert match_periodic_coupled(linearised_one_turn_map(lat, co)).gamma_c == 1.0


def test_a_vertical_bump_couples_the_planes(ref: ReferenceParticle) -> None:
    r"""**The strongest non-rerun gate**: ``y_co`` turns a normal sextupole skew.

    J1 exercised feed-down only in the horizontal, dispersive plane, where the
    product is a normal quadrupole. A *vertical* offset gives ``k1sl_eff = k2l
    y_co`` — betatron coupling, reached here from a direction nothing in the
    package has taken before, and measured with G2's Edwards-Teng machinery.

    The magnitude is pinned against the equivalent lattice built from
    :class:`ThinSkewQuadrupole` (validated against xtrack in G1), and the scaling
    ``1 - gamma_c ~ |C|^2 ~ y_co^2`` is checked because ``gamma_c`` is even in the
    coupling and so cannot be pinned by magnitude alone.
    """
    lat = _lattice(ref, k2l=K2L, kick_y=KICK)
    co = closed_orbit_nonlinear(lat)
    assert abs(co[Y]) > 1e-5  # there really is a vertical orbit

    ct = match_periodic_coupled(linearised_one_turn_map(lat, co))
    assert ct.gamma_c < 1.0 - 1e-9  # the planes are coupled at all

    equiv = _equivalent_lattice(ref, K2L, co, 0.0, KICK)
    ct_ref = match_periodic_coupled(equiv.one_turn_map()[0])
    assert ct.gamma_c == pytest.approx(ct_ref.gamma_c, rel=1e-9)
    assert np.allclose(ct.c_matrix, ct_ref.c_matrix, atol=1e-9, rtol=0.0)

    # Halving the bump halves k1sl_eff, so 1 - gamma_c falls by four.
    half = _lattice(ref, k2l=K2L, kick_y=0.5 * KICK)
    co_h = closed_orbit_nonlinear(half)
    ct_h = match_periodic_coupled(linearised_one_turn_map(half, co_h))
    assert (1.0 - ct.gamma_c) / (1.0 - ct_h.gamma_c) == pytest.approx(4.0, rel=0.02)


def test_a_purely_vertical_steerer_moves_the_horizontal_orbit(
    ref: ReferenceParticle,
) -> None:
    r"""``theta_x = +1/2 k2l y_co^2``: vertical steering produces a *horizontal* orbit.

    Flatly impossible in the linear theory — with ``kick_x = 0`` the horizontal
    inhomogeneity ``k4`` is identically zero and I1 returns exactly ``x = 0`` at
    every monitor, at any kick. The sextupole's ``+y^2`` term is the only thing
    that can produce it, and its **sign is opposite** to the horizontal case, which
    is the ``x^2 - y^2`` structure showing up in the orbit rather than in the tune.
    """
    lat_lin = _lattice(ref, k2l=0.0, kick_y=KICK)
    assert np.array_equal(closed_orbit(lat_lin)[:2], np.zeros(2))

    lat = _lattice(ref, k2l=K2L, kick_y=KICK)
    co = closed_orbit_nonlinear(lat)
    assert abs(co[X]) > 1e-9  # non-vacuous

    y_co = closed_orbit(lat_lin)[Y]
    probe = Lattice(
        [
            Corrector(kick_y=KICK),
            Corrector(kick_x=+0.5 * K2L * y_co**2),
            *_ring(),
        ],
        ref,
    )
    assert co[X] == pytest.approx(closed_orbit(probe)[X], rel=0.02)

    # The horizontal dipole feed-down flips sign between the planes: a horizontal
    # bump of the same size drives x the other way.
    lat_x = _lattice(ref, k2l=K2L, kick_x=KICK)
    dep_x = closed_orbit_nonlinear(lat_x)[X] - closed_orbit(_lattice(ref, kick_x=KICK))[X]
    assert dep_x * co[X] < 0.0


def test_the_vertical_dipole_feeddown_needs_both_planes(ref: ReferenceParticle) -> None:
    r"""``theta_y = +k2l x_co y_co`` is the one term no single bump can switch on.

    With only one plane steered it vanishes identically — no vertical kick from a
    horizontal bump (that is the planar-invariance result above), and no *extra*
    vertical kick from a vertical bump. Steering both planes at once is the only
    configuration in which it is live, and then the vertical orbit departs from
    what either bump alone would give.
    """
    dep_y_only = (
        closed_orbit_nonlinear(_lattice(ref, k2l=K2L, kick_y=KICK))[Y]
        - closed_orbit(_lattice(ref, kick_y=KICK))[Y]
    )
    both = _lattice(ref, k2l=K2L, kick_x=KICK, kick_y=KICK)
    dep_both = (
        closed_orbit_nonlinear(both)[Y] - closed_orbit(_lattice(ref, kick_x=KICK, kick_y=KICK))[Y]
    )

    # Adding a horizontal bump changes the *vertical* departure — only the
    # x_co y_co cross term can do that.
    assert abs(dep_both - dep_y_only) > 0.1 * abs(dep_y_only)


# ---------------------------------------------------------------------------
# The operational punchline: orbit correction becomes a loop
# ---------------------------------------------------------------------------


def test_orbit_correction_stops_being_one_solve_and_starts_iterating(
    ref: ReferenceParticle,
) -> None:
    r"""**Why real orbit correction iterates.**

    I1 lands on machine zero in a single application because the closed orbit is
    strictly affine in the corrector kicks. Feed-down breaks that affineness: the
    steering creates a dipole ``-1/2 k2l x_co^2`` at the sextupole that the model
    response matrix knows nothing about, so one pass leaves an ``O(k2l x_co^2)``
    residual — and that scaling is checked, not just its size.

    **The convergence is linear, and that is the interesting part.** ``R`` is
    rebuilt from the linear model every pass, so it never learns the feed-down
    gradient; the loop is a stale-Jacobian fixed-point iteration whose contraction
    factor is *constant* rather than shrinking. Asserting that the factor repeats
    pass after pass is a much sharper statement than "it gets smaller", and it is
    what distinguishes this from the Newton the module could have implemented.

    The control is the same machine with ``k2l = 0``, which still lands on machine
    zero in one pass, so the residual is feed-down and not a conditioning artefact.
    """
    # Three correctors against three monitors: a square, well-conditioned problem,
    # so "did not reach zero" cannot be least-squares over-determination in
    # disguise (the k2l = 0 control below is what proves that).
    corr_at, monitors = [4, 14, 24], [8, 18, 28]

    def steered(k2l: float, err: float = KICK) -> tuple[Lattice, list[Corrector]]:
        elems: list = [
            Corrector(kick_x=err, name="error"),
            ThinSextupole(k2l, name="sx"),
            *_ring(),
        ]
        corr = []
        for off, at in enumerate(corr_at):
            c = Corrector(kick_x=0.0, name=f"c{off}")
            elems.insert(at + off, c)
            corr.append(c)
        return Lattice(elems, ref), corr

    flat, corr = steered(0.0)
    out = correct_orbit(flat, corr, monitors, "x", nonlinear=True)
    assert out.rms_before > 1e-5
    assert out.rms_after < 1e-18  # no sextupole: still exactly one solve

    lat, corr = steered(K2L)
    passes = [correct_orbit(lat, corr, monitors, "x", nonlinear=True) for _ in range(4)]
    assert passes[0].rms_before > 1e-5
    assert passes[0].rms_after > 1e-9  # NOT machine zero any more — this is feed-down
    assert passes[-1].rms_after < 1e-16  # ...but it is a loop, and a fast one

    contraction = [p.rms_after / p.rms_before for p in passes[1:]]
    for f in contraction:
        assert f == pytest.approx(contraction[0], rel=1e-3)  # constant => linear, not quadratic
    assert contraction[0] < 1e-3

    # The first-pass residual is second order in the orbit that had to be removed.
    resid = []
    for scale in (1.0, 0.5, 0.25, 0.125):
        lat_s, corr_s = steered(K2L, err=scale * KICK)
        resid.append(correct_orbit(lat_s, corr_s, monitors, "x", nonlinear=True).rms_after)
    for a, b in zip(resid[:-1], resid[1:], strict=True):
        assert a / b == pytest.approx(4.0, rel=0.02)


def test_the_linear_correction_path_is_untouched(ref: ReferenceParticle) -> None:
    """``nonlinear=False`` is bit-for-bit I1, sextupole or not.

    The flag defaults off, and with it off :func:`correct_orbit` measures the
    *linear* closed orbit exactly as before — including, deliberately, reporting
    machine zero on a machine whose real orbit is not zero. That gap is the reason
    the flag exists, and it is asserted here so the default's blind spot is
    documented rather than discovered.
    """
    corr_at, monitors = [4, 14, 24], [8, 18, 28]
    elems: list = [Corrector(kick_x=KICK, name="error"), ThinSextupole(K2L, name="sx"), *_ring()]
    corr = []
    for off, at in enumerate(corr_at):
        c = Corrector(kick_x=0.0, name=f"c{off}")
        elems.insert(at + off, c)
        corr.append(c)
    lat = Lattice(elems, ref)

    out = correct_orbit(lat, corr, monitors, "x")
    assert out.rms_after < 1e-18  # the linear orbit is flat...

    # ...and the machine's actual orbit is not, by the feed-down scale.
    real = propagate_orbit_nonlinear(lat)
    rms_real = math.sqrt(float(np.mean([real[m][X] ** 2 for m in monitors])))
    assert rms_real > 1e-9


def test_chromaticity_is_a_design_orbit_quantity(ref: ReferenceParticle) -> None:
    """The blind spot I2 does **not** close, asserted rather than left to be found.

    The milestone statement said the feed-down gradient moves "tunes, beta and
    chromaticity". The first two are gated above. The third is not, and cannot be
    without new machinery: :func:`~accsim.twiss.chromaticity` is built on
    :func:`~accsim.twiss.propagate_twiss`, which walks each element's *on-axis*
    ``matrix()`` — so it is a **design-orbit** quantity and does not move at all
    when the machine is steered, however large the beta-beat actually is.

    This test pins that non-response exactly (it is a property of the code path, so
    the answer is bit-identical), alongside a measurement of the error being made:
    beta really has moved, by the amount ``chromaticity`` is ignoring.
    :func:`linearised_element_maps` is what a corrected version would be built on;
    I2 does not build it. Directly analogous to
    :func:`test_the_linear_correction_path_is_untouched`.
    """
    on_axis = _lattice(ref, k2l=K2L)
    steered = _lattice(ref, k2l=K2L, kick_x=KICK)

    # Bit-identical: no orbit enters the calculation at all.
    assert chromaticity(steered) == chromaticity(on_axis)

    # ...while the beta it is built on has moved by ~0.4%, which is the error.
    beta_design = closed_twiss(steered).beta_x
    beta_real = match_periodic(
        linearised_one_turn_map(steered, closed_orbit_nonlinear(steered))
    ).beta_x
    assert abs(beta_real / beta_design - 1.0) > 1e-3


def test_the_thick_sextupole_goes_through_the_same_machinery(
    ref: ReferenceParticle,
) -> None:
    """A thick :class:`Sextupole` converges too, and feeds down the same way.

    The gates above all use :class:`ThinSextupole` deliberately — a thick body's
    orbit varies across the magnet, so the thin-lens sums would carry an ``O(L^2)``
    error that would read as a loosened tolerance. But
    :func:`closed_orbit_nonlinear` accepts thick sextupoles, and nothing else here
    exercises that path: drift-kick-drift inside ``track()``, several slices, and a
    non-zero length between the kick and the orbit that produced it.
    """
    lat = Lattice(
        [
            Corrector(kick_x=KICK, name="steerer"),
            Sextupole(0.4, K2L / 0.4, name="sx", n_slices=4),
            *_ring(),
        ],
        ref,
    )
    co = closed_orbit_nonlinear(lat)

    state = np.zeros(DIM)
    state[:4] = co
    for elem in lat.elements:
        state = elem.track(state, lat.ref)
    assert np.allclose(state[:4], co, atol=1e-14, rtol=0.0)

    # Same physics: the departure from the linear orbit is still O(x_co^2), and the
    # optics still move. (Its *value* differs from the thin case at O(L^2), which
    # is why the quantitative gates above use a thin sextupole.)
    departures = []
    for kick in (4e-4, 2e-4, 1e-4):
        lat_k = Lattice(
            [
                Corrector(kick_x=kick),
                Sextupole(0.4, K2L / 0.4, n_slices=4),
                *_ring(),
            ],
            ref,
        )
        flat_k = Lattice([Corrector(kick_x=kick), Sextupole(0.4, 0.0), *_ring()], ref)
        departures.append(
            float(np.max(np.abs(closed_orbit_nonlinear(lat_k) - closed_orbit(flat_k))))
        )
    assert departures[0] > 1e-7
    for a, b in zip(departures[:-1], departures[1:], strict=True):
        assert a / b == pytest.approx(4.0, rel=0.02)
