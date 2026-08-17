r"""J3 acceptance: octupole feed-down on a distorted closed orbit.

J2 gave the octupole an exact cubic kick and then **refused** to linearise it about
an orbit: :func:`~accsim.orbit.linearised_lattice` raised rather than report a drift.
This file is what that refusal was deferring. Expanding

    Delta px = -1/6 k3l (x^3 - 3 x y^2),   Delta py = +1/6 k3l (3 x^2 y - y^3)

about an orbit offset ``(x_co, y_co)`` splits **one octupole into six elements**,

    dipole       theta_x  = -1/6 k3l x_co (x_co^2 - 3 y_co^2)
                 theta_y  = +1/6 k3l y_co (3 x_co^2 - y_co^2)
    normal quad  k1l_eff  = +1/2 k3l (x_co^2 - y_co^2)
    skew quad    k1sl_eff = +k3l x_co y_co
    normal sext  k2l_eff  = +k3l x_co
    skew sext    k2sl_eff = +k3l y_co
    octupole     unchanged

every coefficient of which is *derived* in :func:`test_feeddown_expansion_is_derived`.

**The gate is a ladder of three powers, and that is the whole point.** A cubic kick
reaches two orders below itself, where the sextupole's quadratic kick reached one. So
the same single coefficient set has to satisfy three quantities that the package
computes by three unrelated routes, each with a *different* power of the orbit:

    chromaticity   ~ x_co     (the sextupole pair, at dispersion)
    tunes / beta   ~ x_co^2   (the gradient pair)
    the orbit      ~ x_co^3   (the dipole)

Measured over four halvings of the steerer: 2.0, 4.0, 8.0, with residuals falling by
8, 16 and 32 respectively — one order better in each case, which is the statement
that the *coefficient* is right rather than merely the shape.

⚠️ **The ladder alone has no teeth, and neither does a magnitude check alone.** A
uniformly mis-scaled octupole (``1`` in place of ``1/6``, which J2 carries through
every structural check) leaves all three *powers* untouched and is caught only as a
clean factor **6** in size; a single magnitude comparison at a single amplitude is
exactly what J1 and J2 showed a wrong coefficient can survive. Both halves are here
(:func:`test_a_mis_scaled_octupole_is_caught_as_a_factor_of_six`), and neither is
sufficient.

**What is deliberately not re-tested.** J2 already measured the gradient term by
finite-differencing ``track()`` at an offset, so ``k1l_eff`` is used here as an
ingredient rather than as a headline. The three things with no J2 analogue carry the
file: the sextupole pair (a first-order chromatic effect where an on-axis octupole has
*exactly none*), the exact six-way identity, and the fact that ``x = px = 0`` is an
invariant subspace of an octupole but **not** of a sextupole.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Corrector,
    Dipole,
    Drift,
    Lattice,
    Octupole,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    ThinQuadrupole,
    ThinSextupole,
    ThinSkewQuadrupole,
    ThinSkewSextupole,
    chromaticity,
    closed_orbit,
    closed_orbit_nonlinear,
    closed_twiss,
    linearised_element_maps,
    linearised_one_turn_map,
    match_periodic_coupled,
    propagate_orbit,
    propagate_twiss,
)
from accsim.coords import DIM, PX, PY, X, Y
from accsim.orbit import linearised_lattice, propagate_orbit_nonlinear
from accsim.twiss import (
    chromaticity_on_orbit,
    natural_chromaticity_on_orbit,
    tunes_on_orbit,
)

# --------------------------------------------------------------------------
# The rings. Two of them, for the same reason I3 needed two.
# --------------------------------------------------------------------------

# A palindromic thin FODO (alpha = 0 at the entrance, where the octupole sits) with
# no dipoles at all, hence D_x = 0: the orbit and tune gates run here, where the
# sextupole pair contributes nothing and cannot muddy the gradient measurement.
VF = 1.0 / 1.5  # full-quad inverse focal length, F family [m^-1]
VD = 1.0 / 1.6  # ditto, D family [m^-1]
L_HALF = 1.0  # half-cell drift [m]
N_CELLS = 6

K3L = 3.0e5  # integrated octupole strength for the orbit gate [m^-3]
K3L_TUNE = 3.0e4  # weaker for the tune gate: at K3L the fed-down gradient unstables it
KICK = 2e-4  # steerer angle [rad] -> a sub-mm orbit

# The chromaticity ring has dipoles, because an octupole feeds down to chromaticity
# only at dispersion — the same split I3 made for the sextupole.
K3L_CHROMA = 1.0e4


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


def _flat(
    ref: ReferenceParticle, k3l: float = 0.0, kick_x: float = 0.0, kick_y: float = 0.0
) -> Lattice:
    """Steerer + one octupole at the (``alpha = 0``) entrance, then the FODO ring.

    The steerer is thin and touches only ``px``/``py``, so the *position* the octupole
    sees is the closed orbit reported at the lattice entrance (index 1 of the orbit
    table).
    """
    return Lattice(
        [
            Corrector(kick_x=kick_x, kick_y=kick_y, name="steerer"),
            ThinOctupole(k3l, name="oc"),
            *_ring(),
        ],
        ref,
    )


def _dispersive(ref: ReferenceParticle, k3l: float = 0.0, kick_x: float = 0.0) -> Lattice:
    """FODO-with-dipoles (nonzero ``D_x``) carrying one thin octupole per cell."""
    els: list = [Corrector(kick_x=kick_x, name="steerer")]
    for i in range(3):
        els += [
            Quadrupole(0.3, 1.2, name=f"qf_{i}"),
            Drift(0.5),
            ThinOctupole(k3l, name=f"oc_{i}"),
            Drift(0.5),
            Dipole(1.0, 0.12, name=f"b1_{i}"),
            Quadrupole(0.3, -1.2, name=f"qd_{i}"),
            Dipole(1.0, 0.12, name=f"b2_{i}"),
            Drift(0.5),
        ]
    return Lattice(els, ref)


def _fractional_tunes(M: np.ndarray) -> tuple[float, float]:
    """Fractional tunes from the 2x2 diagonal blocks of an uncoupled 6x6 map."""
    out = []
    for a, b in ((X, PX), (Y, PY)):
        half_trace = 0.5 * (M[a, a] + M[b, b])
        assert abs(half_trace) < 1.0, f"unstable linearised map: |tr/2| = {abs(half_trace)}"
        q = math.acos(half_trace) / (2.0 * math.pi)
        out.append(q if M[a, b] >= 0.0 else 1.0 - q)
    return out[0], out[1]


def _ratios(values: list[float]) -> list[float]:
    return [a / b for a, b in zip(values[:-1], values[1:], strict=True)]


# --------------------------------------------------------------------------
# 1. The expansion itself — derived, not recalled
# --------------------------------------------------------------------------


def test_feeddown_expansion_is_derived(ref: ReferenceParticle) -> None:
    r"""Every one of the six coefficients, from the J2 kick, in sympy.

    Each order of the Taylor expansion about ``(x_co, y_co)`` is matched to the
    *existing, separately validated* element it equals — :class:`Corrector`,
    :class:`ThinQuadrupole`, :class:`ThinSkewQuadrupole`, :class:`ThinSextupole`,
    :class:`ThinSkewSextupole`, :class:`ThinOctupole` — so nothing here is a
    remembered formula.

    Two properties are checked that no single component could establish:

    - **The gradient pair is over-determined.** ``k1l_eff`` and ``k1sl_eff`` are each
      read off *twice*, once from ``Delta px`` and once from ``Delta py``, and must
      agree. That is Maxwell showing up as an arithmetic identity: a kick that failed
      it would not be a magnetic field at all.
    - **The dipole/gradient ratio is pure geometry**, ``theta_x / k1l_eff = -x_co/3``
      at ``y_co = 0``, with no ``k3l`` in it — and it is ``-x_co/3`` where the
      sextupole's (I2) is ``-x_co/2``. A gradient measurement alone fixes only the
      product ``k3l x_co^2`` and never the split between the terms.
    """
    xb, yb, k3l, xco, yco = sp.symbols("X Y k3l x_co y_co", real=True)

    x, y = xco + xb, yco + yb
    dpx = sp.expand(-k3l * (x**3 - 3 * x * y**2) / 6)
    dpy = sp.expand(+k3l * (3 * x**2 * y - y**3) / 6)

    def order(expr, n):
        return sum(t for t in expr.as_ordered_terms() if sp.Poly(t, xb, yb).total_degree() == n)

    # Constant term: a Corrector, whose kick() adds directly to (px, py).
    theta_x = -k3l * xco * (xco**2 - 3 * yco**2) / 6
    theta_y = +k3l * yco * (3 * xco**2 - yco**2) / 6
    assert sp.simplify(order(dpx, 0) - theta_x) == 0
    assert sp.simplify(order(dpy, 0) - theta_y) == 0

    # Linear term: ThinQuadrupole is  px -= k1l x, py += k1l y;
    #              ThinSkewQuadrupole is  px += k1sl y, py += k1sl x.
    k1l_eff = k3l * (xco**2 - yco**2) / 2
    k1sl_eff = k3l * xco * yco
    assert sp.simplify(order(dpx, 1) - (-k1l_eff * xb + k1sl_eff * yb)) == 0
    assert sp.simplify(order(dpy, 1) - (+k1l_eff * yb + k1sl_eff * xb)) == 0

    # Quadratic term: ThinSextupole is  px -= 1/2 k2l (x^2 - y^2), py += k2l x y;
    #                 ThinSkewSextupole is  px += k2sl x y, py += 1/2 k2sl (x^2 - y^2).
    k2l_eff, k2sl_eff = k3l * xco, k3l * yco
    assert sp.simplify(order(dpx, 2) - (-k2l_eff * (xb**2 - yb**2) / 2 + k2sl_eff * xb * yb)) == 0
    assert sp.simplify(order(dpy, 2) - (+k2l_eff * xb * yb + k2sl_eff * (xb**2 - yb**2) / 2)) == 0

    # Cubic term: the octupole is unchanged by the shift.
    assert sp.simplify(order(dpx, 3) + k3l * (xb**3 - 3 * xb * yb**2) / 6) == 0
    assert sp.simplify(order(dpy, 3) - k3l * (3 * xb**2 * yb - yb**3) / 6) == 0

    # The gradient pair, read off the *other* component, must be the same numbers.
    assert sp.simplify(sp.diff(order(dpx, 1), xb) + sp.diff(order(dpy, 1), yb)) == 0
    assert sp.simplify(sp.diff(order(dpx, 1), yb) - sp.diff(order(dpy, 1), xb)) == 0

    # The dipole/gradient ratio is pure geometry: no k3l survives, and it is not the
    # sextupole's -x_co/2.
    ratio = sp.simplify(theta_x.subs(yco, 0) / k1l_eff.subs(yco, 0))
    assert sp.simplify(ratio + xco / 3) == 0
    assert sp.simplify(ratio + xco / 2) != 0

    # ...and the whole kick stays curl-free, i.e. it is still a real magnet.
    assert sp.simplify(sp.diff(dpx, yb) - sp.diff(dpy, xb)) == 0


def _split(k3l: float, x_co: float, y_co: float) -> dict[str, float]:
    """The six derived strengths at an orbit offset, in one place."""
    return {
        "theta_x": -k3l * x_co * (x_co**2 - 3.0 * y_co**2) / 6.0,
        "theta_y": +k3l * y_co * (3.0 * x_co**2 - y_co**2) / 6.0,
        "k1l": 0.5 * k3l * (x_co**2 - y_co**2),
        "k1sl": k3l * x_co * y_co,
        "k2l": k3l * x_co,
        "k2sl": k3l * y_co,
    }


def test_the_six_way_split_reproduces_the_kick_exactly(ref: ReferenceParticle) -> None:
    r"""**The exact identity**: the octupole's kick at an offset *is* the six kicks.

    For a thin octupole the expansion terminates at the cubic term, so this is an
    algebraic identity and holds to round-off rather than to a tolerance — which is
    also its danger. An exact residual makes a gate blind (the H2 lesson), so two
    things are done deliberately:

    - **Both planes are steered, and both offsets are asserted nonzero.** With
      ``y_co = 0`` the skew quadrupole and the skew sextupole vanish identically and
      their signs could be anything at all; a horizontal-only version of this test
      would pass with either.
    - **Every one of the six coefficients is flipped in turn** and shown to break the
      identity. That converts "the arithmetic is consistent" into "each term is
      individually load-bearing".
    """
    k3l, x_co, y_co = 2.5e4, 1.7e-3, -1.1e-3
    dev = np.array([3.0e-4, 0.0, -2.0e-4, 0.0, 0.0, 0.0])  # betatron deviation
    assert x_co != 0.0 and y_co != 0.0

    at_orbit = np.array([x_co + dev[X], 0.0, y_co + dev[Y], 0.0, 0.0, 0.0])
    truth = ThinOctupole(k3l).track(at_orbit, ref) - at_orbit

    def chain(s: dict[str, float]) -> np.ndarray:
        """Sum of the six kicks, each evaluated at the *betatron* deviation."""
        total = np.zeros(DIM)
        for elem in (
            Corrector(kick_x=s["theta_x"], kick_y=s["theta_y"]),
            ThinQuadrupole(s["k1l"]),
            ThinSkewQuadrupole(s["k1sl"]),
            ThinSextupole(s["k2l"]),
            ThinSkewSextupole(s["k2sl"]),
            ThinOctupole(k3l),
        ):
            total += elem.track(dev, ref) - dev
        return total

    parts = _split(k3l, x_co, y_co)
    got = chain(parts)
    assert abs(truth[PX]) > 1e-9 and abs(truth[PY]) > 1e-9  # non-vacuous
    assert np.allclose(got, truth, atol=1e-18, rtol=1e-12)

    # Each coefficient is load-bearing: flipping any one of the six breaks it, and by
    # more than round-off in *both* momentum components it touches.
    for key in parts:
        broken = dict(parts)
        broken[key] = -broken[key]
        miss = np.max(np.abs(chain(broken) - truth))
        assert miss > 1e-12, f"flipping {key} changed nothing — the gate cannot see it"


def test_linearised_octupole_is_the_two_thin_quadrupoles(ref: ReferenceParticle) -> None:
    """Element level: the Jacobian at an offset **is** the derived quad pair's product.

    The derivation above made numerical, against two elements the package validates
    independently against xtrack — so the linear half of the decomposition borrows
    their credibility instead of re-deriving its own. The sextupole pair has no linear
    part and correctly does not appear.
    """
    from accsim.symplectic import jacobian

    k3l, x_co, y_co = 2.5e4, 1.7e-3, -1.1e-3
    oc = ThinOctupole(k3l)
    state = np.array([x_co, 0.0, y_co, 0.0, 0.0, 0.0])
    parts = _split(k3l, x_co, y_co)

    got = jacobian(lambda s: oc.track(s, ref), state, step=1e-7)
    want = ThinQuadrupole(parts["k1l"]).matrix(ref) @ ThinSkewQuadrupole(parts["k1sl"]).matrix(ref)
    assert np.allclose(got, want, atol=1e-9, rtol=0.0)

    # The two thin kicks commute exactly, so the order of the product is not a hidden
    # assumption (the same statement I2 makes for the sextupole's pair).
    other = ThinSkewQuadrupole(parts["k1sl"]).matrix(ref) @ ThinQuadrupole(parts["k1l"]).matrix(ref)
    assert np.array_equal(want, other)


# --------------------------------------------------------------------------
# 2. The equivalent lattice: derived coefficients vs differentiated track()
# --------------------------------------------------------------------------


def test_the_equivalent_lattice_matches_the_differentiated_map(ref: ReferenceParticle) -> None:
    r"""``linearised_lattice`` (derived) == ``linearised_one_turn_map`` (finite difference).

    Two routes to the same matrix that share no arithmetic: one builds real elements
    from the coefficients derived above, the other differentiates the octupole's real
    ``track()``. Both planes are steered so the skew quadrupole is live.

    **The residual is shown to be the differencing, not the physics.** It falls as
    ``step^2`` — measured 1.66e-6 / 1.85e-7 / 1.66e-8 / 1.85e-9 at steps 3e-6 down to
    1e-7, on matrix entries up to 3.7 — which is the central difference's truncation
    error for a map with a non-zero third derivative. A disagreement in the derived
    split would sit at a *fixed* size instead, so asserting the scaling is a much
    sharper statement than asserting a tolerance.

    The steps are a constant factor 3 apart on purpose: mixing ``3e-7 -> 1e-7``
    (factor 3.33) with ``3e-6 -> 1e-6`` (factor 3) makes the expected ratio alternate
    between 9 and 11.1, which reads as a broken scaling when it is only arithmetic.
    """
    lat = _flat(ref, k3l=K3L, kick_x=4e-4, kick_y=2.5e-4)
    co = closed_orbit_nonlinear(lat)
    orbit = propagate_orbit_nonlinear(lat, co)
    assert abs(orbit[1][X]) > 1e-5 and abs(orbit[1][Y]) > 1e-5  # both planes live

    M_equiv, _ = linearised_lattice(lat).one_turn_map()
    assert np.max(np.abs(M_equiv)) > 1.0  # non-vacuous: a real map, not near-identity

    steps = [2.7e-6, 9e-7, 3e-7, 1e-7]
    gaps = [
        float(np.max(np.abs(linearised_one_turn_map(lat, co, step=s) - M_equiv))) for s in steps
    ]
    assert gaps[-1] < 1e-8
    for r in _ratios(gaps):
        assert r == pytest.approx(9.0, rel=0.05)  # step^2, at steps a factor 3 apart


def test_the_equivalent_lattice_keeps_the_terms_that_matrices_cannot_see(
    ref: ReferenceParticle,
) -> None:
    """What the returned lattice contains, and why the invisible terms are kept.

    The gradient pair is what a matrix-based optics function reads. The sextupole
    pair has an identity ``matrix()`` and changes nothing linear — but a *normal*
    sextupole is exactly what makes an off-axis octupole a first-order chromaticity
    source, and the chromaticity integrals walk element **types**, so dropping it
    would silently zero the effect this milestone exists for. The octupole itself is
    kept for the same reason I2 keeps its sextupole: the split is the static feed-down
    at the orbit, not a replacement for the element.
    """
    lat = _flat(ref, k3l=K3L, kick_x=4e-4, kick_y=2.5e-4)
    orbit = propagate_orbit_nonlinear(lat)
    x_co, y_co = float(orbit[1][X]), float(orbit[1][Y])
    parts = _split(K3L, x_co, y_co)

    equiv = linearised_lattice(lat)
    by_name = {e.name: e for e in equiv.elements if e.name and e.name.startswith("oc_fd")}
    assert set(by_name) == {"oc_fd_quad", "oc_fd_skew", "oc_fd_sext", "oc_fd_skewsext"}
    assert by_name["oc_fd_quad"].k1l == pytest.approx(parts["k1l"], rel=1e-12)
    assert by_name["oc_fd_skew"].k1sl == pytest.approx(parts["k1sl"], rel=1e-12)
    assert by_name["oc_fd_sext"].k2l == pytest.approx(parts["k2l"], rel=1e-12)
    assert by_name["oc_fd_skewsext"].k2sl == pytest.approx(parts["k2sl"], rel=1e-12)

    # The octupole survives, and the dipole term does not appear: it is what placed
    # the orbit these strengths are read at, and a Corrector's matrix() is the
    # identity anyway, so including it would change nothing and claim something.
    assert any(isinstance(e, ThinOctupole) and e.k3l == K3L for e in equiv.elements)
    assert not any(isinstance(e, Corrector) and e.name and "fd" in e.name for e in equiv.elements)


# --------------------------------------------------------------------------
# 3. THE LADDER, rung by rung
# --------------------------------------------------------------------------


def _chromaticity_feeddown(lat: Lattice) -> tuple[float, float]:
    """The sextupole-pair term: total on-orbit chromaticity minus its natural half."""
    total = chromaticity_on_orbit(lat)
    natural = natural_chromaticity_on_orbit(lat)
    return total[0] - natural[0], total[1] - natural[1]


def _predicted_chromaticity_feeddown(
    ref: ReferenceParticle, kick: float, k3l: float
) -> tuple[float, float]:
    """First-order prediction ``+/- sum beta k2l_eff D_x / (4 pi)`` from the *linear* ring.

    ``beta``, ``D_x`` and ``x_co`` all come from the octupole-free lattice, so this is
    a genuine first-order prediction and not a rearrangement of the measurement.
    """
    lin = _dispersive(ref, 0.0, kick)
    tw = propagate_twiss(lin, closed_twiss(lin))
    orb = propagate_orbit(lin)
    xi_x = xi_y = 0.0
    for i, elem in enumerate(lin.elements):
        if isinstance(elem, ThinOctupole):
            k2l_eff = k3l * float(orb[i][X])  # the derived sextupole pair, normal half
            xi_x += +tw[i].beta_x * k2l_eff * tw[i].disp_x / (4.0 * math.pi)
            xi_y += -tw[i].beta_y * k2l_eff * tw[i].disp_x / (4.0 * math.pi)
    return xi_x, xi_y


def test_an_octupole_on_a_distorted_orbit_is_a_first_order_chromatic_element(
    ref: ReferenceParticle,
) -> None:
    r"""**Rung one, and the headline.** ``Q'`` moves at ``O(x_co)``, from exactly zero.

    On the design orbit an octupole contributes to first-order chromaticity
    *precisely nothing* — J2 derived that its ``delta`` term is a sextupole rather
    than a gradient, so ``Q''`` is the blind spot and ``Q'`` is untouched. That
    non-response is asserted here first, bit-for-bit, because it is what makes the
    rest of this test a measurement of something that did not exist before.

    Steer the machine and the same octupole becomes a *sextupole* of strength
    ``k2l_eff = k3l x_co``, which at dispersion feeds down the ordinary
    ``beta k2l D_x / (4 pi)``. That is **first order in the orbit** — the lowest rung
    of the ladder, and the one that separates this milestone's coefficient set from a
    gradient-only one, since the gradient term cannot produce it at any strength.

    Measured across four halvings: the shift falls by 2.0 (1.88 / 1.97 / 1.99, the
    approach to 2 being the higher-order terms dying) while the residual against the
    first-order prediction falls by 8 — two orders better, because the correction
    enters as a *relative* ``O(x_co^2)`` beat on beta and dispersion.
    """
    on_axis = _dispersive(ref, K3L_CHROMA, 0.0)
    assert chromaticity_on_orbit(on_axis) == chromaticity(on_axis)  # exactly nothing

    measured, residual = [], []
    for kick in (4e-4, 2e-4, 1e-4, 5e-5):
        lat = _dispersive(ref, K3L_CHROMA, kick)
        got = _chromaticity_feeddown(lat)
        want = _predicted_chromaticity_feeddown(ref, kick, K3L_CHROMA)
        if kick == 4e-4:
            assert abs(got[0]) > 1.0  # non-vacuous: a huge chromaticity shift
            assert got[0] == pytest.approx(want[0], rel=0.1)
            assert got[1] == pytest.approx(want[1], rel=0.1)
            # The two planes move in opposite directions, as a normal sextupole makes
            # them — a sign statement the magnitude alone would not carry.
            assert got[0] * got[1] < 0.0
        measured.append(abs(got[0]))
        residual.append(abs(got[0] - want[0]))

    for r in _ratios(measured):
        assert r == pytest.approx(2.0, rel=0.07)  # O(x_co) — first order
    for r in _ratios(residual):
        assert r == pytest.approx(8.0, rel=0.15)  # O(x_co^3) — two orders better


def test_the_tracked_route_reaches_the_same_chromaticity(ref: ReferenceParticle) -> None:
    r"""The chromaticity rung, by a route with no Twiss integral in it at all.

    ``chromaticity_on_orbit`` evaluates F2's integrals on the equivalent lattice built
    from the *derived* split. The tracked route instead Newtons for the nonlinear
    closed orbit at ``delta = +/- h``, differentiates ``track()`` there, and differences
    the resulting tunes — the J1/I3 route, which knows nothing about the split. They
    must agree, and the way they disagree is the content:

    the gap falls by exactly **4 per halving of h** (measured 4.09e-3, 1.02e-3,
    2.56e-4, 6.40e-5 at h = 4e-5 ... 5e-6), i.e. it is the central difference's own
    ``O(h^2)`` truncation and extrapolates to zero. A wrong ``k2l_eff`` would leave a
    gap that *did not move* with ``h``, which no tolerance-based comparison could
    distinguish from a small floor.
    """
    lat = _dispersive(ref, K3L_CHROMA, 2e-4)
    integral = _chromaticity_feeddown(lat)[0]
    assert abs(integral) > 1.0

    gaps = []
    for h in (4e-5, 2e-5, 1e-5, 5e-6):
        qx_p, _ = tunes_on_orbit(lat, delta=+h)
        qx_m, _ = tunes_on_orbit(lat, delta=-h)
        gaps.append(abs((qx_p - qx_m) / (2.0 * h) - integral))
    for r in _ratios(gaps):
        assert r == pytest.approx(4.0, rel=0.1)  # O(h^2): the differencing, not a bias
    assert gaps[-1] / abs(integral) < 1e-5


def test_the_tunes_move_at_second_order_in_the_orbit(ref: ReferenceParticle) -> None:
    r"""**Rung two.** ``Delta Q_x = +beta_x k1l_eff / (4 pi)`` with ``k1l_eff ~ x_co^2``.

    The gradient pair, on the dispersion-free ring so that the sextupole pair
    contributes nothing and this is the gradient alone. The prediction uses ``beta``
    and ``x_co`` from the *unperturbed* lattice, so it is first order in the
    perturbation while being second order in the orbit — the two orders are
    independent and both are checked.

    The vertical shift is the horizontal one with the opposite sign, which is what
    makes this a gradient rather than a general perturbation: a thin quadrupole
    focuses one plane exactly as much as it defocuses the other.
    """
    base = _fractional_tunes(_flat(ref).one_turn_map()[0])
    shifts, residual = [], []
    for kick in (4e-4, 2e-4, 1e-4, 5e-5):
        lat = _flat(ref, k3l=K3L_TUNE, kick_x=kick)
        co = closed_orbit_nonlinear(lat)
        qx, qy = _fractional_tunes(linearised_one_turn_map(lat, co))

        lin = _flat(ref, kick_x=kick)
        tw = propagate_twiss(lin, closed_twiss(lin))[1]
        orb = propagate_orbit(lin)[1]
        k1l_eff = 0.5 * K3L_TUNE * (float(orb[X]) ** 2 - float(orb[Y]) ** 2)
        want_x = +tw.beta_x * k1l_eff / (4.0 * math.pi)
        want_y = -tw.beta_y * k1l_eff / (4.0 * math.pi)

        if kick == 4e-4:
            assert abs(qx - base[0]) > 1e-4  # non-vacuous
            assert (qx - base[0]) == pytest.approx(want_x, rel=0.01)
            assert (qy - base[1]) == pytest.approx(want_y, rel=0.02)
            assert (qx - base[0]) * (qy - base[1]) < 0.0  # focus one, defocus the other
        shifts.append(abs(qx - base[0]))
        residual.append(abs((qx - base[0]) - want_x))

    for r in _ratios(shifts):
        assert r == pytest.approx(4.0, rel=0.02)  # O(x_co^2) — second order
    for r in _ratios(residual):
        assert r == pytest.approx(16.0, rel=0.05)  # O(x_co^4) — two orders better


def _orbit_departure(ref: ReferenceParticle, kick_x: float, kick_y: float, k3l: float):
    """The true departure from the linear orbit, and the linear response to ``theta``.

    The prediction never runs a nonlinear map: ``theta`` is evaluated at the *linear*
    orbit and inserted as an ordinary :class:`Corrector`, then the fixed point is
    solved with I1's linear algebra.
    """
    linear = _flat(ref, kick_x=kick_x, kick_y=kick_y)
    x_lin = closed_orbit(linear)
    x_nl = closed_orbit_nonlinear(_flat(ref, k3l=k3l, kick_x=kick_x, kick_y=kick_y))

    parts = _split(k3l, float(x_lin[X]), float(x_lin[Y]))
    probe = Lattice(
        [
            Corrector(kick_x=kick_x, kick_y=kick_y),
            Corrector(kick_x=parts["theta_x"], kick_y=parts["theta_y"]),
            *_ring(),
        ],
        ref,
    )
    return x_nl - x_lin, closed_orbit(probe) - x_lin


def test_the_orbit_itself_departs_at_third_order(ref: ReferenceParticle) -> None:
    r"""**Rung three.** The closed orbit moves by the dipole term, ``O(x_co^3)``.

    Newton on the tracked octupole map, against a :class:`Corrector` of the derived
    strength ``-1/6 k3l x_co (x_co^2 - 3 y_co^2)`` put through I1's *linear* solve:
    two routes sharing no code. As in I2 the order is the content — halving the
    steerer cuts the departure by 8 where the sextupole's fell by 4, and the residual
    by 32 (the dipole should have been evaluated at the nonlinear orbit, and the
    fed-down gradient perturbs the response that transports it — both ``O(x_co^5)``).

    This is the rung that is *cubic* in the orbit, and it is the reason the ladder
    works: no rescaling of the octupole can turn a cubic response into a quadratic
    one, so the three exponents together identify the expansion uniquely.
    """
    dep, pred = _orbit_departure(ref, KICK, 0.0, K3L)
    assert np.max(np.abs(dep)) > 1e-7  # non-vacuous: there *is* a departure
    assert np.max(np.abs(dep - pred)) < 0.05 * np.max(np.abs(dep))

    departures, residuals = [], []
    for kick in (8e-4, 4e-4, 2e-4, 1e-4, 5e-5):
        d, p = _orbit_departure(ref, kick, 0.0, K3L)
        departures.append(float(np.max(np.abs(d))))
        residuals.append(float(np.max(np.abs(d - p))))

    for r in _ratios(departures[1:]):
        assert r == pytest.approx(8.0, rel=0.05)  # O(x_co^3)
    for r in _ratios(residuals[1:]):
        assert r == pytest.approx(32.0, rel=0.1)  # O(x_co^5) — two orders better


def test_the_three_rungs_are_three_different_powers(ref: ReferenceParticle) -> None:
    """The ladder, stated as the single fact the three tests above share.

    Three quantities the package computes by three unrelated routes — chromaticity
    integrals, a linearised one-turn matrix, and a Newton fixed point — respond to
    the *same* octupole with three consecutive powers of the orbit. The exponents are
    fitted here rather than assumed, and asserted to be 1, 2 and 3 to within 2 %.

    A uniform mis-scaling of the kick leaves all three exponents unchanged (it is
    caught by size instead, in the next test), and no rescaling of any *individual*
    term can produce this pattern: the powers come from the structure of the Taylor
    expansion, not from its coefficients.

    The scan starts at a *smaller* steerer than the individual rungs above, because a
    single fitted exponent has nowhere to put the higher-order contamination: the
    chromatic rung reads 0.962 over a scan starting at 4e-4 and 0.990 starting at
    2e-4, converging on 1 as the orbit shrinks, which is the physics rather than a
    tolerance to be chosen.
    """
    kicks = (2e-4, 1e-4, 5e-5, 2.5e-5)

    def exponent(values: list[float]) -> float:
        """Least-squares slope of log(value) against log(kick), for halved kicks."""
        return float(np.polyfit(np.log([1.0, 0.5, 0.25, 0.125]), np.log(values), 1)[0])

    chroma = [abs(_chromaticity_feeddown(_dispersive(ref, K3L_CHROMA, k))[0]) for k in kicks]
    base = _fractional_tunes(_flat(ref).one_turn_map()[0])[0]
    tune = []
    for k in kicks:
        lat = _flat(ref, k3l=K3L_TUNE, kick_x=k)
        q = _fractional_tunes(linearised_one_turn_map(lat, closed_orbit_nonlinear(lat)))[0]
        tune.append(abs(q - base))
    orbit = [float(np.max(np.abs(_orbit_departure(ref, k, 0.0, K3L)[0]))) for k in kicks]

    assert exponent(chroma) == pytest.approx(1.0, abs=0.02)
    assert exponent(tune) == pytest.approx(2.0, abs=0.02)
    assert exponent(orbit) == pytest.approx(3.0, abs=0.02)


def test_a_mis_scaled_octupole_is_caught_as_a_factor_of_six(ref: ReferenceParticle) -> None:
    r"""The half the ladder is blind to: the size.

    :class:`_Misscaled` uses ``1`` in place of ``1/6`` in *both* components, so it
    remains curl-free, exactly symplectic at every amplitude, and identical to a real
    octupole in every structural respect — J2's lesson, and the reason the powers
    above cannot catch it. Every one of the three rungs is linear in ``k3l``, so all
    three miss by the same clean factor of **6**, and the prediction built from the
    *declared* ``k3l`` sees exactly that.

    The orbit rung is run at the weaker ``K3L_TUNE`` and a smaller steerer than
    :func:`test_the_orbit_itself_departs_at_third_order` uses: a six-times-too-strong
    octupole is six times further from the first-order regime, so at the orbit gate's
    own settings the ratio reads 6.6 — the excess being the mis-scaled element's own
    higher-order terms, not a failure of the factor.
    """

    class _Misscaled(ThinOctupole):
        def _track_body(self, state: np.ndarray, ref_: ReferenceParticle) -> np.ndarray:
            out = np.array(state, dtype=float, copy=True)
            x, y = out[X], out[Y]
            out[PX] -= self.k3l * (x**3 - 3.0 * x * y**2)
            out[PY] += self.k3l * (3.0 * x**2 * y - y**3)
            return out

    # Rung three (the orbit), which is where the whole kick enters undivided.
    kick = 1e-4
    x_lin = closed_orbit(_flat(ref, kick_x=kick))
    bad = Lattice(
        [Corrector(kick_x=kick, name="steerer"), _Misscaled(K3L_TUNE, name="oc"), *_ring()], ref
    )
    dep_bad = closed_orbit_nonlinear(bad) - x_lin
    _, pred = _orbit_departure(ref, kick, 0.0, K3L_TUNE)
    assert np.max(np.abs(dep_bad)) / np.max(np.abs(pred)) == pytest.approx(6.0, rel=0.02)

    # Rung one (the chromaticity), reached through a completely different code path:
    # the derived split feeds a sextupole k2l_eff = k3l x_co into the integrals, while
    # the tracked route feels the real kick. A mis-scaled octupole makes the second
    # six times the first.
    #
    # Weaker again (k3l = 3e2 against the gate's own 1e4), for the same reason: the
    # ratio converges on 6 from below as the machine returns to first order --
    # measured 5.43, 5.81, 5.93, 5.99 at k3l = 1e4, 3e3, 1e3, 3e2. Reading 5.43 as
    # "not a factor of 6" would be mistaking the mis-scaled element's own nonlinearity
    # for a failure of the gate.
    weak, kick_c = 3.0e2, 1e-4
    good = _chromaticity_feeddown(_dispersive(ref, weak, kick_c))[0]
    assert abs(good) > 0.1  # non-vacuous
    lat_bad = Lattice(
        [
            _Misscaled(elem.k3l, name=elem.name) if isinstance(elem, ThinOctupole) else elem
            for elem in _dispersive(ref, weak, kick_c).elements
        ],
        ref,
    )
    h = 1e-5
    tracked_bad = (tunes_on_orbit(lat_bad, delta=+h)[0] - tunes_on_orbit(lat_bad, delta=-h)[0]) / (
        2.0 * h
    )
    assert tracked_bad / good == pytest.approx(6.0, rel=0.01)


# --------------------------------------------------------------------------
# 4. The vertical plane — where the octupole and the sextupole part company
# --------------------------------------------------------------------------


def test_both_planes_are_exact_invariant_subspaces(ref: ReferenceParticle) -> None:
    r"""``x = px = 0`` is invariant for an octupole and **not** for a sextupole.

    The octupole's kick is odd in *both* coordinates — ``Delta px`` carries
    ``x(x^2 - 3y^2)`` and ``Delta py`` carries ``y(3x^2 - y^2)`` — so a beam confined
    to either plane stays there exactly. A sextupole's ``Delta px = -1/2 k2l (x^2 -
    y^2)`` is *even* in ``y``, so a purely vertical orbit drives a horizontal one.

    This is exact arithmetic rather than a small number, so both statements are
    asserted at exact zero. It is also the sharpest single distinction between the
    two elements' feed-down anywhere in the suite: with a vertical bump alone the
    sextupole steers the beam horizontally (I2 gates that) and the octupole does not.
    """
    vertical = _flat(ref, k3l=K3L, kick_y=KICK)
    co = closed_orbit_nonlinear(vertical)
    assert abs(co[Y]) > 1e-5  # there really is a vertical orbit
    assert np.array_equal(co[[X, PX]], np.zeros(2))
    for point in propagate_orbit_nonlinear(vertical, co):
        assert np.array_equal(point[[X, PX]], np.zeros(2))

    horizontal = _flat(ref, k3l=K3L, kick_x=KICK)
    co_h = closed_orbit_nonlinear(horizontal)
    assert abs(co_h[X]) > 1e-5
    assert np.array_equal(co_h[[Y, PY]], np.zeros(2))

    # ...and the contrast: the same vertical bump through a *sextupole* does move x.
    sext = Lattice(
        [Corrector(kick_y=KICK, name="steerer"), ThinSextupole(20.0, name="sx"), *_ring()], ref
    )
    assert abs(closed_orbit_nonlinear(sext)[X]) > 1e-9


def test_a_vertical_bump_flips_the_sign_of_the_fed_down_gradient(
    ref: ReferenceParticle,
) -> None:
    r"""``k1l_eff = +1/2 k3l (x_co^2 - y_co^2)``: the ``x^2 - y^2`` structure, in the tunes.

    A horizontal bump makes the octupole focus horizontally; a vertical bump of the
    same size makes it *defocus* horizontally, because the gradient carries the
    difference of squares. The two tune shifts are therefore equal and opposite for
    equal offsets — a statement no magnitude comparison at a single bump could make,
    and one that pins the relative sign of the two terms in ``k1l_eff``.
    """
    base = _fractional_tunes(_flat(ref).one_turn_map()[0])

    hor = _flat(ref, k3l=K3L_TUNE, kick_x=KICK)
    ver = _flat(ref, k3l=K3L_TUNE, kick_y=KICK)
    q_h = _fractional_tunes(linearised_one_turn_map(hor, closed_orbit_nonlinear(hor)))
    q_v = _fractional_tunes(linearised_one_turn_map(ver, closed_orbit_nonlinear(ver)))

    dq_h = q_h[0] - base[0]
    dq_v = q_v[0] - base[0]
    assert abs(dq_h) > 1e-5 and abs(dq_v) > 1e-5
    assert dq_h * dq_v < 0.0  # opposite senses, which is the x^2 - y^2 structure

    # Equal and opposite once the different beta at the source is divided out: the
    # ring is not round, so the two bumps of equal kick give different offsets.
    x_co = float(propagate_orbit_nonlinear(hor)[1][X])
    y_co = float(propagate_orbit_nonlinear(ver)[1][Y])
    assert dq_h / x_co**2 == pytest.approx(-dq_v / y_co**2, rel=0.02)


def test_a_vertical_bump_couples_the_planes_through_the_skew_pair(
    ref: ReferenceParticle,
) -> None:
    r"""Coupling needs **both** planes here, where the sextupole needed only one.

    The octupole's skew quadrupole is ``k1sl_eff = k3l x_co y_co`` — a product, so it
    vanishes unless the orbit is off-axis in both planes at once. A sextupole's is
    ``k2l y_co``, live with a vertical bump alone. Both statements are asserted, and
    the scaling ``1 - gamma_c ~ |C|^2 ~ (x_co y_co)^2`` is checked because
    ``gamma_c`` is even in the coupling and cannot be pinned by magnitude alone.

    The skew *sextupole* ``k2sl_eff = k3l y_co`` is live with a vertical bump alone,
    but has no linear part at all — so it is invisible here, exactly as its own
    element gates say. That is why the coupling below switches off when the horizontal
    bump does.
    """
    only_vertical = _flat(ref, k3l=K3L_TUNE, kick_y=KICK)
    ct_v = match_periodic_coupled(
        linearised_one_turn_map(only_vertical, closed_orbit_nonlinear(only_vertical))
    )
    assert ct_v.gamma_c == 1.0  # uncoupled, exactly: the product term is zero

    # 1 - gamma_c ~ |C|^2 ~ (x_co y_co)^2, so halving *both* bumps divides it by 16.
    coupling = []
    for kick in (KICK, 0.5 * KICK, 0.25 * KICK):
        lat = _flat(ref, k3l=K3L_TUNE, kick_x=kick, kick_y=kick)
        ct = match_periodic_coupled(linearised_one_turn_map(lat, closed_orbit_nonlinear(lat)))
        coupling.append(1.0 - ct.gamma_c)
    assert coupling[0] > 1e-9  # the planes really are coupled
    for r in _ratios(coupling):
        assert r == pytest.approx(16.0, rel=0.02)


# --------------------------------------------------------------------------
# 5. Scope lines, enforced rather than documented
# --------------------------------------------------------------------------


def test_a_thick_octupole_is_still_refused(ref: ReferenceParticle) -> None:
    """The ``O(L^2)`` line, unchanged from the thick sextupole's.

    A thick body's orbit varies across the magnet, so collapsing it onto a single
    split evaluated at the entrance orbit would carry an ``O(L^2)`` error — the same
    reason :func:`linearised_lattice` refuses a thick :class:`Sextupole`, and the
    reason every quantitative gate above uses thin octupoles.
    :func:`linearised_element_maps` handles it, because it differentiates ``track()``.
    """
    lat = Lattice([Corrector(kick_x=KICK), Octupole(0.2, K3L / 0.2, name="oc"), *_ring()], ref)
    with pytest.raises(NotImplementedError, match="thick Octupole"):
        linearised_lattice(lat)
    with pytest.raises(NotImplementedError, match="thick Octupole"):
        chromaticity_on_orbit(lat)

    # The numerical route works, and sees the same fed-down gradient.
    maps = linearised_element_maps(lat)
    assert not np.allclose(maps[1], lat.elements[1].matrix(ref), atol=1e-12)

    # A zero-strength thick octupole is a drift and passes through unchanged.
    zero = Lattice([Corrector(kick_x=KICK), Octupole(0.2, 0.0), *_ring()], ref)
    assert len(linearised_lattice(zero).elements) == len(zero.elements)


def test_the_thick_octupole_still_closes_an_orbit_and_feeds_down(
    ref: ReferenceParticle,
) -> None:
    """Everything that differentiates ``track()`` works on a thick octupole.

    Newton, the per-element maps and the tune therefore all work; only the derived
    split refuses. This exercises drift-kick-drift with several slices and a non-zero
    length between the kick and the orbit that produced it, which nothing else here
    covers — and the departure is still ``O(x_co^3)``, so it is the same physics with
    a different integrator.
    """
    departures = []
    for kick in (4e-4, 2e-4, 1e-4):
        lat = Lattice([Corrector(kick_x=kick), Octupole(0.2, K3L / 0.2, n_slices=4), *_ring()], ref)
        flat_ring = Lattice([Corrector(kick_x=kick), Octupole(0.2, 0.0), *_ring()], ref)
        departures.append(
            float(np.max(np.abs(closed_orbit_nonlinear(lat) - closed_orbit(flat_ring))))
        )
    assert departures[0] > 1e-8
    for r in _ratios(departures):
        assert r == pytest.approx(8.0, rel=0.1)


def test_chromaticity_on_the_design_orbit_still_ignores_octupoles(
    ref: ReferenceParticle,
) -> None:
    """J2's scope statement survives J3 intact, and the two are not in tension.

    :func:`~accsim.twiss.chromaticity` is a **design-orbit** quantity: it walks each
    element's on-axis ``matrix()``, where an octupole is a drift and contributes to
    ``Q'`` exactly nothing. That remains right — feed-down is a property of the
    *orbit*, not of the element — and it is what makes
    :func:`~accsim.twiss.chromaticity_on_orbit` a different number rather than a
    correction to a wrong one.
    """
    steered = _dispersive(ref, K3L_CHROMA, 2e-4)
    on_axis = _dispersive(ref, K3L_CHROMA, 0.0)
    octupole_free = _dispersive(ref, 0.0, 2e-4)

    # Bit-identical: no orbit enters the design-orbit calculation at all.
    assert chromaticity(steered) == chromaticity(on_axis) == chromaticity(octupole_free)

    # ...while the real machine's chromaticity has moved by more than 100 %.
    assert abs(chromaticity_on_orbit(steered)[0] / chromaticity(steered)[0]) > 2.0
