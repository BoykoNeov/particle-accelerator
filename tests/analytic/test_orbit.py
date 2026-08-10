"""I1 acceptance (part 1): the affine element map and the closed orbit.

A dipole corrector is the one element in the package whose action is **not** a
matrix. A kick of the same angle for every particle is inhomogeneous, so the
element map becomes affine, ``x -> M x + k``, and a lattice composes as

    x -> M2 (M1 x + k1) + k2 = (M2 M1) x + (M2 k1 + k2)

— a kick is transported by everything *downstream* of it. That transport is the
whole content of :meth:`accsim.Lattice.transfer_map`, and it is gated here
against element-by-element :meth:`Element.track` with **two** correctors at
different places: with a single kick only one term exists, so a composition with
the transport applied on the wrong side still produces a perfectly plausible
closed orbit. Two kicks is the smallest configuration that can see the error.

The closed orbit then follows from the same fixed-point condition the matched
dispersion already uses, ``x_co = M4 x_co + k4`` -> ``(I - M4) x_co = k4``. The
classic single-kick closed form

    x_co(s) = theta * sqrt(beta_k beta(s)) / (2 sin(pi Q)) * cos(|dpsi(s)| - pi Q)

is **not** how it is computed; it is derived symbolically here and checked
*against* the exact solve, never the other way round.
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
    Particle,
    ReferenceParticle,
    ThinQuadrupole,
    Tracker,
    closed_orbit,
    closed_twiss,
    propagate_orbit,
    propagate_twiss,
    tunes,
)

# Thin FODO cell, as in the matching suite: half-F | drift | D | drift | half-F.
VF = 1.0 / 1.5  # full-quad inverse focal length, F family [m^-1]
VD = 1.0 / 1.6  # ditto, D family [m^-1]
L_HALF = 1.0  # half-cell drift [m]


@pytest.fixture
def ref() -> ReferenceParticle:
    # Thin quads + drifts are energy-independent; any reference works.
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _cell(*, tag: str = "") -> list:
    """One thin FODO cell, F-centred."""
    return [
        ThinQuadrupole(0.5 * VF, name=f"qf_a{tag}"),
        Drift(L_HALF, name=f"d1{tag}"),
        ThinQuadrupole(-VD, name=f"qd{tag}"),
        Drift(L_HALF, name=f"d2{tag}"),
        ThinQuadrupole(0.5 * VF, name=f"qf_b{tag}"),
    ]


def _ring(n_cells: int = 6) -> list:
    """``n_cells`` FODO cells in series — a closed ring for the orbit tests."""
    return [e for i in range(n_cells) for e in _cell(tag=f"_{i}")]


# ---------------------------------------------------------------------------
# The affine map: composition, and its agreement with element-by-element tracking
# ---------------------------------------------------------------------------


def test_kick_is_zero_for_every_ordinary_element(ref: ReferenceParticle) -> None:
    """Every element but a corrector has a strictly zero constant part."""
    for elem in _ring(2):
        assert np.array_equal(elem.kick(ref), np.zeros(6))


def test_transfer_map_reduces_to_transfer_matrix_without_correctors(
    ref: ReferenceParticle,
) -> None:
    """No corrector -> the affine map *is* the old matrix, with k exactly zero."""
    lat = Lattice(_ring(), ref)
    M, k = lat.transfer_map()
    assert np.array_equal(M, lat.transfer_matrix())
    assert np.array_equal(k, np.zeros(6))


def test_two_corrector_composition_matches_element_by_element_tracking(
    ref: ReferenceParticle,
) -> None:
    """The composed affine map equals tracking, with **two** kicks at different places.

    This is the gate on the composition order. ``M2 k1 + k2`` transports the
    upstream kick through the downstream matrices; writing it the other way
    (``k1 + M1 k2``, or forgetting the transport entirely) leaves a map that is
    still exact for a *single* corrector, so one kick cannot see the bug.
    """
    elems = _ring()
    elems.insert(3, Corrector(kick_x=1.3e-4, kick_y=-4.1e-5, name="c1"))
    elems.insert(19, Corrector(kick_x=-7.7e-5, kick_y=2.2e-4, name="c2"))
    lat = Lattice(elems, ref)

    M, k = lat.transfer_map()
    # A general (non-zero) input state, so the homogeneous and constant parts are
    # both exercised and cannot cancel.
    x0 = np.array([1e-3, 2e-4, -5e-4, 3e-4, 1e-2, 1e-3])
    tracked = x0.copy()
    for elem in lat.elements:
        tracked = elem.track(tracked, ref)

    assert np.allclose(M @ x0 + k, tracked, rtol=0, atol=1e-15)
    # ... and the constant part alone is what a zero input maps to.
    zero_in = np.zeros(6)
    for elem in lat.elements:
        zero_in = elem.track(zero_in, ref)
    assert np.allclose(k, zero_in, rtol=0, atol=1e-18)


def test_composition_order_is_detectable_by_the_two_corrector_test(
    ref: ReferenceParticle,
) -> None:
    """Measure the claim above: the wrong composition really does differ here.

    Guards the previous test against vacuity. If the two placements happened to
    make the transported and untransported sums equal, the composition gate would
    pass on a broken implementation, so the difference is asserted to be large.
    """
    elems = _ring()
    elems.insert(3, Corrector(kick_x=1.3e-4, kick_y=-4.1e-5))
    elems.insert(19, Corrector(kick_x=-7.7e-5, kick_y=2.2e-4))
    lat = Lattice(elems, ref)
    _, k = lat.transfer_map()

    # The "obvious but wrong" version: sum the raw kicks with no transport.
    k_untransported = sum((e.kick(ref) for e in lat.elements), np.zeros(6))
    assert np.linalg.norm(k - k_untransported) > 1e-4  # cf. |k| ~ 1e-4


def test_linear_and_nonlinear_tracking_agree_with_a_corrector(ref: ReferenceParticle) -> None:
    """``Tracker.track`` keeps its promise that the two paths agree, kicks included.

    The linear path uses the accumulated map and the nonlinear path walks the
    elements; before the affine map existed the former would have silently
    dropped every corrector kick.
    """
    elems = _ring()
    elems.insert(7, Corrector(kick_x=2.5e-4, kick_y=1.1e-4))
    tracker = Tracker(Lattice(elems, ref))
    p = Particle(x=1e-3, px=-2e-4, y=4e-4, py=1e-4, zeta=2e-3, delta=5e-4)

    lin = tracker.track(p, nonlinear=False).state
    nonlin = tracker.track(p, nonlinear=True).state
    assert np.allclose(lin, nonlin, rtol=0, atol=1e-15)
    # And the kick actually did something, so the agreement is not trivial.
    no_corr = Tracker(Lattice(_ring(), ref)).track(p, nonlinear=False).state
    assert abs(lin[1] - no_corr[1]) > 1e-5


def test_corrector_matrix_is_the_identity(ref: ReferenceParticle) -> None:
    """A constant kick has no linear part — the optics are untouched by it."""
    c = Corrector(kick_x=1e-3, kick_y=-2e-3)
    assert np.array_equal(c.matrix(ref), np.eye(6))
    assert c.length == 0.0


def test_corrector_kick_broadcasts_over_a_bunch(ref: ReferenceParticle) -> None:
    """``track`` accepts a ``(6, n)`` bunch; the same kick applies to every particle."""
    c = Corrector(kick_x=3e-4, kick_y=-1e-4)
    states = np.zeros((6, 5))
    states[0] = np.linspace(-1e-3, 1e-3, 5)
    out = c.track(states, ref)
    assert np.allclose(out[1], 3e-4)
    assert np.allclose(out[3], -1e-4)
    assert np.allclose(out[0], states[0])  # positions untouched by a thin kick


def test_corrector_does_not_move_the_optics(ref: ReferenceParticle) -> None:
    """beta / alpha / tunes are identical with and without correctors present.

    The physical claim behind ``matrix() = I``: dipoles steer the orbit,
    quadrupoles set the optics. Asserted on the one-turn *matrix* itself, so it
    covers every optics quantity derived from it at once.
    """
    plain = Lattice(_ring(), ref)
    elems = _ring()
    elems.insert(3, Corrector(kick_x=1e-3))
    elems.insert(19, Corrector(kick_y=-2e-3))
    steered = Lattice(elems, ref)
    assert np.array_equal(plain.one_turn_matrix(), steered.one_turn_matrix())


def test_zero_kick_corrector_is_a_no_op(ref: ReferenceParticle) -> None:
    """A corrector set to zero is exactly a drift of zero length."""
    elems = _ring()
    elems.insert(11, Corrector())
    M, k = Lattice(elems, ref).transfer_map()
    assert np.array_equal(M, Lattice(_ring(), ref).transfer_matrix())
    assert np.array_equal(k, np.zeros(6))


def test_same_kick_at_two_places_gives_different_transported_constants(
    ref: ReferenceParticle,
) -> None:
    """The transport is position-dependent — the physics the composition encodes.

    Identical correctors at two different places produce different accumulated
    constants, because each is carried by a different downstream map.
    """
    angle = 5e-4
    a = _ring()
    a.insert(2, Corrector(kick_x=angle))
    b = _ring()
    b.insert(17, Corrector(kick_x=angle))
    _, ka = Lattice(a, ref).transfer_map()
    _, kb = Lattice(b, ref).transfer_map()
    assert not np.allclose(ka, kb, rtol=0, atol=1e-8)
    # Both carry the same total deflection into px only via transport, but the
    # position they arrive with differs — that is the closed orbit's dependence
    # on where the kick sits.
    assert abs(ka[0] - kb[0]) > 1e-5


def test_kick_scales_linearly_with_the_angle(ref: ReferenceParticle) -> None:
    """The accumulated constant is exactly proportional to the kick angle.

    Affine, not merely smooth: this is what makes the response matrix exact and
    the correction a single linear solve rather than an iteration.
    """
    elems = _ring()
    c = Corrector(kick_x=1e-4, kick_y=-3e-5)
    elems.insert(8, c)
    lat = Lattice(elems, ref)
    _, k1 = lat.transfer_map()
    c.kick_x, c.kick_y = 3.7e-4, -1.11e-4  # 3.7x the original, both planes
    _, k2 = lat.transfer_map()
    assert np.allclose(k2, 3.7 * k1, rtol=1e-14, atol=0)


def test_kicks_superpose(ref: ReferenceParticle) -> None:
    """Two correctors together give exactly the sum of each one alone."""
    ka_only = _ring()
    ka_only.insert(3, Corrector(kick_x=1.3e-4))
    kb_only = _ring()
    kb_only.insert(19, Corrector(kick_y=2.2e-4))
    both = _ring()
    both.insert(3, Corrector(kick_x=1.3e-4))
    both.insert(19 + 1, Corrector(kick_y=2.2e-4))  # +1: the first insert shifted it

    _, k_a = Lattice(ka_only, ref).transfer_map()
    _, k_b = Lattice(kb_only, ref).transfer_map()
    _, k_ab = Lattice(both, ref).transfer_map()
    assert np.allclose(k_ab, k_a + k_b, rtol=0, atol=1e-18)


# ---------------------------------------------------------------------------
# The closed orbit: the fixed point, and the single-kick closed form
# ---------------------------------------------------------------------------

KICK = 2.5e-4  # a representative steerer angle [rad]


def _kicked_ring(kick_x: float = KICK, kick_y: float = 0.0, n_cells: int = 6) -> list:
    """A FODO ring with **one** corrector as its last element.

    Placing it last makes the lattice entrance the point *just after* the kick,
    which is where the textbook fixed point ``x+ = (I - M)^-1 kappa`` is written,
    and makes the phase that ``propagate_twiss`` accumulates from the entrance
    exactly the downstream phase ``dpsi`` from the kick — so the closed form below
    needs no phase bookkeeping of its own.
    """
    return [*_ring(n_cells), Corrector(kick_x=kick_x, kick_y=kick_y, name="steerer")]


def _symbolic_fixed_point() -> tuple:
    """Derive ``(x+, px+)`` at a single kick from ``x = M x + kappa``, in sympy.

    Nothing is recalled: the Courant-Snyder one-turn block is written down and the
    2x2 linear system is solved. Returns ``(x, px, symbols)`` as sympy expressions
    in ``(beta, alpha, mu, theta)``.
    """
    sp = pytest.importorskip("sympy")
    beta, alpha, mu, theta = sp.symbols("beta alpha mu theta", real=True, positive=False)
    gamma = (1 + alpha**2) / beta
    M = sp.Matrix(
        [
            [sp.cos(mu) + alpha * sp.sin(mu), beta * sp.sin(mu)],
            [-gamma * sp.sin(mu), sp.cos(mu) - alpha * sp.sin(mu)],
        ]
    )
    sol = sp.simplify((sp.eye(2) - M).inv() * sp.Matrix([0, theta]))
    return sol[0], sol[1], (beta, alpha, mu, theta)


def test_symbolic_fixed_point_is_the_cot_form(ref: ReferenceParticle) -> None:
    """``x+ = (theta beta / 2) cot(mu/2)`` and ``px+ = (theta/2)(1 - alpha cot(mu/2))``.

    Derived, not recalled — and the derivation is what fixes the ``2`` and the
    half-angle. The ``1/sin(pi Q)`` divergence of the usual textbook form is this
    ``cot(mu/2)`` at ``mu = 2 pi Q``.
    """
    sp = pytest.importorskip("sympy")
    x_expr, px_expr, (beta, alpha, mu, theta) = _symbolic_fixed_point()
    cot = sp.cos(mu / 2) / sp.sin(mu / 2)
    assert sp.simplify(x_expr - theta * beta * cot / 2) == 0
    assert sp.simplify(px_expr - theta * (1 - alpha * cot) / 2) == 0


def test_closed_orbit_at_the_kick_matches_the_symbolic_fixed_point(
    ref: ReferenceParticle,
) -> None:
    """The exact ``(I - M4)^-1 k4`` solve reproduces the derived ``(x+, px+)``.

    The module never evaluates ``cot(pi Q)``; it solves the 4D fixed point. This
    is the point of contact between the two.
    """
    sp = pytest.importorskip("sympy")
    lat = Lattice(_kicked_ring(), ref)
    tw = closed_twiss(lat)
    qx, _ = tunes(lat)

    x_expr, px_expr, syms = _symbolic_fixed_point()
    subs = dict(zip(syms, (tw.beta_x, tw.alpha_x, 2 * sp.pi * qx, KICK), strict=True))
    want = np.array([float(x_expr.subs(subs)), float(px_expr.subs(subs))])

    co = closed_orbit(lat)
    assert np.allclose(co[:2], want, rtol=1e-12, atol=0)
    assert np.allclose(co[2:], 0.0, atol=1e-18)  # no vertical kick -> no vertical orbit


def _closed_form_orbit(lat: Lattice, kick: float) -> np.ndarray:
    """Textbook single-kick orbit at every boundary of :func:`_kicked_ring`.

        x(s) = theta sqrt(beta_k beta(s)) / (2 sin(pi Q)) * cos(dpsi(s) - pi Q)

    with ``dpsi`` accumulated **downstream** from the kick (which is why the
    corrector is the last element). Used only as an independent reference.
    """
    tw0 = closed_twiss(lat)
    qx, _ = tunes(lat)
    table = propagate_twiss(lat, tw0)
    amp = kick * math.sqrt(tw0.beta_x) / (2.0 * math.sin(math.pi * qx))
    return np.array([amp * math.sqrt(t.beta_x) * math.cos(t.mu_x - math.pi * qx) for t in table])


def test_orbit_everywhere_matches_the_textbook_single_kick_form(
    ref: ReferenceParticle,
) -> None:
    """The exact solve reproduces ``sqrt(beta_k beta)/(2 sin pi Q) cos(dpsi - pi Q)``.

    Checked at **every** boundary, so the ``sqrt(beta)`` scaling, the ``2 sin(pi Q)``
    denominator and the ``-pi Q`` phase offset are all pinned at once; a wrong
    offset would still fit at the kick point alone.
    """
    lat = Lattice(_kicked_ring(), ref)
    table = propagate_orbit(lat)
    got = np.array([o[0] for o in table])
    want = _closed_form_orbit(lat, KICK)
    assert np.allclose(got, want, rtol=1e-11, atol=0)
    # Non-trivial: the orbit swings through both signs and is not tiny.
    assert got.max() > 1e-4 and got.min() < -1e-4


def test_closed_orbit_actually_closes(ref: ReferenceParticle) -> None:
    """Track the closed orbit one turn and land back on it — the defining property.

    An independent code path: element-by-element :meth:`Element.track`, not the
    accumulated map the solve uses.
    """
    lat = Lattice(_kicked_ring(kick_x=KICK, kick_y=-1.7e-4), ref)
    co = closed_orbit(lat)
    state = np.array([co[0], co[1], co[2], co[3], 0.0, 0.0])
    for elem in lat.elements:
        state = elem.track(state, ref)
    assert np.allclose(state[:4], co, rtol=0, atol=1e-16)


def test_propagate_orbit_closes_on_itself(ref: ReferenceParticle) -> None:
    """The last boundary of a ring equals the first."""
    lat = Lattice(_kicked_ring(kick_x=KICK, kick_y=3.3e-4), ref)
    table = propagate_orbit(lat)
    assert len(table) == len(lat) + 1
    assert np.allclose(table[-1], table[0], rtol=0, atol=1e-16)


def test_no_corrector_gives_exactly_zero_orbit(ref: ReferenceParticle) -> None:
    """A perfect machine sits on the design orbit — exactly zero, not nearly."""
    lat = Lattice(_ring(), ref)
    assert np.array_equal(closed_orbit(lat), np.zeros(4))


def test_vertical_kick_moves_only_the_vertical_plane(ref: ReferenceParticle) -> None:
    """The planes stay decoupled, and the vertical orbit obeys the same cot form.

    ``x`` stays exactly on axis (an uncoupled lattice), while ``y`` takes the
    value ``theta beta_y cot(pi Q_y) / 2`` — the y plane is not a special case,
    it just has its own beta and tune.
    """
    lat = Lattice(_kicked_ring(kick_x=0.0, kick_y=KICK), ref)
    co = closed_orbit(lat)
    assert np.allclose(co[:2], 0.0, atol=1e-18)

    tw = closed_twiss(lat)
    _, qy = tunes(lat)
    want = KICK * tw.beta_y / (2.0 * math.tan(math.pi * qy))
    assert math.isclose(co[2], want, rel_tol=1e-12)
    assert abs(co[2]) > 5e-5  # measured 5.86e-5 — a real displacement, not round-off


def test_closed_orbit_is_exactly_linear_in_the_kicks(ref: ReferenceParticle) -> None:
    """Superposition — the fact that makes the response matrix exact.

    Two correctors, scaled and combined: the closed orbit of the sum is the sum of
    the closed orbits, to machine precision. Correction is therefore a single
    linear solve, not an iteration.
    """

    def orbit_of(kx: float, ky: float) -> np.ndarray:
        elems = _ring()
        elems.insert(4, Corrector(kick_x=kx))
        elems.append(Corrector(kick_y=ky))
        return closed_orbit(Lattice(elems, ref))

    a, b = 1.9e-4, -6.1e-4
    combined = orbit_of(a, b)
    separate = orbit_of(a, 0.0) + orbit_of(0.0, b)
    assert np.allclose(combined, separate, rtol=0, atol=1e-18)
    # Scaling too, with a non-round factor.
    assert np.allclose(orbit_of(2.5 * a, 2.5 * b), 2.5 * combined, rtol=1e-13, atol=0)


def test_unfocused_ring_has_no_closed_orbit(ref: ReferenceParticle) -> None:
    """A ring with no focusing: the kick never comes back, so no orbit closes.

    ``I - M4`` is exactly singular there (zero tune), which is the integer
    resonance in its starkest form. The solve must say so rather than return a
    plausible number.
    """
    lat = Lattice([Drift(2.0), Corrector(kick_x=KICK)], ref)
    with pytest.raises(ClosedOrbitError, match="integer"):
        closed_orbit(lat)


def _weakened_ring(scale: float, kick_y: float = KICK) -> list:
    """The FODO ring with every gradient scaled — a knob that walks ``Q_y`` to zero.

    Weakening the focusing drives the *vertical* tune to the integer first (it is
    the weaker plane in this cell), reaching exactly ``Q_y = 0`` at ``scale =
    0.2``. That gives a physical approach to the resonance in a real focusing
    lattice, rather than the degenerate no-focusing case.
    """
    cells = [
        e
        for _ in range(6)
        for e in (
            ThinQuadrupole(0.5 * scale * VF),
            Drift(L_HALF),
            ThinQuadrupole(-scale * VD),
            Drift(L_HALF),
            ThinQuadrupole(0.5 * scale * VF),
        )
    ]
    return [*cells, Corrector(kick_y=kick_y)]


def test_orbit_blows_up_approaching_the_integer_resonance(ref: ReferenceParticle) -> None:
    """The ``cot(pi Q)`` divergence, measured on a lattice walked toward ``Q_y = 0``.

    At ``scale = 1`` the vertical tune is 0.559 and a 0.25 mrad steerer moves the
    orbit by 59 um; at ``scale = 0.205`` the tune is 0.0197 and the *same* steerer
    moves it by 18 cm — a factor of 3100 from the tune alone. The closed form is
    checked at the near-resonant end too, where it is most stressed and where any
    error in the denominator would be enormous rather than subtle.
    """
    amps = []
    for scale in (1.0, 0.205):
        lat = Lattice(_weakened_ring(scale), ref)
        _, qy = tunes(lat)
        y = closed_orbit(lat)[2]
        want = KICK * closed_twiss(lat).beta_y / (2.0 * math.tan(math.pi * qy))
        assert math.isclose(y, want, rel_tol=1e-11)
        amps.append(abs(y))
    assert amps[1] / amps[0] > 1000.0  # measured 3112


def test_exact_integer_tune_has_no_closed_orbit(ref: ReferenceParticle) -> None:
    """At ``Q_y = 0`` exactly, a focusing lattice still has no closed orbit.

    ``scale = 0.2`` puts the vertical plane exactly on the integer (``|1/2 Tr| =
    1``): the kick repeats in phase every turn and the excursion grows without
    bound. The solve must refuse rather than return a very large number.
    """
    lat = Lattice(_weakened_ring(0.2), ref)
    with pytest.raises(ClosedOrbitError, match="integer"):
        closed_orbit(lat)


def test_propagate_orbit_follows_a_trajectory_when_given_a_start(
    ref: ReferenceParticle,
) -> None:
    """With an explicit ``orbit0`` the walk is a trajectory, not a closed orbit.

    Started away from the closed orbit, the last point differs from the first —
    the transfer-line branch, and the check that closure is a property of the
    *start*, not of the propagation.
    """
    lat = Lattice(_kicked_ring(), ref)
    start = closed_orbit(lat) + np.array([1e-3, 0.0, 0.0, 0.0])
    table = propagate_orbit(lat, start)
    assert np.allclose(table[0], start, rtol=0, atol=0)
    assert not np.allclose(table[-1], table[0], rtol=0, atol=1e-9)
    # The deviation from the closed orbit is transported by the *homogeneous* map.
    M, _ = lat.transfer_map()
    m4 = M[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])]
    assert np.allclose(table[-1] - closed_orbit(lat), m4 @ (start - closed_orbit(lat)), atol=1e-15)


def test_propagate_orbit_rejects_a_wrong_shaped_start(ref: ReferenceParticle) -> None:
    """A 6D state is not a 4D orbit; the mistake is caught, not broadcast."""
    lat = Lattice(_kicked_ring(), ref)
    with pytest.raises(ValueError, match="length-4"):
        propagate_orbit(lat, np.zeros(6))
