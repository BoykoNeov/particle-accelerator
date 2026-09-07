"""Analytic checks for the path-length series ``C(delta)/C`` — momentum compaction
beyond first order (U1).

``momentum_compaction`` reports one number, the slope at ``delta = 0``. The orbit is not
straight in ``delta`` and neither is the path length along it, so the honest object is

    C(delta) / C = 1 + alpha_0 delta + alpha_1 delta^2 + alpha_2 delta^3 + ...

and :func:`~accsim.twiss.momentum_compaction_series` returns its coefficients.

What gates it, in the order the arms below are written:

1. **An identity that lives outside every code involved.** A ring that is nothing but a
   full circle of sector bends has a closed orbit at momentum ``delta`` that is *exactly*
   the concentric circle of radius ``rho (1 + delta)`` — same centre, so ``x = rho delta``
   with no higher-order part — and its length is therefore ``2 pi rho (1 + delta)`` with
   no higher-order part either. ``alpha_0 = 1`` and **every** further coefficient is
   exactly zero, at every ``delta``, for every ``rho``. Nothing is fitted and no
   coefficient is recalled; a package that expands in the wrong variable cannot pass it.
2. **Two disjoint routes.** The ``"map"`` route composes per-element second-order Taylor
   maps and solves two linear systems; the ``"tracked"`` route runs the closed-orbit
   Newton solver at four momenta and differences the result. They share ``Element.track``
   and the definition of ``zeta``, and nothing else.
3. **The convention, pinned here rather than only against PTC.** The *local* compaction
   ``d(C/C_0)/ddelta`` is ``alpha_0 + 2 alpha_1 delta``, so the slope a code reports at
   finite momentum is **twice** ``alpha_1``. That factor is where this milestone can be
   silently wrong, so it is checked internally and not only in ``tests/reference/``.
4. **The velocity factor, re-derived with sympy.** ``beta/beta_0`` carries the series into
   the path length, and its second-order coefficient ``-3 beta_0^2 / (2 gamma_0^2)`` is a
   hard-coded constant in the source. Constants in this project get derived.

The straight-lattice arm cannot test a *sign* (both sides vanish), which is why the circle
and the sextupole-sensitivity arms are here: they are the ones with a number in them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    Sextupole,
    closed_orbit_path_length,
    momentum_compaction,
    momentum_compaction_series,
    slip_factor,
    transition_gamma,
)
from accsim.reference import PROTON_MASS_EV
from accsim.tracking import Particle, Tracker
from accsim.twiss import _velocity_coefficients

RHO = 10.0
"""Bending radius of the exactly-solvable ring [m]."""


def _ref(energy_eV: float = 10e9) -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(PROTON_MASS_EV, energy_eV, charge=1.0)


def _probe_ring(k2: float = 0.5, energy_eV: float = 10e9) -> Lattice:
    """Four FODO cells with thick bends and sextupoles — stable in both planes."""
    ref = _ref(energy_eV)
    angle = 2.0 * math.pi / 8.0
    elements = []
    for _ in range(4):
        elements += [
            Quadrupole(0.5, 0.30),
            Drift(0.6),
            Sextupole(0.2, k2),
            Dipole(1.5, angle),
            Drift(0.6),
            Quadrupole(0.5, -0.30),
            Drift(0.6),
            Sextupole(0.2, -k2),
            Dipole(1.5, angle),
            Drift(0.6),
        ]
    return Lattice(elements, ref)


def _straight_lattice(ref: ReferenceParticle) -> Lattice:
    """A FODO line with no bending — stable in both planes, and dispersion-free."""
    return Lattice([Quadrupole(0.5, 0.3), Drift(1.0), Quadrupole(0.5, -0.3), Drift(1.0)] * 4, ref)


def _circle(n_bends: int = 8) -> Lattice:
    """A ring that is one full circle of sector bends and nothing else."""
    ref = _ref()
    return Lattice(
        [Dipole(2.0 * math.pi * RHO / n_bends, 2.0 * math.pi / n_bends) for _ in range(n_bends)],
        ref,
    )


# --------------------------------------------------------------------------------------
# 4. the velocity factor, derived rather than recalled
# --------------------------------------------------------------------------------------


def test_velocity_coefficients_are_the_sympy_expansion() -> None:
    """``beta/beta_0 = 1 + delta/gamma_0^2 - 3 beta_0^2 delta^2 / (2 gamma_0^2) + ...``."""
    delta, beta0 = sp.symbols("delta beta0", positive=True)
    # delta = dp/p0, so p = p0 (1 + delta) and E/E0 = sqrt(1 + beta0^2 (2 delta + delta^2)).
    beta_over_beta0 = (1 + delta) / sp.sqrt(1 + beta0**2 * (2 * delta + delta**2))
    series = sp.expand(sp.series(beta_over_beta0, delta, 0, 3).removeO())

    b1_sym = sp.simplify(series.coeff(delta, 1))
    b2_sym = sp.simplify(series.coeff(delta, 2))
    # The closed forms the source hard-codes.
    assert sp.simplify(b1_sym - (1 - beta0**2)) == 0
    assert sp.simplify(b2_sym - (-sp.Rational(3, 2) * beta0**2 * (1 - beta0**2))) == 0

    ref = _ref()
    b1, b2 = _velocity_coefficients(ref)
    assert b1 == pytest.approx(float(b1_sym.subs(beta0, ref.beta0)), rel=1e-14)
    assert b2 == pytest.approx(float(b2_sym.subs(beta0, ref.beta0)), rel=1e-14)
    # b2 is not any power of b1 — the mistake this arm exists to catch.
    assert abs(b2) > 100.0 * b1**2


def test_zeta_slip_on_a_drift_is_the_velocity_factor() -> None:
    """The convention the whole milestone rests on: ``Delta zeta = L (1 - beta_0/beta)``."""
    ref = _ref()
    length = 3.7
    lattice = Lattice([Drift(length)], ref)
    for delta in (1e-4, 1e-3, 1e-2):
        out = Tracker(lattice).track(Particle(zeta=0.0, delta=delta), nonlinear=True)
        beta0_over_beta = math.sqrt(1 + ref.beta0**2 * (2 * delta + delta**2)) / (1 + delta)
        assert out.state[4] == pytest.approx(length * (1.0 - beta0_over_beta), rel=1e-12)
        # ... and therefore a straight line's path length is its length, at any momentum.
        # The orbit is supplied: a single drift is not a ring and has no orbit to solve for.
        on_axis = closed_orbit_path_length(lattice, delta, orbit=[0.0, 0.0, 0.0, 0.0])
        assert on_axis == pytest.approx(length, rel=1e-14)


# --------------------------------------------------------------------------------------
# 1. the identity outside every code: a full circle of sector bends
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [-1e-2, -1e-3, 0.0, 1e-3, 1e-2, 5e-2])
def test_pure_circle_path_length_is_exactly_linear(delta: float) -> None:
    """``L(delta) = 2 pi rho (1 + delta)`` exactly — the concentric off-momentum circle.

    The closed orbit is supplied rather than solved for: a ring of pure sector bends has
    **no vertical focusing**, so its vertical tune is zero and ``I - M4`` is singular. That
    is why :func:`closed_orbit_path_length` takes an ``orbit`` argument at all.
    """
    lattice = _circle()
    length = closed_orbit_path_length(lattice, delta, orbit=[RHO * delta, 0.0, 0.0, 0.0])
    assert length == pytest.approx(2.0 * math.pi * RHO * (1.0 + delta), rel=1e-12)


def test_pure_circle_has_alpha_0_one_and_no_higher_coefficients() -> None:
    """Every coefficient past ``alpha_0 = 1`` is zero — nothing fitted, nothing recalled."""
    lattice = _circle()
    circumference = lattice.length

    def f(delta: float) -> float:
        orbit = [RHO * delta, 0.0, 0.0, 0.0]
        return closed_orbit_path_length(lattice, delta, orbit=orbit) / circumference - 1.0

    h = 1e-3
    odd = 0.5 * (f(+h) - f(-h))
    even = 0.5 * (f(+h) + f(-h))
    assert odd / h == pytest.approx(1.0, rel=1e-12)  # alpha_0 = 1
    assert even / h**2 == pytest.approx(0.0, abs=1e-8)  # alpha_1 = 0
    # alpha_2 = 0: the odd part is linear, so its two step sizes agree exactly.
    assert odd / h - 0.5 * (f(+h / 2) - f(-h / 2)) / (h / 2) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# 2. two disjoint routes, and agreement with the shipped first-order number
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["map", "tracked"])
def test_alpha_0_reproduces_momentum_compaction(method: str) -> None:
    """The series' leading term *is* ``alpha_c`` — on a route that never computes it."""
    lattice = _probe_ring()
    series = momentum_compaction_series(lattice, method=method)
    assert series.alpha_0 == pytest.approx(momentum_compaction(lattice), rel=1e-9)


def test_map_and_tracked_routes_agree_on_alpha_1() -> None:
    """The two routes share ``track`` and the ``zeta`` convention, and nothing else."""
    lattice = _probe_ring()
    from_map = momentum_compaction_series(lattice, method="map")
    tracked = momentum_compaction_series(lattice, method="tracked")
    assert from_map.alpha_1 == pytest.approx(tracked.alpha_1, rel=5e-5)
    # ... and alpha_1 is not a rounding artefact of alpha_0 — it is its own number.
    assert abs(tracked.alpha_1) > 1e-4


def test_straight_lattice_has_no_compaction_at_any_order() -> None:
    """No bending, no dispersion, no path lengthening — every coefficient vanishes.

    ``alpha_1 = 0`` here is a **cancellation**, not an absence: the velocity factor
    contributes ``b2 - b1^2`` at second order and the drifts' and quadrupoles' own ``zeta``
    curvature has to cancel it exactly. So this is a real check on the map route's
    bookkeeping even though the answer is zero.

    The two routes are held at different tolerances because their floors differ by five
    orders and both are measured: the map route is algebraic and lands at ``1.6e-14``,
    while the tracked route divides round-off in the path length by ``h^2`` and so cannot
    resolve ``alpha_1`` below ``~eps/h^2 ≈ 1e-9``. That floor is the reason ``"map"`` is
    the default.
    """
    ref = _ref()
    lattice = _straight_lattice(ref)

    from_map = momentum_compaction_series(lattice, method="map")
    assert from_map.alpha_0 == pytest.approx(0.0, abs=1e-13)
    assert from_map.alpha_1 == pytest.approx(0.0, abs=1e-12)

    tracked = momentum_compaction_series(lattice, method="tracked", order=3)
    assert tracked.alpha_0 == pytest.approx(0.0, abs=1e-13)
    assert tracked.alpha_1 == pytest.approx(0.0, abs=1e-8)
    assert tracked.alpha_2 == pytest.approx(0.0, abs=1e-6)

    # The underlying statement, free of any differencing: the length simply does not move.
    for delta in (-1e-2, 1e-2):
        assert closed_orbit_path_length(lattice, delta) == pytest.approx(lattice.length, rel=1e-14)


def test_alpha_1_moves_with_the_sextupoles() -> None:
    """A gate blind to the nonlinear orbit would report the same ``alpha_1`` for both.

    ``alpha_0`` is a *linear* quantity and is untouched by a sextupole on the design orbit;
    ``alpha_1`` is not, because the second-order orbit is driven by the sextupole's kick at
    the dispersion it sits on. So the pair below separates the two orders cleanly.
    """
    off = momentum_compaction_series(_probe_ring(k2=0.0), method="map")
    on = momentum_compaction_series(_probe_ring(k2=0.5), method="map")
    assert on.alpha_0 == pytest.approx(off.alpha_0, rel=1e-12)
    assert abs(on.alpha_1 - off.alpha_1) > 0.5 * abs(off.alpha_1)


# --------------------------------------------------------------------------------------
# 3. the factor of two, pinned inside the analytic suite
# --------------------------------------------------------------------------------------


def test_local_compaction_slope_is_twice_alpha_1() -> None:
    """``d(C/C_0)/ddelta = alpha_0 + 2 alpha_1 delta`` — the factor PTC's spelling needs.

    This is the convention statement of the milestone, so it is checked against the
    tracked path length directly rather than against another accsim quantity.
    """
    lattice = _probe_ring()
    circumference = lattice.length
    series = momentum_compaction_series(lattice, method="tracked")

    def local_slope(delta: float, step: float = 2e-4) -> float:
        plus = closed_orbit_path_length(lattice, delta + step) / circumference
        minus = closed_orbit_path_length(lattice, delta - step) / circumference
        return (plus - minus) / (2.0 * step)

    at = 4e-3
    measured = (local_slope(+at) - local_slope(-at)) / (2.0 * at)
    assert measured == pytest.approx(2.0 * series.alpha_1, rel=1e-3)


# --------------------------------------------------------------------------------------
# transition energy
# --------------------------------------------------------------------------------------


def test_transition_gamma_is_where_the_slip_factor_vanishes() -> None:
    """``gamma_t = 1/sqrt(alpha_c)``, checked by rebuilding the ring at that energy."""
    lattice = _probe_ring()
    gamma_t = transition_gamma(lattice)
    assert gamma_t == pytest.approx(1.0 / math.sqrt(momentum_compaction(lattice)), rel=1e-14)

    at_transition = _probe_ring(energy_eV=gamma_t * PROTON_MASS_EV)
    assert slip_factor(at_transition) == pytest.approx(0.0, abs=1e-12)
    # Below it the slip factor is negative, above it positive — the sign the RF phase needs.
    # gamma_t is only 1.538 on this ring, so "below" has to stay above gamma_0 = 1.
    assert slip_factor(_probe_ring(energy_eV=1.2 * PROTON_MASS_EV)) < 0.0
    assert slip_factor(_probe_ring(energy_eV=3.0 * gamma_t * PROTON_MASS_EV)) > 0.0


def test_transition_gamma_refuses_a_negative_compaction_ring() -> None:
    """A ring that never crosses transition has no ``gamma_t``; ``nan`` would pass silently."""
    lattice = _straight_lattice(_ref())  # stable, but alpha_c == 0
    with pytest.raises(ValueError, match="no transition energy"):
        transition_gamma(lattice)


# --------------------------------------------------------------------------------------
# argument handling
# --------------------------------------------------------------------------------------


def test_third_order_needs_the_tracked_route() -> None:
    """The map route stops at ``alpha_1`` because ``accsim.taylor`` is second order."""
    lattice = _probe_ring()
    with pytest.raises(ValueError, match="order=3 needs method='tracked'"):
        momentum_compaction_series(lattice, order=3, method="map")
    series = momentum_compaction_series(lattice, order=3, method="tracked")
    assert series.alpha_2 is not None
    assert momentum_compaction_series(lattice, order=2, method="tracked").alpha_2 is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "quadrature"}, "method must be"),
        ({"order": 4}, "order must be"),
    ],
)
def test_argument_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        momentum_compaction_series(_probe_ring(), **kwargs)


def test_path_length_rejects_a_malformed_orbit() -> None:
    with pytest.raises(ValueError, match="length-4"):
        closed_orbit_path_length(_circle(), 1e-3, orbit=np.zeros(6))
