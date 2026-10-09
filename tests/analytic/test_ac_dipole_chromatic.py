"""The chromatic driven response (W3): an off-momentum particle is swung by its own amount.

A particle of momentum ``delta`` has its own tune ``Q(delta) = Q + Q' delta + ...`` and its
own beta, so the same AC dipole drives it to a different steady-state amplitude. W1's
closed form holds for it with *its* optics:

    u_hat(delta) = kick beta(delta) sin 2 pi Q(delta) / (2 (cos 2 pi nu - cos 2 pi Q(delta))),

and near the resonance the lever is large — ``u_hat ~ 1 / (Q(delta) - nu)``, a relative
slope of about ``-Q' / (Q - nu)``, ~130 per unit ``delta`` on W1's ring. Two consequences
are the milestone's physics: a beam with a momentum spread swings **more** on average than
its on-momentum particle, by ``(Q' sigma_delta / (Q - nu))^2`` to leading order (the bias an
AC-dipole optics measurement would read — reasoned, gated only at the dipole), and a spread wide enough to reach the drive tune puts
part of the beam on its own resonance (refused, not returned as a number).

The sharp gates need no arbiter. W1's steady-state solve, built from the Jacobians of
``track()`` on the closed orbit at ``delta``, is the off-momentum particle's exact linear
solution; tracking leaves it only through the map's nonlinearity, and the *order* of that
miss is gated — cubic (the exact drift) on a straight ring, quadratic (the bend's
second-order terms) on a bent one. The slope is derived in sympy. A bunch shows the
classic AC-dipole property: its switch-on transient decoheres through the chromatic tune
spread, its driven swing does not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    PROTON_MASS_EV,
    PX,
    PY,
    ACDipole,
    Dipole,
    Drift,
    Lattice,
    Particle,
    ReferenceParticle,
    ThinQuadrupole,
    Tracker,
    X,
    Y,
    driven_amplitude,
)
from accsim.orbit import closed_orbit_nonlinear, linearised_element_maps
from accsim.twiss import (
    ResonantLatticeError,
    chromatic_functions,
    closed_twiss,
    propagate_twiss,
    tunes_on_orbit,
)

REF = ReferenceParticle.from_gamma(PROTON_MASS_EV, 10.0)
LD, NCELL, K1L = 5.0, 6, 0.25
ANGLE = 2.0 * math.pi / (2 * NCELL)  # "bent": every drift a sector bend, D_x != 0
IDX = 2  # after the first drift (or bend), where alpha != 0
OFFSET = 0.012  # |Q - nu| on momentum, as in W1 and W2
LAG = 0.13
DELTA = 4.0e-3  # Q' delta = -0.006 on the straight ring: half the offset


def _ring(kind: str, acd: ACDipole | None = None, first: bool = False) -> Lattice:
    """W2's rings. ``first`` rotates the same ring to start at the dipole."""
    els: list = []
    for _ in range(NCELL):
        body = Dipole(LD, ANGLE) if kind == "bent" else Drift(LD)
        body2 = Dipole(LD, ANGLE) if kind == "bent" else Drift(LD)
        els += [ThinQuadrupole(K1L), body, ThinQuadrupole(-K1L), body2]
    if acd is not None:
        els.insert(IDX, acd)
    if first:
        els = els[IDX:] + els[:IDX]
    return Lattice(els, REF)


def _q(kind: str, plane: str = "x") -> float:
    return tunes_on_orbit(_ring(kind))[0 if plane == "x" else 1]


def _drive(kind: str, kick: float, side: int = 1, plane: str = "x") -> ACDipole:
    """``side = +1``: drive below the tune; ``-1``: above."""
    nu = _q(kind, plane) % 1.0 - side * OFFSET
    return ACDipole(kick=kick, tune=nu, lag=LAG, plane=plane)


def _steady(lat: Lattice, idx: int, acd: ACDipole, delta: float, n: int, solve_at=None):
    """``(orbit, oscillation)``: the closed orbit at ``delta`` and W1's steady state about
    it, from the maps linearised on the orbit at ``solve_at`` (default: ``delta``)."""
    at = delta if solve_at is None else solve_at
    maps = linearised_element_maps(lat, delta=at)
    A = np.eye(6)
    for m in maps[:idx]:
        A = m @ A
    B = np.eye(6)
    for m in maps[idx + 1 :]:
        B = m @ B
    lam = np.exp(2j * np.pi * acd.tune)
    b = np.zeros(6)
    b[PX if acd.plane == "x" else PY] = 1.0
    zhat = np.linalg.solve(
        lam * np.eye(6) - B @ A, B @ b * acd.amplitude * np.exp(2j * np.pi * acd.lag)
    )
    orbit = np.zeros(6)
    orbit[:4] = closed_orbit_nonlinear(lat, delta=delta)
    orbit[5] = delta
    n_ = np.arange(n + 1)
    return orbit, np.imag(zhat[None, :] * lam ** n_[:, None])


def _rel_miss(kind: str, delta: float, kick: float, solve_at=None, n: int = 600) -> float:
    """Exact-path miss from the steady state, x at the lattice start, over its amplitude."""
    acd = _drive(kind, kick)
    lat = _ring(kind, acd)
    orbit, osc = _steady(lat, IDX, acd, delta, n, solve_at)
    got = Tracker(lat).track_turns(Particle.from_array(orbit + osc[0]), n, nonlinear=True)
    return float(np.max(np.abs(got[:, X] - orbit[X] - osc[:, X])) / np.max(np.abs(osc[:, X])))


# --- 1. the named limitation -------------------------------------------------------------
def test_the_linear_walk_is_blind_to_it() -> None:
    """Element matrices carry no ``delta``: on a dispersion-free ring the default walk
    swings every momentum identically, bit for bit. Only ``track()`` sees chromaticity."""
    acd = _drive("fodo", 1.0e-5)
    tr = Tracker(_ring("fodo", acd))
    on = tr.track_turns(Particle(), 300)
    off = tr.track_turns(Particle(delta=DELTA), 300)
    np.testing.assert_array_equal(off[:, :4], on[:, :4])
    exact = tr.track_turns(Particle(delta=DELTA), 300, nonlinear=True)
    assert np.max(np.abs(exact[:, X] - on[:, X])) > 0.1 * np.max(np.abs(on[:, X]))


# --- 2. the identity off momentum, gated on the order of the miss -------------------------
@pytest.mark.parametrize("delta", [DELTA, -DELTA])
@pytest.mark.parametrize(("kind", "order"), [("fodo", 4.0), ("bent", 2.0)])
def test_the_off_momentum_steady_state_holds_to_the_maps_nonlinearity(
    kind: str, order: float, delta: float
) -> None:
    """Halving the kick divides the relative miss by 4 on the straight ring — the exact
    drift's cubic term — and by 2 on the bent one, whose sector bends have second-order
    terms: the bent ring's miss carries a mean shift and a ``2 nu`` line, the signature of a
    quadratic map (localised 2026-10-09; the pre-committed 3.8-4.2 held only for the
    straight ring). Measured ratios 3.9997-3.99999 and 1.998-1.999."""
    big, small = _rel_miss(kind, delta, 4.0e-5), _rel_miss(kind, delta, 2.0e-5)
    assert 1e-9 < small < big < 1e-3  # measured 5.9e-7 .. 6.4e-4: not round-off, not broken
    assert order * 0.95 < big / small < order * 1.05


@pytest.mark.parametrize("delta", [DELTA, -DELTA])
@pytest.mark.parametrize("kind", ["fodo", "bent"])
def test_the_on_momentum_solution_is_not_it(kind: str, delta: float) -> None:
    """Control: the on-momentum steady state, about the same off-momentum orbit, misses by
    order one (measured 0.25-1.99)."""
    assert _rel_miss(kind, delta, 4.0e-5, solve_at=0.0) > 0.2


# --- 3. driven_amplitude is what the particle does ----------------------------------------
@pytest.mark.parametrize("delta", [-DELTA, 0.0, DELTA])
@pytest.mark.parametrize(("kind", "kick", "gate"), [("fodo", 1e-6, 1e-6), ("bent", 1e-7, 1e-5)])
def test_driven_amplitude_is_the_tracked_swing_at_the_dipole(
    kind: str, kick: float, gate: float, delta: float
) -> None:
    """Signed and in phase with the kick, about the orbit at ``delta``. Floors (relative):
    straight 1.6e-9 .. 4.3e-8 at 1e-6 rad; bent ~1.2e-6 at 1e-7 rad (its quadratic term,
    linear in the kick)."""
    acd = _drive(kind, kick)
    lat = _ring(kind, acd, first=True)
    u = driven_amplitude(lat, delta=delta)["x"]
    orbit, osc = _steady(lat, 0, acd, delta, 600)
    got = Tracker(lat).track_turns(Particle.from_array(orbit + osc[0]), 600, nonlinear=True)
    wave = u * np.sin(2.0 * np.pi * (acd.tune * np.arange(601) + LAG))
    assert np.max(np.abs(got[:, X] - orbit[X] - wave)) < gate * abs(u)
    assert u > 0.0  # below the tune: in phase with the kick


@pytest.mark.parametrize("delta", [-DELTA, DELTA])
def test_a_vertical_drive_reads_the_vertical_optics(delta: float) -> None:
    """The ``y`` branch, on the bent ring where the planes differ (``Q_y != Q_x``, and only
    ``x`` has dispersion). Read as the horizontal answer it would be wrong by order one."""
    acd = _drive("bent", 1e-7, plane="y")
    lat = _ring("bent", acd, first=True)
    amp = driven_amplitude(lat, delta=delta)
    assert set(amp) == {"y"}
    u = amp["y"]
    orbit, osc = _steady(lat, 0, acd, delta, 600)
    got = Tracker(lat).track_turns(Particle.from_array(orbit + osc[0]), 600, nonlinear=True)
    wave = u * np.sin(2.0 * np.pi * (acd.tune * np.arange(601) + LAG))
    assert np.max(np.abs(got[:, Y] - orbit[Y] - wave)) < 1e-5 * abs(u)
    # the drive stays in its plane to first order: x moves by u^2-sized amounts (2.4e-12 m,
    # 1.6e-6 of u; the bends' second order), not by anything linear in the drive
    assert np.max(np.abs(got[:, X] - orbit[X])) < 1e-5 * abs(u)
    as_x = ACDipole(kick=1e-7, tune=acd.tune, lag=LAG)
    assert abs(driven_amplitude(_ring("bent", as_x, first=True), delta=delta)["x"] / u - 1) > 0.1


@pytest.mark.parametrize("kind", ["fodo", "bent"])
def test_on_momentum_it_is_w1s_closed_form_on_the_design_optics(kind: str) -> None:
    acd = _drive(kind, 1.0e-5)
    lat = _ring(kind, acd, first=True)
    tw = propagate_twiss(lat, closed_twiss(lat))
    q = tw[-1].mu_x / (2.0 * math.pi)
    closed = 1.0e-5 * tw[0].beta_x * math.sin(2 * math.pi * q)
    closed /= 2.0 * (math.cos(2 * math.pi * acd.tune) - math.cos(2 * math.pi * q))
    assert driven_amplitude(lat)["x"] == pytest.approx(closed, rel=1e-10)  # measured 6e-13


# --- 4. the slope, derived ------------------------------------------------------------------
def test_the_slope_is_derived_not_remembered() -> None:
    """``u_hat = kick beta f(mu)``, so ``d ln u_hat / d delta = b + f'/f * 2 pi Q'`` with
    ``b = beta'/beta`` the MAD8 chromatic beta, and ``f'/f`` as below — whose pole at the
    drive gives the near-resonance shorthand ``-Q' / (Q - nu)``."""
    mu, th, be = sp.symbols("mu theta beta", real=True, positive=True)
    f = be * sp.sin(mu) / (2 * (sp.cos(th) - sp.cos(mu)))
    target = (sp.cos(th) * sp.cos(mu) - 1) / (sp.sin(mu) * (sp.cos(th) - sp.cos(mu)))
    assert sp.simplify(sp.diff(sp.log(f), mu) - target) == 0
    assert sp.simplify(sp.diff(sp.log(f), be) - 1 / be) == 0
    eps = sp.symbols("epsilon")
    pole = sp.series(eps * target.subs(mu, th + eps), eps, 0, 1).removeO()
    assert sp.simplify(pole) == -1  # f'/f = -1/(mu - theta) + regular (sp.limit hangs here)


def _slope_formula(lat: Lattice, nu: float, h: float) -> float:
    qp = (tunes_on_orbit(lat, delta=h)[0] - tunes_on_orbit(lat, delta=-h)[0]) / (2.0 * h)
    b = chromatic_functions(lat, delta=h)[0].b_x
    c, mu = math.cos(2 * math.pi * nu), 2 * math.pi * tunes_on_orbit(lat)[0]
    return b + 2 * math.pi * qp * (c * math.cos(mu) - 1) / (math.sin(mu) * (c - math.cos(mu)))


def _slope_fd(lat: Lattice, h: float) -> float:
    up, dn = driven_amplitude(lat, delta=h)["x"], driven_amplitude(lat, delta=-h)["x"]
    return (math.log(up) - math.log(dn)) / (2.0 * h)


@pytest.mark.parametrize("kind", ["fodo", "bent"])
def test_driven_amplitude_follows_the_derived_slope(kind: str) -> None:
    """Gated on the order: halving the step quarters the residual (measured 4.000)."""
    acd = _drive(kind, 1.0e-5)
    lat = _ring(kind, acd, first=True)
    res = [abs(_slope_fd(lat, h) - _slope_formula(lat, acd.tune, h)) for h in (1e-4, 5e-5)]
    assert 3.8 < res[0] / res[1] < 4.2
    slope = _slope_fd(lat, 5e-5)
    # the step's own O(h^2) truncation, which the ratio above already pins: measured 1.3e-5
    # (straight) and 9e-7 (bent) relative; a wrong coefficient misses by O(1)
    assert res[1] < 1e-4 * abs(slope)
    # the shorthand is the bulk of it, and not it: 1.8% (straight) and 1.2% (bent) off
    qp = (tunes_on_orbit(lat, delta=5e-5)[0] - tunes_on_orbit(lat, delta=-5e-5)[0]) / 1e-4
    short = -qp / OFFSET
    assert 0.005 < abs(short / slope - 1) < 0.05


# --- 5. the direction -----------------------------------------------------------------------
def test_the_drive_pulls_hardest_on_the_particles_whose_tune_it_sits_closest_to() -> None:
    """``Q' < 0``: above momentum the tune falls. Below the tune that is toward the drive (a
    bigger swing), above the tune away from it — and there the swing is in antiphase."""
    below = _ring("fodo", _drive("fodo", 1.0e-5, side=+1))
    u = [driven_amplitude(below, delta=d)["x"] for d in (-DELTA, 0.0, DELTA)]
    assert 0.0 < u[0] < u[1] < u[2]
    above = _ring("fodo", _drive("fodo", 1.0e-5, side=-1))
    u = [driven_amplitude(above, delta=d)["x"] for d in (-DELTA, 0.0, DELTA)]
    assert u[0] < u[1] < u[2] < 0.0  # antiphase, and |u| shrinks with delta


# --- 6. the particle's own resonance is refused -----------------------------------------------
@pytest.mark.parametrize("mirror", [False, True])
def test_a_drive_on_the_particles_own_tune_is_refused(mirror: bool) -> None:
    q = tunes_on_orbit(_ring("fodo"), delta=DELTA)[0] % 1.0
    acd = ACDipole(kick=1.0e-5, tune=1.0 - q if mirror else q)
    lat = _ring("fodo", acd)
    with pytest.raises(ResonantLatticeError):
        driven_amplitude(lat, delta=DELTA)
    assert math.isfinite(driven_amplitude(lat)["x"])  # on momentum it is an ordinary drive


# --- 7 & 8. a beam with a momentum spread -----------------------------------------------------
def _grid(sigma: float, n: int = 201) -> tuple[np.ndarray, np.ndarray]:
    """Momenta on a uniform grid over +-5 sigma, weighted by their Gaussian share: a
    quadrature, so finite-sample noise cannot pose as a residual. (The drive's own
    resonance sits at 7.8 sigma for sigma = 1e-3 and is outside it.)"""
    d = np.linspace(-5.0 * sigma, 5.0 * sigma, n)
    w = np.exp(-0.5 * (d / sigma) ** 2)
    return d, w / w.sum()


def test_a_driven_beam_does_not_decohere_and_its_switch_on_transient_does() -> None:
    """Switched on abruptly from rest, every particle carries a free oscillation as big as
    its driven one. The free parts spread in tune (``Q' sigma = 1.5e-3``) and their centroid
    dies; the driven parts all run at ``nu`` and it does not. Measured: 0.96 of the swing
    in the first 100 turns, 1.4e-6 after 600. Control: the linear walk has no tune spread
    and keeps the transient (0.95)."""
    acd = _drive("fodo", 1.0e-5)
    lat = _ring("fodo", acd, first=True)
    d, w = _grid(1.0e-3)
    u_mean = w @ np.array([driven_amplitude(lat, delta=x)["x"] for x in d])
    n_turns = 3000
    turns = np.arange(n_turns + 1)
    steady = u_mean * np.sin(2.0 * np.pi * (acd.tune * turns + LAG))
    state = np.zeros((6, d.size))
    state[5] = d
    tr = Tracker(lat)
    centroid = np.empty(n_turns + 1)
    for n in turns:
        centroid[n] = w @ state[X]
        state = tr.track_once(state, turn=int(n))
    miss = np.abs(centroid - steady) / abs(u_mean)
    assert miss[:100].max() > 0.5
    assert miss[600:].max() < 1e-5
    # the linear walk: every momentum is the on-momentum particle, transient and all
    lin = tr.track_turns(Particle(delta=d[0]), n_turns)[:, X]
    u0 = driven_amplitude(lat)["x"]
    lin_miss = np.abs(lin - u0 * np.sin(2.0 * np.pi * (acd.tune * turns + LAG))) / abs(u0)
    assert lin_miss[600:].max() > 0.5


def test_a_beam_with_momentum_spread_swings_more_than_its_on_momentum_particle() -> None:
    """``<u_hat> / u_hat(0) - 1 = (1/2) u_hat''/u_hat sigma^2 + O(sigma^4)``, whose
    near-resonance bulk is ``(Q' sigma / (Q - nu))^2``. Gated on two orders: the excess
    quarters when sigma halves (measured 4.04), and what is left after the sigma^2 term
    falls by 16 (measured 16.99) — so the coefficient is the second derivative, not a fit."""
    acd = _drive("fodo", 1.0e-5)
    lat = _ring("fodo", acd, first=True)
    u0 = driven_amplitude(lat)["x"]
    h = 1.0e-4
    k2 = (driven_amplitude(lat, delta=h)["x"] - 2 * u0 + driven_amplitude(lat, delta=-h)["x"]) / (
        h * h * u0
    )

    def excess(sigma: float) -> float:
        d, w = _grid(sigma)
        return float(w @ np.array([driven_amplitude(lat, delta=x)["x"] for x in d]) / u0 - 1.0)

    big, small = excess(5.0e-4), excess(2.5e-4)
    assert big > 0.0 and small > 0.0  # it swings more, never less
    assert 3.9 < big / small < 4.2
    rest_big, rest_small = big - 0.5 * k2 * 5.0e-4**2, small - 0.5 * k2 * 2.5e-4**2
    assert 15.0 < rest_big / rest_small < 18.0
    # the near-resonance shorthand carries all but 0.6% of the coefficient
    qp = (tunes_on_orbit(lat, delta=h)[0] - tunes_on_orbit(lat, delta=-h)[0]) / (2.0 * h)
    assert 0.5 * k2 == pytest.approx((qp / OFFSET) ** 2, rel=0.02)
    # and it is not small at a realistic spread: 1.7% of the swing at sigma = 1e-3
    assert excess(1.0e-3) > 0.015
