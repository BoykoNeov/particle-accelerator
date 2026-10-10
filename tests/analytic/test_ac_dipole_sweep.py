"""The drive-tune sweep (W5): a drive whose tune moves through W4's three steady states.

The roadmap called this "the hysteresis loop". That loop belongs to a *damped* oscillator,
and the rings in W4's scope have no damping, so the two sweep directions do two different
things:

- swept **toward** the tune from the side the detuning pulls to, the particle rides W4's
  small state, lagging the drive by an amount proportional to the rate, and is thrown off
  where that state meets the middle one (the fold, read in tune at fixed kick:
  :func:`accsim.driven_fold_tune`). It lands on **no** steady state;
- swept **through** the tune the way the detuning moves it, the particle can phase-lock and
  be carried out along W4's large branch: **autoresonance**, above a threshold kick
  (:func:`accsim.autoresonance_threshold`) that grows as the rate to the 3/4.

The truth is again the exact driven map (``track()``), with each particle started on an
exact period-``q`` orbit at a rational start tune, or with the drive ramped on far from
resonance: an abrupt switch-on leaves a free oscillation as large as the driven one, and
nothing here damps it. No reference code has the swept element (xtrack's ``ACDipole`` and
MAD-X's ``HACDIPOLE`` have a fixed frequency); xtrack transcribes it in
``tests/reference/test_ac_dipole_sweep_xtrack.py``.

Fixture: W4's split-tune FODO (``Qx = 1.344``) with ``k3l = -2000`` at the dipole.
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from _w4_driven_orbits import IDX, LAG, _acd, _exact_state, _periodic_orbit, _ring, _seed
from accsim import (
    PX,
    ACDipole,
    Lattice,
    Particle,
    ThinSextupole,
    Tracker,
    X,
    autoresonance_threshold,
    driven_amplitude,
    driven_fold_kick,
    driven_fold_tune,
    driven_gradient,
    driven_states,
    driven_twiss,
)
from accsim.twiss import _AUTORESONANCE_MU_C, _ResponseCurve, tunes

K3L = -2000.0
NUF = 13 / 40  # the exact fold of gates 2-4 sits here
NU_UP = 3 / 10  # the up-sweep starts on the exact period-10 small orbit
NU_DOWN = 3 / 8  # the down-sweep starts on the exact period-8 (single) orbit
UP_RATES = (4e-6, 2e-6, 1e-6)


# --- shared machinery ---------------------------------------------------------------------
def _sweep(k3l: float, dipoles: list[ACDipole], z0: np.ndarray, n_turns: int) -> np.ndarray:
    """``x`` at the dipole on every turn, one column per dipole, each driving its own
    particle through the exact map of the fixture with that dipole at ``IDX``.

    ``z0`` is ``(6,)`` or ``(6, len(dipoles))`` at the lattice start. Each column's kick is
    its own element's :meth:`~accsim.elements.acdipole.ACDipole.drive_kick`.
    """
    lat = _ring(dipoles[0], k3l)
    s = np.empty((6, len(dipoles)))
    s[:] = np.asarray(z0, dtype=float).reshape(6, -1)
    xd = np.empty((n_turns, len(dipoles)))
    for turn in range(n_turns):
        for k, elem in enumerate(lat.elements):
            if k == IDX:  # the dipole: identity body, then each column's own kick
                xd[turn] = s[X]
                s[PX] += [d.drive_kick(turn, lat.ref)[PX] for d in dipoles]
            else:
                s = elem.track(s, lat.ref)
    return xd


def _demod(x: np.ndarray, d: ACDipole, centre: int, width: int) -> tuple[float, float]:
    """``(in-phase, quadrature)`` swing about turn ``centre``, against the drive's own phase."""
    n = np.arange(centre - width // 2, centre + width // 2)
    ph = 2.0 * math.pi * (d.tune * n + 0.5 * d.tune_rate * n * n + d.lag)
    (u, v), *_ = np.linalg.lstsq(np.column_stack((np.sin(ph), np.cos(ph))), x[n], rcond=None)
    return float(u), float(v)


def _start(kick: float, nu: float, which: str, k3l: float = K3L) -> np.ndarray:
    """The exact period-``q`` orbit of the fixed drive at ``nu``, seeded at the predicted
    ``"small"`` or ``"large"`` stable state, as a ``(6,)`` state at the lattice start."""
    lat = _ring(_acd(kick, nu), k3l)
    stable = [st.amplitude for st in driven_states(lat)["x"] if st.stable]
    pred = min(stable, key=abs) if which == "small" else max(stable, key=abs)
    q = Fraction(nu).limit_denominator(1000).denominator
    z, _, _ = _periodic_orbit(lat, _seed(lat, pred, kick, nu), q)
    assert z is not None
    out = np.zeros(6)
    out[X], out[PX] = z
    return out


def _turn_at(d: ACDipole, nu: float) -> int:
    """The turn on which the instantaneous drive tune ``tune + tune_rate n`` is ``nu``."""
    return round((nu - d.tune) / d.tune_rate)


@functools.cache
def _exact_fold_kick() -> float:
    """The kick at which the exact small and middle orbits merge at ``13/40`` (W4's
    extrapolation of their squared gap, linear in the kick near a saddle-node)."""
    fc = driven_fold_kick(_ring(_acd(1e-5, NUF), K3L))["x"]
    gaps = []
    for f in (0.996, 0.998):
        lat = _ring(_acd(f * fc, NUF), K3L)
        _, sm, md = driven_states(lat)["x"]
        u_sm, _, _ = _exact_state(lat, sm.amplitude, NUF)
        u_md, _, _ = _exact_state(lat, md.amplitude, NUF)
        gaps.append((u_md - u_sm) ** 2)
    return (0.998 + gaps[1] * (0.998 - 0.996) / (gaps[0] - gaps[1])) * fc


@functools.cache
def _up_sweep() -> tuple[list[ACDipole], np.ndarray]:
    """Gates 2-4: from the exact small orbit at ``3/10``, swept up past ``0.36``."""
    kick = _exact_fold_kick()
    dipoles = [ACDipole(kick, NU_UP, LAG, tune_rate=r) for r in UP_RATES]
    n_turns = math.ceil((0.36 - NU_UP) / min(UP_RATES)) + 1
    xd = _sweep(K3L, dipoles, _start(kick, NU_UP, "small"), n_turns)
    return dipoles, xd


def _departure(x: np.ndarray) -> int:
    """The first turn on which ``|x| > 1.5 cm`` (a lost particle, NaN, has departed too)."""
    gone = ~(np.abs(x) <= 0.015)
    n = int(np.argmax(gone))
    assert gone[n]
    return n


# --- 1. the element -------------------------------------------------------------------
def test_a_fixed_drive_is_w1s_element_bit_for_bit() -> None:
    """``tune_rate = 0`` is W1's element: its kick is W1's formula ``kick ramp(n)
    sin(2 pi (nu n + lag))`` to the last bit (the sweep term is exactly ``0.0``), and it
    tracks exactly as the default-constructed element (both paths)."""
    d = ACDipole(3e-4, 0.325, LAG, ramp=(0, 30, 10**6, 10**6))
    for n in range(500):
        w1_kick = 3e-4 * d.ramp_factor(n) * math.sin(2.0 * math.pi * (0.325 * n + LAG))
        assert d.drive_kick(n, _ring().ref)[PX] == w1_kick
    w1 = _ring(d, K3L)
    w5 = _ring(ACDipole(3e-4, 0.325, LAG, ramp=(0, 30, 10**6, 10**6), tune_rate=0.0), K3L)
    p = Particle(x=1e-3, px=-2e-4)
    for nonlinear in (False, True):
        a = Tracker(w1).track_turns(p, 200, nonlinear=nonlinear)
        b = Tracker(w5).track_turns(p, 200, nonlinear=nonlinear)
        assert np.array_equal(a, b)


def test_the_swept_kick_is_the_accumulated_phase() -> None:
    """The kick on turn ``n`` is ``kick ramp(n) sin(phase_n)``, with ``phase_n`` the sum of
    the per-turn increments ``2 pi (nu + nu_dot (k + 1/2))``: the instantaneous tune on
    turn ``n`` is ``nu + nu_dot n``. The static maps stay the identity."""
    ref = _ring().ref
    d = ACDipole(2e-4, 0.33, 0.21, ramp=(5, 50, 900, 1000), tune_rate=-3.7e-5)
    phase = 0.21
    for n in range(1000):
        expect = 2e-4 * d.ramp_factor(n) * math.sin(2.0 * math.pi * phase)
        assert d.drive_kick(n, ref)[PX] == pytest.approx(expect, rel=1e-9, abs=1e-15)
        phase += 0.33 + -3.7e-5 * (n + 0.5)
    assert np.array_equal(d.matrix(ref), np.eye(6))
    assert np.array_equal(d.kick(ref), np.zeros(6))


def test_the_fixed_drive_optics_refuse_a_swept_dipole() -> None:
    lat = _ring(ACDipole(2e-4, 0.33, LAG, tune_rate=1e-6), K3L)
    for fn in (driven_gradient, driven_twiss, driven_amplitude, driven_states, driven_fold_kick):
        with pytest.raises(ValueError, match="swept"):
            fn(lat)


# --- 2. following the small branch ------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("nu", [5 / 16, 8 / 25])
def test_the_up_sweep_rides_the_small_state_lagging_by_the_rate(nu: float) -> None:
    """The in-phase swing is the exact small orbit's; the quadrature (the lag behind the
    drive) halves with the rate. Measured: in-phase within 1.7e-4; lag x1.94-2.10."""
    dipoles, xd = _up_sweep()
    lat = _ring(_acd(dipoles[0].amplitude, nu), K3L)
    small = driven_states(lat)["x"][1]
    u_exact, _, _ = _exact_state(lat, small.amplitude, nu)
    lags = []
    for p, d in enumerate(dipoles):
        u, v = _demod(xd[:, p], d, _turn_at(d, nu), 80)
        assert abs(u - u_exact) < 1e-3 * abs(u_exact)
        lags.append(v)
    for a, b in zip(lags, lags[1:], strict=False):
        assert 1.8 < a / b < 2.2


# --- 3. the departure -----------------------------------------------------------------
@pytest.mark.slow
def test_the_departure_converges_on_the_exact_fold() -> None:
    """With the kick at the exact fold of ``13/40``, the per-turn departure (``|x| > 1.5
    cm``) lies past ``13/40`` by an overshoot that falls with the rate (ratios in 1.5-2.0,
    under ``2e-4`` at ``1e-6``). Measured 3.76e-4, 2.24e-4, 1.33e-4."""
    dipoles, xd = _up_sweep()
    over = [d.tune + d.tune_rate * _departure(xd[:, p]) - NUF for p, d in enumerate(dipoles)]
    assert all(o > 0.0 for o in over)
    for a, b in zip(over, over[1:], strict=False):
        assert 1.5 < a / b < 2.0
    assert over[-1] < 2e-4
    # driven_fold_tune is the static departure: the cubic's fold, not the exact one
    ft = driven_fold_tune(_ring(dipoles[-1], K3L))["x"]
    assert abs(ft - NUF) < 2e-4


def _averaged_overshoot(alpha: float, radius: float = 1.3) -> float:
    """``i Psi' = -(d - |Psi|^2) Psi + mu`` swept down through its fold: the overshoot
    past ``d_f`` when ``|Psi|`` first exceeds ``radius`` times the fold amplitude."""
    mu = 1.0
    r_f, d_f = (mu / 2.0) ** (1.0 / 3.0), 3.0 * (mu / 2.0) ** (2.0 / 3.0)
    d0 = 2.0 * d_f
    r0 = brentq(lambda r: (d0 - r * r) * r - mu, 1e-9, r_f)

    def rhs(t: float, y: np.ndarray) -> list[float]:
        psi = y[0] + 1j * y[1]
        dpsi = 1j * ((d0 - alpha * t - abs(psi) ** 2) * psi - mu)
        return [dpsi.real, dpsi.imag]

    def escaped(t: float, y: np.ndarray) -> float:
        return y[0] ** 2 + y[1] ** 2 - (radius * r_f) ** 2

    escaped.terminal = True
    sol = solve_ivp(
        rhs,
        (0.0, 2.0 * d0 / alpha),
        [r0, 0.0],
        events=escaped,
        rtol=1e-11,
        atol=1e-13,
        method="DOP853",
    )
    return d_f - (d0 - alpha * sol.t_events[0][0])


@pytest.mark.slow
def test_the_overshoot_goes_as_the_rate_to_the_four_fifths() -> None:
    """Derived on the averaged equation, not read off the ring (it converges too slowly
    there). Near the fold of an *undamped* oscillator the reduced motion is a saddle-centre,
    ``X'' = X^2 + alpha t``; ``X ~ alpha^(2/5)``, ``t ~ alpha^(-1/5)`` make the overshoot
    ``~ alpha^(4/5)``, next term ``alpha^1``. Fitted on the two slowest of six rates, that
    law predicts the other four to 3%; the damped saddle-node's ``2/3`` misses one by more
    than 5%. Measured: <= 2.1%, and 10-13%."""
    alphas = np.array([1e-2 / 2**k for k in range(6)])
    over = np.array([_averaged_overshoot(a) for a in alphas])

    def worst_miss(p: float) -> float:
        basis = np.column_stack((alphas**p, alphas ** (p + 0.2)))
        coef = np.linalg.solve(basis[-2:], over[-2:])
        return float(np.max(np.abs(basis[:-2] @ coef / over[:-2] - 1.0)))

    assert worst_miss(0.8) < 0.03
    assert worst_miss(2.0 / 3.0) > 0.05


# --- 4. no landing --------------------------------------------------------------------
@pytest.mark.slow
def test_past_the_fold_the_particle_lands_on_no_steady_state() -> None:
    """At ``nu = 0.36`` the up-swept particle is lost, or its oscillation exceeds three
    times the single steady state's swing there: undamped, it never settles."""
    dipoles, xd = _up_sweep()
    (only,) = driven_states(_ring(_acd(dipoles[0].amplitude, 0.36), K3L))["x"]
    for p, d in enumerate(dipoles):
        n = _turn_at(d, 0.36)
        window = xd[n - 200 : n, p]
        lost = not np.all(np.isfinite(window))
        assert lost or np.max(np.abs(window)) > 3.0 * abs(only.amplitude)


# --- 5. capture: the reverse sweep, and the open hysteresis ----------------------------
@pytest.mark.slow
def test_the_down_sweep_locks_onto_the_large_branch() -> None:
    """From the exact single orbit at ``3/8``, kick ``2e-4``, swept down: at ``13/40`` and
    ``5/16`` the 2000-turn swing is antiphase and within 6% of the exact large orbit
    (measured 1.4-4.9%; the libration it carries is named, not gated). Swept up from the
    exact small orbit at ``3/10`` with the **same kick**, the particle at ``13/40`` is on
    the small in-phase state instead: the open hysteresis. (Pre-committed against gate 2's
    run, whose kick puts ``13/40`` exactly on the fold; the comparison needs one kick.)"""
    kick = 2e-4
    down = [ACDipole(kick, NU_DOWN, LAG, tune_rate=r) for r in (-4e-6, -2e-6)]
    up = ACDipole(kick, NU_UP, LAG, tune_rate=2e-6)  # the other direction, same kick
    z_down, z_up = _start(kick, NU_DOWN, "large"), _start(kick, NU_UP, "small")
    n_turns = math.ceil((NU_DOWN - 5 / 16) / 2e-6) + 1001
    xd = _sweep(K3L, [*down, up], np.column_stack((z_down, z_down, z_up)), n_turns)
    for nu in (NUF, 5 / 16):
        lat = _ring(_acd(kick, nu), K3L)
        states = driven_states(lat)["x"]
        assert [s.amplitude > 0.0 for s in states] == [False, True, True]
        u_large, _, _ = _exact_state(lat, states[0].amplitude, nu)
        for p, d in enumerate(down):
            u, _ = _demod(xd[:, p], d, _turn_at(d, nu), 2000)
            assert u < 0.0
            assert abs(u - u_large) < 0.06 * abs(u_large)
    lat = _ring(_acd(kick, NUF), K3L)
    u_small, _, _ = _exact_state(lat, driven_states(lat)["x"][1].amplitude, NUF)
    u_up, _ = _demod(xd[:, 2], up, _turn_at(up, NUF), 80)
    assert u_small > 0.0
    assert abs(u_up - u_small) < 1e-3 * u_small


# --- 6. the threshold -----------------------------------------------------------------
def _universal(mu: float, t0: float = -80.0, t_ramp: float = -40.0, t1: float = 80.0) -> bool:
    """``i Psi' + (tau - |Psi|^2) Psi = mu``, ``mu`` ramped on over ``[t0, t_ramp]``:
    captured iff ``|Psi|^2`` follows ``tau``."""

    def rhs(t: float, y: np.ndarray) -> list[float]:
        psi = y[0] + 1j * y[1]
        m = mu * min(1.0, max(0.0, (t - t0) / (t_ramp - t0)))
        dpsi = -1j * (m - (t - abs(psi) ** 2) * psi)
        return [dpsi.real, dpsi.imag]

    sol = solve_ivp(rhs, (t0, t1), [0.0, 0.0], rtol=1e-10, atol=1e-12, method="DOP853")
    return sol.y[0, -1] ** 2 + sol.y[1, -1] ** 2 > 0.5 * t1


@pytest.mark.slow
def test_mu_c_is_the_universal_equations_threshold() -> None:
    """Integrated, not recalled: 0.41060 (Fajans & Friedland quote 0.411)."""
    lo, hi = 0.40, 0.42
    assert not _universal(lo) and _universal(hi)
    while hi - lo > 5e-5:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if _universal(mid) else (mid, hi)
    assert abs(0.5 * (lo + hi) - _AUTORESONANCE_MU_C) < 1e-4


def test_the_ring_maps_onto_the_universal_equation() -> None:
    """Rotating wave, then scaling, in sympy. The thin kick ``theta sin Phi`` at ``beta``
    averages, against ``psi = phi - Phi``, to ``K_drive = theta sqrt(2 J beta) sin(psi) /
    2``, so ``eps = theta sqrt(2 beta) / 4`` in ``i c' = -dK/d conj(c)``. With
    ``nu_dot = -r``, ``a = -A`` (both < 0: the drive and the tune fall together),
    ``c = lam Psi``, ``n - n_x = tau / kap``, ``kap^2 = 2 pi r``, ``lam^2 = kap / (2 pi A)``,
    the equation is the universal one with ``|mu| = eps sqrt(2 pi A) / (2 pi r)^(3/4)``."""
    theta, beta, J, Phi = sp.symbols("theta beta J Phi", positive=True)
    psi = sp.symbols("psi", real=True)
    # -theta x sin(Phi), x = sqrt(2 J beta) cos(phi): the part slow in psi = phi - Phi
    h_kick = -theta * sp.sqrt(2 * J * beta) * sp.cos(psi + Phi) * sp.sin(Phi)
    slow = sp.integrate(h_kick, (Phi, 0, 2 * sp.pi)) / (2 * sp.pi)
    assert sp.simplify(slow - theta * sp.sqrt(2 * J * beta) * sp.sin(psi) / 2) == 0
    # in c = sqrt(J) e^{i psi}, sqrt(2 J) sin(psi) = sqrt(2) Im(c) = sqrt(2) (c - cbar) / 2i
    c, cb = sp.symbols("c cbar")
    k_drive = theta * sp.sqrt(beta) / 2 * sp.sqrt(2) * (c - cb) / (2 * sp.I)
    # i c' = -dK/dcbar = ... - i eps
    eps = sp.simplify(sp.diff(k_drive, cb) / sp.I)
    assert sp.simplify(eps - theta * sp.sqrt(2 * beta) / 4) == 0

    r, A, tau, R, e = sp.symbols("r A tau R epsilon", positive=True)
    Psi, dPsi = sp.symbols("Psi dPsi")
    kap = sp.sqrt(2 * sp.pi * r)
    lam = sp.sqrt(kap / (2 * sp.pi * A))
    # i c' = -2 pi (r (n - n_x) - A |c|^2) c - i eps, with c' = kap lam dPsi, |c|^2 = lam^2 R
    lhs = sp.I * kap * lam * dPsi
    rhs = -2 * sp.pi * (r * tau / kap - A * lam**2 * R) * lam * Psi - sp.I * e
    target = sp.I * dPsi + (tau - R) * Psi + sp.I * e / (kap * lam)
    assert sp.simplify((lhs - rhs) / (kap * lam) - target) == 0
    mu = e / (kap * lam)
    assert sp.simplify(mu - e * sp.sqrt(2 * sp.pi * A) / (2 * sp.pi * r) ** sp.Rational(3, 4)) == 0


def _locked(x: np.ndarray, end: int, backbone: float) -> bool:
    """Carried out: the last 300 turns swing past 0.35 of the backbone. Swept the way the
    detuning moves the tune, a particle that misses the lock keeps a swing of a fifth of
    it; a locked one rides the backbone, librating about it. (Near the threshold that
    libration is large, and a short in-step average can catch it at its low point.)"""
    window = x[end - 300 : end]
    return bool(np.all(np.isfinite(window)) and np.max(np.abs(window)) > 0.35 * backbone)


def _growth(x: np.ndarray, kap: float, n_cross: float, n_end: int) -> float:
    """RMS swing over the last two ``tau`` units over that around ``tau = +30``. A locked
    particle rides the backbone, ``|Psi|^2 = tau``, so this is ~``sqrt 2``; one that crossed
    unlocked keeps its free action, ~1, however large."""
    span = int(2.0 / kap)
    late = x[n_end - span : n_end]
    mid = x[int(n_cross + 29.0 / kap) : int(n_cross + 29.0 / kap) + span]
    return float(np.sqrt(np.mean(late**2) / np.mean(mid**2)))


def _sweep_through(k3l: float, specs: list[tuple[float, np.ndarray]]) -> list[tuple]:
    """For each ``(rate, grid)``, which factors of the threshold lock, swept from rest
    through the tune. The drive is ramped on over ``tau in [-80, -40]`` and the sweep ends
    at ``tau = +60``, ``tau = sqrt(2 pi |rate|) (n - n_cross)``; the backbone there is
    ``sqrt(2 beta |Q - nu| / |a|)``. Against the detuning (no threshold) the grid is on the
    threshold of the same ring swept the other way. Returns
    ``[(rate, threshold, locked, growth), ...]``, per grid point (see :func:`_growth`)."""
    lat0 = _ring(_acd(1e-5, 0.33), k3l)
    _, beta, a = (float(v[0]) for v in _ResponseCurve(lat0, "test")._optics(np.array([0.0])))
    q = tunes(lat0)[0] % 1.0
    runs, dipoles = [], []
    for rate, grid in specs:
        kap = math.sqrt(2.0 * math.pi * abs(rate))
        n_cross, n_ramp, n_end = 80.0 / kap, int(40.0 / kap), int(140.0 / kap)
        th = autoresonance_threshold(_ring(ACDipole(1.0, 0.3, tune_rate=rate), k3l))["x"]
        if th is None:
            th = autoresonance_threshold(_ring(ACDipole(1.0, 0.3, tune_rate=-rate), k3l))["x"]
        start = len(dipoles)
        for f in grid:
            ramp = (0, n_ramp, 10**9, 10**9)
            dipoles.append(ACDipole(f * th, q - rate * n_cross, LAG, ramp=ramp, tune_rate=rate))
        backbone = math.sqrt(2.0 * beta * abs(rate) * 60.0 / kap / abs(a))
        runs.append((rate, th, start, len(grid), (kap, n_cross, n_end), backbone))
    xd = _sweep(k3l, dipoles, np.zeros(6), max(run[4][2] for run in runs))
    return [
        (
            rate,
            th,
            [_locked(xd[:, c], times[2], bb) for c in range(i0, i0 + n)],
            [_growth(xd[:, c], *times) for c in range(i0, i0 + n)],
        )
        for rate, th, i0, n, times, bb in runs
    ]


def _bracket(grid: np.ndarray, locked: list[bool]) -> tuple[float, float]:
    """The grid step where locking switches on, and only once."""
    flips = [k for k in range(1, len(locked)) if locked[k] != locked[k - 1]]
    assert len(flips) == 1 and locked[-1], locked
    return float(grid[flips[0] - 1]), float(grid[flips[0]])


@pytest.mark.slow
def test_the_tracked_threshold_converges_on_the_formula() -> None:
    """The excess of the tracked threshold over :func:`accsim.autoresonance_threshold`
    falls by ``sqrt 2`` per halving of the rate (ratios in 1.33-1.50) and is under 0.5% at
    ``1e-6``: the formula's coefficient is right, its only error the natural-optics
    correction, first order in ``kick / u ~ rate^(1/2)``. Measured 0.825%, 0.585%, 0.415%
    (ratios 1.410, 1.410). **Control:** the thresholds against ``rate^(2/3)`` or
    ``rate^(1/2)`` instead of ``3/4`` drift by more than 5% over the three rates."""
    rates = (-4e-6, -2e-6, -1e-6)
    coarse = 1.0 + 1e-3 * np.arange(16)  # 1.000 ... 1.015
    first = _sweep_through(K3L, [(r, coarse) for r in rates])
    fine = [_bracket(coarse, locked)[0] + 1e-4 * np.arange(11) for _, _, locked, _ in first]
    second = _sweep_through(K3L, list(zip(rates, fine, strict=True)))
    excess, measured = [], []
    for grid, (_, th, locked, _) in zip(fine, second, strict=True):
        f_lo, f_hi = _bracket(grid, locked)
        f = 0.5 * (f_lo + f_hi)
        excess.append(f - 1.0)
        measured.append(f * th)
    for a, b in zip(excess, excess[1:], strict=False):
        assert 1.33 < a / b < 1.50
    assert 0.0 < excess[-1] < 5e-3
    for p in (2.0 / 3.0, 0.5):
        scaled = [m / abs(r) ** p for m, r in zip(measured, rates, strict=True)]
        assert max(scaled) / min(scaled) > 1.05


# --- 7. the mirror ----------------------------------------------------------------------
@pytest.mark.slow
def test_the_mirror_ring_locks_going_up_and_departs_going_down() -> None:
    """``k3l = +2000``: the tune rises with amplitude, so the **up**-sweep locks (at 1.03x
    the threshold, not at 0.97x) and a down-sweep against it does not lock even at 10x
    (``autoresonance_threshold`` says ``None``): its swing after the crossing can be as
    large as a locked one's, but it does not grow with the drive. A down-sweep from the
    exact small orbit at
    ``2/5`` departs past ``driven_fold_tune``, which lies above the tune."""
    k3l = 2000.0
    against = _ring(ACDipole(1.0, 0.3, tune_rate=-2e-6), k3l)
    assert autoresonance_threshold(against)["x"] is None
    (_, _, locked, growth), (_, _, _, against) = _sweep_through(
        k3l, [(2e-6, np.array([0.97, 1.03])), (-2e-6, np.array([10.0]))]
    )
    assert locked == [False, True]
    assert growth[1] > 1.25  # the locked one rides the backbone (~sqrt 2) ...
    assert against[0] < 1.15  # ... the one swept against it, at 10x, keeps its action

    kick = 2e-4
    q = tunes(_ring(None, k3l))[0] % 1.0
    fold = driven_fold_tune(_ring(_acd(kick, 0.4), k3l))["x"]
    assert q < fold < 0.4
    down = ACDipole(kick, 0.4, LAG, tune_rate=-2e-6)
    xd = _sweep(k3l, [down], _start(kick, 0.4, "small", k3l), math.ceil((0.4 - q) / 2e-6))
    nu_dep = 0.4 - 2e-6 * _departure(xd[:, 0])
    assert fold - 2e-3 < nu_dep < fold


# --- 8. driven_fold_tune --------------------------------------------------------------
@pytest.mark.parametrize("k3l", [-2000.0, 2000.0])
def test_driven_fold_tune_inverts_driven_fold_kick(k3l: float) -> None:
    """``driven_fold_kick`` at the returned tune is the dipole's kick, to 1e-9; the tune is
    on the pulling side (below the tune for ``k3l < 0``). ``None`` when the kick puts the
    fold beyond the model's reach."""
    kick = 2e-4
    lat = _ring(ACDipole(kick, 0.3, LAG, tune_rate=5e-6), k3l)  # swept: allowed here
    ft = driven_fold_tune(lat)["x"]
    assert ft is not None
    assert driven_fold_kick(_ring(_acd(1e-5, ft), k3l))["x"] == pytest.approx(kick, rel=1e-9)
    q = tunes(lat)[0] % 1.0
    assert (ft < q) if k3l < 0.0 else (ft > q)
    assert driven_fold_tune(_ring(_acd(2e-2, 0.3), K3L))["x"] is None


def test_the_bare_ring_has_a_fold_from_the_drift_alone() -> None:
    """Pre-committed as ``None`` and corrected: the exact drift detunes upward
    (``3 L gamma^2 / (16 pi)``, W4), so even with no octupole a fold exists, just above
    the tune."""
    ft = driven_fold_tune(_ring(_acd(2e-4, 0.3)))["x"]
    q = tunes(_ring())[0] % 1.0
    assert ft is not None and q < ft < q + 2e-3


def test_the_new_functions_keep_w4s_scope() -> None:
    with pytest.raises(ValueError, match="not swept"):
        autoresonance_threshold(_ring(_acd(2e-4, 0.3), K3L))
    swept = _ring(ACDipole(2e-4, 0.3, tune_rate=-1e-6), K3L)
    with_sextupole = Lattice([*swept.elements, ThinSextupole(1.0)], swept.ref)
    for fn in (autoresonance_threshold, driven_fold_tune):
        with pytest.raises(NotImplementedError):
            fn(with_sextupole)
