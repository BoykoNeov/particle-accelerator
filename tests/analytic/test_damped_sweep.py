"""The damped sweep (W6 part c): with a damper, the hysteresis loop closes.

W5 found no loop: undamped, a particle thrown off the small state lands on no steady
state. With a :class:`~accsim.elements.damper.Damper` in the ring it settles on the large
one, so a slow sweep of the drive tune traces the textbook loop:

- swept **toward** the tune, the particle rides the small state and **jumps up** onto the
  large one at the small state's fold (:func:`accsim.driven_jump_tunes`' first tune);
- swept **back**, it rides the large state and **drops** onto the small one where the
  large state ends (the second tune), a fold that only damping makes.

The truth is the exact driven map with the damper applied, one particle per column of a
batch, started from rest (the damping removes the switch-on transient within a few hundred
turns). The jumps are detected as the demodulated swing (a 40-turn window, centred)
crossing 5 mm; the averaged equation, read with the same rule, is what they are compared
with, because the particle needs ``~1 / Gamma = 333`` turns to settle and the drive moves
as far as the overshoot meanwhile: the rule is part of the measurement.

Fixture: W4's split-tune FODO, ``k3l = -2000`` at the dipole, a damper right after it,
``Gamma = 0.003``.
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import ai_zeros
from test_ac_dipole_sweep import NU_UP, _start, _turn_at, _up_sweep
from test_damped_response import GAMMA, NU17, _exact, _fold_kicks, _lat, _rate

from _w4_driven_orbits import IDX, LAG, _acd, _ring
from accsim import PX, ACDipole, Damper, X, closed_twiss, driven_states, propagate_twiss
from accsim.twiss import _DampedFolds, _DampedResponseCurve, _ResponseCurve, _states_on

K3L = -2000.0
TP = 2.0 * math.pi
WINDOW = 40  # [turns] the demodulation window
THRESHOLD = 5e-3  # [m] a jump is the demodulated swing crossing this
KICK_LOOP = 1.99e-5  # gate 11: bistable between 0.3330 and 0.3414 at 17/50's damping
UP_RATES = (4e-6, 2e-6, 1e-6)
DOWN_RATES = (1e-6, 5e-7, 2.5e-7)
NU_BEYOND = Fraction(43, 125)  # 0.344: past the bistable region, one state


# --- shared machinery ---------------------------------------------------------------------
def _damped_sweep(dipoles: list[ACDipole], rate: float, z0: np.ndarray, n_turns: int) -> np.ndarray:
    """``x`` at the dipole on every turn, one column per dipole, each driving its own
    particle through the exact map of the fixture with a damper of ``rate`` right after
    the dipole (as the turn loops apply it: ``p *= 1 - rate``)."""
    lat = _lat(dipoles[0].amplitude, NU17, gamma=0.0)  # the structure; the rate is below
    assert isinstance(lat.elements[IDX + 1], Damper)
    keep = 1.0 - rate
    s = np.empty((6, len(dipoles)))
    s[:] = np.asarray(z0, dtype=float).reshape(6, -1)
    xd = np.empty((n_turns, len(dipoles)))
    for turn in range(n_turns):
        for k, elem in enumerate(lat.elements):
            if k == IDX:
                xd[turn] = s[X]
                s[PX] += [d.drive_kick(turn, lat.ref)[PX] for d in dipoles]
            elif k == IDX + 1:
                s[PX] *= keep
            else:
                s = elem.track(s, lat.ref)
    return xd


def _demodulated(x: np.ndarray, d: ACDipole) -> tuple[np.ndarray, np.ndarray]:
    """``(centre turns, in-phase + i quadrature)`` on a sliding ``WINDOW``-turn window,
    against the drive's own swept phase: a least-squares fit of ``u sin + v cos`` per
    window (W5's ``_demod``, vectorised). Projecting with ``2 / WINDOW`` instead is biased
    by up to 3.5% here: 40 turns hold 13.6 cycles at 0.34, not a whole number."""
    n = np.arange(len(x))
    ph = TP * (d.tune * n + 0.5 * d.tune_rate * n * n + d.lag)
    s, c = np.sin(ph), np.cos(ph)

    def window_sum(f: np.ndarray) -> np.ndarray:
        cum = np.cumsum(np.concatenate([[0.0], f]))
        return cum[WINDOW:] - cum[:-WINDOW]

    xs, xc = window_sum(x * s), window_sum(x * c)
    ss, cc, sc = window_sum(s * s), window_sum(c * c), window_sum(s * c)
    det = ss * cc - sc * sc
    u = (cc * xs - sc * xc) / det
    v = (ss * xc - sc * xs) / det
    return n[: len(u)] + WINDOW // 2, u + 1j * v


def _ring_overshoot(x: np.ndarray, d: ACDipole) -> float:
    """How far past ``17/50`` the jump is detected: up, where the swing first exceeds
    ``THRESHOLD``; down, just after it last does."""
    n, z = _demodulated(x, d)
    big = np.flatnonzero(np.abs(z) > THRESHOLD)
    assert len(big)
    nu = d.tune + d.tune_rate * n
    if d.tune_rate > 0.0:
        return float(nu[big[0]] - NU17)
    return float(NU17 - nu[big[-1] + 1])


@functools.cache
def _batch() -> tuple[list[ACDipole], np.ndarray]:
    """Gates 11-13 in one batch, all from rest: the loop at ``KICK_LOOP`` and ``1e-6`` (up
    from 0.33, down from 0.35; columns 0, 1), the jumps with each kick at the exact fold of
    17/50 (2-4 up, 5-7 down), and the loop again at ``5e-7`` (8, 9)."""
    folds = _fold_kicks()
    dipoles = [
        ACDipole(KICK_LOOP, 0.33, LAG, tune_rate=1e-6),
        ACDipole(KICK_LOOP, 0.35, LAG, tune_rate=-1e-6),
        *(ACDipole(folds["small_exact"], 0.33, LAG, tune_rate=r) for r in UP_RATES),
        *(ACDipole(folds["large_exact"], 0.35, LAG, tune_rate=-r) for r in DOWN_RATES),
        ACDipole(KICK_LOOP, 0.33, LAG, tune_rate=5e-7),
        ACDipole(KICK_LOOP, 0.35, LAG, tune_rate=-5e-7),
    ]
    n_turns = math.ceil(0.016 / min(DOWN_RATES)) + WINDOW
    return dipoles, _damped_sweep(dipoles, _rate(GAMMA), np.zeros(6), n_turns)


# --- the averaged equation ----------------------------------------------------------------
@functools.cache
def _averaged() -> tuple[float, float, float]:
    """``(a, beta, q_d)``: the natural detuning ``dQ/dJ`` and beta at the dipole, and the
    damped tune (fractional) the averaged equation's distance ``d`` is measured from."""
    lat = _ring(_acd(1e-5, float(NU17)), K3L)
    a = float(_ResponseCurve(lat, "test")._optics(np.array([0.0]))[2][0])
    beta = propagate_twiss(lat, closed_twiss(lat))[IDX].beta_x
    return a, beta, _DampedFolds(_lat(1e-5, NU17), "test").q_d % 1.0


def _cubic(n: float, d: float) -> float:
    """``eps^2`` of the averaged equation's steady state at ``|c|^2 = n``, distance ``d``."""
    a = _averaged()[0]
    return n * ((TP * (d + a * n)) ** 2 + GAMMA**2)


def _turning(d: float) -> tuple[float, float]:
    """``|c|^2`` of the small state's fold (``eps`` maximal) and the large state's end."""
    a = _averaged()[0]
    r = np.roots([3.0 * (TP * a) ** 2, 4.0 * TP**2 * a * d, (TP * d) ** 2 + GAMMA**2])
    assert np.all(np.abs(r.imag) == 0.0)
    lo, hi = sorted(r.real)
    return float(lo), float(hi)


def _steady(d: float, eps: float) -> list[float]:
    a = _averaged()[0]
    r = np.roots([(TP * a) ** 2, 2.0 * TP**2 * a * d, (TP * d) ** 2 + GAMMA**2, -eps * eps])
    return sorted(float(x.real) for x in r if abs(x.imag) < 1e-9 * abs(x))


def _rhs_factory(eps: float, d0: float, d_rate: float):
    """``dc/dn = 2 pi i (d + a |c|^2) c - eps - Gamma c``, ``d = d0 + d_rate n``: W5's
    rotating-wave equation with the damper's ``-Gamma c``."""
    a = _averaged()[0]

    def rhs(n: float, y: np.ndarray) -> list[float]:
        c = y[0] + 1j * y[1]
        dc = 1j * TP * (d0 + d_rate * n + a * abs(c) ** 2) * c - eps - GAMMA * c
        return [dc.real, dc.imag]

    return rhs


def _averaged_crossings(
    eps: float, d0: float, d_rate: float, d_end: float, c0: complex, amp: float, direction: int
) -> np.ndarray:
    """The distances ``d`` at which the swing ``sqrt(2 beta) |c|`` crosses ``amp``."""
    beta = _averaged()[1]

    def crossing(n: float, y: np.ndarray) -> float:
        return y[0] ** 2 + y[1] ** 2 - amp**2 / (2.0 * beta)

    crossing.direction = direction
    sol = solve_ivp(
        _rhs_factory(eps, d0, d_rate),
        (0.0, (d_end - d0) / d_rate),
        [c0.real, c0.imag],
        events=crossing,
        rtol=1e-10,
        atol=1e-14,
        method="DOP853",
        max_step=20.0,
    )
    return d0 + d_rate * sol.t_events[0]


def _averaged_overshoot(rate: float, up: bool) -> float:
    """The averaged equation's overshoot under the ring's rule: its own fold pinned at the
    ring's distance ``q_d - 17/50``, started from rest as far away as the ring's sweep."""
    q_d = _averaged()[2]
    dist = q_d - float(NU17)
    n_small, n_large = _turning(dist)
    eps = math.sqrt(_cubic(n_small if up else n_large, dist))
    if up:  # nu rises from 0.33: d falls from q_d - 0.33
        d = _averaged_crossings(eps, q_d - 0.33, -rate, dist - 0.006, 0j, THRESHOLD, 1)
        return float(dist - d[0])
    d = _averaged_crossings(eps, q_d - 0.35, rate, dist + 0.006, 0j, THRESHOLD, -1)
    return float(d[-1] - dist)


# --- 11. the loop closes --------------------------------------------------------------------
def _exact_states(kick: float, nu: Fraction) -> list[complex]:
    """The exact stable period-q orbits, seeded at the predicted ones, by swing size."""
    lat = _lat(kick, nu)
    curve = _DampedResponseCurve(lat, "test")
    out = []
    for g, state in _states_on(curve):
        if state.stable:
            ex, J = _exact(lat, curve, g, nu)
            assert ex is not None and np.abs(np.linalg.eigvals(J)).max() < 1.0
            out.append(ex)
    return sorted(out, key=abs)


@pytest.mark.slow
def test_the_loop_closes() -> None:
    """At ``KICK_LOOP``: passing 17/50 the up-sweep is on the exact small orbit and the
    down-sweep on the exact large one, each within 3% (complex swing) at ``1e-6``; past the
    region (0.344) both are on the one exact state. Nothing is lost.

    What a swept drive does to the swing is lag the steady state by an amount
    proportional to the rate, in opposite directions going up and coming down. So each
    sweep's miss halves with the rate (ratios 1.8-2.2), and the mean of the two directions
    is the exact state to 1% at 0.344. (Pre-committed: each direction within 1% there; the
    lag at ``1e-6`` is 2.5%.) Measured at ``1e-6``: 1.2% and 2.6% at 17/50; 2.5% and 2.7%
    at 0.344, mean 0.11%; halving ratios 1.87-2.00."""
    dipoles, xd = _batch()
    small, large = _exact_states(KICK_LOOP, NU17)
    (beyond,) = _exact_states(KICK_LOOP, NU_BEYOND)
    assert len(driven_states(_lat(KICK_LOOP, NU_BEYOND))["x"]) == 1
    assert abs(large) > 3.0 * abs(small)  # two different states: the loop is open there

    def at(col: int, nu: Fraction) -> complex:
        d = dipoles[col]
        n, z = _demodulated(xd[:, col], d)
        return complex(z[int(np.argmin(np.abs(d.tune + d.tune_rate * n - float(nu))))])

    miss = {}
    for up, down, rate in ((0, 1, 1e-6), (8, 9, 5e-7)):
        for col in (up, down):
            d = dipoles[col]
            end = _turn_at(d, 0.346 if d.tune_rate > 0.0 else 0.33)
            assert np.all(np.isfinite(xd[:end, col]))
        miss[rate] = (
            abs(at(up, NU17) - small) / abs(small),
            abs(at(down, NU17) - large) / abs(large),
            abs(at(up, NU_BEYOND) - beyond) / abs(beyond),
            abs(at(down, NU_BEYOND) - beyond) / abs(beyond),
        )
        mean = 0.5 * (at(up, NU_BEYOND) + at(down, NU_BEYOND))
        assert abs(mean - beyond) < 0.01 * abs(beyond), abs(mean - beyond) / abs(beyond)
    assert all(m < 0.03 for m in miss[1e-6][:2]), miss[1e-6]
    ratios = [a / b for a, b in zip(miss[1e-6], miss[5e-7], strict=True)]
    assert all(1.8 < r < 2.2 for r in ratios), ratios


# --- 12, 13. the jumps converge -------------------------------------------------------------
@pytest.mark.slow
def test_the_jump_up_converges_as_the_averaged_equation_predicts() -> None:
    """Kick at the exact small fold of 17/50: the swing first exceeds 5 mm past 17/50 by an
    overshoot that falls with the rate and is within 5% of the averaged equation's.
    Measured 0.05-0.26% (6.20, 3.90, 2.49e-4 against 6.18, 3.90, 2.49e-4). The probes
    measured 1.0-3.6% with a plain ``2 / WINDOW`` projection; its bias was most of that."""
    dipoles, xd = _batch()
    over = [_ring_overshoot(xd[:, 2 + k], dipoles[2 + k]) for k in range(len(UP_RATES))]
    want = [_averaged_overshoot(r, up=True) for r in UP_RATES]
    assert all(o > 0.0 for o in over)
    assert over[0] > over[1] > over[2]
    for o, w in zip(over, want, strict=True):
        assert abs(o / w - 1.0) < 0.05, (o, w)


@pytest.mark.slow
def test_the_drop_converges_as_the_averaged_equation_predicts() -> None:
    """Kick at the exact large state's end at 17/50: the swing last exceeds 5 mm past 17/50
    (below it) by an overshoot that falls with the rate and is within 5% of the averaged
    equation's. Measured 0.22-0.28% (5.85, 4.04, 2.70e-4 against 5.84, 4.03, 2.69e-4)."""
    dipoles, xd = _batch()
    base = 2 + len(UP_RATES)
    over = [_ring_overshoot(xd[:, base + k], dipoles[base + k]) for k in range(len(DOWN_RATES))]
    want = [_averaged_overshoot(r, up=False) for r in DOWN_RATES]
    assert all(o > 0.0 for o in over)
    assert over[0] > over[1] > over[2]
    for o, w in zip(over, want, strict=True):
        assert abs(o / w - 1.0) < 0.05, (o, w)


# --- 14. the power, on the averaged equation ------------------------------------------------
def _fold_reduction(eps: float) -> tuple[float, float, float, float]:
    """At the averaged equation's small fold for ``eps``: ``(d_f, fold swing, large swing,
    |k1 k2|)``, where near the fold ``X' = k1 alpha n + k2 X^2`` on the centre manifold
    (``X`` along the Jacobian's null vector, ``d = d_f - alpha n``)."""
    a, beta, _ = _averaged()
    onset = math.sqrt(3.0) * GAMMA / TP  # below it the cubic has no fold at all
    d_f = brentq(
        lambda d: math.sqrt(_cubic(_turning(d)[0], d)) - eps,
        onset * (1.0 + 1e-9),
        0.05,
        xtol=1e-17,
    )
    n1 = _turning(d_f)[0]
    c = eps / (1j * TP * (d_f + a * n1) - GAMMA)
    # dF = da dc + db conj(dc), as a real 2x2 on (Re, Im)
    da = 1j * TP * d_f + 2j * TP * a * abs(c) ** 2 - GAMMA
    db = 1j * TP * a * c * c
    J = np.array([[(da + db).real, (1j * (da - db)).real], [(da + db).imag, (1j * (da - db)).imag]])
    w, V = np.linalg.eig(J)
    i0 = int(np.argmin(np.abs(w)))
    assert abs(w[i0]) < 1e-12 and abs(w[1 - i0] + 2.0 * GAMMA) < 1e-12  # trace -2 Gamma
    e = V[:, i0].real
    wl, U = np.linalg.eig(J.T)
    left = U[:, int(np.argmin(np.abs(wl)))].real
    left = left / (left @ e)
    ec = e[0] + 1j * e[1]
    second = 2j * TP * a * (np.conj(c) * ec * ec + 2.0 * c * abs(ec) ** 2)  # D^2 F (e, e)
    k2 = 0.5 * (left @ np.array([second.real, second.imag]))
    by_d = 1j * TP * c  # dF / dd
    k1 = -(left @ np.array([by_d.real, by_d.imag]))
    n_large = _steady(d_f, eps)[-1]
    return d_f, math.sqrt(2.0 * beta * n1), math.sqrt(2.0 * beta * n_large), abs(k1 * k2)


def _airy() -> float:
    """``Omega_0``, the first zero of ``Ai(-x)``: computed, not recalled."""
    return -float(ai_zeros(1)[0][0])


def _passage(eps: float, rate: float, amp: float) -> float:
    """The averaged equation swept up through its small fold, from its small state 40
    Airy overshoots before it: the overshoot past ``d_f`` where the swing first exceeds
    ``amp``."""
    d_f, _, _, k12 = _fold_reduction(eps)
    offset = 40.0 * _airy() * rate ** (2.0 / 3.0) / k12 ** (1.0 / 3.0)
    d0 = d_f + offset
    n0 = _steady(d0, eps)[0]
    c0 = eps / (1j * TP * (d0 + _averaged()[0] * n0) - GAMMA)
    d = _averaged_crossings(eps, d0, -rate, d_f - offset, c0, amp, 1)
    return float(d_f - d[0])


def test_the_passage_goes_as_the_rate_to_the_two_thirds() -> None:
    """A damped fold reduces to first order, ``X' = k1 alpha n + k2 X^2`` (the other
    direction relaxes at ``2 Gamma``), whose blow-up time is the first zero of ``Ai(-x)``,
    ``Omega_0``: overshoot ``-> Omega_0 alpha^(2/3) / |k1 k2|^(1/3)``, not W5's undamped
    ``alpha^(4/5)``. At kick ``1.99e-5``: local exponents over six halvings from ``1e-6``
    lie in 0.62-0.72 and each more than 0.1 from 4/5 (measured 0.653-0.680, a mid-way
    threshold); convergence is slow (a transit term near ``alpha log(1/alpha)``). At
    ``1.95e-9`` the overshoot over ``alpha^(2/3)`` brackets the Airy value (3.419) between
    a threshold 10% of the way from the fold's swing to the large state's (3.357) and one
    90% of the way (4.195); those two thresholds were chosen after the run (at 20%
    the low side is already 3.72). No tolerance on the Airy value itself."""
    beta = _averaged()[1]
    eps = KICK_LOOP * math.sqrt(2.0 * beta) / 4.0  # W5's eps = theta sqrt(2 beta) / 4
    _, r_f, r_l, k12 = _fold_reduction(eps)
    omega0 = _airy()
    assert omega0 == pytest.approx(2.33811, abs=1e-5)
    airy = omega0 / k12 ** (1.0 / 3.0)

    rates = [1e-6 / 2**k for k in range(7)]
    over = [_passage(eps, r, 0.5 * (r_f + r_l)) for r in rates]
    local = [math.log2(a / b) for a, b in zip(over, over[1:], strict=False)]
    assert all(0.62 < p < 0.72 for p in local), local
    assert all(abs(p - 0.8) > 0.1 for p in local), local

    alpha = 1.95e-9
    low = _passage(eps, alpha, r_f + 0.1 * (r_l - r_f)) / alpha ** (2.0 / 3.0)
    high = _passage(eps, alpha, r_f + 0.9 * (r_l - r_f)) / alpha ** (2.0 / 3.0)
    assert low < airy < high, (low, airy, high)


# --- 15. the control --------------------------------------------------------------------------
@pytest.mark.slow
def test_at_rate_zero_w5s_up_sweep_is_reproduced_bit_for_bit() -> None:
    """With a ``rate = 0`` damper in the ring, W5's up-sweep (its fold kick, its start on
    the exact small orbit at 3/10) is reproduced bit for bit, and W5's gate 4 holds: past
    the fold, at 0.36, the particle is lost or swings more than three times the one steady
    state's swing. Undamped, it lands on nothing."""
    dipoles, xd_w5 = _up_sweep()  # W5's own run (shared, cached)
    kick = dipoles[0].amplitude
    xd = _damped_sweep(dipoles, 0.0, _start(kick, NU_UP, "small"), len(xd_w5))
    assert np.array_equal(xd, xd_w5, equal_nan=True)
    (only,) = driven_states(_ring(_acd(kick, 0.36), K3L))["x"]
    for p, d in enumerate(dipoles):
        n = _turn_at(d, 0.36)
        window = xd[n - 200 : n, p]
        assert not np.all(np.isfinite(window)) or np.max(np.abs(window)) > 3.0 * abs(only.amplitude)
