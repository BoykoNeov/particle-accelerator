"""The island chain that ends a lock early (W8).

W7 found that on W4's ring a damped autoresonant lock is usually lost where the drive tune
reaches 1/3, and called it a resonance of the driven ring. It is the **free** ring's own:

- with no drive and no damper, Newton on the 3-turn map finds six islands (two stable and two
  unstable period-3 orbits), the even-order resonance ``6Q = 8`` an octupole drives at second
  order in ``k3l``; at 3/10 there are ten;
- the lock ends inside that chain, and **whether** it survives depends on the drive's phase as
  it meets it. Survival falls to none as the sweep slows.

A reduced model reproduces the surviving fraction: W5/W7's rotating-wave equation with one
island harmonic added,

    dc/dn = 2 pi i (q_d - nu(n) + D(|c|^2)) c - eps - Gamma c + 3 i (U / J_r^3) c*^5 e^(-i chi(n)),

``chi = 6 Phi_drive - 4 pi n`` (static when the particle's tune is 1/3). It takes two exact
measurements of the free ring and nothing fitted to the lock: the tracked detuning curve
``D(J)`` and the chain's libration ``omega``, through ``U = omega^2 / (2 pi |D'(J_r)| m^2)``.

Fixture: W4's split-tune FODO, ``k3l = -2000``, with W7's columns (the damper right after the
dipole, ``gamma`` held fixed as the rate falls). Survival is read W7's way, ``|Psi|^2 / tau >
1/2``, on a 200-turn least-squares window at two drive tunes past the chain.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from scipy.optimize import brentq
from test_damped_autoresonance import K3L, TP, _Column, _drive, _lat, _optics, _q_d

from _w4_driven_orbits import IDX, _fixed_points, _ring
from accsim import PX, ACDipole, X, closed_twiss

FREE = _ring(None, K3L)  # no drive, no damper
M = 6  # islands at 1/3
GAMMA = 0.05
FS = (1.5, 2.0)
RATES = (-4e-6, -2e-6, -1e-6, -5e-7, -2.5e-7)
NL = 64  # lags over [0, 1/6)
READS = (0.322, 0.318)  # drive tunes past the chain
READ_W = 200  # [turns] the survival read's window
PEAK_W = 40  # [turns] W7's demodulation window, for the loss tune
SYM = (-1e-6, 1.5)  # gate 4's configuration
SYM_TUNE = 0.34  # gate 4: negation checked up to this drive tune


# --- the free ring ------------------------------------------------------------------------
@functools.cache
def _free_optics() -> tuple[float, float]:
    tw = closed_twiss(FREE)
    return float(tw.beta_x), float(tw.alpha_x)


def _launch(J: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """``(6, n)`` states at the lattice start, Courant-Snyder action ``J`` and phase ``phi``."""
    b, al = _free_optics()
    J, phi = np.broadcast_arrays(np.asarray(J, float), np.asarray(phi, float))
    s = np.zeros((6, J.size))
    s[X] = np.sqrt(2.0 * b * J) * np.cos(phi)
    s[PX] = (-np.sqrt(2.0 * J / b) * np.sin(phi)) - al * s[X] / b
    return s


def _cs(z: np.ndarray) -> tuple[float, float]:
    """``(J, phi)`` of ``(x, px)`` at the lattice start."""
    b, al = _free_optics()
    xn, pn = z[0] / math.sqrt(b), (al * z[0] + b * z[1]) / math.sqrt(b)
    return 0.5 * (xn * xn + pn * pn), math.atan2(-pn, xn)


def _free_tunes(s: np.ndarray, n_turns: int) -> np.ndarray:
    """The free ring's tune of each column: the mean phase advance at the lattice start."""
    b, al = _free_optics()
    total, prev = np.zeros(s.shape[1]), None
    for _ in range(n_turns):
        ph = np.arctan2(-(al * s[X] + b * s[PX]), s[X])
        if prev is not None:
            total += (ph - prev) % TP
        prev = ph
        for e in FREE.elements:
            s = e.track(s, FREE.ref)
    return 1.0 + total / (n_turns - 1) / TP


def _one_turn(z: np.ndarray) -> np.ndarray:
    s = np.zeros((6, 1))
    s[X, 0], s[PX, 0] = z
    for e in FREE.elements:
        s = e.track(s, FREE.ref)
    return np.array([s[X, 0], s[PX, 0]])


@functools.cache
def _chain(q: int, J_guess: float) -> tuple[tuple[np.ndarray, float], ...]:
    """The free ring's period-``q`` fixed points near ``J_guess``: ``((z, trace), ...)``. The
    origin, which a seed can also fall onto, is not part of a chain and is dropped."""
    seeds = [
        _launch(J, phi)[[X, PX], 0]
        for J in J_guess * np.array([0.9, 1.0, 1.1])
        for phi in np.linspace(0.0, TP, 36, endpoint=False)
    ]
    found = _fixed_points(FREE, q, seeds)
    return tuple((z, float(np.trace(jac))) for z, jac in found if _cs(z)[0] > 0.25 * J_guess)


def _cycles(points: list[np.ndarray]) -> list[list[int]]:
    """The fixed points grouped into orbits by the one-turn map (each image must be one of
    the points, to ``1e-11``)."""
    image = []
    for z in points:
        w = _one_turn(z)
        d = [float(np.max(np.abs(w - p))) for p in points]
        k = int(np.argmin(d))
        assert d[k] < 1e-11
        image.append(k)
    cycles, seen = [], set()
    for k in range(len(points)):
        if k in seen:
            continue
        cyc, j = [], k
        while j not in cyc:
            cyc.append(j)
            j = image[j]
        assert j == k  # a permutation: every orbit closes on itself
        seen.update(cyc)
        cycles.append(cyc)
    return cycles


def _index_of(points: list[np.ndarray], z: np.ndarray, tol: float) -> int | None:
    d = [float(np.max(np.abs(z - p))) for p in points]
    k = int(np.argmin(d))
    return k if d[k] < tol else None


@functools.cache
def _omega() -> float:
    """The 1/3 chain's libration per turn: ``arccos(tr / 2) / 3`` of the stable orbit."""
    traces = [t for _, t in _chain(3, 2.08e-5) if abs(t) < 2.0]
    return math.acos(float(np.mean(traces)) / 2.0) / 3.0


@functools.cache
def _detuning() -> tuple[float, np.poly1d]:
    """``(Q_0, D)``: the free ring's small-amplitude tune and its tracked detuning curve
    ``D(J) = Q(J) - Q_0``, a quartic fitted away from the 1/3 and 3/10 chains (launches along
    ``phi = 0``, 4000 turns)."""
    x0 = np.linspace(0.5e-3, 40e-3, 400)[::4]
    b, _ = _free_optics()
    J = x0 * x0 / (2.0 * b) * (1.0 + _free_optics()[1] ** 2)  # CS action of (x0, 0)
    s = np.zeros((6, x0.size))
    s[X] = x0
    with np.errstate(all="ignore"):
        Q = _free_tunes(s, 4000) - 1.0
    q0 = float(Q[0])
    ok = np.isfinite(Q) & ((J < 1.85e-5) | ((J > 2.4e-5) & (J < 8.5e-5)))
    return q0, np.poly1d(np.polyfit(J[ok], Q[ok] - q0, 4))


# --- 1. the chain is the free ring's ------------------------------------------------------
def test_the_one_third_chain_has_six_islands() -> None:
    """Twelve fixed points of the free 3-turn map near ``J = 2e-5``: six stable (trace
    ``1.99389``) and six unstable (``2.00597``), two orbits of each, mapped onto each other by
    ``x -> -x``. Six islands: ``6Q = 8``, an even order, second order in ``k3l``."""
    chain = _chain(3, 2.08e-5)
    assert len(chain) == 12
    stable = [z for z, t in chain if abs(t - 1.99389) < 1e-5]
    unstable = [z for z, t in chain if abs(t - 2.00597) < 1e-5]
    assert len(stable) == len(unstable) == 6
    for kind in (stable, unstable):
        assert sorted(len(c) for c in _cycles(kind)) == [3, 3]
        for z in kind:
            assert _index_of(kind, -z, 1e-12) is not None


def test_the_three_tenths_chain_has_ten_islands() -> None:
    """Twenty fixed points of the free 10-turn map near ``J = 9.5e-5``: one stable period-10
    orbit (trace ``1.02002``) and one unstable (``3.1051``), each its own mirror image."""
    chain = _chain(10, 9.5e-5)
    assert len(chain) == 20
    stable = [z for z, t in chain if abs(t - 1.02002) < 1e-4]
    unstable = [z for z, t in chain if abs(t - 3.1051) < 1e-3]
    assert len(stable) == len(unstable) == 10
    for kind in (stable, unstable):
        (cycle,) = _cycles(kind)
        assert len(cycle) == 10
        for z in kind:
            assert _index_of(kind, -z, 1e-12) is not None  # the orbit is its own mirror


# --- 2. two exact widths ------------------------------------------------------------------
@pytest.mark.slow
def test_the_plateau_width_is_the_libration_width() -> None:
    """Along the ray through a stable fixed point, the tracked free tune is 1/3 (to ``3e-5``)
    over a band of ``J``; times the detuning curve's slope at the chain, its half-width in
    tune is within 6% of the pendulum's ``omega / (pi m)`` (measured 2.4%)."""
    stable = [z for z, t in _chain(3, 2.08e-5) if abs(t) < 2.0]
    J_s, phi_s = min((_cs(z) for z in stable), key=lambda jp: abs(jp[1]))
    J = np.linspace(1.6e-5, 2.6e-5, 501)
    Q = _free_tunes(_launch(J, phi_s), 6000)
    on = np.abs(Q - 4.0 / 3.0) < 3e-5
    lo, hi = float(J[on].min()), float(J[on].max())
    assert lo < J_s < hi
    q0, D = _detuning()
    J_r = brentq(lambda j: q0 + D(j) - 1.0 / 3.0, 1e-6, 4e-5)
    width = abs(D.deriv()(J_r)) * (hi - lo) / 2.0
    assert width == pytest.approx(_omega() / (math.pi * M), rel=0.06)


# --- 3-5. the ring ------------------------------------------------------------------------
def _columns() -> list[tuple[tuple, _Column]]:
    """Gate 5's grid (``(rate, f)`` x 64 lags over ``[0, 1/6)``) and gate 4's shifted lags
    (``+1/2``, ``+1/6``) on one configuration: ``[(key, column), ...]``."""
    out = []
    for f in FS:
        for r in RATES:
            out += [((r, f, 0.0, k), _Column(r, GAMMA, f)) for k in range(NL)]
    for shift in (0.5, 1.0 / 6.0, "-kick"):
        out += [((*SYM, shift, k), _Column(SYM[0], GAMMA, SYM[1])) for k in range(NL)]
    for (_, _, shift, k), c in out:
        d = c.dipole
        lag = k / (6.0 * NL) + (0.0 if shift == "-kick" else shift)
        amp = -d.amplitude if shift == "-kick" else d.amplitude
        c.dipole = ACDipole(amp, d.tune, lag, ramp=d.ramp, tune_rate=d.tune_rate)
    return out


def _turn_at(c: _Column, nu: float) -> int:
    return c.turn((nu - _q_d(c.Gamma, c.k3l)) * c.kap / c.rate)


@functools.cache
def _ring_batch() -> dict:
    """One sweep of every column: the survival reads, the loss tune (the peak of W7's 40-turn
    demodulated swing between drive tunes 0.342 and 0.318), and gate 4's negation."""
    keyed = _columns()
    keys = [k for k, _ in keyed]
    cols = [c for _, c in keyed]
    nc = len(cols)
    beta, a = _optics(K3L)
    reads = np.array([[_turn_at(c, nu) for nu in READS] for c in cols])
    lo = reads - READ_W // 2
    t_hi = np.array([_turn_at(c, 0.342) for c in cols])
    t_lo = np.array([_turn_at(c, 0.318) for c in cols])
    t_end = np.maximum(lo[:, -1] + READ_W, t_lo + 1)  # a column's drive stops once read
    n_turns = int(t_end.max())
    d = [c.dipole for c in cols]
    tune = np.array([x.tune for x in d])
    trate = np.array([x.tune_rate for x in d])
    lag = np.array([x.lag for x in d])
    # gate 4: column k of the base SYM grid against its +1/2 partner
    base = {k[3]: i for i, k in enumerate(keys) if k[:3] == (*SYM, 0.0)}
    half = {k[3]: i for i, k in enumerate(keys) if k[:3] == (*SYM, 0.5)}
    flip = {k[3]: i for i, k in enumerate(keys) if k[:3] == (*SYM, "-kick")}
    pa = np.array([base[k] for k in range(NL)])
    pb = np.array([half[k] for k in range(NL)])
    pf = np.array([flip[k] for k in range(NL)])
    t_sym = _turn_at(cols[pa[0]], SYM_TUNE)
    sym_diff, sym_size, flip_exact = 0.0, 0.0, True

    buf = np.zeros((len(READS), READ_W, nc))
    ring = np.zeros((PEAK_W, 5, nc))  # the last 40 turns of x s, x c, s s, c c, s c
    sums = np.zeros((5, nc))
    peak = np.zeros(nc)
    peak_nu = np.full(nc, np.nan)
    keep = np.array([math.exp(-2.0 * c.Gamma) for c in cols])
    lat = _lat(ACDipole(1e-9, 0.3), 0.0)
    s = np.zeros((6, nc))
    for turn in range(n_turns):
        for k, elem in enumerate(lat.elements):
            if k == IDX:
                x = s[X]
                for j in range(len(READS)):
                    m = (turn >= lo[:, j]) & (turn < lo[:, j] + READ_W)
                    if m.any():
                        buf[j, turn - lo[m, j], np.nonzero(m)[0]] = x[m]
                ph = TP * (tune * turn + 0.5 * trate * turn * turn + lag)
                sn, cs = np.sin(ph), np.cos(ph)
                new = np.array([x * sn, x * cs, sn * sn, cs * cs, sn * cs])
                sums += new - ring[turn % PEAK_W]
                ring[turn % PEAK_W] = new
                if turn >= PEAK_W:
                    xs, xc, ss, cc, sc = sums
                    det = ss * cc - sc * sc
                    amp = np.hypot((cc * xs - sc * xc) / det, (ss * xc - sc * xs) / det)
                    centre = turn - PEAK_W + 1 + PEAK_W // 2
                    win = (centre >= t_hi) & (centre <= t_lo) & (amp > peak)
                    peak[win] = amp[win]
                    peak_nu[win] = tune[win] + trate[win] * centre
                if turn <= t_sym:
                    sym_diff = max(sym_diff, float(np.max(np.abs(x[pa] + x[pb]))))
                    sym_size = max(sym_size, float(np.max(np.abs(x[pa]))))
                flip_exact &= bool(np.array_equal(x[pf], -x[pa], equal_nan=True))
                s[PX] += np.where(turn <= t_end, _drive(cols, turn), 0.0)
            elif k == IDX + 1:
                s[PX] *= keep
            else:
                s = elem.track(s, lat.ref)

    r2 = np.zeros((len(READS), nc))
    for i, c in enumerate(cols):
        x = c.dipole
        lam2 = c.kap / (TP * abs(a))
        for j in range(len(READS)):
            n = lo[i, j] + np.arange(READ_W)
            ph = TP * (x.tune * n + 0.5 * x.tune_rate * n * n + x.lag)
            A = np.column_stack([np.sin(ph), np.cos(ph)])
            u = buf[j, :, i]
            if not np.all(np.isfinite(u)):
                r2[j, i] = 0.0  # lost from the ring: lost from the lock
                continue
            uv = np.linalg.lstsq(A, u, rcond=None)[0]
            r2[j, i] = float(uv @ uv) / (2.0 * beta) / lam2 / c.tau(n.mean())
    phase_max = TP * (tune[pb[-1]] * t_sym + 0.5 * trate[pb[-1]] * t_sym * t_sym + lag[pb[-1]])
    return {
        "keys": keys, "r2": r2, "peak_nu": peak_nu,
        "sym_diff": sym_diff, "sym_size": sym_size, "flip_exact": flip_exact,
        "phase_ulp": float(np.spacing(abs(phase_max))),
    }  # fmt: skip


def _survivors(batch: dict, rate: float, f: float, shift: float = 0.0) -> np.ndarray:
    sel = [i for i, k in enumerate(batch["keys"]) if k[:3] == (rate, f, shift)]
    assert len(sel) == NL
    return batch["r2"][-1, sel] > 0.5


@pytest.mark.slow
def test_survival_falls_as_the_sweep_slows() -> None:
    """``gamma = 0.05``, 64 lags: survivors never rise as the rate halves, at least 56 at
    ``4e-6`` and at most 2 at ``2.5e-7`` (measured 60, 52, 43, 8, 0 at ``f = 1.5`` and 60, 43,
    24, 8, 1 at 2). The two reads agree on every column, and they are far apart: locked
    columns read over 0.6 of the backbone, lost ones under 0.05 (measured 0.91 and 0.000)."""
    b = _ring_batch()
    r2 = b["r2"]
    locked = r2 > 0.5
    assert np.array_equal(locked[0], locked[1])
    assert np.all(r2[-1][locked[-1]] > 0.6) and np.all(r2[-1][~locked[-1]] < 0.05)
    for f in FS:
        n = [int(_survivors(b, r, f).sum()) for r in RATES]
        assert all(x >= y for x, y in zip(n, n[1:], strict=False)), (f, n)
        assert n[0] >= 56 and n[-1] <= 2, (f, n)


@pytest.mark.slow
def test_no_lock_ends_past_the_chain() -> None:
    """On every configuration where at least 8 of 64 lose the lock, every loss tune (the peak
    of W7's demodulated swing) lies below the chain's outer edge ``1/3 + omega / (pi m)``
    (measured at most ``+0.51e-3``), and the median within ``omega / (pi m)`` of 1/3
    (measured within ``1.02e-3``). One-sided: a tail below the chain (to ``-3.5e-3``) is not
    gated, and the 40-turn read-out cannot tell a particle caught in the chain from a locked
    one, so part of that tail may be the read-out."""
    b = _ring_batch()
    half = _omega() / (math.pi * M)
    checked = 0
    for f in FS:
        for r in RATES:
            sel = [i for i, k in enumerate(b["keys"]) if k[:3] == (r, f, 0.0)]
            lost = [i for i in sel if b["r2"][-1, i] <= 0.5]
            if len(lost) < 8:
                continue
            nu = b["peak_nu"][lost]
            assert np.all(nu < 1.0 / 3.0 + half), (r, f, nu.max() - 1.0 / 3.0)
            assert abs(float(np.median(nu)) - 1.0 / 3.0) < half, (r, f)
            checked += 1
    assert checked >= 6


@pytest.mark.slow
def test_the_half_lag_symmetry_and_the_kinematic_sixth() -> None:
    """The octupole ring is symmetric under ``x -> -x``. A drive of the opposite sign gives
    the negated trajectory **bit for bit**, through the chain and past it. Lag ``l + 1/2``
    flips the drive too, but through its phase: the same outcome at all 64 lags, and the
    trajectories negatives of each other, until the drive reaches 0.34, to the phase's own
    round-off (the spacing of doubles at the largest phase, about ``1.5e-11`` rad at
    ``8.5e4``, times the swing; pre-committed ``1e-12``, measured ``3.6e-12``). Lag ``l + 1/6``
    also gives the same outcome at all 64 lags: ``l + 1/3`` is nearly a one-turn shift while
    the drive tune is near 1/3, and the damping erases the earlier difference. That is
    kinematics, not the chain's six islands."""
    b = _ring_batch()
    base = _survivors(b, *SYM)
    assert b["flip_exact"]
    assert np.array_equal(base, _survivors(b, *SYM, "-kick"))
    assert np.array_equal(base, _survivors(b, *SYM, 0.5))
    assert np.array_equal(base, _survivors(b, *SYM, 1.0 / 6.0))
    assert b["sym_diff"] < b["phase_ulp"] * b["sym_size"]


# --- 6. the reduced model -----------------------------------------------------------------
MODES = ("curve", "linear", "no-island")
STEP = 0.5  # [turns] the model's RK4 step (0.25 and 0.125 give the same survivors)


def _model_survivors(rate: float, h: float = STEP) -> dict[tuple[str, float], int]:
    """``{(mode, f): survivors of 64}`` on the reduced model, for one rate. ``mode``:
    ``"curve"`` (the tracked ``D``), ``"linear"`` (the small-amplitude ``a J`` in its place),
    ``"no-island"`` (``U = 0``). Columns start at rest, as the ring's do, and are read at the
    survival read's drive tune against the model's own backbone."""
    beta, a = _optics(K3L)
    _, D = _detuning()
    rows = [(mode, f) for mode in MODES for f in FS]
    cols = [_Column(rate, GAMMA, f) for _, f in rows for _ in range(NL)]
    lags = np.tile(np.arange(NL) / (6.0 * NL), len(rows))
    q_d = _q_d(cols[0].Gamma, K3L)
    lin = np.repeat([mode == "linear" for mode, _ in rows], NL)
    J_r = np.where(lin, (1.0 / 3.0 - q_d) / a, brentq(lambda j: q_d + D(j) - 1.0 / 3.0, 1e-6, 4e-5))
    J_bb = np.where(
        lin, (q_d - READS[-1]) / abs(a), brentq(lambda j: q_d + D(j) - READS[-1], 1e-7, 8e-5)
    )
    slope = np.where(lin, abs(a), abs(D.deriv()(J_r)))
    U = _omega() ** 2 / (TP * slope * M * M)
    U[np.repeat([mode == "no-island" for mode, _ in rows], NL)] = 0.0
    K3 = 3j * U / J_r**3
    amp = np.array([c.dipole.amplitude for c in cols]) * math.sqrt(2.0 * beta) / 4.0
    tune0 = cols[0].dipole.tune
    r2 = float(cols[0].dipole.ramp[1])
    Gam = cols[0].Gamma
    coef = D.coeffs[::-1]  # D(J) = sum coef[k] J^k (coef[0], the fit's offset, is kept)
    stop = _turn_at(cols[0], READS[-1])

    def coeffs(t: float) -> tuple[complex, np.ndarray, np.ndarray]:
        """The time-dependent factors at turn ``t``: detuning from the drive, drive, island."""
        phi = TP * (tune0 * t + 0.5 * rate * t * t + lags)
        return (
            TP * 1j * (q_d - (tune0 + rate * t)) - Gam,
            amp * min(1.0, t / r2),
            K3 * np.exp(-1j * (6.0 * phi - 2.0 * TP * t)),
        )

    def rhs(c: np.ndarray, co: tuple) -> np.ndarray:
        J = c.real**2 + c.imag**2
        d_curve = coef[0] + J * (coef[1] + J * (coef[2] + J * (coef[3] + J * coef[4])))
        det = np.where(lin, a * J, d_curve)
        cc = np.conj(c)
        return (co[0] + TP * 1j * det) * c - co[1] + co[2] * (cc * cc) * (cc * cc) * cc

    c = np.zeros(len(cols), complex)
    co_n = coeffs(0.0)
    for step in range(int(round(stop / h))):
        n = step * h
        co_m, co_e = coeffs(n + h / 2), coeffs(n + h)
        k1 = rhs(c, co_n)
        k2 = rhs(c + h / 2 * k1, co_m)
        k3 = rhs(c + h / 2 * k2, co_m)
        k4 = rhs(c + h * k3, co_e)
        c = c + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        co_n = co_e
    surv = np.abs(c) ** 2 > 0.5 * J_bb
    return {row: int(surv[i * NL : (i + 1) * NL].sum()) for i, row in enumerate(rows)}


@functools.cache
def _model_table() -> dict[str, dict[tuple[float, float], int]]:
    """``{mode: {(rate, f): survivors}}``."""
    out: dict = {mode: {} for mode in MODES}
    for r in RATES:
        for (mode, f), n in _model_survivors(r).items():
            out[mode][(r, f)] = n
    return out


@pytest.mark.slow
def test_the_reduced_model_reproduces_the_survival() -> None:
    """Within 10 of 64 at all ten configurations (measured at most 8). **Controls:** the
    small-amplitude linear detuning in place of ``D`` misses by more than 14 somewhere
    (measured 19, at ``2e-6``, ``f = 1.5``: the crossing speed in action matters), and with
    no island term all 64 survive everywhere."""
    b = _ring_batch()
    ring = {(r, f): int(_survivors(b, r, f).sum()) for r in RATES for f in FS}
    table = _model_table()
    curve, linear = table["curve"], table["linear"]
    assert max(abs(curve[k] - ring[k]) for k in ring) <= 10, (curve, ring)
    assert max(abs(linear[k] - ring[k]) for k in ring) > 14, (linear, ring)
    assert all(n == NL for n in table["no-island"].values())
