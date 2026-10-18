"""Damped autoresonance (W7): the threshold rises, and the lock ends.

W5's sweep through the tune locks the particle above a threshold kick and carries it out
along the large branch. With a :class:`~accsim.elements.damper.Damper` (W6) the averaged
equation gains ``-i Gamma c`` and maps onto the damped universal equation

    i Psi' + (tau - |Psi|^2) Psi = mu - i gamma Psi,     gamma = Gamma / sqrt(2 pi |nu_dot|),

so what matters is the damping *against the sweep*, ``gamma``. Two things change:

- the capture threshold ``mu_c`` becomes ``mu_th(gamma)``, integrated here and tabulated in
  the package (:func:`accsim.autoresonance_threshold`);
- a locked particle falls out where the drive can no longer pay for both the damping and
  the growth the sweep demands (:func:`accsim.autoresonance_escape_tune`).

At fixed ``gamma`` the ring maps onto the universal equation exactly, so every ring gate
holds ``gamma`` fixed while the rate falls: one damper strength per column, applied by the
hand loop W6 gate 1 equates with :class:`~accsim.elements.damper.Damper`. The tune falls with
amplitude toward 1/3 here, and a lock that reaches 1/3 is usually lost there (a resonance the
turn average discards): the ring gates read before it, or sweep slowly enough that the escape
comes first.

Fixture: W4's split-tune FODO, ``k3l = -2000`` at the dipole (``+2000`` for the mirror), the
damper right after the dipole.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

from _w4_driven_orbits import IDX, LAG, _ring
from accsim import (
    PX,
    ACDipole,
    Damper,
    Lattice,
    X,
    autoresonance_escape_tune,
    autoresonance_threshold,
    driven_jump_tunes,
)
from accsim.twiss import (
    _AUTORESONANCE_GAMMA_STEP,
    _AUTORESONANCE_MU_C,
    _AUTORESONANCE_MU_TH,
    _autoresonance_mu_th,
    _DampedFolds,
    _ResponseCurve,
    _universal_escape,
)

K3L = -2000.0
TP = 2.0 * math.pi
WINDOW = 40  # [turns] the demodulation window
TH_RATES = (-1e-6, -5e-7, -2.5e-7)  # gate 8
TH_READS = {0.05: 20.0, 0.075: 16.0}  # gate 8: gamma -> tau_read, before the drive reaches 1/3
ESC_GAMMA, ESC_F = 0.1, 1.3  # gate 9
ESC_RATES = (4e-7, 2e-7, 1e-7)  # gate 9, magnitudes (down on W4's ring, up on the mirror)


# --- the damped universal equation (the test's own integrator) ---------------------------
def _universal(mu: float, gamma: float, t_end: float, stop_after_escape: bool = False):
    """``Psi`` from rest, the drive ramped on over ``tau in [-80, -40]``: the solution
    object (dense). With ``stop_after_escape``, it ends once ``tau - |Psi|^2 = 20``."""

    def rhs(t: float, y: np.ndarray) -> list[float]:
        psi = y[0] + 1j * y[1]
        m = mu * min(1.0, max(0.0, (t + 80.0) / 40.0))
        dpsi = -1j * (m - (t - abs(psi) ** 2) * psi) - gamma * psi
        return [dpsi.real, dpsi.imag]

    def gone(t: float, y: np.ndarray) -> float:
        return t - (y[0] ** 2 + y[1] ** 2) - 20.0

    gone.terminal, gone.direction = True, 1.0
    return solve_ivp(
        rhs, (-80.0, t_end), [0.0, 0.0], method="DOP853", rtol=1e-10, atol=1e-12,
        dense_output=True, events=gone if stop_after_escape else None,
    )  # fmt: skip


def _root2(mu: float, gamma: float) -> float | None:
    """``|Psi|_+^2``, the energy balance's upper root (``None`` below ``sqrt(2 gamma)``)."""
    disc = mu * mu - 2.0 * gamma
    return None if disc <= 0.0 else ((mu + math.sqrt(disc)) / (2.0 * gamma)) ** 2


def _captured(mu: float, gamma: float, t_read: float | None = None) -> bool:
    """``|Psi|^2 > tau / 2`` at ``t_read``; by default the table's rule, ``min(40, 0.7
    |Psi|_+^2)``, with no capture below the energy balance's floor or before ``tau = 8``."""
    if t_read is None:
        if gamma == 0.0:
            t_read = 40.0
        else:
            r2 = _root2(mu, gamma)
            if r2 is None or 0.7 * r2 < 8.0:
                return False
            t_read = min(40.0, 0.7 * r2)
    y = _universal(mu, gamma, t_read).y[:, -1]
    return float(y[0] ** 2 + y[1] ** 2) > 0.5 * t_read


@functools.cache
def _mu_th(gamma: float, t_read: float | None = None, wide: bool = False) -> float:
    """Bisected to ``2e-7``, from a bracket around the package's value (``wide``: from
    ``[0.45, 0.9]``, for a ``gamma`` past the table)."""
    if wide:
        lo, hi = 0.45, 0.9
    else:
        guess = _autoresonance_mu_th(gamma)
        lo, hi = guess - 3e-3, guess + 3e-3
    assert not _captured(lo, gamma, t_read) and _captured(hi, gamma, t_read)
    while hi - lo > 2e-7:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if _captured(mid, gamma, t_read) else (mid, hi)
    return 0.5 * (lo + hi)


def _escape(mu: float, gamma: float) -> float:
    """``tau`` at the peak of ``|Psi|``."""
    sol = _universal(mu, gamma, 2.0 * (mu / gamma) ** 2 + 40.0, stop_after_escape=True)
    t_end = float(sol.t_events[0][0])
    grid = np.linspace(max(-80.0, t_end - 60.0), t_end, 60001)
    return float(grid[int(np.argmax(np.sum(sol.sol(grid) ** 2, axis=0)))])


def _peak(mu: float, gamma: float) -> float:
    """The largest ``|Psi|^2`` of the run."""
    sol = _universal(mu, gamma, 2.0 * (mu / gamma) ** 2 + 40.0, stop_after_escape=True)
    t_end = sol.t_events[0][0] if len(sol.t_events[0]) else sol.t[-1]
    grid = np.linspace(-80.0, t_end, int(50 * (t_end + 80.0)))
    return float(np.max(np.sum(sol.sol(grid) ** 2, axis=0)))


# --- 1. the mapping and the energy balance ------------------------------------------------
def test_the_damped_ring_maps_onto_the_damped_universal_equation() -> None:
    """W5's mapping (``tests/analytic/test_ac_dipole_sweep.py``) with ``-i Gamma c`` added:
    ``c = lam Psi``, ``n - n_x = tau / kap``, ``kap^2 = 2 pi r``, ``lam^2 = kap / (2 pi A)``,
    and the damping term becomes ``-i (Gamma / kap) Psi``: ``gamma = Gamma / kap``."""
    r, A, tau, R, e, G = sp.symbols("r A tau R epsilon Gamma", positive=True)
    Psi, dPsi = sp.symbols("Psi dPsi")
    kap = sp.sqrt(2 * sp.pi * r)
    lam = sp.sqrt(kap / (2 * sp.pi * A))
    lhs = sp.I * kap * lam * dPsi
    rhs = (
        -2 * sp.pi * (r * tau / kap - A * lam**2 * R) * lam * Psi - sp.I * e - sp.I * G * lam * Psi
    )
    gamma = G / kap
    target = sp.I * dPsi + (tau - R) * Psi + sp.I * e / (kap * lam) + sp.I * gamma * Psi
    assert sp.simplify((lhs - rhs) / (kap * lam) - target) == 0


def test_the_energy_balance_and_its_roots() -> None:
    """``d|Psi|^2/dtau = -2 gamma |Psi|^2 - 2 mu Im(Psi)``; growing at the sweep's rate 1
    needs ``2 mu s - 2 gamma s^2 >= 1``, ``s = |Psi|``: between ``s_(+-) = (mu +- sqrt(mu^2 -
    2 gamma)) / (2 gamma)``. In the ring's units ``sqrt(J_+) = lam s_+ = (eps + sqrt(eps^2 -
    2 Gamma r / A)) / (2 Gamma)``, which is W6's static end ``eps / Gamma`` as ``r -> 0``."""
    u, v, tau, mu, gamma = sp.symbols("u v tau mu gamma", real=True)
    psi = u + sp.I * v
    r2 = u**2 + v**2
    dpsi = -sp.I * (mu - (tau - r2) * psi) - gamma * psi
    d_r2 = 2 * (u * sp.re(sp.expand(dpsi)) + v * sp.im(sp.expand(dpsi)))
    assert sp.simplify(d_r2 - (-2 * gamma * r2 - 2 * mu * v)) == 0

    s = sp.symbols("s", positive=True)
    m, g = sp.symbols("mu gamma", positive=True)
    roots = sp.solve(sp.Eq(2 * m * s - 2 * g * s**2, 1), s)
    expect = {(m + sp.sqrt(m**2 - 2 * g)) / (2 * g), (m - sp.sqrt(m**2 - 2 * g)) / (2 * g)}
    assert {sp.simplify(x) for x in roots} == {sp.simplify(x) for x in expect}

    r, A, e, G = sp.symbols("r A epsilon Gamma", positive=True)
    kap = sp.sqrt(2 * sp.pi * r)
    lam = sp.sqrt(kap / (2 * sp.pi * A))
    s_plus = (m + sp.sqrt(m**2 - 2 * g)) / (2 * g)
    j_plus = (lam * s_plus).subs({m: e / (kap * lam), g: G / kap})
    ring = (e + sp.sqrt(e**2 - 2 * G * r / A)) / (2 * G)
    assert sp.simplify(j_plus - ring) == 0
    assert sp.limit(ring, r, 0) == e / G


# --- 2-5. the threshold on the averaged equation ------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("gamma", [0.025, 0.05, 0.075, 0.1])
def test_the_table_is_the_integrated_threshold(gamma: float) -> None:
    """Re-derived by bisection, it agrees with the table entry to ``2e-5``."""
    k = round(gamma / _AUTORESONANCE_GAMMA_STEP)
    assert abs(_mu_th(gamma) - _AUTORESONANCE_MU_TH[k]) < 2e-5


@pytest.mark.slow
@pytest.mark.parametrize("gamma", [0.0125, 0.0625, 0.0875])
def test_the_spline_between_table_points(gamma: float) -> None:
    assert abs(_mu_th(gamma) - _autoresonance_mu_th(gamma)) < 2e-5


def test_the_table_starts_at_mu_c() -> None:
    assert _AUTORESONANCE_MU_TH[0] is _AUTORESONANCE_MU_C
    assert _autoresonance_mu_th(0.0) == _AUTORESONANCE_MU_C
    assert len(_AUTORESONANCE_MU_TH) == 41  # gamma = 0 ... 0.1


@pytest.mark.slow
def test_the_read_out_does_not_set_the_threshold() -> None:
    """At ``gamma = 0.05`` read times 20 and 40 agree to ``1e-5`` (measured ``1e-6``); at
    0.1, where the jump has softened, 10 and 14 agree to ``1e-3`` relative (measured
    ``4.7e-4``)."""
    assert abs(_mu_th(0.05, 20.0) - _mu_th(0.05, 40.0)) < 1e-5
    a, b = _mu_th(0.1, 10.0), _mu_th(0.1, 14.0)
    assert abs(a - b) < 1e-3 * a


@pytest.mark.slow
def test_the_threshold_rises_linearly_at_small_damping() -> None:
    """Measured 2.025 (slope 0.94)."""
    ratio = (_mu_th(0.025) - _AUTORESONANCE_MU_C) / (_mu_th(0.0125) - _AUTORESONANCE_MU_C)
    assert 1.95 < ratio < 2.10


def _largest_jump(gamma: float, mus: np.ndarray) -> float:
    """The largest ratio of peak ``|Psi|^2`` across a 4% step in ``mu`` (two grid steps of
    2%)."""
    peaks = np.array([_peak(m, gamma) for m in mus])
    return float(np.max(peaks[2:] / peaks[:-2]))


@pytest.mark.slow
def test_the_lock_softens_past_the_scope() -> None:
    """Over 4%: more than 4x at ``gamma = 0.1`` (measured 4.6), under 3x at 0.15 (2.2).
    The scope stops at 0.1, and the package refuses past it."""
    th10 = _autoresonance_mu_th(0.1)
    assert _largest_jump(0.1, th10 * (0.94 + 0.02 * np.arange(7))) > 4.0
    assert _largest_jump(0.15, np.arange(0.50, 0.701, 0.0115)) < 3.0
    rate = -1e-6
    kap = math.sqrt(TP * abs(rate))
    over = _lat(ACDipole(1e-6, 0.3, LAG, tune_rate=rate), 0.1001 * kap)
    for fn in (autoresonance_threshold, autoresonance_escape_tune):
        with pytest.raises(ValueError, match="gamma"):
            fn(over)


# --- 6. the escape against its closed form ------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("gamma", [0.05, 0.1])
def test_the_escape_lies_between_the_root_and_the_static_end(gamma: float) -> None:
    """Past the root by 1.5-3.5 ``tau`` (measured 1.9-3.0); short of ``(mu / gamma)^2`` by
    between ``0.75 / gamma`` and ``0.95 / gamma`` (measured 0.79-0.82 and 0.89-0.90). The
    averaged equation has no rate: a fixed statement, not a convergence."""
    for f in (1.2, 1.5, 2.0, 3.0):
        mu = f * _autoresonance_mu_th(gamma)
        tau = _escape(mu, gamma)
        assert 1.5 < tau - _root2(mu, gamma) < 3.5
        assert 0.75 < ((mu / gamma) ** 2 - tau) * gamma < 0.95


def test_a_run_that_never_locks_has_no_escape() -> None:
    """Below the threshold the run peaks at the crossing's own response (near ``tau = 5``),
    not on the backbone: the package's integrator says ``None`` rather than call that an
    escape (the guard for kicks just above the table that do not lock in the run)."""
    assert _universal_escape(0.49, 0.1) is None
    assert _universal_escape(0.6, 0.1) is not None


# --- the ring ---------------------------------------------------------------------------
def _lat(acd: ACDipole | None, Gamma: float, k3l: float = K3L) -> Lattice:
    """W4's ring with a damper of amplitude decrement ``Gamma`` right after the dipole."""
    els = list(_ring(acd, k3l).elements)
    els.insert(IDX + 1, Damper(1.0 - math.exp(-2.0 * Gamma)))
    return Lattice(els, _ring().ref)


@functools.cache
def _q_d(Gamma: float, k3l: float) -> float:
    """The damped tune, fractional: where ``tau = 0``."""
    return _DampedFolds(_lat(ACDipole(1e-5, 0.33, LAG), Gamma, k3l), "test").q_d % 1.0


@functools.cache
def _optics(k3l: float) -> tuple[float, float]:
    """``(beta, a)``: the natural beta at the dipole and ``dQ/dJ``."""
    lat = _ring(ACDipole(1e-5, 0.33, LAG), k3l)
    _, beta, a = (float(v[0]) for v in _ResponseCurve(lat, "test")._optics(np.array([0.0])))
    return beta, a


class _Column:
    """One swept particle: rate, gamma and ``mu`` (as a factor ``f`` of ``mu_th``), its
    crossing of the damped tune at ``tau = 0`` and the drive ramped on over ``[-80, -40]``."""

    def __init__(self, rate: float, gamma: float, f: float, k3l: float = K3L) -> None:
        self.rate, self.gamma, self.f, self.k3l = rate, gamma, f, k3l
        self.kap = math.sqrt(TP * abs(rate))
        self.Gamma = gamma * self.kap
        self.n_cross = 80.0 / self.kap
        probe = ACDipole(1.0, 0.3, LAG, tune_rate=rate)
        self.theta_th = autoresonance_threshold(_lat(probe, self.Gamma, k3l))["x"]
        start = _q_d(self.Gamma, k3l) - rate * self.n_cross
        ramp = (0, int(40.0 / self.kap), 10**9, 10**9)
        self.dipole = ACDipole(f * self.theta_th, start, LAG, ramp=ramp, tune_rate=rate)

    def turn(self, tau: float) -> int:
        return math.ceil(self.n_cross + tau / self.kap)

    def tau(self, n: np.ndarray) -> np.ndarray:
        return self.kap * (n - self.n_cross)

    def lattice(self) -> Lattice:
        return _lat(self.dipole, self.Gamma, self.k3l)


def _drive(cols: list[_Column], turn: int) -> np.ndarray:
    """Each column's :meth:`~accsim.elements.acdipole.ACDipole.drive_kick`, vectorised (the
    same arithmetic; checked against the element in :func:`test_the_vectorised_drive`)."""
    p = _drive_params(tuple(cols))
    amp, tune, rate, lag, r2 = p
    ramp = np.minimum(1.0, turn / r2)
    phase = TP * (tune * turn + 0.5 * rate * turn * turn + lag)
    return amp * ramp * np.sin(phase)


@functools.cache
def _drive_params(cols: tuple[_Column, ...]) -> tuple[np.ndarray, ...]:
    d = [c.dipole for c in cols]
    assert all(x.ramp[0] == 0 for x in d)
    return tuple(
        np.array(v, dtype=float)
        for v in (
            [x.amplitude for x in d],
            [x.tune for x in d],
            [x.tune_rate for x in d],
            [x.lag for x in d],
            [x.ramp[1] for x in d],
        )
    )


def _sweep(cols: list[_Column], n_turns: int) -> np.ndarray:
    """``x`` at the dipole on every turn, one column per particle, each with its own drive
    and its own damper (``p *= e^(-2 Gamma)`` right after the dipole, as the turn loops
    apply :class:`~accsim.elements.damper.Damper`). All columns share one ring's ``k3l``."""
    (k3l,) = {c.k3l for c in cols}
    lat = _lat(ACDipole(1e-9, 0.3), 0.0, k3l)  # the structure; drives and dampers below
    assert isinstance(lat.elements[IDX], ACDipole)
    assert isinstance(lat.elements[IDX + 1], Damper)
    keep = np.array([math.exp(-2.0 * c.Gamma) for c in cols])
    s = np.zeros((6, len(cols)))
    xd = np.empty((n_turns, len(cols)))
    for turn in range(n_turns):
        for k, elem in enumerate(lat.elements):
            if k == IDX:
                xd[turn] = s[X]
                s[PX] += _drive(cols, turn)
            elif k == IDX + 1:
                s[PX] *= keep
            else:
                s = elem.track(s, lat.ref)
    return xd


def _demodulated(x: np.ndarray, d: ACDipole) -> tuple[np.ndarray, np.ndarray]:
    """``(centre turns, |swing|)`` on a sliding ``WINDOW``-turn least-squares fit of ``u sin
    + v cos`` against the drive's own swept phase (W6's demodulation)."""
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
    return n[: len(u)] + WINDOW // 2, np.hypot(u, v)


def _r2_at(col: _Column, x: np.ndarray, tau: float) -> float:
    """``|Psi|^2`` at ``tau``: the demodulated swing ``sqrt(2 beta J)``, ``J = lam^2 |Psi|^2``."""
    beta, a = _optics(col.k3l)
    n, amp = _demodulated(x[: col.turn(tau) + WINDOW], col.dipole)
    k = int(np.argmin(np.abs(col.tau(n) - tau)))
    lam2 = col.kap / (TP * abs(a))
    return float(amp[k] ** 2 / (2.0 * beta) / lam2)


def _escape_tau(col: _Column, x: np.ndarray) -> float:
    """``tau`` at the peak of the demodulated swing."""
    n, amp = _demodulated(x, col.dipole)
    return float(col.tau(n[int(np.nanargmax(amp))]))


def _locks(cols: list[_Column], xd: np.ndarray, t_read: float) -> list[bool]:
    return [_r2_at(c, xd[:, k], t_read) > 0.5 * t_read for k, c in enumerate(cols)]


def _bracket(grid: np.ndarray, locked: list[bool]) -> tuple[float, float]:
    """The grid step where locking switches on, and only once."""
    flips = [k for k in range(1, len(locked)) if locked[k] != locked[k - 1]]
    assert len(flips) == 1 and locked[-1], locked
    return float(grid[flips[0] - 1]), float(grid[flips[0]])


COARSE = 1.0 + 1e-3 * np.arange(16)  # 1.000 ... 1.015


@functools.cache
def _w4_batch() -> tuple[list[_Column], np.ndarray]:
    """W4's ring in one sweep: gate 8's coarse grids (2 gammas x 3 rates x 16), gate 9's
    escapes (3), gate 11's lock past 1/3 (1)."""
    cols = [_Column(r, g, f) for g in TH_READS for r in TH_RATES for f in COARSE]
    cols += [_Column(-r, ESC_GAMMA, ESC_F) for r in ESC_RATES]
    cols += [_Column(-1e-6, 0.1, 1.5)]
    n_turns = max(_n_turns(c) for c in cols)
    return cols, _sweep(cols, n_turns)


def _n_turns(c: _Column) -> int:
    if c.gamma in TH_READS and c.f < 1.1:
        return c.turn(TH_READS[c.gamma]) + WINDOW
    return c.turn(1.25 * _model_escape_tau(c) + 10.0) + WINDOW


@functools.cache
def _model_escape_tau(c: _Column) -> float:
    """``tau`` of :func:`accsim.autoresonance_escape_tune` for the column."""
    nu = autoresonance_escape_tune(c.lattice())["x"]
    return (nu - _q_d(c.Gamma, c.k3l)) * c.kap / c.rate


@functools.cache
def _fine_thresholds() -> dict[float, list[tuple[float, float]]]:
    """Gate 8: ``{gamma: [(f, theta_th), ...]}`` per rate, the coarse bracket refined on an
    11-point grid at ``1e-4``."""
    cols, xd = _w4_batch()
    fine_cols, grids = [], []
    for g, t_read in TH_READS.items():
        for r in TH_RATES:
            sel = [k for k, c in enumerate(cols) if c.gamma == g and c.rate == r and c.f < 1.1]
            locked = _locks([cols[k] for k in sel], xd[:, sel], t_read)
            grid = _bracket(COARSE, locked)[0] + 1e-4 * np.arange(11)
            grids.append((g, r, grid))
            fine_cols += [_Column(r, g, f) for f in grid]
    xf = _sweep(fine_cols, max(_n_turns(c) for c in fine_cols))
    out: dict[float, list[tuple[float, float]]] = {g: [] for g in TH_READS}
    for i, (g, _, grid) in enumerate(grids):
        sel = list(range(11 * i, 11 * i + 11))
        locked = _locks([fine_cols[k] for k in sel], xf[:, sel], TH_READS[g])
        lo, hi = _bracket(grid, locked)
        out[g].append((0.5 * (lo + hi), fine_cols[sel[0]].theta_th))
    return out


@functools.cache
def _mirror_batch() -> tuple[list[_Column], np.ndarray]:
    """``k3l = +2000``, swept up: gate 9's escapes (3) and gate 12's lock (2)."""
    cols = [_Column(r, ESC_GAMMA, ESC_F, k3l=-K3L) for r in ESC_RATES]
    cols += [_Column(1e-6, 0.05, f, k3l=-K3L) for f in (0.98, 1.02)]
    return cols, _sweep(cols, max(_n_turns(c) for c in cols))


def test_the_vectorised_drive() -> None:
    """:func:`_drive` is each column's ``drive_kick`` (ramp on, then full; down and up)."""
    cols = [_Column(-1e-6, 0.05, 1.01), _Column(2e-7, 0.1, 1.3, k3l=-K3L)]
    ref = _ring().ref
    for turn in (0, 1, 7, 1000, 15957, 15958, 40000, 123457):
        got = _drive(cols, turn)
        for c, g in zip(cols, got, strict=True):
            assert g == pytest.approx(c.dipole.drive_kick(turn, ref)[PX], rel=1e-12, abs=1e-22)


# --- 7. undamped is W5 ------------------------------------------------------------------
@pytest.mark.parametrize("k3l", [-2000.0, 2000.0])
def test_undamped_it_is_w5s_formula_bit_for_bit(k3l: float) -> None:
    """W5's ``4 mu_c (2 pi |nu_dot|)^(3/4) / (sqrt(2 beta) sqrt(2 pi |a|))``, to the last
    bit; a damper in the *other* plane changes nothing (the planes are uncoupled here)."""
    rate = -2e-6 if k3l < 0.0 else 2e-6
    lat = _ring(ACDipole(1e-6, 0.3, LAG, tune_rate=rate), k3l)
    beta, a = _optics(k3l)
    theta = 4.0 * _AUTORESONANCE_MU_C * (2.0 * math.pi * abs(rate)) ** 0.75
    w5 = theta / (math.sqrt(2.0 * beta) * math.sqrt(2.0 * math.pi * abs(a)))
    assert autoresonance_threshold(lat)["x"] == w5
    other = Lattice([*lat.elements, Damper(0.01, plane="y")], lat.ref)
    assert autoresonance_threshold(other)["x"] == w5
    with pytest.raises(ValueError, match="no damping"):
        autoresonance_escape_tune(lat)


def test_damped_it_is_mu_th_of_gamma() -> None:
    """The same formula with ``mu_th(gamma)``: the ratio to W5's is ``mu_th / mu_c``, and
    ``gamma`` is ``Gamma / sqrt(2 pi |nu_dot|)`` with ``Gamma = -1/2 ln(1 - rate)``."""
    rate = -1e-6
    kap = math.sqrt(TP * abs(rate))
    probe = ACDipole(1e-6, 0.3, LAG, tune_rate=rate)
    w5 = autoresonance_threshold(_ring(probe, K3L))["x"]
    for g in (0.025, 0.05, 0.0875):
        got = autoresonance_threshold(_lat(probe, g * kap))["x"]
        assert got / w5 == pytest.approx(_autoresonance_mu_th(g) / _AUTORESONANCE_MU_C, rel=1e-9)
    against = _lat(ACDipole(1e-6, 0.3, LAG, tune_rate=-rate), 0.05 * kap)
    assert autoresonance_threshold(against)["x"] is None
    assert autoresonance_escape_tune(against)["x"] is None
    with pytest.raises(ValueError, match="not swept"):
        autoresonance_threshold(_lat(ACDipole(1e-6, 0.3, LAG), 1e-4))


# --- 8. the threshold converges --------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("gamma", list(TH_READS))
def test_the_tracked_threshold_converges_on_mu_th(gamma: float) -> None:
    """Read before 1/3, bisected to ``1e-4``: the excess over
    :func:`accsim.autoresonance_threshold` is positive and falls by ``sqrt 2`` per halving
    (ratios in 1.33-1.50), under 0.4% at ``2.5e-7``. Measured 0.486, 0.341, 0.243% at 0.05
    and 0.527, 0.374, 0.263% at 0.075. **Controls:** against W5's undamped formula and
    against ``gamma`` mapped with ``rate`` (about ``2 Gamma``) in place of ``Gamma``, the
    tracked threshold is off by more than 8% at every rate, by an offset that moves less
    than 1% between rates. (At 0.075 the mis-mapped ``gamma`` is 0.15, past the table: its
    threshold is integrated here by the table's rule, which there no longer marks a sharp
    jump; any reading of it is far enough off.)"""
    found = _fine_thresholds()[gamma]
    excess = [f - 1.0 for f, _ in found]
    assert all(e > 0.0 for e in excess)
    for a, b in zip(excess, excess[1:], strict=False):
        assert 1.33 < a / b < 1.50
    assert excess[-1] < 4e-3
    mu_th = _autoresonance_mu_th(gamma)
    mis = _autoresonance_mu_th(2.0 * gamma) if 2.0 * gamma <= 0.1 else _mu_th(2 * gamma, wide=True)
    for control in (_AUTORESONANCE_MU_C, mis):
        offsets = [f * mu_th / control - 1.0 for f, _ in found]
        assert all(abs(o) > 0.08 for o in offsets)
        assert max(offsets) - min(offsets) < 0.01


# --- 9-10. the escape converges --------------------------------------------------------
def _escape_misses() -> dict[str, list[tuple[float, float, _Column]]]:
    """``{ring: [(relative miss, tracked tau, column), ...]}`` per rate of gate 9."""
    out = {}
    for name, (cols, xd) in (("w4", _w4_batch()), ("mirror", _mirror_batch())):
        rows = []
        for k, c in enumerate(cols):
            if c.gamma == ESC_GAMMA and c.f == ESC_F:
                tau = _escape_tau(c, xd[:, k])
                model = _model_escape_tau(c)
                rows.append(((tau - model) / model, tau, c))
        out[name] = sorted(rows, key=lambda row: -abs(row[2].rate))
    return out


@pytest.mark.slow
def test_the_tracked_escape_converges_on_autoresonance_escape_tune() -> None:
    """At ``gamma = 0.1``, ``mu = 1.3 mu_th``, rates ``4e-7, 2e-7, 1e-7``, slow enough that
    the escape comes before 1/3. The part of the miss odd in the detuning's sign (half the
    mirror's minus W4's) falls by ``sqrt 2`` per halving, ratios in 1.30-1.60 (measured
    1.47, 1.39): the uncorrected natural optics. The even part is smaller at every rate, and
    both rings miss by under 7% at ``1e-7`` (measured 5.4% and 6.4%)."""
    m = _escape_misses()
    w4 = [row[0] for row in m["w4"]]
    mirror = [row[0] for row in m["mirror"]]
    odd = [0.5 * (b - a) for a, b in zip(w4, mirror, strict=True)]
    even = [0.5 * (a + b) for a, b in zip(w4, mirror, strict=True)]
    for a, b in zip(odd, odd[1:], strict=False):
        assert 1.30 < a / b < 1.60
    assert all(abs(e) < abs(o) for e, o in zip(even, odd, strict=True))
    assert abs(w4[-1]) < 0.07 and abs(mirror[-1]) < 0.07


@pytest.mark.slow
def test_control_the_static_end_is_not_the_escape() -> None:
    """At ``1e-7`` on both rings, :func:`accsim.driven_jump_tunes`' large-state end lies
    beyond the tracked escape by more than 10% of the escape distance (estimated from the
    averaged end: 28% and 14%)."""
    for rows in _escape_misses().values():
        _, tau, c = rows[-1]
        q_d = _q_d(c.Gamma, c.k3l)
        nu_end = driven_jump_tunes(c.lattice())["x"][1]
        tau_end = (nu_end - q_d) * c.kap / c.rate
        assert tau_end > 1.10 * tau


# --- 11. lost at 1/3 -------------------------------------------------------------------
@pytest.mark.slow
def test_a_lock_that_reaches_one_third_ends_there() -> None:
    """W4's ring, ``gamma = 0.1``, ``mu = 1.5 mu_th``, ``1e-6``: the tracked lock ends with
    the drive tune within ``1.5e-3`` above 1/3, while the model's escape lies more than
    0.005 below it (measured 0.3339 against 0.3234). A resonance of the driven ring the
    turn average discards: named, not modelled."""
    cols, xd = _w4_batch()
    (k,) = [k for k, c in enumerate(cols) if c.f == 1.5]
    c = cols[k]
    q_d = _q_d(c.Gamma, c.k3l)
    nu = q_d + c.rate * _escape_tau(c, xd[:, k]) / c.kap
    assert 1.0 / 3.0 < nu < 1.0 / 3.0 + 1.5e-3
    assert autoresonance_escape_tune(c.lattice())["x"] < 1.0 / 3.0 - 0.005


# --- 12. the mirror locks too -----------------------------------------------------------
@pytest.mark.slow
def test_the_mirror_locks_going_up() -> None:
    """``k3l = +2000``, swept up, ``gamma = 0.05``, ``1e-6``: locked at 1.02x, not 0.98x."""
    cols, xd = _mirror_batch()
    sel = [k for k, c in enumerate(cols) if c.gamma == 0.05]
    assert _locks([cols[k] for k in sel], xd[:, sel], 20.0) == [False, True]
