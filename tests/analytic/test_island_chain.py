"""The free ring's island chain as a package function (W9).

:func:`accsim.island_chain` returns the period-``q`` orbits of a ring with no drive and no
damper, near a given action, that turn ``p/q`` per turn. W8 found its chains with a test
helper (``_fixed_points``); the function adds what made that helper safe only on W4's ring:
the search is centred on the closed orbit, orbits shorter than ``q`` and orbits of the wrong
rotation are dropped, and every orbit is completed by the one-turn map.

The coefficient-sensitive gate is a ring where the chain has a closed form. One thin
octupole drives ``4Q = p`` at **first** order; in the frame turning at ``p/4`` the per-turn
Hamiltonian is

    H = -2 pi Delta J + pi a J^2 - 8 |F| J^2 cos(4 psi - arg F),

``Delta = p/4 - Q``, ``a`` the octupole detuning, ``F = f4000 (e^(-8 pi i Q) - 1)`` (the
generator of the map is ``-H``). Its fixed points, libration and growth are derived below in
sympy; the ring reproduces them to second order in ``Delta``. A swapped sign on the resonant
term swaps the stable and unstable actions (``1.5`` and ``0.75 Delta / a``), so the gate
sees it.
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp
from scipy.optimize import brentq
from test_damped_autoresonance import K3L

from _w4_driven_orbits import REF, _fixed_points, _ring
from accsim import (
    PX,
    ACDipole,
    Corrector,
    CoupledLatticeError,
    Damper,
    Drift,
    Lattice,
    ThinOctupole,
    ThinQuadrupole,
    ThinSkewQuadrupole,
    ThinSkewSextupole,
    X,
    amplitude_detuning,
    closed_orbit_nonlinear,
    closed_twiss,
    closed_twiss_on_orbit,
    island_chain,
    resonance_driving_terms,
    tunes,
    tunes_on_orbit,
)

TP = 2.0 * math.pi
FREE = _ring(None, K3L)  # W8's free ring
K3 = 2e4  # the 4-island ring's octupole
DELTAS = (0.02, 0.01, 0.005)


# --- the 4-island ring --------------------------------------------------------------------
def _quarter_els(kf: float, k3l: float, kick: float = 0.0) -> list:
    els: list = []
    for _ in range(6):
        els += [ThinQuadrupole(kf), Drift(5.0), ThinQuadrupole(-0.22), Drift(5.0)]
    els.insert(3, ThinOctupole(k3l))  # right after the first defocusing quadrupole
    if kick:
        els.insert(1, Corrector(kick_x=kick))
    return els


@functools.cache
def _quarter(delta: float, kick: float = 0.0) -> Lattice:
    """W4's 6-cell FODO with ``kf`` solved for ``Q_x = 5/4 - delta``, one octupole."""
    kf = brentq(lambda k: tunes(Lattice(_quarter_els(k, 0.0), REF))[0] - (1.25 - delta), 0.22, 0.25)
    return Lattice(_quarter_els(kf, K3, kick), REF)


# --- the closed forms ---------------------------------------------------------------------
@functools.cache
def _closed_forms():
    """``(J(s), rate^2(s))`` of the fixed points of ``H`` with ``cos(4 psi - arg F) = s``, as
    functions of ``(Delta, a, A)``, ``A = 8 |F|``. ``rate^2 = H_JJ H_psipsi`` is the
    libration squared when positive (elliptic) and minus the growth squared when negative."""
    J, psi, Dl, a, A, th = sp.symbols("J psi Delta a A theta", real=True)
    H = -2 * sp.pi * Dl * J + sp.pi * a * J**2 - A * J**2 * sp.cos(4 * psi - th)
    out = {}
    for s in (1, -1):
        at = {psi: (th if s == 1 else th + sp.pi) / 4}
        assert sp.simplify(sp.diff(H, psi).subs(at)) == 0
        (Js,) = sp.solve(sp.diff(H, J).subs(at), J)
        rate2 = sp.simplify((sp.diff(H, J, 2) * sp.diff(H, psi, 2)).subs(at).subs(J, Js))
        assert sp.simplify(sp.diff(H, J, psi).subs(at)) == 0
        out[s] = (sp.lambdify((Dl, a, A), Js), sp.lambdify((Dl, a, A), rate2))
    return out


def _theory(delta: float) -> dict:
    lat = _quarter(delta)
    a = float(amplitude_detuning(lat)[0, 0])
    Q = tunes(lat)[0]
    F = complex(resonance_driving_terms(lat)["f4000"]) * (np.exp(-1j * TP * 4 * Q) - 1.0)
    A = 8.0 * abs(F)
    (Js, w2s), (Ju, w2u) = _closed_forms()[1], _closed_forms()[-1]
    assert w2s(delta, a, A) > 0 > w2u(delta, a, A)  # s = +1 is the elliptic one
    return {
        "J_s": Js(delta, a, A),
        "J_u": Ju(delta, a, A),
        "omega": math.sqrt(w2s(delta, a, A)),
        "growth": math.sqrt(-w2u(delta, a, A)),
        "arg_F": float(np.angle(F)),
    }


@functools.cache
def _quarter_chain(delta: float):
    return island_chain(_quarter(delta), Fraction(1, 4), _theory(delta)["J_s"])


def _wrap(x: float) -> float:
    return (x + math.pi) % TP - math.pi


# --- 1. W8's chains, unchanged ------------------------------------------------------------
def _w8_seeds(J_guess: float) -> list:
    tw = closed_twiss(FREE)
    b, al = float(tw.beta_x), float(tw.alpha_x)
    out = []
    for J in J_guess * np.array([0.9, 1.0, 1.1]):
        for phi in np.linspace(0.0, TP, 36, endpoint=False):
            x = math.sqrt(2.0 * b * J) * math.cos(phi)
            out.append((x, -math.sqrt(2.0 * J / b) * math.sin(phi) - al * x / b))
    return out


@pytest.mark.parametrize(
    ("tune", "J", "n_orbits", "m", "tr_s", "tr_u", "tol"),
    [
        (Fraction(1, 3), 2.08e-5, 2, 6, 1.99389, 2.00597, 1e-5),
        (Fraction(3, 10), 9.5e-5, 1, 10, 1.02002, 3.1051, 1e-3),
    ],
)
def test_w8_chains_unchanged(tune, J, n_orbits, m, tr_s, tr_u, tol) -> None:
    """W8's chains through the function: the same orbits and traces, ``m = 6`` and 10, and
    every point within ``1e-12`` of one the W8 helper finds."""
    chain = island_chain(FREE, tune, J)
    q = tune.denominator
    stable = [o for o in chain.orbits if o.stable]
    unstable = [o for o in chain.orbits if not o.stable]
    assert len(stable) == len(unstable) == n_orbits
    assert all(len(o.points) == q for o in chain.orbits)
    assert all(abs(o.trace - tr_s) < 1e-5 for o in stable)
    assert all(abs(o.trace - tr_u) < tol for o in unstable)
    assert chain.islands == m
    helper = [z for z, _ in _fixed_points(FREE, q, _w8_seeds(J))]
    for o in chain.orbits:
        for pt in o.points:
            d = min(float(np.max(np.abs(pt[[X, PX]] - z))) for z in helper)
            assert d < 1e-12


# --- 2. the first-order closed forms ------------------------------------------------------
def _misses(delta: float) -> dict:
    th, chain = _theory(delta), _quarter_chain(delta)
    (s,) = [o for o in chain.orbits if o.stable]
    (u,) = [o for o in chain.orbits if not o.stable]
    assert len(s.points) == len(u.points) == 4 and chain.islands == 4
    shift = [_wrap(4.0 * phi - (th["arg_F"] - 4.0 * math.pi * delta)) for phi in s.phases]
    return {
        "J_s": abs(np.mean(s.actions) / th["J_s"] - 1.0),
        "J_u": abs(np.mean(u.actions) / th["J_u"] - 1.0),
        "omega": abs(s.libration / th["omega"] - 1.0),
        "growth": abs(math.acosh(u.trace / 2.0) / 4.0 / th["growth"] - 1.0),
        "phase": max(abs(x) for x in shift),
    }


def test_the_closed_forms_hold_to_second_order() -> None:
    """``J_s``, ``J_u``, the libration and the unstable growth ``arccosh(trace / 2) / 4`` miss
    the closed forms by under ``70 Delta^2`` and fall by 3.2 to 5 per halving of ``Delta``
    (measured about 4.1). (Pre-committed on the unstable *trace*, which is ``2 + 16
    lambda^2 + ...``: its relative miss carries the constant 2 and falls as ``Delta^4``, so
    it was the wrong quantity, not a wrong orbit.) The
    stable phases sit at ``4 phi = arg F - 4 pi Delta`` to within ``1e-2`` of ``4 pi Delta``,
    the residual falling at least 1.6 per halving: the half-step is the averaged
    Hamiltonian's fixed point rotated back by half a turn's slip."""
    m = [_misses(d) for d in DELTAS]
    for key in ("J_s", "J_u", "omega", "growth"):
        for d, x in zip(DELTAS, m, strict=True):
            assert x[key] < 70.0 * d * d, (key, d, x[key])
        for x0, x1 in zip(m, m[1:], strict=False):
            assert 3.2 < x0[key] / x1[key] < 5.0, (key, x0[key], x1[key])
    for d, x in zip(DELTAS, m, strict=True):
        assert x["phase"] < 1e-2 * 4.0 * math.pi * d
    for x0, x1 in zip(m, m[1:], strict=False):
        assert x0["phase"] / x1["phase"] > 1.6


def test_the_resonant_sign_decides_which_orbit_is_stable() -> None:
    """The stable orbit is the **outer** one, at ``1.5 Delta / a``, the unstable at ``0.75``:
    the generator is ``-H``. With ``+H`` the two would swap."""
    th = _theory(0.01)
    a = float(amplitude_detuning(_quarter(0.01))[0, 0])
    assert th["J_s"] * a / 0.01 == pytest.approx(1.5, rel=1e-12)
    assert th["J_u"] * a / 0.01 == pytest.approx(0.75, rel=1e-12)
    chain = _quarter_chain(0.01)
    (s,) = [o for o in chain.orbits if o.stable]
    (u,) = [o for o in chain.orbits if not o.stable]
    assert min(s.actions) > 1.8 * max(u.actions)


# --- 3. tracking, independent of the solver -----------------------------------------------
def _track(lat: Lattice, s: np.ndarray, n: int) -> np.ndarray:
    for _ in range(n):
        for e in lat.elements:
            s = e.track(s, lat.ref)
    return s


def _tracked_libration(lat: Lattice, pt: np.ndarray, q: int, eps: float, n: int) -> float:
    """The libration per turn of a particle nudged by ``eps`` of ``x - x_co`` off a stable
    point, timed from the upward zero crossings of ``x - x_s`` sampled every ``q`` turns."""
    x_co = _x_co(lat)
    s = pt.reshape(6, 1).copy()
    s[X, 0] += eps * (pt[X] - x_co)
    d = np.empty(n)
    for k in range(n):
        d[k] = s[X, 0] - pt[X]
        s = _track(lat, s, q)
    up = np.nonzero((d[:-1] < 0) & (d[1:] >= 0))[0]
    t = up + (-d[up]) / (d[up + 1] - d[up])
    return TP * (len(t) - 1) / (t[-1] - t[0]) / q


def _x_co(lat: Lattice) -> float:
    return float(closed_orbit_nonlinear(lat)[0])


def _rotation(lat: Lattice, pt: np.ndarray, turns: int) -> float:
    """Mean phase advance per turn about the closed orbit, from the on-orbit optics."""
    co = closed_orbit_nonlinear(lat)
    tw = closed_twiss_on_orbit(lat)
    b, al = float(tw.beta_x), float(tw.alpha_x)
    s, total, prev = pt.reshape(6, 1).copy(), 0.0, None
    for _ in range(turns + 1):
        x, p = s[X, 0] - co[0], s[PX, 0] - co[1]
        ph = math.atan2(-(al * x + b * p), x)
        if prev is not None:
            total += (ph - prev) % TP
        prev = ph
        s = _track(lat, s, 1)
    return total / turns / TP


@pytest.mark.parametrize(
    ("name", "tune", "J"),
    [
        ("W8", Fraction(1, 3), 2.08e-5),
        ("quarter", Fraction(1, 4), None),
        ("steered", Fraction(1, 4), None),
    ],
)
def test_tracking_confirms_the_stable_orbits(name, tune, J) -> None:
    """From each stable point: the tracked rotation over ``100 q`` turns is ``p/q`` to
    ``1e-12``, the point returns after ``q`` turns to ``1e-12`` of its amplitude, and a
    ``1e-4`` nudge librates at ``libration`` to ``1e-4`` (measured ``1.4e-6``, ``1.6e-5``).
    ``steered``: the 4-island ring with a corrector moving the closed orbit."""
    if name == "W8":
        lat = FREE
    elif name == "quarter":
        lat, J = _quarter(0.01), _theory(0.01)["J_s"]
    else:
        lat = _steered()  # the guess from the tune on the steered orbit
        J = 1.5 * (1.25 - tunes_on_orbit(lat)[0]) / float(amplitude_detuning(lat)[0, 0])
    chain = island_chain(lat, tune, J)
    q = tune.denominator
    stable = [o for o in chain.orbits if o.stable]
    assert stable
    tw = closed_twiss_on_orbit(lat)
    b, g = float(tw.beta_x), float(tw.gamma_x)
    for o in stable:
        for pt, J_pt in zip(o.points, o.actions, strict=True):
            back = _track(lat, pt.reshape(6, 1), q)[:, 0]
            assert abs(back[X] - pt[X]) < 1e-12 * math.sqrt(2.0 * b * J_pt)
            assert abs(back[PX] - pt[PX]) < 1e-12 * math.sqrt(2.0 * g * J_pt)
            assert abs(_rotation(lat, pt, 100 * q) - float(tune) % 1.0) < 1e-12
        lib = _tracked_libration(lat, o.points[0], q, 1e-4, 3000)
        assert lib == pytest.approx(o.libration, rel=1e-4)


@functools.cache
def _steered() -> Lattice:
    """The 4-island ring at ``Delta = 0.01`` with a corrector moving the orbit."""
    return _quarter(0.01, kick=STEER)


STEER = 1.2e-4  # [rad]: the orbit moves 1.02 mm at the start


def test_the_steered_ring_has_an_offset_orbit() -> None:
    """The steered fixture's closed orbit at the start is off axis (about a millimetre)."""
    assert 3e-4 < abs(_x_co(_steered())) < 3e-3


# --- 4. the filters -----------------------------------------------------------------------
def _quarter_seeds(lat: Lattice, J_guess: float) -> list:
    tw = closed_twiss(lat)
    b, al = float(tw.beta_x), float(tw.alpha_x)
    out = []
    for J in J_guess * np.array([0.9, 1.0, 1.1]):
        for phi in np.linspace(0.0, TP, 36, endpoint=False):
            x = math.sqrt(2.0 * b * J) * math.cos(phi)
            out.append((x, -math.sqrt(2.0 * J / b) * math.sin(phi) - al * x / b))
    return out


def test_the_wrong_rotation_is_dropped() -> None:
    """At the 1/4 chain, ``tune = 3/4`` returns nothing, though Newton on the same 4-turn
    map reaches the chain's 8 points."""
    lat, J = _quarter(0.01), _theory(0.01)["J_s"]
    assert not island_chain(lat, Fraction(3, 4), J).orbits
    found = _fixed_points(lat, 4, _quarter_seeds(lat, J))
    far = [z for z, _ in found if np.max(np.abs(z)) > 1e-4]
    assert len(far) >= 8


def test_shorter_orbits_and_the_closed_orbit_are_dropped() -> None:
    """``tune = 1/8`` on the 8-turn map returns nothing, though Newton reaches the period-4
    points there; no chain ever contains the closed orbit."""
    lat, J = _quarter(0.01), _theory(0.01)["J_s"]
    assert not island_chain(lat, Fraction(1, 8), J).orbits
    found = _fixed_points(lat, 8, _quarter_seeds(lat, J)[::6])
    assert any(np.max(np.abs(z)) > 1e-4 for z, _ in found)
    for chain in (_quarter_chain(0.01), island_chain(FREE, Fraction(1, 3), 2.08e-5)):
        assert all(min(o.actions) > 0.1 * max(o.actions) for o in chain.orbits)


# --- 5. the refusals ----------------------------------------------------------------------
def _with(lat: Lattice, elem, at: int = 1) -> Lattice:
    els = list(lat.elements)
    els.insert(at, elem)
    return Lattice(els, lat.ref)


@pytest.mark.parametrize(
    ("elem", "exc", "match"),
    [
        (ACDipole(1e-6, 0.3), ValueError, "drive"),
        (Damper(0.01), ValueError, "Damper"),
        (ThinSkewQuadrupole(0.01), CoupledLatticeError, "coupled"),
        (ThinSkewSextupole(1.0), CoupledLatticeError, "coupled"),
    ],
)
def test_the_refusals(elem, exc, match) -> None:
    with pytest.raises(exc, match=match):
        island_chain(_with(_quarter(0.01), elem), Fraction(1, 4), _theory(0.01)["J_s"])


def test_idle_drive_and_damper_change_nothing() -> None:
    """A zero-amplitude ``ACDipole`` and a zero-rate ``Damper`` give the same points."""
    lat = _with(_with(_quarter(0.01), ACDipole(0.0, 0.3)), Damper(0.0), at=4)
    a = island_chain(lat, Fraction(1, 4), _theory(0.01)["J_s"])
    b = _quarter_chain(0.01)
    pa = [p for o in a.orbits for p in o.points]
    pb = [p for o in b.orbits for p in o.points]
    assert len(pa) == len(pb) == 8
    for x, y in zip(pa, pb, strict=True):
        assert np.array_equal(x, y)
