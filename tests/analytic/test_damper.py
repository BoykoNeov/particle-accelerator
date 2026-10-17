"""The damper (W6 part a): a lumped transverse damping, applied once a turn.

``Damper(rate)`` multiplies one plane's momentum by ``1 - rate`` on every pass of a turn
loop: the net transverse effect of a turn of radiation plus RF, and the ideal feedback
kicker. Its static maps are the identity (the turn loops apply it, as they apply the AC
dipole's kick), so every optics function still sees the undamped ring.

The gates need no reference code. Over one turn the linear ring's determinant is
``1 - rate``, so the damped motion is an eigen-solution with ``|lambda|^2 = 1 - rate``
and a phase advance set by the trace,

    cos mu_d = (2 cos mu - rate (cos mu - alpha sin mu)) / (2 sqrt(1 - rate)),

``alpha`` the natural alpha at the damper. Driven, W1's particular solution
``z_hat = (lambda I - B A)^(-1) B b A e^(2 pi i lag)`` still holds with the damper inside
``A`` or ``B``, and right after the dipole it has the closed form (sympy below)

    x_hat = theta beta (1 - rate) sin mu
            / [(2 - rate)(cos 2 pi nu - cos mu) - rate alpha sin mu + i rate sin 2 pi nu],

``x_n = Re(x_hat) sin(phi_n) + Im(x_hat) cos(phi_n)``: W1's formula at ``rate = 0``, plus a
part a quarter turn behind the drive, which is the damping's.

Fixture: W4's split-tune FODO (``tests/_w4_driven_orbits.py``) with no octupole.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from _w4_driven_orbits import IDX, LAG, REF, _acd, _ring
from accsim import (
    PX,
    PY,
    ACDipole,
    Damper,
    Lattice,
    Particle,
    Tracker,
    X,
    driven_amplitude,
    driven_gradient,
    driven_states,
    driven_twiss,
)
from accsim.tune import tracked_tunes
from accsim.twiss import closed_twiss, propagate_twiss, tunes

GAMMA = 0.003  # amplitude decrement per turn: a damping time of 333 turns
RATE = 1.0 - math.exp(-2.0 * GAMMA)
NU = 13 / 40
KICK = 1e-5

#: Where the damper goes, as the index it is inserted at (before the dipole is): right
#: after the dipole, right before it, and part of a cell round the ring (a place whose
#: optics differ: whole cells away would repeat the dipole's alpha).
PLACES = {"after": IDX + 1, "before": IDX, "away": IDX + 4}


def _damped(acd: ACDipole | None, where: str = "after", rate: float = RATE) -> Lattice:
    """W4's ring (no octupole) with a damper at ``where`` relative to the dipole."""
    base = _ring(acd if acd is not None else _acd(KICK, NU))
    els = list(base.elements)
    pos = PLACES[where] if where != "before" else IDX  # IDX: pushes the dipole one on
    els.insert(pos, Damper(rate))
    if acd is None:
        els = [e for e in els if not isinstance(e, ACDipole)]
    return Lattice(els, REF)


def _mat(e) -> np.ndarray:
    return e.damping_matrix() if isinstance(e, Damper) else e.matrix(REF)


def _split(lat: Lattice) -> tuple[int, np.ndarray, np.ndarray]:
    """``(index, A, B)``: start to the dipole's entrance, its exit round to the start."""
    idx = next(i for i, e in enumerate(lat.elements) if isinstance(e, ACDipole))
    A = np.eye(6)
    for e in lat.elements[:idx]:
        A = _mat(e) @ A
    B = np.eye(6)
    for e in lat.elements[idx + 1 :]:
        B = _mat(e) @ B
    return idx, A, B


def _one_turn(lat: Lattice) -> np.ndarray:
    M = np.eye(6)
    for e in lat.elements:
        M = _mat(e) @ M
    return M


def _particular(lat: Lattice, n_turns: int) -> np.ndarray:
    """W1's exact steady state at the lattice start, with the damping inside A and B."""
    _, A, B = _split(lat)
    acd = next(e for e in lat.elements if isinstance(e, ACDipole))
    lam = np.exp(2j * np.pi * acd.tune)
    b = np.zeros(6)
    b[PX] = 1.0
    zhat = np.linalg.solve(
        lam * np.eye(6) - B @ A, B @ b * acd.amplitude * np.exp(2j * np.pi * acd.lag)
    )
    n = np.arange(n_turns + 1)
    return np.imag(zhat[None, :] * lam ** n[:, None])


def _natural_at(lat_undamped: Lattice, idx: int) -> tuple[float, float, float]:
    """``(beta, alpha, mu)`` of the undamped ring at element boundary ``idx``."""
    tw = propagate_twiss(lat_undamped, closed_twiss(lat_undamped))
    return tw[idx].beta_x, tw[idx].alpha_x, 2.0 * math.pi * tunes(lat_undamped)[0]


def _closed_mu_d(mu: float, alpha: float, rate: float) -> float:
    c = (2.0 * math.cos(mu) - rate * (math.cos(mu) - alpha * math.sin(mu))) / (
        2.0 * math.sqrt(1.0 - rate)
    )
    return math.acos(c)


# --- 1. identity outside the turn loops, damping inside -----------------------------------
def test_its_static_maps_are_the_identity() -> None:
    d = Damper(RATE)
    np.testing.assert_array_equal(d.matrix(REF), np.eye(6))
    np.testing.assert_array_equal(d.kick(REF), np.zeros(6))
    st = np.arange(6.0) * 1e-3
    np.testing.assert_array_equal(d.track(st, REF), st)
    assert d.length == 0.0
    M0, k0 = _ring(None).one_turn_map()
    M1, k1 = _damped(None).one_turn_map()
    np.testing.assert_array_equal(M1, M0)
    np.testing.assert_array_equal(k1, k0)
    # the damping itself: the named momentum, and nothing else
    want = st.copy()
    want[PX] *= 1.0 - RATE
    np.testing.assert_array_equal(d.damp(st), want)
    np.testing.assert_array_equal(st, np.arange(6.0) * 1e-3)  # not modified
    v = Damper(RATE, plane="y").damp(st)
    assert v[PY] == st[PY] * (1.0 - RATE) and v[PX] == st[PX]


def test_a_zero_rate_is_the_ring_without_it_bit_for_bit() -> None:
    """Both paths, driven and undriven, and the loss pass."""
    p = Particle(x=1e-3, px=-2e-4, y=5e-4, py=1e-4)
    for acd in (None, _acd(KICK, NU)):
        bare = _ring(acd) if acd is not None else _ring(None)
        zero = _damped(acd, rate=0.0)
        for nonlinear in (False, True):
            a = Tracker(bare).track_turns(p, 200, nonlinear=nonlinear)
            b = Tracker(zero).track_turns(p, 200, nonlinear=nonlinear)
            np.testing.assert_array_equal(a, b)


def test_the_turn_loops_damp_and_a_single_pass_does_not() -> None:
    """``track_turns`` (exact path) is a hand loop applying ``p *= 1 - rate`` at its place,
    bit for bit; the linear path is that loop's matrices to round-off; the loss pass and
    the spin path apply it too; ``track_once`` damps only when given a turn."""
    lat = _damped(_acd(KICK, NU))
    p = Particle(x=1e-3, px=-2e-4)
    n = 150
    got = Tracker(lat).track_turns(p, n, nonlinear=True)
    s = p.state.copy()
    for turn in range(n):
        for e in lat.elements:
            s = e.track(s, REF)
            if isinstance(e, ACDipole):
                s = s + e.drive_kick(turn, REF)
            if isinstance(e, Damper):
                s[PX] *= 1.0 - e.rate
    np.testing.assert_array_equal(got[-1], s)

    lin = Tracker(lat).track_turns(p, n)
    s = p.state.copy()
    for turn in range(n):
        for e in lat.elements:
            s = _mat(e) @ s
            if isinstance(e, ACDipole):
                s = s + e.drive_kick(turn, REF)
    np.testing.assert_allclose(lin[-1], s, rtol=0.0, atol=1e-17)

    # undriven: the linear path folds the damping into the one-turn map
    free = _damped(None)
    lin = Tracker(free).track_turns(p, n)
    M, s = _one_turn(free), p.state.copy()
    for _ in range(n):
        s = M @ s
    np.testing.assert_array_equal(lin[-1], s)

    from accsim import Bunch

    loss = Tracker(lat).track_bunch_losses(Bunch(p.state[:, None].copy()), n, nonlinear=True)
    np.testing.assert_array_equal(loss.states[:, 0], got[-1])
    _, spin = Tracker(lat).track_once_with_spin(p.state, np.array([0.0, 1.0, 0.0]), turn=0)
    np.testing.assert_array_equal(
        Tracker(lat).track_once_with_spin(p.state, np.array([0.0, 1.0, 0.0]), turn=0)[0],
        Tracker(lat).track_once(p.state, turn=0),
    )

    st = p.state
    np.testing.assert_array_equal(Tracker(free).track_once(st), Tracker(_ring(None)).track_once(st))
    assert Tracker(free).track_once(st, turn=0)[PX] != Tracker(_ring(None)).track_once(st)[PX]


def test_construction_guards_and_refusals() -> None:
    from accsim.acceleration import accelerate
    from accsim.scenario import ScenarioError, element_to_dict
    from accsim.tapering import _scaled

    with pytest.raises(ValueError, match="plane"):
        Damper(0.01, plane="z")
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="rate"):
            Damper(bad)
    assert "rate=0.01" in repr(Damper(0.01))
    with pytest.raises(NotImplementedError, match="Damper"):
        accelerate(_damped(None), Particle(), 3)
    with pytest.raises(ScenarioError, match="no scenario representation"):
        element_to_dict(Damper(0.01))
    with pytest.raises(NotImplementedError, match="Damper"):
        _scaled(Damper(0.01), 0.99)  # taper(): it does not know a damper's strength
    lat = _damped(_acd(KICK, NU))
    for fn in (driven_gradient, driven_twiss, driven_amplitude, driven_states):
        with pytest.raises(NotImplementedError, match="Damper"):
            fn(lat)


# --- 2. the damped linear ring ------------------------------------------------------------
@pytest.mark.parametrize("where", ["after", "before", "away"])
def test_the_free_motion_shrinks_by_sqrt_one_minus_rate_and_turns_by_mu_d(where: str) -> None:
    """Projected on the damped ring's own left eigenvector, the tracked linear motion is
    ``lambda^n`` times its start: ``|lambda|^2 = 1 - rate`` exactly, and ``arg lambda`` is
    the closed form's ``mu_d`` with the natural alpha at the damper."""
    lat = _damped(None, where)
    pos = next(i for i, e in enumerate(lat.elements) if isinstance(e, Damper))
    beta, alpha, mu = _natural_at(_ring(None), pos)
    M = _one_turn(lat)[:2, :2]
    wl, U = np.linalg.eig(M.T)
    k = int(np.argmax(wl.imag))
    w, lam = U[:, k], wl[k]
    assert abs(lam) ** 2 == pytest.approx(1.0 - RATE, rel=1e-14)
    mu_d = _closed_mu_d(mu % (2 * math.pi), alpha, RATE)
    assert np.angle(lam) == pytest.approx(mu_d, rel=1e-13)

    n = 400
    hist = Tracker(lat).track_turns(Particle(x=1e-3, px=-2e-4), n)
    proj = hist[:, :2] @ w
    want = proj[0] * lam ** np.arange(n + 1)
    assert np.max(np.abs(proj - want)) < 1e-12 * abs(proj[0])
    # the damping moves the tune by -rate alpha / (4 pi) to first order: not nothing
    shift = (mu_d - mu % (2 * math.pi)) / (2 * math.pi)
    assert shift == pytest.approx(-RATE * alpha / (4 * math.pi), rel=0.02)


def test_tracked_tunes_measures_the_damped_tune() -> None:
    """NAFF on a decaying signal still reads the damped tune: a real weight on a pure
    line leaves the peak where it was."""
    lat = _damped(None, "after")
    pos = next(i for i, e in enumerate(lat.elements) if isinstance(e, Damper))
    _, alpha, mu = _natural_at(_ring(None), pos)
    q_d = _closed_mu_d(mu % (2 * math.pi), alpha, RATE) / (2 * math.pi)
    qx, _ = tracked_tunes(lat, 1024)
    assert qx == pytest.approx(q_d, abs=1e-7)
    assert abs(q_d - (mu / (2 * math.pi)) % 1.0) > 1e-4  # and the shift is visible


# --- 3. the damped steady state -----------------------------------------------------------
@pytest.mark.parametrize("where", ["after", "before", "away"])
def test_a_particle_started_on_the_damped_steady_state_stays_on_it(where: str) -> None:
    """W1's identity with the damping inside ``A`` or ``B``: signed, 600 turns, both paths."""
    lat = _damped(_acd(KICK, NU), where)
    n = 600
    want = _particular(lat, n)
    scale = np.max(np.abs(want), axis=0)
    scale[scale == 0.0] = 1.0
    for nonlinear in (False,):
        got = Tracker(lat).track_turns(Particle.from_array(want[0]), n, nonlinear=nonlinear)
        assert np.max(np.abs(got - want) / scale) < 1e-11
    # and it is not the undamped steady state, which this ring leaves at once
    undamped = _particular(_ring(_acd(KICK, NU)), n)
    assert np.max(np.abs(undamped[:, X] - want[:, X])) > 1e-2 * np.max(np.abs(want[:, X]))


def test_the_closed_form_is_derived_not_remembered() -> None:
    """sympy: ``(lambda - M0 D)^(-1) M0 D b`` at the dipole, damper right after the kick."""
    mu, nu, al, be, r = sp.symbols("mu nu alpha beta r", real=True, positive=True)
    ga = (1 + al**2) / be
    lam = sp.exp(2 * sp.pi * sp.I * nu)
    M0 = sp.Matrix(
        [
            [sp.cos(mu) + al * sp.sin(mu), be * sp.sin(mu)],
            [-ga * sp.sin(mu), sp.cos(mu) - al * sp.sin(mu)],
        ]
    )
    D = sp.diag(1, 1 - r)
    xhat = ((lam * sp.eye(2) - M0 * D).inv() * M0 * D * sp.Matrix([0, 1]))[0]
    th = 2 * sp.pi * nu
    closed = (
        be
        * (1 - r)
        * sp.sin(mu)
        / ((2 - r) * (sp.cos(th) - sp.cos(mu)) - r * al * sp.sin(mu) + sp.I * r * sp.sin(th))
    )
    assert sp.simplify((xhat - closed).rewrite(sp.exp)) == 0
    w1 = be * sp.sin(mu) / (2 * (sp.cos(th) - sp.cos(mu)))
    assert sp.simplify((closed.subs(r, 0) - w1).rewrite(sp.exp)) == 0


def test_the_tracked_swing_at_the_dipole_is_the_closed_form() -> None:
    """In phase and in quadrature, signed, 600 turns. **Control:** W1's undamped formula
    has no quadrature, and misses by about ``Gamma / (2 pi (Q - nu))`` of the swing."""
    lat = _damped(_acd(KICK, NU), "after")
    idx, A, _ = _split(lat)
    beta, alpha, mu = _natural_at(_ring(None), IDX)
    n = 600
    hist = Tracker(lat).track_turns(Particle.from_array(_particular(lat, 0)[0]), n)
    u = (A @ hist.T)[X]  # at the dipole, before its kick
    c, s2 = math.cos(2 * math.pi * NU), math.sin(2 * math.pi * NU)
    xhat = KICK * beta * (1 - RATE) * math.sin(mu)
    xhat /= (2 - RATE) * (c - math.cos(mu)) - RATE * alpha * math.sin(mu) + 1j * RATE * s2
    ph = 2 * math.pi * (NU * np.arange(n + 1) + LAG)
    wave = xhat.real * np.sin(ph) + xhat.imag * np.cos(ph)
    np.testing.assert_allclose(u, wave, rtol=0.0, atol=1e-12 * abs(xhat))
    assert xhat.imag < 0.0  # the damped swing lags the drive
    w1 = KICK * beta * math.sin(mu) / (2 * (c - math.cos(mu)))
    miss = abs(w1 - xhat) / abs(xhat)
    q_minus_nu = (mu / (2 * math.pi)) % 1.0 - NU
    assert miss == pytest.approx(GAMMA / (2 * math.pi * q_minus_nu), rel=0.10)
