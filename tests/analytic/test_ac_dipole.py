"""The AC dipole (W1): a kick that oscillates from turn to turn, and the driven motion.

An AC dipole is a dipole whose field oscillates at a frequency near — not on — the
betatron tune. It makes the whole beam swing coherently at the *drive* tune ``nu``, with
an amplitude set by how close ``nu`` is to the machine tune ``Q``, which is how optics
are measured in real rings. It is the first element whose action depends on the turn,
so it is also the first gate of the turn-aware walkers.

The sharp gate needs no reference code. With the drive's kick ``theta_n = A sin(2 pi
(nu n + lag))`` and a *linear* one-turn map, the driven motion has an exact particular
solution, ``z_n = Im(z_hat lambda^n)`` with ``lambda = e^{2 pi i nu}`` and

    z_hat = (lambda I - B A)^{-1} B b A e^{2 pi i lag},

where ``A`` carries the beam from the lattice start to the dipole, ``B`` from the dipole
round to the start, and ``b`` is the kicked coordinate's unit vector. A particle started
on it stays on it, turn for turn, signed — so an off-by-one turn index or a flipped kick
sign each miss by order one (both asserted). At the dipole itself the amplitude has a
closed form, derived with sympy below:

    x_hat = A beta sin(2 pi Q) / (2 (cos 2 pi nu - cos 2 pi Q))
          = (A beta / 4) (cot pi (Q - nu) + cot pi (Q + nu)),

in phase with the kick, and independent of alpha. Both resonance terms are kept; the
textbook ``A beta / (4 pi (Q - nu))`` is the first one's near-resonance limit.
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
    Aperture,
    Bunch,
    Drift,
    Lattice,
    Particle,
    ReferenceParticle,
    ThinQuadrupole,
    Tracker,
    X,
    Y,
)
from accsim.twiss import closed_twiss, propagate_twiss

REF = ReferenceParticle.from_gamma(PROTON_MASS_EV, 10.0)
K1L, LD, NCELL = 0.25, 5.0, 6  # thin-lens FODO: f = 4 m, 77.4 deg per cell, Q = 1.29
IDX = 2  # the dipole sits after the first drift, where alpha != 0
KICK = 1.0e-5
OFFSET = 0.012  # Q - nu: one beat of the driven motion against the free one is 83 turns
LAG = 0.13  # in turns, as in xtrack and MAD-X


def _cells() -> list:
    out = []
    for _ in range(NCELL):
        out += [ThinQuadrupole(K1L), Drift(LD), ThinQuadrupole(-K1L), Drift(LD)]
    return out


def _ring(acd: ACDipole | None) -> Lattice:
    els = _cells()
    if acd is not None:
        els.insert(IDX, acd)
    return Lattice(els, REF)


def _q(plane: str) -> float:
    tw = closed_twiss(_ring(None))
    mu = propagate_twiss(_ring(None), tw)[-1]
    return (mu.mu_x if plane == "x" else mu.mu_y) / (2.0 * math.pi)


def _drive(plane: str = "x", kick: float = KICK, lag: float = LAG, **kw) -> ACDipole:
    nu = _q(plane) % 1.0 - OFFSET
    return ACDipole(kick=kick, tune=nu, lag=lag, plane=plane, **kw)


def _split(lat: Lattice, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """``A``: lattice start to the dipole's entrance; ``B``: dipole exit round to the start."""
    A = np.eye(6)
    for e in lat.elements[:idx]:
        A = e.matrix(lat.ref) @ A
    B = np.eye(6)
    for e in lat.elements[idx + 1 :]:
        B = e.matrix(lat.ref) @ B
    return A, B


def _particular(lat: Lattice, acd: ACDipole, n_turns: int, shift: int = 0, sign: float = 1.0):
    """The exact steady state at the lattice start, turns ``0..n_turns`` — the identity."""
    A, B = _split(lat, IDX)
    lam = np.exp(2j * np.pi * acd.tune)
    b = np.zeros(6)
    b[PX if acd.plane == "x" else PY] = 1.0
    c = sign * acd.amplitude * np.exp(2j * np.pi * (acd.lag + acd.tune * shift))
    zhat = np.linalg.solve(lam * np.eye(6) - B @ A, B @ b * c)
    n = np.arange(n_turns + 1)
    return np.imag(zhat[None, :] * lam ** n[:, None])


# --- the static maps are the natural machine's ------------------------------------------
def test_its_static_maps_are_the_identity() -> None:
    """Every optics function sees the undriven machine, bit for bit."""
    acd = _drive()
    np.testing.assert_array_equal(acd.matrix(REF), np.eye(6))
    np.testing.assert_array_equal(acd.kick(REF), np.zeros(6))
    st = np.arange(6.0) * 1e-3
    np.testing.assert_array_equal(acd.track(st, REF), st)
    assert acd.length == 0.0
    M0, k0 = _ring(None).one_turn_map()
    M1, k1 = _ring(acd).one_turn_map()
    np.testing.assert_array_equal(M1, M0)
    np.testing.assert_array_equal(k1, k0)


def test_a_single_pass_is_the_natural_machine_unless_given_a_turn() -> None:
    """``track_once`` has no turn of its own: without one it is the undriven ring."""
    acd = _drive()
    st = np.array([1e-3, -2e-4, 5e-4, 1e-4, 0.0, 0.0])
    with_acd, without = Tracker(_ring(acd)), Tracker(_ring(None))
    np.testing.assert_array_equal(with_acd.track_once(st), without.track_once(st))
    driven = with_acd.track_once(st, turn=7)
    assert not np.allclose(driven, without.track_once(st), rtol=0.0, atol=1e-9)


# --- the waveform -------------------------------------------------------------------------
def test_the_kick_is_a_sine_in_turns_with_lag_in_turns() -> None:
    """``theta_n = A sin(2 pi (nu n + lag))`` on the named coordinate, nothing else.

    ``lag = 0.25`` is a quarter turn of the *drive's* phase, so turn 0 carries the full
    amplitude — the unit both reference codes use (xtrack's docstring says radians; its C
    multiplies by 2 pi).
    """
    acd = ACDipole(kick=2.0e-5, tune=0.3, lag=0.25, plane="x")
    k0 = acd.drive_kick(0, REF)
    assert k0[PX] == pytest.approx(2.0e-5, rel=1e-15)
    assert np.count_nonzero(k0) == 1
    for n in (1, 5, 17):
        expected = 2.0e-5 * math.sin(2.0 * math.pi * (0.3 * n + 0.25))
        assert acd.drive_kick(n, REF)[PX] == pytest.approx(expected, rel=1e-13, abs=1e-20)
    v = ACDipole(kick=1.0e-5, tune=0.3, lag=0.25, plane="y").drive_kick(0, REF)
    assert v[PY] == pytest.approx(1.0e-5, rel=1e-15) and v[PX] == 0.0


def test_the_ramp_is_xtracks_trapezoid() -> None:
    """``ramp = (r1, r2, r3, r4)``: zero, linear up, flat, linear down, zero — in turns."""
    acd = ACDipole(kick=1.0, tune=0.0, lag=0.25, plane="x", ramp=(10, 20, 30, 50))
    amp = [acd.drive_kick(n, REF)[PX] for n in (0, 9, 10, 15, 20, 29, 30, 40, 49, 50, 99)]
    assert amp == pytest.approx([0, 0, 0, 0.5, 1, 1, 1, 0.5, 0.05, 0, 0], abs=1e-15)
    flat = ACDipole(kick=1.0, tune=0.0, lag=0.25, plane="x")  # no ramp: on, always
    assert flat.drive_kick(10**7, REF)[PX] == pytest.approx(1.0, rel=1e-15)


# --- the closed form, derived ---------------------------------------------------------------
def test_the_closed_form_is_derived_not_remembered() -> None:
    """sympy: the dipole-location component of ``(lambda - M)^{-1} M b``, over ``beta``."""
    mu, nu, al, be = sp.symbols("mu nu alpha beta", real=True, positive=True)
    ga = (1 + al**2) / be
    lam = sp.exp(2 * sp.pi * sp.I * nu)
    M = sp.Matrix(
        [
            [sp.cos(mu) + al * sp.sin(mu), be * sp.sin(mu)],
            [-ga * sp.sin(mu), sp.cos(mu) - al * sp.sin(mu)],
        ]
    )
    xhat = ((lam * sp.eye(2) - M).inv() * M * sp.Matrix([0, 1]))[0]
    th = 2 * sp.pi * nu
    closed = be * sp.sin(mu) / (2 * (sp.cos(th) - sp.cos(mu)))
    two_terms = (be / 4) * (sp.cot((mu - th) / 2) + sp.cot((mu + th) / 2))
    for form in (closed, two_terms):
        diff = sp.simplify((xhat - form).rewrite(sp.exp))
        assert diff == 0, form


@pytest.mark.parametrize("plane", ["x", "y"])
def test_the_tracked_amplitude_at_the_dipole_is_the_closed_form(plane: str) -> None:
    acd = _drive(plane, lag=0.0)
    lat = _ring(acd)
    n = 600
    hist = Tracker(lat).track_turns(Particle.from_array(_particular(lat, acd, 0)[0]), n)
    A, _ = _split(lat, IDX)
    u = (A @ hist.T)[X if plane == "x" else Y]  # at the dipole, before its kick
    tw = propagate_twiss(lat, closed_twiss(lat))[IDX]
    beta = tw.beta_x if plane == "x" else tw.beta_y
    Q = _q(plane)
    closed = KICK * beta * math.sin(2 * math.pi * Q)
    closed /= 2.0 * (math.cos(2 * math.pi * acd.tune) - math.cos(2 * math.pi * Q))
    # in phase with the kick: u_n = x_hat sin(2 pi nu n), signed
    wave = closed * np.sin(2 * np.pi * acd.tune * np.arange(n + 1))
    np.testing.assert_allclose(u, wave, rtol=0.0, atol=1e-11 * abs(closed))
    # and the near-resonance term alone is not it: the second term is ~1% here
    near = KICK * beta / (4.0 * math.tan(math.pi * (Q % 1.0 - acd.tune)))
    assert abs(near - closed) > 1e-3 * abs(closed)


# --- the identity, turn for turn ------------------------------------------------------------
def test_a_particle_started_on_the_steady_state_stays_on_it() -> None:
    """Signed, every coordinate, 600 turns (7 beats) — no fit, no ramp."""
    acd = _drive()
    lat = _ring(acd)
    n = 600
    want = _particular(lat, acd, n)
    got = Tracker(lat).track_turns(Particle.from_array(want[0]), n)
    scale = np.max(np.abs(want), axis=0)
    scale[scale == 0.0] = 1.0
    assert np.max(np.abs(got - want) / scale) < 1e-11  # measured floor 1.8e-13
    # it really is driven: 55x the kick, in metres (beta ~ 15 m, 1/(4 pi 0.012) ~ 6.6)
    assert np.max(np.abs(want[:, X])) > 50 * KICK


@pytest.mark.parametrize("control", ["off_by_one", "flipped_sign"])
def test_the_identity_sees_the_turn_index_and_the_sign(control: str) -> None:
    """The two errors an amplitude-only gate cannot see each miss by order one."""
    acd = _drive()
    lat = _ring(acd)
    n = 300
    right = _particular(lat, acd, n)
    wrong = (
        _particular(lat, acd, n, shift=1)
        if control == "off_by_one"
        else _particular(lat, acd, n, sign=-1.0)
    )
    got = Tracker(lat).track_turns(Particle.from_array(wrong[0]), n)
    # started on the wrong solution, the tracked particle leaves it at once
    assert np.max(np.abs(got[:, X] - wrong[:, X])) > 0.1 * np.max(np.abs(right[:, X]))


def test_the_linear_walk_and_the_loss_pass_agree() -> None:
    """``track_bunch_losses`` applies the drive with the same turn index."""
    acd = _drive()
    lat = _ring(acd)
    z0 = _particular(lat, acd, 0)[0]
    n = 120
    hist = Tracker(lat).track_turns(Particle.from_array(z0), n)
    out = Tracker(lat).track_bunch_losses(Bunch(z0[:, None].copy()), n_turns=n)
    np.testing.assert_array_equal(out.states[:, 0], hist[-1])


def test_the_drive_can_kill_what_the_natural_ring_keeps() -> None:
    """A near-resonant drive swings the beam into an aperture the free beam clears."""
    acd = _drive(kick=1.0e-4)
    els = _cells()
    els.insert(IDX, acd)
    els.append(Aperture("circular", 3.0e-3))
    lat = Lattice(els, REF)
    bunch = Bunch(np.zeros((6, 1)))
    assert not Tracker(lat).track_bunch_losses(bunch, n_turns=200).alive.any()
    natural = Lattice([e for e in els if e is not acd], REF)
    assert Tracker(natural).track_bunch_losses(Bunch(np.zeros((6, 1))), n_turns=200).alive.all()


def test_the_spin_path_applies_the_same_drive() -> None:
    acd = _drive()
    tr = Tracker(_ring(acd))
    st = np.array([1e-3, -2e-4, 5e-4, 1e-4, 0.0, 0.0])
    got, _ = tr.track_once_with_spin(st, np.array([0.0, 1.0, 0.0]), turn=11)
    np.testing.assert_array_equal(got, tr.track_once(st, turn=11))


# --- the exact path: the identity is linear, the ring is not -------------------------------
def test_on_the_exact_path_the_miss_is_the_drifts_cubic_term() -> None:
    """``track()`` is the exact drift, ``x += L px/pz``: cubic in the angle, so the miss
    *relative to the amplitude* goes as the amplitude squared — halving the kick quarters
    it. Gated on the order, not on a tolerance."""

    def rel_miss(kick: float) -> float:
        acd = _drive(kick=kick)
        lat = _ring(acd)
        want = _particular(lat, acd, 300)
        got = Tracker(lat).track_turns(Particle.from_array(want[0]), 300, nonlinear=True)
        return float(np.max(np.abs(got[:, X] - want[:, X])) / np.max(np.abs(want[:, X])))

    big, small = rel_miss(4.0e-5), rel_miss(2.0e-5)
    # measured 8.1e-6 and 2.0e-6: it is the drift's detuning, accumulating with the turns
    assert 1e-12 < small < big < 1e-4  # not round-off, not broken
    assert 3.8 < big / small < 4.2  # measured 3.99998


# --- switching it on: the ramp law ---------------------------------------------------------
def _free_amplitude(lat: Lattice, acd: ACDipole, n_turns: int, start: int) -> np.ndarray:
    """Courant-Snyder amplitude of (tracked - steady state) at the lattice start, x plane."""
    hist = Tracker(lat).track_turns(Particle(), n_turns)
    free = hist - _particular(lat, acd, n_turns)
    tw = closed_twiss(lat)
    x, px = free[start:, X], free[start:, PX]
    return np.sqrt(tw.gamma_x * x * x + 2 * tw.alpha_x * x * px + tw.beta_x * px * px)


def test_switching_on_abruptly_leaves_a_free_oscillation_as_big_as_the_driven_one() -> None:
    """The control for the ramp law: from rest, the free part is minus the steady state."""
    acd = _drive()
    lat = _ring(acd)
    free = _free_amplitude(lat, acd, 400, 0)
    steady = _particular(lat, acd, 400)
    tw = closed_twiss(lat)
    x, px = steady[:, X], steady[:, PX]
    driven = np.sqrt(tw.gamma_x * x * x + 2 * tw.alpha_x * x * px + tw.beta_x * px * px)
    # the leftover is a FREE oscillation: its invariant does not move
    assert np.ptp(free) < 1e-11 * free.mean()  # measured 1.3e-13
    assert 0.5 * driven.max() < free.mean() < 1.5 * driven.max()


def test_a_linear_ramp_leaves_a_free_oscillation_falling_as_one_over_its_length() -> None:
    """A trapezoid's corners are kinks in the amplitude, so the leftover goes as ``1/N``.

    The ramp lengths put ``(Q - nu) N`` on half-integers, where the two corners' leftovers
    add rather than cancel, so the law is not hidden by the beat between them.
    """
    lengths = [int(round((k + 0.5) / OFFSET)) for k in (1, 3, 7, 15)]
    amps = []
    for N in lengths:
        acd = _drive(ramp=(0, N, 10**9, 10**9))
        lat = _ring(acd)
        free = _free_amplitude(lat, acd, N + 200, N + 1)
        assert np.ptp(free) < 1e-9 * free.mean()  # free once the ramp ends; measured <= 3e-11
        amps.append(free.mean())
    slope = np.polyfit(np.log(lengths), np.log(amps), 1)[0]
    assert -1.05 < slope < -0.95, slope  # measured -1.0006


# --- guards -------------------------------------------------------------------------------
def test_construction_guards() -> None:
    with pytest.raises(ValueError, match="plane"):
        ACDipole(kick=1e-5, tune=0.3, plane="z")
    with pytest.raises(ValueError, match="ramp"):
        ACDipole(kick=1e-5, tune=0.3, ramp=(0, 10, 5, 20))  # not non-decreasing
    with pytest.raises(ValueError, match="ramp"):
        ACDipole(kick=1e-5, tune=0.3, ramp=(0, 10, 20))
    with pytest.raises(ValueError, match="ramp"):
        ACDipole(kick=1e-5, tune=0.3, ramp=(0, 10.5, 20, 30))
    with pytest.raises(TypeError):
        ACDipole(kick=1e-5, tune=0.3, dx=1e-3)  # a uniform kick has no centre to miss
    assert "tune=0.3" in repr(ACDipole(kick=1e-5, tune=0.3))


def test_a_taper_scales_it_like_a_corrector() -> None:
    """A powered dipole: its amplitude follows the beam's momentum, its tune does not."""
    from accsim.tapering import _scaled

    acd = ACDipole(kick=1e-5, tune=0.3, lag=0.1, ramp=(0, 10, 20, 30))
    twin = _scaled(acd, 0.99)
    assert twin.amplitude == pytest.approx(0.99e-5, rel=1e-15)
    assert (twin.tune, twin.lag, twin.ramp) == (acd.tune, acd.lag, acd.ramp)
    assert acd.amplitude == 1e-5  # the original is untouched


def test_the_scenario_format_refuses_it_until_the_editor_has_it() -> None:
    from accsim.scenario import ScenarioError, element_to_dict, load_scenario

    with pytest.raises(ScenarioError, match="no scenario representation"):
        element_to_dict(ACDipole(kick=1e-5, tune=0.3))
    record = {
        "format": "accsim-scenario/1",
        "reference": {"species": "proton", "energy_eV": 1e10, "energy_mode": "total"},
        "elements": [{"type": "ACDipole", "kick": 1e-5, "tune": 0.3}],
    }
    with pytest.raises(ScenarioError, match="unknown element type"):
        load_scenario(record)
