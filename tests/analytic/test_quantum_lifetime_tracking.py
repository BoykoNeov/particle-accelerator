r"""Analytic gates — the acceptance a radiating bunch dies at (B4).

B3 gave the bunch an equilibrium; nothing in accsim had ever *lost* a particle to
radiation, because the excitation only refilled a distribution with no exit. B4 is the
exit: a momentum acceptance, and the decay of the population that survives it.

**The headline is not "the tracking reproduces the closed form", because it does not,
and the reason is physics rather than a bug.** ``quantum_lifetime`` describes a
*continuum* diffusion of the oscillation amplitude. A tracked bunch is a *discrete*
random walk that is looked at once per turn, and at this ring's damping time one turn
moves the normalised action by 0.23 out of ``xi = 3``. Two things follow, both measured
here and both absent from the closed form:

  * a coordinate cut is not an amplitude cut. ``|delta|`` is tested once per turn, so a
    particle whose amplitude has crossed the boundary survives until a *sample* lands
    where ``delta`` is near its extreme. That lengthens the decay by a further 22% at
    this step size, and it is **flat in ``Q_s``** across a factor of four (gate 5),
    which is what identifies it as once-per-turn sampling rather than phase-rotation
    delay — a rotation delay would scale with the synchrotron period;
  * even a true amplitude cut runs 14% long, because the walk's steps are not
    infinitesimal.

Together they put the tracked decay 37% above ``tau/lambda_1`` on this ring.

Both departures **extrapolate to zero as the step vanishes** (gate 6), which is what
makes the closed form the continuum limit of the thing being tracked rather than a
number the tracking misses. So the milestone is a chain of three links, each gated
separately, in B3's shape:

  1. **the map's noise and damping** — B3's Lyapunov solve, re-used here (gate 2);
  2. **the first-passage physics given that map** — accsim's tracked decay against an
     independent 30-line implementation of the same discrete process, which shares no
     code with accsim at all (gate 7);
  3. **that discrete process against the closed forms** — the slowest eigenvalue of the
     Fokker-Planck generator as the step goes to zero (gate 6), and the exact
     mean-first-passage integral against that eigenvalue (gate 4).

What link 2 does *not* buy, stated plainly: the toy shares the conceptual model, so it
cannot catch a wrong noise magnitude. That is link 1's job, and B3 already did it — the
tracked ``sigma_delta`` is the Lyapunov one to better than 0.5%, and the tracked
``delta`` distribution is Gaussian (kurtosis 2.99).

The traps, all of which were walked into while building this:

  * **the cut has to be centred on the closed orbit** (gate 1). Radiation drains
    ``delta`` through the arcs and the cavity restores it in one lump, so the fixed
    point is not ``delta = 0``: a symmetric cut at the worst element is ``xi = 1.73``
    one side and ``7.20`` the other where both should read ``4.00``;
  * **the mean first-passage time is not the decay constant** (gate 4) — 8% apart at
    ``xi = 4``, against a statistical budget of 2%;
  * **the asymptote is not the exact integral** at a gate-sized ``xi``
    (``tests/analytic/test_quantum_lifetime.py``, 29% at ``xi = 4``);
  * **the ring must be built so particles actually die** — at a real ring's ``xi`` of
    tens the lifetime is ``e^50`` damping times and any tracked gate passes vacuously.

The tracked gate costs ~32 s and the whole file ~54 s, which is why the ring is 20 cells
and not 40. It is deliberately *not* the cheaper 10-cell ring: there ``U0/E`` reaches
1.1% per turn, the bunch is not confined at all (tracking it without an acceptance
returns ``NaN``), and the disagreement with the toy that follows is a broken ring rather
than interesting physics. The anharmonicity of the RF bucket was checked as a candidate
explanation for it and **ruled out** — a pendulum version of the toy, matched to the same
``Q_s`` and the same bucket height, returns 521 turns where the linear one returns 523.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal, solve_discrete_lyapunov

from accsim import (
    Aperture,
    Bunch,
    Dipole,
    Lattice,
    MomentumAperture,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    Tracker,
    quantum_lifetime_exact,
)
from accsim.longitudinal import rf_bucket_height
from accsim.radiation import (
    damping_times,
    energy_loss_per_turn,
    equilibrium_energy_spread,
)
from accsim.reference import CLIGHT
from accsim.symplectic import unit_symplectic_matrix

ELECTRON_MASS_EV = 0.51099895069e6

# The B4 ring. 6.5 GeV so the longitudinal amplitude damping time is 220 turns rather
# than thousands, and 90 MV so the acceptance below sits at 0.17 of the bucket height --
# deep in the harmonic region, which is where the closed form lives. The acceptance is
# 2.4-2.8 sigma (xi = 3-4): a real ring's xi of tens would make every tracked gate here
# vacuous. See docs/ROADMAP.md -> B4.
RING = {"cells": 20, "focal": 2.5, "energy": 6.5e9, "voltage": 90.0e6, "harmonic": 20}


def _ring(
    cells: int,
    focal: float,
    energy: float,
    voltage: float,
    harmonic: int,
    cut: MomentumAperture | Aperture | None = None,
    at: int = 0,
) -> Lattice:
    """B3's isomagnetic FODO ring, optionally with one acceptance boundary spliced in."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, energy)
    angle = 2.0 * math.pi / (2 * cells)
    cell = [
        ThinQuadrupole(0.5 / focal),
        Dipole(1.0, angle),
        ThinQuadrupole(-1.0 / focal),
        Dipole(1.0, angle),
        ThinQuadrupole(0.5 / focal),
    ]
    elements = list(cell) * cells
    if cut is not None:
        elements.insert(at, cut)
    lat = Lattice(elements, ref=ref)
    cavity = RFCavity.from_harmonic(voltage, harmonic, lat.length, ref, phi_s=math.pi)
    return Lattice([*elements, cavity], ref=ref)


# ---------------------------------------------------------------------------
# Measurement machinery. This is B3's, deliberately duplicated rather than imported
# across test modules (nothing else in the suite does that): the one-turn Jacobian of
# the *mean* map, its radiation fixed point by Newton, and the equilibrium covariance
# as the exact solution of Sigma = M Sigma M^T + D. No statistics anywhere in it.
# ---------------------------------------------------------------------------
class _OnePhoton:
    """``rng`` double: ``amp`` standard deviations on draw ``k``, nothing on the rest."""

    def __init__(self, k: int, amp: float) -> None:
        self.k, self.amp, self.i = k, amp, 0

    def normal(self, loc: float, scale: np.ndarray | float) -> np.ndarray | float:
        drawn = self.amp * scale if self.i == self.k else 0.0 * np.asarray(scale)
        self.i += 1
        return drawn + loc


def _equilibrium_orbit(tracker: Tracker) -> np.ndarray:
    """Newton on ``track_once(s) = s`` with radiation ``"mean"`` (B2's routine)."""
    state = np.zeros(6)
    for _ in range(60):
        residual = tracker.track_once(state, radiation="mean") - state
        if np.max(np.abs(residual)) < 1e-14:
            break
        jac = np.empty((6, 6))
        for i in range(6):
            plus, minus = state.copy(), state.copy()
            plus[i] += 1e-7
            minus[i] -= 1e-7
            jac[:, i] = (
                tracker.track_once(plus, radiation="mean")
                - tracker.track_once(minus, radiation="mean")
            ) / 2.0e-7
        state = state - np.linalg.solve(jac - np.eye(6), residual)
    return state


def _equilibrium(lattice: Lattice, step: float = 1e-7, amp: float = 1e-3):
    """``(Sigma, M, orbit)`` — B3's envelope, exactly."""
    orbit = _equilibrium_orbit(Tracker(lattice))
    ref, elements = lattice.ref, list(lattice.elements)
    entry, state = [], orbit.copy()
    for elem in elements:
        entry.append(state.copy())
        state = elem.track(state, ref, radiation="mean")

    jacobians, injected = [], []
    for elem, at in zip(elements, entry, strict=True):
        block = np.empty((6, 6))
        for i in range(6):
            plus, minus = at.copy(), at.copy()
            plus[i] += step
            minus[i] -= step
            block[:, i] = (
                elem.track(plus, ref, radiation="mean") - elem.track(minus, ref, radiation="mean")
            ) / (2.0 * step)
        jacobians.append(block)
        up = elem.track(at.copy(), ref, radiation="quantum", rng=_OnePhoton(0, amp))
        down = elem.track(at.copy(), ref, radiation="quantum", rng=_OnePhoton(0, -amp))
        injected.append((up - down) / (2.0 * amp))

    propagator, columns = np.eye(6), []
    for i in range(len(elements) - 1, -1, -1):
        columns.append(propagator @ injected[i])
        propagator = propagator @ jacobians[i]
    cols = np.array(columns[::-1]).T
    return solve_discrete_lyapunov(propagator, cols @ cols.T), propagator, orbit


def _tau_and_qs(m: np.ndarray) -> tuple[float, float]:
    """``(amplitude damping time in turns, Q_s)`` from the longitudinal eigenvalue pair.

    ``tau = -1/ln|lambda|`` **is** the amplitude damping time by definition, which is
    how B4 sidesteps the factor-of-2 between amplitude and action rather than converting
    it: the Lyapunov route's natural output would have been the action's rate.
    """
    values = np.linalg.eigvals(m)
    angles = np.abs(np.angle(values))
    k = int(np.argsort(angles)[0])
    return -1.0 / math.log(abs(values[k])), float(np.sort(angles)[0] / (2.0 * math.pi))


def _mode_parts(sigma: np.ndarray) -> list[np.ndarray]:
    """Split ``Sigma`` into its three eigen-mode contributions, via projectors of ``Sigma S``."""
    values, vectors = np.linalg.eig(sigma @ unit_symplectic_matrix())
    inverse = np.linalg.inv(vectors)
    order = np.argsort(np.abs(values.imag))
    parts = []
    for pair in (order[0:2], order[2:4], order[4:6]):
        select = np.zeros((6, 6), dtype=complex)
        for i in pair:
            select[i, i] = 1.0
        parts.append(np.real(vectors @ select @ inverse @ sigma))
    return parts


_CACHE: dict[float, tuple] = {}


def _cached(voltage: float = RING["voltage"]) -> tuple:
    """``(Sigma, M, orbit, tau, Q_s, lattice)`` — several gates share one expensive ring."""
    if voltage not in _CACHE:
        lat = _ring(**{**RING, "voltage": voltage})
        sigma, m, orbit = _equilibrium(lat)
        _CACHE[voltage] = (sigma, m, orbit, *_tau_and_qs(m), lat)
    return _CACHE[voltage]


def _lambda1(xi: float, n: int = 6000) -> float:
    r"""Slowest decay rate of the amplitude Fokker-Planck generator, in units ``1/tau``.

    The generator whose backward-equation residual ``tests/analytic/test_quantum_lifetime.py``
    already verifies symbolically, ``L = (2/tau)[w d2/dw2 + (1-w) d/dw]``, with an
    **absorbing** wall at ``w = xi``. It is self-adjoint in the weight ``e^-w`` --
    ``L = (1/rho)(p f')'`` with ``rho = e^-w`` and ``p = w e^-w`` -- so the discretisation
    is a *symmetric* tridiagonal matrix and the eigenvalue is exact to the grid rather
    than to an iteration. ``p(0) = 0`` supplies the natural boundary at the core with no
    condition imposed by hand.

    This is what a survival curve measures. It is **not** the mean first-passage time
    (gate 4).
    """
    w = np.linspace(0.0, xi, n + 1)
    h = w[1] - w[0]
    mid = 0.5 * (w[:-1] + w[1:])
    p = mid * np.exp(-mid)
    rho = np.exp(-w)
    diag = np.array([((p[i - 1] if i else 0.0) + p[i]) / (h * h * rho[i]) for i in range(n)])
    off = np.array([-p[i] / (h * h * math.sqrt(rho[i] * rho[i + 1])) for i in range(n - 1)])
    return 2.0 * eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))[0][0]


def _fit_decay(history: list[tuple[int, int]], skip: float) -> float:
    """Decay constant from the *tail* of a survival curve, in turns.

    ``skip`` discards the first damping time. ``e^-xi`` of an equilibrium bunch starts
    with its action already beyond the boundary and dies within a synchrotron period;
    that transient sits entirely at ``t ~ 0`` and would bias a whole-curve fit towards
    too-fast a decay.
    """
    h = np.array(history, dtype=float)
    mask = (h[:, 0] >= skip) & (h[:, 1] > 20)
    return -1.0 / np.polyfit(h[mask, 0], np.log(h[mask, 1]), 1)[0]


def _toy(qs: float, xi: float, tau: float, mode: str, n: int, turns: int, seed: int) -> float:
    """The same first-passage process with **no accelerator in it at all**.

    Normalised longitudinal phase space: rotate by ``2 pi Q_s``, damp both coordinates
    by ``e^-1/tau``, and kick only ``d`` with the variance that holds the equilibrium at
    unit width (``kick^2 = 2(1 - e^-2/tau)``, so that ``<z^2> = <d^2> = 1``). Then remove
    a particle either on the coordinate ``|d| > sqrt(2 xi)`` or on the amplitude
    ``(z^2 + d^2)/2 > xi``.

    Twenty lines, sharing nothing with accsim: no lattice, no elements, no radiation
    model, no photon spectrum. That is the point — it is an independent implementation
    of the *process*, which is what gate 7 compares the tracking against.
    """
    rng = np.random.default_rng(seed)
    r = math.exp(-1.0 / tau)
    kick = math.sqrt(2.0 * (1.0 - r * r))
    c, s = math.cos(2.0 * math.pi * qs), math.sin(2.0 * math.pi * qs)
    acceptance = math.sqrt(2.0 * xi)

    z, d = rng.standard_normal(n), rng.standard_normal(n)
    alive = np.ones(n, dtype=bool)
    history = [(0, n)]
    every = max(1, turns // 40)
    for turn in range(1, turns + 1):
        z, d = c * z - s * d, s * z + c * d
        z *= r
        d = r * d + kick * rng.standard_normal(n)
        alive &= np.abs(d) <= acceptance if mode == "coord" else (z * z + d * d) <= 2.0 * xi
        z[~alive] = 0.0
        d[~alive] = 0.0
        if turn % every == 0:
            history.append((turn, int(alive.sum())))
    return _fit_decay(history, tau)


def _toy_survival(
    qs: float, xi: float, tau: float, marks: tuple[int, ...], n: int, seed: int
) -> dict[int, float]:
    """The same toy, reporting the surviving fraction at chosen turns.

    A survival *fraction* is binomial and needs no fit, so it is a much sharper
    comparison than a fitted decay constant — which is why gate 7 uses it.
    """
    rng = np.random.default_rng(seed)
    r = math.exp(-1.0 / tau)
    kick = math.sqrt(2.0 * (1.0 - r * r))
    c, s = math.cos(2.0 * math.pi * qs), math.sin(2.0 * math.pi * qs)
    acceptance = math.sqrt(2.0 * xi)

    z, d = rng.standard_normal(n), rng.standard_normal(n)
    alive = np.ones(n, dtype=bool)
    out: dict[int, float] = {}
    for turn in range(1, max(marks) + 1):
        z, d = c * z - s * d, s * z + c * d
        z *= r
        d = r * d + kick * rng.standard_normal(n)
        alive &= np.abs(d) <= acceptance
        z[~alive] = 0.0
        d[~alive] = 0.0
        if turn in marks:
            out[turn] = alive.sum() / n
    return out


# ---------------------------------------------------------------------------
# Gate 1 — the closed orbit the cut has to be centred on.
#
# The trap that cost the most: it looks like a detail and it is an order of magnitude.
# ---------------------------------------------------------------------------
def test_the_radiating_closed_orbit_swings_by_two_sigma_in_momentum() -> None:
    r"""``delta_co(s)`` is not zero anywhere but by accident, and the swing is ``~U0/E``.

    Radiation drains ``delta`` steadily through the arcs and the RF cavity restores it
    in one lump, so the periodic fixed point sags below the design momentum and climbs
    back through it once per turn. On this ring ``U0/E = 3.8e-3`` against
    ``sigma_delta = 2.0e-3``.
    """
    _, _, orbit, _, _, lat = _cached()
    sd = equilibrium_energy_spread(lat)  # the design-route width, which is what a user has
    deltas, state = [], orbit.copy()
    for elem in lat.elements:
        deltas.append(state[5])
        state = elem.track(state, lat.ref, radiation="mean")
    d = np.array(deltas) / sd

    assert d.min() == pytest.approx(-0.966, abs=0.01)
    assert d.max() == pytest.approx(+0.921, abs=0.01)
    assert d.max() - d.min() == pytest.approx(1.887, abs=0.02)
    # it is the energy loss per turn that sets the scale, not the beam size
    u0_over_e = energy_loss_per_turn(lat) / lat.ref.total_energy_eV
    assert u0_over_e == pytest.approx(3.8e-3, rel=0.05)
    assert (d.max() - d.min()) * sd == pytest.approx(u0_over_e, rel=0.05)


def test_an_uncentred_cut_is_a_different_xi_on_each_side() -> None:
    r"""Why ``MomentumAperture`` has a ``center``: ``e^xi`` turns 0.9 sigma into 10x.

    A symmetric ``|delta| <= A`` placed at the worst element sits at ``xi = 1.73`` on one
    side and ``7.20`` on the other where both should read ``4.00``. The lifetime is
    dominated by the near side, so the answer is not slightly wrong.
    """
    _, _, orbit, _, _, lat = _cached()
    sd = equilibrium_energy_spread(lat)  # the width a naive cut would be sized in
    deltas, state = [], orbit.copy()
    for elem in lat.elements:
        deltas.append(abs(state[5]))
        state = elem.track(state, lat.ref, radiation="mean")
    worst = max(deltas)

    acceptance = sd * math.sqrt(2.0 * 4.0)  # xi = 4 about the closed orbit
    near = (acceptance - worst) ** 2 / (2.0 * sd**2)
    far = (acceptance + worst) ** 2 / (2.0 * sd**2)
    assert near == pytest.approx(1.734, abs=0.02)
    assert far == pytest.approx(7.199, abs=0.02)
    # and what that costs, in the only units that matter: a 6x shorter lifetime, from a
    # boundary that looks perfectly reasonable on a plot
    penalty = quantum_lifetime_exact(acceptance - worst, sd, 1.0) / quantum_lifetime_exact(
        acceptance, sd, 1.0
    )
    assert penalty == pytest.approx(0.164, rel=0.03)


# ---------------------------------------------------------------------------
# Gate 2 — link 1 of the chain: the map's own damping and width.
# ---------------------------------------------------------------------------
def test_the_tracked_map_and_the_radiation_integrals_agree_on_the_damping_time() -> None:
    r"""``-1/ln|lambda|`` from the one-turn Jacobian is 220.3 turns; ``I2``/``I4`` say 220.2.

    Two routes with nothing in common — an eigenvalue of the tracked map against a
    lattice integral — and, because ``-1/ln|lambda|`` **is** the amplitude damping time
    by definition, this also pins the amplitude-versus-action factor of 2 that
    ``quantum_lifetime`` takes as an input, with no conversion written anywhere.
    """
    _, m, _, tau, qs, lat = _cached()
    t0 = lat.length / (lat.ref.beta0 * CLIGHT)
    assert tau == pytest.approx(min(damping_times(lat)) / t0, rel=3.0e-3)
    assert tau == pytest.approx(219.6, rel=0.01)
    assert qs == pytest.approx(0.1294, rel=0.01)
    assert 1.0 / qs < tau / 10.0  # the synchrotron period is short against the damping


def test_the_momentum_spread_the_cut_is_measured_in_is_the_longitudinal_mode_alone() -> None:
    r"""``Sigma_dd`` is 99.3% longitudinal here, so ``xi = A^2/2 Sigma_dd`` is well posed.

    Dispersion mixes the horizontal mode into the longitudinal plane, and if a
    significant share of the momentum spread belonged to the horizontal mode then a
    ``delta`` cut would be a **two-mode** first-passage problem with no single ``xi``.
    Checked rather than assumed, by projecting ``Sigma`` onto the eigen-modes of
    ``Sigma S``. The projectors are exact: the three parts sum back to ``Sigma``.
    """
    sigma, _, _, _, _, _ = _cached()
    parts = _mode_parts(sigma)
    assert np.max(np.abs(sum(parts) - sigma)) < 1.0e-15 * np.max(np.abs(sigma))

    shares = sorted(part[5, 5] / sigma[5, 5] for part in parts)
    assert shares[0] == pytest.approx(0.0, abs=1e-12)  # the vertical mode, exactly nothing
    assert shares[1] < 0.01  # the horizontal mode's leak
    assert shares[2] > 0.99  # the longitudinal mode owns the momentum spread
    # x is the plane dispersion actually contaminates, and it is not a little
    x_shares = sorted(part[0, 0] / sigma[0, 0] for part in parts)
    assert x_shares[2] > 0.7 and 0.2 < x_shares[1] < 0.35


def test_the_lyapunov_width_carries_b3s_own_departure_and_nothing_new() -> None:
    """The Lyapunov ``sigma_delta`` is 3.6% above the closed form — B3's ``Q_s`` owner.

    Recorded here because ``xi`` is built from the **measured** width, not the design
    one: since the lifetime goes as ``e^xi``, using the closed-form width instead would
    move ``xi`` by 7% and the lifetime by 25%, and that would be a gate on B3's
    smooth-ring approximation wearing B4's clothes.
    """
    sigma, _, _, _, _, lat = _cached()
    measured = math.sqrt(sigma[5, 5])
    assert measured / equilibrium_energy_spread(lat) == pytest.approx(1.036, rel=0.01)
    xi_from_closed_form = (measured * math.sqrt(8.0)) ** 2 / (
        2.0 * equilibrium_energy_spread(lat) ** 2
    )
    assert xi_from_closed_form == pytest.approx(4.29, abs=0.02)  # not 4.00


def test_the_ring_is_built_where_particles_actually_die() -> None:
    r"""``xi = 4`` is 0.17 of the bucket height, and a real ring's ``xi`` would be vacuous.

    Both halves matter. The acceptance must be deep inside the RF bucket, or the
    boundary sits where the motion is anharmonic and the closed form does not describe
    it; and it must be a few sigma, or nothing is ever lost. At a normal ring's
    ``xi = 50`` the lifetime is ``e^50`` damping times and every tracked gate below would
    pass without measuring anything.
    """
    sigma, _, _, tau, _, lat = _cached()
    sd = math.sqrt(sigma[5, 5])
    acceptance = sd * math.sqrt(8.0)
    assert acceptance / rf_bucket_height(lat) == pytest.approx(0.17, abs=0.02)
    assert acceptance / sd == pytest.approx(2.83, rel=0.01)
    # affordable: thousands of turns, not e^50 of them
    assert 1500 < quantum_lifetime_exact(acceptance, sd, tau) < 2500
    assert quantum_lifetime_exact(sd * math.sqrt(100.0), sd, tau) > 1.0e20


# ---------------------------------------------------------------------------
# Gate 4 — link 3: the two closed forms are different numbers here.
# ---------------------------------------------------------------------------
def test_the_mean_first_passage_time_is_not_the_decay_constant() -> None:
    r"""``MFPT/(1/lambda_1)`` = 1.134, 1.079, 1.005, 0.999 at ``xi`` = 3, 4, 8, 12.

    The pre-committed trap of this milestone. ``quantum_lifetime_exact`` is the mean time
    for *one* particle at the core to reach the boundary; what a survival curve measures
    is the slowest eigenvalue of the same generator with an absorbing wall. They agree
    only as ``xi -> infinity``. At ``xi = 4`` they are 8% apart against a statistical
    budget of ~2%, so gating a fitted lifetime against the integral would have failed by
    4x and read as a bug in the tracking.
    """
    ratios = {}
    for xi in (3.0, 4.0, 8.0, 12.0):
        mfpt = quantum_lifetime_exact(math.sqrt(2.0 * xi), 1.0, 1.0)  # in units of tau
        ratios[xi] = mfpt * _lambda1(xi)
    assert ratios[3.0] == pytest.approx(1.134, abs=0.003)
    assert ratios[4.0] == pytest.approx(1.079, abs=0.003)
    assert ratios[8.0] == pytest.approx(1.005, abs=0.003)
    assert ratios[12.0] == pytest.approx(1.000, abs=0.003)
    # monotone towards 1: the gap is the same O(1/xi) that separates the asymptote
    assert ratios[3.0] > ratios[4.0] > ratios[8.0] > 0.999
    # a real ring's xi: the distinction stops mattering, which is why it was never noticed
    for xi in (14.0, 16.0, 18.0):
        assert quantum_lifetime_exact(math.sqrt(2.0 * xi), 1.0, 1.0) * _lambda1(
            xi
        ) == pytest.approx(1.0, abs=0.005)


def test_the_eigenvalue_route_has_a_ceiling_and_the_integral_does_not() -> None:
    r"""``_lambda1`` is unusable past ``xi ~ 20``, and that is gated rather than discovered.

    The symmetrising weight is ``e^-w``, so the tridiagonal entries span ``e^xi`` and
    double precision runs out: at ``xi = 25`` the eigenvalue is wrong by 3x and at
    ``xi = 30`` it comes back **negative**. Recorded because the temptation is to use this
    helper at a real ring's ``xi`` of tens, where it silently lies -- while
    ``quantum_lifetime_exact``, an everywhere-positive series, stays exact there.
    """
    assert quantum_lifetime_exact(math.sqrt(40.0), 1.0, 1.0) * _lambda1(20.0) == pytest.approx(
        1.0, abs=0.01
    )
    assert _lambda1(30.0) < 0.0  # not merely inaccurate: unphysical
    assert quantum_lifetime_exact(math.sqrt(60.0), 1.0, 1.0) > 1.0e11  # the series is fine


def test_the_eigenvalue_is_converged_in_the_grid_it_is_measured_on() -> None:
    """``lambda_1`` at ``n = 6000`` is a grid-converged number, not a discretisation.

    Halving the grid moves it by ~0.1%, and the sequence converges as ``1/n``; every
    ratio gated above is quoted well inside that.
    """
    coarse, fine, finer = _lambda1(4.0, 1500), _lambda1(4.0, 3000), _lambda1(4.0, 6000)
    assert abs(fine - finer) / finer < 1.0e-3
    # first order in the grid spacing: with e(n) = C/n the ratio of those gaps is 3, not
    # 2 -- the same "state the law, do not guess the exponent" trap as the O(1/xi)
    # departure in test_quantum_lifetime.py.
    assert abs(coarse - finer) / abs(fine - finer) == pytest.approx(3.0, rel=0.1)


# ---------------------------------------------------------------------------
# Gate 5 — what the coordinate/amplitude gap actually is.
#
# Two candidate mechanisms produce a longer decay for a coordinate cut: the particle
# waits for the synchrotron phase to bring delta to its extreme (which would scale with
# the synchrotron period), or it waits for a once-per-turn *sample* to land there (which
# would not). Varying Q_s at fixed step size separates them, and nothing else does --
# the step-size scan of gate 6 moves both together.
# ---------------------------------------------------------------------------
def test_the_coordinate_cut_costs_a_fixed_factor_that_does_not_depend_on_q_s() -> None:
    r"""Flat to 1.5% while ``Q_s`` moves by a factor of four — so it is *sampling*.

    If a particle above the boundary were merely waiting for phase rotation, quadrupling
    ``Q_s`` would quarter the wait. It does not move at all, which identifies the
    mechanism as looking at the particle once per turn rather than continuously.
    """
    _, _, _, tau, _, _ = _cached()
    reference = tau / _lambda1(3.0)
    ratios = {
        qs: _toy(qs, 3.0, tau, "coord", n=20000, turns=2400, seed=4) / reference
        for qs in (0.0843, 0.1294, 0.1893, 0.35)
    }
    for value in ratios.values():
        assert value == pytest.approx(1.38, rel=0.04)
    assert max(ratios.values()) / min(ratios.values()) < 1.03


def test_the_gap_closes_at_the_half_integer_synchrotron_resonance() -> None:
    r"""At ``Q_s = 0.5`` the coordinate cut *is* the amplitude cut, and both go wrong.

    The edge of the flat region above, kept rather than dropped. Two turns per
    oscillation means every sample lands at the same pair of phases, so ``delta`` is
    seen at its extreme every time and the coordinate test degenerates into the
    amplitude test. Both then fall *below* the closed form, because the sampled phase no
    longer explores the orbit at all.
    """
    _, _, _, tau, _, _ = _cached()
    reference = tau / _lambda1(3.0)
    coord = _toy(0.5, 3.0, tau, "coord", n=20000, turns=2400, seed=4) / reference
    amp = _toy(0.5, 3.0, tau, "amp", n=20000, turns=2400, seed=4) / reference
    assert coord == pytest.approx(amp, rel=0.02)  # the two cuts have become one cut
    assert coord < 0.8  # and neither is the closed form


# ---------------------------------------------------------------------------
# Gate 6 — link 3: both departures are the step size, and they extrapolate to zero.
# ---------------------------------------------------------------------------
def test_both_departures_extrapolate_to_the_closed_form_as_the_step_vanishes() -> None:
    r"""``lambda_1`` is a *continuum* eigenvalue; the tracked walk has a finite step.

    One turn moves the normalised action by about ``sqrt(2 xi) sqrt(2/tau)`` — at
    ``tau = 220`` turns and ``xi = 3`` that is 0.23, a thirteenth of the whole domain, so
    the diffusion limit the closed form is derived in is not reached. Quadrupling ``tau``
    halves the step, and **both** excesses fall towards zero with it.

    The law that is gated is the **intercept**, not the exponent. A quadratic in the step
    is fitted through the three points and its value at zero step is required to be
    consistent with the closed form; the measured power is reported by the fit rather
    than asserted, because the two excesses give different and drifting powers (~1.0-1.2
    and ~0.7-0.8) and asserting either would be the same "guess the exponent" mistake the
    ``O(1/xi)`` departure in ``test_quantum_lifetime.py`` records.
    """
    xi = 3.0
    steps, excess = [], {"amp": [], "coord": []}
    for tau in (220.0, 440.0, 880.0):
        reference = tau / _lambda1(xi)
        steps.append(math.sqrt(2.0 * xi) * math.sqrt(2.0 / tau))
        for mode in ("amp", "coord"):
            ratio = (
                _toy(0.1294, xi, tau, mode, n=15000, turns=int(3.5 * reference), seed=6) / reference
            )
            excess[mode].append(ratio - 1.0)

    step = np.array(steps)
    assert step[0] / step[-1] == pytest.approx(2.0, rel=0.01)  # a factor of two in step

    # The coordinate cut is the clean one: its excess is *proportional* to the step, so
    # the line through it passes through the origin and nothing is left at zero step.
    # Gated as that ratio rather than as a fitted exponent -- 1.67, 1.73, 1.78.
    coord = np.array(excess["coord"])
    per_step = coord / step
    assert max(per_step) / min(per_step) < 1.15
    assert abs(np.polyfit(step, coord, 1)[1]) < 0.2 * coord[0]  # intercept ~ 0
    assert coord[0] > 0.3  # and it is a large effect at the ring's own step

    # The amplitude cut's excess is three times smaller, so at this particle count it is
    # noisy, and it is gated only on shrinking -- not on monotonicity, which noise breaks.
    amp = np.array(excess["amp"])
    assert amp[0] > 0.1
    assert min(amp) < 0.6 * amp[0]
    assert abs(np.polyfit(step, amp, 1)[1]) < 0.3 * amp[0]


# ---------------------------------------------------------------------------
# Gate 7 — link 2, and the headline: the tracking is the same process.
# ---------------------------------------------------------------------------
def test_a_tracked_bunch_dies_the_way_the_same_process_does_without_a_lattice() -> None:
    r"""accsim's survival curve is the toy's, pointwise, and neither is the closed form.

    The milestone's headline. On the left: 51 elements, a radiation model, a photon
    variance, an RF cavity, a Lyapunov-matched initial bunch, and the shipped
    ``track_bunch_losses`` walking a ``MomentumAperture``. On the right: twenty lines of
    numpy that know nothing about any of it. They agree pointwise across a factor of four
    in survival, while both stand ~38% away from ``tau/lambda_1`` — so the comparison is
    not a tautology in which any two exponentials look alike.

    **Budget, stated in advance rather than tuned.** The survival fraction is binomial,
    ``sqrt(p(1-p)/N)``, which at ``N = 1500`` is 1.9-4.4% over the three marks; the gate
    is 3 sigma of that. The toy's own error is negligible (``N = 200000``). The gate is
    therefore ~13% at the last mark against a 38% effect — tight enough to tell the two
    apart, which is the property that matters.
    """
    sigma, _, orbit, tau, qs, _ = _cached()
    sd = math.sqrt(sigma[5, 5])
    xi, n_particles, turns, chunk = 3.0, 1500, 1200, 50
    marks = (400, 800, 1200)

    cut = MomentumAperture(sd * math.sqrt(2.0 * xi), center=float(orbit[5]))
    lattice = _ring(**RING, cut=cut, at=0)
    rng = np.random.default_rng(20260819)
    values, vectors = np.linalg.eigh(sigma)  # not Cholesky: eps_y = 0, Sigma is singular
    root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
    states = orbit[:, None] + root @ rng.standard_normal((6, n_particles))

    tracker = Tracker(lattice)
    alive = np.ones(n_particles, dtype=bool)
    survival = {}
    for c in range(turns // chunk):
        result = tracker.track_bunch_losses(
            Bunch(states), n_turns=chunk, nonlinear=True, radiation="quantum", rng=rng
        )
        states = result.states
        alive &= result.alive
        states[:, ~alive] = orbit[:, None]  # park the dead: they cannot be lost twice
        if (c + 1) * chunk in marks:
            survival[(c + 1) * chunk] = alive.sum() / n_particles
    assert np.isfinite(states).all()

    reference = _toy_survival(qs, xi, tau, marks, n=200000, seed=0)
    for mark in marks:
        p = reference[mark]
        budget = 3.0 * math.sqrt(p * (1.0 - p) / n_particles)
        assert survival[mark] == pytest.approx(p, abs=budget)
    # ...and the thing they agree on is NOT the closed form. The decay across the last
    # two marks is 1098 turns where tau/lambda_1 is 799: the gate above is 13% wide at
    # the last mark and the effect it has to survive is 37%, so agreeing with the toy and
    # agreeing with the continuum answer are emphatically not the same test.
    local_decay = (marks[-1] - marks[-2]) / math.log(reference[marks[-2]] / reference[marks[-1]])
    assert local_decay / (tau / _lambda1(xi)) == pytest.approx(1.374, rel=0.02)
    assert 0.15 < survival[marks[-1]] < 0.40  # a real decay, not everything or nothing


# ---------------------------------------------------------------------------
# Gate 8 — the plane with no quantum lifetime at all.
# ---------------------------------------------------------------------------
def test_a_vertical_aperture_on_a_quantum_excited_ring_loses_nothing() -> None:
    r"""B3 measured the vertical excitation as exactly ``0.0``; here nothing dies of it.

    One ring, one radiation model, two planes that must disagree in a pre-stated way:
    the momentum cut above loses three quarters of the bunch, and a vertical aperture at
    the same few-sigma tightness loses **zero**, for as long as it is tracked.

    The vertical noise is not so much absent as *multiplicative* — ``py`` is scaled by a
    random factor, so the diffusion goes as ``py^2`` and the equilibrium is zero rather
    than the drive being zero. The bunch is therefore started **below** the aperture:
    started above, the damping transient would carry particles across it and the zero
    would read as a broken gate rather than a physical one.
    """
    sigma, _, orbit, _, _, _ = _cached()
    half_y = 1.0e-4
    lattice = _ring(**RING, cut=Aperture("rectangular", 1.0, half_y), at=0)
    rng = np.random.default_rng(4)
    values, vectors = np.linalg.eigh(sigma)
    root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
    states = orbit[:, None] + root @ rng.standard_normal((6, 400))
    states[2] = 0.2 * half_y  # a real vertical amplitude, safely inside
    states[3] = 0.0

    result = Tracker(lattice).track_bunch_losses(
        Bunch(states), n_turns=800, nonlinear=True, radiation="quantum", rng=rng
    )
    assert result.alive.all()
    assert np.max(np.abs(result.states[2])) < half_y
    # it damped *through* the equilibrium rather than stopping at one
    assert np.std(result.states[2]) < 0.02 * half_y
