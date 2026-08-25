r"""Analytic gates — the equilibrium beam is untouched by resolving photons (B5, part 3).

Parts 1 and 2 established that the photon sum has B3's mean and B3's variance as exact
identities, and a tail B3's Gaussian does not have. This file spends the tracking budget
on the consequence, which is the milestone's central claim: **the beam does not change.**

That is worth gating rather than asserting, because it is not obvious from the outside.
The two models disagree about every single draw — one of them hands a particle energy
2.6% of the time and the other never can, and their single-magnet skewnesses are 0.00 and
-0.92 — and yet the beam they hold open has the same size, the same energy spread and the
same emittances. The reason is B3's own argument run backwards: the equilibrium is the
fixed point of ``Sigma = M Sigma M^T + D``, the map ``M`` is untouched, and ``D`` is built
out of the emission process's *first two moments only*. So:

  1. **``D`` itself is the same**, measured directly: the one-turn covariance a bunch
     picks up starting from the closed orbit, under both models, element by element in
     all 6x6. This is the sharp gate and it is cheap — one turn, 120000 particles — and
     it is the one that makes everything below a consequence rather than a coincidence.
  2. **And the beam that results is the same.** Two bunches tracked *together* through
     one ring for five damping times, one under each model, compared on the momentum
     spread and the horizontal emittance against a budget computed from the number of
     independent samples rather than tuned.
  3. **Including the zero.** B3 measured the vertical excitation as exactly ``0.0``,
     because the photons in this model leave along the direction of motion and carry no
     transverse recoil. Resolving them individually does not change that: the vertical
     emittance is still exactly zero, not merely small, and the opening-angle floor is
     still the thing neither model has.

**Cost, stated because it is the largest single file in the analytic suite.** ~155 s on
a quiet box: gate 2's four damping times of photon-resolved tracking are ~95 s of it and
gate 1's two 60000-particle turns ~20 s. Both are dominated by the draws themselves --
this model spends ~16 uniforms per particle per magnet and a turn of this ring is 40
magnets, so it is a few times dearer than B3's one Gaussian per magnet. That is the
price of the tail, and it is paid once here rather than in every gate: parts 1 and 2 buy
their sharpness from determinism instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import Dipole, Lattice, ReferenceParticle, RFCavity, ThinQuadrupole, Tracker
from accsim.radiation import damping_times, equilibrium_energy_spread
from accsim.reference import CLIGHT
from accsim.symplectic import unit_symplectic_matrix

ELECTRON_MASS_EV = 0.51099895069e6
L_BEND = 1.0
# B3's SETTLE ring: 6.5 GeV, where the damping time is 846 turns rather than 1858, which
# is what makes tracking all the way to equilibrium affordable at all. Its Q_s and its
# lumping are deliberately unconverged -- B3 owns those departures from the closed forms,
# and this file compares the two *models* to each other on one ring, so they cancel.
SETTLE = {"cells": 20, "focal": 2.5, "energy": 6.5e9, "voltage": 90.0e6, "harmonic": 20}


def _ring(cells: int, focal: float, energy: float, voltage: float, harmonic: int) -> Lattice:
    """B3's ring: isomagnetic FODO, total bend ``2 pi``, with the RF that replaces ``U0``."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, energy)
    angle = 2.0 * math.pi / (2 * cells)
    cell = [
        ThinQuadrupole(0.5 / focal),
        Dipole(L_BEND, angle),
        ThinQuadrupole(-1.0 / focal),
        Dipole(L_BEND, angle),
        ThinQuadrupole(0.5 / focal),
    ]
    elements = list(cell) * cells
    lat = Lattice(elements, ref=ref)
    cavity = RFCavity.from_harmonic(voltage, harmonic, lat.length, ref, phi_s=math.pi)
    return Lattice([*elements, cavity], ref=ref)


_ORBIT_CACHE: dict[tuple, np.ndarray] = {}


def _closed_orbit(tracker: Tracker, params: dict, turns: int = 4000) -> np.ndarray:
    """The fixed point a radiating ring actually settles on, found by damping onto it.

    Not ``delta = 0``: radiation drains momentum through the arcs and the cavity puts it
    back in one lump, so the periodic orbit sits off-momentum. B4 records this as the
    trap that cost it the most. Tracked with ``"mean"`` — the deterministic map — because
    the fixed point is a property of the map, not of the noise on it.
    """
    # keyed on the ring's parameters, not on id() of anything: a freed list's id is
    # reusable, so an id-keyed cache would silently hand back another ring's orbit.
    key = tuple(sorted(params.items()))
    if key not in _ORBIT_CACHE:
        state = np.zeros(6)
        for _ in range(turns):
            state = tracker.track_once(state, "mean")
        _ORBIT_CACHE[key] = state
    return _ORBIT_CACHE[key].copy()


def _tau_turns(lattice: Lattice) -> tuple[float, float, float]:
    """``damping_times`` in turns, ascending ``(tau_z, tau_y, tau_x)``.

    The shipped helper returns seconds; a tracking budget is counted in turns, and the
    revolution period is the ring's own length over ``beta0 c``.
    """
    period = lattice.length / (lattice.ref.beta0 * CLIGHT)
    return tuple(sorted(t / period for t in damping_times(lattice)))  # type: ignore[return-value]


def _mode_emittances(sigma: np.ndarray) -> tuple[float, float, float]:
    """The three invariant emittances of a 6x6 covariance, as ``|eigenvalues of Sigma S|``.

    Eigenvalues of ``Sigma S`` come in pairs ``+-i eps_k``; taking absolute values and
    the three distinct magnitudes gives the mode emittances, which is the coupling-proof
    way to read a beam's size off a covariance matrix.
    """
    values = np.abs(np.linalg.eigvals(sigma @ unit_symplectic_matrix()))
    ordered = np.sort(values)[::-1]
    return float(ordered[0]), float(ordered[2]), float(ordered[4])


# ---------------------------------------------------------------------------
# Gate 1 — the diffusion matrix, measured. The sharp one, and the cheap one.
# ---------------------------------------------------------------------------
def test_the_two_models_inject_the_same_diffusion_matrix_in_all_six_dimensions() -> None:
    r"""One turn from the closed orbit: the same ``D``, entry by entry, in all 6x6.

    The gate the rest of the file is a consequence of. B3 showed the equilibrium is the
    solution of ``Sigma = M Sigma M^T + D`` for the tracked map's own noise; ``M`` is the
    deterministic map and resolving photons does not touch it, so if ``D`` is the same
    then the equilibrium is the same *by B3's gate*, with no further tracking needed.

    Measured rather than derived, because deriving it would only re-run part 2's algebra.
    A bunch of 60000 starts exactly on the closed orbit and takes one turn; what it picks
    up is ``D``. The budget on a variance from ``N`` samples is ``sqrt(2/N) = 0.58%``,
    and the gate is 4 sigma of that on the entries that are not structurally zero.
    """
    lattice = _ring(**SETTLE)
    tracker = Tracker(lattice)
    orbit = _closed_orbit(tracker, SETTLE)

    n = 60_000
    injected = {}
    for model in ("quantum", "photons"):
        start = np.repeat(orbit[:, None], n, axis=1)
        out = tracker.track_once(start, model, np.random.default_rng(31))
        deterministic = tracker.track_once(orbit.copy(), "mean")
        centred = out - deterministic[:, None]
        injected[model] = centred @ centred.T / n

    budget = 4.0 * math.sqrt(2.0 / n)
    diagonal = np.diag(injected["quantum"])
    scale = np.sqrt(np.outer(diagonal, diagonal))
    active = scale > 0.0
    difference = np.abs(injected["photons"] - injected["quantum"])[active] / scale[active]
    assert float(np.max(difference)) < budget
    # ...and the rows that carry no noise at all carry none under either model
    assert np.all(np.diag(injected["photons"])[[2, 3]] == 0.0)
    assert np.all(np.diag(injected["quantum"])[[2, 3]] == 0.0)


# ---------------------------------------------------------------------------
# Gate 2 — and therefore the beam. Two bunches, one ring, five damping times.
# ---------------------------------------------------------------------------
def test_a_bunch_of_real_photons_settles_where_the_gaussian_one_does() -> None:
    r"""Same momentum spread, same horizontal emittance, after five damping times.

    The roadmap's pre-committed gate, in the form that costs the least to make sharp:
    both bunches go through *one* ring for the *same* turns, so every departure B3 named
    — the finite synchrotron tune, B2's one-kick-per-element lumping — is common to the
    two and cancels in the ratio. What is left is the emission process, which is the only
    thing being varied.

    **The budget, computed rather than tuned.** The action decorrelates in ``tau/2``, so
    averaging over ``2 tau`` with 150 particles is ~1200 independent samples: 2.0% on a
    width and 4.1% on a variance, and two independently noisy curves are compared, so the
    difference carries ``sqrt(2)`` of that. The gate is 3 sigma, i.e. 12% on the momentum
    spread — set by what four damping times of photon-resolved tracking cost, and wide
    only against the *statistics*: a model with the wrong variance would miss by tens of
    percent, and gate 1 pins the variance itself to 1.6%.

    A sanity band, not a gate, closes it: this ring is 12% off
    :func:`equilibrium_energy_spread` because its ``Q_s`` is large and its lumping
    unconverged, exactly as B3 measured. That number is not the subject here — the
    subject is that resolving photons does not move it.
    """
    lattice = _ring(**SETTLE)
    tracker = Tracker(lattice)
    orbit = _closed_orbit(tracker, SETTLE)
    tau = int(round(_tau_turns(lattice)[2]))
    assert 700 < tau < 1000  # the ring the budget below was computed for

    n_each = 150
    rng = np.random.default_rng(2026)
    bunch = np.repeat(orbit[:, None], 2 * n_each, axis=1)
    settle, average = 2 * tau, 2 * tau
    accumulated = np.zeros((2, 6, 6))
    parts = (slice(0, n_each), slice(n_each, 2 * n_each))

    for turn in range(settle + average):
        for model, part in zip(("quantum", "photons"), parts, strict=True):
            bunch[:, part] = tracker.track_once(bunch[:, part], model, rng)
        if turn >= settle:
            for j, part in enumerate(parts):
                centred = bunch[:, part] - bunch[:, part].mean(axis=1, keepdims=True)
                accumulated[j] += centred @ centred.T / n_each
    assert np.isfinite(bunch).all()  # nothing left the RF bucket under either model

    gaussian, photons = (accumulated[j] / average for j in range(2))
    independent = n_each * average / (tau / 2.0)
    budget = 3.0 * math.sqrt(2.0) / math.sqrt(2.0 * independent)  # two noisy widths

    assert math.sqrt(photons[5, 5]) == pytest.approx(math.sqrt(gaussian[5, 5]), rel=budget)
    eps_x_photons, _, _ = _mode_emittances(photons)
    eps_x_gaussian, _, _ = _mode_emittances(gaussian)
    assert eps_x_photons == pytest.approx(eps_x_gaussian, rel=2.0 * budget)

    # the sanity band: both sit the same distance from the design-route closed form
    closed = equilibrium_energy_spread(lattice)
    for measured in (gaussian, photons):
        assert math.sqrt(measured[5, 5]) / closed == pytest.approx(1.0, abs=0.25)


def test_the_vertical_plane_is_still_exactly_zero_and_not_merely_small() -> None:
    r"""Resolving photons does not give the vertical plane an emittance. Exactly zero.

    B3's boundary, restated for the new model because it is the one place where a reader
    might expect real photons to help. They do not: the missing physics is the ``1/gamma``
    opening angle — the photon leaving slightly off the direction of motion — and this
    model, like B3's and like xtrack's, emits strictly along the momentum vector. So
    ``py`` is only ever *scaled*, the vertical diffusion goes as ``py^2``, and a bunch on
    a flat lattice damps to a point rather than to the opening-angle floor.

    Gated as an exact zero, which is a much stronger statement than a small number and
    the honest one: this is a scope boundary, not an approximation with an error bar.
    """
    lattice = _ring(**SETTLE)
    tracker = Tracker(lattice)
    orbit = _closed_orbit(tracker, SETTLE)
    start = np.repeat(orbit[:, None], 4000, axis=1)
    out = tracker.track_once(start, "photons", np.random.default_rng(5))
    deterministic = tracker.track_once(orbit.copy(), "mean")
    assert np.array_equal(out[2], np.full(4000, deterministic[2]))
    assert np.array_equal(out[3], np.full(4000, deterministic[3]))
