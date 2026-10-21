"""Change B, step 2: the spin closed state is timed so the cavity sees the cavity-last slip.

``spin._closed_state`` puts a spin's carrier on the 4D closed orbit at ``zeta = 0`` and tracks it
**live** (the spin map needs the real turn). With the cavity between the bends the cavity then
sees the slip made before it, kicks ``delta``, and the carrier is not a fixed point of the turn
its spin is carried around. The state now starts at ``zeta = s - z_c`` (``z_c`` the ``zeta`` at
the first cavity, ``s`` the turn's slip, from one held-``delta`` walk), so the cavity sees what
the cavity-last ring's sees; with the cavity last the start is exactly ``0.0``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from accsim import spin as spin_module
from accsim.coords import ZETA
from accsim.orbit import _tracked_turn_6d, closed_orbit_6d
from accsim.scenario import load_scenario
from accsim.spin import _closed_state, spin_one_turn_matrix, spin_tune

sys.path.insert(0, os.path.dirname(__file__))

from test_rf_frozen_orbit import KICK, MID, PERIODIC, PRESETS, _ring  # noqa: E402


def _at_zeta_zero(lat, delta=None):
    """The state as it was built before: the same orbit and momentum, ``zeta = 0``."""
    state = _closed_state(lat, None, delta)
    state[ZETA] = 0.0
    return state


# ---------------------------------------------------------------------- gate 1
@pytest.mark.parametrize("pos", [None, *MID])
def test_the_spin_carrier_is_a_fixed_point_of_the_live_turn(pos: int | None) -> None:
    """Gate 1: one live turn returns the state, and it is the 6D closed orbit, both to the 4D
    solve's own ``tol`` (on the cavity-last ring it stops ``1.4e-15`` short). ``zeta`` closes to
    ``closed_orbit_delta``'s own stopping test, ``|slip| < 1e-15 C``: the first run found the
    cavity-last ring, which this change does not touch, at ``3.5e-14`` there."""
    lat = _ring(pos, kick=KICK)
    state = _closed_state(lat, None)
    residual = _tracked_turn_6d(lat, state) - state
    assert np.max(np.abs(np.delete(residual, ZETA))) < 1e-14
    assert abs(residual[ZETA]) < 1e-15 * lat.length
    # The 6D orbit's cavity sees zeta = 0; this one's sees the slip left by that test.
    miss = state - closed_orbit_6d(lat)
    assert np.max(np.abs(np.delete(miss, ZETA))) < 1e-14
    assert abs(miss[ZETA]) < 1e-15 * lat.length


@pytest.mark.parametrize("pos", MID)
def test_at_zeta_zero_the_cavity_kicked_it(pos: int) -> None:
    """Gate 1's control: started at ``zeta = 0`` the live turn moves ``delta`` by > 1e-9."""
    lat = _ring(pos, kick=KICK)
    state = _at_zeta_zero(lat)
    assert abs(_tracked_turn_6d(lat, state)[-1] - state[-1]) > 1e-9


# ---------------------------------------------------------------------- gate 2
@pytest.mark.parametrize("pos", MID)
def test_the_spin_does_not_move_with_the_cavity(pos: int) -> None:
    """Gate 2: the one-turn spin rotation and the spin tune are the cavity-last ring's."""
    last, lat = _ring(None, kick=KICK), _ring(pos, kick=KICK)
    np.testing.assert_allclose(
        spin_one_turn_matrix(lat), spin_one_turn_matrix(last), rtol=0, atol=1e-12
    )
    assert spin_tune(lat) == pytest.approx(spin_tune(last), rel=0, abs=1e-12)


# ---------------------------------------------------------------------- gate 3
@pytest.mark.parametrize("pid", PERIODIC)
@pytest.mark.parametrize("delta", [None, 1e-4])
def test_nothing_else_moves_on_the_presets(pid: str, delta: float | None) -> None:
    """Gate 3: cavity-last and cavity-free rings start at ``zeta = 0.0`` exactly, as before."""
    lat = load_scenario(PRESETS[pid]).lattice
    state = _closed_state(lat, None, delta)
    assert state[ZETA] == 0.0
    assert np.array_equal(state, _at_zeta_zero(lat, delta))


def test_nothing_else_moves_on_the_steered_cavity_last_ring() -> None:
    """Gate 3: the steered ring with its cavity last, bit for bit."""
    lat = _ring(None, kick=KICK)
    assert _closed_state(lat, None)[ZETA] == 0.0
    assert spin_module._closed_state(lat, None, 1e-4)[ZETA] == 0.0
