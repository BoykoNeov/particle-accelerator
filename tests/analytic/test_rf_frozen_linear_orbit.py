"""Change B, step 3: the linear closed orbit holds the RF cavity fixed (``docs/ROADMAP.md``).

``closed_orbit`` solved ``(I - M4) x = k4 + d delta`` on ``lattice.one_turn_map()``, the cavity
live. With the cavity between the bends its ``zeta -> delta`` row reaches ``x`` through the
dispersion downstream, so the solve was skewed the same way the tracked orbit was (2.7% at
``delta = 1e-4``, 0.6% steered). It now solves on the frozen affine product, composed in
``Lattice.transfer_map``'s order, so a cavity-last ring reads the same floats.
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest

from accsim.coords import DELTA
from accsim.elements.corrector import Corrector
from accsim.lattice import Lattice
from accsim.orbit import (
    _affine_4d,
    closed_orbit,
    correct_orbit,
    orbit_response_matrix,
    propagate_orbit,
)
from accsim.scenario import load_scenario

sys.path.insert(0, os.path.dirname(__file__))

from test_rf_frozen_orbit import (  # noqa: E402
    CASE_IDS,
    CASES,
    KICK,
    MID,
    PERIODIC,
    PRESETS,
    _drop_cavity_point,
    _ring,
)

STEERERS = (40, 60)  # element indices of the two zero-kick correctors gate 2 steers with
MONITORS = list(range(1, 40))  # boundaries upstream of every cavity position used here


def _old_closed_orbit(lattice: Lattice, *, delta: float = 0.0) -> np.ndarray:
    """The solve as it was: the live one-turn map."""
    one_turn, kick = lattice.one_turn_map()
    m4, k4 = _affine_4d(one_turn, kick)
    rhs = k4 + one_turn[[0, 1, 2, 3], DELTA] * delta
    if not rhs.any():
        return np.zeros(4)
    return np.linalg.solve(np.eye(4) - m4, rhs)


def _steerable(pos: int | None, kick: float | None) -> tuple[Lattice, list[Corrector]]:
    """``_ring`` with two zero-kick horizontal correctors inserted before the cavity goes in."""
    sc = copy.deepcopy(PRESETS["electron-ring"])
    cav = next(e for e in sc["elements"] if e["type"] == "RFCavity")
    els = [e for e in sc["elements"] if e["type"] != "RFCavity"]
    if kick is not None:
        els = [*els[:27], {"type": "Corrector", "name": "CH", "kick_x": kick}, *els[27:]]
    for n, at in enumerate(STEERERS):
        els = [*els[:at], {"type": "Corrector", "name": f"C{n}", "kick_x": 0.0}, *els[at:]]
    els = [*els, cav] if pos is None else [*els[:pos], cav, *els[pos:]]
    sc["elements"] = els
    lat = load_scenario(sc).lattice
    return lat, [e for e in lat.elements if isinstance(e, Corrector) and e.name in ("C0", "C1")]


# ---------------------------------------------------------------------- gate 1
@pytest.mark.parametrize(("kick", "pos", "delta"), CASES, ids=CASE_IDS)
def test_the_linear_orbit_does_not_move_with_the_cavity(kick, pos: int, delta: float) -> None:
    """Gate 1: the same linear orbit wherever the cavity sits, along the ring and closed."""
    last, lat = _ring(None, kick=kick), _ring(pos, kick=kick)
    assert np.max(np.abs(closed_orbit(lat, delta=delta) - closed_orbit(last, delta=delta))) < 1e-17
    if delta == 0.0:  # propagate_orbit is the delta = 0 trajectory
        along = propagate_orbit(lat)
        assert np.max(np.abs(along[-1] - along[0])) < 1e-17
        want = np.array(propagate_orbit(last)[:-1])
        assert np.max(np.abs(np.array(_drop_cavity_point(along, pos)) - want)) < 1e-17


@pytest.mark.parametrize(("kick", "pos", "delta"), CASES, ids=CASE_IDS)
def test_the_old_linear_orbit_moved(kick, pos: int, delta: float) -> None:
    """Gate 1's control: the live one-turn map misses by more than 4e-3 of the orbit."""
    want = closed_orbit(_ring(None, kick=kick), delta=delta)
    got = _old_closed_orbit(_ring(pos, kick=kick), delta=delta)
    assert np.max(np.abs(got - want)) > 4e-3 * np.max(np.abs(want))


# ---------------------------------------------------------------------- gate 2
@pytest.mark.parametrize("pos", MID)
def test_the_response_matrix_does_not_move_with_the_cavity(pos: int) -> None:
    """Gate 2: the model response of two correctors, upstream monitors."""
    want = orbit_response_matrix(*_steerable(None, None), monitors=MONITORS)
    got = orbit_response_matrix(*_steerable(pos, None), monitors=MONITORS)
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))


@pytest.mark.parametrize("pos", MID)
def test_the_orbit_correction_does_not_move_with_the_cavity(pos: int) -> None:
    """Gate 2: steering out the ``KICK`` orbit asks the same of the two correctors."""
    want = correct_orbit(*_steerable(None, KICK), monitors=MONITORS).kicks
    got = correct_orbit(*_steerable(pos, KICK), monitors=MONITORS).kicks
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))


# ---------------------------------------------------------------------- gate 3
@pytest.mark.parametrize("pid", PERIODIC)
@pytest.mark.parametrize("delta", [0.0, 1e-4])
def test_nothing_else_moves_on_the_presets(pid: str, delta: float) -> None:
    """Gate 3: on every periodic preset as shipped the new solve is the old one, bit for bit."""
    lat = load_scenario(PRESETS[pid]).lattice
    assert np.array_equal(closed_orbit(lat, delta=delta), _old_closed_orbit(lat, delta=delta))


@pytest.mark.parametrize("delta", [0.0, 1e-4])
def test_nothing_else_moves_on_the_steered_cavity_last_ring(delta: float) -> None:
    """Gate 3: the steered ring with its cavity last, bit for bit."""
    lat = _ring(None, kick=KICK)
    assert np.array_equal(closed_orbit(lat, delta=delta), _old_closed_orbit(lat, delta=delta))
