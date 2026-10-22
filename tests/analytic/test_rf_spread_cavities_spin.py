"""The spin carrier refuses a ring whose RF cavities do not all see the same timing.

``spin._cavity_timing`` (change B step 2) starts the carrier so the **first** cavity sees what a
cavity-last ring's sees. A second cavity elsewhere sees the slip made between the two, kicks
``delta``, and the carrier is not a fixed point of the live turn its spin is carried around: no
start can fix that, because the ring's 6D closed orbit then changes ``delta`` part-way round and
is not the held-``delta`` 4D orbit at all. Before this file such rings returned a number. Now
every spin entry point raises ``NotImplementedError`` when, on the held-``delta`` walk, a later
cavity's ``zeta`` differs from the first's by more than ``closed_orbit_delta``'s own stopping
test, ``1e-15 C``. Cavities together (or first and last, which the turn joins) are still served.

The ring is ``electron-ring`` with its cavity split into two half-voltage cavities. Measured on
the steered ring: the spread cavities differ by ``8e-5`` to ``1e-4`` in ``zeta`` (nine orders
over the bound), the co-located ones by ``0.0``, first-and-last by ``3.5e-14``; the unsteered
ring closes at ``delta = 0`` and every cavity sees ``0.0``.
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest

from accsim import radiation
from accsim import spin as spin_module
from accsim.orbit import _tracked_turn_6d
from accsim.scenario import load_scenario
from accsim.spin import _closed_state, spin_tune

sys.path.insert(0, os.path.dirname(__file__))

from test_rf_frozen_orbit import KICK, PRESETS  # noqa: E402

SPREAD = [[30, 90], [50, 100], [None, 100]]
TOGETHER = [[None, None], [100, 100], [0, None]]

ENTRY_POINTS = {
    "spin_one_turn_matrix": spin_module.spin_one_turn_matrix,
    "closed_spin_solution": spin_module.closed_spin_solution,
    "spin_tune": spin_module.spin_tune,
    "propagate_spin_solution": spin_module.propagate_spin_solution,
    "spin_orbit_coupling": spin_module.spin_orbit_coupling,
    "propagate_spin_orbit_coupling": spin_module.propagate_spin_orbit_coupling,
    "polarization_integrals": radiation.polarization_integrals,
    "depolarization_integrals": radiation.depolarization_integrals,
}


def _ring(positions: list[int | None], kick: float | None = KICK):
    """``electron-ring`` with its cavity split in equal parts at ``positions`` (``None``: last),
    and ``kick`` a horizontal corrector after the 27th element, as in ``test_rf_frozen_orbit``."""
    sc = copy.deepcopy(PRESETS["electron-ring"])
    cav = next(e for e in sc["elements"] if e["type"] == "RFCavity")
    els = [e for e in sc["elements"] if e["type"] != "RFCavity"]
    if kick is not None:
        corrector = {"type": "Corrector", "name": "CH", "kick_x": kick, "kick_y": 0.0}
        els = [*els[:27], corrector, *els[27:]]
    parts = []
    for i, pos in enumerate(positions):
        part = {**cav, "name": f"RF{i}", "voltage": cav["voltage"] / len(positions)}
        parts.append((len(els) if pos is None else pos, i, part))
    # Insert from the back so earlier positions still index the cavity-free list.
    for pos, _, part in sorted(parts, key=lambda t: (t[0], t[1]), reverse=True):
        els.insert(pos, part)
    sc["elements"] = els
    return load_scenario(sc).lattice


def _ids(cases):
    return ["-".join("last" if p is None else str(p) for p in c) for c in cases]


@pytest.fixture
def unrefused(monkeypatch):
    """The carrier as it was before this file: no check on the later cavities."""
    monkeypatch.setattr(spin_module, "_refuse_spread_cavities", lambda lattice, zetas: None)


# ---------------------------------------------------------------------- gate 1
@pytest.mark.parametrize("name", ENTRY_POINTS)
@pytest.mark.parametrize("positions", SPREAD, ids=_ids(SPREAD))
def test_every_spin_entry_point_refuses_spread_cavities(positions, name: str) -> None:
    """Gate 1: all eight public routes onto the carrier refuse, none returns a number."""
    lat = _ring(positions)
    with pytest.raises(NotImplementedError, match="cavities"):
        ENTRY_POINTS[name](lat)


def test_an_explicit_momentum_is_refused_too() -> None:
    """Gate 1: off the closed momentum the unsteered ring's spread cavities see different slips."""
    with pytest.raises(NotImplementedError, match="cavities"):
        _closed_state(_ring([30, 90], kick=None), None, 1e-4)


# ------------------------------------------------------------ gate 1's control
@pytest.mark.parametrize("positions", SPREAD, ids=_ids(SPREAD))
def test_without_the_refusal_the_carrier_did_not_close(unrefused, positions) -> None:
    """Gate 1's control: unchecked, one live turn misses the carrier by more than 1e-7. The
    witness is the orbit, not the spin tune: with ``[last, 100]`` the spin tune came out the
    single cavity's to the bit while the carrier missed by ``1.5e-6``."""
    lat = _ring(positions)
    state = _closed_state(lat, None)
    assert np.max(np.abs(_tracked_turn_6d(lat, state) - state)) > 1e-7


# ---------------------------------------------------------------------- gate 2
@pytest.mark.parametrize("positions", TOGETHER, ids=_ids(TOGETHER))
def test_cavities_together_are_still_served(positions) -> None:
    """Gate 2: co-located halves (and first-and-last, which the turn joins) close the carrier to
    the momentum solve's stopping test and give the single cavity's spin tune."""
    lat = _ring(positions)
    state = _closed_state(lat, None)
    assert np.max(np.abs(_tracked_turn_6d(lat, state) - state)) < 1e-15 * lat.length
    assert spin_tune(lat) == pytest.approx(spin_tune(_ring([None])), rel=0, abs=1e-12)


@pytest.mark.parametrize("positions", SPREAD, ids=_ids(SPREAD))
def test_spread_cavities_on_the_unsteered_ring_are_still_served(positions) -> None:
    """Gate 2: unsteered, the ring closes at ``delta = 0``, every cavity sees ``zeta = 0`` and the
    carrier closes exactly; the check must not refuse what is right."""
    lat = _ring(positions, kick=None)
    state = _closed_state(lat, None)
    assert np.array_equal(_tracked_turn_6d(lat, state), state)
    assert spin_tune(lat) == spin_tune(_ring([None], kick=None))
