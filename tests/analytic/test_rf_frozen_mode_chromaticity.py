"""Change B, step 4: ``mode_chromaticity`` with the RF cavity between the bends (``docs/ROADMAP.md``).

The scan's orbit at ``+-delta`` holds the cavity fixed since step 1, and ``propagate_normal_form``
already froze. The leftover was the start's normal form, taken of the *unfrozen* product of the
on-orbit maps: ``-6.09`` against ``-6.42`` in mode 1 on the skew ``electron-ring``, cavity at 100.
With that frozen too the refusal goes.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

import accsim as ac
from accsim import orbit as orbit_module
from accsim import twiss as twiss_module
from accsim.coords import DIM
from accsim.scenario import load_scenario
from accsim.twiss import (
    UnstableLatticeError,
    _mode_tune_scan,
    mode_chromaticity,
    normal_form,
    propagate_normal_form,
)

sys.path.insert(0, os.path.dirname(__file__))

from test_rf_frozen_orbit import MID, PERIODIC, PRESETS, SKEW, _ring  # noqa: E402


def _old_scan(lattice, delta: float = 1e-4, step: float = 1e-7) -> tuple[float, float]:
    """``_mode_tune_scan`` as it was: the start's normal form of the unfrozen product."""
    d = twiss_module._chromatic_step(delta)
    full = []
    for s in (+d, -d):
        maps = twiss_module._on_orbit_maps(lattice, s, step)
        one_turn = np.eye(DIM)
        for m in maps:
            one_turn = m @ one_turn
        end = propagate_normal_form(lattice, normal_form(one_turn, method="4d"), maps=maps)[-1]
        full.append(np.array(end.mu) / (2.0 * math.pi))
    slope = (full[0] - full[1]) / (2.0 * d)
    return float(slope[0]), float(slope[1])


# ---------------------------------------------------------------------- gate 1
@pytest.mark.parametrize("pos", MID)
def test_the_mode_chromaticity_does_not_move_with_the_cavity(pos: int) -> None:
    """Gate 1: both modes are the cavity-last ring's, and the refusal is gone."""
    want = mode_chromaticity(_ring(None, SKEW))
    got = mode_chromaticity(_ring(pos, SKEW))
    assert got == pytest.approx(want, rel=1e-8)


@pytest.mark.parametrize("pos", MID)
def test_the_live_orbit_was_off(monkeypatch: pytest.MonkeyPatch, pos: int) -> None:
    """Gate 1's control: the frozen start on the old, live orbit puts mode 1 more than 1% away
    (``-6.09`` / ``-6.19`` against ``-6.42``, the measurement that made it refuse)."""
    want = mode_chromaticity(_ring(None, SKEW))
    monkeypatch.setattr(orbit_module, "_step", lambda elem, state, ref: elem.track(state, ref))
    old = _mode_tune_scan(_ring(pos, SKEW), 1e-4, 1e-7)
    assert abs(old[0] / want[0] - 1.0) > 1e-2


@pytest.mark.parametrize("pos", MID)
def test_half_the_old_route_has_no_normal_form(pos: int) -> None:
    """Found on the first run: the held orbit with the unfrozen start is not a wrong slope but
    no slope. That start's 4D block is not symplectic (eigenvalue moduli 1.0024 at 100), so the
    normal form refuses it."""
    with pytest.raises(UnstableLatticeError):
        _old_scan(_ring(pos, SKEW))


# ---------------------------------------------------------------------- gate 3
@pytest.mark.parametrize("pid", PERIODIC)
def test_uncoupled_rings_still_return_the_chromaticity(pid: str) -> None:
    """Gate 3: on an uncoupled preset nothing is scanned; the per-plane value is returned."""
    lat = load_scenario(PRESETS[pid]).lattice
    frozen = twiss_module._frozen_one_turn(twiss_module._element_maps(lat))
    if twiss_module._coupling_norm(frozen) > 1e-9:
        pytest.skip("a design-coupled preset: scanned, not this gate's")
    assert mode_chromaticity(lat) == ac.chromaticity(lat)


def test_the_cavity_last_skew_ring_moves_only_by_round_off() -> None:
    """Gate 3: freezing replaces the finite-difference ``delta`` rows' noise, nothing more."""
    lat = _ring(None, SKEW)
    assert _mode_tune_scan(lat, 1e-4, 1e-7) == pytest.approx(_old_scan(lat), rel=1e-9)
