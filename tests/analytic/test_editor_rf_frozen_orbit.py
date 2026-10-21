"""Change B, step 5: the editor's orbit and mode chromaticity hold the RF cavity fixed.

The editor's copies of the 4D closed orbit had the package's leftovers: the main view's affine
orbit (``closedOrbit``) solved on the live ``transferMap``, and the Injection tab's per-mode
chromaticity refused a coupled ring with its cavity between the bends because its off-momentum
orbit (``closedOrbit4``) and its Jacobian (``orbitMap``) tracked the cavity live. They now use a
frozen affine product and a held-``delta`` turn (``trackTurnHeld``), the package's steps 1 to 4.

Node-driven (``editor/inject-selftest.js``); skipped without Node.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_rf_frozen_orbit import KICK, MID, PRESETS, SKEW

import accsim as ac
from accsim.scenario import load_scenario

EDITOR = Path(__file__).resolve().parents[2] / "editor"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on the PATH")


def _scenario(pos: int | None, k1sl: float | None = None, kick: float | None = None) -> dict:
    """``test_rf_frozen_orbit._ring``'s rings, as scenario dicts for the editor."""
    sc = copy.deepcopy(PRESETS["electron-ring"])
    cav = next(e for e in sc["elements"] if e["type"] == "RFCavity")
    els = [e for e in sc["elements"] if e["type"] != "RFCavity"]
    if kick is not None:
        corrector = {"type": "Corrector", "name": "CH", "kick_x": kick, "kick_y": 0.0}
        els = [*els[:27], corrector, *els[27:]]
    if k1sl is not None:
        els = [*els[:5], {"type": "ThinSkewQuadrupole", "name": "SQ", "k1sl": k1sl}, *els[5:]]
    sc["elements"] = [*els, cav] if pos is None else [*els[:pos], cav, *els[pos:]]
    return sc


def _node(cases: list[dict]) -> list[dict]:
    proc = subprocess.run(
        [NODE, str(EDITOR / "inject-selftest.js")],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------- gate 1
@pytest.mark.parametrize("pos", MID)
def test_the_main_views_orbit_does_not_move_with_the_cavity(pos: int) -> None:
    """Gate 1: the editor's affine orbit is the package's, and the cavity-last ring's."""
    mid, last = _node(
        [
            {"scenario": _scenario(pos, kick=KICK), "mode": "design"},
            {"scenario": _scenario(None, kick=KICK), "mode": "design"},
        ]
    )
    want = ac.closed_orbit(load_scenario(_scenario(pos, kick=KICK)).lattice)
    assert np.max(np.abs(np.array(mid["closedOrbit0"]) - want)) < 1e-17
    assert np.max(np.abs(np.array(mid["closedOrbit0"]) - np.array(last["closedOrbit0"]))) < 1e-17


# ---------------------------------------------------------------------- gate 2
@pytest.mark.parametrize("pos", MID)
def test_the_editors_mode_chromaticity_does_not_move_with_the_cavity(pos: int) -> None:
    """Gate 2: answered, the package's to the port gate's tolerance, the cavity-last ring's."""
    mid, last = _node(
        [
            {"scenario": _scenario(pos, SKEW), "mode": "chroma"},
            {"scenario": _scenario(None, SKEW), "mode": "chroma"},
        ]
    )
    assert mid["error"] is None, mid["error"]
    want = ac.mode_chromaticity(load_scenario(_scenario(pos, SKEW)).lattice)
    np.testing.assert_allclose(mid["chromaticity"], want, rtol=3e-8)
    np.testing.assert_allclose(mid["chromaticity"], last["chromaticity"], rtol=1e-8)
