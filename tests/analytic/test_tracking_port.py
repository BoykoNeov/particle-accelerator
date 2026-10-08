"""The editor's turn-by-turn tracker, held against the package's element-by-element one.

``editor/accsim-track.js`` is a port of every element's ``track()`` — the exact drift, the
momentum-dependent thick quadrupole, the exact and the curved combined-function bend, the
pole-face fringes, the multipole kicks, the solenoid, the wiggler, the RF cavity and the
misalignment wrapper. It exists so the editor can show a beam *filament* after injection,
which the one-turn matrix cannot do: a matrix has the same tune at every energy.

Same rule as the optics port (``test_scenario.py``): **disagreement is a bug on one side,
never a tolerance to loosen.** The JavaScript is the same arithmetic in the same order, and
the floor it reaches is round-off — measured 2026-10-07 at ``<= 5e-14`` relative after 30
turns of every ring preset and ``<= 1e-15`` through one pass of every element type. The
gates sit two orders above that floor.

A port gate is only as good as the states it is fed, so each case also asserts that the
package's *exact* tracker and its *matrix* path disagree by far more than the gate on those
same states. Without that, a port that silently linearised (dropped ``k1/(1+delta)``, say)
could pass against particles too close to the axis to tell.

Skipped, not failed, when Node is not on the PATH (CI runs without it).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from accsim.lattice import Lattice
from accsim.scenario import load_scenario
from accsim.tracking import Bunch, Tracker

ROOT = Path(__file__).resolve().parents[2]
EDITOR = ROOT / "editor"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on the PATH")

GATE = 1e-12  # relative, per coordinate; the floor is <= 5e-14 (see the module docstring)


def _presets() -> list[dict]:
    text = (EDITOR / "presets.js").read_text(encoding="utf-8")
    marker = "\nvar ACCSIM_PRESETS = "
    start = text.index(marker) + len(marker)
    return json.loads(text[start : text.rindex("];") + 1])


PRESETS = _presets()
PRESET_IDS = [p["id"] for p in PRESETS]

_REF = {"species": "electron", "energy_eV": 1.0e9, "energy_mode": "total"}

# Every element type the editor offers, with the options it exposes switched on: a
# displacement, a roll, a combined-function bend with pole faces, both bodies with the
# nonlinear fringe faces, a rolled straight gradient magnet, an off-crest RF cavity.
ZOO = {
    "format": "accsim-scenario/1",
    "name": "every element",
    "periodic": False,
    "reference": _REF,
    "elements": [
        {"type": "Drift", "length": 0.7},
        {"type": "Quadrupole", "length": 0.4, "k1": 1.3, "dx": 2e-4, "dy": -1e-4},
        {"type": "Quadrupole", "length": 0.4, "k1": -1.1, "roll": 0.03, "dx": 1e-4},
        {"type": "ThinQuadrupole", "k1l": 0.6, "roll": -0.02, "dy": 3e-4},
        {"type": "Dipole", "length": 1.2, "angle": 0.15},
        {"type": "Dipole", "length": 1.0, "angle": 0.1, "k1": 0.4, "e1": 0.05, "e2": 0.03},
        {"type": "Dipole", "length": 1.0, "angle": 0.12, "e1": 0.06, "e2": 0.04, "fringe": True},
        {
            "type": "Dipole",
            "length": 0.9,
            "angle": 0.08,
            "k1": -0.3,
            "e1": 0.02,
            "e2": 0.05,
            "fringe": True,
        },
        {"type": "Dipole", "length": 0.5, "angle": 0.0, "k1": 0.7, "roll": 0.04},
        {"type": "Sextupole", "length": 0.3, "k2": 12.0, "dx": 1e-4},
        {"type": "ThinSextupole", "k2l": -3.0, "roll": 0.1},
        {"type": "Octupole", "length": 0.2, "k3": 300.0},
        {"type": "ThinOctupole", "k3l": -40.0, "dy": 2e-4},
        {"type": "SkewQuadrupole", "length": 0.3, "k1s": 0.5},
        {"type": "ThinSkewQuadrupole", "k1sl": -0.2, "dx": 1e-4},
        {"type": "Solenoid", "length": 1.5, "ks": 0.8, "roll": 0.2},
        {"type": "Corrector", "kick_x": 1e-4, "kick_y": -2e-4, "dx": 1e-4},
        {"type": "Wiggler", "period": 0.1, "h0": 0.45, "periods": 10, "dy": 1e-4},
        {"type": "RFCavity", "voltage": 2.0e6, "frequency": 5.0e8, "phi_s": 2.8},
        {"type": "Aperture", "shape": "elliptical", "half_x": 0.05, "half_y": 0.03},
        {
            "type": "Collimator",
            "shape": "rectangular",
            "half_x": 0.05,
            "half_y": 0.03,
            "length": 0.5,
        },
        {"type": "Drift", "length": 0.3},
    ],
}
# Types whose map is exactly linear (or the identity): a matrix-vs-exact contrast on them
# would be zero by construction, so they are left out of the "not blind" check. A thick
# Collimator is not among them: it is the exact drift of its length (2026-10-08), and the
# zoo's is long enough that the identity it used to be fails the gate by orders.
_LINEAR = {"ThinQuadrupole", "ThinSkewQuadrupole", "Corrector", "Aperture"}


def _states(n: int, seed: int, ax: float, ap: float, az: float, ad: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (n, 6)) * np.array([ax, ap, ax, ap, az, ad])


def _node(cases: list[dict]) -> list[dict]:
    assert NODE is not None
    proc = subprocess.run(
        [NODE, str(EDITOR / "track-selftest.js")],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _rel_dev(a: np.ndarray, b: np.ndarray) -> float:
    """Largest deviation of each coordinate relative to that coordinate's own largest
    magnitude, maxed over the six — px is ~beta times smaller than x, so one shared
    absolute scale would be far looser on the slopes."""
    scale = np.max(np.abs(b), axis=tuple(range(b.ndim - 1)))
    dev = np.max(np.abs(a - b), axis=tuple(range(a.ndim - 1)))
    return float(np.max(dev / np.where(scale > 0.0, scale, 1.0)))


# --- one pass, element by element ------------------------------------------------------
def test_every_element_type_matches_its_python_map() -> None:
    """One pass through every element type, compared after **each** element.

    Element-by-element so that a failure names the map that broke rather than a turn
    number. The states reach ``|delta| = 5e-3`` and millimetres of amplitude, where every
    nonlinear and chromatic term in the zoo is far above the gate (asserted per element).
    """
    states = _states(40, seed=7, ax=3e-3, ap=3e-4, az=5e-3, ad=5e-3)
    (res,) = _node([{"scenario": ZOO, "states": states.tolist(), "mode": "elements"}])
    assert "error" not in res, res.get("error")
    js = np.array(res["states"], dtype=float)  # (particle, element, 6)

    lat = load_scenario(ZOO).lattice
    py = np.zeros_like(js)
    lin = np.zeros_like(js)
    for p in range(states.shape[0]):
        s = states[p].copy()
        for i, elem in enumerate(lat.elements):
            lin[p, i] = elem.matrix(lat.ref) @ s + elem.kick(lat.ref)
            s = elem.track(s, lat.ref)
            py[p, i] = s

    for i, elem in enumerate(lat.elements):
        name = f"#{i} {type(elem).__name__}"
        assert _rel_dev(js[:, i], py[:, i]) < GATE, f"{name}: the port disagrees"
        if ZOO["elements"][i]["type"] not in _LINEAR:
            # Not blind: on these states the exact map is far from its own linearisation.
            assert _rel_dev(lin[:, i], py[:, i]) > 1e3 * GATE, f"{name}: states too tame"


# --- many turns, every preset ------------------------------------------------------------
@pytest.mark.parametrize("preset", PRESETS, ids=PRESET_IDS)
def test_every_preset_tracks_turn_for_turn(preset: dict) -> None:
    """30 turns of every ring preset (one pass of a transfer line), state after each turn.

    Apertures ignored here — this is the maps. The particles carry ``|delta| <= 2e-3``,
    so each ring's whole natural chromaticity is in play, and after 30 turns a mis-scaled
    ``1/(1+delta)`` anywhere would have rotated them visibly off the package's.
    """
    turns = 30 if preset.get("periodic", True) else 1
    states = _states(6, seed=11, ax=1e-3, ap=1e-4, az=1e-3, ad=2e-3)
    (res,) = _node(
        [{"scenario": preset, "states": states.tolist(), "turns": turns, "mode": "turns"}]
    )
    assert "error" not in res, res.get("error")
    js = np.array(res["states"], dtype=float)  # (particle, turn, 6)

    lat = load_scenario(preset).lattice
    tracker = Tracker(lat)
    M, k = lat.transfer_map()
    py = np.zeros_like(js)
    lin = np.zeros_like(js)
    for p in range(states.shape[0]):
        s, m = states[p].copy(), states[p].copy()
        for t in range(turns):
            s = tracker.track_once(s)
            m = M @ m + k
            py[p, t], lin[p, t] = s, m

    assert _rel_dev(js, py) < GATE
    assert _rel_dev(lin, py) > 1e3 * GATE  # the matrix path is visibly different physics


# --- losses ------------------------------------------------------------------------------
def test_losses_match_the_package_turn_and_element() -> None:
    """A tight circular aperture on the electron ring: who is lost, on which turn, where.

    The rule is ``Tracker.track_bunch_losses``: the survival test runs on the state *after*
    the acceptance element, and a lost particle is frozen there. The bunch is spread so the
    aperture takes some particles on the first turn, some later (as their betatron phase
    brings them out) and spares the rest — each outcome is asserted to occur.
    """
    preset = next(p for p in PRESETS if p["id"] == "electron-ring")
    scenario = dict(preset)
    scenario["elements"] = list(preset["elements"]) + [
        {"type": "Aperture", "name": "AP", "shape": "circular", "half_x": 1.5e-3}
    ]
    states = _states(60, seed=3, ax=2.2e-3, ap=1.5e-4, az=1e-3, ad=1e-3)
    states[:, 0] *= 0.6  # keep the start inside, so turn 0 is not every loss
    turns = 40
    (res,) = _node(
        [{"scenario": scenario, "states": states.tolist(), "turns": turns, "mode": "losses"}]
    )
    assert "error" not in res, res.get("error")

    lat = load_scenario(scenario).lattice
    out = Tracker(lat).track_bunch_losses(Bunch(states.T.copy()), n_turns=turns, nonlinear=True)

    js_turn = np.array(res["lostTurn"])
    assert np.array_equal(js_turn, out.loss_turn)
    assert np.array_equal(np.array(res["lostAt"]), out.loss_element)
    assert _rel_dev(np.array(res["states"], dtype=float), out.states.T) < GATE

    assert (js_turn == 0).any() and (js_turn > 0).any() and (js_turn < 0).any()
    assert set(np.array(res["lostAt"])[js_turn >= 0]) == {len(lat.elements) - 1}


def test_a_thick_collimator_loses_at_the_same_face() -> None:
    """A 2 m jaw tested at both faces, as ``track_bunch_losses`` does.

    The face is read off the frozen state: a particle lost at the entry face is frozen
    before the jaw drifts it, one lost at the exit face after — 1 mm apart here, which a
    port checking the wrong face (or only one) could not hide below the gate. Each of the
    three outcomes — entry face, exit face, survivor — is asserted to occur.
    """
    a, L = 5.0e-3, 2.0
    scenario = {
        "format": "accsim-scenario/1",
        "name": "a thick jaw",
        "periodic": False,
        "reference": _REF,
        "elements": [
            {"type": "Drift", "length": 1.0},
            {"type": "Collimator", "shape": "rectangular", "half_x": a, "half_y": a, "length": L},
            {"type": "Drift", "length": 1.0},
        ],
    }
    # entry face position = x0 + px * 1 m; exit = that + px * L. Each is >= 0.1a off the edge.
    x_entry = np.array([1.2, 0.8, -0.8, 0.9, 0.3]) * a
    x_exit = np.array([0.8, 1.2, -1.2, -0.9, 0.6]) * a
    states = np.zeros((5, 6))
    states[:, 1] = (x_exit - x_entry) / L
    states[:, 0] = x_entry - states[:, 1]
    (res,) = _node(
        [{"scenario": scenario, "states": states.tolist(), "turns": 1, "mode": "losses"}]
    )
    assert "error" not in res, res.get("error")

    lat = load_scenario(scenario).lattice
    out = Tracker(lat).track_bunch_losses(Bunch(states.T.copy()), n_turns=1, nonlinear=True)
    assert list(out.loss_s) == pytest.approx([1.0, 1.0 + L, 1.0 + L, np.nan, np.nan], nan_ok=True)
    assert np.array_equal(np.array(res["lostTurn"]), out.loss_turn)
    assert np.array_equal(np.array(res["lostAt"]), out.loss_element)
    assert _rel_dev(np.array(res["states"], dtype=float), out.states.T) < GATE


def test_the_tracker_refuses_what_the_editor_refuses() -> None:
    """A displaced bend, a design tilt: the editor's optics refuse both, so must its tracker
    — loudly, as an error string, never a silently wrong answer."""
    bad = [
        {"type": "Dipole", "length": 1.0, "angle": 0.1, "roll": 0.01},
        {"type": "Dipole", "length": 1.0, "angle": 0.1, "tilt": 0.5},
    ]
    cases = [
        {
            "scenario": {"reference": _REF, "elements": [el]},
            "states": [[0.0] * 6],
            "turns": 1,
            "mode": "turns",
        }
        for el in bad
    ]
    for res in _node(cases):
        assert "error" in res


def test_the_bunch_states_round_trip_the_lattice_unchanged() -> None:
    """Housekeeping the gates above lean on: the Python side of every comparison is the
    package's own tracker on the lattice the scenario loader builds, and that lattice is
    the editor's — same element count, same order (a dropped element would shift every
    per-element comparison by one and could still pass on a drift)."""
    lat: Lattice = load_scenario(ZOO).lattice
    assert len(lat.elements) == len(ZOO["elements"])
    assert [type(e).__name__ for e in lat.elements] == [e["type"] for e in ZOO["elements"]]
