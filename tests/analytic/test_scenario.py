"""Scenario files and the editor's optics core, held against the package.

Two halves:

1. **Round trip.** Every bundled preset loads into a ``Lattice`` through
   :mod:`accsim.scenario`, dumps back to a dict, and reloads to the same maps. The
   format is the seam between the editor and the package, so this is the seam test.

2. **Cross-check.** ``editor/accsim-optics.js`` is a port of the package's linear
   optics. Run under Node, it emits every element matrix, the one-turn map, the
   matched Twiss, the tunes, chromaticity, momentum compaction, synchrotron tune,
   survey and closed orbit for each preset; each is compared with the package's own
   number to ``1e-9``. **Disagreement is a bug on one side, never a tolerance to
   loosen** — the JavaScript is meant to be the same arithmetic. Skipped, not failed,
   when Node is not on the PATH (CI runs without it).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import accsim as ac
from accsim.scenario import (
    SCENARIO_FORMAT,
    ScenarioError,
    dump_scenario,
    element_from_dict,
    element_to_dict,
    load_scenario,
    scenario_from_lattice,
)

ROOT = Path(__file__).resolve().parents[2]
EDITOR = ROOT / "editor"
NODE = shutil.which("node")


def _presets() -> list[dict]:
    text = (EDITOR / "presets.js").read_text(encoding="utf-8")
    marker = "\nvar ACCSIM_PRESETS = "  # the header comment names the variable too
    start = text.index(marker) + len(marker)
    return json.loads(text[start : text.rindex("];") + 1])


PRESETS = _presets()
PRESET_IDS = [p["id"] for p in PRESETS]


def _maps(lat: ac.Lattice) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return [e.matrix(lat.ref) for e in lat.elements], [e.kick(lat.ref) for e in lat.elements]


# --- round trip -------------------------------------------------------------------
@pytest.mark.parametrize("preset", PRESETS, ids=PRESET_IDS)
def test_preset_round_trip(preset: dict) -> None:
    sc = load_scenario(preset)
    assert sc.name == preset["name"]
    assert sc.periodic == preset["periodic"]
    assert len(sc.lattice) == len(preset["elements"])
    again = load_scenario(json.loads(dump_scenario(sc)))
    m1, k1 = _maps(sc.lattice)
    m2, k2 = _maps(again.lattice)
    for a, b in zip(m1, m2, strict=True):
        assert np.array_equal(a, b)
    for a, b in zip(k1, k2, strict=True):
        assert np.array_equal(a, b)
    assert again.lattice.ref == sc.lattice.ref


def test_electron_ring_preset_is_the_example_machine() -> None:
    """The bundled ring is examples/build_a_machine.py's store ring, number for number."""
    import sys

    sys.path.insert(0, str(ROOT / "examples"))
    import build_a_machine as bam

    sc = load_scenario(next(p for p in PRESETS if p["id"] == "electron-ring"))
    ours, theirs = sc.lattice, bam.store_ring()
    # Same optics: the cavity phase differs (the example replenishes U0), the arc is the same.
    assert ac.tunes(ours) == pytest.approx(ac.tunes(theirs), abs=1e-12)
    assert ours.length == theirs.length
    cav = next(e for e in ours.elements if isinstance(e, ac.RFCavity))
    assert cav.harmonic_number(ours.ref, ours.length) == pytest.approx(400.0, abs=1e-9)


def test_harmonic_resolves_against_the_whole_circumference() -> None:
    """``harmonic`` -> frequency uses the full element list's length, cavity included."""
    ref = ac.ReferenceParticle.from_total_energy(ac.PROTON_MASS_EV, 10e9)
    data = {
        "format": SCENARIO_FORMAT,
        "reference": {"species": "proton", "energy_eV": 10e9},
        "elements": [
            {"type": "Drift", "length": 30.0},
            {"type": "RFCavity", "voltage": 1e6, "harmonic": 7},
            {"type": "Drift", "length": 20.0},
        ],
    }
    lat = load_scenario(data).lattice
    want = ac.RFCavity.from_harmonic(1e6, 7, 50.0, ref)
    assert lat[1].frequency == pytest.approx(want.frequency, rel=1e-15)


def test_every_registered_type_round_trips() -> None:
    ref = ac.ReferenceParticle.from_total_energy(ac.ELECTRON_MASS_EV, 1e9, charge=-1.0)
    records = [
        {"type": "Drift", "length": 1.0, "dx": 1e-4, "dy": -2e-4, "roll": 0.01},
        {"type": "Quadrupole", "length": 0.5, "k1": 1.2, "roll": 0.02},
        {"type": "ThinQuadrupole", "k1l": 0.3, "dx": 1e-3},
        {
            "type": "Dipole",
            "length": 1.0,
            "angle": 0.1,
            "k1": 0.05,
            "e1": 0.02,
            "e2": 0.03,
            "fringe": True,
        },
        {"type": "Sextupole", "length": 0.2, "k2": 5.0},
        {"type": "ThinSextupole", "k2l": 1.0},
        {"type": "Octupole", "length": 0.2, "k3": 50.0},
        {"type": "ThinOctupole", "k3l": 10.0},
        {"type": "SkewQuadrupole", "length": 0.3, "k1s": 0.4},
        {"type": "ThinSkewQuadrupole", "k1sl": 0.1},
        {"type": "Solenoid", "length": 2.0, "ks": 0.3},
        {"type": "RFCavity", "voltage": 2e6, "frequency": 5e8, "phi_s": 0.1},
        {"type": "Corrector", "kick_x": 1e-4, "kick_y": -2e-4},
        {"type": "Wiggler", "period": 0.1, "h0": 0.2, "periods": 5},
        {"type": "Aperture", "shape": "elliptical", "half_x": 0.02, "half_y": 0.01},
        {
            "type": "Collimator",
            "shape": "rectangular",
            "half_x": 0.01,
            "half_y": 0.02,
            "length": 0.5,
        },
    ]
    for rec in records:
        elem = element_from_dict(rec)
        back = element_to_dict(elem)
        assert back["type"] == rec["type"]
        elem2 = element_from_dict(back)
        assert np.array_equal(elem.matrix(ref), elem2.matrix(ref)), rec["type"]
        assert np.array_equal(elem.kick(ref), elem2.kick(ref)), rec["type"]
        for key, val in rec.items():
            if key in ("type",):
                continue
            assert back.get(key) == val, (rec["type"], key)


def test_scenario_from_lattice_names_the_species() -> None:
    ref = ac.ReferenceParticle.from_total_energy(ac.ELECTRON_MASS_EV, 2e9, charge=-1.0)
    d = scenario_from_lattice(ac.Lattice([ac.Drift(1.0)], ref), name="x")
    assert d["reference"] == {"species": "electron", "energy_mode": "total", "energy_eV": 2e9}
    odd = ac.ReferenceParticle.from_total_energy(1.0e9, 2e9, charge=2.0)
    d = scenario_from_lattice(ac.Lattice([ac.Drift(1.0)], odd))
    assert d["reference"]["species"] == "custom"
    assert load_scenario(d).lattice.ref == odd


def test_bad_records_raise_scenario_error() -> None:
    with pytest.raises(ScenarioError):
        load_scenario({"format": "other/9", "reference": {}, "elements": []})
    with pytest.raises(ScenarioError):
        element_from_dict({"type": "Warp", "length": 1.0})
    with pytest.raises(ScenarioError):
        element_from_dict({"type": "Drift", "length": "long"})
    with pytest.raises(ScenarioError):  # accsim refuses a displaced bend; so must the loader
        element_from_dict({"type": "Dipole", "length": 1.0, "angle": 0.1, "dx": 1e-3})
    with pytest.raises(ScenarioError):
        element_from_dict({"type": "RFCavity", "voltage": 1.0, "harmonic": 3})


# --- the JavaScript core against the package ---------------------------------------
@pytest.fixture(scope="module")
def js_results() -> dict[str, dict]:
    if NODE is None:
        pytest.skip("node is not on the PATH; the editor cross-check needs it")
    proc = subprocess.run(
        [NODE, str(EDITOR / "selftest.js")], capture_output=True, text=True, check=True
    )
    return {r["id"]: r for r in json.loads(proc.stdout)}


@pytest.mark.parametrize("preset", PRESETS, ids=PRESET_IDS)
def test_editor_core_matches_accsim(preset: dict, js_results: dict[str, dict]) -> None:
    js = js_results[preset["id"]]
    assert js["errors"] == []
    sc = load_scenario(preset)
    lat = sc.lattice
    ref = lat.ref
    tol = {"rtol": 1e-9, "atol": 1e-12}

    # Reference particle and geometry.
    assert js["ref"]["gamma0"] == pytest.approx(ref.gamma0, rel=1e-14)
    assert js["ref"]["beta0"] == pytest.approx(ref.beta0, rel=1e-14)
    assert js["length"] == pytest.approx(lat.length, rel=1e-14)

    # Every element's matrix and kick.
    mats, kicks = _maps(lat)
    for i, (m, k) in enumerate(zip(mats, kicks, strict=True)):
        np.testing.assert_allclose(np.array(js["matrices"][i]), m, err_msg=f"matrix {i}", **tol)
        np.testing.assert_allclose(np.array(js["kicks"][i]), k, err_msg=f"kick {i}", **tol)
    M, kk = lat.transfer_map()
    np.testing.assert_allclose(np.array(js["oneTurn"]), M, **tol)
    np.testing.assert_allclose(np.array(js["oneTurnKick"]), kk, **tol)

    # Survey.
    sv = ac.survey(lat)
    np.testing.assert_allclose(np.array(js["survey"]["X"]), sv.X, **tol)
    np.testing.assert_allclose(np.array(js["survey"]["Z"]), sv.Z, **tol)
    np.testing.assert_allclose(np.array(js["survey"]["theta"]), sv.theta, **tol)

    if not sc.periodic:
        t = sc.initial_twiss
        tw0 = ac.Twiss(
            0.0,
            t["beta_x"],
            t["alpha_x"],
            0.0,
            t["beta_y"],
            t["alpha_y"],
            0.0,
            t["disp_x"],
            t["disp_px"],
        )
        pts = ac.propagate_twiss(lat, tw0)
        _compare_twiss(js["twissBoundaries"], pts)
        traj = ac.propagate_orbit(lat, np.zeros(4))
        np.testing.assert_allclose(np.array(js["orbit"]), np.array(traj), **tol)
        return

    # Periodic optics.
    tw0 = ac.closed_twiss(lat)
    pts = ac.propagate_twiss(lat, tw0)
    _compare_twiss(js["twissBoundaries"], pts)
    np.testing.assert_allclose(js["tunes"], ac.tunes(lat), **tol)
    np.testing.assert_allclose(js["chromaticity"], ac.chromaticity(lat), **tol)
    np.testing.assert_allclose(js["naturalChromaticity"], ac.natural_chromaticity(lat), **tol)
    assert js["alpha_c"] == pytest.approx(ac.momentum_compaction(lat), rel=1e-9)
    if any(isinstance(e, ac.RFCavity) for e in lat.elements):
        assert js["Qs"] == pytest.approx(ac.synchrotron_tune(lat), rel=1e-9)
    else:
        assert js["Qs"] is None
    co = ac.closed_orbit(lat)
    if js["closedOrbit0"] is None:
        assert not co.any()
    else:
        np.testing.assert_allclose(np.array(js["closedOrbit0"]), co, **tol)
        np.testing.assert_allclose(np.array(js["orbit"]), np.array(ac.propagate_orbit(lat)), **tol)


def _compare_twiss(js_pts: list[dict], pts: list[ac.Twiss]) -> None:
    assert len(js_pts) == len(pts)
    for name in (
        "s",
        "beta_x",
        "alpha_x",
        "mu_x",
        "beta_y",
        "alpha_y",
        "mu_y",
        "disp_x",
        "disp_px",
        "disp_y",
        "disp_py",
    ):
        got = np.array([p[name] for p in js_pts])
        want = np.array([getattr(p, name) for p in pts])
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, err_msg=name)


@pytest.mark.skipif(NODE is None, reason="node is not on the PATH")
def test_editor_core_refuses_what_accsim_refuses() -> None:
    """A coupled ring: accsim raises CoupledLatticeError, the core reports code 'coupled'."""
    data = {
        "format": SCENARIO_FORMAT,
        "name": "skew",
        "reference": {"species": "proton", "energy_eV": 10e9},
        "elements": [
            {"type": "ThinQuadrupole", "k1l": 0.2},
            {"type": "Drift", "length": 2.0},
            {"type": "ThinSkewQuadrupole", "k1sl": 0.05},
            {"type": "Drift", "length": 2.0},
            {"type": "ThinQuadrupole", "k1l": -0.2},
            {"type": "Drift", "length": 2.0},
        ],
    }
    with pytest.raises(ac.CoupledLatticeError):
        ac.closed_twiss(load_scenario(data).lattice)
    proc = subprocess.run(
        [NODE, str(EDITOR / "selftest.js"), "-"],
        input=json.dumps(data),
        capture_output=True,
        text=True,
        check=True,
    )
    js = json.loads(proc.stdout)
    assert js["coupled"] is True
    assert js["opticsError"]["code"] == "coupled"
    assert js["twiss0"] is None
