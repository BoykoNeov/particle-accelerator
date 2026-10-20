"""The editor's per-mode chromaticity and what reads it (``editor/inject.js``, ``modeChromaticity``).

Change 4 of *coupled optics in the editor's main view* (``docs/ROADMAP.md``, gates pre-committed
there). On a coupled ring the editor now has a chromaticity for each mode: the slope of the
tracked turn's 4D mode tunes in momentum, on the closed orbit at ``+-1e-4``. It is a different
route from :func:`~accsim.twiss.mode_chromaticity` (the whole turn rather than element by
element) to the same number. The Injection tab's decoherence prediction reads it, so a coupled
ring's beam is now predicted to smear at its MODE chromaticity, which on change 1's skew ring is
23% away from the per-plane one.

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
from test_coupled_compaction import skew_ring
from test_editor_coupled_optics import FIXTURES, PRESETS, _equal_tune_ring

import accsim as ac
from accsim.scenario import load_scenario, scenario_from_lattice

EDITOR = Path(__file__).resolve().parents[2] / "editor"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on the PATH")


def _scenario(lat: ac.Lattice, name: str) -> dict:
    sc = scenario_from_lattice(lat, name=name, periodic=True)
    return sc if isinstance(sc, dict) else sc.to_dict()


SKEW = _scenario(skew_ring(), "change-1 skew ring")
PORT = {**FIXTURES, "change-1 skew ring": SKEW}


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
@pytest.mark.parametrize("name", list(PORT))
def test_the_editor_mode_chromaticity_is_the_packages(name: str) -> None:
    """Gate 1: whole-turn route against the package's element-by-element one."""
    (r,) = _node([{"scenario": PORT[name], "mode": "chroma"}])
    assert r.get("error") is None, r.get("error")
    want = ac.mode_chromaticity(load_scenario(PORT[name]).lattice)
    np.testing.assert_allclose(r["chromaticity"], want, rtol=3e-8)


# ---------------------------------------------------------------------- gate 2
@pytest.mark.parametrize("pid", [k for k, p in PRESETS.items() if p.get("periodic", True)], ids=str)
def test_an_uncoupled_preset_keeps_its_integral_chromaticity(pid: str) -> None:
    """Gate 2: modeChromaticity IS a.chromaticity where the planes are the modes."""
    (r,) = _node([{"scenario": PRESETS[pid], "mode": "chroma"}])
    assert r["coupled"] is False
    assert r["chromaticity"] == r["analysisChromaticity"]


def test_equal_tunes_on_an_uncoupled_ring_take_no_normal_form() -> None:
    """Gate 2: Q_x = Q_y exactly, uncoupled: the integral, not a refused normal form."""
    (r,) = _node([{"scenario": _equal_tune_ring(0.0), "mode": "chroma"}])
    assert r.get("error") is None, r.get("error")
    assert r["chromaticity"] == r["analysisChromaticity"]


# ---------------------------------------------------------------------- gate 3
def _gradient_bend_ring() -> dict:
    sc = copy.deepcopy(SKEW)
    i = next(n for n, e in enumerate(sc["elements"]) if e["type"] == "Dipole")
    sc["elements"][i]["k1"] = 0.01
    return sc


def test_a_coupled_ring_with_a_bending_gradient_magnet_has_no_chromaticity() -> None:
    """Gate 3: refused as the package refuses, the reason naming the metric group."""
    with pytest.raises(NotImplementedError):
        ac.mode_chromaticity(load_scenario(_gradient_bend_ring()).lattice)
    (r,) = _node([{"scenario": _gradient_bend_ring(), "mode": "chroma"}])
    assert r["chromaticity"] is None
    assert "metric" in r["error"]
    (inj,) = _node([{"scenario": _gradient_bend_ring(), "mode": "inject", "inj": {"n": 16}}])
    assert inj["optics"]["chromaticity"] is None


# ---------------------------------------------------------------------- gate 4
def _decoherence(sigma_delta: float) -> tuple[list[float], list[float], list[float]]:
    """Max |centroid_n / centroid_0 - envelope_n| per mode, with the MODE chromaticity the tab
    now carries and with the uncoupled ring's per-plane one (the control), and the envelope's
    final depth."""
    turns = int(round(2.6 / sigma_delta))
    inj = {
        "n": 400,
        "seed": 5,
        "emit_x": 0.0,
        "emit_y": 0.0,
        "sigma_delta": sigma_delta,
        "dx": 1e-3,
        "dy": 3e-4,
    }
    (r,) = _node([{"scenario": SKEW, "mode": "stats", "inj": inj, "turns": turns}])
    assert r.get("error") is None, r.get("error")
    deltas = np.array(r["deltas"])
    n = np.arange(turns + 1)
    plane = ac.chromaticity(skew_ring(0.0))
    mode_miss, plane_miss, depth = [], [], []
    for i, pl in enumerate(("x", "y")):
        c = np.array([s[pl]["centroid"] for s in r["stats"]])
        for xi, out in ((r["optics"]["chromaticity"][i], mode_miss), (plane[i], plane_miss)):
            env = np.abs(np.exp(2j * np.pi * xi * np.outer(n, deltas)).mean(axis=1))
            out.append(float(np.max(np.abs(c / c[0] - env))))
            if out is mode_miss:
                depth.append(float(env[-1]))
    return mode_miss, plane_miss, depth


def test_a_coupled_beam_decoheres_at_the_mode_chromaticity() -> None:
    """Gate 4: first-order misses with Q'_k, an order-one miss with the per-plane xi."""
    m1, p1, d1 = _decoherence(7.5e-4)
    m2, p2, d2 = _decoherence(3.75e-4)
    assert max(d1 + d2) < 0.05  # the runs really decohere (else the gate is idle)
    assert max(m1) < 3e-3
    assert max(m2) < 1.5e-3
    for a, b in zip(m1, m2, strict=True):
        assert 1.6 < a / b < 2.4
    assert p1[0] > 0.1
    assert p2[0] > 0.1


# ---------------------------------------------------------------------- gate 5
def test_a_coupled_cavity_ring_keeps_its_6d_synchrotron_tune() -> None:
    """Gate 5: Q_s is still the 6D form's third tune (not change 3's lumped tile value)."""
    (r,) = _node([{"scenario": FIXTURES["electron-ring+skew"], "mode": "inject", "inj": {"n": 16}}])
    o = r["optics"]
    q3 = o["normal"]["tunes"][2]
    assert o["Qs"] == min(q3, 1.0 - q3)
    assert o["eta"] is None
    lumped = ac.synchrotron_tune(load_scenario(FIXTURES["electron-ring+skew"]).lattice)
    assert abs(o["Qs"] / lumped - 1.0) > 1e-6
    assert o["chromaticity"] is not None
