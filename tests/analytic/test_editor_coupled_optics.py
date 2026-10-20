"""The editor's main view on a design-coupled ring (``editor/accsim-optics.js``, ``modes``).

Change 3 of *coupled optics in the editor's main view* (``docs/ROADMAP.md``, gates
pre-committed there). A ring with a skew quadrupole, a solenoid or a rolled magnet used to stop
at ``matchPeriodic``. ``analyse`` now adds ``modes``: the 4D normal form carried element by
element (a port of :func:`~accsim.twiss.propagate_normal_form`), the Mais-Ripken betas and
alphas at every boundary, the full mode tunes, the 4D dispersion, and the compaction, ``eta``
and ``Q_s`` of change 1. The per-plane fields stay empty and ``opticsError`` stays ``coupled``,
so nothing that reads them changes meaning before change 4 re-points it.

Node-driven (``editor/selftest.js``); skipped without Node.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import accsim as ac
from accsim.scenario import load_scenario
from accsim.twiss import _matched_dispersion, _transverse_4d

EDITOR = Path(__file__).resolve().parents[2] / "editor"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on the PATH")


def _presets() -> dict[str, dict]:
    text = (EDITOR / "presets.js").read_text(encoding="utf-8")
    marker = "\nvar ACCSIM_PRESETS = "
    start = text.index(marker) + len(marker)
    return {p["id"]: p for p in json.loads(text[start : text.rindex("];") + 1])}


PRESETS = _presets()


def _insert(pid: str, at: int, el: dict) -> dict:
    q = copy.deepcopy(PRESETS[pid])
    q["elements"] = q["elements"][:at] + [el] + q["elements"][at:]
    return q


def _skew(pid: str, k1sl: float = 0.01) -> dict:
    return _insert(pid, 5, {"type": "ThinSkewQuadrupole", "name": "SQ", "k1sl": k1sl})


def _rolled_qd() -> dict:
    q = copy.deepcopy(PRESETS["fodo-cell"])
    q["elements"][4]["roll"] = 0.05
    return q


FIXTURES = {
    "electron-ring+skew": _skew("electron-ring"),
    "kicked-ring+skew": _skew("kicked-ring"),
    "wiggler-ring+skew": _skew("wiggler-ring"),
    "dba-cell+solenoid": _insert(
        "dba-cell", 1, {"type": "Solenoid", "name": "SOL", "length": 0.5, "ks": 0.2}
    ),
    "fodo-cell+rolled-QD": _rolled_qd(),
}


def _js(sc: dict) -> dict:
    proc = subprocess.run(
        [NODE, str(EDITOR / "selftest.js"), "-"],
        input=json.dumps(sc),
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(proc.stdout)
    assert out["errors"] == []
    return out


@pytest.fixture(scope="module")
def js() -> dict[str, dict]:
    return {name: _js(sc) for name, sc in FIXTURES.items()}


def _package(sc: dict):
    lat = load_scenario(sc).lattice
    form = ac.normal_form(lat.one_turn_matrix(), method="4d")
    return lat, ac.propagate_normal_form(lat, form)


def _package_dispersion(lat: ac.Lattice) -> np.ndarray:
    """The matched 4D dispersion carried through the package's element matrices."""
    M = lat.one_turn_matrix()
    d = _matched_dispersion(M)
    out = [d]
    for e in lat.elements:
        m = e.matrix(lat.ref)
        d = _transverse_4d(m) @ d + m[:4, 5]
        out.append(d)
    return np.array(out)


# ---------------------------------------------------------------------- gate 1
@pytest.mark.parametrize("name", list(FIXTURES))
def test_the_mode_optics_along_the_ring_are_the_packages(name: str, js: dict) -> None:
    """Gate 1: W(s), mu, the full tunes, B and A at every boundary."""
    lat, pts = _package(FIXTURES[name])
    got = js[name]["modes"]["boundaries"]
    assert len(got) == len(pts)
    wmax = max(np.abs(p.w).max() for p in pts)
    for g, p in zip(got, pts, strict=True):
        assert g["s"] == pytest.approx(p.s, rel=1e-14, abs=1e-14)
        assert np.abs(np.array(g["W"]) - p.w).max() < 1e-11 * wmax
        np.testing.assert_allclose(g["mu"], p.mu, rtol=0, atol=1e-9)
        bmax = np.abs(p.betas).max()
        np.testing.assert_allclose(g["B"], p.betas, rtol=1e-9, atol=1e-12 * bmax)
        np.testing.assert_allclose(g["A"], p.alphas, rtol=1e-9, atol=1e-12 * bmax)
    full = np.array(pts[-1].mu) / (2.0 * np.pi)
    np.testing.assert_allclose(js[name]["modes"]["tunes"], full, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------- gate 2
@pytest.mark.parametrize("name", list(FIXTURES))
def test_the_dispersion_along_the_ring_is_the_packages(name: str, js: dict) -> None:
    """Gate 2: the 4D dispersion, with the D_y that coupling makes."""
    lat = load_scenario(FIXTURES[name]).lattice
    want = _package_dispersion(lat)
    got = np.array([g["disp"] for g in js[name]["modes"]["boundaries"]])
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12 * np.abs(want).max())


# ---------------------------------------------------------------------- gate 3
@pytest.mark.parametrize("name", list(FIXTURES))
def test_compaction_eta_and_qs_are_the_packages(name: str, js: dict) -> None:
    """Gate 3: change 1's numbers, through the editor's own identity."""
    lat = load_scenario(FIXTURES[name]).lattice
    m = js[name]["modes"]
    a = ac.momentum_compaction(lat)
    assert m["alpha_c"] == pytest.approx(a, rel=1e-9)
    assert m["eta"] == pytest.approx(ac.slip_factor(lat), rel=1e-9)
    if a > 0:
        assert m["gamma_t"] == pytest.approx(ac.transition_gamma(lat), rel=1e-9)
    if any(isinstance(e, ac.RFCavity) for e in lat.elements):
        assert m["Qs"] == pytest.approx(ac.synchrotron_tune(lat), rel=1e-9)
    else:
        assert m["Qs"] is None


def test_the_vertical_pair_is_load_bearing_in_the_editor(js: dict) -> None:
    """Gate 3: on the rolled-QD cell R53 D_y + R54 D_py is far above the port's floor."""
    lat = load_scenario(FIXTURES["fodo-cell+rolled-QD"]).lattice
    M, D = lat.one_turn_matrix(), _matched_dispersion(lat.one_turn_matrix())
    share = abs(M[4, 2] * D[2] + M[4, 3] * D[3]) / lat.length
    a = ac.momentum_compaction(lat)
    assert share > 1e3 * 1e-9 * abs(a)
    assert js["fodo-cell+rolled-QD"]["modes"]["alpha_c"] == pytest.approx(a, rel=1e-9)


# ---------------------------------------------------------------------- gate 4
def _without_cavity(sc: dict) -> dict:
    q = copy.deepcopy(sc)
    q["elements"] = [e for e in q["elements"] if e["type"] != "RFCavity"]
    return q


@pytest.mark.parametrize("name", list(FIXTURES))
def test_the_beam_size_matches_the_edwards_teng_route(name: str) -> None:
    """Gate 4: sigma_x, sigma_y from W against coupled_beam_sigma from gamma_c, C, beta_i.

    On the cavity-free copy of each ring: propagate_coupled_twiss re-matches the transverse
    block of the local 6x6 one-turn map, which on a ring with a cavity carries an RF-times-
    dispersion term the 4D optics does not (see the next test), so there it is a different
    model rather than a second route to the same one.
    """
    sc = _without_cavity(FIXTURES[name])
    lat = load_scenario(sc).lattice
    beam = sc["beam"]
    sx, sy, _tilt = ac.coupled_beam_sigma(
        ac.propagate_coupled_twiss(lat), beam["emit_x"], beam["emit_y"], beam["sigma_delta"]
    )
    sig = _js(sc)["modes"]["sigma"]
    np.testing.assert_allclose(sig["x"], sx, rtol=1e-9)
    np.testing.assert_allclose(sig["y"], sy, rtol=1e-9)


def test_the_edwards_teng_route_sees_the_cavity_where_the_4d_optics_does_not() -> None:
    """Why gate 4 drops the cavity: a package finding, pinned so it is not rediscovered.

    On the UNCOUPLED electron-ring, propagate_coupled_twiss's beta_1 misses propagate_twiss's
    beta_x by 1.8% with the cavity and by round-off without it. At the entrance the local map
    is the one-turn map itself, so the two agree there; downstream the local 6x6 map's
    transverse block picks up the cavity's slope through the dispersion.
    """
    for sc, lo, hi in (
        (PRESETS["electron-ring"], 1e-2, 3e-2),
        (_without_cavity(PRESETS["electron-ring"]), 0.0, 1e-13),
    ):
        lat = load_scenario(sc).lattice
        tw = ac.propagate_twiss(lat, ac.closed_twiss(lat))
        et = ac.propagate_coupled_twiss(lat)
        miss = max(abs(t.beta_1 / w.beta_x - 1.0) for t, w in zip(et, tw, strict=True))
        assert lo <= miss < hi
        assert et[0].beta_1 == pytest.approx(tw[0].beta_x, rel=1e-12)


@pytest.mark.parametrize("pid", ["fodo-cell", "electron-ring", "dba-cell"])
def test_mode_sigma_of_an_uncoupled_ring_is_todays_envelope(pid: str) -> None:
    """Gate 4: the same helper on the per-plane Twiss is sqrt(eps beta + (D sigma_delta)^2)."""
    proc = subprocess.run(
        [NODE, str(EDITOR / "selftest.js"), "-", "--uncoupled-sigma"],
        input=json.dumps(PRESETS[pid]),
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(proc.stdout)
    b = PRESETS[pid]["beam"]
    tw = out["twissBoundaries"]
    want_x = [
        np.sqrt(b["emit_x"] * p["beta_x"] + (b["sigma_delta"] * p["disp_x"]) ** 2) for p in tw
    ]
    want_y = [
        np.sqrt(b["emit_y"] * p["beta_y"] + (b["sigma_delta"] * p["disp_y"]) ** 2) for p in tw
    ]
    np.testing.assert_allclose(out["uncoupledSigma"]["x"], want_x, rtol=1e-14)
    np.testing.assert_allclose(out["uncoupledSigma"]["y"], want_y, rtol=1e-14)


# ---------------------------------------------------------------------- gate 5
def test_uncoupled_presets_carry_no_modes() -> None:
    """Gate 5: the new branch is reached only by a coupled ring."""
    proc = subprocess.run(
        [NODE, str(EDITOR / "selftest.js")], capture_output=True, text=True, check=True
    )
    for r in json.loads(proc.stdout):
        assert r["modes"] is None, r["id"]
        assert r["modesError"] is None, r["id"]


@pytest.mark.parametrize("name", list(FIXTURES))
def test_the_per_plane_fields_stay_empty_on_a_coupled_ring(name: str, js: dict) -> None:
    """Gate 5: every consumer of the per-plane record still sees a refused ring."""
    r = js[name]
    assert r["coupled"] is True
    assert r["opticsError"]["code"] == "coupled"
    for field in ("twiss0", "twissBoundaries", "tunes", "chromaticity", "alpha_c", "Qs"):
        assert r[field] is None, field


# ---------------------------------------------------------------------- gate 6
def test_the_mode_optics_meet_the_uncoupled_ring_at_second_order() -> None:
    """Gate 6: misses of the betas and tunes from the uncoupled ring fall as k1sl^2."""
    base = _js(PRESETS["electron-ring"])
    b0 = np.array([[p["beta_x"], p["beta_y"]] for p in base["twissBoundaries"]])
    q0 = np.array(base["tunes"])
    beta_miss, tune_miss = [], []
    for k in (1e-4, 5e-5, 2.5e-5):
        m = _js(_skew("electron-ring", k))["modes"]
        pts = [p for i, p in enumerate(m["boundaries"]) if i != 6]  # drop the inserted element
        b = np.array([[p["B"][0][0], p["B"][1][1]] for p in pts])
        beta_miss.append(np.abs(b / b0 - 1.0).max())
        tune_miss.append(np.abs(np.array(m["tunes"]) - q0).max())
    assert beta_miss[0] == pytest.approx(2.03e-7, rel=1e-2)
    assert tune_miss[0] == pytest.approx(1.23e-8, rel=1e-2)
    for miss in (beta_miss, tune_miss):
        assert 3.9 < miss[0] / miss[1] < 4.1
        assert 3.9 < miss[1] / miss[2] < 4.1


def _equal_tune_ring(k1sl: float) -> dict:
    """A bend-free thin FODO ring (Q_x = Q_y exactly) with one weak skew quadrupole."""
    cell = [
        {"type": "ThinQuadrupole", "k1l": 0.2},
        {"type": "Drift", "length": 2.0},
        {"type": "ThinQuadrupole", "k1l": -0.2},
        {"type": "Drift", "length": 2.0},
    ]
    return {
        "format": ac.SCENARIO_FORMAT,
        "name": "equal tunes",
        "reference": {"species": "proton", "energy_eV": 10e9},
        "beam": {"emit_x": 1e-6, "emit_y": 1e-6, "sigma_delta": 1e-3},
        "elements": [{"type": "ThinSkewQuadrupole", "k1sl": k1sl}, *cell * 4],
    }


def test_a_cavity_between_the_bends_is_refused_where_the_package_refuses() -> None:
    """Found by the final review: with electron-ring's cavity moved to mid-ring, the one-turn
    map's transverse block misses symplecticity by 4.9e-3 and the package refuses the ring
    (momentum_compaction: UnstableLatticeError). Unchecked, the editor drew modes anyway (full
    tunes 5.7809, 5.5746, with nothing to hold them against); it now refuses with its own code,
    not as an instability."""
    sc = copy.deepcopy(FIXTURES["electron-ring+skew"])
    cav = next(e for e in sc["elements"] if e["type"] == "RFCavity")
    els = [e for e in sc["elements"] if e["type"] != "RFCavity"]
    sc["elements"] = [*els[:100], cav, *els[100:]]
    with pytest.raises(ac.UnstableLatticeError):
        ac.momentum_compaction(load_scenario(sc).lattice)
    r = _js(sc)
    assert r["modes"] is None
    assert r["modesError"]["code"] == "rf-4d"


def _uncoupled_with_cavity_at(pos: int | None) -> dict:
    sc = copy.deepcopy(PRESETS["electron-ring"])
    cav = next(e for e in sc["elements"] if e["type"] == "RFCavity")
    els = [e for e in sc["elements"] if e["type"] != "RFCavity"]
    sc["elements"] = [*els, cav] if pos is None else [*els[:pos], cav, *els[pos:]]
    return sc


def test_an_uncoupled_ring_with_a_cavity_between_the_bends_is_flagged_approximate() -> None:
    """Found after the entry closed (a side agent's note): the per-plane optics of an UNCOUPLED
    ring whose cavity sits between bends do not close on themselves (beta after one turn misses
    its start by 0.59% with electron-ring's cavity at element 100, 2.8% at element 50), and
    neither the editor nor the package's closed_twiss said so. The user chose a warning: the
    numbers stay the package's (test_scenario holds them to 1e-9) and analyse adds rfWarning
    with the measured miss; the cavity-last preset carries none."""
    assert _js(_uncoupled_with_cavity_at(None))["rfWarning"] is None
    for pos, miss in ((100, 5.93e-3), (50, 2.82e-2)):
        sc = _uncoupled_with_cavity_at(pos)
        r = _js(sc)
        assert r["opticsError"] is None and r["twiss0"] is not None
        assert r["rfWarning"]["betaMiss"] == pytest.approx(miss, rel=1e-2)
        lat = load_scenario(sc).lattice
        pts = ac.propagate_twiss(lat, ac.closed_twiss(lat))
        assert r["rfWarning"]["betaMiss"] == pytest.approx(
            max(abs(pts[-1].beta_x / pts[0].beta_x - 1), abs(pts[-1].beta_y / pts[0].beta_y - 1)),
            rel=1e-9,
        )


def test_equal_tunes_just_above_the_threshold_are_refused_as_degenerate() -> None:
    """Gate 6: coupled by the 1e-9 test, but the modes cannot be separated."""
    q = ac.tunes(load_scenario(_equal_tune_ring(0.0)).lattice)
    assert q[0] == pytest.approx(q[1], abs=1e-12)
    r = _js(_equal_tune_ring(1e-8))
    assert r["coupled"] is True
    assert r["modes"] is None
    assert r["modesError"]["code"] == "degenerate"
