"""The injected beam on a ring without RF, taken on its tracked closed orbit (``editor/inject.js``).

A ring with no cavity has no 6D normal form, and the Injection tab used to draw and read its
beam with the **design** Twiss and static dispersion about the **affine** 4D orbit. A beam on a
displaced orbit sees a different linear map, the Jacobian of the tracked turn at that orbit,
and the change is first order in the orbit: on ``kicked-ring`` the static dispersion at the
injection point moves 1.9% (the exact drifts and bends at the orbit's angle, as on the cavity
path). A vertical orbit also couples the planes, which per-plane Twiss cannot describe.

The editor now centres the beam on the tracked 4D orbit (``closedOrbit4``) and draws and reads
it in the **4D normal modes** ``W4`` of that map (``normalForm4``, a real-arithmetic route with
no eigen-solver), with the map's own static dispersion ``D = (I - J_4)^-1 J[:4, delta]``. These
gates hold it:

  * **the port** — orbit, Jacobian, ``W4``, tunes and ``D`` against the package
    (``closed_orbit_nonlinear``, ``linearised_one_turn_map``, ``normal_form(J, "4d")``) at the
    measured cross-route floor;
  * **the dispersion by a route that takes no Jacobian** — ``d x_co / d delta``;
  * **the coupled formulas reduce to today's** — on an uncoupled map ``W4`` is the per-plane
    Courant-Snyder matrix;
  * **the matched beam is the map's fixed point**; the design beam misses at first order;
  * **one particle on ``D delta0`` makes no betatron oscillation** — fourth order, where the
    design ``D`` leaves second;
  * **nothing else moves** — kick-free rings keep the design optics, cavity rings their 4D
    design record.

Node-driven; skipped (not failed) without Node.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess

import numpy as np
import pytest
from test_injection import EDITOR, NODE, PRESETS, RING, _centred_ring, _node
from test_injection_orbit6 import _scaled_kicks

from accsim.orbit import closed_orbit_6d, closed_orbit_nonlinear, linearised_one_turn_map
from accsim.scenario import load_scenario
from accsim.twiss import normal_form

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not on the PATH")


def _kick_y(sc: dict, kick: float) -> dict:
    """The same ring with ``kick_y`` set on every corrector: a vertical orbit, which couples the
    on-orbit map (off-block ``3.2e-5`` at ``3e-5`` rad) while the design map stays uncoupled."""
    q = copy.deepcopy(sc)
    for e in q["elements"]:
        if e["type"] == "Corrector":
            e["kick_y"] = kick
    return q


def _skew(sc: dict, k1sl: float = 0.01, at: int = 5) -> dict:
    """The same ring with a thin skew quadrupole after its ``at``-th element: DESIGN coupling
    (off-block ``M`` ``0.17``, off-block ``W4`` ``3.5e-2`` of ``max |W|``, ``D_y = 1.2`` cm on the
    rolled kicked ring), far from the coupling resonance (fractional tunes 0.78 and 0.57)."""
    q = copy.deepcopy(sc)
    sq = {"type": "ThinSkewQuadrupole", "name": "SQ", "k1sl": k1sl}
    q["elements"] = q["elements"][:at] + [sq] + q["elements"][at:]
    return q


RINGS = {
    "kicked-ring": RING,
    "kicks-x4": _scaled_kicks(RING, 4.0),
    "kick-y": _kick_y(RING, 3e-5),
    "skew-quad": _skew(RING),
}
COUPLED = {"skew-quad": RINGS["skew-quad"], "centred-skew-quad": _skew(_centred_ring())}
# Horizontal kicks only: the on-orbit J is exactly uncoupled.
UNCOUPLED = ("kicked-ring", "kicks-x4")
_IDX5 = [0, 1, 2, 3, 5]


def _editor(sc: dict, inj: dict | None = None, sampler: str | None = None) -> dict:
    case = {"scenario": sc, "mode": "inject", "inj": inj or {"n": 8}}
    if sampler:
        case["sampler"] = sampler
    (r,) = _node([case])
    return r


def _package(sc: dict):
    """The package's tracked 4D orbit, the Jacobian of the turn there, its 4D normal form and
    its static dispersion."""
    lat = load_scenario(sc).lattice
    co = closed_orbit_nonlinear(lat)
    J = linearised_one_turn_map(lat, co, delta=0.0)
    D = np.linalg.solve(np.eye(4) - J[:4, :4], J[:4, 5])
    return lat, co, J, normal_form(J, method="4d"), D


def _S4() -> np.ndarray:
    S = np.zeros((4, 4))
    S[0, 1] = S[2, 3] = 1.0
    S[1, 0] = S[3, 2] = -1.0
    return S


def _design_dispersion(sc: dict) -> np.ndarray:
    M, _k = load_scenario(sc).lattice.one_turn_map()
    return np.linalg.solve(np.eye(4) - M[:4, :4], M[:4, 5])


# --- 1. the port ------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(RINGS), ids=str)
def test_the_editor_4d_on_orbit_optics_are_the_packages(name: str) -> None:
    """``closedOrbit4`` against ``closed_orbit_nonlinear`` to ``1e-12`` of ``max |x_co|``;
    ``orbitMap`` against ``linearised_one_turn_map`` to ``5e-9`` per entry; ``W4`` against
    ``normal_form(J, "4d")`` to ``1e-9`` of ``max |W|``, mode tunes to ``1e-9``; ``D`` against the
    package's static dispersion of its own ``J`` to ``1e-9`` of ``|D_x|``; ``W4`` symplectic to
    ``1e-9``. Measured at worst ``2.5e-14``, ``4.1e-10``, ``3.2e-11``, ``5.7e-12``, ``6.4e-11``,
    ``4.8e-12``. Not blind: the design ``D_x`` is 1.9% (7.2% on ``kicks-x4``) off."""
    sc = RINGS[name]
    o = _editor(sc)["optics"]
    _lat, co, J, nf, D = _package(sc)
    assert o["normal4"] is not None
    orbit = np.array(o["orbit0"])
    assert orbit[4] == 0.0 and orbit[5] == 0.0
    assert np.max(np.abs(orbit[:4] - co)) < 1e-12 * np.max(np.abs(co))
    assert np.max(np.abs(np.array(o["orbitMap"]) - J)) < 5e-9
    n4 = o["normal4"]
    W = np.array(n4["W"])
    assert np.max(np.abs(W - nf.w)) < 1e-9 * np.max(np.abs(nf.w))
    for a, b in zip(n4["tunes"], nf.tunes, strict=True):
        assert abs((a - b + 0.5) % 1.0 - 0.5) < 1e-9
    Djs = np.array(n4["dispersion"])
    assert np.max(np.abs(Djs - D)) < 1e-9 * abs(D[0])
    assert np.max(np.abs(W.T @ _S4() @ W - _S4())) < 1e-9
    t = o["twiss"]
    assert [t["disp_x"], t["disp_px"], t["disp_y"], t["disp_py"]] == list(n4["dispersion"])
    assert (t["beta_x"], t["beta_y"]) == pytest.approx(nf.mode_beta, rel=1e-9, abs=0)
    assert abs(_design_dispersion(sc)[0] / D[0] - 1.0) > 1e-2  # not blind


# --- 2. the dispersion by a route that takes no Jacobian -----------------------------------------
@pytest.mark.parametrize("name", list(RINGS), ids=str)
def test_the_4d_dispersion_is_the_closed_orbits_momentum_slope(name: str) -> None:
    """The editor's ``D`` equals ``d x_co / d delta`` from ``closed_orbit_nonlinear`` at ``+-3e-6``
    to ``1e-8`` of ``|D_x|`` (measured ``<= 3.3e-10``)."""
    sc = RINGS[name]
    D = np.array(_editor(sc)["optics"]["normal4"]["dispersion"])
    lat = load_scenario(sc).lattice
    h = 3e-6
    scan = (closed_orbit_nonlinear(lat, delta=h) - closed_orbit_nonlinear(lat, delta=-h)) / (2 * h)
    assert np.max(np.abs(D - scan)) < 1e-8 * abs(D[0])


# --- 3. the coupled formulas reduce to today's ---------------------------------------------------
def _cs_block(beta: float, alpha: float) -> np.ndarray:
    sb = np.sqrt(beta)
    return np.array([[sb, 0.0], [-alpha / sb, 1.0 / sb]])


@pytest.mark.parametrize("name", UNCOUPLED, ids=str)
def test_on_an_uncoupled_map_w4_is_the_per_plane_courant_snyder_matrix(name: str) -> None:
    """``J`` is exactly uncoupled here (horizontal kicks only), and ``W4`` equals
    ``blockdiag(N(beta_x, alpha_x), N(beta_y, alpha_y))`` from the per-plane periodic match of the
    same ``J`` to ``1e-10`` of ``max |W|``. This is what makes the mode sampler and the mode reading
    the per-plane ones on such a ring.

    As pre-committed, ``normalForm4`` produced ``W4`` here and met the gate at ``4.9e-12`` /
    ``1.5e-11`` (the finite-difference ``J``'s own non-symplecticity, which ``M^-1 = -S M^T S``
    assumes away). After pre-commitment a bend-free ring with ``Q_x = Q_y`` was found to fail
    in ``normalForm4`` (gate 7 below), so an exactly uncoupled ``J`` now takes the per-plane match
    itself, and this holds to round-off."""
    o = _editor(RINGS[name])["optics"]
    J = np.array(o["orbitMap"])
    assert np.max(np.abs(J[0:2, 2:4])) == 0.0 and np.max(np.abs(J[2:4, 0:2])) == 0.0
    Wcs = np.zeros((4, 4))
    for i in (0, 2):
        C = J[i : i + 2, i : i + 2]
        c = 0.5 * (C[0, 0] + C[1, 1])
        s = np.sign(C[0, 1]) * np.sqrt(1.0 - c * c)
        Wcs[i : i + 2, i : i + 2] = _cs_block(C[0, 1] / s, 0.5 * (C[0, 0] - C[1, 1]) / s)
    W = np.array(o["normal4"]["W"])
    assert np.max(np.abs(W - Wcs)) < 1e-10 * np.max(np.abs(W))


@pytest.mark.parametrize(
    "preset_id", [k for k, p in PRESETS.items() if p.get("periodic", True)], ids=str
)
def test_normal_form4_of_a_design_matrix_is_the_packages(preset_id: str) -> None:
    """``normalForm4`` of each ring preset's design transverse block against ``normal_form(M,
    "4d")`` of the same matrix, to ``1e-12`` of ``max |W|``: the port of the route alone, on an
    exactly symplectic matrix."""
    (d,) = _node([{"scenario": PRESETS[preset_id], "mode": "design"}])
    M4 = np.array(d["oneTurn"])[:4, :4]
    (n,) = _node([{"mode": "nf4", "matrix": M4.tolist()}])
    nf = normal_form(M4, method="4d")
    W = np.array(n["normal"]["W"])
    assert np.max(np.abs(W - nf.w)) < 1e-12 * np.max(np.abs(nf.w))


# --- 4. the matched beam is the map's fixed point ------------------------------------------------
def _covariance5(states: list, co: np.ndarray) -> np.ndarray:
    X = np.array(states)[:, _IDX5].copy()
    X[:, :4] -= co
    return X.T @ X / len(X)


def _fixed_point_miss(J: np.ndarray, Sig: np.ndarray) -> float:
    J5 = J[np.ix_(_IDX5, _IDX5)]
    norm = np.sqrt(np.outer(np.diag(Sig), np.diag(Sig)))
    return float(np.max(np.abs(J5 @ Sig @ J5.T - Sig) / norm))


@pytest.mark.parametrize("name", UNCOUPLED, ids=str)
def test_the_4d_matched_beam_is_the_on_orbit_maps_fixed_point(name: str) -> None:
    """The drawn beam (default settings, ``n = 600``, ``seed = 7``), its 5D covariance ``(x, px, y,
    py, delta)`` about ``closed_orbit_nonlinear`` (``zeta`` is drawn at 0 and only drifts without
    RF), satisfies ``J5 Sigma J5^T = Sigma`` with the package's ``J`` to ``1e-9`` of ``sqrt(Sigma_ii
    Sigma_jj)`` (measured ``2.3e-11``, ``9.7e-11``). The design beam (``sampler: "design"``) misses by
    more than ``1e-2`` (measured ``3.7e-2``, ``0.15``): first order in the orbit."""
    sc = RINGS[name]
    _lat, co, J, _nf, _D = _package(sc)
    inj = {"n": 600, "seed": 7}
    assert _fixed_point_miss(J, _covariance5(_editor(sc, inj)["initial"], co)) < 1e-9
    design = _editor(sc, inj, sampler="design")["initial"]
    assert _fixed_point_miss(J, _covariance5(design, co)) > 1e-2


# --- 5. one particle on the static dispersion ----------------------------------------------------
@pytest.mark.parametrize("name", ["kicked-ring", "skew-quad"], ids=str)
def test_a_particle_on_the_4d_dispersion_makes_no_betatron_oscillation(name: str) -> None:
    """Put at ``x_co + D delta0`` with the editor's orbit and ``D``, ``delta0 = 1e-5`` and ``2e-5``,
    and tracked 1000 turns by the editor, a particle's largest betatron action (the package's
    on-orbit modes, its own ``D`` removed) stays below ``1e-8`` of ``(D_x delta0)^2 / beta_1``
    (measured ``2.7e-11``) and rises at least 14 when ``delta0`` doubles (16.0: fourth order, the
    second-order dispersion). Placed with the design orbit and ``D`` it is above ``1e-5`` of that
    scale (``2.5e-4``) and rises by 3.5 to 4.1 (4.0: second order, the dispersion error). On the
    skew-quadrupole ring the same, coupled (measured ``1.3e-10``)."""
    sc = RINGS[name]
    o = _editor(sc)["optics"]
    _lat, co, _J, nf, D = _package(sc)
    beta1 = nf.mode_beta[0]

    def actions(orbit: list, disp: list) -> list[float]:
        starts = [[orbit[i] + disp[i] * d0 for i in range(4)] + [0.0, d0] for d0 in (1e-5, 2e-5)]
        (t,) = _node([{"scenario": sc, "mode": "tbt", "states": starts, "turns": 1000}])
        out = []
        for s0, traj in zip(starts, t["tbt"], strict=True):
            d0 = s0[5]
            u = (np.vstack([s0[:4], traj]) - co - D * d0) @ nf.w_inv.T
            jb = 0.5 * np.sum(u**2, axis=1).max()
            out.append(float(jb / ((D[0] * d0) ** 2 / beta1)))
        return out

    t = o["twiss"]
    a1, a2 = actions(o["orbit0"], [t["disp_x"], t["disp_px"], t["disp_y"], t["disp_py"]])
    assert a1 < 1e-8
    assert a2 / a1 * 4.0 >= 14.0  # the scale itself carries delta0^2
    t4 = o["twiss4"]
    b1, b2 = actions(o["orbit4"], [t4["disp_x"], t4["disp_px"], t4["disp_y"], t4["disp_py"]])
    assert b1 > 1e-5
    assert 3.5 <= b2 / b1 * 4.0 <= 4.1


# --- 7. nothing else moves -----------------------------------------------------------------------
def _without_cavity(p: dict) -> dict:
    q = dict(p)
    q["elements"] = [e for e in p["elements"] if e.get("type") != "RFCavity"]
    return q


KICK_FREE = {
    "fodo-cell": PRESETS["fodo-cell"],
    "dba-cell": PRESETS["dba-cell"],
    "electron-ring-no-rf": _without_cavity(PRESETS["electron-ring"]),
    "centred-rolled": _centred_ring(),
}
_TWISS_KEYS = ("beta_x", "alpha_x", "beta_y", "alpha_y", "disp_x", "disp_px", "disp_y", "disp_py")


@pytest.mark.parametrize("name", list(KICK_FREE), ids=str)
def test_a_kick_free_ring_without_rf_keeps_the_design_optics(name: str) -> None:
    """Orbit exactly zero, no 4D form attached, and ``twiss`` IS the analysis' ``twiss0``, bit
    for bit."""
    sc = KICK_FREE[name]
    o = _editor(sc)["optics"]
    (d,) = _node([{"scenario": sc, "mode": "design"}])
    assert o["orbit0"] == [0.0] * 6 and o["normal4"] is None
    assert [o["twiss"][k] for k in _TWISS_KEYS] == [d["twiss0"][k] for k in _TWISS_KEYS]


@pytest.mark.parametrize("preset_id", ["electron-ring", "wiggler-ring"], ids=str)
def test_a_cavity_ring_keeps_its_design_4d_record(preset_id: str) -> None:
    """On a cavity ring the 4D path is not taken: no 4D form, and ``twiss4`` / ``orbit4`` are the
    analysis' design Twiss and affine orbit, bit for bit (the 6D path is held by
    ``test_injection_on_orbit.py``)."""
    sc = PRESETS[preset_id]
    o = _editor(sc)["optics"]
    (d,) = _node([{"scenario": sc, "mode": "design"}])
    assert o["normal4"] is None and o["normal"] is not None
    assert [o["twiss4"][k] for k in _TWISS_KEYS] == [d["twiss0"][k] for k in _TWISS_KEYS]
    assert o["orbit4"] == (d["closedOrbit0"] or [0.0] * 4) + [0.0, 0.0]


# --- 9. the coupled matched beam is the map's fixed point -----------------------------------------
@pytest.mark.parametrize("name", list(COUPLED), ids=str)
def test_the_coupled_matched_beam_is_the_maps_fixed_point(name: str) -> None:
    """On a design-coupled ring (refused by the tab before) the drawn beam is the fixed point of
    the package's on-orbit map, as in gate 4, to ``1e-9`` (measured ``1.7e-11``, ``7e-14``). The
    control is the same beam built with ``W4``'s off-diagonal blocks dropped, the coupling
    ignored: it misses by more than ``0.1`` (measured ``0.27``). This is the arbiter-free proof
    that the coupling is handled, not just ported."""
    sc = COUPLED[name]
    r = _editor(sc, {"n": 600, "seed": 7})
    o, p = r["optics"], r["inj"]
    _lat, co, J, _nf, _D = _package(sc)
    assert _fixed_point_miss(J, _covariance5(r["initial"], co)) < 1e-9

    W = np.array(o["normal4"]["W"])
    D = np.array(o["normal4"]["dispersion"])
    Wd = W.copy()
    Wd[0:2, 2:4] = 0.0
    Wd[2:4, 0:2] = 0.0
    eps = np.diag([p["emit_x"], p["emit_x"], p["emit_y"], p["emit_y"]])
    sd2 = p["sigma_delta"] ** 2
    Sig = np.zeros((5, 5))
    Sig[:4, :4] = Wd @ eps @ Wd.T + np.outer(D, D) * sd2
    Sig[:4, 4] = Sig[4, :4] = D * sd2
    Sig[4, 4] = sd2
    assert _fixed_point_miss(J, Sig) > 0.1


# --- 10. a particle in one mode stays in it --------------------------------------------------------
@pytest.mark.parametrize("name", list(COUPLED), ids=str)
def test_a_particle_in_one_coupled_mode_stays_in_it(name: str) -> None:
    """Put at ``x_co + W4 (sqrt(2 J1), 0, 0, 0)``, ``J1 = 1e-8``, tracked 1000 turns by the editor
    and read with the editor's ``W4^-1``, the particle's mode-2 action stays below ``1e-9`` of
    ``J1`` (measured ``3.7e-11``). The control, the same amplitude put in lab ``x`` with the mode-1
    Twiss (what a per-plane sampler would do), carries above ``1e-5`` of it (``1.3e-4``) into
    mode 2."""
    sc = COUPLED[name]
    o = _editor(sc)["optics"]
    n4, t = o["normal4"], o["twiss"]
    W, Wi = np.array(n4["W"]), np.array(n4["Winv"])
    x4 = np.array(o["orbit0"][:4])
    j1 = 1e-8
    a = np.sqrt(2.0 * j1)
    s_mode = list(x4 + W @ np.array([a, 0.0, 0.0, 0.0])) + [0.0, 0.0]
    b1, a1 = t["beta_x"], t["alpha_x"]
    s_lab = list(x4 + np.array([a * np.sqrt(b1), -a1 * a / np.sqrt(b1), 0.0, 0.0])) + [0.0, 0.0]
    (tr,) = _node([{"scenario": sc, "mode": "tbt", "states": [s_mode, s_lab], "turns": 1000}])

    def mode2(s0: list, traj: list) -> float:
        u = (np.vstack([s0[:4], traj]) - x4) @ Wi.T
        return float(0.5 * (u[:, 2] ** 2 + u[:, 3] ** 2).max() / j1)

    assert mode2(s_mode, tr["tbt"][0]) < 1e-9
    assert mode2(s_lab, tr["tbt"][1]) > 1e-5


# --- 11. the kick-free coupled ring --------------------------------------------------------------
def test_a_kick_free_coupled_ring_uses_the_design_modes_bit_for_bit() -> None:
    """The centre is exactly zero, so ``orbitMap`` is the design matrix itself and ``W4`` is
    ``normalForm4`` of its transverse block, bit for bit."""
    sc = COUPLED["centred-skew-quad"]
    o = _editor(sc)["optics"]
    assert o["orbit0"] == [0.0] * 6 and o["coupled"] is True
    (d,) = _node([{"scenario": sc, "mode": "design"}])
    assert o["orbitMap"] == d["oneTurn"]
    M4 = [row[:4] for row in d["oneTurn"][:4]]
    (n,) = _node([{"mode": "nf4", "matrix": M4}])
    assert o["normal4"]["W"] == n["normal"]["W"]


# --- 12. the coupled cavity ring ------------------------------------------------------------------
def test_a_coupled_cavity_ring_is_matched_in_its_6d_modes() -> None:
    """``electron-ring`` with the skew quadrupole: the analysis has no ``Q_s`` for it, and the 6D
    path now runs on the cavity alone. ``W`` against ``normal_form(J, "6d")`` at the package's 6D
    orbit to ``1e-9`` of ``max |W|`` (measured ``6.7e-14``); the drawn beam is the 6D map's fixed
    point to ``1e-9`` (``1.3e-13``; with ``W``'s x-y blocks dropped ``0.41``); ``Q_s`` is the third
    mode's; ``twiss4``, shown beside the dynamic dispersion, is the design matrix's 4D mode
    Twiss and static ``D``, against the package to ``1e-9``."""
    sc = _skew(PRESETS["electron-ring"])
    r = _editor(sc, {"n": 600, "seed": 7})
    o = r["optics"]
    assert o["coupled"] is True and o["normal"] is not None and o["normal4"] is None
    lat = load_scenario(sc).lattice
    co6 = closed_orbit_6d(lat)
    J = linearised_one_turn_map(lat, co6[:4], delta=co6[5])
    nf = normal_form(J, method="6d")
    W = np.array(o["normal"]["W"])
    assert np.max(np.abs(W - nf.w)) < 1e-9 * np.max(np.abs(nf.w))
    assert max(abs(W[i, j]) for i in (0, 1) for j in (2, 3)) > 1e-2 * np.max(np.abs(W))  # coupled

    X = np.array(r["initial"]) - co6
    Sig = X.T @ X / len(X)
    norm = np.sqrt(np.outer(np.diag(Sig), np.diag(Sig)))
    assert np.max(np.abs(J @ Sig @ J.T - Sig) / norm) < 1e-9

    q3 = o["normal"]["tunes"][2]
    assert o["Qs"] == min(q3, 1.0 - q3)

    M, _k = lat.one_turn_map()
    nf4 = normal_form(M, method="4d")
    D4 = np.linalg.solve(np.eye(4) - M[:4, :4], M[:4, 5])
    t4 = o["twiss4"]
    got = np.array([t4["beta_x"], t4["alpha_x"], t4["beta_y"], t4["alpha_y"]])
    want = np.array([nf4.mode_beta[0], nf4.mode_alpha[0], nf4.mode_beta[1], nf4.mode_alpha[1]])
    assert np.max(np.abs(got - want) / np.abs(want)) < 1e-9
    Dt = np.array([t4["disp_x"], t4["disp_px"], t4["disp_y"], t4["disp_py"]])
    assert np.max(np.abs(Dt - D4)) < 1e-9 * abs(D4[0])


# Found after pre-commitment (2026-10-20, the advisor's review): a bend-free thin-lens FODO ring
# has Q_x = Q_y exactly, where normalForm4 must refuse (its projectors divide by cos mu_x -
# cos mu_y). With a horizontal corrector it injected before the 4D path moved onto the orbit;
# the exactly uncoupled on-orbit map now takes the per-plane match and still does.
def _equal_tune_ring(kick_y: float = 0.0) -> dict:
    cell = [
        {"type": "ThinQuadrupole", "name": "QF", "k1l": 0.1},
        {"type": "Drift", "name": "D", "length": 5.0},
        {"type": "ThinQuadrupole", "name": "QD", "k1l": -0.1},
        {"type": "Drift", "name": "D", "length": 5.0},
    ]
    els = [dict(e) for _ in range(12) for e in cell]
    els.insert(3, {"type": "Corrector", "name": "CH", "kick_x": 1e-5, "kick_y": kick_y})
    return {
        "format": "accsim-scenario/1",
        "name": "equal-tunes",
        "periodic": True,
        "reference": {"species": "proton", "energy_mode": "total", "energy_eV": 1e10},
        "beam": {"emit_x": 1e-8, "emit_y": 1e-8, "sigma_delta": 1e-4},
        "elements": els,
    }


def test_a_steered_ring_with_equal_tunes_still_injects() -> None:
    """Horizontal kick only: the on-orbit map is exactly uncoupled, the per-plane match builds
    the modes, and the drawn beam is the package's on-orbit map's fixed point to ``1e-9``. With a
    vertical kick too the map is coupled ON the coupling resonance (``Q_x = Q_y``), where the
    modes cannot be separated: refused, and the message says why."""
    sc = _equal_tune_ring()
    r = _editor(sc, {"n": 600, "seed": 7})
    o = r["optics"]
    assert o["normal4"] is not None
    # Inside normalForm4's refusal band, |cos mu_x - cos mu_y| < 1e-6 (measured 2e-9: the orbit
    # moves Q_x by 1.6e-9 off the design's exact equality).
    qx, qy = o["normal4"]["tunes"]
    assert abs(np.cos(2 * np.pi * qx) - np.cos(2 * np.pi * qy)) < 1e-6
    _lat, co, J, _nf, _D = _package(sc)
    assert _fixed_point_miss(J, _covariance5(r["initial"], co)) < 1e-9

    case = {"scenario": _equal_tune_ring(kick_y=1e-5), "mode": "inject", "inj": {"n": 8}}
    proc = subprocess.run(
        [NODE, str(EDITOR / "inject-selftest.js")],
        input=json.dumps([case]),
        capture_output=True,
        text=True,
        check=True,
    )
    (res,) = json.loads(proc.stdout)
    assert "coupling resonance" in res["error"]
