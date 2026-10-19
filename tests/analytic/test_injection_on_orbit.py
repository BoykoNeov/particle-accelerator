"""The injected beam's optics taken on the closed orbit (``orbitMap`` in ``editor/inject.js``).

On a cavity ring the 6D-matched beam is centred on the tracked 6D closed orbit. Its shape
used to come from ``W``, the normal form of the one-turn matrix about the **design** orbit.
A beam on a displaced orbit sees a different linear map, the Jacobian of the tracked turn at
that orbit (I3's optics on orbit), and the change is first order in the orbit: the exact
drifts at the orbit's angle (54%) and the exact bends (46%) make dispersion of their own, 1.9%
at the injection point of the kicked fixture. (Not the thin quadrupoles: their kick has no
``delta`` in it.)

The editor now takes ``W`` from that Jacobian (central differences of its ported tracked
turn). These gates hold it:

  * **the port** — against the package's ``linearised_one_turn_map`` (a product of
    per-element Jacobians, a different route) at the measured cross-route floor;
  * **the dispersion by a route that takes no Jacobian** — ``d x_co / d delta`` from closed
    orbits at shifted momenta;
  * **the matched beam is the map's fixed point** — ``J Sigma J^T = Sigma`` with the package's
    ``J``; the design-``W`` beam misses at first order;
  * **one particle on the dynamic dispersion makes no betatron oscillation** — tracked, its
    residual is fourth order in the energy offset; the design ``W``'s is second;
  * **the old beam's mismatch is the dispersion's** — second order in kick x energy spread;
  * **nothing else moves** — a kick-free ring keeps its one-turn matrix exactly.

Node-driven; skipped (not failed) without Node.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from test_injection_6d import _S, PRESETS, _eps, _node, _rolled
from test_injection_orbit6 import KICKED, _scaled_kicks

from accsim.orbit import closed_orbit_6d, closed_orbit_nonlinear, linearised_one_turn_map
from accsim.scenario import load_scenario
from accsim.twiss import normal_form

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not on the PATH")

RINGS = {
    "kicked-ring-cavity-rolled": KICKED,
    "kicks-x4": _scaled_kicks(KICKED, 4.0),
    "wiggler-ring": PRESETS["wiggler-ring"],
}
STEERED = ("kicked-ring-cavity-rolled", "kicks-x4")


def _editor(sc: dict, inj: dict | None = None) -> dict:
    (r,) = _node([{"scenario": sc, "mode": "inject", "inj": inj or {"n": 8}}])
    assert r["optics"]["normal"] is not None, r["optics"].get("normalError")
    return r


def _package(sc: dict):
    """The package's 6D closed orbit, its on-orbit map (product of per-element Jacobians) and
    that map's normal form, and the design normal form."""
    lat = load_scenario(sc).lattice
    co = closed_orbit_6d(lat)
    J = linearised_one_turn_map(lat, co[:4], delta=co[5])
    M, _k = lat.one_turn_map()
    return lat, co, J, normal_form(J, method="6d"), normal_form(M, method="6d")


def _static_dispersion(J: np.ndarray) -> np.ndarray:
    return np.linalg.solve(np.eye(4) - J[:4, :4], J[:4, 5])


# --- 1. the port ------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(RINGS), ids=str)
def test_the_editor_on_orbit_optics_are_the_packages(name: str) -> None:
    """``orbitMap`` against ``linearised_one_turn_map`` to ``5e-9`` per entry, and ``W`` against
    ``normal_form`` of it to ``1e-9`` of ``max |W|``. The two Jacobians come by different routes
    (a whole-turn central difference against a product of per-element ones), which agree to a
    measured floor of ``1e-10`` to ``6.4e-10`` in ``J`` and ``3e-11`` in ``W``: the amount by which
    the package's own product moves when its step changes. Not blind: the design ``W`` differs
    by ``2.4e-3``, ``9.6e-3`` and ``6.6e-6`` of ``max |W|``."""
    sc = RINGS[name]
    r = _editor(sc)
    _lat, _co, J, nf, nf0 = _package(sc)
    Jjs = np.array(r["optics"]["orbitMap"])
    assert np.max(np.abs(Jjs - J)) < 5e-9
    js = r["optics"]["normal"]
    W = np.array(js["W"])
    scale = np.max(np.abs(nf.w))
    assert np.max(np.abs(W - nf.w)) < 1e-9 * scale
    for a, b in zip(js["tunes"], nf.tunes, strict=True):
        assert abs((a - b + 0.5) % 1.0 - 0.5) < 1e-9
    assert js["modeBeta"] == pytest.approx(nf.mode_beta, rel=1e-9, abs=0)
    assert np.max(np.abs(np.array(js["dispersion"]) - nf.dispersion)) < 1e-9 * abs(nf.dispersion[0])
    S = _S()
    assert np.max(np.abs(W.T @ S @ W - S)) < 1e-9
    assert np.max(np.abs(nf0.w - nf.w)) > 6e3 * 1e-9 * scale  # not blind


# --- 2. the dispersion by a route that takes no Jacobian -----------------------------------------
@pytest.mark.parametrize("name", STEERED, ids=str)
def test_the_on_orbit_dispersion_is_the_closed_orbits_momentum_slope(name: str) -> None:
    """The static dispersion of the editor's ``orbitMap``, ``(I - J_4)^-1 J[:4, delta]``,
    equals ``d x_co / d delta`` from ``closed_orbit_nonlinear`` at ``delta_co +- 3e-6`` to ``1e-8``
    of ``|D_x|`` (measured ``1.6e-10`` and ``5.7e-10``). Not blind: the design static dispersion
    differs by 1.9% and 7.8%."""
    sc = RINGS[name]
    r = _editor(sc)
    lat = load_scenario(sc).lattice
    co = closed_orbit_6d(lat)
    h = 3e-6
    scan = (
        closed_orbit_nonlinear(lat, delta=co[5] + h) - closed_orbit_nonlinear(lat, delta=co[5] - h)
    ) / (2.0 * h)
    D = _static_dispersion(np.array(r["optics"]["orbitMap"]))
    assert np.max(np.abs(D - scan)) < 1e-8 * abs(D[0])
    M, _k = lat.one_turn_map()
    assert abs(_static_dispersion(M)[0] / D[0] - 1.0) > 1e-2  # not blind


# --- 3. the matched beam is the map's fixed point ------------------------------------------------
@pytest.mark.parametrize("name", STEERED, ids=str)
def test_the_matched_beam_is_the_on_orbit_maps_fixed_point(name: str) -> None:
    """The editor's drawn beam (moment-matched, default settings), covariance about
    ``closed_orbit_6d``, satisfies ``J Sigma J^T = Sigma`` with the PACKAGE's ``J``, each entry to
    ``1e-9`` of ``sqrt(Sigma_ii Sigma_jj)`` (measured ``9e-12`` / ``6.5e-11`` analytically). The
    control, the design-``W`` beam, misses by more than ``1e-2`` (measured ``3.4e-2`` / ``0.14``):
    first order in the orbit."""
    sc = RINGS[name]
    r = _editor(sc, {"n": 600, "seed": 7})
    _lat, co, J, _nf, nf0 = _package(sc)
    X = np.array(r["initial"]) - co
    Sig = X.T @ X / len(X)
    norm = np.sqrt(np.outer(np.diag(Sig), np.diag(Sig)))
    assert np.max(np.abs(J @ Sig @ J.T - Sig) / norm) < 1e-9

    p = r["inj"]
    w0 = nf0.w
    Sig0 = w0 @ np.diag(_eps(w0, p["emit_x"], p["emit_y"], p["sigma_delta"])) @ w0.T
    norm0 = np.sqrt(np.outer(np.diag(Sig0), np.diag(Sig0)))
    assert np.max(np.abs(J @ Sig0 @ J.T - Sig0) / norm0) > 1e-2


# --- 4. one particle on the dynamic dispersion ---------------------------------------------------
def _mode3_particle(W: np.ndarray, co: np.ndarray, d0: float) -> list[float]:
    """The orbit plus the mode-3 point with ``zeta = 0``, ``delta = d0``: what the sampler's energy
    offset puts a particle on."""
    det = W[4, 4] * W[5, 5] - W[4, 5] * W[5, 4]
    u = np.zeros(6)
    u[4], u[5] = -W[4, 5] * d0 / det, W[4, 4] * d0 / det
    return list(co + W @ u)


@pytest.mark.parametrize("name", STEERED, ids=str)
def test_a_particle_on_the_dynamic_dispersion_makes_no_betatron_oscillation(name: str) -> None:
    """Placed with the editor's ``W`` at ``delta0 = 1e-5`` and ``2e-5`` and tracked 1000 turns by the
    editor's tracker, a particle's betatron action (the package's on-orbit modes) stays below
    ``1e-8`` of its synchrotron action, and doubling ``delta0`` raises it by at least 12: fourth
    order, the nonlinearity (measured 18.6). The design ``W``'s particle rises by ``4.0`` to 5%
    (second order, the dispersion error) and is above ``1e-6`` of its synchrotron action."""
    sc = RINGS[name]
    r = _editor(sc)
    _lat, co, _J, nf, nf0 = _package(sc)
    Wjs = np.array(r["optics"]["normal"]["W"])
    Wi = np.linalg.inv(nf.w)
    co_js = np.array(r["optics"]["orbit0"])

    def actions(W: np.ndarray) -> list[tuple[float, float]]:
        starts = [_mode3_particle(W, co_js, d0) for d0 in (1e-5, 2e-5)]
        (t,) = _node(
            [{"scenario": sc, "mode": "tbt", "states": starts, "turns": 1000, "full": True}]
        )
        out = []
        for s0, traj in zip(starts, t["tbt"], strict=True):
            u = (np.vstack([s0, traj]) - co) @ Wi.T
            jb = 0.5 * (u[:, 0] ** 2 + u[:, 1] ** 2 + u[:, 2] ** 2 + u[:, 3] ** 2)
            js = 0.5 * (u[0, 4] ** 2 + u[0, 5] ** 2)
            out.append((float(jb.max()), float(js)))
        return out

    (a1, s1), (a2, _s2) = actions(Wjs)
    assert a1 < 1e-8 * s1
    assert a2 / a1 >= 12.0

    (b1, t1), (b2, _t2) = actions(nf0.w)
    assert b2 / b1 == pytest.approx(4.0, rel=5e-2)
    assert b1 > 1e-6 * t1


# --- 5. the old beam's mismatch is the dispersion's ----------------------------------------------
def test_the_design_beams_mismatch_is_second_order_in_the_dispersion_error() -> None:
    """Package-side: the design-``W`` beam's ``<J_1> / eps - 1``, read in the on-orbit modes,
    scales by ``4.0`` when ``sigma_delta`` halves and by ``16.0`` when the kicks are scaled by 4,
    each to 2% (measured 4.05 and 16.04): the square of (dispersion error x energy spread),
    with the dispersion error first order in the orbit."""

    def excess(sc: dict, sd: float) -> float:
        _lat, _co, _J, nf, nf0 = _package(sc)
        b = sc["beam"]
        w0, w1 = nf0.w, nf.w
        Sig = w0 @ np.diag(_eps(w0, b["emit_x"], b["emit_y"], sd)) @ w0.T
        Wi = np.linalg.inv(w1)
        Su = Wi @ Sig @ Wi.T
        return 0.5 * (Su[0, 0] + Su[1, 1]) / _eps(w1, b["emit_x"], b["emit_y"], sd)[0] - 1.0

    base = excess(KICKED, 7e-4)
    assert base > 1e-4
    assert base / excess(KICKED, 3.5e-4) == pytest.approx(4.0, rel=2e-2)
    assert excess(RINGS["kicks-x4"], 7e-4) / base == pytest.approx(16.0, rel=2e-2)


# --- 6. nothing else moves -----------------------------------------------------------------------
KICK_FREE = {
    "electron-ring": PRESETS["electron-ring"],
    "electron-ring-rolled": _rolled(PRESETS["electron-ring"]),
}


@pytest.mark.parametrize("name", list(KICK_FREE), ids=str)
def test_a_kick_free_ring_keeps_its_one_turn_matrix(name: str) -> None:
    """The closed orbit is exactly zero, so ``orbitMap`` IS the design one-turn matrix (exact
    equality, not a finite difference at zero) and ``W`` is ``normalForm6`` of it, bit for bit.
    (``normalForm6`` itself stays a port test of the design matrix, at ``1e-12``, in
    ``test_injection_6d.py``.)"""
    sc = KICK_FREE[name]
    r = _editor(sc)
    assert r["optics"]["orbit0"] == [0.0] * 6
    (d,) = _node([{"scenario": sc, "mode": "design"}])
    assert r["optics"]["orbitMap"] == d["oneTurn"]
    (n,) = _node([{"mode": "nf6", "matrix": d["oneTurn"]}])
    assert r["optics"]["normal"]["W"] == n["normal"]["W"]
