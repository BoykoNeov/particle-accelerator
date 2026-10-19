"""The injected beam centred on the 6D closed orbit (``closedOrbit6`` in ``editor/inject.js``).

On a ring with a kick (a corrector, a displaced magnet) and an RF cavity the closed orbit is
not at ``delta = 0``. The kicked orbit is longer than the design circumference while the RF
is locked to the design revolution, so the beam closes only at the momentum ``delta_co`` that
brings its path back into step (N5's ``closed_orbit_delta``; I4's ``closed_orbit_6d`` with
radiation off gives the same point, ``zeta_co = 0``). A beam centred on the 4D orbit at
``delta = 0`` starts a synchrotron oscillation of amplitude ``|delta_co|`` nobody asked for.

The editor finds the point by Newton on its own ported tracked turn, started from the affine
solve of the one-turn map. These gates hold it:

  * **the port** — against ``closed_orbit_6d`` to round-off, where the affine start alone
    misses ``delta_co`` by 5% (so the Newton is what passes);
  * **centred means it stays put** — a particle on it returns to itself every turn; one on the
    old 4D centre swings in ``delta`` by ``2 |delta_co|``;
  * **the energy offset is about the ring's own momentum** — ``<delta> = delta_co + delta0``;
  * **nothing else moves** — on a cavity ring with no kick at all (no corrector, no
    displaced magnet, no wiggler: its snaking path is a kick in ``zeta``) the orbit is
    exactly zero.

The sampler's own gates (``test_injection_6d.py``) were re-referenced to ``closed_orbit_6d``
at unchanged tolerances; before that they failed by the predicted ``(delta_co /
sigma_delta)^2 = 6.5e-5`` and ``5e-3``. Node-driven; skipped (not failed) without Node.
"""

from __future__ import annotations

import copy
import shutil

import numpy as np
import pytest
from test_injection_6d import FIXTURES, PRESETS, _node, _rolled

from accsim.orbit import closed_orbit, closed_orbit_6d
from accsim.scenario import load_scenario
from accsim.twiss import normal_form

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not on the PATH")

KICKED = FIXTURES["kicked-ring-cavity-rolled"]


def _scaled_kicks(sc: dict, f: float) -> dict:
    """The same ring with every corrector kick and every magnet displacement scaled by ``f``."""
    q = copy.deepcopy(sc)
    for e in q["elements"]:
        for key in ("kick_x", "kick_y", "dx", "dy"):
            if e.get(key):
                e[key] *= f
    return q


# The wiggler preset has no corrector, but its snaking path is 6.4 um longer than its length
# (the wiggler's own affine zeta kick, T1), so it too closes off momentum, at -1.0e-6. Found
# when its pre-committed "kick-free, exactly zero" gate failed; it is the same physics.
ORBIT_RINGS = {
    "kicked-ring-cavity-rolled": KICKED,
    "kicks-x4": _scaled_kicks(KICKED, 4.0),
    "wiggler-ring": PRESETS["wiggler-ring"],
}
STEERED = ("kicked-ring-cavity-rolled", "kicks-x4")


def _editor(sc: dict, inj: dict | None = None) -> dict:
    (r,) = _node([{"scenario": sc, "mode": "inject", "inj": inj or {"n": 8}}])
    return r


def _affine(sc: dict) -> np.ndarray:
    M, k = load_scenario(sc).lattice.one_turn_map()
    return np.linalg.solve(np.eye(6) - M, k)


@pytest.mark.parametrize("name", list(ORBIT_RINGS), ids=str)
def test_the_editor_orbit_is_the_packages_6d_closed_orbit(name: str) -> None:
    """``closedOrbit6`` against ``closed_orbit_6d(lattice)``: transverse to ``1e-12`` of the
    orbit, ``zeta`` to ``1e-15`` m, ``delta`` to ``1e-10`` of ``|delta_co|`` (measured
    ``4.4e-16`` and ``1.1e-14``). Not blind: the affine start misses ``delta_co`` by more than
    1% (measured 5%) and ``x_co`` by more than ``1e-4`` of itself on the steered rings, the
    orbit's second-order path length, so only a converged Newton passes. On ``wiggler-ring``
    the path length is a constant and the affine start is already close; that ring is here
    for the port, not the Newton."""
    sc = ORBIT_RINGS[name]
    r = _editor(sc)
    assert r["optics"]["normal"] is not None, r["optics"].get("normalError")
    js = np.array(r["optics"]["orbit0"])
    py = closed_orbit_6d(load_scenario(sc).lattice)
    scale = np.max(np.abs(py[:4]))
    assert np.max(np.abs(js[:4] - py[:4])) < 1e-12 * scale
    assert abs(js[4] - py[4]) < 1e-15
    assert abs(js[5] - py[5]) < 1e-10 * abs(py[5])

    assert abs(py[5]) > 5e-7  # the ring really sits off momentum
    if name not in STEERED:
        return
    aff = _affine(sc)
    assert abs(aff[5] / py[5] - 1.0) > 1e-2
    assert abs(aff[0] / py[0] - 1.0) > 1e-4


@pytest.mark.parametrize("name", ["kicked-ring-cavity-rolled", "wiggler-ring"], ids=str)
def test_centred_means_it_stays_put(name: str) -> None:
    """1000 turns. A particle on the editor's orbit stays within ``1e-15`` of its start in
    every coordinate (measured ``1.8e-17``). The control, a particle on the old 4D centre
    (the 4D orbit at ``zeta = delta = 0``), swings in ``delta`` by ``2 |delta_co|`` to 5%
    (measured 2.03): it oscillates about the fixed point it was started ``delta_co`` from."""
    sc = ORBIT_RINGS[name]
    r = _editor(sc)
    co6 = r["optics"]["orbit0"]
    lat = load_scenario(sc).lattice
    co4 = list(np.asarray(closed_orbit(lat))[:4]) + [0.0, 0.0]
    (t,) = _node(
        [{"scenario": sc, "mode": "tbt", "states": [co6, co4], "turns": 1000, "full": True}]
    )
    on, off = (np.array(traj) for traj in t["tbt"])
    assert np.max(np.abs(on - np.array(co6))) < 1e-15
    swing = np.max(np.abs(off[:, 5] - co4[5]))
    assert swing / (2.0 * abs(co6[5])) == pytest.approx(1.0, abs=5e-2)


def test_an_energy_offset_is_about_the_rings_own_momentum() -> None:
    """``delta0`` is an offset from ``delta_co``: the drawn beam's ``<delta>`` is
    ``delta_co + delta0`` to ``1e-12``, and its ``<zeta>`` is ``zeta_co`` (zero)."""
    d0 = 1e-3
    r = _editor(KICKED, {"n": 500, "seed": 9, "delta0": d0})
    X = np.array(r["initial"])
    co = r["optics"]["orbit0"]
    assert co[5] != 0.0
    assert X[:, 5].mean() == pytest.approx(co[5] + d0, rel=1e-12)
    assert abs(X[:, 4].mean() - co[4]) < 1e-15


KICK_FREE = {
    "electron-ring": PRESETS["electron-ring"],
    "electron-ring-rolled": _rolled(PRESETS["electron-ring"]),
}


@pytest.mark.parametrize("name", list(KICK_FREE), ids=str)
def test_a_ring_with_no_kick_is_centred_on_zero(name: str) -> None:
    """Nothing kicks the beam, so the 6D closed orbit is exactly zero and is not iterated:
    ``electron-ring`` draws the beam it drew before (bit-identical states were
    checked once against a saved draw when this landed; this is the property that makes it
    so)."""
    r = _editor(KICK_FREE[name])
    assert r["optics"]["normal"] is not None
    assert r["optics"]["orbit0"] == [0.0] * 6


@pytest.mark.parametrize("name", ["kicked-ring-cavity-rolled", "wiggler-ring"], ids=str)
def test_the_readings_are_about_the_6d_centre(name: str) -> None:
    """The page's numbers (``beamStats`` through ``betatronPart``) read the beam about the same
    centre it was drawn on. A matched, centred beam's measured centroid is zero to round-off
    in both planes, and its ``<J>`` is the package's mode action of ``W^-1 (x - x_co6)`` to
    ``1e-12``. Without this gate a reading that forgot ``delta_co`` passed every other test
    here (a planted bug, 2026-10-19): it shows as a centroid of about 1% of the beam size."""
    sc = ORBIT_RINGS[name]
    (r,) = _node([{"scenario": sc, "mode": "stats", "inj": {"n": 400, "seed": 4}, "turns": 0}])
    st = r["stats"][0]
    lat = load_scenario(sc).lattice
    M, _k = lat.one_turn_map()
    nf = normal_form(M, method="6d")
    u = (np.array(r["initial"]) - closed_orbit_6d(lat)) @ np.linalg.inv(nf.w).T
    for k, pl in enumerate(("x", "y")):
        eps = r["inj"][f"emit_{pl}"]
        assert st[pl]["centroid"] < 1e-9 * np.sqrt(eps)
        J = 0.5 * (u[:, 2 * k] ** 2 + u[:, 2 * k + 1] ** 2)
        assert st[pl]["meanJ"] == pytest.approx(J.mean(), rel=1e-12)
