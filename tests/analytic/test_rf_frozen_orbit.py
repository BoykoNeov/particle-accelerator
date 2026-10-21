"""Change B, step 1: the 4D closed orbit holds the RF cavity fixed (``docs/ROADMAP.md``).

The 4D closed orbit is a fixed point at a momentum the caller chooses. Its tracked turn used
to carry ``delta`` through the ring, so a cavity between the bends -- the one element that
reads ``zeta`` -- turned the orbit's path-length slip into a ``delta`` kick part-way round, and
the orbit downstream rode the wrong momentum: 2.7% / 4.0% of its size at ``delta = 1e-4``
(cavity at 100 / 50 on ``electron-ring``), 0.6% / 3.0% on a steered ring at ``delta = 0``. Every
walk along the closed orbit now re-imposes ``delta`` after each element, the model the optics
already match on (xtrack's ``twiss(method="4d")``).

The arbiter that needs no second ring: without radiation a synchronous cavity gives no kick on
its own closed orbit, so the transverse part of the **live** 6D orbit (``closed_orbit_6d``,
untouched here) *is* the frozen 4D orbit at its momentum. The same argument fixes
``closed_orbit_delta``, whose slip was tracked live from ``zeta = 0`` and so was 12% / 38% off.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import accsim as ac
from accsim import orbit as orbit_module
from accsim.coords import DELTA
from accsim.lattice import Lattice
from accsim.orbit import (
    closed_orbit_6d,
    closed_orbit_delta,
    closed_orbit_nonlinear,
    linearised_element_maps,
    propagate_orbit_nonlinear,
)
from accsim.scenario import load_scenario
from accsim.twiss import closed_orbit_path_length, second_order_dispersion

EDITOR = Path(__file__).resolve().parents[2] / "editor"


def _presets() -> dict[str, dict]:
    text = (EDITOR / "presets.js").read_text(encoding="utf-8")
    marker = "\nvar ACCSIM_PRESETS = "
    start = text.index(marker) + len(marker)
    return {p["id"]: p for p in json.loads(text[start : text.rindex("];") + 1])}


PRESETS = _presets()
PERIODIC = [pid for pid, p in PRESETS.items() if p.get("periodic", True) is not False]
SKEW = 0.01
KICK = -6e-5
MID = [100, 50]


def _ring(pos: int | None, k1sl: float | None = None, kick: float | None = None) -> Lattice:
    """``electron-ring`` with its thin cavity at element ``pos`` (``None``: last, as shipped).
    ``k1sl`` adds a thin skew quadrupole after the fifth element, ``kick`` a horizontal
    corrector after the 27th. The same rings as ``test_rf_frozen_on_orbit.py``."""
    sc = copy.deepcopy(PRESETS["electron-ring"])
    cav = next(e for e in sc["elements"] if e["type"] == "RFCavity")
    els = [e for e in sc["elements"] if e["type"] != "RFCavity"]
    if kick is not None:
        corrector = {"type": "Corrector", "name": "CH", "kick_x": kick, "kick_y": 0.0}
        els = [*els[:27], corrector, *els[27:]]
    if k1sl is not None:
        els = [*els[:5], {"type": "ThinSkewQuadrupole", "name": "SQ", "k1sl": k1sl}, *els[5:]]
    els = [*els, cav] if pos is None else [*els[:pos], cav, *els[pos:]]
    sc["elements"] = els
    return load_scenario(sc).lattice


def _drop_cavity_point(points: list, pos: int) -> list:
    """Boundaries of the cavity-at-``pos`` ring aligned with the cavity-last ring's.

    The cavity is transparent in the frozen walk, so the point after it repeats the one before;
    dropping it leaves ``N + 1`` points against the cavity-last ring's ``N + 2``, whose last two
    are the same for the same reason.
    """
    return [*points[: pos + 1], *points[pos + 2 :]]


def _live_step(elem, state, ref):
    """The walk as it was: ``delta`` carried, so a cavity kicks it."""
    return elem.track(state, ref)


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch):
    """Run the package with the old, live walk -- the control and the bit-for-bit reference."""
    monkeypatch.setattr(orbit_module, "_step", _live_step)


DELTAS = [0.0, 1e-4, -1e-3]
# The unsteered ring at ``delta = 0`` is the design orbit, zero either way, so it is left out.
CASES = [(k, p, d) for k in (None, KICK) for p in MID for d in DELTAS if k is not None or d != 0.0]
CASE_IDS = [f"{'steered' if k else 'unsteered'}-cavity-at-{p}-delta-{d:g}" for k, p, d in CASES]


# ---------------------------------------------------------------------- gate 1
@pytest.mark.parametrize(("kick", "pos", "delta"), CASES, ids=CASE_IDS)
def test_the_orbit_does_not_move_with_the_cavity(kick, pos: int, delta: float) -> None:
    """Gate 1: the same orbit wherever the cavity sits, along the ring and closed."""
    want = closed_orbit_nonlinear(_ring(None, kick=kick), delta=delta)
    lat = _ring(pos, kick=kick)
    got = closed_orbit_nonlinear(lat, delta=delta)
    assert np.max(np.abs(got - want)) < 1e-16

    along = propagate_orbit_nonlinear(lat, delta=delta)
    ref_along = propagate_orbit_nonlinear(_ring(None, kick=kick), delta=delta)
    assert np.max(np.abs(along[-1] - along[0])) < 1e-16
    aligned = np.array(_drop_cavity_point(along, pos))
    assert np.max(np.abs(aligned - np.array(ref_along[:-1]))) < 1e-16


@pytest.mark.parametrize(("kick", "pos", "delta"), CASES, ids=CASE_IDS)
def test_the_live_walk_really_moved_the_orbit(live, kick, pos: int, delta: float) -> None:
    """Gate 1's control: the old walk misses the cavity-last orbit by more than 4e-3."""
    want = closed_orbit_nonlinear(_ring(None, kick=kick), delta=delta)
    got = closed_orbit_nonlinear(_ring(pos, kick=kick), delta=delta)
    assert np.max(np.abs(got - want)) > 4e-3 * np.max(np.abs(want))


# ---------------------------------------------------------------------- gate 2
@pytest.mark.parametrize("pos", [None, *MID])
def test_the_live_6d_orbit_is_the_frozen_4d_orbit(pos: int | None) -> None:
    """Gate 2: the synchronous cavity does nothing on its own orbit, so the two agree.

    The 4D solve runs at ``tol = 1e-16``: at its default ``1e-14`` Newton stopped on the
    cavity-last ring with a residual of ``9.9e-16`` and the orbit ``1.4e-15`` short, its own
    floor and not the cavity's (that ring's live and held walks are the same bits). The 6D
    solve was already at ``1.6e-18``."""
    lat = _ring(pos, kick=KICK)
    co6 = closed_orbit_6d(lat)
    co4 = closed_orbit_nonlinear(lat, delta=co6[DELTA], tol=1e-16)
    assert np.max(np.abs(co6[:4] - co4)) < 1e-16
    assert abs(closed_orbit_delta(lat) - co6[DELTA]) < 1e-12


@pytest.mark.parametrize("pos", MID)
def test_the_closed_orbit_momentum_does_not_move_with_the_cavity(pos: int) -> None:
    """Gate 2: the momentum a bunched ring sits at is the cavity-last ring's."""
    want = closed_orbit_delta(_ring(None, kick=KICK))
    assert abs(closed_orbit_delta(_ring(pos, kick=KICK)) - want) < 1e-12


@pytest.mark.parametrize("pos", MID)
def test_the_old_closed_orbit_momentum_was_far_off(live, pos: int) -> None:
    """Gate 2's control: 12% (100) and 38% (50) below the 6D orbit's momentum."""
    lat = _ring(pos, kick=KICK)
    assert abs(closed_orbit_delta(lat) / closed_orbit_6d(lat)[DELTA] - 1.0) > 0.1


# ---------------------------------------------------------------------- gate 3
@pytest.mark.parametrize("pos", MID)
def test_the_path_length_follows_the_orbit(pos: int) -> None:
    """Gate 3: the off-momentum orbit's length, read off ``zeta``, is the cavity-last ring's."""
    want = closed_orbit_path_length(_ring(None, kick=KICK), 1e-4)
    got = closed_orbit_path_length(_ring(pos, kick=KICK), 1e-4)
    assert got == pytest.approx(want, rel=1e-14)


@pytest.mark.parametrize("pos", MID)
def test_the_second_order_dispersion_follows_the_orbit(pos: int) -> None:
    """Gate 3: ``d`` and ``d^2`` of the orbit in ``delta``, boundary by boundary."""
    want = second_order_dispersion(_ring(None, kick=KICK))[:-1]
    got = _drop_cavity_point(second_order_dispersion(_ring(pos, kick=KICK)), pos)
    for name in ("disp_x", "disp_px", "disp_y", "disp_py"):
        diff = max(abs(getattr(g, name) - getattr(w, name)) for g, w in zip(got, want, strict=True))
        assert diff < 1e-12, name
    for name in ("ddisp_x", "ddisp_px", "ddisp_y", "ddisp_py"):
        diff = max(abs(getattr(g, name) - getattr(w, name)) for g, w in zip(got, want, strict=True))
        assert diff < 1e-10, name


@pytest.mark.parametrize("pos", MID)
@pytest.mark.parametrize("delta", [0.0, 1e-4])
def test_the_on_orbit_optics_follow_the_orbit(pos: int, delta: float) -> None:
    """Gate 3: the steered on-orbit optics are the cavity-last ring's (the leftover of the
    on-orbit Twiss fix, 2.5e-6 / 7.8e-6 in beta, is gone)."""
    want = ac.propagate_twiss_on_orbit(_ring(None, kick=KICK), delta=delta)[:-1]
    got = _drop_cavity_point(ac.propagate_twiss_on_orbit(_ring(pos, kick=KICK), delta=delta), pos)
    for g, w in zip(got, want, strict=True):
        assert g.beta_x == pytest.approx(w.beta_x, rel=1e-9)
        assert g.beta_y == pytest.approx(w.beta_y, rel=1e-9)
        assert g.disp_x == pytest.approx(w.disp_x, rel=0, abs=1e-10)
        assert g.disp_px == pytest.approx(w.disp_px, rel=0, abs=1e-10)
    closed = ac.closed_twiss_on_orbit(_ring(pos, kick=KICK), delta=delta)
    assert closed.beta_x == pytest.approx(want[0].beta_x, rel=1e-9)


@pytest.mark.parametrize("pos", MID)
def test_the_coupled_on_orbit_optics_follow_the_orbit(pos: int) -> None:
    """Gate 3, coupled: the steered skew ring's mode betas are the cavity-last ring's."""
    want = ac.coupled_twiss_on_orbit(_ring(None, SKEW, kick=KICK))
    got = ac.coupled_twiss_on_orbit(_ring(pos, SKEW, kick=KICK))
    assert got.beta_1 == pytest.approx(want.beta_1, rel=1e-9)
    assert got.beta_2 == pytest.approx(want.beta_2, rel=1e-9)


# ---------------------------------------------------------------------- gate 4
def _results(lat: Lattice) -> dict[str, object]:
    out: dict[str, object] = {"delta_co": closed_orbit_delta(lat)}
    for d in (0.0, 1e-4):
        out[f"co{d}"] = closed_orbit_nonlinear(lat, delta=d)
        out[f"along{d}"] = np.array(propagate_orbit_nonlinear(lat, delta=d))
        out[f"maps{d}"] = np.array(linearised_element_maps(lat, delta=d))
    out["path"] = closed_orbit_path_length(lat, 1e-4)
    return out


def _assert_same_bits(new: dict, old: dict) -> None:
    assert new.keys() == old.keys()
    for key in new:
        assert np.array_equal(np.asarray(new[key]), np.asarray(old[key])), key


@pytest.mark.parametrize("pid", PERIODIC)
def test_nothing_else_moves_on_the_presets(monkeypatch: pytest.MonkeyPatch, pid: str) -> None:
    """Gate 4: on every periodic preset as shipped the frozen walk is the live one, bit for bit."""
    lat = load_scenario(PRESETS[pid]).lattice
    new = _results(lat)
    monkeypatch.setattr(orbit_module, "_step", _live_step)
    _assert_same_bits(new, _results(lat))


def test_nothing_else_moves_on_the_steered_cavity_last_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate 4: the steered ring with its cavity last, bit for bit."""
    lat = _ring(None, kick=KICK)
    new = _results(lat)
    monkeypatch.setattr(orbit_module, "_step", _live_step)
    _assert_same_bits(new, _results(lat))
