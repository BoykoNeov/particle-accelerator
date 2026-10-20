"""The on-orbit Twiss holds the RF cavity fixed (``docs/ROADMAP.md``, gates pre-committed).

``closed_twiss_on_orbit`` and ``coupled_twiss_on_orbit`` matched on ``linearised_one_turn_map``,
the unfrozen product of the per-element on-orbit maps. With ``electron-ring``'s cavity moved
between the bends that missed the design optics by 0.23% (element 100) and 1.6% (element 50) in
beta even unsteered, and disagreed with ``propagate_twiss_on_orbit``'s own start, which freezes.
Both now match on the frozen product of the same maps. ``linearised_one_turn_map`` itself stays
the tracked turn's honest Jacobian.

What this does not fix (change B): on a *steered* ring the 4D closed orbit itself moves with the
cavity's position, so the on-orbit optics there still differ from the cavity-last ring's, first
order in the kick. Those rings are gated only for self-consistency below.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import accsim as ac
from accsim.lattice import Lattice
from accsim.scenario import load_scenario
from accsim.twiss import match_periodic, match_periodic_coupled

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
    corrector after the 27th (``kicked-ring``'s place for it)."""
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


def _with_skew(sc: dict) -> dict:
    sc = copy.deepcopy(sc)
    sc["elements"] = [{"type": "ThinSkewQuadrupole", "name": "SQ", "k1sl": 0.02}, *sc["elements"]]
    return sc


# ---------------------------------------------------------------------- gate 1
@pytest.mark.parametrize("pos", MID)
def test_unsteered_the_on_orbit_optics_are_the_design_optics(pos: int) -> None:
    """Gate 1: the orbit is zero, so the on-orbit maps are the element matrices."""
    lat = _ring(pos)
    got, want = ac.closed_twiss_on_orbit(lat), ac.closed_twiss(lat)
    for name in ("beta_x", "alpha_x", "beta_y", "alpha_y"):
        assert getattr(got, name) == pytest.approx(getattr(want, name), rel=1e-12), name
    for name in ("disp_x", "disp_px"):
        assert getattr(got, name) == pytest.approx(getattr(want, name), rel=0, abs=1e-12), name


@pytest.mark.parametrize("pos", MID)
def test_unsteered_the_coupled_on_orbit_optics_are_the_design_optics(pos: int) -> None:
    """Gate 1, coupled: the skew ring's on-orbit modes are ``coupled_twiss``'s."""
    lat = _ring(pos, SKEW)
    got, want = ac.coupled_twiss_on_orbit(lat), ac.coupled_twiss(lat)
    for name in ("beta_1", "alpha_1", "beta_2", "alpha_2"):
        assert getattr(got, name) == pytest.approx(getattr(want, name), rel=1e-12), name


@pytest.mark.parametrize("pos", MID)
def test_the_unfrozen_product_really_was_wrong(pos: int) -> None:
    """Gate 1's control: the full Jacobian missed beta by 0.23% (100) and 1.6% (50)."""
    lat = _ring(pos)
    old = match_periodic(ac.linearised_one_turn_map(lat))
    assert abs(old.beta_x / ac.closed_twiss(lat).beta_x - 1.0) > 1e-3
    skew = _ring(pos, SKEW)
    old_c = match_periodic_coupled(ac.linearised_one_turn_map(skew))
    assert abs(old_c.beta_1 / ac.coupled_twiss(skew).beta_1 - 1.0) > 1e-3


# ---------------------------------------------------------------------- gate 2
STEERED = [(pos, d) for pos in (None, *MID) for d in (0.0, 1e-4)]
STEERED_IDS = [f"cavity-at-{p or 'end'}-delta-{d:g}" for p, d in STEERED]


@pytest.mark.parametrize(("pos", "delta"), STEERED, ids=STEERED_IDS)
def test_the_closed_form_is_the_propagated_start(pos: int | None, delta: float) -> None:
    """Gate 2: one model, one computation, the same floats, steered or not."""
    lat = _ring(pos, kick=KICK)
    assert (
        ac.closed_twiss_on_orbit(lat, delta=delta)
        == ac.propagate_twiss_on_orbit(lat, delta=delta)[0]
    )


@pytest.mark.parametrize(("pos", "delta"), STEERED, ids=STEERED_IDS)
def test_the_coupled_closed_form_is_the_propagated_start(pos: int | None, delta: float) -> None:
    """Gate 2, coupled: the same for the skew ring's modes."""
    lat = _ring(pos, SKEW, kick=KICK)
    assert (
        ac.coupled_twiss_on_orbit(lat, delta=delta)
        == ac.propagate_coupled_twiss_on_orbit(lat, delta=delta)[0]
    )


# ---------------------------------------------------------------------- gate 3
def _cavity_last() -> dict[str, Lattice]:
    out = {pid: load_scenario(PRESETS[pid]).lattice for pid in PERIODIC}
    out["electron-ring+kick"] = _ring(None, kick=KICK)
    return out


@pytest.mark.parametrize(("name", "lat"), list(_cavity_last().items()))
def test_a_cavity_last_or_absent_moves_nothing(name: str, lat: Lattice) -> None:
    """Gate 3: every non-cavity on-orbit map's ``delta`` row is exactly the unit row, so rows
    0-4 of the frozen product are the unfrozen one's and every number is the same float."""
    assert ac.closed_twiss_on_orbit(lat) == match_periodic(ac.linearised_one_turn_map(lat))


@pytest.mark.parametrize("pid", [*PERIODIC, "electron-ring+kick"])
def test_a_cavity_last_or_absent_moves_nothing_coupled(pid: str) -> None:
    """Gate 3, coupled: the same on each ring with a skew quadrupole in front."""
    if pid == "electron-ring+kick":
        lat = _ring(None, SKEW, kick=KICK)
    else:
        lat = load_scenario(_with_skew(PRESETS[pid])).lattice
    want = match_periodic_coupled(ac.linearised_one_turn_map(lat))
    assert ac.coupled_twiss_on_orbit(lat) == want
