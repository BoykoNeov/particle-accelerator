"""The rest of the 4D model holds the RF cavity fixed (``docs/ROADMAP.md``, gates pre-committed).

``test_rf_frozen_optics.py`` moved the matched optics onto the one-turn product with ``delta``'s
row frozen. This file is the consumers that entry left on the full 6x6 map: the momentum
compaction (and so ``eta`` and the lumped synchrotron tune), the 4D normal-mode tunes and the 4D
normal form. With ``electron-ring``'s cavity moved between the bends they were 3.6-5.2% wrong
(the compaction), 2.5% wrong (``Q_s``), or refused the ring as unstable (the 4D tunes; and
everything on a coupled ring). The 6D normal form keeps the live cavity: it is the
synchro-betatron answer, and the arbiter for ``Q_s`` below.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import accsim as ac
from accsim.lattice import Lattice
from accsim.scenario import load_scenario

EDITOR = Path(__file__).resolve().parents[2] / "editor"


def _presets() -> dict[str, dict]:
    text = (EDITOR / "presets.js").read_text(encoding="utf-8")
    marker = "\nvar ACCSIM_PRESETS = "
    start = text.index(marker) + len(marker)
    return {p["id"]: p for p in json.loads(text[start : text.rindex("];") + 1])}


PRESETS = _presets()
PERIODIC = [pid for pid, p in PRESETS.items() if p.get("periodic", True) is not False]
SKEW = 0.01
MID = [100, 50]


def _ring(pos: int | None | str, k1sl: float | None = None) -> Lattice:
    """``electron-ring`` with its thin cavity at element ``pos``: ``None`` last (as shipped),
    ``"none"`` removed. ``k1sl`` adds a thin skew quadrupole after the fifth element."""
    sc = copy.deepcopy(PRESETS["electron-ring"])
    cav = next(e for e in sc["elements"] if e["type"] == "RFCavity")
    els = [e for e in sc["elements"] if e["type"] != "RFCavity"]
    if k1sl is not None:
        els = [*els[:5], {"type": "ThinSkewQuadrupole", "name": "SQ", "k1sl": k1sl}, *els[5:]]
    if pos is None:
        els = [*els, cav]
    elif pos != "none":
        els = [*els[:pos], cav, *els[pos:]]
    sc["elements"] = els
    return load_scenario(sc).lattice


def _old_identity(lat: Lattice) -> float:
    """``momentum_compaction``'s identity route as it was: the full 6x6 one-turn map."""
    M = lat.one_turn_matrix()
    m4 = M[:4, :4]
    d = np.linalg.solve(np.eye(4) - m4, M[:4, 5])
    slip = M[4, 0] * d[0] + M[4, 1] * d[1] + M[4, 5] + M[4, 2] * d[2] + M[4, 3] * d[3]
    return 1.0 / lat.ref.gamma0**2 - slip / lat.length


def _eigen_qs(lat: Lattice) -> float:
    q3 = ac.closed_normal_form(lat, method="6d").tunes[2]
    return min(q3, 1.0 - q3)


RINGS = [(pos, k) for k in (None, SKEW) for pos in MID]
IDS = [f"cavity-at-{p}{'-skew' if k else ''}" for p, k in RINGS]


# ---------------------------------------------------------------------- gate 1
@pytest.mark.parametrize(("pos", "k1sl"), RINGS, ids=IDS)
def test_the_compaction_does_not_depend_on_where_the_cavity_sits(pos: int, k1sl) -> None:
    """Gate 1: held fixed, a thin cavity is the identity, so the ring is the cavity-free ring."""
    lat, bare = _ring(pos, k1sl), _ring("none", k1sl)
    assert ac.momentum_compaction(lat) == pytest.approx(ac.momentum_compaction(bare), rel=1e-12)
    quad = ac.momentum_compaction(lat, method="quadrature")
    assert quad == pytest.approx(ac.momentum_compaction(bare, method="quadrature"), rel=1e-12)


@pytest.mark.parametrize(("pos", "k1sl"), RINGS, ids=IDS)
def test_the_two_compaction_routes_still_meet_at_one_over_n_squared(pos: int, k1sl) -> None:
    """Gate 1: the quadrature route starts from the frozen dispersion too, so it converges onto
    the identity route rather than onto a position-dependent number."""
    lat = _ring(pos, k1sl)
    exact = ac.momentum_compaction(lat)
    miss = [
        ac.momentum_compaction(lat, method="quadrature", slices=n) - exact for n in (32, 64, 128)
    ]
    assert 3.9 < miss[0] / miss[1] < 4.1
    assert 3.9 < miss[1] / miss[2] < 4.1


@pytest.mark.parametrize("pos", MID)
def test_the_old_compaction_really_was_wrong(pos: int) -> None:
    """Gate 1's control: the full map's identity route missed by 5.2% (100) and 3.6% (50)."""
    want = ac.momentum_compaction(_ring("none"))
    assert abs(_old_identity(_ring(pos)) / want - 1.0) > 3e-2


# ---------------------------------------------------------------------- gate 2
@pytest.mark.parametrize(("pos", "k1sl"), RINGS, ids=IDS)
def test_the_synchrotron_tune_does_not_depend_on_where_the_cavity_sits(pos: int, k1sl) -> None:
    """Gate 2: the lumped formula meets the 6D eigen-tune at its documented Q_s^2 order."""
    lat = _ring(pos, k1sl)
    qs = ac.synchrotron_tune(lat)
    assert qs == pytest.approx(ac.synchrotron_tune(_ring(None, k1sl)), rel=1e-12)
    assert ac.slip_factor(lat) == pytest.approx(ac.slip_factor(_ring("none", k1sl)), rel=1e-12)
    assert abs(qs / _eigen_qs(lat) - 1.0) < 3e-3


@pytest.mark.parametrize("pos", MID)
def test_the_old_synchrotron_tune_really_was_wrong(pos: int) -> None:
    """Gate 2's control: built on the full map's eta it missed the eigen-tune by 2.5% and 1.6%."""
    lat = _ring(pos)
    eta = _old_identity(lat) - 1.0 / lat.ref.gamma0**2
    cav = next(e for e in lat.elements if isinstance(e, ac.RFCavity))
    old = np.arccos(1.0 - 0.5 * cav.slope(lat.ref) * eta * lat.length) / (2.0 * np.pi)
    assert abs(old / _eigen_qs(lat) - 1.0) > 1e-2


# ---------------------------------------------------------------------- gate 3
@pytest.mark.parametrize(("pos", "k1sl"), RINGS, ids=IDS)
def test_the_4d_tunes_and_normal_form_do_not_depend_on_where_the_cavity_sits(
    pos: int, k1sl
) -> None:
    """Gate 3: these refused the ring as unstable (moduli 1.0024 at 100, 1.0061 at 50)."""
    lat, last = _ring(pos, k1sl), _ring(None, k1sl)
    np.testing.assert_allclose(
        ac.normal_mode_tunes(lat), ac.normal_mode_tunes(last), rtol=0, atol=1e-12
    )
    got, want = ac.closed_normal_form(lat, method="4d"), ac.closed_normal_form(last, method="4d")
    np.testing.assert_allclose(got.tunes, want.tunes, rtol=0, atol=1e-12)
    assert np.abs(got.w - want.w).max() < 1e-12 * np.abs(want.w).max()
    if k1sl is None:
        np.testing.assert_allclose(
            ac.normal_mode_tunes(lat), np.mod(ac.tunes(lat), 1.0), atol=1e-12
        )


@pytest.mark.parametrize(("pos", "k1sl"), RINGS, ids=IDS)
def test_the_4d_normal_form_carried_round_the_ring_closes(pos: int, k1sl) -> None:
    """Found by gate 5, not pre-committed: ``propagate_normal_form`` carried the 4D form in the
    full running transfer, so past a mid-ring cavity the mode phase did not close (mode 1's full
    tune 5.7792 against 5.7815 on the skew ring at 100). Frozen, it ends where it started and
    accumulates the cavity-last ring's phase."""
    lat, last = _ring(pos, k1sl), _ring(None, k1sl)
    pts = ac.propagate_normal_form(lat, ac.closed_normal_form(lat, method="4d"))
    ref = ac.propagate_normal_form(last, ac.closed_normal_form(last, method="4d"))
    assert np.abs(pts[-1].w - pts[0].w).max() < 1e-12 * np.abs(pts[0].w).max()
    np.testing.assert_allclose(pts[-1].mu, ref[-1].mu, rtol=1e-12)


@pytest.mark.parametrize(("pos", "k1sl"), RINGS, ids=IDS)
def test_the_6d_normal_form_keeps_the_live_cavity(pos: int, k1sl) -> None:
    """Gate 3: the 6D form is the synchro-betatron answer and reads the full map, as before."""
    lat = _ring(pos, k1sl)
    got, want = ac.closed_normal_form(lat), ac.normal_form(lat.one_turn_matrix())
    assert got.tunes == want.tunes
    assert np.array_equal(got.w, want.w)


@pytest.mark.parametrize("pos", MID)
def test_mode_chromaticity_refuses_a_coupled_ring_with_its_cavity_mid_ring(pos: int) -> None:
    """Gate 3: freezing the maps is not enough there; the off-momentum orbit itself is skewed
    (change B), and the scan would return -6.09 against -6.42. Refused rather than wrong."""
    with pytest.raises(NotImplementedError, match="cavity"):
        ac.mode_chromaticity(_ring(pos, SKEW))
    ac.mode_chromaticity(_ring(None, SKEW))  # the cavity last is still answered


# ---------------------------------------------------------------------- gate 4
def _scenarios() -> dict[str, dict]:
    out = {pid: PRESETS[pid] for pid in PERIODIC}
    for pid in ("electron-ring", "kicked-ring", "wiggler-ring"):
        sc = copy.deepcopy(PRESETS[pid])
        sc["elements"] = [
            {"type": "ThinSkewQuadrupole", "name": "SQ", "k1sl": 0.02},
            *sc["elements"],
        ]
        out[f"{pid}+skew"] = sc
    sc = copy.deepcopy(PRESETS["dba-cell"])
    sc["elements"].insert(1, {"type": "Solenoid", "name": "SOL", "length": 0.5, "ks": 0.2})
    out["dba-cell+solenoid"] = sc
    sc = copy.deepcopy(PRESETS["fodo-cell"])
    sc["elements"][4]["roll"] = 0.05
    out["fodo-cell+rolled-QD"] = sc
    return out


@pytest.mark.parametrize(("name", "sc"), list(_scenarios().items()))
def test_a_cavity_last_or_absent_moves_nothing(name: str, sc: dict) -> None:
    """Gate 4: every shipped ring puts its cavity last or has none. Rows 0-4 of the frozen
    product are then the full map's bit for bit, so every number here is the same float."""
    from accsim.twiss import _element_maps, _frozen_one_turn

    lat = load_scenario(sc).lattice
    M = lat.one_turn_matrix()
    assert np.array_equal(_frozen_one_turn(_element_maps(lat))[:5], M[:5])
    assert ac.momentum_compaction(lat) == _old_identity(lat)
    form = ac.closed_normal_form(lat, method="4d")
    old = ac.normal_form(M, method="4d")
    assert form.tunes == old.tunes and np.array_equal(form.w, old.w)
