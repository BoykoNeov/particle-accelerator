"""The 4D optics hold the RF cavity fixed (``docs/ROADMAP.md``, gates pre-committed there).

An :class:`~accsim.elements.rfcavity.RFCavity` is the one element whose linear map changes
``delta``. In the full 6x6 one-turn map that change, made between bends, comes back into ``x``
through the dispersion downstream, so the map's transverse blocks are not symplectic and optics
matched on them do not close on themselves. :func:`~accsim.twiss.closed_twiss`,
:func:`~accsim.twiss.coupled_twiss` and :func:`~accsim.twiss.propagate_coupled_twiss` now match
on the product with ``delta``'s row frozen, the model :func:`~accsim.twiss.propagate_twiss`
already transports in (and xtrack's ``twiss(method="4d")``, ``tests/reference``).
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
from accsim.twiss import match_periodic_coupled

EDITOR = Path(__file__).resolve().parents[2] / "editor"


def _presets() -> dict[str, dict]:
    text = (EDITOR / "presets.js").read_text(encoding="utf-8")
    marker = "\nvar ACCSIM_PRESETS = "
    start = text.index(marker) + len(marker)
    return {p["id"]: p for p in json.loads(text[start : text.rindex("];") + 1])}


PRESETS = _presets()
PERIODIC = [pid for pid, p in PRESETS.items() if p.get("periodic", True) is not False]


def _ring(pos: int | None, k1sl: float | None = None) -> Lattice:
    """``electron-ring`` with its (thin) cavity at element ``pos`` (``None``: last, as shipped)."""
    sc = copy.deepcopy(PRESETS["electron-ring"])
    cav = next(e for e in sc["elements"] if e["type"] == "RFCavity")
    els = [e for e in sc["elements"] if e["type"] != "RFCavity"]
    if k1sl is not None:
        els = [*els[:5], {"type": "ThinSkewQuadrupole", "name": "SQ", "k1sl": k1sl}, *els[5:]]
    sc["elements"] = [*els, cav] if pos is None else [*els[:pos], cav, *els[pos:]]
    return load_scenario(sc).lattice


def _drop_cavity_point(points: list, pos: int | None) -> list:
    """Boundaries of a ring with a thin cavity at ``pos``, minus the cavity's own exit."""
    if pos is None:
        return points[:-1]
    return points[: pos + 1] + points[pos + 2 :]


def _old_propagate_coupled(lat: Lattice) -> list:
    """``propagate_coupled_twiss`` as it was: the local 6x6 map, nothing frozen."""
    one_turn = lat.one_turn_matrix()
    points = [match_periodic_coupled(one_turn, s=0.0)]
    transfer, s = np.eye(6), 0.0
    for elem in lat.elements:
        transfer = elem.matrix(lat.ref) @ transfer
        s += elem.length
        points.append(match_periodic_coupled(transfer @ one_turn @ np.linalg.inv(transfer), s=s))
    return points


# ---------------------------------------------------------------------- gate 1
@pytest.mark.parametrize("pos", [100, 50])
def test_the_optics_close_on_themselves_with_the_cavity_between_the_bends(pos: int) -> None:
    """Gate 1: today's closure miss was 5.9e-3 (element 100) and 2.8e-2 (element 50) in beta."""
    lat = _ring(pos)
    pts = ac.propagate_twiss(lat, ac.closed_twiss(lat))
    a, b = pts[0], pts[-1]
    assert b.beta_x == pytest.approx(a.beta_x, rel=1e-12)
    assert b.beta_y == pytest.approx(a.beta_y, rel=1e-12)
    for f in ("alpha_x", "alpha_y", "disp_x", "disp_px"):
        assert getattr(b, f) == pytest.approx(getattr(a, f), abs=1e-12), f
    # The old match (2x2 blocks of the full product) really did not close: the gate can see it.
    old = ac.propagate_twiss(lat, ac.match_periodic(lat.one_turn_matrix()))
    assert abs(old[-1].beta_x / old[0].beta_x - 1.0) > 1e-3


# ---------------------------------------------------------------------- gate 2
@pytest.mark.parametrize("pos", [100, 50])
def test_where_the_cavity_sits_does_not_move_the_optics(pos: int) -> None:
    """Gate 2: with the cavity held fixed it is a thin identity, so the ring is the same ring."""
    ref_lat, lat = _ring(None), _ring(pos)
    want = _drop_cavity_point(ac.propagate_twiss(ref_lat, ac.closed_twiss(ref_lat)), None)
    got = _drop_cavity_point(ac.propagate_twiss(lat, ac.closed_twiss(lat)), pos)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        for f in ("beta_x", "beta_y", "disp_x"):
            assert getattr(g, f) == pytest.approx(getattr(w, f), rel=1e-12), f
        for f in ("alpha_x", "alpha_y", "disp_px"):
            assert getattr(g, f) == pytest.approx(getattr(w, f), rel=1e-12, abs=1e-12), f
    np.testing.assert_allclose(ac.tunes(lat), ac.tunes(ref_lat), rtol=1e-12)
    np.testing.assert_allclose(
        ac.natural_chromaticity(lat), ac.natural_chromaticity(ref_lat), rtol=1e-12
    )


# ---------------------------------------------------------------------- gate 3
@pytest.mark.parametrize("pid", PERIODIC)
def test_a_cavity_last_or_absent_moves_nothing(pid: str) -> None:
    """Gate 3: every shipped ring puts its cavity last or has none; the answer is the same float."""
    lat = load_scenario(PRESETS[pid]).lattice
    M = lat.one_turn_matrix()
    assert ac.closed_twiss(lat) == ac.match_periodic(M)
    assert ac.coupled_twiss(lat) == match_periodic_coupled(M)


@pytest.mark.parametrize("pid", PERIODIC)
def test_a_cavity_free_ring_propagates_its_coupled_optics_as_before(pid: str) -> None:
    """Gate 3: freezing ``delta``'s row is a no-op on a ring with no cavity."""
    sc = copy.deepcopy(PRESETS[pid])
    sc["elements"] = [e for e in sc["elements"] if e["type"] != "RFCavity"]
    lat = load_scenario(sc).lattice
    assert ac.propagate_coupled_twiss(lat) == _old_propagate_coupled(lat)


# ---------------------------------------------------------------------- gate 4
@pytest.mark.parametrize("pos", [None, 100, 50])
def test_the_coupled_route_is_the_per_plane_model(pos: int | None) -> None:
    """Gate 4: on the uncoupled ring the Edwards-Teng betas are the per-plane betas.

    Before, the local 6x6 map's transverse block carried the cavity's slope through the
    dispersion, and the two parted by 1.8% even with the cavity last (the control below).
    """
    lat = _ring(pos)
    tw = ac.propagate_twiss(lat, ac.closed_twiss(lat))
    et = ac.propagate_coupled_twiss(lat)
    for t, w in zip(et, tw, strict=True):
        assert t.beta_1 == pytest.approx(w.beta_x, rel=1e-12)
        assert t.beta_2 == pytest.approx(w.beta_y, rel=1e-12)
        assert t.disp_x == pytest.approx(w.disp_x, abs=1e-12)
    old = _old_propagate_coupled(lat)
    assert max(abs(t.beta_1 / w.beta_x - 1.0) for t, w in zip(old, tw, strict=True)) > 1e-2


# ---------------------------------------------------------------------- gate 5
@pytest.mark.parametrize("pos", [100, 50])
def test_a_coupled_ring_with_its_cavity_mid_ring_closes(pos: int) -> None:
    """Gate 5: the frozen 4D block is symplectic (the full map's missed by 4.9e-3, 1.2e-2)."""
    lat = _ring(pos, k1sl=0.01)
    S = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=float)
    m4 = lat.one_turn_matrix()[:4, :4]
    assert np.abs(m4.T @ S @ m4 - S).max() > 1e-3  # the full map's block: the old input
    ct = ac.coupled_twiss(lat)
    pts = ac.propagate_coupled_twiss(lat)
    assert pts[0] == ct
    assert pts[-1].beta_1 == pytest.approx(pts[0].beta_1, rel=1e-12)
    assert pts[-1].beta_2 == pytest.approx(pts[0].beta_2, rel=1e-12)
    assert ct.gamma_c < 1.0  # really coupled
