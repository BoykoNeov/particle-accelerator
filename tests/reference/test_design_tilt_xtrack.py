r"""V1 — the design tilt against xtrack's ``rot_s_rad``: survey, optics, and the map's sense.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

The second arbiter beside ``test_design_tilt_madx.py``, and the one that sees the **map**
directly. MAD-X pins the survey and the optics; xtrack additionally hands back the one-turn
matrix, which is where the sense of the phase-space rotation is pinned on its own:

- the F2 ring (every element turned by ``t = 0.1`` with its frame) has the one-turn map
  ``R(-t) M_flat R(+t)``. accsim's matrix lands on xtrack's to the finite-difference floor
  the probe measured (``1.6e-9``), while the same conjugation with the opposite sign misses
  by order one — so this gate is sharp in the sign where the tune invariance is not;
- the 3D survey, all six columns, on F3 and on the dogleg ring;
- the dogleg's signed ``D_y``/``D_py`` and its momentum compaction.

**Cost.** Each ``xt.Line`` build JIT-compiles a fresh C kernel (12-70 s apiece on this box —
``docs/CONVENTIONS.md`` -> *Test-suite cost*), so the three lines are module-scoped fixtures
and every claim is read off them. F3 is never given a tracker: a survey needs no map.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))

from test_design_tilt import (  # noqa: E402
    ANG,
    ELECTRON_MASS_EV,
    F3,
    NCELL,
    VA,
    dogleg_ring,
    f3_line,
    flat_ring,
    rolled_ring,
)

from accsim.elements.alignment import s_rotation  # noqa: E402
from accsim.geometry import survey  # noqa: E402
from accsim.twiss import closed_twiss, momentum_compaction, propagate_twiss, tunes  # noqa: E402

PI2 = math.pi / 2
T_ROLL = 0.1
COLUMNS = ("X", "Y", "Z", "theta", "phi", "psi")
_T4 = [0, 1, 2, 3]


def _line(elements, names):
    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(mass0=ELECTRON_MASS_EV, q0=-1.0, energy0=3.0e9)
    return line


def _cell(i: int, t: float = 0.0):
    def q(k1):
        return xt.Quadrupole(length=0.5, k1=k1, rot_s_rad=t)

    def b():
        return xt.Bend(length=1.5, angle=ANG, k0_from_h=True, rot_s_rad=t)

    def d():
        return xt.Drift(length=0.6)

    parts = (
        (f"qf{i}", q(0.30)), (f"d{i}a", d()), (f"b{i}a", b()), (f"d{i}b", d()),
        (f"qd{i}", q(-0.30)), (f"d{i}c", d()), (f"b{i}b", b()), (f"d{i}d", d()),
    )  # fmt: skip
    return [p[1] for p in parts], [p[0] for p in parts]


def _ring(t: float = 0.0, dogleg: bool = False):
    els, nms = [], []
    for i in range(NCELL):
        e, n = _cell(i, t)
        els += e
        nms += n
        if dogleg and i == 0:
            els += [
                xt.Bend(length=0.3, angle=VA, k0_from_h=True, rot_s_rad=PI2),
                xt.Drift(length=1.0),
                xt.Bend(length=0.3, angle=-VA, k0_from_h=True, rot_s_rad=PI2),
            ]
            nms += ["vb1", "dd", "vb2"]
    return _line(els, nms)


@pytest.fixture(scope="module")
def f3_survey():
    els, nms = [], []
    for i, (length, angle, tilt) in enumerate(F3):
        els += [xt.Bend(length=length, angle=angle, k0_from_h=True, rot_s_rad=tilt)]
        els += [xt.Drift(length=0.8)]
        nms += [f"b{i}", f"d{i}"]
    sv = _line(els, nms).survey()
    return {c: np.asarray(sv[c], dtype=float) for c in COLUMNS}


@pytest.fixture(scope="module")
def dogleg():
    line = _ring(dogleg=True)
    line.build_tracker()
    sv = line.survey()
    return {c: np.asarray(sv[c], dtype=float) for c in COLUMNS}, line.twiss(method="4d")


@pytest.fixture(scope="module")
def rolled_one_turn():
    line = _ring(t=T_ROLL)
    line.build_tracker()
    res = line.get_R_matrix(particle_on_co=line.build_particles(x=0.0))
    return np.asarray(res["R_matrix"])[np.ix_(_T4, _T4)]


def _ours(table) -> dict[str, np.ndarray]:
    return {
        "X": table.X, "Y": table.Y, "Z": table.Z,
        "theta": table.theta, "phi": table.phi, "psi": table.psi,
    }  # fmt: skip


@pytest.mark.parametrize("column", COLUMNS)
def test_the_f3_survey_matches_xtrack_in_every_column(f3_survey, column: str) -> None:
    ours = _ours(survey(f3_line()))[column]
    assert ours.shape == f3_survey[column].shape
    assert np.allclose(ours, f3_survey[column], rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("column", COLUMNS)
def test_the_dogleg_survey_matches_xtrack_in_every_column(dogleg, column: str) -> None:
    surv, _ = dogleg
    assert np.allclose(_ours(survey(dogleg_ring()))[column], surv[column], rtol=0.0, atol=1e-12)


def test_the_dogleg_optics_match_xtrack(dogleg) -> None:
    """Tunes, betas and the **signed** dispersion in both planes, boundary for boundary —
    xtrack's rows are the ``N + 1`` boundaries, as accsim's are.

    Measured 2026-10-07: ``D_x`` to ``2.9e-9`` (of ``2.17`` m), ``D_y`` to ``1.4e-10`` (of
    ``2.4e-2`` m), betas to ``4.3e-9`` relative — xtrack's finite-difference floor, which the
    flat ring shows as well; the tilt adds nothing measurable to it."""
    _, tw = dogleg
    lat = dogleg_ring()
    qx, qy = tunes(lat)
    assert qx == pytest.approx(float(tw.qx), abs=1e-9)
    assert qy == pytest.approx(float(tw.qy), abs=1e-9)
    pts = propagate_twiss(lat, closed_twiss(lat))
    pairs = {
        "betx": [p.beta_x for p in pts], "bety": [p.beta_y for p in pts],
        "dx": [p.disp_x for p in pts], "dpx": [p.disp_px for p in pts],
        "dy": [p.disp_y for p in pts], "dpy": [p.disp_py for p in pts],
    }  # fmt: skip
    for col, ours in pairs.items():
        theirs = np.asarray(tw[col], dtype=float)
        if col.startswith("bet"):
            assert np.allclose(ours, theirs, rtol=1e-8, atol=0.0), col
        else:
            assert np.allclose(ours, theirs, rtol=0.0, atol=1e-8), col
    assert np.max(np.abs(np.asarray(tw["dy"]))) > 1e-2


def test_the_dogleg_momentum_compaction_matches_xtrack(dogleg) -> None:
    _, tw = dogleg
    assert momentum_compaction(dogleg_ring()) == pytest.approx(
        float(tw.momentum_compaction_factor), rel=1e-8
    )


def test_the_one_turn_map_is_conjugated_in_xtracks_sense(rolled_one_turn) -> None:
    """The sign gate on the **map**: accsim's rolled-ring matrix *is* xtrack's, and the
    opposite conjugation is not — by order one.

    The floor is xtrack's finite-difference ``R_matrix``, not the tilt: measured 2026-10-07
    at ``1.03e-8`` on this ring and ``8.8e-9`` on the **flat** ring, against ``> 0.1`` for
    the wrong sense."""
    ours = rolled_ring(T_ROLL).one_turn_matrix()[np.ix_(_T4, _T4)]
    assert np.allclose(ours, rolled_one_turn, rtol=0.0, atol=3e-8)
    flat = flat_ring().one_turn_matrix()
    wrong = (s_rotation(T_ROLL) @ flat @ s_rotation(-T_ROLL))[np.ix_(_T4, _T4)]
    assert np.max(np.abs(wrong - rolled_one_turn)) > 0.1
