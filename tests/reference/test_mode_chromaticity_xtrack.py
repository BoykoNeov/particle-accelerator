"""Cross-check the per-mode chromaticity against xtrack (change 2, gate 5).

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

On a coupled ring xtrack's 4D ``twiss`` reports the **mode** tunes as ``qx``/``qy`` and
differentiates them in momentum for ``dqx``/``dqy``, so those are what
:func:`~accsim.twiss.mode_chromaticity` returns. The ring is change 1's skew ring
(``tests/analytic/test_coupled_compaction.py``), built from thin multipoles and sector bends
whose models match accsim's: ``rot-kick-rot`` (the exact bend) and exact drifts, since a
chromaticity on a bendy ring is what those models decide. The uncoupled ring is the control:
there ``mode_chromaticity`` is :func:`~accsim.twiss.chromaticity`, and the agreement is the
one ``test_chromaticity_xtrack.py`` already holds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from accsim import chromaticity, mode_chromaticity

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))

import test_coupled_compaction as cc  # noqa: E402


def _twiss(k1sl: float):
    def quad(k1l: float):
        return xt.Multipole(knl=[0.0, k1l], length=0.0)

    def bend():
        b = xt.Bend(length=cc.L_BEND, angle=cc.ANGLE, k1=0.0)
        b.edge_entry_active = 0
        b.edge_exit_active = 0
        b.model = "rot-kick-rot"
        return b

    elements = []
    for i in range(cc.N_CELLS):
        cell = [
            quad(0.5 / cc.F_FOCAL),
            bend(),
            quad(-1.0 / cc.F_FOCAL),
            bend(),
            quad(0.5 / cc.F_FOCAL),
        ]
        if i == 1:
            skew = xt.Multipole(knl=[0.0, 0.0], ksl=[0.0, k1sl], length=0.0)
            cell = [*cell[:2], skew, *cell[2:]]
        elements += cell
    elements.append(quad(cc.SPLIT))
    line = xt.Line(elements=elements)
    line.particle_ref = xt.Particles(mass0=cc.MASS0, q0=1, gamma0=cc.GAMMA0)
    line.configure_drift_model("exact")
    try:
        line.build_tracker()
        return line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


def test_mode_chromaticity_matches_xtrack_on_a_coupled_ring() -> None:
    tw = _twiss(cc.K1SL)
    xi1, xi2 = mode_chromaticity(cc.skew_ring())
    assert xi1 == pytest.approx(tw.dqx, rel=1e-5)
    assert xi2 == pytest.approx(tw.dqy, rel=1e-5)
    # The comparison can tell the mode from the plane: the uncoupled xi_x is 23% away.
    assert abs(tw.dqx / chromaticity(cc.skew_ring(0.0))[0] - 1.0) > 0.2


def test_the_uncoupled_control_matches_xtrack() -> None:
    tw = _twiss(0.0)
    xi1, xi2 = mode_chromaticity(cc.skew_ring(0.0))
    assert xi1 == pytest.approx(tw.dqx, rel=1e-5)
    assert xi2 == pytest.approx(tw.dqy, rel=1e-5)
