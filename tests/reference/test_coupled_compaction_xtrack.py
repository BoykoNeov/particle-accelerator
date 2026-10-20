"""Cross-check momentum compaction on a design-coupled ring against xtrack (gate 4).

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

The ring is ``tests/analytic/test_coupled_compaction.py``'s skew fixture, built in both
codes from thin lenses and edge-free sector bends, so the two maps agree to round-off and
any gap is in the compaction itself. The thin skew quadrupole is
``ThinSkewQuadrupole(k1sl) == xt.Multipole(ksl=[0, k1sl])`` (``px += k1sl y``, ``py +=
k1sl x`` in both). xtrack's ``momentum_compaction_factor`` is its own tracking-based
number, so this is the external check that the coupled dispersion and the vertical pair
are right: dropping the pair would miss by ``4.5e-3``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from accsim import momentum_compaction, slip_factor

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))

import test_coupled_compaction as cc  # noqa: E402


@pytest.fixture(scope="module")
def twiss():
    """xtrack's 4D twiss of the skew ring (module-scoped: one JIT build)."""

    def quad(k1l: float):
        return xt.Multipole(knl=[0.0, k1l], length=0.0)

    def bend():
        b = xt.Bend(length=cc.L_BEND, angle=cc.ANGLE, k1=0.0)
        b.edge_entry_active = 0
        b.edge_exit_active = 0
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
            cell = (
                cell[:2] + [xt.Multipole(knl=[0.0, 0.0], ksl=[0.0, cc.K1SL], length=0.0)] + cell[2:]
            )
        elements += cell
    elements.append(quad(cc.SPLIT))
    line = xt.Line(elements=elements)
    line.particle_ref = xt.Particles(mass0=cc.MASS0, q0=1, gamma0=cc.GAMMA0)
    try:
        line.build_tracker()
        return line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


def test_coupled_momentum_compaction_matches_xtrack(twiss) -> None:
    lat = cc.skew_ring()
    assert momentum_compaction(lat) == pytest.approx(twiss.momentum_compaction_factor, rel=1e-7)
    assert slip_factor(lat) == pytest.approx(twiss.slip_factor, rel=1e-7)
    # The gate can see the vertical pair: without it accsim would be 4.5e-3 away.
    assert abs(cc._vertical_share(lat)) / momentum_compaction(lat) > 4e-3


def test_coupled_dispersion_matches_xtrack(twiss) -> None:
    """The 4D dispersion the compaction is built on, including the leaked D_y."""
    want = np.array([twiss.dx[0], twiss.dpx[0], twiss.dy[0], twiss.dpy[0]])
    np.testing.assert_allclose(cc._dispersion(cc.skew_ring()), want, rtol=1e-7, atol=1e-10)
    assert abs(want[2]) > 0.2
