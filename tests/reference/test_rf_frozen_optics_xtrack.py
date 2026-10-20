"""The 4D optics with the cavity mid-ring against xtrack's ``twiss(method="4d")`` (gate 6).

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

xtrack's 4D twiss holds ``delta`` fixed, the model ``closed_twiss`` and
``propagate_coupled_twiss`` now match on (``tests/analytic/test_rf_frozen_optics.py``). The ring
is thin quadrupoles and edge-free sector bends in both codes, so the two maps agree to
round-off, with a 2 MV cavity between the bends. The agreement (``5.6e-9`` in beta) is the same
number as with the cavity last, xtrack's own floor; the old match, on the 2x2 blocks of the full
6x6 product, misses by 6.6% and 9.3%.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import accsim as ac
from accsim import Dipole, Lattice, RFCavity, ThinQuadrupole, ThinSkewQuadrupole

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0, GAMMA0 = 938.27208816e6, 5.0
F_FOCAL, L_BEND, ANGLE, N_CELLS = 2.5, 1.0, 0.15, 6
VOLTAGE, HARMONIC = 2.0e6, 10
EMIT_1, EMIT_2, SIGMA_DELTA = 1e-8, 1e-9, 1e-3


def _pair(k1sl: float, pos: int):
    """``(accsim lattice, xtrack 4D twiss)``: the skew ring of test_coupled_compaction, cavity at ``pos``."""
    ref = ac.ReferenceParticle.from_gamma(MASS0, GAMMA0)
    acc, xtk = [], []
    for i in range(N_CELLS):
        for k in (0.5 / F_FOCAL, None, -1.0 / F_FOCAL, None, 0.5 / F_FOCAL):
            if k is None:
                acc.append(Dipole(L_BEND, ANGLE))
                b = xt.Bend(length=L_BEND, angle=ANGLE, k1=0.0)
                b.edge_entry_active = 0
                b.edge_exit_active = 0
                xtk.append(b)
            else:
                acc.append(ThinQuadrupole(k))
                xtk.append(xt.Multipole(knl=[0.0, k], length=0.0))
        if i == 1:  # after the first bend of cell 2
            acc.insert(len(acc) - 3, ThinSkewQuadrupole(k1sl))
            xtk.insert(len(xtk) - 3, xt.Multipole(knl=[0.0, 0.0], ksl=[0.0, k1sl], length=0.0))
    acc.append(ThinQuadrupole(-0.03))
    xtk.append(xt.Multipole(knl=[0.0, -0.03], length=0.0))
    # eta > 0 (above transition), so phi_s = pi; q0 = +1 makes xtrack's phase pi the same.
    cav = RFCavity.from_harmonic(VOLTAGE, HARMONIC, Lattice(acc, ref).length, ref, phi_s=math.pi)
    acc.insert(pos, cav)
    xtk.insert(pos, xt.Cavity(voltage=VOLTAGE, frequency=cav.frequency, phase=math.pi))
    line = xt.Line(elements=xtk)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        return Lattice(acc, ref), line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("pos", [8, 17])
def test_per_plane_optics_with_the_cavity_mid_ring_match_xtrack_4d(pos: int) -> None:
    lat, tw = _pair(0.0, pos)
    pts = ac.propagate_twiss(lat, ac.closed_twiss(lat))
    np.testing.assert_allclose([p.beta_x for p in pts], tw.betx, rtol=2e-8)
    np.testing.assert_allclose([p.beta_y for p in pts], tw.bety, rtol=2e-8)
    np.testing.assert_allclose([p.disp_x for p in pts], tw.dx, rtol=0.0, atol=1e-8)
    # Control: the old match (2x2 blocks of the full 6x6 product) is far off, so the gate sees it.
    old = ac.propagate_twiss(lat, ac.match_periodic(lat.one_turn_matrix()))
    assert np.abs(np.array([p.beta_x for p in old]) / tw.betx - 1.0).max() > 5e-2


@pytest.mark.parametrize("pos", [8, 17])
def test_coupled_beam_size_with_the_cavity_mid_ring_matches_xtrack_4d(pos: int) -> None:
    lat, tw = _pair(0.05, pos)
    pts = ac.propagate_coupled_twiss(lat)
    assert min(p.gamma_c for p in pts) < 0.999  # really coupled
    sx, sy, _tilt = ac.coupled_beam_sigma(pts, EMIT_1, EMIT_2, SIGMA_DELTA)
    want_x = np.sqrt(EMIT_1 * tw.betx1 + EMIT_2 * tw.betx2 + (tw.dx * SIGMA_DELTA) ** 2)
    want_y = np.sqrt(EMIT_1 * tw.bety1 + EMIT_2 * tw.bety2 + (tw.dy * SIGMA_DELTA) ** 2)
    np.testing.assert_allclose(sx, want_x, rtol=2e-8)
    np.testing.assert_allclose(sy, want_y, rtol=2e-8)
    np.testing.assert_allclose([p.disp_x for p in pts], tw.dx, rtol=0.0, atol=1e-8)
