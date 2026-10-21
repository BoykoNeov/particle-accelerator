"""Change B, gate 5: the 4D closed orbit with the cavity mid-ring against xtrack's 4D twiss.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

xtrack's ``twiss(method="4d")`` closes the orbit with ``delta`` frozen, the model accsim's 4D
walk now uses (``tests/analytic/test_rf_frozen_orbit.py``). The ring is
``test_rf_frozen_optics_xtrack.py``'s -- thin quadrupoles and edge-free sector bends -- with a
horizontal corrector, closed off momentum. With the cavity between the bends the agreement must
be no worse than with it last (twice that, at most), and xtrack's own orbit must not move with
the cavity; the old live walk missed by more than 1e-3.

The bends are xtrack's ``model = "bend-kick-bend"``, the exact sector bend accsim's ``Dipole``
is. xtrack's default for this bend (``rot-kick-rot``) is not, and on this steered off-momentum
orbit it parts from accsim by ``6.1e-10`` (``1.3e-6`` of the orbit) at every cavity position;
with the exact model the shared floor is ``2.2e-16`` (measured 2026-10-22).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import accsim as ac
from accsim import Corrector, Dipole, Lattice, RFCavity, ThinQuadrupole, ThinSkewQuadrupole
from accsim import orbit as orbit_module

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0, GAMMA0 = 938.27208816e6, 5.0
F_FOCAL, L_BEND, ANGLE, N_CELLS = 2.5, 1.0, 0.15, 6
VOLTAGE, HARMONIC = 2.0e6, 10
KICK, DELTA0 = 1e-4, 1e-4
CORRECTOR_AT = 3  # after the first focusing quadrupole's bend, cell 1


def _pair(pos: int | None, k1sl: float = 0.0, kick: float = KICK, delta0: float = DELTA0):
    """``(accsim lattice, xtrack 4D twiss at delta0)``, cavity at ``pos`` (``None``: last).

    ``k1sl`` adds ``test_rf_frozen_optics_xtrack.py``'s skew quadrupole (after the first bend
    of cell 2) in both codes."""
    ref = ac.ReferenceParticle.from_gamma(MASS0, GAMMA0)
    acc, xtk = [], []
    for i in range(N_CELLS):
        for k in (0.5 / F_FOCAL, None, -1.0 / F_FOCAL, None, 0.5 / F_FOCAL):
            if k is None:
                acc.append(Dipole(L_BEND, ANGLE))
                b = xt.Bend(length=L_BEND, angle=ANGLE, k1=0.0)
                b.model = "bend-kick-bend"  # the exact sector bend, as accsim's
                b.edge_entry_active = 0
                b.edge_exit_active = 0
                xtk.append(b)
            else:
                acc.append(ThinQuadrupole(k))
                xtk.append(xt.Multipole(knl=[0.0, k], length=0.0))
        if i == 1 and k1sl != 0.0:
            acc.insert(len(acc) - 3, ThinSkewQuadrupole(k1sl))
            xtk.insert(len(xtk) - 3, xt.Multipole(knl=[0.0, 0.0], ksl=[0.0, k1sl], length=0.0))
    acc.append(ThinQuadrupole(-0.03))
    xtk.append(xt.Multipole(knl=[0.0, -0.03], length=0.0))
    # accsim Corrector(kick_x=+k) == xt.Multipole(knl=[-k]) (test_feeddown_xtrack.py).
    acc.insert(CORRECTOR_AT, Corrector(kick_x=kick))
    xtk.insert(CORRECTOR_AT, xt.Multipole(knl=[-kick]))
    cav = RFCavity.from_harmonic(VOLTAGE, HARMONIC, Lattice(acc, ref).length, ref, phi_s=math.pi)
    at = len(acc) if pos is None else pos
    acc.insert(at, cav)
    xtk.insert(at, xt.Cavity(voltage=VOLTAGE, frequency=cav.frequency, phase=math.pi))
    line = xt.Line(elements=xtk)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        return Lattice(acc, ref), line.twiss(method="4d", delta0=delta0)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


def _miss(lat: Lattice, tw) -> float:
    """accsim's orbit along the ring against xtrack's, in ``x`` and ``px``, max abs."""
    along = np.array(ac.propagate_orbit_nonlinear(lat, delta=DELTA0))
    return float(max(np.abs(along[:, 0] - tw.x).max(), np.abs(along[:, 1] - tw.px).max()))


@pytest.fixture(scope="module")
def last():
    return _pair(None)


@pytest.mark.parametrize("pos", [8, 17])
def test_the_off_momentum_orbit_mid_ring_matches_xtrack_4d(last, pos: int) -> None:
    lat_last, tw_last = last
    lat, tw = _pair(pos)
    floor = _miss(lat_last, tw_last)
    assert _miss(lat, tw) <= 2.0 * floor
    # xtrack's 4D orbit is the cavity-last one: the cavity is frozen out of it.
    assert abs(tw.x[0] - tw_last.x[0]) < 1e-15
    assert abs(tw.px[0] - tw_last.px[0]) < 1e-15


@pytest.mark.parametrize("pos", [8, 17])
def test_the_live_walk_missed_xtrack(monkeypatch: pytest.MonkeyPatch, last, pos: int) -> None:
    """Control: carrying ``delta`` through the cavity misses by more than 1e-3 of the orbit."""
    lat, tw = _pair(pos)
    monkeypatch.setattr(orbit_module, "_step", lambda elem, state, ref: elem.track(state, ref))
    assert _miss(lat, tw) > 1e-3 * np.abs(tw.x).max()


# ---------------------------------------------------------------- step 4, gate 2
SKEW_K1SL = 0.05  # test_rf_frozen_optics_xtrack.py's coupled ring


def _chroma_miss(lat: Lattice, tw) -> float:
    """accsim's per-mode chromaticity against xtrack's 4D dqx / dqy, max relative."""
    got = np.array(ac.mode_chromaticity(lat))
    want = np.array([tw.dqx, tw.dqy])
    return float(np.max(np.abs(got / want - 1.0)))


@pytest.fixture(scope="module")
def skew_last():
    return _pair(None, SKEW_K1SL, kick=0.0, delta0=0.0)


@pytest.mark.parametrize("pos", [8, 17])
def test_the_mode_chromaticity_mid_ring_matches_xtrack_4d(skew_last, pos: int) -> None:
    """Change B step 4: no longer refused, and as close to xtrack as with the cavity last."""
    lat_last, tw_last = skew_last
    lat, tw = _pair(pos, SKEW_K1SL, kick=0.0, delta0=0.0)
    assert _chroma_miss(lat, tw) <= 2.0 * _chroma_miss(lat_last, tw_last)
    assert tw.dqx == pytest.approx(tw_last.dqx, rel=1e-6)
    assert tw.dqy == pytest.approx(tw_last.dqy, rel=1e-6)
