"""Cross-check the chromatic driven response (W3) against xtrack's ``ACDipole``.

Marked ``reference``; skips when xtrack or its JIT is unavailable.

W1's ring and drive (ramped up and down, ``lag = 0.13``), now with an **off-momentum**
particle at ``delta = 4e-3``: its tune sits 0.006 closer to the drive, so it is swung
about twice as hard. Turn-by-turn ``x`` for 1500 turns against accsim's exact path, with
xtrack's drift matched (``model="exact"``). The on-momentum particle is asserted to be a
different answer, so the match is not chromaticity-blind.

The kick convention is asserted first, in one pass at ``delta = 0.05``: xtrack adds the
full ``volt * 0.3 / p0c`` to ``px``, with the *reference* ``p0c`` — no ``1/(1 + delta)``,
as accsim (a field kick changes ``P_x`` whatever the momentum, and ``px = P_x / P0``).
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    PROTON_MASS_EV,
    ACDipole,
    Drift,
    Lattice,
    Particle,
    ReferenceParticle,
    ThinQuadrupole,
    Tracker,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

GAMMA0, K1L, LD, NCELL, IDX = 10.0, 0.25, 5.0, 6, 2
NU = 0.277406248449648  # frac(Q) - 0.012 on this ring, on momentum
KICK, LAG, RAMP = 2.0e-5, 0.13, (100, 500, 900, 1300)
NT, DELTA = 1500, 4.0e-3
GATE = 1e-11  # relative to the largest |x|; measured 9.4e-14 (on-momentum: 0.64)
X0, PX0 = 1.0e-4, -2.0e-6


def _accsim(delta: float) -> np.ndarray:
    els = []
    for _ in range(NCELL):
        els += [ThinQuadrupole(K1L), Drift(LD), ThinQuadrupole(-K1L), Drift(LD)]
    els.insert(IDX, ACDipole(kick=KICK, tune=NU, lag=LAG, plane="x", ramp=RAMP))
    lat = Lattice(els, ReferenceParticle.from_gamma(PROTON_MASS_EV, GAMMA0))
    p = Particle(x=X0, px=PX0, delta=delta)
    return Tracker(lat).track_turns(p, NT, nonlinear=True)[:NT, 0]


def _line(els: list, names: list) -> object:
    line = xt.Line(elements=els, element_names=names)
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1.0, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


def _p0c_gev() -> float:
    ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1.0, gamma0=GAMMA0)
    return float(ref.p0c[0]) / 1e9


def test_xtracks_kick_does_not_scale_with_momentum() -> None:
    """One pass at ``delta = 0.05``, the drive's sine held at 1: ``px`` gains ``kick``."""
    # xtrack's default ramp (0, 0, 0, 0) is "off"; its ramp is uint16, so 60000 not 1e5
    on = [0, 0, 60000, 60000]
    acd = xt.ACDipole(volt=KICK * _p0c_gev() / 0.3, freq=0.0, lag=0.25, ramp=on, plane="h")
    line = _line([acd], ["acd"])
    p = line.build_particles(x=0.0, px=0.0, delta=0.05)
    line.track(p, num_turns=1)
    assert float(p.px[0]) == pytest.approx(KICK, rel=1e-14)  # not KICK / 1.05
    ref = ReferenceParticle.from_gamma(PROTON_MASS_EV, GAMMA0)
    ours = ACDipole(kick=KICK, tune=0.0, lag=0.25).drive_kick(0, ref)[1]
    assert ours == pytest.approx(KICK, rel=1e-15)


def test_an_off_momentum_particle_is_driven_as_xtrack_drives_it() -> None:
    els, names = [], []
    for i in range(NCELL):
        els += [
            xt.Multipole(knl=[0, K1L]),
            xt.Drift(length=LD, model="exact"),
            xt.Multipole(knl=[0, -K1L]),
            xt.Drift(length=LD, model="exact"),
        ]
        names += [f"qf{i}", f"da{i}", f"qd{i}", f"db{i}"]
    acd = xt.ACDipole(volt=KICK * _p0c_gev() / 0.3, freq=NU, lag=LAG, ramp=list(RAMP), plane="h")
    els.insert(IDX, acd)
    names.insert(IDX, "acd")
    line = _line(els, names)
    p = line.build_particles(x=X0, px=PX0, delta=DELTA)
    line.track(p, num_turns=NT, turn_by_turn_monitor=True)
    theirs = np.asarray(line.record_last_track.x[0])
    ours = _accsim(DELTA)
    scale = np.max(np.abs(theirs))
    assert np.max(np.abs(ours - theirs)) / scale < GATE
    # not chromaticity-blind: the on-momentum particle is another answer by tens of percent
    assert np.max(np.abs(_accsim(0.0) - theirs)) / scale > 0.1
