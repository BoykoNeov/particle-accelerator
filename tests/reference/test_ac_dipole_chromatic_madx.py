"""Cross-check the chromatic driven response (W3) against MAD-X ``HACDIPOLE`` + ``TRACK``.

Marked ``reference``; skips when cpymad is absent. W1's MAD-X leg, with an **off-momentum**
particle at ``delta = 4e-3``. MAD-X's ``TRACK`` takes the momentum as ``pt``
(energy deviation over ``p0 c``), converted exactly from ``delta`` by
``(1 + delta)^2 = 1 + 2 pt / beta0 + pt^2``. W1's conventions carry over: MAD-X counts the
dipole's turns from 1 (``lag - nu``, ``ramp + 1``), and ``TRACK``'s drift is the exact one.

The kick convention is asserted first, in one pass at ``delta = 0.05``: ``HACDIPOLE`` adds
the full ``volt * 0.3 / pc`` to ``px`` — no ``1/(1 + delta)``, as accsim and xtrack.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from _madx import madx_session

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

GAMMA0, K1L, LD, NCELL, IDX = 10.0, 0.25, 5.0, 6, 2
NU = 0.277406248449648  # frac(Q) - 0.012 on this ring, on momentum
KICK, LAG, RAMP = 1.0e-5, 0.13, (100, 500, 900, 1300)
NT, DELTA = 1500, 4.0e-3
GATE = 1e-10  # relative to the largest |x|; measured 8.1e-13 (on-momentum: 0.77)
X0, PX0 = 1.0e-4, -2.0e-6


def _pt(delta: float) -> float:
    beta0 = math.sqrt(1.0 - 1.0 / GAMMA0**2)
    return -1.0 / beta0 + math.sqrt(1.0 / beta0**2 + 2.0 * delta + delta * delta)


def _accsim(delta: float) -> np.ndarray:
    els = []
    for _ in range(NCELL):
        els += [ThinQuadrupole(K1L), Drift(LD), ThinQuadrupole(-K1L), Drift(LD)]
    els.insert(IDX, ACDipole(kick=KICK, tune=NU, lag=LAG, plane="x", ramp=RAMP))
    lat = Lattice(els, ReferenceParticle.from_gamma(PROTON_MASS_EV, GAMMA0))
    p = Particle(x=X0, px=PX0, delta=delta)
    return Tracker(lat).track_turns(p, NT, nonlinear=True)[:, 0]


def test_madxs_kick_does_not_scale_with_momentum(tmp_path: Path) -> None:
    """One pass at ``delta = 0.05``, the sine held at 1 (``freq = 0``, ``lag = 0.25``)."""
    with madx_session() as m:
        m.chdir(str(tmp_path))
        m.input(f"beam, particle=proton, gamma={GAMMA0!r};")
        pc = float(m.eval("beam->pc"))
        m.input(
            f"acd: hacdipole, volt={KICK * pc / 0.3!r}, freq=0, lag=0.25, "
            "ramp1=0, ramp2=0, ramp3=100000, ramp4=100000;"
            "mend: marker; ring: line=(acd, mend); use, sequence=ring;"
        )
        m.input(
            "track, onetable=true, onepass=true, dump=false, aperture=false, recloss=false;"
            f"start, x=0, px=0, pt={_pt(0.05)!r}; observe, place=mend; run, turns=1; endtrack;"
        )
        px = float(np.array(m.table.trackone.px)[-1])
    assert px == pytest.approx(KICK, rel=1e-12)  # not KICK / 1.05


def _madx(workdir: Path, delta: float) -> np.ndarray:
    """``x`` at the end of MAD-X turns 1..NT; the drive in MAD-X's numbering (W1)."""
    lag = LAG - NU
    r = [ri + 1 for ri in RAMP]
    with madx_session() as m:
        m.chdir(str(workdir))
        m.input(f"beam, particle=proton, gamma={GAMMA0!r};")
        pc = float(m.eval("beam->pc"))
        cells, members = [], []
        for i in range(NCELL):
            cells.append(
                f"qf{i}: multipole, knl={{0, {K1L!r}}}; da{i}: drift, l={LD!r}; "
                f"qd{i}: multipole, knl={{0, {-K1L!r}}}; db{i}: drift, l={LD!r};"
            )
            members += [f"qf{i}", f"da{i}"] + (["acd"] if i == 0 else []) + [f"qd{i}", f"db{i}"]
        assert members.index("acd") == IDX
        m.input("\n".join(cells))
        m.input(
            f"acd: hacdipole, volt={KICK * pc / 0.3!r}, freq={NU!r}, lag={lag!r}, "
            f"ramp1={r[0]}, ramp2={r[1]}, ramp3={r[2]}, ramp4={r[3]};"
        )
        m.input(f"mend: marker; ring: line=({', '.join(members)}, mend); use, sequence=ring;")
        m.input(
            "track, onetable=true, onepass=true, dump=false, aperture=false, recloss=false;"
            f"start, x={X0!r}, px={PX0!r}, pt={_pt(delta)!r}; observe, place=mend; "
            f"run, turns={NT}; endtrack;"
        )
        t = m.table.trackone
        x, turn, s = np.array(t.x), np.array(t.turn, dtype=int), np.array(t.s)
    out = np.full(NT + 1, np.nan)
    out[0] = X0
    end = s > 0.0
    out[turn[end]] = x[end]
    assert not np.isnan(out).any()
    return out


def test_an_off_momentum_particle_is_driven_as_madx_drives_it(tmp_path: Path) -> None:
    theirs = _madx(tmp_path, DELTA)
    ours = _accsim(DELTA)
    scale = np.max(np.abs(theirs))
    assert np.max(np.abs(ours - theirs)) / scale < GATE
    # not chromaticity-blind: the on-momentum particle is another answer by tens of percent
    assert np.max(np.abs(_accsim(0.0) - theirs)) / scale > 0.1
