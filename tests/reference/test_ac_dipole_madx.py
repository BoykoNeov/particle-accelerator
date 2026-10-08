"""Cross-check the AC dipole's driven tracking against MAD-X ``HACDIPOLE`` + ``TRACK`` (W1).

Marked ``reference``; skips when cpymad is absent. The second arbiter, independent of
xtrack, and two of its conventions were found by running it (2026-10-08):

- **MAD-X counts the AC dipole's turns from 1.** Pass ``n`` of accsim (and xtrack) is
  MAD-X turn ``n + 1``, for the sine's phase *and* for the ramp. The same physical drive
  is therefore ``lag_madx = lag - nu`` and ``ramp_madx = ramp + 1``. Without that shift the
  two miss by order one (asserted).
- **``TRACK``'s drift is the exact one.** With the drive off, MAD-X's free motion matches
  accsim's exact ``track()`` path to ``3e-12`` over 1500 turns and its *linear* walk only to
  ``7e-6``. The ``1.1e-6`` residual quadratic in amplitude that the 2026-10-06 filter run
  saw between MAD-X and xtrack (whose default drift is linear) is this drift, localised.

``lag`` is in turns here too, and ``volt`` carries the same ``0.3 / pc[GeV]`` scale.
"""

from __future__ import annotations

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
NU = 0.277406248449648  # frac(Q) - 0.012 on this ring
KICK, LAG, RAMP = 1.0e-5, 0.13, (100, 500, 900, 1300)
NT = 1500
GATE = 1e-10  # relative to the largest |x|; measured 1.0e-12 against the exact path
X0, PX0 = 1.0e-4, -2.0e-6


def _accsim(nonlinear: bool, kick: float = KICK) -> np.ndarray:
    els = []
    for _ in range(NCELL):
        els += [ThinQuadrupole(K1L), Drift(LD), ThinQuadrupole(-K1L), Drift(LD)]
    els.insert(IDX, ACDipole(kick=kick, tune=NU, lag=LAG, plane="x", ramp=RAMP))
    lat = Lattice(els, ReferenceParticle.from_gamma(PROTON_MASS_EV, GAMMA0))
    return Tracker(lat).track_turns(Particle(x=X0, px=PX0), NT, nonlinear=nonlinear)[:, 0]


def _madx(workdir: Path, turn_shift: int, kick: float = KICK) -> np.ndarray:
    """``x`` at the end of MAD-X turns 1..NT, the drive given in MAD-X's own numbering.

    Run in ``workdir``: ``TRACK`` writes a ``checkpoint_restart.dat`` where it stands.
    """
    lag = LAG - NU * turn_shift
    r = [ri + turn_shift for ri in RAMP]
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
            f"acd: hacdipole, volt={kick * pc / 0.3!r}, freq={NU!r}, lag={lag!r}, "
            f"ramp1={r[0]}, ramp2={r[1]}, ramp3={r[2]}, ramp4={r[3]};"
        )
        m.input(f"mend: marker; ring: line=({', '.join(members)}, mend); use, sequence=ring;")
        m.input(
            "track, onetable=true, onepass=true, dump=false, aperture=false, recloss=false;"
            f"start, x={X0!r}, px={PX0!r}; observe, place=mend; run, turns={NT}; endtrack;"
        )
        t = m.table.trackone
        x, turn, s = np.array(t.x), np.array(t.turn, dtype=int), np.array(t.s)
    # rows: the start (s = 0, turn 0), then the end of every turn (s = 60), recorded twice
    out = np.full(NT + 1, np.nan)
    out[0] = X0
    end = s > 0.0
    out[turn[end]] = x[end]
    assert not np.isnan(out).any()
    return out


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_the_driven_motion_matches_madx_track(tmp_path: Path) -> None:
    exact = _accsim(nonlinear=True)
    madx = _madx(tmp_path, turn_shift=1)
    assert _rel(madx, exact) < GATE
    # TRACK's drift is the exact one: the linear walk is not what it agrees with
    assert _rel(madx, _accsim(nonlinear=False)) > 1e3 * GATE
    # the one-turn shift is load-bearing: in accsim's numbering the drive is another one
    assert _rel(_madx(tmp_path, turn_shift=0), exact) > 0.5


def test_with_the_drive_off_madx_is_the_exact_drift(tmp_path: Path) -> None:
    """The localisation of the filter run's residual: no drive, still exact vs linear."""
    madx = _madx(tmp_path, turn_shift=1, kick=0.0)
    assert _rel(madx, _accsim(nonlinear=True, kick=0.0)) < GATE
    assert _rel(madx, _accsim(nonlinear=False, kick=0.0)) > 1e3 * GATE
