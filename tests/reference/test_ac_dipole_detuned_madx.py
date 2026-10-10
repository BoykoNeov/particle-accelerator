"""Cross-check W4's exact truth against MAD-X ``HACDIPOLE`` + thin ``MULTIPOLE`` + ``TRACK``.

Marked ``reference``; skips when cpymad is absent. The xtrack leg's three steady states
(0.5 ``F_c``, octupole ``k3l = -2000`` at the dipole, ``nu = 13/40``), handed to MAD-X and
tracked 40 turns. W1's conventions carry over: MAD-X counts the dipole's turns from 1, so
the same drive is ``lag - nu`` (always on here, so the ramp needs no shift), and ``TRACK``'s
drift is the exact one. Like the xtrack leg it checks the exact map the analytic gates
compare against, not the cubic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _madx import madx_session

from _w4_driven_orbits import (
    IDX,
    LAG,
    NU,
    REF,
    _acd,
    _periodic_orbit,
    _ring,
    _seed,
)
from accsim import ACDipole, driven_fold_kick, driven_states

pytestmark = pytest.mark.reference

K3L, Q = -2000.0, 40
LD, NCELL, KF, KD = 5.0, 6, 0.25, -0.22
GATE = 1e-11  # relative to the orbit's largest |x|; measured 3.7e-14 on the stable states,
# 2.4e-13 on the unstable middle one (its 40-turn growth ~12 carries the round-off)


def _accsim(kick: float, starts: np.ndarray) -> np.ndarray:
    """``(Q + 1, 2, n)``: ``(x, px)`` at the start of turns 0..Q, accsim's exact driven map."""
    lat = _ring(_acd(kick), K3L)
    s = np.zeros((6, starts.shape[1]))
    s[:2] = starts
    out = [s[:2].copy()]
    for turn in range(Q):
        for e in lat.elements:
            s = e.track(s, lat.ref)
            if isinstance(e, ACDipole):
                s = s + e.drive_kick(turn, lat.ref)[:, None]
        out.append(s[:2].copy())
    return np.array(out)


def test_the_exact_steady_states_are_madxs_too(tmp_path: Path) -> None:
    fc = driven_fold_kick(_ring(_acd(1e-5), K3L))["x"]
    kick = 0.5 * fc
    lat = _ring(_acd(kick), K3L)
    starts = []
    for st in driven_states(lat)["x"]:
        z, _, _ = _periodic_orbit(lat, _seed(lat, st.amplitude, kick, NU), Q)
        assert z is not None
        starts.append(z)
    starts = np.array(starts).T  # (2, 3)

    with madx_session() as m:
        m.chdir(str(tmp_path))
        m.input(f"beam, particle=proton, gamma={REF.gamma0!r};")
        pc = float(m.eval("beam->pc"))
        cells, members = [], []
        for i in range(NCELL):
            cells.append(
                f"qf{i}: multipole, knl={{0, {KF!r}}}; da{i}: drift, l={LD!r}; "
                f"qd{i}: multipole, knl={{0, {KD!r}}}; db{i}: drift, l={LD!r};"
            )
            members += [f"qf{i}", f"da{i}"] + (["acd", "oct"] if i == 0 else [])
            members += [f"qd{i}", f"db{i}"]
        assert members.index("acd") == IDX
        m.input("\n".join(cells))
        m.input(
            f"acd: hacdipole, volt={kick * pc / 0.3!r}, freq={NU!r}, lag={LAG - NU!r}, "
            "ramp1=0, ramp2=0, ramp3=100000, ramp4=100000;"
            f"oct: multipole, knl={{0, 0, 0, {K3L!r}}};"
        )
        m.input(f"mend: marker; ring: line=({', '.join(members)}, mend); use, sequence=ring;")
        track = "track, onetable=true, onepass=true, dump=false, aperture=false, recloss=false;"
        for k in range(starts.shape[1]):
            track += f"start, x={float(starts[0, k])!r}, px={float(starts[1, k])!r};"
        m.input(track + f"observe, place=mend; run, turns={Q}; endtrack;")
        t = m.table.trackone
        num = np.array(t.number, dtype=int)
        turn = np.array(t.turn, dtype=int)
        x, px, s = np.array(t.x), np.array(t.px), np.array(t.s)

    ours = _accsim(kick, starts)
    theirs = np.full_like(ours, np.nan)
    theirs[0] = starts
    end = s > 0.0
    theirs[turn[end], 0, num[end] - 1] = x[end]
    theirs[turn[end], 1, num[end] - 1] = px[end]
    assert not np.isnan(theirs).any()
    scale = np.max(np.abs(ours[:, 0, :]), axis=0)
    tbt = np.max(np.abs(ours[:, 0, :] - theirs[:, 0, :]), axis=0) / scale
    back = np.max(np.abs(theirs[Q] - starts), axis=0) / scale
    assert np.all(tbt < GATE), tbt
    assert np.all(back < GATE), back
