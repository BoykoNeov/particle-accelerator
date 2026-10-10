"""Cross-check W4's exact truth against xtrack: the driven, detuned map is the same map.

Marked ``reference``; skips when xtrack or its JIT is unavailable.

W4's gates compare :func:`accsim.driven_states` with the period-40 orbits of accsim's
**own** exact driven map. This checks that map, not the cubic: the three steady states at
0.5 ``F_c`` (octupole ``k3l = -2000`` at the dipole, drive ``nu = 13/40`` below the tune)
are handed to xtrack — ``Multipole(knl=[0, 0, 0, k3l])``, ``Drift(model="exact")``,
``ACDipole`` always on — and tracked 40 turns. Each returns to itself, and turn by turn
xtrack follows accsim, at the measured floor. The middle state is unstable (its 40-turn
Jacobian has an eigenvalue ~12), so its floor is the round-off grown by that.
"""

from __future__ import annotations

import numpy as np
import pytest

from _w4_driven_orbits import (
    IDX,
    LAG,
    NU,
    REF,
    _acd,
    _exact_state,
    _periodic_orbit,
    _q_turns,
    _ring,
    _seed,
)
from accsim import ACDipole, X, driven_fold_kick, driven_states

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

K3L, Q = -2000.0, 40
LD, NCELL, KF, KD = 5.0, 6, 0.25, -0.22
GATE = 1e-12  # relative to the orbit's largest |x|; measured 8.6e-15 turn by turn,
# 1.9e-14 back at the start (the unstable middle state)


def _period_orbits() -> tuple[float, np.ndarray]:
    """``(kick, (6, 3) start states)`` of the three exact steady states at 0.5 ``F_c``."""
    fc = driven_fold_kick(_ring(_acd(1e-5), K3L))["x"]
    kick = 0.5 * fc
    lat = _ring(_acd(kick), K3L)
    starts = []
    for st in driven_states(lat)["x"]:
        z, _, _ = _periodic_orbit(lat, _seed(lat, st.amplitude, kick, NU), Q)
        assert z is not None
        u, _, _ = _exact_state(lat, st.amplitude)
        assert abs(u - st.amplitude) < 0.01 * abs(u)
        s = np.zeros(6)
        s[0], s[1] = z
        starts.append(s)
    return kick, np.array(starts).T


def test_the_exact_steady_states_are_xtracks_too() -> None:
    kick, starts = _period_orbits()
    ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1.0, gamma0=REF.gamma0)
    p0c_gev = float(ref.p0c[0]) / 1e9
    els, names = [], []
    for i in range(NCELL):
        els += [
            xt.Multipole(knl=[0, KF]),
            xt.Drift(length=LD, model="exact"),
            xt.Multipole(knl=[0, KD]),
            xt.Drift(length=LD, model="exact"),
        ]
        names += [f"qf{i}", f"da{i}", f"qd{i}", f"db{i}"]
    # accsim's order: the dipole at IDX, the octupole right after it
    els[IDX:IDX] = [
        xt.ACDipole(
            volt=kick * p0c_gev / 0.3, freq=NU, lag=LAG, ramp=[0, 0, 60000, 60000], plane="h"
        ),
        xt.Multipole(knl=[0, 0, 0, K3L]),
    ]
    names[IDX:IDX] = ["acd", "oct"]
    line = xt.Line(elements=els, element_names=names)
    line.particle_ref = ref
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    p = line.build_particles(x=starts[0], px=starts[1])
    line.track(p, num_turns=Q, turn_by_turn_monitor=True)
    theirs = np.asarray(line.record_last_track.x)  # (3, Q): x at the start of turns 0..Q-1
    end_x, end_px = np.asarray(p.x), np.asarray(p.px)

    lat = _ring(_acd(kick), K3L)
    assert isinstance(lat.elements[IDX], ACDipole)
    ours = np.empty_like(theirs)
    s = starts.copy()
    for turn in range(Q):
        ours[:, turn] = s[X]
        s = _q_turns_from(lat, s, turn)
    scale = np.max(np.abs(ours), axis=1)
    tbt = np.max(np.abs(ours - theirs), axis=1) / scale
    back = np.maximum(np.abs(end_x - starts[0]), np.abs(end_px - starts[1])) / scale
    assert np.all(tbt < GATE), tbt
    assert np.all(back < GATE), back


def _q_turns_from(lat, s: np.ndarray, turn: int) -> np.ndarray:
    """One turn, numbered ``turn``, of accsim's exact driven map (``_q_turns`` counts from 0)."""
    out = s.copy()
    for e in lat.elements:
        out = e.track(out, lat.ref)
        if isinstance(e, ACDipole):
            out = out + e.drive_kick(turn, lat.ref)[:, None]
    return out


def test_q_turns_is_the_turn_loop() -> None:
    """The helper above and the shared ``_q_turns`` are the same map (no xtrack needed)."""
    lat = _ring(_acd(1e-4), K3L)
    s = np.zeros((6, 1))
    s[0, 0], s[1, 0] = 1e-3, -2e-5
    a = s.copy()
    for turn in range(Q):
        a = _q_turns_from(lat, a, turn)
    np.testing.assert_array_equal(a, _q_turns(lat, s, Q))
