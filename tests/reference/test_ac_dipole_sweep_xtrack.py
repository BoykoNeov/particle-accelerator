"""Transcribe W5's swept AC dipole into xtrack: the same chirped kicks, turn by turn.

Marked ``reference``; skips when xtrack or its JIT is unavailable.

**No reference code has the swept element.** xtrack's ``ACDipole`` kicks with
``sin(2 pi (freq * at_turn + lag))`` at a fixed ``freq`` and ``lag``, and MAD-X's
``HACDIPOLE`` likewise. A sweep ``2 pi (nu n + nu_dot n^2 / 2 + lag)`` can still be handed
to xtrack by resetting its ``lag`` to ``lag + nu_dot n^2 / 2`` before tracking turn ``n``.
That checks the transcription of the element — accsim's phase convention, its turn count
and the exact detuned map it drives — not the physics of the sweep, whose truth in W5 is
accsim's own exact map.
"""

from __future__ import annotations

import numpy as np
import pytest

from _w4_driven_orbits import IDX, LAG, REF, _ring
from accsim import ACDipole, X

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

K3L, KICK, NU0, RATE, NT = -2000.0, 3e-4, 0.30, 4e-5, 600
LD, NCELL, KF, KD = 5.0, 6, 0.25, -0.22
GATE = 1e-12  # relative to the largest |x| (1.8 cm); measured 4.7e-13


def test_xtrack_follows_the_swept_drive_with_its_lag_reset_each_turn() -> None:
    """Swept from 0.30 through the tune (1.344) and beyond at ``4e-5`` per turn, 600 turns,
    with the octupole: the particle's ``x`` at the start of every turn agrees."""
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
    els[IDX:IDX] = [
        xt.ACDipole(
            volt=KICK * p0c_gev / 0.3, freq=NU0, lag=LAG, ramp=[0, 0, 10**6, 10**6], plane="h"
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
    p = line.build_particles(x=[1e-3], px=[-1e-4])
    theirs = np.empty(NT)
    for n in range(NT):
        theirs[n] = float(p.x[0])
        line["acd"].lag = LAG + 0.5 * RATE * n * n  # the sweep, as a per-turn lag
        line.track(p, num_turns=1)
        assert int(p.at_turn[0]) == n + 1

    lat = _ring(ACDipole(KICK, NU0, LAG, tune_rate=RATE), K3L)
    s = np.zeros((6, 1))
    s[0, 0], s[1, 0] = 1e-3, -1e-4
    ours = np.empty(NT)
    for n in range(NT):
        ours[n] = s[X, 0]
        for e in lat.elements:
            s = e.track(s, lat.ref)
            if isinstance(e, ACDipole):
                s = s + e.drive_kick(n, lat.ref)[:, None]
    assert np.all(np.isfinite(ours))
    assert np.max(np.abs(ours - theirs)) / np.max(np.abs(ours)) < GATE
