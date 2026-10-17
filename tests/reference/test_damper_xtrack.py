"""Cross-check the damper (W6 part a) against xtrack: the same element, transcribed.

Marked ``reference``; skips when xtrack or its JIT is unavailable.

xtrack has the damper's map as a ``LineSegmentMap`` with ``damping_rate_px``: its source
fills ``damping_factors[1, 1] = 1 - damping_rate_px`` and applies ``px *= factor``
(``xtrack/beam_elements/elements.py``, ``linesegmentmap.h``). With zero length, zero
phase advance and equal betas at both ends its linear part is the identity, so it is
the damping alone. The ring is W4's split-tune FODO with the octupole and the AC dipole,
exact drifts in both codes, the damper right after the dipole and, in a second case,
part of a cell away. A particle started off the steady state is tracked 600 turns (two
damping times): turn by turn xtrack follows accsim. This checks the transcription of
the element, not the physics; the physics is gated in ``tests/analytic/test_damper.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from _w4_driven_orbits import IDX, REF, _acd, _ring
from accsim import ACDipole, Damper, Lattice, Particle, Tracker, X

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

K3L, NU, KICK = -2000.0, 13 / 40, 2e-5
GAMMA = 0.003
RATE = 1.0 - math.exp(-2.0 * GAMMA)
LD, NCELL, KF, KD = 5.0, 6, 0.25, -0.22
N_TURNS = 600
GATE = 1e-12  # relative to the largest |x|; measured 1.2e-14 (after the dipole), 9.6e-15 (away)


def _accsim_ring(damper_at: int) -> Lattice:
    els = list(_ring(_acd(KICK, NU), K3L).elements)  # [..., acd (IDX), oct, ...]
    els.insert(damper_at, Damper(RATE))
    return Lattice(els, REF)


def _xtrack_line(lat: Lattice) -> object:
    ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1.0, gamma0=REF.gamma0)
    p0c_gev = float(ref.p0c[0]) / 1e9
    els, names = [], []
    for i, e in enumerate(lat.elements):
        kind = type(e).__name__
        if kind == "ThinQuadrupole":
            els.append(xt.Multipole(knl=[0, e.k1l]))
        elif kind == "Drift":
            els.append(xt.Drift(length=e.length, model="exact"))
        elif kind == "ThinOctupole":
            els.append(xt.Multipole(knl=[0, 0, 0, e.k3l]))
        elif kind == "ACDipole":
            els.append(
                xt.ACDipole(
                    volt=e.amplitude * p0c_gev / 0.3,
                    freq=e.tune,
                    lag=e.lag,
                    ramp=[0, 0, 60000, 60000],
                    plane="h",
                )
            )
        elif kind == "Damper":
            els.append(xt.LineSegmentMap(length=0.0, qx=0.0, qy=0.0, damping_rate_px=e.rate))
        else:  # pragma: no cover - the fixture has nothing else
            raise AssertionError(kind)
        names.append(f"e{i}")
    line = xt.Line(elements=els, element_names=names)
    line.particle_ref = ref
    return line


@pytest.mark.parametrize("damper_at", [IDX + 1, IDX + 5], ids=["after_dipole", "away"])
def test_xtrack_damps_the_driven_ring_the_same_way(damper_at: int) -> None:
    lat = _accsim_ring(damper_at)
    assert isinstance(lat.elements[IDX], ACDipole)
    line = _xtrack_line(lat)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    x0, px0 = 2e-3, -1e-4  # off the steady state: a free oscillation damps toward it
    p = line.build_particles(x=[x0], px=[px0])
    line.track(p, num_turns=N_TURNS, turn_by_turn_monitor=True)
    theirs = np.asarray(line.record_last_track.x)[0]  # x at the start of turns 0..N-1

    ours = Tracker(lat).track_turns(Particle(x=x0, px=px0), N_TURNS, nonlinear=True)[:N_TURNS, X]
    scale = np.max(np.abs(ours))
    miss = np.max(np.abs(ours - theirs)) / scale
    assert miss < GATE
    # and the run is long enough to see the damping: the free swing has mostly gone
    assert abs(ours[-50:]).max() < 0.5 * abs(ours[:50]).max()
