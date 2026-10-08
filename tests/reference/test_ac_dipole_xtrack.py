"""Cross-check the AC dipole's driven tracking against xtrack's ``ACDipole`` (W1).

Marked ``reference``; skips when xtrack or its JIT is unavailable.

The same thin-lens FODO ring in both codes, the drive ramped up and down (xtrack's
trapezoid) with a non-zero ``lag``, a particle started off the steady state so the free
and driven motion mix: turn-by-turn ``x, px`` for 1500 turns. Three of xtrack's
conventions had to be matched first, each found by running it (2026-10-06 filter run,
``W:\\temp\\claude\\accsim-filter-2026-10-06``):

- ``lag`` is in **turns** — the docstring says radians, the C multiplies by ``2 pi``;
- the kick is ``volt * 0.3 / p0c[GeV]``, with ``0.3`` exactly, not ``c``;
- ``at_turn`` counts from 0, as accsim's turn index does.

And the drift model: xtrack's default drift is linear in the angle, accsim's ``track()``
is the exact drift. So the linear walk is held against ``model="expanded"`` and the
exact path against ``model="exact"`` — and the crossed pair is asserted to differ, so the
match is not a drift-blind one.
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
NU = 0.277406248449648  # frac(Q) - 0.012 on this ring
KICK, LAG, RAMP = 2.0e-5, 0.13, (100, 500, 900, 1300)
NT = 1500
GATE = 1e-11  # relative to the largest |x|; measured 1.2e-13 (linear), 2.1e-13 (exact)
X0, PX0 = 1.0e-4, -2.0e-6


def _accsim(nonlinear: bool) -> np.ndarray:
    els = []
    for _ in range(NCELL):
        els += [ThinQuadrupole(K1L), Drift(LD), ThinQuadrupole(-K1L), Drift(LD)]
    els.insert(IDX, ACDipole(kick=KICK, tune=NU, lag=LAG, plane="x", ramp=RAMP))
    lat = Lattice(els, ReferenceParticle.from_gamma(PROTON_MASS_EV, GAMMA0))
    hist = Tracker(lat).track_turns(Particle(x=X0, px=PX0), NT, nonlinear=nonlinear)
    return hist[:NT, :2]  # row k: the start of pass k, as xtrack's monitor records it


def _xtrack(model: str) -> np.ndarray:
    els, names = [], []
    for i in range(NCELL):
        els += [
            xt.Multipole(knl=[0, K1L]),
            xt.Drift(length=LD, model=model),
            xt.Multipole(knl=[0, -K1L]),
            xt.Drift(length=LD, model=model),
        ]
        names += [f"qf{i}", f"da{i}", f"qd{i}", f"db{i}"]
    ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1.0, gamma0=GAMMA0)
    p0c_gev = float(ref.p0c[0]) / 1e9
    acd = xt.ACDipole(volt=KICK * p0c_gev / 0.3, freq=NU, lag=LAG, ramp=list(RAMP), plane="h")
    els.insert(IDX, acd)
    names.insert(IDX, "acd")
    line = xt.Line(elements=els, element_names=names)
    line.particle_ref = ref
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    p = line.build_particles(x=X0, px=PX0)
    line.track(p, num_turns=NT, turn_by_turn_monitor=True)
    rec = line.record_last_track
    return np.stack([rec.x[0], rec.px[0]], axis=1)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a[:, 0] - b[:, 0])) / np.max(np.abs(b[:, 0])))


def test_the_driven_motion_matches_xtrack_on_both_paths() -> None:
    lin, exact = _accsim(nonlinear=False), _accsim(nonlinear=True)
    xt_lin, xt_exact = _xtrack("expanded"), _xtrack("exact")
    assert _rel(lin, xt_lin) < GATE
    assert _rel(exact, xt_exact) < GATE
    # not drift-blind: the crossed pair differs by far more than the gate
    assert _rel(lin, xt_exact) > 1e3 * GATE
    # and the drive is really on: 1.3 mm of swing from a 0.1 mm start
    assert np.max(np.abs(xt_lin[:, 0])) > 10 * X0
