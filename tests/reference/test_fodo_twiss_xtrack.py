"""Stage 1 acceptance cross-check: FODO Twiss vs. xtrack to < 1e-6.

Builds the *same* thick-quadrupole FODO ring in accsim and in xtrack and compares
the full matched optics — beta and alpha in both planes at every element
boundary, the accumulated phase mu/2pi, and the tunes. The thick quad is used
(rather than a thin lens) because xtrack's ``Quadrupole`` is unambiguous and its
6x6 map is already pinned entrywise in ``test_quadrupole_xtrack.py``.

Marked ``reference``: skips (not fails) when xtrack or its JIT compiler is
unavailable — see ``docs/CONVENTIONS.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    closed_twiss,
    propagate_twiss,
    tunes,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

LQ = 0.3  # quad length [m]
K1 = 1.2  # quad gradient [m^-2]
LD = 1.0  # drift length [m]
MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
N_CELLS = 4


def _accsim_lattice() -> Lattice:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    cell = [Quadrupole(LQ, K1), Drift(LD), Quadrupole(LQ, -K1), Drift(LD)]
    return Lattice(cell * N_CELLS, ref)


def _xtrack_twiss():
    """Matched 4D Twiss of the identical xtrack ring, or skip if the JIT can't build."""
    ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    cell = [
        xt.Quadrupole(length=LQ, k1=K1),
        xt.Drift(length=LD),
        xt.Quadrupole(length=LQ, k1=-K1),
        xt.Drift(length=LD),
    ]
    line = xt.Line(elements=cell * N_CELLS)
    line.particle_ref = ref
    try:
        line.build_tracker()
        return line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


def test_fodo_twiss_matches_xtrack() -> None:
    tw = _xtrack_twiss()

    lat = _accsim_lattice()
    pts = propagate_twiss(lat, closed_twiss(lat))

    s = np.array([p.s for p in pts])
    betx = np.array([p.beta_x for p in pts])
    bety = np.array([p.beta_y for p in pts])
    alfx = np.array([p.alpha_x for p in pts])
    alfy = np.array([p.alpha_y for p in pts])
    mux = np.array([p.mu_x for p in pts]) / (2.0 * np.pi)
    muy = np.array([p.mu_y for p in pts]) / (2.0 * np.pi)

    # Same s-grid (element boundaries) so the comparison is point-for-point.
    np.testing.assert_allclose(s, np.array(tw.s), atol=1e-12)

    # The Stage 1 acceptance gate: matched optics agree to < 1e-6.
    np.testing.assert_allclose(betx, np.array(tw.betx), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(bety, np.array(tw.bety), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(alfx, np.array(tw.alfx), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(alfy, np.array(tw.alfy), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(mux, np.array(tw.mux), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(muy, np.array(tw.muy), rtol=1e-6, atol=1e-9)

    qx, qy = tunes(lat)
    assert qx == pytest.approx(tw.qx, abs=1e-6)
    assert qy == pytest.approx(tw.qy, abs=1e-6)


def test_fodo_beta_oscillates_against_xtrack() -> None:
    # Sanity beyond the numeric match: xtrack's own optics show the FODO beat —
    # beta_x peaks where beta_y troughs (the F and D quads), confirming the
    # max-at-F / min-at-D structure the analytic test asserts in accsim.
    tw = _xtrack_twiss()
    betx, bety = np.array(tw.betx), np.array(tw.bety)
    assert betx.max() > 1.3 * betx.min()
    # Where beta_x peaks (an F quad), beta_y is at its trough — the planes beat
    # in antiphase. (Several F quads tie for the max across the ring, so compare
    # values, not indices.)
    assert bety[np.argmax(betx)] == pytest.approx(bety.min())
