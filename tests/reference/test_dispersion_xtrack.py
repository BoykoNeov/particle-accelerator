"""Cross-check matched dispersion against xtrack on an arc FODO cell.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

Convention note (the one thing that could have differed): xtrack's twiss ``dx``
is ``dx/ddelta`` (momentum dispersion), the *same* variable as our matched ``D``.
It does **not** use the MAD-X ``pt``-based ``DX = (1/beta0) dx/ddelta``. Verified
at gamma0 = 5 (beta0 ~ 0.98), where a ``1/beta0`` factor would show as an
unmistakable ~2% gap — the measured ratio is 1.0 to ~1e-11. See
``docs/CONVENTIONS.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    Dipole,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    closed_twiss,
    propagate_twiss,
    tunes,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6
GAMMA0 = 5.0  # beta0 ~ 0.98 -> a 1/beta0 convention factor would be a clear 2%
F_FOCAL = 2.5
L_BEND = 1.0
ANGLE = 0.15


def _accsim_cell() -> list:
    return [
        ThinQuadrupole(0.5 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(-1.0 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]


def _xtrack_twiss():
    def quad(k1l: float):
        return xt.Multipole(knl=[0.0, k1l], length=0.0)

    def bend():
        b = xt.Bend(length=L_BEND, angle=ANGLE, k1=0.0)
        b.edge_entry_active = 0
        b.edge_exit_active = 0
        return b

    line = xt.Line(
        elements=[quad(0.5 / F_FOCAL), bend(), quad(-1.0 / F_FOCAL), bend(), quad(0.5 / F_FOCAL)]
    )
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        return line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


def test_dispersion_matches_xtrack() -> None:
    tw = _xtrack_twiss()

    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    lat = Lattice(_accsim_cell(), ref)
    pts = propagate_twiss(lat, closed_twiss(lat))

    dx = np.array([p.disp_x for p in pts])
    dpx = np.array([p.disp_px for p in pts])

    # Same momentum convention (ratio 1, not 1/beta0): agree to < 1e-6.
    np.testing.assert_allclose(dx, np.array(tw.dx), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(dpx, np.array(tw.dpx), rtol=1e-6, atol=1e-9)

    # Also pin beta-propagation and the tune THROUGH the dipoles directly (the
    # quad-only FODO cross-check cannot exercise a bend): B = C B C^T transport
    # and continuous phase must match xtrack on this dipole-containing cell.
    betx = np.array([p.beta_x for p in pts])
    bety = np.array([p.beta_y for p in pts])
    np.testing.assert_allclose(betx, np.array(tw.betx), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(bety, np.array(tw.bety), rtol=1e-6, atol=1e-9)
    qx, qy = tunes(lat)
    assert qx == pytest.approx(tw.qx, abs=1e-6)
    assert qy == pytest.approx(tw.qy, abs=1e-6)

    # Pin the convention explicitly: ratio is 1, decisively not 1/beta0 (~1.02).
    beta0 = ref.beta0
    ratio = tw.dx[0] / pts[0].disp_x
    assert ratio == pytest.approx(1.0, abs=1e-6)
    assert abs(ratio - 1.0 / beta0) > 0.01
