"""Cross-check the Dipole (pure sector bend) 6x6 against xtrack.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

The xtrack ``Bend`` is configured as a *pure sector* — edges disabled
(``edge_entry/exit_active = 0``) and ``k1 = 0`` — so its R-matrix is the bare
sector map with no edge focusing or gradient, apples-to-apples with
:class:`accsim.elements.dipole.Dipole`.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import DELTA, PX, ZETA, Dipole, ReferenceParticle, X

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 5.0  # non-ultrarelativistic so the L/gamma0^2 slip is sizeable
LENGTH = 2.0
ANGLE = 0.2


def _xtrack_bend_rmatrix() -> np.ndarray:
    """6x6 R-matrix of a pure sector xtrack Bend, or skip if the JIT can't build."""
    ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    bend = xt.Bend(length=LENGTH, angle=ANGLE, k1=0.0)
    bend.edge_entry_active = 0
    bend.edge_exit_active = 0
    line = xt.Line(elements=[bend])
    line.particle_ref = ref
    try:
        line.build_tracker()
        res = line.get_R_matrix(particle_on_co=ref.copy())
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return np.asarray(res["R_matrix"])


def test_dipole_matrix_matches_xtrack() -> None:
    R_xt = _xtrack_bend_rmatrix()
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    R_us = Dipole(LENGTH, ANGLE).matrix(ref)

    # Whole 6x6: horizontal focusing, dispersion, vertical drift, and the
    # longitudinal path-length row all agree with the reference tracker.
    np.testing.assert_allclose(R_us, R_xt, rtol=1e-6, atol=1e-9)

    # Convention pins stated against the reference:
    assert R_us[X, DELTA] > 0.0 and R_xt[X, DELTA] > 0.0  # dispersion outward
    # R56 carries the momentum-variable slip L/gamma0^2 minus the design-orbit arc,
    # NOT an energy-variable value — xtrack confirms the exact number.
    assert R_us[ZETA, DELTA] == pytest.approx(R_xt[ZETA, DELTA], rel=1e-6)
    assert R_us[PX, DELTA] == pytest.approx(R_xt[PX, DELTA], rel=1e-6)
