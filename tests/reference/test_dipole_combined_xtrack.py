"""Cross-check the combined-function Dipole (bend + gradient) against xtrack.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

The xtrack ``Bend`` takes a body gradient ``k1`` directly; edges are disabled
(``edge_*_active = 0``) so the R-matrix is the pure combined-function body,
apples-to-apples with ``Dipole(L, theta, k1=...)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import PX, PY, Dipole, ReferenceParticle, X, Y

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 5.0
LENGTH = 2.0
ANGLE = 0.2


def _xtrack_combined_rmatrix(k1: float) -> np.ndarray:
    ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    bend = xt.Bend(length=LENGTH, angle=ANGLE, k1=k1)
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


@pytest.mark.parametrize("k1", [0.3, -0.25])
def test_combined_dipole_matches_xtrack(k1: float) -> None:
    R_xt = _xtrack_combined_rmatrix(k1)
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    R_us = Dipole(LENGTH, ANGLE, k1=k1).matrix(ref)

    np.testing.assert_allclose(R_us, R_xt, rtol=1e-6, atol=1e-9)
    assert np.sign(R_us[PX, X]) == np.sign(R_xt[PX, X])  # K_x = h^2 + k1
    assert np.sign(R_us[PY, Y]) == np.sign(R_xt[PY, Y])  # K_y = -k1
