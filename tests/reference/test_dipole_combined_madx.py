"""Cross-check the combined-function Dipole (bend + gradient) against MAD-X (D3).

Marked ``reference``: skips when cpymad is absent.

The MAD-X ``sbend`` carries a body gradient ``k1`` with the pole faces left at
``e1 = e2 = 0``, so its map is the pure combined-function body -- apples-to-apples
with ``Dipole(L, theta, k1=...)``. This pins the ``K_x = h^2 + k1`` /
``K_y = -k1`` sign split (and the gradient's effect on dispersion and the
longitudinal slip) against an independent Fortran implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from _madx import single_element_rmatrix

from accsim import DELTA, PX, PY, Dipole, ReferenceParticle, X, Y

pytestmark = pytest.mark.reference

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 5.0
LENGTH = 2.0
ANGLE = 0.2


@pytest.mark.parametrize("k1", [0.3, -0.25])
def test_combined_dipole_matches_madx(k1: float) -> None:
    m = single_element_rmatrix(
        f"sbend, l={LENGTH}, angle={ANGLE}, k1={k1}, e1=0, e2=0",
        LENGTH,
        particle="proton",
        gamma0=GAMMA0,
    )
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    R_us = Dipole(LENGTH, ANGLE, k1=k1).matrix(ref)

    np.testing.assert_allclose(R_us, m.accsim, rtol=1e-9, atol=1e-12)

    # The gradient enters K_x = h^2 + k1: k1 > 0 strengthens horizontal focus
    # (R21 more negative), k1 < 0 weakens it -- pinned against the second code.
    assert np.sign(R_us[PX, X]) == np.sign(m.accsim[PX, X])
    # Vertical focus/defocus is set by K_y = -k1.
    assert np.sign(R_us[PY, Y]) == np.sign(m.accsim[PY, Y])
    assert R_us[X, DELTA] > 0.0 and m.accsim[X, DELTA] > 0.0  # dispersion still outward
