"""Cross-check the Quadrupole 6x6 against MAD-X -- the second reference (D3).

Marked ``reference``: skips when cpymad is absent.

Beyond entrywise agreement, this re-pins the two convention choices the xtrack
check pins, now against an independently written code: (1) ``k1 > 0`` focuses in
x and defocuses in y, and (2) the longitudinal slip is carried *inside* the thick
quad rather than sliced out of it.
"""

from __future__ import annotations

import numpy as np
import pytest
from _madx import single_element_rmatrix

from accsim import DELTA, PX, PY, ZETA, Quadrupole, ReferenceParticle, X, Y

pytestmark = pytest.mark.reference

MASS0 = 0.51099895069e6  # electron, eV
GAMMA0 = 2.0  # non-ultrarelativistic so the longitudinal block is unambiguous
LENGTH = 0.5
K1 = 1.2


def test_quadrupole_matrix_matches_madx() -> None:
    m = single_element_rmatrix(
        f"quadrupole, l={LENGTH}, k1={K1}", LENGTH, particle="electron", gamma0=GAMMA0
    )
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0, charge=-1.0)
    R_us = Quadrupole(LENGTH, K1).matrix(ref)

    np.testing.assert_allclose(R_us, m.accsim, rtol=1e-9, atol=1e-12)

    # Convention pins, restated against the second code:
    assert R_us[PX, X] < 0.0 and m.accsim[PX, X] < 0.0  # k1>0 focuses x
    assert R_us[PY, Y] > 0.0 and m.accsim[PY, Y] > 0.0  # k1>0 defocuses y
    # The slip lives inside the thick quad, and it is the drift-like value: a
    # quad has no dispersion, so nothing else contributes to R56 at first order.
    assert R_us[ZETA, DELTA] == pytest.approx(LENGTH / GAMMA0**2, rel=1e-9)
