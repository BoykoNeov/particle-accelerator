"""Cross-check the SkewQuadrupole 6x6 against MAD-X -- the exact-map second reference.

Marked ``reference``: skips when cpymad is absent.

This is the *strong* skew-quad pin. accsim's ``SkewQuadrupole`` is the exact
hard-edge 45-deg roll of a normal quad; MAD-X's ``quadrupole, k1s=...`` is the same
exact tilt, so the **whole transverse 4x4 agrees to ~2e-16** -- diagonal
``(F+D)/2`` focusing included. (xtrack's ``Quadrupole(k1s)`` keeps only the linear
coupling, so it can pin the *sign* but not this full map -- see
``test_betatron_coupling_xtrack.py``. That both codes agree on the coupling sign,
and MAD-X on the exact magnitude, is what fixes the ``k1s`` convention.)
"""

from __future__ import annotations

import numpy as np
import pytest
from _madx import single_element_rmatrix

from accsim import PX, PY, ReferenceParticle, SkewQuadrupole, X, Y

pytestmark = pytest.mark.reference

MASS0 = 0.51099895069e6  # electron, eV
GAMMA0 = 5.0
LENGTH = 0.5
K1S = 0.3
_T = [X, PX, Y, PY]


def test_skew_quadrupole_matrix_matches_madx() -> None:
    m = single_element_rmatrix(
        f"quadrupole, l={LENGTH}, k1s={K1S}", LENGTH, particle="electron", gamma0=GAMMA0
    )
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0, charge=-1.0)
    R_us = SkewQuadrupole(LENGTH, K1S).matrix(ref)

    # exact roll == exact tilt: the whole transverse 4x4 agrees to machine precision
    np.testing.assert_allclose(R_us[np.ix_(_T, _T)], m.accsim[np.ix_(_T, _T)], rtol=0, atol=1e-13)

    # convention pins, restated against the independent code:
    # the coupling is symmetric R[px,y] == R[py,x] and same sign in both
    assert R_us[PX, Y] == pytest.approx(m.accsim[PX, Y], abs=1e-13)
    assert R_us[PY, X] == pytest.approx(m.accsim[PY, X], abs=1e-13)
    assert R_us[PX, Y] * m.accsim[PX, Y] > 0.0
    # and the diagonal carries the O(k1s^2) focusing that a drift model would miss
    assert abs(m.accsim[PX, X]) > 1e-4 and R_us[PX, X] == pytest.approx(m.accsim[PX, X], abs=1e-13)
