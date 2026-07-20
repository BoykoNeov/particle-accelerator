"""Cross-check the Drift 6x6 against MAD-X -- the *second* reference (D3).

Marked ``reference``: skips when cpymad is absent. See ``_madx`` for the
``(T, PT) -> (zeta, delta)`` change of variables and how it was pinned.

The drift is where the longitudinal scale factor is unambiguous, and it is the
one convention this project has repeatedly had to defend: accsim carries
``R56 = L / gamma0^2`` (a *momentum*-variable slip). MAD-X, working in the energy
variable ``PT``, reports ``L / (beta0^2 gamma0^2)`` for the same physical drift.
The two differ by exactly ``beta0^2``, and this test asserts that both the raw
MAD-X number and the transformed one land where they should -- so a future edit
that "fixes" the transform by fudging a factor fails here rather than silently
propagating.
"""

from __future__ import annotations

import numpy as np
import pytest
from _madx import single_element_rmatrix

from accsim import DELTA, ZETA, Drift, ReferenceParticle

pytestmark = pytest.mark.reference

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 3.0  # non-ultrarelativistic so the L/gamma0^2 slip is sizeable
LENGTH = 0.75


def test_drift_matrix_matches_madx() -> None:
    m = single_element_rmatrix(f"drift, l={LENGTH}", LENGTH, particle="proton", gamma0=GAMMA0)
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    R_us = Drift(LENGTH).matrix(ref)

    # The whole 6x6 -- transverse blocks *and* the longitudinal slip.
    np.testing.assert_allclose(R_us, m.accsim, rtol=1e-9, atol=1e-12)

    # The convention pin: accsim carries the momentum-variable slip L/gamma0^2.
    assert R_us[ZETA, DELTA] == pytest.approx(LENGTH / GAMMA0**2, rel=1e-12)

    # And MAD-X's *raw* frame carries the energy-variable one, beta0^2 larger.
    # This is a genuine second-code number, not a restatement of our transform:
    # it is what makes the change of variables necessary rather than cosmetic.
    beta0 = m.beta0
    assert m.madx[ZETA, DELTA] == pytest.approx(LENGTH / (beta0**2 * GAMMA0**2), rel=1e-9)
    assert m.madx[ZETA, DELTA] > R_us[ZETA, DELTA]  # they really do differ
