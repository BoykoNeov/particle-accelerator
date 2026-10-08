"""Cross-check a thick acceptance element against MAD-X's collimators.

Marked ``reference``: skips when cpymad is absent. See ``_madx`` for the
``(T, PT) -> (zeta, delta)`` change of variables.

The question this settles (open in ``docs/CONVENTIONS.md`` until 2026-10-08): is a
collimator of length ``L`` the identity, as accsim used to say, or a drift of ``L``,
as the editor's optics drew it? MAD-X answers for all three of its collimator types
with the whole 6x6, slip included — and xtrack's MAD-X loader converts all three with
``convert_drift_like``, the second code's answer read off its source.
"""

from __future__ import annotations

import numpy as np
import pytest
from _madx import single_element_rmatrix

from accsim import DELTA, ZETA, Aperture, Collimator, Drift, MomentumAperture, ReferenceParticle

pytestmark = pytest.mark.reference

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 3.0  # non-ultrarelativistic so the L/gamma0^2 slip is sizeable
LENGTH = 1.3

_MADX_JAWS = [
    f"rcollimator, l={LENGTH}, xsize=0.01, ysize=0.02",
    f"ecollimator, l={LENGTH}, xsize=0.01, ysize=0.02",
    f"collimator, l={LENGTH}, apertype=rectangle, aperture={{0.01, 0.02}}",
]


@pytest.mark.parametrize("definition", _MADX_JAWS, ids=["rcollimator", "ecollimator", "collimator"])
def test_a_madx_collimator_is_accsims_collimator(definition: str) -> None:
    m = single_element_rmatrix(definition, LENGTH, particle="proton", gamma0=GAMMA0)
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)

    for ours in (
        Collimator("rectangular", 0.01, 0.02, length=LENGTH),
        Aperture("rectangular", 0.01, 0.02, length=LENGTH),
        MomentumAperture(1.0e-3, length=LENGTH),
    ):
        np.testing.assert_allclose(ours.matrix(ref), m.accsim, rtol=1e-9, atol=1e-12)

    # Not blind: the identity accsim used to return misses by the drift's own entries.
    assert m.accsim[0, 1] == pytest.approx(LENGTH, rel=1e-12)
    assert m.accsim[ZETA, DELTA] == pytest.approx(LENGTH / GAMMA0**2, rel=1e-9)
    np.testing.assert_allclose(m.accsim, Drift(LENGTH).matrix(ref), rtol=1e-9, atol=1e-12)
