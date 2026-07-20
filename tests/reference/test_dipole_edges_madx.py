"""Cross-check Dipole pole-face (edge) focusing against MAD-X -- second reference (D3).

Marked ``reference``: skips when cpymad is absent.

The MAD-X ``sbend`` is given pole-face angles ``e1``/``e2`` with **fringe fields
off** (``fint = hgap = 0``), which is exactly accsim's hard-edge model -- so the
two maps are apples-to-apples with no fringe correction to reconcile. This is the
independent-implementation pin for the edge sign and plane: a positive edge angle
must defocus ``x`` (``R21 > 0``) and focus ``y`` (``R43 < 0``), and MAD-X, a
separate Fortran code, has to agree.

The rectangular-bend identity (``e1 = e2 = theta/2`` -> horizontal focusing
cancels, ``R21 = 0``) is also checked against MAD-X, so the analytic gate's
strongest claim is confirmed by the reference tracker and not just self-derived.
"""

from __future__ import annotations

import numpy as np
import pytest
from _madx import single_element_rmatrix

from accsim import DELTA, PX, PY, ZETA, Dipole, ReferenceParticle, X, Y

pytestmark = pytest.mark.reference

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 5.0
LENGTH = 2.0
ANGLE = 0.2
E1 = 0.12
E2 = 0.20


def test_dipole_edges_match_madx() -> None:
    m = single_element_rmatrix(
        f"sbend, l={LENGTH}, angle={ANGLE}, e1={E1}, e2={E2}, fint=0, hgap=0",
        LENGTH,
        particle="proton",
        gamma0=GAMMA0,
    )
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    R_us = Dipole(LENGTH, ANGLE, e1=E1, e2=E2).matrix(ref)

    # Whole 6x6, including the edge-modified horizontal/vertical blocks and the
    # dispersion/longitudinal rows they feed through composition.
    np.testing.assert_allclose(R_us, m.accsim, rtol=1e-9, atol=1e-12)

    # Sign/plane pins stated against the independent code:
    assert R_us[PX, X] > 0.0 and m.accsim[PX, X] > 0.0  # edge defocuses x
    assert R_us[PY, Y] < 0.0 and m.accsim[PY, Y] < 0.0  # edge focuses y
    # The longitudinal sign-sensitive entries still agree (edges don't corrupt them):
    assert R_us[X, DELTA] > 0.0 and m.accsim[X, DELTA] > 0.0
    assert R_us[ZETA, X] < 0.0 and m.accsim[ZETA, X] < 0.0


def test_rectangular_bend_horizontal_cancels_vs_madx() -> None:
    """e1 = e2 = theta/2: MAD-X confirms R21 = 0 (horizontal focusing cancels)."""
    m = single_element_rmatrix(
        f"sbend, l={LENGTH}, angle={ANGLE}, e1={ANGLE / 2}, e2={ANGLE / 2}, fint=0, hgap=0",
        LENGTH,
        particle="proton",
        gamma0=GAMMA0,
    )
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    R_us = Dipole(LENGTH, ANGLE, e1=ANGLE / 2, e2=ANGLE / 2).matrix(ref)

    np.testing.assert_allclose(R_us, m.accsim, rtol=1e-9, atol=1e-12)
    assert R_us[PX, X] == pytest.approx(0.0, abs=1e-12)
    assert m.accsim[PX, X] == pytest.approx(0.0, abs=1e-9)
    assert R_us[X, X] == pytest.approx(1.0) and R_us[PX, PX] == pytest.approx(1.0)
