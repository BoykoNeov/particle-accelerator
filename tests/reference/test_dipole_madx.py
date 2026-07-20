"""Cross-check the Dipole (pure sector bend) 6x6 against MAD-X -- second reference (D3).

Marked ``reference``: skips when cpymad is absent.

The MAD-X ``sbend`` is configured as a *pure sector* -- pole-face angles
``e1 = e2 = 0`` and no gradient (``k1 = 0``) -- so its map is the bare sector
bend with no edge focusing, apples-to-apples with
:class:`accsim.elements.dipole.Dipole`.

This is also the element that fixes the **sign** of the ``(T, PT) -> (zeta,
delta)`` transform. A drift only exercises the diagonal-adjacent longitudinal
term, which is even under flipping both longitudinal coordinates; the dipole's
``R51``/``R52`` (path lengthening with transverse offset and angle) and
``R16``/``R26`` (dispersion) are odd under that flip, so agreement here is what
rules out a sign error hiding in the change of variables. See ``_madx``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _madx import single_element_rmatrix

from accsim import DELTA, PX, ZETA, Dipole, ReferenceParticle, X

pytestmark = pytest.mark.reference

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 5.0
LENGTH = 2.0
ANGLE = 0.2


def test_dipole_matrix_matches_madx() -> None:
    m = single_element_rmatrix(
        f"sbend, l={LENGTH}, angle={ANGLE}, e1=0, e2=0",
        LENGTH,
        particle="proton",
        gamma0=GAMMA0,
    )
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    R_us = Dipole(LENGTH, ANGLE).matrix(ref)

    np.testing.assert_allclose(R_us, m.accsim, rtol=1e-9, atol=1e-12)

    # Convention pins against the second code:
    assert R_us[X, DELTA] > 0.0 and m.accsim[X, DELTA] > 0.0  # dispersion outward

    # The sign-sensitive entries. These are non-zero *only* for a bend, and they
    # flip sign if the zeta<->T mapping is taken with the wrong sign, so their
    # agreement is the evidence that the transform's sign is right rather than
    # merely self-consistent.
    assert R_us[ZETA, X] < 0.0 and m.accsim[ZETA, X] < 0.0
    assert R_us[ZETA, PX] < 0.0 and m.accsim[ZETA, PX] < 0.0
    np.testing.assert_allclose(R_us[ZETA, X], m.accsim[ZETA, X], rtol=1e-9)
    np.testing.assert_allclose(R_us[ZETA, PX], m.accsim[ZETA, PX], rtol=1e-9)


def test_dipole_symplectic_in_both_frames() -> None:
    """The transform is a genuine change of variables, not a fudge factor.

    A similarity transform preserves symplecticity only if it is itself
    symplectic. Checking that MAD-X's raw map and its transformed image are
    *both* symplectic confirms ``M = diag(1,1,1,1,beta0,1/beta0)`` is a
    legitimate canonical rescaling -- the ``beta0`` and ``1/beta0`` pairing is
    exactly what keeps the longitudinal conjugate pair's area intact.
    """
    from accsim import is_symplectic

    m = single_element_rmatrix(
        f"sbend, l={LENGTH}, angle={ANGLE}, e1=0, e2=0",
        LENGTH,
        particle="proton",
        gamma0=GAMMA0,
    )
    assert is_symplectic(m.madx, atol=1e-10)
    assert is_symplectic(m.accsim, atol=1e-10)
