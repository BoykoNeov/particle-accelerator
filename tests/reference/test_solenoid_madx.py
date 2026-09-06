"""Cross-check the Solenoid 6x6 against MAD-X (S1) — the second, independent reference.

Marked ``reference``: skips when cpymad is absent.

This is the sharp leg for the **matrix**. MAD-X reports the transfer map directly as its
``re<ij>`` columns, so nothing here is differenced and nothing is tracked: the comparison is
entrywise against a Fortran code that shares no source with accsim, and it lands at the level
of double-precision arithmetic (`2.2e-16` when the probe that chose this milestone was run).

Two conventions are pinned here that no analytic test can settle:

* **the sign of the coupling** — which way a positive ``ks`` turns the beam;
* **``ks`` is charge-free** — an electron and a proton at the same ``ks`` get the *same*
  map. A solenoid's coupling sense is physically set by the sign of the charge, so it could
  as easily have lived in the reference particle; MAD-X puts it in ``KS``, xtrack agrees, and
  :meth:`~accsim.elements.solenoid.Solenoid._matrix_body` therefore never reads
  ``ref.charge``. Getting this wrong flips the coupling for every electron machine while
  leaving every proton machine right, which is exactly the kind of error that survives a
  single-species test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from _madx import single_element_rmatrix

from accsim import DELTA, PX, PY, ZETA, ReferenceParticle, Solenoid, X, Y

pytestmark = pytest.mark.reference

MASS_E = 0.51099895069e6  # electron, eV
GAMMA0 = 20.0
LENGTH = 0.9
KS = 0.6


def _madx(ks: float, particle: str = "electron"):
    return single_element_rmatrix(
        f"solenoid, l={LENGTH}, ks={ks}", LENGTH, particle=particle, gamma0=GAMMA0
    )


def test_solenoid_matrix_matches_madx() -> None:
    """Entrywise, at round-off — and the coupling's sign with it.

    The tolerance is the residual the filter probe measured (`2.2e-16`) with headroom, not a
    round number: this is two implementations of the same closed form in double precision,
    and anything above that level is a disagreement rather than a tolerance.
    """
    ref = ReferenceParticle.from_gamma(MASS_E, GAMMA0, charge=-1.0)
    R_us = Solenoid(LENGTH, KS).matrix(ref)
    m = _madx(KS)

    np.testing.assert_allclose(R_us, m.accsim, rtol=0, atol=1e-14)

    # Convention pins, restated as signs so a future refactor cannot drift past them:
    assert R_us[X, Y] > 0.0 and m.accsim[X, Y] > 0.0  # ks > 0 turns y into x this way
    assert R_us[Y, X] < 0.0 and m.accsim[Y, X] < 0.0  # ... and x into y the other
    assert R_us[PX, X] < 0.0 and m.accsim[PX, X] < 0.0  # both planes focus
    assert R_us[PY, Y] < 0.0 and m.accsim[PY, Y] < 0.0
    # The slip is a straight element's: a solenoid has no dispersion, so nothing else
    # contributes to R56 at first order.
    assert R_us[ZETA, DELTA] == pytest.approx(LENGTH / GAMMA0**2, rel=1e-12)


def test_the_coupling_reverses_with_ks_in_madx_too() -> None:
    """``ks -> -ks`` exchanges the planes, in MAD-X as in accsim.

    The diagonal blocks are even in ``ks`` (a solenoid focuses either way round) and the
    coupling blocks are odd. Asserted against MAD-X rather than only internally, because the
    internal version of this test cannot tell a sign convention from its mirror image.
    """
    plus, minus = _madx(KS).accsim, _madx(-KS).accsim
    idx = np.ix_([X, PX], [X, PX])
    np.testing.assert_allclose(minus[idx], plus[idx], rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        minus[np.ix_([X, PX], [Y, PY])], -plus[np.ix_([X, PX], [Y, PY])], rtol=0, atol=1e-14
    )

    ref = ReferenceParticle.from_gamma(MASS_E, GAMMA0, charge=-1.0)
    np.testing.assert_allclose(Solenoid(LENGTH, -KS).matrix(ref), minus, rtol=0, atol=1e-14)


def test_ks_is_charge_free_in_madx() -> None:
    """MAD-X gives an electron and a proton the same solenoid map — and so does accsim.

    The convention trap this milestone's filter probe was run to find. Both reference codes
    put the charge inside ``ks``; if accsim ever starts consulting ``ref.charge`` in the
    solenoid's map, the second assert fails immediately.
    """
    electron, proton = _madx(KS, "electron").accsim, _madx(KS, "proton").accsim
    np.testing.assert_allclose(electron, proton, rtol=0, atol=1e-14)

    e_ref = ReferenceParticle.from_gamma(MASS_E, GAMMA0, charge=-1.0)
    p_ref = ReferenceParticle.from_gamma(MASS_E, GAMMA0, charge=+1.0)
    sol = Solenoid(LENGTH, KS)
    assert np.array_equal(sol.matrix(e_ref), sol.matrix(p_ref))


def test_a_cyclotron_rate_solenoid_would_fail_madx() -> None:
    """The control: the half that matters.

    A solenoid built at the cyclotron rate ``ks`` instead of the Larmor rate ``ks/2`` is the
    classic error, and it is not a small one — it misses MAD-X by ``O(1)``. Stated here so
    that the passing test above is known to be discriminating rather than merely loose.
    """
    ref = ReferenceParticle.from_gamma(MASS_E, GAMMA0, charge=-1.0)
    wrong = Solenoid(LENGTH, 2.0 * KS).matrix(ref)
    assert np.abs(wrong - _madx(KS).accsim).max() > 0.1
