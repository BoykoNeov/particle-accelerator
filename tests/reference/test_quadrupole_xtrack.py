"""Cross-check the Quadrupole transfer matrix against xtrack.

Marked ``reference``: skipped when xtrack is absent, and skipped (not failed)
when xtrack's JIT C-kernel compilation is unavailable. See
``tests/reference/test_drift_xtrack.py`` and ``docs/CONVENTIONS.md`` for the
toolchain story.

The purpose beyond entrywise agreement: pin two convention choices against the
reference tracker — (1) ``k1 > 0`` focuses in x / defocuses in y, and (2) the
longitudinal slip ``R56 = L/gamma0^2`` is carried *inside* the thick quad (some
codes slice it differently), the gotcha that motivated this check.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import DELTA, PX, PY, ZETA, Quadrupole, ReferenceParticle, X, Y

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")


def _xtrack_quad_rmatrix(
    length: float, k1: float, mass0_eV: float, q0: float, gamma0: float
) -> np.ndarray:
    """One-turn 6x6 R-matrix of a single xtrack Quadrupole, or skip if JIT can't build."""
    ref = xt.Particles(mass0=mass0_eV, q0=q0, gamma0=gamma0)
    line = xt.Line(elements=[xt.Quadrupole(length=length, k1=k1)])
    line.particle_ref = ref
    try:
        line.build_tracker()
        res = line.get_R_matrix(particle_on_co=ref.copy())
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return np.asarray(res["R_matrix"])


def test_quadrupole_matrix_matches_xtrack() -> None:
    L, k1 = 0.5, 1.2
    mass0 = 0.51099895069e6  # electron, eV
    gamma0 = 2.0  # non-ultrarelativistic so R56 = L/gamma0^2 is unambiguous

    R_xt = _xtrack_quad_rmatrix(L, k1, mass0, q0=-1, gamma0=gamma0)
    ref = ReferenceParticle.from_gamma(mass0, gamma0, charge=-1.0)
    R_us = Quadrupole(L, k1).matrix(ref)

    # Whole 6x6 agrees: focusing structure, both planes, and the longitudinal block.
    np.testing.assert_allclose(R_us, R_xt, rtol=1e-6, atol=1e-9)

    # Convention pins, stated explicitly against the reference:
    assert R_us[PX, X] < 0.0 and R_xt[PX, X] < 0.0  # k1>0 focuses x (R21 < 0)
    assert R_us[PY, Y] > 0.0 and R_xt[PY, Y] > 0.0  # k1>0 defocuses y (R43 > 0)
    assert R_xt[ZETA, DELTA] == pytest.approx(L / gamma0**2, rel=1e-6)  # slip carried here
