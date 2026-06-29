"""Cross-check the Drift transfer matrix against xtrack (Xsuite core tracker).

Marked ``reference``: skipped when xtrack is not installed, and skipped (not
failed) when xtrack's just-in-time C-kernel compilation is unavailable. On this
Windows toolchain the JIT compile is enabled by the ``_xtrack_jit`` fix-up
(applied in ``tests/reference/conftest.py``), which routes the build through
clang-cl — see ``docs/CONVENTIONS.md`` for the full diagnosis. On machines
without clang-cl the fix-up is a no-op and this cross-check skips gracefully.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import DELTA, ZETA, Drift, ReferenceParticle

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")


def _xtrack_drift_rmatrix(length: float, mass0_eV: float, q0: float, gamma0: float) -> np.ndarray:
    """Full one-turn 6x6 R-matrix of a single xtrack Drift, or skip if the JIT can't build."""
    ref = xt.Particles(mass0=mass0_eV, q0=q0, gamma0=gamma0)
    line = xt.Line(elements=[xt.Drift(length=length)])
    line.particle_ref = ref
    try:
        line.build_tracker()
        res = line.get_R_matrix(particle_on_co=ref.copy())
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return np.asarray(res["R_matrix"])


def test_drift_matrix_matches_xtrack() -> None:
    L = 2.0
    mass0 = 0.51099895069e6  # electron, eV
    gamma0 = 2.0  # non-ultrarelativistic so R56 = L/gamma0^2 is unambiguous

    R_xt = _xtrack_drift_rmatrix(L, mass0, q0=-1, gamma0=gamma0)
    ref = ReferenceParticle.from_gamma(mass0, gamma0, charge=-1.0)
    R_us = Drift(L).matrix(ref)

    # Gold check: every entry of the 6x6 map agrees with xtrack, not just the two
    # non-trivial couplings. This pins the whole drift convention (R12 = R34 = L,
    # the identity structure elsewhere) against the reference tracker.
    np.testing.assert_allclose(R_us, R_xt, rtol=1e-6, atol=1e-9)

    # The physics that distinguishes our convention: R56 = L/gamma0^2 (momentum
    # variable delta), NOT L/(beta0^2 gamma0^2) (energy variable). For gamma0 = 2
    # those are 0.5 vs 0.667 — xtrack confirms the momentum-variable value, and
    # its +sign (delta > 0 arrives earlier, so zeta increases).
    assert R_us[ZETA, DELTA] == pytest.approx(L / gamma0**2, rel=1e-9)
    assert R_xt[ZETA, DELTA] == pytest.approx(L / gamma0**2, rel=1e-6)
