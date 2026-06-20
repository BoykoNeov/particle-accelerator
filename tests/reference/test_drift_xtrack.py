"""Cross-check the Drift transfer matrix against xtrack (Xsuite core tracker).

Marked ``reference``: skipped when xtrack is not installed, and skipped (not
failed) when xtrack's just-in-time C-kernel compilation is unavailable on the
current interpreter. As of Python 3.14 on Windows the xtrack JIT compile fails
with a missing-include-path error (and the spaced project path likely compounds
it) — see docs/CONVENTIONS.md. This test will start exercising the cross-check
automatically once that is resolved (a Stage 1 prerequisite).
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import DELTA, PX, ZETA, Drift, ReferenceParticle, X

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")


def _xtrack_drift_rmatrix(length: float, mass0_eV: float, q0: float, gamma0: float) -> np.ndarray:
    """One-turn R-matrix of a single xtrack Drift, or skip if the JIT can't build."""
    ref = xt.Particles(mass0=mass0_eV, q0=q0, gamma0=gamma0)
    line = xt.Line(elements=[xt.Drift(length=length)])
    line.particle_ref = ref
    try:
        line.build_tracker()
        res = line.compute_one_turn_matrix_finite_differences(particle_on_co=ref.copy())
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return np.array(res["R_matrix"])


def test_drift_matrix_matches_xtrack() -> None:
    L = 2.0
    mass0 = 0.51099895069e6  # electron, eV
    gamma0 = 2.0  # non-ultrarelativistic so R56 is unambiguous

    R_xt = _xtrack_drift_rmatrix(L, mass0, q0=-1, gamma0=gamma0)
    ref = ReferenceParticle.from_gamma(mass0, gamma0, charge=-1.0)
    R_us = Drift(L).matrix(ref)

    # Key transfer-matrix entries must agree to high precision.
    assert R_us[X, PX] == pytest.approx(R_xt[X, PX], rel=1e-6, abs=1e-9)
    assert R_us[ZETA, DELTA] == pytest.approx(R_xt[ZETA, DELTA], rel=1e-6, abs=1e-9)
