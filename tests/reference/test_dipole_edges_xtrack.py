"""Cross-check Dipole pole-face (edge) focusing against xtrack.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

The xtrack ``Bend`` carries a **linear** edge model by default with the fringe
integrals off (``edge_*_fint = edge_*_hgap = 0``, verified below), which is the
same hard-edge kick accsim implements. Setting ``edge_entry_angle``/
``edge_exit_angle`` and leaving the fringe defaults untouched gives an
apples-to-apples map -- so this pins the edge focusing against a second,
independent tracker on top of the MAD-X check.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import PX, PY, Dipole, ReferenceParticle, X, Y

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 5.0
LENGTH = 2.0
ANGLE = 0.2
E1 = 0.12
E2 = 0.20


def _xtrack_bend_edges_rmatrix(e1: float, e2: float) -> np.ndarray:
    """6x6 R-matrix of an xtrack Bend with hard-edge pole faces, or skip."""
    ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    bend = xt.Bend(
        length=LENGTH,
        angle=ANGLE,
        k1=0.0,
        edge_entry_angle=e1,
        edge_exit_angle=e2,
    )
    # Guard the hard-edge assumption: fringe integrals must be zero, or the
    # vertical focusing would carry an e -> e - psi correction we do not model.
    assert bend.edge_entry_fint == 0.0 and bend.edge_exit_fint == 0.0
    assert bend.edge_entry_hgap == 0.0 and bend.edge_exit_hgap == 0.0
    line = xt.Line(elements=[bend])
    line.particle_ref = ref
    try:
        line.build_tracker()
        res = line.get_R_matrix(particle_on_co=ref.copy())
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return np.asarray(res["R_matrix"])


def test_dipole_edges_match_xtrack() -> None:
    R_xt = _xtrack_bend_edges_rmatrix(E1, E2)
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    R_us = Dipole(LENGTH, ANGLE, e1=E1, e2=E2).matrix(ref)

    np.testing.assert_allclose(R_us, R_xt, rtol=1e-6, atol=1e-9)
    assert R_us[PX, X] > 0.0 and R_xt[PX, X] > 0.0  # edge defocuses x
    assert R_us[PY, Y] < 0.0 and R_xt[PY, Y] < 0.0  # edge focuses y


def test_rectangular_bend_horizontal_cancels_vs_xtrack() -> None:
    R_xt = _xtrack_bend_edges_rmatrix(ANGLE / 2, ANGLE / 2)
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    R_us = Dipole(LENGTH, ANGLE, e1=ANGLE / 2, e2=ANGLE / 2).matrix(ref)

    np.testing.assert_allclose(R_us, R_xt, rtol=1e-6, atol=1e-9)
    assert R_us[PX, X] == pytest.approx(0.0, abs=1e-12)
    assert R_xt[PX, X] == pytest.approx(0.0, abs=1e-7)
