"""Analytic checks for the 1-sigma beam envelope ``sigma_u = sqrt(eps*beta + (D*sd)^2)``.

The whole physics claim is that betatron width and momentum-spread offset add in
**quadrature** (they are statistically independent in a matched beam). The
discriminating test needs dispersion, so it runs on an arc cell *with a dipole*
(``D_x != 0``) and asserts the exact decomposition ``sigma_x^2 - eps_x beta_x ==
(D_x sigma_delta)^2`` at every point. A drift/quad-only lattice has ``D = 0`` and
cannot test the dispersive term at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Dipole,
    Drift,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    beam_sigma,
    closed_twiss,
    propagate_twiss,
)

F_FOCAL = 2.5
L_BEND = 1.0
ANGLE = 0.15


def _arc_cell() -> list:
    """Symmetric arc FODO with bends — generates nonzero horizontal dispersion."""
    return [
        ThinQuadrupole(0.5 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(-1.0 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(938.27208816e6, 5.0)


def test_quadrature_decomposition_on_dispersive_lattice(ref: ReferenceParticle) -> None:
    # THE gate: sigma_x^2 - eps_x beta_x == (D_x sigma_delta)^2 exactly, everywhere.
    emit_x, emit_y, sd = 3e-9, 1e-9, 1e-3
    pts = propagate_twiss(Lattice(_arc_cell(), ref), closed_twiss(Lattice(_arc_cell(), ref)))
    sx, sy = beam_sigma(pts, emit_x, emit_y, sd)

    # Dispersion must actually be present, else this test proves nothing.
    assert max(abs(t.disp_x) for t in pts) > 1e-3

    for t, sxi, syi in zip(pts, sx, sy, strict=True):
        disp_term = (t.disp_x * sd) ** 2
        assert sxi**2 - emit_x * t.beta_x == pytest.approx(disp_term, rel=1e-12, abs=1e-18)
        # Flat, uncoupled lattice: no vertical dispersion, so y is betatron-only.
        assert syi**2 == pytest.approx(emit_y * t.beta_y, rel=1e-12, abs=1e-18)


def test_zero_momentum_spread_is_pure_betatron(ref: ReferenceParticle) -> None:
    # sigma_delta = 0 must collapse to sqrt(eps*beta) even where D_x != 0.
    emit = 2e-9
    pts = propagate_twiss(Lattice(_arc_cell(), ref), closed_twiss(Lattice(_arc_cell(), ref)))
    sx, sy = beam_sigma(pts, emit, sigma_delta=0.0)
    for t, sxi, syi in zip(pts, sx, sy, strict=True):
        assert sxi == pytest.approx(math.sqrt(emit * t.beta_x), rel=1e-12)
        assert syi == pytest.approx(math.sqrt(emit * t.beta_y), rel=1e-12)


def test_emit_y_defaults_to_emit_x(ref: ReferenceParticle) -> None:
    # A dispersion-free quad-only lattice, so only the betatron term is exercised.
    lat = Lattice([ThinQuadrupole(0.4), Drift(1.0), ThinQuadrupole(-0.4), Drift(1.0)], ref)
    pts = propagate_twiss(lat, closed_twiss(lat))
    sx_def, sy_def = beam_sigma(pts, 1e-9)
    sx_exp, sy_exp = beam_sigma(pts, 1e-9, 1e-9)
    np.testing.assert_allclose(sy_def, sy_exp)
    # emit_y=emit_x by default; sigma_y then differs from sigma_x only via beta.
    for t, syi in zip(pts, sy_def, strict=True):
        assert syi == pytest.approx(math.sqrt(1e-9 * t.beta_y))
    assert sx_def == sx_exp


def test_larger_momentum_spread_widens_only_dispersive_plane(ref: ReferenceParticle) -> None:
    # Physical monotonicity: more sigma_delta grows sigma_x where D_x != 0, never y.
    emit = 2e-9
    pts = propagate_twiss(Lattice(_arc_cell(), ref), closed_twiss(Lattice(_arc_cell(), ref)))
    sx_lo, sy_lo = beam_sigma(pts, emit, sigma_delta=1e-4)
    sx_hi, sy_hi = beam_sigma(pts, emit, sigma_delta=1e-3)
    # sigma_x grows wherever there is dispersion; sigma_y is unchanged.
    assert any(hi > lo for hi, lo in zip(sx_hi, sx_lo, strict=True))
    for hi, lo in zip(sx_hi, sx_lo, strict=True):
        assert hi >= lo
    np.testing.assert_allclose(sy_hi, sy_lo)
