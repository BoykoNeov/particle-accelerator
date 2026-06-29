"""Stage 1 acceptance: a thin-lens FODO cell vs. the symbolically-derived closed form.

The expected phase advance and beta extrema are re-derived from scratch with
sympy *inside the test* (no remembered FODO coefficient), for the exact layout
built here: a symmetric cell starting at the centre of a focusing quad, with the
focusing quad split into two half-strength lenses at the ends —

    QF/2(f_lens=2f) - Drift(L) - QD(f_lens=-f) - Drift(L) - QF/2(f_lens=2f)

so the full-quad focal length is ``f`` and ``L`` is the half-cell drift. The
derivation yields cos mu = 1 - L^2/(2 f^2), beta_max at the F centre and beta_min
at the D centre; the code must reproduce all of them and show the
max-at-F / min-at-D oscillation in both planes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Drift,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    closed_twiss,
    propagate_twiss,
    tunes,
)
from accsim.coords import PX, X

F_FOCAL = 1.5  # full-quad focal length [m]
L_HALF = 1.0  # half-cell drift length [m]


def _fodo_cell() -> list:
    """The exact layout the closed form is derived for (half-F at both ends)."""
    return [
        ThinQuadrupole(0.5 / F_FOCAL, name="qf_half"),
        Drift(L_HALF, name="d1"),
        ThinQuadrupole(-1.0 / F_FOCAL, name="qd"),
        Drift(L_HALF, name="d2"),
        ThinQuadrupole(0.5 / F_FOCAL, name="qf_half"),
    ]


def _symbolic_fodo(f_val: float, ll_val: float) -> dict[str, float]:
    """Re-derive (mu, beta_max, beta_min) for this layout from the thin matrices."""
    sp = pytest.importorskip("sympy")
    f, ll = sp.symbols("f L", positive=True)
    qfh = sp.Matrix([[1, 0], [-1 / (2 * f), 1]])  # half-F, focusing in x
    qd = sp.Matrix([[1, 0], [1 / f, 1]])  # full-D, defocusing in x
    drift = sp.Matrix([[1, ll], [0, 1]])
    m = qfh * drift * qd * drift * qfh  # one-turn x-block from the F centre
    cos_mu = (m[0, 0] + m[1, 1]) / 2
    sin_mu = sp.sqrt(1 - cos_mu**2)
    beta_max = m[0, 1] / sin_mu  # alpha = 0 at the symmetry point
    half = drift * qfh  # F-centre -> D-centre half cell
    beta_min = half[0, 0] ** 2 * beta_max + half[0, 1] ** 2 / beta_max
    subs = {f: f_val, ll: ll_val}
    return {
        "cos_mu": float(cos_mu.subs(subs)),
        "mu": float(sp.acos(cos_mu).subs(subs)),
        "beta_max": float(sp.simplify(beta_max).subs(subs)),
        "beta_min": float(sp.simplify(beta_min).subs(subs)),
    }


@pytest.fixture
def ref() -> ReferenceParticle:
    # Optics is energy-independent for thin quads + drifts; any ref works.
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def test_phase_advance_matches_closed_form(ref: ReferenceParticle) -> None:
    sym = _symbolic_fodo(F_FOCAL, L_HALF)
    lat = Lattice(_fodo_cell(), ref)
    M = lat.one_turn_matrix()

    # cos mu = 1/2 Tr of the x 2x2 block, straight from the one-turn map.
    half_trace = 0.5 * (M[X, X] + M[PX, PX])
    assert half_trace == pytest.approx(sym["cos_mu"])
    assert half_trace == pytest.approx(1.0 - L_HALF**2 / (2.0 * F_FOCAL**2))

    # Phase advance per cell from continuous accumulation == derived mu.
    qx, qy = tunes(lat)
    assert qx == pytest.approx(sym["mu"] / (2.0 * math.pi))
    assert qy == pytest.approx(sym["mu"] / (2.0 * math.pi))  # symmetric cell


def test_beta_extrema_match_closed_form(ref: ReferenceParticle) -> None:
    sym = _symbolic_fodo(F_FOCAL, L_HALF)
    lat = Lattice(_fodo_cell(), ref)
    tw0 = closed_twiss(lat)

    # At the F centre (cell start): beta_x is the maximum, beta_y the minimum.
    assert tw0.beta_x == pytest.approx(sym["beta_max"])
    assert tw0.beta_y == pytest.approx(sym["beta_min"])
    assert tw0.alpha_x == pytest.approx(0.0, abs=1e-12)  # symmetry point

    # At the D centre: beta_x min, beta_y max. The full D quad is a single thin
    # kick, so no element boundary sits exactly at its centre — beta is preserved
    # across the kick (points[2] before == points[3] after) while alpha flips
    # sign antisymmetrically about the symmetry point. (The F quad, by contrast,
    # is split half-and-half at the cell ends, so its centre IS a boundary with
    # alpha = 0, asserted above.)
    points = propagate_twiss(lat, tw0)
    before_d, after_d = points[2], points[3]
    assert before_d.beta_x == pytest.approx(sym["beta_min"])
    assert after_d.beta_x == pytest.approx(sym["beta_min"])
    assert before_d.beta_y == pytest.approx(sym["beta_max"])
    assert before_d.alpha_x == pytest.approx(-after_d.alpha_x)  # symmetry straddled


def test_beta_oscillation_is_max_at_F_min_at_D(ref: ReferenceParticle) -> None:
    # Slice the drifts finely and confirm beta_x peaks AT the F quad (s=0) and
    # troughs AT the D quad (mid-cell), with the y-plane mirror-imaged.
    n = 20
    sub = L_HALF / n
    cell = (
        [ThinQuadrupole(0.5 / F_FOCAL)]
        + [Drift(sub) for _ in range(n)]
        + [ThinQuadrupole(-1.0 / F_FOCAL)]
        + [Drift(sub) for _ in range(n)]
        + [ThinQuadrupole(0.5 / F_FOCAL)]
    )
    lat = Lattice(cell, ref)
    pts = propagate_twiss(lat, closed_twiss(lat))
    s = np.array([p.s for p in pts])
    bx = np.array([p.beta_x for p in pts])
    by = np.array([p.beta_y for p in pts])

    # The two cell ends are both F centres (degenerate beta_x maxima), so check
    # the F centre attains the max and the unique trough sits at the D centre.
    assert bx[0] == pytest.approx(bx.max())  # beta_x maximal at the F centre
    assert s[np.argmin(bx)] == pytest.approx(L_HALF)  # minimal at the D centre
    assert by[0] == pytest.approx(by.min())  # beta_y minimal at F
    assert s[np.argmax(by)] == pytest.approx(L_HALF)  # maximal at D
    assert bx.max() > bx.min() * 1.5  # genuine, sizeable oscillation


def test_fodo_phase_advance_in_stable_range(ref: ReferenceParticle) -> None:
    # sin(mu/2) = L/(2f) must be < 1 for a stable cell; here L/(2f) = 1/3.
    sym = _symbolic_fodo(F_FOCAL, L_HALF)
    assert math.sin(sym["mu"] / 2.0) == pytest.approx(L_HALF / (2.0 * F_FOCAL))
    assert L_HALF / (2.0 * F_FOCAL) < 1.0
