"""Per-mode chromaticity: how each coupled mode's tune moves with momentum.

Change 2 of *coupled optics in the editor's main view* (``docs/ROADMAP.md``, gates
pre-committed there). On a coupled ring the two modes are not the two planes, and neither
is their chromaticity: on the skew ring below, mode 1's ``dQ/ddelta`` is 23% away from the
uncoupled ring's ``xi_x``. :func:`~accsim.twiss.mode_chromaticity` central-differences the
full mode tunes over the tracked maps on the closed orbit at ``+-delta``.

The analytic leg is a closed form derived here with sympy: an uncoupled one-turn map with one
thin skew quadrupole at the observation point. The xtrack leg is
``tests/reference/test_mode_chromaticity_xtrack.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp
from test_coupled_compaction import _ref, skew_ring

from accsim import (
    Dipole,
    Drift,
    Lattice,
    ThinQuadrupole,
    ThinSkewQuadrupole,
    chromatic_functions,
    chromaticity,
    closed_twiss,
    mode_chromaticity,
    tunes,
)
from accsim.twiss import _mode_tune_scan

K1SL = 0.05


def bend_free(k1sl: float, split: bool = True) -> Lattice:
    """change 1's ring with every bend a drift, the thin skew quadrupole at the entrance."""
    els = [Drift(e.length) if isinstance(e, Dipole) else e for e in skew_ring(0.0).elements]
    if not split:
        els = els[:-1]  # drop the tune-splitting quadrupole: Q_x = Q_y exactly
    return Lattice([ThinSkewQuadrupole(k1sl), *els], _ref())


# ---------------------------------------------------------------------- gate 1
def _closed_form_cosines():
    """``(e1, disc)``: ``cos mu_1 + cos mu_2 = e1/2`` and ``cos mu_1,2 = (e1 +- sqrt(disc))/4``."""
    mx, my, bx, by, ax, ay, k = sp.symbols("mu_x mu_y beta_x beta_y alpha_x alpha_y k", real=True)

    def cs(mu, b, a):
        g = (1 + a**2) / b
        return sp.Matrix(
            [
                [sp.cos(mu) + a * sp.sin(mu), b * sp.sin(mu)],
                [-g * sp.sin(mu), sp.cos(mu) - a * sp.sin(mu)],
            ]
        )

    kick = sp.eye(4)
    kick[1, 2] = k
    kick[3, 0] = k
    M = kick * sp.diag(cs(mx, bx, ax), cs(my, by, ay))
    S = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    N = M + (-S * M.T * S)
    p1 = sp.simplify(N.trace() / 2)
    p2 = sp.simplify((N * N).trace() / 2)
    e2 = sp.simplify((p1 * p1 - p2) / 2)
    return (mx, my, bx, by, ax, ay, k), sp.simplify(p1), sp.simplify(p1**2 - 4 * e2)


def test_the_closed_form_is_the_one_used_below() -> None:
    """The sympy result, in the form gate 1 differentiates (no alpha anywhere)."""
    (mx, my, bx, by, ax, ay, k), e1, disc = _closed_form_cosines()
    assert sp.simplify(e1 - 2 * sp.cos(mx) - 2 * sp.cos(my)) == 0
    want = 4 * ((sp.cos(mx) - sp.cos(my)) ** 2 + k**2 * bx * by * sp.sin(mx) * sp.sin(my))
    assert sp.simplify(disc - want) == 0


def _closed_form_xi(include_dbeta: bool = True) -> np.ndarray:
    """dQ_k/ddelta of the closed form, from the uncoupled ring's per-plane optics."""
    lat0 = bend_free(0.0)
    tw = closed_twiss(lat0)
    qx, qy = tunes(lat0)
    xix, xiy = chromaticity(lat0)
    ch = chromatic_functions(lat0, delta=1e-4)[0]
    mx, my = 2 * math.pi * qx, 2 * math.pi * qy
    cx, cy, sx, sy = math.cos(mx), math.cos(my), math.sin(mx), math.sin(my)
    bx, by, k = tw.beta_x, tw.beta_y, K1SL
    dcx, dcy = -sx * 2 * math.pi * xix, -sy * 2 * math.pi * xiy
    R = (cx - cy) ** 2 + k * k * bx * by * sx * sy
    # d(sin mu_x sin mu_y) = cos mu_x dmu_x sin mu_y + sin mu_x cos mu_y dmu_y, dmu = 2 pi xi
    dR = 2 * (cx - cy) * (dcx - dcy)
    dR += k * k * bx * by * (2 * math.pi * xix * cx * sy + 2 * math.pi * xiy * sx * cy)
    if include_dbeta:
        dR += k * k * (ch.dbeta_x * by + bx * ch.dbeta_y) * sx * sy
    sign = math.copysign(1.0, cx - cy)  # mode 1 is the x-like root
    out = []
    for s in (+sign, -sign):
        c = (cx + cy) / 2 + s * math.sqrt(R) / 2
        dc = (dcx + dcy) / 2 + s * dR / (4 * math.sqrt(R))
        out.append(-dc / (2 * math.pi * math.sqrt(1 - c * c)))
    return np.array(out)


def test_mode_chromaticity_matches_the_closed_form() -> None:
    """Gate 1: the scan reproduces the closed form's derivative to 1e-8."""
    xi = np.array(mode_chromaticity(bend_free(K1SL)))
    np.testing.assert_allclose(xi, [-0.39600114, -0.39806748], atol=1e-8)
    np.testing.assert_allclose(xi, _closed_form_xi(), rtol=0, atol=1e-8)


def test_the_beta_term_of_the_closed_form_is_load_bearing() -> None:
    """Gate 1: without dbeta/ddelta at the skew quadrupole the closed form moves by 5e-4."""
    gap = np.abs(_closed_form_xi(include_dbeta=False) - _closed_form_xi())
    assert gap.min() > 4e-4


# ---------------------------------------------------------------------- gate 2
def test_the_mode_is_not_the_plane() -> None:
    """Gate 2: on the bendy skew ring mode 1's slope is 23% off the uncoupled xi_x."""
    xi = np.array(mode_chromaticity(skew_ring()))
    plane = np.array(chromaticity(skew_ring(0.0)))
    np.testing.assert_allclose(xi, [-0.167256, -0.238061], atol=2e-6)
    assert abs(xi[0] / plane[0] - 1.0) > 0.2


def test_the_step_error_is_second_order() -> None:
    """Gate 2: halving delta quarters the change, so the default step is in the flat region."""
    lat = skew_ring()
    xi = [np.array(_mode_tune_scan(lat, d, 1e-7)) for d in (1e-3, 5e-4, 2.5e-4)]
    r1, r2 = xi[0] - xi[1], xi[1] - xi[2]
    assert np.all((3.9 < r1 / r2) & (r1 / r2 < 4.1))


# ---------------------------------------------------------------------- gate 3
def test_the_scan_is_the_integral_on_an_uncoupled_ring() -> None:
    """Gate 3: forced through the coupled machinery, the scan meets chromaticity() at 1/n^2."""
    lat = skew_ring(0.0)
    a, b = (np.array(_mode_tune_scan(lat, d, 1e-7)) for d in (2.5e-4, 1.25e-4))
    limit = b + (b - a) / 3.0
    miss = [limit - np.array(chromaticity(lat, slices=n)) for n in (64, 128, 256)]
    for m1, m2 in zip(miss, miss[1:], strict=False):
        assert np.all((3.9 < m1 / m2) & (m1 / m2 < 4.1))


# ---------------------------------------------------------------------- gate 4
def test_mode_chromaticity_is_second_order_in_the_coupling() -> None:
    """Gate 4: (xi_mode - xi_plane) / k1sl^2 settles to a constant."""
    plane = np.array(chromaticity(skew_ring(0.0), slices=512))
    coef = [(np.array(mode_chromaticity(skew_ring(k))) - plane) / k**2 for k in (0.005, 0.0025)]
    np.testing.assert_allclose(coef[1], [-8.203, 2.102], rtol=1e-2)
    np.testing.assert_allclose(coef[0], coef[1], rtol=1e-2)


# ---------------------------------------------------------------------- gate 6
def test_an_uncoupled_ring_returns_the_integral_even_at_equal_tunes() -> None:
    """Gate 6: no normal form is taken where the planes are the modes."""
    lat = bend_free(0.0, split=False)
    qx, qy = tunes(lat)
    assert qx == pytest.approx(qy, abs=1e-12)
    assert mode_chromaticity(lat) == chromaticity(lat)
    assert mode_chromaticity(skew_ring(0.0)) == chromaticity(skew_ring(0.0))


def test_a_coupled_ring_with_a_bending_gradient_magnet_is_refused() -> None:
    """Gate 6: the tracked map lacks L4's (1 + h x) metric group there."""
    els = list(skew_ring().elements)
    i = next(n for n, e in enumerate(els) if isinstance(e, Dipole))
    els[i] = Dipole(els[i].length, els[i].angle, k1=0.01)
    with pytest.raises(NotImplementedError, match="metric"):
        mode_chromaticity(Lattice(els, _ref()))


def test_the_bend_free_fixture_is_what_gate_1_says() -> None:
    """Guard: the per-plane tunes and the coupled modes the gates were measured on."""
    np.testing.assert_allclose(tunes(bend_free(0.0)), [0.370617, 0.394681], atol=1e-6)
    assert isinstance(bend_free(K1SL).elements[-1], ThinQuadrupole)
