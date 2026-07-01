"""Stage 2 acceptance: natural chromaticity vs. the symbolically-derived dQ/ddelta.

Natural chromaticity ``Q' = dQ/ddelta`` comes from the off-momentum weakening of
the quadrupole gradient, ``k1 -> k1/(1 + delta)``. The production
:func:`natural_chromaticity` returns the textbook β-weighted integral
``Q'_x = -(1/4pi) ∮ beta_x k1 ds`` / ``Q'_y = +(1/4pi) ∮ beta_y k1 ds``.

The strong, **independent** check (mirroring how the Stage 1 FODO test re-derived
``cos mu = 1 - L^2/2f^2`` from the matrix rather than a remembered formula) does
NOT reuse the β-sum: it builds the thin one-turn map *as a function of delta*,
forms ``cos mu(delta) = 1/2 Tr M(delta)`` symbolically, and differentiates
``Q(delta) = mu(delta)/2pi`` at ``delta = 0``. That derivative knows nothing about
β or ``4pi`` — they are properties of the perturbation formula, not the map — so
first-order perturbation theory being exact at first order means the two must
agree to machine precision. Sum-vs-sum would only check the same thing twice.

A separate always-on test validates the *thick*-quad β-integration path against a
finite-difference tune derivative (the thick body is not a single β-at-the-quad
point), so the thick machinery is covered even when the xtrack cross-check skips.
"""

from __future__ import annotations

import math

import pytest

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinQuadrupole,
    natural_chromaticity,
    tunes,
)

F_FOCAL = 1.5  # full-quad focal length [m]
L_HALF = 1.0  # half-cell drift length [m]


@pytest.fixture
def ref() -> ReferenceParticle:
    # Thin quads + drifts are energy-independent; any ref works.
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _thin_fodo(f_val: float = F_FOCAL, ll_val: float = L_HALF) -> list:
    """Symmetric thin FODO from the F centre: half-F | drift | D | drift | half-F."""
    return [
        ThinQuadrupole(0.5 / f_val, name="qf_half"),
        Drift(ll_val, name="d1"),
        ThinQuadrupole(-1.0 / f_val, name="qd"),
        Drift(ll_val, name="d2"),
        ThinQuadrupole(0.5 / f_val, name="qf_half"),
    ]


def _symbolic_chromaticity(f_val: float, ll_val: float) -> dict[str, float]:
    """Re-derive (Q'_x, Q'_y) = dQ/ddelta from the delta-dependent thin one-turn map.

    Independent of the β-sum formula: the tune comes straight from the trace of
    the momentum-dependent map, differentiated at delta = 0.
    """
    sp = pytest.importorskip("sympy")
    f, ll, d = sp.symbols("f L delta", positive=False)

    def q_of_delta(sign: int):
        # sign = +1 for the x plane (F focuses x), -1 for y (F defocuses y).
        # A thin quad kicks p by -k1l*u in x and +k1l*u in y; off momentum the
        # integrated strength weakens by 1/(1+delta).
        w = 1 / (1 + d)
        qfh = sp.Matrix([[1, 0], [-sign * (1 / (2 * f)) * w, 1]])  # half-F
        qd = sp.Matrix([[1, 0], [sign * (1 / f) * w, 1]])  # full-D
        drift = sp.Matrix([[1, ll], [0, 1]])
        m = qfh * drift * qd * drift * qfh
        cos_mu = (m[0, 0] + m[1, 1]) / 2
        return sp.acos(cos_mu) / (2 * sp.pi)

    subs = {f: f_val, ll: ll_val}
    dqx = sp.diff(q_of_delta(+1), d).subs(d, 0).subs(subs)
    dqy = sp.diff(q_of_delta(-1), d).subs(d, 0).subs(subs)
    return {"dqx": float(dqx), "dqy": float(dqy)}


def test_thin_fodo_chromaticity_matches_symbolic_derivative(ref: ReferenceParticle) -> None:
    sym = _symbolic_chromaticity(F_FOCAL, L_HALF)
    lat = Lattice(_thin_fodo(), ref)
    xi_x, xi_y = natural_chromaticity(lat)

    # Thin quads are exact single-point contributions — no slicing — so this must
    # equal the symbolic dQ/ddelta to machine precision (a clean equality).
    assert xi_x == pytest.approx(sym["dqx"], rel=1e-12, abs=1e-12)
    assert xi_y == pytest.approx(sym["dqy"], rel=1e-12, abs=1e-12)


def test_thin_fodo_natural_chromaticity_is_negative(ref: ReferenceParticle) -> None:
    # Physical sanity: an ordinary FODO of pure quads has negative natural
    # chromaticity in both planes (off-momentum particles are under-focused).
    lat = Lattice(_thin_fodo(), ref)
    xi_x, xi_y = natural_chromaticity(lat)
    assert xi_x < 0.0
    assert xi_y < 0.0


def test_thin_fodo_chromaticity_equals_beta_weighted_sum(ref: ReferenceParticle) -> None:
    # The value the function reports IS the closed-form β-sum for this layout:
    #   Q'_x = -(1/4pi) [ beta_max/(2f) - beta_min/f + beta_max/(2f) ]
    #        = -(1/4pi) (beta_max - beta_min)/f,
    # with beta_max at the F centre (cell start) and beta_min at the D centre.
    from accsim import closed_twiss, propagate_twiss

    lat = Lattice(_thin_fodo(), ref)
    pts = propagate_twiss(lat, closed_twiss(lat))
    beta_max = pts[0].beta_x  # F centre
    beta_min = pts[2].beta_x  # at the D quad (continuous across the thin kick)
    expected_x = -(1.0 / (4.0 * math.pi)) * (beta_max - beta_min) / F_FOCAL

    xi_x, _ = natural_chromaticity(lat)
    assert xi_x == pytest.approx(expected_x, rel=1e-12)


def _finite_difference_chromaticity(
    build_cell, ref: ReferenceParticle, h: float = 1e-6
) -> tuple[float, float]:
    """dQ/ddelta by central difference, weakening every quad's k1 by 1/(1+delta).

    Test scaffolding only: rebuilds the lattice with weakened *copies* of each
    quad (no ``delta`` threaded through the element API), then differentiates the
    tune numerically. Shares the k1/(1+delta) physics with the analytic formula —
    it checks the β-integration coefficient, not the physics itself (that is what
    the xtrack cross-check is for).
    """

    def weakened_tunes(delta: float) -> tuple[float, float]:
        scale = 1.0 / (1.0 + delta)
        cell = []
        for e in build_cell():
            if isinstance(e, Quadrupole):
                cell.append(Quadrupole(e.length, e.k1 * scale))
            elif isinstance(e, ThinQuadrupole):
                cell.append(ThinQuadrupole(e.k1l * scale))
            else:
                cell.append(e)
        return tunes(Lattice(cell, ref))

    qx_p, qy_p = weakened_tunes(+h)
    qx_m, qy_m = weakened_tunes(-h)
    return (qx_p - qx_m) / (2.0 * h), (qy_p - qy_m) / (2.0 * h)


def test_thick_fodo_chromaticity_matches_finite_difference(ref: ReferenceParticle) -> None:
    # Validate the THICK-quad β-integration path (the body is not a single
    # β-at-the-quad point) against a finite-difference tune derivative. Always-on,
    # so the thick machinery is covered even when the xtrack cross-check skips.
    lq, k1, ld = 0.3, 1.2, 1.0

    def cell() -> list:
        return [Quadrupole(lq, k1), Drift(ld), Quadrupole(lq, -k1), Drift(ld)]

    lat = Lattice(cell() * 4, ref)
    xi_x, xi_y = natural_chromaticity(lat)
    fd_x, fd_y = _finite_difference_chromaticity(lambda: cell() * 4, ref)

    # Both negative, and the β-integral matches the tune derivative. Tolerance
    # covers trapezoidal slicing + central-difference truncation, both « the value.
    assert xi_x < 0.0 and xi_y < 0.0
    assert xi_x == pytest.approx(fd_x, rel=1e-4)
    assert xi_y == pytest.approx(fd_y, rel=1e-4)
