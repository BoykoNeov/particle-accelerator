"""Stage 2: sextupole linear map (a drift) and its chromaticity feed-down.

A sextupole is nonlinear; its *linear* transfer matrix is a drift (a thin
sextupole is the identity), so it leaves beta, dispersion, and the tunes of the
linear lattice untouched. The Stage-2 "linear effect" is **feed-down**: at a point
of dispersion ``x = x_beta + D_x delta`` the quadratic kick
``Delta px = -1/2 k2l (x^2 - y^2)`` acquires a ``delta``-dependent linear gradient
``k1_eff = k2 D_x delta``, shifting the chromaticity by

    dQ_x/ddelta += +(1/4pi) ∮ beta_x k2 D_x ds,
    dQ_y/ddelta += -(1/4pi) ∮ beta_y k2 D_x ds.

The strong, **independent** check mirrors the natural-chromaticity test: it does
NOT reuse the beta-sum. It models the sextupole as the ``delta``-dependent thin
quad ``k1l_eff = k2l D_x delta``, builds the ``delta``-dependent one-turn map,
forms ``cos mu(delta) = 1/2 Tr M(delta)`` symbolically, and differentiates
``Q = mu/2pi`` at ``delta = 0``. That derivative never touches ``beta`` or ``4pi``
(properties of the perturbation formula, not the map), so it pins the coefficient
and per-plane sign to machine precision. It *does* share the feed-down **model**
(sextupole at ``D_x`` == extra quad ``k2 D_x delta``) with the formula — which is
exactly why the xtrack cross-check (real nonlinear kick) is not redundant.

``D_x`` here is accsim's matched dispersion (validated in Stage 1); it enters both
sides as the *same* input number, so this isolates the feed-down coefficient, not
the dispersion.
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
    Sextupole,
    ThinQuadrupole,
    ThinSextupole,
    chromaticity,
    closed_twiss,
    natural_chromaticity,
    propagate_twiss,
    tunes,
)
from accsim.coords import DELTA, PX, PY, ZETA, X, Y
from accsim.twiss import _sextupole_feeddown

F_FOCAL = 2.5  # thin-quad focal length [m]
L_DRIFT = 1.0  # drift / dipole length [m]
BEND = 0.15  # dipole bend angle [rad] (gives nonzero dispersion)
K2L = 8.0  # sextupole integrated strength [m^-2]


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _dispersive_cell(sext=None) -> list:
    """A FODO-with-bends cell; optionally a sextupole just after the F quad.

    Layout: half-F | drift | [sextupole] | dipole | D | dipole | drift | half-F.
    The two bends create dispersion, so a sextupole placed after the first drift
    sees ``D_x != 0`` and contributes feed-down.
    """
    els: list = [ThinQuadrupole(0.5 / F_FOCAL, name="qfh"), Drift(L_DRIFT, name="d1")]
    if sext is not None:
        els.append(sext)
    els += [
        Dipole(L_DRIFT, BEND, name="b1"),
        ThinQuadrupole(-1.0 / F_FOCAL, name="qd"),
        Dipole(L_DRIFT, BEND, name="b2"),
        Drift(L_DRIFT, name="d2"),
        ThinQuadrupole(0.5 / F_FOCAL, name="qfh"),
    ]
    return els


# --- linear map: a sextupole is a drift ------------------------------------


def test_thin_sextupole_linear_map_is_identity(ref: ReferenceParticle) -> None:
    assert np.allclose(ThinSextupole(3.3).matrix(ref), np.eye(6))


def test_thick_sextupole_linear_map_is_a_drift(ref: ReferenceParticle) -> None:
    L = 0.4
    M = Sextupole(L, 12.0).matrix(ref)
    drift = np.eye(6)
    drift[X, PX] = L
    drift[Y, PY] = L
    drift[ZETA, DELTA] = L / ref.gamma0**2  # same longitudinal slip as a drift
    assert np.allclose(M, drift)


def test_sextupole_k2l_is_k2_times_length() -> None:
    assert Sextupole(0.25, 8.0).k2l == pytest.approx(2.0)


# --- a sextupole does not perturb the linear optics ------------------------


def test_sextupole_leaves_tunes_beta_dispersion_unchanged(ref: ReferenceParticle) -> None:
    # The drift-map guard the feed-down isolation relies on: adding a sextupole
    # changes no linear quantity (tunes, matched beta, dispersion) at all.
    lat0 = Lattice(_dispersive_cell(), ref)
    lat1 = Lattice(_dispersive_cell(ThinSextupole(K2L)), ref)
    assert tunes(lat1) == pytest.approx(tunes(lat0), rel=1e-14)
    tw0, tw1 = closed_twiss(lat0), closed_twiss(lat1)
    assert tw1.beta_x == pytest.approx(tw0.beta_x, rel=1e-14)
    assert tw1.beta_y == pytest.approx(tw0.beta_y, rel=1e-14)
    assert tw1.disp_x == pytest.approx(tw0.disp_x, rel=1e-14)


# --- feed-down chromaticity: independent symbolic derivative ----------------


def _symbolic_feeddown(lat: Lattice, ref: ReferenceParticle) -> dict[str, float]:
    """Re-derive the sextupole feed-down (dQ/ddelta) from the delta-dependent trace.

    Fixed numeric 2x2 blocks for every element; the sextupole becomes a thin quad
    whose strength is ``k1l_eff = k2l D_x delta`` (``c_x = -k2l D_x delta`` in x,
    ``c_y = +k2l D_x delta`` in y). Independent of the (1/4pi)*beta formula.
    """
    sp = pytest.importorskip("sympy")
    d = sp.symbols("delta")

    pts = propagate_twiss(lat, closed_twiss(lat))
    # D_x at the entrance of the (single) sextupole in the cell.
    sext_idx = next(i for i, e in enumerate(lat.elements) if isinstance(e, ThinSextupole))
    dx = pts[sext_idx].disp_x

    def block(elem, plane: str):
        M = elem.matrix(ref)
        i, j = (X, PX) if plane == "x" else (Y, PY)
        return sp.Matrix([[M[i, i], M[i, j]], [M[j, i], M[j, j]]])

    def q_of_delta(plane: str):
        m = sp.eye(2)
        for elem in lat.elements:
            if isinstance(elem, ThinSextupole):
                c = (-elem.k2l * dx * d) if plane == "x" else (elem.k2l * dx * d)
                blk = sp.Matrix([[1, 0], [c, 1]])
            else:
                blk = block(elem, plane)
            m = blk * m  # M_last @ ... @ M_first
        cos_mu = (m[0, 0] + m[1, 1]) / 2
        return sp.acos(cos_mu) / (2 * sp.pi)

    return {
        "dqx": float(sp.diff(q_of_delta("x"), d).subs(d, 0)),
        "dqy": float(sp.diff(q_of_delta("y"), d).subs(d, 0)),
    }


def test_thin_sextupole_feeddown_matches_symbolic_derivative(ref: ReferenceParticle) -> None:
    lat = Lattice(_dispersive_cell(ThinSextupole(K2L)), ref)
    sym = _symbolic_feeddown(lat, ref)
    fx, fy = _sextupole_feeddown(lat)
    # A single thin sextupole is an exact single-point contribution, so this is a
    # clean machine-precision equality with the trace derivative.
    assert fx == pytest.approx(sym["dqx"], rel=1e-12, abs=1e-12)
    assert fy == pytest.approx(sym["dqy"], rel=1e-12, abs=1e-12)


def test_thin_sextupole_feeddown_equals_closed_form(ref: ReferenceParticle) -> None:
    # The value IS the closed form +/-(1/4pi) beta k2l D_x at the sextupole.
    lat = Lattice(_dispersive_cell(ThinSextupole(K2L)), ref)
    pts = propagate_twiss(lat, closed_twiss(lat))
    idx = next(i for i, e in enumerate(lat.elements) if isinstance(e, ThinSextupole))
    bx, by, dx = pts[idx].beta_x, pts[idx].beta_y, pts[idx].disp_x
    fx, fy = _sextupole_feeddown(lat)
    assert fx == pytest.approx(+(1.0 / (4.0 * math.pi)) * bx * K2L * dx, rel=1e-12)
    assert fy == pytest.approx(-(1.0 / (4.0 * math.pi)) * by * K2L * dx, rel=1e-12)


def test_chromaticity_is_natural_plus_feeddown(ref: ReferenceParticle) -> None:
    lat = Lattice(_dispersive_cell(ThinSextupole(K2L)), ref)
    nat = natural_chromaticity(lat)
    fx, fy = _sextupole_feeddown(lat)
    tot = chromaticity(lat)
    assert tot[0] == pytest.approx(nat[0] + fx)
    assert tot[1] == pytest.approx(nat[1] + fy)


# --- thick sextupole: converges to the thin limit as length -> 0 ------------


def test_thick_sextupole_feeddown_converges_to_thin(ref: ReferenceParticle) -> None:
    # The thick trapezoidal integration path -> the exact thin single-point value
    # as the length shrinks at fixed k2l (quadratically in the length).
    k2l = 2.0

    def centred_cell(sext, ls):
        return [
            ThinQuadrupole(0.5 / F_FOCAL),
            Drift(L_DRIFT - ls / 2),
            sext,
            Drift(L_DRIFT - ls / 2),
            Dipole(L_DRIFT, BEND),
            ThinQuadrupole(-1.0 / F_FOCAL),
            Dipole(L_DRIFT, BEND),
            Drift(L_DRIFT),
            ThinQuadrupole(0.5 / F_FOCAL),
        ]

    thin = _sextupole_feeddown(Lattice(centred_cell(ThinSextupole(k2l), 0.0), ref))
    prev_err = None
    for ls in (0.2, 0.05):
        thick = _sextupole_feeddown(Lattice(centred_cell(Sextupole(ls, k2l / ls), ls), ref))
        err = abs(thick[0] - thin[0]) / abs(thin[0])
        if prev_err is not None:
            # 4x shorter -> ~16x smaller error (second-order midpoint/trapezoid).
            assert err < prev_err / 8.0
        prev_err = err
    assert prev_err < 1e-3  # already sub-per-mille at ls = 0.05 m


# --- chromaticity correction: the point of a sextupole ----------------------


def test_sextupole_corrects_horizontal_chromaticity(ref: ReferenceParticle) -> None:
    # Feed-down is linear in k2l, so a single family can zero one plane. Pick k2l
    # to cancel the (negative) natural Q'_x, and confirm the total lands at zero.
    lat0 = Lattice(_dispersive_cell(), ref)
    nat_x, _ = natural_chromaticity(lat0)
    assert nat_x < 0.0  # ordinary FODO: negative natural chromaticity

    # feed-down per unit k2l (from a unit-strength probe), then solve for zero.
    unit = Lattice(_dispersive_cell(ThinSextupole(1.0)), ref)
    fx_per_k2l, _ = _sextupole_feeddown(unit)
    assert fx_per_k2l > 0.0  # k2l > 0 at D_x > 0 pushes chromaticity toward zero
    k2l_correct = -nat_x / fx_per_k2l

    corrected = Lattice(_dispersive_cell(ThinSextupole(k2l_correct)), ref)
    tot_x, _ = chromaticity(corrected)
    assert tot_x == pytest.approx(0.0, abs=1e-12)
