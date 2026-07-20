"""D3 acceptance: matched FODO-with-bends optics vs MAD-X, the second reference.

Marked ``reference``: skips when cpymad is absent.

Where the xtrack FODO check uses a straight (bend-free) cell, this ring carries
**dipoles**, so dispersion and momentum compaction are non-zero and actually
carry information -- a bend-free lattice has ``D_x = 0`` and ``alpha_c = 0`` in
any code, which would compare two zeros and prove nothing.

Two MAD-X frame conventions are pinned here, both consistent with the R-matrix
transform in ``_madx`` rather than assumed independently:

* ``DX``/``DPX`` in the twiss table are derivatives with respect to ``PT`` (the
  energy variable), so accsim's momentum dispersion is ``beta0 * DX``. This is
  the same ``beta0`` that relates ``R16`` in the two frames, which is why it is a
  consistency check and not a free fudge factor.
* ``MUX``/``MUY`` are already in units of the tune (turns), not radians.
"""

from __future__ import annotations

import numpy as np
import pytest
from _madx import beam_beta0, madx_session

from accsim import (
    DELTA,
    PX,
    ZETA,
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    X,
    closed_twiss,
    momentum_compaction,
    propagate_twiss,
    tunes,
)

pytestmark = pytest.mark.reference

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
LQ, K1 = 0.3, 1.2  # quad length [m], gradient [m^-2]
LD = 0.6  # drift length [m]
LB = 1.0  # dipole length [m]
N_CELLS = 8
ANGLE = 2.0 * np.pi / (2 * N_CELLS)  # two bends per cell close the ring exactly


def _accsim_lattice() -> Lattice:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    cell = [
        Quadrupole(LQ, K1),
        Drift(LD),
        Dipole(LB, ANGLE),
        Drift(LD),
        Quadrupole(LQ, -K1),
        Drift(LD),
        Dipole(LB, ANGLE),
        Drift(LD),
    ]
    return Lattice(cell * N_CELLS, ref)


def _identity_alpha_c(lat: Lattice) -> float:
    """alpha_c via the one-turn longitudinal row -- exact, no quadrature.

    Mirrors ``tests/analytic/test_momentum_compaction.py``. Used here because
    :func:`momentum_compaction` integrates ``D_x/rho`` by trapezoid, which is
    only O((ds)^2)-accurate; comparing the *exact* route to MAD-X keeps this a
    physics check rather than a quadrature-error check.
    """
    M = lat.one_turn_matrix()
    tw = closed_twiss(lat)
    r = M[ZETA, X] * tw.disp_x + M[ZETA, PX] * tw.disp_px + M[ZETA, DELTA]
    return 1.0 / lat.ref.gamma0**2 - r / lat.length


def _madx_twiss():
    """Matched periodic optics of the identical ring, plus beta0 and the summary."""
    with madx_session() as madx:
        madx.input(f"""
            beam, particle=proton, gamma={GAMMA0};
            qf: quadrupole, l={LQ}, k1={K1};
            qd: quadrupole, l={LQ}, k1={-K1};
            dr: drift, l={LD};
            bb: sbend, l={LB}, angle={ANGLE}, e1=0, e2=0;
            cell: line=(qf, dr, bb, dr, qd, dr, bb, dr);
            seq: line=({N_CELLS}*cell);
            use, sequence=seq;
            select, flag=twiss, clear;
            twiss;
        """)
        t = madx.table.twiss
        cols = {
            c: np.array(t[c], dtype=float)
            for c in ("s", "betx", "bety", "alfx", "alfy", "mux", "muy", "dx", "dpx")
        }
        summ = {c: float(madx.table.summ[c][0]) for c in ("q1", "q2", "alfa")}
        return cols, summ, beam_beta0(madx, "seq")


def _drop_end_marker(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """MAD-X appends a zero-length ``$end`` marker row duplicating the last s.

    Asserted rather than assumed: if MAD-X ever stops emitting it, this fails
    loudly instead of silently misaligning the two s-grids by one row.
    """
    s = cols["s"]
    assert s[-1] == s[-2], "expected a duplicated final row (the $end marker)"
    return {k: v[:-1] for k, v in cols.items()}


def test_fodo_ring_twiss_matches_madx() -> None:
    cols, summ, beta0 = _madx_twiss()
    cols = _drop_end_marker(cols)

    lat = _accsim_lattice()
    pts = propagate_twiss(lat, closed_twiss(lat))

    # Point-for-point on the same s-grid (element boundaries).
    np.testing.assert_allclose([p.s for p in pts], cols["s"], atol=1e-12)

    np.testing.assert_allclose([p.beta_x for p in pts], cols["betx"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose([p.beta_y for p in pts], cols["bety"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose([p.alpha_x for p in pts], cols["alfx"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose([p.alpha_y for p in pts], cols["alfy"], rtol=1e-9, atol=1e-12)

    # MAD-X reports phase in turns; accsim in radians.
    mux = np.array([p.mu_x for p in pts]) / (2.0 * np.pi)
    muy = np.array([p.mu_y for p in pts]) / (2.0 * np.pi)
    np.testing.assert_allclose(mux, cols["mux"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(muy, cols["muy"], rtol=1e-9, atol=1e-12)

    qx, qy = tunes(lat)
    assert qx == pytest.approx(summ["q1"], abs=1e-9)
    assert qy == pytest.approx(summ["q2"], abs=1e-9)


def test_fodo_ring_dispersion_matches_madx() -> None:
    """Dispersion, with the energy-vs-momentum rescaling that D3 exists to pin."""
    cols, _, beta0 = _madx_twiss()
    cols = _drop_end_marker(cols)

    lat = _accsim_lattice()
    pts = propagate_twiss(lat, closed_twiss(lat))
    dx = np.array([p.disp_x for p in pts])
    dpx = np.array([p.disp_px for p in pts])

    # D_accsim = beta0 * DX_madx -- the *same* beta0 that maps R16 between the
    # two frames in the dipole R-matrix check.
    np.testing.assert_allclose(dx, beta0 * cols["dx"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(dpx, beta0 * cols["dpx"], rtol=1e-9, atol=1e-12)

    # The rescaling is not cosmetic: at gamma0 = 20 it is a ~1e-3 relative shift,
    # far above the 1e-9 tolerance above, so an omitted beta0 would fail here.
    assert not np.allclose(dx, cols["dx"], rtol=1e-9, atol=1e-12)

    # Dispersion is outward and non-trivial -- guards against comparing two zeros.
    assert dx.min() > 0.0 and dx.max() > 1.0


def test_momentum_compaction_matches_madx() -> None:
    """alpha_c against MAD-X, exact route first, then the quadrature's convergence.

    MAD-X evaluates ``(1/C) integral D_x/rho ds`` in closed form per element;
    accsim's :func:`momentum_compaction` uses a trapezoid. Rather than loosen a
    tolerance to hide that, the exact identity is compared tightly, and the
    quadrature is then shown to *converge onto MAD-X's number* under refinement
    -- which upgrades the existing analytic convergence test from self-
    consistency to agreement with an independent code.
    """
    _, summ, _ = _madx_twiss()
    lat = _accsim_lattice()
    alfa_mx = summ["alfa"]

    # Exact route: agrees with MAD-X to near machine precision.
    assert _identity_alpha_c(lat) == pytest.approx(alfa_mx, rel=1e-10)

    # The shipped quadrature at its default slice count: close, but only ~1e-6,
    # and that gap is quadrature error, not a convention disagreement.
    coarse = abs(momentum_compaction(lat) / alfa_mx - 1.0)
    fine = abs(momentum_compaction(lat, slices=1024) / alfa_mx - 1.0)
    assert coarse < 1e-5
    # Refining by 16x cuts the error by ~256x (trapezoid is O(ds^2)); assert a
    # conservative 50x so the claim is about convergence, not a lucky constant.
    assert fine < coarse / 50.0
