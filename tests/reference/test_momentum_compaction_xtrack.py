"""Cross-check momentum compaction / slip factor against xtrack on an arc cell.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

This is the independent physics anchor for ``alpha_c``. The analytic suite pins
``alpha_c`` via the symplecticity identity (one-turn longitudinal row) and a
symbolic re-derivation — both correct-*by-construction* given the element maps.
xtrack computes it by its own tracking-based method, so agreement here is a
genuine external check on the absolute value and sign.

Convention note: xtrack's ``momentum_compaction_factor`` is the geometric
``alpha_c`` (no ``gamma0``), and its ``slip_factor`` is ``eta = alpha_c -
1/gamma0^2`` — the *same* sign convention as accsim's :func:`slip_factor`
(verified here to ~1e-6, where a flipped sign or a ``beta0^2`` factor would be an
unmistakable gap).
"""

from __future__ import annotations

import pytest

from accsim import (
    Dipole,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    momentum_compaction,
    slip_factor,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6
GAMMA0 = 5.0  # 1/gamma0^2 = 0.04, a large, unmistakable slip-vs-compaction offset
F_FOCAL = 2.5
L_BEND = 1.0
ANGLE = 0.15


def _accsim_cell() -> list:
    return [
        ThinQuadrupole(0.5 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(-1.0 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]


def _xtrack_twiss():
    def quad(k1l: float):
        return xt.Multipole(knl=[0.0, k1l], length=0.0)

    def bend():
        b = xt.Bend(length=L_BEND, angle=ANGLE, k1=0.0)
        b.edge_entry_active = 0
        b.edge_exit_active = 0
        return b

    line = xt.Line(
        elements=[quad(0.5 / F_FOCAL), bend(), quad(-1.0 / F_FOCAL), bend(), quad(0.5 / F_FOCAL)]
    )
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        return line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


def test_momentum_compaction_matches_xtrack() -> None:
    tw = _xtrack_twiss()

    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    lat = Lattice(_accsim_cell(), ref)

    alpha_c = momentum_compaction(lat)
    eta = slip_factor(lat)

    assert alpha_c == pytest.approx(tw.momentum_compaction_factor, rel=1e-6)
    assert eta == pytest.approx(tw.slip_factor, rel=1e-6)

    # Pin the eta convention explicitly: eta = alpha_c - 1/gamma0^2, decisively
    # NOT alpha_c (offset 0.04) and NOT the beta0^2-scaled variant.
    assert eta == pytest.approx(alpha_c - 1.0 / GAMMA0**2, rel=1e-12)
    assert abs(tw.slip_factor - tw.momentum_compaction_factor) == pytest.approx(
        1.0 / GAMMA0**2, abs=1e-6
    )
