"""Stage 2 cross-check: sextupole chromaticity feed-down vs. xtrack.

accsim omits the dipole's own weak-focusing / edge chromaticity (out of Stage 2
scope), so a bent lattice's *absolute* ``dqx`` differs from xtrack by that
uncomputed term — a direct comparison would fail on the dipole, not the
sextupole. The fix is to compare the **with-minus-without-sextupole difference**:
because a sextupole's linear map is a drift, switching its ``k2`` on/off (at fixed
geometry — the element stays in place, only its strength changes) leaves beta,
dispersion, and the tunes identical in *both* codes, so the shared (dipole, quad)
chromaticity cancels term-for-term and the difference isolates exactly the
feed-down. Toggling ``k2`` rather than inserting/removing the element keeps the
cell length (hence the tunes) fixed — a ``k2 = 0`` sextupole is exactly a drift.

xtrack tracks the real nonlinear sextupole kick at finite ``delta``, so the
difference of its ``tw.dqx`` is a genuinely independent check of accsim's
first-order ``+/-(1/4pi) beta k2 D_x`` feed-down — validating the *model*
(sextupole == extra quad ``k2 D_x delta``) that the symbolic analytic test shares
with the formula.

``k2`` is kept modest so the (linear-in-``k2``) feed-down is not contaminated by
``k2``-nonlinear geometric aberration. The observed accsim-vs-xtrack residual is
``rel ≈ 5e-4`` and is **independent of ``k2``** (checked at ``k2 = 6, 3, 1``), so it
is not a ``delta^2`` artefact but accsim's trapezoidal slicing of ``beta·D_x``
across the thick body against xtrack's thick-sextupole model and finite-``delta``
chromaticity step; the ``rel = 2e-3`` gate leaves ~4x headroom. Marked
``reference``: skips when xtrack or its JIT compiler is unavailable.
"""

from __future__ import annotations

import pytest

from accsim import (
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    Sextupole,
    chromaticity,
    natural_chromaticity,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
LQ = 0.3  # quad length [m]
K1 = 1.2  # quad gradient [m^-2]
LD = 1.0  # drift length [m]
LB = 1.0  # dipole length [m]
ANG = 0.12  # dipole bend angle [rad] -> nonzero dispersion
LS = 0.2  # sextupole length [m]
K2 = 6.0  # sextupole strength [m^-3] (modest: avoid delta^2 pollution)
N_CELLS = 3


def _accsim_feeddown() -> tuple[float, float]:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    cell = [
        Quadrupole(LQ, K1),
        Drift(LD),
        Sextupole(LS, K2),
        Dipole(LB, ANG),
        Quadrupole(LQ, -K1),
        Dipole(LB, ANG),
        Drift(LD),
    ]
    lat = Lattice(cell * N_CELLS, ref)
    tot = chromaticity(lat)
    nat = natural_chromaticity(lat)  # sextupole treated as its drift map
    return tot[0] - nat[0], tot[1] - nat[1]  # the isolated feed-down


def _xtrack_cell(k2: float) -> list:
    # The sextupole is always present; k2 = 0 makes it an exact drift, so the "off"
    # line has identical geometry (length, tunes) to the "on" line.
    return [
        xt.Quadrupole(length=LQ, k1=K1),
        xt.Drift(length=LD),
        xt.Sextupole(length=LS, k2=k2),
        xt.Bend(length=LB, angle=ANG, k0=ANG / LB),
        xt.Quadrupole(length=LQ, k1=-K1),
        xt.Bend(length=LB, angle=ANG, k0=ANG / LB),
        xt.Drift(length=LD),
    ]


def _xtrack_line(k2: float):
    line = xt.Line(elements=_xtrack_cell(k2) * N_CELLS)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        line.twiss(method="4d")  # probe: raises here if the JIT can't build
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


def test_sextupole_has_drift_linear_map() -> None:
    # The premise of the difference method: turning k2 on must not move the linear
    # tunes in xtrack (its linear map is a drift). If this fails, the dipole/quad
    # terms would NOT cancel and the difference would be meaningless.
    tw0 = _xtrack_line(k2=0.0).twiss(method="4d")
    tw1 = _xtrack_line(k2=K2).twiss(method="4d")
    assert tw1.qx == pytest.approx(tw0.qx, abs=1e-9)
    assert tw1.qy == pytest.approx(tw0.qy, abs=1e-9)


def test_sextupole_feeddown_matches_xtrack_difference() -> None:
    tw0 = _xtrack_line(k2=0.0).twiss(method="4d")
    tw1 = _xtrack_line(k2=K2).twiss(method="4d")
    xt_dfx = tw1.dqx - tw0.dqx  # xtrack's own with-minus-without feed-down
    xt_dfy = tw1.dqy - tw0.dqy

    acc_fx, acc_fy = _accsim_feeddown()

    # The Stage 2 acceptance gate for the sextupole: accsim's first-order feed-down
    # matches xtrack's real-tracking difference. Tolerance covers accsim's
    # trapezoidal slicing and xtrack's finite-delta chromaticity step.
    assert acc_fx == pytest.approx(xt_dfx, rel=2e-3, abs=1e-4)
    assert acc_fy == pytest.approx(xt_dfy, rel=2e-3, abs=1e-4)
    # Feed-down has opposite signs in the two planes (the x^2 - y^2 structure).
    assert acc_fx > 0.0 and acc_fy < 0.0
