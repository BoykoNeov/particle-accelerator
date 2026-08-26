r"""M3 second reference: MAD-X's ``DDX``, which is a different number — and reconcilable.

This is the first place on axis M where MAD-X can be **reconciled rather than named**.
M1 and M2 both ended with MAD-X's ``Q''`` unreachable by construction: its TWISS drift
is paraxial with no exact option, and M2 showed that splits ``Q''`` on any ring that
bends. Second-order dispersion is not split by the drift model at all (the reason is
derived in ``tests/analytic/test_second_order_dispersion.py``), so what is left between
accsim and MAD-X here is pure bookkeeping — and the bookkeeping is entirely a change of
momentum variable, resolved below to ``2e-7``.

**Two conventions are pinned by probe, and both would be easy to get wrong:**

1. ``DDX`` is the coefficient of ``pt^2`` in ``x = DX pt + DDX pt^2``, where
   ``pt = (E - E0)/(p0 c)`` is the **energy** deviation. It is therefore
   ``1/2 d^2x/dpt^2`` — half of xtrack's ``ddx`` *and* in a different variable:

       DDX = (d^2x/ddelta^2 - (dx/ddelta)/gamma0^2) / (2 beta0^2),

   using ``pt = beta0 delta + beta0 delta^2/(2 gamma0^2)``. Reading it as a plain
   ``1/2 d^2x/ddelta^2`` is wrong by ``4.6e-4`` at ``gamma0 = 20`` — small enough to
   pass for round-off — and by ``7.6e-3`` at ``gamma0 = 5``. Both rings are run below
   precisely so the ``gamma0``-dependence is what carries the proof, and a single-ring
   fit cannot masquerade as agreement.

2. **MAD-X renormalises ``PX`` when ``DELTAP`` is non-zero.** Its table's ``X`` at
   ``deltap`` is the closed orbit, but its ``PX`` is divided by the shifted reference
   momentum, so a naive second difference of MAD-X's own ``PX`` returns
   ``d^2px/ddelta^2 - 2 dpx/ddelta`` — on this ring, ``-0.3083`` where the derivative
   is ``+0.4381``, i.e. the wrong sign. That trap is asserted below rather than merely
   avoided, because the sampled-``DELTAP`` trick is exactly how M1 and M2 checked
   MAD-X's tunes and it does **not** transfer to the transverse momentum.

MAD-X is cheap compared with xtrack (no JIT compile), which is why the two-``gamma0``
sweep lives in this file rather than next door.

Marked ``reference``: skips (not fails) when cpymad is unavailable.
"""

from __future__ import annotations

import pytest
from _madx import madx_session

from accsim import (
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    second_order_chromaticity,
    second_order_dispersion,
)

pytestmark = pytest.mark.reference

MASS0 = 938.27208816e6  # proton, eV
LQ, K1, LD, LB, ANG, N_CELLS = 0.3, 1.2, 0.5, 1.0, 0.12, 3
DELTA = 1e-3

_CELL = ["qf", "dl", "dl", "bb", "qd", "bb", "dl"]


def _lattice(gamma: float) -> Lattice:
    ref = ReferenceParticle.from_gamma(MASS0, gamma)
    els: list = []
    for _ in range(N_CELLS):
        els += [
            Quadrupole(LQ, K1),
            Drift(LD),
            Drift(LD),
            Dipole(LB, ANG),
            Quadrupole(LQ, -K1),
            Dipole(LB, ANG),
            Drift(LD),
        ]
    return Lattice(els, ref)


def _build_madx(madx, gamma: float) -> None:
    """The same arc in MAD-X, element for element."""
    madx.input(f"qf: quadrupole, l={LQ}, k1={K1};")
    madx.input(f"qd: quadrupole, l={LQ}, k1={-K1};")
    madx.input(f"bb: sbend, l={LB}, angle={ANG};")
    madx.input(f"dl: drift, l={LD};")
    madx.input("ring: line=({});".format(", ".join(_CELL * N_CELLS)))
    madx.input(f"beam, particle=proton, gamma={gamma};")
    madx.input("use, sequence=ring;")


def _madx_chrom_columns(madx) -> dict[str, float]:
    """``DX``/``DPX``/``DDX``/``DDPX`` at the start of the ring, from ``TWISS, CHROM``."""
    madx.input("twiss, chrom;")
    table = madx.table.twiss
    return {
        "dx": float(table.dx[0]),
        "dpx": float(table.dpx[0]),
        "ddx": float(table.ddx[0]),
        "ddpx": float(table.ddpx[0]),
    }


def _to_madx_pt_convention(first: float, second: float, beta0: float, gamma0: float) -> float:
    r"""``d/ddelta`` derivatives -> MAD-X's ``pt^2`` coefficient.

    ``x(pt) = DX pt + DDX pt^2`` with ``pt = beta0 delta + beta0 delta^2/(2 gamma0^2)``
    gives ``d^2x/ddelta^2 = DX beta0/gamma0^2 + 2 DDX beta0^2``, and ``DX beta0`` is
    just ``dx/ddelta``. Derived here rather than recalled — the ``1/gamma0^2`` term is
    the whole reason two values of ``gamma0`` are run.
    """
    return (second - first / gamma0**2) / (2.0 * beta0**2)


# ---------------------------------------------------------------------------
# 1. the convention, pinned by probe at two energies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [20.0, 5.0])
def test_madxs_ddx_is_the_pt_squared_coefficient_not_half_the_delta_derivative(
    gamma: float,
) -> None:
    r"""MAD-X's ``DDX`` and ``DDPX``, reconstructed from accsim's ``d/ddelta`` derivatives.

    The transform is asserted at ``gamma0 = 20`` **and** ``gamma0 = 5``, where the
    ``1/gamma0^2`` term is sixteen times larger. That is what makes this a convention
    check rather than a curve fit: the naive "half of the second derivative" reading is
    wrong by a factor that *moves with the beam energy*, and both the size and the
    direction of that movement are pinned here.
    """
    lattice = _lattice(gamma)
    point = second_order_dispersion(lattice, delta=DELTA)[0]
    beta0, gamma0 = lattice.ref.beta0, lattice.ref.gamma0

    predicted_ddx = _to_madx_pt_convention(point.disp_x, point.ddisp_x, beta0, gamma0)
    predicted_ddpx = _to_madx_pt_convention(point.disp_px, point.ddisp_px, beta0, gamma0)

    with madx_session() as madx:
        _build_madx(madx, gamma)
        cols = _madx_chrom_columns(madx)

    assert cols["ddx"] == pytest.approx(predicted_ddx, rel=2e-6)
    assert cols["ddpx"] == pytest.approx(predicted_ddpx, rel=2e-6)

    # The first-order column is the ``pt`` dispersion this project already recorded,
    # ``DX = (1/beta0) dx/ddelta`` — asserted alongside so the two orders are shown to
    # use the *same* momentum variable rather than each being fixed up separately.
    assert cols["dx"] * beta0 == pytest.approx(point.disp_x, rel=1e-6)

    # And the naive "half the second derivative" reading fails by an amount that
    # *moves with the beam energy*: the algebra above collapses to
    # ``|1 - (dx/ddelta)/(d^2x/ddelta^2)| / (beta0 gamma0)^2``, which is 4.6e-4 at
    # gamma0 = 20 and 7.6e-3 at gamma0 = 5 on this ring. Neither is round-off, the two
    # differ by the sixteen-fold ratio of ``(beta0 gamma0)^2``, and asserting the
    # closed form rather than a threshold is what rules out a coincidence.
    naive_error = abs(cols["ddx"] / (0.5 * point.ddisp_x) - 1.0)
    predicted_error = abs(1.0 - point.disp_x / point.ddisp_x) / (beta0 * gamma0) ** 2
    assert naive_error > 3e-4
    assert naive_error == pytest.approx(predicted_error, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. the trap: sampling MAD-X's own orbit at three DELTAP values
# ---------------------------------------------------------------------------


def test_sampling_madxs_own_orbit_works_for_x_and_is_renormalised_for_px() -> None:
    r"""``X`` at three ``DELTAP`` values twice-differences correctly; ``PX`` does not.

    M1 and M2 both cleared their tune comparisons by sampling MAD-X's own ``Q1`` at
    three ``DELTAP`` values instead of reading a ``DD`` column, on the grounds that
    second-difference conventions differ between codes. That trick is sound for ``X``
    — asserted first — and silently wrong for ``PX``, because MAD-X normalises the
    transverse momentum to the *shifted* reference momentum: what comes back is
    ``d^2/ddelta^2 [px/(1+delta)] = d^2px/ddelta^2 - 2 dpx/ddelta``. On this ring that
    is ``-0.308`` against a true ``+0.438``, so it does not merely lose precision, it
    changes sign.
    """
    gamma = 20.0
    point = second_order_dispersion(_lattice(gamma), delta=DELTA)[0]

    with madx_session() as madx:
        _build_madx(madx, gamma)
        samples = {}
        for sign in (+1, 0, -1):
            madx.input(f"twiss, deltap={sign * DELTA:.12g};")
            samples[sign] = (float(madx.table.twiss.x[0]), float(madx.table.twiss.px[0]))

    def second_difference(index: int) -> float:
        return (samples[1][index] - 2.0 * samples[0][index] + samples[-1][index]) / DELTA**2

    assert second_difference(0) == pytest.approx(point.ddisp_x, rel=1e-5)

    renormalised = point.ddisp_px - 2.0 * point.disp_px
    assert second_difference(1) == pytest.approx(renormalised, rel=1e-5)
    assert second_difference(1) < 0.0 < point.ddisp_px  # the sign really does flip


# ---------------------------------------------------------------------------
# 3. what this buys: agreement where Q'' could not have it
# ---------------------------------------------------------------------------


def test_madx_and_accsim_agree_on_second_order_dispersion_where_they_cannot_on_qpp() -> None:
    r"""Both statements on one ring, so the contrast is a measurement and not a claim.

    MAD-X's TWISS drift is paraxial and cannot be changed, which M2 showed puts its
    ``Q''`` permanently out of reach on a ring that bends — the residual there is
    ``1.4e-2``. On second-order dispersion the same two codes agree to ``2e-7`` once
    the momentum variable is matched, because the drift model does not reach this
    quantity. One machine, two neighbouring quantities, five orders of magnitude
    between the residuals.
    """
    gamma = 20.0
    lattice = _lattice(gamma)
    point = second_order_dispersion(lattice, delta=DELTA)[0]
    beta0, gamma0 = lattice.ref.beta0, lattice.ref.gamma0

    with madx_session() as madx:
        _build_madx(madx, gamma)
        cols = _madx_chrom_columns(madx)
        qpp = _madx_second_order_chromaticity(madx)

    residual_dispersion = abs(
        cols["ddx"] / _to_madx_pt_convention(point.disp_x, point.ddisp_x, beta0, gamma0) - 1.0
    )
    residual_chromaticity = abs(qpp[0] / second_order_chromaticity(lattice)[0] - 1.0)

    assert residual_dispersion < 1e-6
    assert residual_chromaticity > 1e-2
    assert residual_chromaticity / residual_dispersion > 1e4


def _madx_second_order_chromaticity(madx) -> tuple[float, float]:
    """``Q''`` from MAD-X's own ``Q1``/``Q2`` at three ``DELTAP`` values (M1's method)."""
    tunes = {}
    for sign in (+1, 0, -1):
        madx.input(f"twiss, deltap={sign * DELTA:.12g};")
        tunes[sign] = (float(madx.table.summ.q1[0]), float(madx.table.summ.q2[0]))
    return (
        (tunes[1][0] - 2.0 * tunes[0][0] + tunes[-1][0]) / DELTA**2,
        (tunes[1][1] - 2.0 * tunes[0][1] + tunes[-1][1]) / DELTA**2,
    )
