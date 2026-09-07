r"""U1 against MAD-X PTC: the path-length series a second, independent code computes.

PTC's ``ptc_twiss`` emits the higher-order momentum compaction *directly*, as
``alpha_c_p`` and ``alpha_c_p2`` in its summary table. That makes it the only external
check this milestone has — and the checks below are as much about **decoding** it as about
comparing to it, because MAD-X ships two answers to this question and they disagree.

**MAD-X contradicts itself, and the disagreement is not small.** On the fixture ring
*with sextupoles*:

* differencing ``summ.alfa`` across a ``twiss, deltap=`` scan gives ``+4.156e-01``;
* PTC's own ``alpha_c_p``, from the same MAD-X session, gives ``-3.479e-03``.

Opposite signs, two orders of magnitude apart. (Without sextupoles both are positive and
the gap is a factor of 16 — the magnitude split is the general statement, the sign flip is
this ring's.) Scanning ``deltap`` **re-references the
machine** rather than moving a particle along its off-momentum closed orbit, so
``summ.alfa`` at ``deltap`` is the compaction of a different reference momentum — a
perfectly good number, and not this one. The same is true of scanning ``xtrack``'s
``twiss(delta0=...)``, which gives a third value again (``+1.177e-02``). ``alpha_c_p`` is
the one that means what :func:`~accsim.twiss.momentum_compaction_series` means, and
:func:`test_madx_twiss_scan_is_a_different_quantity` pins that split so a future session
does not read it as a bug in accsim.

**The factor is measured, not recalled.** ``alpha_c_p`` is ``2 alpha_1`` and
``alpha_c_p2`` is ``6 alpha_2`` — the derivatives of the *local* compaction, where accsim
ships the Taylor coefficients of the path length. Both factors are calibrated below on a
**sextupole-free** ring, where accsim and PTC implement the same maps and the comparison
has nothing else in it. This is the same trap as PTC's ``anhx`` being ``dQ/d(2J)``.

**The arbiter's reach is stated, not implied.** ``alpha_c_p`` and ``alpha_c_p2`` appear
only under ``icase=56`` *and* ``deltap_dependency``; with ``icase=5`` or ``icase=6`` all
three come back as the ``-1e6`` "not computed" sentinel. And ``alpha_c_p3`` is that
sentinel even at ``no=3``, so **PTC reaches second order in delta and no further** —
``alpha_2`` is the last coefficient anything outside accsim can see.

With sextupoles the two codes part company by ``0.086%``, and that gap is **not**
integration error (PTC is unchanged from ``nst=5`` to ``nst=80``): it is accsim's
single-slice thick sextupole against PTC's converged one, and it closes as ``1/n^2`` when
``Sextupole(n_slices=n)`` is refined — the same second-order-integrator law P2 (ii)
established.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _madx import madx_session

from accsim import (
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    Sextupole,
    momentum_compaction,
    momentum_compaction_series,
)
from accsim.reference import PROTON_MASS_EV

pytestmark = pytest.mark.reference

ENERGY_GEV = 10.0
ANGLE = 2.0 * math.pi / 8.0
"""Eight sector bends closing the ring; four FODO cells."""


def _lattice(k2: float, n_slices: int = 1) -> Lattice:
    ref = ReferenceParticle.from_total_energy(PROTON_MASS_EV, ENERGY_GEV * 1e9, charge=1.0)
    elements = []
    for _ in range(4):
        elements += [
            Quadrupole(0.5, 0.30),
            Drift(0.6),
            Sextupole(0.2, k2, n_slices=n_slices),
            Dipole(1.5, ANGLE),
            Drift(0.6),
            Quadrupole(0.5, -0.30),
            Drift(0.6),
            Sextupole(0.2, -k2, n_slices=n_slices),
            Dipole(1.5, ANGLE),
            Drift(0.6),
        ]
    return Lattice(elements, ref)


def _sequence(k2: float) -> str:
    """The same ring in MAD-X input, element for element."""
    return f"""
    qf: quadrupole, l=0.5, k1=0.30; qd: quadrupole, l=0.5, k1=-0.30;
    sf: sextupole, l=0.2, k2={k2};  sd: sextupole, l=0.2, k2={-k2};
    b: sbend, l=1.5, angle={ANGLE}; d: drift, l=0.6;
    cell: line=(qf,d,sf,b,d,qd,d,sd,b,d);
    ring: line=(4*cell);
    beam, particle=proton, energy={ENERGY_GEV};
    use, sequence=ring;
    """


def _ptc_summary(madx, k2: float, *, icase: int = 56, no: int = 3, nst: int = 20) -> dict:
    """``ptc_twiss``'s summary row for the ring, as a plain dict."""
    madx.input(_sequence(k2))
    madx.input(
        f"""
        ptc_create_universe;
        ptc_create_layout, model=2, method=6, nst={nst}, exact=true;
        ptc_twiss, closed_orbit, icase={icase}, no={no}, deltap_dependency, summary_table=ps;
        ptc_end;
        """
    )
    table = madx.table.ps
    return {c: float(table[c][0]) for c in ("alpha_c", "alpha_c_p", "alpha_c_p2", "alpha_c_p3")}


PTC_SENTINEL = -1e6
"""What PTC writes when it did not compute a coefficient."""


def test_ptc_alpha_c_matches_the_first_order_compaction() -> None:
    """The two codes agree on ``alpha_0`` to 11 digits — the baseline the rest rests on."""
    with madx_session() as madx:
        ptc = _ptc_summary(madx, k2=0.0)
    assert ptc["alpha_c"] == pytest.approx(momentum_compaction(_lattice(0.0)), rel=1e-11)


def test_ptc_alpha_c_p_is_twice_alpha_1() -> None:
    """The factor of two, calibrated where the two codes implement the same maps.

    Sextupole-free on purpose: with sextupoles the comparison also carries the difference
    between a single-kick body and a converged one, which is a separate finding
    (:func:`test_sextupole_gap_closes_as_the_slice_law`).
    """
    series = momentum_compaction_series(_lattice(0.0), method="map")
    with madx_session() as madx:
        ptc = _ptc_summary(madx, k2=0.0)
    assert ptc["alpha_c_p"] == pytest.approx(2.0 * series.alpha_1, rel=1e-5)
    # ... and the factor really is two: alpha_1 alone would miss by a clear factor.
    assert abs(ptc["alpha_c_p"] / series.alpha_1 - 2.0) < 1e-4


def test_ptc_alpha_c_p2_is_six_times_alpha_2() -> None:
    """``alpha_c_p2 = d^2 alpha_c / ddelta^2 = 6 alpha_2`` — measured, not assumed.

    Six rather than two because the third Taylor coefficient of ``C(delta)`` appears in the
    *second* derivative of the local slope with a factor ``3 * 2``.
    """
    series = momentum_compaction_series(_lattice(0.0), method="tracked", order=3)
    assert series.alpha_2 is not None
    with madx_session() as madx:
        ptc = _ptc_summary(madx, k2=0.0)
    assert ptc["alpha_c_p2"] == pytest.approx(6.0 * series.alpha_2, rel=1e-3)


def test_ptc_reaches_second_order_and_says_so() -> None:
    """``alpha_c_p3`` is the sentinel even at ``no=3``: nothing outside accsim sees ``alpha_3``.

    And the two coefficients that *are* emitted need both ``icase=56`` and
    ``deltap_dependency`` — with ``icase=5`` PTC silently returns the sentinel for all
    three, which is what a comparison written from the manual would have compared against.
    """
    with madx_session() as madx:
        full = _ptc_summary(madx, k2=0.0, icase=56)
        assert full["alpha_c_p"] != pytest.approx(PTC_SENTINEL)
        assert full["alpha_c_p2"] != pytest.approx(PTC_SENTINEL)
        assert full["alpha_c_p3"] == pytest.approx(PTC_SENTINEL)

        narrow = _ptc_summary(madx, k2=0.0, icase=5)
        for key in ("alpha_c_p", "alpha_c_p2", "alpha_c_p3"):
            assert narrow[key] == pytest.approx(PTC_SENTINEL)
        # icase=5 still gets alpha_c itself right, so the sentinel is the only signal.
        assert narrow["alpha_c"] == pytest.approx(full["alpha_c"], rel=1e-12)


@pytest.mark.parametrize("k2", [0.0, 0.5])
def test_madx_twiss_scan_is_a_different_quantity(k2: float) -> None:
    """MAD-X's own two answers disagree, and on the sextupole ring they disagree in *sign*.

    Nothing about accsim is asserted here. The point is entirely about the arbiter: a
    session that reaches for ``twiss, deltap=`` and differences ``summ.alfa`` gets a number
    that is not ``2 alpha_1`` and never converges to it, so the disagreement must not be
    read as a bug to hunt.

    The magnitude gap is the part that holds on **both** rings — a factor of about 16
    without sextupoles and 120 with them. The sign flip is a property of the sextupole ring
    alone (there ``alpha_c_p`` is negative while the scan stays positive), which is why it
    is asserted only there rather than claimed as a general rule.
    """
    with madx_session() as madx:
        madx.input(_sequence(k2))
        alfa = {}
        for deltap in (-1e-3, 1e-3):
            madx.input(f"twiss, deltap={deltap};")
            alfa[deltap] = madx.table.summ.alfa[0]
        scan_slope = (alfa[1e-3] - alfa[-1e-3]) / 2e-3
        ptc = _ptc_summary(madx, k2=k2)

    assert abs(scan_slope) > 10.0 * abs(ptc["alpha_c_p"])
    if k2 != 0.0:
        assert scan_slope * ptc["alpha_c_p"] < 0.0


def test_sextupole_gap_closes_as_the_slice_law() -> None:
    """With sextupoles the codes differ by 0.09%, and it is accsim's single-slice body.

    Not PTC's integration: its ``alpha_c_p`` is unchanged from ``nst=5`` to ``nst=80``
    (asserted below), so the moving part is on accsim's side. Refining ``n_slices`` closes
    the gap as ``1/n^2`` — the second-order-integrator law, gated on the *order* rather
    than on a tolerance.
    """
    with madx_session() as madx:
        coarse = _ptc_summary(madx, k2=0.5, nst=5)
        fine = _ptc_summary(madx, k2=0.5, nst=80)
    # PTC has converged: the gap is not its quadrature.
    assert fine["alpha_c_p"] == pytest.approx(coarse["alpha_c_p"], rel=1e-9)
    target = fine["alpha_c_p"]

    gaps = []
    for n_slices in (1, 2, 4):
        series = momentum_compaction_series(_lattice(0.5, n_slices=n_slices), method="tracked")
        gaps.append(2.0 * series.alpha_1 - target)

    # Single-slice: a real disagreement, ~0.09% -- large enough that the law has something
    # to close, small enough that it is plainly a body model and not a wrong coefficient.
    assert 3e-4 < abs(gaps[0] / target) < 3e-3
    # ... and each doubling divides it by four.
    for coarse_gap, fine_gap in zip(gaps[:-1], gaps[1:], strict=True):
        assert coarse_gap / fine_gap == pytest.approx(4.0, rel=0.15)


def test_ptc_agrees_on_alpha_1_once_the_body_is_refined() -> None:
    """The endpoint of the slice law: a refined sextupole lands on PTC's number."""
    series = momentum_compaction_series(_lattice(0.5, n_slices=64), method="tracked")
    with madx_session() as madx:
        ptc = _ptc_summary(madx, k2=0.5, nst=80)
    assert 2.0 * series.alpha_1 == pytest.approx(ptc["alpha_c_p"], rel=1e-5)


def test_alpha_0_is_untouched_by_any_of_this() -> None:
    """A control: the first-order number does not move with the sextupole slicing.

    ``alpha_0`` is linear and a sextupole contributes nothing to it on the design orbit, so
    a slice count that changed it would mean the refinement was altering the lattice rather
    than integrating it better.
    """
    reference = momentum_compaction(_lattice(0.5, n_slices=1))
    for n_slices in (2, 8, 64):
        assert momentum_compaction(_lattice(0.5, n_slices=n_slices)) == pytest.approx(
            reference, rel=1e-12
        )
    assert reference == pytest.approx(momentum_compaction(_lattice(0.0)), rel=1e-12)


def test_series_alpha_0_matches_ptc_on_the_sextupole_ring() -> None:
    """Both routes' ``alpha_0`` against PTC on the full ring, as a closing control."""
    lattice = _lattice(0.5)
    with madx_session() as madx:
        ptc = _ptc_summary(madx, k2=0.5)
    for method in ("map", "tracked"):
        series = momentum_compaction_series(lattice, method=method)
        assert series.alpha_0 == pytest.approx(ptc["alpha_c"], rel=1e-9)
    assert np.isfinite(ptc["alpha_c_p"])
