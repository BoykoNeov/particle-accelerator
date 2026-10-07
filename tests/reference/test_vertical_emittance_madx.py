r"""V2 — vertical emittance from design ``D_y`` against MAD-X: one arbiter, and its reach.

Marked ``reference``: skips when cpymad is unavailable.

MAD-X can arbitrate **no** vertical radiation integral: ``TWISS`` tabulates only the
horizontal ``synch_1..5`` (its ``synch_6``/``synch_8`` are horizontal quantities, nonzero on
a flat ring). What it has is ``EMIT``, the eigen-analysis of the radiating one-turn map, and
on the dogleg ring that departs from accsim's integral as ``c (2 pi Q_s)^2`` with the same
``c`` accsim's tracked Lyapunov equilibrium shows — the gate here.

Its **reach is stated, not implied**, because the filter run's "MAD-X ``EMIT`` and xtrack
agree to ``9e-5``" invited reading more into it than is there:

- that agreement was at ``Q_s = 0.087``, where both sit ``14%`` above the integral;
- on a ring turned over by ``pi/2`` — the same machine standing on its side — ``EMIT``'s
  ``ey`` is ``1.9%`` of the flat ring's ``ex``, and ``TWISS``'s ``synch_1`` is ``10%`` of its
  own ``alfa * C``. Both are asserted below as measured facts, written to fail the day MAD-X
  changes. Where between the dogleg's ``0.02`` rad and a full ``pi/2`` ring this sets in was
  not swept.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))

import test_vertical_emittance as v2  # noqa: E402
from _madx import import_madx, madx_session  # noqa: E402
from test_design_tilt import ANG, VA, dogleg_ring  # noqa: E402

from accsim.radiation import equilibrium_vertical_emittance  # noqa: E402

PI2 = math.pi / 2
#: RF voltages of the scan [MV], harmonic 10 — the same three Q_s as the analytic gate.
_SCAN = (3.78, 3.9, 4.5)


def _sequence(*, dogleg: bool, tilt: float = 0.0, volt: float | None = None) -> str:
    tl = f", tilt={tilt!r}" if tilt else ""
    body = "cell, vb1, dd, vb2, 3*cell" if dogleg else "4*cell"
    rf = ""
    if volt is not None:
        rf = f"rf: rfcavity, l=0, volt={volt!r}, harmon=10, lag=0.5;\n"
        body += ", rf"
    return f"""
    qf: quadrupole, l=0.5, k1=0.30{tl};
    qd: quadrupole, l=0.5, k1=-0.30{tl};
    b:  sbend, l=1.5, angle={ANG!r}{tl};
    d:  drift, l=0.6;
    vb1: sbend, l=0.3, angle={VA!r}, tilt={PI2!r};
    vb2: sbend, l=0.3, angle={-VA!r}, tilt={PI2!r};
    dd: drift, l=1.0;
    cell: line=(qf, d, b, d, qd, d, b, d);
    {rf}probe: line=({body});
    beam, particle=electron, energy=3.0{", radiate=true" if volt is not None else ""};
    """


def _emit(**kw) -> dict[str, float]:
    import_madx()
    with madx_session() as madx:
        madx.input(_sequence(**kw))
        madx.use(sequence="probe")
        madx.input("emit, deltap=0;")
        es = madx.table.emitsumm
        return {c: float(es[c][0]) for c in ("ex", "ey", "qs")}


def _summ(**kw) -> dict[str, float]:
    import_madx()
    with madx_session() as madx:
        madx.input(_sequence(**kw))
        madx.use(sequence="probe")
        madx.input("twiss, chrom;")  # synch_* are filled only with chrom
        su = madx.table.summ
        return {c: float(su[c][0]) for c in ("alfa", "length", "synch_1")}


def test_emit_departs_with_accsims_tracked_slope() -> None:
    """``EMIT``'s ``ey`` against the integral, over three ``Q_s``: a line in ``(2 pi Q_s)^2``
    whose slope is accsim's tracked one. The intercepts are not compared (CONVENTIONS ->
    *Vertical emittance*: accsim's carries the energy sag, ``-U0/E``, and MAD-X's
    ``+9e-4`` is its own)."""
    eps = equilibrium_vertical_emittance(dogleg_ring())
    runs = [_emit(dogleg=True, volt=v) for v in _SCAN]
    xs = [(2 * math.pi * r["qs"]) ** 2 for r in runs]
    ds = [r["ey"] / eps - 1.0 for r in runs]
    slope, _ = np.polyfit(xs, ds, 1)
    _, slope_accsim, _ = v2._departure_law(3.0e9)
    assert slope == pytest.approx(slope_accsim, rel=5e-2)
    # ...and at the lowest Q_s, EMIT is already within 0.5% of the integral.
    assert abs(ds[0]) < 5e-3


def test_emit_does_not_turn_a_ring_over() -> None:
    """A measured limit of the arbiter, not an accsim claim: the flat ring stood on its side
    is the same machine, yet ``EMIT`` gives it a vertical emittance ``~50x`` smaller than the
    flat ring's horizontal one. accsim's identity holds to ``1e-12``
    (``test_vertical_emittance.py``) and so does xtrack's eigen route."""
    flat = _emit(dogleg=False, volt=8.0)
    rolled = _emit(dogleg=False, tilt=PI2, volt=8.0)
    assert rolled["ey"] / flat["ex"] < 0.1


def test_twiss_radiation_integrals_ignore_the_tilt() -> None:
    """``synch_1`` is ``∮ h D_x`` and should equal ``alfa * C``; on the flat ring it does (to
    MAD-X's own 0.4%), on the same ring turned over it is ``10%`` of it, while ``alfa`` — read
    off the map — is unchanged. So no ``synch_*`` arbitrates a tilted ring."""
    flat, rolled = _summ(dogleg=False), _summ(dogleg=False, tilt=PI2)
    assert flat["synch_1"] / (flat["alfa"] * flat["length"]) == pytest.approx(1.0, rel=1e-2)
    assert rolled["alfa"] == pytest.approx(flat["alfa"], rel=1e-9)
    assert rolled["synch_1"] / (rolled["alfa"] * rolled["length"]) < 0.2
