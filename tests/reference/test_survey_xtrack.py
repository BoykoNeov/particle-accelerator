r"""Cross-check the survey (R1) against xtrack's ``line.survey()``.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**What this leg proves that the analytic file cannot.** ``tests/analytic/test_survey.py``
pins the walk against a closed form and against a second walk written in the complex plane —
but both of those are *this* project's reading of what a survey is. xtrack's is an
independent implementation of the same object, and it fixes the two things a self-consistent
walk can still get wrong: the **handedness** (a positive bend angle moves the machine towards
negative ``X``) and the **sense of ``theta``** (it runs the other way from the accumulated
bend, unwrapped past ``-2 pi``).

**The two codes walk the same boundaries and only name them differently.** xtrack's table has
``N + 1`` rows for ``N`` elements, naming each row after the element it *begins* and calling
the last one ``_end_point``; accsim's table is the same ``N + 1`` poses with the ``N`` names
kept alongside. So this is a row-for-row comparison with no offset — which is the whole
reason the table was built that way, and it was measured before it was decided.

**The fixture is three identical cells of three different bends.** A regular polygon would
agree with a walker that composed its rotations in the wrong order; this one does not, and
the three-fold symmetry is there because position closure needs it — angles summing to
``2 pi`` close the direction only.

**Cost.** An ``xt.Line`` build plus its first ``survey()`` costs 30-70 s on this machine
(measured 2026-09-06, well above the 12 s on record for a plain build), so both lines here
are module-scoped fixtures built exactly once and every claim is read off the two tables
they produce. See ``docs/CONVENTIONS.md`` -> *Test-suite cost*.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))

from test_closed_orbit_6d import ELECTRON_MASS_EV, RING, ring  # noqa: E402
from test_survey import BEND_ANGLES, BEND_LENGTHS, DRIFT_LENGTHS, uneven_ring  # noqa: E402

from accsim.geometry import survey  # noqa: E402


def _uneven_line():
    """The uneven ring as an ``xt.Line``. ``k0`` is set from the geometry — see below."""
    elements, names = [], []
    for i, (angle, lb, ld) in enumerate(zip(BEND_ANGLES, BEND_LENGTHS, DRIFT_LENGTHS, strict=True)):
        elements.append(xt.Bend(length=lb, angle=angle, k0=angle / lb))
        names.append(f"b{i}")
        elements.append(xt.Drift(length=ld))
        names.append(f"d{i}")
    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(mass0=ELECTRON_MASS_EV, q0=1.0, energy0=1.0e9)
    return line


def _i4_line():
    """I4's ring in xtrack — the realistic leg: 40 bends, 60 thin quadrupoles, a cavity.

    Built without a tracker: this file only ever surveys, and a survey needs no map. The
    element mix is Q1's, thin quadrupoles included, because a zero-length element advancing
    the curve by nothing is exactly the case a boundary-row table has to get right.
    """
    cells, focal = RING["cells"], RING["focal"]
    angle = 2.0 * math.pi / (2 * cells)
    elements, names = [], []
    for i in range(cells):
        for j, strength in enumerate((0.5 / focal, angle, -1.0 / focal, angle, 0.5 / focal)):
            if j % 2:
                elements.append(xt.Bend(length=1.0, angle=strength, k0=strength))
            else:
                elements.append(xt.Multipole(knl=[0.0, strength], length=0.0))
            names.append(f"e{i}_{j}")
    elements.append(xt.Cavity(voltage=RING["voltage"], frequency=1.0, lag=180.0))
    names.append("cav")
    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(mass0=ELECTRON_MASS_EV, q0=1.0, energy0=RING["energy"])
    return line


@pytest.fixture(scope="module")
def uneven():
    """``(accsim table, xtrack table)`` for the asymmetric ring — built once."""
    return survey(uneven_ring()), _uneven_line().survey()


@pytest.fixture(scope="module")
def i4():
    """``(accsim table, xtrack table)`` for I4's 6.5 GeV ring — built once."""
    return survey(ring()[0]), _i4_line().survey()


def _compare(mine, theirs, atol: float) -> None:
    """Tolerances are the **measured** residuals with a factor ~30 of headroom, not round
    numbers: 4.4e-16 on the asymmetric ring and 3.6e-15 on I4's, against a 12.7 m extent."""
    assert len(mine) == len(theirs.X)
    assert np.allclose(mine.X, theirs.X, rtol=0.0, atol=atol)
    assert np.allclose(mine.Y, theirs.Y, rtol=0.0, atol=atol)
    assert np.allclose(mine.Z, theirs.Z, rtol=0.0, atol=atol)
    assert np.allclose(mine.theta, theirs.theta, rtol=0.0, atol=atol)


def test_the_asymmetric_ring_agrees_row_for_row(uneven) -> None:
    """Every boundary, both codes, no offset — the fixture a square cannot replace."""
    mine, theirs = uneven
    _compare(mine, theirs, atol=1e-14)
    # The rows really are spread out: this is not agreement at a single point.
    assert np.ptp(np.asarray(theirs.X)) > 2.0


def test_the_handedness_is_xtracks(uneven) -> None:
    """A positive bend angle moves the machine towards **negative** ``X`` in both codes.

    The one statement a self-consistent walk cannot check on itself: flip the sign and every
    analytic gate in this milestone still passes.
    """
    mine, theirs = uneven
    assert mine.X[1] < 0.0
    assert float(theirs.X[1]) < 0.0
    assert mine.X[1] == pytest.approx(float(theirs.X[1]), rel=0.0, abs=1e-14)


def test_theta_runs_against_the_bend_and_is_not_wrapped(uneven) -> None:
    """Both codes end a full turn at ``-2 pi``, not at ``0`` and not at ``+2 pi``."""
    mine, theirs = uneven
    assert mine.theta[-1] == pytest.approx(-2.0 * math.pi, rel=0.0, abs=1e-14)
    assert float(theirs.theta[-1]) == pytest.approx(-2.0 * math.pi, rel=0.0, abs=1e-12)


def test_the_i4_ring_agrees_element_by_element(i4) -> None:
    """101 elements — 40 bends, 60 zero-length quadrupoles and a cavity — around 40 m."""
    mine, theirs = i4
    _compare(mine, theirs, atol=1e-13)
    # The realistic fixture is large, so the agreement above is relative to a big number.
    assert np.ptp(np.asarray(theirs.Z)) > 10.0


def test_xtrack_is_planar_here_too(i4, uneven) -> None:
    """``Y`` is zero in xtrack's own table, which is what makes accsim's refusal safe.

    accsim reports ``Y = phi = psi = 0`` by construction because it has no design tilt. That
    is only *equivalent* to xtrack's answer as long as xtrack agrees the machine is flat —
    it does, and this asserts it rather than assuming it.
    """
    for _, theirs in (i4, uneven):
        assert np.allclose(np.asarray(theirs.Y), 0.0, rtol=0.0, atol=1e-15)
        assert np.allclose(np.asarray(theirs.phi), 0.0, rtol=0.0, atol=1e-15)
        assert np.allclose(np.asarray(theirs.psi), 0.0, rtol=0.0, atol=1e-15)
