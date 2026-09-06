r"""Cross-check the survey (R1) against MAD-X's ``SURVEY`` — the second, independent leg.

Marked ``reference``: skips when cpymad is unavailable.

xtrack and accsim share a coordinate convention by design, so a shared convention error
would survive that leg. MAD-X is a separate Fortran implementation, and what it buys here is
the same thing D3 buys everywhere else: the handedness and the sense of ``theta`` have to be
reproduced by a code that does not share a line of source with either of the other two.

**MAD-X names the boundaries the other way round, and that was measured before this file was
written.** Its ``SURVEY`` table opens with a ``$start`` row at the origin and then names each
row after the element it *ends*, closing with a duplicate ``$end``; xtrack names each row
after the element it *begins* and closes with ``_end_point``. Both walk the **same** ``N + 1``
boundaries, which is exactly why :class:`~accsim.geometry.SurveyTable` carries ``N + 1`` poses
with the ``N`` names kept beside them: the only thing to do here is drop MAD-X's trailing
duplicate row.

**What is refused.** MAD-X's ``TILT`` — a *design* tilt, which rolls the reference frame with
the magnet and tips the ring out of the horizontal plane (measured: ``phi = -pi/2`` for a
90-degree bend tilted by ``pi/2``). accsim has no design tilt; its ``roll`` is a
misalignment, which moves the magnet and not the curve. The test below asserts that MAD-X
*does* respond to ``TILT``, so it fails — deliberately — the day accsim grows one and this
milestone's planarity claim stops being true.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))

from _madx import import_madx, madx_session  # noqa: E402
from test_closed_orbit_6d import RING, ring  # noqa: E402
from test_survey import BEND_ANGLES, BEND_LENGTHS, DRIFT_LENGTHS, uneven_ring  # noqa: E402

from accsim.geometry import survey  # noqa: E402


def _madx_survey(sequence: str, name: str = "probe"):
    """``(X, Y, Z, theta, phi, psi)`` from a MAD-X ``SURVEY``, trailing ``$end`` dropped."""
    import_madx()
    with madx_session() as madx:
        madx.input(sequence)
        madx.use(sequence=name)
        madx.survey()
        t = madx.table.survey
        cols = [
            np.array(getattr(t, c), dtype=float) for c in ("x", "y", "z", "theta", "phi", "psi")
        ]
    # The last row repeats the one before it ($end at the same place as the final element's
    # exit), so the boundary sequence is everything but that.
    return tuple(c[:-1] for c in cols)


def _uneven_sequence() -> str:
    """The three-cell ring: three unequal bends per cell, three unequal drifts."""
    defs = []
    for i, (angle, lb, ld) in enumerate(zip(BEND_ANGLES, BEND_LENGTHS, DRIFT_LENGTHS, strict=True)):
        defs.append(f"b{i}: sbend, l={lb!r}, angle={angle!r};")
        defs.append(f"d{i}: drift, l={ld!r};")
    members = ", ".join(f"b{i}, d{i}" for i in range(len(BEND_ANGLES)))
    return (
        "\n".join(defs)
        + f"""
    probe: line=({members});
    beam, particle=electron, energy=1.0;
    use, sequence=probe;
    """
    )


def _i4_sequence() -> str:
    """I4's ring, in the same line-construction form Q1's MAD-X leg uses."""
    cells, focal = RING["cells"], RING["focal"]
    angle = 2.0 * math.pi / (2 * cells)
    return f"""
    b: sbend, l=1.0, angle={angle!r}, k0={angle!r};
    qf: multipole, knl:={{0, {0.5 / focal!r}}};
    qd: multipole, knl:={{0, {-1.0 / focal!r}}};
    rf: rfcavity, l=0.0, volt={RING["voltage"] / 1e6!r}, harmon={RING["harmonic"]}, lag=0.5;
    probe: line=({cells}*(qf, b, qd, b, qf), rf);
    beam, particle=electron, energy={RING["energy"] / 1e9!r};
    use, sequence=probe;
    """


@pytest.fixture(scope="module")
def uneven():
    return survey(uneven_ring()), _madx_survey(_uneven_sequence())


@pytest.fixture(scope="module")
def i4():
    return survey(ring()[0]), _madx_survey(_i4_sequence())


def _compare(mine, theirs, atol: float) -> None:
    """Tolerances are the **measured** residuals with headroom: 4.4e-16 on the asymmetric
    ring and 2.9e-15 on I4's — MAD-X reproduces this geometry to round-off, not to a
    tolerance, which is worth stating because it is a Fortran code that shares nothing with
    either of the other two."""
    X, Y, Z, theta, _phi, _psi = theirs
    assert len(mine) == X.size
    assert np.allclose(mine.X, X, rtol=0.0, atol=atol)
    assert np.allclose(mine.Y, Y, rtol=0.0, atol=atol)
    assert np.allclose(mine.Z, Z, rtol=0.0, atol=atol)
    assert np.allclose(mine.theta, theta, rtol=0.0, atol=atol)


def test_the_asymmetric_ring_agrees_row_for_row(uneven) -> None:
    """Every boundary of the fixture a regular polygon cannot replace."""
    mine, theirs = uneven
    _compare(mine, theirs, atol=1e-14)
    assert np.ptp(theirs[0]) > 2.0


def test_the_handedness_and_the_sense_of_theta_are_madx_s_too(uneven) -> None:
    """A positive bend goes to negative ``X``; ``theta`` runs the other way, unwrapped."""
    mine, theirs = uneven
    X, _Y, _Z, theta, _phi, _psi = theirs
    assert mine.X[1] < 0.0 and X[1] < 0.0
    assert theta[-1] == pytest.approx(-2.0 * math.pi, rel=0.0, abs=1e-12)
    assert mine.theta[-1] == pytest.approx(theta[-1], rel=0.0, abs=1e-12)


def test_the_i4_ring_agrees_element_by_element(i4) -> None:
    """101 elements including 60 zero-length quadrupoles that must advance nothing."""
    mine, theirs = i4
    _compare(mine, theirs, atol=1e-13)
    assert np.ptp(theirs[2]) > 10.0


def test_madx_is_planar_here_and_a_design_tilt_is_what_would_break_it() -> None:
    """The refusal, as a test with a consequence.

    Without a ``TILT`` MAD-X reports ``Y = phi = psi = 0``, which is what makes accsim's
    planar-by-construction survey a complete answer rather than a projection of one. Give the
    same bend a design tilt and MAD-X tips the ring straight out of the plane — so if accsim
    ever gains a design tilt, the first assertion here is the one that stops being enough.
    """
    _X, Y, _Z, _theta, phi, psi = _madx_survey(_uneven_sequence())
    assert np.allclose(Y, 0.0, rtol=0.0, atol=1e-15)
    assert np.allclose(phi, 0.0, rtol=0.0, atol=1e-15)
    assert np.allclose(psi, 0.0, rtol=0.0, atol=1e-15)

    tilted = _madx_survey(f"""
    b: sbend, l=1.0, angle={math.pi / 2!r}, tilt={math.pi / 2!r};
    probe: line=(b);
    beam, particle=electron, energy=1.0;
    use, sequence=probe;
    """)
    assert abs(tilted[1][-1]) > 0.6  # Y — the ring has left the horizontal plane
    assert tilted[4][-1] == pytest.approx(-math.pi / 2, rel=0.0, abs=1e-9)  # phi
