r"""Cross-check the taper profile (Q1) against xtrack's ``delta_taper``.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

xtrack computes the same object by the same route — track one turn with the cavity kick
killed, average each element's entrance and exit momenta, and shift the start until the
profile's mean over ``s`` is zero (``delta0='zero_mean'``, a documented argument that can be
given a number instead). So this is a comparison of two implementations of one convention,
and B2's rule governs it: ``integrator="uniform"`` and ``num_multipole_kicks=1`` on every
bend, or the two codes are integrating different maps and the disagreement is ``7/8`` of the
loss rather than anything to do with tapering.

**What agrees and to what.** The sawtooth's *height* agrees to ``6.4e-6`` relative — a
tighter number than the milestone's analytic span gate, because both sides are tracked
rather than one being a design-route integral. Its *centring* agrees to round-off on both
sides independently, with **no** constant component between the profiles (``1.7e-9`` of the
span over the radiating elements, ``2.3e-7`` over all of them).

**The element-by-element disagreement is a law, not a tolerance, and that is the point of
this file.** It is ``0.1446 span^2`` — *second* order in the sag — measured to ``0.5%``
across a factor eight in span, and three quarters of it is a parabola in ``s`` whose sagitta
is ``0.0946 span^2``. That shape is *consistent with* a difference in where along the element
the loss is evaluated, the radiated power following the local ``E^2`` — inferred from the
shape and the scaling rather than localised, so it is not what the gate rests on. What the
gate rests on is the order: this vanishes quadratically as the ring stops radiating, which a
genuine coefficient error would not. Two
energies are run for exactly that reason — a single one could only ever be a tolerance.

**The control matters more than usual here.** The span is fixed by conservation, so a
profile that is wrong everywhere in the same way still has the right span; the gate that
discriminates is the element-by-element one, and the deliberate break below (an uncentred
profile) misses it by half the span — a thousand times the agreement.
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

from accsim.tapering import taper_profile  # noqa: E402

#: ``max|accsim - xtrack| = SHAPE_LAW * span^2``. Constant to 0.5% over a factor 8 in span.
SHAPE_LAW = 0.1446


def _line(energy: float = RING["energy"], voltage: float = RING["voltage"]):
    """The I4 ring in xtrack, with a real RF frequency.

    I4's own cross-check builds this line with ``frequency=1.0`` — it only ever tracks, and
    a near-DC cavity is harmless there. The tapering routine runs a full ``twiss``, and a
    ring whose synchrotron tune is ~0 has no 6D closed orbit to find: it raises
    ``ClosedOrbitSearchError``. So the frequency is the ring's own, ``h beta0 c / C``, which
    is what :meth:`RFCavity.from_harmonic` gives the accsim side.
    """
    from scipy.constants import c as clight

    cells, focal = RING["cells"], RING["focal"]
    angle = 2.0 * math.pi / (2 * cells)
    elements, names = [], []
    for i in range(cells):
        for j, strength in enumerate((0.5 / focal, angle, -1.0 / focal, angle, 0.5 / focal)):
            if j % 2:
                elements.append(
                    xt.Bend(
                        length=1.0,
                        angle=strength,
                        k0=strength,
                        model="bend-kick-bend",
                        integrator="uniform",
                        num_multipole_kicks=1,
                    )
                )
            else:
                elements.append(xt.Multipole(knl=[0.0, strength], length=0.0))
            names.append(f"e{i}_{j}")
    elements.append(
        xt.Cavity(voltage=voltage, frequency=RING["harmonic"] * clight / (2.0 * cells), lag=180.0)
    )
    names.append("cav")

    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(mass0=ELECTRON_MASS_EV, q0=1.0, energy0=energy)
    line.build_tracker()
    return line


@pytest.fixture(scope="module")
def theirs() -> np.ndarray:
    """xtrack's ``delta_taper``, one per magnet, after its own compensation converges."""
    line = _line()
    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss(verbose=False)
    return np.array([getattr(line[n], "delta_taper", np.nan) for n in line.element_names[:-1]])


@pytest.fixture(scope="module")
def mine():
    return taper_profile(ring()[0])


def test_the_sawtooth_has_the_same_height_in_both_codes(mine, theirs) -> None:
    """Peak to peak, ``6.4e-6`` relative — both sides tracked, neither an integral."""
    span_theirs = float(np.nanmax(theirs) - np.nanmin(theirs))
    span_mine = float(mine.delta[: len(theirs)].max() - mine.delta[: len(theirs)].min())
    assert span_mine == pytest.approx(span_theirs, rel=1e-5)


def test_both_codes_centre_the_profile_on_the_design_momentum(mine, theirs) -> None:
    """Independently zero-mean: accsim's own mean and xtrack's bend mean are both round-off.

    xtrack averages with ``trapz`` over ``s`` and accsim with element lengths — the same
    integral written twice — so this is a convention check as well as a numerical one.
    """
    bends = ~np.isnan(theirs) & (mine.lengths[: len(theirs)] > 0.0)
    span = mine.span
    assert abs(mine.mean) < 1e-11 * span
    assert abs(float(np.mean(theirs[bends]))) < 1e-8 * span


def test_the_profiles_agree_element_by_element(mine, theirs) -> None:
    """The gate that discriminates, and it is not the span.

    The two codes place the ramp identically and distribute the loss along it slightly
    differently. "Identically" is asserted where it is sharpest — over the **radiating**
    elements, where the constant component is ``1.7e-9`` of the span, five orders below the
    shape term. Over all elements it is ``2.3e-7``, because the thin quadrupoles sample the
    ramp at its steps rather than across them and so weight the two codes' step placement
    differently; that is still three orders below the shape term, and both are gated so the
    difference between them cannot be mistaken for slack.
    """
    ours = mine.delta[: len(theirs)]
    keep = ~np.isnan(theirs)
    radiating = keep & (mine.lengths[: len(theirs)] > 0.0)
    gap = ours[keep] - theirs[keep]
    offset = float(np.mean(gap))

    assert abs(float(np.mean(ours[radiating] - theirs[radiating]))) / mine.span < 1e-8
    assert abs(offset) / mine.span < 1e-6
    assert np.max(np.abs(gap)) / mine.span**2 == pytest.approx(SHAPE_LAW, rel=0.02)


@pytest.mark.parametrize(("energy", "voltage"), [(3.25e9, 400.0e6), (6.5e9, 90.0e6)])
def test_the_disagreement_is_second_order_in_the_sag(energy: float, voltage: float) -> None:
    """``max|gap| = 0.1446 span^2`` across a factor eight in the span, and 3/4 a parabola.

    This is the difference between a coefficient error and a model difference. A wrong
    coefficient in the radiation kick would scale with the span itself — first order — and
    would survive the ring's loss going to zero. This vanishes quadratically, which is what
    evaluating the same ``E^2`` power law at a slightly different point along each magnet
    produces, and it is the same family of per-element gap B2 measured and owns.
    """
    line = _line(energy, voltage)
    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss(verbose=False)
    theirs = np.array([getattr(line[n], "delta_taper", np.nan) for n in line.element_names[:-1]])

    profile = taper_profile(ring(energy=energy, voltage=voltage)[0])
    keep = ~np.isnan(theirs)
    gap = profile.delta[: len(theirs)][keep] - theirs[keep]
    s_mid = profile.s_mid[: len(theirs)][keep]

    assert np.max(np.abs(gap)) / profile.span**2 == pytest.approx(SHAPE_LAW, rel=0.02)

    curve = np.polyfit(s_mid, gap, 2)
    sagitta = abs(curve[0]) * (s_mid[-1] - s_mid[0]) ** 2 / 8.0
    assert sagitta / profile.span**2 == pytest.approx(0.0946, rel=0.02)


def test_an_uncentred_profile_misses_xtrack_by_half_the_span(theirs) -> None:
    """The control: the deliberate break is a thousand times the agreement above.

    Without this the element-by-element gate would only be showing that two ramps of the
    same height are the same ramp, which they need not be.
    """
    raw = taper_profile(ring()[0], delta0=0.0)
    ours = raw.delta[: len(theirs)]
    keep = ~np.isnan(theirs)
    gap = ours[keep] - theirs[keep]
    assert np.max(np.abs(gap)) / raw.span == pytest.approx(0.5, rel=0.02)


def test_the_untapered_ring_carries_the_seven_millimetre_orbit(mine) -> None:
    """The profile's consequence, in the reference code and in this one.

    xtrack's radiation twiss puts the closed orbit at ``7.0898e-3``; accsim's own 6D closed
    orbit is gated on the same number in ``tests/analytic/test_tapering.py``, and MAD-X's
    ``TWISS`` with ``radiate=true`` gives ``7.0882e-3``. Three codes, one orbit, and it is
    what Q2 removes.
    """
    line = _line()
    line.configure_radiation(model="mean")
    tw = line.twiss(method="6d")
    assert float(np.max(np.abs(tw.x))) == pytest.approx(7.0898e-3, rel=2e-3)
