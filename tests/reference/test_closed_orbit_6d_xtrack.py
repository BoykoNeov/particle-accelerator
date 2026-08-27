r"""Cross-check the 6D closed orbit (I4) against xtrack.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

xtrack finds its own 6D closed orbit by **tracking** the line (``twiss`` ->
``find_closed_orbit_line`` -> ``_one_turn_map`` -> ``line.track``), so the element
settings that govern its tracker govern this comparison too, and B2's rule applies
unchanged: set ``integrator="uniform"`` and ``num_multipole_kicks=1`` on every bend or the
two codes are integrating different maps. See ``test_radiation_tracking_xtrack.py`` for
the diagnosis.

**The pre-commitment, written into ``docs/ROADMAP.md`` -> I4 before this file was run, and
it held to five digits.** With the integration order matched, the whole disagreement
between the two codes' closed orbits should be the residual B2 already owns — xtrack's
pre-2019 elementary charge (``1.0639e-8``, a constants vintage that lands on ``C_gamma``
and so on the loss) plus the ``2/gamma0^2`` of its ultra-relativistic approximations — and
nothing else. Measured on the reconstructed loss: ``2.29997e-8`` against a predicted
``2.29997e-8``.

**The comparison is made on the loss, not on ``zeta_co``.** They are the same statement
through ``V sin(k zeta_co) = U``, but the arcsine folds a relative disagreement in ``U``
into a *different* relative disagreement in ``zeta`` by ``tan(k zeta)/(k zeta)`` — 1.027 on
this ring, and ring-dependent. Comparing the losses removes a factor nobody would think to
divide out; the factor itself is then asserted separately, so the choice is gated rather
than merely made.

**A third, independent confirmation falls out.** xtrack reports its own energy loss per
turn as ``twiss.energy_loss``, computed inside xtrack from its own radiation bookkeeping
and never from the closed orbit. Reconstructing that number from **xtrack's own**
``zeta`` through accsim's closed form reproduces it to ``2e-10`` — so the closed form this
milestone is built on is not an accsim convention, and holds inside the reference code as
well.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))

from test_closed_orbit_6d import (  # noqa: E402
    ELECTRON_MASS_EV,
    RING,
    at_cavity,
    cavity_kick_eV,
    ring,
)

from accsim.coords import ZETA  # noqa: E402
from accsim.orbit import closed_orbit_6d  # noqa: E402

#: B2's two named owners of the accsim/xtrack radiation gap, both on xtrack's side: the
#: CODATA-2014 elementary charge in its ``r0``, and its ultra-relativistic approximations.
CONSTANTS_VINTAGE = 1.0639e-8


def _line():
    """The I4 ring in xtrack: thin quads (which do not radiate), bends, one cavity.

    The cavity is **last**, as it is in accsim's ``ring()``, so a turn ends at its entrance
    and ``twiss.zeta[0]`` is the arrival time the closed form is written in. ``q0 = +1``,
    which is accsim's default charge too, so the two RF conventions coincide and the phase
    maps straight over as ``phase = phi_s = pi`` (see ``rfcavity.py``: for a
    *negative* charge they would be exact negatives of each other).
    """
    cells, focal = RING["cells"], RING["focal"]
    angle = 2.0 * math.pi / (2 * cells)
    elements, names = [], []
    for i in range(cells):
        for j, strength in enumerate((0.5 / focal, angle, -1.0 / focal, angle, 0.5 / focal)):
            if j % 2:
                elements.append(
                    xt.Bend(length=1.0, angle=strength, k0=strength, model="bend-kick-bend")
                )
            else:
                elements.append(xt.Multipole(knl=[0.0, strength], length=0.0))
            names.append(f"e{i}_{j}")
    elements.append(xt.Cavity(voltage=RING["voltage"], frequency=1.0, phase=math.pi))
    names.append("cav")

    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(
        mass0=ELECTRON_MASS_EV,
        p0c=math.sqrt(RING["energy"] ** 2 - ELECTRON_MASS_EV**2),
        q0=1.0,
    )
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    for name in names[:-1]:
        if isinstance(line[name], xt.Bend):
            line[name].integrator = "uniform"
            line[name].num_multipole_kicks = 1  # one kick per element, like accsim
    return line


@pytest.fixture(scope="module")
def pair():
    """``(accsim lattice, cavity index, xtrack line, xtrack radiating twiss)``.

    One expensive build shared by the whole file: every ``xt.Line`` JIT-compiles a fresh
    C kernel at ~12 s (see CONVENTIONS.md -> *Test-suite cost*).
    """
    lattice, icav = ring()
    line = _line()
    line["cav"].frequency = lattice.elements[icav].frequency  # accsim's own h*beta0*c/C
    line.configure_radiation(model="mean")
    return lattice, icav, line, line.twiss(radiation_analysis=True)


def _losses(pair):
    """``(accsim, xtrack)`` reconstructed loss per turn [eV], from each code's own orbit."""
    lattice, icav, _, twiss = pair
    orbit = closed_orbit_6d(lattice, radiation="mean")
    mine = float(at_cavity(lattice, orbit, icav)[ZETA])
    theirs = float(twiss.zeta[0])
    return (
        cavity_kick_eV(lattice, mine, icav),
        cavity_kick_eV(lattice, theirs, icav),
        mine,
        theirs,
    )


def test_the_two_codes_agree_on_the_arrival_time_to_eight_digits(pair) -> None:
    """First, the headline: 8.887901 cm on both sides, of a 40 m ring.

    Before any residual is named, the effect itself has to be reproduced — the whole of it,
    by a code that solves for it a different way. Both also put the momentum at the lattice
    start at ``1.862083e-3`` and the horizontal orbit at ``-5.47e-7 m``.
    """
    lattice, _, _, twiss = pair
    _, _, mine, theirs = _losses(pair)
    orbit = closed_orbit_6d(lattice, radiation="mean")

    assert mine == pytest.approx(theirs, rel=1e-7)
    assert mine == pytest.approx(8.887901e-2, rel=1e-6)
    assert float(orbit[5]) == pytest.approx(float(twiss.delta[0]), rel=1e-7)
    assert float(orbit[0]) == pytest.approx(float(twiss.x[0]), rel=1e-6)


def test_the_whole_residual_is_b2s_two_named_owners_and_nothing_else(pair) -> None:
    """The pre-commitment, and it held: ``2.29997e-8`` predicted, ``2.29997e-8`` measured.

    Not a tolerance — a *prediction*, made from two mechanisms that were localised on a
    different milestone (B2, a single magnet) and are asserted here on a quantity B2 never
    touched (a whole ring's fixed point). The agreement between prediction and measurement
    is ``2e-6`` of the residual itself, which leaves no room for a third owner.
    """
    mine, theirs, _, _ = _losses(pair)
    gamma = RING["energy"] / ELECTRON_MASS_EV
    predicted = CONSTANTS_VINTAGE + 2.0 / gamma**2

    assert mine / theirs - 1.0 == pytest.approx(predicted, rel=1e-4)


def test_the_arcsine_is_why_the_comparison_is_made_on_the_loss_and_not_on_zeta(pair) -> None:
    """The same disagreement reads 2.36e-8 in ``zeta`` and 2.30e-8 in the loss.

    ``V sin(k zeta) = U`` maps one relative error onto the other through
    ``tan(k zeta)/(k zeta)``, which is 1.0270 here and would be a different number on a
    different ring. Comparing ``zeta`` directly is not wrong, it is *ring-dependent* — so
    the factor is asserted rather than divided out silently.
    """
    lattice, icav, _, _ = pair
    mine_loss, theirs_loss, mine, theirs = _losses(pair)
    k = lattice.elements[icav].k_rf(lattice.ref)

    in_zeta = mine / theirs - 1.0
    in_loss = mine_loss / theirs_loss - 1.0
    assert in_zeta / in_loss == pytest.approx(math.tan(k * mine) / (k * mine), rel=1e-3)
    assert in_zeta / in_loss == pytest.approx(1.0270, rel=1e-3)


def test_the_closed_form_holds_inside_xtrack_against_its_own_energy_loss(pair) -> None:
    """The third route, and the one that makes the closed form not an accsim convention.

    ``twiss.energy_loss`` is xtrack's own energy loss per turn, from its own radiation
    bookkeeping and not from any closed orbit. Feeding **xtrack's** ``zeta`` through
    accsim's ``q V [sin(phi_s - k zeta) - sin(phi_s)]`` reproduces it to ``2e-10``, so both
    codes' fixed points satisfy the same equation and the equation belongs to neither.
    """
    _, _, _, twiss = pair
    _, theirs_loss, _, _ = _losses(pair)

    assert theirs_loss == pytest.approx(float(twiss.energy_loss), rel=1e-9)
    assert theirs_loss == pytest.approx(2.48046780e7, rel=1e-8)
