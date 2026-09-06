r"""Cross-check the *applied* taper (Q2) against xtrack, at both ends of the milestone.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

Q1's sibling file compares the taper *profile*. This one compares what the profile is for,
and it has two legs that answer different questions:

- **the element.** ``xt.Bend`` carries ``k0`` and ``angle`` as separate parameters, so it
  is a direct arbiter for the object this milestone exists to create — a bending magnet
  whose field is not its geometry. This is the sharpest number in the milestone: the two
  maps agree to ``7.5e-17``, and they agree at ``gamma = 1.6`` as well as ``gamma = 21``,
  which is what puts the ``zeta`` row under a gate at all. On I4's 6.5 GeV electron ring
  the taper's ``zeta`` correction is ``O(t/gamma^2) ~ 1e-11`` of the length and *nothing*
  measured there could see it.
- **the ring.** ``line.compensate_radiation_energy_loss()`` is xtrack's own version of
  :func:`~accsim.tapering.taper`, and it is a genuinely independent implementation: it
  stores a per-element ``delta_taper`` and scales the strengths at tracking time where
  accsim builds a new lattice with new ``k0``. Both collapse the ``7.09 mm`` sag orbit to
  ``1e-10`` m and stop there.

**The roadmap entry's own number is corrected here.** Q1's filter recorded xtrack
collapsing this orbit "by a factor 14,300", to ``4.96e-7``. Run against a twiss rather
than the routine's own report it collapses it by **7.3e7**, to ``9.70e-11`` — which is
accsim's third round to 3%, and the reason ``rounds`` defaults to three rather than two.
The larger residual belonged to a measurement, not to the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))

from test_closed_orbit_6d import ring  # noqa: E402
from test_tapering_applied import orbit_excursion  # noqa: E402
from test_tapering_xtrack import _line  # noqa: E402

from accsim import Dipole, ReferenceParticle, taper  # noqa: E402

PROTON_MASS_EV = 938.27208816e6

#: A tapered bend outside anything a ring needs, so the comparison is not a comparison of
#: two nearly-untapered magnets. I4 runs at ``4e-3``.
TAPERS = (0.0, 4.0e-3, -0.05)

STATES = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0e-3, 2.0e-3, -1.5e-3, 8.0e-4, 0.0, 3.0e-3),
    (-4.0e-3, 1.0e-3, 2.0e-3, -2.0e-3, 0.0, -1.0e-2),
)


def _tracked(energy: float, length: float, h: float, k0: float, states) -> np.ndarray:
    """Every state through one ``xt.Bend(angle=h L, k0=k0)``, in accsim's coordinate order.

    All the states go through a **single** line: every ``xt.Line`` build JIT-compiles a
    fresh C kernel at ~12 s and one leaked ``.pyd``, so the loop is over magnets and never
    over particles (``docs/CONVENTIONS.md`` -> *Test-suite cost*).
    """
    bend = xt.Bend(
        length=length,
        angle=h * length,
        k0=k0,
        model="bend-kick-bend",
        integrator="uniform",
        num_multipole_kicks=1,
    )
    line = xt.Line(elements=[bend], element_names=["b"])
    line.particle_ref = xt.Particles(mass0=PROTON_MASS_EV, q0=1.0, energy0=energy)
    line.build_tracker()
    grid = np.asarray(states, dtype=float)
    part = xt.Particles(
        mass0=PROTON_MASS_EV,
        q0=1.0,
        energy0=energy,
        x=grid[:, 0],
        px=grid[:, 1],
        y=grid[:, 2],
        py=grid[:, 3],
        zeta=grid[:, 4],
        delta=grid[:, 5],
    )
    line.track(part)
    order = np.argsort(part.particle_id)
    return np.column_stack(
        [
            part.x[order],
            part.px[order],
            part.y[order],
            part.py[order],
            part.zeta[order],
            part.delta[order],
        ]
    )


@pytest.mark.parametrize(("energy", "tag"), [(1.5e9, "gamma 1.6"), (20.0e9, "gamma 21")])
def test_a_tapered_bend_is_xtracks_bend_with_k0_given_separately(energy: float, tag: str) -> None:
    r"""``k0 != h``, entry for entry, to ``7.5e-17``.

    ``xt.Bend`` refuses to be handed ``h`` directly (``length`` and ``angle`` set it) but
    takes ``k0`` on its own, which is exactly accsim's split since Q2. Both codes then run
    the same physical magnet: uniform field ``k0``, reference arc of curvature
    ``h = angle/length``.

    The **low-energy leg is the one that gates ``zeta``.** accsim reaches a tapered map by
    an exact rescaling of the design magnet's, and that symmetry carries the transverse
    coordinates untouched while leaving ``zeta`` needing a clock correction of order
    ``t/gamma^2``. At ``gamma = 21`` that is ``1e-5`` of the correction's own size and a
    map with the row simply omitted would pass; at ``gamma = 1.6`` it does not.
    """
    ref = ReferenceParticle.from_total_energy(PROTON_MASS_EV, energy)
    length, h = 1.0, 0.3
    for taper_t in TAPERS:
        k0 = h * (1.0 + taper_t)
        bend = Dipole(length, h * length, k0=k0)
        theirs = _tracked(energy, length, h, k0, STATES)
        for state, want in zip(STATES, theirs, strict=True):
            mine = bend.track(np.array(state, dtype=float), ref)
            assert np.abs(mine - want).max() < 1.0e-15, (taper_t, state)


@pytest.fixture(scope="module")
def xtrack_orbits() -> tuple[float, float]:
    """``(untapered, tapered)`` max |x| from xtrack's own radiation twiss."""
    line = _line()
    line.configure_radiation(model="mean")
    before = float(np.max(np.abs(line.twiss(method="6d").x)))
    line.compensate_radiation_energy_loss(verbose=False)
    after = float(np.max(np.abs(line.twiss(method="6d").x)))
    return before, after


def test_both_codes_see_the_same_seven_millimetre_sag_orbit(xtrack_orbits) -> None:
    """The thing being removed, before anything removes it.

    accsim's 6D closed orbit and xtrack's radiation twiss are different solves of the same
    fixed point, and they agree to six figures — which is what makes the collapse below a
    comparison rather than two unrelated numbers.
    """
    before, _ = xtrack_orbits
    lattice, _ = ring()
    assert orbit_excursion(lattice) == pytest.approx(before, rel=1.0e-5)
    assert before == pytest.approx(7.0904e-3, rel=1.0e-3)


def test_both_codes_collapse_the_sag_orbit_to_the_same_floor(xtrack_orbits) -> None:
    r"""``9.70e-11`` (xtrack) against ``1.00e-10`` (accsim), from unrelated machinery.

    xtrack keeps the design strengths and a per-element ``delta_taper`` that its kernels
    apply at tracking time; accsim builds a **new lattice** whose bends carry a different
    ``k0`` and whose quadrupoles carry a different ``k1``, and never touches the caller's.
    Neither code's residual is a tolerance either side has chosen — both are where their
    own closed-orbit solve stops resolving, seven orders below the distortion.

    This is also where the roadmap's ``14,300`` is corrected: that figure came from an
    earlier probe of the same routine and is a property of that measurement, not of
    xtrack.
    """
    before, after = xtrack_orbits
    lattice, _ = ring()
    mine = orbit_excursion(taper(lattice))
    assert after == pytest.approx(9.70e-11, rel=0.1)
    assert mine == pytest.approx(after, rel=0.15)
    assert before / after > 1.0e7
    assert before / mine > 1.0e7
