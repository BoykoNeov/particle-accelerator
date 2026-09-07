r"""Cross-check the wiggler's *tracked* radiation against xtrack (T3).

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**xtrack has no wiggler, and it can still arbitrate this milestone.** T1 established that
neither reference code has the magnet, and that is true of its **map**: xtrack's two ways of
spelling a wiggler put the vertical focusing in no plane or in the wrong one, which is why
T1 and T2 pre-committed no tolerance gate against either. But **radiated energy needs only
``|B_perp|`` along the path, not the plane the focusing lands in** — and a stack of curved
bends following the segment-mean of ``h0 cos(ks)`` *is* the wiggler's trajectory,
discretised. So the arbiter that was structurally blind to T1's physics is not blind to
T3's, and this file is the first external check anything on axis T has had.

Two separable statements, gated separately:

1. **On the shared stack the two codes agree, and the residual is already named.** It is
   ``5.33e-07``, *constant in the pole count* across an eightfold refinement (40 → 320
   poles), and it is the sum of the two owners ``test_radiation_tracking_xtrack.py`` records
   on xtrack's side: its pre-2019 elementary charge (``1.064e-08``) and its
   ultra-relativistic approximations (``2/gamma0^2 = 5.223e-07``). Nothing new appears when
   the field alternates twenty times, and nothing accumulates over 320 elements.
2. **The stack reaches the wiggler by a law, not by a tolerance.** The segment-mean field
   keeps ``int h ds`` (the geometry) exact but under-counts ``int h^2 ds`` by the field's own
   variance across a segment, so the stack's loss approaches ``C_gamma E^4 I2 / 2 pi`` from
   **below** as ``(k ds)^2 / 12``. Measured ratio of deficit to law: ``0.980, 0.995, 0.999``.

Each ``xt.Line`` build costs 30–70 s on this box (``docs/CONVENTIONS.md`` -> *Test-suite
cost*), so the module builds **three** and shares them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import Dipole, Lattice, ReferenceParticle
from accsim.radiation_kick import radiation_constant_cgamma
from accsim.tracking import Tracker

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 0.51099895069e6  # electron, eV
ENERGY = 1e9
PERIOD, PERIODS, H0 = 0.1, 10, 0.449689
LENGTH = PERIOD * PERIODS
WAVENUMBER = 2.0 * math.pi / PERIOD

#: Poles per period. Three points, because a law needs more than two.
RESOLUTIONS = (8, 16, 32)


def _ref() -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(MASS0, ENERGY, 1.0)


def _cos_stack(per_period: int) -> list[tuple[float, float]]:
    """``(length, curvature)`` of each pole: the **segment mean** of ``h0 cos(ks)``.

    The segment mean rather than the mid-point value, because it is the choice that keeps
    ``int h ds`` — the geometry, and hence the closing of the trajectory — exact at every
    resolution. What it does *not* keep exact is ``int h^2 ds``, which is the whole of the
    convergence law below.
    """
    n = per_period * PERIODS
    ds = LENGTH / n
    out = []
    for j in range(n):
        s0, s1 = j * ds, (j + 1) * ds
        h = H0 * (math.sin(WAVENUMBER * s1) - math.sin(WAVENUMBER * s0)) / (WAVENUMBER * ds)
        out.append((ds, h))
    return out


def _accsim_loss(stack: list[tuple[float, float]]) -> float:
    ref = _ref()
    lattice = Lattice([Dipole(ds, h * ds) for ds, h in stack], ref)
    out = Tracker(lattice).track_once(np.zeros(6), radiation="mean")
    d1 = float(out[5])
    e1 = math.hypot(ref.momentum_eV * (1.0 + d1), MASS0)
    return ref.momentum_eV**2 * (-d1) * (2.0 + d1) / (ENERGY + e1)


def _xtrack_loss(stack: list[tuple[float, float]]) -> float:
    elements, names = [], []
    for j, (ds, h) in enumerate(stack):
        bend = xt.Bend(length=ds, angle=h * ds, k0=h)
        # One kick per element at the entry energy, like accsim -- without this xtrack's
        # adaptive integrator sub-steps and the 3.8e-5 integration-order difference
        # ``test_radiation_tracking_xtrack.py`` documents would swamp everything here.
        bend.integrator = "uniform"
        bend.num_multipole_kicks = 1
        elements.append(bend)
        names.append(f"p{j}")
    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1.0, energy0=ENERGY)
    line.configure_radiation(model="mean")
    line.build_tracker()
    part = line.build_particles(x=0, px=0, y=0, py=0, zeta=0, delta=0)
    line.track(part)
    return ENERGY - float(part.energy[0])


@pytest.fixture(scope="module")
def losses() -> dict[int, tuple[float, float]]:
    """``{poles per period: (accsim, xtrack)}`` — three xtrack line builds, shared."""
    return {n: (_accsim_loss(_cos_stack(n)), _xtrack_loss(_cos_stack(n))) for n in RESOLUTIONS}


def test_the_two_codes_agree_on_the_alternating_stack_at_the_residual_already_on_record(
    losses: dict[int, tuple[float, float]],
) -> None:
    """``5.33e-07``, and it is xtrack's charge vintage plus its ``2/gamma0^2``.

    Both numbers are established in ``test_radiation_tracking_xtrack.py`` on a *single*
    bend. What this asserts is that a field which reverses twenty times, spread over up to
    320 elements, adds nothing to them — which is the only way the stack can be used as an
    arbiter for the wiggler at all.
    """
    gamma0 = ENERGY / MASS0
    predicted = 1.064e-8 + 2.0 / (gamma0 * gamma0)

    for n in RESOLUTIONS:
        accsim, xtrack = losses[n]
        assert (accsim - xtrack) / accsim == pytest.approx(predicted, rel=0.01)


def test_that_residual_does_not_grow_with_the_number_of_elements(
    losses: dict[int, tuple[float, float]],
) -> None:
    """Constant in ``n`` to four digits over 40 → 320 poles: a vintage, not an accumulation.

    Asserted as a *mechanism* rather than a bound, in the manner T1 gate 7 established for
    this axis. A per-element error would grow by 8x across this range.
    """
    residuals = [(losses[n][0] - losses[n][1]) / losses[n][0] for n in RESOLUTIONS]
    assert max(residuals) / min(residuals) == pytest.approx(1.0, rel=1e-3)


def test_the_stack_reaches_the_wigglers_closed_form_as_the_step_squared(
    losses: dict[int, tuple[float, float]],
) -> None:
    """The deficit divided by ``(k ds)^2/12`` approaches 1 — the law, not a tolerance.

    This is what turns an arbiter that only knows about hard-edge bends into an arbiter for
    a cosine field. Both codes are checked against it, because the point is a property of
    the discretisation rather than of either implementation.

    The wiggler's own tracked loss is *above* ``C_gamma E^4 I2 / 2 pi`` by the path
    lengthening ``theta^2/4 = 1.28e-05`` (T3's analytic suite), which at these resolutions
    is 250x smaller than the deficit being measured and is deliberately not subtracted: the
    stack's own design orbit is exactly its length, so the two effects belong to different
    objects.
    """
    exact = radiation_constant_cgamma(_ref()) / (2.0 * math.pi) * ENERGY**4 * (0.5 * H0**2 * LENGTH)

    gamma0 = ENERGY / MASS0
    code_residual = 1.064e-8 + 2.0 / (gamma0 * gamma0)

    ratios = []
    for n in RESOLUTIONS:
        accsim, xtrack = losses[n]
        law = (WAVENUMBER * LENGTH / (n * PERIODS)) ** 2 / 12.0
        assert accsim < exact  # the segment mean under-counts h^2: it approaches from below
        ratios.append((1.0 - accsim / exact) / law)
        # The two codes' *losses* agree at the named residual, but this quantity divides
        # by a law that is itself only 3.2e-03 at the finest resolution, so the residual is
        # amplified 300x into the ratio. The bound is that amplification and nothing more.
        assert abs((1.0 - xtrack / exact) / law - ratios[-1]) < 1.1 * code_residual / law

    assert ratios == pytest.approx([0.979688, 0.994981, 0.999154], rel=1e-4)
    for coarse, fine in zip(ratios, ratios[1:], strict=False):
        assert abs(fine - 1.0) < abs(coarse - 1.0)  # monotone towards the law


def test_the_wigglers_own_tracked_loss_is_where_that_law_extrapolates_to(
    losses: dict[int, tuple[float, float]],
) -> None:
    """Richardson: two resolutions of the arbiter, extrapolated, land on the shipped magnet.

    With the error going as ``ds^2``, the ``n`` and ``2n`` results combine into an estimate
    with that term removed. It is xtrack's answer for a magnet xtrack cannot build, and it
    lands on the shipped wiggler's tracked loss to ``3.1e-05`` — a hundredfold improvement
    on the raw ``3.2e-03`` at the finest resolution.

    **And it stops there, which is the honest half.** A second Richardson stage only reaches
    ``1.4e-05``, so the remaining error is not a clean ``ds^4``: the segment mean is taken
    across a field that changes sign twenty times, and at those zeros the discretisation has
    no smooth even expansion. The consequence is that this arbiter resolves the wiggler's
    mean loss to a few parts in ``1e5`` and is therefore **blind to the ``theta^2/4 =
    1.28e-05`` path lengthening** that separates the tracked route from the design route —
    which stays an accsim-internal statement, gated in the analytic suite. A clean reference
    half and a half the arbiter cannot see, which is S2's and T1's shape on this axis.
    """
    from accsim import Wiggler

    ref = _ref()
    out = Tracker(Lattice([Wiggler(PERIOD, H0, PERIODS, "w")], ref)).track_once(
        np.zeros(6), radiation="mean"
    )
    d1 = float(out[5])
    shipped = (
        ref.momentum_eV**2
        * (-d1)
        * (2.0 + d1)
        / (ENERGY + math.hypot(ref.momentum_eV * (1.0 + d1), MASS0))
    )

    coarse, fine = losses[16][1], losses[32][1]  # xtrack's, at n and 2n
    richardson = (4.0 * fine - coarse) / 3.0
    assert richardson / shipped == pytest.approx(1.0, rel=5e-5)
    assert abs(richardson / shipped - 1.0) < 0.02 * abs(fine / shipped - 1.0)

    # ...and the second stage does not clean it up, so the limit is stated, not implied.
    second = (16.0 * richardson - (4.0 * losses[16][1] - losses[8][1]) / 3.0) / 15.0
    assert abs(second / shipped - 1.0) > 0.5 * 1.28e-5  # still above the path factor
