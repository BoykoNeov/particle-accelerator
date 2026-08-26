r"""M3 cross-check: second-order dispersion against xtrack, on both of its drift models.

Two things are established here, in order:

1. **The convention, by probe rather than by recall.** xtrack's ``ddx`` is the *full*
   second derivative ``d^2x/ddelta^2``, not half of it. Half is the other live
   candidate — it is what MAD-X reports, and it is what a Taylor-coefficient reading
   of "second-order dispersion" would give — so the factor is measured here by
   twice-differencing xtrack's **own** closed orbit and comparing with its own
   reported column.

2. **The finding: on this ring the quantity does not care which drift model the
   reference uses.** The roadmap pre-committed the opposite — that a ``ddx`` comparison
   would have to force ``xt.Drift(model="exact")`` or reproduce M1's 5% ``Q''``
   disagreement in a new place. It does not here: xtrack's default (paraxial) and exact
   drifts give ``ddx`` values that differ in the **ninth** significant digit, and
   accsim's exact-drift answer sits the same tiny distance from both.

   **"On this ring" is load-bearing.** The analytic suite derives the condition with no
   reference code present (``test_second_order_dispersion.py``): the ``delta^2`` part of
   the two drifts' difference is ``3 a b^2``, with ``a`` the **on-momentum** closed-orbit
   angle and ``b`` the dispersion angle. This ring closes on the axis, so ``a = 0`` and
   the split is gone; a steered ring has ``a != 0`` and splits ``ddx`` by ``6.8e-3``
   relative at a 10 mrad kick. ``Q''`` is split either way, because it differentiates the
   *Jacobian* about the orbit, where the same term contributes at ``O(b^2 delta^2)`` —
   one order lower, and free of ``a``.

Both drift models are therefore asserted to agree **on this unsteered ring**, which is
the exact opposite of what ``test_chromatic_optics_xtrack.py`` asserts about ``Q''`` on
the same one. The pair is the point: one quantity splits, the neighbouring one does not,
and the reason is a single power of ``delta``.

The edge model is ``suppressed`` for the same reason as in M1 — accsim's ``Dipole``
uses the linear hard-edge kick, which is the identity for a sector bend.

Marked ``reference``: skips (not fails) when xtrack or its JIT compiler is
unavailable — see ``docs/CONVENTIONS.md``.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from accsim import (
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    second_order_dispersion,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
LQ, K1, LD, LB, ANG, N_CELLS = 0.3, 1.2, 0.5, 1.0, 0.12, 3

# The same step M1 uses on this ring: above xtrack's own second-difference noise and
# below where truncation bites. Both codes finite-difference, so the residual between
# them is set by their two steps, not by a physics difference.
DELTA = 1e-3


def _accsim_lattice() -> Lattice:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
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


@functools.cache
def _line(drift_model: str | None = None):
    """A built xtrack line, cached: every ``xt.Line`` build JIT-compiles a fresh kernel.

    ``drift_model=None`` is xtrack's default (paraxial); ``"exact"`` matches accsim's.
    Which one is in use is the subject of this file, so it is a parameter rather than
    a default left implicit.
    """

    def drift(length: float):
        return xt.Drift(length=length, model=drift_model)

    els: list = []
    for _ in range(N_CELLS):
        els += [
            xt.Quadrupole(length=LQ, k1=K1),
            drift(LD),
            drift(LD),
            xt.Bend(length=LB, angle=ANG, k0=ANG / LB),
            xt.Quadrupole(length=LQ, k1=-K1),
            xt.Bend(length=LB, angle=ANG, k0=ANG / LB),
            drift(LD),
        ]
    line = xt.Line(elements=els)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    line.configure_bend_model(core="bend-kick-bend", edge="suppressed")
    try:
        line.build_tracker()
        line.twiss(method="4d")  # probe: raises here if the JIT cannot build
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


# ---------------------------------------------------------------------------
# 1. the convention, pinned by probe
# ---------------------------------------------------------------------------


def test_xtracks_ddx_is_the_full_second_derivative_and_not_half_of_it() -> None:
    r"""``ddx = d^2x/ddelta^2``, measured against xtrack's own closed orbit.

    The factor of two is exactly the kind of thing a milestone like this exists to
    catch: MAD-X's ``DDX`` really is half of this (in a different momentum variable
    again), so "second-order dispersion" names two different numbers depending on who
    is asked. Nothing is recalled here — xtrack's ``x`` is sampled at three momenta and
    twice-differenced, and the ratio to its reported ``ddx`` is asserted to be one.
    """
    line = _line("exact")
    x0 = line.twiss(method="4d")
    xp = line.twiss(method="4d", delta0=+DELTA)
    xm = line.twiss(method="4d", delta0=-DELTA)

    measured = (xp.x[0] - 2.0 * x0.x[0] + xm.x[0]) / DELTA**2
    assert measured == pytest.approx(x0.ddx[0], rel=1e-4)
    assert abs(measured / x0.ddx[0] - 0.5) > 0.4  # decisively not the halved convention

    # ...and its first-order column is d/ddelta in the same variable, ratio one.
    assert (xp.x[0] - xm.x[0]) / (2.0 * DELTA) == pytest.approx(x0.dx[0], rel=1e-6)


# ---------------------------------------------------------------------------
# 2. the deliverable: element by element, around a ring with bends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drift_model", [None, "exact"])
def test_second_order_dispersion_matches_xtrack_around_the_ring(drift_model: str | None) -> None:
    r"""``ddx`` and ``ddpx`` agree at **every** boundary, on either xtrack drift model.

    A one-turn scalar can be right by cancellation; a curve around the ring cannot.
    The parametrisation is the milestone's finding rather than thoroughness: the
    agreement is the same to within the two codes' own steps whether xtrack uses the
    drift accsim uses or the one M2 showed splits ``Q''`` by 5%.

    Scope, stated because it is easy to over-read: this ring's on-momentum closed orbit
    is on the axis, which is the condition under which the two drift models agree here at
    all. The ``disp_x`` comparison below is safe for the same reason — first-order
    dispersion is split by a steered orbit too, and by a *lower* power.
    """
    lattice = _accsim_lattice()
    points = second_order_dispersion(lattice, delta=DELTA)
    tw = _line(drift_model).twiss(method="4d")

    assert len(points) == len(tw.x)
    ddx = np.array([p.ddisp_x for p in points])
    ddpx = np.array([p.ddisp_px for p in points])
    np.testing.assert_allclose(ddx, np.array(tw.ddx), rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(ddpx, np.array(tw.ddpx), rtol=2e-6, atol=1e-6)

    # The first-order part of the same object, against the column Stage 1 pinned.
    np.testing.assert_allclose(
        np.array([p.disp_x for p in points]), np.array(tw.dx), rtol=1e-6, atol=1e-9
    )

    # The curves genuinely vary around the ring, so the agreement is not the trivial
    # one of two nearly-constant lines. ``ddx`` swings by ~11% of its own size and
    # ``ddpx`` changes sign, both far above the 2e-6 the comparison is made at.
    assert (ddx.max() - ddx.min()) / abs(ddx).max() > 0.1
    assert ddpx.min() < 0.0 < ddpx.max()

    # A flat ring has no vertical dispersion at either order, in either code.
    assert np.abs(np.array([p.ddisp_y for p in points])).max() < 1e-9
    assert np.abs(np.array(tw.ddy)).max() < 1e-9


# ---------------------------------------------------------------------------
# 3. the finding: the drift model splits Q'' and does not split this
# ---------------------------------------------------------------------------


def test_the_drift_model_moves_ddx_in_the_ninth_digit_where_it_moves_qpp_by_5_percent() -> None:
    r"""The contrast, measured inside one reference code so nothing else can differ.

    Same line, same bend model, same tune — only ``xt.Drift``'s ``model`` changes. M2
    established that this moves ``Q''`` from ``0.75202`` to about ``0.79``, a 5% split
    that took a milestone to explain. On the neighbouring quantity it is worth eight
    orders of magnitude less, because the ``delta^2`` part of the two models' difference
    carries a factor of the **on-momentum** orbit angle, which this ring does not have.
    Steer it and the eight orders come back — asserted in the analytic suite, where the
    arbiter can be given a steerer without a second JIT build.
    """
    paraxial = _line(None).twiss(method="4d")
    exact = _line("exact").twiss(method="4d")

    assert exact.qx == pytest.approx(paraxial.qx, abs=1e-9)  # the same machine
    split = abs(exact.ddx[0] - paraxial.ddx[0]) / abs(exact.ddx[0])
    assert split < 1e-7

    # And the accompanying second-order *chromaticity* on the same pair of lines is
    # the 5% split M2 named — asserted here so the contrast is one measurement.
    chrom_split = abs(exact.ddqx - paraxial.ddqx) / abs(exact.ddqx)
    assert chrom_split > 1e-2
