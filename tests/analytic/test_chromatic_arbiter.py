"""M2: which code's ``Q''`` is right, and why the other two are not.

M1 measured three different second-order chromaticities on a ring with bends —
accsim ``0.79307``, xtrack ``0.75202``, MAD-X ``0.70441`` — while all three agreed
on the tune to ten digits and on ``Q'`` to seven. It concluded that the spread could
not be accsim's element maps, and shipped the number as an unarbitrated boundary.

**That conclusion was wrong, and correcting it is this milestone.** The reasoning
was valid; its premise was not. M1 established that accsim's *dipole* Jacobian
equals ``xt.Bend``'s off-momentum and inferred "identical maps". It never checked
the **drift** off-momentum — because L1 had shipped the drift exact, which read as
settled. L1 validated the drift's map; it did not validate its agreement with
xtrack's *default configuration*, and those are different claims.

The split is the drift model, entirely:

    exact     x += L px / sqrt((1+delta)^2 - px^2 - py^2)      accsim, and xtrack's
                                                              ``Drift(model="exact")``
    paraxial  x += L px / (1 + delta)                          xtrack's default, MAD-X

The two differ by the factor ``1 + (px^2 + py^2)/2 + ...``. On a ring with no bends
the closed orbit has ``px = 0`` at every momentum and the two maps coincide, which
is why M1's bend-free control showed a genuine three-code agreement. With bends the
closed orbit acquires ``px ~ D_px delta``, so the difference enters at
``O(delta^2)`` — invisible in ``Q`` and ``Q'``, landing squarely on ``Q''`` — and is
proportional to ``D_px^2``, hence to the **square of the bending angle**. That is
exactly the scaling law M1 measured and could not explain, and the sweep below
reproduces it with no reference code involved at all.

These gates use ``tests/_m2_minimal_ring.py``: a five-element ring whose ``Q''`` is
derived there from lab-frame geometry at sixty digits, so accsim is checked against
a number rather than against another implementation. The reference suites carry the
other half — that xtrack's ``model="exact"`` collapses the disagreement to noise and
its default reproduces the paraxial arbiter instead.
"""

from __future__ import annotations

import numpy as np
import pytest
from mpmath import mp, mpf

from _m2_minimal_ring import (
    ANG,
    LB,
    bend_map,
    drift_map,
    lattice,
)
from _m2_minimal_ring import (
    second_order_chromaticity as arbiter_qpp,
)
from accsim import second_order_chromaticity, tunes_on_orbit
from accsim.elements.dipole import exact_sector_bend_map

# ---------------------------------------------------------------------------
# 0. the ring is a fair place to ask the question
# ---------------------------------------------------------------------------


def test_the_minimal_ring_is_stable_in_both_planes_and_off_the_half_integer() -> None:
    """Asserted before anything is measured on it, because a second difference is fragile.

    A second difference of the tune divides by ``delta^2``, so it amplifies whatever
    the tune extraction gets wrong; near a half integer ``sin 2 pi Q -> 0`` and every
    derivative of ``Q`` with respect to a map entry blows up. This ring keeps both
    tunes near ``0.13`` and both traces near ``1.39``, comfortably inside ``|Tr| < 2``.

    A sector bend focuses horizontally and **not** vertically, so the vertical plane's
    entire stability comes from the two thin quadrupoles. One quadrupole would leave
    it unstable, which is why the roadmap's pre-committed "single thin quadrupole plus
    single sector bend" is not the ring used here.
    """
    from _m2_minimal_ring import design_traces, design_tunes

    traces, tunes = design_traces(), design_tunes()
    for plane in ("x", "y"):
        assert abs(traces[plane]) < 2.0
        assert 0.05 < tunes[plane] < 0.45  # far from both the integer and the half integer

    # accsim builds the same ring: its tunes agree with the arbiter's to twelve digits.
    ours = tunes_on_orbit(lattice())
    assert ours[0] == pytest.approx(tunes["x"], rel=1e-11)
    assert ours[1] == pytest.approx(tunes["y"], rel=1e-11)


def test_the_arbiters_vertical_plane_equals_a_hand_derived_thin_lens_trace() -> None:
    r"""A closed form the arbiter must hit exactly, written out by hand.

    A sector bend has **no** vertical focusing — it only advances ``y`` by
    ``py phi / h``, which on the design orbit is ``py Lb``. So the vertical plane of
    this ring is a pure thin-lens FODO: ``D(b) Q(-k) D(a) Q(+k)`` with
    ``a = Lb + LD = 1.5``, ``b = LD = 0.5``. Multiplying those four ``2x2`` matrices
    out, every term in the trace cancels except

        Tr = 2 - a b k^2 = 2 - 1.5 * 0.5 * 0.81 = 1.3925

    exactly. This checks the arbiter's bend, drift and thin quadrupole *together* in
    the one plane where the answer is a rational number, so it cannot be right by a
    compensating pair of errors.
    """
    from _m2_minimal_ring import KF, design_traces
    from _m2_minimal_ring import LB as BEND_L
    from _m2_minimal_ring import LD as DRIFT_L

    a, b = BEND_L + DRIFT_L, DRIFT_L
    assert design_traces()["y"] == pytest.approx(2.0 - a * b * KF**2, abs=1e-14)


def test_the_geometric_bend_reproduces_accsims_rearranged_map() -> None:
    r"""The arbiter's bend is a *derivation*, and this is what licenses calling it one.

    :func:`accsim.elements.dipole.exact_sector_bend_map` is written so that no two
    numbers of size one are ever subtracted — a rationalised ``pz - 1``, an ``arcsinc``
    standing in for a difference of two arcsines, and no ``1/h`` anywhere. Porting that
    arrangement into the arbiter would have tested it against itself. The arbiter
    instead solves the raw geometry: the circle of radius ``p_perp/h`` meeting the exit
    face, with the intersection found numerically at sixty digits.

    Two independent constructions of the same element, agreeing to ``1e-14`` over
    random states and at large amplitude.
    """
    ref = lattice().ref
    rng = np.random.default_rng(0)
    worst = 0.0
    with mp.workdps(50):
        for _ in range(150):
            state = np.array(
                [
                    rng.normal(0.0, 2e-3),
                    rng.normal(0.0, 1e-3),
                    rng.normal(0.0, 2e-3),
                    rng.normal(0.0, 1e-3),
                    0.0,
                    rng.normal(0.0, 2e-3),
                ]
            )
            ours = exact_sector_bend_map(state, LB, ANG / LB, ref)[:4]
            theirs = bend_map([mpf(v) for v in state[:4]] + [mpf(state[5])], mpf(LB), mpf(ANG))
            worst = max(worst, max(abs(float(theirs[i]) - ours[i]) for i in range(4)))

        big = np.array([0.05, 0.02, 0.04, 0.01, 0.0, 0.05])
        ours = exact_sector_bend_map(big, LB, ANG / LB, ref)[:4]
        theirs = bend_map([mpf(v) for v in big[:4]] + [mpf(big[5])], mpf(LB), mpf(ANG))
        worst_big = max(abs(float(theirs[i]) - ours[i]) for i in range(4))

    assert worst < 1e-14
    assert worst_big < 1e-14


# ---------------------------------------------------------------------------
# 1. the arbitration
# ---------------------------------------------------------------------------


def test_accsim_lands_on_the_independently_derived_second_order_chromaticity() -> None:
    r"""The milestone's headline: accsim's ``Q''`` **is** the exact-drift answer.

    Gated on the **order** rather than on a value at one step, because
    :func:`~accsim.twiss.second_order_chromaticity` is a central second difference:
    its residual against the true value falls as ``delta^2``, so halving ``delta``
    must quarter it. A single-step tolerance would pass just as happily on a map that
    was wrong by a constant.

    Measured residuals against the arbiter's ``0.307378890874``: ``4.1e-5``, ``1.0e-5``,
    ``2.6e-6`` at ``delta = 1e-2, 5e-3, 2.5e-3``. Below ``~1e-3`` the closed-orbit
    solve's own noise starts to enter as ``1/delta^2`` and the ratio degrades — the
    same window M1 documented for xtrack's ``ddqx``.
    """
    exact = arbiter_qpp(exact_drift=True)
    lat = lattice()
    steps = (1e-2, 5e-3, 2.5e-3)

    for plane, index in (("x", 0), ("y", 1)):
        residuals = [
            abs(second_order_chromaticity(lat, delta=d)[index] - exact[plane]) for d in steps
        ]
        for coarse, fine in zip(residuals, residuals[1:], strict=False):
            assert coarse / fine == pytest.approx(4.0, rel=0.2)
        assert residuals[-1] < 5e-6  # and it is small in absolute terms as well


def test_accsim_is_not_on_the_paraxial_drifts_answer() -> None:
    r"""The other half of the arbitration, and the reason the gate above discriminates.

    Feeding the *paraxial* drift into the same arbiter gives a different number —
    ``0.293224`` against ``0.307379`` in ``x``, a 4.6% split. accsim sits on the exact
    one, roughly three thousand times closer to it than to the paraxial one, so the
    gate above is not merely a loose tolerance that both would pass.

    This number is not a curiosity: the reference suite shows xtrack's **default**
    drift reproduces it to ``4e-6``.
    """
    exact, paraxial = arbiter_qpp(exact_drift=True), arbiter_qpp(exact_drift=False)
    lat = lattice()

    for plane, index in (("x", 0), ("y", 1)):
        ours = second_order_chromaticity(lat, delta=2.5e-3)[index]
        to_exact = abs(ours - exact[plane])
        to_paraxial = abs(ours - paraxial[plane])
        assert to_paraxial / to_exact > 1000.0

    assert exact["x"] - paraxial["x"] == pytest.approx(0.0141553115, rel=1e-6)
    assert exact["y"] - paraxial["y"] == pytest.approx(0.0047755244, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. the mechanism, established without any reference code
# ---------------------------------------------------------------------------


def test_the_two_drift_models_agree_until_the_particle_has_an_angle() -> None:
    r"""Why the split needs bends: the two drifts differ at ``O(px^2)``, not ``O(delta)``.

    ``L px/pz`` over ``L px/(1+delta)`` is ``1 + (px^2 + py^2)/(2 (1+delta)^2) + ...``.
    At ``px = py = 0`` the two maps are identical **at every momentum** — asserted to
    round-off below — so a ring whose closed orbit is straight cannot tell them apart
    no matter how far off-momentum it is asked about. That is why M1's bend-free
    control was a genuine three-code agreement rather than a lucky one.

    Turn the angle on and the difference grows as its square, measured here on the
    displacement itself.
    """
    with mp.workdps(50):
        for delta in (mpf(0), mpf("1e-3"), mpf("1e-2")):
            state = [mpf(0), mpf(0), mpf(0), mpf(0), delta]
            a = drift_map(state, mpf(1), exact=True)
            b = drift_map(state, mpf(1), exact=False)
            assert max(abs(float(a[i] - b[i])) for i in range(4)) < 1e-45

        gaps = []
        for px in (mpf("1e-4"), mpf("2e-4"), mpf("4e-4")):
            state = [mpf(0), px, mpf(0), mpf(0), mpf("1e-3")]
            a = drift_map(state, mpf(1), exact=True)
            b = drift_map(state, mpf(1), exact=False)
            gaps.append(abs(float(a[0] - b[0])))
    # doubling px multiplies the *displacement* gap by 8 (px times px^2), not by 2
    for coarse, fine in zip(gaps[1:], gaps, strict=False):
        assert coarse / fine == pytest.approx(8.0, rel=1e-3)


def test_the_split_is_zero_without_bends_and_quadratic_in_the_bending_angle() -> None:
    r"""M1's scaling law, reproduced from the drift model alone.

    M1 swept the bending angle against MAD-X and found the gap exactly zero at zero
    angle and quadratic in the angle as it turned on, and named that "the signature of
    the longitudinal constraint". It is the signature of the drift model: the closed
    orbit's ``px`` is proportional to ``D_px delta``, ``D_px`` is proportional to the
    bending angle, and the two drift maps differ at ``O(px^2)``.

    Swept here **inside the arbiter**, with both drift models and no reference code in
    the room, the same law appears: ``0`` at zero angle, and ``gap/angle^2`` tending to
    a constant near ``1.07`` as the angle shrinks (``1.074``, ``1.069``, ``1.051`` at
    ``0.015``, ``0.03``, ``0.06`` rad; higher orders take over by ``0.12``).
    """

    def gap(angle: float) -> float:
        return (
            arbiter_qpp(exact_drift=True, angle=angle)["x"]
            - arbiter_qpp(exact_drift=False, angle=angle)["x"]
        )

    assert abs(gap(0.0)) < 1e-12  # no bend, no gap: the two maps are the same ring

    angles = (0.015, 0.03, 0.06)
    coefficients = [gap(a) / a**2 for a in angles]
    assert all(c > 0.0 for c in coefficients)
    for coarse, fine in zip(coefficients, coefficients[1:], strict=False):
        assert coarse == pytest.approx(fine, rel=0.03)  # stable as the angle shrinks
    assert gap(0.03) / gap(0.015) == pytest.approx(4.0, rel=0.02)


def test_the_answer_does_not_depend_on_the_reference_particle() -> None:
    """This ring's ``Q''`` is a property of the geometry, not of the beam energy.

    Nothing in the transverse map of a thin quadrupole, an exact drift or a sector
    bend refers to the reference particle: ``beta0`` and ``gamma0`` enter only through
    ``zeta``, and without an RF cavity ``zeta`` never feeds back. So the arbiter can
    omit a reference particle entirely, and accsim must return the same number from
    ``gamma0 = 5`` to ``gamma0 = 4000``.

    Worth asserting rather than assuming: a stray ``1/gamma0^2`` in a path-length term
    is exactly the kind of error this project exists to catch, and it would show up
    here and nowhere else in the milestone.
    """
    values = [second_order_chromaticity(lattice(g), delta=2.5e-3)[0] for g in (5.0, 20.0, 4000.0)]
    for v in values[1:]:
        assert v == pytest.approx(values[0], rel=1e-13)
