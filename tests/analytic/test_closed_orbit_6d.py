r"""The closed orbit in all six coordinates (I4).

Every closed orbit in :mod:`accsim.orbit` up to here is a fixed point of the
*transverse* map at a ``delta`` the caller chooses, with ``zeta`` held at zero:
:func:`~accsim.orbit.closed_orbit` solves ``(I - M4) x = k4``,
:func:`~accsim.orbit.closed_orbit_nonlinear` Newtons on the same 4D subspace, and N5's
:func:`~accsim.orbit.closed_orbit_delta` adds the one scalar that makes ``zeta`` close on
a bunched ring. That is exact for a ring whose cavity has nothing to make up. It is wrong
the moment the beam radiates in tracking: a ring that loses ``U`` per turn has to arrive
off the zero crossing of the RF wave, far enough up it that the cavity hands ``U`` back.

**The closed form, and it is read at the cavity.** ``zeta_co`` solves

    ``q V [sin(phi_s - k_rf zeta_co) - sin(phi_s)] = U_turn``

at the **cavity**, not at the lattice start — the two differ by whatever share of the
loss is accumulated upstream of it, which is why one ring here has the cavity spliced
mid-lattice.

**Two routes supply ``U_turn`` and the file asserts both.** The *tracked* loss, summed
element by element around the converged orbit, satisfies the equation at round-off — an
identity, and what makes the solve testable without a second model. The package's own
``energy_loss_per_turn``, a design-route radiation integral, is an independent number and
lands nearby but not exactly.

**The discriminating content is the order of that second departure, and it is a statement
about the fixed point rather than about the radiation.** On the *design* orbit the two
differ at **first** order in ``U_0/E``: a lumped per-element kick makes the particle
poorer as it goes, so every element after the first radiates below the design energy. On
the *closed* orbit that error is gone, because the fixed point is exactly where the sag is
centred — the beam sits high at the cavity's exit and low at its entrance, and the
linear-in-``delta`` part of the loss averages away over the turn. The departure is
therefore **quadratic**, and fitting the exponent reads a wrong fixed point off directly
where a tolerance would not.

**One shipped claim is corrected here.** ``closed_orbit_delta``'s docstring and
``docs/ROADMAP.md`` -> N5 both said a ring needs a 6D fixed point when ``phi_s != 0``
**or** radiation is tracked. The first half does not hold in this package: the cavity kick
``sin(phi_s - k zeta) - sin(phi_s)`` vanishes at ``zeta = 0`` for *every* ``phi_s``, and
the reference-energy ramp that would make an accelerating bucket mean something lives
inside :func:`accsim.accelerate`, which builds its own reference per turn and never
touches the tracking path. Tracked radiation is the only thing in accsim that moves
``zeta_co``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Corrector,
    Dipole,
    Lattice,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
)
from accsim.coords import DELTA, ZETA
from accsim.orbit import (
    ClosedOrbitError,
    closed_orbit_6d,
    closed_orbit_delta,
    closed_orbit_nonlinear,
)
from accsim.radiation import energy_loss_per_turn

ELECTRON_MASS_EV = 0.51099895069e6

#: The I4 ring: B4's isomagnetic FODO, whose ``U0/E = 3.8e-3`` per turn is large enough
#: that the arrival time is a visible 8.9 cm and small enough that the ring is not broken.
#: (``charge`` is the package default ``+1``, as everywhere on axis B, so the stable
#: stationary phase above transition is ``pi``.)
RING = {"cells": 20, "focal": 2.5, "energy": 6.5e9, "voltage": 90.0e6, "harmonic": 20}


def ring(
    cells: int = RING["cells"],
    focal: float = RING["focal"],
    energy: float = RING["energy"],
    voltage: float = RING["voltage"],
    harmonic: int = RING["harmonic"],
    phi_s: float = math.pi,
    at: int | None = None,
    kick: float = 0.0,
) -> tuple[Lattice, int]:
    """``(lattice, index of the cavity)``. ``at`` splices the cavity mid-lattice."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, energy)
    angle = 2.0 * math.pi / (2 * cells)
    cell = [
        ThinQuadrupole(0.5 / focal),
        Dipole(1.0, angle),
        ThinQuadrupole(-1.0 / focal),
        Dipole(1.0, angle),
        ThinQuadrupole(0.5 / focal),
    ]
    elements: list = list(cell) * cells
    if kick:
        elements.insert(1, Corrector(kick_x=kick))
    plain = Lattice(elements, ref=ref)
    cavity = RFCavity.from_harmonic(voltage, harmonic, plain.length, ref, phi_s=phi_s)
    if at is None:
        return Lattice([*elements, cavity], ref=ref), len(elements)
    spliced = list(elements)
    spliced.insert(at, cavity)
    return Lattice(spliced, ref=ref), at


def turn(lattice: Lattice, state: np.ndarray, radiation: str = "mean") -> np.ndarray:
    """One turn of the real map, radiation and all."""
    out = np.asarray(state, dtype=float).copy()
    for elem in lattice.elements:
        out = elem.track(out, lattice.ref, radiation=radiation)
    return out


def at_cavity(lattice: Lattice, orbit: np.ndarray, icav: int) -> np.ndarray:
    """The orbit propagated from the lattice start to the cavity's **entrance**."""
    out = np.asarray(orbit, dtype=float).copy()
    for elem in lattice.elements[:icav]:
        out = elem.track(out, lattice.ref, radiation="mean")
    return out


def tracked_loss(lattice: Lattice, orbit: np.ndarray) -> float:
    """The turn's radiated energy [eV], summed element by element along ``orbit``.

    Each element's loss is the ``delta`` it removes *at the state it actually sees*, which
    is the whole point: the sum is a property of the trajectory, not of the design orbit.
    """
    ref = lattice.ref
    total, state = 0.0, np.asarray(orbit, dtype=float).copy()
    for elem in lattice.elements:
        plain = elem.track(state, ref)
        state = elem.track(state, ref, radiation="mean")
        total += float(plain[DELTA] - state[DELTA])
    return total * ref.beta0**2 * ref.total_energy_eV


def cavity_kick_eV(lattice: Lattice, zeta: float, icav: int) -> float:
    """``q V [sin(phi_s - k zeta) - sin(phi_s)]`` [eV] — the left side of the gate."""
    ref, cav = lattice.ref, lattice.elements[icav]
    k = cav.k_rf(ref)
    return ref.charge * cav.voltage * (math.sin(cav.phi_s - k * zeta) - math.sin(cav.phi_s))


# ---------------------------------------------------------------------------
# Gate 1 — the fixed point exists in six coordinates, and the four-coordinate
# answer is not it.
# ---------------------------------------------------------------------------
def test_the_six_dimensional_orbit_closes_to_round_off() -> None:
    """All six coordinates come back to themselves, with radiation dissipating energy."""
    lat, _ = ring()
    orbit = closed_orbit_6d(lat, radiation="mean")

    assert orbit.shape == (6,)
    assert np.max(np.abs(turn(lat, orbit) - orbit)) < 1e-14


def test_the_four_dimensional_orbit_misses_in_arrival_time_first_and_not_in_momentum() -> None:
    """The pre-commitment here was **refuted**, and the refutation is the content.

    Written down before measuring: the 4D answer (:func:`closed_orbit_nonlinear` for the
    transverse part, N5's :func:`closed_orbit_delta` for the momentum, ``zeta = 0``) knows
    nothing about the light the beam emits, so its one-turn residual should be dominated by
    ``delta`` and equal to the whole bill, ``U0/(beta0^2 E0) = 3.8e-3``.

    Neither half holds. The **largest** residual is ``zeta`` — 2.7 cm of arrival time
    against 2.6e-3 of momentum — and the momentum miss is only **0.686** of the turn's
    loss. Both have the same cause and it is a feedback the pre-commitment ignored: losing
    ``delta`` through the arc makes the orbit slip in ``zeta``, and by the time the particle
    reaches the cavity that slip is already large enough for the RF to hand back a third of
    what was lost. The residual is not the bill; it is the bill minus what the ring
    accidentally pays on the way. Reconstructed from that mechanism below, which is what
    turns a refuted guess into a gate.
    """
    lat, icav = ring()
    old = np.zeros(6)
    old[:4] = closed_orbit_nonlinear(lat)
    old[DELTA] = closed_orbit_delta(lat)
    residual = turn(lat, old) - old
    ref = lat.ref

    assert int(np.argmax(np.abs(residual))) == ZETA
    assert residual[ZETA] == pytest.approx(2.720e-2, rel=1e-3)

    scale = ref.beta0**2 * ref.total_energy_eV
    bill = tracked_loss(lat, old) / scale
    assert residual[DELTA] / bill == pytest.approx(-0.6889, rel=0.01)
    # and the missing third is the cavity kick the slipped zeta already collects: the
    # turn ends at the cavity's entrance, so that slip *is* the phase the RF sees. The
    # reconstruction closes because between them these two are the only elements of the
    # ring that change delta at all.
    zeta_at_cavity = turn(lat, old)[ZETA]
    collected = cavity_kick_eV(lat, float(zeta_at_cavity), icav) / scale
    assert residual[DELTA] == pytest.approx(collected - bill, rel=1e-10)
    # It was bit-for-bit until P2 (iii) and is not any more, by a **derivable** amount
    # rather than a tolerated one. ``collected`` is the cavity's kick expressed in
    # ``p_zeta``; the cavity now converts that into ``delta`` exactly, and
    # ``delta = p_zeta - p_zeta^2/(2 gamma0^2)`` is nonlinear, so the delta it actually
    # delivers falls short of ``collected`` by ``d_cav*g/gamma0^2 + g^2/(2 gamma0^2)``
    # with ``d_cav`` the momentum arriving at the cavity and ``g = collected``. That is
    # ``-2.4e-14``, i.e. ``9.0e-12`` of the residual -- this is a ``gamma0 = 12720``
    # electron ring, which is exactly where the milestone said the term would be
    # invisible. Asserted as its closed form so that a *wrong* conversion would still
    # fail here rather than hide under a loosened equality.
    d_cav = at_cavity(lat, old, icav)[DELTA]
    leftover = (collected - bill) - residual[DELTA]
    predicted = d_cav * collected / ref.gamma0**2 + collected**2 / (2 * ref.gamma0**2)
    assert leftover == pytest.approx(predicted, rel=0.02)
    assert abs(leftover / residual[DELTA]) == pytest.approx(9.0e-12, rel=0.05)
    # it has to be the *tracked* loss: the design-route integral is 4.4e-3 away here, the
    # first-order lumping error that only the closed orbit cancels (see the exponent gate)
    assert energy_loss_per_turn(lat) / scale / bill - 1.0 == pytest.approx(4.40e-3, rel=0.01)


# ---------------------------------------------------------------------------
# Gate 2 — the closed form, on both of its routes.
# ---------------------------------------------------------------------------
def test_the_cavity_hands_back_exactly_the_turns_tracked_loss() -> None:
    """The identity arm: an equality at round-off, not a tolerance.

    This is the definition of the fixed point written out, so it *must* hold to round-off
    — and that is what makes it useful: it is the arm that needs no second model, and it
    fails the instant the solve returns something that is not a fixed point.
    """
    lat, icav = ring()
    orbit = closed_orbit_6d(lat, radiation="mean")
    zeta = float(at_cavity(lat, orbit, icav)[ZETA])

    # An equality in eV, bounded by the only thing that can spoil it: the residual the
    # Newton solve was allowed to stop at. ``tol`` is in state units, so a delta of ``tol``
    # is ``tol * beta0^2 E0`` of energy -- 6.5e-5 eV here, against a measured 1.7e-5.
    ref = lat.ref
    budget = 10.0 * 1e-14 * ref.beta0**2 * ref.total_energy_eV
    assert abs(cavity_kick_eV(lat, zeta, icav) - tracked_loss(lat, orbit)) < budget
    # and the arrival time is a real, visible distance, not a rounding correction
    assert zeta == pytest.approx(8.888e-2, rel=1e-3)


def test_the_design_route_integral_lands_on_the_tracked_loss_but_not_exactly() -> None:
    """The independent arm: ``energy_loss_per_turn`` is a different calculation entirely.

    A radiation integral over the design orbit against a sum of lumped kicks along the
    real one. They agree to 1.7e-7 on this ring, and the *reason* they do not agree
    exactly is the subject of the next test.
    """
    lat, icav = ring()
    orbit = closed_orbit_6d(lat, radiation="mean")
    zeta = float(at_cavity(lat, orbit, icav)[ZETA])

    u0 = energy_loss_per_turn(lat)
    assert cavity_kick_eV(lat, zeta, icav) == pytest.approx(u0, rel=1e-6)
    assert u0 / tracked_loss(lat, orbit) - 1.0 == pytest.approx(1.743e-7, rel=0.05)


def test_the_departure_is_second_order_on_the_closed_orbit_and_first_on_the_design_one() -> None:
    """The sharp gate, and it measures the fixed point rather than the radiation.

    The lumping error is first order in ``U0/E`` about the design orbit and second order
    about the closed one, because the closed orbit is *where the sag is centred*: the
    linear-in-``delta`` part of the loss cancels between the half turn spent above the
    design energy and the half spent below. An orbit that is wrong by any fraction of the
    sag puts the first-order term back and the exponent falls to 1.
    """
    losses, on_orbit, on_design = [], [], []
    for energy in (1.625e9, 3.25e9, 6.5e9, 13.0e9):
        lat, _ = ring(energy=energy, voltage=400.0e6)
        u0 = energy_loss_per_turn(lat)
        orbit = closed_orbit_6d(lat, radiation="mean")
        losses.append(u0 / energy)
        on_orbit.append(abs(u0 / tracked_loss(lat, orbit) - 1.0))
        on_design.append(abs(u0 / tracked_loss(lat, np.zeros(6)) - 1.0))

    x = np.log(losses)
    assert np.polyfit(x, np.log(on_orbit), 1)[0] == pytest.approx(2.0, abs=0.02)
    assert np.polyfit(x, np.log(on_design), 1)[0] == pytest.approx(1.0, abs=0.02)
    # over three decades of loss the two routes never cross: the closed orbit is better
    # everywhere, by between three and four orders of magnitude
    assert min(d / o for d, o in zip(on_design, on_orbit, strict=True)) > 1.0e3


# ---------------------------------------------------------------------------
# Gate 3 — where the closed form is read, which is not the lattice start.
# ---------------------------------------------------------------------------
def test_the_closed_form_holds_at_the_cavity_and_not_at_the_lattice_start() -> None:
    """Splice the cavity mid-lattice and the two arrival times differ by 10%.

    ``zeta`` slips through the arc as the orbit's own path length, so reading it at the
    lattice start solves an equation about a place the cavity is not. With the cavity last
    the distinction is invisible — the cavity is thin, so the turn *ends* at its entrance
    and the two points coincide exactly, which is why this ring exists.
    """
    lat, icav = ring(at=50)
    orbit = closed_orbit_6d(lat, radiation="mean")
    zeta_cav = float(at_cavity(lat, orbit, icav)[ZETA])
    zeta_start = float(orbit[ZETA])

    loss = tracked_loss(lat, orbit)
    assert cavity_kick_eV(lat, zeta_cav, icav) == pytest.approx(loss, rel=1e-11)
    assert zeta_start / zeta_cav - 1.0 == pytest.approx(-0.1025, rel=0.02)
    assert cavity_kick_eV(lat, zeta_start, icav) / loss - 1.0 == pytest.approx(-0.1002, rel=0.02)


def test_the_arrival_time_at_the_cavity_does_not_depend_on_where_the_ring_is_cut() -> None:
    """The same physical ring started at element 50 gives the same orbit, transported.

    A fixed point is a property of the ring, not of the arbitrary place the element list
    begins. Rotating the sequence and re-solving must reproduce the first solve propagated
    to the same point — including through the cavity, which the rotation moves.
    """
    lat, icav = ring()
    orbit = closed_orbit_6d(lat, radiation="mean")

    els = list(lat.elements)
    rotated = Lattice([*els[50:100], els[icav], *els[:50]], ref=lat.ref)
    theirs = closed_orbit_6d(rotated, radiation="mean")

    mine = np.asarray(orbit, dtype=float).copy()
    for elem in els[:50]:
        mine = elem.track(mine, lat.ref, radiation="mean")
    assert np.max(np.abs(mine - theirs)) < 1e-13


# ---------------------------------------------------------------------------
# Gate 4 — the reductions, and the shipped claim this milestone corrects.
# ---------------------------------------------------------------------------
def test_without_radiation_it_reduces_to_n5s_scalar_and_the_four_dimensional_solve() -> None:
    """Radiation off, on a ring steered off axis: the new solve contains the old ones.

    The corrector lengthens the closed orbit, which is what gives N5's ``delta_co`` a
    non-zero value to reproduce; without it every coordinate would be zero and the test
    would assert nothing.
    """
    lat, _ = ring(kick=2.0e-4)
    orbit = closed_orbit_6d(lat)

    # round-off, not exactly zero: unlike the undistorted ring below, here the seed is not
    # already the answer, so Newton takes real steps and ``zeta`` inherits their arithmetic
    assert abs(orbit[ZETA]) < 1e-16
    assert float(orbit[DELTA]) == pytest.approx(closed_orbit_delta(lat), rel=1e-9)
    assert float(orbit[DELTA]) != 0.0
    transverse = closed_orbit_nonlinear(lat, delta=float(orbit[DELTA]))
    assert np.max(np.abs(orbit[:4] - transverse)) < 1e-13


def test_an_accelerating_synchronous_phase_alone_does_not_move_the_arrival_time() -> None:
    """The correction to a shipped claim, asserted with ``==`` rather than a tolerance.

    ``closed_orbit_delta``'s docstring and ROADMAP N5 both said ``phi_s != 0`` needs a 6D
    fixed point. It does not, in this package: the cavity's kick vanishes at ``zeta = 0``
    for every ``phi_s``, and the reference-energy ramp that would make an accelerating
    bucket mean something lives inside :func:`accsim.accelerate`, which builds its own
    reference per turn and never touches this path. Tracked radiation is the only thing
    that moves ``zeta_co``.
    """
    for phi_s in (math.pi, math.pi - 0.3, math.pi - 1.0):
        lat, _ = ring(phi_s=phi_s)
        orbit = closed_orbit_6d(lat)
        assert orbit[ZETA] == 0.0
        assert orbit[DELTA] == 0.0
    # and the same ring with radiation on is nowhere near zero, so the assertion above
    # is about the phase and not about the ring being trivial
    lat, _ = ring(phi_s=math.pi - 0.3)
    assert abs(closed_orbit_6d(lat, radiation="mean")[ZETA]) > 1.0e-2


def test_a_ring_without_an_rf_cavity_is_refused_rather_than_iterated() -> None:
    """The fourth appearance of one degeneracy: ``J - I`` is singular in the long plane.

    Without a cavity nothing reads ``zeta``, so ``zeta`` and ``delta`` are both
    eigenvalue-1 directions and there is no isolated fixed point to converge to. N5 met
    this, N4 explained it, N3 first hit it. Refused, not iterated into nonsense.
    """
    lat, icav = ring()
    unbunched = Lattice(list(lat.elements[:icav]), ref=lat.ref)
    with pytest.raises(ClosedOrbitError, match="RF cavity"):
        closed_orbit_6d(unbunched)
    with pytest.raises(ClosedOrbitError, match="RF cavity"):
        closed_orbit_6d(unbunched, radiation="mean")


def test_a_stochastic_radiation_model_has_no_fixed_point_and_is_refused() -> None:
    """A random map has no fixed point; Newton on one would return whatever it drew."""
    lat, _ = ring()
    for model in ("quantum", "photons"):
        with pytest.raises(ValueError, match="stochastic"):
            closed_orbit_6d(lat, radiation=model)


# ---------------------------------------------------------------------------
# Gate 5 — the two-sided scaling: the knob that turns the effect off, and the
# shape of the curve it traces.
# ---------------------------------------------------------------------------
def test_the_reconstructed_loss_is_flat_in_the_voltage_while_the_arrival_time_is_not() -> None:
    """Raising the voltage moves ``zeta_co`` and changes nothing about the physics.

    The loss is a property of the magnets, so ``V sin(k zeta_co)`` must be the same number
    at every voltage while ``zeta_co`` itself sweeps a factor of seven. That is the
    two-sided gate: the effect is turned off by raising ``V``, not by anything about the
    lattice.
    """
    losses, zetas, voltages = [], [], (60.0e6, 90.0e6, 200.0e6, 400.0e6)
    for voltage in voltages:
        lat, icav = ring(voltage=voltage)
        orbit = closed_orbit_6d(lat, radiation="mean")
        zeta = float(at_cavity(lat, orbit, icav)[ZETA])
        losses.append(cavity_kick_eV(lat, zeta, icav))
        zetas.append(zeta)

    assert max(losses) / min(losses) - 1.0 < 1e-12
    assert zetas[0] / zetas[-1] == pytest.approx(6.87, rel=0.02)


def test_the_arrival_time_is_the_arcsine_and_not_its_linearisation() -> None:
    """``k zeta_co = arcsin(U/V)``, which is 3% above ``U/(kV)`` at the lowest voltage.

    The distinction matters because the linearisation is what a first attempt would write
    and it is right to within a few percent — small enough to hide inside a loose
    tolerance, large enough to be a wrong model. Both forms are checked at both ends, and
    the linear one is excluded at the low voltage while agreeing at the high one.
    """
    for voltage, expected in ((60.0e6, 0.03093), (400.0e6, 6.4e-4)):
        lat, icav = ring(voltage=voltage)
        orbit = closed_orbit_6d(lat, radiation="mean")
        zeta = float(at_cavity(lat, orbit, icav)[ZETA])
        k = lat.elements[icav].k_rf(lat.ref)
        u_over_v = tracked_loss(lat, orbit) / voltage

        assert k * zeta == pytest.approx(math.asin(u_over_v), rel=1e-12)
        assert k * zeta / u_over_v - 1.0 == pytest.approx(expected, rel=0.02)


def test_the_momentum_swings_by_exactly_the_turns_loss_across_the_cavity() -> None:
    """The other half of the fixed point: what the cavity restores, the arcs took.

    An exact statement, because the cavity is the only element that adds ``delta`` and the
    orbit closes. The *size* of the swing is the one number a user reads off this milestone
    that Stage 7 already predicts — ``U0/E`` — so it is checked against that too.
    """
    lat, icav = ring()
    orbit = closed_orbit_6d(lat, radiation="mean")
    ref = lat.ref

    before = at_cavity(lat, orbit, icav)
    after = lat.elements[icav].track(before, ref, radiation="mean")
    swing = float(after[DELTA] - before[DELTA])

    assert swing == pytest.approx(tracked_loss(lat, orbit) / (ref.beta0**2 * ref.total_energy_eV))
    assert swing == pytest.approx(energy_loss_per_turn(lat) / ref.total_energy_eV, rel=1e-5)
    assert swing == pytest.approx(3.816e-3, rel=1e-3)


# ---------------------------------------------------------------------------
# Gate 6 — the interface: seeding, convergence and the errors it raises.
# ---------------------------------------------------------------------------
def test_a_far_guess_lands_on_the_unstable_fixed_point_of_the_bucket_before() -> None:
    """The map has more than one fixed point, and this one is named rather than avoided.

    ``sin(k zeta) = U/V`` has two roots per RF period, and only the near one is the orbit a
    beam actually rides: the other is the **unstable** point on the far side of the bucket,
    where the RF slope has the wrong sign. Newton does not know the difference. Started
    half a metre away it converges, cleanly and to round-off, onto
    ``k zeta = pi - arcsin(U/V) - 2 pi``, i.e. the unstable point of the *previous* bucket
    — asserted here against that closed form, because a fixed point the function can
    return is worth pinning even when it is not the one anybody wants.

    So the contract matches :func:`closed_orbit_nonlinear`'s: the default seed is the 4D
    answer, which is the right orbit in the no-radiation limit and lands on the stable
    point; a far guess makes **no claim** about which fixed point comes back.
    """
    lat, icav = ring()
    default = closed_orbit_6d(lat, radiation="mean")
    k = lat.elements[icav].k_rf(lat.ref)

    near = default + np.array([0.0, 0.0, 0.0, 0.0, 1.0e-3, 0.0])
    assert np.max(np.abs(closed_orbit_6d(lat, near, radiation="mean") - default)) < 1e-13

    far = np.array([1.0e-3, 0.0, 5.0e-4, 0.0, 0.5, 5.0e-3])
    other = closed_orbit_6d(lat, far, radiation="mean")
    u_over_v = tracked_loss(lat, other) / lat.elements[icav].voltage
    assert float(other[ZETA]) == pytest.approx(-(math.pi + math.asin(u_over_v)) / k, rel=1e-9)
    # it *is* a fixed point -- the solve did not fail, it found a different answer
    assert np.max(np.abs(turn(lat, other) - other)) < 1e-14


def test_a_budget_too_small_to_converge_raises_rather_than_returning_a_near_miss() -> None:
    """One Newton step from the 4D seed is not enough, and the failure says so."""
    lat, _ = ring()
    with pytest.raises(ClosedOrbitError, match="did not converge"):
        closed_orbit_6d(lat, radiation="mean", max_iter=1, tol=1e-16)
