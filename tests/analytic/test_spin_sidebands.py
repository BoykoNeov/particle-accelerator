r"""N5: the invariant spin field once the ring has an RF cavity in it.

Every ring on N1-N4 was **unbunched** -- no cavity, ``delta`` an exact constant of the
motion, ``zeta`` read by nothing. N4's own write-up named what that bought and what it
cost: ``N[:, ZETA]`` is exactly zero "because nothing reads ``zeta`` -- which is what
makes accsim's six-column equation and xtrack's five-column one the same object, and
stops being true the moment an RF cavity enters". This file enters it.

**What actually breaks, and it is not the map.** An RF cavity is thin, so it does not
precess a spin, and N4's differencing machinery was six-column general from the start.
What breaks is one level down. :func:`accsim.spin._closed_state` carries the spin around
on the **4D** closed orbit with ``zeta = delta = 0``, and on a bunched ring that state is
**not a fixed point**: the closed orbit is longer than the design circumference, so one
turn moves ``zeta`` (by ``-8.3e-7 m`` on this file's ring), and the cavity converts that
slip into a ``delta`` kick. N4's exact one-turn rotation ``A`` and its shared ``(R, D)``
bundle then describe a trajectory that does not close.

**The fix is one scalar, and it has a closed form.** ``zeta_co = 0`` exactly -- accsim's
cavity subtracts ``sin(phi_s)``, so the synchronous particle sits at the zero crossing
whatever the RF frequency is -- and the ring instead locks its revolution period by
sitting **off momentum** until the path length matches:

    ``delta_co = -(Delta C / C) / alpha_c``.

That is :func:`accsim.orbit.closed_orbit_delta`, and it is the whole implementation of
this milestone. It matters here rather than being a rounding correction:
``nu_0 = G gamma (1 + delta)``, so ``delta_co`` moves the spin tune by ``5.4e-7`` -- five
percent of the distance at the closest point of the resonance scan below.

**The headline.** With RF the orbital spectrum is ``exp(+-2 pi i Q_x)``,
``exp(+-2 pi i Q_y)``, ``exp(+-2 pi i Q_s)``; the doubled eigenvalue ``1`` that N4's whole
design was arranged around is **gone**, and the Sylvester equation gains a pole at
``nu_0 = k +- Q_s``. Those are the synchrotron sidebands, and they are what actually
limits the polarization of a real high-energy electron ring. They are gated here exactly
as N4 gated its vertical one -- as a *location* (the pole extrapolates to ``Q_s``) and as
a *residue* (the denominator is ``2|sin(pi (nu_0 - Q_s))|`` and not one of four
alternatives) -- because a tolerance on a divergent quantity says nothing.

**What replaces N4's primary gate, which does not survive.** N4's sharpest check was
``N (D, 0, 1) = d/ddelta [n_0 closed at delta]``, and it leaned on ``delta`` being
conserved. With a cavity it is not, and there is no off-momentum closed orbit to
differentiate. What anchors the RF-on field instead is **continuity onto the RF-off one**:
both the new ``zeta`` column and the shift in ``dn/ddelta`` vanish as ``Q_s^2`` when the
voltage is taken down, landing on the value N4's identity already pinned. Gated as an
*order* by a fit, with the linear-in-``Q_s`` correction that rides on it named rather
than absorbed into a tolerance.

**And what nothing here can see.** The Derbenev-Kondratenko coefficient ``11/18``, for
the third milestone running -- every gate below is a ratio it survives, a location it
does not enter, or a scaling order it is constant against. It is reachable only from
``tests/reference/test_spin_sidebands_xtrack.py``, behind the skippable ``reference``
marker.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from accsim.coords import DELTA, PX, PY, ZETA, X, Y
from accsim.elements.rfcavity import RFCavity
from accsim.lattice import Lattice
from accsim.orbit import closed_orbit_delta, closed_orbit_nonlinear
from accsim.radiation import derbenev_kondratenko_polarization, sokolov_ternov_polarization
from accsim.reference import ELECTRON_ANOMALOUS_MOMENT as G_E
from accsim.reference import ELECTRON_MASS_EV, ReferenceParticle
from accsim.spin import (
    SpinResonanceError,
    _closed_state,
    closed_spin_solution,
    spin_orbit_coupling,
)
from accsim.tracking import Tracker
from accsim.twiss import momentum_compaction

# N2's bump ring and N4's re-energising helper are imported rather than rebuilt: this
# milestone is a *change of ring*, not a change of lattice, and a second copy of either
# would be a second chance to get one wrong. tests/ dirs are not import packages.
sys.path.insert(0, os.path.dirname(__file__))

import test_depolarization as n4  # noqa: E402
import test_polarization as n2  # noqa: E402

ENERGY = 5e9

#: The cavity. ``phi_s = 0`` is the stationary bucket this ring is stable in (it is above
#: transition with a *negative* charge, so the two signs in ``Qs^2 = -h eta q V cos phi_s``
#: cancel). ``harmonic`` rather than a raw frequency, because ``h`` is the design quantity
#: -- and because ``Q_s ~ sqrt(h V)`` makes it the cheap half of the knob: this ``Q_s`` of
#: ``0.05`` is reached at 2.3 MV instead of the 230 MV a harmonic of 1 would have needed
#: in a 51 m ring, which keeps the synchro-betatron coupling ``R56 . R65`` small enough
#: for the three orbital modes to stay identifiable.
HARMONIC = 100
VOLTAGE = 2.3e6


def bunched(energy_eV: float = ENERGY, voltage: float = VOLTAGE) -> Lattice:
    """N2's vertical-bump ring with a cavity appended."""
    lattice = n4.ring(energy_eV)
    cavity = RFCavity.from_harmonic(voltage, HARMONIC, lattice.length, lattice.ref, phi_s=0.0)
    return Lattice([*lattice.elements, cavity], lattice.ref)


def flat_bunched(energy_eV: float = ENERGY) -> Lattice:
    """The same ring with the vertical bump turned off -- the control for ``delta_co``."""
    lattice = n2.gate_ring(0.0, n2.electron(energy_eV))
    cavity = RFCavity.from_harmonic(VOLTAGE, HARMONIC, lattice.length, lattice.ref, phi_s=0.0)
    return Lattice([*lattice.elements, cavity], lattice.ref)


def orbital_modes(orbit_matrix: np.ndarray) -> dict[str, tuple[float, np.ndarray]]:
    r"""``{'x': (Q_x, E_x), 'y': (Q_y, E_y), 's': (Q_s, E_s)}`` from a 6D one-turn Jacobian.

    N4's two-mode version identified a mode by which plane its eigenvector lived *mostly*
    in, and that rule does not survive the third mode. On a dispersive ring the
    **horizontal** eigenvector's largest component is ``zeta`` -- a betatron oscillation
    changes the path length -- so "most content in ``(zeta, delta)``" picks out the wrong
    mode, and the first version of this file's scan found no pole at all because of it.

    The rule that does work: only the synchrotron mode has ``delta`` oscillating at its own
    frequency, so ``s`` is the mode with the largest ``|delta|``. ``y`` is taken first (it
    is uncoupled here and unambiguous), and ``x`` is what is left. The
    eigenvalue-``1`` directions N4 skipped are gone -- with RF there are none, which is
    :func:`test_the_bunched_spectrum_has_no_eigenvalue_one`.
    """
    values, vectors = np.linalg.eig(orbit_matrix)
    upper = [
        (float(np.angle(v)) / (2.0 * math.pi), w)
        for v, w in zip(values, vectors.T, strict=True)
        if np.angle(v) > 1e-9
    ]
    if len(upper) != 3:
        raise AssertionError(f"expected three oscillating modes, got {len(upper)}")
    vertical = max(upper, key=lambda m: abs(m[1][Y]) ** 2 + abs(m[1][PY]) ** 2)
    rest = [m for m in upper if m is not vertical]
    longitudinal = max(rest, key=lambda m: abs(m[1][DELTA]))
    horizontal = next(m for m in rest if m is not longitudinal)
    return {"x": horizontal, "y": vertical, "s": longitudinal}


def _sideband_ring(distance: float, sign: int = +1, energy_eV: float = ENERGY) -> Lattice:
    r"""The bunched ring re-energised so that ``nu_0 = k + sign * Q_s + distance``.

    **The knob is no longer clean, and this function is where that is paid for.** N4 could
    set the energy in two Newton steps on the spin tune alone, because every optical tune
    of its ring was energy-independent. ``Q_s`` is not: ``Q_s^2 ~ 1/E``, so moving the beam
    energy to place ``nu_0`` also moves the target ``nu_0`` is being placed against. The
    solve is therefore a *self-consistent* one -- ``Q_s`` re-measured from the same one-turn
    Jacobian the spin field is built from, at every step -- and it is what makes the
    location claim below mean anything.
    """
    energy = energy_eV
    for _ in range(6):
        coupling = spin_orbit_coupling(bunched(energy))
        target = sign * orbital_modes(coupling.orbit_matrix)["s"][0]
        gap = (target + distance - coupling.spin_tune + 0.5) % 1.0 - 0.5
        energy += gap * ELECTRON_MASS_EV / G_E
    return bunched(energy)


def _one_turn(lattice: Lattice, state: np.ndarray) -> np.ndarray:
    """One turn of the real map on a full 6D state."""
    return Tracker(lattice).track_once(np.asarray(state, dtype=float)[:, None])[:, 0]


# --- the closed orbit gains a momentum -----------------------------------------------


def test_a_bunched_ring_closes_off_momentum_and_the_closed_form_says_by_how_much():
    r"""``delta_co = -(Delta C / C) / alpha_c``, to seven digits.

    The physics in one line: the RF is locked to the *reference* revolution, the closed
    orbit is longer than the design circumference, and the only way for the beam to arrive
    at the same RF phase every turn is to sit off momentum until the path length matches.
    Both sides of the identity are the package's own -- ``Delta C`` is the one-turn ``zeta``
    slip of the on-momentum orbit, ``alpha_c`` is :func:`accsim.twiss.momentum_compaction`
    -- so this is not a remembered formula but two independent routes to one number.

    The sign is the content. A **longer** orbit needs a **lower** momentum (above
    transition, a lower-momentum particle takes a shorter path), so ``delta_co < 0`` where
    the slip is negative; getting ``alpha_c``'s sign or the ``zeta`` convention wrong flips
    it, and both flips are excluded by comparing the signed numbers rather than magnitudes.
    """
    lattice = bunched()
    on_momentum = np.zeros(6)
    on_momentum[[X, PX, Y, PY]] = closed_orbit_nonlinear(lattice, delta=0.0)
    excess = _one_turn(lattice, on_momentum)[ZETA] - on_momentum[ZETA]

    predicted = excess / (lattice.length * momentum_compaction(lattice))
    measured = closed_orbit_delta(lattice)

    assert excess < 0.0  # zeta = s - beta0 c t: a longer path arrives late
    assert measured < 0.0
    assert abs(measured - predicted) < 1e-6 * abs(measured)


def test_the_state_the_spin_rides_is_now_a_genuine_fixed_point():
    """One turn returns the carrier state to round-off, where before it did not.

    This is the defect the milestone exists to fix, measured from both sides on the same
    ring: at ``delta = 0`` the state is out by ``~1e-6`` in ``zeta`` (and, through the
    cavity, in ``delta``), and at ``delta_co`` it is out by round-off in all six.

    The ``1e-6`` looks harmless and is not:
    :func:`test_carrying_the_momentum_offset_moves_the_measured_tune_distance` shows what
    it does to this file's own headline.
    """
    lattice = bunched()
    naive = np.zeros(6)
    naive[[X, PX, Y, PY]] = closed_orbit_nonlinear(lattice, delta=0.0)

    delta_co = closed_orbit_delta(lattice)
    closed = np.zeros(6)
    closed[[X, PX, Y, PY]] = closed_orbit_nonlinear(lattice, delta=delta_co)
    closed[DELTA] = delta_co

    assert np.abs(_one_turn(lattice, naive) - naive).max() > 1e-7
    assert np.abs(_one_turn(lattice, closed) - closed).max() < 1e-16


def test_an_undistorted_ring_has_no_momentum_offset_at_all():
    """Bump off, ``delta_co == 0.0`` -- asserted with ``==``, not a tolerance.

    The control that makes the previous test a measurement of something. With the vertical
    bump at zero the closed orbit *is* the design orbit, every element is traversed on
    axis, the path length is the design circumference **identically**, and a milestone that
    returned some small non-zero number here would be reporting its own numerical noise as
    physics. Nothing about the cavity, the harmonic number or the voltage changes that.
    """
    assert closed_orbit_delta(flat_bunched()) == 0.0


def test_an_unbunched_ring_is_left_exactly_where_n4_left_it():
    """No cavity, ``delta_co == 0.0`` -- which is what keeps N1-N4 from moving.

    Without RF, ``delta`` is a free constant: *every* momentum closes, and the one that
    also closes ``zeta`` is a perfectly well-defined number (the same
    ``-(Delta C / C)/alpha_c``) that N1-N4 deliberately did not use, because nothing on
    those milestones reads ``zeta``. Adopting it here would silently re-baseline four
    milestones' worth of gates for no physics. The offset is therefore applied only when
    the one-turn map actually couples ``zeta`` into ``delta``, and this test is that
    boundary.
    """
    assert closed_orbit_delta(n4.ring()) == 0.0
    assert closed_orbit_delta(n2.gate_ring(2e-3, n2.electron(ENERGY))) == 0.0


def test_the_offset_is_quadratic_in_the_bump_amplitude():
    r"""``delta_co ~ amplitude^2``, because a path length is even in the orbit angle.

    A trajectory at angle ``theta`` to the axis is longer by ``theta^2/2`` per unit length
    and the bump's angles are all linear in its amplitude, so the excess -- and with it
    ``delta_co`` -- must be **quadratic**. Gated on the exponent rather than on a
    coefficient, in the shape J2 established: the coefficient is a property of this
    particular bump, the exponent is a property of the mechanism.
    """
    amplitudes = [4e-3, 2e-3, 1e-3, 5e-4]
    offsets = []
    for amplitude in amplitudes:
        lattice = n2.gate_ring(amplitude, n2.electron(ENERGY))
        cavity = RFCavity.from_harmonic(VOLTAGE, HARMONIC, lattice.length, lattice.ref, phi_s=0.0)
        offsets.append(abs(closed_orbit_delta(Lattice([*lattice.elements, cavity], lattice.ref))))

    for (a1, d1), (a2, d2) in zip(
        zip(amplitudes, offsets, strict=True),
        zip(amplitudes[1:], offsets[1:], strict=True),
        strict=False,
    ):
        assert abs(math.log(d1 / d2) / math.log(a1 / a2) - 2.0) < 0.02


# --- the field is no longer blind to zeta --------------------------------------------


def test_the_spin_field_now_has_a_longitudinal_column():
    r"""``N[:, ZETA]`` is non-zero with RF, where N4 asserted it bit-for-bit zero.

    The mechanism is exactly the one N4 named: ``zeta`` reaches the spin only through an
    element that turns it into ``delta``, and until this milestone the package had none in
    any spin lattice. It is still small -- ``2.7e-5`` against a matrix whose largest entry
    is order ``1e-2`` -- because it is the ``delta`` column seen through one cavity's
    ``R65``, which is what the ``Q_s^2`` law below says formally.

    The perpendicularity N4 gated survives untouched: ``n`` is a unit vector, so
    ``n_0 . N = 0`` for the new column as much as for the old six.
    """
    coupling = spin_orbit_coupling(bunched())
    assert np.linalg.norm(coupling.matrix[:, ZETA]) > 1e-6
    assert np.abs(coupling.n0 @ coupling.matrix).max() < 1e-9 * np.abs(coupling.matrix).max()


def test_the_bunched_spectrum_has_no_eigenvalue_one():
    r"""``R``'s six eigenvalues are three conjugate pairs, none of them ``1``.

    The structural difference this whole milestone follows from, and the third appearance
    of the same degeneracy on this axis. Without RF, ``zeta`` and ``delta`` are both fixed
    directions of the one-turn map and ``R`` carries a doubled eigenvalue ``1``; N3 met it
    as xtrack's ``twiss`` refusing a flat ring, N4 as ``inv(I - A)`` being singular for
    every ring, and this milestone's own 6D fixed-point solve would meet it a third time as
    ``I - J``.

    With a cavity it is simply not there: the closest approach to ``1`` is
    ``2|sin(pi Q_s)| = 0.31``, and that is asserted as the *identity* it is rather than as
    "far from 1".
    """
    coupling = spin_orbit_coupling(bunched())
    values = np.linalg.eigvals(coupling.orbit_matrix)
    q_s = orbital_modes(coupling.orbit_matrix)["s"][0]

    closest = float(np.min(np.abs(values - 1.0)))
    assert abs(closest - 2.0 * abs(math.sin(math.pi * q_s))) < 1e-6
    assert np.allclose(np.abs(values), 1.0, atol=1e-8)


def test_the_new_column_and_the_momentum_shift_both_vanish_as_the_synchrotron_tune_squared():
    r"""The gate that replaces N4's dispersion identity, and it is an **order**.

    N4's primary check -- ``N (D, 0, 1) = d/ddelta [n_0 closed at delta]`` -- does not exist
    on a bunched ring: ``delta`` is not conserved and there is no off-momentum closed orbit
    to differentiate. What anchors the RF-on field instead is continuity onto the RF-off
    one, which *that* identity pinned to ``5e-9``. Taking the voltage down at fixed energy,
    both quantities the cavity created must die as ``Q_s^2``:

    - ``|N[:, ZETA]|``, which exists only through the cavity's ``R65 ~ Q_s^2``;
    - ``|N[:, DELTA] - N4's|``, the shift in the column that already existed.

    **Measured as a fitted exponent, and the residual is named rather than absorbed.** The
    ratios ``|.|/Q_s^2`` are not flat -- they run ``1.030e-2, 1.087e-2, 1.103e-2, 1.107e-2``
    over a factor of eight in ``Q_s``, with steps that halve as ``Q_s`` halves. That is a
    linear-in-``Q_s`` correction riding on a quadratic law, exactly what a next-order term
    looks like, and quoting the ratio as "constant to 0.4%" would have been the easy
    mistake. The exponent is therefore fitted between successive voltages and required to
    *approach* 2, which it does: ``2.09, 2.03, 2.01``.
    """
    reference = spin_orbit_coupling(n4.ring(ENERGY)).matrix[:, DELTA]

    tunes, zeta_column, momentum_shift = [], [], []
    for voltage in (9.2e6, 2.3e6, 5.75e5, 1.4375e5):
        coupling = spin_orbit_coupling(bunched(ENERGY, voltage))
        tunes.append(orbital_modes(coupling.orbit_matrix)["s"][0])
        zeta_column.append(float(np.linalg.norm(coupling.matrix[:, ZETA])))
        momentum_shift.append(float(np.linalg.norm(coupling.matrix[:, DELTA] - reference)))

    for series in (zeta_column, momentum_shift):
        exponents = [
            math.log(series[i] / series[i + 1]) / math.log(tunes[i] / tunes[i + 1])
            for i in range(len(series) - 1)
        ]
        assert all(abs(e - 2.0) < 0.12 for e in exponents), exponents
        # and it is *approaching* 2 rather than sitting near it by luck
        assert abs(exponents[-1] - 2.0) < abs(exponents[0] - 2.0)


def test_a_spin_launched_on_the_field_stays_on_it_for_forty_turns():
    r"""The primary gate: track the field and check it is actually invariant.

    N4's primary gate was an *identity* -- ``N (D, 0, 1) = d/ddelta [n_0 closed at delta]``
    -- reachable without going near the Sylvester solve. On a bunched ring it does not
    exist. What replaces it is the definition itself, checked the only way that needs no
    solve at all: put a particle at ``x`` off the closed orbit with its spin at
    ``n_0 + N x``, **track it**, and require that at every later turn its spin is still
    ``n_0 + N x(turn)`` for the orbit deviation it then has. That is what "invariant spin
    field" means, and nothing but the tracking map goes into it.

    **It discriminates at first order, which is what makes it the primary gate.** The field
    is first-order in ``x``, so the true residual is ``O(x^2)`` and the *relative* residual
    -- measured against ``|N x|`` -- must fall **linearly with the amplitude**. It does:
    ``5.6e-4``, ``5.6e-5``, ``5.6e-6`` over three decades. A matrix wrong by a fraction
    ``f`` instead leaves a relative residual that sits at ``f`` and does not move with the
    amplitude at all, so this gate reads a wrong ``N`` off directly rather than through a
    tolerance. ``tests/reference/test_spin_sidebands_xtrack.py`` is where that distinction
    does real work.

    The amplitudes stop at ``1e-5`` because the spin deviation being measured is only
    ``3e-10`` there and the residual has reached its round-off floor: below that the ratio
    stops improving, which is a property of double precision and not of the field.

    The displacement is along the **synchrotron** eigenvector, because that is the mode
    this milestone adds and the one the sideband makes large.
    """
    lattice = bunched()
    coupling = spin_orbit_coupling(lattice)
    closed = _closed_state(lattice, coupling.orbit, coupling.delta)
    eigenvector = np.real(orbital_modes(coupling.orbit_matrix)["s"][1])
    direction = eigenvector / np.linalg.norm(eigenvector)

    def residual(amplitude: float, turns: int = 40) -> float:
        """Worst departure from ``n_0 + N x`` over ``turns``, relative to ``|N x|``."""
        offset = amplitude * direction
        state = (closed + offset)[:, None]
        spin = coupling.n0 + coupling.matrix @ offset
        spin = (spin / np.linalg.norm(spin))[:, None]

        tracker = Tracker(lattice)
        worst = 0.0
        for _ in range(turns):
            state, spin = tracker.track_once_with_spin(state, spin)
            expected = coupling.n0 + coupling.matrix @ (state[:, 0] - closed)
            expected /= np.linalg.norm(expected)
            worst = max(worst, float(np.linalg.norm(spin[:, 0] - expected)))
        return worst / float(np.linalg.norm(coupling.matrix @ offset))

    relative = [residual(a) for a in (1e-3, 1e-4, 1e-5)]
    assert relative[0] < 1e-2
    for coarse, fine in zip(relative, relative[1:], strict=False):
        assert 5.0 < coarse / fine < 20.0


# --- the sidebands, which are the milestone ------------------------------------------


@pytest.fixture(scope="module")
def upper_scan() -> list[tuple[float, float, float, float, float, float]]:
    """``(distance, nu_0, Q_s, Q_x, Q_y, |N E_s|)`` approaching ``nu_0 = k + Q_s``."""
    rows = []
    for distance in (1e-2, 1e-3, 1e-4, 1e-5):
        lattice = _sideband_ring(distance)
        coupling = spin_orbit_coupling(lattice)
        modes = orbital_modes(coupling.orbit_matrix)
        q_s, e_s = modes["s"]
        rows.append(
            (
                distance,
                coupling.spin_tune,
                q_s,
                modes["x"][0],
                modes["y"][0],
                float(np.linalg.norm(coupling.matrix @ e_s)),
            )
        )
    return rows


def test_the_pole_extrapolates_to_the_synchrotron_tune(upper_scan):
    r"""``1/|N E_s|`` is linear in the distance and hits zero at ``Q_s``, not elsewhere.

    N4's location gate, shifted from ``Q_y`` to ``Q_s``. A first-order pole means
    ``1/|N E_s|`` is a straight line through the resonance, so the two closest points
    extrapolate to it; the claim is that the zero of that line is the *synchrotron* tune,
    a quantity measured off the same one-turn Jacobian and never assumed.

    A tolerance on ``|N E_s|`` itself would say nothing -- it is divergent, so any
    implementation with a pole roughly here passes. The extrapolated zero is what pins
    *which* pole.
    """
    (_, nu1, _, _, _, amp1), (_, nu2, q_s, _, _, amp2) = upper_scan[-2], upper_scan[-1]
    y1, y2 = 1.0 / amp1, 1.0 / amp2
    zero = nu1 - y1 * (nu2 - nu1) / (y2 - y1)

    assert abs(zero - q_s) < 1e-6
    # ... and not at the integer resonance N2 owns, a quarter of a tune away
    assert abs(zero - round(zero)) > 0.04


def test_the_residue_identifies_the_denominator_as_the_synchrotron_sideband(upper_scan):
    r"""``|N E_s| . 2|sin(pi (nu_0 - Q_s))|`` is constant; four alternatives are not.

    The sharper half of the pair. ``|N E_s|`` runs over a factor of ``1000`` across the
    scan, so a *constant* product is a strong statement about the denominator -- and the
    alternatives fail loudly, because each of them is regular across the scan and therefore
    scales with ``|N E_s|`` itself.

    The four excluded denominators are chosen to be the ones a plausible bug produces: the
    other sideband ``nu_0 + Q_s`` (a sign), the imperfection resonance ``nu_0`` (forgetting
    the orbital mode), and the two *other* modes' sidebands ``nu_0 - Q_y`` and
    ``nu_0 - Q_x`` (identifying the wrong eigenvector -- which is not hypothetical: the
    first version of :func:`orbital_modes` did exactly that, see its docstring).

    **The far end of the scan is left out, and named rather than hidden.** At ``1e-2`` the
    residue is still ``19%`` below its limit, because a pole is only the whole of the answer
    where it dominates and there is a regular background underneath it. N4 met the same
    thing one decade closer in (``32%`` at ``1e-3`` on its ring). So the constancy is
    asserted over the three *resonant* points, and the far point's departure is asserted
    too -- as a measured size, so that a version of this file which quietly widened the
    tolerance to swallow it would fail here instead.
    """

    def residues(phase_of) -> list[float]:
        return [row[5] * 2.0 * abs(math.sin(math.pi * phase_of(row))) for row in upper_scan]

    def spread(values) -> float:
        return (max(values) - min(values)) / min(values)

    correct = residues(lambda r: r[1] - r[2])
    assert spread(correct[1:]) < 0.02
    # ... and it is converging onto that limit from below, not sitting near it by luck
    assert correct[0] < correct[1] < correct[2] < correct[3]
    assert 0.15 < (correct[-1] - correct[0]) / correct[-1] < 0.25

    for name, phase_of in (
        ("nu_0 + Q_s", lambda r: r[1] + r[2]),
        ("nu_0", lambda r: r[1]),
        ("nu_0 - Q_y", lambda r: r[1] - r[4]),
        ("nu_0 - Q_x", lambda r: r[1] - r[3]),
    ):
        assert spread(residues(phase_of)[1:]) > 20.0, name


def test_the_lower_sideband_is_there_too_and_is_a_separate_pole():
    r"""``nu_0 = k - Q_s`` diverges the same way, at a different energy.

    A Sylvester equation is singular when the two spectra meet, and ``A``'s reduced
    eigenvalues are a **conjugate pair** ``exp(-+2 pi i nu_0)`` -- so each orbital mode
    contributes *two* poles, not one. Building only the upper one would leave a
    sign-flipped implementation of the reduction passing every other gate in this file.
    """
    coupling = spin_orbit_coupling(_sideband_ring(1e-4, sign=-1))
    modes = orbital_modes(coupling.orbit_matrix)
    q_s, e_s = modes["s"]

    amplitude = float(np.linalg.norm(coupling.matrix @ e_s))
    assert amplitude > 1.0  # divergent: the on-resonance ring below refuses outright
    assert abs((coupling.spin_tune + q_s) % 1.0 - 1e-4) < 1e-6


def test_the_vertical_plane_ignores_the_sideband_while_the_horizontal_one_is_dragged_in(
    upper_scan,
):
    r"""``N``'s vertical columns are flat across the scan; its horizontal ones diverge with it.

    Written expecting *all four* transverse columns to stay put, on the reasoning that a
    longitudinal resonance is a longitudinal thing. Half of that was wrong, and the half
    that was wrong is the more interesting one.

    The resonant part of ``N`` lives entirely along the synchrotron eigenvector -- it is
    ``(N E_s)`` times the covector dual to ``E_s`` -- so which *columns* it appears in is
    decided by which coordinates ``E_s`` has content in. On a **dispersive** ring the
    synchrotron mode is not purely longitudinal: a momentum oscillation drives a horizontal
    one through the dispersion. So the horizontal columns carry the pole too, growing by a
    factor of ``900`` across the scan alongside ``|N E_s|``'s ``1200``, while the vertical
    columns -- whose mode has no dispersion here to couple through -- move by ``3%``.

    That is a sharper statement than the one it replaced: it says the sideband reaches the
    horizontal plane *through the dispersion*, and it would fail on a ring where the two
    were confused.
    """
    horizontal, vertical = [], []
    for distance in (1e-2, 1e-5):
        coupling = spin_orbit_coupling(_sideband_ring(distance))
        horizontal.append(float(np.abs(coupling.matrix[:, [X, PX]]).max()))
        vertical.append(float(np.abs(coupling.matrix[:, [Y, PY]]).max()))

    assert 0.95 < vertical[0] / vertical[1] < 1.05
    assert horizontal[1] / horizontal[0] > 500.0
    assert upper_scan[-1][5] / upper_scan[0][5] > 500.0


def test_a_ring_exactly_on_the_sideband_refuses():
    """``SpinResonanceError`` rather than a large number, at ``nu_0 = k + Q_s``.

    N4 wired the guard generically -- every orbital eigenvalue against
    ``exp(+-2 pi i nu_0)`` -- so the synchrotron mode is caught by machinery that was
    written before there was a synchrotron mode to catch. That is worth asserting rather
    than assuming: the guard could have been quietly specific to the two betatron pairs.
    """
    with pytest.raises(SpinResonanceError, match="spin-orbit resonance"):
        spin_orbit_coupling(_sideband_ring(0.0))


def test_the_vertical_tune_is_the_only_one_that_survives_the_scan(upper_scan):
    r"""``Q_y`` is fixed to ``1e-9``; ``Q_s`` and ``Q_x`` move, and that is why the solve iterates.

    N4's ``test_the_orbital_tunes_do_not_move_across_the_scan`` asserted that the beam
    energy moves ``nu_0 = G gamma`` and nothing else, which is what let it set the energy in
    two Newton steps. On a bunched ring that is **false**, and the milestone has to say so
    rather than inherit the claim:

    - ``Q_s^2 ~ 1/E``, so the target moves with the thing being aimed at it;
    - ``Q_x`` picks up a synchro-betatron shift through ``R56 . R65``.

    ``Q_y`` is untouched because the vertical plane has no dispersion here. The test asserts
    all three -- that the one is fixed and that the other two are not -- so that a future
    edit which accidentally froze the energy would fail rather than pass more easily.
    """
    q_s = [row[2] for row in upper_scan]
    q_x = [row[3] for row in upper_scan]
    q_y = [row[4] for row in upper_scan]

    assert max(q_y) - min(q_y) < 1e-9
    assert max(q_s) - min(q_s) > 1e-5
    assert max(q_x) - min(q_x) > 1e-6


def test_carrying_the_momentum_offset_moves_the_measured_tune_distance():
    r"""Dropping ``delta_co`` shifts ``nu_0`` by ``5.4e-7`` -- 5% of the closest scan point.

    The reason this milestone is an implementation and not only a test file, quantified on
    its own headline. ``nu_0 = G gamma (1 + delta)``, so a closed orbit sitting at
    ``delta_co = -4.8e-8`` moves the spin tune by ``(d nu_0/d delta) delta_co``. At the
    ``1e-5`` end of the scan that is a five percent systematic on the very quantity the
    location gate extrapolates, and it would have been invisible: it biases the *whole*
    scan the same way, so the pole would still look like a clean straight line, just one
    aimed slightly off ``Q_s``.

    **And the slope is not ``G gamma``, which is the finding inside the finding.** The
    obvious guess -- ``nu_0 = G gamma`` and ``gamma`` scales with the momentum, so
    ``d nu_0/d delta = G gamma`` -- is **43% too big**. Measured on this ring the slope is
    ``0.7003 G gamma``, and it is ``0.7003`` with the vertical bump on *or off*, so it is a
    property of the arc rather than of the distortion: an off-momentum closed orbit rides
    the dispersion through the thin quadrupoles, sees their feed-down field, and takes a
    different path length through the dipoles. The gate therefore *measures* the slope, by
    the same off-momentum closed spin solution N4 built its dispersion identity on, rather
    than quoting ``G gamma`` -- and asserts the bump-independence that identifies it as the
    arc's, so the difference from ``G gamma`` is recorded rather than hidden inside a loose
    tolerance.
    """
    lattice = _sideband_ring(1e-4)
    delta_co = closed_orbit_delta(lattice)
    gamma = lattice.ref.total_energy_eV / ELECTRON_MASS_EV

    step = 1e-7
    slope = (
        closed_spin_solution(lattice, delta=step).spin_tune
        - closed_spin_solution(lattice, delta=-step).spin_tune
    ) / (2.0 * step)

    carried = closed_spin_solution(lattice).spin_tune
    dropped = closed_spin_solution(lattice, delta=0.0).spin_tune

    assert abs(carried - dropped - slope * delta_co) < 1e-3 * abs(slope * delta_co)
    assert 1e-7 < abs(carried - dropped) < 1e-6

    # The slope is a property of the *arc*, not of the bump, so it is gated as that
    # insensitivity plus a loose band rather than as four digits: the ring is shared with
    # N2-N4, and a future edit to its cell count or bend length should not fail here with a
    # confusing message about a hard-coded constant.
    naked = n2.gate_ring(0.0, n2.electron(lattice.ref.total_energy_eV))
    bare = (
        closed_spin_solution(naked, delta=step).spin_tune
        - closed_spin_solution(naked, delta=-step).spin_tune
    ) / (2.0 * step)
    assert slope == pytest.approx(bare, rel=1e-4)
    assert 0.5 < slope / (G_E * gamma) < 0.9


# --- and what it does to the polarization --------------------------------------------


def test_the_polarization_collapses_on_the_synchrotron_sideband():
    r"""``P_eq`` dies on the sideband while Sokolov-Ternov's ``P_inf`` does not notice.

    The physical point of the whole milestone: a synchrotron sideband of the imperfection
    resonance depolarizes a stored electron beam, and this is the mechanism by which it
    does -- through ``dn/ddelta``, which is what the Derbenev-Kondratenko rate is built
    from, and which the sideband pole drives.

    ``P_inf`` is the control, in the shape N3 established. It is a *ratio*, and its
    numerator and denominator are both integrals over the closed orbit alone, so it cannot
    see a resonance in the field around that orbit. Its only motion across these two rings
    is its own ``1/(G gamma)`` energy dependence -- a sixth-digit drift here, since this
    scan moves the beam energy further than N4's did -- and asserting that is what stops a
    future edit from reporting a change of energy as a change of physics.

    **The sideband is already doing most of its damage a hundredth of a tune out.** At
    ``1e-2`` the polarization is ``-0.463``, half of ``P_inf``'s ``-0.9238``; by ``1e-4``
    it is ``-4e-5``. The "off-resonance" ring in this test is therefore not an unspoiled
    one, and it is not claimed to be -- what is asserted is the *collapse*, four orders of
    magnitude of it, against a control that does not move at all.
    """
    off = _sideband_ring(1e-2)
    on = _sideband_ring(1e-4)

    weak, strong = derbenev_kondratenko_polarization(off), derbenev_kondratenko_polarization(on)
    assert abs(weak) > 0.4
    assert abs(strong) < 1e-3
    assert abs(weak / strong) > 1e4

    control = abs(sokolov_ternov_polarization(off) - sokolov_ternov_polarization(on))
    assert 0.0 < control < 1e-5 * abs(sokolov_ternov_polarization(off))


def test_n0_and_the_spin_tune_do_not_notice_the_sideband(upper_scan):
    r"""N2's objects are regular across a scan that makes N4's object diverge.

    The separation N4 drew between ``n_0`` and the field around it, re-drawn on the new
    resonance. ``n_0`` rides the closed orbit and therefore sees only one-turn-periodic
    perturbations -- integer harmonics -- so a *sideband* cannot touch it. Its tilt is
    **monotone** across the scan (a pole is not), and the spin tune tracks what
    :func:`_sideband_ring` set it to.
    """
    tilts = []
    for distance in (1e-2, 1e-3, 1e-4, 1e-5):
        solution = closed_spin_solution(_sideband_ring(distance))
        tilts.append(float(solution.n0[X]))

    differences = np.diff(tilts)
    assert np.all(differences > 0) or np.all(differences < 0)
    assert max(abs(t) for t in tilts) / min(abs(t) for t in tilts) < 2.0


def test_the_reference_particle_is_the_species_this_file_thinks_it_is():
    """A guard, not physics: the whole file is an electron ring above transition.

    ``Qs^2 = -(h eta q V cos phi_s)/(2 pi beta0^2 E0)`` is positive here only because the
    charge is negative and the ring is above transition -- two sign flips that cancel. A
    ``phi_s = 0`` cavity on a *positive* particle would give this ring no bucket at all, and
    every gate above would fail in a confusing place rather than here.
    """
    lattice = bunched()
    assert isinstance(lattice.ref, ReferenceParticle)
    assert lattice.ref.charge < 0.0
    assert momentum_compaction(lattice) > 1.0 / lattice.ref.gamma0**2
