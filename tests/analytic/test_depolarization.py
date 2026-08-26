r"""N4 -- the invariant spin field, its resonance, and the depolarization it drives.

N3 built the polarization synchrotron radiation *creates*. This is what fights it. A
particle that is not on the closed orbit has its own periodic spin direction --- the
**invariant spin field** ``n(x) = n_0 + N x`` --- and every photon it emits jumps its
``delta``, and therefore its ``n``, somewhere else. The spread that opens up is
depolarization, and it enters the Derbenev-Kondratenko rates as two more ``kappa^3``
integrals over ``dn/ddelta``:

    ``alpha_plus  = alpha_plus_co  + (11/18) <kappa^3 |dn/ddelta|^2>``,
    ``alpha_minus = alpha_minus_co -        <kappa^3  dn/ddelta . b>``.

**Where the milestone's content actually is.** ``N`` solves ``A N - N R = -D`` --- a
Sylvester equation in the one-turn spin rotation ``A`` (N2's, exact), the one-turn
orbital Jacobian ``R``, and ``D = d(spin out)/d(orbit in)``. A Sylvester equation is
solvable exactly when the two spectra are disjoint, so ``N`` blows up when the spin
comes back in step with an orbital mode:

    ``nu_0 = k``            --- integer: N2's *imperfection* resonance, which moves ``n_0``,
    ``nu_0 = k +- Q_y``     --- the **intrinsic** resonance, which does not.

That second family is what N2 was written expecting and did not find, because ``n_0``
rides the closed orbit and sees only one-turn-periodic perturbations. It lives here
instead, in the field *around* ``n_0``, and this file's central gate is its **location**:
a pole that extrapolates to ``Q_y`` and not to an integer, with no coefficient anywhere
in the claim.

Four gates carry the file, in the order they need to hold:

- **The field itself, checked without the solve.** Without RF, ``delta`` is an exact
  constant of the motion, so the eigenvalue-``1`` direction of ``R`` *is* the dispersion
  and a particle on it sits on the closed orbit of a different momentum. Therefore
  ``N (D_x, D_px, D_y, D_py, 0, 1)`` must equal the derivative of the *off-momentum
  closed spin solution*, which is computed by closing an orbit at ``delta`` and reading
  ``n_0`` off it --- a route that touches neither ``A``'s off-diagonal structure, nor
  ``D``, nor the Sylvester solve
  (:func:`test_the_dispersion_direction_reproduces_the_off_momentum_solution`).

- **The pole's location, twice over.** Extrapolating ``1/|N E_y|`` to zero lands on
  ``Q_y`` to ``1e-6`` in tune, a quarter of a unit away from the nearest integer; and the
  residue ``|N E_y| * 2 |sin(pi (nu_0 - Q_y))|`` is constant to 1.5% while ``|N E_y|``
  itself varies thirtyfold. Both alternative denominators --- ``sin(pi nu_0)`` (the
  integer resonance) and ``sin(pi (nu_0 + Q_y))`` --- are asserted excluded, by a factor
  of twenty.

- **The separation from N2.** At the intrinsic resonance ``n_0`` and ``nu_0`` do not
  move, the horizontal mode does not diverge, and only the vertical one does. That is
  the entire reason this milestone is not N2's.

- **The collapse.** ``P_eq`` falls from ``-0.92`` to ``-0.02`` as the spin tune closes to
  ``1e-5`` of ``Q_y``, while ``P_inf`` --- N3's number, which cannot see any of this ---
  sits unmoved at nine digits. The depolarization grows as the **inverse square** of the
  tune distance, gated as an *order* rather than a tolerance (J2's lesson) and measured
  close enough in that the non-resonant background has died away.

**What is degenerate here, for the fourth time on this axis.** A flat ring has no
vertical motion, so nothing on the orbit ever produces a horizontal field, every rotation
is about ``y``, and a ``delta`` perturbation only changes how fast a spin turns about the
axis it already lies along. ``dn/ddelta`` is then **exactly zero** and ``P_eq == P_inf``
bit for bit. Asserted, as N3 asserted its own blindnesses.

**And what nothing here can see.** ``11/18`` is the Derbenev-Kondratenko coefficient. It
multiplies one of four integrals, and every gate in this file is either a *ratio* it
survives, a *location* it does not enter, or a *scaling order* it is constant against. It
is reachable only from ``tests/reference/test_depolarization_xtrack.py``, behind the
skippable ``reference`` marker --- the same warning N3 ended on, and for the same reason.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from accsim.coords import DELTA, DIM, PX, PY, ZETA, X, Y
from accsim.lattice import Lattice
from accsim.radiation import (
    depolarization_integrals,
    derbenev_kondratenko_polarization,
    polarization_buildup_time,
    polarization_integrals,
    polarization_time,
    sokolov_ternov_polarization,
)
from accsim.reference import ELECTRON_ANOMALOUS_MOMENT as G_E
from accsim.reference import ELECTRON_MASS_EV
from accsim.spin import (
    SpinResonanceError,
    closed_spin_solution,
    propagate_spin_orbit_coupling,
    spin_orbit_coupling,
)

# The bump ring is N2's and N3's, imported rather than rebuilt: the milestone is about a
# *tilted* n_0, and a second copy of a closed-bump solver is a second chance to get it
# wrong. tests/ dirs are not import packages, so it is reached by path.
sys.path.insert(0, os.path.dirname(__file__))

import test_polarization as gate  # noqa: E402

ENERGY = 5e9
AMPLITUDE = 2e-3

#: ``8 / (5 sqrt3)``, Sokolov-Ternov's ratio -- the polarization a ring reaches when
#: nothing depolarizes it.
P_ST = gate.P_ST


# --- the lattices, and the one knob this file turns ----------------------------------


def ring(energy_eV: float = ENERGY) -> Lattice:
    """N2's vertical-bump ring at a chosen energy."""
    return gate.gate_ring(AMPLITUDE, gate.electron(energy_eV))


def orbital_modes(orbit_matrix: np.ndarray) -> dict[str, tuple[float, np.ndarray]]:
    r"""``{'x': (Q_x, E_x), 'y': (Q_y, E_y)}`` from a one-turn Jacobian.

    The modes are identified by **eigenvector content** -- which plane the eigenvector
    lives mostly in -- and never by position in :func:`numpy.linalg.eig`'s output, whose
    ordering is not stable across the small changes in ``R`` an energy scan makes. Only
    the positive-frequency member of each conjugate pair is kept; the eigenvalue-``1``
    directions (``delta`` and ``zeta``, in a ring with no RF) are skipped.
    """
    values, vectors = np.linalg.eig(orbit_matrix)
    modes: dict[str, tuple[float, np.ndarray]] = {}
    for value, vector in zip(values, vectors.T, strict=True):
        phase = float(np.angle(value))
        if phase <= 1e-9:
            continue
        vertical = abs(vector[Y]) ** 2 + abs(vector[PY]) ** 2
        horizontal = abs(vector[X]) ** 2 + abs(vector[PX]) ** 2
        modes["y" if vertical > horizontal else "x"] = (phase / (2.0 * math.pi), vector)
    return modes


def vertical_tune() -> float:
    """``Q_y`` of the bump ring -- energy-independent, and this file leans on that."""
    return orbital_modes(spin_orbit_coupling(ring()).orbit_matrix)["y"][0]


def ring_at_tune_distance(distance: float, energy_eV: float = ENERGY) -> Lattice:
    r"""The bump ring re-energised so that ``nu_0 = k + Q_y + distance``.

    ``nu_0 = G gamma`` on a flat ring, so the beam energy is the knob that moves the spin
    tune, and it is the *only* knob that does: the lattice is specified by normalised
    strengths, so every optical quantity -- ``Q_y`` included -- is untouched by it. That
    is what makes the resonance-location claim sharp, and
    :func:`test_the_orbital_tunes_do_not_move_across_the_scan` checks it rather than
    assuming it.

    The bump shifts ``nu_0`` off ``G gamma`` at second order in its own kick (N2), so the
    energy is refined by two Newton steps on the *measured* spin tune, which is linear
    in the energy to far better than the ``1e-9`` these gates need.
    """
    target = vertical_tune() + distance
    energy = energy_eV
    for _ in range(2):
        current = closed_spin_solution(ring(energy)).spin_tune
        energy += ((target - current + 0.5) % 1.0 - 0.5) * ELECTRON_MASS_EV / G_E
    return ring(energy)


# --- the field itself, and the route that does not go through the solve ---------------


def test_the_dispersion_direction_reproduces_the_off_momentum_solution():
    r"""``N (D, 0, 1)`` is ``d/ddelta`` of the closed spin solution -- checked two ways.

    **The one gate on ``N`` that does not use ``N``'s machinery.** Without an RF cavity
    ``delta`` is an exact constant of the motion, so a particle displaced along
    ``(D_x, D_px, D_y, D_py, 0, 1) * eps`` is not merely near a closed orbit -- it *is*
    on the closed orbit of momentum ``eps``. Its invariant spin direction is therefore
    that ring's ``n_0``, and

        ``N (D, 0, 1) = d/ddelta [ n_0 closed at delta ]``.

    The right-hand side is computed by closing an orbit at ``+-ddelta`` and taking the
    null space of ``R_spin - I`` on it: no Sylvester solve, no ``D`` matrix, no
    perpendicular reduction, and a different closed orbit each time. The two agree to
    ``5e-8``, which is the central difference's own accuracy and not this milestone's.

    **The second assertion is what stops this from being a tautology.** The quantity the
    Derbenev-Kondratenko formula wants is ``N[:, DELTA]`` -- the partial derivative at
    *fixed transverse coordinates*, because a photon emission is instantaneous and moves
    nothing but ``delta``. That is a different vector from the dispersion-direction
    derivative above, by the ``N[:, :4] D`` the dispersion contributes, and on this ring
    they differ by a factor of 2.1. A version of this package that quietly used one for the
    other would pass every *shape* gate in this file and be wrong by that factor, so the
    gap is asserted -- at 1.5 rather than at the measured 2.1, since the ratio is a property
    of this particular ring's dispersion and not a number the physics pins.
    """
    from accsim.orbit import closed_orbit_nonlinear

    lattice = ring()
    coupling = spin_orbit_coupling(lattice)

    step = 1e-6
    dispersion = (
        closed_orbit_nonlinear(lattice, delta=+step) - closed_orbit_nonlinear(lattice, delta=-step)
    ) / (2.0 * step)
    measured = (
        closed_spin_solution(lattice, delta=+step).n0
        - closed_spin_solution(lattice, delta=-step).n0
    ) / (2.0 * step)

    predicted = coupling.matrix[:, [X, PX, Y, PY]] @ dispersion + coupling.dn_ddelta
    assert np.linalg.norm(measured - predicted) < 5e-8 * np.linalg.norm(measured)

    # ... and it is *not* the same object as dn/ddelta at fixed transverse coordinates.
    assert np.linalg.norm(measured) > 1.5 * np.linalg.norm(coupling.dn_ddelta)


def test_the_field_is_perpendicular_to_n0_and_exactly_blind_to_zeta():
    r"""``n_0 . N = 0`` by construction, and ``N[:, ZETA] == 0`` because nothing reads ``zeta``.

    Two structural facts, both load-bearing rather than decorative.

    ``n`` is a *unit* vector, so its component along ``n_0`` is not small but meaningless,
    and :func:`accsim.spin.spin_orbit_coupling` solves in the perpendicular plane rather
    than solving in three dimensions and hoping. The consistency condition that makes that
    legitimate -- ``n_0 . D = 0``, i.e. perturbing the orbit can only move the returned
    spin *sideways* -- is not exact in floating point and is asserted at the differencing
    accuracy it actually reaches.

    The ``zeta`` column is **exactly** zero, bit for bit, because no element in this
    lattice reads ``zeta`` at all. That is what makes this package's six-column Sylvester
    equation and ``xtrack``'s five-column (``zeta``-deleted) formulation the same object,
    which is the only reason the reference comparison is meaningful. **It stops being
    true the moment an RF cavity enters**, and then the two formulations part company --
    stated here because the failure would look like a coefficient disagreement.
    """
    coupling = spin_orbit_coupling(ring())

    assert np.abs(coupling.n0 @ coupling.matrix).max() < 1e-15
    assert np.abs(coupling.n0 @ coupling.spin_response).max() < 1e-9  # the precondition
    assert list(coupling.matrix[:, ZETA]) == [0.0, 0.0, 0.0]  # exactly, not nearly


def test_the_coupling_is_flat_in_the_differencing_step():
    r"""``N`` does not depend on ``step`` between ``1e-7`` and ``1e-5``.

    The only free numerical parameter this milestone introduces, and the one place a
    plausible-looking number could be quietly wrong. ``R`` and ``D`` are central
    differences, so the truncation falls as ``step^2`` and the round-off grows as
    ``eps/step``; the two cross near ``1e-5``, and the default ``1e-6`` sits in the flat
    bottom. Gated as a *plateau* -- three decades apart agreeing to ``1e-6`` relative --
    rather than by asserting the default is best, because a plateau is what says the
    answer is the derivative and not the step.
    """
    reference = spin_orbit_coupling(ring(), step=1e-6).matrix
    for step in (1e-7, 1e-5):
        other = spin_orbit_coupling(ring(), step=step).matrix
        assert np.abs(other - reference).max() < 1e-6 * np.abs(reference).max()


def test_the_propagated_field_closes_on_itself():
    r"""``N(s)`` round the ring: it starts where the solve put it and comes back to it.

    :func:`accsim.spin.propagate_spin_orbit_coupling` launches the differencing bundle
    *on* the field and reads ``N(s)`` off its central differences at every element
    boundary. Two things follow that nothing else checks:

    - the first matrix is the solve's own, which says the launch used the field it was
      given (it agrees to ``1e-10``, the cancellation floor of adding ``step * N`` to a
      unit vector -- not to round-off, and that is worth knowing);
    - the last equals the first, which is what "invariant" *means* and is the only
      end-to-end statement that the Sylvester solution is periodic at all. A sign error
      in ``A`` or ``D`` gives a field that drifts every turn and fails here.
    """
    lattice = ring()
    coupling = spin_orbit_coupling(lattice)
    field = propagate_spin_orbit_coupling(lattice)

    assert len(field) == len(lattice.elements) + 1
    assert np.abs(field[0] - coupling.matrix).max() < 1e-9
    assert np.abs(field[-1] - field[0]).max() < 1e-7 * np.abs(field[0]).max()


# --- the resonance: a location, with no coefficient in it -----------------------------

#: Tune distances from ``nu_0 = k + Q_y`` the scan is run at. Kept no closer than
#: ``1e-3``: the Sylvester solve amplifies ``D``'s differencing noise by
#: ``1/|2 sin(pi (nu_0 - Q_y))|``, so a residue fitted from ``1e-5`` would be fitting the
#: noise. The *collapse* gates below go closer, because there the quantity of interest is
#: large rather than delicate.
SCAN = (3e-2, 1e-2, 3e-3, 1e-3, -1e-3, -3e-3, -1e-2)


def scan_vertical_response() -> list[tuple[float, float, float, float]]:
    """``(distance, nu_0, |N E_y|, |N E_x|)`` across :data:`SCAN`."""
    rows = []
    for distance in SCAN:
        coupling = spin_orbit_coupling(ring_at_tune_distance(distance))
        modes = orbital_modes(coupling.orbit_matrix)
        rows.append(
            (
                distance,
                coupling.spin_tune,
                float(np.linalg.norm(coupling.matrix @ modes["y"][1])),
                float(np.linalg.norm(coupling.matrix @ modes["x"][1])),
            )
        )
    return rows


def test_the_orbital_tunes_do_not_move_across_the_scan():
    r"""Scanning the energy moves ``nu_0`` and **nothing** optical -- to twelve digits.

    The precondition for every location claim below. ``nu_0 = G gamma`` scales with the
    beam energy; the lattice is specified by normalised strengths (``k0``, ``k1``), so its
    transfer matrices do not depend on the energy at all and ``Q_x``, ``Q_y`` are frozen.
    If they moved, "the pole is at ``Q_y``" would be a statement about a moving target and
    the extrapolation below would mean nothing.

    Asserted at ``1e-12`` rather than at a physics tolerance because the expected residual
    is round-off in the eigenvalue solve, not physics.
    """
    tunes = [
        orbital_modes(spin_orbit_coupling(ring_at_tune_distance(d)).orbit_matrix) for d in SCAN
    ]
    for plane in ("x", "y"):
        values = [t[plane][0] for t in tunes]
        assert max(values) - min(values) < 1e-12


def test_the_pole_extrapolates_to_the_vertical_tune_and_not_to_an_integer():
    r"""The milestone's location gate: ``1/|N E_y|`` crosses zero at ``Q_y``, not at ``k``.

    Near the pole ``|N E_y| ~ C / |nu_0 - Q_y|``, so its **reciprocal** is linear in
    ``nu_0`` and its zero is the resonance's position. Two points at ``+1e-3`` and
    ``+3e-3`` -- close enough to be in the linear regime, far enough that the solve is not
    amplifying noise -- extrapolate to within ``2e-6`` of ``Q_y`` in tune.

    That is a *location*, with no coefficient in it: ``C`` cancels out of the
    extrapolation exactly, so a version of this package whose spin-orbit coupling were
    uniformly ten times too strong would land in the same place. It is the same shape as
    N2's gate (an integer-indexed position) shifted by ``Q_y``, which is the whole
    difference between the imperfection resonance and the intrinsic one.

    The discrimination is asserted as well as the agreement: the extrapolated position is
    a quarter of a unit from the nearest integer, so the two hypotheses are apart by a
    factor of ``10^5`` rather than by a tolerance.
    """
    qy = vertical_tune()
    points = []
    for distance in (1e-3, 3e-3):
        coupling = spin_orbit_coupling(ring_at_tune_distance(distance))
        modes = orbital_modes(coupling.orbit_matrix)
        points.append(
            (coupling.spin_tune, 1.0 / float(np.linalg.norm(coupling.matrix @ modes["y"][1])))
        )

    (nu_1, f_1), (nu_2, f_2) = points
    crossing = nu_1 - f_1 * (nu_2 - nu_1) / (f_2 - f_1)

    assert abs(crossing - qy) < 2e-6
    assert abs(crossing - round(crossing)) > 0.25  # nowhere near an integer resonance


def test_the_residue_identifies_the_denominator_as_the_intrinsic_one():
    r"""``|N E_y| * 2 |sin(pi (nu_0 - Q_y))|`` is constant; the other two candidates are not.

    The location gate above says *where* the pole is. This says what the denominator's
    **shape** is, over a range in which ``|N E_y|`` itself changes thirtyfold: the
    Sylvester solve's resonant factor is ``|lambda_y - exp(2 pi i nu_0)|``, which is
    ``2 |sin(pi (nu_0 - Q_y))|`` and nothing else. Multiplied back out, what is left is
    the residue -- a property of the lattice's vertical spin-orbit coupling strength --
    and it is flat to 1.5% across the scan.

    Both alternatives are asserted *excluded* rather than left unmentioned, which is the
    part that makes this a gate:

    - ``2 |sin(pi nu_0)|``, the integer/imperfection denominator, varies by a factor of
      27 over the same points;
    - ``2 |sin(pi (nu_0 + Q_y))|``, the sum resonance, by a factor of 31.

    Neither is nearly constant, and no tolerance-loosening could confuse them with the
    right answer.
    """
    rows = scan_vertical_response()
    qy = vertical_tune()

    def spread(denominator) -> float:
        values = [response * denominator(nu) for _, nu, response, _ in rows]
        return max(values) / min(values)

    def two_sin(phase: float) -> float:
        return 2.0 * abs(math.sin(math.pi * phase))

    assert spread(lambda nu: two_sin(nu - qy)) < 1.02
    assert spread(lambda nu: two_sin(nu)) > 20.0
    assert spread(lambda nu: two_sin(nu + qy)) > 20.0


def test_only_the_vertical_mode_diverges():
    r"""``|N E_y|`` blows up thirtyfold; ``|N E_x|`` does not move at all.

    A resonance is a statement about *one* mode, and a bug that simply made the whole
    coupling matrix large near a particular energy would look identical on ``|N E_y|``
    alone. The horizontal response is flat to 20% across the same scan while the vertical
    one runs over a factor of 30, so the divergence is attributable to the mode whose tune
    the resonance condition names.
    """
    rows = scan_vertical_response()
    vertical = [row[2] for row in rows]
    horizontal = [row[3] for row in rows]

    assert max(vertical) / min(vertical) > 25.0
    assert max(horizontal) / min(horizontal) < 1.2


def test_n0_and_the_spin_tune_do_not_notice_the_intrinsic_resonance():
    r"""N2's two objects sit through the resonance untouched -- which is why this is N4.

    The roadmap put ``nu_0 = k +- Q_y`` in N2 and N2 found it was not there: ``n_0``
    lives on the closed orbit, which is one-turn periodic, so it can only resonate with
    one-turn-periodic drive -- integer harmonics. The intrinsic resonance is a property of
    the field *around* ``n_0``, and this asserts the separation directly rather than
    arguing it.

    **The claim is "no feature", not "no change".** ``n_0``'s tilt is *not* constant across
    the scan -- it carries N2's ``cot(pi nu_0)``, and the scan moves ``nu_0``, so it drifts
    by 10% from one end to the other. What it does not do is diverge: it is monotone in
    ``nu_0``, its two innermost points (``+-1e-3``, straddling the pole) agree with each
    other to 1%, and that whole 10% drift is dwarfed by the factor of thirty ``|N E_y||``
    covers over the same points. A quantity with a pole in the middle of the scan cannot be
    monotone through it, so monotonicity is the sharp form of the statement.
    """
    tilts = []
    for distance in SCAN:
        coupling = spin_orbit_coupling(ring_at_tune_distance(distance))
        tilts.append(float(math.hypot(coupling.n0[0], coupling.n0[2])))
        assert coupling.spin_tune == pytest.approx(vertical_tune() + distance, abs=1e-9)

    assert tilts == sorted(tilts)  # smooth in nu_0, with nothing happening at the pole
    assert max(tilts) / min(tilts) < 1.15
    inner, outer = tilts[SCAN.index(1e-3)], tilts[SCAN.index(-1e-3)]
    assert outer / inner - 1.0 < 1e-2


def test_a_ring_exactly_on_the_resonance_refuses():
    r"""``nu_0 = k + Q_y`` to machine precision raises :class:`SpinResonanceError`.

    The invariant spin field does not exist there, and a solve run anyway returns a large
    number rather than failing, which is the worst of the three possible behaviours. The
    error subclasses N2's :class:`accsim.spin.SpinSolutionError` because it is the same
    statement -- this ring has no periodic spin object of the kind you asked for -- but it
    is a *different* degeneracy: ``n_0`` is still perfectly well defined here, and the
    test asserts that too, since a caller who catches this must not conclude the ring has
    no ``n_0``.
    """
    lattice = ring_at_tune_distance(0.0)
    assert closed_spin_solution(lattice).n0[1] > 0.99  # n_0 is fine; the field is not

    with pytest.raises(SpinResonanceError, match="k \\+- Q"):
        spin_orbit_coupling(lattice)


# --- the depolarization: the integrals, the degeneracy, and the collapse --------------


def test_the_flat_ring_depolarizes_exactly_not_at_all():
    r"""``dn/ddelta = 0`` bit for bit on a flat ring, so ``P_eq == P_inf`` bit for bit.

    This axis's degeneracy for the fourth time, and the sharpest instance of it yet. On a
    flat, unsteered ring the closed orbit never has a vertical excursion, so no element
    ever produces a horizontal field on it, so every rotation a spin meets is about ``y``
    -- and a ``delta`` perturbation only changes *how fast* a spin turns about the axis it
    already lies along, which does nothing whatever to ``n_0 = y``.

    The consequence is not "small": both depolarization integrals are ``0.0`` exactly, the
    Derbenev-Kondratenko polarization equals Sokolov-Ternov's to the last bit, and the two
    time constants are the same float. Asserted with ``==`` rather than ``approx``,
    because anything else would be a bug, and because a milestone whose gate ring were
    accidentally flat would otherwise pass everything below by returning N3's answers.
    """
    flat = gate.flat_ring(ref=gate.electron(ENERGY))
    integrals = depolarization_integrals(flat)

    assert integrals.depolarization_plus == 0.0
    assert integrals.depolarization_minus == 0.0
    assert derbenev_kondratenko_polarization(flat) == sokolov_ternov_polarization(flat)
    assert polarization_time(flat) == polarization_buildup_time(flat)


def test_the_closed_orbit_pair_is_bit_identical_to_n3s():
    r"""The two routes' ``alpha_plus_co`` / ``alpha_minus_co`` are the *same floats*.

    N4 walks the ring with a thirteen-column differencing bundle where N3 walks it with
    one particle, and both accumulate the closed-orbit rate integrals on the way past.
    Since they share :func:`accsim.radiation._quadrature_nodes` and N3's own integrand --
    arithmetic order included -- the two must agree exactly, not nearly. Asserted with
    ``==`` so that any future change to either quadrature is caught here rather than
    surfacing as a slow drift between two numbers that are supposed to be one number.
    """
    lattice = ring()
    theirs = polarization_integrals(lattice)
    ours = depolarization_integrals(lattice)

    assert ours.alpha_plus_co == theirs.alpha_plus
    assert ours.alpha_minus_co == theirs.alpha_minus


def test_depolarization_can_only_make_things_worse():
    r"""It adds a square to the rate, so the polarization falls and the time shortens.

    Two signs, neither of which is a convention this package may choose.
    ``(11/18) <kappa^3 |dn/ddelta|^2>`` is an average of a square: it is non-negative for
    every ring that exists, it *adds* to ``alpha_plus``, and since ``alpha_plus`` is the
    denominator of the polarization and the numerator of the rate, the beam ends up **less
    polarized** and gets there **faster**.

    That last pairing is the thing most easily got backwards -- a fast-polarizing ring is
    not a well-polarized one -- so both halves are asserted, and at a size (``2e-4`` of the
    rate on this ring) rather than merely in direction, so a term that silently vanished
    would not pass.
    """
    lattice = ring()
    integrals = depolarization_integrals(lattice)

    assert integrals.depolarization_plus > 0.0
    assert integrals.depolarization_plus / integrals.alpha_plus_co > 1e-4

    assert abs(derbenev_kondratenko_polarization(lattice)) < abs(
        sokolov_ternov_polarization(lattice)
    )
    assert polarization_time(lattice) < polarization_buildup_time(lattice)


def test_the_quadrature_is_already_converged_because_the_integrand_is_smooth():
    r"""Eight sub-slices per dipole is enough here, and the reason is worth recording.

    N3 needed **Simpson's rule** and 64 sub-slices because its integrand carries
    ``(n_0 . v)^2``, and ``n_0``'s horizontal part turns through ``G gamma theta = 4.4``
    radians across one of this ring's bends -- the quadrature has to resolve the *spin*
    phase, not the optics.

    The dominant new integrand does not oscillate at all: ``|dn/ddelta|^2`` is the squared
    **modulus** of a vector that rotates about ``n_0``, and a modulus is blind to the
    rotation. It is converged to twelve digits at eight sub-slices, where N3's own term is
    still moving. The subtracted ``dn/ddelta . b`` term *does* oscillate, and it is the one
    quantity in this milestone that is quadrature-limited rather than converged -- but it is
    a hundred times smaller than the term above it, so it cannot set the requirement
    either. Both facts are asserted, because "it is small" is the reason the coarse
    quadrature is allowed, and it would stop being true on a ring with a larger tilt.

    Recorded as a gate rather than a remark because it is what justifies the reference
    file's element-granularity comparison: if this needed thousands of slices, no
    like-for-like comparison with ``xtrack``'s per-element sum would be possible at all.
    """
    lattice = ring_at_tune_distance(1e-3)
    coarse = depolarization_integrals(lattice, 8)
    fine = depolarization_integrals(lattice, 128)

    assert coarse.depolarization_plus == pytest.approx(fine.depolarization_plus, rel=1e-10)
    assert abs(coarse.depolarization_minus) < 2e-2 * coarse.depolarization_plus


def test_the_depolarization_grows_as_the_inverse_square_of_the_tune_distance():
    r"""``<kappa^3 |dn/ddelta|^2> ~ 1/distance^2`` -- gated on the **order**, not a value.

    ``dn/ddelta`` inherits the resonant denominator ``1/|nu_0 - Q_y|`` through the
    vertical mode's share of the ``delta`` direction, and the integral squares it. So
    halving the distance to the resonance should quadruple the depolarization, and a
    tenfold approach should raise it a hundredfold.

    Gated as J2 gated the octupole detuning: on the **power**, fitted across a decade,
    rather than on any single number. Every coefficient in the chain -- ``11/18``, the
    coupling strength, the bend's ``kappa^3`` -- is constant against a change of distance
    and drops out of the fitted slope exactly. What the slope *can* catch is the thing
    that matters: a ``dn/ddelta`` that inherited the wrong denominator, or picked up its
    resonance from the wrong mode, has a different power or none at all.

    **Measured close in, and that is not a convenience.** ``dn/ddelta`` is a resonant term
    *plus* a non-resonant background -- the horizontal and integer contributions, which do
    not care where ``Q_y`` is -- so the square of the sum only reaches a clean ``1/d^2``
    once the pole dominates. At ``d = 1e-3`` the background is still worth 32% and the
    fitted power comes out ``-1.89``; by ``1e-4`` it is worth 3%. Fitting at the
    comfortable distance and reporting ``-1.89`` as agreement would have been the easy
    mistake here, so the scan is taken to ``1e-5`` and the *residue* ``d^2 * integral`` is
    asserted flat as well -- the stronger statement, since a pure power law has a constant
    residue and a power law contaminated by anything else does not.
    """
    distances = (1e-4, 3e-5, 1e-5)
    values = [
        depolarization_integrals(ring_at_tune_distance(d), 16).depolarization_plus
        for d in distances
    ]
    slope, _ = np.polyfit(np.log(distances), np.log(values), 1)
    residues = [value * d * d for value, d in zip(values, distances, strict=True)]

    assert slope == pytest.approx(-2.0, abs=0.03)
    assert max(residues) / min(residues) < 1.03


def test_the_polarization_collapses_at_the_resonance_while_p_inf_does_not():
    r"""The headline: ``-0.9238 -> -0.15`` over ``3e-5`` in tune, with ``P_inf`` unmoved.

    Everything above is machinery; this is what the machinery is for. Approaching
    ``nu_0 = k + Q_y``, the invariant spin field's ``delta`` derivative diverges, the
    depolarization term overwhelms the Sokolov-Ternov rate, and the equilibrium
    polarization goes to zero -- while N3's ``P_inf``, which is built from the closed orbit
    alone and cannot see the spin field at all, returns ``-0.92376022...`` at every energy
    in the scan. It is not *quite* frozen -- it drifts by ``3e-9`` across the scan, which is
    N3's ``(n_0 . v)^2`` correction following the energy as ``1/(G gamma)`` -- and the
    assertion is set at that measured size rather than at a round number, so a P_inf that
    started responding to the resonance would be caught.

    That contrast *is* the milestone. It is also why ``nu_0 = G gamma`` makes the beam
    energy of a polarized ring a quantity you steer rather than merely read: the machine
    spends its life avoiding these lines, and the same divergence that destroys the
    polarization is what lets a ring measure its own energy to a part in ``10^4``.

    The assertions are on the *shape* -- monotone collapse, a factor of six between the
    ends, ``P_inf`` frozen -- not on any particular value, so no coefficient in the chain
    can make them pass or fail.
    """
    scan = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    equilibrium, sokolov = [], []
    for distance in scan:
        lattice = ring_at_tune_distance(distance)
        equilibrium.append(derbenev_kondratenko_polarization(lattice))
        sokolov.append(sokolov_ternov_polarization(lattice))

    # The polarization is negative -- the beam polarizes antiparallel to the guide field
    # (N3) -- so "collapsing towards zero" is a monotone *rise*. Getting that backwards is
    # the one way this assertion could be written so as to pass on a ring that never moved.
    assert equilibrium == sorted(equilibrium)
    assert abs(equilibrium[0]) > 0.9
    assert abs(equilibrium[-1]) < 0.05
    assert abs(equilibrium[0] / equilibrium[-1]) > 20.0

    # N3's number, meanwhile, has no idea any of this is happening.
    assert max(sokolov) - min(sokolov) < 1e-8  # N3's own 1/(G gamma) energy drift, and nothing else
    assert sokolov[-1] == pytest.approx(-P_ST, abs=1e-5)


def test_the_time_constant_shortens_as_the_polarization_dies():
    r"""Near the resonance the beam polarizes *faster* and *less*, and both by the same factor.

    ``alpha_plus`` is one number doing two jobs: it is the denominator of the equilibrium
    polarization and the numerator of the rate. So the depolarization term cannot shorten
    the time without lowering the polarization by exactly the same factor, and the ratio
    ``tau_with / tau_without`` must equal ``P_eq / P_inf`` up to the (much smaller)
    ``dn/ddelta . b`` correction in the numerator.

    Asserted as an identity to 1% rather than as two independent trends, because two
    trends agreeing in direction is a much weaker statement than one quantity appearing
    twice -- and because it is the cheapest available check that the same ``alpha_plus``
    really is being used in both places.
    """
    lattice = ring_at_tune_distance(1e-4)
    time_ratio = polarization_time(lattice) / polarization_buildup_time(lattice)
    polarization_ratio = derbenev_kondratenko_polarization(lattice) / sokolov_ternov_polarization(
        lattice
    )

    assert time_ratio < 0.7  # it really is much faster
    assert time_ratio == pytest.approx(polarization_ratio, rel=1e-2)


def test_the_derbenev_kondratenko_coefficient_is_invisible_to_this_whole_file():
    r"""``11/18`` cannot be reached from here, and the assertions say so out loud.

    N3 ended on this warning and N4 inherits a sharper version of it. Every gate above is
    one of three things: a *ratio* the coefficient cancels out of, a *location* it does not
    enter, or a *scaling order* it is constant against. Rescaling it would move no gate in
    this file past its tolerance except the ones that merely bound the term's size.

    Demonstrated rather than claimed: recomputing the polarization with the depolarization
    term multiplied by an arbitrary factor still collapses at the resonance, still stays
    below ``P_inf``, and still scales as ``1/distance^2``. Only ``xtrack``'s
    ``spin_polarization_eq`` discriminates the factor, and that lives behind the skippable
    ``reference`` marker -- so a green analytic suite is, once again, weaker evidence here
    than it looks.
    """
    integrals = depolarization_integrals(ring_at_tune_distance(1e-4))
    honest = 8.0 / (5.0 * math.sqrt(3.0)) * integrals.alpha_minus / integrals.alpha_plus

    for factor in (0.5, 2.0):
        rescaled = (
            8.0
            / (5.0 * math.sqrt(3.0))
            * integrals.alpha_minus
            / (integrals.alpha_plus_co + factor * integrals.depolarization_plus)
        )
        assert abs(rescaled) < P_ST  # still a collapse, still the right direction
        assert abs(rescaled - honest) > 0.1  # and the gates above cannot tell them apart


def test_a_lattice_that_does_not_bend_still_refuses():
    r"""The N3 refusal survives the depolarization route, and by the same construction.

    A lattice with no bending radiates nothing, so ``alpha_plus`` is exactly zero and
    there is no polarization to speak of. N3 found that this refusal is nearly unreachable
    -- a ring of drifts and on-axis quadrupoles never gets there, because nothing
    precesses and N2's :class:`accsim.spin.SpinSolutionError` fires first -- and that
    exactly one construction separates the two conditions: a **quadrupole traversed
    off-axis**, which has real field on the orbit but does not bend.

    That construction is reused here rather than rebuilt, and it exercises a path N3's
    could not: this route must assemble the whole spin-orbit coupling matrix *before* it
    discovers there is nothing to integrate.
    """
    from accsim.elements.corrector import Corrector
    from accsim.elements.drift import Drift
    from accsim.elements.quadrupole import Quadrupole

    steered = Lattice(
        [
            Corrector(kick_x=1e-3),
            Drift(1.0),
            Quadrupole(0.4, 1.2),
            Drift(1.0),
            Quadrupole(0.4, -1.2),
            Drift(1.0),
        ],
        gate.electron(ENERGY),
    )
    assert depolarization_integrals(steered).alpha_plus == 0.0
    with pytest.raises(ValueError, match="no bending"):
        derbenev_kondratenko_polarization(steered)
    with pytest.raises(ValueError, match="no bending"):
        polarization_time(steered)


def test_the_state_vector_dimension_is_what_the_bundle_assumes():
    """``DIM == 6``, and the differencing bundle is ``2 * DIM + 1`` columns wide.

    A one-line guard on the only assumption :func:`accsim.spin._bundle` makes about the
    coordinate system, kept because the bundle indexes columns arithmetically and a
    seventh coordinate would corrupt ``N`` silently rather than raising.
    """
    coupling = spin_orbit_coupling(ring())
    assert DIM == 6
    assert coupling.matrix.shape == (3, DIM)
    assert coupling.orbit_matrix.shape == (DIM, DIM)
    assert coupling.spin_response.shape == (3, DIM)
    assert coupling.dn_ddelta.shape == (3,)
    assert list(coupling.dn_ddelta) == list(coupling.matrix[:, DELTA])
