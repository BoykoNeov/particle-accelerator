r"""Cross-check the bunched-ring spin field (N5) against ``xtrack``'s 6D ``twiss``.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**This file's job changed halfway through writing it, and that is the milestone's result.**
It was written to confirm a prediction. N4 found the two codes' ``dn/ddelta`` differing by
``2e-6`` in absolute terms (``1e-5`` on Linux: it is round-off, and N4 gates the mechanism
rather than the number) where every other column agreed to ``1e-8`` relative, and
attributed the gap to xtrack's mode-by-mode ``inv(lambda I - A)`` on the ``delta`` mode,
whose orbital eigenvalue is exactly ``1``: entries of order ``1e11``, and ``1e11 x 1e-16``
of cancellation debris left in what survives. **With RF there is no eigenvalue ``1`` at
all**, so the prediction written into ``docs/ROADMAP.md`` before any of this was measured
was that xtrack's momentum column would come into line.

It does not. What happens instead is that a **new and much larger** disagreement appears,
one that is zero without RF and grows as ``Q_s^2`` -- reaching ``14%`` on this file's gate
ring. The prediction is therefore recorded as **refuted**
(:func:`test_the_predicted_agreement_does_not_happen_and_a_bigger_gap_appears`), which the
roadmap entry said in advance would be the larger finding.

**Whose gap it is, decided without either code's spin-field machinery.** The arbiter N4
used -- an identity that goes nowhere near the Sylvester solve -- does not exist on a
bunched ring. What replaces it is the *definition* of an invariant spin field: a particle
launched at ``x`` with spin ``n_0 + N x`` must still have spin ``n_0 + N x(turn)`` many
turns later. That is a first-order statement, so a matrix wrong by a fraction ``f`` leaves
a relative residual stuck at ``f`` while a right one leaves ``O(x)``. Run on **xtrack's
own tracking map**, xtrack's own field sits at ``3.56%`` at every amplitude and accsim's
falls with the amplitude. The same verdict comes out of accsim's map. There is no
configuration in which the reference's matrix is the invariant one.

**And the gap is downstream of everything both codes agree on.** Differencing xtrack's own
map gives a spin response ``D`` matching accsim's to ``1.6e-9`` and a one-turn Jacobian
``R`` matching to ``1.2e-10``; the mode-by-mode formula transcribed from xtrack's own
source reproduces accsim's Sylvester solve to ``7.6e-11``; and feeding **xtrack's** ``D``
and ``R`` through it returns accsim's matrix to ``1.0e-7`` -- three orders inside the
``14%`` at issue. So the
ingredients are right and the method is right, and the error enters somewhere after them,
in the stage where xtrack rescales its eigenvectors, tracks them as finite-amplitude
particles and reassembles ``NN = EE_spin @ inv(EE_orb)`` -- *which* of those steps, this
file does not determine. One tempting explanation is excluded by the
data already here: cancellation in reading tiny tracked deviations against a finite closed
orbit would grow as the resonance is approached, and the discrepancy is **flat** in the
tune distance (``1.1435`` at ``1e-2``, ``1.1434`` at ``1e-3``). The mechanism is not
claimed beyond that.

**The sixth silent switch on this axis, and the first that is a documented convention
rather than a default.** xtrack's RF kernel takes ``q = fabs(q0) * charge_ratio``
(``track_rf.h``) -- the **absolute** charge -- while accsim's :class:`RFCavity` multiplies
by the signed ``ref.charge``. For an electron the two cavities are exact negatives of each
other, so the correspondence is ``phase = phi_s + pi``, not ``phase = phi_s``. With
the naive mapping the xtrack line is longitudinally **unstable** (eigenvalues ``1.373`` and
``0.728`` off the unit circle) and its 6D ``twiss`` dies inside the normal form with
``Invalid n3`` -- loudly, which is the only reason this one did not become a quiet wrong
number. ``src/accsim/elements/rfcavity.py`` used to claim the phase conventions simply
matched; it now states the charge-sign caveat, and
:func:`test_the_cavity_correspondence_needs_half_a_turn_of_phase_for_an_electron` is why.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from accsim.coords import DELTA, PY, ZETA, X, Y
from accsim.elements.rfcavity import RFCavity
from accsim.orbit import closed_orbit_delta
from accsim.radiation import derbenev_kondratenko_polarization
from accsim.spin import _closed_state, spin_orbit_coupling
from accsim.tracking import Tracker

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

# tests/ dirs are not import packages, so N5's analytic rings and N3's xtrack twin are
# reached by path. Both are imported rather than rebuilt: the two codes must be given the
# *same* lattice, and N3's ``_build`` carries five hard-won ``xt.Particles`` settings.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "analytic"))

import test_depolarization as n4  # noqa: E402
import test_depolarization_xtrack as n4_xt  # noqa: E402
import test_polarization_xtrack as n3  # noqa: E402
import test_spin_sidebands as gate  # noqa: E402

#: How far in tune this file's ring sits from ``nu_0 = k + Q_s``.
DISTANCE = 1e-3

_ELEMENT = n3._xtrack_element


def _element_with_cavity(elem):
    """N3's element correspondence, plus the cavity this milestone needs.

    ``phase`` carries the extra ``pi`` the module docstring explains: xtrack's RF kernel
    uses ``fabs(q0)``, so an electron's cavity is the negative of accsim's at the same
    phase. ``phase`` (radians) rather than the deprecated ``lag`` (degrees), which is also
    what ``test_synchrotron_tune_xtrack.py`` uses.
    """
    if isinstance(elem, RFCavity):
        return xt.Cavity(
            voltage=elem.voltage,
            frequency=elem.frequency,
            phase=math.pi + elem.phi_s,
        )
    return _ELEMENT(elem)


n3._xtrack_element = _element_with_cavity


def _tunes(matrix: np.ndarray) -> list[float]:
    """The three positive eigen-tunes of a 6D one-turn matrix, sorted."""
    angles = np.angle(np.linalg.eigvals(matrix)) / (2.0 * math.pi)
    return sorted(float(a) for a in angles if a > 1e-9)


@pytest.fixture(scope="module")
def ring():
    """The sideband ring and its xtrack twin, twissed in 6D with spin."""
    lattice = gate._sideband_ring(DISTANCE)
    line = n3._build(lattice)
    return lattice, line, line.twiss(spin=True, polarization_analysis=True)


# --- the correspondence itself -------------------------------------------------------


def test_the_cavity_correspondence_needs_half_a_turn_of_phase_for_an_electron(ring):
    r"""``phase = phi_s`` makes the reference ring longitudinally unstable; ``+pi`` fixes it.

    The switch is in ``xtrack/beam_elements/elements_src/track_rf.h``:

        ``q = fabs(LocalParticle_get_q0(part)) * charge_ratio``

    accsim's :meth:`RFCavity.energy_kick_delta` uses ``ref.charge`` **signed**, because an
    RF cavity accelerates a physical charge in a physical field and a negative particle at
    a given phase gains the opposite energy. xtrack's choice makes ``lag`` mean the same
    thing for both species, which is defensible and is *not* what accsim's docstring used
    to say it did.

    Measured both ways on one line, with the cavity's ``phase`` flipped in place so that
    nothing else can differ: at ``phase = 0`` two eigenvalues leave the unit circle
    (``1.373`` and its reciprocal ``0.728``) and there is no synchrotron tune at all; at
    ``phase = pi`` all six sit on it and the three tunes reproduce accsim's to nine
    digits.
    """
    lattice, line, twiss = ring
    cavity = line.element_names[-1]
    modes = gate.orbital_modes(spin_orbit_coupling(lattice).orbit_matrix)
    ours = sorted(q for q, _ in modes.values())

    matched = line.get_R_matrix(particle_on_co=twiss.particle_on_co)["R_matrix"]
    assert np.allclose(np.abs(np.linalg.eigvals(matched)), 1.0, atol=1e-7)
    assert _tunes(matched) == pytest.approx(ours, abs=1e-9)

    line[cavity].phase = 0.0
    try:
        naive = np.abs(
            np.linalg.eigvals(line.get_R_matrix(particle_on_co=twiss.particle_on_co)["R_matrix"])
        )
        assert naive.max() > 1.3
        assert naive.min() < 0.8
    finally:
        line[cavity].phase = math.pi


def test_the_closed_orbit_momentum_is_confirmed_by_the_reference(ring):
    r"""``delta_co`` against xtrack's 6D closed orbit, and ``zeta_co`` against zero.

    The new implementation, checked by a code that solves the whole 6D fixed point rather
    than the one scalar accsim solves for. They agree to **seven digits**
    (``-4.7788823e-8`` against ``-4.7788829e-8``), and xtrack's ``zeta`` on the closed orbit
    is ``3e-13`` -- confirming the argument that the synchronous particle sits at the RF
    zero crossing, which is what let this milestone get away with a scalar.
    """
    lattice, _, twiss = ring
    assert float(twiss.delta[0]) == pytest.approx(closed_orbit_delta(lattice), rel=1e-6)
    assert abs(float(twiss.zeta[0])) < 1e-11


def test_the_tunes_and_the_closed_spin_solution_agree_to_nine_digits(ring):
    """Everything except the spin field matches, which is what makes the gap attributable.

    Three orbital tunes, the fractional spin tune and ``n_0`` itself. ``Q_y`` is folded:
    xtrack's normal form reports this ring's vertical mode as ``0.7214``, the conjugate of
    accsim's ``0.2786``, which is a choice of which member of the pair to call the mode and
    not a disagreement. If any of these moved
    with the ``14%`` below, the disagreement could be blamed on the lattice or on the map;
    none of them does, so it belongs to the object the milestone is about.
    """
    lattice, _, twiss = ring
    coupling = spin_orbit_coupling(lattice)
    modes = gate.orbital_modes(coupling.orbit_matrix)

    assert float(twiss.qs) == pytest.approx(modes["s"][0], abs=1e-9)
    their_qy = float(twiss.qy) % 1.0
    assert min(their_qy, 1.0 - their_qy) == pytest.approx(modes["y"][0], abs=1e-9)
    folded = min(coupling.spin_tune % 1.0, 1.0 - coupling.spin_tune % 1.0)
    assert float(twiss.spin_tune_fractional) == pytest.approx(folded, abs=1e-9)

    theirs = np.array([twiss.spin_x[0], twiss.spin_y[0], twiss.spin_z[0]])
    assert np.abs(theirs - coupling.n0).max() < 1e-9


def test_the_vertical_columns_of_the_spin_field_still_agree(ring):
    """``N``'s vertical columns match to ``1e-5`` relative while the rest do not.

    The vertical mode is the one the sideband does not reach (it has no dispersion here to
    couple through), so its columns carry no resonant part -- and there the two codes agree
    as they did throughout N4. That is what turns the disagreement below from "the two
    codes differ" into "the two codes differ *in the resonant direction*".
    """
    _, _, twiss = ring
    lattice = ring[0]
    ours = spin_orbit_coupling(lattice).matrix
    theirs = np.array(twiss.spin_n_matrix)[0]

    for column in (Y, PY):
        gap = np.abs(ours[:, column] - theirs[:, column]).max()
        assert gap < 1e-5 * np.abs(ours[:, column]).max()


# --- the prediction, and its refutation ----------------------------------------------


def test_the_predicted_agreement_does_not_happen_and_a_bigger_gap_appears(ring):
    r"""The pre-committed claim, measured: **refuted**, and the replacement is quantified.

    Written into ``docs/ROADMAP.md`` before measurement: with RF there is no eigenvalue
    ``1``, so xtrack's ``inv(lambda I - A)`` is no longer singular and its momentum column
    should come into line with accsim's at the ``1e-8`` the betatron columns already reach.

    What is measured instead: on the sideband ring the momentum and longitudinal columns
    differ by a factor of ``1.143``, and the horizontal ones -- which carry the resonant
    term through the dispersion -- by ``1.14`` as well. N4's ``2e-6`` was never repaired
    because it was never the mechanism at work here; it is a *different*, larger gap that
    the cavity switches on.

    The scaling says which knob it hangs on, and the gate is the **order**: the two codes
    agree exactly without RF and their disagreement grows as ``Q_s^2``, which is the same
    law the physical RF correction itself obeys. So xtrack is not wrong about the ring -- it
    is wrong about the *part of the field the RF creates*, by roughly a third of it.
    """
    lattice, _, twiss = ring
    ours = spin_orbit_coupling(lattice).matrix
    theirs = np.array(twiss.spin_n_matrix)[0]

    ratio = np.linalg.norm(theirs, axis=0) / np.linalg.norm(ours, axis=0)
    assert ratio[DELTA] == pytest.approx(1.1434, abs=0.002)
    assert ratio[ZETA] == pytest.approx(1.1434, abs=0.002)
    assert ratio[X] > 1.10
    # the prediction, stated as the assertion it failed
    assert abs(ratio[DELTA] - 1.0) > 1e-8, "the pre-committed agreement would fail this"


def test_the_disagreement_is_switched_on_by_the_cavity_and_grows_as_the_synchrotron_tune():
    r"""Zero without RF, ``Q_s^2`` with it -- three voltages on N4's own ring.

    Deliberately run on **N4's** gate ring rather than this file's, so that the RF-off end
    of the scan is the exact comparison N4 published: there the two codes' momentum columns
    differ by N4's cancellation debris and nothing else -- ``1.8e-6`` absolute on
    Windows/clang-cl, ``1.1e-5`` on Linux/gcc, and on either box equal to xtrack's own
    dispersion-identity miss vector for vector, which is how N4's number is reproduced
    without asserting a compiler's round-off. Switching the cavity on at ``Q_s = 0.005``
    moves them apart by ``2.6e-4`` relative, at ``0.0158`` by ``2.6e-3``, at ``0.05`` by
    ``2.9e-2`` -- a factor of ten per factor of ``3.16`` in ``Q_s``, which is the square.

    This is the test that makes the disagreement a *statement about the RF* rather than
    about this milestone's particular ring, and it is why the file does not simply widen a
    tolerance around ``14%``.
    """
    from accsim.lattice import Lattice

    base = n4.ring_at_tune_distance(1e-3)
    unbunched = n3._build(base).twiss(method="4d", spin=True, polarization_analysis=True)
    off_ours = spin_orbit_coupling(base).matrix[:, DELTA]
    off_theirs_matrix = np.array(unbunched.spin_n_matrix)[0]
    off_theirs = off_theirs_matrix[:, DELTA]
    off_gap = off_ours - off_theirs
    assert np.linalg.norm(off_gap) < n4_xt.DEBRIS_ESTIMATE  # N4's floor, reproduced ...
    miss = n4_xt.dispersion_identity_miss(base, off_theirs_matrix)
    assert np.linalg.norm(off_gap + miss) < 1e-2 * np.linalg.norm(off_gap)  # ... as xtrack's

    tunes, gaps = [], []
    for voltage in (2.3e4, 2.3e5, 2.3e6):
        cavity = RFCavity.from_harmonic(voltage, gate.HARMONIC, base.length, base.ref, phi_s=0.0)
        lattice = Lattice([*base.elements, cavity], base.ref)
        twiss = n3._build(lattice).twiss(spin=True, polarization_analysis=True)
        ours = spin_orbit_coupling(lattice).matrix[:, DELTA]
        theirs = np.array(twiss.spin_n_matrix)[0][:, DELTA]
        tunes.append(float(twiss.qs))
        gaps.append(float(np.linalg.norm(theirs) / np.linalg.norm(ours) - 1.0))

    for (q1, g1), (q2, g2) in zip(
        zip(tunes, gaps, strict=True), zip(tunes[1:], gaps[1:], strict=True), strict=False
    ):
        assert abs(math.log(g2 / g1) / math.log(q2 / q1) - 2.0) < 0.15


# --- which code is right, decided outside both spin-field machineries ----------------


def test_the_invariant_field_is_accsims_on_xtracks_own_tracking_map(ring):
    r"""The arbitration, run in the reference's own tracker so it cannot be blamed on ours.

    An invariant spin field is *defined* by this: launch at ``x`` with spin ``n_0 + N x``,
    track, and the spin is still ``n_0 + N x(turn)`` at every later turn. It is a
    first-order statement, so the relative residual falls with the amplitude for a correct
    ``N`` and sits at a constant for one that is wrong by a fraction.

    Measured over three decades of amplitude, tracking with ``xtrack``: accsim's matrix
    gives ``5.6e-4, 5.8e-5, 2.4e-5`` (falling until it meets the round-off floor), xtrack's
    own gives ``3.56e-2`` three times over. Neither code's spin-field solver is involved --
    only its tracker, and the two trackers give the same verdict
    (``tests/analytic/test_spin_sidebands.py`` runs the same check in accsim's).
    """
    lattice, line, twiss = ring
    coupling = spin_orbit_coupling(lattice)
    theirs = np.array(twiss.spin_n_matrix)[0]

    n0 = np.array([twiss.spin_x[0], twiss.spin_y[0], twiss.spin_z[0]])
    closed = np.array(
        [twiss.x[0], twiss.px[0], twiss.y[0], twiss.py[0], twiss.zeta[0], twiss.delta[0]]
    )
    eigenvector = np.real(gate.orbital_modes(coupling.orbit_matrix)["s"][1])
    direction = eigenvector / np.linalg.norm(eigenvector)
    names = ("x", "px", "y", "py", "zeta", "delta")

    def residual(matrix: np.ndarray, amplitude: float, turns: int = 40) -> float:
        offset = amplitude * direction
        launch = closed + offset
        spin = n0 + matrix @ offset
        spin /= np.linalg.norm(spin)

        particle = line.build_particles(**{k: launch[i] for i, k in enumerate(names)})
        particle.spin_x, particle.spin_y, particle.spin_z = spin[0], spin[1], spin[2]
        monitor = xt.ParticlesMonitor(num_particles=1, start_at_turn=0, stop_at_turn=turns)
        line.track(particle, num_turns=turns, turn_by_turn_monitor=monitor)

        worst = 0.0
        for turn in range(turns):
            tracked = np.array([getattr(monitor, k)[0, turn] for k in names]) - closed
            expected = n0 + matrix @ tracked
            expected /= np.linalg.norm(expected)
            got = np.array(
                [monitor.spin_x[0, turn], monitor.spin_y[0, turn], monitor.spin_z[0, turn]]
            )
            worst = max(worst, float(np.linalg.norm(got - expected)))
        return worst / float(np.linalg.norm(coupling.matrix @ offset))

    amplitudes = (1e-4, 1e-5, 1e-6)
    mine = [residual(coupling.matrix, a) for a in amplitudes]
    reference = [residual(theirs, a) for a in amplitudes]

    assert mine[0] < 1e-3
    assert mine[1] < 0.3 * mine[0]  # falls with the amplitude, as a first-order field must
    assert all(r == pytest.approx(3.56e-2, rel=0.02) for r in reference)
    assert min(reference) > 50.0 * mine[0]


def test_the_two_codes_ingredients_agree_so_the_gap_is_downstream_of_them(ring):
    r"""``D``, ``R`` and the mode-by-mode formula all check out; the gap is after them.

    Three separate claims, and together they localise the disagreement without guessing at
    it:

    - Differencing xtrack's **own** map with a spin started along ``n_0`` gives the spin
      response ``D`` to ``1.6e-9`` of accsim's and the one-turn Jacobian ``R`` to
      ``1.2e-10``, both relative to their own largest entry. (``D``'s ``zeta`` column is
      **identically** zero in both -- asserted entry by entry, since a ``zeta``
      displacement reaches the spin only through the cavity, which sits at the *end* of the
      turn.)
    - The mode-by-mode construction transcribed from xtrack's own source --
      ``N E_i = inv(lambda_i I - A) D E_i``, assembled as ``N = [N E_i] inv(E)`` --
      reproduces accsim's Sylvester solve to ``7.6e-11``. The two formulations are the same
      equation, confirmed numerically rather than argued.
    - Feeding xtrack's ``D`` and ``R`` through that construction returns **accsim's**
      matrix to ``1.0e-7`` -- three orders inside the ``14%`` at issue.

    So the published ``spin_n_matrix`` is not what xtrack's own ingredients and own method
    produce. The error is somewhere in the stage between -- xtrack rescales its
    eigenvectors, tracks them at finite amplitude and reassembles
    ``NN = EE_spin @ inv(EE_orb)`` -- but *which* of those steps is not determined here, and
    the file does not claim it.
    """
    lattice, line, twiss = ring
    coupling = spin_orbit_coupling(lattice)
    closed = np.array(
        [twiss.x[0], twiss.px[0], twiss.y[0], twiss.py[0], twiss.zeta[0], twiss.delta[0]]
    )
    n0 = coupling.n0
    names = ("x", "px", "y", "py", "zeta", "delta")

    step = 1e-6
    response = np.zeros((3, 6))
    jacobian = np.zeros((6, 6))
    for j in range(6):
        ends = {}
        for sign in (+1, -1):
            launch = closed.copy()
            launch[j] += sign * step
            particle = line.build_particles(**{k: launch[i] for i, k in enumerate(names)})
            particle.spin_x, particle.spin_y, particle.spin_z = n0[0], n0[1], n0[2]
            line.track(particle, num_turns=1)
            ends[sign] = (
                np.array([getattr(particle, k)[0] for k in names]),
                np.array([particle.spin_x[0], particle.spin_y[0], particle.spin_z[0]]),
            )
        jacobian[:, j] = (ends[1][0] - ends[-1][0]) / (2.0 * step)
        response[:, j] = (ends[1][1] - ends[-1][1]) / (2.0 * step)

    scale = np.abs(coupling.spin_response).max()
    assert np.abs(response - coupling.spin_response).max() < 1e-8 * scale
    assert np.all(response[:, ZETA] == 0.0)
    assert np.all(coupling.spin_response[:, ZETA] == 0.0)
    assert (
        np.abs(jacobian - coupling.orbit_matrix).max() < 1e-9 * np.abs(coupling.orbit_matrix).max()
    )

    def mode_by_mode(orbit_matrix: np.ndarray, spin_response: np.ndarray) -> np.ndarray:
        """xtrack's own construction: one 3x3 inverse per orbital eigenvector."""
        values, vectors = np.linalg.eig(orbit_matrix)
        columns = np.empty((3, 6), dtype=complex)
        for i in range(6):
            shifted = values[i] * np.eye(3) - coupling.one_turn_matrix
            columns[:, i] = np.linalg.inv(shifted) @ (spin_response @ vectors[:, i])
        return np.real(columns @ np.linalg.inv(vectors))

    ours = mode_by_mode(coupling.orbit_matrix, coupling.spin_response)
    assert np.abs(ours - coupling.matrix).max() < 1e-9 * np.abs(coupling.matrix).max()

    theirs = mode_by_mode(jacobian, response)
    assert np.abs(theirs - coupling.matrix).max() < 1e-6 * np.abs(coupling.matrix).max()


def test_the_equilibrium_polarizations_differ_by_what_the_fields_do(ring):
    r"""``P_eq`` differs by the same amount ``dn/ddelta`` does -- so ``11/18`` stays N4's.

    N4's reference file could reach the Derbenev-Kondratenko coefficient because the two
    codes agreed on the field and differed only in the constant. Here they disagree about
    the field itself, so the polarization comparison inherits it: xtrack's ``-0.00569``
    against accsim's ``-0.00747``, which is the ``14%`` in ``dn/ddelta`` squared and folded
    through a ratio.

    This test exists to say that plainly rather than to check a physics constant. The
    ``11/18`` remains anchored where N4 anchored it -- on an unbunched ring, where the two
    codes' fields agree -- and this milestone does not re-derive it.
    """
    lattice, _, twiss = ring
    ours = derbenev_kondratenko_polarization(lattice)
    theirs = float(twiss.spin_polarization_eq)

    assert abs(theirs) < abs(ours)  # a larger dn/ddelta depolarizes harder
    assert 0.7 < abs(theirs / ours) < 0.85
    # ... and both are far below the Sokolov-Ternov value the same twiss reports
    assert abs(float(twiss.spin_polarization_inf_no_depol)) > 0.9


def test_the_spin_field_matches_accsims_own_tracker_too(ring):
    """The same invariance check in accsim's tracker, so the verdict is not one-sided.

    ``tests/analytic/test_spin_sidebands.py`` runs this on its own ring; here it runs on
    *this* file's ring with *xtrack's* matrix as the comparison, so that the two trackers
    are asked the identical question about the identical pair of matrices.
    """
    lattice, _, twiss = ring
    coupling = spin_orbit_coupling(lattice)
    theirs = np.array(twiss.spin_n_matrix)[0]
    closed = _closed_state(lattice, coupling.orbit, coupling.delta)
    eigenvector = np.real(gate.orbital_modes(coupling.orbit_matrix)["s"][1])
    direction = eigenvector / np.linalg.norm(eigenvector)

    def residual(matrix: np.ndarray, amplitude: float, turns: int = 40) -> float:
        offset = amplitude * direction
        state = (closed + offset)[:, None]
        spin = coupling.n0 + matrix @ offset
        spin = (spin / np.linalg.norm(spin))[:, None]
        tracker = Tracker(lattice)
        worst = 0.0
        for _ in range(turns):
            state, spin = tracker.track_once_with_spin(state, spin)
            expected = coupling.n0 + matrix @ (state[:, 0] - closed)
            expected /= np.linalg.norm(expected)
            worst = max(worst, float(np.linalg.norm(spin[:, 0] - expected)))
        return worst / float(np.linalg.norm(coupling.matrix @ offset))

    assert residual(coupling.matrix, 1e-4) < 1e-3
    assert residual(theirs, 1e-4) == pytest.approx(3.56e-2, rel=0.02)
    assert residual(theirs, 1e-6) == pytest.approx(3.56e-2, rel=0.02)
