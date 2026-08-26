r"""Cross-check the closed spin solution (N2) against ``xtrack``'s own.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

The ring is N2's analytic gate ring rebuilt element for element in xtrack -- a closed
vertical bump around one thick quadrupole, inside a bend-free straight, followed by a
thin-lens FODO arc whose bends sum to ``2 pi``. Rebuilding it rather than using something
simpler is the point: on a *flat, unsteered* ring ``n_0`` is exactly ``(0, 1, 0)`` in both
codes for reasons that have nothing to do with either code being right, so an agreement
there measures nothing at all. Only the tilt is a comparison.

**Four things have to be right before the comparison means anything.** Three are N1's, and
they are re-asserted here rather than assumed: ``line.configure_spin(...)``, an explicit
``anomalous_magnetic_moment`` (xtrack defaults it to ``0``, which is the cyclotron rotation
and not "spin off"), and an exact bend model. The fourth is new and is the one this
milestone can be destroyed by: **the two codes must agree about the orbit first**. With a
vertical closed orbit in play, a spin disagreement is an orbit disagreement until that has
been excluded -- which is N1's finding 2 in a ring instead of a magnet -- so the orbit is
compared, element by element, before a single spin component is looked at.

**The two codes build the one-turn spin rotation differently, and that sets the tolerance.**
accsim's is exact: the spin map is linear in the spin, so carrying the three Cartesian basis
vectors around once *is* the matrix. xtrack finite-differences it with ``ds = 1e-5``
(``twiss.py``, ``_get_spin_polarization``) and finds ``n_0`` with a two-knob optimiser to a
tolerance of ``1e-12``. The agreement is therefore expected near ``1e-9``, and which side
the error is on is known rather than guessed.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from accsim.orbit import closed_orbit_nonlinear, propagate_orbit_nonlinear
from accsim.reference import ELECTRON_ANOMALOUS_MOMENT as G
from accsim.reference import ELECTRON_MASS_EV as MASS0
from accsim.spin import closed_spin_solution, propagate_spin_solution

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

# tests/ dirs are not import packages, so the analytic gate ring is reached by path. It is
# imported rather than rebuilt on purpose: the two codes must be given the *same* ring, and
# a second copy of a bump solver is a second chance to get it wrong.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "analytic"))

import test_spin_solution as gate  # noqa: E402

ENERGY = 5e9
P0C = math.sqrt(ENERGY**2 - MASS0**2)
AMPLITUDE = 2e-3


def _accsim_ring():
    return gate.gate_ring(AMPLITUDE, gate.electron(ENERGY))


def _xtrack_element(elem):
    """The xtrack twin of one accsim element, using the correspondences the suite has pinned.

    ``Corrector(kick_y=+k) == xt.Multipole(ksl=[+k])`` and
    ``ThinQuadrupole(k1l) == xt.Multipole(knl=[0, k1l])`` are
    ``tests/reference/test_feeddown_xtrack.py``'s and ``test_misalignment_xtrack.py``'s,
    not this file's; the bend's ``model="bend-kick-bend"`` is N1's.
    """
    from accsim.elements.corrector import Corrector
    from accsim.elements.dipole import Dipole
    from accsim.elements.drift import Drift
    from accsim.elements.quadrupole import Quadrupole, ThinQuadrupole

    if isinstance(elem, Drift):
        # ``model="exact"`` is not cosmetic: xtrack's *default* drift is the paraxial
        # one (M2's finding), and a paraxial ring closes this bump exactly while an
        # exact one does not -- so with the default the two codes' orbits differ by
        # accsim's own bump leak and every spin comparison inherits it.
        return xt.Drift(length=elem.length, model="exact")
    if isinstance(elem, Corrector):
        return xt.Multipole(knl=[-elem.kick_x], ksl=[elem.kick_y], length=0.0)
    if isinstance(elem, ThinQuadrupole):
        return xt.Multipole(knl=[0.0, elem.k1l], length=0.0)
    if isinstance(elem, Quadrupole):
        return xt.Quadrupole(length=elem.length, k1=elem.k1)
    if isinstance(elem, Dipole):
        return xt.Bend(
            length=elem.length,
            angle=elem.angle,
            k0=elem.angle / elem.length,
            model="bend-kick-bend",
        )
    raise AssertionError(f"no xtrack twin wired up for {type(elem).__name__}")


@pytest.fixture(scope="module")
def line():
    lattice = _accsim_ring()
    elements = [_xtrack_element(e) for e in lattice.elements]
    built = xt.Line(elements=elements, element_names=[f"e{i}" for i in range(len(elements))])
    built.particle_ref = xt.Particles(mass0=MASS0, p0c=P0C, anomalous_magnetic_moment=G, spin_y=1.0)
    # Without this the kernel is compiled with spin off and track() is a no-op on it.
    built.configure_spin("auto")
    try:
        built.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    for name in built.element_names:
        element = built[name]
        if isinstance(element, xt.Bend | xt.Quadrupole):
            element.integrator = "uniform"
            element.num_multipole_kicks = 1
    return built


@pytest.fixture(scope="module")
def spin_twiss(line):
    return line.twiss(method="4d", spin=True, polarization_analysis=True)


# --- the orbit, before any spin is looked at ----------------------------------------


def test_the_two_codes_agree_about_the_closed_orbit_first(spin_twiss):
    """Element by element, ``(x, px, y, py)``, before a spin component is compared.

    N1's second finding was an alarming spin disagreement that turned out to be the
    *orbit*: xtrack's default bend splitting put the design orbit slightly off axis and the
    spin honestly followed it. That failure mode is far more available here, where the
    orbit is deliberately not zero, so it is excluded first rather than diagnosed later.
    """
    lattice = _accsim_ring()
    ours = np.array(propagate_orbit_nonlinear(lattice))[:-1]
    theirs = np.array(
        [spin_twiss.x[:-1], spin_twiss.px[:-1], spin_twiss.y[:-1], spin_twiss.py[:-1]]
    ).T

    assert theirs.shape == ours.shape
    assert np.abs(theirs[:, 2]).max() > 0.2 * AMPLITUDE  # the bump is really there
    assert np.abs(theirs - ours).max() < 1e-12


# --- the closed spin solution -------------------------------------------------------


def test_n0_agrees_with_xtrack_element_by_element(spin_twiss):
    r"""``n_0(s)`` from :func:`accsim.spin.propagate_spin_solution` against ``spin_x/y/z``.

    Both codes orient the solution upward -- xtrack's fixed-point search sets
    ``s_y = +sqrt(1 - s_x^2 - s_z^2)`` and so can only return an upward one, which is why
    accsim adopts the same rule -- so the vectors are directly comparable with no sign
    reconciliation.

    The claim that carries the test is the **tilt**, not the agreement: the two transverse
    components are ``1e-4``-sized and vary around the ring, and it is those that agree. A
    comparison on an unsteered ring would be two zeros matching two zeros.
    """
    lattice = _accsim_ring()
    ours = np.array(propagate_spin_solution(lattice))[:-1]
    theirs = np.array([spin_twiss.spin_x[:-1], spin_twiss.spin_y[:-1], spin_twiss.spin_z[:-1]]).T

    assert theirs.shape == ours.shape
    assert np.abs(ours[:, [0, 2]]).max() > 1e-5  # there is a tilt to compare at all
    assert np.ptp(np.abs(ours[:, [0, 2]])) > 1e-5  # and it moves around the ring
    assert np.abs(theirs - ours).max() < 1e-8


def test_the_spin_tune_agrees_with_the_polarization_analysis(spin_twiss):
    r"""``nu_0`` against ``spin_tune_fractional``, folded the way xtrack folds it.

    xtrack takes ``max(angle(eigvals(A)))/(2 pi)`` of its finite-differenced spin matrix,
    and ``angle`` returns ``(-pi, pi]``, so its answer is always in ``[0, 0.5]``: it is
    ``|nu_0|`` folded, with the sign and the half-turn thrown away. accsim keeps the whole
    fraction in ``[0, 1)`` (see :func:`accsim.spin.spin_axis_and_tune` for why that sign),
    so the comparison folds accsim's rather than unfolding xtrack's -- the information is
    not there to unfold.
    """
    lattice = _accsim_ring()
    ours = closed_spin_solution(lattice).spin_tune
    folded = min(ours, 1.0 - ours)

    assert spin_twiss.spin_tune_fractional == pytest.approx(folded, abs=1e-9)
    assert folded == pytest.approx((G * lattice.ref.gamma0) % 1.0, abs=1e-6)


def test_the_one_turn_rotation_is_the_matrix_xtrack_differences(line, spin_twiss):
    r"""accsim's exact 3x3 against xtrack's ``ds = 1e-5`` central difference of the same map.

    The asymmetry is the finding. accsim's matrix costs one turn and is exact, because a
    rotation is linear in the spin; xtrack rebuilds it from six tracked spins at
    ``+-1e-5`` about the closed orbit. The two therefore agree to about ``1e-9``, and the
    residual is xtrack's differencing, not accsim's map -- which is why the tolerance here
    is looser than N1's element-by-element round-off.
    """
    lattice = _accsim_ring()
    ours = closed_spin_solution(lattice, closed_orbit_nonlinear(lattice)).one_turn_matrix

    step = 1e-5
    orbit = spin_twiss.particle_on_co
    columns = []
    for axis in range(3):
        images = []
        for sign in (+1.0, -1.0):
            spin = [0.0, 0.0, 0.0]
            spin[axis] = sign * step
            particle = orbit.copy()
            particle.spin_x, particle.spin_y, particle.spin_z = spin
            line.track(particle)
            images.append(np.array([particle.spin_x[0], particle.spin_y[0], particle.spin_z[0]]))
        columns.append((images[0] - images[1]) / (2.0 * step))
    theirs = np.array(columns).T

    assert np.abs(theirs @ theirs.T - np.eye(3)).max() < 1e-8
    assert np.abs(theirs - ours).max() < 1e-8


def test_the_paraxial_drift_closes_the_bump_that_the_exact_one_does_not(spin_twiss):
    r"""M2's drift model decides whether this ring's vertical bump closes at all.

    The bump's corrector strengths are solved from the elements' **matrices**, so a ring of
    *paraxial* drifts closes it exactly and a ring of *exact* drifts does not -- the exact
    map departs from its own Jacobian at third order in the excursion, which the analytic
    suite measures independently as a leak of ``1.6e-9`` at this amplitude
    (``test_the_bump_closes_so_the_arc_stays_on_the_design_orbit``).

    So with xtrack's **default** drift the two codes' closed orbits disagree -- and the
    disagreement is not a spin question, not a tolerance, and not a bug in either code: it
    is that number, reproduced here to two digits from a completely separate line. M2's
    lesson, in the axis that came after it: localise before deriving.
    """
    lattice = _accsim_ring()
    ours = np.array(propagate_orbit_nonlinear(lattice))[:-1]

    paraxial = xt.Line(
        elements=[
            xt.Drift(length=e.length) if isinstance(t, xt.Drift) else t
            for e, t in zip(
                lattice.elements, (_xtrack_element(e) for e in lattice.elements), strict=True
            )
        ],
        element_names=[f"e{i}" for i in range(len(lattice.elements))],
    )
    paraxial.particle_ref = xt.Particles(mass0=MASS0, p0c=P0C, anomalous_magnetic_moment=G)
    try:
        paraxial.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    for name in paraxial.element_names:
        element = paraxial[name]
        if isinstance(element, xt.Bend | xt.Quadrupole):
            element.integrator = "uniform"
            element.num_multipole_kicks = 1
    tw = paraxial.twiss(method="4d")

    theirs = np.array([tw.x[:-1], tw.px[:-1], tw.y[:-1], tw.py[:-1]]).T
    gap = float(np.abs(theirs[:, 2] - ours[:, 2]).max())
    leak = float(np.abs(ours[gate.n_straight() :, 2]).max())

    assert abs(theirs[:, 2]).max() > 0.2 * AMPLITUDE  # it is the same bump, either way
    assert gap == pytest.approx(leak, rel=0.05)  # and the whole gap is the leak
    # while the exact-drift line, which does not close it either, agrees to a thousandth of it
    assert np.abs(np.array(spin_twiss.y[:-1]) - ours[:, 2]).max() < 1e-3 * gap
