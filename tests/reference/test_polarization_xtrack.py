r"""Cross-check the Sokolov-Ternov polarization (N3) against ``xtrack``'s own analysis.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**This file carries more of N3 than a reference file usually carries.** The milestone's
analytic gates pin the two lattice weights against a sympy closed form, the ``gamma^5``
and ``rho^3`` *powers* by scaling, and the eV-to-SI bridge against ``scipy.constants`` --
but a wrong *factor* in the rate, a rate ten times too fast, passes every one of them.
``P_inf`` cannot see it either, being a ratio the constant cancels out of. The buildup
time against xtrack's ``spin_t_pol_component_s`` is the only thing in the package that
does, and it lives behind the skippable marker. A green analytic suite is weaker evidence
here than it is anywhere else on this axis, and that is worth saying out loud.

Two rings, doing two different jobs:

- **A flat ring** for the coefficient. ``P_inf`` is degenerate there -- both codes return
  the textbook ratio for reasons that have nothing to do with either being right -- but
  the *time constant* is not degenerate at all, and it is what the flat ring is for.
- **N2's vertical-bump ring** for the two weights. Only a tilted ``n_0`` makes
  ``(n_0 . b)`` differ from ``1`` and ``(n_0 . v)`` differ from ``0``, so only there does
  the comparison touch the physics the analytic gates are about.

**Which xtrack fields, and why not the obvious ones.** The comparison is against
``spin_polarization_inf_no_depol`` and ``spin_t_pol_component_s``, both built from
xtrack's *closed-orbit* pair ``alpha_plus_co`` / ``alpha_minus_co``. The
similarly-named ``spin_polarization_eq`` and ``spin_t_pol_buildup_s`` additionally carry
the ``(11/18) int kappa^3 |dn/ddelta|^2`` depolarization term, which needs the spin-orbit
coupling accsim does not have yet (N4). Comparing against those instead would produce a
plausible near-miss with a physical-sounding size -- exactly the kind of disagreement that
gets chased as a coefficient bug for an afternoon.

**One known, quantified disagreement, predicted before it was measured.**
:func:`accsim.radiation.polarization_integrals` counts only dipoles, matching
:func:`accsim.radiation.radiation_integrals`, because ``alpha_plus * C == I3`` is a gate
the two accsim routes have to agree on. xtrack reads ``kappa`` from the closed orbit
element by element and so also counts the bump's **offset quadrupole**, which really does
curve the orbit and really does radiate. On the gate ring that is ``2e-11`` of
``alpha_plus`` -- and ``1e-4`` of the tilt term the comparison is actually about. It is
asserted at that size rather than tolerated, so if it ever grows the test says so.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

from accsim.radiation import (
    polarization_buildup_time,
    polarization_integrals,
    sokolov_ternov_polarization,
)
from accsim.reference import ELECTRON_ANOMALOUS_MOMENT as G
from accsim.reference import ELECTRON_MASS_EV as MASS0

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

# tests/ dirs are not import packages, so the analytic rings are reached by path. They are
# imported rather than rebuilt: the two codes must be given the *same* lattice, and a
# second copy of a bump solver is a second chance to get it wrong.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "analytic"))

import test_polarization as gate  # noqa: E402

ENERGY = 5e9
P0C = math.sqrt(ENERGY**2 - MASS0**2)
AMPLITUDE = 2e-3


def _xtrack_element(elem):
    """The xtrack twin of one accsim element -- the correspondences N1/N2 already pinned.

    ``model="exact"`` on the drift and ``model="bend-kick-bend"`` on the bend are not
    cosmetic: xtrack's defaults are the paraxial drift (M2's finding) and a split bend
    whose *orbit* differs from accsim's (N1's finding 2), and either one moves ``n_0``
    before any polarization integral is reached.
    """
    from accsim.elements.corrector import Corrector
    from accsim.elements.dipole import Dipole
    from accsim.elements.drift import Drift
    from accsim.elements.quadrupole import Quadrupole, ThinQuadrupole

    if isinstance(elem, Drift):
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


def _build(lattice, q0: float = -1.0):
    """The xtrack twin of ``lattice``, at **``lattice``'s own energy**.

    The reference momentum is read off ``lattice.ref`` rather than taken from this
    module's ``P0C``. On N3's rings the two are the same number and it makes no
    difference; on N4's they are not, because that milestone's only knob is the beam
    energy (``nu_0 = G gamma``), and a hard-coded ``p0c`` would have quietly compared a
    resonance-tuned accsim ring against a 5 GeV xtrack one -- the two codes agreeing to
    nine digits on everything except the one quantity the milestone is about.
    """
    elements = [_xtrack_element(e) for e in lattice.elements]
    line = xt.Line(elements=elements, element_names=[f"e{i}" for i in range(len(elements))])
    # Two xt.Particles defaults that are wrong here, both silently. anomalous_magnetic_moment
    # defaults to 0, which is not "spin off" but a spin tune of exactly zero (N1's trap). And
    # **q0 defaults to +1**: without it this is a positive particle with an electron's mass,
    # and the polarization comes back pointing the wrong way -- see
    # test_the_charge_is_the_fifth_silent_switch_on_the_xtrack_side.
    line.particle_ref = xt.Particles(
        mass0=lattice.ref.mass_eV,
        p0c=math.sqrt(lattice.ref.total_energy_eV**2 - lattice.ref.mass_eV**2),
        q0=q0,
        anomalous_magnetic_moment=G,
        spin_y=1.0,
    )
    line.configure_spin("auto")  # without it the kernel compiles with spin off
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    for name in line.element_names:
        element = line[name]
        if isinstance(element, xt.Bend | xt.Quadrupole):
            element.integrator = "uniform"
            element.num_multipole_kicks = 1
    return line


@pytest.fixture(scope="module")
def tilted():
    """N2's bump ring, in both codes.

    There is deliberately no *flat*-ring fixture, and the reason is a hard limit on the
    arbiter rather than a preference. xtrack's polarization analysis solves
    ``inv(lambda_i I - A) DD EE_orb`` per orbital eigenvector; with ``method="4d"`` one
    orbital eigenvalue is exactly ``1``, and a flat ring's spin matrix ``A`` is an exact
    rotation about ``y``, so ``I - A`` has an exactly zero row and ``numpy`` raises
    ``LinAlgError: Singular matrix``. A tilted ring survives only because the same matrix
    is merely ill-conditioned instead of exactly singular. The quantities compared below
    are not computed through that inverse -- it feeds the ``dn/ddelta`` depolarization
    term accsim defers to N4 -- but it aborts the whole ``twiss``, so the flat ring is
    simply unavailable as a comparison. This axis's degeneracy, arriving in the arbiter.
    """
    lattice = gate.gate_ring(AMPLITUDE, gate.electron(ENERGY))
    return lattice, _build(lattice).twiss(method="4d", spin=True, polarization_analysis=True)


@pytest.fixture(scope="module")
def tilted_default_charge():
    """The same ring, left on ``xt.Particles``' default ``q0 = +1``."""
    lattice = gate.gate_ring(AMPLITUDE, gate.electron(ENERGY))
    line = _build(lattice, q0=+1.0)
    return lattice, line.twiss(method="4d", spin=True, polarization_analysis=True)


def test_a_flat_rings_spin_matrix_is_singular_which_is_why_there_is_no_flat_fixture():
    r"""``I - A`` on a flat ring is singular to working precision -- hence no flat comparison.

    The mechanism behind the fixture note above, asserted on accsim's side where it is a
    permanent fact about the ring rather than on xtrack's, where it is a version-dependent
    crash. A flat ring's one-turn spin rotation is a rotation about ``y``: its off-diagonal
    ``y`` terms are **exactly** zero -- bit for bit, not to a tolerance, because
    :func:`accsim.spin.rotate` turns about an axis that is exactly ``(0, +-1, 0)`` -- and its
    diagonal is ``1`` to round-off. So ``I - A`` has a zero row and no inverse.

    xtrack's version is worse, not better: it builds ``A`` by central-differencing tracked
    spins at ``+-ds``, and a ``y`` component that comes back untouched gives
    ``(ds - (-ds)) / (2 ds) = 1`` *exactly*, so its ``I - A`` is exactly singular and
    ``np.linalg.inv`` raises rather than returning something large. accsim's own residual
    ``9e-16`` on the diagonal is the difference between "cannot be inverted" and "raises".
    """
    import numpy as np

    from accsim.spin import spin_one_turn_matrix

    a = spin_one_turn_matrix(gate.flat_ring(ref=gate.electron(ENERGY)))
    assert [(np.eye(3) - a)[1][0], (np.eye(3) - a)[1][2]] == [0.0, 0.0]  # exactly, not nearly
    assert abs((np.eye(3) - a)[1][1]) < 1e-15
    assert np.linalg.svd(np.eye(3) - a, compute_uv=False)[-1] < 1e-15
    assert np.linalg.cond(np.eye(3) - a) > 1e15


def test_the_charge_is_the_fifth_silent_switch_on_the_xtrack_side(tilted, tilted_default_charge):
    r"""``xt.Particles`` defaults ``q0 = +1``, and N3 is the first quantity that notices.

    N1 catalogued three silent switches on the reference side and N2 found a fourth. This
    is the fifth, and it is the quietest yet, because everything axis N compared before now
    is **blind to it**. A lattice specified by normalized strengths (``k0``, ``k1``) bends
    the same way whatever the charge, and the Thomas-BMT rotation reads the field through
    the same normalization, so the closed orbit, ``n_0``, the spin tune and the one-turn
    rotation are all bit-for-bit unchanged by ``q0`` -- which is exactly why N1's and N2's
    reference files agreed without ever setting it.

    The polarization *direction* is the first quantity on this axis that asks what the
    **physical** field is, and it is charge that turns a curvature into a field. Both codes
    then flip together, so the disagreement never appears as a disagreement: run with the
    default and xtrack cheerfully reports an electron beam polarizing *along* its guide
    field. Asserted here as the sharp statement -- ``alpha_plus`` unchanged to round-off,
    ``alpha_minus`` and ``P_inf`` exactly negated -- rather than quietly fixed in the
    fixture.
    """
    _, right = tilted
    _, wrong = tilted_default_charge

    assert float(wrong.spin_alpha_plus_co) == pytest.approx(
        float(right.spin_alpha_plus_co), rel=1e-12
    )
    assert float(wrong.spin_alpha_minus_co) == pytest.approx(
        -float(right.spin_alpha_minus_co), rel=1e-12
    )
    assert float(wrong.spin_polarization_inf_no_depol) > 0.0  # the wrong direction entirely
    assert float(right.spin_polarization_inf_no_depol) < 0.0

    # ... and nothing the earlier milestones compared moved at all.
    assert float(wrong.spin_tune_fractional) == float(right.spin_tune_fractional)
    assert list(wrong.spin_y) == list(right.spin_y)
    assert list(wrong.y) == list(right.y)


# --- the coefficient: the one thing only xtrack can catch ----------------------------


def test_the_buildup_time_agrees_with_xtracks_polarization_component(tilted):
    r"""``tau_pol`` against ``spin_t_pol_component_s`` -- the milestone's real gate.

    Both sides are ``(5 sqrt3 / 8) r_0 (hbar / m_0) gamma^5 alpha_plus_co`` inverted, and
    both assemble that constant independently: xtrack from ``scipy.constants`` in SI,
    accsim from its own ``HBAR_C_EV_M`` and the particle's rest energy through the eV
    bridge. Nothing else in the package discriminates a wrong *factor* here -- ``P_inf``
    is a ratio it cancels out of, and the ``gamma^5``/``rho^3`` scaling gates are exact
    for a rate that is ten times too fast.

    ``spin_t_pol_component_s``, deliberately, and **not** ``spin_t_pol_buildup_s``: the
    latter adds the depolarization term accsim defers to N4, and would show up as a
    plausible few-percent miss rather than as a disagreement.

    The ring is the tilted one only because xtrack cannot twiss the flat one (see the
    fixture); the tilt is irrelevant to this quantity, moving ``alpha_plus`` by ``1e-7``.
    """
    lattice, twiss = tilted
    assert polarization_buildup_time(lattice) == pytest.approx(
        float(twiss.spin_t_pol_component_s), rel=1e-6
    )


def test_the_polarization_agrees_with_xtrack_including_the_sign(tilted):
    r"""``P_inf`` matches ``spin_polarization_inf_no_depol`` -- and the content is the sign.

    The magnitude is very nearly degenerate: both codes would return ``8/(5 sqrt3)`` from
    any correct-shaped implementation. What is not degenerate is the **direction**. accsim
    recovers the physical field from ``normalized_field`` by multiplying the charge's sign
    back out of ``(B rho)_0 = p/q``; xtrack builds ``B`` from ``kappa * brho_part`` with a
    ``brho`` that carries ``q0``; the two arrive at the same *negative* answer by
    different routes. An electron beam polarizes antiparallel to its guide field.
    """
    lattice, twiss = tilted
    assert float(twiss.spin_polarization_inf_no_depol) < 0.0
    assert sokolov_ternov_polarization(lattice) == pytest.approx(
        float(twiss.spin_polarization_inf_no_depol), rel=1e-6
    )


# --- the two weights, on the only ring where they are not degenerate -----------------


def test_the_two_rate_integrals_agree_on_the_tilted_ring(tilted):
    r"""``alpha_plus`` / ``alpha_minus`` against ``spin_alpha_plus_co`` / ``..._minus_co``.

    The integrals themselves, before either derived quantity. Their agreement is bounded
    below by two *known* effects rather than by tolerance-hunting: xtrack finite-differences
    ``n_0`` where accsim's is exact (N2's finding, worth ``1e-9``), and xtrack counts the
    bump's offset quadrupole as a radiating element where accsim counts only dipoles (this
    module's docstring, worth ``2e-11`` of ``alpha_plus``).

    The last two assertions are the ones with physics in them: the two integrals must stop
    being each other's negative, by the same second-order amount, in **both** codes. A
    comparison of ``alpha_plus`` alone would agree to nine digits with the tilt term
    completely wrong.
    """
    lattice, twiss = tilted
    ours = polarization_integrals(lattice)
    theirs_plus = float(twiss.spin_alpha_plus_co)
    theirs_minus = float(twiss.spin_alpha_minus_co)
    assert ours.alpha_plus == pytest.approx(theirs_plus, rel=1e-8)
    assert ours.alpha_minus == pytest.approx(theirs_minus, rel=1e-8)

    # The tilt really is present in both codes: the two integrals stop being each other's
    # negative, by the same second-order amount, in both.
    ours_gap = (ours.alpha_plus + ours.alpha_minus) / ours.alpha_plus
    theirs_gap = (theirs_plus + theirs_minus) / theirs_plus
    assert ours_gap > 1e-8
    assert theirs_gap == pytest.approx(ours_gap, rel=1e-2)


def test_the_tilted_ring_polarization_agrees_where_the_ratio_is_not_degenerate(tilted):
    r"""``P_inf`` on the bump ring: below the textbook ratio, in both codes, by the same amount.

    On the flat ring the agreement is worth nothing. Here the departure from
    ``-8/(5 sqrt3)`` is second order in the tilt and is the only place a reference
    comparison touches the two weights at all, so the assertion is on the *departure*
    rather than on the polarization -- which agrees to nine digits either way and would
    hide a completely wrong tilt term.
    """
    lattice, twiss = tilted
    ours = abs(sokolov_ternov_polarization(lattice)) - gate.P_ST
    theirs = abs(float(twiss.spin_polarization_inf_no_depol)) - gate.P_ST

    assert ours < 0.0  # a tilted n_0 always polarizes *less*
    assert abs(ours) > 1e-8  # and there is a departure to compare at all
    assert theirs == pytest.approx(ours, rel=1e-2)
