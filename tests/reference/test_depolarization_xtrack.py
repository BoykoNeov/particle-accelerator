r"""Cross-check the invariant spin field and its depolarization (N4) against ``xtrack``.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**What only this file can see.** N4's analytic gates pin a resonance *location*, a scaling
*order*, and a *ratio*. None of them can reach ``11/18`` --- the Derbenev-Kondratenko
coefficient in front of ``int kappa^3 |dn/ddelta|^2`` --- for the same reason N3's could
not reach its own constant: it multiplies one term of a ratio whose shape is fixed by
other things. ``xtrack``'s ``spin_polarization_eq`` is the only arbiter in the project
that discriminates it, and it lives behind the skippable marker.

**The field choice is the opposite trap from N3's, and much sharper.** N3 warned that
comparing against ``spin_polarization_eq`` instead of ``spin_polarization_inf_no_depol``
would produce a plausible near-miss. Here the two must be swapped: this milestone *is*
the ``eq`` one, and on the resonant ring below the pair differ by a factor of **46**
(``-0.0199`` against ``-0.9238``). What was a quiet few-percent trap in N3 is an
unmissable one here, and :func:`test_the_two_xtrack_fields_are_not_interchangeable` says
so with an assertion so that a future edit cannot silently swap them back.

Two rings, doing two different jobs:

- **A ring a thousandth of a tune from the resonance**, where the spin field is
  well-behaved. This is where the ``(3, 6)`` matrix itself is compared, element by
  element, quadrature-free --- and where the *one* real disagreement on this axis shows
  up (below).
- **A ring a hundred-thousandth away**, where depolarization outweighs Sokolov-Ternov
  fifty to one and the polarization has collapsed to ``-0.02``. That is where the
  integrals, the equilibrium polarization and the time constant are compared, and where
  the agreement is *best* --- which is the reverse of the usual situation and is
  explained by the disagreement below.

**The one disagreement, measured and attributed rather than tolerated.** The two codes'
``dn/ddelta`` differ by ``2e-6`` in absolute terms (Windows, clang-cl; ``1.1e-5`` on
Linux/gcc -- see below) while every other column of the matrix agrees to ``1e-8``
relative. It is xtrack's, and the attribution is not a matter of opinion: without RF, ``delta`` is exactly conserved, so ``N (D, 0, 1)`` must equal the
momentum derivative of the *off-momentum closed spin solution*, which is a quantity
neither code's spin-field machinery is involved in computing. accsim's matrix satisfies
that identity to ``5e-9``; xtrack's misses it by ``1e-4``
(:func:`test_the_dispersion_identity_says_which_code_carries_the_gap`).

The mechanism is the one N3 met from the other side. xtrack solves for the spin field
mode by mode, and the ``delta`` mode has orbital eigenvalue exactly ``1``, so it needs
``inv(I - A)`` --- singular for *every* ring, since ``A n_0 = n_0``. Its finite-differenced
``A`` misses that zero by ``1e-11`` rather than exactly, the inverse comes back with
entries of order ``1e11``, and subtracting the unphysical ``n_0`` component afterwards
leaves ``1e11 * 1e-16 ~ 1e-5`` of cancellation debris in what survives. accsim solves the
whole matrix at once in the plane perpendicular to ``n_0``, where that eigenvalue does not
exist, and never forms the inverse. Which is also why accsim can do a flat ring and xtrack
raises ``LinAlgError`` on one --- N3 recorded that as an observation; here it is the same
fact, quantified.

Because the debris is **absolute**, it matters least where ``dn/ddelta`` is largest. On
the resonant ring it is ``2e-6`` of a quantity of order ``9``, and the two codes' equilibrium
polarizations agree to ``7e-5``.

**The debris is round-off, so its size is a property of the platform, not of the ring.**
Measured 2026-09-02: ``2e-6`` on Windows (clang-cl, Python 3.14) and ``1.07e-5`` on Linux
(gcc, Python 3.11, numpy 2.4.6) -- bit-identical on the latter across xtrack ``0.106.4``
and ``0.111.6`` and across two accsim commits, so it is neither the arbiter's version nor
ours. A floor asserted as a number would therefore be a statement about one compiler. The
gates below assert the *mechanism* instead, which is platform-independent: the gap is
xtrack's dispersion-identity miss itself (to ``6e-4`` of its norm), it lies in the plane
perpendicular to ``n_0`` (the ``n_0`` component is what the subtraction removed), and it is
transported round the ring as a vector (norm conserved to ``0.7%``) rather than
accumulating. Its absolute size is bounded only by the mechanism's own estimate,
``1e11 * eps ~ 2e-5``, which both platforms sit under.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from accsim.coords import DELTA, PX, PY, X, Y
from accsim.orbit import closed_orbit_nonlinear
from accsim.radiation import (
    depolarization_integrals,
    derbenev_kondratenko_polarization,
    polarization_time,
    sokolov_ternov_polarization,
)
from accsim.spin import closed_spin_solution, propagate_spin_orbit_coupling, spin_orbit_coupling

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

# tests/ dirs are not import packages, so the analytic rings and N3's xtrack twin are
# reached by path. Both are imported rather than rebuilt: the two codes must be given the
# *same* lattice, and N3's `_build` already carries four hard-won xt.Particles settings.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "analytic"))

import test_depolarization as gate  # noqa: E402
import test_polarization_xtrack as n3  # noqa: E402

#: How far in vertical tune each fixture sits from ``nu_0 = k + Q_y``.
NEAR = 1e-5
FAR = 1e-3

#: The mechanism's own estimate of xtrack's cancellation debris, ``1e11 * eps``: the only
#: absolute bound placed on the ``dn/ddelta`` gap, because its exact size is round-off.
DEBRIS_ESTIMATE = 3e-5


def dispersion_identity_miss(lattice, matrix: np.ndarray, step: float = 1e-6) -> np.ndarray:
    r"""``N (D, 0, 1) - d/ddelta [n_0 closed at delta]`` for a spin-field matrix ``N``.

    Without RF this vanishes for any correct matrix (see
    :func:`test_the_dispersion_identity_says_which_code_carries_the_gap`), and the
    right-hand side involves neither code's spin-field machinery. Returned as the vector,
    not its norm, because the gates use its *direction* as well as its size.
    """
    dispersion = (
        closed_orbit_nonlinear(lattice, delta=+step) - closed_orbit_nonlinear(lattice, delta=-step)
    ) / (2.0 * step)
    truth = (
        closed_spin_solution(lattice, delta=+step).n0
        - closed_spin_solution(lattice, delta=-step).n0
    ) / (2.0 * step)
    return matrix[:, [X, PX, Y, PY]] @ dispersion + matrix[:, DELTA] - truth


def _twiss(lattice):
    return n3._build(lattice).twiss(method="4d", spin=True, polarization_analysis=True)


@pytest.fixture(scope="module")
def far():
    """A thousandth of a tune out: the spin field is ordinary, and comparable term by term."""
    lattice = gate.ring_at_tune_distance(FAR)
    return lattice, _twiss(lattice)


@pytest.fixture(scope="module")
def near():
    """A hundred-thousandth out: depolarization outweighs Sokolov-Ternov fifty to one."""
    lattice = gate.ring_at_tune_distance(NEAR)
    return lattice, _twiss(lattice)


# --- the matrix itself, which is where the physics is --------------------------------


def test_the_orbital_columns_of_the_spin_field_agree_to_eight_digits(far):
    r"""``N``'s four transverse columns match ``spin_n_matrix`` to ``1e-8`` of its scale.

    The whole ``(3, 6)`` object, quadrature-free, before any integral is taken over it.
    These four columns are the ones reached through *betatron* eigenvalues
    ``exp(+-2 pi i Q)``, which are nowhere near ``1``, so xtrack's mode-by-mode inverse is
    well conditioned for them and the comparison is limited only by its
    finite-differenced ``A`` (N2's ``1e-10``) and by the two codes' shared bend geometry.

    **Two assertions, because either one alone leaves half the matrix unconstrained.** The
    vertical columns are four orders of magnitude larger than the horizontal ones on this
    ring (the resonance is a *vertical* one, a thousandth of a tune away). Scaling the whole
    comparison by the matrix's largest entry therefore says nothing about the horizontal
    columns -- it would admit a 1% error in them -- and a per-entry relative test would be
    dominated by them and say nothing about the vertical ones. So the columns are checked
    **individually** against their own norms, and the matrix as a whole against its largest
    entry. Measured: the horizontal columns agree to ``2e-6``, the vertical ones to
    ``4e-9``.
    """
    lattice, twiss = far
    ours = spin_orbit_coupling(lattice).matrix
    theirs = np.array(twiss.spin_n_matrix)[0]

    columns = [X, PX, Y, PY]
    assert np.abs(ours[:, columns] - theirs[:, columns]).max() < 1e-8 * np.abs(ours).max()
    for column in columns:
        gap = float(np.linalg.norm(ours[:, column] - theirs[:, column]))
        assert gap < 1e-5 * float(np.linalg.norm(ours[:, column]))


def test_the_momentum_column_agrees_only_to_the_arbiters_cancellation_floor(far):
    r"""``dn/ddelta`` differs by ``1e-5`` absolute -- a thousand times worse than the rest.

    The disagreement this file exists to pin down, asserted so that it can neither grow
    nor quietly vanish unnoticed. Every other column agrees to ``1e-8`` of the matrix
    scale; this one is out by ``1e-4`` in relative terms, and it is the only column whose
    orbital eigenvalue is exactly ``1``.

    **Its size is not a number this test can own.** It was first measured at ``2e-6`` on
    Windows/clang-cl and asserted at that size; the same ring on Linux/gcc gives ``1.07e-5``,
    bit for bit across two xtrack versions and two accsim commits. The gap is round-off in
    xtrack's near-singular ``inv(I - A)`` (the next two tests), so it is the *compiler's*
    number, and asserting it to a factor of two would only ever pass on one box. What is
    asserted instead is the thing that does not depend on the platform: the gap **is**
    xtrack's dispersion-identity miss, vector for vector -- the two agree to ``6e-4`` of
    their norm, the residue being the transverse columns' ``1e-8`` share -- and it is
    bounded above only by the mechanism's own estimate, ``1e11 * eps``.

    The next two tests establish that the gap is xtrack's and explain where it comes from.
    It is stated here first, without attribution, because that is the order the evidence
    actually arrived in.
    """
    lattice, twiss = far
    ours = spin_orbit_coupling(lattice).matrix
    theirs = np.array(twiss.spin_n_matrix)[0]

    gap = ours[:, DELTA] - theirs[:, DELTA]
    size = float(np.linalg.norm(gap))
    assert 1e-7 < size < DEBRIS_ESTIMATE
    assert size > 1e-6 * float(np.abs(ours[:, DELTA]).max())  # far above the other columns

    # The gap is the arbiter's own inconsistency, as a vector: what xtrack's column misses
    # of the identity is exactly what it misses of ours.
    miss = dispersion_identity_miss(lattice, theirs)
    assert float(np.linalg.norm(gap + miss)) < 1e-2 * size


def test_the_dispersion_identity_says_which_code_carries_the_gap(far):
    r"""A third quantity, computed by neither code's spin-field machinery, breaks the tie.

    Two matrices disagree; nothing in a two-way comparison says which is wrong. There is a
    third route here, and it is exact. Without an RF cavity ``delta`` is a constant of the
    motion, so a particle displaced along ``(D_x, D_px, D_y, D_py, 0, 1)`` sits *on* the
    closed orbit of a different momentum, and its invariant spin direction is that ring's
    ``n_0``. Therefore, for any correct spin-field matrix,

        ``N (D, 0, 1) = d/ddelta [ n_0 closed at delta ]``,

    and the right-hand side is obtained by closing an orbit at ``+-ddelta`` and reading off
    the periodic spin direction --- no spin field, no eigenvector inverse, no Sylvester
    solve, in either code.

    accsim's matrix satisfies it to ``5e-9``. xtrack's misses it by ``1e-4``, four orders
    of magnitude worse and in exactly the amount the previous test measured. The gap is
    xtrack's.

    This is the same shape as M2's three-code split and F2's combined-function
    disagreement: localise the difference against something *outside* the disagreement
    before deciding which side to change.
    """
    lattice, twiss = far
    step = 1e-6
    truth = (
        closed_spin_solution(lattice, delta=+step).n0
        - closed_spin_solution(lattice, delta=-step).n0
    ) / (2.0 * step)

    def residual(matrix: np.ndarray) -> float:
        miss = dispersion_identity_miss(lattice, matrix, step)
        return float(np.linalg.norm(miss) / np.linalg.norm(truth))

    ours = residual(spin_orbit_coupling(lattice).matrix)
    theirs = residual(np.array(twiss.spin_n_matrix)[0])

    assert ours < 1e-7
    assert theirs > 1e-5
    assert theirs > 1000.0 * ours


def test_the_singularity_behind_that_gap_is_the_one_n3_could_only_observe(far):
    r"""``I - A`` is singular for **every** ring, and that is why the ``delta`` mode suffers.

    N3 found that xtrack cannot twiss an exactly flat ring: it inverts ``lambda_i I - A``
    per orbital eigenvector, ``method="4d"`` leaves an orbital eigenvalue at exactly ``1``,
    and a flat ring's spin matrix is an exact rotation about ``y``, so ``I - A`` has a zero
    row. That was recorded as a fact about flat rings. It is not --- it is a fact about
    *all* rings, because ``A n_0 = n_0`` by the definition of ``n_0``, so ``I - A`` always
    has ``n_0`` in its kernel and is always singular.

    A tilted ring survives only because xtrack's ``A`` is finite-differenced and misses the
    zero by round-off, leaving an inverse with enormous entries whose ``n_0`` part is then
    subtracted off. What that subtraction cannot remove is its own cancellation debris,
    and that debris is the previous three tests' gap.

    Asserted on accsim's side, where it is a permanent property of the ring rather than a
    version-dependent crash: this ring is *tilted*, and ``I - A`` is still singular to
    ``1e-16``, with a condition number above ``1e15``.
    """
    lattice, twiss = far
    coupling = spin_orbit_coupling(lattice)
    residual = np.eye(3) - coupling.one_turn_matrix

    smallest = float(np.linalg.svd(residual, compute_uv=False)[-1])
    assert smallest < 1e-15
    assert np.linalg.cond(residual) > 1e14

    # And the debris has the direction the mechanism says: what survives the subtraction
    # of the ``n_0`` component is perpendicular to ``n_0`` (measured ``5e-4`` of its norm).
    gap = coupling.dn_ddelta - np.array(twiss.spin_n_matrix)[0][:, DELTA]
    assert abs(float(coupling.n0 @ gap)) < 1e-2 * float(np.linalg.norm(gap))


def test_the_field_agrees_all_the_way_round_the_ring(far):
    r"""``N(s)`` against ``spin_n_matrix`` at every element boundary, not only the entrance.

    A matrix that matched at ``s = 0`` and drifted afterwards would still produce the right
    ``dn/ddelta`` there and the wrong integral over the ring, so the pointwise comparison
    is the one that protects the integrals below. accsim propagates the field by tracking
    a bundle launched on it; xtrack tracks its scaled eigenvectors and re-fits the matrix
    at each element, having rephased and averaged the two signs. Two genuinely different
    transports, agreeing element by element to the same absolute floor the entrance shows
    --- so the floor really is a property of the ``delta`` mode and not something that
    accumulates.

    "Does not accumulate" is made exact by the mechanism. The debris is a vector in the
    entrance ``dn/ddelta``; downstream, the ``delta`` column is the spin rotation applied
    to it plus the transverse columns weighted by the orbit's dispersion, and those agree
    to ``1e-8``. So the gap's **norm** is transported unchanged round the ring (measured:
    ``0.7%`` spread over 81 boundaries, which is xtrack's per-element re-fit) while its
    largest component, which is what a per-entry comparison sees, swings by ``1.3x`` as the
    rotation turns it. Asserted on the norm, with the same mechanism bound as the entrance.
    """
    lattice, twiss = far
    ours = propagate_spin_orbit_coupling(lattice)
    theirs = np.array(twiss.spin_n_matrix)

    # xtrack's twiss table carries an ``_end_point`` row after the last element, so the
    # two sequences are boundary-for-boundary the same length: entrance, then every exit.
    assert len(ours) == len(theirs) == len(lattice.elements) + 1
    gaps = np.array(
        [np.linalg.norm(ours[i][:, DELTA] - theirs[i][:, DELTA]) for i in range(len(theirs))]
    )
    assert gaps.max() < DEBRIS_ESTIMATE
    assert gaps.max() < 1.05 * gaps[0]  # it does not accumulate down the ring
    assert gaps.min() > 0.95 * gaps[0]  # nor is it lost: a rotation carries it round


# --- the integrals, and the coefficient nothing else can reach -----------------------


def test_the_two_depolarization_integrals_agree_on_the_resonant_ring(near):
    r"""``(11/18) int kappa^3 |dn/ddelta|^2`` and ``int kappa^3 dn/ddelta . b`` against xtrack.

    The milestone's real gate, and the only place ``11/18`` is checked at all. Both sides
    build the same two integrals from their own spin field; a coefficient that were, say,
    ``1/2`` instead of ``11/18`` --- a 10% error, entirely plausible from memory --- fails
    here and passes every analytic gate in the package.

    **The resonant ring is used because the agreement is *better* there**, which is worth
    stating plainly because it is the reverse of the usual arrangement. xtrack's
    ``dn/ddelta`` carries an *absolute* ``2e-6`` to ``1e-5`` of cancellation debris (see
    above), so it matters in proportion to how small ``dn/ddelta`` is. A hundred-thousandth
    of a tune from the resonance ``|dn/ddelta| ~ 9``, and the debris is worth ``1e-6`` of
    it at most.

    The residual is bounded below by two known effects rather than by tolerance-hunting,
    exactly as N3's was:

    - **xtrack's quadrature**, which carries essentially all of it. It sums ``kappa^3``
      times the integrand at each element *entrance* times that element's length -- a
      rectangle rule over 2 m bends -- where accsim sub-slices each bend with Simpson's rule
      and is converged to twelve digits by eight sub-slices (N4's analytic file measures
      that).
    - **the bump's off-axis quadrupole**, which xtrack counts as radiating and accsim does
      not (only dipoles radiate, matching ``radiation_integrals``). N3 measured its share of
      ``alpha_plus`` at ``2e-11`` and warned that it grows as the *cube* of the orbit
      offset. It had to be re-measured here, because it now carries ``|dn/ddelta|^2`` as
      well and that is of order ``80`` on this ring: it comes to ``3e-11`` of the
      depolarization integral, and -- this is the reason it does not grow -- it stays there
      at every distance from the resonance, because the new weight multiplies the
      quadrupole's contribution and the ring's by the same diverging factor. Seven orders
      of magnitude below the quadrature, so the attribution above is safe.
    """
    lattice, twiss = near
    integrals = depolarization_integrals(lattice)
    circumference = lattice.length

    assert integrals.depolarization_plus * circumference == pytest.approx(
        float(twiss.spin_int_kappa3_11_18_dn_ddelta_sq), rel=2e-4
    )
    assert integrals.depolarization_minus * circumference == pytest.approx(
        float(twiss.spin_int_kappa3_dn_ddelta_ib), rel=2e-4
    )

    # ... and they really are the dominant thing here, not a correction.
    assert integrals.depolarization_plus > 40.0 * integrals.alpha_plus_co


def test_the_equilibrium_polarization_and_its_time_constant_agree(near):
    r"""``P_eq`` and ``tau`` against ``spin_polarization_eq`` / ``spin_t_pol_buildup_s``.

    The two assembled quantities, both of which carry ``11/18`` and neither of which any
    analytic gate can reach. They agree to ``1e-4``, which is the quadrature difference
    above propagated through.

    ``tau`` additionally re-checks the eV-to-SI bridge that N3 built --- xtrack assembles
    its constant from ``scipy.constants`` in SI and accsim from its own ``HBAR_C_EV_M``
    through eV --- but on a ring where ``alpha_plus`` is now dominated by the
    depolarization term rather than by the closed-orbit rate, so it is a different linear
    combination of the same pieces than N3 checked.
    """
    lattice, twiss = near

    assert derbenev_kondratenko_polarization(lattice) == pytest.approx(
        float(twiss.spin_polarization_eq), rel=1e-4
    )
    assert polarization_time(lattice) == pytest.approx(float(twiss.spin_t_pol_buildup_s), rel=1e-4)


def test_the_two_xtrack_fields_are_not_interchangeable(near):
    r"""``spin_polarization_eq`` and ``spin_polarization_inf_no_depol`` differ by 46x here.

    N3 compared against the ``_co`` / ``no_depol`` pair and warned that reaching for the
    other one would produce "a plausible near-miss with a physical-sounding size --
    exactly the kind of disagreement that gets chased as a coefficient bug for an
    afternoon". N4 is the milestone that must use the other one, and on the resonant ring
    the two are not a near-miss at all: ``-0.0199`` against ``-0.9238``.

    Asserted as a **factor** rather than a value so that a future edit which swaps the
    field names back fails loudly and immediately rather than drifting. The same assertion
    also confirms the two codes agree about which quantity is which: accsim's
    :func:`~accsim.radiation.sokolov_ternov_polarization` matches xtrack's ``no_depol``
    number on the very ring where its
    :func:`~accsim.radiation.derbenev_kondratenko_polarization` has collapsed away from
    it.
    """
    lattice, twiss = near
    equilibrium = float(twiss.spin_polarization_eq)
    undepolarized = float(twiss.spin_polarization_inf_no_depol)

    assert abs(undepolarized / equilibrium) > 40.0
    assert undepolarized == pytest.approx(-8.0 / (5.0 * math.sqrt(3.0)), abs=1e-5)
    assert sokolov_ternov_polarization(lattice) == pytest.approx(undepolarized, rel=1e-6)


def test_both_codes_see_the_collapse_and_neither_sees_it_in_p_inf(far, near):
    r"""The milestone's headline, verified against the arbiter rather than only asserted.

    N4's analytic file shows accsim's ``P_eq`` collapsing from ``-0.92`` to ``-0.02`` as the
    spin tune closes on ``Q_y`` while ``P_inf`` sits unmoved. That is a statement about
    accsim alone. Here the same two rings are run through xtrack, and xtrack collapses by
    the same factor between them while *its* ``P_inf`` moves in the ninth digit.

    A shared error would have to be a shared error in two independently written spin-field
    solvers --- one a Sylvester solve in the plane perpendicular to ``n_0``, the other a
    mode-by-mode eigenvector inverse --- that happened to produce the same resonant
    denominator, the same coefficient and the same lattice weighting. That is the strongest
    statement available on this axis, and it is what the reference suite is for.
    """
    far_lattice, far_twiss = far
    near_lattice, near_twiss = near

    ours = [
        derbenev_kondratenko_polarization(far_lattice),
        derbenev_kondratenko_polarization(near_lattice),
    ]
    theirs = [float(far_twiss.spin_polarization_eq), float(near_twiss.spin_polarization_eq)]

    assert ours[0] / ours[1] == pytest.approx(theirs[0] / theirs[1], rel=1e-3)
    assert abs(ours[0] / ours[1]) > 40.0

    undepolarized = [
        float(far_twiss.spin_polarization_inf_no_depol),
        float(near_twiss.spin_polarization_inf_no_depol),
    ]
    assert abs(undepolarized[0] - undepolarized[1]) < 1e-8
