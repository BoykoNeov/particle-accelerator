r"""N3 -- Sokolov-Ternov: the polarization synchrotron radiation builds up.

A bending electron radiates, and a tiny fraction of that radiation flips its spin. The
two flip directions are not equally likely, so a stored beam slowly polarizes -- to
``8/(5 sqrt3) = 92.376%``, antiparallel to the guide field, with a time constant that
runs from a second in a small strong-bending ring to hours at LEP. Two lattice averages
carry all of it (A. Chao; ``xtrack``'s ``spin_alpha_plus_co`` / ``spin_alpha_minus_co``):

    ``alpha_plus  = (1/C) int kappa^3 (1 - (2/9) (n_0 . v)^2) ds``   -- the rate,
    ``alpha_minus = (1/C) int kappa^3 (n_0 . b) ds``                 -- the direction.

**The headline number is a control, not a gate, and this file says so with assertions
rather than in prose.** ``P_inf = 8/(5 sqrt3)`` is a *ratio* of those two integrals, and
on any flat ring ``n_0`` is parallel to the field everywhere it bends, so the ratio is
``-1`` before either integral is evaluated. It comes back at ``-0.9237604307034013`` --
the same sixteen digits -- at every energy, every bending radius, every quadrupole
strength and every slice count this file tries
(:func:`test_the_flat_ring_ratio_is_the_same_sixteen_digits_whatever_the_ring_is`). Any
uniform mis-scale of the pair cancels out of it exactly. This is B5's "three arbiters
lying quietly" and J1's "structural gates are blind to the kick coefficient", arriving on
a third axis.

Three gates carry the milestone instead, each aimed at something the ratio cannot see:

- **The two weights pulling apart.** On N2's vertical-bump ring ``n_0`` tilts away from
  the field by ``t``, and the two integrals then differ by
  ``t^2 (1/2 - (2/9) <cos^2>)`` -- second order, and of *opposite sign* in the two, since
  ``(n_0 . b)`` loses ``t^2/2`` while ``(n_0 . v)^2`` gains ``(2/9) t^2 <cos^2>``. One
  assertion on that difference pins **both** weights and cannot pass on a normalization
  coincidence, which is what an ``alpha_plus * C == I3`` check on its own would do
  (:func:`test_the_tilted_ring_separates_the_two_weights`). The arc average
  ``<cos^2>`` is integrated in sympy, not recalled.

- **The direction.** ``alpha_minus`` is signed and ``alpha_plus`` is not: the size of the
  bend lives in ``kappa^3``, its sense lives in ``b``. The sign is anchored on two
  independent knobs -- swap the charge, or reverse every bend -- and each flips the
  answer alone (:func:`test_the_beam_polarizes_antiparallel_to_the_guide_field`).

- **The time constant's coefficient**, which is the one quantity ``P_inf`` provably
  cannot reach. It is guarded at three removes: the ``gamma^5`` and ``rho^3`` *powers*
  by scaling, the eV-to-SI bridge by an independent assembly from ``scipy.constants``,
  and the machine-scale answer by LEP, whose Sokolov-Ternov time is ~5.5 hours. A wrong
  *factor* that survives all three is caught only by ``xtrack``
  (``tests/reference/test_polarization_xtrack.py``) -- which lives behind the skippable
  ``reference`` marker, so a green analytic suite is weaker evidence here than usual.

Two facts found on the way, both asserted rather than assumed. The horizontal part of
``n_0`` **counter-rotates** against the bend: its phase advances as ``-G gamma`` per unit
bend angle, and taking that sign the other way puts the arc average of ``cos^2`` 1.5% out
-- small enough to read as a quadrature error and large enough to be a wrong answer
(:func:`test_the_spin_phase_counter_rotates_against_the_bend`). And the quadrature has to
resolve the **spin** phase rather than the optics: ``G gamma theta`` is 4.4 radians across
one bend of the gate ring where the dispersion turns through ``theta = 0.39``, which is
why this integral uses Simpson's rule where :func:`accsim.radiation.radiation_integrals`
uses the trapezoid (:func:`test_the_quadrature_converges_at_fourth_order`).
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
import sympy as sp
from scipy import constants as sci

from accsim.coords import PY, Y
from accsim.elements.corrector import Corrector
from accsim.elements.dipole import Dipole
from accsim.elements.drift import Drift
from accsim.elements.quadrupole import Quadrupole, ThinQuadrupole
from accsim.lattice import Lattice
from accsim.radiation import (
    polarization_buildup_time,
    polarization_integrals,
    radiation_integrals,
    sokolov_ternov_polarization,
)
from accsim.reference import (
    ELECTRON_ANOMALOUS_MOMENT,
    ELECTRON_MASS_EV,
    ReferenceParticle,
)
from accsim.spin import propagate_spin_solution

G_E = ELECTRON_ANOMALOUS_MOMENT

#: ``8 / (5 sqrt3)``, the ratio of the two spin-flip rates. Sokolov-Ternov's number.
P_ST = 8.0 / (5.0 * math.sqrt(3.0))


def electron(energy_eV: float = 5e9, charge: float = -1.0) -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(
        ELECTRON_MASS_EV, energy_eV, charge=charge, anomalous_moment=G_E
    )


# --- the lattices -------------------------------------------------------------------

N_CELLS = 8  # FODO cells in the arc; two bends each, summing to 2 pi
BEND_L = 2.0  # arc dipole length [m]
KL = 0.20  # thin-quadrupole strength [1/m] -- thin, so it does not precess
K1 = 0.6  # the one thick quadrupole, which is the whole spin perturbation
LQ = 0.4  # its length [m]
SLICES = 8  # slices it is cut into
DRIFT = 0.6  # drift between the bump's correctors [m]


def flat_ring(
    k1: float = 1.2,
    ref: ReferenceParticle | None = None,
    n_cells: int = 8,
    scale: float = 1.0,
    bend_sign: float = +1.0,
) -> Lattice:
    """A flat, unsteered ring -- the degenerate case, where ``n_0`` is ``+y`` everywhere.

    ``scale`` stretches every length by the same factor while shrinking ``k1`` as
    ``1/scale^2``, which leaves every transfer matrix identical and multiplies the
    bending radius by ``scale``: a *geometrically similar* machine, and the clean way to
    move ``rho`` without also moving the optics.
    """
    ref = ref if ref is not None else electron()
    elements: list = []
    for _ in range(n_cells):
        elements += [
            Dipole(length=BEND_L * scale, angle=bend_sign * 2.0 * math.pi / n_cells),
            Drift(0.5 * scale),
            Quadrupole(0.4 * scale, k1 / scale**2),
            Drift(0.5 * scale),
            Quadrupole(0.4 * scale, -k1 / scale**2),
        ]
    return Lattice(elements, ref)


def _closure_pattern(elements: list, ref: ReferenceParticle) -> np.ndarray:
    """Corrector strengths that leave ``(y, py) = 0`` at the end of ``elements``."""
    columns = []
    for i, elem in enumerate(elements):
        if not isinstance(elem, Corrector):
            continue
        downstream = np.eye(2)
        for later in elements[i + 1 :]:
            downstream = later.matrix(ref)[np.ix_([Y, PY], [Y, PY])] @ downstream
        columns.append(downstream @ np.array([0.0, 1.0]))
    _, _, right = np.linalg.svd(np.array(columns).T)
    pattern = right[-1]
    return pattern / np.max(np.abs(pattern))


def gate_ring(amplitude: float, ref: ReferenceParticle | None = None) -> Lattice:
    """N2's ring: a closed vertical bump round one thick quadrupole, then a flat arc.

    The bump tilts ``n_0`` away from the vertical by a small angle while leaving the arc
    exactly on the design orbit, which is the only construction in this package that
    makes ``(n_0 . b)`` differ from ``1`` and ``(n_0 . v)`` differ from ``0``. The
    straight comes **first**, so the arc entrance -- where the closed forms below read
    the tilt off -- is element :func:`n_straight`.
    """
    ref = ref if ref is not None else electron()
    quadrupole = [Quadrupole(LQ / SLICES, K1) for _ in range(SLICES)]

    def build(t1: float, t2: float, t3: float) -> list:
        return [
            Drift(DRIFT),
            Corrector(kick_y=t1),
            Drift(DRIFT),
            Corrector(kick_y=t2),
            Drift(DRIFT),
            *quadrupole,
            Drift(DRIFT),
            Corrector(kick_y=t3),
            Drift(DRIFT),
        ]

    elements = build(*(amplitude * _closure_pattern(build(0.0, 0.0, 0.0), ref)))
    for _ in range(N_CELLS):
        for sign in (+1, -1):
            elements += [
                ThinQuadrupole(sign * KL),
                Drift(0.5),
                Dipole(length=BEND_L, angle=math.pi / N_CELLS),
                Drift(0.5),
            ]
    return Lattice(elements, ref)


def n_straight() -> int:
    """Elements in the straight -- five before the thick quadrupole, three after."""
    return 8 + SLICES


def arc_curvature() -> float:
    """``h = 1/rho`` of every arc dipole in :func:`gate_ring`."""
    return (math.pi / N_CELLS) / BEND_L


# --- the closed forms, derived rather than recalled ----------------------------------


@cache
def derive_arc_average_of_cos_squared() -> sp.Expr:
    r"""``(1/2pi) int_0^{2pi} cos^2(phi_0 - g phi) dphi`` as an exact expression in
    ``(phi_0, g)``.

    The one integral the tilted ring's ``alpha_plus`` needs, and the one place a
    remembered "the average of ``cos^2`` is ``1/2``" would be quietly wrong. It is
    ``1/2`` plus a term that falls as ``1/g``, and at ``g = G gamma = 11.35`` that term is
    worth 0.6% of the answer -- comparable to the 1.5% a flipped phase sign costs, and
    both are far above the ``3e-8`` the quadrature reaches. Integrated symbolically, then
    evaluated; never assumed.
    """
    phi, phi_0, g = sp.symbols("phi phi_0 g", real=True)
    integral = sp.integrate(sp.cos(phi_0 - g * phi) ** 2, (phi, 0, 2 * sp.pi))
    return sp.simplify(integral / (2 * sp.pi))


def arc_average_of_cos_squared(phi_0: float, g: float) -> float:
    """:func:`derive_arc_average_of_cos_squared`, evaluated."""
    expr = derive_arc_average_of_cos_squared()
    symbols = {str(s): s for s in expr.free_symbols}
    return float(expr.subs({symbols["phi_0"]: phi_0, symbols["g"]: g}))


@cache
def derive_the_ev_to_si_bridge() -> sp.Expr:
    r"""``hbar / m_0`` in ``m^2/s``, from ``hbar c`` [eV m], ``c`` [m/s] and ``mc^2`` [eV].

    The step where this milestone could silently lose a factor: everything in the
    accelerator core is in eV and metres, and the Sokolov-Ternov rate is written with an
    ``hbar`` in joule-seconds and a mass in kilograms. Rather than trust
    ``hbar_c * c / mass_eV``, the substitution ``hbar = (hbar c)/c`` and
    ``m_0 = (mc^2)/c^2`` is made symbolically and the ratio simplified, so the identity
    is *shown*. :func:`test_the_ev_to_si_bridge_is_an_identity_not_a_coincidence` then
    evaluates both sides against ``scipy.constants``, which never passes through eV at
    all.
    """
    hbar, m_0, hbar_c, mc2, c = sp.symbols("hbar m_0 hbar_c mc2 c", positive=True)
    return sp.simplify((hbar / m_0).subs({hbar: hbar_c / c, m_0: mc2 / c**2}))


def independent_buildup_time(alpha_plus: float, gamma: float) -> float:
    """``tau_pol`` [s] assembled from ``scipy.constants`` alone -- no accsim constant.

    The second route to the same number, and the reason it exists: every constant the
    package uses is expressed in eV, so a slip in the eV-to-SI bridge would be invisible
    to any check written in the package's own units. This one is in kilograms and
    joule-seconds throughout.
    """
    rate = (
        5.0
        * math.sqrt(3.0)
        / 8.0
        * sci.value("classical electron radius")
        * sci.hbar
        / sci.m_e
        * gamma**5
        * alpha_plus
    )
    return 1.0 / rate


# --- the control: what the headline number cannot see --------------------------------


@pytest.mark.parametrize(
    ("k1", "n_cells", "scale", "energy_eV", "slices"),
    [
        (1.2, 8, 1.0, 5e9, 64),
        (0.7, 8, 1.0, 5e9, 64),  # different focusing
        (1.2, 12, 1.0, 5e9, 64),  # different cell count, so a different bend angle
        (1.2, 8, 3.0, 5e9, 64),  # a ring three times the size
        (1.2, 8, 1.0, 1.7e9, 64),  # a different energy, so a different spin tune
        (1.2, 8, 1.0, 5e9, 8),  # a coarser quadrature
    ],
)
def test_the_flat_ring_ratio_is_the_same_sixteen_digits_whatever_the_ring_is(
    k1: float, n_cells: int, scale: float, energy_eV: float, slices: int
):
    r"""``P_inf = -8/(5 sqrt3)`` on every flat ring, bit for bit -- which proves almost nothing.

    On a flat, unsteered lattice ``n_0`` is ``+y`` everywhere and every field is vertical,
    so ``(n_0 . b) = -1`` and ``(n_0 . v) = 0`` at every point, and the ratio of the two
    integrals is ``-1`` before either is computed. The bending radius, the focusing, the
    energy, the number of bends and even the slice count all cancel out of it exactly.

    This is asserted, and asserted across six rings, precisely so that the milestone's
    headline number is on record as a **control**: a uniform mis-scale of both rates --
    a wrong power of ``kappa``, a wrong circumference, a factor of two in the
    accumulation -- would leave every one of these passing. What it *does* test is that
    the two integrals travel the same code path and that ``n_0`` really is parallel to
    the field, which is worth exactly that much and no more.
    """
    ring = flat_ring(k1, electron(energy_eV), n_cells=n_cells, scale=scale)
    assert sokolov_ternov_polarization(ring, slices=slices) == pytest.approx(-P_ST, abs=1e-15)


def test_alpha_plus_is_the_third_radiation_integral_over_the_circumference():
    r"""``alpha_plus * C == I3`` on a flat ring -- the normalisation, and nothing else.

    With ``(n_0 . v) = 0`` the ``2/9`` weight is switched off and ``alpha_plus`` reduces
    to ``(1/C) int kappa^3 ds``, which is ``I3/C`` by definition. The two routes are
    genuinely independent code -- ``I3`` is ``|h|^3 L`` per dipole from the element's
    geometry, ``alpha_plus`` is a Simpson sum over the tracked closed orbit reading
    ``normalized_field`` at each sub-slice -- so agreement to round-off pins the
    circumference, the cube, and the fact that both routes count the same magnets.

    It is deliberately *not* the milestone's gate: it is blind to both weights, which is
    the whole physics. That is what :func:`test_the_tilted_ring_separates_the_two_weights`
    is for.
    """
    ring = flat_ring()
    integrals = polarization_integrals(ring)
    assert integrals.alpha_plus * ring.length == pytest.approx(
        radiation_integrals(ring).i3, rel=1e-13
    )
    assert integrals.alpha_minus == pytest.approx(-integrals.alpha_plus, rel=1e-15)


# --- the direction --------------------------------------------------------------------


def test_the_beam_polarizes_antiparallel_to_the_guide_field():
    r"""The sign of ``P_inf``, anchored on two knobs that move it independently.

    ``alpha_minus`` is signed and ``alpha_plus`` is not: the magnitude of the bend lives
    in ``kappa^3``, its sense lives in the unit vector ``b`` along the **physical** field.
    Recovering that field from ``normalized_field`` means multiplying the charge's sign
    back in, because the normalisation ``(B rho)_0 = p/q`` carries it -- and a package
    that skipped that step would produce every magnitude in this file correctly and the
    direction backwards.

    So the claim is tested by moving each of the two things the field direction depends
    on, one at a time. An electron ring polarizes **antiparallel** to its guide field
    (the textbook direction, and not a convention this package is free to choose);
    swapping to a positron flips it, and reversing every bend on the electron ring flips
    it too. Magnitudes are untouched in all three, which is what makes it a test of the
    direction alone.
    """
    assert sokolov_ternov_polarization(flat_ring()) == pytest.approx(-P_ST, abs=1e-15)
    assert sokolov_ternov_polarization(flat_ring(ref=electron(charge=+1.0))) == pytest.approx(
        +P_ST, abs=1e-15
    )
    assert sokolov_ternov_polarization(flat_ring(bend_sign=-1.0)) == pytest.approx(+P_ST, abs=1e-15)


def test_the_spin_phase_counter_rotates_against_the_bend():
    r"""Through one bend, ``n_0``'s horizontal phase advances by ``-G gamma theta``.

    The sign found while deriving the tilted ring's ``alpha_plus``, and worth its own
    assertion because of how it failed: taking it the other way leaves the arc average of
    ``cos^2`` 1.5% out, which reads exactly like a quadrature error and is not one. The
    quantity is ``atan2(n_x, n_z)``, measured at the arc entrance and again one dipole
    later, and it goes **down** by ``G gamma theta`` where the bend angle ``theta`` goes
    up -- the spin's horizontal part turns the opposite way from the trajectory.

    ``|G gamma theta| = 4.5`` radians for a single bend here, so the phase is unwrapped
    against the prediction rather than compared raw.

    It is measured on a **flat** ring, with a horizontal spin injected by hand rather than
    ``n_0`` read off the gate ring, and the second half of the test says why. On the gate
    ring the same phase misses ``-G gamma theta`` by ``7.4e-8``, which is not round-off:
    it grows as the **square** of the bump amplitude, to five digits, and numerical noise
    does not scale at all. It has to originate in the bump's vertical orbit leaking into
    the arc (``py = 2.2e-11`` there), since that is the only asymmetry the ring has -- but
    the ``h i_y G (gamma - 1)`` precession N2 pins is *first* order in ``py``, and would
    therefore be amplitude-**independent** once divided by a tilt itself linear in the
    bump. The observed second order does not match it, so the precise mechanism is **not
    established here**; it is stated as measured rather than explained away.

    Which is why the assertion is on the exponent and a bound, not on the number. That is
    enough for the purpose it serves: the residual is a hundred times smaller than the
    tilt term this milestone gates, and it explains why the counter-rotation sign is
    measured on a flat ring instead of on the bump ring.
    """
    ref = electron()
    theta = 2.0 * math.pi / N_CELLS
    predicted = -G_E * ref.gamma0 * theta
    assert predicted < -8.9  # a large negative phase, not a rounding sign

    flat = propagate_spin_solution(flat_ring(ref=ref), n0=np.array([0.0, 0.0, 1.0]))
    phase = math.atan2(flat[1][0], flat[1][2]) - math.atan2(flat[0][0], flat[0][2])
    assert abs((phase - predicted + math.pi) % (2.0 * math.pi) - math.pi) < 1e-14

    def arc_leak(amplitude: float) -> float:
        """How far one arc bend's phase misses ``-G gamma theta`` on the bump ring."""
        n0 = propagate_spin_solution(gate_ring(amplitude, ref))
        # The arc cell is ThinQuadrupole, Drift, Dipole, Drift; only the dipole precesses.
        entrance, after_one_bend = n0[n_straight() + 2], n0[n_straight() + 3]
        phase = math.atan2(after_one_bend[0], after_one_bend[2]) - math.atan2(
            entrance[0], entrance[2]
        )
        predicted = -G_E * ref.gamma0 * (math.pi / N_CELLS)
        return (phase - predicted + math.pi) % (2.0 * math.pi) - math.pi

    amplitudes = [1e-3, 2e-3, 4e-3]
    leaks = [abs(arc_leak(a)) for a in amplitudes]
    assert 1e-8 < leaks[0] < 1e-6  # far above round-off, far below the phase itself
    slope = np.polyfit(np.log(amplitudes), np.log(leaks), 1)[0]
    assert slope == pytest.approx(2.0, abs=1e-3)


# --- the gate: the two weights pull apart --------------------------------------------


def _tilt_of_the_arc(ring: Lattice) -> tuple[float, float, float]:
    """``(n_y, t, phi_0)`` at the arc entrance: the vertical part, the tilt, its phase.

    ``n_y`` is invariant through the whole arc -- every rotation there is about ``y`` --
    so one reading fixes ``(n_0 . b)`` everywhere it matters, and ``t = sqrt(1 - n_y^2)``
    is likewise constant. Only the *phase* of the horizontal part moves, and it moves at
    the rate :func:`test_the_spin_phase_counter_rotates_against_the_bend` pins.
    """
    n_arc = propagate_spin_solution(ring)[n_straight()]
    return (
        float(n_arc[1]),
        float(math.hypot(n_arc[0], n_arc[2])),
        float(math.atan2(n_arc[0], n_arc[2])),
    )


@pytest.mark.parametrize("amplitude", [1e-3, 2e-3, 4e-3])
def test_the_tilted_ring_separates_the_two_weights(amplitude: float):
    r"""The milestone's gate: ``alpha_plus + alpha_minus`` on a ring whose ``n_0`` tilts.

    On N2's vertical-bump ring the closed spin direction leaves the vertical by a small
    angle ``t``, and the two integrals stop being each other's negative. Each departs from
    ``I3/C`` on its own account, and in *opposite* directions:

        ``|alpha_minus| C / I3 = n_y            = 1 - t^2/2 + O(t^4)``   (the ``(n_0 . b)`` weight)
        ``alpha_plus     C / I3 = 1 - (2/9) t^2 <cos^2>``                (the ``(n_0 . v)^2`` weight)

    so their **sum** is ``t^2 (1/2 - (2/9) <cos^2>)`` -- one number carrying both weights,
    with different coefficients and opposite signs, which is what makes it impossible to
    pass by a normalisation coincidence. ``alpha_plus * C == I3`` and
    ``alpha_minus = -n_y I3/C`` are each true of a wrong pairing of weights; this is not.

    Both alternatives are asserted to be *excluded*, not merely different: dropping the
    ``2/9`` term, or keeping it alone, each lands hundreds of tolerances away. ``<cos^2>``
    comes from :func:`derive_arc_average_of_cos_squared` (sympy, not the remembered
    ``1/2``), ``I3`` from the arc's geometry rather than from
    :func:`accsim.radiation.radiation_integrals`, and the amplitude is scanned so the
    ``t^2`` is a measured power and not a fitted one.
    """
    ring = gate_ring(amplitude)
    n_y, t, phi_0 = _tilt_of_the_arc(ring)
    g = G_E * ring.ref.gamma0
    mean_cos_sq = arc_average_of_cos_squared(phi_0, g)

    # I3 from the geometry of the arc alone: 2 N_CELLS dipoles of length BEND_L.
    i3 = arc_curvature() ** 3 * (2 * N_CELLS * BEND_L)
    integrals = polarization_integrals(ring, slices=64)
    residual = (integrals.alpha_plus + integrals.alpha_minus) * ring.length / i3

    both = t**2 * (0.5 - 2.0 / 9.0 * mean_cos_sq)
    assert residual == pytest.approx(both, rel=2e-6)

    # ... and the two one-legged alternatives are excluded, not merely different.
    # Excluded by a wide margin, not merely by more than the tolerance: dropping the
    # ``2/9`` term overshoots by 28% of the answer, and keeping it alone misses by 128%.
    # A threshold set just above the tolerance would pass with the ``2/9`` coefficient
    # wrong by a factor of three; this one does not.
    only_b = t**2 * 0.5
    only_v = -(t**2) * 2.0 / 9.0 * mean_cos_sq
    assert abs(residual - only_b) > 0.2 * abs(both)
    assert abs(residual - only_v) > 0.2 * abs(both)


def test_the_departure_from_the_textbook_ratio_is_second_order_in_the_tilt():
    r"""``P_inf`` misses ``-8/(5 sqrt3)`` by ``O(t^2)``, and the *order* is the assertion.

    A tilted ``n_0`` cannot depolarize the beam at first order: ``(n_0 . b)`` is a cosine
    of the tilt and ``(n_0 . v)^2`` is a square, so both weights are even in ``t``. J2's
    lesson applies -- gate on the order, not on a tolerance -- because a coefficient that
    is wrong by a factor still produces a curve of the right shape, while a term that is
    wrong in *order* (a stray first-order piece from, say, using ``n_0`` where ``|n_0|``
    was meant) does not.

    The bump amplitude is doubled twice; the tilt ``t`` it produces is linear in it, and
    the departure must go up by four each time. Fitted as a power law, the exponent is
    ``2`` to three decimals.
    """
    amplitudes = [1e-3, 2e-3, 4e-3, 8e-3]
    tilts, departures = [], []
    for amplitude in amplitudes:
        ring = gate_ring(amplitude)
        _, t, _ = _tilt_of_the_arc(ring)
        tilts.append(t)
        departures.append(abs(sokolov_ternov_polarization(ring) + P_ST))

    assert np.allclose(np.diff(np.log(tilts)), math.log(2.0), atol=1e-6)  # t is linear in it
    slope = np.polyfit(np.log(tilts), np.log(departures), 1)[0]
    assert slope == pytest.approx(2.0, abs=2e-3)


# --- the coefficient: what the ratio provably cannot see ------------------------------


def test_the_ev_to_si_bridge_is_an_identity_not_a_coincidence():
    r"""``hbar / m_0 = (hbar c) c / (mc^2)``, shown symbolically and checked in SI.

    The package keeps every constant in eV and metres; the Sokolov-Ternov rate wants an
    ``hbar`` in joule-seconds over a mass in kilograms. That conversion is the one place
    this milestone could lose a factor without any *ratio* noticing, and B2's pre-2019
    charge constant is the same failure mode one axis over.

    So it is done twice. :func:`derive_the_ev_to_si_bridge` substitutes
    ``hbar = (hbar c)/c`` and ``m_0 = (mc^2)/c^2`` in sympy and simplifies, which shows
    the identity; then both sides are evaluated -- one from the package's own
    ``HBAR_C_EV_M`` and electron rest energy, the other from ``scipy.constants``, which
    never passes through eV at all -- and must agree to the accuracy of the CODATA values
    the two carry.
    """
    from accsim.radiation import HBAR_C_EV_M
    from accsim.reference import CLIGHT

    expr = derive_the_ev_to_si_bridge()
    hbar_c, mc2, c = (sp.Symbol(name, positive=True) for name in ("hbar_c", "mc2", "c"))
    assert sp.simplify(expr - hbar_c * c / mc2) == 0

    from_accsim = HBAR_C_EV_M * CLIGHT / ELECTRON_MASS_EV
    from_scipy = sci.hbar / sci.m_e
    assert from_accsim == pytest.approx(from_scipy, rel=1e-9)


def test_the_buildup_time_matches_an_assembly_that_never_touches_electronvolts():
    """``tau_pol`` from the package, against the same rate built from ``scipy.constants``.

    The package's route runs ``r_0`` from the particle's own rest-energy ratio and
    ``hbar/m_0`` through the eV bridge above; the check runs the classical electron
    radius, ``hbar`` and ``m_e`` straight out of CODATA in SI. Only ``alpha_plus`` is
    shared, which is the point: this isolates the *constant* from the *integral*.
    """
    ring = flat_ring()
    alpha_plus = polarization_integrals(ring).alpha_plus
    assert polarization_buildup_time(ring) == pytest.approx(
        independent_buildup_time(alpha_plus, ring.ref.gamma0), rel=1e-8
    )


@pytest.mark.parametrize(
    ("knob", "expected_slope"),
    [("energy", -5.0), ("size", 3.0)],
)
def test_the_buildup_time_carries_the_gamma_fifth_and_rho_cubed_powers(
    knob: str, expected_slope: float
):
    r"""``tau_pol ~ gamma^-5 rho^3`` -- the two powers, measured as slopes.

    These are what scaling *can* catch, and the reason to bother: a wrong power hides
    perfectly inside a single-ring comparison and inside ``P_inf`` entirely. The energy
    scan holds the geometry fixed, so only ``gamma^5`` moves. The size scan stretches
    every length by ``f`` while shrinking ``k1`` as ``1/f^2``, which leaves every transfer
    matrix -- and therefore the optics, the closed orbit and ``n_0`` -- **identical** and
    multiplies the bending radius by ``f``; ``alpha_plus`` then falls as ``f^-3`` purely
    through ``kappa^3``.

    What neither catches is a wrong *factor*: both slopes are exact for a rate that is
    ten times too fast. That is xtrack's job.
    """
    factors = [1.0, 2.0, 4.0]
    if knob == "energy":
        rings = [flat_ring(ref=electron(5e9 * f)) for f in factors]
        x = [ring.ref.gamma0 for ring in rings]
    else:
        rings = [flat_ring(scale=f) for f in factors]
        x = list(factors)

    times = [polarization_buildup_time(ring) for ring in rings]
    slope = np.polyfit(np.log(x), np.log(times), 1)[0]
    assert slope == pytest.approx(expected_slope, abs=1e-9)


def test_a_lep_scale_ring_polarizes_in_about_five_and_a_half_hours():
    r"""The machine-scale sanity number, against a real ring's published one.

    LEP at 45.6 GeV, with an arc bending radius of 3096 m in a 26.66 km circumference,
    has a Sokolov-Ternov buildup time of about 5.5 hours -- the number transverse
    polarization for energy calibration was actually built on. It is a weak *tolerance*
    (this ring shares LEP's geometry and nothing else) and a strong *exponent* check:
    ``gamma^5`` over ``rho^3`` spans nineteen orders of magnitude between this and the
    one-second 5 GeV ring above, so a mis-assembled constant does not land in the right
    hour by accident.

    The lattice is a bare isomagnetic ring with LEP's radius and circumference, not a
    model of LEP: only ``alpha_plus`` and ``gamma`` reach the answer, and both are fixed
    by geometry and energy alone.
    """
    rho, circumference, n_cells = 3096.0, 26658.9, 16
    bend_length = 2.0 * math.pi * rho / (2 * n_cells)
    straight = (circumference - 2.0 * math.pi * rho) / (4 * n_cells)
    half_cell = bend_length + 2.0 * straight
    kl = 2.0 * math.sin(math.radians(60.0) / 2.0) / half_cell  # 60 deg/cell, thin-FODO

    elements: list = []
    for _ in range(n_cells):
        for sign in (+1, -1):
            elements += [
                ThinQuadrupole(sign * kl),
                Drift(straight),
                Dipole(length=bend_length, angle=2.0 * math.pi / (2 * n_cells)),
                Drift(straight),
            ]
    ring = Lattice(elements, electron(45.6e9))
    assert ring.length == pytest.approx(circumference, rel=1e-12)

    hours = polarization_buildup_time(ring) / 3600.0
    assert 5.0 < hours < 6.0


# --- the quadrature, and the refusals -------------------------------------------------


def test_the_quadrature_converges_at_fourth_order():
    r"""Simpson's rule on the ``(n_0 . v)^2`` term, halving the step four bits at a time.

    The sub-slicing here has to resolve the **spin** phase, not the optics:
    ``G gamma theta = 4.4`` radians across one bend of the gate ring where the dispersion
    :func:`accsim.radiation.radiation_integrals` sub-steps turns through ``theta = 0.39``.
    That is why this integral uses Simpson where that one uses the trapezoid, and the
    difference is not cosmetic -- at the shared default of 64 slices the trapezoid is
    still 1.5% short of the ``(n_0 . v)^2`` term while Simpson has reached round-off.

    Measured on the term itself rather than on ``alpha_plus``, which it is one part in
    ``10^8`` of: a convergence test on ``alpha_plus`` would report machine precision at
    every slice count and see nothing. The error must fall by ``16`` per doubling.
    """
    ring = gate_ring(1e-3)
    n_y, t, phi_0 = _tilt_of_the_arc(ring)
    i3 = arc_curvature() ** 3 * (2 * N_CELLS * BEND_L)
    exact = -2.0 / 9.0 * t**2 * arc_average_of_cos_squared(phi_0, G_E * ring.ref.gamma0)

    errors = []
    for slices in (8, 16, 32):
        integrals = polarization_integrals(ring, slices=slices)
        term = integrals.alpha_plus * ring.length / i3 - 1.0
        errors.append(abs(term / exact - 1.0))

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(ratio > 10.0 for ratio in ratios), ratios  # fourth order, not second


def test_a_ring_that_does_not_bend_cannot_polarize():
    r"""No bending, no radiation, no spin flips -- and so no polarization to report.

    ``alpha_plus`` is exactly zero rather than nearly zero, so this is a refusal and not a
    tolerance: reporting ``0/0`` as ``8/(5 sqrt3)`` would claim a physical equilibrium
    that a machine with no bending magnets does not have. Both entry points refuse.

    **Reaching that refusal takes more care than it looks.** The obvious lattice -- drifts
    and on-axis quadrupoles -- never gets to it: with no field anywhere on the orbit
    nothing precesses, the one-turn spin rotation is the identity, and N2's
    :class:`accsim.spin.SpinSolutionError` fires first, because the integrals are weighted
    by an ``n_0`` that does not exist. The guard is live only where the two conditions
    come apart, and exactly one construction does that: a **quadrupole traversed
    off-axis**. Its field on the orbit is real, so the spin precesses and ``n_0`` is
    unique; it is not a dipole, so nothing in this milestone's scope radiates. Both
    branches are asserted here, because "the code raises" is worth much less than "the
    code raises *this*".
    """
    from accsim.spin import SpinSolutionError

    field_free = Lattice(
        [Drift(1.0), Quadrupole(0.4, 1.2), Drift(1.0), Quadrupole(0.4, -1.2)], electron()
    )
    with pytest.raises(SpinSolutionError, match="integer"):
        polarization_integrals(field_free)

    steered = Lattice(
        [
            Corrector(kick_x=1e-3),
            Drift(1.0),
            Quadrupole(0.4, 1.2),
            Drift(1.0),
            Quadrupole(0.4, -1.2),
            Drift(1.0),
        ],
        electron(),
    )
    assert polarization_integrals(steered).alpha_plus == 0.0  # nothing here is a dipole
    with pytest.raises(ValueError, match="no bending"):
        sokolov_ternov_polarization(steered)
    with pytest.raises(ValueError, match="no bending"):
        polarization_buildup_time(steered)


def test_an_integer_spin_tune_refuses_here_too():
    """The polarization inherits N2's degeneracy: no ``n_0``, no ``(n_0 . b)`` to weight by.

    Both integrals are weighted by the periodic spin direction, so a ring that has no
    unique one has no polarization integrals either. The failure surfaces as N2's own
    :class:`accsim.spin.SpinSolutionError` rather than being caught and turned into
    something vaguer -- the caller needs to know it is the *spin* solution that does not
    exist, not the integral that failed to converge.
    """
    from accsim.spin import SpinSolutionError

    ref = electron(11.0 / G_E * ELECTRON_MASS_EV)
    with pytest.raises(SpinSolutionError, match="integer"):
        polarization_integrals(flat_ring(ref=ref))
