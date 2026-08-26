r"""N2 -- the closed spin solution ``n_0`` and the spin tune ``nu_0``, gated against closed forms.

The milestone's own headline quantities are **degenerate on every ring the package could
casually build**. On a flat, unsteered lattice every field a spin meets is vertical, every
rotation is about ``y``, and ``n_0`` comes back exactly ``(0, 1, 0)`` -- bit for bit, at any
energy, for any quadrupole strength, and for a precession coefficient scaled by seven. That
is asserted here (:func:`test_an_unsteered_ring_returns_vertical_whatever_the_map_says`)
rather than hoped against; it is N1's "the spin tune is a control" arriving one milestone
later in a second quantity, and M3's degeneracy in a third guise.

What carries the milestone is a ring built to break it: a **closed vertical bump**, holding
exactly one thick quadrupole, inside a bend-free straight. Two facts make that construction
the only honest one, and both are asserted:

- a bend with a vertical *angle* precesses the spin about ``z`` at rate
  ``h i_y G (gamma - 1)`` -- first order in ``py``, and comparable in size to the
  quadrupole kick this gate is about. So a vertical orbit that leaks into the arc is a
  second, distributed, uncontrolled driving term; the bump must actually close, and
  :func:`test_the_bump_closes_so_the_arc_stays_on_the_design_orbit` checks it does, at
  round-off, element by element.
- thin elements do not precess (N1's stated scope), so the arc's focusing can be thin and
  contribute nothing. The one thick quadrupole is then the entire spin perturbation, and
  the ring reduces to a single localized rotation ``chi`` about ``x`` composed with a
  uniform ``-2 pi nu_0`` about ``y``.

For that ring the closed form -- derived with sympy in
:func:`derive_tilt_of_the_closed_solution`, not recalled -- is

    ``n_0 = ( -(chi/2) cot(pi nu_0),  1,  -chi/2 )``   to first order in ``chi``,

and its two components are two independent gates. The ``z`` component has **no** resonance
denominator, so it measures ``chi`` itself and through it the ``(1 + G gamma)`` factor; the
``x`` component carries ``cot(pi nu_0)``, which diverges at every **integer** spin tune and
nowhere else. Scanning the beam energy therefore predicts an *integer-indexed family of
locations* separably from any coefficient, which is J2's "gate on the order, not a
tolerance" in another guise -- and it is where a polarized ring's energy calibration comes
from.

The resonance is at an **integer**, not at ``nu_0 = k +- Q_y``. ``n_0`` lives on the closed
orbit, so the perturbation it sees is one-turn periodic and has only integer harmonics;
``k +- Q_y`` is a statement about the invariant spin field of a particle with vertical
betatron *amplitude*, which is a different object and is not built here (see
``docs/ROADMAP.md`` under N3).
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
import sympy as sp

from accsim.coords import PY, X, Y
from accsim.elements.corrector import Corrector
from accsim.elements.dipole import Dipole
from accsim.elements.drift import Drift
from accsim.elements.quadrupole import Quadrupole, ThinQuadrupole
from accsim.lattice import Lattice
from accsim.orbit import closed_orbit, closed_orbit_nonlinear, propagate_orbit_nonlinear
from accsim.reference import (
    ELECTRON_ANOMALOUS_MOMENT,
    ELECTRON_MASS_EV,
    ReferenceParticle,
)
from accsim.spin import (
    ClosedSpinSolution,
    SpinSolutionError,
    closed_spin_solution,
    propagate_spin_solution,
    spin_axis_and_tune,
    spin_one_turn_matrix,
    spin_tune,
)
from accsim.tracking import Tracker

G_E = ELECTRON_ANOMALOUS_MOMENT

# The gate ring. The straight comes **first** so that the observation point is the ring
# entrance and the one thick quadrupole's kick sits at the top of the turn -- which is the
# composition order the closed form below is derived for.
N_CELLS = 8  # FODO cells in the arc; two bends each, summing to 2 pi
KL = 0.20  # thin-quadrupole strength [1/m] -- thin, so it does not precess
K1 = 0.6  # the one thick quadrupole, which is the whole spin perturbation
LQ = 0.4  # its length [m]
SLICES = 8  # slices it is cut into (the midpoint rule's quadrature)
DRIFT = 0.6  # drift between the bump's correctors [m]


def electron(energy_eV: float = 5e9, g: float | None = G_E) -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(
        ELECTRON_MASS_EV, energy_eV, charge=-1.0, anomalous_moment=g
    )


def _closure_pattern(elements: list, ref: ReferenceParticle) -> np.ndarray:
    """Corrector strengths that leave ``(y, py) = 0`` at the end of ``elements``.

    Three correctors, two closure conditions, so the answer is a one-dimensional null
    space and the free parameter is the bump amplitude. Solved from the elements' own
    vertical transfer matrices rather than from a drift-space formula, because the bump
    has a quadrupole in the middle of it.
    """
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


def straight_section(amplitude: float, ref: ReferenceParticle) -> list:
    """A bend-free straight holding a closed vertical bump around one thick quadrupole."""
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

    return build(*(amplitude * _closure_pattern(build(0.0, 0.0, 0.0), ref)))


def gate_ring(amplitude: float, ref: ReferenceParticle) -> Lattice:
    """The straight, then a thin-lens FODO arc whose bends sum to ``2 pi``."""
    elements = straight_section(amplitude, ref)
    for _ in range(N_CELLS):
        for sign in (+1, -1):
            elements += [
                ThinQuadrupole(sign * KL),
                Drift(0.5),
                Dipole(length=2.0, angle=math.pi / N_CELLS),
                Drift(0.5),
            ]
    return Lattice(elements, ref)


#: Index of the thick quadrupole's first slice inside the straight: it is preceded by
#: Drift, Corrector, Drift, Corrector, Drift, whatever the slicing is.
QUAD_ENTRANCE = 5


def n_straight() -> int:
    """Elements in the straight -- five before the quadrupole, three after, plus its slices."""
    return 8 + SLICES


def flat_ring(k1: float, ref: ReferenceParticle, n_cells: int = 8) -> Lattice:
    """A closed, flat, *unsteered* ring -- the degenerate case, kept as the control."""
    elements: list = []
    for _ in range(n_cells):
        elements += [
            Dipole(length=2.0, angle=2.0 * math.pi / n_cells),
            Drift(0.5),
            Quadrupole(0.4, k1),
            Drift(0.5),
            Quadrupole(0.4, -k1),
        ]
    return Lattice(elements, ref)


# --- the closed forms, derived rather than recalled ---------------------------------


@cache
def derive_tilt_of_the_closed_solution() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    r"""``(n_0.x/chi, n_0.z/chi, dnu_0/chi^2)`` for ``R = R_y(-2 pi nu) R_x(chi)``.

    The whole gate ring is that one matrix: a lone rotation ``chi`` about ``x`` from the
    thick quadrupole at a vertical offset, then the arc's uniform ``-2 pi nu_0`` about
    ``y``. Every other rotation in the ring is about ``y`` and commutes with it, so they
    lump exactly. The fixed axis is read off the antisymmetric part, oriented upward, and
    expanded to first order in ``chi``; the tune comes from the trace.
    """
    nu, chi = sp.symbols("nu chi", real=True)

    def rot_y(a: sp.Expr) -> sp.Matrix:
        c, s = sp.cos(a), sp.sin(a)
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rot_x(a: sp.Expr) -> sp.Matrix:
        c, s = sp.cos(a), sp.sin(a)
        return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])

    R = rot_y(-2 * sp.pi * nu) * rot_x(chi)

    axis = sp.Matrix([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    axis = axis / sp.sqrt((axis.T * axis)[0])
    # orient upward: the raw axis has n_y = -sign(sin 2 pi nu)
    axis = -sp.sign(sp.sin(2 * sp.pi * nu)) * axis
    first_order = [sp.simplify(sp.diff(component, chi).subs(chi, 0)) for component in axis]

    # the tune, from the trace: cos(2 pi nu') = cos(2 pi nu) - chi^2 cos^2(pi nu)/2 + ...
    cos_turn = sp.simplify((sp.trace(R) - 1) / 2)
    shift = sp.series(cos_turn, chi, 0, 3).removeO().coeff(chi, 2)
    # d(cos 2 pi nu)/d nu = -2 pi sin(2 pi nu)  =>  d nu = shift * chi^2 / (-2 pi sin 2 pi nu)
    d_nu = sp.simplify(shift / (-2 * sp.pi * sp.sin(2 * sp.pi * nu)))
    return sp.simplify(first_order[0]), sp.simplify(first_order[2]), d_nu


def tilt_coefficients(nu0: float) -> tuple[float, float, float]:
    """The derived coefficients evaluated at ``nu_0`` -- ``(n_x/chi, n_z/chi, dnu/chi^2)``."""
    nu = sp.symbols("nu", real=True)
    return tuple(float(expr.subs(nu, nu0)) for expr in derive_tilt_of_the_closed_solution())


@cache
def _integral_of_y_symbols() -> tuple[sp.Expr, sp.Expr]:
    r"""``int_0^L y ds`` for ``y'' = +k y``, as the coefficients of ``y0`` and ``py0``."""
    s, k, ell = sp.symbols("s k L", positive=True)
    y0, py0 = sp.symbols("y0 py0", real=True)
    y = y0 * sp.cosh(sp.sqrt(k) * s) + py0 * sp.sinh(sp.sqrt(k) * s) / sp.sqrt(k)
    integral = sp.integrate(y, (s, 0, ell))
    return sp.simplify(integral.coeff(y0)), sp.simplify(integral.coeff(py0))


def integral_of_y(k1: float, length: float, y0: float, py0: float) -> float:
    """``int y ds`` through a quadrupole entered at ``(y0, py0)``, from the sympy solution."""
    k, ell = sp.symbols("k L", positive=True)
    subs = {k: k1, ell: length}
    a, b = (float(expr.subs(subs)) for expr in _integral_of_y_symbols())
    return a * y0 + b * py0


# --- the degeneracy, asserted before anything is built on it ------------------------


def test_an_unsteered_ring_returns_vertical_whatever_the_map_says(monkeypatch):
    """``n_0 = (0, 1, 0)`` **bit for bit**, and stays there under a broken precession.

    The point of the milestone's gate ring, stated as its opposite. Three separate
    mutilations of the map -- a five-fold change of every quadrupole, a seven-fold
    mis-scale of the quadrupole *field* the precession reads, and a sign flip of the
    whole precession vector -- leave ``n_0`` exactly vertical. Not to a tolerance:
    exactly, because on the design orbit a quadrupole's field is identically zero and
    every remaining rotation is about ``y``, so the ``y`` axis is a fixed point of each
    factor separately.
    """
    ref = electron()
    vertical = np.array([0.0, 1.0, 0.0])
    assert np.array_equal(closed_spin_solution(flat_ring(1.2, ref)).n0, vertical)
    assert np.array_equal(closed_spin_solution(flat_ring(6.0, ref)).n0, vertical)

    original = Quadrupole.normalized_field
    monkeypatch.setattr(
        Quadrupole,
        "normalized_field",
        lambda self, x, y: tuple(7.0 * np.asarray(c) for c in original(self, x, y)),
    )
    assert np.array_equal(closed_spin_solution(flat_ring(1.2, ref)).n0, vertical)

    import accsim.spin as spin_module

    unflipped = spin_module.precession_vector
    monkeypatch.setattr(spin_module, "precession_vector", lambda *a, **k: -unflipped(*a, **k))
    assert np.array_equal(closed_spin_solution(flat_ring(1.2, ref)).n0, vertical)


@pytest.mark.parametrize("energy_eV", [1e9, 5e9, 45.6e9])
def test_the_unsteered_spin_tune_is_g_gamma_and_agrees_with_n1s_route(energy_eV):
    """``nu_0 = G gamma mod 1``, by two independent implementations of the same number.

    N1 read the tune off the *angle a single spin turned through*
    (``atan2`` on a tracked ``x`` vector); N2 reads it off the eigen-structure of the
    ``3x3``. Same ring, same number, and they must agree to round-off -- which is worth
    more than either agreeing with ``G gamma`` alone, since that is a control both are
    blind in.
    """
    ref = electron(energy_eV)
    lattice = flat_ring(1.2, ref)

    _, spin = Tracker(lattice).track_once_with_spin(np.zeros(6), np.array([1.0, 0.0, 0.0]))
    n1_route = -math.atan2(-spin[2], spin[0]) / (2.0 * math.pi) % 1.0

    assert spin_tune(lattice) == pytest.approx(n1_route, abs=1e-14)
    assert spin_tune(lattice) == pytest.approx((G_E * ref.gamma0) % 1.0, abs=1e-9)


def test_an_integer_spin_tune_has_no_closed_solution():
    """At ``G gamma`` integer the one-turn rotation is the identity and ``n_0`` is undefined.

    The spin twin of :class:`accsim.orbit.ClosedOrbitError` on an integer betatron tune,
    and the same statement: the fixed point stops being unique, so returning one would
    claim a uniqueness the map does not have. The energy is chosen to put ``G gamma``
    on ``11`` exactly.
    """
    ref = electron(11.0 / G_E * ELECTRON_MASS_EV)
    assert G_E * ref.gamma0 == pytest.approx(11.0, abs=1e-9)
    with pytest.raises(SpinSolutionError, match="integer"):
        closed_spin_solution(flat_ring(1.2, ref))


# --- the gate ring: its premise, checked before its conclusion ----------------------


def test_the_bump_closes_so_the_arc_stays_on_the_design_orbit():
    r"""The vertical orbit lives inside the straight, and what leaks past it cannot matter.

    This is the gate's premise, and it is not decoration. A bend traversed with a vertical
    *angle* precesses the spin about ``z`` at ``h i_y G (gamma - 1)`` -- first order in
    ``py``, scaling with ``G gamma``, and of the same size as the quadrupole kick the gate
    is built to measure (:func:`test_the_bend_really_would_precess_on_a_vertical_angle`).
    Vertical orbit leaking into the arc would therefore be a second, distributed,
    uncontrolled driving term, and the closed form would still fit -- with a wrong
    coefficient. So the leak is bounded here rather than assumed away.

    It is not exactly zero, and the reason is worth naming: the bump is closed using the
    elements' **matrices**, while the orbit is the fixed point of their exact **tracked**
    maps, and axis L's exact maps depart from their own Jacobians at third order in the
    excursion. So the leak is asserted at that order (a factor 8 per doubling), which is
    what says it is the exact map and not a broken closure, and the arc's whole spin
    driving that follows from it -- ``2 pi`` of bending at the leaked angle -- is bounded
    against ``chi``. That bound grows as the *square* of the bump (leak cubic over kick
    linear) and reaches ``5e-5`` at the largest amplitude any gate in this file uses,
    which is below every tolerance any of them assert.
    """
    ref = electron()
    leaks, driving = [], []
    for amplitude in (1e-3, 2e-3, 4e-3):
        lattice = gate_ring(amplitude, ref)
        orbit = np.array(propagate_orbit_nonlinear(lattice))
        arc = orbit[n_straight() :]

        assert np.abs(orbit[:, Y]).max() > 0.2 * amplitude  # there really is a bump
        # not exactly zero: axis L's exact maps couple the planes on a vertical orbit --
        # but thirteen orders below the bump, so the ring is horizontally on the design orbit
        assert np.abs(orbit[:, X]).max() < 1e-16
        leaks.append(float(np.abs(arc[:, [Y, PY]]).max()))

        # 2 pi of bending, every bit of it at the leaked vertical angle
        drive = 2.0 * math.pi * float(np.abs(arc[:, PY]).max()) * G_E * (ref.gamma0 - 1.0)
        driving.append(abs(drive / kick_of(amplitude, ref)))
        assert driving[-1] < 1e-4

    for small, large in zip(leaks, leaks[1:], strict=False):
        assert large / small == pytest.approx(8.0, rel=0.05)
    for small, large in zip(driving, driving[1:], strict=False):
        assert large / small == pytest.approx(4.0, rel=0.05)


def test_the_closed_orbit_does_not_depend_on_the_beam_energy():
    """Which is what lets one bump serve a whole energy scan, and is not obvious.

    accsim's magnets are specified by *normalised* strengths -- ``k1`` and a bend angle,
    not a field -- so the transfer maps at ``delta = 0`` carry no energy at all and the
    closed orbit is the same at every energy the resonance scan visits. Only ``G gamma``
    moves. Asserted bit for bit, because it is the assumption behind reusing one orbit and
    one ``int y ds`` across the sweep.
    """
    reference = closed_orbit_nonlinear(gate_ring(2e-3, electron(5e9)))
    for energy in (1e9, 4.8e9, 45.6e9):
        assert np.array_equal(closed_orbit_nonlinear(gate_ring(2e-3, electron(energy))), reference)


def test_the_bend_really_would_precess_on_a_vertical_angle():
    """The reason the bump has to close, measured: a bend tilts a spin at first order in ``py``.

    ``Omega_z = h i_y i_z G (gamma - 1)`` -- the difference between the ``(1 + G gamma)``
    perpendicular coefficient and the ``(1 + G)`` parallel one, which is what survives when
    a vertical angle gives a horizontal-field magnet a component along the motion. Asserted
    as its order in ``py`` (a factor 2 per doubling) rather than as a size, and against the
    cone the composition predicts.
    """
    ref = electron()
    bend = Dipole(length=2.0, angle=0.3)
    tilts = []
    for py in (1e-3, 2e-3, 4e-3):
        state = np.array([0.0, 0.0, 0.0, py, 0.0, 0.0])
        _, spin = bend.track_with_spin(state, np.array([0.0, 1.0, 0.0]), ref)
        tilts.append(math.hypot(spin[0], spin[2]))
    for small, large in zip(tilts, tilts[1:], strict=False):
        assert large / small == pytest.approx(2.0, rel=1e-3)

    # the cone: axis tilt Omega_z/Omega_y, opened by the BMT rotation angle |Omega| L
    g, gamma, h = G_E, ref.gamma0, 0.3 / 2.0
    omega_y, omega_z = -(1.0 + g * gamma) * h, h * 1e-3 * g * (gamma - 1.0)
    assert tilts[0] == pytest.approx(
        2.0 * abs(omega_z / omega_y) * abs(math.sin(abs(omega_y) * 2.0 / 2.0)), rel=2e-3
    )


# --- the two components of the tilt, which are two different gates ------------------


def solution_of(amplitude: float, ref: ReferenceParticle, orbit: np.ndarray) -> ClosedSpinSolution:
    return closed_spin_solution(gate_ring(amplitude, ref), orbit)


@cache
def _quadrupole_entrance(amplitude: float, _slices: int) -> tuple[float, float]:
    """``(y, py)`` of the closed orbit at the thick quadrupole's entrance.

    Cached on the bump amplitude and the slicing alone: the closed orbit carries no energy
    (:func:`test_the_closed_orbit_does_not_depend_on_the_beam_energy`), which is what makes
    one Newton solve serve the whole resonance scan.
    """
    orbit = propagate_orbit_nonlinear(gate_ring(amplitude, electron()))
    entrance = orbit[QUAD_ENTRANCE]
    return float(entrance[Y]), float(entrance[PY])


def kick_of(amplitude: float, ref: ReferenceParticle) -> float:
    r"""``chi = -(1 + G gamma) k1 int y ds``: the ring's whole spin perturbation, predicted.

    ``int y ds`` comes from the sympy solution of ``y'' = +k1 y`` started at the closed
    orbit's ``(y, py)`` at the quadrupole's entrance -- so nothing about the tracked spin
    enters, and the factor under test is the ``(1 + G gamma)`` in front of it.
    """
    y0, py0 = _quadrupole_entrance(amplitude, SLICES)
    return -(1.0 + G_E * ref.gamma0) * K1 * integral_of_y(K1, LQ, y0, py0)


def test_the_z_component_measures_the_kick_with_no_resonance_denominator():
    r"""``n_0 . z = -chi/2``, with ``chi`` predicted from ``(1 + G gamma) k1 int y ds``.

    The half of the closed form that carries **no** ``cot(pi nu_0)``: whatever the spin
    tune is doing, this component is the kick itself, divided by two. It is therefore the
    component that pins the ``(1 + G gamma)`` factor inside a *ring* -- N1 pinned it in a
    single element -- and the three candidate readings of that factor are an order of
    magnitude apart, so no tolerance is needed to tell them apart.

    The residual is the midpoint rule's quadrature of ``int y ds`` over the quadrupole's
    slices, so it is asserted as second order in the slice length rather than as a size.
    """
    ref = electron()
    amplitude = 2e-3
    chi = kick_of(amplitude, ref)

    # the three readings this gate separates, exactly as N1's single-element version
    assert abs(chi / (chi / (1.0 + G_E * ref.gamma0) * G_E * ref.gamma0)) > 1.08
    assert abs(chi / (chi / (1.0 + G_E * ref.gamma0) * (1.0 + G_E))) > 10.0

    n0 = solution_of(amplitude, ref, closed_orbit_nonlinear(gate_ring(amplitude, ref))).n0
    _, per_chi, _ = tilt_coefficients((G_E * ref.gamma0) % 1.0)
    assert per_chi == -0.5  # derived, not assumed: no resonance denominator at all
    assert n0[2] == pytest.approx(per_chi * chi, rel=2e-4)


def test_the_z_component_converges_at_second_order_in_the_slice_length():
    """The residual of the gate above is the quadrature, and it is asserted as its order.

    Slicing the one thick quadrupole finer is what turns the midpoint rule into a
    convergent integral of ``y ds``; a mis-scaled coefficient would leave a residual that
    does not converge at all. Same shape as N1's single-element version, now inside a ring
    whose closed orbit has to be re-solved at every slicing.
    """
    global SLICES
    ref = electron()
    amplitude = 2e-3
    original = SLICES
    residuals = []
    try:
        for slices in (2, 4, 8, 16):
            SLICES = slices
            lattice = gate_ring(amplitude, ref)
            n0 = closed_spin_solution(lattice, closed_orbit_nonlinear(lattice)).n0
            residuals.append(abs(n0[2] + 0.5 * kick_of(amplitude, ref)))
    finally:
        SLICES = original

    for coarse, fine in zip(residuals, residuals[1:], strict=False):
        assert coarse / fine == pytest.approx(4.0, rel=0.05)


@pytest.mark.parametrize("energy_eV", [4.4e9, 5e9, 6.2e9, 7.7e9])
def test_the_tilt_direction_alone_measures_the_spin_tune(energy_eV):
    r"""``n_0.x / n_0.z = cot(pi nu_0)`` -- the ratio in which ``chi`` cancels completely.

    The other half of the closed form. Because the strength of the perturbation divides
    out, the *direction* the closed solution leans in is a measurement of the spin tune
    with nothing about the ring's imperfection left in it -- and it is the quantity that
    diverges on resonance. Compared against ``G gamma`` at four energies, i.e. four
    different spin tunes on one unchanged lattice (the optics do not move with energy,
    because ``k1`` and the bend angles are normalised).
    """
    ref = electron(energy_eV)
    amplitude = 1e-3
    lattice = gate_ring(amplitude, ref)
    n0 = closed_spin_solution(lattice, closed_orbit_nonlinear(lattice)).n0

    nu0 = (G_E * ref.gamma0) % 1.0
    per_chi_x, per_chi_z, _ = tilt_coefficients(nu0)
    assert n0[0] / n0[2] == pytest.approx(per_chi_x / per_chi_z, rel=1e-5)
    assert n0[0] / n0[2] == pytest.approx(1.0 / math.tan(math.pi * nu0), rel=1e-5)


def test_the_tilt_is_first_order_in_the_steering():
    """Halve the bump and both components of the tilt halve -- ratio 2, not 4, not 1.

    The degeneracy asserted as an exponent. At zero steering ``n_0`` is exactly vertical,
    so the only thing a ring can gate is the *order* in which it leaves vertical, which is
    what M3 concluded one axis earlier about second-order dispersion.
    """
    ref = electron()
    tilts = []
    for amplitude in (5e-4, 1e-3, 2e-3, 4e-3):
        lattice = gate_ring(amplitude, ref)
        n0 = closed_spin_solution(lattice, closed_orbit_nonlinear(lattice)).n0
        tilts.append((abs(n0[0]), abs(n0[2])))
    for (small_x, small_z), (large_x, large_z) in zip(tilts, tilts[1:], strict=False):
        assert large_x / small_x == pytest.approx(2.0, rel=1e-3)
        assert large_z / small_z == pytest.approx(2.0, rel=1e-3)


def test_the_spin_tune_is_unmoved_at_first_order_in_the_steering():
    r"""``nu_0`` shifts by ``chi^2 cot(pi nu_0) / (8 pi)`` -- second order, so blind at first.

    N1's "the spin tune is a control" arriving in N2: the number that gets quoted is
    precisely the one the imperfection cannot move, while the tilt beside it moves
    linearly. Gated as the exponent (a factor 4 per doubling) *and* as the derived
    coefficient, since the exponent alone would survive a mis-scale.
    """
    ref = electron()
    nu0 = (G_E * ref.gamma0) % 1.0
    _, _, per_chi2 = tilt_coefficients(nu0)
    assert per_chi2 == pytest.approx(1.0 / math.tan(math.pi * nu0) / (8.0 * math.pi), rel=1e-12)

    shifts = []
    for amplitude in (1e-3, 2e-3, 4e-3):
        lattice = gate_ring(amplitude, ref)
        solution = closed_spin_solution(lattice, closed_orbit_nonlinear(lattice))
        shifts.append((solution.spin_tune - nu0, kick_of(amplitude, ref)))

    for (small, _), (large, _) in zip(shifts, shifts[1:], strict=False):
        assert large / small == pytest.approx(4.0, rel=1e-2)
    for shift, chi in shifts:
        assert shift == pytest.approx(per_chi2 * chi * chi, rel=5e-3)


# --- the resonance: an integer-indexed family of locations --------------------------


def test_the_tilt_resonates_at_every_integer_spin_tune_and_the_kick_does_not():
    r"""Scan the beam energy: ``|n_0.x|`` peaks exactly where ``G gamma`` is an integer.

    The discriminating gate of the milestone, and it is a statement about *locations*, not
    about a size. ``nu_0 = G gamma`` on a flat ring, so sweeping the energy sweeps the spin
    tune; ``cot(pi nu_0)`` diverges at every integer and only there. The lattice, the bump
    and the optical tunes are held fixed throughout -- accsim's maps are normalised, so the
    orbital tunes do not move with energy at all -- which is what makes an integer-indexed
    family of *spin* tunes the only thing the scan can be reading.

    The companion claim is what makes it discriminating rather than decorative. Divided by
    the kick ``chi`` that drives it, the ``z`` component is ``-1/2`` across the entire
    sweep, to four digits, resonance crossings included; the ``x`` component over the same
    sweep spans more than two orders of magnitude. Two components of one vector, one with a
    denominator and one without.
    """
    amplitude = 1e-3
    orbit = closed_orbit_nonlinear(gate_ring(amplitude, electron()))
    energies = np.linspace(4.80e9, 5.35e9, 111)

    tune, per_chi_x, per_chi_z = [], [], []
    for energy in energies:
        ref = electron(float(energy))
        n0 = closed_spin_solution(gate_ring(amplitude, ref), orbit).n0
        chi = kick_of(amplitude, ref)
        tune.append(G_E * ref.gamma0)
        per_chi_x.append(abs(n0[0] / chi))
        per_chi_z.append(n0[2] / chi)
    tune = np.array(tune)
    per_chi_x, per_chi_z = np.array(per_chi_x), np.array(per_chi_z)

    assert tune[0] < 11.0 and tune[-1] > 12.0  # the sweep really crosses two integers
    peaks = [
        i
        for i in range(1, len(tune) - 1)
        if per_chi_x[i] > per_chi_x[i - 1] and per_chi_x[i] > per_chi_x[i + 1]
    ]
    spacing = float(np.diff(tune).mean())
    assert [round(tune[i]) for i in peaks] == [11, 12]
    for i in peaks:
        assert abs(tune[i] - round(tune[i])) < spacing

    assert per_chi_x.max() / per_chi_x.min() > 200.0  # the resonant component
    assert np.abs(per_chi_z + 0.5).max() < 2e-4  # the one with no denominator, at all


def test_the_resonance_is_the_1_over_sin_law_and_not_merely_a_peak():
    r"""Off resonance the tilt *is* ``chi/(2 |sin(pi nu_0)|)``, at every energy of the scan.

    A peak at the right place would also come out of a code with the wrong coefficient in
    front of it. This asserts the whole law: at eight energies spread across a resonance
    crossing, the measured tilt magnitude equals the derived one from the independently
    predicted ``chi``, with a single tolerance and no fit.
    """
    amplitude = 1e-3
    orbit = closed_orbit_nonlinear(gate_ring(amplitude, electron()))
    for energy in np.linspace(4.90e9, 5.24e9, 8):
        ref = electron(float(energy))
        n0 = closed_spin_solution(gate_ring(amplitude, ref), orbit).n0
        chi = kick_of(amplitude, ref)
        nu0 = (G_E * ref.gamma0) % 1.0
        assert math.hypot(n0[0], n0[2]) == pytest.approx(
            abs(chi) / (2.0 * abs(math.sin(math.pi * nu0))), rel=2e-3
        )


# --- what the solution is, structurally ---------------------------------------------


def test_a_spin_started_along_n0_comes_back_to_itself():
    """The definition, checked by tracking rather than by the algebra that produced it."""
    ref = electron()
    lattice = gate_ring(2e-3, ref)
    orbit = closed_orbit_nonlinear(lattice)
    solution = closed_spin_solution(lattice, orbit)

    state = np.zeros(6)
    state[[0, 1, 2, 3]] = orbit
    _, spin = Tracker(lattice).track_once_with_spin(state, solution.n0.copy())
    assert np.abs(spin - solution.n0).max() < 1e-14


def test_propagate_returns_a_closed_chain_of_unit_vectors():
    """``len(lattice) + 1`` points, all unit, and the last is the first."""
    ref = electron()
    lattice = gate_ring(2e-3, ref)
    points = propagate_spin_solution(lattice)

    assert len(points) == len(lattice.elements) + 1
    assert np.abs([np.linalg.norm(p) - 1.0 for p in points]).max() < 1e-14
    assert np.abs(points[-1] - points[0]).max() < 1e-14


def test_a_thin_lens_ring_never_moves_the_spin_at_all():
    """Thin elements do not precess, so every point of the chain is the same vector.

    N1's stated omission, restated as the property N2 relies on: the gate ring's arc is
    thin-lens *on purpose*, so that the one thick quadrupole is the entire perturbation.
    Bit for bit, not to a tolerance.
    """
    ref = electron()
    lattice = Lattice([ThinQuadrupole(0.2), Drift(1.0), ThinQuadrupole(-0.2), Drift(1.0)] * 4, ref)
    points = propagate_spin_solution(lattice, np.array([0.3, 0.6, -0.2]))
    assert all(np.array_equal(p, points[0]) for p in points)


def test_the_solution_needs_the_tracked_closed_orbit_not_the_linear_one():
    r"""The spin rides ``track()``, so its orbit must be ``track()``'s fixed point.

    The default is the expensive one for a reason. On the *linear* closed orbit the
    trajectory does not quite close -- the exact maps of axis L differ from their Jacobians
    at third order in the excursion -- so the "one-turn" spin rotation is really a rotation
    between two different points. Asserted as that exponent: the linear orbit's one-turn
    residual grows by a factor 8 per doubling of the bump, while the tracked orbit's stays
    at round-off.
    """
    ref = electron()
    linear, tracked = [], []
    for amplitude in (1e-3, 2e-3, 4e-3):
        lattice = gate_ring(amplitude, ref)
        for orbit, into in (
            (closed_orbit(lattice), linear),
            (closed_orbit_nonlinear(lattice), tracked),
        ):
            state = np.zeros(6)
            state[[0, 1, 2, 3]] = orbit
            out, _ = Tracker(lattice).track_once_with_spin(state, np.array([0.0, 1.0, 0.0]))
            into.append(np.abs(out[[0, 1, 2, 3]] - orbit).max())

    for small, large in zip(linear, linear[1:], strict=False):
        assert large / small == pytest.approx(8.0, rel=0.05)
    assert max(tracked) < 1e-15


def test_the_one_turn_matrix_is_orthogonal_and_that_is_not_a_gate(monkeypatch):
    """A product of rotations is orthogonal whatever field it was built from.

    Kept, and labelled: it catches a broken Rodrigues formula and nothing else. The proof
    that it is blind is that it survives multiplying every quadrupole field by seven.
    """
    ref = electron()
    lattice = gate_ring(2e-3, ref)
    assert (
        np.abs(spin_one_turn_matrix(lattice).T @ spin_one_turn_matrix(lattice) - np.eye(3)).max()
        < 1e-14
    )

    original = Quadrupole.normalized_field
    monkeypatch.setattr(
        Quadrupole,
        "normalized_field",
        lambda self, x, y: tuple(7.0 * np.asarray(c) for c in original(self, x, y)),
    )
    matrix = spin_one_turn_matrix(lattice)
    assert np.abs(matrix.T @ matrix - np.eye(3)).max() < 1e-14


def test_n0_is_oriented_upward_and_the_tune_is_a_fraction():
    """The two sign conventions, checked where they could silently flip.

    ``n_0 . y > 0`` matches xtrack (whose fixed-point search can only return an upward
    solution), and ``nu_0`` is the fraction in ``[0, 1)`` defined by ``R = R(n_0, -2 pi
    nu_0)`` -- the sign that makes a flat ring's answer ``+G gamma`` rather than
    ``-G gamma``. Both are read off a hand-built rotation, so the test does not depend on
    any lattice.
    """
    for nu in (0.1, 0.4, 0.6, 0.9):
        c, s = math.cos(-2.0 * math.pi * nu), math.sin(-2.0 * math.pi * nu)
        n0, tune = spin_axis_and_tune(np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]))
        assert n0[1] > 0.0
        assert tune == pytest.approx(nu, abs=1e-14)

    # a rotation about -y is the same rotation seen the other way: n_0 flips up, nu -> 1-nu
    c, s = math.cos(0.8 * math.pi), math.sin(0.8 * math.pi)
    n0, tune = spin_axis_and_tune(np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]))
    assert n0[1] > 0.0
    assert tune == pytest.approx(1.0 - 0.4, abs=1e-14)
