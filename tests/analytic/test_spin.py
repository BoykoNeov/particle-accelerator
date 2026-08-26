r"""N1 — the Thomas-BMT spin precession, gated against closed forms.

The headline number of this axis, the spin tune ``nu_0 = G gamma``, is a **control**
and not a gate: it depends only on the ring's bends summing to ``2 pi`` and on the beam
energy, so an implementation whose transverse coefficient is mis-scaled — or whose
quadrupole contribution is missing altogether — reproduces it exactly. That blindness is
asserted here (:func:`test_the_spin_tune_is_blind_to_every_quadrupole`) rather than
hoped against, which is J1's lesson applied to a rotation.

Three gates carry the milestone instead:

- **the Dirac identity**: with ``G = 0`` the BMT rotation *is* the cyclotron rotation, so
  a spin started along the momentum stays along it. It needs no reference, holds at any
  amplitude and any momentum, and is broken by a wrong sign, a mis-scaled coefficient
  *and* a missing bend frame rotation. Its teeth are asserted too: each of those three
  faults leaves a residual that does **not** converge under slicing.
- **a sector bend on the design orbit**, whose net rotation is exactly ``-G gamma theta``
  about ``y`` — the BMT rotation ``-(1 + G gamma) theta`` plus the frame's own
  ``+theta`` — with no quadrature error at all, because the field is constant.
- **a quadrupole at a vertical offset**, which pins the ``(1 + G gamma)`` factor itself
  against a sympy-derived ``int y ds``. The Dirac identity cannot: the factor is ``1``
  there by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim.elements.corrector import Corrector
from accsim.elements.dipole import Dipole
from accsim.elements.drift import Drift
from accsim.elements.quadrupole import Quadrupole, ThinQuadrupole
from accsim.elements.skew_quadrupole import SkewQuadrupole
from accsim.lattice import Lattice
from accsim.reference import (
    ELECTRON_ANOMALOUS_MOMENT,
    ELECTRON_MASS_EV,
    PROTON_ANOMALOUS_MOMENT,
    PROTON_MASS_EV,
    ReferenceParticle,
)
from accsim.spin import (
    along_direction_of_motion,
    anomalous_moment,
    direction_of_motion,
    precession_vector,
    rotate,
)
from accsim.tracking import Tracker

G_E = ELECTRON_ANOMALOUS_MOMENT


def electron(energy_eV: float = 5e9, g: float | None = G_E) -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(
        ELECTRON_MASS_EV, energy_eV, charge=-1.0, anomalous_moment=g
    )


def rotation_about_y(angle: float) -> np.ndarray:
    """Right-handed rotation by ``angle`` about ``y``, as a 3x3."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def sliced_arc(n: int, ref: ReferenceParticle) -> Lattice:
    """One FODO-ish cell with a bend, cut into ``n`` slices per element.

    Slicing an element is what turns the midpoint rule into a convergent quadrature, so
    every order-gate below is a sweep over ``n``. Thin elements are deliberately absent:
    they do not precess at all (see :mod:`accsim.spin`), so a thin-lens ring would make
    the whole sweep vacuous.
    """
    elements: list = []
    for proto in (
        Dipole(length=2.0, angle=0.3),
        Drift(1.0),
        Quadrupole(0.5, 1.2),
        Drift(1.0),
        Quadrupole(0.5, -1.2),
    ):
        for _ in range(n):
            if isinstance(proto, Dipole):
                elements.append(Dipole(length=proto.length / n, angle=proto.angle / n))
            elif isinstance(proto, Quadrupole):
                elements.append(Quadrupole(length=proto.length / n, k1=proto.k1))
            else:
                elements.append(Drift(proto.length / n))
    return Lattice(elements, ref)


# --- the Dirac identity: G = 0 locks the spin to the direction of motion -----------


def dirac_residual(n: int, state: np.ndarray) -> float:
    """``|S - p_hat|`` after one pass of the ``n``-sliced arc, starting from ``S = p_hat``."""
    tracker = Tracker(sliced_arc(n, electron(g=0.0)))
    out, spin = tracker.track_once_with_spin(state.copy(), along_direction_of_motion(state))
    return float(np.linalg.norm(spin - along_direction_of_motion(out)))


@pytest.mark.parametrize(
    "state",
    [
        np.array([2e-3, 1e-3, -1.5e-3, 0.7e-3, 0.0, 1e-3]),
        np.array([-4e-3, -2e-3, 3e-3, 1.1e-3, 1e-3, -2e-3]),
    ],
)
def test_no_anomalous_moment_locks_the_spin_to_the_direction_of_motion(state):
    """With ``G = 0`` the spin follows the momentum, to the accuracy of the quadrature.

    This is the sharpest gate on the axis and the only one that needs no reference at
    all. It is second-order accurate wherever the field varies along the path — the
    midpoint rule's own order, asserted as a ratio of 4 per halving rather than as a
    tolerance.

    **Only off-orbit states are parametrised here, deliberately.** On the design orbit
    the residual is identically zero at every slicing, which would make the order-gate
    pass *vacuously* — and would go on passing if a regression made ``_precess`` return
    its input untouched. The exact case is worth asserting and gets its own test below,
    where "exactly zero" is the claim rather than an escape hatch.
    """
    residuals = [dirac_residual(n, state) for n in (2, 4, 8, 16)]
    assert residuals[0] > 1e-8  # the sweep is measuring something
    for coarse, fine in zip(residuals, residuals[1:], strict=False):
        assert coarse / fine == pytest.approx(4.0, rel=0.02)
    assert residuals[-1] < 1e-5  # and small in absolute terms, for a unit vector


def test_the_dirac_identity_is_exact_on_the_design_orbit():
    """No slicing needed where the field and the direction are both constant.

    A particle down the axis of the arc sees ``b = (0, h, 0)`` and moves along ``s``
    through every element, so the midpoint rule is not an approximation of anything and
    the identity holds to round-off at a single slice. That is what makes the sector
    bend's closed form below exact rather than convergent.
    """
    assert dirac_residual(1, np.zeros(6)) < 1e-15


def test_a_mis_scaled_precession_does_not_converge(monkeypatch):
    """The teeth: scale ``Omega`` by 1.01 and the Dirac residual stops converging.

    Without this the order-gate above could be passed by an implementation that is
    uniformly wrong — the residual would still fall as ``1/n^2`` towards the *wrong*
    limit. Here it plateaus instead, because the identity is broken at zeroth order.
    """
    import accsim.spin as spin_mod

    true = spin_mod.precession_vector
    monkeypatch.setattr(spin_mod, "precession_vector", lambda *a, **k: 1.01 * true(*a, **k))

    state = np.array([2e-3, 1e-3, -1.5e-3, 0.7e-3, 0.0, 1e-3])
    residuals = [dirac_residual(n, state) for n in (2, 4, 8, 16)]
    assert min(residuals) > 1e-3  # nowhere near converging
    assert residuals[0] / residuals[-1] < 1.5  # and flat, not falling as 1/n^2


def test_a_flipped_precession_sign_does_not_converge(monkeypatch):
    """The same teeth for the sign, which the ``|Omega|`` in the rotation angle hides.

    :func:`accsim.spin.rotate` turns ``Omega`` into an axis and a *magnitude*, so a
    global sign error survives every check that looks at rotation size alone. It does
    not survive this one.
    """
    import accsim.spin as spin_mod

    true = spin_mod.precession_vector
    monkeypatch.setattr(spin_mod, "precession_vector", lambda *a, **k: -true(*a, **k))

    state = np.array([2e-3, 1e-3, -1.5e-3, 0.7e-3, 0.0, 1e-3])
    residuals = [dirac_residual(n, state) for n in (2, 8)]
    assert min(residuals) > 1e-2


def test_a_missing_bend_frame_rotation_does_not_converge(monkeypatch):
    """And for the frame rotation, which is the half of a bend that is pure geometry.

    Drop it and a dipole rotates a Dirac spin by the full ``-theta`` instead of leaving
    it alone; the residual is then ``O(theta)`` per cell and slicing cannot help,
    because each slice contributes its own share of the same total angle.
    """
    monkeypatch.setattr(Dipole, "frame_rotation_angle", property(lambda self: 0.0))

    state = np.array([2e-3, 1e-3, -1.5e-3, 0.7e-3, 0.0, 1e-3])
    residuals = [dirac_residual(n, state) for n in (2, 8)]
    assert min(residuals) > 0.1
    assert residuals[0] == pytest.approx(residuals[1], rel=0.05)


# --- the sector bend: an exact closed form ----------------------------------------


@pytest.mark.parametrize("angle", [0.3, -0.3, 0.05, 1.0])
@pytest.mark.parametrize("energy_eV", [1e9, 5e9])
def test_a_sector_bend_rotates_the_spin_by_exactly_minus_g_gamma_theta(angle, energy_eV):
    r"""``S_out = R_y(-G gamma theta) S_in`` on the design orbit, to round-off.

    The BMT rotation about the constant field is ``-(1 + G gamma) theta``; the
    curvilinear frame's own turn is ``+theta``; the ``1`` cancels and what is left is
    the spin's *excess* precession over the orbit's, which is the whole content of the
    spin tune. Compared as a rotation applied to three independent starting spins
    rather than as an angle, so a ``2 pi`` wrap cannot be mistaken for agreement.
    """
    ref = electron(energy_eV)
    bend = Dipole(length=2.0, angle=angle)
    expected = rotation_about_y(-G_E * ref.gamma0 * angle)

    for start in np.eye(3):
        _, spin = bend.track_with_spin(np.zeros(6), start, ref)
        np.testing.assert_allclose(spin, expected @ start, atol=1e-13)


def test_the_bend_closed_form_needs_no_slicing():
    """Cutting the bend into 64 pieces changes nothing: the field is constant.

    Worth asserting because it separates the two error sources on this axis. Everything
    that converges below converges because the *field varies along the path*, not
    because a rotation is being composed approximately — rotations about a fixed axis
    compose exactly.
    """
    ref = electron()
    whole = Dipole(length=2.0, angle=0.3)
    _, spin_1 = whole.track_with_spin(np.zeros(6), np.array([1.0, 0.0, 0.0]), ref)

    spin = np.array([1.0, 0.0, 0.0])
    state = np.zeros(6)
    for _ in range(64):
        state, spin = Dipole(length=2.0 / 64, angle=0.3 / 64).track_with_spin(state, spin, ref)
    np.testing.assert_allclose(spin, spin_1, atol=1e-14)


# --- the quadrupole: what pins the (1 + G gamma) factor -----------------------------


def integral_of_y_through_a_defocusing_quadrupole(k1: float, length: float, y0: float) -> float:
    r"""``int_0^L y ds`` for ``y'' = +k1 y``, derived with sympy rather than recalled."""
    s, k, ell, y_0 = sp.symbols("s k L y0", positive=True)
    closed = sp.integrate(y_0 * sp.cosh(sp.sqrt(k) * s), (s, 0, ell))
    return float(closed.subs({k: k1, ell: length, y_0: y0}))


def test_a_quadrupole_at_a_vertical_offset_pins_the_one_plus_g_gamma_factor():
    r"""``phi = -(1 + G gamma) k1 int y ds`` about ``x``, converged at second order.

    This is the gate the Dirac identity cannot be: there ``G = 0`` makes the factor
    exactly ``1``, so nothing about its size is tested. Here the same number appears
    multiplied by ``gamma ~ 9785``, so the three candidate readings are an order of
    magnitude apart — ``(1 + G gamma) = 12.35``, ``G gamma = 11.35``, ``(1 + G) =
    1.0012`` — and no tolerance is needed to tell them apart.

    A quadrupole is the right element for it because at ``x = px = 0`` the field is
    ``b = (k1 y, 0, 0)``, purely horizontal and purely perpendicular to the motion: the
    precession axis is ``x`` throughout, so the rotations commute and the *only* error
    is the quadrature of ``int y ds``.
    """
    ref = electron()
    k1, length, y0 = 1.2, 0.8, 1e-4
    integral = integral_of_y_through_a_defocusing_quadrupole(k1, length, y0)
    expected = -(1.0 + G_E * ref.gamma0) * k1 * integral

    # the three readings this gate exists to separate
    assert abs(expected / (-G_E * ref.gamma0 * k1 * integral)) > 1.08
    assert abs(expected / (-(1.0 + G_E) * k1 * integral)) > 10.0

    residuals = []
    for n in (2, 4, 8, 16):
        state = np.array([0.0, 0.0, y0, 0.0, 0.0, 0.0])
        spin = np.array([0.0, 1.0, 0.0])
        for _ in range(n):
            state, spin = Quadrupole(length=length / n, k1=k1).track_with_spin(state, spin, ref)
        residuals.append(math.atan2(spin[2], spin[1]) - expected)

    for coarse, fine in zip(residuals, residuals[1:], strict=False):
        assert coarse / fine == pytest.approx(4.0, rel=0.02)
    assert abs(residuals[-1] / expected) < 3e-4


# --- the (1 + G) parallel term, with no solenoid in the package ---------------------


def test_the_parallel_term_is_the_projection_of_omega_on_the_direction_of_motion():
    r"""``Omega . i_hat = -(1 + G) (b . i_hat) / (1 + delta)``, exactly.

    Every element in this package has a purely transverse field, which makes it tempting
    to read the ``(1 + G)`` term as dead code awaiting a solenoid. It is not:
    ``b_par`` is the component of ``b`` along the **direction of motion**, and a
    transverse field has one as soon as the particle has an angle. Projecting ``Omega``
    back onto ``i_hat`` isolates that coefficient exactly, since ``b_perp . i_hat`` is
    zero by construction — so this reads ``(1 + G)`` off with no other term in the way.
    """
    ref = electron()
    g = anomalous_moment(ref)
    bx, by, delta = 0.7, -1.3, 3e-3
    px, py = 2e-3, -5e-4

    omega = precession_vector(bx, by, px, py, delta, ref)
    i_hat = direction_of_motion(px, py, delta)
    b_dot_i = bx * i_hat[0] + by * i_hat[1]

    assert float(omega @ i_hat) == pytest.approx(-(1.0 + g) * b_dot_i / (1.0 + delta), rel=1e-13)


def test_the_parallel_term_vanishes_with_the_angle_and_is_first_order_in_it():
    """It is zero on the design orbit, and grows linearly with ``px``.

    The same degeneracy M3 found for second-order dispersion, in another place: a
    quantity that is identically zero on every unsteered, on-axis test ring is a
    quantity no such ring can gate. So the order in ``px`` is asserted, not the value.
    """
    ref = electron()
    i_hat = direction_of_motion(0.0, 0.0, 0.0)
    assert float(precession_vector(0.4, -0.9, 0.0, 0.0, 0.0, ref) @ i_hat) == 0.0

    parallel = []
    for px in (1e-3, 2e-3, 4e-3):
        omega = precession_vector(0.4, -0.9, px, 0.0, 0.0, ref)
        parallel.append(abs(float(omega @ direction_of_motion(px, 0.0, 0.0))))
    for small, large in zip(parallel, parallel[1:], strict=False):
        assert large / small == pytest.approx(2.0, rel=1e-3)


# --- the control, and its blindness ------------------------------------------------


def flat_ring(k1: float, ref: ReferenceParticle, n_cells: int = 8) -> Lattice:
    """A closed, flat ring: ``n_cells`` bends summing to ``2 pi``, plus a quad doublet."""
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


def spin_tune_on_the_design_orbit(k1: float, ref: ReferenceParticle) -> float:
    """The angle a design-orbit spin turns through in one turn, divided by ``2 pi``."""
    tracker = Tracker(flat_ring(k1, ref))
    _, spin = tracker.track_once_with_spin(np.zeros(6), np.array([1.0, 0.0, 0.0]))
    return -math.atan2(-spin[2], spin[0]) / (2.0 * math.pi)


@pytest.mark.parametrize("energy_eV", [1e9, 5e9, 45.6e9])
def test_the_spin_tune_is_g_gamma(energy_eV):
    """The control: a flat ring turns a spin ``G gamma`` times per turn.

    Correct, famous, and *not* a gate — see the module docstring, and the two tests
    below for exactly what it fails to see. It is kept because it is the number the
    rest of the axis is built on: at LEP's 45.6 GeV it is ``103.5``, which is why half
    a percent of energy calibration is worth a milestone.
    """
    ref = electron(energy_eV)
    assert spin_tune_on_the_design_orbit(1.2, ref) % 1.0 == pytest.approx(
        (G_E * ref.gamma0) % 1.0, abs=1e-9
    )


def test_the_spin_tune_is_blind_to_every_quadrupole():
    """Change the quadrupoles by a factor of five and the spin tune does not move a bit.

    Asserted **bit-for-bit**, because that is the honest strength of the statement: on
    the design orbit a quadrupole has no field, so its spin kick is exactly zero and an
    implementation that omitted it entirely would pass :func:`test_the_spin_tune_is_g_gamma`
    at every energy. J1 recorded the same shape — structural gates that cannot see the
    coefficient they appear to test.
    """
    ref = electron()
    assert spin_tune_on_the_design_orbit(1.2, ref) == spin_tune_on_the_design_orbit(6.0, ref)


def test_the_spin_tune_is_blind_to_a_mis_scaled_quadrupole_precession(monkeypatch):
    """The same blindness stated as a fault injection: break the quadrupole, tune unmoved.

    :func:`test_a_quadrupole_at_a_vertical_offset_pins_the_one_plus_g_gamma_factor` is
    what catches this; the control cannot.
    """
    ref = electron()
    before = spin_tune_on_the_design_orbit(1.2, ref)

    original = Quadrupole.normalized_field
    monkeypatch.setattr(
        Quadrupole,
        "normalized_field",
        lambda self, x, y: tuple(7.0 * np.asarray(c) for c in original(self, x, y)),
    )
    assert spin_tune_on_the_design_orbit(1.2, ref) == before


# --- scope, stated as tests --------------------------------------------------------


def test_thin_elements_do_not_precess():
    """The axis's one real omission, asserted rather than left to be discovered.

    Unlike radiation — where a zero-length magnet genuinely radiates nothing, since
    ``U ~ kappa^2 L`` — a thin quadrupole's *integrated* field is finite and its true
    spin rotation is not zero. It is dropped because xtrack's thin ``Multipole`` does
    not rotate spin either, so there would be no arbiter (L5's reason). The cost: a
    thin-lens ring has no spin dynamics at all.
    """
    ref = electron()
    state = np.array([1e-3, 2e-4, -3e-3, 1e-4, 0.0, 1e-3])
    spin = np.array([0.3, 0.6, math.sqrt(1 - 0.09 - 0.36)])
    for thin in (ThinQuadrupole(k1l=0.5), Corrector(kick_x=1e-3), Drift(1.0)):
        _, out = thin.track_with_spin(state, spin, ref)
        np.testing.assert_array_equal(out, spin)


def test_a_rolled_quadrupole_precesses_a_spin_like_the_skew_quadrupole_it_is():
    """The same magnet spelled two ways must rotate a spin the same way.

    G1's identity, carried onto the new quantity: a normal quadrupole rolled by 45
    degrees *is* a skew quadrupole. (By ``-45`` in the base class's sense:
    ``SkewQuadrupole._track_body`` is ``s_rotation(+45) . Q . s_rotation(-45)``, and
    ``Element._track_impl`` enters the body with ``s_rotation(roll)``.)

    For the spin this exercises two different code paths — the roll rotates the spin
    into and out of the body frame, while ``SkewQuadrupole`` instead rotates its own
    field — so agreeing is a real check and not a tautology. **It failed when first
    written**: ``SkewQuadrupole.normalized_field`` rolled the *opposite* way from its
    own map, flipping the field vector while leaving ``|b|`` alone. Radiation, the only
    consumer a field had until now, takes the magnitude and could not see it.
    """
    ref = electron()
    k1, length = 1.4, 0.6
    state = np.array([1e-3, 2e-4, -3e-3, 1e-4, 0.0, 5e-4])
    spin = np.array([0.3, 0.6, math.sqrt(1 - 0.09 - 0.36)])

    rolled = Quadrupole(length, k1, roll=-math.pi / 4)
    skew = SkewQuadrupole(length, k1s=k1)
    state_r, spin_r = rolled.track_with_spin(state, spin, ref)
    state_s, spin_s = skew.track_with_spin(state, spin, ref)

    np.testing.assert_allclose(state_r, state_s, atol=1e-15)
    np.testing.assert_allclose(spin_r, spin_s, atol=1e-15)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda length: Quadrupole(length, 1.4), id="quadrupole"),
        pytest.param(lambda length: SkewQuadrupole(length, k1s=1.4), id="skew-quadrupole"),
        pytest.param(lambda length: Quadrupole(length, 1.4, roll=0.37), id="rolled-quadrupole"),
        pytest.param(lambda length: Dipole(length, angle=0.0, k1=1.4), id="straight-gradient"),
    ],
)
def test_a_straight_magnets_field_agrees_with_its_own_momentum_kick(make):
    r"""``(dpx, dpy)/L -> (-b_y, +b_x)`` as ``L -> 0``, for every straight magnet.

    The structural gate this axis needed and the package did not have.
    :meth:`~accsim.elements.element.Element.normalized_field` had exactly one consumer
    — the radiation kick — and that consumer takes ``|b_perp|``, so a field could point
    the wrong way indefinitely without a single test noticing. Spin is the first
    quantity that reads the field's *direction*, and the first thing it found was
    :class:`~accsim.elements.skew_quadrupole.SkewQuadrupole` rolling its field the
    opposite way from its map.

    Compared against ``_track_body``, not ``track``, because that is where the
    invariant actually lives: ``normalized_field`` is the element's field **in its own
    frame**, and both consumers — the radiation kick and the spin precession — evaluate
    it on *body-frame* coordinates. A rolled element makes the distinction visible, and
    checking it against the lab-frame kick would fail for a correct implementation.

    Stated as a limit rather than an equality because a thick magnet's kick is the
    integral of its field along a trajectory that is itself curving away from the point
    the field was sampled at; the two agree only as the element is shortened. Entered
    with ``px = py = 0`` the leading error cancels and the convergence is **second**
    order in ``L`` (a ratio of 4 per halving), which is asserted rather than the first
    order a one-sided sample would give — an order gate that had to be measured is
    still an order gate, and it discriminates where a tolerance would not.

    Bends are excluded on purpose: there the curvilinear frame's own turn cancels the
    design field, so a sector bend's kick is zero while its field is ``h``.
    """
    ref = electron()
    x, y = 2e-3, -1.3e-3
    state = np.array([x, 0.0, y, 0.0, 0.0, 0.0])

    bx, by = make(1.0).normalized_field(x, y)
    residuals = []
    for length in (1e-2, 5e-3, 2.5e-3):
        out = make(length)._track_body(state, ref)
        measured = np.array([out[1] - state[1], out[3] - state[3]]) / length
        residuals.append(float(np.linalg.norm(measured - np.array([-by, bx]))))

    for coarse, fine in zip(residuals, residuals[1:], strict=False):
        assert coarse / fine == pytest.approx(4.0, rel=0.05)
    assert residuals[-1] < 1e-4 * math.hypot(bx, by)


def test_a_rolled_bend_refuses_rather_than_guessing():
    """K2's rigid-body geometry moves a bend's exit face, so the frame turn is not ``R_y``.

    Applying the aligned formula anyway would be a plausible answer with no arbiter.
    Raising is the recorded scope of N1.
    """
    ref = electron()
    with pytest.raises(NotImplementedError, match="rolled"):
        Dipole(length=2.0, angle=0.3, roll=0.1).track_with_spin(
            np.zeros(6), np.array([1.0, 0.0, 0.0]), ref
        )


def test_a_displaced_quadrupole_precesses_by_where_it_really_is():
    """K1's conjugation reaches the spin too, through the body coordinates.

    A quadrupole shifted down by ``dy`` gives a spin on the axis the same rotation an
    aligned one gives a spin at ``-dy``: the magnet's field is what it is, and the
    offset only changes which part of it the particle crosses.
    """
    ref = electron()
    k1, length, dy = 1.2, 0.8, 2e-4

    displaced = Quadrupole(length, k1, dy=dy)
    _, spin_d = displaced.track_with_spin(np.zeros(6), np.array([0.0, 1.0, 0.0]), ref)

    aligned = Quadrupole(length, k1)
    shifted = np.array([0.0, 0.0, -dy, 0.0, 0.0, 0.0])
    _, spin_a = aligned.track_with_spin(shifted, np.array([0.0, 1.0, 0.0]), ref)

    np.testing.assert_allclose(spin_d, spin_a, atol=1e-15)


# --- the API, the invariants, and the species -------------------------------------


def test_an_unset_anomalous_moment_raises_rather_than_precessing_at_zero():
    """The trap xtrack sets: ``anomalous_magnetic_moment`` defaults to ``0`` there.

    A silent zero is not "no spin physics" — it is the cyclotron rotation and a spin
    tune of exactly zero, which looks like a working simulation. accsim refuses; an
    explicit ``0.0`` is still accepted, because it is the Dirac limit the sharpest gate
    above runs in.
    """
    with pytest.raises(ValueError, match="anomalous_moment"):
        Dipole(length=1.0, angle=0.1).track_with_spin(
            np.zeros(6), np.array([1.0, 0.0, 0.0]), electron(g=None)
        )
    anomalous_moment(electron(g=0.0))  # explicit zero is fine


def test_carrying_a_spin_leaves_the_orbit_bit_for_bit_alone():
    """The axis's structural claim: nothing on axes A-M moves because of this one.

    Checked on a misaligned, rolled, dispersive lattice so that every branch of
    ``_track_impl`` is exercised, and with ``==`` rather than a tolerance — a spin does
    not act back on the orbit, so "unchanged" here means *identical*, not *close*.
    """
    ref = electron()
    elements = [
        Dipole(length=2.0, angle=0.3),
        Drift(1.0),
        Quadrupole(0.5, 1.2, dx=1e-4, dy=-2e-4),
        Quadrupole(0.5, -1.2, roll=0.05),
        ThinQuadrupole(k1l=0.1),
    ]
    state = np.array([1e-3, 2e-4, -3e-3, 1e-4, 5e-4, 1e-3])

    plain, carried = state.copy(), state.copy()
    spin = np.array([0.0, 1.0, 0.0])
    for element in elements:
        plain = element.track(plain, ref)
        carried, spin = element.track_with_spin(carried, spin, ref)
    np.testing.assert_array_equal(plain, carried)


def test_the_spin_stays_a_unit_vector():
    """Cheap, blind, and worth having: every map on this axis is a rotation."""
    ref = electron()
    tracker = Tracker(sliced_arc(3, ref))
    spin = np.array([0.3, 0.6, math.sqrt(1 - 0.09 - 0.36)])
    state = np.array([2e-3, 1e-3, -1.5e-3, 0.7e-3, 0.0, 1e-3])
    for _ in range(20):
        state, spin = tracker.track_once_with_spin(state, spin)
    assert float(np.linalg.norm(spin)) == pytest.approx(1.0, abs=1e-14)


def test_the_precession_follows_the_particle_gamma_not_the_reference_one():
    r"""``gamma = E(delta)/m``, so the precession is chromatic at first order in ``delta``.

    Using ``gamma0`` instead would be a silent, uniform 1% error at ``delta = 1e-2`` on
    an ultra-relativistic beam — invisible to every on-momentum gate above, and exactly
    the kind of thing the spin tune's blindness would wave through. Asserted against
    both readings so the test cannot pass under either.
    """
    ref = electron()
    delta, h = 1e-2, 0.15
    gamma = math.sqrt((ref.momentum_eV * (1 + delta)) ** 2 + ref.mass_eV**2) / ref.mass_eV

    omega = precession_vector(0.0, h, 0.0, 0.0, delta, ref)
    assert float(omega[1]) == pytest.approx(-(1.0 + G_E * gamma) * h / (1.0 + delta), rel=1e-13)
    wrong = -(1.0 + G_E * ref.gamma0) * h / (1.0 + delta)
    assert abs(float(omega[1]) / wrong - 1.0) > 5e-3


def test_the_normalisation_makes_the_species_charge_drop_out():
    r"""An electron (``q = -1``) and a proton at the same ``G gamma`` precess identically.

    The textbook ``Omega`` divides by the rigidity ``B rho = p/q`` and so carries the
    charge; the package's ``b`` already *is* ``B q / p_0``, and the two cancel. Worth
    a test because a stray ``ref.charge`` would be invisible on any single-species ring
    — every other gate in this file runs on electrons.
    """
    e_ref = electron(5e9)
    target_g_gamma = G_E * e_ref.gamma0
    p_gamma = target_g_gamma / PROTON_ANOMALOUS_MOMENT
    p_ref = ReferenceParticle.from_gamma(
        PROTON_MASS_EV, p_gamma, charge=1.0, anomalous_moment=PROTON_ANOMALOUS_MOMENT
    )

    bend = Dipole(length=2.0, angle=0.3)
    _, spin_e = bend.track_with_spin(np.zeros(6), np.array([1.0, 0.0, 0.0]), e_ref)
    _, spin_p = bend.track_with_spin(np.zeros(6), np.array([1.0, 0.0, 0.0]), p_ref)
    np.testing.assert_allclose(spin_e, spin_p, atol=1e-11)


def test_a_zero_precession_returns_the_spin_bit_for_bit():
    """No threshold, no branch: ``|Omega| = 0`` gives back the input exactly.

    xtrack skips its rotation below ``|Omega| = 1e-10``, which is a discontinuity in a
    quantity that ought to be smooth. Rodrigues' formula needs no such cut — the angle
    is zero, so the sine is zero and the cosine is one.
    """
    spin = np.array([0.3, 0.6, math.sqrt(1 - 0.09 - 0.36)])
    np.testing.assert_array_equal(rotate(spin, np.zeros(3), 1.0), spin)


def test_a_bunch_precesses_column_by_column():
    """A ``(3, n)`` spin array tracks a ``(6, n)`` bunch, and agrees particle by particle."""
    ref = electron()
    tracker = Tracker(sliced_arc(2, ref))
    states = np.array(
        [
            [1e-3, -2e-3, 0.0],
            [2e-4, 1e-4, 0.0],
            [-3e-3, 1e-3, 0.0],
            [1e-4, -2e-4, 0.0],
            [0.0, 0.0, 0.0],
            [1e-3, -1e-3, 0.0],
        ]
    )
    spins = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=float)
    spins /= np.linalg.norm(spins, axis=0)

    out_states, out_spins = tracker.track_once_with_spin(states.copy(), spins.copy())
    for column in range(states.shape[1]):
        one_state, one_spin = tracker.track_once_with_spin(
            states[:, column].copy(), spins[:, column].copy()
        )
        np.testing.assert_allclose(out_states[:, column], one_state, atol=1e-15)
        np.testing.assert_allclose(out_spins[:, column], one_spin, atol=1e-15)
