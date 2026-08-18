r"""L4 — the curved quadrupole's expanded map, and the model boundary it lands on.

The last element in the package whose ``track`` was its ``matrix``. L3 proved the split
at ``k1``: with ``k1 = 0`` a bend's flow is a circle and has a closed form, with
``k1 != 0`` it does not, so a *curved* quadrupole cannot have an exact map at all. What
it can have is the **expanded** one — MAD-X's ``track_thick_cfd``, xtrack's
``mat-kick-mat`` — which is the exact flow of the *paraxial* combined-function
Hamiltonian, plus F2's Maxwell curvature-sextupole term as one centred thin kick.

Three gate shapes, in the order they bind:

1. **The origin Jacobian is the linear matrix**, entry for entry. It needs no reference
   implementation and it confirms ``K_x``, the dispersion drive ``G``, the three path
   integrals *and* the ``zeta`` split in one go, against a ``matrix()`` that is already
   xtrack- and MAD-X-validated.
2. **The closed form is the equations of motion's own solution.** The map is compared
   against a direct ``solve_ivp`` integration of the paraxial equations, which shares no
   arithmetic with it — including at ``K_x = 0`` exactly, where every path integral has a
   removable singularity.
3. **Feed-down pins the Maxwell coefficients.** Linearising ``track`` about an orbit
   offset must reproduce F2's derived generator — ``-2 h k1 x_0`` horizontally,
   ``+h k1 x_0`` vertically, ``+h k1 y_0`` in both cross terms. That ``2:-1`` is not an
   ordinary sextupole's ratio and nothing structural (symplecticity, Maxwell alone, the
   Jacobian identity) can see it.

And then the honest part, which is the most useful thing in this file. **The expanded
family drops the curvilinear metric factor** ``(1 + h x)`` from ``x' = px(1+hx)/pz``,
keeping it only in the path length. That is exactly F2's ``h(gamma_x D_x - 2 alpha_x
D_px)`` / ``gamma_y h D_x`` group, so a *bending* combined-function magnet's tracked
chromaticity converges to **F2 minus that group** and not to F2 — measured here in closed
form, and confirmed against xtrack's own converged ``mat-kick-mat`` in the reference
suite. A *straight* gradient magnet has ``h = 0``, the group vanishes identically, and
tracking reaches 100%: that is the case
``tests/analytic/test_exact_quadrupole.py``'s 56% control measures.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

from accsim import (
    PROTON_MASS_EV,
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    is_symplectic_map,
    is_symplectic_map_canonical,
    jacobian,
    natural_chromaticity,
    tunes_on_orbit,
)
from accsim.coords import DELTA, DIM, PX, PY, ZETA, X, Y
from accsim.elements.dipole import curvature_sextupole_kick, expanded_cfd_map
from accsim.elements.element import Element
from accsim.twiss import closed_twiss

L_B = 1.0
ANGLE = 0.3927  # ~ 2 pi / 16: sin, theta and tan differ in the third digit
K1 = 0.6

# Every coordinate nonzero, so no term can hide behind a zero.
STATE = np.array([3.0e-3, 8.0e-3, -2.0e-3, 5.0e-3, 1.0e-3, 2.0e-3])
# A second one at large ``delta``, the coordinate this map is exact in.
STATE_BIG = np.array([2.0e-2, -5.0e-3, 1.5e-2, 4.0e-3, 0.0, 5.0e-2])


@pytest.fixture
def ref() -> ReferenceParticle:
    """gamma0 = 20: a real machine, and the ``1/gamma0^2`` slip still visible."""
    return ReferenceParticle.from_gamma(PROTON_MASS_EV, 20.0)


# --------------------------------------------------------------------------
# The independent derivation: the equations of motion, integrated
# --------------------------------------------------------------------------


def _ode_map(
    state: np.ndarray,
    length: float,
    h: float,
    k1: float,
    ref: ReferenceParticle,
    *,
    cubic: bool = False,
) -> np.ndarray:
    r"""The paraxial combined-function equations of motion, integrated numerically.

    Nothing here is a transcription of the map under test: no ``cos``/``cosh`` family, no
    path integrals, no series. It is the system the map is the *solution of*, written
    straight from the Hamiltonian

        H = (px^2 + py^2)/(2q) + q(K_x x^2 + K_y y^2)/2 - q G x,

    ``q = 1 + delta``, ``K_x = (h^2+k1)/q``, ``K_y = -k1/q``, ``G = h delta/q``, whose
    Hamilton equations are the five lines below. Every ``q`` cancels out of the momentum
    equations, which is the statement that a **field** changes momentum by an amount
    independent of the particle's rigidity; it is the *angles* that respond, through
    ``x' = px/q``.

    ``zeta' = 1 - (1/rvv)(1 + h x + (x'^2 + y'^2)/2)`` is the path length: the design
    length, plus the extra arc a particle on the outside of the bend travels, plus the
    extra distance its angle makes it travel, all against its own speed.

    With ``cubic=True`` the Maxwell curvature-sextupole term of
    :func:`~accsim.elements.dipole.curvature_sextupole_kick` is added to the momentum
    equations, so the *element* — which applies it as one centred thin kick — can be
    checked to converge to it.
    """
    x0, px0, y0, py0, z0, delta = (float(v) for v in state)
    q = 1.0 + delta
    e_over_e0 = np.hypot(ref.momentum_eV * q, ref.mass_eV) / ref.total_energy_eV
    inv_rvv = e_over_e0 / q

    def rhs(_s: float, u: list[float]) -> list[float]:
        x, px, y, py, _z = u
        dpx = -(h * h + k1) * x + h * delta
        dpy = k1 * y
        if cubic:
            dpx += h * k1 * (0.5 * y * y - x * x)
            dpy += h * k1 * x * y
        xp, yp = px / q, py / q
        return [xp, dpx, yp, dpy, 1.0 - inv_rvv * (1.0 + h * x + 0.5 * (xp * xp + yp * yp))]

    sol = solve_ivp(
        rhs, (0.0, length), [x0, px0, y0, py0, z0], rtol=1e-13, atol=1e-16, method="DOP853"
    )
    out = sol.y[:, -1]
    return np.array([out[0], out[1], out[2], out[3], out[4], delta])


# --------------------------------------------------------------------------
# 1. The gate that binds first: matrix() is the origin Jacobian
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("length", "angle", "k1", "e1", "e2"),
    [
        (L_B, ANGLE, K1, 0.0, 0.0),
        (L_B, ANGLE, -K1, 0.0, 0.0),
        (2.0, 0.0, 0.7, 0.0, 0.0),  # a straight gradient magnet
        (L_B, ANGLE, K1, 0.1, 0.15),  # with pole faces
        (1.5, 0.6, 1.2, 0.0, 0.0),  # a strong, strongly-bent one
    ],
)
def test_the_origin_jacobian_is_the_linear_matrix_entry_for_entry(
    ref: ReferenceParticle, length: float, angle: float, k1: float, e1: float, e2: float
) -> None:
    r"""The invariant that bounds the whole axis, and the cheapest gate in the file.

    ``matrix()`` is the Jacobian of ``track()`` at the origin — exactly, not to first
    order. It is what every optics function in the package is built on, so a milestone
    that broke it would move numbers everywhere, and it is what rules out slicing
    families (a sliced combined-function bend's Jacobian is the *product* of the slices'
    matrices, which is not ``exp(L A)``).

    Two structural facts make it survive here. The ``mat . kick . mat`` composition
    splits the body into two halves, and two Hill solutions over ``L/2`` compose to the
    one over ``L`` **identically** — the dispersion drive is the same inhomogeneous
    equation and the path integrals simply add. And the curvature-sextupole kick is
    *quadratic* in the coordinates, so its Jacobian at the origin is exactly zero. Neither
    is obvious enough to leave unasserted.

    It is also a free cross-check on the algebra: ``matrix()`` here is
    :meth:`~accsim.elements.dipole.Dipole._combined_function_body`, which is validated
    against xtrack's R-matrix *and* MAD-X's, so agreement pins ``K_x = h^2 + k1``, the
    dispersion pair ``(R16, R26) = (h c1, h s1)``, their symplectic partners
    ``R51/R52``, and ``R56 = L/gamma0^2 + h^2 c2`` — all of which this map builds by
    completely different arithmetic.

    The residual is stated as a **floor that improves with the step**: truncation, which
    is what a correct map gives, not cancellation, which is what a badly-arranged one
    gives (the failure mode L3 measured at ``3.2e-9`` and *growing*).
    """
    elem = Dipole(length, angle, k1=k1, e1=e1, e2=e2)
    M = elem.matrix(ref)

    coarse = float(np.abs(jacobian(lambda s: elem.track(s, ref), np.zeros(DIM), 1e-6) - M).max())
    fine = float(np.abs(jacobian(lambda s: elem.track(s, ref), np.zeros(DIM), 1e-8) - M).max())
    assert coarse < 1.0e-12
    assert fine < 1.0e-14
    assert fine < coarse  # truncation, not cancellation


def test_two_halves_compose_to_the_whole_body(ref: ReferenceParticle) -> None:
    """The composition claim above, run rather than argued.

    If ``mat(L/2) . mat(L/2)`` were not ``mat(L)``, the Jacobian identity would hold only
    to the difference and the previous test would be measuring the wrong thing. It is an
    identity of the closed form, so it holds at any amplitude and any ``delta``, not just
    at the origin.
    """
    h = ANGLE / L_B
    for st in (STATE, STATE_BIG):
        halves = expanded_cfd_map(
            expanded_cfd_map(st, 0.5 * L_B, h, K1, ref), 0.5 * L_B, h, K1, ref
        )
        np.testing.assert_allclose(halves, expanded_cfd_map(st, L_B, h, K1, ref), atol=2e-17)


# --------------------------------------------------------------------------
# 2. The closed form is the equations of motion's own solution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("length", "h", "k1"),
    [
        (1.0, 0.4, 0.5),
        (2.0, 0.0, 0.3),  # straight: the pure-quadrupole limit
        (1.0, 0.4, -0.16),  # K_x = h^2 + k1 = 0 EXACTLY: every path integral is 0/0
        (1.5, 0.25, 1.1),
        (1.0, 0.4, 0.16),  # K_y = 0 is not special, but K_x = 2 h^2 checks the other side
    ],
)
def test_the_map_solves_the_equations_of_motion_it_claims_to(
    ref: ReferenceParticle, length: float, h: float, k1: float
) -> None:
    r"""The discriminating gate, and it shares no arithmetic with the implementation.

    :func:`_ode_map` integrates the paraxial combined-function equations of motion
    directly; :func:`~accsim.elements.dipole.expanded_cfd_map` writes down their closed
    solution. They agree to the integrator's own tolerance on states with **every**
    coordinate nonzero and at ``delta`` up to 10%, which pins in one shot: the rigidity
    scaling ``K = (h^2+k1)/(1+delta)``, the dispersion drive ``G = h delta/(1+delta)``,
    the ``(1+delta)`` factors on the momenta, the curvature's own path term ``h int x ds``,
    the angle's ``int u'^2/2 ds``, and the ``zeta`` rearrangement.

    ``k1 = -h^2`` is in the list deliberately: there ``K_x`` is **exactly zero**, so
    ``c1``, ``c2`` and ``t1`` are all ``0/0``. The half-angle form of ``c1`` and the
    series branch of :func:`~accsim.elements.dipole._cfd_path_integrals` are what make
    that a value rather than a NaN, and nothing else in the suite would notice if they
    were not — a combined-function magnet tuned exactly to cancel its own weak focusing
    is a perfectly ordinary design.
    """
    worst = 0.0
    for st in (STATE, STATE_BIG, np.array([0.0, 0.0, 0.0, 0.0, 0.0, -3.0e-2])):
        got = expanded_cfd_map(st, length, h, k1, ref)
        want = _ode_map(st, length, h, k1, ref)
        worst = max(worst, float(np.abs(got - want).max()))
    assert worst < 1.0e-14


def test_the_weak_gradient_limit_has_no_cliff(ref: ReferenceParticle) -> None:
    r"""``K_x -> 0`` is approached, not special-cased — the series/closed-form switch.

    :func:`~accsim.elements.dipole._cfd_path_integrals` swaps to a Taylor series at
    ``|K L^2| = 1e-2``. A switch is a place to hide a discontinuity, so the map is checked
    against the integrated equations of motion on a sweep that crosses it, including
    exactly at zero. The point is that the error does **not** spike anywhere: a naive
    ``(S - L)/K`` degrades to ``1e-7`` relative near the pole while the series is still
    exact, and the seam is where either mistake would show.
    """
    h = 0.4
    for eps in (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 0.0, -1e-3, -1e-2, -1e-1):
        k1 = -h * h + eps  # K_x = eps / (1 + delta)
        got = expanded_cfd_map(STATE, L_B, h, k1, ref)
        want = _ode_map(STATE, L_B, h, k1, ref)
        assert float(np.abs(got - want).max()) < 1.0e-14, f"eps={eps}"


def test_a_straight_gradient_dipole_is_a_quadrupole_by_two_arithmetic_routes(
    ref: ReferenceParticle,
) -> None:
    r"""``Dipole(L, 0, k1)`` **is** ``Quadrupole(L, k1)``, in ``track`` as in ``matrix``.

    The two maps are written independently — L2's carries no curvature at all, so its
    path lengthening substitutes ``A = -K x`` and cancels every ``1/K``, while this one
    keeps ``A = -K x + G`` and needs ``t1`` explicitly. At ``h = 0`` the drive ``G``
    vanishes and they must land on the same numbers, which is a real check on both rather
    than a tautology: a factor of ``(1+delta)`` misplaced in either would show here and
    nowhere else in this file.

    Their **matrices** were already asserted byte-identical elsewhere; this is the
    tracking half of that statement, and it is the gate that says a gradient bend and a
    quadrupole are one magnet.
    """
    for length, k1 in ((1.0, 0.4), (2.0, -0.7), (0.5, 1.3)):
        straight = Dipole(length, 0.0, k1=k1).track(STATE_BIG, ref)
        quad = Quadrupole(length, k1).track(STATE_BIG, ref)
        np.testing.assert_allclose(straight, quad, rtol=0.0, atol=1e-17)


def test_the_map_broadcasts_over_a_bunch(ref: ReferenceParticle) -> None:
    """A ``(6, n)`` bunch with a momentum spread must equal ``n`` single particles.

    ``K_x``, ``K_y`` and ``G`` are all per-particle here, which is the whole reason this
    is not a matrix multiply — so the vectorisation is load-bearing, not a convenience.
    The zero column also checks that a design particle takes the same code path.
    """
    bunch = np.stack([STATE, STATE_BIG, -0.5 * STATE, np.zeros(DIM)], axis=1)
    elem = Dipole(L_B, ANGLE, k1=K1, e1=0.1, e2=0.05)
    got = elem.track(bunch, ref)
    for j in range(bunch.shape[1]):
        np.testing.assert_allclose(got[:, j], elem.track(bunch[:, j], ref), atol=1e-18)


# --------------------------------------------------------------------------
# 3. The Maxwell curvature-sextupole term, and its 2:-1
# --------------------------------------------------------------------------


def test_the_kicks_potential_is_maxwell_forced_and_the_ratio_is_derived() -> None:
    r"""``psi_3 = -(h k1/3) x^3 + (h k1/2) x y^2``, re-derived rather than recalled.

    The field of a combined-function *sector* magnet cannot be exactly
    ``B_y = B0(h + k1 x)``, ``B_x = B0 k1 y``: in the curved frame that has
    ``div B = h k1 y != 0``. Maxwell forces a third-order correction, and in the
    curvilinear metric with scale factor ``(1 + h x)`` the condition on
    ``psi_3 = c1 x^3 + c2 x y^2`` is ``6 c1 + 2 c2 + h k1 = 0`` — **one** equation for two
    unknowns, so Maxwell alone does not fix the split. F2 pinned the horizontal
    coefficient against xtrack and MAD-X; the vertical one then follows with no further
    freedom, which is why matching it is a confirmation and not a fit.

    This test does the algebra with sympy and then checks the **kick actually applied**
    against ``-L grad H_3``, ``H_3 = -psi_3``. That last step is the one no other gate in
    the package makes: ``tests/analytic/test_dipole_chromaticity.py`` derives the same
    ``psi_3`` for the *chromaticity integral*, and until L4 nothing tracked it.
    """
    x, y, h, k1, c1, c2 = sp.symbols("x y h k1 c1 c2", real=True)
    psi3 = c1 * x**3 + c2 * x * y**2
    # Maxwell in the curved frame: the transverse Laplacian of the potential picks up the
    # metric's h, and the 3rd-order part must cancel the k1 term's divergence.
    maxwell = sp.Eq(6 * c1 + 2 * c2 + h * k1, 0)
    horizontal = sp.Eq(c1, -h * k1 / 3)  # F2's xtrack/MAD-X-pinned coefficient
    sol = sp.solve([maxwell, horizontal], [c1, c2], dict=True)[0]
    assert sp.simplify(sol[c1] - (-h * k1 / 3)) == 0
    assert sp.simplify(sol[c2] - (+h * k1 / 2)) == 0  # follows; not fitted

    # The kick the element applies IS -L grad(-psi_3), with the derived coefficients.
    psi3_solved = psi3.subs(sol)
    want_px = sp.lambdify((x, y, h, k1), sp.diff(psi3_solved, x), "numpy")
    want_py = sp.lambdify((x, y, h, k1), sp.diff(psi3_solved, y), "numpy")

    st = np.array([3.0e-3, 0.0, -2.0e-3, 0.0, 0.0, 0.0])
    hv, k1v, length = ANGLE / L_B, K1, L_B
    got = curvature_sextupole_kick(st, hv * k1v * length)
    assert got[PX] == pytest.approx(length * want_px(st[X], st[Y], hv, k1v), rel=1e-14)
    assert got[PY] == pytest.approx(length * want_py(st[X], st[Y], hv, k1v), rel=1e-14)
    # ...and it moves nothing else, which is what makes it a momentum kick.
    for idx in (X, Y, ZETA, DELTA):
        assert got[idx] == st[idx]


def test_the_kick_is_momentum_independent_like_a_thin_quadrupole(ref: ReferenceParticle) -> None:
    """No ``1/(1+delta)``: a field changes every particle's *momentum* by the same amount.

    The same fact that makes a
    :class:`~accsim.elements.quadrupole.ThinQuadrupole` chromatically exact on its own. It
    is worth a gate because the surrounding ``mat`` halves are full of ``1/(1+delta)``
    factors, and adding one here would look consistent and be wrong.
    """
    base = np.array([4.0e-3, 0.0, -3.0e-3, 0.0, 0.0, 0.0])
    hk1l = (ANGLE / L_B) * K1 * L_B
    kicked = curvature_sextupole_kick(base, hk1l)
    for delta in (0.0, 1.0e-2, 5.0e-2, -3.0e-2):
        st = base.copy()
        st[DELTA] = delta
        got = curvature_sextupole_kick(st, hk1l)
        assert got[PX] == pytest.approx(kicked[PX], rel=0.0, abs=1e-18)
        assert got[PY] == pytest.approx(kicked[PY], rel=0.0, abs=1e-18)


@pytest.mark.parametrize(("x0", "y0"), [(2.0e-3, 0.0), (0.0, 3.0e-3), (2.0e-3, 3.0e-3)])
def test_feed_down_reproduces_f2s_generator_and_its_two_to_minus_one(
    ref: ReferenceParticle, x0: float, y0: float
) -> None:
    r"""**The gate that pins the coefficients**, and the only structural one that can.

    Symplecticity cannot see them (any ``(c1, c2)`` gives a gradient kick), Maxwell alone
    cannot (one equation, two unknowns), and the origin-Jacobian identity cannot (the kick
    is quadratic, so it contributes nothing there). What discriminates is **feed-down**:
    on an orbit the cubic potential linearises into a gradient, and F2 derived exactly
    which one from the exact Hamiltonian (``tests/analytic/test_dipole_chromaticity.py``,
    the ``a21``/``a43`` entries of its off-momentum generator):

        a21 gains -2 h k1 x_0        a43 gains + h k1 x_0
        a23 gains + h k1 y_0         a41 gains + h k1 y_0

    Four numbers from two coefficients, so a uniform mis-scale is caught by the size and a
    wrong split by the ``2:-1``; the vertical orbit is what makes the plane-coupling pair
    visible at all, and a bend on a vertical orbit is where L3 found the package's newest
    coupling source. The measurement is ``(J(orbit) - J(0))/ds`` on a short slice, taken
    to the ``ds -> 0`` limit because a finite slice also transports.
    """
    h = 0.4
    want = {
        (PX, X): -2.0 * h * K1 * x0,
        (PY, Y): +h * K1 * x0,
        (PX, Y): +h * K1 * y0,
        (PY, X): +h * K1 * y0,
    }
    s0 = np.array([x0, 0.0, y0, 0.0, 0.0, 0.0])
    errs = []
    for ds in (0.1, 0.05, 0.025):
        elem = Dipole(ds, h * ds, k1=K1)
        d_j = (
            jacobian(lambda s, e=elem: e.track(s, ref), s0, 1e-7)
            - jacobian(lambda s, e=elem: e.track(s, ref), np.zeros(DIM), 1e-7)
        ) / ds
        errs.append(max(abs(d_j[idx] - value) for idx, value in want.items()))
    scale = max(abs(v) for v in want.values())
    assert errs[-1] < 3.0e-4 * scale  # the limit is reached...
    for coarse, fine in zip(errs[:-1], errs[1:], strict=True):
        assert fine < 0.75 * coarse  # ...and approached, so it is a limit and not luck


def test_a_straight_gradient_magnet_has_no_curvature_sextupole_at_all(
    ref: ReferenceParticle,
) -> None:
    """``h = 0`` removes the term entirely — it is *curvature*-sextupole, not gradient.

    Stated because the 56% control in ``test_exact_quadrupole.py`` uses a zero-angle
    gradient bend, so that test measures the ``mat`` half **only** and can say nothing
    about the Maxwell half. Naming the reason here is what stops the Maxwell half being
    read as gated by it.
    """
    assert np.array_equal(curvature_sextupole_kick(STATE, 0.0), STATE)
    h = 0.0
    straight = Dipole(2.0, 0.0, k1=K1)
    np.testing.assert_allclose(
        straight.track(STATE_BIG, ref), expanded_cfd_map(STATE_BIG, 2.0, h, K1, ref), atol=1e-18
    )


def test_the_split_converges_to_the_equations_of_motion_at_second_order(
    ref: ReferenceParticle,
) -> None:
    r"""One centred kick is a Strang splitting: **second order**, and the order is pinned.

    ``mat(L/2) . kick(L) . mat(L/2)`` is exactly what xtrack's ``mat-kick-mat`` does with
    one uniform kick, and it is an approximation to the flow of the *whole* paraxial
    Hamiltonian including the cubic term. Slicing the magnet into ``N`` pieces must reduce
    the error as ``1/N^2``; a first-order splitting (kick at one end) or a mis-weighted
    one would show up as ``1/N``, and no amount of agreement at ``N = 1`` would reveal it.

    accsim does **not** offer a slice-count knob: an element is one ``mat-kick-mat``, and a
    user who wants the flow more accurately slices the lattice, which is what this test
    does. That keeps the element's Jacobian identity exact and puts the accuracy/cost
    trade where the user can see it.
    """
    h = ANGLE / L_B
    errs = []
    for n in (1, 2, 4, 8, 16):
        worst = 0.0
        for st in (STATE, STATE_BIG):
            s = st.copy()
            for _ in range(n):
                s = Dipole(L_B / n, ANGLE / n, k1=K1).track(s, ref)
            worst = max(worst, float(np.abs(s - _ode_map(st, L_B, h, K1, ref, cubic=True)).max()))
        errs.append(worst)
    assert errs[0] < 1.0e-4  # already small at N = 1
    for coarse, fine in zip(errs[:-1], errs[1:], strict=True):
        assert coarse / fine == pytest.approx(4.0, rel=0.1)


# --------------------------------------------------------------------------
# 4. Symplecticity — structural, and blind to everything above
# --------------------------------------------------------------------------


def test_symplectic_in_the_canonical_pair_and_rejected_by_the_other_one(
    ref: ReferenceParticle,
) -> None:
    r"""The exact flow of an approximate Hamiltonian, so it is *exactly* symplectic.

    That is the property worth having and the reason a truncated map is preferable to a
    more accurate non-symplectic one: it is safe to iterate for a million turns. It holds
    factor by factor — each ``mat`` half is a Hamiltonian flow and the kick is the
    gradient of a potential — so the composition is too, edges included (a thin edge kick
    has unit determinant in each plane).

    The **canonical** check is the right one, exactly as for L2's quadrupole and L3's
    bend: ``(zeta, delta)`` is not a canonical pair, so plain
    :func:`~accsim.symplectic.is_symplectic_map` rejects this correct map. A
    combined-function bend has just **moved** from the first group to the second, which is
    a live change to ``tests/analytic/test_roll.py`` and not a formality — before L4 its
    ``track`` was its ``matrix`` and the plain check passed.

    This test is also the demonstration that symplecticity is **blind** to everything
    section 3 gates: it would pass just as well with the Maxwell coefficients doubled, or
    swapped between the planes.
    """
    for elem in (
        Dipole(L_B, ANGLE, k1=K1),
        Dipole(L_B, ANGLE, k1=-K1, e1=0.1, e2=0.15),
        Dipole(2.0, 0.0, k1=0.7),
    ):
        assert is_symplectic_map_canonical(lambda s, e=elem: e.track(s, ref), STATE, ref)

    cf = Dipole(L_B, ANGLE, k1=K1)
    assert not is_symplectic_map(lambda s: cf.track(s, ref), STATE, atol=1e-8)

    # ...and the blindness, run rather than asserted in prose.
    class _WrongMaxwell(Dipole):
        def _track_body(self, state: np.ndarray, r: ReferenceParticle) -> np.ndarray:
            half = 0.5 * self.length
            st = expanded_cfd_map(state, half, self.curvature, self.k1, r)
            st = curvature_sextupole_kick(st, 2.0 * self.curvature * self.k1 * self.length)
            return expanded_cfd_map(st, half, self.curvature, self.k1, r)

    wrong = _WrongMaxwell(L_B, ANGLE, k1=K1)
    assert is_symplectic_map_canonical(lambda s: wrong.track(s, ref), STATE, ref)
    assert np.abs(wrong.track(STATE, ref) - cf.track(STATE, ref)).max() > 1.0e-8


# --------------------------------------------------------------------------
# 5. The model boundary: what the expanded family drops, in closed form
# --------------------------------------------------------------------------


class _LinearBend(Dipole):
    """The pre-L4 combined-function bend: ``track()`` is ``matrix()``."""

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        return Element._track_body(self, state, ref)


def _tracked_chromaticity(lattice: Lattice, h: float = 1.0e-5) -> tuple[float, float]:
    """``dQ/ddelta`` by tracking alone. Shares no code with any Twiss integral."""
    qx_p, qy_p = tunes_on_orbit(lattice, delta=+h)
    qx_m, qy_m = tunes_on_orbit(lattice, delta=-h)
    return (qx_p - qx_m) / (2.0 * h), (qy_p - qy_m) / (2.0 * h)


def _cf_ring(ref: ReferenceParticle, n_slices: int = 1, kf: float = 0.35) -> Lattice:
    """Eight alternating-gradient sector bends and eight drifts — a real AG arc."""
    els: list = []
    for i in range(8):
        k1 = +kf if i % 2 == 0 else -kf
        els += [
            Dipole(1.0 / n_slices, (2.0 * np.pi / 16) / n_slices, k1=k1) for _ in range(n_slices)
        ]
        els.append(Drift(0.4))
    return Lattice(els, ref)


def _chromaticity_without_the_metric_group(
    lattice: Lattice, slices: int = 512
) -> tuple[float, float]:
    r"""F2's natural chromaticity with the curvilinear-metric group removed.

    F2's per-length integrand inside a bend body is

        integrand_x = -beta_x (k1 + h^2) + h (gamma_x D_x - 2 alpha_x D_px) + 2 h k1 beta_x D_x
        integrand_y = +beta_y k1         +    gamma_y h D_x                 -   h k1 beta_y D_x

    and the **middle** term of each is the curvilinear metric: it is the ``(1 + h x)``
    factor of ``x' = px(1+hx)/pz`` evaluated on the dispersed orbit ``x = D_x delta``.
    That is precisely the factor the expanded family drops from the transverse map (it
    keeps it only in the path length), so dropping it here should predict what tracking
    sees. Written out in full rather than imported, so this gate does not depend on a
    private symbol staying where it is.

    The ring is built from bends and drifts only, and a drift contributes nothing to
    either integrand, so this needs no element dispatch beyond the bends.
    """
    ref = lattice.ref
    tw = closed_twiss(lattice)
    bx, ax, by, ay = tw.beta_x, tw.alpha_x, tw.beta_y, tw.alpha_y
    dx, dpx = tw.disp_x, tw.disp_px
    state = np.array([bx, ax, by, ay, dx, dpx])
    acc_x = acc_y = 0.0

    def advance(state: np.ndarray, M: np.ndarray) -> np.ndarray:
        b_x, a_x, b_y, a_y, d_x, d_px = state
        c, s = M[X, X], M[X, PX]
        cp, sp = M[PX, X], M[PX, PX]
        nb_x = c * c * b_x - 2.0 * c * s * a_x + s * s * (1.0 + a_x * a_x) / b_x
        na_x = -(c * cp * b_x - (c * sp + s * cp) * a_x + s * sp * (1.0 + a_x * a_x) / b_x)
        c, s = M[Y, Y], M[Y, PY]
        cp, sp = M[PY, Y], M[PY, PY]
        nb_y = c * c * b_y - 2.0 * c * s * a_y + s * s * (1.0 + a_y * a_y) / b_y
        na_y = -(c * cp * b_y - (c * sp + s * cp) * a_y + s * sp * (1.0 + a_y * a_y) / b_y)
        nd_x = M[X, X] * d_x + M[X, PX] * d_px + M[X, DELTA]
        nd_px = M[PX, X] * d_x + M[PX, PX] * d_px + M[PX, DELTA]
        return np.array([nb_x, na_x, nb_y, na_y, nd_x, nd_px])

    for elem in lattice.elements:
        if not isinstance(elem, Dipole):
            state = advance(state, elem.matrix(ref))
            continue
        ds = elem.length / slices
        sub = Dipole(ds, elem.curvature * ds, k1=elem.k1).matrix(ref)
        for i in range(slices + 1):
            b_x, a_x, b_y, a_y, d_x, _d_px = state
            weight = 0.5 if i in (0, slices) else 1.0
            acc_x += (
                weight
                * ds
                * (
                    -b_x * (elem.k1 + elem.curvature**2)
                    + 2.0 * elem.curvature * elem.k1 * b_x * d_x
                )
            )
            acc_y += weight * ds * (b_y * elem.k1 - elem.curvature * elem.k1 * b_y * d_x)
            if i < slices:
                state = advance(state, sub)
    return acc_x / (4.0 * np.pi), acc_y / (4.0 * np.pi)


def test_tracking_a_bending_gradient_magnet_lands_on_f2_minus_the_metric_group(
    ref: ReferenceParticle,
) -> None:
    r"""**The milestone's most useful result, and it is a limit rather than a match.**

    The expanded family solves ``x' = px/(1+delta)``, where the exact curvilinear equation
    is ``x' = px (1 + h x)/p_z``. The dropped ``(1 + h x)`` is not a small tidying-up: on
    the dispersed orbit ``x = D_x delta`` it *is* F2's
    ``h(gamma_x D_x - 2 alpha_x D_px)`` / ``gamma_y h D_x`` group, and that group is what
    largely **cancels** the geometric ``-beta_x h^2`` focusing (CONVENTIONS ->
    *Dipole chromaticity*: a pure sector bend contributes almost nothing, and a partial fix
    is worse than none).

    So tracking a bending combined-function magnet does **not** converge to
    :func:`~accsim.natural_chromaticity`. It converges to F2 *minus* that group — which
    this test computes in closed form and confirms to better than 0.2% at 16 slices,
    approaching as ``1/N^2`` (the splitting error, which is L4's and does vanish). The
    remainder is the model family's and does not.

    That makes the boundary a number rather than a caveat, and it is confirmed from the
    other side in the reference suite: xtrack's own converged ``mat-kick-mat`` lands on
    the same value, and its exact families land on F2.

    ⚠️ The practical consequence, and the reason this is not buried in a docstring: on a
    ring of *bending* gradient magnets the tracked chromaticity can be **further** from the
    truth after L4 than the pre-L4 blind map was, because the blind map contributed
    nothing at all where this one contributes an uncancelled ``-beta_x h^2``. That is the
    F1 failure mode, and it is the price of the expanded family. ``natural_chromaticity``
    remains the deliverable and is untouched.
    """
    full = natural_chromaticity(_cf_ring(ref), slices=1024)
    partial = _chromaticity_without_the_metric_group(_cf_ring(ref))

    # The two really are different — otherwise this test would be measuring nothing.
    assert abs(full[0] - partial[0]) > 0.2
    assert abs(full[1] - partial[1]) > 0.1

    errs = []
    for n in (1, 2, 4, 8, 16):
        tracked = _tracked_chromaticity(_cf_ring(ref, n_slices=n))
        errs.append(max(abs(tracked[0] - partial[0]), abs(tracked[1] - partial[1])))
        if n == 16:
            assert tracked[0] == pytest.approx(partial[0], rel=2e-3)
            assert tracked[1] == pytest.approx(partial[1], rel=2e-3)
            # ...and it is nowhere near the full answer, which is the whole point.
            assert abs(tracked[0] - full[0]) > 0.2
    for coarse, fine in zip(errs[:-1], errs[1:], strict=True):
        assert coarse / fine == pytest.approx(4.0, rel=0.25)  # the splitting error, 1/N^2


def test_the_gradient_bend_is_no_longer_chromatically_ideal(ref: ReferenceParticle) -> None:
    """The plain statement of what changed, against the map it replaced.

    Before L4 a combined-function bend's ``track`` was its ``matrix``, so it was *exactly*
    momentum-independent — a magnet that focuses every particle identically, which is not
    what a magnet is. The gate is that the tracked chromaticity **moves**, and moves by
    something of the size of the gradient's own share; where it moves *to* is the previous
    test's business.
    """
    lat, blind = (
        _cf_ring(ref),
        Lattice(
            [
                _LinearBend(e.length, e.angle, k1=e.k1) if isinstance(e, Dipole) else e
                for e in _cf_ring(ref).elements
            ],
            ref,
        ),
    )
    # Identical design optics: same matrices, so this is a controlled experiment.
    np.testing.assert_allclose(blind.one_turn_matrix(), lat.one_turn_matrix(), atol=1e-15)

    moved = abs(_tracked_chromaticity(lat)[0] - _tracked_chromaticity(blind)[0])
    assert moved > 0.02


# --------------------------------------------------------------------------
# 6. The discontinuity at k1 = 0, measured
# --------------------------------------------------------------------------


def test_a_bend_is_discontinuous_in_k1_and_the_gap_is_second_order_not_third(
    ref: ReferenceParticle,
) -> None:
    r"""**L2 refused a discontinuity in ``k1``; L3 forced one, and here is its size.**

    ``Dipole(L, theta, 0)`` takes L3's exact circle; ``Dipole(L, theta, eps)`` takes this
    expanded map however small ``eps`` is. So the map is genuinely discontinuous at
    ``k1 = 0`` for a *bending* magnet, and the jump does not shrink with ``eps`` — it
    converges to a fixed limit, which is asserted first.

    L3 argued the split is *forced* rather than chosen (with ``k1 = 0``, ``p_y`` is
    conserved and the vertical equation is a quadrature; with ``k1 != 0`` it is an ODE with
    an ``s``-dependent coefficient), so the sub-case has a strictly better map that the
    general family provably cannot express. That justifies the discontinuity; it does not
    excuse leaving its size unmeasured.

    **The obvious guess about its order is wrong**, which is why the order is probed per
    coordinate rather than assumed. L2's residual against the exact drift is
    ``O(angle^3)``; this one is **quadratic** — a factor of two in the amplitude gives a
    factor of four, not eight — because a bend loses *two* things at once, and both are
    second order:

    - the **expanded square root**, which for a bend enters ``px' = h p_z - h`` already at
      ``O(p^2)`` rather than at third order as it does in a straight quadrupole. The exact
      bend's ``px`` therefore falls short of the expanded one's by ``h px^2 L/2``, probed
      at ``x = 0`` and matched below;
    - the **dropped metric factor** ``(1 + h x)``, whose signature is a *bilinear* ``x px``
      term that neither coordinate alone produces, and which is the very term the
      chromaticity gate above names. The exact map's extra ``h x px`` in ``x'`` makes its
      ``x`` the larger, so the mixed second difference is ``-h x px L``, matched below.

    Two residuals, one of them this milestone's own model boundary — which makes the whole
    thing one story rather than two.
    """
    h = ANGLE / L_B

    def gap(state: np.ndarray, eps: float = 1.0e-9) -> np.ndarray:
        return Dipole(L_B, ANGLE, k1=eps).track(state, ref) - Dipole(L_B, ANGLE).track(state, ref)

    # 1. It is a limit, not a vanishing: the jump converges as eps -> 0.
    sizes = [float(np.abs(gap(STATE, eps)).max()) for eps in (1e-4, 1e-6, 1e-8)]
    assert sizes[-1] == pytest.approx(1.821e-5, rel=1e-2)
    assert sizes[-1] == pytest.approx(sizes[-2], rel=1e-3)  # converged, not shrinking

    # 2. Quadratic in the amplitude, not cubic. At delta = 0, so that the (also
    # quadratic) delta^2 share of the gap cannot masquerade as an amplitude scaling.
    transverse = np.array([3.0e-3, 8.0e-3, -2.0e-3, 5.0e-3, 0.0, 0.0])
    worst = [float(np.abs(gap(transverse * s)).max()) for s in (1.0, 0.5, 0.25)]
    for coarse, fine in zip(worst[:-1], worst[1:], strict=True):
        assert coarse / fine == pytest.approx(4.0, rel=0.1)  # NOT 8

    # 3. The angle half: the expanded square root, at x = 0.
    px = 8.0e-3
    only_px = gap(np.array([0.0, px, 0.0, 0.0, 0.0, 0.0]))
    assert only_px[PX] == pytest.approx(+h * px * px * L_B / 2.0, rel=0.05)

    # 4. The metric half: the bilinear x*px term, which neither coordinate makes alone.
    x0 = 3.0e-3
    both = np.array([x0, px, 0.0, 0.0, 0.0, 0.0])
    mixed = (
        gap(both)
        - gap(np.array([x0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        - gap(np.array([0.0, px, 0.0, 0.0, 0.0, 0.0]))
    )
    assert mixed[X] == pytest.approx(-h * x0 * px * L_B, rel=0.15)
