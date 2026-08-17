r"""L3 — the dipole's exact map, and the close of K2's account.

A uniform field has a closed-form flow, and it is a **circle**: a particle of momentum
``1 + delta`` in ``B0 = P0 h / q`` moves, in projection onto the bend plane, on a circle
of radius ``r = p_perp / h`` with ``p_perp = sqrt((1+delta)^2 - py^2)``. Everything the
map does is that circle meeting the exit face. So unlike L2's quadrupole — where the
square root and the quadratic potential do not commute and no closed form exists at all
— this map is exact in the **angles as well as in** ``delta``, and the discriminating
gate is both at once.

The gate shape neither L1 nor L2 prescribed, and it is an *identity*: the map is
re-derived here from plane geometry (:func:`_circle_map`), sharing no arithmetic with
the implementation, and the two agree to ``1e-15`` at bend angles up to ``1.5 rad`` and
``delta`` up to ``0.3``, where the linear matrix is wrong by ``2.3e-2``. The exact
Hamiltonian is checked to be an invariant, which no wrong map of this shape would be.

**What it closes.** K2 measured a gap it could not represent and wrote its specification
as ``Delta d_y = p_y L (h <D_x> - 1)`` — a ``-1`` from ``1/pz`` and a ``+h <D_x>`` from
the extra arc a dispersed particle travels on the outside of a bend. L1 delivered the
first half on a bend-free ring; this delivers the second. It also turns out that
formula was **incomplete**, in a way only an exact bend could show: see
:func:`_first_order_dispersion` and the test that uses it.

The combined-function bend is deliberately *not* touched — there is no closed form for
a curved quadrupole either, and the reason the split is forced rather than chosen is
recorded on :class:`~accsim.elements.dipole.Dipole`.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Corrector,
    Dipole,
    Drift,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    closed_twiss,
    is_symplectic_map,
    is_symplectic_map_canonical,
    jacobian,
    natural_chromaticity,
    tunes_on_orbit,
)
from accsim.coords import DELTA, DIM, PX, PY, ZETA, X, Y
from accsim.elements.element import Element
from accsim.orbit import (
    closed_orbit_nonlinear,
    linearised_element_maps,
    linearised_one_turn_map,
    propagate_orbit_nonlinear,
)
from accsim.twiss import _matched_dispersion, coupled_twiss_on_orbit

L_B = 1.0
ANGLE = 2.0 * np.pi / 16.0  # 0.3927 rad: sin, theta and tan differ in the third digit
N_CELLS = 8
F_FOCAL = 2.2
STEER = 1.0e-4  # a vertical steerer, to give the ring an orbit *angle*

# Every coordinate nonzero, so no term can hide behind a zero.
STATE = np.array([3.0e-3, 8.0e-3, -2.0e-3, 5.0e-3, 1.0e-3, 2.0e-3])


@pytest.fixture
def ref() -> ReferenceParticle:
    """gamma0 = 20: a real machine, and the ``1/gamma0^2`` slip still visible."""
    from accsim import PROTON_MASS_EV

    return ReferenceParticle.from_gamma(PROTON_MASS_EV, 20.0)


# --------------------------------------------------------------------------
# The independent derivation: a circle meeting a plane
# --------------------------------------------------------------------------


def _circle_map(state: np.ndarray, length: float, h: float, ref: ReferenceParticle) -> np.ndarray:
    r"""The exact bend map from plane geometry, sharing no arithmetic with the package.

    In a Cartesian ``(x, z)`` frame pinned to the entrance face, with the entry point at
    ``(x0, 0)`` and ``z`` along the reference direction:

    - the trajectory projects onto a circle of radius ``r = p_perp / h`` whose centre is
      ``(x0 - pz0/h, px0/h)`` — one radius along the inward normal to the momentum;
    - the exit face passes through ``rho (cos t - 1, sin t)`` with normal
      ``(-sin t, cos t)``, ``t = h L``;
    - ``x_out`` is where the circle meets that face, and ``px_out`` is ``p_perp`` along
      the exit transverse direction;
    - the swept angle ``phi`` gives everything else: ``y`` advances by ``py phi / h`` and
      the three-dimensional path is ``(1 + delta) phi / h``, because a vertical velocity
      is constant and the arc length in the bend plane is ``r phi``.

    Nothing here is a transcription of the map under test: no ``sinc``, no rationalised
    ``pz - 1``, no arcsine difference. It is the geometry the map is *of*.
    """
    x0, px0, y0, py, zeta0, delta = (float(v) for v in state)
    rho, t = 1.0 / h, h * length
    p_perp = np.sqrt((1.0 + delta) ** 2 - py**2)
    pz0 = np.sqrt(p_perp**2 - px0**2)
    r = p_perp / h

    entry = np.array([x0, 0.0])
    centre = np.array([x0 - pz0 / h, px0 / h])
    face_point = np.array([rho * (np.cos(t) - 1.0), rho * np.sin(t)])
    face_x = np.array([np.cos(t), np.sin(t)])  # outward transverse direction at the exit

    d = face_point - centre
    along = float(np.dot(d, face_x))
    x_out = -along + np.sqrt(along**2 - float(np.dot(d, d)) + r**2)  # the near root
    exit_point = face_point + x_out * face_x

    inward = (centre - exit_point) / r
    px_out = p_perp * float(np.dot(np.array([inward[1], -inward[0]]), face_x))

    v0, v1 = entry - centre, exit_point - centre
    phi = float(np.arctan2(v0[0] * v1[1] - v0[1] * v1[0], float(np.dot(v0, v1))))

    E_over_E0 = np.hypot(ref.momentum_eV * (1.0 + delta), ref.mass_eV) / ref.total_energy_eV
    path3d = (1.0 + delta) * phi / h
    return np.array(
        [
            x_out,
            px_out,
            y0 + py * phi / h,
            py,
            zeta0 + length - path3d * E_over_E0 / (1.0 + delta),
            delta,
        ]
    )


def _hamiltonian(state: np.ndarray, h: float) -> float:
    """``H = -(1+hx) sqrt((1+d)^2 - px^2 - py^2) + h(x + h x^2/2)``, the exact invariant.

    The potential is F2's ``psi`` at ``k1 = 0``: the ``-h^2 x^2 / 2`` is the curvilinear
    metric correction, and without it the on-momentum focusing comes out ``2h^2``
    instead of the validated ``h^2`` (``docs/CONVENTIONS.md`` -> *Dipole chromaticity*).
    """
    x, px, _y, py, _zeta, delta = (float(v) for v in state)
    return -(1.0 + h * x) * np.sqrt((1.0 + delta) ** 2 - px**2 - py**2) + h * (x + h * x * x / 2.0)


# --------------------------------------------------------------------------
# 1. The map, against geometry rather than against a transcription
# --------------------------------------------------------------------------


def test_the_map_is_the_circle_that_plane_geometry_says_it_is(ref: ReferenceParticle) -> None:
    r"""The identity, at large angle **and** large ``delta`` together.

    L1 discriminated at large angles and L2 at large ``delta``; a bend needs both,
    because it is exact in both and the two axes fail differently. A map that expanded
    the square root (MAD-X's and xtrack's ``mat-kick-mat`` combined-function form, which
    is what a *curved quadrupole* is stuck with) would agree at small amplitude and part
    company at ``O(angle^3)``; a map that mishandled the rigidity would agree at
    ``delta = 0`` and part company at ``O(delta)``. The grid below closes both doors,
    and the linear matrix — the thing this replaces — is off by up to ``2.3e-2`` on the
    same states, so the comparison has teeth rather than merely a tolerance.
    """
    for angle in (0.3, 0.8, 1.5):
        h = angle / L_B
        for delta in (0.0, 0.02, 0.10, 0.30):
            state = STATE.copy()
            state[DELTA] = delta
            bend = Dipole(L_B, angle)
            got = bend.track(state, ref)
            want = _circle_map(state, L_B, h, ref)
            np.testing.assert_allclose(got, want, rtol=0, atol=5e-15)

            # ...and it is not the linear map, by four to five orders.
            linear = bend.matrix(ref) @ state
            assert np.abs(linear - want).max() > 3.0e-5


def test_the_map_is_the_same_circle_for_a_whole_bunch_at_once(ref: ReferenceParticle) -> None:
    """Vectorisation is a property of the map, not of the loop that calls it.

    A ``(6, n)`` bunch with a momentum spread takes the same code path as a ``(6,)``
    state, and every particle in it must get its *own* radius — the failure mode being a
    map that quietly uses one particle's ``delta`` for all of them.
    """
    rng = np.random.default_rng(19)
    bunch = rng.normal(scale=3.0e-3, size=(DIM, 24))
    bunch[DELTA] = rng.uniform(-0.05, 0.05, size=24)
    bend = Dipole(L_B, ANGLE)

    together = bend.track(bunch, ref)
    for i in range(bunch.shape[1]):
        np.testing.assert_allclose(
            together[:, i], _circle_map(bunch[:, i], L_B, ANGLE / L_B, ref), rtol=0, atol=5e-15
        )


def test_the_exact_hamiltonian_is_an_invariant_of_the_map(ref: ReferenceParticle) -> None:
    r"""``H`` is ``s``-independent, so the true map conserves it — and this one does.

    A check the map cannot pass by accident and that needs no reference implementation
    at all: it uses only the Hamiltonian the physics is written from. A map with a wrong
    coefficient anywhere — the metric term, the rigidity, the path length — moves ``H``
    at the order of the error. Measured at machine precision across amplitudes and two
    very different bend angles.
    """
    rng = np.random.default_rng(23)
    for angle in (0.3, 1.0):
        h = angle / L_B
        bend = Dipole(L_B, angle)
        worst = 0.0
        for _ in range(200):
            state = rng.normal(scale=[2e-3, 2e-3, 2e-3, 2e-3, 1e-3, 5e-3])
            out = bend.track(state, ref)
            worst = max(worst, abs(_hamiltonian(out, h) - _hamiltonian(state, h)))
        assert worst < 5.0e-15


def test_the_origin_jacobian_is_the_linear_matrix_entry_for_entry(ref: ReferenceParticle) -> None:
    r"""The invariant that bounds this milestone as it bounded L1 and L2.

    ``matrix()`` must be the **exact** Jacobian of ``track()`` at the origin, or design
    optics and the tracked machine describe different rings. It is what rules out the
    slicing family — a sliced map's origin Jacobian is the sliced *approximation* to the
    sector block, and every "tracking agrees with the matrix on the design orbit" gate in
    the package would have moved.

    The number is also the whole of this milestone's numerical work. Written as xtrack
    writes it, ``x = (pz_out h - dpx/ds - k)/(h k)`` forms an answer of size ``x`` out of
    a numerator of size ``h``, and the same difference comes out at ``3.2e-9`` —
    *degrading* as the step shrinks, which is the signature of cancellation rather than
    truncation. Rearranged so nothing of size one is ever subtracted, it improves with
    the step, as a truncation error must. Both halves are asserted: the size, and the
    direction it moves in.
    """
    bend = Dipole(L_B, ANGLE)
    want = bend.matrix(ref)

    coarse = jacobian(lambda st: bend.track(st, ref), np.zeros(DIM), step=1.0e-6)
    fine = jacobian(lambda st: bend.track(st, ref), np.zeros(DIM), step=1.0e-7)
    assert np.abs(fine - want).max() < 1.0e-13
    # It is truncation, not round-off: ten times the step is a hundred times the error.
    assert np.abs(coarse - want).max() > 10.0 * np.abs(fine - want).max()

    # The design orbit is a fixed point exactly, not to a tolerance.
    assert np.abs(bend.track(np.zeros(DIM), ref)).max() == 0.0


def test_the_straight_limit_is_the_exact_drift_and_needs_no_branch(ref: ReferenceParticle) -> None:
    r"""No division by the curvature survives, so ``h -> 0`` is continuous.

    The map is written entirely through ``sinc(theta)`` and ``(1 - cos theta)/h``, both
    of which are analytic at ``h = 0``, so the straight limit is reached rather than
    special-cased. Two consequences:

    - a **zero-angle** ``Dipole`` *is* a :class:`~accsim.elements.drift.Drift` — the same
      map by two arithmetic routes, agreeing to a few ulp. That removes a documented
      inconsistency (L1's "a zero-strength magnet is a drift in ``matrix`` and no longer
      in ``track``") for this element, and it is why L2's 48%-vs-100% control had to be
      re-baselined: its straight ``Dipole`` stand-in is no longer chromatically blind;
    - a **weak** bend degrades gracefully instead of falling off a cliff. At
      ``h = 1e-4`` the origin Jacobian is still good to ``1e-12``; the transcribed form
      is ``1.4e-5`` there, which is what "no branch" has to mean to be worth anything.
    """
    rng = np.random.default_rng(29)
    straight, drift = Dipole(L_B, 0.0), Drift(L_B)
    worst = 0.0
    for _ in range(200):
        state = rng.normal(scale=1.0e-3, size=DIM)
        worst = max(worst, np.abs(straight.track(state, ref) - drift.track(state, ref)).max())
    assert worst < 1.0e-17  # a few ulp of a 1e-3 coordinate, not a modelling difference

    for h in (1.0e-2, 1.0e-4):
        weak = Dipole(L_B, h * L_B)
        got = jacobian(lambda st, e=weak: e.track(st, ref), np.zeros(DIM), step=1.0e-7)
        assert np.abs(got - weak.matrix(ref)).max() < 1.0e-12


def test_symplectic_in_the_canonical_pair_and_rejected_by_the_other_one(
    ref: ReferenceParticle,
) -> None:
    r"""The exact flow of the exact Hamiltonian, checked where the check is valid.

    ``(zeta, delta)`` is not a canonically conjugate pair, so
    :func:`~accsim.symplectic.is_symplectic_map` **rejects this correct map**, exactly as
    it rejects the exact drift (L1) and the momentum-dependent quadrupole (L2). Asserting
    the rejection matters as much as asserting the acceptance: L2 found the same check
    silently *accepting* a correct map at small amplitude, so neither verdict from it
    means anything on its own.
    """
    bend = Dipole(L_B, ANGLE)
    assert is_symplectic_map_canonical(lambda st: bend.track(st, ref), STATE, ref)
    assert not is_symplectic_map(lambda st: bend.track(st, ref), STATE)

    # The straight limit is symplectic too — the same statement L1 makes about the drift.
    straight = Dipole(L_B, 0.0)
    assert is_symplectic_map_canonical(lambda st: straight.track(st, ref), STATE, ref)


# --------------------------------------------------------------------------
# 2. The first-order content, derived — and the plane symmetry that is false
# --------------------------------------------------------------------------


def test_the_horizontal_response_is_not_the_plane_swap_of_the_vertical_one() -> None:
    r"""Derived in sympy from the equations of motion, because the planes differ.

    In the curved frame the exact equations of motion are

        x' = (1 + h x) px / pz,     px' = h (pz - 1 - h x),
        y' = (1 + h x) py / pz,     py' = 0.

    The vertical one is a plain **quadrature**: ``py`` is conserved, so integrating the
    linearised planar motion ``x(s) = x0 cos + px0 sin/h + delta (1-cos)/h`` straight
    through gives, to first order in the orbit,

        dy = py [ L + x0 sin(t) + px0 rho (1 - cos t) - delta rho sin(t) ].

    The horizontal one is **not**, because ``px`` is not conserved: the response feeds
    back through the bend's own focusing and dispersion. Collecting the ``px0 delta``
    terms gives a *driven* oscillator at twice the bend frequency,

        xi'' + h^2 xi = (3h/2) sin(2 h s),      xi(0) = 0,  xi'(0) = -1,

    whose solution is ``xi(s) = -sin(2 h s)/(2h)``, i.e. ``-rho sin(t) cos(t)`` at the
    exit — the vertical answer times an extra ``cos t``.

    **That extra cosine is the trap of this milestone.** Symmetrising the planes gives
    ``-px rho sin(t)``, which is 8% wrong at this bend angle, right at ``delta = 0``,
    right on the design orbit, and right in every design-optics gate in the package.
    The ring-level test below shows what it costs there.
    """
    s, h = sp.symbols("s h", positive=True)
    theta = sp.symbols("theta", positive=True)

    xi = -sp.sin(2 * h * s) / (2 * h)
    assert (
        sp.simplify(sp.diff(xi, s, 2) + h**2 * xi - sp.Rational(3, 2) * h * sp.sin(2 * h * s)) == 0
    )
    assert sp.simplify(xi.subs(s, 0)) == 0
    assert sp.simplify(sp.diff(xi, s).subs(s, 0)) == -1
    assert sp.simplify(xi.subs(s, theta / h) + sp.sin(theta) * sp.cos(theta) / h) == 0

    # The vertical quadrature, for the same reason and by the same route.
    x0, px0, d = sp.symbols("x0 px0 delta", real=True)
    x_of_s = x0 * sp.cos(h * s) + px0 * sp.sin(h * s) / h + d * (1 - sp.cos(h * s)) / h
    L = theta / h
    dy_over_py = sp.integrate((1 + h * x_of_s) * (1 - d), (s, 0, L)).expand()
    dy_over_py = sp.expand(dy_over_py.removeO() if hasattr(dy_over_py, "removeO") else dy_over_py)
    want = L + x0 * sp.sin(theta) + px0 * (1 - sp.cos(theta)) / h - d * sp.sin(theta) / h
    # Equal once the second-order products (x0 delta, px0 delta) are dropped.
    residual = sp.simplify(sp.expand(dy_over_py - want))
    assert sp.simplify(residual.subs(d, 0)) == 0
    assert sp.simplify(sp.diff(residual, d).subs([(x0, 0), (px0, 0), (d, 0)])) == 0


def test_the_new_jacobian_entries_have_the_derived_values_and_pair_up(
    ref: ReferenceParticle,
) -> None:
    r"""Every first-order entry, against the closed forms derived above.

    Six statements, each of which can fail on its own:

    - ``M[y, delta] = -py rho sin t`` — the ``h <D_x> - 1`` combination K2 specified,
      per element and exactly, rather than averaged over a magnet;
    - ``M[zeta, py]`` equals it, the conjugate pairing L1 found for the drift;
    - ``M[y, x] = +py sin t`` and ``M[y, px] = +py rho (1 - cos t)`` — these are ``py``
      times the bend's *own* dispersion entries ``R26`` and ``R16``, and they are
      **plane coupling**: a bend on a vertical orbit couples ``x`` into ``y``. K2's
      formula did not have them, and on a real arc they dominate;
    - each of those two has its own partner, ``M[px, py]`` and ``M[x, py]``, with the
      opposite sign;
    - ``M[x, delta] = -px rho sin t cos t``, with the extra cosine that makes the
      horizontal plane *not* the mirror of the vertical.

    Everything is asserted at two orbit amplitudes so "first order in the orbit" is
    measured rather than assumed.
    """
    bend = Dipole(L_B, ANGLE)
    rho, sin_t, cos_t = bend.rho, np.sin(ANGLE), np.cos(ANGLE)

    for scale in (1.0, 0.5):
        px_co, py_co = 3.0e-5 * scale, -4.5e-5 * scale
        orbit = np.array([0.0, px_co, 0.0, py_co, 0.0, 0.0])
        D = jacobian(lambda st: bend.track(st, ref), orbit, step=1.0e-7) - bend.matrix(ref)

        # The vertical source, and its conjugate partner.
        assert D[Y, DELTA] == pytest.approx(-py_co * rho * sin_t, rel=1e-4)
        assert D[ZETA, PY] == pytest.approx(D[Y, DELTA], rel=1e-8)

        # The coupling pair — py times the bend's own dispersion entries.
        assert D[Y, X] == pytest.approx(py_co * sin_t, rel=1e-6)
        assert D[PX, PY] == pytest.approx(-D[Y, X], rel=1e-8)
        assert D[Y, PX] == pytest.approx(py_co * rho * (1.0 - cos_t), rel=1e-3)
        assert D[X, PY] == pytest.approx(-D[Y, PX], rel=1e-3)

        # The horizontal source, with the cosine, and its own (unequal) partner.
        assert D[X, DELTA] == pytest.approx(-px_co * rho * sin_t * cos_t, rel=1e-3)
        assert D[ZETA, PX] == pytest.approx(-px_co * rho * sin_t, rel=1e-3)
        # The two are *not* equal here, unlike the drift's pair: cos t apart, measured.
        assert D[X, DELTA] / D[ZETA, PX] == pytest.approx(cos_t, rel=1e-3)


# --------------------------------------------------------------------------
# 3. The ring: what K2 asked for, and what its formula was missing
# --------------------------------------------------------------------------


def _ring(ref: ReferenceParticle, steer: float = STEER) -> Lattice:
    """Thin quadrupoles and thick bends, and **no drifts**: the bend is the only map.

    K2's own arc shape, and the point of it is isolation — a thin quadrupole's kick is
    momentum-independent and exact already (L2), so every new term in this ring belongs
    to the bend and nothing has to be subtracted off.
    """
    els: list = []
    for _ in range(N_CELLS):
        els += [
            ThinQuadrupole(0.5 / F_FOCAL),
            Dipole(L_B, ANGLE),
            ThinQuadrupole(-1.0 / F_FOCAL),
            Dipole(L_B, ANGLE),
            ThinQuadrupole(0.5 / F_FOCAL),
        ]
    if steer != 0.0:
        els.insert(1, Corrector(kick_y=steer))
    return Lattice(els, ref)


def _first_order_dispersion(lat: Lattice, ref: ReferenceParticle, *, naive: bool = False):
    """``D`` re-solved with the bend's derived first-order entries put into the matrices.

    Shares no arithmetic with :func:`~accsim.orbit.linearised_element_maps`, which gets
    there by differencing ``track()``. ``naive=True`` builds K2's specification instead —
    the two ``delta``-column terms with the planes symmetric and no coupling at all —
    which is what the package would have been asked to reproduce had this milestone
    taken that formula at its word.
    """
    orbit = propagate_orbit_nonlinear(lat)
    M = np.eye(DIM)
    for elem, o in zip(lat.elements, orbit, strict=False):
        m = elem.matrix(ref).copy()
        if isinstance(elem, Dipole) and elem.angle != 0.0:
            rho, sin_t, cos_t = elem.rho, np.sin(elem.angle), np.cos(elem.angle)
            px, py = float(o[PX]), float(o[PY])
            m[Y, DELTA] += -py * rho * sin_t
            if naive:
                m[X, DELTA] += -px * rho * sin_t
            else:
                m[ZETA, PY] += -py * rho * sin_t
                m[Y, X] += py * sin_t
                m[PX, PY] += -py * sin_t
                m[Y, PX] += py * rho * (1.0 - cos_t)
                m[X, PY] += -py * rho * (1.0 - cos_t)
                m[X, DELTA] += -px * rho * sin_t * cos_t
                m[ZETA, PX] += -px * rho * sin_t
        M = m @ M
    return _matched_dispersion(M)


def test_a_bend_on_a_vertical_orbit_makes_vertical_dispersion(ref: ReferenceParticle) -> None:
    r"""K2's other half, delivered: ``D_y`` goes from exactly ``0`` to ``8.6e-5``.

    This ring has no rolled magnet, no skew quadrupole and no drift — the only element
    with length in it is a sector bend, and the only thing wrong with the machine is a
    vertical steerer. The design optics still reports exactly ``0``, and *correctly*: the
    terms are bilinear in ``(p, delta)`` and no 6x6 can carry them. The on-orbit optics
    reports the physical answer.

    L1 made the same statement on a ring with no bend at all, where ``D_x`` is identically
    zero and the drift's ``-1`` is the whole effect. This is the complementary machine:
    the bend's ``+h <D_x>`` is switched on and ``D_x`` is 2.1 m.
    """
    lat = _ring(ref)
    co = closed_orbit_nonlinear(lat)
    assert abs(co[Y]) > 1.0e-5
    assert co[PY] == pytest.approx(-0.5 * STEER, rel=1e-6)  # symmetric ring: -kick/2

    assert closed_twiss(lat).disp_y == 0.0  # the design optics cannot see it, and says so

    tw = coupled_twiss_on_orbit(lat)
    assert tw.disp_y == pytest.approx(8.600147e-05, rel=1e-5)
    assert tw.disp_x == pytest.approx(2.1271486, rel=1e-6)  # the horizontal is untouched


def test_the_dispersion_matches_the_derived_closed_form_and_k2s_formula_does_not(
    ref: ReferenceParticle,
) -> None:
    r"""The value, against the derivation — and the correction to K2's specification.

    Putting the derived first-order entries into the linear matrices reproduces the
    tracked answer to ``5e-9`` relative, which is the finite-difference floor of the
    route it is compared against and not a physical residual. So the number is
    *understood*, not merely measured.

    **K2's specification was incomplete, and only an exact bend could show it.**
    ``Delta d_y = p_y L (h <D_x> - 1)`` is the ``delta`` column alone; the exact bend
    also couples the planes, ``M[y, x] = py sin t`` and ``M[y, px] = py rho (1 - cos t)``,
    and those entries transport the ring's 2.1 m of *horizontal* dispersion into the
    vertical. On this ring that path is the larger one: K2's formula gives ``3.3e-4``
    where the answer is ``8.6e-5``, wrong by a factor of 3.8 and in the wrong direction.
    It reproduced xtrack to 0.2% on K2's own rings, so this is a correction to the
    formula's *scope*, not a contradiction of that measurement — and the reference
    cross-check (``tests/reference/test_roll_xtrack.py``) is where the two are reconciled.
    """
    lat = _ring(ref)
    tracked = coupled_twiss_on_orbit(lat).disp_y
    derived = _first_order_dispersion(lat, ref)[2]

    assert derived == pytest.approx(tracked, rel=1.0e-7)

    naive = _first_order_dispersion(lat, ref, naive=True)[2]
    assert abs(naive / tracked) > 3.0  # the incomplete formula is not close

    # First order in the orbit: halving the steerer halves the answer.
    halved = coupled_twiss_on_orbit(_ring(ref, steer=STEER / 2)).disp_y
    assert tracked / halved == pytest.approx(2.0, rel=1e-4)


def test_the_planes_are_coupled_by_a_bend_on_a_vertical_orbit(ref: ReferenceParticle) -> None:
    r"""A consequence with no precedent in the package, stated rather than absorbed.

    Before L3 the only ways to couple the planes were a skew quadrupole (G1), a rolled
    magnet (K2) or a displaced/rolled multipole. An **upright sector bend on a vertical
    orbit** now does it too, because the vertical velocity ``dy/ds = py (1 + h x)/pz``
    depends on ``x``. It is first order in the orbit angle and it is why
    :func:`~accsim.twiss.closed_twiss_on_orbit` raises on this ring while the design
    :func:`~accsim.twiss.closed_twiss` does not — the two are describing different maps,
    and the coupled route is the one that has to be used.
    """
    from accsim.twiss import CoupledLatticeError, closed_twiss_on_orbit

    lat = _ring(ref)
    one_turn = linearised_one_turn_map(lat)
    off_block = max(
        np.abs(one_turn[np.ix_([X, PX], [Y, PY])]).max(),
        np.abs(one_turn[np.ix_([Y, PY], [X, PX])]).max(),
    )
    assert off_block > 1.0e-5

    # The design matrix has none of it: this is the exact map's doing, not the lattice's.
    design = lat.one_turn_matrix()
    assert np.abs(design[np.ix_([X, PX], [Y, PY])]).max() == 0.0

    with pytest.raises(CoupledLatticeError):
        closed_twiss_on_orbit(lat)


# --------------------------------------------------------------------------
# 4. Chromaticity: the last share, and the control that attributes it
# --------------------------------------------------------------------------


class _LinearBend(Dipole):
    """The pre-L3 bend: ``track()`` is ``matrix()``, which is what tracking used to see."""

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        return Element._track_body(self, state, ref)


def _tracked_chromaticity(lattice: Lattice, h: float = 1.0e-5) -> tuple[float, float]:
    """``dQ/ddelta`` by tracking alone. Shares no code with any Twiss integral."""
    qx_p, qy_p = tunes_on_orbit(lattice, delta=+h)
    qx_m, qy_m = tunes_on_orbit(lattice, delta=-h)
    return (qx_p - qx_m) / (2.0 * h), (qy_p - qy_m) / (2.0 * h)


def test_tracking_now_sees_all_of_the_natural_chromaticity(ref: ReferenceParticle) -> None:
    r"""45% (L1) -> 100% (L2, bend-free) -> 100% **with bends**, and the residual is owned.

    On this ring every element with length is a sector bend and every quadrupole is thin,
    so before L3 tracking could see *none* of the natural chromaticity: a thin kick is
    momentum-independent and the linear bend was chromatically ideal. The control below
    is that statement, run rather than remembered — the same ring with the same matrices
    and the pre-L3 ``track``.

    What is left over is **the trapezoid error of** ``natural_chromaticity``'s own
    integral, not the map's: it falls with the slice count, and is asserted as that order
    rather than to a tolerance.
    """
    lat = _ring(ref, steer=0.0)
    analytic = natural_chromaticity(lat, slices=1024)
    tracked = _tracked_chromaticity(lat)

    assert tracked[0] == pytest.approx(analytic[0], rel=2e-5)
    assert tracked[1] == pytest.approx(analytic[1], rel=2e-5)

    # The residual belongs to the integral's slicing, and shrinks when that is refined.
    coarse = abs(tracked[0] - natural_chromaticity(lat, slices=64)[0])
    fine = abs(tracked[0] - natural_chromaticity(lat, slices=512)[0])
    assert coarse / fine > 8.0

    # The control: identical matrices, identical design optics, identical analytic
    # chromaticity — and the pre-L3 map, which sees none of it.
    els = [_LinearBend(e.length, e.angle) if isinstance(e, Dipole) else e for e in lat.elements]
    blind = Lattice(els, ref)
    np.testing.assert_allclose(blind.one_turn_matrix(), lat.one_turn_matrix(), atol=1e-15)
    np.testing.assert_allclose(natural_chromaticity(blind, slices=1024), analytic, rtol=1e-12)
    blind_tracked = _tracked_chromaticity(blind)
    assert abs(blind_tracked[0]) < 1.0e-3 * abs(analytic[0])
    assert abs(blind_tracked[1]) < 1.0e-3 * abs(analytic[1])


def test_an_unsteered_ring_reports_exactly_the_design_optics(ref: ReferenceParticle) -> None:
    r"""The control that bounds the whole change: no orbit, no new terms.

    Every new entry is proportional to an orbit *angle*, so on a machine whose closed
    orbit is exactly zero the exact map must be indistinguishable from the linear one.
    The design optics is bit-for-bit unchanged because it is built on ``matrix()``, which
    this milestone did not touch; the on-orbit route agrees to its finite-difference
    floor, and the honest statement is the size of that floor.
    """
    lat = _ring(ref, steer=0.0)
    assert np.abs(closed_orbit_nonlinear(lat)).max() == 0.0
    assert closed_twiss(lat).disp_y == 0.0

    residual = float(np.abs(linearised_one_turn_map(lat) - lat.one_turn_matrix()).max())
    assert residual < 1.0e-11
    assert residual > 1.0e-16  # it is a floor, not an identity — say so

    for m, elem in zip(linearised_element_maps(lat), lat.elements, strict=True):
        assert np.abs(m - elem.matrix(ref)).max() < 1.0e-13


def test_a_combined_function_bend_is_deliberately_left_on_the_linear_map(
    ref: ReferenceParticle,
) -> None:
    r"""The scope line, enforced rather than documented.

    ``k1 != 0`` keeps the affine map, and the reason is that the split is **forced**, not
    chosen: with ``k1 = 0`` the vertical equation ``y' = py (1 + h x)/pz`` is a quadrature
    over a known ``x(s)`` because ``py`` is conserved, and with ``k1 != 0`` the same
    equation becomes a second-order ODE with an ``s``-dependent coefficient. The
    geometric term and vertical focusing are mutually exclusive in closed form, and a
    closed form is what keeps ``matrix()`` the exact origin Jacobian.

    So a combined-function bend is still chromatically ideal in ``track``, and this test
    is the statement of that limit — it will change when the expanded curved-quadrupole
    map lands, and it should fail loudly then rather than drift.
    """
    cf = Dipole(L_B, ANGLE, k1=0.4)
    got = cf.track(STATE, ref)
    np.testing.assert_allclose(got, cf.matrix(ref) @ STATE, rtol=0, atol=1e-16)

    # The pure bend at the same angle is *not* its matrix, which is what says the
    # difference above is the gradient's doing and not a dead code path.
    pure = Dipole(L_B, ANGLE)
    assert np.abs(pure.track(STATE, ref) - pure.matrix(ref) @ STATE).max() > 1.0e-6
