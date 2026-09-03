r"""P3 (a) — the rotated pole face: the wedge, and what it does *not* move.

P2 (i) shipped the hard-edge fringe of an **unrotated** face and refused the rotated
one, on the grounds that "a rotated face's nonlinear map is the fringe **plus a wedge**,
and the wedge is *first* order in the face angle where the fringe is second". The first
half of that is true and the second half is a non-sequitur, and finding out which was
this milestone's whole content: the wedge *is* first order in the face angle, and its
first-order content is precisely :func:`~accsim.elements.dipole._edge_matrix` — the
``h tan(e)`` kick F2 shipped years of milestones ago. The composed face therefore
reproduces a quantity the package already has before it is allowed to add one, and
``matrix()``, the tunes, ``beta``, the dispersion and the chromaticity do not move.

**What is derived here and what is imported.** The wedge is *derived*, from the same
uniform-field circle that :func:`~accsim.elements.dipole.exact_sector_bend_map`
integrates, and the derivation is done twice over: once symbolically (the plane-crossing
condition collapses to the algebraic momentum map with no transcendental solve left) and
once numerically against an independent implementation that walks the circle through
``arcsin`` rather than using the collapsed form. ``xt.Bend``'s closed form appears in
``tests/reference/test_wedge_xtrack.py`` as a *check*, not as the source — transcribing
it and then agreeing with it would be a transcription test, the circularity P2 (iv) had
to rule out for PTC.

**The gate with teeth is the decomposition, not the total.** Each of the three pieces of
a face has a non-identity Jacobian; the product is the identity plus ``_edge_matrix``.
Reporting only the product would read as "the pieces I composed compose", so the pieces
are reported separately here, and three deliberate breakages — the wedge's sign mirrored
at the exit, the rotation dropped, the wedge dropped — are shown to move it by
``O(h tan e)``, four orders above the residual.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Dipole,
    Drift,
    Lattice,
    ReferenceParticle,
    closed_twiss,
    is_symplectic_map_canonical,
    jacobian,
    natural_chromaticity,
    tunes,
)
from accsim.coords import DELTA, DIM, PX, PY, X, Y
from accsim.elements.dipole import _arcsinc, _edge_matrix, hard_edge_fringe_map, wedge_map

MASS0, GAMMA0 = 938.27208816e6, 20.0
L_B, ANGLE = 1.0, 0.2
H = ANGLE / L_B
E1, E2 = 0.15, -0.09

#: Every coordinate live at once: the wedge's entries are products, and a state with a
#: zero in it cannot separate one from another.
STATES = [
    np.array([1.3e-3, 7.0e-4, -9.0e-4, 5.0e-4, 2.0e-3, 4.0e-3]),
    np.array([-2.0e-3, 3.0e-3, 4.0e-3, -1.5e-3, -1.0e-3, -2.0e-2]),
    np.array([5.0e-3, -8.0e-3, 6.0e-3, 9.0e-3, 3.0e-3, 5.0e-2]),
]


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


@pytest.fixture
def slow_ref() -> ReferenceParticle:
    """A *low-energy* reference. The wedge's ``zeta`` carries ``beta0/beta``, which is
    ``1`` to within the suite's own tolerance at ``gamma0 = 20`` — P2 (i)'s finding, and
    it applies one map over."""
    return ReferenceParticle.from_gamma(MASS0, 1.5)


# ---------------------------------------------------------------------------
# The derivation: a circle, a plane, and no transcendental left
# ---------------------------------------------------------------------------


def test_the_plane_crossing_collapses_to_the_algebraic_momentum_map() -> None:
    r"""The wedge's ``px`` line is *derived*, symbolically, from where the circle lands.

    In a uniform vertical field ``dpx/dz = -h``, so with ``q = sqrt((1+d)^2 - py^2)``,
    ``px = q sin(a)`` and ``pz = q cos(a)`` the trajectory is the circle

        x(a) = x0 + (q/h)(cos a - cos a0),      z(a) = (q/h)(sin a0 - sin a).

    The wedge ends where that meets the plane ``z = x tan(theta)`` and reports in the
    frame whose ``x`` axis lies along ``(cos theta, sin theta)``, so the outgoing
    horizontal momentum is ``q sin(a_end + theta)``. Eliminating ``a_end`` between the
    two ought to need a transcendental solve. It does not: the crossing condition, times
    ``h cos(theta)/q``, **is** the statement

        sin(a_end + theta) = sin(a0 + theta) - (h x0 / q) sin(theta),

    i.e. ``new_px = px cos(theta) + (pz - h x) sin(theta)`` — algebraic, exact, and the
    line the code evaluates. sympy is asked to confirm the two are the same expression,
    not to check them at a point.
    """
    x0, a0, ae, th, q, h = sp.symbols("x0 alpha0 alpha_e theta q h", real=True)
    crossing = (q / h) * (sp.sin(a0) - sp.sin(ae)) - sp.tan(th) * (
        x0 + (q / h) * (sp.cos(ae) - sp.cos(a0))
    )
    collapsed = sp.sin(ae + th) - sp.sin(a0 + th) + (h * x0 / q) * sp.sin(th)
    assert sp.simplify(sp.expand_trig(crossing * sp.cos(th) * h / q + collapsed)) == 0


def test_every_component_matches_a_walk_around_the_circle(ref: ReferenceParticle) -> None:
    r"""The collapsed form, against the circle walked through ``arcsin``.

    The independent leg for the derivation above: rather than the algebraic map, find
    ``a_end`` explicitly, step along the arc to it, and rotate the frame by hand. ``y``
    and the path length are then the arc angle times ``py/h`` and ``(1+delta)/h`` — which
    is where ``zeta``'s share comes from, since the wedge advances the *reference*
    particle by nothing at all.

    This is not a re-implementation of the same algebra: it inverts a sine where the
    shipped map cancels it, and it divides by ``h`` where the shipped map refuses to.
    """
    rng = np.random.default_rng(3)
    worst = np.zeros(DIM)
    for _ in range(400):
        st = np.zeros(DIM)
        st[[X, PX, Y, PY]] = rng.normal(scale=2e-3, size=4)
        st[DELTA] = rng.normal(scale=1e-3)
        theta = rng.uniform(-0.4, 0.4)
        h = rng.uniform(0.05, 0.4) * rng.choice([-1.0, 1.0])

        one_plus = 1.0 + st[DELTA]
        q = np.sqrt(one_plus**2 - st[PY] ** 2)
        a0 = np.arcsin(st[PX] / q)
        a_end = np.arcsin(np.sin(a0 + theta) - (h * st[X] / q) * np.sin(theta)) - theta
        x_arc = st[X] + (q / h) * (np.cos(a_end) - np.cos(a0))
        z_arc = (q / h) * (np.sin(a0) - np.sin(a_end))
        e_over_e0 = np.hypot(ref.momentum_eV * one_plus, ref.mass_eV) / ref.total_energy_eV

        want = st.copy()
        want[X] = x_arc * np.cos(theta) + z_arc * np.sin(theta)
        want[PX] = q * np.sin(a_end + theta)
        want[Y] = st[Y] - (st[PY] / h) * (a_end - a0)
        want[4] = st[4] + (e_over_e0 / h) * (a_end - a0)
        worst = np.maximum(worst, np.abs(wedge_map(st, theta, h, ref) - want))
    assert worst.max() < 1e-13, worst


def test_arcsinc_is_the_series_it_claims_to_be() -> None:
    """``arcsin(t)/t`` across the branch, and the branch itself is invisible."""
    t = np.array([-0.6, -1e-3, -1e-4, -1e-5, 0.0, 1e-5, 1e-4, 1e-3, 0.6])
    got = _arcsinc(t)
    want = np.where(
        t == 0.0, 1.0, np.arcsin(np.where(t == 0.0, 1.0, t)) / np.where(t == 0.0, 1.0, t)
    )
    assert np.allclose(got, want, rtol=0.0, atol=1e-15)
    assert _arcsinc(np.array(0.0)) == 1.0


# ---------------------------------------------------------------------------
# The structure: no division by h, and a rotation is a wedge with the field off
# ---------------------------------------------------------------------------


def test_theta_zero_is_the_identity_bit_for_bit(ref: ReferenceParticle) -> None:
    """An unrotated face has no wedge, and that has to be *exactly* nothing.

    P2 (ii)'s finding, one element over: a short-circuit that returns something merely
    close to the limit is a **discontinuity**, and the way to avoid arguing about it is
    for the short-circuit to be the identity and for the limit to reach it.
    """
    for st in STATES:
        assert np.array_equal(wedge_map(st, 0.0, H, ref), st)
        assert np.array_equal(wedge_map(st, -0.0, H, ref), st)


def test_the_zero_field_wedge_is_the_rotation_into_the_face_frame(
    ref: ReferenceParticle,
) -> None:
    r"""``h = 0`` needs no branch, and what comes out is the plain frame rotation.

    This is why a face is built from **one** map and not two. The rotation into the
    plane of the real pole face is the exact canonical transformation

        x -> x pz / (pz cos e - px sin e),      px -> px cos e + pz sin e,

    with ``y`` and the arrival time picking up ``x sin e / (pz cos e - px sin e)`` times
    ``py`` and ``E/E0``. Every one of those falls out of :func:`wedge_map` at ``h = 0``
    with nothing special done — the ``h``-free construction of the arc is what buys it.
    """
    for st in STATES:
        for e in (0.15, -0.3, 0.4):
            one_plus = 1.0 + st[DELTA]
            pz = np.sqrt(one_plus**2 - st[PX] ** 2 - st[PY] ** 2)
            npz0 = pz * np.cos(e) - st[PX] * np.sin(e)
            e_over_e0 = np.hypot(ref.momentum_eV * one_plus, ref.mass_eV) / ref.total_energy_eV
            want = st.copy()
            want[X] = st[X] * pz / npz0
            want[PX] = st[PX] * np.cos(e) + pz * np.sin(e)
            want[Y] = st[Y] + st[X] * st[PY] * np.sin(e) / npz0
            want[4] = st[4] - e_over_e0 * st[X] * np.sin(e) / npz0
            assert np.allclose(wedge_map(st, e, 0.0, ref), want, rtol=0.0, atol=1e-15)


def test_the_arc_is_built_without_dividing_by_h(ref: ReferenceParticle) -> None:
    r"""The wedge reaches its zero-field limit **linearly**, and a naive form does not.

    ``y`` and ``zeta`` are ``(alpha0 - alpha_end)/h``, a ratio of two things that both
    vanish with ``h``. Written as ``(theta + D)/h`` with ``D`` a difference of arcsines —
    which is how the reference codes spell it, and it is exact algebra — the numerator's
    round-off is amplified by ``1/h``. Written as ``arcsin(h u)/h = u arcsinc(h u)`` with
    ``u`` free of ``h``, as here, nothing is amplified.

    The gate is the *approach*: ``|wedge(h) - wedge(0)|`` must fall by ten per decade of
    ``h``, all the way to ``h = 1e-12``, where the naive form is already wrong in the
    ninth digit.
    """
    st = STATES[0]
    limit = wedge_map(st, 0.2, 0.0, ref)
    previous = None
    for h in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        gap = np.abs(wedge_map(st, 0.2, h, ref) - limit).max()
        if previous is not None:
            assert previous / gap == pytest.approx(100.0, rel=5e-3)
        previous = gap
    assert np.abs(wedge_map(st, 0.2, 1e-14, ref) - limit).max() < 1e-15


# ---------------------------------------------------------------------------
# The face: three pieces, and the product is a quantity F2 already shipped
# ---------------------------------------------------------------------------


def face(state: np.ndarray, h: float, e: float, ref: ReferenceParticle, *, exit_face: bool):
    """The composition under test, written out rather than imported, so a change to
    :meth:`Dipole._face` has to be made here too before this file agrees with it."""
    if exit_face:
        st = wedge_map(state, -e, h, ref)
        st = hard_edge_fringe_map(st, -h, ref)
        return wedge_map(st, e, 0.0, ref)
    st = wedge_map(state, e, 0.0, ref)
    st = hard_edge_fringe_map(st, h, ref)
    return wedge_map(st, -e, h, ref)


def test_each_piece_carries_part_of_the_edge_and_the_product_is_all_of_it(
    ref: ReferenceParticle,
) -> None:
    r"""Where ``h tan(e)`` actually comes from — separately, then together.

    The shipped edge matrix is ``R21 = +h tan e``, ``R43 = -h tan e`` and nothing else.
    Neither is written anywhere in the composed face; both are produced, and by
    different pieces:

    - **``R43`` is the fringe's**, and it is the fringe *of P2 (i)*, unchanged. Its
      Jacobian at the origin is the identity — that was P2 (i)'s headline — but the
      rotation does not fix the origin: it sends the design orbit to ``px = sin(e)``, and
      the fringe's kick ``py -> py - h tan(x') y`` evaluated *there* is exactly
      ``-h tan(e) y``. So the vertical edge focusing is the fringe seen in the tilted
      frame, which is the physics, and it is asserted here as the single non-identity
      entry of the fringe's Jacobian at the rotated design orbit.
    - **``R21`` is the wedge's**, as ``h sin(e)``, scaled to ``h tan(e)`` by the
      rotation's own ``x -> x/cos(e)``.
    - the rotation's ``sin(e)`` momentum-dispersion and ``-tan(e)`` path-length entries
      are cancelled by the wedge's, which is why the product has no ``delta`` column.
    """
    rot = wedge_map(np.zeros(DIM), E1, 0.0, ref)
    assert rot[PX] == pytest.approx(np.sin(E1), rel=1e-14)

    # R43 is the fringe's, at the rotated design orbit.
    off = jacobian(lambda s: hard_edge_fringe_map(s, H, ref), rot) - np.eye(DIM)
    assert off[PY, Y] == pytest.approx(-H * np.tan(E1), rel=1e-8)
    off[PY, Y] = 0.0
    assert np.abs(off).max() < 1e-9

    # R21 is the wedge's h sin(e), and the rotation's x scaling turns it into h tan(e).
    wedge_only = jacobian(lambda s: wedge_map(s, -E1, H, ref), rot) - np.eye(DIM)
    assert wedge_only[PX, X] == pytest.approx(H * np.sin(E1), rel=1e-6)
    rotate_only = jacobian(lambda s: wedge_map(s, E1, 0.0, ref), np.zeros(DIM))
    assert rotate_only[X, X] == pytest.approx(1.0 / np.cos(E1), rel=1e-8)
    assert wedge_only[PX, X] * rotate_only[X, X] == pytest.approx(H * np.tan(E1), rel=1e-6)

    # ...and the whole face is _edge_matrix, at both ends.
    for exit_face in (False, True):
        e = E2 if exit_face else E1
        got = jacobian(
            lambda s, e=e, x=exit_face: face(s, H, e, ref, exit_face=x), np.zeros(DIM), 1e-5
        )
        assert np.abs(got - _edge_matrix(H, e)).max() < 1e-10


def test_three_deliberate_breakages_move_it_by_the_edge_kick(ref: ReferenceParticle) -> None:
    r"""The control. Without it the previous test reads as "the pieces I composed compose".

    Each of the three is a mistake that is easy to make and that no other gate in this
    file would catch:

    - **mirroring the wedge's sign at the exit.** The *fringe* takes ``-h`` at the exit,
      because the field switches on at one face and off at the other; the wedge does not,
      because it is a slice of the body's own field. Doing both breaks it by ``2 h tan e``.
    - **dropping the rotation** — the fringe is then evaluated on axis, where it does
      nothing, and ``R43`` disappears.
    - **dropping the wedge** — ``R21`` disappears and the rotation's ``delta`` column
      survives uncancelled.
    """
    reference_gap = H * np.tan(E1)

    def mirrored(s):
        s = wedge_map(s, -E1, -H, ref)
        s = hard_edge_fringe_map(s, -H, ref)
        return wedge_map(s, E1, 0.0, ref)

    def no_rotation(s):
        return wedge_map(hard_edge_fringe_map(s, H, ref), -E1, H, ref)

    def no_wedge(s):
        return hard_edge_fringe_map(wedge_map(s, E1, 0.0, ref), H, ref)

    edge = _edge_matrix(H, E1)
    gaps = {
        name: np.abs(jacobian(broken, np.zeros(DIM), 1e-5) - edge).max()
        for name, broken in (
            ("mirrored", mirrored),
            ("no_rotation", no_rotation),
            ("no_wedge", no_wedge),
        )
    }
    # Mirroring the wedge doubles the horizontal kick instead of cancelling nothing:
    # exactly 2 h tan(e) out, which is the sharpest of the three because it is a number.
    assert gaps["mirrored"] == pytest.approx(2.0 * reference_gap, rel=1e-6)
    # The other two leave the rotation's own tan(e) path-length column uncancelled, which
    # is 1/h times larger than the edge kick it was supposed to produce.
    for name in ("no_rotation", "no_wedge"):
        assert gaps[name] == pytest.approx(np.tan(E1), rel=1e-6), name
    assert min(gaps.values()) > 1.9 * reference_gap


def test_the_face_leaves_the_design_orbit_where_it_found_it(ref: ReferenceParticle) -> None:
    """A face is a rotation *and its undoing*; a particle on axis must not feel it.

    The intermediate states are nowhere near the origin — ``px = sin(e) = 0.15`` after the
    rotation — so this is a real cancellation between the three pieces, not three
    identities in a row.
    """
    for exit_face in (False, True):
        for e in (E1, E2, 0.4):
            assert np.abs(face(np.zeros(DIM), H, e, ref, exit_face=exit_face)).max() < 1e-15


# ---------------------------------------------------------------------------
# Symplecticity — the only gate that reaches zeta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exit_face", [False, True])
def test_the_composed_face_is_symplectic_at_amplitude(
    ref: ReferenceParticle, slow_ref: ReferenceParticle, exit_face: bool
) -> None:
    r"""Every piece is a canonical transformation, so the face is one — checked, not argued.

    ``zeta``'s share of the wedge is the extra flight time across the sliver, and no
    second-order reference leg can see it (it is the *path length*, which ``sectormap``
    reports in its own frame and which the composed ``T`` of a thin face reaches only
    through products of three coordinates). Symplecticity in the canonical
    ``(zeta, p_zeta)`` pair is what gates it, exactly as in P2 (i), and it is checked at
    a **low-energy** reference too, where the ``beta0/beta`` conversion is not ``1``.
    """
    for state in STATES:
        for r in (ref, slow_ref):
            assert is_symplectic_map_canonical(
                lambda s, r=r: face(s, H, E1, r, exit_face=exit_face), state, r, atol=1e-9
            )


def test_dropping_the_time_conversion_breaks_symplecticity_at_every_energy(
    ref: ReferenceParticle, slow_ref: ReferenceParticle
) -> None:
    r"""The ``E/E0`` factor in ``zeta`` is real — and here, unlike P2 (i), it is worst at
    **high** energy.

    ``zeta = s - beta0 c t``, and the wedge advances the *reference* particle by nothing
    while the particle flies an extra path ``ell``, so the cost is ``-ell beta0/beta``
    and not ``-ell``. P2 (i) measured the same conversion inside the fringe and found it
    visible **only** on a low-energy fixture (``9.1e-8`` at ``gamma0 = 1.5``, below
    tolerance at ``20``). The wedge inverts that, twice over, and neither reversal is a
    convention:

    - **it is visible at any energy**, because of the *order*, not the energy: the
      fringe's ``zeta`` share is ``Phi_delta ybar^2 / 2``, cubic in the coordinates, while
      the wedge's is the flight time across the sliver — linear in ``x`` and first order
      in ``e``, four orders larger at millimetre amplitudes;
    - **and it is worse the faster the beam is.** What symplecticity sees is
      ``d(E/E0)/d(delta) = beta0^2``, so dropping the factor costs ``beta0^2`` times the
      wedge's own ``zeta`` share. The gate is that closed form: the ratio of the two
      fixtures' failures is ``beta0(20)^2 / beta0(1.5)^2``, not a tolerance.
    """

    def naive(state, r):
        """The wedge with ``E/E0`` replaced by 1 in the ``zeta`` line."""
        st = wedge_map(state, E1, 0.0, r)
        st = hard_edge_fringe_map(st, H, r)
        good = wedge_map(st, -E1, H, r)
        one_plus = 1.0 + st[DELTA]
        e_over_e0 = np.hypot(r.momentum_eV * one_plus, r.mass_eV) / r.total_energy_eV
        bad = good.copy()
        bad[4] = st[4] + (good[4] - st[4]) / e_over_e0
        return bad

    honest = _worst_symplectic_gap(lambda s: face(s, H, E1, ref, exit_face=False), STATES[0], ref)
    fast_gap = _worst_symplectic_gap(lambda s: naive(s, ref), STATES[0], ref)
    slow_gap = _worst_symplectic_gap(lambda s: naive(s, slow_ref), STATES[0], slow_ref)
    assert honest < 1e-9
    assert fast_gap > 1e5 * honest
    assert fast_gap / slow_gap == pytest.approx(ref.beta0**2 / slow_ref.beta0**2, rel=2e-2)


def _worst_symplectic_gap(map_fn, state, ref: ReferenceParticle) -> float:
    """``max |M^T J M - J|`` for ``map_fn`` at ``state``, in the canonical pair."""
    from accsim.symplectic import from_canonical, to_canonical, unit_symplectic_matrix

    j = unit_symplectic_matrix()
    m = jacobian(
        lambda c: to_canonical(map_fn(from_canonical(c, ref)), ref), to_canonical(state, ref)
    )
    return float(np.abs(m.T @ j @ m - j).max())


# ---------------------------------------------------------------------------
# The Dipole, and the first-order quantities that must not move
# ---------------------------------------------------------------------------


def test_an_unrotated_bend_is_unchanged_bit_for_bit(ref: ReferenceParticle) -> None:
    """P3 must not disturb P2 (i). At ``e1 = e2 = 0`` the wedges are the identity."""
    rng = np.random.default_rng(7)
    bunch = rng.normal(scale=1e-3, size=(DIM, 16))
    with_faces = Dipole(L_B, ANGLE, fringe=True).track(bunch, ref)
    bare_fringe = hard_edge_fringe_map(
        Dipole(L_B, ANGLE).track(hard_edge_fringe_map(bunch, H, ref), ref), -H, ref
    )
    assert np.array_equal(with_faces, bare_fringe)


def test_a_bunch_takes_the_same_path_as_a_state(ref: ReferenceParticle) -> None:
    """Vectorised over a trailing particle axis, at a face angle where nothing short-circuits.

    P2 (i) asserts this for the fringe because it has a scalar-looking quadratic solve in
    it. The wedge has the same shape of hazard one map over — :func:`_arcsinc` picks its
    branch with :func:`numpy.where`, which is per-element for a bunch and per-scalar for a
    state — so a wedge that worked on one particle and silently mixed branches across a
    bunch would pass every other test in this file. Asserted at ``e = E1``, since at
    ``e = 0`` both wedges return a copy and the check is vacuous.
    """
    rng = np.random.default_rng(23)
    bunch = rng.normal(scale=2e-3, size=(DIM, 24))
    bunch[DELTA] *= 10.0  # delta out to a few percent, where _arcsinc's argument moves
    for theta, h in ((E1, H), (E2, 0.0), (0.4, -H)):
        together = wedge_map(bunch, theta, h, ref)
        for i in range(bunch.shape[1]):
            assert np.array_equal(together[:, i], wedge_map(bunch[:, i], theta, h, ref))
    faced = Dipole(L_B, ANGLE, e1=E1, e2=E2, fringe=True)
    out = faced.track(bunch, ref)
    for i in range(bunch.shape[1]):
        assert np.array_equal(out[:, i], faced.track(bunch[:, i], ref))


def test_a_straight_magnet_with_a_rotated_face_still_feels_nothing(
    ref: ReferenceParticle,
) -> None:
    r"""``angle = 0`` with ``e1 != 0``: two exact-inverse rotations, and no guard for them.

    The one corner where P3 is *not* bit-identical to what came before. With no field
    there is no fringe and no wedge, so a face is the rotation into the pole-face plane
    followed by its own inverse — mathematically the identity, numerically two round-offs.
    :meth:`Dipole._matrix_body` returns ``_edge_matrix(0, e) = I``, so this is the only
    place in the package where ``matrix`` is the origin Jacobian of ``track`` up to
    round-off rather than exactly.

    It is left unguarded on purpose, and the residual is gated on its **mechanism** rather
    than on a number: the rotation puts the design orbit at ``px = sin(e)``, so what is
    lost is one bit of ``sin(e)`` — not one bit of the coordinate. Measured at ``1.2``
    machine epsilons of ``sin(e) + |state|`` across face angles from ``0.05`` to ``0.3``,
    where the *coordinate*-relative error is 84 and 496 epsilons respectively and would
    read as a bug. A ``h != 0`` branch would buy bit-identity at the price of the
    discontinuity P2 (ii) found in the sextupole's short-circuit, for a residual in the
    last bit of a quantity the map does not even return.

    A *bending* magnet has no such corner: there the two wedges are not inverses, and the
    agreement with ``_edge_matrix`` is structural rather than numerical — which is why the
    same comparison on a bend is ten orders larger and is the milestone's whole point.
    """
    for e in (0.05, 0.3):
        faced = Dipole(2.0, 0.0, e1=e, e2=-e, fringe=True)
        plain = Dipole(2.0, 0.0, e1=e, e2=-e)
        assert np.array_equal(faced.matrix(ref), plain.matrix(ref))
        assert np.abs(faced.track(np.zeros(DIM), ref)).max() == 0.0
        for st in STATES:
            gap = np.abs(faced.track(st, ref) - plain.track(st, ref)).max()
            scale = np.sin(e) + np.abs(st).max()
            assert gap < 8.0 * np.finfo(float).eps * scale, (e, gap, gap / scale)

    # ...and the contrast: the same two elements *with* a bend angle differ by the face,
    # measured 3.0e-7 against the straight corner's 1.2e-16 — nine orders, which is what
    # says the corner is arithmetic and the milestone is not.
    bent = np.abs(
        Dipole(2.0, 0.4, e1=0.3, e2=-0.3, fringe=True).track(STATES[0], ref)
        - Dipole(2.0, 0.4, e1=0.3, e2=-0.3).track(STATES[0], ref)
    ).max()
    assert bent > 1e8 * 8.0 * np.finfo(float).eps * (np.sin(0.3) + np.abs(STATES[0]).max())


def test_no_first_order_quantity_moves(ref: ReferenceParticle) -> None:
    r"""``matrix``, the tunes, ``beta``, the dispersion and the chromaticity, by equality.

    The claim P3 exists to make. It is asserted with ``array_equal`` and not with a
    tolerance, because the code path is *identical*: :meth:`Dipole._matrix_body` never
    consults ``fringe`` at all. What makes that legitimate rather than a convenient
    omission is the Jacobian test above — the composed face's origin Jacobian **is**
    ``_edge_matrix``, so the package's central invariant (``matrix`` is the exact origin
    Jacobian of ``track``) survives the change, and it is re-measured here on the ring.
    """

    def ring(fringe: bool) -> Lattice:
        return Lattice(
            [
                Dipole(L_B, ANGLE, e1=E1, e2=E2, fringe=fringe, name="b1"),
                Drift(0.8),
                Dipole(L_B, -ANGLE, e1=E2, e2=E1, fringe=fringe, name="b2"),
                Drift(0.8),
            ],
            ref,
        )

    off, on = ring(False), ring(True)
    assert np.array_equal(off.one_turn_matrix(), on.one_turn_matrix())
    for element_off, element_on in zip(off.elements, on.elements, strict=True):
        assert np.array_equal(element_off.matrix(ref), element_on.matrix(ref))

    bend_on = Dipole(L_B, ANGLE, e1=E1, e2=E2, fringe=True)
    got = jacobian(lambda s: bend_on.track(s, ref), np.zeros(DIM), 1e-5)
    assert np.abs(got - bend_on.matrix(ref)).max() < 1e-9


def test_the_face_does_change_tracking_and_by_a_measured_amount(ref: ReferenceParticle) -> None:
    """The flag has to *do* something, and the something has to be the face.

    A gate P2 (iv) had to learn to distrust — "the flag moves the answer" is satisfied by
    a map that moves it wrongly — so the size is pinned against the *fringe alone*, which
    is what the previous milestone would have applied to this bend if it had not refused.
    The wedge is the larger contribution, and the difference is first order in ``e``.
    """
    linear = Dipole(L_B, ANGLE, e1=E1, e2=E2)
    full = Dipole(L_B, ANGLE, e1=E1, e2=E2, fringe=True)
    gap = np.abs(full.track(STATES[0], ref) - linear.track(STATES[0], ref)).max()
    assert 1e-8 < gap < 1e-4


def test_the_face_effect_is_first_order_in_the_rotation(ref: ReferenceParticle) -> None:
    r"""``|face(e) - face(0)|`` falls by ten per decade of ``e`` — the wedge's own order.

    P2 (i) was right that the wedge is *first* order in the face angle where the fringe
    is second; that is measured here rather than repeated. What it was wrong about is the
    consequence, since the first-order part lands on a matrix the package already had.
    """
    previous = None
    for e in (1e-1, 1e-2, 1e-3, 1e-4):
        gap = np.abs(
            face(STATES[0], H, e, ref, exit_face=False) - hard_edge_fringe_map(STATES[0], H, ref)
        ).max()
        if previous is not None:
            assert previous / gap == pytest.approx(10.0, rel=2e-2)
        previous = gap
    assert np.array_equal(
        face(STATES[0], H, 0.0, ref, exit_face=False), hard_edge_fringe_map(STATES[0], H, ref)
    )


def test_a_ring_of_rotated_faces_still_has_its_optics(ref: ReferenceParticle) -> None:
    """A smoke test with teeth: a real cell, tracked, with the faces on.

    Closed Twiss, the tunes and the natural chromaticity are all computed from
    ``matrix``, so they are the *same numbers* with the flag on — the point is that the
    lattice still builds, still closes, and still tracks a bunch to a finite answer with
    the nonlinear faces in the ring.
    """
    from accsim import Quadrupole

    cell = [
        Quadrupole(0.3, 1.2),
        Drift(0.5),
        Dipole(L_B, ANGLE, e1=ANGLE / 2, e2=ANGLE / 2, fringe=True),
        Drift(0.5),
        Quadrupole(0.3, -1.2),
        Drift(0.5),
        Dipole(L_B, ANGLE, e1=ANGLE / 2, e2=ANGLE / 2, fringe=True),
        Drift(0.5),
    ]
    ring = Lattice(cell * 6, ref)
    tw = closed_twiss(ring)
    assert np.all(np.asarray(tw.beta_x) > 0.0)
    qx, qy = tunes(ring)
    assert 0.0 < qx < 10.0 and 0.0 < qy < 10.0
    assert np.all(np.isfinite(natural_chromaticity(ring)))

    rng = np.random.default_rng(19)
    bunch = rng.normal(scale=2e-4, size=(DIM, 64))
    out = bunch
    for element in ring.elements:
        out = element.track(out, ref)
    assert np.all(np.isfinite(out))
    assert np.abs(out).max() < 1.0
