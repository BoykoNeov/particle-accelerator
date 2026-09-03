r"""P2 (iv) — the quadrupole's kinematic term, opt in.

L2 shipped the thick quadrupole as the exact flow of the **paraxial** Hamiltonian:
exact in ``delta`` to all orders, and dropping ``O(angle^3)``. The reason was not
laziness — there is no closed form for the exact quadrupole Hamiltonian, and the
alternative on offer (split it and integrate) moves ``matrix()`` off the origin
Jacobian of ``track()``, which every design-optics gate in the package rests on.

P2 (iv) takes the kinematic term back **without** paying that price. The exact
Hamiltonian splits as

    H_exact  =  H_paraxial  +  H_kin
    H_kin    =  (1+delta) - sqrt((1+delta)^2 - p^2) - p^2/(2(1+delta))
             =  p^4 / (8 (1+delta)^3) + O(p^6),        p^2 = px^2 + py^2

and *both* halves are exactly solvable: the first by L2's cos/sin flow, the second
because ``H_kin`` depends on the **momenta alone**, so ``px``, ``py`` and ``delta``
are its constants of motion. The composition is symmetric,
``[kin(h/2) . para(h) . kin(h/2)]^n`` with ``h = L/n``, reached through
``Quadrupole(..., kinematic_slices=n)``; ``n = 0`` (the default) is L2's map
untouched.

**Why the origin Jacobian survives, in one line.** ``H_kin``'s flow moves ``x`` and
``y`` by amounts *cubic* in the angles and ``zeta`` by a *quartic*, so its Jacobian
at zero angle is the identity — at any ``delta``. At ``n = 1`` the paraxial factor is
not sliced at all, so ``matrix()`` is still the exact slope of ``track()`` at the
reference particle, bit for bit. That is the whole difference from the
drift-kick-drift family, which buys the same physics and loses that.

**What gates what.** Symplecticity is blind to the *size* of the term (any multiple
of ``H_kin``'s flow is a symplectic map — J1's lesson), and so is the ``k1 -> 0``
drift identity, which the paraxial factor does not enter. The gate that pins the
coefficient with no arbiter at all is the first one below: it was written before the
implementation and predicts P1's already-measured PTC gap. The reference legs live in
``tests/reference/test_kinematic_quadrupole_xtrack.py`` and
``tests/reference/test_second_order_map_madx.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    SkewQuadrupole,
    closed_twiss,
    is_symplectic_map,
    is_symplectic_map_canonical,
    natural_chromaticity,
    taylor_expand,
    tunes,
)
from accsim.coords import DELTA, PX, PY, ZETA, X, Y
from accsim.elements.quadrupole import kinematic_drift

#: P1's fixture, element and expansion point, verbatim
#: (``tests/reference/test_second_order_map_madx.py``).
MASS0, GAMMA0 = 938.27208816e6, 20.0
LQ, KF = 0.3, 1.2
Z0 = np.array([3.8e-4, -6.6e-5, 0.0, 0.0, 0.0, 0.0])

#: The five rows PTC returns at ``icase=5`` — the frame P1's gap was measured in.
FIVE = [X, PX, Y, PY, DELTA]

#: A longer, stronger magnet and a generic state, for everything that is not P1's
#: measurement: every coordinate nonzero, so no term can hide behind a zero.
L_Q, K1 = 0.7, 1.2
STATE = np.array([2.0e-3, 1.5e-3, -1.0e-3, 8.0e-4, 5.0e-4, 1.0e-3])


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


# --------------------------------------------------------------------------
# 1. The pre-committed gate: the size of the term, with no arbiter
# --------------------------------------------------------------------------


def test_switching_the_kinematic_term_on_moves_t_by_p1s_measured_ptc_gap(
    ref: ReferenceParticle,
) -> None:
    r"""**Written before the implementation, and it is the only gate on the size.**

    P1 measured accsim's thick quadrupole missing PTC's exact one by ``5.6e-5`` on
    ``T[x, px, px]`` about ``x = 3.8e-4, px = -6.6e-5`` of a ``Quadrupole(0.3, 1.2)``
    at ``gamma0 = 20``, and *attributed* it to the ``p^4/8`` that L2's paraxial
    Hamiltonian drops. If that attribution is right, switching the term on has to move
    ``T`` by that same amount — a prediction checkable with no reference code at all,
    accsim against accsim. Half the coefficient would give ``2.8e-5``; a wrong power of
    ``(1+delta)`` moves it too. Neither symplecticity nor the ``k1 -> 0`` drift
    identity below can see any of that.

    It lands: ``5.373e-5`` at one slice, ``5.626e-5`` converged, and the entry that
    moves is ``T[x, px, px]`` — the one P1 named. The residual ``4.5%`` at ``n = 1``
    is the split's own error, not the coefficient's; it is gated as a ``1/n^2`` below.

    The number is not fitted, either. On a *drift* the kinematic map is
    ``x += L px (px^2+py^2)/2``, whose second derivative in ``px`` at ``px_co`` is
    ``3 L px_co = 3 * 0.3 * 6.6e-5 = 5.94e-5`` — within ``6%`` of the measured
    ``5.622e-5``, the remainder being the focusing this estimate ignores.
    """
    para = Quadrupole(LQ, KF)
    F = np.ix_(FIVE, FIVE, FIVE)
    t_para = taylor_expand(lambda s: para.track(s, ref), Z0, step=2.5e-4).T

    moved = {}
    for n in (1, 8):
        kin = Quadrupole(LQ, KF, kinematic_slices=n)
        t_kin = taylor_expand(lambda s, e=kin: e.track(s, ref), Z0, step=2.5e-4).T
        moved[n] = np.abs(t_kin[F] - t_para[F])

    assert np.max(moved[1]) == pytest.approx(5.6e-5, rel=0.10)  # the pre-committed number
    assert np.max(moved[8]) == pytest.approx(5.6e-5, rel=0.02)  # sharper, converged
    # And it is P1's entry, not some other one that happens to be the same size.
    assert np.argmax(moved[8]) == np.ravel_multi_index((X, PX, PX), moved[8].shape)
    # The naive geometric estimate, for a term that was pinned rather than fitted.
    assert np.max(moved[8]) == pytest.approx(3.0 * LQ * abs(Z0[PX]), rel=0.08)


# --------------------------------------------------------------------------
# 2. The map is the flow of the Hamiltonian remainder
# --------------------------------------------------------------------------


def test_the_kinematic_remainder_is_quartic_in_the_momenta(ref: ReferenceParticle) -> None:
    r"""``H_kin = p^4 / (8 (1+delta)^3) + O(p^6)`` — derived, not recalled.

    The leading coefficient is what makes the whole term ``O(angle^3)`` in the map and
    the ``zeta`` share ``O(angle^4)``, and it is why the origin Jacobian cannot move:
    a quartic Hamiltonian has a vanishing Hessian at ``p = 0``.
    """
    px, py, t = sp.symbols("px py t", real=True)
    # ``1 + delta`` as a *positive* symbol: sympy will not reduce sqrt((1+delta)^2)
    # without that, and the whole expansion is in powers of it.
    w = sp.Symbol("one_plus_delta", positive=True)
    p2 = px**2 + py**2
    h_kin = w - sp.sqrt(w**2 - p2) - p2 / (2 * w)

    # Order by order in the momenta (t scales both), with delta kept exact.
    scaled = h_kin.subs({px: px * t, py: py * t})
    series = sp.expand(sp.series(scaled, t, 0, 8).removeO())
    assert sp.simplify(series.coeff(t, 2)) == 0  # the paraxial part is subtracted off
    assert sp.simplify(series.coeff(t, 3)) == 0  # no odd term
    assert sp.simplify(series.coeff(t, 4) - p2**2 / (8 * w**3)) == 0
    assert sp.simplify(series.coeff(t, 6) - p2**3 / (16 * w**5)) == 0


def test_the_kinematic_drift_is_that_hamiltonians_exact_flow(ref: ReferenceParticle) -> None:
    r"""The implemented map equals ``s * dH_kin/d(canonical momenta)`` at 60 digits.

    ``x`` and ``y`` move by ``s dH_kin/dpx``, ``s dH_kin/dpy``; ``zeta`` by
    ``s dH_kin/dp_zeta``, and it is ``p_zeta`` — not ``delta`` — because that is
    ``zeta``'s canonical partner (see :mod:`accsim.symplectic`). ``px``, ``py`` and
    ``delta`` do not move at all, which is what makes the flow explicit.

    Evaluated in ``mpmath`` at 60 digits rather than in float64, because the
    implementation reaches the term by *subtracting* the paraxial drift from the exact
    one and the answer is nine orders below the two it is the difference of. The
    measured agreement is ``~1e-19`` on all three moved coordinates — about one ulp of
    the coordinates themselves — which says the subtraction costs digits of the
    correction and never reaches the coordinate. (A float64 ``sympy`` substitution is
    *not* a valid arbiter here and was tried first: ``sympy`` keeps the precision of
    the ``Float`` it is given, so the same cancellation eats its answer too, and it
    disagreed at the seventh digit for reasons that had nothing to do with the map.)
    """
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60

    length = 0.37
    st = np.array([1.0e-3, 3.0e-3, -2.0e-3, -1.7e-3, 5.0e-4, 2.0e-3])
    got = kinematic_drift(st, length, ref) - st

    d = mp.mpf(st[DELTA])
    p2 = mp.mpf(st[PX]) ** 2 + mp.mpf(st[PY]) ** 2
    w = 1 + d
    u = mp.sqrt((mp.mpf(ref.momentum_eV) * w) ** 2 + mp.mpf(ref.mass_eV) ** 2) / mp.mpf(
        ref.total_energy_eV
    )
    pz = mp.sqrt(w * w - p2)
    s = mp.mpf(length)
    # dH_kin/dpx = px/pz - px/(1+delta);  dH_kin/dp_zeta uses ddelta/dp_zeta = (E/E0)/(1+delta).
    want_x = s * (mp.mpf(st[PX]) / pz - mp.mpf(st[PX]) / w)
    want_y = s * (mp.mpf(st[PY]) / pz - mp.mpf(st[PY]) / w)
    want_z = s * (u / w + p2 * u / (2 * w**3) - u / pz)

    assert abs(got[X] - float(want_x)) < 1e-18
    assert abs(got[Y] - float(want_y)) < 1e-18
    assert abs(got[ZETA] - float(want_z)) < 1e-18
    # The momenta are the constants of the motion: they do not move at all.
    assert got[PX] == 0.0 and got[PY] == 0.0 and got[DELTA] == 0.0
    # Non-vacuous: the moved coordinates are far above the tolerance just used.
    assert abs(got[X]) > 1e-9 and abs(got[ZETA]) > 1e-12


# --------------------------------------------------------------------------
# 3. What it closes: L2's zero-strength inconsistency, with no branch
# --------------------------------------------------------------------------


def test_a_zero_strength_quadrupole_is_now_the_exact_drift(ref: ReferenceParticle) -> None:
    r"""**L2's documented open gap, closed — and closed structurally.**

    L2's write-up records that ``Quadrupole(L, 0).track`` is the *expanded* drift and
    not :class:`~accsim.elements.drift.Drift`'s exact one, "narrowed rather than
    closed", and that short-circuiting ``k1 == 0`` would close it only by making the
    map discontinuous in ``k1`` — the same trap P2 (ii) then walked into with the
    sextupole's zero-strength body.

    With the term on there is nothing to short-circuit. At ``k1 = 0`` both factors are
    flows of momentum-only Hamiltonians over the same length, so they **commute**, the
    interleaving telescopes, and ``kin(L) . para(L)`` is the exact drift identically —
    at any ``n``, for any ``L``, with no branch anywhere. Measured to a few ulps
    (``1e-17`` absolute on coordinates of ``1e-3``), the residue of adding the
    increments in pieces rather than at once.

    The continuity that L2 refused to break is asserted directly: the residual against
    the exact drift goes to zero **linearly in** ``k1``, so the ``k1 = 0`` answer is
    the limit of the map rather than a special case bolted onto it.
    """
    st = np.array([1.0e-3, 3.7e-2, 1.0e-3, -1.3e-2, 5.0e-4, 1.0e-3])
    for length in (0.7, 1.3, 2.0):
        exact = Drift(length).track(st, ref)
        for n in (1, 2, 3, 5):
            got = Quadrupole(length, 0.0, kinematic_slices=n).track(st, ref)
            assert np.max(np.abs(got - exact)) < 1e-16, (length, n)
        # Non-vacuous: the shipped default still differs, by the O(angle^3) L2 named.
        assert np.max(np.abs(Quadrupole(length, 0.0).track(st, ref) - exact)) > 1e-6

    # No k1 == 0 branch: the map is continuous through zero gradient.
    def gap(k1: float) -> float:
        return float(
            np.max(
                np.abs(
                    Quadrupole(2.0, k1, kinematic_slices=1).track(st, ref)
                    - Drift(2.0).track(st, ref)
                )
            )
        )

    assert gap(1.0e-9) / gap(1.0e-10) == pytest.approx(10.0, rel=0.01)  # linear in k1
    assert gap(1.0e-9) < 1e-10
    assert gap(0.0) < 1e-16


def test_the_term_is_cubic_in_the_angle_and_independent_of_the_gradient(
    ref: ReferenceParticle,
) -> None:
    """The shape that identifies it as the angle expansion: ``x8`` per doubling."""
    sizes = []
    for scale in (1.0, 2.0, 4.0):
        st = np.array([0.0, 1.0e-3 * scale, 0.0, 0.0, 0.0, 0.0])
        sizes.append(
            float(
                np.max(
                    np.abs(
                        Quadrupole(L_Q, 0.0, kinematic_slices=1).track(st, ref)
                        - Quadrupole(L_Q, 0.0).track(st, ref)
                    )
                )
            )
        )
    for small, big in zip(sizes[:-1], sizes[1:], strict=True):
        assert big / small == pytest.approx(8.0, rel=0.01)


# --------------------------------------------------------------------------
# 4. What it costs: a second-order split whose error is not small
# --------------------------------------------------------------------------


def test_the_splitting_error_falls_as_one_over_n_squared(ref: ReferenceParticle) -> None:
    r"""``4.00`` per doubling of ``kinematic_slices`` — the order, not a tolerance.

    P2 (ii)'s lesson applied to a second element: a fixed tolerance would accept a map
    that is merely small-and-wrong, so what is asserted is the *mechanism*. The
    symmetric composition's Baker-Campbell-Hausdorff remainder is ``O(h^2)`` overall,
    and it comes out as ``3.73, 3.95, 3.99, 4.00, 4.01`` across five doublings.
    """
    converged = Quadrupole(L_Q, K1, kinematic_slices=512).track(STATE, ref)
    residuals = [
        float(np.max(np.abs(Quadrupole(L_Q, K1, kinematic_slices=n).track(STATE, ref) - converged)))
        for n in (1, 2, 4, 8, 16, 32)
    ]
    for coarse, fine in zip(residuals[1:-1], residuals[2:], strict=True):
        assert coarse / fine == pytest.approx(4.0, rel=0.03)
    assert residuals[0] / residuals[1] == pytest.approx(3.7, rel=0.05)  # n=1 is off the asymptote


def test_one_slice_is_not_enough_and_the_reason_is_not_the_terms_smallness(
    ref: ReferenceParticle,
) -> None:
    r"""**The trap this milestone had to avoid naming a default.**

    ``H_kin`` is quartic in the momenta and its effect here is ``2.6e-10``, so the
    tempting inference is that one symmetric slice must be plenty — the split's error
    being "a small correction to a small term". It is not. The leading commutator
    ``[H_para, H_kin]`` carries the gradient, and relative to ``H_kin``'s own effect it
    scales as ``k1 L x / p``, which is order **one** for an ordinary trajectory
    (``x ~ p/(k1 L)`` is what a matched particle looks like). Measured on ``STATE``:
    the term is ``2.58e-10`` and the ``n = 1`` error is ``3.64e-10`` — *larger* than
    the thing being added.

    So ``kinematic_slices`` is a real knob and the gate above is the ``1/n^2``, not a
    number. Recorded here rather than left as a surprise for the reference legs, which
    have to be converged on both sides before they mean anything.
    """
    converged = Quadrupole(L_Q, K1, kinematic_slices=512).track(STATE, ref)
    paraxial = Quadrupole(L_Q, K1).track(STATE, ref)
    one_slice = Quadrupole(L_Q, K1, kinematic_slices=1).track(STATE, ref)

    term = float(np.max(np.abs(converged - paraxial)))
    error = float(np.max(np.abs(one_slice - converged)))
    assert term == pytest.approx(2.58e-10, rel=0.05)
    assert error > term  # the headline: not a small correction to a small term

    # And it really is the *offset* that drives it, not a coincidence of this state:
    # shrinking x and y at fixed angle takes the ratio from 1.42 down to 0.14, an order
    # of magnitude, and it is monotone. (It does not reach zero, because a quadrupole
    # builds an offset out of the entry angle inside its own body.)
    ratios = []
    for scale in (1.0, 0.5, 0.25, 0.0):
        st = STATE.copy()
        st[[X, Y]] *= scale
        best = Quadrupole(L_Q, K1, kinematic_slices=512).track(st, ref)
        ratios.append(
            float(np.max(np.abs(Quadrupole(L_Q, K1, kinematic_slices=1).track(st, ref) - best)))
            / float(np.max(np.abs(best - Quadrupole(L_Q, K1).track(st, ref))))
        )
    assert ratios[0] > 1.0 and ratios[-1] < 0.2
    for big, small in zip(ratios[:-1], ratios[1:], strict=True):
        assert big > small


def test_the_composition_is_symplectic_in_the_canonical_variables(
    ref: ReferenceParticle,
) -> None:
    r"""Exactly symplectic at every ``n``, because both factors are exact flows.

    Not "symplectic to the order of the splitting": each of ``kin`` and ``para`` is the
    exact flow of its own Hamiltonian, so the composition is a composition of
    symplectic maps whatever ``h`` is. That is the property worth having over a more
    accurate non-symplectic map — a truncated-but-symplectic map is safe to iterate.

    The plain :func:`~accsim.symplectic.is_symplectic_map` still **rejects** it, as it
    rejects the exact drift and L2's own map, because ``(zeta, delta)`` is not a
    canonical pair. Asserted so that a future reader does not read the rejection as a
    regression.
    """
    for n in (1, 2, 4, 16):
        q = Quadrupole(L_Q, K1, kinematic_slices=n)
        assert is_symplectic_map_canonical(lambda s, e=q: e.track(s, ref), STATE, ref, atol=1e-11)
        assert not is_symplectic_map(lambda s, e=q: e.track(s, ref), STATE, atol=1e-11)


# --------------------------------------------------------------------------
# 5. Blast radius: nothing on the design orbit moves
# --------------------------------------------------------------------------


def _ring(ref: ReferenceParticle, kinematic_slices: int = 0) -> Lattice:
    els: list = []
    for _ in range(6):
        els += [
            Quadrupole(0.3, K1, kinematic_slices=kinematic_slices),
            Drift(0.5),
            Quadrupole(0.3, -K1, kinematic_slices=kinematic_slices),
            Drift(0.5),
        ]
    return Lattice(els, ref)


def test_nothing_on_the_design_orbit_moves(ref: ReferenceParticle) -> None:
    r"""**The blast radius, and why it is zero.** ``array_equal``, not a tolerance.

    ``H_kin``'s flow is the identity at zero angle *at any* ``delta``, so the whole
    design-optics half of the package cannot move: :meth:`Quadrupole._matrix_body` is
    untouched code, and every quantity built on it — beta, alpha, the tunes, the
    dispersion, the natural chromaticity — is bit-for-bit what it was. The reference
    particle itself tracks to bit-identical coordinates, because at ``px = py = 0``
    and ``delta = 0`` every increment either factor adds is exactly ``0.0``.

    This is the same statement P2 (i) made about the dipole fringe and for the same
    structural reason, and it is asserted the same way — with ``array_equal``, which a
    tolerance-based check would let drift.
    """
    plain, kin = _ring(ref), _ring(ref, kinematic_slices=4)

    np.testing.assert_array_equal(plain.one_turn_matrix(), kin.one_turn_matrix())
    np.testing.assert_array_equal(np.array(tunes(plain)), np.array(tunes(kin)))
    np.testing.assert_array_equal(
        np.array(natural_chromaticity(plain)), np.array(natural_chromaticity(kin))
    )
    tw_a, tw_b = closed_twiss(plain), closed_twiss(kin)
    for name in ("beta_x", "beta_y", "alpha_x", "alpha_y", "mu_x", "mu_y", "disp_x", "disp_px"):
        assert getattr(tw_a, name) == getattr(tw_b, name), name

    # The reference particle tracks identically, and so does a purely longitudinal one.
    def tracked(lat: Lattice, st: np.ndarray) -> np.ndarray:
        out = np.asarray(st, dtype=float)
        for element in lat.elements:
            out = element.track(out, ref)
        return out

    np.testing.assert_array_equal(tracked(plain, np.zeros(6)), tracked(kin, np.zeros(6)))

    # **Off momentum but still on axis, it is one ulp and not zero**, and the scope of
    # the bit-identity claim has to say so. ``kin`` reaches its (analytically exact)
    # zero by subtracting the paraxial drift's ``zeta`` from the exact drift's, and at
    # ``px = py = 0`` those two are the same quantity written with a different grouping
    # of the arithmetic — equal to the last decimal, not to the last bit. Measured
    # ``6.5e-19`` on ``zeta`` at ``delta = 2e-3``, i.e. ``1.9e-15`` relative. Masking
    # ``p = 0`` would make it exact and is not worth a branch; what matters is that the
    # *design orbit*, where every increment is exactly ``0.0``, is bit-identical above.
    on_axis = np.array([0.0, 0.0, 0.0, 0.0, 3.0e-4, 2.0e-3])
    np.testing.assert_allclose(tracked(plain, on_axis), tracked(kin, on_axis), rtol=0.0, atol=1e-18)

    # Non-vacuous: a particle with an *angle* does move, which is the whole point.
    angled = np.array([1.0e-3, 2.0e-3, -5.0e-4, 1.0e-3, 0.0, 0.0])
    assert np.max(np.abs(tracked(plain, angled) - tracked(kin, angled))) > 1e-11


# --------------------------------------------------------------------------
# 6. The same magnet, however it is spelled — and the knob itself
# --------------------------------------------------------------------------


def test_a_skew_quadrupole_carries_the_term_through_the_roll(ref: ReferenceParticle) -> None:
    r"""``SkewQuadrupole(k1s, kinematic_slices=n)`` is the normal one rolled 45 degrees.

    L2's rule — the same magnet must not behave two ways depending on how it is
    spelled — carries here for a reason worth stating rather than assuming:
    ``H_kin`` depends on the momenta only through ``px^2 + py^2``, which any
    ``s``-rotation leaves invariant. So the term commutes with the roll and the
    conjugation is exact, not approximate.
    """
    st = STATE.copy()
    skew = SkewQuadrupole(L_Q, K1, kinematic_slices=4)
    rolled = Quadrupole(L_Q, K1, kinematic_slices=4, roll=-np.pi / 4.0)
    np.testing.assert_allclose(skew.track(st, ref), rolled.track(st, ref), rtol=0.0, atol=1e-15)

    # Non-vacuous: the term is present in both, not absent from both.
    assert np.max(np.abs(skew.track(st, ref) - SkewQuadrupole(L_Q, K1).track(st, ref))) > 1e-11


def test_zero_slices_is_the_shipped_map_and_a_negative_count_is_refused(
    ref: ReferenceParticle,
) -> None:
    """The default is L2's map to the last bit, and the knob validates its input."""
    plain = Quadrupole(L_Q, K1)
    np.testing.assert_array_equal(
        plain.track(STATE, ref), Quadrupole(L_Q, K1, kinematic_slices=0).track(STATE, ref)
    )
    assert plain.kinematic_slices == 0
    assert "kinematic_slices" not in repr(plain)
    assert "kinematic_slices=3" in repr(Quadrupole(L_Q, K1, kinematic_slices=3))

    with pytest.raises(ValueError, match="kinematic_slices"):
        Quadrupole(L_Q, K1, kinematic_slices=-1)
    with pytest.raises(ValueError, match="kinematic_slices"):
        SkewQuadrupole(L_Q, K1, kinematic_slices=-1)


def test_the_map_broadcasts_over_a_bunch(ref: ReferenceParticle) -> None:
    """A ``(6, n)`` bunch takes the same path as ``n`` single states."""
    rng = np.random.default_rng(20260903)
    bunch = np.zeros((6, 7))
    bunch[:4] = rng.normal(scale=1.0e-3, size=(4, 7))
    bunch[DELTA] = rng.normal(scale=2.0e-3, size=7)
    q = Quadrupole(L_Q, K1, kinematic_slices=3)
    together = q.track(bunch, ref)
    for i in range(7):
        np.testing.assert_allclose(together[:, i], q.track(bunch[:, i], ref), rtol=0.0, atol=1e-18)
