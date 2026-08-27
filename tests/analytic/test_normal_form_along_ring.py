"""Analytic gates for the normal form along the ring (O2).

O1 built ``W`` at one point. This file builds it at every element boundary,
``W(s) = M(0->s) W(0)`` re-phased back into O1's convention, and gates the three things
that only exist once ``W`` is a function of ``s``: the accumulated phase advance
``mu(s)``, the Mais-Ripken **cross-plane** betas (how much of mode 2 is carried in ``x``),
and the **crab dispersion** (how the transverse orbit depends on arrival time).

**The blindness is worse here than in O1, and again this file says so with a test.** Every
new quantity the milestone delivers is invariant under the very re-phasing that is O2's
whole operation: ``betx2 = |v2[x]|^2`` and ``alfx2 = -Re(v2[x] conj(v2[px]))`` both have
the phase cancel between their two factors, and ``dx_zeta`` is a ratio of components
*within one eigenvector*. So a wrong re-phasing changes none of them.
``test_the_new_quantities_are_blind_to_the_re_phasing`` builds a mis-phased ``W(s)`` and
shows exactly that. Only two things can see it:

  * the convention itself, ``W[2p, 2p+1] = 0`` with ``W[2p, 2p] > 0``, at every point;
  * ``mu(s)`` -- and that gate is **quantised, not a tolerance**. ``tunes()`` returns the
    *full* integer-plus-fractional tune, so if the unwrap ever drops a branch the answer
    is wrong by exactly ``1`` and no tolerance can absorb it.
    ``test_no_element_advances_the_phase_by_more_than_pi`` is the localiser that says
    *why* if it ever fires.

The rest, ordered by how much they can catch:

  * **The Edwards-Teng tie along the ring.** ``W(s) = V(s) . diag(B1(s), B2(s))`` at every
    point on a coupled ring -- G2's decoupling transform, computed by re-matching the
    *local* one-turn map rather than by transporting anything. That is what ties
    ``betx2``/``bety1`` to a quantity derived on a completely different route.
  * **The crab dispersion's two exponents.** ``dx_zeta`` is not zero on an ordinary ring
    with no crab cavity: it is the *phase lag* of the dispersion, because the transverse
    response to a momentum oscillating at ``Q_s`` is not in phase with it. The lag is
    **linear** in ``Q_s``; ``dx_zeta`` itself is **quadratic**, because the longitudinal
    mode's momentum content is linear in ``Q_s`` too. Both exponents are gated, and so is
    the identity that multiplies them -- one mechanism, two orders, which is a much
    tighter statement than either number alone. On a **bend-free** ring it is exactly
    zero, which is what makes it dispersive.
  * **What ``dx_zeta`` means, checked by tracking.** A particle launched on pure
    longitudinal motion has ``x(s) = D(s) delta(s) + dx_zeta(s) zeta(s)`` at every point.
    That regression runs off the lattice's own element maps and never touches ``W``.
  * **Action invariance along the ring.** The mode actions of a tracked particle are the
    same at every ``s`` when each point's own ``W`` is used -- which gates the whole
    propagation, normalisation included.
  * **Structural, and labelled as such** -- reconstruction of the local one-turn map,
    symplecticity at every point, and the ring closing on itself. None of them see the
    phase.
  * **A guard for the element set, not the algebra.**
    ``_transverse_4d(A B) = _transverse_4d(A) _transverse_4d(B)`` only because no accsim
    element makes the transverse coordinates depend on ``zeta`` and none makes ``delta``
    depend on the transverse ones. A crab cavity would break it, so it is asserted rather
    than assumed.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from accsim import (
    DELTA,
    ZETA,
    Drift,
    Lattice,
    NormalFormError,
    NormalFormPoint,
    Quadrupole,
    ReferenceParticle,
    ThinQuadrupole,
    ThinSkewQuadrupole,
    actions,
    closed_normal_form,
    closed_twiss,
    propagate_coupled_twiss,
    propagate_normal_form,
    propagate_twiss,
    tunes,
)
from accsim.coords import PX, X
from accsim.twiss import _transverse_4d

sys.path.insert(0, os.path.dirname(__file__))

import test_closed_orbit_6d as i4  # noqa: E402

ELECTRON_MASS_EV = 0.51099895069e6


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(0.938272e9, 5.0)


def _fodo(kq: float, cell_len: float = 2.0, nq_len: float = 0.3) -> list:
    d = (cell_len - 2 * nq_len) / 2.0
    return [Quadrupole(nq_len, kq), Drift(d), Quadrupole(nq_len, -kq), Drift(d)]


def _fodo_ring(ref: ReferenceParticle) -> Lattice:
    return Lattice(_fodo(1.2) * 4, ref)


def _coupled_ring(k1sl: float, ref: ReferenceParticle, split: float = 0.05) -> Lattice:
    """FODO ring with one thin skew kick and a tune split -- no bends, so no dispersion."""
    base = _fodo(1.2) * 4
    return Lattice(base[:4] + [ThinSkewQuadrupole(k1sl)] + base[4:] + [ThinQuadrupole(split)], ref)


def _bend_free_rf_ring(ref: ReferenceParticle) -> Lattice:
    """A FODO ring with a cavity and no dipoles: 6D optics, but nothing dispersive."""
    els = _fodo(1.2) * 4
    plain = Lattice(els, ref)
    from accsim import RFCavity

    return Lattice([*els, RFCavity.from_harmonic(1.0e5, 8, plain.length, ref, phi_s=0.0)], ref)


def _rot(theta: float) -> np.ndarray:
    return np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])


def _S(dim: int) -> np.ndarray:
    s = np.zeros((dim, dim))
    for p in range(dim // 2):
        s[2 * p, 2 * p + 1] = 1.0
        s[2 * p + 1, 2 * p] = -1.0
    return s


def _cs_block(beta: float, alpha: float) -> np.ndarray:
    rb = math.sqrt(beta)
    return np.array([[rb, 0.0], [-alpha / rb, 1.0 / rb]])


def _synchrotron_tune(lattice: Lattice) -> float:
    """The fractional synchrotron tune, folded into ``[0, 0.5]``."""
    return min(min(t, 1.0 - t) for t in closed_normal_form(lattice, method="6d").tunes)


# --------------------------------------------------------------------------------------
# Structural: true of any valid propagation, and blind to the phase
# --------------------------------------------------------------------------------------


def test_every_point_normalises_its_own_local_one_turn_map(ref: ReferenceParticle) -> None:
    """``W(s) R W(s)^-1`` must be the one-turn map *starting at s*, not the one at 0."""
    lat = _coupled_ring(0.1, ref)
    form = closed_normal_form(lat, method="4d")
    points = propagate_normal_form(lat, form)
    one_turn = _transverse_4d(lat.one_turn_matrix())
    transfer = np.eye(6)
    for i, point in enumerate(points):
        if i:
            transfer = lat.elements[i - 1].matrix(lat.ref) @ transfer
        t4 = _transverse_4d(transfer)
        local = t4 @ one_turn @ np.linalg.inv(t4)
        assert np.allclose(point.w @ form.rotation @ point.w_inv, local, atol=1e-12)


def test_w_is_symplectic_at_every_point(ref: ReferenceParticle) -> None:
    """No renormalisation is done in the library, so this is a real measurement."""
    for lat, method in ((_coupled_ring(0.1, ref), "4d"), (i4.ring()[0], "6d")):
        points = propagate_normal_form(lat, closed_normal_form(lat, method=method))
        dim = points[0].dim
        worst = max(np.abs(p.w.T @ _S(dim) @ p.w - _S(dim)).max() for p in points)
        assert worst < 1e-12, f"{method}: symplecticity drifted to {worst:.3e}"


def test_the_ring_closes_on_itself(ref: ReferenceParticle) -> None:
    """``W(C) = W(0)`` -- the last point is the first, which the phase convention forces."""
    lat, _ = i4.ring()
    points = propagate_normal_form(lat, closed_normal_form(lat, method="6d"))
    assert np.allclose(points[-1].w, points[0].w, atol=1e-9)
    assert points[0].s == 0.0
    assert points[-1].s == pytest.approx(lat.length)
    assert len(points) == len(lat.elements) + 1


def test_the_convention_holds_at_every_point(ref: ReferenceParticle) -> None:
    """``W[2p, 2p+1] = 0`` and ``W[2p, 2p] > 0`` -- one of the two things re-phasing does."""
    lat, _ = i4.ring()
    for point in propagate_normal_form(lat, closed_normal_form(lat, method="6d")):
        for p in range(3):
            assert abs(point.w[2 * p, 2 * p + 1]) < 1e-13
            assert point.w[2 * p, 2 * p] > 0.0


# --------------------------------------------------------------------------------------
# The blindness, and the two things that are not blind
# --------------------------------------------------------------------------------------


def test_the_new_quantities_are_blind_to_the_re_phasing(ref: ReferenceParticle) -> None:
    """Re-phasing is O2's whole operation, and every quantity O2 delivers cannot see it.

    ``betx2``, ``alfx2``, ``bety1`` and the crab dispersion are all invariant under
    ``W -> W diag(Rot, Rot, Rot)``: the phase cancels between the two factors of each
    product, and the crab dispersion is a ratio taken *inside* one eigenvector. So a
    wrong re-phasing would ship silently through all of them. What catches it is the
    convention (``W[2p, 2p+1] = 0``) and, around the ring, ``mu``.
    """
    lat, _ = i4.ring()
    point = propagate_normal_form(lat, closed_normal_form(lat, method="6d"))[7]
    spoil = np.zeros((6, 6))
    for p, theta in enumerate((0.7, -1.3, 2.1)):
        spoil[2 * p : 2 * p + 2, 2 * p : 2 * p + 2] = _rot(theta)
    spoiled = NormalFormPoint(
        point.s, point.w @ spoil, np.linalg.inv(point.w @ spoil), point.mu, "6d"
    )

    assert np.allclose(spoiled.betas, point.betas, atol=1e-12)
    assert np.allclose(spoiled.alphas, point.alphas, atol=1e-12)
    assert np.allclose(spoiled.gammas, point.gammas, atol=1e-12)
    assert np.allclose(spoiled.crab_dispersion, point.crab_dispersion, atol=1e-12)
    assert np.allclose(spoiled.dispersion, point.dispersion, atol=1e-12)
    # ... and it is still symplectic, so that check cannot see it either.
    assert np.allclose(spoiled.w.T @ _S(6) @ spoiled.w, _S(6), atol=1e-12)
    # What does see it:
    assert abs(spoiled.w[0, 1]) > 1e-3
    assert abs(point.w[0, 1]) < 1e-13


def test_mu_over_one_turn_is_the_full_integer_plus_fractional_tune(ref: ReferenceParticle) -> None:
    """A quantised gate: a dropped unwrap branch is wrong by exactly 1, not by a little.

    ``tunes()`` accumulates the phase advance through :func:`propagate_twiss`'s
    Courant-Snyder recursion, so its integer part is a genuinely independent count of how
    many times the phase went round.

    Both rings are chosen for having an integer part at all -- the four-cell FODO used
    elsewhere in this file reaches only ``0.206`` per turn, where the gate would be
    vacuous. The I4 ring adds bends and a cavity, so it also exercises the 4D path on a
    lattice whose full map is anything but transverse.
    """
    for lat in (Lattice(_fodo(1.2) * 24, ref), i4.ring()[0]):
        points = propagate_normal_form(lat, closed_normal_form(lat, method="4d"))
        qx, qy = tunes(lat)
        assert qx > 1.0 and qy > 1.0, "a gate on the integer part needs an integer part"
        assert points[-1].mu[0] / (2.0 * math.pi) == pytest.approx(qx, abs=1e-12)
        assert points[-1].mu[1] / (2.0 * math.pi) == pytest.approx(qy, abs=1e-12)


def test_no_element_advances_the_phase_by_more_than_pi(ref: ReferenceParticle) -> None:
    """The localiser for the gate above: the unwrap is only safe while this holds."""
    for lat, method in (
        (_fodo_ring(ref), "4d"),
        (_coupled_ring(0.1, ref), "4d"),
        (i4.ring()[0], "6d"),
    ):
        points = propagate_normal_form(lat, closed_normal_form(lat, method=method))
        mu = np.array([p.mu for p in points])
        step = np.abs(np.diff(mu, axis=0)).max()
        assert step < math.pi, f"{method}: an element advances the phase by {step:.3f} rad"


def test_mu_reproduces_propagate_twiss_everywhere(ref: ReferenceParticle) -> None:
    """Against Stage 1's phase advance, which is a Courant-Snyder recursion, not an angle."""
    lat = _fodo_ring(ref)
    points = propagate_normal_form(lat, closed_normal_form(lat, method="4d"))
    for point, tw in zip(points, propagate_twiss(lat, closed_twiss(lat)), strict=True):
        assert point.s == pytest.approx(tw.s, abs=1e-12)
        assert point.mu[0] == pytest.approx(tw.mu_x, abs=1e-12)
        assert point.mu[1] == pytest.approx(tw.mu_y, abs=1e-12)


def test_beta_and_alpha_reproduce_propagate_twiss_everywhere(ref: ReferenceParticle) -> None:
    """The O1 Courant-Snyder tie, now at every point rather than only at the entrance."""
    lat = _fodo_ring(ref)
    points = propagate_normal_form(lat, closed_normal_form(lat, method="4d"))
    for point, tw in zip(points, propagate_twiss(lat, closed_twiss(lat)), strict=True):
        assert np.allclose(point.w[0:2, 0:2], _cs_block(tw.beta_x, tw.alpha_x), atol=1e-13)
        assert np.allclose(point.w[2:4, 2:4], _cs_block(tw.beta_y, tw.alpha_y), atol=1e-13)
        assert point.mode_beta[0] == pytest.approx(tw.beta_x, rel=1e-13)
        assert point.mode_alpha[1] == pytest.approx(tw.alpha_y, abs=1e-13)


# --------------------------------------------------------------------------------------
# Mais-Ripken: the cross-plane betas
# --------------------------------------------------------------------------------------


def test_an_uncoupled_ring_carries_no_cross_plane_beta(ref: ReferenceParticle) -> None:
    """``betx2`` and ``bety1`` are exactly zero without coupling -- they exist only for it."""
    lat = _fodo_ring(ref)
    for point in propagate_normal_form(lat, closed_normal_form(lat, method="4d")):
        assert abs(point.betas[0, 1]) < 1e-28  # betx2
        assert abs(point.betas[1, 0]) < 1e-28  # bety1
        assert abs(point.alphas[0, 1]) < 1e-28
        assert abs(point.alphas[1, 0]) < 1e-28


def test_gamma_ties_to_stage_ones_beta_and_alpha(ref: ReferenceParticle) -> None:
    """``gammas`` is read off the *momentum* row, so nothing else in this file gates it.

    ``betas`` and ``alphas`` are pinned by ``propagate_twiss``, by Edwards-Teng and by
    xtrack. ``gammas`` is pinned by none of those -- writing row ``2p`` where row
    ``2p+1`` belongs would make it equal ``betas`` and every other test in both files
    would still pass, because they only ever check that it is *invariant* under the
    re-phasing. So it gets its own tie, to the same Stage 1 quantity: on an uncoupled
    ring each mode's ``gamma`` must be ``(1 + alpha^2)/beta``.
    """
    lat = _fodo_ring(ref)
    points = propagate_normal_form(lat, closed_normal_form(lat, method="4d"))
    for point, tw in zip(points, propagate_twiss(lat, closed_twiss(lat)), strict=True):
        assert point.gammas[0, 0] == pytest.approx((1.0 + tw.alpha_x**2) / tw.beta_x, rel=1e-13)
        assert point.gammas[1, 1] == pytest.approx((1.0 + tw.alpha_y**2) / tw.beta_y, rel=1e-13)
    # ...and it is genuinely a different number from beta, so the tie is not vacuous.
    assert points[3].gammas[0, 0] != pytest.approx(points[3].betas[0, 0], rel=1e-3)


def test_the_three_ripken_matrices_close_on_the_symplectic_identity(ref: ReferenceParticle) -> None:
    r"""``beta gamma - alpha^2 = det^2`` per entry, and the determinants sum to one.

    The coupled counterpart of ``beta gamma - alpha^2 = 1``, which holds per plane only
    when the planes are uncoupled. What survives coupling comes straight from ``W`` being
    symplectic: for each **mode** ``m``,

        sum over planes p of  det [[W[2p,2m],   W[2p,2m+1]  ],
                                   [W[2p+1,2m], W[2p+1,2m+1]]]  =  1,

    and each of those determinants squared is ``B[p,m] G[p,m] - A[p,m]^2`` by Lagrange's
    identity. Written on a coupled ring so all four entries of every matrix are nonzero --
    which is what makes this a gate on ``gammas``' *formula* rather than on its symmetry.
    """
    lat = _coupled_ring(0.1, ref)
    for point in propagate_normal_form(lat, closed_normal_form(lat, method="4d")):
        w, b, a, g = point.w, point.betas, point.alphas, point.gammas
        assert np.abs(b * g).min() > 1e-6, "a gate on all four entries needs all four"
        for m in range(2):
            det = [
                w[2 * p, 2 * m] * w[2 * p + 1, 2 * m + 1]
                - w[2 * p, 2 * m + 1] * w[2 * p + 1, 2 * m]
                for p in range(2)
            ]
            assert sum(det) == pytest.approx(1.0, abs=1e-13)
            for p in range(2):
                assert b[p, m] * g[p, m] - a[p, m] ** 2 == pytest.approx(det[p] ** 2, abs=1e-13)


def test_the_ripken_matrix_ties_to_edwards_teng_along_the_ring(ref: ReferenceParticle) -> None:
    r"""``W(s) = V(s) . diag(B1(s), B2(s))`` at every point on a coupled ring.

    G2's :func:`propagate_coupled_twiss` re-matches the **local** one-turn map at each
    point and transports nothing, so this is the same object reached by a route that
    shares no code with the eigenvector solve. It is what ties ``betx2``/``bety1`` --
    which have no closed form of their own -- to something already validated.
    """
    for k1sl in (0.02, 0.1):
        lat = _coupled_ring(k1sl, ref)
        points = propagate_normal_form(lat, closed_normal_form(lat, method="4d"))
        for point, ct in zip(points, propagate_coupled_twiss(lat), strict=True):
            blocks = np.zeros((4, 4))
            blocks[0:2, 0:2] = _cs_block(ct.beta_1, ct.alpha_1)
            blocks[2:4, 2:4] = _cs_block(ct.beta_2, ct.alpha_2)
            assert np.allclose(point.w, ct.v_matrix @ blocks, atol=1e-12)


def test_cross_plane_beta_grows_as_the_square_of_the_skew_strength(ref: ReferenceParticle) -> None:
    """``betx2 = |v2[x]|^2`` and ``v2[x]`` is first order in the skew kick, so this is 2.

    A uniform mis-scale of ``W``'s off-diagonal block would leave the ratio alone, so the
    exponent is the discriminating statement -- not the value at any one strength.

    The window matters and is chosen deliberately. ``2`` is the **asymptotic** exponent;
    the next term is fourth order and still worth measuring at the strengths the coupling
    tests elsewhere use, where the fitted slope is ``1.90`` at ``k1sl = 0.01`` and
    ``1.67`` at ``0.04``. Fitting there and calling it ``2`` would need a tolerance three
    times this one. The window below is where the fourth-order term has left.
    """
    strengths = [0.0025, 0.00125, 0.000625, 0.0003125]
    peak = []
    for k1sl in strengths:
        lat = _coupled_ring(k1sl, ref)
        points = propagate_normal_form(lat, closed_normal_form(lat, method="4d"))
        peak.append(max(p.betas[0, 1] for p in points))
    slope = np.polyfit(np.log(strengths), np.log(peak), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.02)


# --------------------------------------------------------------------------------------
# Crab dispersion: the transverse orbit's dependence on arrival time
# --------------------------------------------------------------------------------------


def test_crab_dispersion_is_exactly_zero_on_a_bend_free_ring(ref: ReferenceParticle) -> None:
    """It is dispersive: with no bends the transverse rows never see ``delta`` at all."""
    lat = _bend_free_rf_ring(ReferenceParticle.from_gamma(ELECTRON_MASS_EV, 5.0))
    for point in propagate_normal_form(lat, closed_normal_form(lat, method="6d")):
        assert np.abs(point.crab_dispersion).max() < 1e-25


def test_the_dispersion_phase_lag_is_linear_in_the_synchrotron_tune() -> None:
    """Why an ordinary ring has crab dispersion at all: the response lags the drive.

    ``c0 = v3[x] / v3[delta]`` is the transverse response to the oscillating momentum. In
    the ``Q_s -> 0`` limit it is the real 4D matched dispersion; at finite ``Q_s`` the
    ring is driven off-resonance and the response acquires a phase, first order.
    """
    qs, lag = [], []
    for voltage in (1.40625e6, 3.515625e5, 8.7890625e4, 2.197265625e4):
        lat, _ = i4.ring(voltage=voltage)
        w = closed_normal_form(lat, method="6d").w
        c0 = complex(w[X, 4], w[X, 5]) / complex(w[DELTA, 4], w[DELTA, 5])
        qs.append(_synchrotron_tune(lat))
        lag.append(abs(math.atan2(c0.imag, c0.real)))
    assert np.polyfit(np.log(qs), np.log(lag), 1)[0] == pytest.approx(1.0, abs=0.01)


def test_crab_dispersion_is_quadratic_in_the_synchrotron_tune() -> None:
    """Two exponents from one mechanism -- and this is the one that is *not* the lag.

    ``dx_zeta`` is the lag times the longitudinal mode's momentum content, and that
    content is linear in ``Q_s`` too (the longitudinal ellipse elongates as the cavity
    weakens). One power each, so the product is quadratic. Getting only the lag right
    would give ``1``; getting only the ellipse right would give ``1``; this asks for both.
    """
    qs, dxz, dpxz = [], [], []
    for voltage in (1.40625e6, 3.515625e5, 8.7890625e4, 2.197265625e4):
        lat, _ = i4.ring(voltage=voltage)
        crab = closed_normal_form(lat, method="6d").crab_dispersion
        qs.append(_synchrotron_tune(lat))
        dxz.append(abs(crab[0]))
        dpxz.append(abs(crab[1]))
    assert np.polyfit(np.log(qs), np.log(dxz), 1)[0] == pytest.approx(2.0, abs=0.02)
    assert np.polyfit(np.log(qs), np.log(dpxz), 1)[0] == pytest.approx(2.0, abs=0.02)


def test_crab_dispersion_factorises_into_the_lag_and_the_longitudinal_ellipse() -> None:
    r"""The identity behind the two exponents, asserted rather than left in prose.

        dx_zeta = - gamma_3 Im(c0) / sigma_3

    with ``gamma_3 = |v3[delta]|^2`` the mode's momentum content (linear in ``Q_s``),
    ``Im(c0)`` the lag (linear in ``Q_s``) and ``sigma_3`` the longitudinal share of the
    unit symplectic norm (order one). This is algebra on ``W``, so it proves no physics
    on its own -- it is here because it is what makes the two exponents above one
    statement instead of two coincidences.
    """
    lat, _ = i4.ring(voltage=1.40625e6)
    point = propagate_normal_form(lat, closed_normal_form(lat, method="6d"))[5]
    w = point.w
    b = complex(w[DELTA, 4], w[DELTA, 5])
    c = complex(w[ZETA, 4], w[ZETA, 5])
    for i, row in enumerate((X, PX)):
        c0 = complex(w[row, 4], w[row, 5]) / b
        gamma_3 = abs(b) ** 2
        sigma_3 = (b * c.conjugate()).imag
        assert point.crab_dispersion[i] == pytest.approx(-gamma_3 * c0.imag / sigma_3, rel=1e-11)


def test_crab_dispersion_is_the_tracked_dependence_of_x_on_arrival_time() -> None:
    """What the number *means*, measured off the lattice's own element maps.

    A particle launched on pure longitudinal motion has, at every point around the ring,

        x(s) = D(s) delta(s) + dx_zeta(s) zeta(s),

    with ``D`` the dynamic dispersion and ``dx_zeta`` the crab dispersion. Regressing the
    tracked ``x`` on the tracked ``(delta, zeta)`` recovers both, and never touches ``W``.
    """
    lat, _ = i4.ring(voltage=22.5e6)
    points = propagate_normal_form(lat, closed_normal_form(lat, method="6d"))
    w = points[0].w
    starts = [
        w @ np.array([0.0, 0.0, 0.0, 0.0, r * math.cos(a), r * math.sin(a)])
        for r, a in ((1.0, 0.0), (1.0, 1.1), (0.6, 2.3))
    ]
    trajectories = []
    for state in starts:
        row = [np.asarray(state, dtype=float)]
        for elem in lat.elements:
            row.append(elem.matrix(lat.ref) @ row[-1])
        trajectories.append(row)
    for k, point in enumerate(points):
        sample = np.array([t[k] for t in trajectories])
        design = np.column_stack([sample[:, DELTA], sample[:, ZETA]])
        for i, row in enumerate((X, PX)):
            coeff, *_ = np.linalg.lstsq(design, sample[:, row], rcond=None)
            assert coeff[0] == pytest.approx(point.dispersion[i], abs=1e-9)
            assert coeff[1] == pytest.approx(point.crab_dispersion[i], abs=1e-9)


def test_the_dynamic_dispersion_reaches_the_matched_one_as_the_cavity_weakens() -> None:
    """Element by element, and quadratically -- O1's ``Q_s^2`` result, now along ``s``."""
    qs, gap = [], []
    for voltage in (1.40625e6, 3.515625e5, 8.7890625e4, 2.197265625e4):
        lat, _ = i4.ring(voltage=voltage)
        points = propagate_normal_form(lat, closed_normal_form(lat, method="6d"))
        matched = propagate_twiss(lat, closed_twiss(lat))
        qs.append(_synchrotron_tune(lat))
        gap.append(
            max(abs(p.dispersion[0] - t.disp_x) for p, t in zip(points, matched, strict=True))
        )
    assert np.polyfit(np.log(qs), np.log(gap), 1)[0] == pytest.approx(2.0, abs=0.02)


# --------------------------------------------------------------------------------------
# The propagation itself
# --------------------------------------------------------------------------------------


def test_actions_are_invariant_along_the_ring(ref: ReferenceParticle) -> None:
    """The strongest single gate on the propagation: a global invariant, measured locally.

    The mode actions belong to the particle, not to the point. Tracking element by
    element and reading the actions off *each point's own* ``W`` must give the same three
    numbers everywhere -- which can only happen if both the transport and the
    normalisation of ``W(s)`` are right.
    """
    lat, _ = i4.ring()
    points = propagate_normal_form(lat, closed_normal_form(lat, method="6d"))
    state = points[0].w @ np.array([1e-3, 0.0, 5e-4, 0.0, 2e-3, 0.0])
    reference = actions(points[0], state)
    for i, point in enumerate(points):
        if i:
            state = lat.elements[i - 1].matrix(lat.ref) @ state
        for got, want in zip(actions(point, state), reference, strict=True):
            assert got == pytest.approx(want, rel=1e-9)


def test_the_transverse_block_of_a_product_is_the_product_of_the_blocks() -> None:
    """A guard on accsim's element set, not on the algebra.

    ``_transverse_4d(A B) = _transverse_4d(A) _transverse_4d(B)`` needs
    ``A[0:4,4] B[4,0:4] + A[0:4,5] B[5,0:4] = 0``. Both terms vanish only because no
    element makes the transverse coordinates depend on ``zeta`` (there is no crab cavity)
    and none makes ``delta`` depend on the transverse ones (the cavity kicks ``delta``
    from ``zeta`` alone). A crab cavity, or a radiation map passed through ``maps``, would
    break it -- so the 4D propagation is asserted to be unambiguous rather than assumed.
    """
    lat, _ = i4.ring()
    full = np.eye(6)
    blocks = np.eye(4)
    for elem in lat.elements:
        m = elem.matrix(lat.ref)
        full = m @ full
        blocks = _transverse_4d(m) @ blocks
    assert np.abs(_transverse_4d(full) - blocks).max() == 0.0


def test_maps_substitutes_the_transport_bit_for_bit(ref: ReferenceParticle) -> None:
    """Passing the on-axis matrices explicitly must reproduce the default exactly."""
    lat = _coupled_ring(0.1, ref)
    maps = [e.matrix(lat.ref) for e in lat.elements]
    form = closed_normal_form(lat, method="4d")
    for a, b in zip(
        propagate_normal_form(lat, form),
        propagate_normal_form(lat, form, maps=maps),
        strict=True,
    ):
        assert np.array_equal(a.w, b.w)
        assert a.mu == b.mu


def test_a_wrong_number_of_maps_is_rejected(ref: ReferenceParticle) -> None:
    lat = _fodo_ring(ref)
    with pytest.raises(ValueError, match="one matrix per element"):
        propagate_normal_form(lat, closed_normal_form(lat, method="4d"), maps=[np.eye(6)])


def test_a_four_dimensional_point_has_no_dispersion(ref: ReferenceParticle) -> None:
    """The 4D form knows nothing about momentum, so both dispersions must refuse."""
    lat = _fodo_ring(ref)
    point = propagate_normal_form(lat, closed_normal_form(lat, method="4d"))[3]
    with pytest.raises(NormalFormError, match="6D"):
        _ = point.dispersion
    with pytest.raises(NormalFormError, match="6D"):
        _ = point.crab_dispersion
    assert point.dim == 4
    assert point.betas.shape == (2, 2)


def test_an_rf_free_ring_still_propagates_its_four_dimensional_form(ref: ReferenceParticle) -> None:
    """The fifth-time degeneracy is O1's, not O2's: ``method='4d'`` is the answer here too."""
    lat = _fodo_ring(ref)
    points = propagate_normal_form(lat, closed_normal_form(lat, method="4d"))
    assert len(points) == len(lat.elements) + 1
    assert all(p.dim == 4 for p in points)
