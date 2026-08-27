"""Analytic gates for the linear normal form (O1).

The object under test is the symplectic ``W`` for which the one-turn map is a plain
rotation, ``M = W R W^-1`` with ``R`` block-diagonal in 2x2 rotations by the tunes.

**Two of the three obvious checks cannot see the answer, and this file says so with a
test rather than a comment.** ``M = W R W^-1`` is invariant under ``W -> W D`` for any
``D`` that commutes with ``R``, which includes a per-plane rescaling *and* a per-plane
rotation; requiring ``W`` symplectic kills the rescaling and leaves three free angles.
``test_definition_and_symplecticity_are_blind_to_the_phase`` builds exactly that wrong
``W``, shows it passes both, and shows what does catch it.

Gates, ordered by how much they can catch:

  * **The Courant-Snyder tie (primary).** Under the phase convention chosen here — each
    eigenvector rotated until its own plane's *position* component is real and positive —
    the 2x2 diagonal blocks of ``W`` must *be* ``[[sqrt(beta), 0], [-alpha/sqrt(beta),
    1/sqrt(beta)]]`` built from :func:`accsim.closed_twiss`'s ``beta``/``alpha``. That is
    a Stage-1 quantity computed on a different route (a matched 2x2 block, not an
    eigenvector), so the agreement is what makes the convention a *choice* rather than a
    copy of xtrack's.
  * **The Edwards-Teng tie.** On a coupled, dispersion-free ring the 4D ``W`` must equal
    ``V . diag(B1, B2)`` — G2's decoupling transformation times each mode's
    Courant-Snyder block — with **no** residual per-mode rotation.
  * **The Courant-Snyder invariant.** ``2 J_x`` in normalised coordinates must equal
    ``(x^2 + (alpha x + beta px)^2)/beta``. This one gates the *normalisation*, which the
    phase-blind checks above do see, and does it against a closed form rather than
    against ``W`` itself.
  * **The order at which the 6D optics leaves the 4D optics.** The 6D normal form does
    not reproduce ``closed_twiss``: its ``beta_x``, its tune and its dispersion are the
    ring's response to a momentum *oscillating* at the synchrotron tune rather than a
    momentum held fixed. The claim gated here is the **exponent** — all three departures
    vanish as ``Q_s^2``, not ``Q_s``.
  * **Structural, and labelled as such** — reconstruction, symplecticity, the
    round-trip, and action invariance under tracking. None of them can see the phase.
  * **A refusal.** An RF-free ring has no 6D normal form: ``zeta`` and ``delta`` are a
    repeated unit eigenvalue and the third mode's symplectic norm is exactly zero. Fifth
    appearance of that one degeneracy (N3, N4, N5, I4).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from accsim import (
    DELTA,
    Drift,
    Lattice,
    NormalFormError,
    Quadrupole,
    ReferenceParticle,
    ThinQuadrupole,
    ThinSkewQuadrupole,
    UnstableLatticeError,
    actions,
    closed_twiss,
    coupled_twiss,
    from_normalized,
    normal_form,
    normal_mode_tunes,
    pzeta_from_delta,
    synchrotron_tune,
    to_normalized,
    tunes,
)
from accsim.twiss import _transverse_4d

# I4's radiating FODO ring, imported rather than rebuilt: it is the one ring in the suite
# with an RF cavity strong enough that the 6D and 4D optics visibly disagree, and a second
# copy of it is a second chance to get the harmonic number wrong. tests/ dirs are not
# import packages, so it is reached by path.
sys.path.insert(0, os.path.dirname(__file__))

import test_closed_orbit_6d as i4  # noqa: E402

rf_ring = i4.ring


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(0.938272e9, 5.0)


def _fodo(kq: float, cell_len: float = 2.0, nq_len: float = 0.3) -> list:
    d = (cell_len - 2 * nq_len) / 2.0
    return [Quadrupole(nq_len, kq), Drift(d), Quadrupole(nq_len, -kq), Drift(d)]


def _fodo_ring(ref: ReferenceParticle) -> Lattice:
    return Lattice(_fodo(1.2) * 4, ref)


def _coupled_ring(k1sl: float, ref: ReferenceParticle, split: float = 0.05) -> Lattice:
    """FODO ring with one thin skew kick and a tune split — no bends, so no dispersion."""
    base = _fodo(1.2) * 4
    return Lattice(base[:4] + [ThinSkewQuadrupole(k1sl)] + base[4:] + [ThinQuadrupole(split)], ref)


def _cs_block(beta: float, alpha: float) -> np.ndarray:
    """The Courant-Snyder normalising block ``[[sqrt(b), 0], [-a/sqrt(b), 1/sqrt(b)]]``."""
    rb = math.sqrt(beta)
    return np.array([[rb, 0.0], [-alpha / rb, 1.0 / rb]])


def _rot(theta: float) -> np.ndarray:
    return np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])


def _unit_symplectic(dim: int) -> np.ndarray:
    S = np.zeros((dim, dim))
    for p in range(dim // 2):
        S[2 * p, 2 * p + 1] = 1.0
        S[2 * p + 1, 2 * p] = -1.0
    return S


# --------------------------------------------------------------------------------------
# The primary gate: the convention is the Courant-Snyder one, verified against Stage 1
# --------------------------------------------------------------------------------------


def test_uncoupled_blocks_are_the_courant_snyder_matrix(ref: ReferenceParticle) -> None:
    """``W``'s diagonal blocks equal ``[[sqrt(b), 0], [-a/sqrt(b), 1/sqrt(b)]]`` exactly.

    ``beta``/``alpha`` come from :func:`closed_twiss`, which never forms an eigenvector.
    This is the only gate in the file that pins the three phase angles.
    """
    lat = _fodo_ring(ref)
    nf = normal_form(lat.one_turn_matrix(), method="4d")
    tw = closed_twiss(lat)
    for plane, (beta, alpha) in enumerate([(tw.beta_x, tw.alpha_x), (tw.beta_y, tw.alpha_y)]):
        block = nf.w[2 * plane : 2 * plane + 2, 2 * plane : 2 * plane + 2]
        assert np.allclose(block, _cs_block(beta, alpha), atol=1e-13, rtol=0.0)


def test_uncoupled_off_diagonal_blocks_vanish(ref: ReferenceParticle) -> None:
    """No x-y mixing without a coupling element — the off blocks are zero, not small."""
    nf = normal_form(_fodo_ring(ref).one_turn_matrix(), method="4d")
    assert np.abs(nf.w[0:2, 2:4]).max() < 1e-14
    assert np.abs(nf.w[2:4, 0:2]).max() < 1e-14


def test_mode_betas_reproduce_closed_twiss(ref: ReferenceParticle) -> None:
    """The ``beta``/``alpha`` read back off ``W`` are the ones that went in."""
    lat = _fodo_ring(ref)
    nf = normal_form(lat.one_turn_matrix(), method="4d")
    tw = closed_twiss(lat)
    assert nf.mode_beta[0] == pytest.approx(tw.beta_x, rel=1e-13)
    assert nf.mode_beta[1] == pytest.approx(tw.beta_y, rel=1e-13)
    assert nf.mode_alpha[0] == pytest.approx(tw.alpha_x, abs=1e-13)
    assert nf.mode_alpha[1] == pytest.approx(tw.alpha_y, abs=1e-13)


def test_mode_tunes_are_the_uncoupled_fractional_tunes(ref: ReferenceParticle) -> None:
    lat = _fodo_ring(ref)
    nf = normal_form(lat.one_turn_matrix(), method="4d")
    qx, qy = tunes(lat)
    assert nf.tunes[0] == pytest.approx(qx % 1.0, abs=1e-13)
    assert nf.tunes[1] == pytest.approx(qy % 1.0, abs=1e-13)


def test_definition_and_symplecticity_are_blind_to_the_phase(ref: ReferenceParticle) -> None:
    """The two structural checks pass on a ``W`` whose parameterisation is wrong.

    ``W' = W . diag(Rot(t1), Rot(t2))`` commutes with ``R``, so it reconstructs the same
    map and is just as symplectic — and its ``beta`` is not ``closed_twiss``'s. This is
    what the primary gate above is for, and it is the J1 lesson in a new place.
    """
    lat = _fodo_ring(ref)
    m4 = _transverse_4d(lat.one_turn_matrix())
    nf = normal_form(m4, method="4d")
    S = _unit_symplectic(4)

    spoiled = nf.w @ np.block([[_rot(0.7), np.zeros((2, 2))], [np.zeros((2, 2)), _rot(-1.3)]])
    # ...still reconstructs the map...
    assert np.allclose(spoiled @ nf.rotation @ np.linalg.inv(spoiled), m4, atol=1e-12)
    # ...and is still symplectic...
    assert np.allclose(spoiled.T @ S @ spoiled, S, atol=1e-12)
    # ...but its optics is wrong, and only the Courant-Snyder tie sees it.
    tw = closed_twiss(lat)
    assert spoiled[0, 0] ** 2 + spoiled[0, 1] ** 2 == pytest.approx(tw.beta_x, rel=1e-12)
    assert spoiled[0, 1] != pytest.approx(0.0, abs=1e-6)
    assert not np.allclose(spoiled[0:2, 0:2], _cs_block(tw.beta_x, tw.alpha_x), atol=1e-6, rtol=0.0)


# --------------------------------------------------------------------------------------
# The Edwards-Teng tie: the mixing, pinned against G2
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("k1sl", [0.02, 0.1])
def test_coupled_w_is_edwards_teng_times_courant_snyder(
    k1sl: float, ref: ReferenceParticle
) -> None:
    """On a dispersion-free coupled ring, ``W = V . diag(B1, B2)`` exactly.

    ``V`` is G2's decoupling transformation and ``B_i`` each mode's Courant-Snyder
    block. Nothing is fitted: the eigenvector route and the Edwards-Teng route are
    independent and must land on the same matrix, not merely on the same map.
    """
    lat = _coupled_ring(k1sl, ref)
    nf = normal_form(lat.one_turn_matrix(), method="4d")
    ct = coupled_twiss(lat)
    blocks = np.block(
        [
            [_cs_block(ct.beta_1, ct.alpha_1), np.zeros((2, 2))],
            [np.zeros((2, 2)), _cs_block(ct.beta_2, ct.alpha_2)],
        ]
    )
    assert np.allclose(nf.w, ct.v_matrix @ blocks, atol=1e-13, rtol=0.0)


@pytest.mark.parametrize("k1sl", [0.02, 0.1])
def test_coupled_mode_tunes_match_the_eigenvalue_route(k1sl: float, ref: ReferenceParticle) -> None:
    lat = _coupled_ring(k1sl, ref)
    nf = normal_form(lat.one_turn_matrix(), method="4d")
    q1, q2 = normal_mode_tunes(lat)
    assert nf.tunes[0] == pytest.approx(q1, abs=1e-13)
    assert nf.tunes[1] == pytest.approx(q2, abs=1e-13)


# --------------------------------------------------------------------------------------
# The normalisation, gated against a closed form rather than against W
# --------------------------------------------------------------------------------------


def test_action_is_the_courant_snyder_invariant(ref: ReferenceParticle) -> None:
    """``2 J_x = (x^2 + (alpha x + beta px)^2)/beta`` — the textbook invariant.

    The phase-blind checks see the *scale* of ``W`` but only through symplecticity;
    this ties it to a formula with no free constant.
    """
    lat = _fodo_ring(ref)
    tw = closed_twiss(lat)
    nf = normal_form(lat.one_turn_matrix(), method="4d")
    state = np.array([1.3e-3, -2.1e-4, 0.7e-3, 1.1e-4, 0.0, 0.0])
    jx, jy = actions(nf, state)
    x, px, y, py = state[0], state[1], state[2], state[3]
    cs_x = (x**2 + (tw.alpha_x * x + tw.beta_x * px) ** 2) / tw.beta_x
    cs_y = (y**2 + (tw.alpha_y * y + tw.beta_y * py) ** 2) / tw.beta_y
    assert 2.0 * jx == pytest.approx(cs_x, rel=1e-12)
    assert 2.0 * jy == pytest.approx(cs_y, rel=1e-12)


def test_normalised_round_trip(ref: ReferenceParticle) -> None:
    nf = normal_form(_fodo_ring(ref).one_turn_matrix(), method="4d")
    state = np.array([1.3e-3, -2.1e-4, 0.7e-3, 1.1e-4, 0.0, 0.0])
    assert np.allclose(from_normalized(nf, to_normalized(nf, state)), state[:4], atol=1e-18)


def test_actions_are_invariant_under_tracking(ref: ReferenceParticle) -> None:
    """A linear lattice moves the normalised point around a circle and nothing else.

    Structural — blind to the phase convention, since a rotated ``W`` gives the same
    radius — but it is the statement the whole change of variables exists to make.
    """
    lat = _fodo_ring(ref)
    m4 = _transverse_4d(lat.one_turn_matrix())
    nf = normal_form(m4, method="4d")
    state = np.array([1.3e-3, -2.1e-4, 0.7e-3, 1.1e-4])
    j0 = actions(nf, state)
    for _ in range(512):
        state = m4 @ state
        j = actions(nf, state)
        assert j[0] == pytest.approx(j0[0], rel=1e-11)
        assert j[1] == pytest.approx(j0[1], rel=1e-11)


def test_normalised_turn_is_a_rotation_by_the_tune(ref: ReferenceParticle) -> None:
    """One turn advances the normalised angle by exactly ``2 pi Q``, both planes."""
    lat = _fodo_ring(ref)
    m4 = _transverse_4d(lat.one_turn_matrix())
    nf = normal_form(m4, method="4d")
    state = np.array([1.3e-3, -2.1e-4, 0.7e-3, 1.1e-4])
    u0 = to_normalized(nf, state)
    u1 = to_normalized(nf, m4 @ state)
    for plane in range(2):
        a0 = math.atan2(u0[2 * plane + 1], u0[2 * plane])
        a1 = math.atan2(u1[2 * plane + 1], u1[2 * plane])
        assert (a0 - a1) % (2.0 * math.pi) == pytest.approx(
            2.0 * math.pi * nf.tunes[plane], abs=1e-11
        )


# --------------------------------------------------------------------------------------
# 6D: what the RF changes, and the order at which it stops mattering
# --------------------------------------------------------------------------------------


def test_6d_normal_form_reconstructs_and_is_symplectic() -> None:
    """Structural, on the 6D form: both checks that cannot see the parameterisation."""
    lat, _ = rf_ring()
    m6 = lat.one_turn_matrix()
    nf = normal_form(m6, method="6d")
    assert np.allclose(nf.w @ nf.rotation @ nf.w_inv, m6, atol=1e-12)
    S = _unit_symplectic(6)
    assert np.allclose(nf.w.T @ S @ nf.w, S, atol=1e-12)


def test_6d_optics_is_not_the_4d_optics() -> None:
    """On a strong-RF ring the two disagree by percents, and that is not a bug.

    Pinned so that a later "fix" that makes them agree fails here: the 4D quantities
    answer a momentum held fixed, the 6D ones a momentum oscillating at ``Q_s``.
    """
    lat, _ = rf_ring()
    tw = closed_twiss(lat)
    nf6 = normal_form(lat.one_turn_matrix(), method="6d")
    nf4 = normal_form(lat.one_turn_matrix(), method="4d")
    # the 4D form is the one that must equal closed_twiss...
    assert nf4.mode_beta[0] == pytest.approx(tw.beta_x, rel=1e-12)
    assert nf4.tunes[0] == pytest.approx(tunes(lat)[0] % 1.0, abs=1e-12)
    # ...and the 6D one must not, on this ring.
    assert nf6.mode_beta[0] / tw.beta_x == pytest.approx(0.925, abs=0.01)
    assert nf6.tunes[0] - nf4.tunes[0] == pytest.approx(-6.5e-3, abs=2e-4)
    assert nf6.dispersion[0] / tw.disp_x == pytest.approx(1.239, abs=0.01)


def test_6d_departs_from_4d_quadratically_in_the_synchrotron_tune() -> None:
    """The milestone's claim: exponent 2, not 1, in ``Q_s`` — for all three quantities.

    Fitted over a decade in ``Q_s`` (the RF voltage down by a factor 64). An exponent
    of 1 would mean the 6D form carried a linear-in-``Q_s`` error; a wrong
    normalisation shows up here where a tolerance at any single voltage would not.
    """
    voltages = [1.40625e6, 3.515625e5, 8.7890625e4, 2.197265625e4]
    qs, d_disp, d_beta, d_tune = [], [], [], []
    for voltage in voltages:
        lat, _ = rf_ring(voltage=voltage)
        m6 = lat.one_turn_matrix()
        nf6 = normal_form(m6, method="6d")
        nf4 = normal_form(m6, method="4d")
        tw = closed_twiss(lat)
        qs.append(synchrotron_tune(lat))
        d_disp.append(abs(nf6.dispersion[0] / tw.disp_x - 1.0))
        d_beta.append(abs(nf6.mode_beta[0] / tw.beta_x - 1.0))
        d_tune.append(abs(nf6.tunes[0] - nf4.tunes[0]))
    log_qs = np.log(np.array(qs))
    for name, dev in (("dispersion", d_disp), ("beta", d_beta), ("tune", d_tune)):
        slope = float(np.polyfit(log_qs, np.log(np.array(dev)), 1)[0])
        assert slope == pytest.approx(2.0, abs=0.02), f"{name}: exponent {slope}"


def test_rf_free_ring_has_no_6d_normal_form() -> None:
    """``zeta`` and ``delta`` are a repeated unit eigenvalue — refused, not divided by.

    Fifth appearance of the one degeneracy: N3 met it, N4 explained it, N5 hit it in the
    spin field, I4 in the 6D closed orbit.
    """
    lat, icav = rf_ring()
    plain = Lattice(lat.elements[:icav] + lat.elements[icav + 1 :], ref=lat.ref)
    with pytest.raises(NormalFormError):
        normal_form(plain.one_turn_matrix(), method="6d")
    # ...and the 4D form of the very same ring is fine.
    nf = normal_form(plain.one_turn_matrix(), method="4d")
    assert nf.mode_beta[0] == pytest.approx(closed_twiss(plain).beta_x, rel=1e-12)


def test_unstable_lattice_is_rejected(ref: ReferenceParticle) -> None:
    lat = Lattice(_fodo(8.0) * 4, ref)  # past the |Tr| < 2 stability boundary
    with pytest.raises(UnstableLatticeError):
        normal_form(lat.one_turn_matrix(), method="4d")


def test_pzeta_and_delta_are_the_same_variable_at_linear_order(
    ref: ReferenceParticle,
) -> None:
    """Asserted, not assumed: ``dpzeta/ddelta = 1`` exactly at ``delta = 0``.

    xtrack writes ``W`` in ``(x, px, y, py, zeta, pzeta)`` where accsim's one-turn matrix
    is in ``delta``. The two linear maps coincide because ``dE/ddelta = beta0 P0`` and
    ``pzeta = (E - E0)/(beta0^2 E0)``, so there is no ``beta0^2`` anywhere in the
    comparison.
    """
    step = 1e-8
    slope = (pzeta_from_delta(step, ref) - pzeta_from_delta(-step, ref)) / (2.0 * step)
    assert slope == pytest.approx(1.0, abs=1e-12)
    # ...and the same statement in closed form, with no finite difference.
    assert ref.momentum_eV**2 / (ref.beta0**2 * ref.total_energy_eV**2) == pytest.approx(
        1.0, rel=1e-15
    )


def test_method_must_be_named() -> None:
    lat, _ = rf_ring()
    with pytest.raises(ValueError, match="method"):
        normal_form(lat.one_turn_matrix(), method="5d")


def test_6d_state_round_trips_through_the_full_form() -> None:
    lat, _ = rf_ring()
    nf = normal_form(lat.one_turn_matrix(), method="6d")
    state = np.zeros(6)
    state[0], state[1], state[2], state[3] = 1.3e-4, -2.1e-5, 0.7e-4, 1.1e-5
    state[4], state[DELTA] = 1.0e-3, 2.0e-4
    assert np.allclose(from_normalized(nf, to_normalized(nf, state)), state, atol=1e-18)
    assert len(actions(nf, state)) == 3
