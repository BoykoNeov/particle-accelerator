"""H1 acceptance (part 2): chromaticity matching as an *exact* linear solve.

Two sextupole families -> a target ``(Q'_x, Q'_y)``. The structural claim that
makes this a one-shot solve rather than an iteration is:

    a sextupole's *linear* map is a drift,

so changing ``k2`` moves neither ``beta``, nor the dispersion, nor the tunes. The
total chromaticity is therefore **strictly affine** in the sextupole strengths,
``Q'(v) = Q'(v0) + S (v - v0)``, with a response matrix ``S`` that is a genuine
constant rather than a local linearisation.

That claim is what these tests attack, from three sides:

1. ``S`` computed at two *different* ``k2`` baselines is bit-for-bit identical —
   if it were only a local derivative, it would drift.
2. The post-solve residual is at machine precision, not merely inside a
   convergence tolerance.
3. The tunes and beta functions are unchanged by the match, so the "match tunes
   first, chromaticity second" ordering really is non-destructive.

The response coefficients themselves are pinned against a symbolically
differentiated ``dQ'/dv`` in the same style as part 1's tune Jacobian.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Dipole,
    Drift,
    Knob,
    Lattice,
    MatchingError,
    ReferenceParticle,
    Sextupole,
    ThinQuadrupole,
    ThinSextupole,
    chromaticity,
    chromaticity_response_matrix,
    closed_twiss,
    match_chromaticity,
    natural_chromaticity,
    propagate_twiss,
    tunes,
)

F_FOCAL = 2.2  # full-quad focal length [m]
L_DRIFT = 1.0  # element spacing [m]
BEND = 0.12  # dipole bend angle [rad]


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _dispersive_cell(k2l_f: float = 0.0, k2l_d: float = 0.0) -> list:
    """A dispersive FODO cell with an SF next to the F quad and an SD next to the D.

    Dipoles give ``D_x != 0`` (without which a sextupole has no chromaticity
    response at all), and placing SF at high ``beta_x`` / SD at high ``beta_y``
    is what makes the 2x2 response well-conditioned — the standard two-family
    correction scheme.
    """
    return [
        ThinQuadrupole(0.5 / F_FOCAL, name="qfh_a"),
        ThinSextupole(k2l_f, name="sf"),
        Drift(L_DRIFT, name="d1"),
        Dipole(L_DRIFT, BEND, name="b1"),
        ThinQuadrupole(-1.0 / F_FOCAL, name="qd"),
        ThinSextupole(k2l_d, name="sd"),
        Dipole(L_DRIFT, BEND, name="b2"),
        Drift(L_DRIFT, name="d2"),
        ThinQuadrupole(0.5 / F_FOCAL, name="qfh_b"),
    ]


def _sext_knobs(lat: Lattice) -> tuple[Knob, Knob]:
    """The (SF, SD) knobs. SD carries a negative weight, as a real scheme does."""
    return Knob([lat[1]], name="ksf"), Knob([lat[5]], weights=[-1.0], name="ksd")


def _symbolic_chroma_jacobian(lat: Lattice) -> np.ndarray:
    """Re-derive ``dQ'_u/dv`` from the feed-down definition, symbolically in the knobs.

    The kick ``Delta px = -1/2 k2l (x^2 - y^2)`` at ``x = x_beta + D_x delta``
    carries the linear gradient ``k1_eff = k2l D_x delta``. Feeding that through
    the *thin-quad* tune shift ``dQ_x = +beta_x dk1l / 4pi`` (the relation part 1
    already pinned symbolically) gives the response — built here from the
    propagated ``(beta, D_x)`` at each sextupole, without touching the production
    response matrix.
    """
    sp = pytest.importorskip("sympy")
    vf, vd = sp.symbols("v_f v_d", real=True)
    tw = propagate_twiss(lat, closed_twiss(lat))
    # Twiss points are entrance, then the exit of each element: index i is the
    # entrance of element i. beta and D_x are continuous across a thin kick.
    sf, sd = tw[1], tw[5]
    weights = {1: vf, 5: -vd}  # knob weights: SD is driven with weight -1
    xi_x = xi_y = sp.Integer(0)
    for point, idx in ((sf, 1), (sd, 5)):
        k2l = weights[idx]
        xi_x += +k2l * point.beta_x * point.disp_x / (4 * sp.pi)
        xi_y += -k2l * point.beta_y * point.disp_x / (4 * sp.pi)
    return np.array(
        [
            [float(sp.diff(xi_x, v)) for v in (vf, vd)],
            [float(sp.diff(xi_y, v)) for v in (vf, vd)],
        ]
    )


# --------------------------------------------------------------------------
# The response matrix: coefficients, and the exactness that makes it constant
# --------------------------------------------------------------------------


def test_chromaticity_response_matches_symbolic_feeddown(ref: ReferenceParticle) -> None:
    """The production response == the feed-down definition, built independently."""
    lat = Lattice(_dispersive_cell(), ref)
    resp = chromaticity_response_matrix(lat, _sext_knobs(lat))
    sym = _symbolic_chroma_jacobian(lat)
    assert resp == pytest.approx(sym, rel=1e-12, abs=1e-14)


def test_chromaticity_response_signs_are_the_correcting_ones(ref: ReferenceParticle) -> None:
    """``+`` in x and ``-`` in y at ``D_x > 0`` — the ``x^2 - y^2`` structure.

    This opposite-sign split is *why* two sextupole families can pull two negative
    natural chromaticities toward zero independently.
    """
    lat = Lattice(_dispersive_cell(), ref)
    tw = propagate_twiss(lat, closed_twiss(lat))
    assert tw[1].disp_x > 0.0 and tw[5].disp_x > 0.0  # both sextupoles are dispersive
    resp = chromaticity_response_matrix(lat, _sext_knobs(lat))
    assert resp[0, 0] > 0.0  # SF (weight +1) raises Q'_x
    assert resp[1, 0] < 0.0  # ... and lowers Q'_y
    assert resp[0, 1] < 0.0  # SD (weight -1) does the reverse
    assert resp[1, 1] > 0.0


def test_response_matrix_is_the_same_at_every_baseline(ref: ReferenceParticle) -> None:
    """The affineness claim, tested directly: ``S`` does not depend on ``k2``.

    A first-order Jacobian would drift with the baseline. This one must not — it
    is the exact constant gradient of an affine map, and that is the whole reason
    :func:`match_chromaticity` can solve in one shot.
    """
    zero = Lattice(_dispersive_cell(0.0, 0.0), ref)
    loaded = Lattice(_dispersive_cell(3.5, -4.25), ref)
    s_zero = chromaticity_response_matrix(zero, _sext_knobs(zero))
    s_loaded = chromaticity_response_matrix(loaded, _sext_knobs(loaded))
    assert s_loaded == pytest.approx(s_zero, rel=1e-14, abs=1e-16)


def test_chromaticity_is_affine_in_the_knobs(ref: ReferenceParticle) -> None:
    """``Q'(v) - Q'(0) == S v`` exactly, for a large, arbitrary ``v``."""
    base = Lattice(_dispersive_cell(0.0, 0.0), ref)
    resp = chromaticity_response_matrix(base, _sext_knobs(base))
    xi0 = np.array(chromaticity(base))

    v = np.array([7.0, 5.0])  # SD weight is -1, so k2l_d = -5.0
    moved = Lattice(_dispersive_cell(v[0], -v[1]), ref)
    predicted = xi0 + resp @ v
    assert np.array(chromaticity(moved)) == pytest.approx(predicted, rel=1e-12, abs=1e-14)


# --------------------------------------------------------------------------
# The matcher
# --------------------------------------------------------------------------


def test_match_chromaticity_to_zero_from_zero_strength(ref: ReferenceParticle) -> None:
    """The canonical job: cancel the natural chromaticity, starting from ``k2 = 0``.

    Starting from zero is the case a multiplicative knob could never solve, and
    the residual is asserted at machine precision — the solve is exact, not
    converged.
    """
    lat = Lattice(_dispersive_cell(0.0, 0.0), ref)
    natural = natural_chromaticity(lat)
    assert natural[0] < 0.0 and natural[1] < 0.0  # there is something to correct

    result = match_chromaticity(lat, (0.0, 0.0), _sext_knobs(lat))
    assert chromaticity(lat) == pytest.approx((0.0, 0.0), abs=1e-13)
    assert result.residual < 1e-13
    assert result.initial == (0.0, 0.0)
    assert result.iterations == 1  # one solve, by construction


def test_match_chromaticity_to_a_positive_target(ref: ReferenceParticle) -> None:
    """Head-tail stability wants slightly positive chromaticity, not zero."""
    lat = Lattice(_dispersive_cell(), ref)
    target = (1.5, 2.0)
    match_chromaticity(lat, target, _sext_knobs(lat))
    assert chromaticity(lat) == pytest.approx(target, abs=1e-13)


def test_match_chromaticity_is_independent_of_the_start(ref: ReferenceParticle) -> None:
    """One shot from anywhere: two very different starts land on the same strengths."""
    target = (0.0, 0.0)
    a = Lattice(_dispersive_cell(0.0, 0.0), ref)
    b = Lattice(_dispersive_cell(11.0, -9.0), ref)
    ra = match_chromaticity(a, target, _sext_knobs(a))
    rb = match_chromaticity(b, target, _sext_knobs(b))
    assert ra.values == pytest.approx(rb.values, rel=1e-12)


def test_matching_chromaticity_leaves_the_tunes_and_beta_alone(ref: ReferenceParticle) -> None:
    """The ordering guarantee: correcting chromaticity cannot undo a tune match.

    This is the linear-map-is-a-drift claim expressed as the property a user
    actually relies on.
    """
    lat = Lattice(_dispersive_cell(), ref)
    q_before = tunes(lat)
    tw_before = closed_twiss(lat)
    match_chromaticity(lat, (0.0, 0.0), _sext_knobs(lat))
    assert tunes(lat) == pytest.approx(q_before, abs=1e-15)
    assert closed_twiss(lat).beta_x == pytest.approx(tw_before.beta_x, rel=1e-15)
    assert closed_twiss(lat).beta_y == pytest.approx(tw_before.beta_y, rel=1e-15)
    assert closed_twiss(lat).disp_x == pytest.approx(tw_before.disp_x, rel=1e-15)


def test_match_chromaticity_works_on_thick_sextupoles(ref: ReferenceParticle) -> None:
    """The sliced beta*D_x integral path, mirroring ``chromaticity``'s own quadrature.

    Both integrators must use the same trapezoid and the same drift transport, or
    the "exact" solve would only be exact up to the discretisation difference.
    """
    ls = 0.4
    elems = [
        ThinQuadrupole(0.5 / F_FOCAL),
        Sextupole(ls, 0.0, name="sf"),
        Drift(L_DRIFT - ls),
        Dipole(L_DRIFT, BEND),
        ThinQuadrupole(-1.0 / F_FOCAL),
        Sextupole(ls, 0.0, name="sd"),
        Dipole(L_DRIFT, BEND),
        Drift(L_DRIFT - ls),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]
    lat = Lattice(elems, ref)
    knobs = (Knob([elems[1]]), Knob([elems[5]], weights=[-1.0]))
    result = match_chromaticity(lat, (0.0, 0.0), knobs)
    assert chromaticity(lat) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert result.residual < 1e-12


# --------------------------------------------------------------------------
# Refusals and rollback
# --------------------------------------------------------------------------


def test_quadrupole_knobs_are_refused_for_chromaticity(ref: ReferenceParticle) -> None:
    """A quad knob would break affineness — refuse rather than silently linearise."""
    lat = Lattice(_dispersive_cell(), ref)
    with pytest.raises(MatchingError, match="not be linear"):
        match_chromaticity(lat, (0.0, 0.0), (Knob([lat[0]]), Knob([lat[4]])))


def test_sextupole_knobs_are_refused_for_tune_matching(ref: ReferenceParticle) -> None:
    """The mirror refusal, with the reason: a sextupole's linear map is a drift."""
    from accsim import match_tunes

    lat = Lattice(_dispersive_cell(), ref)
    with pytest.raises(MatchingError, match="cannot change the tunes"):
        match_tunes(lat, (0.2, 0.2), _sext_knobs(lat))


def test_sextupole_at_zero_dispersion_is_refused(ref: ReferenceParticle) -> None:
    """No dispersion, no feed-down: the response row is zero, so the solve is singular.

    A drift+quad lattice has ``D_x = 0`` everywhere, so both sextupoles are
    perfectly useless and a bare solve would return infinities.
    """
    elems = [
        ThinQuadrupole(0.5 / F_FOCAL),
        ThinSextupole(0.0, name="sf"),
        Drift(L_DRIFT),
        ThinQuadrupole(-1.0 / F_FOCAL),
        ThinSextupole(0.0, name="sd"),
        Drift(L_DRIFT),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]
    lat = Lattice(elems, ref)
    assert closed_twiss(lat).disp_x == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(MatchingError, match="singular|conditioned"):
        match_chromaticity(lat, (0.0, 0.0), (Knob([elems[1]]), Knob([elems[4]])))


def test_failed_chromaticity_match_restores_strengths(ref: ReferenceParticle) -> None:
    """Rollback after the strengths were *already applied*, not just before.

    The only failure that can happen post-apply is the residual check, so it is
    provoked with an unsatisfiable ``tol``: the solve is exact to ~1e-16 (asserted
    elsewhere), so ``tol = 1e-20`` is guaranteed to reject a correct answer and
    force the restore path. A refusal raised *before* the first ``apply`` would
    pass this assertion vacuously.
    """
    lat = Lattice(_dispersive_cell(2.0, -3.0), ref)
    before = [lat[1].k2l, lat[5].k2l]
    with pytest.raises(MatchingError, match="missed by"):
        match_chromaticity(lat, (0.0, 0.0), _sext_knobs(lat), tol=1e-20)
    assert [lat[1].k2l, lat[5].k2l] == before


def test_desynced_sextupole_family_is_caught_at_match_time(ref: ReferenceParticle) -> None:
    """The exact linear solve re-checks ganging too, not only the Newton matcher.

    Same hazard as on the quadrupole side: ``Knob.value`` reads back the first
    member, so a member set directly behind the knob's back makes ``value``
    misreport where the lattice is — and the solve would then overwrite that
    member from a wrong ``initial``. The SF family is split into two co-located
    thin halves (physically the original single kick, since thin kicks at the
    same ``s`` add) purely so there is a second member to desync.
    """
    cell = _dispersive_cell()
    cell.insert(2, ThinSextupole(0.0, name="sf_b"))
    lat = Lattice(cell, ref)
    ksf = Knob([lat[1], lat[2]], weights=[0.5, 0.5], name="ksf")
    ksd = Knob([lat[6]], weights=[-1.0], name="ksd")

    lat[2].k2l = 0.7  # one half moved behind the knob's back, after construction
    with pytest.raises(MatchingError, match="not consistent"):
        match_chromaticity(lat, (0.0, 0.0), (ksf, ksd))
    assert lat[2].k2l == 0.7  # and nothing was overwritten
    assert lat[1].k2l == 0.0


def test_overlapping_chromaticity_knobs_are_refused(ref: ReferenceParticle) -> None:
    """One sextupole in two knobs is not two independent variables."""
    lat = Lattice(_dispersive_cell(2.0, -3.0), ref)
    with pytest.raises(MatchingError, match="two knobs"):
        match_chromaticity(lat, (0.0, 0.0), (Knob([lat[1]]), Knob([lat[1]])))


def test_wrong_number_of_chromaticity_knobs_is_refused(ref: ReferenceParticle) -> None:
    lat = Lattice(_dispersive_cell(), ref)
    sf, _ = _sext_knobs(lat)
    with pytest.raises(MatchingError, match="exactly two"):
        match_chromaticity(lat, (0.0, 0.0), (sf,))


# --------------------------------------------------------------------------
# The two halves together
# --------------------------------------------------------------------------


def test_tunes_then_chromaticity_is_a_stable_ordering(ref: ReferenceParticle) -> None:
    """Match the tunes, then the chromaticity; the tunes survive to machine precision.

    The end-to-end statement of what H1 delivers: the second match cannot undo the
    first, because sextupoles do not move the tunes.
    """
    from accsim import match_tunes

    lat = Lattice(_dispersive_cell(), ref)
    q_target = (0.2050, 0.1400)
    quad_knobs = (
        Knob([lat[0], lat[8]], weights=[0.5, 0.5], name="kqf"),
        Knob([lat[4]], weights=[-1.0], name="kqd"),
    )
    match_tunes(lat, q_target, quad_knobs)
    assert tunes(lat) == pytest.approx(q_target, abs=1e-12)

    xi_target = (0.5, 0.5)
    match_chromaticity(lat, xi_target, _sext_knobs(lat))
    assert chromaticity(lat) == pytest.approx(xi_target, abs=1e-13)
    assert tunes(lat) == pytest.approx(q_target, abs=1e-13)  # untouched


def test_matched_cell_has_the_textbook_correction_sign(ref: ReferenceParticle) -> None:
    """SF ends up positive and SD negative — the classic scheme, not an artefact.

    ``k2l_sf > 0`` at high ``beta_x`` pushes ``Q'_x`` up; ``k2l_sd < 0`` at high
    ``beta_y`` pushes ``Q'_y`` up (the ``-`` sign in the y row). Both natural
    chromaticities are negative, so both corrections must be "up".
    """
    lat = Lattice(_dispersive_cell(), ref)
    tw = propagate_twiss(lat, closed_twiss(lat))
    assert tw[1].beta_x > tw[1].beta_y  # SF sits where beta_x dominates
    assert tw[5].beta_y > tw[5].beta_x  # SD where beta_y does
    match_chromaticity(lat, (0.0, 0.0), _sext_knobs(lat))
    assert lat[1].k2l > 0.0  # SF focusing-plane corrector
    assert lat[5].k2l < 0.0  # SD defocusing-plane corrector
    assert math.isfinite(lat[1].k2l) and math.isfinite(lat[5].k2l)
