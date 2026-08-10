"""H1 acceptance (part 1): tune matching by Newton on the beta-weighted Jacobian.

Two quadrupole families -> a target ``(Q_x, Q_y)``. The physics content is the
**response matrix**

    dQ_x/dv = +(1/4pi) sum_i w_i (integral of beta_x over element i),
    dQ_y/dv = -(1/4pi) sum_i w_i (integral of beta_y over element i),

for a knob that sets each family member's strength to ``w_i * v``. Everything
else is bookkeeping around it.

The strong test does **not** re-sum that formula. It builds the thin one-turn map
as a symbolic function of the knob value, forms ``cos mu(v) = 1/2 Tr M(v)``, and
differentiates ``Q(v) = mu(v)/2pi`` — a derivative that knows nothing about beta,
about ``4pi``, or about the weights. Those three are properties of the
perturbation formula, not of the map, so agreement to machine precision pins all
of them at once (sign, coefficient, and the ``w_i`` weighting) the same way the
Stage-2 chromaticity test pins ``dQ/ddelta``.

The Newton step uses that first-order Jacobian but the *residual* is the exact
:func:`tunes`, so the converged answer is exact even though the Jacobian is
first-order — the recovery test asserts the strengths themselves, not just a
small residual.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Drift,
    Knob,
    Lattice,
    MatchingError,
    Quadrupole,
    ReferenceParticle,
    ThinQuadrupole,
    closed_twiss,
    match_tunes,
    tune_response_matrix,
    tunes,
)

# Nominal thin FODO: full-quad inverse focal lengths, half-cell drift.
VF_NOMINAL = 1.0 / 1.5  # F family knob value [m^-1]
VD_NOMINAL = 1.0 / 1.6  # D family knob value [m^-1]
L_HALF = 1.0  # half-cell drift length [m]


@pytest.fixture
def ref() -> ReferenceParticle:
    # Thin quads + drifts are energy-independent; any ref works.
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _thin_fodo(vf: float = VF_NOMINAL, vd: float = VD_NOMINAL) -> list:
    """Symmetric thin FODO from the F centre: half-F | drift | D | drift | half-F.

    Written so the two knobs below reproduce it exactly: the F family is the two
    half-quads at weight ``0.5`` (so ``v_f`` is the *full*-quad strength), and the
    D family is the single full quad at weight ``-1`` (so ``v_d`` is also a
    positive inverse focal length). Both fractional and negative weights are
    therefore exercised by every test in this file.
    """
    return [
        ThinQuadrupole(0.5 * vf, name="qf_half_a"),
        Drift(L_HALF, name="d1"),
        ThinQuadrupole(-1.0 * vd, name="qd"),
        Drift(L_HALF, name="d2"),
        ThinQuadrupole(0.5 * vf, name="qf_half_b"),
    ]


def _knobs(lat: Lattice) -> tuple[Knob, Knob]:
    """The (F, D) knobs of ``_thin_fodo``, with their non-unit weights."""
    kf = Knob([lat[0], lat[4]], weights=[0.5, 0.5], name="kqf")
    kd = Knob([lat[2]], weights=[-1.0], name="kqd")
    return kf, kd


def _symbolic_tune_jacobian(vf: float, vd: float, ll_val: float) -> np.ndarray:
    """Re-derive ``dQ_u/dv`` for both knobs from the v-dependent thin one-turn map.

    Independent of the beta-weighted response formula: the tune comes straight
    from the trace of the map, differentiated at the nominal strengths. Returns
    the 2x2 ``[[dQx/dvf, dQx/dvd], [dQy/dvf, dQy/dvd]]``.
    """
    sp = pytest.importorskip("sympy")
    f, d, ll = sp.symbols("v_f v_d L", positive=True)

    def q_of(sign: int):
        # sign = +1 for the x plane (F focuses x), -1 for y.
        # ThinQuadrupole(k1l) maps px -> px - k1l x and py -> py + k1l y, so the
        # 2x2 block is [[1, 0], [-sign*k1l, 1]]. The F members carry k1l = 0.5*v_f
        # (weight 0.5) and the D member k1l = -v_d (weight -1).
        qfh = sp.Matrix([[1, 0], [-sign * f / 2, 1]])
        qd = sp.Matrix([[1, 0], [+sign * d, 1]])
        drift = sp.Matrix([[1, ll], [0, 1]])
        m = qfh * drift * qd * drift * qfh  # one-turn block from the F centre
        cos_mu = (m[0, 0] + m[1, 1]) / 2
        return sp.acos(cos_mu) / (2 * sp.pi)  # mu in (0, pi) for this cell

    subs = {f: vf, d: vd, ll: ll_val}
    return np.array(
        [
            [float(sp.diff(q_of(+1), v).subs(subs)) for v in (f, d)],
            [float(sp.diff(q_of(-1), v).subs(subs)) for v in (f, d)],
        ]
    )


# --------------------------------------------------------------------------
# The response matrix: sign, coefficient and weights, against a symbolic dQ/dv
# --------------------------------------------------------------------------


def test_response_matrix_matches_symbolic_tune_derivative(ref: ReferenceParticle) -> None:
    """The beta-weighted Jacobian == d(tune)/d(knob) from the symbolic map.

    This is the acceptance test for the response matrix: it pins the sign
    (+ for x, - for y), the 1/4pi, and the fact that each member enters weighted
    by its own w_i — none of which the symbolic side knows about.
    """
    lat = Lattice(_thin_fodo(), ref)
    jac = tune_response_matrix(lat, _knobs(lat))
    sym = _symbolic_tune_jacobian(VF_NOMINAL, VD_NOMINAL, L_HALF)
    assert jac == pytest.approx(sym, rel=1e-10, abs=1e-12)

    # And the signs are the physical ones: more F focusing raises Q_x, lowers Q_y.
    assert jac[0, 0] > 0.0
    assert jac[1, 0] < 0.0


def test_response_matrix_is_beta_over_4pi_at_the_quad(ref: ReferenceParticle) -> None:
    """Spell the F column out in closed form: (1/4pi) * beta at the F centre.

    Both half-quads sit at a periodic cell's start/end, so both see the same
    ``beta``, and their weights sum to 1 — the column is exactly ``+beta_x/4pi``
    and ``-beta_y/4pi`` with no residual weight factor left over.
    """
    lat = Lattice(_thin_fodo(), ref)
    tw = closed_twiss(lat)
    jac = tune_response_matrix(lat, _knobs(lat))
    assert jac[0, 0] == pytest.approx(tw.beta_x / (4.0 * math.pi))
    assert jac[1, 0] == pytest.approx(-tw.beta_y / (4.0 * math.pi))


def test_response_matrix_weights_are_not_ignored(ref: ReferenceParticle) -> None:
    """Halving every weight halves the column — the easy miss is dropping w_i."""
    lat = Lattice(_thin_fodo(), ref)
    kf, kd = _knobs(lat)
    full = tune_response_matrix(lat, (kf, kd))
    half = Knob([lat[0], lat[4]], weights=[0.25, 0.25])
    scaled = tune_response_matrix(lat, (half, kd))
    assert scaled[:, 0] == pytest.approx(0.5 * full[:, 0])
    assert scaled[:, 1] == pytest.approx(full[:, 1])


def test_thick_quad_response_matches_finite_difference(ref: ReferenceParticle) -> None:
    """The thick-quad beta-integral path, against a finite-difference dQ/dk1.

    A thick quad's response is ``(1/4pi) * integral of beta over the body``, not
    ``beta`` at a point, so it exercises the sliced integration rather than the
    thin single-point term.
    """
    elems = [
        Quadrupole(0.3, 1.2, name="qf"),
        Drift(1.0),
        Quadrupole(0.3, -1.2, name="qd"),
        Drift(1.0),
    ]
    lat = Lattice(elems, ref)
    kf = Knob([elems[0]], name="kqf")
    kd = Knob([elems[2]], name="kqd")
    jac = tune_response_matrix(lat, (kf, kd), slices=256)

    h = 1e-6
    fd = np.zeros((2, 2))
    for j, knob in enumerate((kf, kd)):
        v0 = knob.value
        knob.apply(v0 + h)
        qp = np.array(tunes(lat))
        knob.apply(v0 - h)
        qm = np.array(tunes(lat))
        knob.apply(v0)
        fd[:, j] = (qp - qm) / (2.0 * h)
    assert jac == pytest.approx(fd, rel=2e-5)


# --------------------------------------------------------------------------
# The matcher: exact recovery of a known solution
# --------------------------------------------------------------------------


def test_match_tunes_recovers_the_known_strengths(ref: ReferenceParticle) -> None:
    """Target the tunes of a *known* lattice, start detuned, recover the strengths.

    Stronger than a small residual: the exact answer is known independently, so
    this catches a matcher that lands on a different (Q_x, Q_y)-equivalent root
    or that converges to the right tunes with the wrong optics.
    """
    known = Lattice(_thin_fodo(VF_NOMINAL, VD_NOMINAL), ref)
    target = tunes(known)

    # A substantial detune that is still stable: (0.149, 0.048) vs (0.115, 0.093).
    lat = Lattice(_thin_fodo(1.0 / 1.3, 1.0 / 1.7), ref)
    result = match_tunes(lat, target, _knobs(lat))

    assert result.values == pytest.approx((VF_NOMINAL, VD_NOMINAL), abs=1e-9)
    assert tunes(lat) == pytest.approx(target, abs=1e-12)
    assert result.residual < 1e-10
    assert result.iterations >= 1
    # The lattice really was mutated in place, to exactly the reported values.
    assert lat[0].k1l == pytest.approx(0.5 * VF_NOMINAL)
    assert lat[4].k1l == pytest.approx(0.5 * VF_NOMINAL)
    assert lat[2].k1l == pytest.approx(-VD_NOMINAL)


def test_match_tunes_hits_a_freely_chosen_target(ref: ReferenceParticle) -> None:
    """A target that is not the tune of any lattice we built by hand."""
    lat = Lattice(_thin_fodo(), ref)
    target = (0.2500, 0.1750)
    result = match_tunes(lat, target, _knobs(lat))
    assert tunes(lat) == pytest.approx(target, abs=1e-12)
    assert result.achieved == pytest.approx(target, abs=1e-12)
    assert result.initial == pytest.approx((VF_NOMINAL, VD_NOMINAL))


def test_match_tunes_already_on_target_is_a_no_op(ref: ReferenceParticle) -> None:
    lat = Lattice(_thin_fodo(), ref)
    target = tunes(lat)
    result = match_tunes(lat, target, _knobs(lat))
    assert result.iterations == 0
    assert result.values == pytest.approx((VF_NOMINAL, VD_NOMINAL))


def test_match_tunes_works_on_thick_quads(ref: ReferenceParticle) -> None:
    """The thick path end-to-end: first-order Jacobian, exact residual."""
    elems = [Quadrupole(0.3, 1.2), Drift(1.0), Quadrupole(0.3, -1.2), Drift(1.0)]
    lat = Lattice(elems, ref)
    target = (0.2100, 0.1600)
    match_tunes(lat, target, (Knob([elems[0]]), Knob([elems[2]])))
    assert tunes(lat) == pytest.approx(target, abs=1e-12)


def test_match_tunes_survives_a_step_into_instability(ref: ReferenceParticle) -> None:
    """A far-from-target start makes the full Newton step unstable; backtracking saves it.

    Without step-halving the loop dies on :class:`UnstableLatticeError` instead of
    converging — a FODO pushed toward the stability boundary is exactly where an
    overshooting first-order step lands.
    """
    lat = Lattice(_thin_fodo(1.0 / 0.55, 1.0 / 0.55), ref)  # near the boundary
    target = (0.1100, 0.1050)
    result = match_tunes(lat, target, _knobs(lat))
    assert tunes(lat) == pytest.approx(target, abs=1e-12)
    assert result.iterations >= 1


# --------------------------------------------------------------------------
# Refusals and rollback
# --------------------------------------------------------------------------


def test_knob_rejects_mixed_thick_and_thin(ref: ReferenceParticle) -> None:
    """``k1`` [m^-2] and ``k1l`` [m^-1] are different units — ganging them is a bug."""
    with pytest.raises(MatchingError, match="same strength attribute"):
        Knob([Quadrupole(0.3, 1.2), ThinQuadrupole(0.4)])


def test_knob_rejects_inconsistent_initial_strengths(ref: ReferenceParticle) -> None:
    """If the members are not already ganged, the knob's current value is undefined."""
    with pytest.raises(MatchingError, match="not consistent"):
        Knob([ThinQuadrupole(0.5), ThinQuadrupole(0.7)])


def test_knob_rejects_an_unsupported_element(ref: ReferenceParticle) -> None:
    with pytest.raises(MatchingError, match="no matchable strength"):
        Knob([Drift(1.0)])


def test_knob_rejects_a_zero_weight(ref: ReferenceParticle) -> None:
    """A zero-weight member can never move; silently keeping it hides a typo."""
    with pytest.raises(MatchingError, match="zero weight"):
        Knob([ThinQuadrupole(0.5), ThinQuadrupole(0.0)], weights=[1.0, 0.0])


def test_degenerate_knobs_are_refused_not_solved(ref: ReferenceParticle) -> None:
    """Knobs at equivalent optics give proportional columns — raise, don't return garbage.

    The two half-quads sit at the two ends of a periodic cell, so they see the
    same ``(beta_x, beta_y)``: driving them separately is one variable wearing two
    hats, and a bare 2x2 solve would hand back a huge meaningless step.
    """
    lat = Lattice(_thin_fodo(), ref)
    twins = (Knob([lat[0]], weights=[0.5]), Knob([lat[4]], weights=[0.5]))
    with pytest.raises(MatchingError, match="singular|conditioned"):
        match_tunes(lat, (0.25, 0.17), twins)


def test_overlapping_knobs_are_refused(ref: ReferenceParticle) -> None:
    """One element in two knobs is not two independent variables."""
    lat = Lattice(_thin_fodo(), ref)
    kf, _ = _knobs(lat)
    with pytest.raises(MatchingError, match="two knobs"):
        match_tunes(lat, (0.25, 0.17), (kf, Knob([lat[0]], weights=[0.5])))


def test_failure_restores_the_original_strengths(ref: ReferenceParticle) -> None:
    """A failed match must leave the lattice exactly as it found it."""
    lat = Lattice(_thin_fodo(), ref)
    before = [e.k1l for e in (lat[0], lat[2], lat[4])]
    with pytest.raises(MatchingError):
        # Unreachable: Q_x = Q_y = 2 is far outside this one-cell FODO's range.
        match_tunes(lat, (2.0, 2.0), _knobs(lat), max_iter=6)
    assert [e.k1l for e in (lat[0], lat[2], lat[4])] == before


def test_wrong_number_of_knobs_is_refused(ref: ReferenceParticle) -> None:
    lat = Lattice(_thin_fodo(), ref)
    kf, _ = _knobs(lat)
    with pytest.raises(MatchingError, match="exactly two"):
        match_tunes(lat, (0.25, 0.17), (kf,))
