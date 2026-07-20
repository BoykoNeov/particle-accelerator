r"""Analytic gate -- the LO ``gamma*/Z`` angular structure and the ``sin^2(theta_W)``
extraction from ``A_FB(m)`` (milestone A2).

Layered so that a wrong model and a wrong fitter cannot cancel:

1. **Symbolic derivation** (the physics). The ``(1 + cos^2 t)`` / ``cos t``
   decomposition that :mod:`accsim.events.electroweak` implements is re-derived
   here from explicit Dirac gamma matrices with *symbolic* couplings -- no
   remembered textbook amplitude -- and the module's ``S``/``D`` combination is
   checked against it term by term. This is what makes the model trustworthy;
   everything below only tests machinery.
2. **Coupling algebra.** ``g_V^l`` vanishes at ``sin^2(theta_W) = 1/4``, which is
   the origin of the measurement's sensitivity.
3. **Physics anchors.** The ``A_FB = (3/8) A_4`` identity against the *independent*
   extractor in :mod:`accsim.events.kinematics`, and the sign gate recorded in
   ``docs/CONVENTIONS.md`` (``A_FB < 0`` below the pole, ``> 0`` above).
4. **Round-trip.** Sample events from the model's own angular distribution at a
   known ``sin^2(theta_W)``, run them through the *real* ``A_FB`` extractor, and
   fit the angle back. Recovery to MC precision, with a pull check so the quoted
   error is not silently inflated -- "within fit error" is only a gate if the
   error is honest.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
import sympy as sp

from accsim.events import forward_backward_asymmetry
from accsim.events.electroweak import (
    CHARGED_LEPTON,
    DOWN_TYPE,
    GAMMA_Z,
    M_Z,
    UP_TYPE,
    Sin2ThetaWFit,
    _s_and_d,  # noqa: PLC2701 -- pinning internals
    afb_hadronic,
    afb_parton,
    fit_sin2_theta_w,
    neutral_current_couplings,
)

# ===========================================================================
# 1. Symbolic derivation of the angular structure
# ===========================================================================

_I2, _Z2 = sp.eye(2), sp.zeros(2, 2)
_SX = sp.Matrix([[0, 1], [1, 0]])
_SY = sp.Matrix([[0, -sp.I], [sp.I, 0]])
_SZ = sp.Matrix([[1, 0], [0, -1]])


def _blk(a, b, c, d):
    return sp.Matrix(sp.BlockMatrix([[a, b], [c, d]]))


_G = [
    _blk(_I2, _Z2, _Z2, -_I2),
    _blk(_Z2, _SX, -_SX, _Z2),
    _blk(_Z2, _SY, -_SY, _Z2),
    _blk(_Z2, _SZ, -_SZ, _Z2),
]
_G5 = sp.simplify(sp.I * _G[0] * _G[1] * _G[2] * _G[3])
_MET = sp.diag(1, -1, -1, -1)


def _slash(p):
    return _G[0] * p[0] - _G[1] * p[1] - _G[2] * p[2] - _G[3] * p[3]


@functools.lru_cache(maxsize=1)
def _symbolic_bilinear():
    """Spin-summed ``|M|^2`` for one mediator pair, as a polynomial in ``cos(theta)``.

    Massless fermions in the ``q qbar`` CM frame (which *is* the Collins-Soper
    frame at ``q_T = 0``). Couplings are symbols, so both mediators and every
    interference term are covered by the single expression.
    """
    E, c, s = sp.symbols("E c s", positive=True)
    sn = sp.sqrt(1 - c**2)
    vq, aq, vl, al = sp.symbols("vq aq vl al")
    vq2, aq2, vl2, al2 = sp.symbols("vq2 aq2 vl2 al2")

    p1 = [E, 0, 0, E]  # quark   -> +z defines the CS axis
    p2 = [E, 0, 0, -E]  # antiquark
    p3 = [E, E * sn, 0, E * c]  # l-
    p4 = [E, -E * sn, 0, -E * c]  # l+

    def vertex(v, a):
        return [_G[m] * (v * sp.eye(4) - a * _G5) for m in range(4)]

    def tensor(pa, pb, v, a, v2, a2):
        vm, vn = vertex(v, a), vertex(v2, a2)
        sa, sb = _slash(pa), _slash(pb)
        return sp.Matrix(4, 4, lambda m, n: sp.expand(sp.trace(sa * vm[m] * sb * vn[n])))

    had = tensor(p2, p1, vq, aq, vq2, aq2)
    lep = tensor(p3, p4, vl, al, vl2, al2)
    lep_down = _MET * lep * _MET
    m2 = sp.expand(sum(had[m, n] * lep_down[m, n] for m in range(4) for n in range(4)))
    m2 = sp.expand(sp.simplify(m2.subs(E, sp.sqrt(s) / 2)))
    return m2, (c, s, vq, aq, vl, al, vq2, aq2, vl2, al2)


def test_symbolic_angular_decomposition():
    r"""``|M|^2 = 4 s^2 [ (1+cos^2) * SYM + 2 cos * ASYM ]`` -- derived, not assumed.

    The coefficients of ``cos^0`` and ``cos^2`` must be *equal* (that is what makes
    the symmetric part ``1 + cos^2``), and the coupling structure of each piece is
    exactly what :func:`_s_and_d` builds.
    """
    m2, (c, s, vq, aq, vl, al, vq2, aq2, vl2, al2) = _symbolic_bilinear()
    poly = sp.Poly(m2, c)
    coeffs = {p[0]: sp.simplify(co) for p, co in poly.terms()}

    assert set(coeffs) == {0, 1, 2}, f"unexpected powers of cos(theta): {set(coeffs)}"

    sym = (vl * vl2 + al * al2) * (vq * vq2 + aq * aq2)
    asym = (al * vl2 + al2 * vl) * (aq * vq2 + aq2 * vq)

    # 1 + cos^2 structure: the constant and quadratic coefficients coincide.
    assert sp.simplify(coeffs[0] - coeffs[2]) == 0
    assert sp.simplify(coeffs[2] - 4 * s**2 * sym) == 0
    assert sp.simplify(coeffs[1] - 2 * (4 * s**2) * asym) == 0


def test_symbolic_afb_is_three_quarters_d_over_s():
    """``A_FB = (3/4) D / S`` follows from integrating the derived distribution."""
    m2, (c, s, vq, aq, vl, al, vq2, aq2, vl2, al2) = _symbolic_bilinear()
    fwd = sp.integrate(m2, (c, 0, 1))
    bwd = sp.integrate(m2, (c, -1, 0))
    afb = sp.simplify((fwd - bwd) / (fwd + bwd))

    sym = (vl * vl2 + al * al2) * (vq * vq2 + aq * aq2)
    asym = (al * vl2 + al2 * vl) * (aq * vq2 + aq2 * vq)
    assert sp.simplify(afb - sp.Rational(3, 4) * asym / sym) == 0


@pytest.mark.parametrize("mass", [70.0, 91.1876, 110.0])
@pytest.mark.parametrize("quark", [UP_TYPE, DOWN_TYPE])
def test_module_s_and_d_match_symbolic_two_mediator_sum(mass, quark):
    r"""The module's ``S``/``D`` equal the symbolic bilinear summed over mediators.

    Independent path: build the ``gamma``/``Z`` propagators and couplings here,
    substitute them into the *symbolic* expression, and sum the four mediator
    pairs -- then compare against :func:`_s_and_d`.
    """
    s2w = 0.2312
    m2, (c, s, vq, aq, vl, al, vq2, aq2, vl2, al2) = _symbolic_bilinear()

    gv_q, ga_q = neutral_current_couplings(quark, s2w)
    gv_l, ga_l = neutral_current_couplings(CHARGED_LEPTON, s2w)
    s_val = mass**2
    kappa = 1.0 / (4.0 * s2w * (1.0 - s2w))
    p_gamma = 1.0 / s_val
    p_z = kappa / (s_val - M_Z**2 + 1j * M_Z * GAMMA_Z)

    mediators = [
        (p_gamma, quark.charge, 0.0, CHARGED_LEPTON.charge, 0.0),
        (p_z, float(gv_q), float(ga_q), float(gv_l), float(ga_l)),
    ]

    poly = sp.Poly(m2, c)
    coeffs = {p[0]: co for p, co in poly.terms()}
    sym_expr, asym_expr = coeffs[2] / (4 * s**2), coeffs[1] / (8 * s**2)

    s_ref = d_ref = 0.0
    for pa, vqa, aqa, vla, ala in mediators:
        for pb, vqb, aqb, vlb, alb in mediators:
            sub = {
                vq: vqa,
                aq: aqa,
                vl: vla,
                al: ala,
                vq2: vqb,
                aq2: aqb,
                vl2: vlb,
                al2: alb,
                s: s_val,
            }
            w = float(np.real(pa * np.conj(pb)))
            s_ref += w * float(sym_expr.subs(sub))
            d_ref += w * float(asym_expr.subs(sub))

    s_mod, d_mod = _s_and_d(np.array([mass]), quark, s2w)
    assert s_mod[0] == pytest.approx(s_ref, rel=1e-12)
    assert d_mod[0] == pytest.approx(d_ref, rel=1e-12)


# ===========================================================================
# 2. Coupling algebra -- where the sensitivity comes from
# ===========================================================================


def test_axial_coupling_is_isospin_only():
    """``g_A = T3`` carries no ``sin^2(theta_W)`` dependence, for any angle."""
    for f in (UP_TYPE, DOWN_TYPE, CHARGED_LEPTON):
        for s2w in (0.1, 0.23, 0.4):
            _, g_a = neutral_current_couplings(f, s2w)
            assert float(g_a) == pytest.approx(f.t3, abs=0.0)


def test_leptonic_vector_coupling_vanishes_at_one_quarter():
    """``g_V^l = -1/2 + 2 s2w`` has a zero at ``s2w = 1/4`` -- the sensitivity engine.

    Near the physical value ``g_V^l`` is small, so its *relative* response to the
    mixing angle is large, and ``A_FB`` inherits that amplification.
    """
    g_v, _ = neutral_current_couplings(CHARGED_LEPTON, 0.25)
    assert float(g_v) == pytest.approx(0.0, abs=1e-15)

    # ...and it is linear with slope 2 (= -2 Q_l).
    lo, _ = neutral_current_couplings(CHARGED_LEPTON, 0.20)
    hi, _ = neutral_current_couplings(CHARGED_LEPTON, 0.30)
    assert float(hi - lo) == pytest.approx(0.2, rel=1e-12)


# ===========================================================================
# 3. Physics anchors -- sign gate and the A_FB = (3/8) A_4 identity
# ===========================================================================


@pytest.mark.parametrize("quark", [UP_TYPE, DOWN_TYPE])
def test_afb_sign_flips_across_the_z_pole(quark):
    """The recorded physics gate: ``A_FB < 0`` below ``M_Z``, ``> 0`` above.

    This is the sign structure ``docs/CONVENTIONS.md`` pins the pipeline against;
    the model must reproduce it independently.
    """
    s2w = 0.2312
    below = afb_parton(np.array([70.0, 80.0, 85.0]), quark, s2w)
    above = afb_parton(np.array([97.0, 105.0, 115.0]), quark, s2w)
    assert np.all(below < 0.0), below
    assert np.all(above > 0.0), above


@pytest.mark.parametrize("quark", [UP_TYPE, DOWN_TYPE])
def test_afb_has_a_zero_crossing_just_below_the_pole(quark):
    """The interference zero sits slightly under ``M_Z`` -- bracket it by bisection."""
    s2w = 0.2312
    lo, hi = 80.0, M_Z
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if afb_parton(mid, quark, s2w) < 0.0:
            lo = mid
        else:
            hi = mid
    crossing = 0.5 * (lo + hi)
    assert 80.0 < crossing < M_Z


def test_afb_matches_three_eighths_a4_of_the_derived_distribution():
    r"""``A_FB = (3/8) A_4``, with ``A_4 = 2 D / S`` read off the model.

    Anchors the model against the identity that
    :func:`accsim.events.angular_coefficients` is independently pinned to.
    """
    s2w = 0.2312
    masses = np.array([70.0, 85.0, 91.0, 100.0, 120.0])
    s_tot, d_tot = _s_and_d(masses, UP_TYPE, s2w)
    a4 = 2.0 * d_tot / s_tot
    np.testing.assert_allclose(afb_parton(masses, UP_TYPE, s2w), 0.375 * a4, rtol=1e-14)


def test_hadronic_mix_is_bracketed_by_its_flavours():
    """A luminosity-weighted ``A_FB`` lies between the pure up- and down-type curves.

    Guards the ``S``/``D``-level combination: averaging the *ratios* instead would
    still bracket, but weighting ``S`` and ``D`` separately is the correct rule and
    this at least catches a sum that escapes the physical envelope.
    """
    s2w = 0.2312
    masses = np.array([75.0, 91.0, 110.0])
    up = afb_parton(masses, UP_TYPE, s2w)
    down = afb_parton(masses, DOWN_TYPE, s2w)
    mixed = afb_hadronic(masses, s2w, {UP_TYPE: 2.0, DOWN_TYPE: 1.0})
    assert np.all(mixed >= np.minimum(up, down) - 1e-12)
    assert np.all(mixed <= np.maximum(up, down) + 1e-12)


def test_flavour_weight_scale_cancels():
    """Only relative parton luminosities matter -- an overall factor drops out."""
    s2w = 0.2312
    masses = np.array([75.0, 91.0, 110.0])
    a = afb_hadronic(masses, s2w, {UP_TYPE: 2.0, DOWN_TYPE: 1.0})
    b = afb_hadronic(masses, s2w, {UP_TYPE: 200.0, DOWN_TYPE: 100.0})
    np.testing.assert_allclose(a, b, rtol=1e-14)


# ===========================================================================
# 4. Round-trip -- recover an injected sin^2(theta_W)
# ===========================================================================

_BIN_EDGES = np.linspace(70.0, 110.0, 11)
_BIN_CENTRES = 0.5 * (_BIN_EDGES[:-1] + _BIN_EDGES[1:])
_WEIGHTS = {UP_TYPE: 2.0, DOWN_TYPE: 1.0}


def _sample_costheta(rng, s_tot, d_tot, n):
    """Accept-reject sampling from ``dsigma/dcos ~ S (1+cos^2) + 2 D cos``."""
    ceiling = 2.0 * s_tot + 2.0 * abs(d_tot)
    out = np.empty(n)
    filled = 0
    while filled < n:
        c = rng.uniform(-1.0, 1.0, size=2 * (n - filled) + 64)
        f = s_tot * (1.0 + c**2) + 2.0 * d_tot * c
        keep = c[rng.uniform(0.0, ceiling, size=c.size) < f]
        take = min(keep.size, n - filled)
        out[filled : filled + take] = keep[:take]
        filled += take
    return out


def _pseudo_experiment(rng, s2w_true, n_per_bin):
    """Generate events per mass bin and measure ``A_FB`` with the real extractor."""
    afb = np.empty(_BIN_CENTRES.size)
    err = np.empty(_BIN_CENTRES.size)
    for i, m in enumerate(_BIN_CENTRES):
        s_sum = d_sum = 0.0
        for quark, w in _WEIGHTS.items():
            s_tot, d_tot = _s_and_d(np.array([m]), quark, s2w_true)
            s_sum += w * float(s_tot[0])
            d_sum += w * float(d_tot[0])
        cos_theta = _sample_costheta(rng, s_sum, d_sum, n_per_bin)
        afb[i], err[i] = forward_backward_asymmetry(cos_theta)
    return afb, err


def test_noiseless_round_trip_is_exact():
    """Fit the model's own curve with no noise -- recovery must be near-exact.

    Isolates the fitter from statistics: any bias here is a bug in
    :func:`fit_sin2_theta_w`, not a fluctuation.
    """
    s2w_true = 0.2312
    truth = afb_hadronic(_BIN_CENTRES, s2w_true, _WEIGHTS)
    fit = fit_sin2_theta_w(_BIN_CENTRES, truth, np.full_like(truth, 1e-4), _WEIGHTS, initial=0.20)
    assert isinstance(fit, Sin2ThetaWFit)
    assert fit.sin2_theta_w == pytest.approx(s2w_true, abs=1e-9)
    assert fit.chi2 == pytest.approx(0.0, abs=1e-12)
    assert fit.ndof == _BIN_CENTRES.size - 1


@pytest.mark.parametrize("s2w_true", [0.2200, 0.2312, 0.2450])
def test_round_trip_recovers_injected_angle_from_sampled_events(s2w_true):
    """The full chain: sample -> extract ``A_FB`` -> fit. Recovery to MC precision.

    Deliberately routed through the *real* :func:`forward_backward_asymmetry`, so
    the model, the extractor and the fitter must all agree. Several injected values
    are used -- a fitter that ignored the data and returned its starting point
    would pass one, never three.
    """
    rng = np.random.default_rng(20260720)
    afb, err = _pseudo_experiment(rng, s2w_true, n_per_bin=200_000)
    fit = fit_sin2_theta_w(_BIN_CENTRES, afb, err, _WEIGHTS, initial=0.20)

    assert abs(fit.sin2_theta_w - s2w_true) < 4.0 * fit.error
    # The error must also be *small* -- "within error" is meaningless otherwise.
    assert fit.error < 2.0e-3
    assert fit.chi2_per_dof < 3.0


def test_fit_error_is_honest_pull_distribution():
    r"""Pulls ``(fit - truth)/error`` over many pseudo-experiments are unit-width.

    This is the guard against the trap that makes "recovered within fit error"
    vacuous: an inflated error would pass every individual round-trip while the
    pull width collapsed well below 1.
    """
    s2w_true = 0.2312
    rng = np.random.default_rng(7)
    pulls = []
    for _ in range(25):
        afb, err = _pseudo_experiment(rng, s2w_true, n_per_bin=20_000)
        fit = fit_sin2_theta_w(_BIN_CENTRES, afb, err, _WEIGHTS, initial=0.20)
        pulls.append((fit.sin2_theta_w - s2w_true) / fit.error)
    pulls = np.array(pulls)

    assert abs(float(np.mean(pulls))) < 0.6, f"biased pull mean: {np.mean(pulls)}"
    assert 0.5 < float(np.std(pulls)) < 1.8, f"pull width off: {np.std(pulls)}"


def test_fit_drops_unusable_bins():
    """Empty bins arrive as ``nan`` from the extractor and must not poison the fit."""
    s2w_true = 0.2312
    truth = afb_hadronic(_BIN_CENTRES, s2w_true, _WEIGHTS)
    err = np.full_like(truth, 1e-4)
    truth[2] = np.nan
    err[5] = np.nan
    err[7] = 0.0

    fit = fit_sin2_theta_w(_BIN_CENTRES, truth, err, _WEIGHTS, initial=0.20)
    assert fit.n_bins == _BIN_CENTRES.size - 3
    assert fit.sin2_theta_w == pytest.approx(s2w_true, abs=1e-9)


def test_fit_is_starting_point_independent():
    """Distinct starting guesses in the same basin land on the same minimum.

    A fit whose answer tracked its own initial value would sail through a single
    round-trip; this is what catches it.
    """
    s2w_true = 0.2312
    truth = afb_hadronic(_BIN_CENTRES, s2w_true, _WEIGHTS)
    err = np.full_like(truth, 1e-4)
    for x0 in (0.12, 0.18, 0.23, 0.30):
        fit = fit_sin2_theta_w(_BIN_CENTRES, truth, err, _WEIGHTS, initial=x0)
        assert fit.sin2_theta_w == pytest.approx(s2w_true, abs=1e-8), f"x0={x0}"


def test_fit_refuses_to_return_a_pinned_bound():
    r"""Converging onto a search bound is a failed fit, not a measurement.

    ``scipy.optimize.least_squares`` sets ``success=True`` when it settles on a
    bound, so without this guard a far-off starting guess silently returns the
    edge of the window (observed: ``initial=0.40`` -> ``0.45`` with
    ``chi^2 ~ 6e6``) dressed up as a result.
    """
    truth = afb_hadronic(_BIN_CENTRES, 0.2312, _WEIGHTS)
    err = np.full_like(truth, 1e-4)
    with pytest.raises(RuntimeError, match="converged onto the bound"):
        fit_sin2_theta_w(_BIN_CENTRES, truth, err, _WEIGHTS, initial=0.40, bounds=(0.05, 0.45))


def test_fit_recovers_a_wrong_hypothesis_not_its_prior():
    """Data generated at one angle must not be pulled to the starting guess."""
    generated_at = 0.2450
    data = afb_hadronic(_BIN_CENTRES, generated_at, _WEIGHTS)
    fit = fit_sin2_theta_w(_BIN_CENTRES, data, np.full_like(data, 1e-4), _WEIGHTS, initial=0.2312)
    assert fit.sin2_theta_w == pytest.approx(generated_at, abs=1e-8)


def test_chi2_has_real_curvature_around_the_minimum():
    r"""The ``chi^2`` must rise steeply away from the truth.

    A flat ``chi^2`` means the data carry no information about the angle, and any
    "recovered within error" claim would be empty. With ``sigma = 1e-3`` per bin a
    1e-3 shift in ``sin^2(theta_W)`` costs ``chi^2 >> 1``.
    """
    s2w_true = 0.2312
    truth = afb_hadronic(_BIN_CENTRES, s2w_true, _WEIGHTS)
    sigma = 1e-3

    def chi2(s2w):
        return float(np.sum(((truth - afb_hadronic(_BIN_CENTRES, s2w, _WEIGHTS)) / sigma) ** 2))

    assert chi2(s2w_true) == pytest.approx(0.0, abs=1e-12)
    assert chi2(s2w_true - 0.001) > 100.0
    assert chi2(s2w_true + 0.001) > 100.0


def test_fit_rejects_too_few_bins():
    with pytest.raises(ValueError, match="at least 2 usable bins"):
        fit_sin2_theta_w(
            np.array([91.0, 95.0]),
            np.array([0.01, np.nan]),
            np.array([1e-3, 1e-3]),
            _WEIGHTS,
        )


def test_hadronic_requires_a_flavour():
    with pytest.raises(ValueError, match="at least one quark flavour"):
        afb_hadronic(np.array([91.0]), 0.23, {})
