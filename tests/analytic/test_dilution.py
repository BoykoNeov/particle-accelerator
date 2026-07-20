r"""Analytic gates for the ``pp`` dilution unfolding (milestone A3).

The gate the roadmap asks for: *the unfolded proxy reproduces the undiluted
true-quark-direction curve*. Here the undiluted curve is
:func:`accsim.events.afb_hadronic` -- A2's independently validated model -- so
the two sides of the closure are not the same code path.

The tests are layered so that a wrong dilution model and a wrong unfolding
cannot cancel:

* the two exactly-known limits (perfect axis, no information) pin
  :func:`afb_diluted` against ``afb_hadronic`` and against zero;
* the formula-level closure runs a **multi-flavour** toy with genuinely
  different per-flavour dilution *and* different per-flavour ``A_FB``, so the
  flavour weighting is load bearing -- and the naive PDF-only factor is asserted
  to give a **provably wrong** answer on the same input, which is what makes the
  closure a test of the method rather than of arithmetic;
* a sampled Monte-Carlo closure drives real four-vectors through the actual
  ``sign(Q_z)`` proxy (:func:`accsim.events.collins_soper_costheta`) and the real
  :func:`accsim.events.forward_backward_asymmetry`, so the analytic split into
  "aligned" and "reversed" luminosities is checked against event kinematics
  rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim.events import (
    DOWN_TYPE,
    UP_TYPE,
    afb_diluted,
    afb_hadronic,
    afb_parton,
    collins_soper_costheta,
    dilution_factor,
    forward_backward_asymmetry,
    parton_x,
    pdf_dilution,
    unfold_afb,
)
from accsim.events.electroweak import _s_and_d

SQRT_S = 13000.0
S2W = 0.2315  # the effective leptonic angle, near the measured value

# Discrete mass points, deliberately straddling the Z pole so the sign change of
# A_FB is exercised (below the pole A_FB < 0, above it > 0).
MASSES = np.array([72.0, 82.0, 91.1876, 100.0, 112.0])

# Rapidity nodes for the toy luminosity integral. The central region is excluded
# on purpose: that is where the proxy carries no information (D_eff -> 0) and is
# covered by its own degeneracy test below.
RAPIDITY = np.linspace(0.3, 2.4, 22)


# --------------------------------------------------------------------------
# A toy proton: two flavours with different valence hardness, so that the two
# quark species have different *dilution* as well as different A_FB.
# --------------------------------------------------------------------------
def _u(x: np.ndarray) -> np.ndarray:
    return 2.0 * x**-0.4 * (1.0 - x) ** 3.0 + _sea_u(x)


def _sea_u(x: np.ndarray) -> np.ndarray:
    return 0.30 * x**-1.0 * (1.0 - x) ** 8.0


def _d(x: np.ndarray) -> np.ndarray:
    return 1.0 * x**-0.4 * (1.0 - x) ** 5.5 + _sea_d(x)


def _sea_d(x: np.ndarray) -> np.ndarray:
    return 0.45 * x**-1.0 * (1.0 - x) ** 7.0


_QUARK = {UP_TYPE: _u, DOWN_TYPE: _d}
_ANTIQUARK = {UP_TYPE: _sea_u, DOWN_TYPE: _sea_d}


def _luminosity_grid(masses: np.ndarray, rapidity: np.ndarray) -> tuple[dict, dict]:
    """Per-(mass, rapidity) aligned/reversed luminosities, shape ``(n_m, n_y)``.

    For ``y > 0`` the higher-``x`` beam is beam 1, so the proxy-correct
    ("aligned") configuration is quark-from-beam-1: ``q(x1) qbar(x2)``.
    """
    m = masses[:, None]
    y = rapidity[None, :]
    x1, x2 = parton_x(m, y, SQRT_S)
    aligned = {f: _QUARK[f](x1) * _ANTIQUARK[f](x2) for f in _QUARK}
    reversed_ = {f: _ANTIQUARK[f](x1) * _QUARK[f](x2) for f in _QUARK}
    return aligned, reversed_


def _luminosity(masses: np.ndarray = MASSES) -> tuple[dict, dict]:
    """Rapidity-summed aligned/reversed luminosities per mass bin."""
    aligned, reversed_ = _luminosity_grid(masses, RAPIDITY)
    return (
        {f: v.sum(axis=1) for f, v in aligned.items()},
        {f: v.sum(axis=1) for f, v in reversed_.items()},
    )


def _undiluted(masses: np.ndarray, aligned: dict, reversed_: dict) -> np.ndarray:
    """The parton-level curve: A2's ``afb_hadronic`` at the total luminosity."""
    return afb_hadronic(masses, S2W, {f: aligned[f] + reversed_[f] for f in aligned})


# --------------------------------------------------------------------------
# Kinematics helper
# --------------------------------------------------------------------------
def test_parton_x_inverts_mass_and_rapidity() -> None:
    """``x1 x2 = m^2/s`` and ``y = (1/2) ln(x1/x2)`` -- the LO definition."""
    m = np.array([50.0, 91.1876, 200.0])
    y = np.array([-1.7, 0.0, 2.2])
    x1, x2 = parton_x(m, y, SQRT_S)
    assert np.allclose(x1 * x2, m**2 / SQRT_S**2, rtol=0, atol=1e-15)
    assert np.allclose(0.5 * np.log(x1 / x2), y, rtol=0, atol=1e-13)
    # x1 > x2 exactly when y > 0 -- the statement that makes sign(Q_z) a proxy
    # for "the quark came from the higher-x beam".
    assert np.all(np.sign(x1 - x2) == np.sign(y))


# --------------------------------------------------------------------------
# The two exactly-known limits
# --------------------------------------------------------------------------
def test_perfect_orientation_reproduces_afb_hadronic() -> None:
    """No reversed luminosity => no dilution => exactly A2's hadronic curve."""
    aligned, _ = _luminosity()
    zero = {f: np.zeros_like(v) for f, v in aligned.items()}
    assert np.allclose(
        afb_diluted(MASSES, S2W, aligned, zero),
        afb_hadronic(MASSES, S2W, aligned),
        rtol=0,
        atol=1e-15,
    )


def test_no_orientation_information_gives_zero_asymmetry() -> None:
    """Equal aligned/reversed luminosity: the proxy is a coin flip, A_FB = 0."""
    aligned, _ = _luminosity()
    assert np.allclose(afb_diluted(MASSES, S2W, aligned, aligned), 0.0, atol=1e-16)


def test_dilution_shrinks_the_asymmetry_without_flipping_it() -> None:
    """The observed asymmetry is the true one, pulled toward zero -- same sign."""
    aligned, reversed_ = _luminosity()
    observed = afb_diluted(MASSES, S2W, aligned, reversed_)
    truth = _undiluted(MASSES, aligned, reversed_)
    d_eff = dilution_factor(MASSES, S2W, aligned, reversed_)

    assert np.all(np.abs(observed) < np.abs(truth))
    assert np.all(np.sign(observed) == np.sign(truth))
    assert np.all((d_eff > 0.0) & (d_eff < 1.0))
    # The A2 sign gate must survive dilution: below the Z pole A_FB < 0, above > 0.
    assert observed[0] < 0.0 < observed[-1]


# --------------------------------------------------------------------------
# The gate: formula-level closure
# --------------------------------------------------------------------------
def test_unfolding_recovers_the_undiluted_curve() -> None:
    """**The A3 gate.** Unfold the diluted curve; get ``afb_hadronic`` back."""
    aligned, reversed_ = _luminosity()
    observed = afb_diluted(MASSES, S2W, aligned, reversed_)
    d_eff = dilution_factor(MASSES, S2W, aligned, reversed_)
    err = np.full_like(observed, 1e-3)

    unfolded, unfolded_err = unfold_afb(observed, err, d_eff)
    truth = _undiluted(MASSES, aligned, reversed_)

    assert np.allclose(unfolded, truth, rtol=0, atol=1e-14)
    # Unfolding inflates the error by the same factor it inflates the value:
    # dilution destroys information, it does not merely rescale it.
    assert np.allclose(unfolded_err, err / d_eff, rtol=0, atol=1e-18)
    assert np.all(unfolded_err > err)


def test_flavour_blind_unfolding_is_measurably_wrong() -> None:
    """The naive PDF-only factor misses -- which is what makes the gate a test.

    ``pdf_dilution`` weights the flavours by luminosity instead of by their
    contribution ``L D_q`` to the asymmetry. With up and down carrying different
    valence hardness *and* different ``A_FB``, that mis-weighting does not
    cancel. If this test ever passes trivially, the toy has become effectively
    single-flavour and :func:`test_unfolding_recovers_the_undiluted_curve` is
    no longer testing the method.
    """
    aligned, reversed_ = _luminosity()
    observed = afb_diluted(MASSES, S2W, aligned, reversed_)
    truth = _undiluted(MASSES, aligned, reversed_)
    err = np.full_like(observed, 1e-3)

    naive, _ = unfold_afb(observed, err, pdf_dilution(aligned, reversed_))
    correct, _ = unfold_afb(observed, err, dilution_factor(MASSES, S2W, aligned, reversed_))

    assert np.allclose(correct, truth, rtol=0, atol=1e-14)
    # The naive answer is wrong by far more than the correct one, and by more
    # than a plausible measurement error on A_FB.
    residual = np.abs(naive - truth)
    assert np.all(residual > 1e-3), f"naive unfolding residuals too small: {residual}"

    # The two quark species really do have different asymmetries and different
    # dilution -- the premises the assertion above rests on.
    assert not np.allclose(
        afb_parton(MASSES, UP_TYPE, S2W), afb_parton(MASSES, DOWN_TYPE, S2W), rtol=1e-2
    )
    per_flavour = [
        (aligned[f] - reversed_[f]) / (aligned[f] + reversed_[f]) for f in (UP_TYPE, DOWN_TYPE)
    ]
    assert not np.allclose(per_flavour[0], per_flavour[1], rtol=1e-2)


def test_single_flavour_collapses_to_the_pdf_ratio() -> None:
    """With one flavour the ``D_q`` cancel and the naive factor becomes exact."""
    aligned, reversed_ = _luminosity()
    one_a = {UP_TYPE: aligned[UP_TYPE]}
    one_r = {UP_TYPE: reversed_[UP_TYPE]}
    assert np.allclose(
        dilution_factor(MASSES, S2W, one_a, one_r),
        pdf_dilution(one_a, one_r),
        rtol=0,
        atol=1e-15,
    )


# --------------------------------------------------------------------------
# The honest caveats, asserted rather than only documented
# --------------------------------------------------------------------------
def test_dilution_factor_depends_on_sin2_theta_w_when_flavours_mix() -> None:
    """``D_eff`` is not a PDF-only quantity -- it carries ``sin^2(theta_W)``.

    This is the coupling back into the parameter A2 fits, so it is asserted to
    exist (multi-flavour) and to vanish (single flavour), and its size is
    quantified rather than waved at.
    """
    aligned, reversed_ = _luminosity()
    lo = dilution_factor(MASSES, 0.2250, aligned, reversed_)
    hi = dilution_factor(MASSES, 0.2380, aligned, reversed_)
    shift = np.abs(hi - lo)
    assert np.all(shift > 0.0), "D_eff must move with sin^2(theta_W) when flavours mix"
    # Weak, but not negligible next to a per-mille A_FB measurement: it belongs
    # in the systematic budget, or the fit should be iterated.
    assert np.max(shift) < 5e-2

    one_a, one_r = {UP_TYPE: aligned[UP_TYPE]}, {UP_TYPE: reversed_[UP_TYPE]}
    assert np.allclose(
        dilution_factor(MASSES, 0.2250, one_a, one_r),
        dilution_factor(MASSES, 0.2380, one_a, one_r),
        rtol=0,
        atol=1e-15,
    )


def test_central_rapidity_is_masked_not_divided_by() -> None:
    """``x1 -> x2`` kills the proxy; the answer must be ``nan``, not a big number."""
    aligned, reversed_ = _luminosity_grid(MASSES, np.array([1e-9]))
    aligned = {f: v[:, 0] for f, v in aligned.items()}
    reversed_ = {f: v[:, 0] for f, v in reversed_.items()}

    d_eff = dilution_factor(MASSES, S2W, aligned, reversed_)
    assert np.all(np.isnan(d_eff))

    observed = afb_diluted(MASSES, S2W, aligned, reversed_)
    assert np.allclose(observed, 0.0, atol=1e-8)  # nothing left to unfold

    unfolded, unfolded_err = unfold_afb(observed, np.full_like(observed, 1e-3), d_eff)
    assert np.all(np.isnan(unfolded)) and np.all(np.isnan(unfolded_err))


def test_unfold_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        unfold_afb(np.zeros(3), np.zeros(3), np.zeros(4))


def test_luminosity_dicts_must_cover_the_same_flavours() -> None:
    aligned, reversed_ = _luminosity()
    with pytest.raises(ValueError, match="same flavours"):
        afb_diluted(MASSES, S2W, aligned, {UP_TYPE: reversed_[UP_TYPE]})
    with pytest.raises(ValueError, match="at least one quark flavour"):
        afb_diluted(MASSES, S2W, {}, {})


# --------------------------------------------------------------------------
# Sampled closure: real four-vectors through the real sign(Q_z) proxy
# --------------------------------------------------------------------------
def _sample(rng: np.random.Generator, n_events: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw Drell-Yan events from the toy, returning ``(m, p_mu-, p_mu+)``.

    Sampling mirrors the physics, not the master formula: a (mass, rapidity,
    flavour, orientation) cell is drawn with weight ``L * S_q`` (the *rate*,
    which is dilution-blind), then ``cos(theta)`` about the **true quark axis**
    from ``S(1+c^2) + 2 D c``. Whether the proxy then gets the axis right is
    left entirely to :func:`collins_soper_costheta` to work out from the
    four-vectors.
    """
    # Sample both signs of rapidity, so the proxy has to pick a direction rather
    # than always being handed +z.
    y_nodes = np.concatenate([-RAPIDITY[::-1], RAPIDITY])
    aligned, reversed_ = _luminosity_grid(MASSES, np.abs(y_nodes))

    flavours = [UP_TYPE, DOWN_TYPE]
    s_q = np.stack([_s_and_d(MASSES, f, S2W)[0] for f in flavours])  # (n_f, n_m)
    d_q = np.stack([_s_and_d(MASSES, f, S2W)[1] for f in flavours])

    # weight[flavour, orientation, mass, rapidity]
    lum = np.stack(
        [
            np.stack([aligned[f] for f in flavours]),
            np.stack([reversed_[f] for f in flavours]),
        ],
        axis=1,
    )  # (n_f, 2, n_m, n_y)
    weight = lum * s_q[:, None, :, None]
    flat = (weight / weight.sum()).ravel()
    idx = rng.choice(flat.size, size=n_events, p=flat)
    i_f, i_o, i_m, i_y = np.unravel_index(idx, weight.shape)

    m = MASSES[i_m]
    y = y_nodes[i_y]
    s_ev, d_ev = s_q[i_f, i_m], d_q[i_f, i_m]

    # Rejection-sample cos(theta) about the true quark axis from S(1+c^2)+2Dc.
    cos_true = np.empty(n_events)
    todo = np.ones(n_events, dtype=bool)
    ceiling = 2.0 * s_ev + 2.0 * np.abs(d_ev)
    while todo.any():
        c = rng.uniform(-1.0, 1.0, size=todo.sum())
        p = s_ev[todo] * (1.0 + c**2) + 2.0 * d_ev[todo] * c
        keep = rng.uniform(0.0, 1.0, size=c.size) * ceiling[todo] < p
        where = np.flatnonzero(todo)[keep]
        cos_true[where] = c[keep]
        todo[where] = False

    # The true quark axis: along sign(y) for the aligned orientation (i_o == 0),
    # against it for the reversed one.
    axis = np.where(i_o == 0, 1.0, -1.0) * np.sign(y)

    # Build the pair in its rest frame (Q_T = 0), then boost along z by y.
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_events)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_true**2))
    e_star = m / 2.0
    pz_star = e_star * cos_true * axis
    px = e_star * sin_t * np.cos(phi)
    py = e_star * sin_t * np.sin(phi)

    gamma, beta_gamma = np.cosh(y), np.sinh(y)
    e_minus = gamma * e_star + beta_gamma * pz_star
    pz_minus = beta_gamma * e_star + gamma * pz_star
    e_plus = gamma * e_star - beta_gamma * pz_star
    pz_plus = beta_gamma * e_star - gamma * pz_star

    p_minus = np.stack([e_minus, px, py, pz_minus], axis=-1)
    p_plus = np.stack([e_plus, -px, -py, pz_plus], axis=-1)
    return m, p_minus, p_plus


def test_sampled_proxy_matches_the_orientation_bookkeeping() -> None:
    """The ``sign(Q_z)`` proxy flips ``cos(theta)`` on exactly the reversed events.

    This is the bridge between the four-vector world and the ``L^+ / L^-`` split
    the master formula is written in: it is *derived* here from
    :func:`collins_soper_costheta` rather than assumed.
    """
    rng = np.random.default_rng(20260720)
    m, p_minus, p_plus = _sample(rng, 20_000)
    cos_proxy = collins_soper_costheta(p_minus, p_plus)
    # Q_T = 0 here, so cos(theta) about +z in the pair rest frame is recoverable
    # analytically; the proxy must equal it times sign(Q_z).
    q = p_minus + p_plus
    y = np.arctanh(q[:, 3] / q[:, 0])
    pz_star = np.cosh(y) * p_minus[:, 3] - np.sinh(y) * p_minus[:, 0]
    expected = (pz_star / (m / 2.0)) * np.sign(q[:, 3])
    assert np.allclose(cos_proxy, expected, rtol=0, atol=1e-9)


def test_sampled_events_unfold_to_the_undiluted_curve() -> None:
    """End-to-end: sample -> ``sign(Q_z)`` proxy -> measure -> unfold -> truth.

    The measurement side uses only what an experiment has (four-vectors and the
    proxy); the prediction side is A2's ``afb_hadronic``. Agreement is asserted
    as a **pull**, so the unfolded errors are tested too -- a closure that only
    checked central values could be satisfied by an error that is wrong.
    """
    rng = np.random.default_rng(20260721)
    m, p_minus, p_plus = _sample(rng, 400_000)
    cos_proxy = collins_soper_costheta(p_minus, p_plus)

    aligned, reversed_ = _luminosity()
    truth = _undiluted(MASSES, aligned, reversed_)
    d_eff = dilution_factor(MASSES, S2W, aligned, reversed_)

    observed = np.empty(MASSES.size)
    error = np.empty(MASSES.size)
    for i, mass in enumerate(MASSES):
        observed[i], error[i] = forward_backward_asymmetry(cos_proxy[m == mass])

    # The measured (diluted) curve must agree with the analytic diluted model...
    predicted = afb_diluted(MASSES, S2W, aligned, reversed_)
    assert np.all(np.abs(observed - predicted) / error < 4.0)

    # ...and unfolding it must land on the undiluted curve.
    unfolded, unfolded_err = unfold_afb(observed, error, d_eff)
    pull = (unfolded - truth) / unfolded_err
    assert np.all(np.abs(pull) < 4.0), f"unfolded pulls {pull}"

    # The dilution is a real effect at this rapidity range, not a rounding
    # correction: the raw proxy measurement is nowhere near the truth.
    assert np.all(np.abs(observed - truth) / error > 10.0)
