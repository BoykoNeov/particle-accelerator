r"""Analytic gate — the Collins–Soper ``cos θ*`` frame transform.

The Drell-Yan pipeline's ``A_FB`` deliverable rests on one piece of physics code:
:func:`accsim.events.collins_soper_costheta`, the closed lab-frame form of the
polar angle in the Collins–Soper (CS) frame. There is **no** clean closed-form
gate for the *magnitude* of ``A_FB`` (it folds in γ*/Z interference and the ``pp``
dilution), so the magnitude is validated at the pipeline level by the physics
sign guard (``A_FB > 0`` above ``M_Z``, ``< 0`` below). What *is* gated here — and
must be, because a flipped sign or a wrong ``2/(Q√(Q²+Q_T²))`` coefficient would
still produce a convincing plot — is the frame transform itself:

  1. **Closed form == explicit construction** over many random ``μ⁻/μ⁺`` pairs:
     the shipped closed form equals an *independent* boost-into-rest-frame
     bisector construction of the CS axis. Testing over random pairs (not one
     config) pins the coefficient generally — "derive, don't trust the constant".
  2. **Orientation** — hand configs with a known answer pin the overall sign the
     two derivations could *share* an error on: ``μ⁻`` along the quark (boost)
     direction ⇒ ``cos θ* = +1`` (forward), against it ⇒ ``−1``.
  3. **The ``pp`` proxy vs. undiluted orientation** — the default ``sign(Q_z)``
     proxy equals the raw ``+ẑ`` form times ``sign(Q_z)``; passing the true quark
     direction overrides it (the dilution knob).
  4. **``A_FB`` counting** helper reduces to ``(N_F − N_B)/N``.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim.events import collins_soper_costheta, forward_backward_asymmetry

# The Collins–Soper closed form is the standard **massless-lepton** definition: it
# returns the geometric momentum-direction cos θ* only in the m_ℓ → 0 limit (for a
# massive lepton exactly along the axis it gives p/E = β_ℓ, not 1). At the real
# muon mass vs the ~45 GeV Z-decay momentum this is a ~1e-6 effect — negligible, and
# it is the definition every DY experiment uses. So the frame math is pinned here in
# the massless limit, where closed form and geometric construction agree exactly.


# ---------------------------------------------------------------------------
# An independent CS-angle construction: boost into the di-lepton rest frame,
# build the CS axis as the bisector of beam-1 and reversed-beam-2, project.
# This shares NO code with the closed form — that is the point.
# ---------------------------------------------------------------------------
def _boost_to_rest(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Lorentz-boost four-vector ``p`` into the rest frame of ``q`` (both ``(E,px,py,pz)``)."""
    e_q = q[0]
    q_vec = q[1:]
    m = np.sqrt(e_q**2 - q_vec @ q_vec)
    beta = q_vec / e_q
    b2 = beta @ beta
    if b2 == 0.0:
        return p.copy()
    gamma = e_q / m
    e_p = p[0]
    p_vec = p[1:]
    bp = beta @ p_vec
    e_new = gamma * (e_p - bp)
    p_new = p_vec + ((gamma - 1.0) * bp / b2 - gamma * e_p) * beta
    return np.array([e_new, *p_new])


def _costheta_by_construction(p_minus: np.ndarray, p_plus: np.ndarray) -> float:
    """CS ``cos θ*`` via explicit boost + bisector, beam A (``+ẑ``) as the quark side."""
    # Light-like beams along ±z in the lab (only their directions enter the CS axis;
    # the closed form is derived in this massless-beam limit, so the result is
    # independent of the chosen beam energy).
    beam_a = np.array([1.0, 0.0, 0.0, 1.0])
    beam_b = np.array([1.0, 0.0, 0.0, -1.0])
    q = p_minus + p_plus

    k1 = _boost_to_rest(beam_a, q)[1:]
    k2 = _boost_to_rest(beam_b, q)[1:]
    k1 /= np.linalg.norm(k1)
    k2 /= np.linalg.norm(k2)
    z_cs = k1 - k2  # bisector of beam-1 and reversed-beam-2
    z_cs /= np.linalg.norm(z_cs)

    pm = _boost_to_rest(p_minus, q)[1:]
    pm /= np.linalg.norm(pm)
    return float(pm @ z_cs)


def _muon(px: float, py: float, pz: float) -> np.ndarray:
    """Close a **massless** lepton four-vector ``(|p|, px, py, pz)`` from its 3-momentum.

    Massless because the CS closed form is the massless-lepton definition; see the
    module note. This is the limit in which it equals the geometric construction.
    """
    e = np.sqrt(px**2 + py**2 + pz**2)
    return np.array([e, px, py, pz])


# ---------------------------------------------------------------------------
# Gate 1 — closed form equals the independent construction over random pairs.
# ---------------------------------------------------------------------------
def test_closed_form_matches_explicit_boost_construction() -> None:
    rng = np.random.default_rng(20260710)
    for _ in range(3000):
        pm = _muon(*rng.uniform(-60.0, 60.0, size=3))
        pp = _muon(*rng.uniform(-60.0, 60.0, size=3))
        # quark_direction=+1 selects the raw +ẑ-referenced form (beam A side),
        # matching the construction's beam-A-as-quark orientation.
        closed = collins_soper_costheta(pm, pp, quark_direction=1.0)
        built = _costheta_by_construction(pm, pp)
        assert abs(closed - built) < 1e-10
        assert abs(closed) <= 1.0 + 1e-12  # a genuine cosine


def test_closed_form_batched_matches_loop() -> None:
    rng = np.random.default_rng(11)
    pm = np.array([_muon(*rng.uniform(-40, 40, 3)) for _ in range(50)])
    pp = np.array([_muon(*rng.uniform(-40, 40, 3)) for _ in range(50)])
    batched = collins_soper_costheta(pm, pp, quark_direction=1.0)
    for i in range(50):
        assert abs(batched[i] - collins_soper_costheta(pm[i], pp[i], quark_direction=1.0)) < 1e-12


# ---------------------------------------------------------------------------
# Gate 2 — orientation (the sign the two derivations could share an error on).
# ---------------------------------------------------------------------------
def test_muon_along_quark_direction_is_forward() -> None:
    # In the rest frame, μ⁻ along +z, μ⁺ along −z (massless ⇒ E=p); boost the pair
    # along +z so the quark proxy sign(Q_z) = +1. μ⁻ then travels along the quark
    # direction ⇒ +1.
    p = 45.0  # rest-frame (massless) muon momentum/energy along z
    e = p
    # Boost along +z with a modest γ (β=0.6). Boosting (E,±p ẑ) → still along z.
    beta = 0.6
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    minus = np.array([gamma * (e + beta * p), 0.0, 0.0, gamma * (p + beta * e)])
    plus = np.array([gamma * (e - beta * p), 0.0, 0.0, gamma * (-p + beta * e)])
    assert (minus + plus)[3] > 0.0  # Q_z > 0 ⇒ proxy quark direction is +z
    cs = collins_soper_costheta(minus, plus)  # default proxy
    assert cs == pytest.approx(1.0, abs=1e-12)


def test_muon_against_quark_direction_is_backward() -> None:
    # Same but μ⁻ starts along −z in the rest frame ⇒ against the +z boost ⇒ −1.
    p = 45.0
    e = p  # massless
    beta = 0.6
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    minus = np.array([gamma * (e - beta * p), 0.0, 0.0, gamma * (-p + beta * e)])
    plus = np.array([gamma * (e + beta * p), 0.0, 0.0, gamma * (p + beta * e)])
    assert (minus + plus)[3] > 0.0
    cs = collins_soper_costheta(minus, plus)
    assert cs == pytest.approx(-1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Gate 3 — the pp proxy vs the undiluted (true-quark-direction) orientation.
# ---------------------------------------------------------------------------
def test_proxy_is_raw_times_sign_qz() -> None:
    rng = np.random.default_rng(7)
    for _ in range(500):
        pm = _muon(*rng.uniform(-60, 60, 3))
        pp = _muon(*rng.uniform(-60, 60, 3))
        qz = (pm + pp)[3]
        raw = collins_soper_costheta(pm, pp, quark_direction=1.0)  # +ẑ-referenced
        proxy = collins_soper_costheta(pm, pp)  # default sign(Q_z)
        assert proxy == pytest.approx(raw * np.sign(qz), abs=1e-12)


def test_true_quark_direction_flips_when_opposite_to_boost() -> None:
    # When the true quark went along −z but the di-lepton boost is +z, the proxy
    # and the true-direction orientation disagree by a sign — the dilution in the
    # extreme. Passing quark_direction=-1 gives the undiluted (true) value.
    pm = _muon(10.0, 3.0, 25.0)
    pp = _muon(-8.0, -2.0, 40.0)
    assert (pm + pp)[3] > 0.0  # boost is +z ⇒ proxy assumes quark +z
    proxy = collins_soper_costheta(pm, pp)  # sign(Q_z) = +1
    true_dir = collins_soper_costheta(pm, pp, quark_direction=-1.0)
    assert true_dir == pytest.approx(-proxy, abs=1e-12)


# ---------------------------------------------------------------------------
# Gate 4 — the A_FB counting helper.
# ---------------------------------------------------------------------------
def test_afb_counting() -> None:
    cos = np.array([0.5, 0.9, -0.3, 0.1, -0.8])  # 3 forward, 2 backward
    afb, err = forward_backward_asymmetry(cos)
    assert afb == pytest.approx((3 - 2) / 5)
    assert err == pytest.approx(np.sqrt((1 - afb**2) / 5))


def test_afb_drops_nonfinite_and_zeros() -> None:
    cos = np.array([0.5, np.nan, 0.0, -0.5])  # nan + exact-zero dropped ⇒ 1 vs 1
    afb, _ = forward_backward_asymmetry(cos)
    assert afb == pytest.approx(0.0)


def test_afb_empty_is_nan() -> None:
    afb, err = forward_backward_asymmetry(np.array([np.nan, 0.0]))
    assert np.isnan(afb) and np.isnan(err)
