r"""Analytic gate — the Drell-Yan angular coefficients ``A₀…A₇`` (machinery).

This file pins the *machinery* of the A1 milestone (DY angular coefficients + the
Lam–Tung relation): the Collins–Soper azimuth ``φ*`` and the moment-projection
extraction of ``A₀…A₇``. The **physics** gate — the Lam–Tung relation ``A₀ = A₂`` —
is a *dynamical* statement (the DY analog of Callan–Gross) and cannot be checked by
the extraction machinery alone; it lives in :mod:`test_lam_tung` (closed-form
O(α_s) derivation) and, empirically, in the Pythia pipeline. A round-trip that
injects ``A₀ = A₂`` and recovers it would be **circular** as a Lam–Tung test, so it
is labelled here as machinery validation only.

What is gated here, and must be, because a wrong moment weight or a flipped frame
sign would still produce a convincing plot:

  1. **φ\* construction == closed-form cos θ\***: the new construction-based
     :func:`collins_soper_angles` reproduces the independent massless closed form
     :func:`collins_soper_costheta` (cross-validating two separate implementations),
     and ``φ*`` lands in ``(−π, π]``.
  2. **Moment closure (sympy)**: each projection weight ``P_i`` obeys
     ``⟨P_i⟩_f = A_i`` over the full solid angle — derived, not memorised.
  3. **Round-trip recovery**: events sampled (accept-reject) from a distribution
     with *chosen* ``A_i`` recover those ``A_i`` within Monte-Carlo error.
  4. **Quark-flip parity**: swapping the quark direction sends
     ``cos θ* → −cos θ*``, ``φ* → −φ*``, hence even coefficients ``A₀,A₂,A₃,A₆`` are
     invariant and odd ones ``A₁,A₄,A₅,A₇`` flip sign — so ``A₀,A₂`` (Lam–Tung) are
     **immune to the pp dilution**.
  5. **``A_FB = (3/8) A₄``**: the counting asymmetry equals the ``A₄`` moment.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from accsim.events import (
    angular_coefficients,
    collins_soper_angles,
    collins_soper_costheta,
    forward_backward_asymmetry,
)


# ---------------------------------------------------------------------------
# Helpers: massless lepton four-vectors and the CS angular distribution.
# ---------------------------------------------------------------------------
def _muon(px: float, py: float, pz: float) -> np.ndarray:
    """Massless lepton ``(|p|, px, py, pz)`` — the limit the CS closed form assumes."""
    e = np.sqrt(px**2 + py**2 + pz**2)
    return np.array([e, px, py, pz])


def _f_of_angles(ct: np.ndarray, ph: np.ndarray, a: dict[str, float]) -> np.ndarray:
    """The CS angular distribution ``dσ/dΩ`` (unnormalised) at coefficients ``a``."""
    st = np.sqrt(np.maximum(1.0 - ct**2, 0.0))
    s2 = 2.0 * st * ct
    return (
        (1.0 + ct**2)
        + a["A0"] * 0.5 * (1.0 - 3.0 * ct**2)
        + a["A1"] * s2 * np.cos(ph)
        + a["A2"] * 0.5 * st**2 * np.cos(2.0 * ph)
        + a["A3"] * st * np.cos(ph)
        + a["A4"] * ct
        + a["A5"] * st**2 * np.sin(2.0 * ph)
        + a["A6"] * s2 * np.sin(ph)
        + a["A7"] * st * np.sin(ph)
    )


def _sample_cs_distribution(
    a: dict[str, float], n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Accept-reject sample ``(cos θ*, φ*)`` from ``dσ/dΩ`` with coefficients ``a``.

    Proposal is *uniform on the sphere* (cos θ ~ U[-1,1], φ ~ U[0,2π]); accept with
    probability ``f/M``. ``M`` is a safe envelope over the sphere. Returns the
    accepted ``(cos θ, φ)`` — distributed as ``dσ/dΩ``, the input the moment
    extraction assumes.
    """
    # Envelope: bound each term by its max |amplitude| × |A_i| on top of the ≤2 core.
    m = 2.0 + sum(abs(v) for v in a.values()) + 1.0
    out_c: list[float] = []
    out_p: list[float] = []
    while len(out_c) < n:
        block = max(n, 4096)
        ct = rng.uniform(-1.0, 1.0, block)
        ph = rng.uniform(-np.pi, np.pi, block)
        keep = rng.uniform(0.0, m, block) < _f_of_angles(ct, ph, a)
        out_c.extend(ct[keep].tolist())
        out_p.extend(ph[keep].tolist())
    return np.asarray(out_c[:n]), np.asarray(out_p[:n])


# ---------------------------------------------------------------------------
# Gate 1 — the construction-based angles reproduce the closed-form cos θ*.
# ---------------------------------------------------------------------------
def test_construction_matches_closed_form_costheta() -> None:
    rng = np.random.default_rng(20260710)
    max_diff = 0.0
    for _ in range(3000):
        pm = _muon(*rng.uniform(-60.0, 60.0, 3))
        pp = _muon(*rng.uniform(-60.0, 60.0, 3))
        c_con, phi = collins_soper_angles(pm, pp, quark_direction=1.0)
        c_cf = collins_soper_costheta(pm, pp, quark_direction=1.0)
        max_diff = max(max_diff, abs(c_con - c_cf))
        assert abs(c_con) <= 1.0 + 1e-12
        assert -np.pi - 1e-12 <= phi <= np.pi + 1e-12
    # Two independent routes (boost construction vs light-cone closed form) agree.
    assert max_diff < 1e-10


def test_angles_batched_matches_loop() -> None:
    rng = np.random.default_rng(3)
    pm = np.array([_muon(*rng.uniform(-40, 40, 3)) for _ in range(64)])
    pp = np.array([_muon(*rng.uniform(-40, 40, 3)) for _ in range(64)])
    cb, phib = collins_soper_angles(pm, pp, quark_direction=1.0)
    for i in range(64):
        c1, p1 = collins_soper_angles(pm[i], pp[i], quark_direction=1.0)
        assert abs(cb[i] - c1) < 1e-12
        assert abs(phib[i] - p1) < 1e-12


# ---------------------------------------------------------------------------
# Gate 2 — moment closure: each weight P_i projects out exactly A_i (sympy).
# This is the "derive the coefficient, don't trust the constant" gate: the same
# eight prefactors the code uses are proven here by exact symbolic integration.
# ---------------------------------------------------------------------------
def test_moment_weights_close_symbolically() -> None:
    th, ph = sp.symbols("theta phi", real=True)
    a = sp.symbols("A0:8", real=True)
    c, s = sp.cos(th), sp.sin(th)
    f = (
        (1 + c**2)
        + a[0] * sp.Rational(1, 2) * (1 - 3 * c**2)
        + a[1] * sp.sin(2 * th) * sp.cos(ph)
        + a[2] * sp.Rational(1, 2) * s**2 * sp.cos(2 * ph)
        + a[3] * s * sp.cos(ph)
        + a[4] * c
        + a[5] * s**2 * sp.sin(2 * ph)
        + a[6] * sp.sin(2 * th) * sp.sin(ph)
        + a[7] * s * sp.sin(ph)
    )
    # The eight projection polynomials — identical prefactors to _moment_weights.
    weights = [
        4 - 10 * c**2,
        5 * sp.sin(2 * th) * sp.cos(ph),
        10 * s**2 * sp.cos(2 * ph),
        4 * s * sp.cos(ph),
        4 * c,
        5 * s**2 * sp.sin(2 * ph),
        5 * sp.sin(2 * th) * sp.sin(ph),
        4 * s * sp.sin(ph),
    ]

    def solid_angle(expr: sp.Expr) -> sp.Expr:
        return sp.integrate(sp.integrate(expr * s, (ph, 0, 2 * sp.pi)), (th, 0, sp.pi))

    norm = solid_angle(f)
    assert sp.simplify(norm - sp.Rational(16, 3) * sp.pi) == 0
    for i, w in enumerate(weights):
        proj = sp.simplify(solid_angle(w * f) / norm)
        assert sp.simplify(proj - a[i]) == 0, f"weight P{i} does not project A{i}"


# ---------------------------------------------------------------------------
# Gate 3 — round-trip recovery (MACHINERY validation, NOT a Lam–Tung test).
# Chosen A_i are recovered from an accept-reject sample within Monte-Carlo error.
# ---------------------------------------------------------------------------
def test_round_trip_recovers_injected_coefficients() -> None:
    rng = np.random.default_rng(1234)
    # Deliberately all-distinct, modest values (so the coefficients are individually
    # identifiable and f stays positive). NOT A0==A2 — that would smell circular.
    injected = {
        "A0": 0.30,
        "A1": 0.12,
        "A2": 0.18,
        "A3": 0.08,
        "A4": 0.25,
        "A5": 0.06,
        "A6": 0.05,
        "A7": 0.04,
    }
    n = 400_000
    ct, ph = _sample_cs_distribution(injected, n, rng)
    got = angular_coefficients(ct, ph)
    # MC error on a mean-of-weight is std(P_i)/√N; the weights are O(1–10), so the
    # per-coefficient error is a few ×1e-3. Require recovery to 0.02 (comfortably
    # >4σ margin) — a loose-but-honest bound, not a tuned tolerance.
    for key, want in injected.items():
        assert abs(got[key] - want) < 0.02, f"{key}: got {got[key]:.4f}, want {want}"


def test_round_trip_isotropic_gives_zero_coefficients() -> None:
    # A pure 1+cos²θ distribution (all A_i = 0) must return all-zero coefficients.
    rng = np.random.default_rng(77)
    zero = {f"A{i}": 0.0 for i in range(8)}
    ct, ph = _sample_cs_distribution(zero, 300_000, rng)
    got = angular_coefficients(ct, ph)
    for key, val in got.items():
        assert abs(val) < 0.02, f"{key} = {val:.4f} should be ~0"


# ---------------------------------------------------------------------------
# Gate 4 — quark-flip parity: even {A0,A2,A3,A6} invariant, odd {A1,A4,A5,A7} flip.
# This is why A0,A2 (Lam–Tung) are immune to the pp dilution.
# ---------------------------------------------------------------------------
def test_quark_flip_parity_of_angles() -> None:
    rng = np.random.default_rng(555)
    pm = np.array([_muon(*rng.uniform(-60, 60, 3)) for _ in range(2000)])
    pp = np.array([_muon(*rng.uniform(-60, 60, 3)) for _ in range(2000)])
    c_p, phi_p = collins_soper_angles(pm, pp, quark_direction=1.0)
    c_m, phi_m = collins_soper_angles(pm, pp, quark_direction=-1.0)
    assert np.allclose(c_m, -c_p, atol=1e-12)
    # φ* → −φ*, compared modulo 2π (handles the ±π branch of atan2).
    wrap = np.angle(np.exp(1j * (phi_m + phi_p)))
    assert np.allclose(wrap, 0.0, atol=1e-10)


def test_quark_flip_parity_of_coefficients() -> None:
    rng = np.random.default_rng(999)
    injected = {
        "A0": 0.30,
        "A1": 0.15,
        "A2": 0.20,
        "A3": 0.10,
        "A4": 0.28,
        "A5": 0.07,
        "A6": 0.06,
        "A7": 0.05,
    }
    ct, ph = _sample_cs_distribution(injected, 300_000, rng)
    a_plus = angular_coefficients(ct, ph)
    # Flip orientation: cos θ* → −cos θ*, φ* → −φ*.
    a_minus = angular_coefficients(-ct, -ph)
    even, odd = {"A0", "A2", "A3", "A6"}, {"A1", "A4", "A5", "A7"}
    for key in even:
        assert abs(a_minus[key] - a_plus[key]) < 1e-9, f"{key} should be quark-flip even"
    for key in odd:
        assert abs(a_minus[key] + a_plus[key]) < 1e-9, f"{key} should be quark-flip odd"


# ---------------------------------------------------------------------------
# Gate 5 — the forward-backward asymmetry is (3/8) A₄.
# ---------------------------------------------------------------------------
def test_afb_equals_three_eighths_a4() -> None:
    # For a pure A₄ (+ isotropic core) distribution, the counting A_FB = (3/8)A₄.
    rng = np.random.default_rng(2024)
    a = {f"A{i}": 0.0 for i in range(8)}
    a["A4"] = 0.4
    ct, ph = _sample_cs_distribution(a, 500_000, rng)
    coeffs = angular_coefficients(ct, ph)
    afb, err = forward_backward_asymmetry(ct)
    assert abs(coeffs["A4"] - 0.4) < 0.02
    # A_FB = (3/8) A₄ ; both are measured on the same sample, agree within ~error.
    assert abs(afb - (3.0 / 8.0) * coeffs["A4"]) < 5.0 * max(err, 1e-3)
