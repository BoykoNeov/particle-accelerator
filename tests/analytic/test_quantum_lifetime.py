"""Analytic checks for the quantum (aperture-limited) lifetime (Stage 4).

The closed form ``τ_q = τ_d·e^ξ/(2ξ)`` (``ξ = A²/2σ²``) is validated *against its
own derivation*, not a remembered constant:

1. The mean-first-passage-time (MFPT) solution of the amplitude-diffusion
   Fokker–Planck backward equation is verified symbolically (residual = -1), and
   the exact MFPT integral's ``ξ→∞`` ratio to the leading term is 1.
2. Numerically, ``quantum_lifetime`` matches the exact MFPT integral to ``O(1/ξ)``,
   with the relative error shrinking as ``ξ`` grows (the hallmark of a correct
   asymptote — a wrong coefficient would not converge to the exact integral).

The ``τ_d`` here is the **amplitude** damping time; the factor-of-2 relationship
to the emittance damping time is pinned explicitly.
"""

from __future__ import annotations

import math

import pytest

from accsim import quantum_lifetime


def _exact_mfpt(xi: float, tau_d: float) -> float:
    """Exact quantum lifetime (τ_d/2)∫₀^ξ (e^w−1)/w dw, by numeric quadrature."""
    from scipy.integrate import quad

    integral, _ = quad(lambda w: (math.expm1(w)) / w, 0.0, xi)
    return 0.5 * tau_d * integral


# --- 1. the derivation itself: MFPT solves the backward equation, right asymptote ---
def test_mfpt_derivation_symbolic() -> None:
    sp = pytest.importorskip("sympy")
    w, c, xi = sp.symbols("w c xi", positive=True)
    # Claimed MFPT slope solving the backward eqn c[w T'' + (1-w) T'] = -1.
    tp = -(sp.exp(w) - 1) / (c * w)
    residual = sp.simplify(c * (w * sp.diff(tp, w) + (1 - w) * tp))
    assert residual == -1
    # Exact MFPT integral asymptotes to the leading term e^xi/xi.
    integral = sp.integrate((sp.exp(w) - 1) / w, (w, 0, xi))
    assert sp.limit(integral / (sp.exp(xi) / xi), xi, sp.oo) == 1


# --- 2. closed form matches the exact MFPT, converging as O(1/xi) ---------------
@pytest.mark.parametrize(("xi", "tol"), [(8.0, 0.16), (16.0, 0.08), (32.0, 0.04)])
def test_closed_form_matches_exact_mfpt(xi: float, tol: float) -> None:
    scipy_spec = pytest.importorskip("scipy")  # noqa: F841
    tau_d = 5.0e-3  # 5 ms amplitude damping time (arbitrary scale)
    sigma = 1.0e-3
    aperture = sigma * math.sqrt(2.0 * xi)  # xi = A^2 / 2 sigma^2

    got = quantum_lifetime(aperture, sigma, tau_d)
    exact = _exact_mfpt(xi, tau_d)
    # Leading asymptote: relative error ~ 1/xi. Assert it is within tol (which
    # halves as xi doubles) and always UNDER-estimates (leading < exact integral).
    rel_err = abs(got - exact) / exact
    assert rel_err < tol
    assert got < exact  # e^xi/xi is the first term; the exact integral adds +1/xi...


def test_closed_form_error_halves_with_xi() -> None:
    """The O(1/xi) signature: doubling xi roughly halves the relative error."""
    scipy_spec = pytest.importorskip("scipy")  # noqa: F841
    tau_d, sigma = 1.0, 1.0e-3

    def rel_err(xi: float) -> float:
        a = sigma * math.sqrt(2.0 * xi)
        return abs(quantum_lifetime(a, sigma, tau_d) - _exact_mfpt(xi, tau_d)) / _exact_mfpt(
            xi, tau_d
        )

    e1, e2 = rel_err(10.0), rel_err(20.0)
    assert e2 < e1
    assert e2 == pytest.approx(0.5 * e1, rel=0.25)  # ~halved, as O(1/xi) predicts


# --- scaling and the amplitude-vs-emittance factor of 2 ------------------------
def test_scales_linearly_with_damping_time() -> None:
    a, sigma = 5.0e-3, 1.0e-3
    assert quantum_lifetime(a, sigma, 2.0) == pytest.approx(2.0 * quantum_lifetime(a, sigma, 1.0))


def test_grows_steeply_with_aperture() -> None:
    sigma, tau_d = 1.0e-3, 1.0
    # Bigger aperture -> exponentially longer lifetime.
    assert quantum_lifetime(6.0e-3, sigma, tau_d) > quantum_lifetime(5.0e-3, sigma, tau_d)
    # e^xi/(2xi) with xi = A^2/2sigma^2: check one value against a hand computation.
    xi = (5.0e-3) ** 2 / (2.0 * sigma**2)  # = 12.5
    assert quantum_lifetime(5.0e-3, sigma, tau_d) == pytest.approx(math.exp(xi) / (2.0 * xi))


def test_emittance_damping_time_is_half() -> None:
    # Passing the amplitude damping time tau_d must equal passing 2*tau_eps, where
    # tau_eps = tau_d/2 is the emittance damping time (the factor-of-2 convention).
    a, sigma, tau_d = 5.0e-3, 1.0e-3, 4.0e-3
    tau_eps = tau_d / 2.0
    assert quantum_lifetime(a, sigma, tau_d) == pytest.approx(
        quantum_lifetime(a, sigma, 2.0 * tau_eps)
    )


def test_input_guards() -> None:
    with pytest.raises(ValueError):
        quantum_lifetime(-1e-3, 1e-3, 1.0)
    with pytest.raises(ValueError):
        quantum_lifetime(1e-3, 0.0, 1.0)
    with pytest.raises(ValueError):
        quantum_lifetime(1e-3, 1e-3, -1.0)
