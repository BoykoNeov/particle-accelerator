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

import numpy as np
import pytest

from accsim import quantum_lifetime, quantum_lifetime_exact


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


# --- 3. the exact MFPT integral as a shipped function (B4) ---------------------
#
# B4 gates a tracked lifetime against a closed form at xi ~ 4, where the shipped
# asymptote is not the answer. These pin the exact integral three ways: against
# quadrature, against Ei where that form is well conditioned, and against the two
# hard numbers at xi = 4 that make the departure a measured law rather than a
# tolerance.
def test_the_exact_lifetime_is_the_quadrature_of_its_own_integrand() -> None:
    """Series ``Σ ξⁿ/(n·n!)`` == ``∫₀^ξ (e^w−1)/w dw``, over five decades of ξ."""
    tau_d = 5.0e-3
    sigma = 1.0e-3
    for xi in (1.0e-3, 0.1, 1.0, 4.0, 20.0, 100.0):
        aperture = sigma * math.sqrt(2.0 * xi)
        got = quantum_lifetime_exact(aperture, sigma, tau_d)
        assert got == pytest.approx(_exact_mfpt(xi, tau_d), rel=1e-12)


def test_the_series_and_the_exponential_integral_are_the_same_function() -> None:
    """``∫₀^ξ (e^w−1)/w dw = Ei(ξ) − γ − ln ξ`` — an independent implementation.

    Checked only for ``ξ ≥ 1``: below that the ``Ei`` form is a difference of large
    near-equal terms and loses digits, which is *why* the series is what ships.
    """
    special = pytest.importorskip("scipy.special")
    tau_d, sigma = 5.0e-3, 1.0e-3
    for xi in (1.0, 4.0, 20.0, 100.0):
        aperture = sigma * math.sqrt(2.0 * xi)
        ei_form = 0.5 * tau_d * (special.expi(xi) - float(np.euler_gamma) - math.log(xi))
        assert quantum_lifetime_exact(aperture, sigma, tau_d) == pytest.approx(ei_form, rel=1e-12)


def test_at_the_working_point_the_asymptote_is_wrong_by_a_named_factor() -> None:
    """At ξ = 4 the exact integral is 17.6674 and the asymptote 13.6495: ratio 1.29436.

    The number B4's ring is designed around. It is asserted to five figures because
    it is a *pure* number — no lattice, no units, no tolerance to loosen.
    """
    tau_d, sigma = 1.0, 1.0
    aperture = math.sqrt(8.0)  # xi = 4
    exact = quantum_lifetime_exact(aperture, sigma, tau_d)
    asymptote = quantum_lifetime(aperture, sigma, tau_d)
    assert 2.0 * exact == pytest.approx(17.667364, rel=1e-6)
    assert 2.0 * asymptote == pytest.approx(13.649538, rel=1e-6)
    assert exact / asymptote == pytest.approx(1.2943563, rel=1e-6)


def test_the_asymptotic_series_brackets_the_truth_instead_of_reaching_it() -> None:
    """``1 + 1/ξ`` and ``1 + 1/ξ + 2/ξ²`` straddle 1.29436 at ξ = 4.

    Pre-committed because it is the trap: the obvious way to gate the departure is
    against a truncated ``O(1/ξ)`` expression, and at this ξ *that test would itself
    be wrong* — one term undershoots by 3.5%, two overshoot by 6.2%. The exact
    integral is the gate; the expansion is not an approximation to it here.
    """
    xi = 4.0
    ratio = quantum_lifetime_exact(math.sqrt(2.0 * xi), 1.0, 1.0) / quantum_lifetime(
        math.sqrt(2.0 * xi), 1.0, 1.0
    )
    one_term, two_term = 1.0 + 1.0 / xi, 1.0 + 1.0 / xi + 2.0 / xi**2
    assert one_term < ratio < two_term
    assert ratio / one_term == pytest.approx(1.0355, rel=1e-3)
    assert ratio / two_term == pytest.approx(0.9414, rel=1e-3)


def test_the_asymptote_becomes_the_exact_integral_where_a_real_ring_lives() -> None:
    """The departure is ``1/ξ`` exactly: ``ξ·(exact/asymptote − 1) → 1``.

    Stated as the law rather than as "the departure halves when ξ doubles", which
    is the same claim only in the limit: at ξ = 8 the next order is still worth 40%
    and successive departures are in the ratio 2.42, not 2. What converges cleanly
    is ``ξ·departure``, and its *excess over 1* is what halves.
    """
    scaled = {}
    for k in range(3, 10):  # xi = 8 .. 512
        xi = 2.0**k
        aperture = math.sqrt(2.0 * xi)
        ratio = quantum_lifetime_exact(aperture, 1.0, 1.0) / quantum_lifetime(aperture, 1.0, 1.0)
        scaled[xi] = xi * (ratio - 1.0)

    assert all(scaled[2.0**k] > scaled[2.0 ** (k + 1)] > 1.0 for k in range(3, 9))  # ↓ to 1
    assert scaled[512.0] == pytest.approx(1.0, abs=5.0e-3)
    # and the excess over 1 halves, converging on 2 rather than merely near it
    excess = {xi: value - 1.0 for xi, value in scaled.items()}
    assert excess[16.0] / excess[32.0] == pytest.approx(2.28, rel=0.02)
    assert excess[128.0] / excess[256.0] == pytest.approx(2.02, rel=0.02)
    # a normal ring's xi is tens: there the shipped asymptote is already fine
    assert quantum_lifetime_exact(math.sqrt(2.0 * 64.0), 1.0, 1.0) / quantum_lifetime(
        math.sqrt(2.0 * 64.0), 1.0, 1.0
    ) == pytest.approx(1.0, abs=0.02)


def test_the_exact_form_is_stable_where_the_leading_one_diverges() -> None:
    """As ξ → 0 the lifetime → 0 (nothing survives); the asymptote → ∞ instead.

    ``e^ξ/(2ξ)`` blows up at small ξ, which is nonsense — an aperture inside the
    beam loses everything immediately. The exact integral goes to zero linearly.
    """
    tau_d = 1.0
    for xi in (1.0e-4, 1.0e-3, 1.0e-2):
        aperture = math.sqrt(2.0 * xi)
        exact = quantum_lifetime_exact(aperture, 1.0, tau_d)
        assert exact == pytest.approx(0.5 * tau_d * xi, rel=0.02)  # ∫ → ξ as ξ → 0
        assert quantum_lifetime(aperture, 1.0, tau_d) > 100.0 * exact  # asymptote: nonsense


def test_the_exact_form_scales_with_the_damping_time_and_guards_its_inputs() -> None:
    aperture, sigma = 3.0e-3, 1.0e-3
    a = quantum_lifetime_exact(aperture, sigma, 1.0e-3)
    b = quantum_lifetime_exact(aperture, sigma, 7.0e-3)
    assert b / a == pytest.approx(7.0, rel=1e-12)
    for bad in ((0.0, sigma, 1.0), (aperture, 0.0, 1.0), (aperture, sigma, 0.0)):
        with pytest.raises(ValueError):
            quantum_lifetime_exact(*bad)
    with pytest.raises(ValueError, match="did not converge"):
        quantum_lifetime_exact(aperture, sigma, 1.0, terms=2)
