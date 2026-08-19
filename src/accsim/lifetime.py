r"""Simple beam-lifetime models (Stage 4).

Currently the **quantum (aperture-limited) lifetime**: in a radiation-damped ring,
quantum excitation continually repopulates the Gaussian tail of the betatron
distribution; particles that diffuse past a physical aperture ``A`` are lost, and
the balance of diffusion against damping sets a finite lifetime.

**Derivation (not a remembered constant).** With the normalized action
``w = a²/2σ²`` (betatron amplitude ``a``, rms beam size ``σ``), the equilibrium is
``f_eq(w) = e^{-w}`` and the aperture sits at ``w = ξ = A²/2σ²``. Radiation damps
the amplitude as ``a ∝ e^{-t/τ_d}`` (so ``d(a²)/dt|_damp = -2a²/τ_d``), which fixes
the amplitude-diffusion Fokker–Planck generator; its mean-first-passage time from
the core to the aperture is, exactly,

    τ_q(exact) = (τ_d/2) ∫₀^ξ (e^w − 1)/w dw,

whose ``ξ ≫ 1`` asymptote (the standard quantum lifetime) is

    τ_q = τ_d · e^ξ / (2ξ),      ξ = A²/2σ².

Both the MFPT solution and the asymptote are verified symbolically in
``tests/analytic/test_quantum_lifetime.py`` (the backward-equation residual is
exactly ``-1``; the exact/leading ratio → 1). See ``docs/CONVENTIONS.md`` →
*Quantum lifetime*.

**Damping-time convention — the factor-of-2 trap.** ``τ_d`` here is the
**amplitude** damping time (the ``τ`` for which the betatron *amplitude* decays
``e^{-t/τ_d}``). The action / emittance damps *twice* as fast, so the emittance
damping time is ``τ_ε = τ_d/2``; if you hold ``τ_ε``, pass ``2·τ_ε``. accsim has
no radiation model yet (Stage 5+), so ``τ_d`` is a caller-supplied input.

The aperture ``A`` and beam size ``σ`` are in the same length units, and ``ξ``
shares its ``·/2σ²`` structure with the circular-transmission formula
``1 − exp(−R²/2σ²)`` — the same aperture-to-sigma ratio governs both.
"""

from __future__ import annotations

import math


def quantum_lifetime(aperture: float, sigma: float, amplitude_damping_time: float) -> float:
    r"""Aperture-limited quantum lifetime ``τ_q = τ_d · e^ξ/(2ξ)``, ``ξ = A²/2σ²``.

    Parameters
    ----------
    aperture
        Half-aperture ``A`` [m] — the transverse distance from the reference orbit
        to the physical limit (same units as ``sigma``).
    sigma
        RMS betatron beam size ``σ`` [m] at the aperture.
    amplitude_damping_time
        Radiation **amplitude** damping time ``τ_d`` [s] (amplitude ``∝ e^{-t/τ_d}``;
        the emittance damps at ``τ_d/2`` — see the module docstring). Same time unit
        as the returned lifetime.

    Returns
    -------
    float
        Quantum lifetime in the units of ``amplitude_damping_time``. This is the
        ``ξ ≫ 1`` closed form; it is the asymptote of the exact mean-first-passage
        integral and is accurate to ``O(1/ξ)`` (the regime of any real ring, where
        ``ξ`` is typically tens).
    """
    if aperture <= 0:
        raise ValueError(f"aperture must be > 0, got {aperture}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if amplitude_damping_time <= 0:
        raise ValueError(f"amplitude_damping_time must be > 0, got {amplitude_damping_time}")
    xi = aperture**2 / (2.0 * sigma**2)
    return amplitude_damping_time * math.exp(xi) / (2.0 * xi)


def quantum_lifetime_exact(
    aperture: float, sigma: float, amplitude_damping_time: float, *, terms: int = 10000
) -> float:
    r"""Exact aperture-limited quantum lifetime — the MFPT integral, not its asymptote.

    ``τ_q = (τ_d/2) ∫₀^ξ (e^w − 1)/w dw``, ``ξ = A²/2σ²`` — the mean first-passage
    time from the core to the aperture of the amplitude-diffusion Fokker–Planck
    generator, of which :func:`quantum_lifetime` returns the ``ξ ≫ 1`` asymptote.

    Use this whenever ``ξ`` is not large. The asymptote is accurate to ``O(1/ξ)``,
    which at the ``ξ ≈ 4`` of a deliberately tight acceptance is **29%** — not a
    tolerance but a different number (``17.667`` against ``13.650`` at ``ξ = 4``).
    The two-term expansion does not rescue it either: ``1 + 1/ξ = 1.25`` and
    ``1 + 1/ξ + 2/ξ² = 1.375`` *bracket* the true ratio ``1.2944``, so the
    asymptotic series is only good to 5–10% there. Gate against this function.

    Evaluated as the everywhere-positive series ``Σ_{n≥1} ξⁿ/(n·n!)``, which is
    stable at every ``ξ``. The equivalent ``Ei(ξ) − γ − ln ξ`` is not: it is a
    difference of large near-equal terms as ``ξ → 0``. The suite cross-checks the
    two against each other where both are well conditioned.

    Parameters
    ----------
    aperture, sigma, amplitude_damping_time
        As :func:`quantum_lifetime` — ``A`` and ``σ`` in the same length unit,
        ``τ_d`` the **amplitude** damping time (the emittance damps at ``τ_d/2``).
    terms
        Series-term cap. The terms peak near ``n ≈ ξ`` and the default is far
        beyond the ``ξ ≲ 709`` at which ``e^ξ`` overflows a float anyway.

    Returns
    -------
    float
        Quantum lifetime in the units of ``amplitude_damping_time``.

    Raises
    ------
    ValueError
        On a non-positive input, or if the series has not converged in ``terms``.

    Notes
    -----
    This is the mean time for *one* particle starting at the core to reach the
    aperture. The decay constant of a *surviving population* is the slowest
    eigenvalue of the same generator with an absorbing boundary, and the two agree
    only as ``ξ → ∞``: measured, ``τ_q/τ_decay`` is 1.135 at ``ξ = 3``, 1.080 at
    ``ξ = 4``, 1.005 at ``ξ = 8`` and 1.0004 at ``ξ = 12``. A fitted survival curve
    is the *decay* constant; do not gate it against this function at small ``ξ``.
    """
    if aperture <= 0:
        raise ValueError(f"aperture must be > 0, got {aperture}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if amplitude_damping_time <= 0:
        raise ValueError(f"amplitude_damping_time must be > 0, got {amplitude_damping_time}")
    xi = aperture**2 / (2.0 * sigma**2)

    total = 0.0
    term = 1.0
    for n in range(1, terms + 1):
        term *= xi / n
        total += term / n
        if term / n <= 1e-17 * total:
            break
    else:
        raise ValueError(f"series did not converge in {terms} terms at xi={xi}")
    return 0.5 * amplitude_damping_time * total
