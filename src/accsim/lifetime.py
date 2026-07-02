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
