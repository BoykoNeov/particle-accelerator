r"""RAMBO: flat Lorentz-invariant phase-space sampling (Phase 2 learning module).

The n-body Lorentz-invariant phase space (LIPS) of a decay of total invariant mass
``W = sqrt(s)`` into ``n`` massless particles is

    dPhi_n = [prod_i d^3 p_i / ((2 pi)^3 2 E_i)] (2 pi)^4 delta^4(P - sum_i p_i).

**RAMBO** (Kleiss, Stirling & Ellis, *Comput. Phys. Commun.* **40** (1986) 359)
generates momenta distributed *uniformly* over this phase space with a **constant**
weight equal to the total volume, so a Monte-Carlo integral is just
``int f dPhi_n ~= volume * <f>``. The algorithm:

1. Draw ``n`` massless "raw" vectors ``q_i`` isotropically, each with energy
   ``q_i^0 = -ln(r_a r_b)`` (a unit-mean, correctly-weighted energy spectrum).
2. The sum ``Q = sum q_i`` is a timelike vector with mass ``M``; boost and rescale
   every ``q_i`` by the (unique) proper transformation that maps ``Q -> (W, 0)``.
   The result ``p_i`` is a valid, momentum-conserving, on-shell massless
   configuration with the flat LIPS weight.

**Massless-only.** This module implements the massless RAMBO. Massive final states
need the extra iterative rescaling step (out of scope for the ``e+e- -> mu+ mu-``
toy at ``sqrt(s) >> m_mu``). The **analytic volume**
:func:`massless_phase_space_volume` is the closed form used to gate the sampler
*before* any matrix element is trusted (see ``docs/CONVENTIONS.md`` -> *Toy event
generator*): for ``n = 2`` it is ``1/(8 pi)``, independent of ``s``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["RamboResult", "rambo", "massless_phase_space_volume"]


@dataclass(frozen=True)
class RamboResult:
    """One RAMBO batch: momenta and the (constant) phase-space weight.

    ``momenta`` has shape ``(n_events, n_particles, 4)`` with the mostly-minus
    metric of :mod:`accsim.events.kinematics` (energy in index 0). ``weight`` is the
    total massless LIPS volume :func:`massless_phase_space_volume` — identical for
    every event, since RAMBO is a *flat* sampler.
    """

    momenta: npt.NDArray[np.float64]
    weight: float


def massless_phase_space_volume(n_particles: int, sqrt_s: float) -> float:
    r"""Analytic volume of the massless ``n``-body Lorentz-invariant phase space.

        Phi_n = (pi/2)^(n-1) * (s)^(n-2) * (2 pi)^(4 - 3n) / (Gamma(n) Gamma(n-1)),

    with ``s = sqrt_s^2``. This is the RAMBO normalisation (KSE 1986, eq. 4.11).
    For ``n = 2`` it collapses to ``1/(8 pi)`` independent of ``s``; for ``n = 3``
    it is ``s / (256 pi^3)``. Derived/verified symbolically in the test suite — not
    a remembered constant.
    """
    if n_particles < 2:
        raise ValueError(f"n_particles must be >= 2, got {n_particles}")
    if sqrt_s <= 0:
        raise ValueError(f"sqrt_s must be positive, got {sqrt_s}")
    n = n_particles
    s = sqrt_s * sqrt_s
    return (
        (math.pi / 2.0) ** (n - 1)
        * s ** (n - 2)
        * (2.0 * math.pi) ** (4 - 3 * n)
        / (math.gamma(n) * math.gamma(n - 1))
    )


def rambo(
    n_particles: int,
    sqrt_s: float,
    n_events: int,
    rng: np.random.Generator,
) -> RamboResult:
    """Generate ``n_events`` flat massless phase-space configurations at ``sqrt_s``.

    Returns a :class:`RamboResult` whose ``momenta`` conserve four-momentum
    (``sum_i p_i = (sqrt_s, 0, 0, 0)`` per event, to floating precision) and are
    massless on-shell, with the constant flat-LIPS ``weight``.
    """
    if n_particles < 2:
        raise ValueError(f"n_particles must be >= 2, got {n_particles}")
    if sqrt_s <= 0:
        raise ValueError(f"sqrt_s must be positive, got {sqrt_s}")
    if n_events < 1:
        raise ValueError(f"n_events must be >= 1, got {n_events}")

    n, ne = n_particles, n_events

    # --- Step 1: isotropic massless raw vectors q_i with energy -ln(r_a r_b). ---
    c = 2.0 * rng.random((ne, n)) - 1.0  # cos(theta) uniform in [-1, 1]
    phi = 2.0 * math.pi * rng.random((ne, n))
    r_a = rng.random((ne, n))
    r_b = rng.random((ne, n))
    q0 = -np.log(r_a * r_b)  # energy; sum of two exponentials
    s_theta = np.sqrt(1.0 - c * c)
    q = np.empty((ne, n, 4))
    q[..., 0] = q0
    q[..., 1] = q0 * s_theta * np.cos(phi)
    q[..., 2] = q0 * s_theta * np.sin(phi)
    q[..., 3] = q0 * c

    # --- Step 2: boost/scale so that sum q_i -> (sqrt_s, 0). ---
    Q = q.sum(axis=1)  # (ne, 4)
    M = np.sqrt(Q[:, 0] ** 2 - Q[:, 1] ** 2 - Q[:, 2] ** 2 - Q[:, 3] ** 2)  # (ne,)
    b = -Q[:, 1:] / M[:, None]  # boost 3-vector, (ne, 3)
    x = sqrt_s / M  # rescale
    gamma = Q[:, 0] / M  # = sqrt(1 + |b|^2)
    a = 1.0 / (1.0 + gamma)  # (ne,)

    q_spatial = q[..., 1:]  # (ne, n, 3)
    q_energy = q[..., 0]  # (ne, n)
    bq = np.einsum("ej,enj->en", b, q_spatial)  # b . q_i, (ne, n)

    p = np.empty((ne, n, 4))
    p[..., 0] = x[:, None] * (gamma[:, None] * q_energy + bq)
    p[..., 1:] = x[:, None, None] * (
        q_spatial + b[:, None, :] * (q_energy[..., None] + a[:, None, None] * bq[..., None])
    )

    return RamboResult(momenta=p, weight=massless_phase_space_volume(n, sqrt_s))
