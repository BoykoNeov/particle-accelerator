r"""Relativistic kinematics for the toy event generator (Phase 2 learning module).

**Natural units.** Unlike the beam-dynamics core (SI metres, eV; see
``docs/CONVENTIONS.md`` -> *Units*), the event-physics module works in **natural
units** ``hbar = c = 1`` with energies/momenta in **GeV**. This is the universal
convention of particle-physics cross-section calculations, and keeping it local to
``accsim.events`` avoids threading ``c`` factors through every Mandelstam. The
one boundary crossing back to laboratory units is the cross-section, converted from
``GeV^-2`` to barns via ``(hbar c)^2`` in :mod:`accsim.events.generator`.

**Metric.** Mostly-minus ``(+, -, -, -)``, so a four-vector ``p = (E, px, py, pz)``
has invariant ``p.p = E^2 - |p_vec|^2 = m^2``. Four-vectors are plain
``numpy`` arrays of shape ``(4,)`` (single) or ``(..., 4)`` (batched, energy in the
last axis's index 0).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "METRIC",
    "minkowski_dot",
    "invariant_mass_squared",
    "mandelstam_s",
    "mandelstam_t",
    "mandelstam_u",
    "collins_soper_costheta",
    "forward_backward_asymmetry",
]

# Mostly-minus metric diag(+1, -1, -1, -1). Contracting with it turns the naive
# Euclidean dot into the Minkowski one.
METRIC: npt.NDArray[np.float64] = np.array([1.0, -1.0, -1.0, -1.0])


def minkowski_dot(
    p: npt.NDArray[np.float64], q: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | float:
    r"""Minkowski inner product ``p.q = E_p E_q - vec p . vec q``.

    Accepts single four-vectors (shape ``(4,)``) or batches (shape ``(..., 4)``);
    contraction is over the last axis, so batched inputs return the array of dots.
    """
    return np.sum(p * METRIC * q, axis=-1)


def invariant_mass_squared(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | float:
    """``p.p = m^2`` — the squared invariant mass of a (possibly summed) four-vector."""
    return minkowski_dot(p, p)


def mandelstam_s(
    p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | float:
    r"""``s = (p1 + p2)^2`` — the squared CM energy of the two incoming particles."""
    return invariant_mass_squared(p1 + p2)


def mandelstam_t(
    p1: npt.NDArray[np.float64], p3: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | float:
    r"""``t = (p1 - p3)^2`` — the momentum transfer from incoming ``p1`` to outgoing ``p3``."""
    return invariant_mass_squared(p1 - p3)


def mandelstam_u(
    p1: npt.NDArray[np.float64], p4: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | float:
    r"""``u = (p1 - p4)^2`` — the crossed momentum transfer (incoming ``p1`` to outgoing ``p4``)."""
    return invariant_mass_squared(p1 - p4)


def collins_soper_costheta(
    p_minus: npt.NDArray[np.float64],
    p_plus: npt.NDArray[np.float64],
    quark_direction: npt.NDArray[np.float64] | float | None = None,
) -> npt.NDArray[np.float64] | float:
    r"""``cos θ*`` of the ``μ⁻`` in the **Collins–Soper frame** of the di-lepton pair.

    The Collins–Soper (CS) frame is the di-lepton rest frame whose polar axis
    bisects the angle between beam 1 and the *reversed* beam 2 direction — the
    choice that minimises sensitivity to the pair's transverse momentum ``Q_T``.
    For a lepton ``ℓ⁻`` (``p_minus``, **particle 1**) and antilepton ``ℓ⁺``
    (``p_plus``) with the beams along ``±ẑ`` in the lab, ``cos θ*`` has the
    closed lab-frame form (Collins & Soper, *Phys. Rev.* **D16** (1977) 2219)

    .. math::

        \cos\theta^*_{\rm CS}
          = \frac{2\,(p^-_z E^+ - E^- p^+_z)}{m_{\ell\ell}\,\sqrt{m_{\ell\ell}^2 + Q_T^2}},

    i.e. ``2 (ℓ⁺_1 ℓ⁻_2 − ℓ⁻_1 ℓ⁺_2)/(Q√(Q²+Q_T²))`` in the light-cone
    components ``ℓ^± = (E ± p_z)/√2`` — the ``√2`` factors cancel to leave
    ``p^-_z E^+ − E^- p^+_z`` in the numerator. Derived, not memorised: the
    equality of this closed form to an explicit boost-into-rest-frame bisector
    construction is pinned over random pairs in
    ``tests/analytic/test_collins_soper.py``.

    **Quark-direction orientation (the ``pp`` dilution).** The formula above is
    referenced to ``+ẑ``. In a symmetric ``pp`` collision the quark direction is
    not known event-by-event, so the axis must be oriented by a proxy:

    - ``quark_direction=None`` (default) uses ``sign(Q_z)`` — the direction of
      the di-lepton longitudinal boost, the standard experimental proxy (the
      valence quark statistically carries more momentum than the sea antiquark).
      This is a *probabilistic* assignment, so the resulting ``A_FB`` is
      **diluted** below the parton-level value.
    - passing ``quark_direction`` (``±1``, e.g. the true incoming-quark ``p_z``
      sign from the generator record) orients the axis exactly, giving the
      **undiluted** parton-level ``cos θ*``. The difference between the two is
      the dilution — see the Drell-Yan pipeline.

    This is the standard **massless-lepton** closed form (the one every Drell-Yan
    experiment uses): it returns the geometric momentum-direction ``cos θ*`` in the
    ``m_ℓ → 0`` limit; at the real muon mass vs the ~45 GeV Z-decay momentum the
    difference is a ~1e-6 (``β_μ``) effect, negligible.

    Four-vectors are ``(E, px, py, pz)`` in natural units (GeV); single
    ``(4,)`` or batched ``(..., 4)``. Returns ``cos θ*`` with matching shape.
    """
    p_minus = np.asarray(p_minus, dtype=np.float64)
    p_plus = np.asarray(p_plus, dtype=np.float64)

    q = p_minus + p_plus
    e_q, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m2 = e_q**2 - qx**2 - qy**2 - qz**2  # m_ℓℓ²
    qt2 = qx**2 + qy**2  # Q_T²

    e_m, pz_m = p_minus[..., 0], p_minus[..., 3]  # ℓ⁻ energy, p_z
    e_p, pz_p = p_plus[..., 0], p_plus[..., 3]  # ℓ⁺ energy, p_z

    denom = np.sqrt(m2) * np.sqrt(m2 + qt2)  # m_ℓℓ · √(m_ℓℓ² + Q_T²)
    raw = 2.0 * (pz_m * e_p - e_m * pz_p) / denom

    if quark_direction is None:
        orient = np.sign(qz)  # proxy: the di-lepton boost direction
    else:
        orient = np.sign(np.asarray(quark_direction, dtype=np.float64))
    return raw * orient


def forward_backward_asymmetry(
    costheta: npt.NDArray[np.float64],
) -> tuple[float, float]:
    r"""Forward-backward asymmetry ``A_FB = (N_F − N_B)/(N_F + N_B)`` + its error.

    ``N_F`` (``N_B``) counts ``cos θ* > 0`` (``< 0``); events with ``cos θ* == 0``
    or non-finite (degenerate ``Q_z = 0`` proxy) are dropped. The returned error
    is the binomial ``√((1 − A_FB²)/N)`` on ``N = N_F + N_B`` counted pairs.

    Returns ``(A_FB, err)``; ``(nan, nan)`` if no pair survives.
    """
    c = np.asarray(costheta, dtype=np.float64)
    c = c[np.isfinite(c)]
    n_f = int(np.count_nonzero(c > 0.0))
    n_b = int(np.count_nonzero(c < 0.0))
    n = n_f + n_b
    if n == 0:
        return float("nan"), float("nan")
    afb = (n_f - n_b) / n
    err = float(np.sqrt(max(0.0, 1.0 - afb**2) / n))
    return float(afb), err
