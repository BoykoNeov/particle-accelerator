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
    "collins_soper_angles",
    "angular_coefficients",
    "forward_backward_asymmetry",
    "transverse_mass",
    "transverse_mass_from_vectors",
    "jacobian_peak_pdf",
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


def _boost_to_rest(
    p: npt.NDArray[np.float64], q: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Lorentz-boost four-vector(s) ``p`` into the rest frame of ``q`` (batched).

    Both ``(..., 4)`` with matching leading shape (or ``q`` broadcastable to ``p``).
    Pure active boost by ``-vec β`` with ``vec β = vec q / E_q``; where ``q`` is
    already at rest (``|vec β| = 0``) the vector is returned unchanged.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    e_q = q[..., 0]
    q_vec = q[..., 1:]
    m = np.sqrt(np.maximum(e_q**2 - np.sum(q_vec**2, axis=-1), 0.0))
    beta = q_vec / e_q[..., None]  # (..., 3)
    b2 = np.sum(beta**2, axis=-1)  # (...,)
    gamma = np.where(m > 0.0, e_q / np.where(m > 0.0, m, 1.0), 1.0)

    e_p = p[..., 0]
    p_vec = p[..., 1:]
    bp = np.sum(beta * p_vec, axis=-1)  # (...,)
    safe_b2 = np.where(b2 > 0.0, b2, 1.0)
    coeff = ((gamma - 1.0) * bp / safe_b2 - gamma * e_p)[..., None]
    p_new = np.where(b2[..., None] > 0.0, p_vec + coeff * beta, p_vec)
    return p_new


def collins_soper_angles(
    p_minus: npt.NDArray[np.float64],
    p_plus: npt.NDArray[np.float64],
    quark_direction: npt.NDArray[np.float64] | float | None = None,
) -> tuple[npt.NDArray[np.float64] | float, npt.NDArray[np.float64] | float]:
    r"""``(cos θ*, φ*)`` of the ``ℓ⁻`` in the **Collins–Soper frame**, by construction.

    Unlike :func:`collins_soper_costheta` (a memorised-free but *closed-form*
    massless-lepton projection), this builds the CS frame **explicitly** and
    projects — the geometric route, which needs no closed form for the
    error-prone azimuth ``φ*``. Both muons and the two beam directions are boosted
    into the di-lepton rest frame; there the axes are

    .. math::

        \hat z_{\rm CS} = \widehat{\hat k_1 - \hat k_2}, \quad
        \hat y_{\rm CS} = \widehat{\hat k_1 \times \hat k_2}, \quad
        \hat x_{\rm CS} = \hat y_{\rm CS} \times \hat z_{\rm CS},

    with ``k̂₁, k̂₂`` the boosted directions of beam 1 (``+ẑ`` in the lab) and beam 2
    (``−ẑ``). ``ẑ_CS`` bisects beam 1 and the *reversed* beam 2 (the CS choice that
    minimises ``Q_T`` sensitivity); ``ŷ_CS`` is normal to the beam plane; ``x̂_CS``
    completes a right-handed triad and points along ``+Q_T``. Then

    .. math::

        \cos\theta^* = \hat p_{\ell^-}\cdot\hat z_{\rm CS}, \qquad
        \varphi^* = \operatorname{atan2}(\hat p_{\ell^-}\cdot\hat y_{\rm CS},\;
                                          \hat p_{\ell^-}\cdot\hat x_{\rm CS}),

    with ``p̂`` the ``ℓ⁻`` direction in the rest frame. ``φ* ∈ (−π, π]``.

    **Quark-direction orientation (``pp`` dilution).** As for
    :func:`collins_soper_costheta`, the axis must be oriented by a proxy in a
    symmetric ``pp`` collision. Swapping which beam is the "quark" side sends
    ``ẑ_CS → −ẑ_CS`` and ``ŷ_CS → −ŷ_CS`` (``x̂_CS`` invariant), hence
    ``cos θ* → −cos θ*`` and ``φ* → −φ*``. ``quark_direction=None`` (default) uses
    ``sign(Q_z)``; a passed ``±1`` orients exactly (the undiluted parton-level
    angles). This construction uses the **full lepton kinematics** (no ``m_ℓ → 0``
    approximation): its ``cos θ*`` equals the massless closed form to ``O(β_ℓ)``,
    pinned in ``tests/analytic/test_angular_coefficients.py``.

    Four-vectors are ``(E, px, py, pz)`` in natural units (GeV); single ``(4,)`` or
    batched ``(..., 4)``. Returns ``(cos θ*, φ*)`` with the matching shape.
    """
    p_minus = np.asarray(p_minus, dtype=np.float64)
    p_plus = np.asarray(p_plus, dtype=np.float64)
    q = p_minus + p_plus

    # Lightlike beam reference directions in the lab (only their directions enter
    # the CS axes, so unit lightlike vectors suffice — beam energy cancels).
    beam_a = np.array([1.0, 0.0, 0.0, 1.0])  # beam 1, +ẑ (default quark side)
    beam_b = np.array([1.0, 0.0, 0.0, -1.0])  # beam 2, −ẑ

    def _unit(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(n > 0.0, n, 1.0)

    k1 = _unit(_boost_to_rest(np.broadcast_to(beam_a, q.shape), q))
    k2 = _unit(_boost_to_rest(np.broadcast_to(beam_b, q.shape), q))
    z_cs = _unit(k1 - k2)
    y_cs = _unit(np.cross(k1, k2))
    x_cs = np.cross(y_cs, z_cs)  # already unit (ŷ ⟂ ẑ, both unit)

    pm = _unit(_boost_to_rest(p_minus, q))
    cos_theta = np.sum(pm * z_cs, axis=-1)
    p_y = np.sum(pm * y_cs, axis=-1)
    p_x = np.sum(pm * x_cs, axis=-1)

    if quark_direction is None:
        orient = np.sign(q[..., 3])  # proxy: di-lepton boost direction
    else:
        orient = np.sign(np.asarray(quark_direction, dtype=np.float64))

    cos_theta = orient * cos_theta
    phi = np.arctan2(orient * p_y, p_x)  # φ* → −φ* under the same flip
    if cos_theta.ndim == 0:
        return float(cos_theta), float(phi)
    return cos_theta, phi


# ---------------------------------------------------------------------------
# Angular-coefficient (A_0..A_7) moment-projection weights.
#
# The Drell-Yan CS-frame angular distribution is
#   dσ/dΩ ∝ (1+cos²θ) + A₀·½(1−3cos²θ) + A₁ sin2θ cosφ + A₂·½ sin²θ cos2φ
#           + A₃ sinθ cosφ + A₄ cosθ + A₅ sin²θ sin2φ + A₆ sin2θ sinφ + A₇ sinθ sinφ.
# Each coefficient is recovered as the sample mean of a weight polynomial P_i(θ,φ):
# by orthogonality over 4π, ⟨P_i⟩ = A_i and every other term integrates to zero.
# The prefactors below are DERIVED symbolically (not memorised) — the derivation
# is pinned in tests/analytic/test_angular_coefficients.py (moment closure gate).
# ---------------------------------------------------------------------------
def _moment_weights(
    cos_theta: npt.NDArray[np.float64], phi: npt.NDArray[np.float64]
) -> dict[str, npt.NDArray[np.float64]]:
    """The eight projection polynomials ``P_i(θ, φ)`` evaluated per event."""
    ct = np.asarray(cos_theta, dtype=np.float64)
    st = np.sqrt(np.maximum(1.0 - ct**2, 0.0))  # sin θ ≥ 0
    ph = np.asarray(phi, dtype=np.float64)
    sin2t = 2.0 * st * ct  # sin 2θ
    return {
        # A₀: ⟨P₀⟩ = A₀ directly, P₀ = 4 − 10 cos²θ (⟨cos²θ⟩ = (4−A₀)/10).
        "A0": 4.0 - 10.0 * ct**2,
        "A1": 5.0 * sin2t * np.cos(ph),
        "A2": 10.0 * st**2 * np.cos(2.0 * ph),
        "A3": 4.0 * st * np.cos(ph),
        "A4": 4.0 * ct,
        "A5": 5.0 * st**2 * np.sin(2.0 * ph),
        "A6": 5.0 * sin2t * np.sin(ph),
        "A7": 4.0 * st * np.sin(ph),
    }


def angular_coefficients(
    cos_theta: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
) -> dict[str, float]:
    r"""Extract the Drell-Yan angular coefficients ``A₀ … A₇`` by moment projection.

    Given per-event Collins–Soper ``cos θ*`` and ``φ*`` of the ``ℓ⁻`` (from
    :func:`collins_soper_angles`) for events distributed as ``dσ/dΩ``, returns the
    eight coefficients of the standard decomposition

    .. math::

        \frac{d\sigma}{d\Omega} \propto (1+\cos^2\theta)
          + A_0\,\tfrac12(1-3\cos^2\theta) + A_1\,\sin2\theta\cos\varphi
          + A_2\,\tfrac12\sin^2\theta\cos2\varphi + A_3\,\sin\theta\cos\varphi
          + A_4\,\cos\theta + A_5\,\sin^2\theta\sin2\varphi
          + A_6\,\sin2\theta\sin\varphi + A_7\,\sin\theta\sin\varphi .

    Each ``A_i`` is the sample mean of a projection polynomial; by orthogonality of
    the polynomials over the full solid angle every other term integrates to zero,
    so ``⟨P_i⟩ = A_i``. The polynomial prefactors are derived symbolically (moment
    closure gate). **This requires full ``4π`` acceptance** — the inversion is
    biased by detector cuts, so extract at truth level; acceptance-corrected
    (folded) reco extraction is out of scope. Events must be distributed as
    ``dσ/dΩ`` (unweighted, or accept-rejected to it).

    Returns ``{"A0": …, …, "A7": …}`` (Python floats). The forward-backward
    asymmetry is ``A_FB = (3/8) A₄``.
    """
    w = _moment_weights(np.asarray(cos_theta), np.asarray(phi))
    out: dict[str, float] = {}
    for key, poly in w.items():
        out[key] = float(np.mean(poly))
    return out


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


def transverse_mass(
    pt_lep: npt.NDArray[np.float64],
    phi_lep: npt.NDArray[np.float64],
    pt_miss: npt.NDArray[np.float64],
    phi_miss: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Transverse mass ``m_T`` of a lepton + missing-transverse-momentum pair.

        m_T^2 = 2 p_T^l p_T^nu (1 - cos(phi_l - phi_nu))

    This is *the* W-mass observable: the neutrino escapes, so no full invariant
    mass exists — only the transverse projection is reconstructible. For an
    on-shell ``W -> l nu`` decay ``m_T <= M_W``, and the phase-space pile-up
    against that endpoint is the **Jacobian edge** (see
    :func:`jacobian_peak_pdf`).

    Angles are radians; the ``(1 - cos Δφ)`` form is periodic, so no wrapping of
    ``Δφ`` is needed. All four arguments broadcast against each other.

    **Why ``m_T``, not ``p_T^l``.** The lepton-``p_T`` spectrum also has a
    Jacobian peak, but at ``M_W/2``, and it is smeared to first order by the
    ``W``'s recoil transverse momentum. The ``m_T`` edge sits at ``M_W`` and is
    insensitive to that recoil at first order — which is why hadron-collider
    ``W``-mass measurements are built on it. See ``docs/CONVENTIONS.md`` ->
    *Transverse mass and the W Jacobian edge*.
    """
    ptl = np.asarray(pt_lep, dtype=np.float64)
    ptm = np.asarray(pt_miss, dtype=np.float64)
    dphi = np.asarray(phi_lep, dtype=np.float64) - np.asarray(phi_miss, dtype=np.float64)
    # Clip at zero: 1 - cos is non-negative analytically, but rounding can push
    # the product to ~-1e-17 for a collinear pair and NaN the sqrt.
    return np.sqrt(np.maximum(0.0, 2.0 * ptl * ptm * (1.0 - np.cos(dphi))))


def transverse_mass_from_vectors(
    p_lep: npt.NDArray[np.float64],
    p_miss: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""``m_T`` from four-vectors ``(E, px, py, pz)`` — only the transverse parts are used.

    Convenience wrapper over :func:`transverse_mass` for the pipeline, where both
    the truth neutrino and the reco MET arrive as four-vectors. The energies and
    ``p_z`` components are **ignored by construction**: ``m_T`` is a transverse
    observable, and the missing-momentum estimator has no ``p_z`` at all.
    """
    lep = np.asarray(p_lep, dtype=np.float64)
    mis = np.asarray(p_miss, dtype=np.float64)
    return transverse_mass(
        np.hypot(lep[..., 1], lep[..., 2]),
        np.arctan2(lep[..., 2], lep[..., 1]),
        np.hypot(mis[..., 1], mis[..., 2]),
        np.arctan2(mis[..., 2], mis[..., 1]),
    )


def jacobian_peak_pdf(m_t: npt.NDArray[np.float64], mass: float) -> npt.NDArray[np.float64]:
    r"""Idealised ``m_T`` density for an **on-shell, zero-``p_T``, isotropic** two-body decay.

        dN/dm_T = m_T / (M sqrt(M^2 - m_T^2)),   0 <= m_T <= M

    Derived symbolically (sympy), not remembered: in the ``W`` rest frame the two
    massless daughters are back-to-back, so ``Δφ = π`` exactly and both carry
    ``p_T = (M/2) sin θ``; hence ``m_T = M sin θ``. Pushing the isotropic
    ``cos θ ~ U(-1, 1)`` through that gives the density above (normalised to 1 on
    ``[0, M]``, CDF ``1 - sqrt(1 - m_T^2/M^2)``).

    The ``1/sqrt(M^2 - m_T^2)`` **integrable singularity at ``m_T = M``** is the
    Jacobian edge: ``dm_T/dcos θ -> 0`` at ``θ = 90°``, so a broad swathe of decay
    angles piles into a narrow ``m_T`` interval just below ``M``.

    **Scope — this is the idealised gate, not the collider truth.** Three real
    effects round the edge: the finite width ``Γ_W`` (``M`` is not one number),
    the ``W``'s recoil ``p_T`` from ISR (Sudakov-suppressed at low ``p_T``), and
    the MET resolution. The **endpoint location** survives all three; the shape
    does not. A ``V-A`` angular weight likewise changes the shape but **not** the
    endpoint, which is why the endpoint is what the pipeline gates on.

    Returns 0 outside ``[0, M)`` and ``inf`` exactly at the endpoint.
    """
    if mass <= 0:
        raise ValueError(f"mass must be positive, got {mass}")
    x = np.asarray(m_t, dtype=np.float64)
    out = np.zeros(np.broadcast(x, np.float64(0.0)).shape, dtype=np.float64)
    inside = (x >= 0.0) & (x < mass)
    with np.errstate(divide="ignore"):
        out = np.where(inside, x / (mass * np.sqrt(np.abs(mass**2 - x**2))), 0.0)
        out = np.where(x == mass, np.inf, out)
    return out
