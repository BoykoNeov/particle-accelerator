r"""Analytic gate for the transverse mass and the W Jacobian edge (E1).

Layered so that a wrong ``m_T`` definition and a wrong test sample cannot cancel:

1. **Definition identities** — hand-computable configurations (back-to-back,
   collinear), plus the two exact symmetries that pin the *form* of ``m_T``:
   invariance under a common azimuthal rotation and under a **longitudinal**
   boost.
2. **The endpoint** ``m_T <= M`` — asserted on explicitly constructed two-body
   decays, including ``W``s with recoil ``p_T`` (where it must still hold) and
   under a ``V-A`` angular weight (where it must still hold too). This is the
   convention-independent anchor the pipeline gates on.
3. **The shape** ``dN/dm_T = m_T/(M sqrt(M^2 - m_T^2))`` — checked against a
   sampled isotropic decay, with the isotropy assumption stated. Derived
   symbolically; re-derived here from the CDF so the closed form is not merely
   remembered.
4. **The ``M/2`` trap** — the lepton-``p_T`` peak sits at ``M/2`` while the
   ``m_T`` edge sits at ``M``. Asserted together, because confusing the two is
   the classic error this observable invites.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim.events import (
    invariant_mass_squared,
    jacobian_peak_pdf,
    transverse_mass,
    transverse_mass_from_vectors,
)

M_W = 80.379  # GeV, PDG-ish; the tests are scale-free apart from this label.


# --------------------------------------------------------------------------
# helpers: explicit two-body decay four-vectors, built here (not imported), so
# a generator bug cannot make the kinematics module look right.
# --------------------------------------------------------------------------
def _two_body_decay(
    mass: float, n: int, rng: np.random.Generator, costheta: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``mass`` -> two massless daughters, back-to-back in the rest frame.

    Returns ``(p_lep, p_nu)``, each shape ``(n, 4)`` as ``(E, px, py, pz)``.
    ``costheta`` may be supplied to impose a non-isotropic angular weight.
    """
    ct = rng.uniform(-1.0, 1.0, n) if costheta is None else np.asarray(costheta)
    st = np.sqrt(1.0 - ct**2)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    e = mass / 2.0
    px, py, pz = e * st * np.cos(phi), e * st * np.sin(phi), e * ct
    lep = np.stack([np.full(n, e), px, py, pz], axis=-1)
    nu = np.stack([np.full(n, e), -px, -py, -pz], axis=-1)
    return lep, nu


def _boost(p: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Lorentz-boost four-vectors ``p`` (shape ``(n, 4)``) by velocity ``beta`` (``(3,)``)."""
    b2 = float(np.dot(beta, beta))
    gamma = 1.0 / np.sqrt(1.0 - b2)
    e, vec = p[..., 0], p[..., 1:]
    bp = vec @ beta
    e_new = gamma * (e + bp)
    vec_new = vec + ((gamma - 1.0) * bp / b2 + gamma * e)[..., None] * beta
    return np.concatenate([e_new[..., None], vec_new], axis=-1)


# --------------------------------------------------------------------------
# 1. definition identities
# --------------------------------------------------------------------------
def test_back_to_back_pair_gives_twice_the_pt() -> None:
    """Exactly back-to-back (``Δφ = π``): ``m_T = 2 sqrt(p_T1 p_T2)``, hand-checkable."""
    assert transverse_mass(45.0, 0.0, 45.0, np.pi) == pytest.approx(90.0, rel=1e-15)
    assert transverse_mass(30.0, 1.3, 12.0, 1.3 + np.pi) == pytest.approx(
        2.0 * np.sqrt(30.0 * 12.0), rel=1e-15
    )


def test_collinear_pair_vanishes_without_nan() -> None:
    """``Δφ = 0`` gives ``m_T = 0`` — and the rounding clip must not leak a NaN."""
    dphis = np.array([0.0, 2.0 * np.pi, -2.0 * np.pi, 1e-9])
    m_t = transverse_mass(40.0, dphis, 40.0, 0.0)
    assert np.all(np.isfinite(m_t))
    assert m_t[0] == 0.0 and m_t[1] == pytest.approx(0.0, abs=1e-6)


def test_invariant_under_common_azimuthal_rotation() -> None:
    """``m_T`` depends on ``Δφ`` only — a global rotation must not move it."""
    rng = np.random.default_rng(20260720)
    ptl, ptn = rng.uniform(1, 60, 500), rng.uniform(1, 60, 500)
    phil, phin = rng.uniform(-np.pi, np.pi, 500), rng.uniform(-np.pi, np.pi, 500)
    base = transverse_mass(ptl, phil, ptn, phin)
    rotated = transverse_mass(ptl, phil + 0.77, ptn, phin + 0.77)
    assert np.allclose(base, rotated, rtol=0, atol=1e-12)


def test_invariant_under_longitudinal_boost() -> None:
    """The reason ``m_T`` is usable at a hadron collider: the unknown longitudinal
    boost of the ``qqbar`` system leaves it exactly unchanged."""
    rng = np.random.default_rng(7)
    lep, nu = _two_body_decay(M_W, 2000, rng)
    rest = transverse_mass_from_vectors(lep, nu)
    for beta_z in (0.3, -0.6, 0.95):
        b = np.array([0.0, 0.0, beta_z])
        boosted = transverse_mass_from_vectors(_boost(lep, b), _boost(nu, b))
        assert np.allclose(rest, boosted, rtol=1e-12, atol=1e-10)


def test_ignores_energy_and_pz_by_construction() -> None:
    """The four-vector wrapper must use *only* the transverse components — the MET
    estimator has no ``p_z`` at all, so leaking one in would be unphysical."""
    rng = np.random.default_rng(11)
    lep, nu = _two_body_decay(M_W, 200, rng)
    scrambled = nu.copy()
    scrambled[:, 0] *= 3.7  # nonsense energy
    scrambled[:, 3] = rng.normal(0, 50, 200)  # nonsense p_z
    assert np.allclose(
        transverse_mass_from_vectors(lep, nu),
        transverse_mass_from_vectors(lep, scrambled),
        rtol=0,
        atol=0,
    )


# --------------------------------------------------------------------------
# 2. the endpoint — the anchor
# --------------------------------------------------------------------------
def test_endpoint_is_the_parent_mass_at_rest() -> None:
    """``m_T <= M``, saturated at ``θ = 90°``. This is the Jacobian edge location."""
    rng = np.random.default_rng(3)
    lep, nu = _two_body_decay(M_W, 200_000, rng)
    m_t = transverse_mass_from_vectors(lep, nu)
    assert np.max(m_t) <= M_W * (1.0 + 1e-12)
    # dense sampling gets arbitrarily close to it
    assert np.max(m_t) > M_W * (1.0 - 1e-4)
    # the saturating configuration is exactly transverse
    lep90, nu90 = _two_body_decay(M_W, 1, rng, costheta=np.zeros(1))
    assert transverse_mass_from_vectors(lep90, nu90)[0] == pytest.approx(M_W, rel=1e-14)


def test_endpoint_survives_transverse_recoil() -> None:
    """A ``W`` with recoil ``p_T`` (ISR) still respects ``m_T <= M_W``.

    This is the property that makes ``m_T`` the ``W``-mass observable: unlike the
    lepton-``p_T`` peak, the edge does not walk with the boson's transverse
    momentum. Checked with a large transverse boost, far beyond real ISR.
    """
    rng = np.random.default_rng(5)
    lep, nu = _two_body_decay(M_W, 100_000, rng)
    for beta in ([0.4, 0.0, 0.0], [0.2, 0.3, 0.5]):
        b = np.array(beta)
        lb, nb = _boost(lep, b), _boost(nu, b)
        # sanity: the boost preserved the parent invariant mass
        assert np.allclose(np.sqrt(invariant_mass_squared(lb + nb)), M_W, rtol=1e-10)
        assert np.max(transverse_mass_from_vectors(lb, nb)) <= M_W * (1.0 + 1e-10)


def test_endpoint_survives_a_v_minus_a_angular_weight() -> None:
    """``V-A`` reweights the decay angle — the *shape* moves, the endpoint does not."""
    rng = np.random.default_rng(13)
    # (1 - cos θ)^2 weight, accept-reject
    ct = rng.uniform(-1.0, 1.0, 400_000)
    keep = rng.uniform(0.0, 4.0, ct.size) < (1.0 - ct) ** 2
    ct = ct[keep]
    lep, nu = _two_body_decay(M_W, ct.size, rng, costheta=ct)
    m_t = transverse_mass_from_vectors(lep, nu)
    assert np.max(m_t) <= M_W * (1.0 + 1e-12)
    assert np.max(m_t) > M_W * (1.0 - 1e-3)
    # and the shape really did move, so this is not a vacuous restatement
    iso_lep, iso_nu = _two_body_decay(M_W, ct.size, rng)
    iso = transverse_mass_from_vectors(iso_lep, iso_nu)
    assert abs(np.median(m_t) - np.median(iso)) > 1.0


# --------------------------------------------------------------------------
# 3. the shape (isotropic assumption stated)
# --------------------------------------------------------------------------
def test_pdf_is_normalised_and_matches_its_cdf() -> None:
    """``dN/dm_T = m_T/(M sqrt(M^2-m_T^2))`` integrates to 1 with CDF
    ``1 - sqrt(1 - m_T^2/M^2)`` — both derived symbolically (sympy), re-checked
    here numerically so the closed form is never merely remembered."""
    # Substitution m_T = M sin(a) removes the endpoint singularity: the integrand
    # is analytically sin(a). Integrate by the **midpoint** rule — in factored
    # form the exact endpoint is inf * 0, so a grid that lands on a = pi/2
    # evaluates 0 * inf = nan/inf. That is a quadrature artifact, not a physics
    # one; the midpoint rule simply never samples the singular point.
    n = 200_000
    edges = np.linspace(0.0, np.pi / 2, n + 1)
    a = 0.5 * (edges[:-1] + edges[1:])
    integrand = jacobian_peak_pdf(M_W * np.sin(a), M_W) * M_W * np.cos(a)
    assert np.all(np.isfinite(integrand))
    assert float(np.sum(integrand) * (np.pi / 2 / n)) == pytest.approx(1.0, rel=1e-8)

    for frac in (0.25, 0.5, 0.9, 0.99):
        x = frac * M_W
        a_up = np.linspace(0.0, np.arcsin(frac), 100_001)
        cdf_num = np.trapezoid(
            jacobian_peak_pdf(M_W * np.sin(a_up), M_W) * M_W * np.cos(a_up), a_up
        )
        assert cdf_num == pytest.approx(1.0 - np.sqrt(1.0 - x**2 / M_W**2), rel=1e-7)


def test_sampled_isotropic_decay_reproduces_the_jacobian_shape() -> None:
    """The sampled ``m_T`` histogram follows the analytic density.

    **Stated assumption:** isotropic decay, on-shell parent, zero ``p_T``. A
    ``V-A`` weight or a finite width changes this shape (but not the endpoint —
    see the endpoint tests).
    """
    rng = np.random.default_rng(20260720)
    lep, nu = _two_body_decay(M_W, 400_000, rng)
    m_t = transverse_mass_from_vectors(lep, nu)

    edges = np.linspace(0.0, 0.98 * M_W, 41)  # stop short of the singularity
    counts, _ = np.histogram(m_t, bins=edges)
    width = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # expected fraction per bin from the exact CDF, not from the pdf at the centre
    cdf = 1.0 - np.sqrt(1.0 - (edges / M_W) ** 2)
    expected = np.diff(cdf) * m_t.size

    assert np.all(counts > 50), "bins too sparse for a meaningful chi2"
    chi2 = np.sum((counts - expected) ** 2 / expected)
    assert chi2 / counts.size < 2.0, f"chi2/ndf = {chi2 / counts.size:.2f}"

    # and the density itself agrees where the bins are narrow
    density = counts / (width * m_t.size)
    ref = jacobian_peak_pdf(centres, M_W)
    assert np.allclose(density, ref, rtol=0.12)


def test_pdf_edges_and_domain() -> None:
    """Zero below/above the support, divergent exactly at the endpoint."""
    assert jacobian_peak_pdf(np.array([0.0]), M_W)[0] == 0.0
    assert np.isinf(jacobian_peak_pdf(np.array([M_W]), M_W)[0])
    assert jacobian_peak_pdf(np.array([1.01 * M_W, 500.0]), M_W).tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        jacobian_peak_pdf(np.array([1.0]), -1.0)


# --------------------------------------------------------------------------
# 4. the M/2 trap
# --------------------------------------------------------------------------
def test_lepton_pt_peaks_at_half_the_mass_but_the_mt_edge_is_at_the_mass() -> None:
    """Both Jacobian peaks in one assertion, because confusing them is *the*
    error this observable invites: ``p_T^l`` edges at ``M/2``, ``m_T`` at ``M``."""
    rng = np.random.default_rng(17)
    lep, nu = _two_body_decay(M_W, 300_000, rng)
    pt_lep = np.hypot(lep[:, 1], lep[:, 2])
    m_t = transverse_mass_from_vectors(lep, nu)

    assert np.max(pt_lep) <= M_W / 2.0 * (1.0 + 1e-12)
    assert np.max(pt_lep) > M_W / 2.0 * (1.0 - 1e-4)
    assert np.max(m_t) > M_W * (1.0 - 1e-4)
    # a factor of exactly two apart, in the idealised limit
    assert np.max(m_t) / np.max(pt_lep) == pytest.approx(2.0, rel=1e-3)
