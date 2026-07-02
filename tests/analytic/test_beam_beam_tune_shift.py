"""Stage 6 acceptance (gate 2): the beam-beam tune shift ``xi`` matches the
analytic expression, established as the **small-amplitude limit of the kick**
(not a standalone remembered formula).

Three non-circular checks:

1. **Coefficient derived symbolically.** A thin lens ``[[1,0],[-k1l,1]]`` composed
   with a Courant-Snyder one-turn rotation ``R(mu; beta, alpha)`` has
   ``d mu/d k1l = beta/2`` (from ``cos mu' = 1/2 Tr``), so ``dQ/dk1l = beta/(4 pi)``.
   Sympy derives this; it never assumes the tune-shift formula.
2. **Through a real ring.** Insert the linearised ``BeamBeam`` into a FODO cell and
   measure the tune change with :func:`accsim.tunes` (the ``atan2`` accumulation,
   independent of the ``beta k1l/4pi`` formula). It reproduces
   ``dQ_u = -beta_u K/(4 pi)`` as ``K -> 0`` (residual scales like ``K^2``).
3. **The formula and its LHC value.** ``beam_beam_tune_shift`` returns exactly
   ``-beta_u K/(4 pi)``; for LHC-nominal round optics its magnitude is the standard
   ``xi ~ 0.0037`` per IP, and its sign is negative (proton-proton defocusing).
"""

from __future__ import annotations

import math

import pytest

from accsim import (
    BeamBeam,
    Drift,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    beam_beam_tune_shift,
    closed_twiss,
    tunes,
)

REF = ReferenceParticle.from_total_energy(938.272e6, 7.0e12)


def _thin_fodo(f: float, ll: float = 1.0) -> list:
    """Symmetric thin FODO (focusing quads split half-and-half at the ends)."""
    return [
        ThinQuadrupole(0.5 / f, name="qf/2"),
        Drift(ll, name="d1"),
        ThinQuadrupole(-1.0 / f, name="qd"),
        Drift(ll, name="d2"),
        ThinQuadrupole(0.5 / f, name="qf/2"),
    ]


def test_tune_shift_coefficient_is_beta_over_4pi_symbolic() -> None:
    """dQ/dk1l = beta/(4 pi) from the one-turn trace (derives the formula)."""
    sp = pytest.importorskip("sympy")
    beta, alpha, mu, k = sp.symbols("beta alpha mu k1l", positive=True)
    gamma = (1 + alpha**2) / beta
    R = sp.Matrix(
        [
            [sp.cos(mu) + alpha * sp.sin(mu), beta * sp.sin(mu)],
            [-gamma * sp.sin(mu), sp.cos(mu) - alpha * sp.sin(mu)],
        ]
    )
    lens = sp.Matrix([[1, 0], [-k, 1]])  # thin quad: Delta px = -k1l x
    m = lens * R
    cos_mu_new = m.trace() / 2
    # The one-turn trace shifts cleanly: 1/2 Tr(lens R) = cos mu - k1l beta sin mu / 2.
    assert sp.simplify(cos_mu_new - (sp.cos(mu) - k * beta * sp.sin(mu) / 2)) == 0
    # Implicit differentiation of cos mu'(k): d(cos mu')/dk = -sin mu' dmu'/dk, so at
    # k=0 (mu'=mu, sin mu != 0) dmu'/dk = -(d cos mu'/dk)/sin mu = beta/2 -- no Abs.
    dcos_dk = sp.diff(cos_mu_new, k).subs(k, 0)
    dmu_dk = -dcos_dk / sp.sin(mu)
    assert sp.simplify(dmu_dk - beta / 2) == 0
    # Q = mu/2pi  ->  dQ/dk1l = beta/(4 pi).


def test_tune_shift_through_ring_matches_formula() -> None:
    """Measured tune change from an inserted BeamBeam -> -beta K/(4 pi) as K -> 0."""
    fodo = _thin_fodo(f=1.5)
    tw0 = closed_twiss(Lattice(fodo, ref=REF))  # unperturbed beta at s=0
    q0x, q0y = tunes(Lattice(fodo, ref=REF))

    # A tiny strong-bunch strength so the linearised shift dominates the O(K^2)
    # amplitude term; scale sigma/N to land K at a small value.
    bb = BeamBeam(n_particles=1.0e9, sigma=2.0e-4, strong_charge=1.0)
    k = bb.strength(REF)
    ring = Lattice([bb, *fodo], ref=REF)  # beam-beam at s=0 where beta = tw0.beta_*
    q1x, q1y = tunes(ring)

    dqx_meas, dqy_meas = q1x - q0x, q1y - q0y
    dqx_pred = -tw0.beta_x * k / (4.0 * math.pi)
    dqy_pred = -tw0.beta_y * k / (4.0 * math.pi)
    assert math.isclose(dqx_meas, dqx_pred, rel_tol=1e-4)
    assert math.isclose(dqy_meas, dqy_pred, rel_tol=1e-4)
    # And this is exactly what beam_beam_tune_shift returns.
    dqx_fn, dqy_fn = beam_beam_tune_shift(bb, REF, tw0.beta_x, tw0.beta_y)
    assert math.isclose(dqx_fn, dqx_pred, rel_tol=1e-14)
    assert math.isclose(dqy_fn, dqy_pred, rel_tol=1e-14)


def test_residual_scales_quadratically() -> None:
    """Halving K quarters the (measured - linear) residual: confirms K^2 error."""
    fodo = _thin_fodo(f=1.5)
    tw0 = closed_twiss(Lattice(fodo, ref=REF))
    q0x, _ = tunes(Lattice(fodo, ref=REF))

    def residual(n_particles: float) -> float:
        bb = BeamBeam(n_particles=n_particles, sigma=1.0e-3, strong_charge=1.0)
        k = bb.strength(REF)
        q1x, _ = tunes(Lattice([bb, *fodo], ref=REF))
        return abs((q1x - q0x) - (-tw0.beta_x * k / (4.0 * math.pi)))

    r1 = residual(4.0e11)
    r2 = residual(2.0e11)  # half the strength
    assert r1 > 0
    assert math.isclose(r1 / r2, 4.0, rel_tol=0.15)  # quadratic: ratio ~ 4


def test_lhc_xi_value_and_pp_sign() -> None:
    """LHC-nominal round optics -> |dQ| = xi ~ 0.0037; sign negative (pp defocus)."""
    eps_n = 3.75e-6
    beta_star = 0.55
    eps_geom = eps_n / (REF.beta0 * REF.gamma0)
    sigma = math.sqrt(eps_geom * beta_star)
    bb = BeamBeam(n_particles=1.15e11, sigma=sigma, strong_charge=1.0)  # pp

    dqx, dqy = beam_beam_tune_shift(bb, REF, beta_star)  # round: beta_y = beta_x
    xi_analytic = (
        1.15e11 * REF.classical_radius_m * beta_star / (4.0 * math.pi * REF.gamma0 * sigma**2)
    )
    assert dqx == dqy  # round IP
    assert dqx < 0  # proton-proton: defocusing lowers the tune
    assert math.isclose(abs(dqx), xi_analytic, rel_tol=1e-14)
    assert math.isclose(abs(dqx), 0.0037, rel_tol=0.05)

    # Opposite charges (e.g. p-pbar) focus -> positive shift, same magnitude.
    bb_opp = BeamBeam(n_particles=1.15e11, sigma=sigma, strong_charge=-1.0)
    dqx_opp, _ = beam_beam_tune_shift(bb_opp, REF, beta_star)
    assert dqx_opp > 0
    assert math.isclose(dqx_opp, -dqx, rel_tol=1e-14)
