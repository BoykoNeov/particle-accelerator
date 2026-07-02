"""Stage 6 acceptance (gate 1): luminosity.

Two independent checks, neither circular with the implementation:

1. **Gaussian overlap integral (symbolic).** The head-on luminosity per unit
   bunch-collision rate is the transverse overlap ``int rho1 rho2 dx dy`` of two
   normalized 2D Gaussian densities. Sympy evaluates it in closed form and it
   equals ``1/(4 pi sigma_x sigma_y)`` for equal beams — the coefficient the
   ``luminosity`` formula uses. This derives the ``4 pi``; it is not remembered.

2. **LHC worked example (LHC Design Report Vol I, Table 2.1).** Nominal design
   parameters reproduce the published head-on ``~1.2e34`` and, with the nominal
   crossing angle, the design ``1.0e34 cm^-2 s^-1``. This is the roadmap's
   "textbook worked example for a known machine".
"""

from __future__ import annotations

import math

import sympy as sp

from accsim import ReferenceParticle
from accsim.collider import luminosity, piwinski_reduction


def test_gaussian_overlap_gives_four_pi_sigma_coefficient() -> None:
    """int rho1 rho2 d^2r for equal 2D Gaussians == 1/(4 pi sigma_x sigma_y)."""
    x, y, sx, sy = sp.symbols("x y sigma_x sigma_y", positive=True)

    def gaussian_2d(sig_x: sp.Symbol, sig_y: sp.Symbol) -> sp.Expr:
        norm = 1 / (2 * sp.pi * sig_x * sig_y)
        return norm * sp.exp(-(x**2) / (2 * sig_x**2) - y**2 / (2 * sig_y**2))

    overlap = sp.integrate(
        sp.integrate(gaussian_2d(sx, sy) * gaussian_2d(sx, sy), (x, -sp.oo, sp.oo)),
        (y, -sp.oo, sp.oo),
    )
    assert sp.simplify(overlap - 1 / (4 * sp.pi * sx * sy)) == 0

    # And the formula multiplies that overlap by N1*N2*f_rev*n_b (single crossing
    # rate), so luminosity == rate * overlap. Check numerically against the symbol.
    n1, n2, f, nb = 3.0e11, 2.0e11, 1.1e4, 5
    sxv, syv = 2.0e-5, 1.0e-5
    expected = n1 * n2 * f * nb * float(overlap.subs({sx: sxv, sy: syv}))
    assert math.isclose(luminosity(n1, n2, sxv, syv, f, nb), expected, rel_tol=1e-15)


def test_lhc_worked_example_headon_and_with_crossing() -> None:
    """LHC nominal -> head-on ~1.2e34 and design 1.0e34 cm^-2 s^-1 with crossing."""
    # LHC Design Report Vol I, Table 2.1 (nominal, 7 TeV per beam).
    n_bunch = 1.15e11
    n_bunches = 2808
    f_rev = 11245.0  # Hz
    beta_star = 0.55  # m
    eps_n = 3.75e-6  # m (normalized)
    proton = ReferenceParticle.from_total_energy(mass_eV=938.272e6, total_energy_eV=7.0e12)

    # Geometric emittance (the stray-gamma trap: divide the *normalized* eps by
    # beta0*gamma0, not gamma0 alone) -> IP beam size sqrt(eps * beta*).
    eps_geom = eps_n / (proton.beta0 * proton.gamma0)
    sigma = math.sqrt(eps_geom * beta_star)

    l_headon = luminosity(n_bunch, n_bunch, sigma, sigma, f_rev, n_bunches)
    # Published head-on ~1.2e34 cm^-2 s^-1 (i.e. 1.2e38 m^-2 s^-1).
    assert math.isclose(l_headon * 1e-4, 1.20e34, rel_tol=0.02)

    # With the nominal full crossing angle 285 urad and bunch length 7.55 cm the
    # Piwinski factor brings it to the design peak 1.0e34 cm^-2 s^-1.
    l_cross = luminosity(
        n_bunch,
        n_bunch,
        sigma,
        sigma,
        f_rev,
        n_bunches,
        crossing_angle=285e-6,
        sigma_z=0.0755,
    )
    assert math.isclose(l_cross * 1e-4, 1.0e34, rel_tol=0.03)
    # The crossing reduction is exactly the standalone Piwinski factor.
    assert math.isclose(
        l_cross / l_headon, piwinski_reduction(285e-6, 0.0755, sigma), rel_tol=1e-12
    )


def test_piwinski_limits_and_half_angle() -> None:
    """S -> 1 head-on / point bunch; uses tan(phi/2), not tan(phi)."""
    sigma, sigma_z = 2.0e-5, 0.08
    assert piwinski_reduction(0.0, sigma_z, sigma) == 1.0  # head-on
    assert piwinski_reduction(3e-4, 0.0, sigma) == 1.0  # point-like bunch
    # Half-angle: the reduction at full angle phi equals the naive tan(phi/2) form.
    phi = 6e-4
    piw = sigma_z * math.tan(phi / 2) / sigma
    assert math.isclose(
        piwinski_reduction(phi, sigma_z, sigma), 1 / math.hypot(1, piw), rel_tol=1e-14
    )


def test_classical_radius_electron_and_proton() -> None:
    """r0 = r_e for the electron and r_e*(m_e/m_p) ~ 1.535e-18 m for the proton."""
    from accsim.reference import ELECTRON_MASS_EV, ELECTRON_RADIUS_M

    electron = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 10e9)
    proton = ReferenceParticle.from_total_energy(938.272e6, 7e12)
    assert math.isclose(electron.classical_radius_m, ELECTRON_RADIUS_M, rel_tol=1e-15)
    assert math.isclose(proton.classical_radius_m, 1.5347e-18, rel_tol=1e-3)
    # charge^2 scaling: a hypothetical q=2 proton has 4x the classical radius.
    alpha = ReferenceParticle.from_total_energy(938.272e6, 7e12, charge=2.0)
    assert math.isclose(alpha.classical_radius_m, 4 * proton.classical_radius_m, rel_tol=1e-15)
