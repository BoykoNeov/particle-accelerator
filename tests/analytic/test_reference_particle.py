"""Analytic checks on reference-particle kinematics.

These pin the relativistic identities that every transfer matrix depends on. A
silent error here (e.g. a momentum/energy mix-up) would propagate into every
element, so they are checked directly against closed-form relations.
"""

from __future__ import annotations

import math

import pytest

from accsim import ELECTRON_MASS_EV, PROTON_MASS_EV, ReferenceParticle


def test_beta_gamma_identity() -> None:
    # beta^2 = 1 - 1/gamma^2 must hold exactly for any gamma.
    ref = ReferenceParticle.from_gamma(PROTON_MASS_EV, 7.0)
    assert ref.gamma0 == pytest.approx(7.0)
    assert ref.beta0**2 == pytest.approx(1.0 - 1.0 / 7.0**2)


def test_energy_momentum_relation() -> None:
    # E0^2 = (p0 c)^2 + (m c^2)^2  — the relativistic dispersion relation.
    ref = ReferenceParticle.from_total_energy(PROTON_MASS_EV, 10.0e9)
    assert ref.total_energy_eV**2 == pytest.approx(ref.momentum_eV**2 + ref.mass_eV**2)


def test_from_momentum_roundtrip() -> None:
    # Build from a known momentum, recover that momentum.
    p0c = 3.0e9  # eV
    ref = ReferenceParticle.from_momentum(PROTON_MASS_EV, p0c)
    assert ref.momentum_eV == pytest.approx(p0c)
    # ... and gamma = E/m with E = sqrt(p^2 + m^2).
    expected_gamma = math.hypot(p0c, PROTON_MASS_EV) / PROTON_MASS_EV
    assert ref.gamma0 == pytest.approx(expected_gamma)


def test_from_kinetic_energy() -> None:
    ke = 50.0e6  # 50 MeV electron
    ref = ReferenceParticle.from_kinetic_energy(ELECTRON_MASS_EV, ke)
    assert ref.kinetic_energy_eV == pytest.approx(ke)
    assert ref.total_energy_eV == pytest.approx(ELECTRON_MASS_EV + ke)


def test_ultrarelativistic_beta_approaches_one() -> None:
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 2.0e9)
    assert ref.beta0 < 1.0
    assert ref.beta0 > 0.9999999


def test_rejects_energy_below_rest_mass() -> None:
    with pytest.raises(ValueError):
        ReferenceParticle(mass_eV=PROTON_MASS_EV, total_energy_eV=0.5 * PROTON_MASS_EV)


def test_rejects_nonpositive_mass() -> None:
    with pytest.raises(ValueError):
        ReferenceParticle(mass_eV=0.0, total_energy_eV=1.0)
