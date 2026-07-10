"""Shared pytest fixtures for the accsim test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from accsim import ELECTRON_MASS_EV, PROTON_MASS_EV, ReferenceParticle, features


@pytest.fixture(autouse=True)
def _reset_feature_switches() -> Iterator[None]:
    """Keep optional-addon switches from leaking across tests.

    ``accsim.features`` overrides are process-global; without this a test that
    ``enable``s an addon would flip it on for every later test. Reset before and
    after each test so every test starts from the default (all addons OFF).
    """
    features.reset()
    yield
    features.reset()


@pytest.fixture
def electron_2gev() -> ReferenceParticle:
    """A 2 GeV (total energy) electron — ultrarelativistic, gamma0 ~ 3914."""
    return ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 2.0e9, charge=-1.0)


@pytest.fixture
def proton_gamma5() -> ReferenceParticle:
    """A proton at gamma0 = 5 — non-ultrarelativistic, so 1/gamma0^2 is sizeable."""
    return ReferenceParticle.from_gamma(PROTON_MASS_EV, 5.0)
