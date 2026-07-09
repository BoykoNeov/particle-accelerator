r"""Toy 2->2 event-physics learning module (Phase 2).

A clearly-labelled, from-scratch Monte-Carlo generator for ``e+ e- -> mu+ mu-`` at
tree level: matrix element + RAMBO flat phase space + MC integration. It exists to
demonstrate the *event side* of a collision end-to-end and to hit the Phase 2
acceptance gate (toy total cross-section matches the analytic value within
Monte-Carlo error) — **not** to replace a real generator. The roadmap's
*orchestrate, don't rebuild* rule stands for physics-grade work (Pythia/MadGraph ->
Delphes -> analysis); those tools do not build on this Windows/Python 3.14 host, so
the toy is the local realisation of Phase 2. See ``docs/ROADMAP.md`` (Phase 2) and
``docs/CONVENTIONS.md`` -> *Toy event generator*.

**Units.** This module works in natural units ``hbar = c = 1`` with GeV, unlike the
SI/eV beam-dynamics core; the boundary crossing back to laboratory units is the
cross-section, converted from ``GeV^-2`` to barns via ``(hbar c)^2``.
"""

from __future__ import annotations

from .generator import (
    GEV2_TO_MBARN,
    AngularDistribution,
    CrossSectionEstimate,
    ee_to_mumu_cross_section,
    ee_to_mumu_events,
    gev2_to_barn,
)
from .kinematics import (
    invariant_mass_squared,
    mandelstam_s,
    mandelstam_t,
    mandelstam_u,
    minkowski_dot,
)
from .matrix_element import ALPHA_EM, EEtoMuMu
from .phase_space import RamboResult, massless_phase_space_volume, rambo

__all__ = [
    "ALPHA_EM",
    "GEV2_TO_MBARN",
    "AngularDistribution",
    "CrossSectionEstimate",
    "EEtoMuMu",
    "RamboResult",
    "ee_to_mumu_cross_section",
    "ee_to_mumu_events",
    "gev2_to_barn",
    "invariant_mass_squared",
    "mandelstam_s",
    "mandelstam_t",
    "mandelstam_u",
    "massless_phase_space_volume",
    "minkowski_dot",
    "rambo",
]
