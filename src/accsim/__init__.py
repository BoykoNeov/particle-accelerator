"""accsim — a modular, physics-correct particle accelerator simulator.

Public API (Stage 0). See ``docs/ROADMAP.md`` for the staged plan and
``docs/CONVENTIONS.md`` for the coordinate, unit, and sign conventions that the
whole codebase depends on.
"""

from __future__ import annotations

from .coords import COORD_NAMES, DELTA, DIM, PX, PY, ZETA, X, Y
from .elements import (
    Dipole,
    Drift,
    Element,
    Quadrupole,
    Sextupole,
    ThinQuadrupole,
    ThinSextupole,
)
from .lattice import Lattice, matrix_of
from .reference import ELECTRON_MASS_EV, PROTON_MASS_EV, ReferenceParticle
from .symplectic import J6, is_symplectic
from .tracking import Bunch, Particle, Tracker
from .twiss import (
    Twiss,
    UnstableLatticeError,
    chromaticity,
    closed_twiss,
    is_stable,
    match_periodic,
    natural_chromaticity,
    propagate_twiss,
    tunes,
)

__version__ = "0.0.1"

__all__ = [
    # coordinates
    "X",
    "PX",
    "Y",
    "PY",
    "ZETA",
    "DELTA",
    "DIM",
    "COORD_NAMES",
    # reference particle
    "ReferenceParticle",
    "ELECTRON_MASS_EV",
    "PROTON_MASS_EV",
    # elements
    "Element",
    "Drift",
    "Quadrupole",
    "ThinQuadrupole",
    "Dipole",
    "Sextupole",
    "ThinSextupole",
    # lattice
    "Lattice",
    "matrix_of",
    # tracking
    "Particle",
    "Bunch",
    "Tracker",
    # symplectic helpers
    "is_symplectic",
    "J6",
    # twiss / optics
    "Twiss",
    "UnstableLatticeError",
    "match_periodic",
    "closed_twiss",
    "propagate_twiss",
    "tunes",
    "is_stable",
    "natural_chromaticity",
    "chromaticity",
    "__version__",
]
