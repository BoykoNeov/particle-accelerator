"""accsim — a modular, physics-correct particle accelerator simulator.

Public API (Stage 0). See ``docs/ROADMAP.md`` for the staged plan and
``docs/CONVENTIONS.md`` for the coordinate, unit, and sign conventions that the
whole codebase depends on.
"""

from __future__ import annotations

from . import features
from .acceleration import (
    RampResult,
    accelerate,
    energy_gain_per_turn,
    synchronous_phase,
)
from .collider import beam_beam_tune_shift, luminosity, piwinski_reduction
from .coords import COORD_NAMES, DELTA, DIM, PX, PY, ZETA, X, Y
from .elements import (
    Aperture,
    BeamBeam,
    Collimator,
    Dipole,
    Drift,
    Element,
    Quadrupole,
    RFCavity,
    Sextupole,
    ThinQuadrupole,
    ThinSextupole,
)
from .lattice import Lattice, matrix_of
from .lifetime import quantum_lifetime
from .radiation import (
    RadiationIntegrals,
    damping_partition_numbers,
    damping_times,
    energy_loss_per_turn,
    equilibrium_emittance,
    equilibrium_energy_spread,
    quantum_constant_cq,
    radiation_constant_cgamma,
    radiation_integrals,
)
from .longitudinal import (
    longitudinal_hamiltonian,
    rf_bucket_height,
    separatrix,
)
from .reference import (
    CLIGHT,
    ELECTRON_MASS_EV,
    ELECTRON_RADIUS_M,
    PROTON_MASS_EV,
    ReferenceParticle,
)
from .symplectic import J6, is_symplectic
from .tracking import Bunch, LossResult, Particle, Tracker
from .twiss import (
    Twiss,
    UnstableLatticeError,
    beam_sigma,
    chromaticity,
    closed_twiss,
    is_stable,
    match_periodic,
    momentum_compaction,
    natural_chromaticity,
    propagate_twiss,
    slip_factor,
    synchrotron_tune,
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
    "ELECTRON_RADIUS_M",
    "CLIGHT",
    # elements
    "Element",
    "Drift",
    "Quadrupole",
    "ThinQuadrupole",
    "Dipole",
    "Sextupole",
    "ThinSextupole",
    "RFCavity",
    "Aperture",
    "Collimator",
    "BeamBeam",
    # lattice
    "Lattice",
    "matrix_of",
    # tracking
    "Particle",
    "Bunch",
    "Tracker",
    "LossResult",
    # lifetime models
    "quantum_lifetime",
    "RadiationIntegrals",
    "radiation_integrals",
    "radiation_constant_cgamma",
    "quantum_constant_cq",
    "energy_loss_per_turn",
    "damping_partition_numbers",
    "damping_times",
    "equilibrium_emittance",
    "equilibrium_energy_spread",
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
    "momentum_compaction",
    "slip_factor",
    "synchrotron_tune",
    # longitudinal (nonlinear RF bucket)
    "longitudinal_hamiltonian",
    "rf_bucket_height",
    "separatrix",
    "beam_sigma",
    # collider (Stage 6)
    "luminosity",
    "piwinski_reduction",
    "beam_beam_tune_shift",
    # acceleration (Stage 5)
    "energy_gain_per_turn",
    "synchronous_phase",
    "accelerate",
    "RampResult",
    # runtime feature switches (optional addons; default OFF)
    "features",
    "__version__",
]
