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
from .collider import (
    beam_beam_tune_shift,
    hourglass_reduction,
    luminosity,
    piwinski_reduction,
)
from .coords import COORD_NAMES, DELTA, DIM, PX, PY, ZETA, X, Y
from .elements import (
    Aperture,
    BeamBeam,
    Collimator,
    Corrector,
    Dipole,
    Drift,
    Element,
    Octupole,
    Quadrupole,
    RFCavity,
    Sextupole,
    SkewQuadrupole,
    ThinOctupole,
    ThinQuadrupole,
    ThinSextupole,
    ThinSkewQuadrupole,
    ThinSkewSextupole,
)
from .lattice import Lattice, matrix_of
from .lifetime import quantum_lifetime
from .longitudinal import (
    longitudinal_hamiltonian,
    rf_bucket_height,
    separatrix,
)
from .matching import (
    InsertionMatchResult,
    Knob,
    MatchingError,
    MatchResult,
    Target,
    chromaticity_response_matrix,
    insertion_response_matrix,
    match_chromaticity,
    match_insertion,
    match_tunes,
    tune_response_matrix,
)
from .orbit import (
    ClosedOrbitError,
    OrbitConvergenceError,
    OrbitCorrection,
    OrbitCorrectionError,
    OrbitStatistics,
    closed_orbit,
    closed_orbit_nonlinear,
    correct_orbit,
    linearised_element_maps,
    linearised_lattice,
    linearised_one_turn_map,
    misalign,
    misalignment_response,
    orbit_response_matrix,
    orbit_statistics,
    propagate_orbit,
    propagate_orbit_nonlinear,
)
from .radiation import (
    RadiationIntegrals,
    damping_partition_numbers,
    damping_times,
    energy_loss_per_turn,
    equilibrium_emittance,
    equilibrium_emittances_coupled,
    equilibrium_energy_spread,
    quantum_constant_cq,
    radiation_constant_cgamma,
    radiation_integrals,
)
from .reference import (
    CLIGHT,
    ELECTRON_MASS_EV,
    ELECTRON_RADIUS_M,
    PROTON_MASS_EV,
    ReferenceParticle,
)
from .symplectic import (
    J6,
    delta_from_pzeta,
    is_symplectic,
    is_symplectic_map,
    is_symplectic_map_canonical,
    jacobian,
    pzeta_from_delta,
)
from .tracking import Bunch, LossResult, Particle, Tracker
from .tune import ellipse_from_trajectory, naff, tracked_tunes
from .twiss import (
    CoupledLatticeError,
    CoupledTwiss,
    Twiss,
    UnstableLatticeError,
    amplitude_detuning,
    beam_sigma,
    chromaticity,
    chromaticity_on_orbit,
    closed_twiss,
    closed_twiss_on_orbit,
    closest_tune_approach,
    coupled_beam_sigma,
    coupled_twiss,
    coupled_twiss_on_orbit,
    is_stable,
    match_periodic,
    match_periodic_coupled,
    momentum_compaction,
    natural_chromaticity,
    natural_chromaticity_on_orbit,
    normal_mode_tunes,
    propagate_coupled_twiss,
    propagate_coupled_twiss_on_orbit,
    propagate_twiss,
    propagate_twiss_on_orbit,
    slip_factor,
    synchrotron_tune,
    tunes,
    tunes_on_orbit,
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
    "Corrector",
    "Sextupole",
    "ThinSextupole",
    "Octupole",
    "ThinOctupole",
    "SkewQuadrupole",
    "ThinSkewQuadrupole",
    "ThinSkewSextupole",
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
    "equilibrium_emittances_coupled",
    "equilibrium_energy_spread",
    # symplectic helpers
    "is_symplectic",
    "is_symplectic_map",
    "is_symplectic_map_canonical",
    "pzeta_from_delta",
    "delta_from_pzeta",
    "jacobian",
    "J6",
    # twiss / optics
    "Twiss",
    "UnstableLatticeError",
    "CoupledLatticeError",
    "match_periodic",
    "closed_twiss",
    "propagate_twiss",
    "closed_twiss_on_orbit",
    "propagate_twiss_on_orbit",
    "tunes_on_orbit",
    "amplitude_detuning",
    "tunes",
    "is_stable",
    "normal_mode_tunes",
    "closest_tune_approach",
    "CoupledTwiss",
    "coupled_twiss",
    "coupled_twiss_on_orbit",
    "match_periodic_coupled",
    "propagate_coupled_twiss",
    "propagate_coupled_twiss_on_orbit",
    "coupled_beam_sigma",
    "natural_chromaticity",
    "chromaticity",
    "natural_chromaticity_on_orbit",
    "chromaticity_on_orbit",
    "momentum_compaction",
    "slip_factor",
    "synchrotron_tune",
    # matching (H1)
    "Knob",
    "MatchResult",
    "MatchingError",
    "tune_response_matrix",
    "match_tunes",
    "chromaticity_response_matrix",
    "match_chromaticity",
    # matching (H2) — local optics at a point, N knobs -> M targets
    "Target",
    "InsertionMatchResult",
    "insertion_response_matrix",
    "match_insertion",
    # closed orbit & its correction (I1)
    "ClosedOrbitError",
    "closed_orbit",
    "propagate_orbit",
    "OrbitCorrection",
    "OrbitCorrectionError",
    "orbit_response_matrix",
    "correct_orbit",
    # sextupole feed-down on a distorted orbit (I2)
    "OrbitConvergenceError",
    "closed_orbit_nonlinear",
    "propagate_orbit_nonlinear",
    "linearised_element_maps",
    "linearised_one_turn_map",
    "linearised_lattice",
    # misalignments & the statistical orbit (K1)
    "misalignment_response",
    "orbit_statistics",
    "OrbitStatistics",
    "misalign",
    # tracking-based tune (NAFF)
    "naff",
    "ellipse_from_trajectory",
    "tracked_tunes",
    # longitudinal (nonlinear RF bucket)
    "longitudinal_hamiltonian",
    "rf_bucket_height",
    "separatrix",
    "beam_sigma",
    # collider (Stage 6)
    "luminosity",
    "piwinski_reduction",
    "hourglass_reduction",
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
