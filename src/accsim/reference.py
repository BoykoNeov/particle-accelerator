"""The reference (synchronous) particle.

Transfer maps depend on the beam's reference kinematics (``beta0``, ``gamma0``),
so a :class:`ReferenceParticle` is threaded through every ``Element.matrix`` call.

Internal unit convention (see ``docs/CONVENTIONS.md``): energies and momenta are
stored in **eV** (momentum as ``p0*c`` in eV), lengths in metres. Only the
dimensionless ratios ``beta0``/``gamma0`` enter the transfer matrices, so the eV
choice is a boundary convenience, not a physics commitment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Rest energies (m c^2) in eV — CODATA-rounded, adequate for a teaching simulator.
ELECTRON_MASS_EV: float = 0.51099895069e6
PROTON_MASS_EV: float = 938.27208816e6


@dataclass(frozen=True)
class ReferenceParticle:
    """Immutable reference particle.

    Construct via :meth:`from_total_energy`, :meth:`from_kinetic_energy`,
    :meth:`from_momentum`, or :meth:`from_gamma` rather than the raw fields, so
    the energy specification is unambiguous.
    """

    mass_eV: float
    """Rest energy m c^2 in eV (must be > 0)."""
    total_energy_eV: float
    """Total energy E0 = gamma0 * m c^2 in eV (must be >= mass_eV)."""
    charge: float = 1.0
    """Charge in units of the elementary charge e."""

    def __post_init__(self) -> None:
        if self.mass_eV <= 0:
            raise ValueError(f"mass_eV must be positive, got {self.mass_eV}")
        if self.total_energy_eV < self.mass_eV:
            raise ValueError(
                f"total_energy_eV ({self.total_energy_eV}) must be >= mass_eV "
                f"({self.mass_eV}); a particle cannot have less than its rest energy."
            )

    # --- constructors --------------------------------------------------------
    @classmethod
    def from_total_energy(
        cls, mass_eV: float, total_energy_eV: float, charge: float = 1.0
    ) -> ReferenceParticle:
        return cls(mass_eV=mass_eV, total_energy_eV=total_energy_eV, charge=charge)

    @classmethod
    def from_kinetic_energy(
        cls, mass_eV: float, kinetic_energy_eV: float, charge: float = 1.0
    ) -> ReferenceParticle:
        return cls(mass_eV=mass_eV, total_energy_eV=mass_eV + kinetic_energy_eV, charge=charge)

    @classmethod
    def from_momentum(
        cls, mass_eV: float, momentum_eV: float, charge: float = 1.0
    ) -> ReferenceParticle:
        """``momentum_eV`` is p0*c expressed in eV."""
        total = math.hypot(momentum_eV, mass_eV)
        return cls(mass_eV=mass_eV, total_energy_eV=total, charge=charge)

    @classmethod
    def from_gamma(cls, mass_eV: float, gamma0: float, charge: float = 1.0) -> ReferenceParticle:
        if gamma0 < 1:
            raise ValueError(f"gamma0 must be >= 1, got {gamma0}")
        return cls(mass_eV=mass_eV, total_energy_eV=gamma0 * mass_eV, charge=charge)

    # --- derived kinematics --------------------------------------------------
    @property
    def gamma0(self) -> float:
        """Lorentz factor gamma0 = E0 / (m c^2)."""
        return self.total_energy_eV / self.mass_eV

    @property
    def beta0(self) -> float:
        """Relativistic beta0 = v0 / c = sqrt(1 - 1/gamma0^2)."""
        g = self.gamma0
        return math.sqrt(1.0 - 1.0 / (g * g))

    @property
    def momentum_eV(self) -> float:
        """Reference momentum as p0*c in eV: sqrt(E0^2 - (m c^2)^2)."""
        return math.sqrt(self.total_energy_eV**2 - self.mass_eV**2)

    @property
    def kinetic_energy_eV(self) -> float:
        return self.total_energy_eV - self.mass_eV
