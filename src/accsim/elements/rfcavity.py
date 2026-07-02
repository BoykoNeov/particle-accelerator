"""RF cavity: a thin longitudinal kick that provides synchrotron focusing.

An RF cavity gives each particle an energy kick that depends on its arrival phase
relative to the RF wave. In the accsim state vector (``delta`` is a *momentum*
deviation, see ``docs/CONVENTIONS.md``) the kick is

    Delta delta = (q V / (beta0^2 E0)) * [ sin(phi_s - k_rf * zeta) - sin(phi_s) ],

with

    k_rf = 2*pi*frequency / (beta0 * c)     [1/m]   (RF wavenumber),
    phi_s                                    [rad]   (synchronous phase),
    q = ref.charge, V = voltage [V], E0 = ref.total_energy_eV [eV].

The energy-to-``delta`` factor is ``1/(beta0^2 E0)`` (**not** ``1/E0``): with the
momentum variable ``delta``, ``dE = beta0^2 E0 * delta`` at the reference, the same
``beta0^2`` that distinguishes ``R56 = L/gamma0^2`` from the energy-variable form.

**Phase convention matches xtrack's ``Cavity``**: xtrack applies
``energy_kick = q V sin(lag_rad - (2*pi f / c) * zeta / beta0)`` — i.e. the same
``phi = phi_s - k_rf zeta`` used here, with ``phi_s`` playing the role of xtrack's
``lag`` (xtrack takes ``lag`` in *degrees*; accsim uses radians, consistent with
the rest of the codebase). Cross-checked in ``tests/reference/``.

**Stationary bucket only (Stage 3).** The synchronous particle (``zeta = 0``)
receives the constant offset ``-sin(phi_s)``; a stationary (non-accelerating)
bucket takes ``phi_s = 0`` below transition and ``phi_s = pi`` above it, so the
synchronous particle gets zero net kick and the small-amplitude motion is stable
(``Qs^2 = -(h eta q V cos phi_s)/(2 pi beta0^2 E0) > 0``). The acceleration ramp
and the ``q V sin(phi_s)`` energy gain per turn are **Stage 5**, not built here;
the ``voltage``/``frequency``/``phi_s`` interface is forward-compatible with it.

The **linear** map (:meth:`matrix`) is the small-amplitude limit — a longitudinal
shear ``R65 = d(Delta delta)/d(zeta)|_0 = -(q V k_rf cos phi_s)/(beta0^2 E0)`` —
on which the synchrotron tune :func:`accsim.synchrotron_tune` is built. The full
nonlinear ``sin`` kick (the pendulum whose separatrix is the RF bucket) is the
tracking map added with longitudinal tracking.
"""

from __future__ import annotations

import math

import numpy as np

from ..coords import DELTA, DIM, ZETA
from ..reference import CLIGHT, ReferenceParticle
from .element import Element


class RFCavity(Element):
    r"""A thin RF cavity of peak ``voltage`` [V] at ``frequency`` [Hz].

    ``phi_s`` [rad] is the synchronous phase (xtrack's ``lag``, but in radians).
    Zero length: only ``delta`` is affected, via the longitudinal shear ``R65``.
    """

    def __init__(
        self,
        voltage: float,
        frequency: float,
        phi_s: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(0.0, name=name)
        if frequency < 0:
            raise ValueError(f"frequency must be >= 0, got {frequency}")
        self.voltage = float(voltage)
        self.frequency = float(frequency)
        self.phi_s = float(phi_s)

    def k_rf(self, ref: ReferenceParticle) -> float:
        """RF wavenumber ``k_rf = 2*pi*frequency / (beta0 * c)`` [1/m]."""
        return 2.0 * math.pi * self.frequency / (ref.beta0 * CLIGHT)

    def slope(self, ref: ReferenceParticle) -> float:
        """Small-amplitude longitudinal focusing ``R65 = d(Delta delta)/d(zeta)|_0``.

        ``R65 = -(q V k_rf cos phi_s) / (beta0^2 E0)`` [1/m]. Negative for a
        below-transition stationary bucket (``phi_s = 0``), which combines with the
        (also negative) slip factor into a positive ``Qs^2``.
        """
        return -(ref.charge * self.voltage * self.k_rf(ref) * math.cos(self.phi_s)) / (
            ref.beta0**2 * ref.total_energy_eV
        )

    def energy_kick_delta(self, zeta: float, ref: ReferenceParticle) -> float:
        """Full nonlinear momentum kick ``Delta delta`` at longitudinal ``zeta``.

        ``(q V / (beta0^2 E0)) * [sin(phi_s - k_rf zeta) - sin(phi_s)]`` — the
        ``sin`` (not its linearisation) that gives the RF bucket its separatrix.
        Used by the nonlinear tracker; :meth:`matrix` is its ``zeta -> 0`` slope.
        """
        k = self.k_rf(ref)
        amp = ref.charge * self.voltage / (ref.beta0**2 * ref.total_energy_eV)
        return amp * (math.sin(self.phi_s - k * zeta) - math.sin(self.phi_s))

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        # Thin longitudinal shear: identity except the small-amplitude R65 kick.
        M = np.eye(DIM)
        M[DELTA, ZETA] = self.slope(ref)
        return M

    def track(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        # Full nonlinear map: a thin momentum kick with the exact ``sin`` (not its
        # R65 linearisation). Only ``delta`` changes; the kick depends on ``zeta``.
        # This delta-kick is a symplectic shear -- composed with the linear arc it
        # forms the pendulum whose separatrix is the RF bucket.
        out = np.array(state, dtype=float, copy=True)
        out[DELTA] += self.energy_kick_delta(float(out[ZETA]), ref)
        return out

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return (
            f"RFCavity(voltage={self.voltage}, frequency={self.frequency}, "
            f"phi_s={self.phi_s}{name})"
        )
