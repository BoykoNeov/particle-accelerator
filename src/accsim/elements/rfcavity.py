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

**Phase convention matches xtrack's ``Cavity`` — for a positive charge only.**
xtrack applies ``energy_kick = q V sin(phase + lag_rad - (2 pi f / c) zeta / beta0)``,
the same ``phi = phi_s - k_rf zeta`` used here — **but its** ``q`` **is**
``fabs(q0) * charge_ratio`` (``beam_elements/elements_src/track_rf.h``), the
*absolute* charge, where the expression above uses the signed ``ref.charge``. For a
**negative** particle the two cavities are therefore exact negatives of each other,
and the correspondence is ``phase = phi_s + pi`` (equivalently
``lag = degrees(phi_s) + 180``), not ``phase = phi_s``. Neither convention is wrong:
this one is the physical ``q E . v``, xtrack's makes ``lag`` mean the same thing for
every species. Getting it backwards makes an electron ring come out longitudinally
*unstable* rather than merely shifted — see ``docs/CONVENTIONS.md`` → *RF cavity /
synchrotron tune* and ``tests/reference/test_spin_sidebands_xtrack.py``.

**Synchronous phase and acceleration.** The synchronous particle (``zeta = 0``)
receives the constant offset ``-sin(phi_s)``, so its net kick is zero *in the
frame that follows the reference energy*. A stationary (non-accelerating) bucket
has ``sin phi_s = 0``, and **which** of the two roots is the stable one depends on
the **sign of the charge**: stability needs ``eta q cos phi_s < 0``, so a *proton*
takes ``phi_s = 0`` below transition and ``phi_s = pi`` above it, and an *electron*
— for which ``q < 0`` — takes them the other way round.

For ``sin(phi_s) != 0`` the same ``[sin(phi_s - k_rf zeta) - sin(phi_s)]`` kick is the
**accelerating** kick measured relative to a *ramping* reference: the synchronous
particle still gets zero net Delta delta and stays at ``delta = 0``, while the
reference energy climbs by ``q V sin(phi_s)`` per turn.
That ramp — plus the accompanying adiabatic damping — is driven by
:func:`accsim.accelerate` (Stage 5); this element's ``matrix``/``slope``/
``energy_kick_delta`` are unchanged. The small-amplitude motion is stable when
``Qs^2 = -(h eta q V cos phi_s)/(2 pi beta0^2 E0) > 0``.

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

    ``phi_s`` [rad] is the synchronous phase (xtrack's ``phase``, plus ``pi`` for a
    negative charge -- see the module docstring).
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

    @classmethod
    def from_harmonic(
        cls,
        voltage: float,
        harmonic: int,
        circumference: float,
        ref: ReferenceParticle,
        phi_s: float = 0.0,
        name: str | None = None,
    ) -> RFCavity:
        """Build a cavity from the **harmonic number** ``h`` (Stage-5 interface).

        The harmonic number is the (integer) number of RF wavelengths that fit in
        one revolution: ``h = f * C / (beta0 c)``, so the frequency is

            frequency = harmonic * beta0 * c / circumference    [Hz],

        which makes ``k_rf * C = 2*pi*h`` exactly. ``circumference`` is the ring
        length ``C`` [m] and ``ref`` fixes ``beta0``; ``harmonic`` must be a
        positive integer. This is the natural way to specify a ring cavity, where
        ``h`` (not the raw frequency) is the design quantity.
        """
        if harmonic <= 0:
            raise ValueError(f"harmonic number must be a positive integer, got {harmonic}")
        if circumference <= 0:
            raise ValueError(f"circumference must be > 0, got {circumference}")
        frequency = harmonic * ref.beta0 * CLIGHT / circumference
        return cls(voltage=voltage, frequency=frequency, phi_s=phi_s, name=name)

    def k_rf(self, ref: ReferenceParticle) -> float:
        """RF wavenumber ``k_rf = 2*pi*frequency / (beta0 * c)`` [1/m]."""
        return 2.0 * math.pi * self.frequency / (ref.beta0 * CLIGHT)

    def harmonic_number(self, ref: ReferenceParticle, circumference: float) -> float:
        """Harmonic number ``h = f C / (beta0 c) = k_rf C / (2 pi)`` for this ring.

        The inverse of :meth:`from_harmonic`. Returns a float; for a physical ring
        cavity it should be (very close to) an integer.
        """
        return self.frequency * circumference / (ref.beta0 * CLIGHT)

    def slope(self, ref: ReferenceParticle) -> float:
        """Small-amplitude longitudinal focusing ``R65 = d(Delta delta)/d(zeta)|_0``.

        ``R65 = -(q V k_rf cos phi_s) / (beta0^2 E0)`` [1/m]. Its sign carries the
        charge's: for a **proton** at ``phi_s = 0`` it is negative, combining with the
        (also negative) below-transition slip factor into a positive ``Qs^2``; for an
        **electron** at the same phase it is positive, and pairs with a positive
        above-transition slip factor instead. See the module docstring.
        """
        return -(ref.charge * self.voltage * self.k_rf(ref) * math.cos(self.phi_s)) / (
            ref.beta0**2 * ref.total_energy_eV
        )

    def energy_kick_delta(self, zeta: float | np.ndarray, ref: ReferenceParticle) -> float:
        """Full nonlinear momentum kick ``Delta delta`` at longitudinal ``zeta``.

        ``(q V / (beta0^2 E0)) * [sin(phi_s - k_rf zeta) - sin(phi_s)]`` — the
        ``sin`` (not its linearisation) that gives the RF bucket its separatrix.
        Used by the nonlinear tracker; :meth:`matrix` is its ``zeta -> 0`` slope.

        ``zeta`` may be an array (one entry per particle), in which case the kick
        is returned elementwise — the bunch path of the nonlinear tracker.
        """
        k = self.k_rf(ref)
        amp = ref.charge * self.voltage / (ref.beta0**2 * ref.total_energy_eV)
        return amp * (np.sin(self.phi_s - k * np.asarray(zeta)) - math.sin(self.phi_s))

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        # Thin longitudinal shear: identity except the small-amplitude R65 kick.
        M = np.eye(DIM)
        M[DELTA, ZETA] = self.slope(ref)
        return M

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        # Full nonlinear map: a thin momentum kick with the exact ``sin`` (not its
        # R65 linearisation). Only ``delta`` changes; the kick depends on ``zeta``.
        # This delta-kick is a symplectic shear -- composed with the linear arc it
        # forms the pendulum whose separatrix is the RF bucket.
        # ``state`` may be a (6,) vector or a (6, n) bunch; the zeta-dependent kick
        # is elementwise either way.
        out = np.array(state, dtype=float, copy=True)
        out[DELTA] += self.energy_kick_delta(out[ZETA], ref)
        return out

    def __repr__(self) -> str:
        return (
            f"RFCavity(voltage={self.voltage}, frequency={self.frequency}, "
            f"phi_s={self.phi_s}{self._repr_tail()})"
        )
