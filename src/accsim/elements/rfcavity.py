"""RF cavity: a thin **energy** kick that provides synchrotron focusing.

An RF cavity gives each particle an energy kick that depends on its arrival phase
relative to the RF wave. That is a statement about ``E``, not about ``delta``, and
this element applies it as one:

    Delta E = q V [ sin(phi_s - k_rf * zeta) - sin(phi_s) ]                [eV]

with

    k_rf = 2*pi*frequency / (beta0 * c)     [1/m]   (RF wavenumber),
    phi_s                                    [rad]   (synchronous phase),
    q = ref.charge, V = voltage [V], E0 = ref.total_energy_eV [eV].

**The energy kick expressed in accsim's variables.** ``p_zeta = (E - E0)/(beta0^2 E0)``
is the coordinate canonically conjugate to ``zeta`` (see :mod:`accsim.symplectic`), so
in ``p_zeta`` the kick is *linear in the energy* and reads

    Delta p_zeta = (q V / (beta0^2 E0)) [ sin(phi_s - k_rf zeta) - sin(phi_s) ],

which is :meth:`RFCavity.energy_kick_pzeta`. The state carries the *momentum*
deviation ``delta = Delta p / p0``, and ``delta`` is a **nonlinear** function of the
energy, so the kick is applied by converting through it:

    (1 + delta')^2 = (1 + delta)^2 + Delta ptau (2 E/(p0 c) + Delta ptau),
    Delta ptau = beta0 * Delta p_zeta = Delta E / (p0 c),
    E/(p0 c)   = sqrt((1 + delta)^2 + (m/p0)^2).

:meth:`_track_body` evaluates that as an *increment* on ``delta`` so that nothing
cancels and a zero kick is exactly the identity — see its comments.

**Why not just add the kick to ``delta``.** Until P2 (iii) it was added straight to
``delta``, with the amplitude above. That is right at ``delta = 0`` — ``d delta/d p_zeta
= 1`` there — and wrong off-momentum by the curvature ``d^2 delta/d p_zeta^2 =
-1/gamma0^2``. As a second-order map the difference is exactly

    T[delta, zeta, delta] = T[delta, delta, zeta] = -R65 / (2 gamma0^2),

which is ``1/(2 gamma0^2)`` of the cavity's own slope: ``1.25e-3`` of it on a
``gamma0 = 20`` proton ring, and invisible on a high-energy electron ring. It is also
the whole of the difference at first order in ``delta``, so the *linearised* form is
**not canonically symplectic**: it is a shear in ``(zeta, delta)``, which is not a
conjugate pair, while the energy kick is a shear in ``(zeta, p_zeta)``, which is.
See ``docs/CONVENTIONS.md`` -> *The cavity gives energy, not momentum*.

The small-amplitude **bucket** model in :mod:`accsim.longitudinal` still uses the
``p_zeta`` amplitude above as if it were a ``delta`` kick. That is not this bug
repeated: the bucket Hamiltonian is a small-amplitude construction to begin with, and
the two agree to the order it is written at.

**Phase convention matches xtrack's ``Cavity`` — for a positive charge only.**
xtrack applies ``energy_kick = q V sin(phase + lag_rad - (2 pi f / c) zeta / beta0)``,
the same ``phi = phi_s - k_rf zeta`` used here, and applies it through
``LocalParticle_add_to_energy(..., pz_only=1)`` — the same energy kick at fixed
``px``/``py``, so accsim's map and xtrack's are now the *same map*, gated by tracking
in ``tests/reference/test_rf_energy_kick_xtrack.py``. **But its** ``q`` **is**
``fabs(q0) * charge_ratio`` (``beam_elements/elements_src/track_rf.h``), the
*absolute* charge, where the expression above uses the signed ``ref.charge``. For a
**negative** particle the two cavities are therefore exact negatives of each other,
and the correspondence is ``phase = phi_s + pi`` (equivalently
``lag = degrees(phi_s) + 180``), not ``phase = phi_s``. Neither convention is wrong:
this one is the physical ``q E . v``, xtrack's makes ``lag`` mean the same thing for
every species. Getting it backwards makes an electron ring come out longitudinally
*unstable* rather than merely shifted — see ``docs/CONVENTIONS.md`` -> *RF cavity /
synchrotron tune* and ``tests/reference/test_spin_sidebands_xtrack.py``.

**Synchronous phase and acceleration.** The synchronous particle (``zeta = 0``)
receives the constant offset ``-sin(phi_s)``, so its net kick is zero *in the
frame that follows the reference energy* — and exactly zero, at any ``delta``, since
a vanishing energy kick leaves ``delta`` untouched bit for bit. A stationary
(non-accelerating) bucket has ``sin phi_s = 0``, and **which** of the two roots is the
stable one depends on the **sign of the charge**: stability needs
``eta q cos phi_s < 0``, so a *proton* takes ``phi_s = 0`` below transition and
``phi_s = pi`` above it, and an *electron* — for which ``q < 0`` — takes them the
other way round.

For ``sin(phi_s) != 0`` the same ``[sin(phi_s - k_rf zeta) - sin(phi_s)]`` kick is the
**accelerating** kick measured relative to a *ramping* reference: the synchronous
particle still gets zero net Delta E and stays at ``delta = 0``, while the
reference energy climbs by ``q V sin(phi_s)`` per turn.
That ramp — plus the accompanying adiabatic damping — is driven by
:func:`accsim.accelerate` (Stage 5); this element's ``matrix``/``slope``/
``energy_kick_pzeta`` are unchanged. The small-amplitude motion is stable when
``Qs^2 = -(h eta q V cos phi_s)/(2 pi beta0^2 E0) > 0``.

The **linear** map (:meth:`matrix`) is the small-amplitude limit — a longitudinal
shear ``R65 = d(Delta delta)/d(zeta)|_0 = -(q V k_rf cos phi_s)/(beta0^2 E0)`` —
on which the synchrotron tune :func:`accsim.synchrotron_tune` is built. It is
**unchanged** by the energy-kick form: ``d delta/d p_zeta = 1`` at the origin, and the
synchronous particle sits at ``Delta p_zeta = 0``, so both ``R65`` and ``R66`` are
what they were. The full nonlinear ``sin`` kick (the pendulum whose separatrix is the
RF bucket) is the tracking map added with longitudinal tracking.
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

    **``track`` returns NaN for a particle the kick would drive below rest energy.**
    The map applies an *energy* kick, and ``(1 + delta')^2 = (E/(p0 c) + Delta ptau)^2 -
    (m/p0)^2`` has no real root once the total energy would fall under ``m c^2``. On axis
    that edge is exactly ``Delta p_zeta = -(gamma0 - 1)/(beta0^2 gamma0)``, i.e.
    ``Delta E = -(gamma0 - 1) m c^2`` -- the whole kinetic energy. It is a reported
    condition, the same contract :class:`~accsim.elements.drift.Drift` and
    :class:`~accsim.elements.dipole.Dipole` carry for their exact maps, and in practice it
    is reached only by a trajectory already far outside any RF bucket. The pre-P2 (iii)
    momentum kick had no such domain and did not merely lose accuracy near it: it would
    report a finite ``delta`` below ``-1``, a negative total momentum.
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

    def energy_kick_pzeta(self, zeta: float | np.ndarray, ref: ReferenceParticle) -> float:
        """Full nonlinear energy kick ``Delta p_zeta`` at longitudinal ``zeta``.

        ``(q V / (beta0^2 E0)) * [sin(phi_s - k_rf zeta) - sin(phi_s)]`` — the
        ``sin`` (not its linearisation) that gives the RF bucket its separatrix.
        ``p_zeta = (E - E0)/(beta0^2 E0)`` is linear in the energy, so this *is* the
        energy kick ``q V [sin ... ]`` divided by ``beta0^2 E0``, and it is the
        quantity the cavity actually delivers. :meth:`_track_body` converts it into
        the state's ``delta``, which is a nonlinear function of it; only at the
        origin, where ``d delta / d p_zeta = 1``, are the two the same number, which
        is why :meth:`matrix` reads this slope unchanged.

        Named ``..._pzeta`` since P2 (iii); it was ``energy_kick_delta`` while the
        kick was added straight to ``delta``, which is the thing that changed.

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
        # Full nonlinear map: a thin ENERGY kick with the exact ``sin`` (not its R65
        # linearisation), converted into ``delta``. Only ``delta`` changes -- ``px``
        # and ``py`` are momenta normalised to the *reference* P0, which the cavity
        # does not move, so they are untouched (xtrack's ``pz_only=1``).
        # A shear in ``(zeta, p_zeta)``, the canonically conjugate pair -- composed
        # with the linear arc it forms the pendulum whose separatrix is the RF bucket.
        # ``state`` may be a (6,) vector or a (6, n) bunch; the zeta-dependent kick
        # is elementwise either way.
        out = np.array(state, dtype=float, copy=True)
        one_p_d = 1.0 + out[DELTA]
        # ``Delta ptau = Delta E/(p0 c) = beta0 Delta p_zeta``, and
        # ``E/(p0 c) = sqrt((1+delta)^2 + (m/p0)^2)`` -- by hypot, so nothing cancels.
        dptau = ref.beta0 * self.energy_kick_pzeta(out[ZETA], ref)
        e_over_p0c = np.hypot(one_p_d, ref.mass_eV / ref.momentum_eV)
        # ``(E + dE)^2 - m^2 = P'^2`` gives the exact new momentum as a *difference*
        # of squares, ``s = (1+delta')^2 - (1+delta)^2 = dptau (2 E/(p0c) + dptau)``,
        # which is then divided by ``(1+delta') + (1+delta)`` rather than square-rooted
        # and subtracted. So ``delta`` keeps full relative precision, and a zero kick
        # (V = 0, or the synchronous particle anywhere in a bunch) adds exactly 0.0
        # -- the identity bit for bit, with no special case to get wrong.
        s = dptau * (2.0 * e_over_p0c + dptau)
        # NaN for a particle the kick would drive below rest energy is a *documented*
        # return value (class docstring): ``(1+delta')^2 < 0`` has no real root, and the
        # edge is exactly ``Delta E = -(gamma0 - 1) m c^2``, the whole kinetic energy.
        # Same contract as ``Drift``/``Dipole``'s exact maps, so the same treatment --
        # the sqrt's warning is noise rather than a signal and is silenced here only,
        # never the value itself, which still propagates as NaN.
        with np.errstate(invalid="ignore"):
            out[DELTA] += s / (np.sqrt(one_p_d * one_p_d + s) + one_p_d)
        return out

    def __repr__(self) -> str:
        return (
            f"RFCavity(voltage={self.voltage}, frequency={self.frequency}, "
            f"phi_s={self.phi_s}{self._repr_tail()})"
        )
