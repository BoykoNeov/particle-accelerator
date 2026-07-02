"""Acceleration: energy ramp, energy gain per turn, adiabatic damping (Stage 5).

Stage 3 built the RF cavity and the *stationary* bucket (``sin phi_s = 0``: no net
energy gain). Stage 5 turns the ramp on. The physics is already in the Stage-3
cavity kick

    Delta delta = (q V / beta0^2 E0) [ sin(phi_s - k_rf zeta) - sin(phi_s) ],

whose ``- sin(phi_s)`` term is exactly the energy the *reference* particle absorbs
each turn. A synchronous particle (``zeta = 0``) therefore gets zero net kick and
stays at ``delta = 0`` **by construction**, while the reference (design) energy
climbs by

    Delta E_s = sum_cav  q V sin(phi_s)                     [eV]   (per turn).

Two things must then follow the ramp, and this module supplies them:

1. **The reference energy program** ``E0(n) = E0(0) + n * Delta E_s`` — a fresh
   immutable :class:`~accsim.reference.ReferenceParticle` is built each turn from
   the ramped ``E0`` (the lattice's own ``ref`` is never mutated). Because the beam
   energy is constant around the ring *except* across the cavity, using the
   turn-entry ``ref`` for that turn's arc is exact when the cavity sits at the end
   of the lattice (as in the standard ring), and correct to ``O(Delta E_s / E0)``
   per turn otherwise — negligible for a real ramp (keV gain on GeV energy).

2. **Adiabatic damping.** The normalised transverse momenta ``px = Px / P0`` and
   the momentum deviation ``delta`` are referenced to ``P0``; when ``P0`` grows to
   ``P0'`` while the *physical* ``Px, Py`` are untouched by the (longitudinal) RF,
   re-referencing multiplies them by ``r = P0 / P0' < 1``. Derived from the
   coordinate definitions (see :func:`accelerate`), the per-turn map is

       (px, py, delta)  ->  r * (px, py, delta_after_kick),   r = P0(n) / P0(n+1).

   Position ``(x, y, zeta)`` is a spatial coordinate, not normalised by ``P0``, so
   it is not rescaled at the thin cavity; the betatron/synchrotron motion then
   converts the momentum damping into overall amplitude damping over a period,
   conserving the adiabatic invariant ``P0 * J`` (canonical action). This is *not*
   a symplecticity violation — the geometric emittance genuinely shrinks.

**Scope (Stage 5).** Energy ramp, energy gain per turn, adiabatic damping, and the
synchronous-phase relation. Constant magnetic optics are assumed (``k1``/bend
angles held fixed — i.e. the magnets ramp with the beam energy, the physical
"tracking" ramp), so the transverse Twiss is energy-invariant. The moving-bucket
*acceptance* (bucket area vs. ``phi_s``), beam loading, and transition crossing
are out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .coords import DELTA, DIM, PX, PY
from .elements.rfcavity import RFCavity
from .lattice import Lattice
from .reference import ReferenceParticle
from .tracking import Particle


def _cavities(lattice: Lattice) -> list[RFCavity]:
    cavities = [elem for elem in lattice.elements if isinstance(elem, RFCavity)]
    if not cavities:
        raise ValueError(
            "acceleration requires at least one RFCavity in the lattice; without "
            "RF there is no energy gain."
        )
    return cavities


def energy_gain_per_turn(lattice: Lattice) -> float:
    """Synchronous energy gain per turn ``Delta E_s = sum_cav q V sin(phi_s)`` [eV].

    The energy the reference (synchronous) particle gains each revolution — the
    Stage-5 acceptance quantity. ``q = ref.charge`` (e-units) and ``V`` in volts,
    so ``q V`` is in eV. Sums over every cavity, so a multi-cavity ring adds the
    contributions (cavities may differ in voltage or phase). Zero for a stationary
    bucket (``phi_s = 0`` or ``pi``), which recovers the Stage-3 model.
    """
    q = lattice.ref.charge
    return q * sum(cav.voltage * math.sin(cav.phi_s) for cav in _cavities(lattice))


def synchronous_phase(
    voltage: float,
    energy_gain_per_turn: float,
    above_transition: bool,
    charge: float = 1.0,
) -> float:
    """Stable synchronous phase ``phi_s`` for a target energy gain per turn.

    Inverts ``Delta E_s = q V sin(phi_s)`` for the **stable** root. Requiring both
    net acceleration (``sin phi_s > 0``) and a stable bucket fixes the branch: the
    small-amplitude stability condition ``Qs^2 = -(h eta q V cos phi_s)/(2 pi
    beta0^2 E0) > 0`` needs ``eta cos phi_s < 0``, so

    - **below transition** (``eta < 0``): ``cos phi_s > 0`` -> ``phi_s in (0, pi/2)``
      -> ``phi_s = asin(s)``;
    - **above transition** (``eta > 0``): ``cos phi_s < 0`` -> ``phi_s in (pi/2, pi)``
      -> ``phi_s = pi - asin(s)``,

    with ``s = Delta E_s / (q V)``. At zero gain this returns ``0`` below transition
    and ``pi`` above — exactly the Stage-3 stationary phases. ``eta``'s sign is a
    lattice property (:func:`accsim.slip_factor`), independent of the cavity phase,
    so it can be evaluated before the cavities are built.
    """
    qv = charge * voltage
    if qv <= 0:
        raise ValueError(f"charge*voltage must be > 0, got {qv}")
    s = energy_gain_per_turn / qv
    if not -1.0 <= s <= 1.0:
        raise ValueError(
            f"energy gain per turn ({energy_gain_per_turn}) exceeds the available "
            f"q V ({qv}); |sin phi_s| = {abs(s)} > 1, no synchronous phase exists."
        )
    base = math.asin(s)  # in [-pi/2, pi/2]
    return (math.pi - base) if above_transition else base


@dataclass
class RampResult:
    """Outcome of an acceleration ramp (see :func:`accelerate`).

    - ``states``    — ``(n_turns + 1, 6)`` phase-space states, including the initial
      one, in the accsim coordinates ``(x, px, y, py, zeta, delta)``. ``px``, ``py``
      and ``delta`` are referenced to the *turn's* reference momentum ``P0(n)``.
    - ``energy_eV`` — ``(n_turns + 1,)`` reference **total energy** program
      ``E0(n) = E0(0) + n * Delta E_s`` [eV]. The energy gain per turn is the
      constant first difference.
    - ``mass_eV``   — the (fixed) rest energy, so momenta/kinematics can be derived.
    """

    states: np.ndarray
    energy_eV: np.ndarray
    mass_eV: float

    @property
    def n_turns(self) -> int:
        return self.states.shape[0] - 1

    @property
    def momentum_eV(self) -> np.ndarray:
        """Reference momentum program ``P0(n) c`` [eV] = ``sqrt(E0^2 - m^2)``."""
        return np.sqrt(self.energy_eV**2 - self.mass_eV**2)

    @property
    def gamma0(self) -> np.ndarray:
        """Reference Lorentz factor program ``gamma0(n) = E0(n) / m``."""
        return self.energy_eV / self.mass_eV


def accelerate(lattice: Lattice, particle: Particle, n_turns: int) -> RampResult:
    r"""Track ``particle`` for ``n_turns`` turns with the reference energy ramping.

    Each turn: push the state element-by-element through the lattice at the current
    reference (so the cavity's full ``sin`` kick acts — Stage-3 nonlinear tracking),
    then advance the reference energy by :func:`energy_gain_per_turn` and apply the
    adiabatic-damping re-referencing.

    **Adiabatic damping factor, derived.** With ``delta = (P - P0)/P0`` and
    ``px = Px/P0``: after the cavity (at fixed ``P0``) a particle sits at
    ``delta_A = delta + A[sin phi - sin phi_s]`` relative to the old ``P0``, i.e. its
    momentum is ``P = P0 (1 + delta_A)``. Re-referencing to ``P0' = P0 + Delta P_s``
    (the reference's own gain) gives

        delta' = (P - P0')/P0' = (P0/P0') (1 + delta_A) - 1
               = (P0/P0') delta_A - Delta P_s/P0'   -> to the working order,
               = (P0/P0') * (delta + A[sin phi - sin phi_s]),

    because ``Delta P_s/P0' = (P0/P0') * Delta P_s/P0`` and ``A sin phi_s`` is exactly
    ``Delta P_s/P0``, so the reference-gain terms cancel. The physical ``Px, Py`` are
    untouched by the longitudinal kick, so ``px' = Px/P0' = (P0/P0') px`` and likewise
    ``py``. Hence the single factor ``r = P0/P0'`` multiplies ``(px, py, delta)`` once
    per turn; ``r = 1`` when there is no gain, recovering Stage-3 tracking exactly.

    Returns a :class:`RampResult`. A synchronous particle ``(0, 0, 0, 0, 0, 0)`` stays
    there while ``energy_eV`` climbs linearly.
    """
    if n_turns < 0:
        raise ValueError(f"n_turns must be >= 0, got {n_turns}")
    ref0 = lattice.ref
    mass = ref0.mass_eV
    charge = ref0.charge
    dE = energy_gain_per_turn(lattice)  # constant reference gain per turn [eV]

    states = np.empty((n_turns + 1, DIM))
    energy = np.empty(n_turns + 1)
    states[0] = particle.state
    energy[0] = ref0.total_energy_eV

    state = particle.state.astype(float, copy=True)
    ref = ref0
    for turn in range(1, n_turns + 1):
        # 1. Track one turn at the turn-entry reference (nonlinear: sin kick acts).
        for elem in lattice.elements:
            state = elem.track(state, ref)
        # 2. Ramp the reference energy and re-reference (adiabatic damping).
        e_new = ref.total_energy_eV + dE
        ref_new = ReferenceParticle.from_total_energy(mass, e_new, charge)
        r = ref.momentum_eV / ref_new.momentum_eV  # = P0(n)/P0(n+1) <= 1
        state[PX] *= r
        state[PY] *= r
        state[DELTA] *= r
        ref = ref_new
        states[turn] = state
        energy[turn] = e_new

    return RampResult(states, energy, mass)
