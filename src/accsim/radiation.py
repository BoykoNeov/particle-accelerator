r"""Synchrotron radiation & radiation damping (Stage 7) — the radiation integrals.

A relativistic charged beam on a curved orbit radiates. That radiation both
**damps** the phase-space oscillations (the average energy loss is restored on-axis
by the RF, but the transverse/longitudinal *amplitudes* shrink) and **excites** them
(the loss comes in discrete quanta, a random walk). The equilibrium between the two
sets the beam's emittance and energy spread. All of it is captured by five lattice
integrals ``I1..I5`` (Sands, *The Physics of Electron Storage Rings*, SLAC-121);
this module computes them and the damping quantities they feed. Commit 1 delivers
``I1..I4`` + energy loss + partition numbers + damping times; ``I5`` and the
equilibrium emittance/energy spread follow (they carry the one integral with no
clean within-baseline closed form — see :func:`radiation_integrals`).

**This is baseline core physics** (numpy only), not a gated addon: it needs no
external tool, just the lattice already in hand.

**Units.** SI throughout the accelerator core: energies in **eV**, lengths in
**m**, times in **s** (see ``docs/CONVENTIONS.md`` → *Units*). ``C_gamma`` is then
in ``m/eV^3`` and ``U0`` comes out in **eV**; ``C_q`` is in **m**.

**Scope.** Dipoles are **pure sector bends** (no combined-function gradient, no
pole-face edge focusing — both out of Stage-1 scope), so ``I4 = ∮ D_x h^3 ds`` (the
``2 k1`` body term and the ``-D_x h^2 tan(edge)`` face term both vanish). The lattice
is assumed a **periodic ring** (``closed_twiss`` enforces stability); the
*isomagnetic* closed forms additionally assume total bend ``2*pi``. Vertical bending
and betatron coupling are absent, so ``J_y = 1`` exactly and the equilibrium
vertical emittance is ~0 (a flat-lattice statement — real rings get ``eps_y`` from
coupling/vertical dispersion, out of scope).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lattice import Lattice
from .reference import CLIGHT, ReferenceParticle
from .twiss import _blocks, _dispersive_kick, _propagate_block, _transverse_4d, closed_twiss

# CODATA hbar*c = 197.3269804 MeV*fm = 1.9732698045e-7 eV*m. The one physical
# constant the radiation module adds beyond the reference particle's own.
HBAR_C_EV_M: float = 1.9732698045e-7


@dataclass(frozen=True)
class RadiationIntegrals:
    r"""The synchrotron-radiation lattice integrals (Sands / Chao conventions).

    - ``i1 = ∮ D_x h ds``           — links to momentum compaction ``alpha_c = I1/C``.
    - ``i2 = ∮ h^2 ds``             — sets the energy loss ``U0`` (and radiated power).
    - ``i3 = ∮ |h|^3 ds``           — sets the quantum-excitation / energy spread.
    - ``i4 = ∮ D_x h^3 ds``         — the damping-partition redistribution term.
    - ``i5 = ∮ curlyH |h|^3 ds``    — sets the equilibrium horizontal emittance, with
      the dispersion invariant ``curlyH = gamma_x D_x^2 + 2 alpha_x D_x D_x' +
      beta_x D_x'^2``.

    ``h = 1/rho`` is the signed orbit curvature; ``i3``/``i5`` use ``|h|^3``
    (excitation is bend-sign-blind) while ``i4`` keeps the sign of ``h^3``.
    """

    i1: float
    i2: float
    i3: float
    i4: float
    i5: float


def radiation_constant_cgamma(ref: ReferenceParticle) -> float:
    r"""``C_gamma = 4*pi*r0 / (3 (m c^2)^3)`` [m/eV^3] for the reference species.

    Computed from the particle's own classical radius and rest energy, so it is
    correct for any species (``r0 ∝ 1/m`` ⇒ ``C_gamma ∝ 1/m^3``). For the electron
    this is the familiar ``8.846e-5 m/GeV^3``.
    """
    return 4.0 * math.pi * ref.classical_radius_m / (3.0 * ref.mass_eV**3)


def quantum_constant_cq(ref: ReferenceParticle) -> float:
    r"""``C_q = 55/(32 sqrt3) * hbar c / (m c^2)`` [m] for the reference species.

    The quantum-excitation constant; ``55/(32 sqrt3)`` is the ratio of moments of the
    synchrotron-radiation spectrum (Sands). For the electron, ``3.832e-13 m``.
    """
    return 55.0 / (32.0 * math.sqrt(3.0)) * HBAR_C_EV_M / ref.mass_eV


def _curly_h(beta: float, alpha: float, dx: float, dpx: float) -> float:
    r"""The dispersion invariant ``curlyH = gamma D_x^2 + 2 alpha D_x D_x' + beta D_x'^2``."""
    gamma = (1.0 + alpha * alpha) / beta
    return gamma * dx * dx + 2.0 * alpha * dx * dpx + beta * dpx * dpx


def radiation_integrals(lattice: Lattice, slices: int = 64) -> RadiationIntegrals:
    r"""Compute ``I1..I5`` for a periodic ``lattice``.

    Only bending magnets contribute (``h = 0`` elsewhere). Inside each thick sector
    dipole the matched dispersion ``D_x(s)`` **and** the beta functions ``beta_x,
    alpha_x`` are co-transported by ``slices``-fold trapezoidal sub-stepping of the
    sub-bend map (the dispersion machinery of
    :func:`accsim.twiss.momentum_compaction` plus the ``beta`` transport of
    :func:`accsim.twiss.natural_chromaticity`). ``h`` is constant across the body, so
    ``∮ D_x h ds = h ∮ D_x ds`` and ``∮ D_x h^3 ds = h^3 ∮ D_x ds`` reuse one accumulated
    ``∮ D_x ds``; ``I5 = |h|^3 ∮ curlyH ds`` needs ``curlyH`` re-evaluated per sub-slice
    from the local ``beta_x, alpha_x, D_x, D_x'``. The ``h``-only pieces ``∮ h^2 ds`` /
    ``∮ |h|^3 ds`` are ``h^2 L`` / ``|h|^3 L`` per dipole.

    ``I1 == alpha_c * C`` cross-checks the dispersion transport within the baseline;
    ``I5`` (curly-``H``, needing the co-transported ``beta``) has no clean within-baseline
    absolute check, so it is gated by energy-scaling (``eps_x ∝ gamma^2``) + xtrack
    (``tests/analytic/test_radiation.py``, ``tests/reference/``).
    """
    from .elements.dipole import Dipole

    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    disp = np.array([tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py])
    i1 = i2 = i3 = i4 = i5 = 0.0
    for elem in lattice.elements:
        M = elem.matrix(lattice.ref)
        if isinstance(elem, Dipole) and elem.angle != 0.0 and elem.length > 0.0:
            h = elem.curvature  # 1/rho, signed
            ds = elem.length / slices
            sub = Dipole(ds, h * ds).matrix(lattice.ref)  # one sector sub-slice
            sub4, subk = _transverse_4d(sub), _dispersive_kick(sub)
            xblock = _blocks(sub)[0]  # x 2x2 of the sub-slice (incl. weak focusing)
            acc_dx = 0.5 * disp[0]  # trapezoid: half-weight the entrance samples
            acc_h = 0.5 * _curly_h(bx, ax, disp[0], disp[1])
            for i in range(slices):
                disp = sub4 @ disp + subk
                bx, ax, _ = _propagate_block(xblock, bx, ax)
                w = 0.5 if i == slices - 1 else 1.0  # half-weight the exit sample
                acc_dx += w * disp[0]
                acc_h += w * _curly_h(bx, ax, disp[0], disp[1])
            int_dx = acc_dx * ds  # ∮ D_x ds across the body
            i1 += h * int_dx
            i2 += h * h * elem.length
            i3 += abs(h) ** 3 * elem.length
            i4 += h**3 * int_dx  # pure sector bend: no 2*k1 body term, no edge term
            i5 += abs(h) ** 3 * acc_h * ds  # ∮ curlyH |h|^3 ds
            continue
        # Non-dipole: co-transport beta/alpha and dispersion across the element.
        bx, ax, _ = _propagate_block(_blocks(M)[0], bx, ax)
        disp = _transverse_4d(M) @ disp + _dispersive_kick(M)
    return RadiationIntegrals(i1, i2, i3, i4, i5)


def energy_loss_per_turn(lattice: Lattice) -> float:
    r"""Energy radiated per revolution ``U0 = (C_gamma / 2*pi) E^4 I2`` [eV].

    For an isomagnetic ring this is the textbook ``U0 = C_gamma E^4 / rho``
    (``I2 = 2*pi/rho``): ~88.5 keV per turn for a 1 GeV electron at ``rho = 1 m``.
    """
    ri = radiation_integrals(lattice)
    cg = radiation_constant_cgamma(lattice.ref)
    return cg / (2.0 * math.pi) * lattice.ref.total_energy_eV**4 * ri.i2


def damping_partition_numbers(lattice: Lattice) -> tuple[float, float, float]:
    r"""Damping partition numbers ``(J_x, J_y, J_z) = (1 - I4/I2, 1, 2 + I4/I2)``.

    They apportion the radiated damping among the three planes. **Robinson's
    theorem** ``J_x + J_y + J_z = 4`` is exact by construction (the ``I4/I2`` cancels)
    — the structural gate on the integrals. ``J_y = 1`` holds for a flat lattice with
    no vertical bending or gradient (this module's scope).
    """
    ri = radiation_integrals(lattice)
    d = ri.i4 / ri.i2
    return (1.0 - d, 1.0, 2.0 + d)


def damping_times(lattice: Lattice) -> tuple[float, float, float]:
    r"""Radiation **amplitude** damping times ``(tau_x, tau_y, tau_z)`` [s].

    ``tau_i = 2 E T0 / (J_i U0)``, with ``T0 = C / (beta0 c)`` the revolution period.
    These are the times for the oscillation **amplitude** to damp by ``1/e``; the
    action/emittance damps twice as fast (at ``tau_i / 2``). This matches the
    ``amplitude_damping_time`` convention of :func:`accsim.lifetime.quantum_lifetime`
    (Stage 4), so the two compose without a stray factor of 2 — Stage 4's quantum
    lifetime, which took the damping time as an input, is now computable from the
    lattice.
    """
    ri = radiation_integrals(lattice)
    cg = radiation_constant_cgamma(lattice.ref)
    e = lattice.ref.total_energy_eV
    u0 = cg / (2.0 * math.pi) * e**4 * ri.i2
    d = ri.i4 / ri.i2
    partitions = (1.0 - d, 1.0, 2.0 + d)
    t0 = lattice.length / (lattice.ref.beta0 * CLIGHT)
    return tuple(2.0 * e * t0 / (j * u0) for j in partitions)  # type: ignore[return-value]


def equilibrium_energy_spread(lattice: Lattice) -> float:
    r"""Equilibrium RMS relative energy spread ``sigma_delta`` (dimensionless).

    ``sigma_delta^2 = C_q gamma^2 I3 / (J_z I2)`` — the balance between quantum
    excitation (``I3``) and longitudinal radiation damping (``J_z``). Energy-only
    dependence is the ``gamma^2`` prefactor (``I3``, ``I2``, ``J_z`` are geometry), so
    ``sigma_delta ∝ gamma`` — the machine-precision scaling gate; the absolute value is
    pinned against xtrack.
    """
    ri = radiation_integrals(lattice)
    cq = quantum_constant_cq(lattice.ref)
    g2 = lattice.ref.gamma0**2
    jz = 2.0 + ri.i4 / ri.i2
    return math.sqrt(cq * g2 * ri.i3 / (jz * ri.i2))


def equilibrium_emittance(lattice: Lattice) -> float:
    r"""Equilibrium **geometric** horizontal emittance ``eps_x`` [m·rad].

    ``eps_x = C_q gamma^2 I5 / (J_x I2)`` — quantum excitation of the horizontal
    betatron motion (the curly-``H`` integral ``I5``) balanced against horizontal
    damping (``J_x``). Geometric (not normalized) emittance; multiply by ``beta0 gamma0``
    for the normalized value. Energy dependence is the ``gamma^2`` prefactor, so
    ``eps_x ∝ gamma^2`` — the machine-precision scaling gate (``I5``, ``I2``, ``J_x`` are
    pure geometry); the absolute value is pinned against xtrack.

    The equilibrium **vertical** emittance is ~0 here: with no vertical bending or
    betatron coupling there is no vertical quantum excitation. Real rings set ``eps_y``
    by coupling / vertical dispersion — out of scope (flat-lattice assumption).
    """
    ri = radiation_integrals(lattice)
    cq = quantum_constant_cq(lattice.ref)
    g2 = lattice.ref.gamma0**2
    jx = 1.0 - ri.i4 / ri.i2
    return cq * g2 * ri.i5 / (jx * ri.i2)
