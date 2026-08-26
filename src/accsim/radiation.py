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

**Scope.** ``I4`` carries the general **combined-function + edge** form
``I4 = ∮ D_x h (h^2 + 2 k1) ds - Σ_faces D_x h^2 tan(e)``: the ``2 k1`` body term
(from the quadrupole gradient) and the ``-D_x h^2 tan(e)`` pole-face term now
contribute, reducing to the pure-sector ``∮ D_x h^3 ds`` when ``k1 = e1 = e2 = 0``.
The dispersion/beta transport inside a dipole is co-transported through the
*actual* combined-function body and edge kicks, so ``I1``/``I4``/``I5`` are correct
for such magnets; ``I2 = ∮ h^2 ds`` and ``I3 = ∮ |h|^3 ds`` are pure geometry and
unchanged. The coefficient and edge sign are pinned against MAD-X's own
integral-method ``synch_4`` (not xtrack, whose damped-map eigenanalysis differs
from the integral method at the ~1% level — the size of the effect). The lattice
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

from .coords import X, Y
from .elements.element import Element
from .lattice import Lattice
from .radiation_kick import HBAR_C_EV_M as HBAR_C_EV_M  # noqa: PLC0414  (re-export)
from .radiation_kick import quantum_constant_cq as _cq
from .radiation_kick import radiation_constant_cgamma as _cgamma
from .reference import CLIGHT, ReferenceParticle
from .twiss import _blocks, _dispersive_kick, _propagate_block, _transverse_4d, closed_twiss


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

    Defined in :mod:`accsim.radiation_kick` and re-exported here: the design route
    (these integrals) and the tracked route (B2's per-element kick) set the size of
    the same effect, and must not carry two copies of the constant that does it.
    """
    return _cgamma(ref)


def quantum_constant_cq(ref: ReferenceParticle) -> float:
    r"""``C_q = 55/(32 sqrt3) * hbar c / (m c^2)`` [m] for the reference species.

    The quantum-excitation constant; ``55/(32 sqrt3)`` is the ratio of moments of the
    synchrotron-radiation spectrum (Sands). For the electron, ``3.832e-13 m``.

    Defined in :mod:`accsim.radiation_kick` and re-exported here, for the same reason
    ``C_gamma`` is: the equilibrium these integrals predict and the graininess B3's
    per-element kick injects are the same effect, and one constant sets both.
    """
    return _cq(ref)


def _curly_h(beta: float, alpha: float, dx: float, dpx: float) -> float:
    r"""The dispersion invariant ``curlyH = gamma D_x^2 + 2 alpha D_x D_x' + beta D_x'^2``."""
    gamma = (1.0 + alpha * alpha) / beta
    return gamma * dx * dx + 2.0 * alpha * dx * dpx + beta * dpx * dpx


def radiation_integrals(lattice: Lattice, slices: int = 64) -> RadiationIntegrals:
    r"""Compute ``I1..I5`` for a periodic ``lattice``.

    Only bending magnets contribute (``h = 0`` elsewhere). Inside each thick dipole
    the matched dispersion ``D_x(s)`` **and** the beta functions ``beta_x,
    alpha_x`` are co-transported by ``slices``-fold trapezoidal sub-stepping of the
    sub-bend map (the dispersion machinery of
    :func:`accsim.twiss.momentum_compaction` plus the ``beta`` transport of
    :func:`accsim.twiss.natural_chromaticity`). For a **combined-function** magnet the
    sub-slices carry the gradient ``k1`` (so the focusing that reshapes ``D_x`` inside
    the body is included), and the thin **pole-face edge** kicks are applied to
    ``(D_x', alpha_x)`` at entry/exit (``D_x`` and ``beta_x`` are continuous across a
    thin edge). ``h`` and ``k1`` are constant across the body, so ``∮ D_x h ds =
    h ∮ D_x ds`` and the ``I4`` body term ``∮ D_x h (h^2 + 2 k1) ds =
    h (h^2 + 2 k1) ∮ D_x ds`` reuse one accumulated ``∮ D_x ds``; the ``I4`` edge term
    subtracts ``D_x h^2 tan(e)`` at each face using the face-local ``D_x``.
    ``I5 = |h|^3 ∮ curlyH ds`` needs ``curlyH`` re-evaluated per sub-slice from the
    local ``beta_x, alpha_x, D_x, D_x'``. The ``h``-only pieces ``∮ h^2 ds`` /
    ``∮ |h|^3 ds`` are ``h^2 L`` / ``|h|^3 L`` per dipole (gradient/edge-independent).

    ``I1 == alpha_c * C`` cross-checks the dispersion transport within the baseline;
    ``I5`` (curly-``H``, needing the co-transported ``beta``) has no clean within-baseline
    absolute check, so it is gated by energy-scaling (``eps_x ∝ gamma^2``) + xtrack
    (``tests/analytic/test_radiation.py``, ``tests/reference/``).
    """
    from .elements.dipole import Dipole, _edge_matrix

    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    disp = np.array([tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py])
    i1 = i2 = i3 = i4 = i5 = 0.0
    for elem in lattice.elements:
        M = elem.matrix(lattice.ref)
        if isinstance(elem, Dipole) and elem.angle != 0.0 and elem.length > 0.0:
            h = elem.curvature  # 1/rho, signed
            k1 = elem.k1
            ds = elem.length / slices
            sub = Dipole(ds, h * ds, k1=k1).matrix(lattice.ref)  # combined-function sub-slice
            sub4, subk = _transverse_4d(sub), _dispersive_kick(sub)
            xblock = _blocks(sub)[0]  # x 2x2 of the sub-slice (incl. weak focusing + k1)

            # Entrance pole face: a thin kick on (D_x', alpha_x); D_x, beta_x are
            # continuous, so read the face-local D_x for the edge term either side.
            dx_entrance = disp[0]
            if elem.e1 != 0.0:
                ent = _edge_matrix(h, elem.e1)
                disp = _transverse_4d(ent) @ disp
                bx, ax, _ = _propagate_block(_blocks(ent)[0], bx, ax)

            acc_dx = 0.5 * disp[0]  # trapezoid: half-weight the entrance samples
            acc_h = 0.5 * _curly_h(bx, ax, disp[0], disp[1])
            for i in range(slices):
                disp = sub4 @ disp + subk
                bx, ax, _ = _propagate_block(xblock, bx, ax)
                w = 0.5 if i == slices - 1 else 1.0  # half-weight the exit sample
                acc_dx += w * disp[0]
                acc_h += w * _curly_h(bx, ax, disp[0], disp[1])
            int_dx = acc_dx * ds  # ∮ D_x ds across the body

            # Exit pole face (D_x still continuous ⇒ read before the kick).
            dx_exit = disp[0]
            if elem.e2 != 0.0:
                ext = _edge_matrix(h, elem.e2)
                disp = _transverse_4d(ext) @ disp
                bx, ax, _ = _propagate_block(_blocks(ext)[0], bx, ax)

            i1 += h * int_dx
            i2 += h * h * elem.length
            i3 += abs(h) ** 3 * elem.length
            # I4 = ∮ D_x h (h^2 + 2 k1) ds  -  Σ_faces D_x h^2 tan(e)
            i4 += h * (h * h + 2.0 * k1) * int_dx
            i4 -= h * h * (dx_entrance * math.tan(elem.e1) + dx_exit * math.tan(elem.e2))
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


def _coupling_off_lattice(lattice: Lattice) -> Lattice:
    """Copy of ``lattice`` with every skew quadrupole replaced by its ``k1s = 0``
    limit (a :class:`Drift` of the same length), i.e. the coupling turned off.

    A :class:`~accsim.elements.skew_quadrupole.SkewQuadrupole` at ``k1s = 0`` *is* a
    drift (``F = D`` → the map is a plain drift), and a
    :class:`~accsim.elements.skew_quadrupole.ThinSkewQuadrupole` at ``k1sl = 0`` is
    the identity, so this is exact for a thin skew. For a thick skew it additionally
    drops the magnet's own ``O(k1s^2)`` self-focusing ``(F + D) / 2`` — the natural
    coupling-off reference at the leading order the sharing formula already works to.
    The result is transversely uncoupled, so the Courant-Snyder path
    (:func:`equilibrium_emittance`, :func:`accsim.twiss.tunes`) applies.
    """
    from .elements.drift import Drift
    from .elements.skew_quadrupole import SkewQuadrupole, ThinSkewQuadrupole

    elems = []
    for e in lattice.elements:
        if isinstance(e, (SkewQuadrupole, ThinSkewQuadrupole)):
            if e.length > 0.0:
                elems.append(Drift(e.length))  # thick skew → its k1s=0 drift limit
            # thin skew → identity, drop it
        else:
            elems.append(e)
    return Lattice(elems, ref=lattice.ref)


def equilibrium_emittances_coupled(lattice: Lattice) -> tuple[float, float]:
    r"""Eigen-mode equilibrium geometric emittances ``(eps_1, eps_2)`` under linear
    betatron coupling [m·rad], near the difference resonance ``Q_x = Q_y``.

    A skew quadrupole couples the horizontal and vertical betatron motion, so the
    horizontal quantum excitation (which alone sets ``eps_x`` on a flat lattice) is
    shared between the two coupled **normal modes**. Diagonalising the excitation /
    damping balance in the mode basis, with the mode mixing fixed by the
    difference-resonance geometry ``tan(2 phi) = |C^-| / Delta``, gives

        G = sqrt(Delta^2 + |C^-|^2),
        eps_1 = eps_x0 (G + Delta) / (2 G) = eps_x0 cos^2 phi,   (the x-like mode)
        eps_2 = eps_x0 (G - Delta) / (2 G) = eps_x0 sin^2 phi,   (the y-like mode)

    where

    - ``eps_x0`` is the **coupling-off** horizontal equilibrium emittance
      (:func:`equilibrium_emittance` of :func:`_coupling_off_lattice`),
    - ``Delta`` is the distance of the decoupled tune split ``Q_x - Q_y`` to the
      nearest integer (the difference-resonance detuning), from
      :func:`accsim.twiss.tunes` of the coupling-off lattice,
    - ``|C^-| =`` :func:`accsim.twiss.closest_tune_approach` is the coupling strength.

    The sum is conserved exactly (``eps_1 + eps_2 = eps_x0``): coupling redistributes
    the horizontal excitation, it does not add any. ``eps_2`` (the smaller, y-like
    mode) is the **vertical emittance from coupling** — off the resonance
    ``eps_2 / eps_1 = (G - Delta)/(G + Delta) -> |C^-|^2 / (4 Delta^2)``. This closes the
    Stage-7 flat-lattice gap (``eps_y ≈ 0``). On the resonance (``Delta = 0``) the modes
    share equally (``eps_1 = eps_2 = eps_x0 / 2``).

    **Approximation, stated honestly.** This is the leading-order two-mode result and
    assumes (i) equal transverse damping ``J_x ≈ J_y`` — true for a weak ring
    (``I4/I2 ≪ 1``), *not* for a strongly combined-function one; (ii) a single
    difference-resonance coupling coefficient ``|C^-|`` (leading order in the skew
    strength, so most accurate for a thin skew). It is **not** a symbolic closed form
    but a physics model, validated against xtrack's radiation-envelope eigen-emittances
    (``eq_gemitt_x``/``eq_gemitt_y``) to ~3-4% off resonance — near the resonance both
    this model and the envelope eigenanalysis degrade (near-degenerate modes). The
    full radiation-envelope (Sigma-matrix) eigen-emittance treatment, which would drop
    both assumptions, is the rigorous alternative. Returns ``(eps_x0, 0.0)`` for an
    uncoupled lattice (``|C^-| = 0``). See ``docs/CONVENTIONS.md`` → *Betatron coupling*.
    """
    from .twiss import closest_tune_approach, tunes

    lat0 = _coupling_off_lattice(lattice)
    eps_x0 = equilibrium_emittance(lat0)
    qx, qy = tunes(lat0)
    split = qx - qy
    delta = abs(split - round(split))  # distance to the nearest difference resonance
    cminus = closest_tune_approach(lattice)
    g = math.hypot(delta, cminus)
    if g == 0.0:  # no coupling and exactly on the integer resonance: nothing to share
        return eps_x0, 0.0
    eps_1 = eps_x0 * (g + delta) / (2.0 * g)
    eps_2 = eps_x0 * (g - delta) / (2.0 * g)
    return float(eps_1), float(eps_2)


# ---------------------------------------------------------------------------
# N3: Sokolov-Ternov -- the polarization the radiation builds up
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolarizationIntegrals:
    r"""The two Derbenev-Kondratenko rate integrals, on the closed orbit [1/m^3].

    The spin-flip channel of the same synchrotron radiation ``I1..I5`` describe. Its
    two rates -- flips *out of* ``n_0`` and flips *into* it -- differ slightly, and that
    asymmetry is the whole of the Sokolov-Ternov effect. Written as arc-length averages
    over the ring (A. Chao, *Evaluation of Radiative Spin Polarization in an Electron
    Storage Ring*; the same pair ``xtrack`` reports as ``spin_alpha_plus_co`` /
    ``spin_alpha_minus_co``):

    - ``alpha_plus  = (1/C) int kappa^3 (1 - (2/9) (n_0 . v)^2) ds``  -- sets the **rate**,
    - ``alpha_minus = (1/C) int kappa^3 (n_0 . b) ds``               -- sets the **direction**,

    with ``kappa = |B_perp| / (B rho)_0`` the local orbit curvature, ``b`` the unit vector
    along the **physical** magnetic field, ``v`` the unit vector along the particle's
    motion, and ``n_0`` the periodic spin direction of :mod:`accsim.spin`.

    ``alpha_minus`` is **signed** and ``alpha_plus`` is not: the magnitude of the
    curvature lives in ``kappa^3``, the sense of the bend lives in ``b``. On a flat ring
    ``n_0`` is ``+y`` while an electron's guide field points ``-y``, so
    ``alpha_minus = -alpha_plus`` and the equilibrium polarization comes out
    **negative** -- the beam polarizes *antiparallel* to the field, which is the textbook
    direction and not a sign convention this package is free to choose.
    """

    alpha_plus: float
    alpha_minus: float


def _polarization_integrand(
    elem: Element, state: np.ndarray, spin: np.ndarray, qsign: float
) -> tuple[float, float]:
    """``(kappa^3 (1 - 2/9 (n_0.v)^2), kappa^3 (n_0.b))`` at one point on the orbit."""
    from .spin import along_direction_of_motion

    bx, by = elem.normalized_field(state[X], state[Y])
    # normalized_field is B/(B rho)_0, and (B rho)_0 = p/q carries the charge's sign, so
    # the *physical* field direction needs that multiplied back out. Getting it wrong
    # flips the polarization's direction while leaving every magnitude untouched.
    b = np.array([float(bx), float(by), 0.0]) * qsign
    kappa = float(np.linalg.norm(b))
    if kappa == 0.0:
        return 0.0, 0.0
    n_dot_v = float(spin @ along_direction_of_motion(state))
    return kappa**3 * (1.0 - 2.0 / 9.0 * n_dot_v * n_dot_v), kappa**3 * float(spin @ (b / kappa))


def polarization_integrals(lattice: Lattice, slices: int = 64) -> PolarizationIntegrals:
    r"""Compute ``(alpha_plus, alpha_minus)`` for a periodic ``lattice`` [1/m^3].

    Both integrals are accumulated **per sub-slice**, carrying ``n_0(s)`` from
    :func:`accsim.spin.closed_spin_solution` along with the closed orbit, because the
    weights ``(n_0 . b)`` and ``(n_0 . v)`` are properties of the *local* spin direction.
    On a flat ring the shortcut ``alpha_plus = I3 / C`` is exactly right and the
    accumulation is wasted work; it stops being right the moment ``n_0`` tilts, and no
    flat ring can tell the two apart.

    **``slices`` must resolve the spin phase, not the optics.** Inside a bend of angle
    ``theta`` the horizontal part of ``n_0`` turns through ``G gamma theta`` relative to
    the direction of motion -- 4.4 radians per bend on N2's 5 GeV gate ring, an order more
    at LEP energies -- while the dispersion :func:`radiation_integrals` sub-steps turns
    through ``theta``. The quadrature here is **Simpson's rule** over the sub-slice
    boundaries rather than the trapezoid used there, for exactly that reason: it
    converges as ``slices^-4`` and reaches the round-off floor of the ``(n_0 . v)^2``
    term at the default 64, where the trapezoid is still ``1.5e-2`` short of it.

    **Scope: only dipoles radiate** -- the same restriction :func:`radiation_integrals`
    carries, and deliberately the same one, because ``alpha_plus * C == I3`` is a gate
    the two routes must agree on and so they must agree on what radiates. A quadrupole
    traversed off-axis really does curve the orbit and really does radiate (``xtrack``
    counts it, reading ``kappa`` from the closed orbit element by element); on N2's gate
    ring the one offset quadrupole would add ``3e-12`` of ``alpha_plus``, which is
    negligible against both ``alpha_plus`` and the ``1e-8`` tilt term this milestone is
    about -- but only *there*: it grows as the **cube** of the orbit offset where the
    tilt term grows as its square, so the margin closes on a badly steered ring. Lifting
    the restriction means lifting it in both places, which moves axis B's numbers, so it
    is a separate change.
    """
    from .elements.dipole import Dipole
    from .spin import _closed_state, closed_spin_solution

    ref = lattice.ref
    spin = closed_spin_solution(lattice).n0.copy()
    state = _closed_state(lattice, None)
    qsign = math.copysign(1.0, ref.charge)
    alpha_plus = alpha_minus = 0.0
    for elem in lattice.elements:
        if isinstance(elem, Dipole) and elem.length > 0.0 and elem.curvature != 0.0:
            n = slices + (slices % 2)  # Simpson's rule needs an even number of steps
            ds = elem.length / n
            sub = Dipole(ds, elem.curvature * ds, k1=elem.k1)
            f_plus, f_minus = _polarization_integrand(sub, state, spin, qsign)
            alpha_plus += f_plus * ds / 3.0
            alpha_minus += f_minus * ds / 3.0
            for i in range(1, n + 1):
                state, spin = sub.track_with_spin(state, spin, ref)
                weight = (1.0 if i == n else (4.0 if i % 2 else 2.0)) / 3.0
                f_plus, f_minus = _polarization_integrand(sub, state, spin, qsign)
                alpha_plus += weight * f_plus * ds
                alpha_minus += weight * f_minus * ds
            continue
        state, spin = elem.track_with_spin(state, spin, ref)
    return PolarizationIntegrals(alpha_plus / lattice.length, alpha_minus / lattice.length)


def _alpha_plus_or_raise(lattice: Lattice, slices: int) -> PolarizationIntegrals:
    """The integrals, refusing a lattice that does not bend and so cannot polarize."""
    integrals = polarization_integrals(lattice, slices)
    if integrals.alpha_plus == 0.0:
        raise ValueError(
            "a lattice with no bending radiates nothing and has no Sokolov-Ternov "
            "polarization: alpha_plus is exactly zero"
        )
    return integrals


def sokolov_ternov_polarization(lattice: Lattice, slices: int = 64) -> float:
    r"""Equilibrium beam polarization ``P_inf = (8 / (5 sqrt3)) alpha_minus / alpha_plus``.

    Signed and dimensionless, measured along ``n_0``: **negative** on an ordinary
    electron ring, where the beam polarizes antiparallel to the guide field (see
    :class:`PolarizationIntegrals`). ``8 / (5 sqrt3) = 0.9237604...`` is the ratio of the
    two spin-flip rates, and it is reached exactly whenever ``n_0`` is parallel to the
    field everywhere the ring bends -- which every flat, unsteered ring satisfies.

    **That number is a control, not a gate.** It is a *ratio* of the two integrals, so
    any uniform mis-scale of the pair -- a wrong power of ``kappa``, a wrong
    circumference, a stray factor in the accumulation -- cancels out of it exactly and it
    returns ``-0.9237604...`` regardless. What it does measure is the two weights pulling
    apart: on a ring whose ``n_0`` tilts away from the field by ``t`` the departure is
    ``t^2 (1/2 - (2/9) <cos^2>)``, second order, and of *opposite sign* in the two
    integrals. That is the gate.

    **No depolarization.** This is ``xtrack``'s ``spin_polarization_inf_no_depol``, not
    its ``spin_polarization_eq``: the ``(11/18) int kappa^3 |dn/ddelta|^2`` term that
    fights the buildup needs the spin-orbit coupling ``dn/d(x, px, ...)``, which this
    package does not have yet (``docs/ROADMAP.md`` -> N4).
    """
    integrals = _alpha_plus_or_raise(lattice, slices)
    return 8.0 / (5.0 * math.sqrt(3.0)) * integrals.alpha_minus / integrals.alpha_plus


def polarization_buildup_time(lattice: Lattice, slices: int = 64) -> float:
    r"""The Sokolov-Ternov buildup time constant ``tau_pol`` [s].

    ``1 / tau = (5 sqrt3 / 8) r_0 (hbar / m_0) gamma^5 alpha_plus`` -- the rate at which
    the polarization approaches :func:`sokolov_ternov_polarization`, as
    ``1 - exp(-t / tau)``. The ``gamma^5``, and the ``kappa^3`` inside ``alpha_plus``,
    make it violently energy- and radius-dependent: a second in a small strong-bending
    ring, hours at LEP.

    **The coefficient is the discriminating quantity, and ``P_inf`` cannot see it** --
    that is a ratio in which any uniform mis-scale of the two rates cancels. Here it does
    not, so this is where a wrong constant shows up. Two things guard it: ``gamma^5`` and
    ``rho^3`` scaling, which catch a wrong *power*, and ``xtrack``'s
    ``spin_t_pol_component_s``, which catches a wrong *factor* and is the only thing that
    does. The eV-to-SI bridge is the risky step and is written out rather than folded
    into a literal: ``hbar / m_0 = (hbar c) c / (m c^2)``, in ``m^2/s``, assembled from
    the package's own ``HBAR_C_EV_M`` and the particle's own rest energy -- so a slip here
    is a slip axis B would feel too.
    """
    integrals = _alpha_plus_or_raise(lattice, slices)
    ref = lattice.ref
    hbar_over_mass = HBAR_C_EV_M * CLIGHT / ref.mass_eV  # [m^2/s]
    rate = (
        5.0
        * math.sqrt(3.0)
        / 8.0
        * ref.classical_radius_m
        * hbar_over_mass
        * ref.gamma0**5
        * integrals.alpha_plus
    )
    return 1.0 / rate
