r"""D1 — build a machine: inject, accelerate, store, collide, account the losses.

A narrated end-to-end run of the whole accsim stack on **one** electron storage
ring, in the order a real machine is commissioned::

    inject (Stage 1-2 optics)
      -> accelerate  (Stage 5 RF ramp)
      -> store       (Stage 7 radiation damping sets the equilibrium)
      -> collide     (Stage 6 luminosity)
      -> account     (Stage 4 aperture losses + quantum lifetime)

Run it with ``python examples/build_a_machine.py``.

**What this file is for.** Every stage already has its own analytic gate, and each
of those gates is a pure function of one lattice — re-asserting them here would be
green forever and prove nothing. The gates that matter live in
``tests/analytic/test_end_to_end.py`` and are all **seams**: statements about what
one stage hands the next, which no single-stage test can make. This module owns the
machine (so the test and the narration cannot drift apart) and the narration; the
test owns the assertions.

**The machine.** A 192 m, 24-cell FODO ring of thin quadrupoles and rectangular-ish
sector bends (total bend ``2*pi``, isomagnetic), one RF cavity on harmonic ``h``, and
one aperture. Injection at 0.6 GeV, store at 2.0 GeV. Electrons, because radiation
damping is the point and it is negligible for protons at this scale.

**Magnet strengths are geometric, so the optics does not move during the ramp.**
``ThinQuadrupole`` carries an integrated ``k1l`` and ``Dipole`` a bend *angle* —
both energy-independent as matrices. Physically that is the statement that the
magnets ramp in step with the beam, which is exactly what a real ramp does; so
``beta``, the tunes, ``D_x`` and ``alpha_c`` are the same at injection and at store,
and every energy dependence below is the *beam's*, not the lattice's.

Scope, stated honestly (each of these is a real limitation, not a simplification of
convenience):

- ``accelerate`` is **radiation-free** — it ramps the reference energy and
  re-references, with no ``U0`` term. A real ramp needs ``q V sin phi_s = Delta E_s
  + U0(E)``; here the cavity supplies only ``Delta E_s``. It is also
  **single-particle**, so the ramp below tracks one particle on an invariant
  ellipse, not a bunch.
- **Radiation damping is closed-form, never tracked.** accsim has no damped or
  stochastic map, so "store with damping" is a *data-flow handoff* — the store
  energy's equilibrium emittance and energy spread — and not a tracked
  ``eps -> eps_eq`` convergence. Do not read the store step as a simulation of
  damping; the damping *times* say how long it would take.
- **The IP is a design point, not a matched insertion.** Stage 6's ``luminosity`` /
  ``hourglass_reduction`` are closed forms in ``(eps, beta*, sigma_z)``, so ``beta*``
  here is a specified collider parameter. Matching a real low-beta insertion into
  this lattice (and re-matching the ring around it) is not attempted.
- **There is no vertical-emittance model.** ``equilibrium_emittance`` is the
  horizontal one; a flat uncoupled lattice has ``eps_y = 0``. The vertical emittance
  of a real ring comes from betatron coupling and vertical dispersion, neither
  modelled here, so ``eps_y`` is an input — set by ``COUPLING`` below.

Two limitations of the *existing* stages that this chain surfaced, both documented
in ``docs/CONVENTIONS.md``:

1. ``synchronous_phase`` keyed its stable branch on ``eta`` alone, which is the
   ``q V > 0`` (proton) special case. A lepton ring driven by a positive voltage has
   ``q V < 0`` and needs the other branch. Fixed before this example was written.
2. ``rf_bucket_height`` / ``separatrix`` / ``longitudinal_hamiltonian`` model the
   **stationary** bucket only. A store ring whose RF replenishes ``U0`` has
   ``sin phi_s = U0 / (q V) != 0``, so they reject it. Here the small parameter is
   ``U0 / |q V| ~ 1.9%``, and the acceptance is quoted from the stationary twin;
   the moving-bucket overvoltage factor is out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from accsim import (
    ELECTRON_MASS_EV,
    Aperture,
    Dipole,
    Drift,
    Lattice,
    Particle,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    Twiss,
    accelerate,
    closed_twiss,
    damping_times,
    energy_gain_per_turn,
    energy_loss_per_turn,
    equilibrium_emittance,
    equilibrium_energy_spread,
    hourglass_reduction,
    luminosity,
    momentum_compaction,
    quantum_lifetime,
    rf_bucket_height,
    slip_factor,
    synchronous_phase,
    synchrotron_tune,
    tunes,
)
from accsim.reference import CLIGHT

# --- machine parameters ---------------------------------------------------------
N_CELLS = 24
F_FOCAL = 3.0  # thin-quad focal length [m]
L_BEND = 2.0  # dipole length [m]
L_DRIFT = 1.0  # drift between quad and bend [m]
BEND_ANGLE = 2.0 * math.pi / (2 * N_CELLS)  # two bends per cell -> total bend 2*pi

E_INJECT = 0.6e9  # injection total energy [eV]
E_STORE = 2.0e9  # store total energy [eV]
HARMONIC = 400  # RF harmonic number
V_RAMP = 2.0e6  # ramp cavity voltage [V]
V_STORE = 5.0e6  # store cavity voltage [V]
RAMP_GAIN_PER_TURN = 3.0e5  # net synchronous gain during the ramp [eV/turn]

EMIT_INJECT = 5.0e-7  # geometric emittance delivered by the injector [m rad]
COUPLING = 0.01  # eps_y / eps_x — an input, not a model (see the docstring)
BETA_STAR = 5.0e-3  # design beta* at the IP [m]
BUNCH_POPULATION = 1.0e10  # particles per bunch
N_BUNCHES = 1
APERTURE_SIGMAS = 2.5  # vertical half-aperture, in equilibrium sigmas


def arc_cell() -> list:
    """One symmetric FODO cell: half-QF, bend, QD, bend, half-QF (two bends/cell)."""
    return [
        ThinQuadrupole(0.5 / F_FOCAL),
        Drift(L_DRIFT),
        Dipole(L_BEND, BEND_ANGLE),
        Drift(L_DRIFT),
        ThinQuadrupole(-1.0 / F_FOCAL),
        Drift(L_DRIFT),
        Dipole(L_BEND, BEND_ANGLE),
        Drift(L_DRIFT),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]


def arc(total_energy_eV: float) -> Lattice:
    """The bare arc (no RF, no aperture) at a given reference energy."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, total_energy_eV, charge=-1.0)
    return Lattice(arc_cell() * N_CELLS, ref=ref)


def rf_cavity(lattice: Lattice, voltage: float, energy_gain: float) -> RFCavity:
    """A cavity on harmonic :data:`HARMONIC` phased for ``energy_gain`` per turn.

    ``phi_s`` comes from the **assembled ring's own** ``eta`` and the reference's
    **signed** charge. Both matter: taking ``eta`` from a single arc cell would still
    work here (this ring is all arc), but the sign of ``q V`` is what puts a lepton
    machine on the opposite branch from a proton one.
    """
    ref = lattice.ref
    eta = slip_factor(lattice)
    phi_s = synchronous_phase(voltage, energy_gain, above_transition=eta > 0.0, charge=ref.charge)
    return RFCavity.from_harmonic(voltage, HARMONIC, lattice.length, ref, phi_s)


def ring(
    total_energy_eV: float,
    voltage: float,
    energy_gain: float,
    aperture: Aperture | None = None,
) -> Lattice:
    """The full ring: arc + one RF cavity (+ an optional aperture at the cell start)."""
    base = arc(total_energy_eV)
    elements = list(base.elements) + [rf_cavity(base, voltage, energy_gain)]
    if aperture is not None:
        elements = [aperture] + elements
    return Lattice(elements, ref=base.ref)


def store_ring(total_energy_eV: float = E_STORE, voltage: float = V_STORE) -> Lattice:
    """The store lattice: RF phased to replenish exactly the radiated ``U0``.

    This is the one place the accelerator core and the radiation module meet in the
    lattice itself — ``U0`` is a Stage-7 quantity and it sets a Stage-5 cavity phase.
    """
    base = arc(total_energy_eV)
    return ring(total_energy_eV, voltage, energy_loss_per_turn(base))


def stationary_twin(lattice: Lattice) -> Lattice:
    """The same ring with ``phi_s`` forced to the stationary value (no net gain).

    :func:`rf_bucket_height` models the stationary bucket only, and the store ring's
    is *moving* by ``sin phi_s = U0/(q V)``. The twin is what the acceptance is
    quoted from; see the module docstring, limitation 2.
    """
    return ring(lattice.ref.total_energy_eV, _voltage(lattice), 0.0)


def _voltage(lattice: Lattice) -> float:
    return sum(e.voltage for e in lattice.elements if isinstance(e, RFCavity))


# --- the chain ------------------------------------------------------------------
@dataclass(frozen=True)
class Machine:
    """Everything the chained run produces, with each number's stage of origin."""

    # optics (energy-independent — geometric magnets)
    circumference: float
    tune_x: float
    tune_y: float
    twiss: Twiss
    alpha_c: float
    # injection / ramp (Stage 5)
    ref_inject: ReferenceParticle
    ref_store: ReferenceParticle
    n_ramp_turns: int
    emit_injected: float
    emit_adiabatic: float
    emit_adiabatic_tracked_y: float
    emit_adiabatic_tracked_x: float
    # store (Stage 7 + Stage 3/5)
    u0: float
    phi_s: float
    eta: float
    qs: float
    emit_eq_x: float
    emit_eq_y: float
    sigma_delta: float
    sigma_z: float
    bucket_height: float
    tau_x: float
    tau_y: float
    tau_z: float
    # collide (Stage 6)
    f_rev: float
    sigma_x_ip: float
    sigma_y_ip: float
    lumi_geometric: float
    hourglass: float
    lumi: float
    lumi_injected_beam: float
    # account (Stage 4 + lifetime)
    sigma_y_aperture: float
    half_aperture: float
    xi: float
    tau_quantum: float


def bunch_length(lattice: Lattice, sigma_delta: float) -> float:
    r"""RMS bunch length ``sigma_z`` [m] of a matched bunch.

    ``sigma_z = sigma_delta * |eta| * C / (2 pi Qs)`` — the matched ellipse of the
    linear synchrotron rotation, whose aspect ratio is fixed by the same ``eta C``
    slip and ``Qs`` that :func:`accsim.synchrotron_tune` is built from.

    **This is the three-stage number of the whole example.** ``sigma_delta`` is
    radiation (Stage 7), ``Qs`` is the RF (Stage 3/5), ``eta`` and ``C`` are the
    lattice (Stage 1-2) — and the result then sets the hourglass reduction of the
    luminosity (Stage 6). Nothing about the bunch length is an input.
    """
    slip = abs(slip_factor(lattice)) * lattice.length
    return sigma_delta * slip / (2.0 * math.pi * synchrotron_tune(lattice))


def tracked_adiabatic_action(
    lattice: Lattice,
    emit0: float,
    n_turns: int,
    plane: str = "y",
    phase: float = 0.0,
) -> float:
    r"""Track one particle on the ``emit0`` invariant ellipse through the ramp.

    Launched at the closed-Twiss point with Courant-Snyder action ``J = emit0`` at
    betatron ``phase``, everything else zero; returns the CS action of the *tracked*
    state at the end of the ramp, evaluated with the ring's own Twiss. Adiabatic
    damping predicts ``emit0 * P0(0)/P0(n)`` — read out of the tracking, not from
    the reference-momentum program.

    **``plane`` defaults to ``"y"``, and that is a physics choice, not a style one.**
    The *horizontal* action is not cleanly adiabatic on this ring, and the reason is
    a coupling the chain surfaced: a betatron oscillation in ``x`` feeds ``zeta``
    through the dispersive one-turn entries ``R51 x + R52 px``, the cavity converts
    that ``zeta`` into a ``delta`` kick, and dispersion feeds ``D_x delta`` straight
    back into ``x``. The loop is closed, so the horizontal action carries a
    synchro-betatron ripple of a percent or so that does **not** shrink as the ramp
    slows — it is real physics, not ramp error. ``D_y = 0`` on a flat uncoupled
    lattice, so the vertical plane has no such path and shows the ``1/P0`` law
    cleanly, with a residual that *is* the finite ramp rate (``propto 1/n_turns``).
    """
    tw = closed_twiss(lattice)
    beta = tw.beta_y if plane == "y" else tw.beta_x
    alpha = tw.alpha_y if plane == "y" else tw.alpha_x
    gamma = tw.gamma_y if plane == "y" else tw.gamma_x
    amplitude = math.sqrt(emit0 * beta)
    u0 = amplitude * math.cos(phase)
    pu0 = -amplitude / beta * (alpha * math.cos(phase) + math.sin(phase))
    kwargs = {"y": u0, "py": pu0} if plane == "y" else {"x": u0, "px": pu0}
    result = accelerate(lattice, Particle(**kwargs), n_turns)
    idx = 2 if plane == "y" else 0
    u, pu = result.states[-1, idx], result.states[-1, idx + 1]
    return gamma * u * u + 2.0 * alpha * u * pu + beta * pu * pu


def build() -> Machine:
    """Run the whole chain and return every number it produced."""
    # --- optics, once: geometric magnets => energy-independent -------------------
    lat_inject = arc(E_INJECT)
    lat_store_arc = arc(E_STORE)
    tw = closed_twiss(lat_store_arc)
    qx, qy = tunes(lat_store_arc)

    # --- inject -> accelerate (Stage 5) -----------------------------------------
    ramp = ring(E_INJECT, V_RAMP, RAMP_GAIN_PER_TURN)
    n_turns = int(round((E_STORE - E_INJECT) / RAMP_GAIN_PER_TURN))
    ref_inject, ref_store = lat_inject.ref, lat_store_arc.ref
    damping = ref_inject.momentum_eV / ref_store.momentum_eV
    emit_adiabatic = EMIT_INJECT * damping
    emit_tracked_y = tracked_adiabatic_action(ramp, EMIT_INJECT, n_turns, plane="y")
    emit_tracked_x = tracked_adiabatic_action(ramp, EMIT_INJECT, n_turns, plane="x")

    # --- store (Stage 7 sets the equilibrium; Stage 3/5 holds the bunch) ---------
    lat = store_ring()
    u0 = energy_loss_per_turn(lat_store_arc)
    phi_s = next(e.phi_s for e in lat.elements if isinstance(e, RFCavity))
    eta = slip_factor(lat)
    qs = synchrotron_tune(lat)
    emit_eq_x = equilibrium_emittance(lat)
    sigma_delta = equilibrium_energy_spread(lat)
    sigma_z = bunch_length(lat, sigma_delta)
    tau_x, tau_y, tau_z = damping_times(lat)

    # --- collide (Stage 6) ------------------------------------------------------
    f_rev = ref_store.beta0 * CLIGHT / lat.length
    emit_eq_y = COUPLING * emit_eq_x
    sigma_x_ip = math.sqrt(emit_eq_x * BETA_STAR)
    sigma_y_ip = math.sqrt(emit_eq_y * BETA_STAR)
    lumi_geom = luminosity(
        BUNCH_POPULATION, BUNCH_POPULATION, sigma_x_ip, sigma_y_ip, f_rev, N_BUNCHES
    )
    hg = hourglass_reduction(sigma_z, BETA_STAR)
    # The same IP with the *injected* beam's emittance — what luminosity would be if
    # the collider had been handed the un-damped beam.
    lumi_injected = luminosity(
        BUNCH_POPULATION,
        BUNCH_POPULATION,
        math.sqrt(emit_adiabatic * BETA_STAR),
        math.sqrt(COUPLING * emit_adiabatic * BETA_STAR),
        f_rev,
        N_BUNCHES,
    )

    # --- account the losses (Stage 4 + quantum lifetime) ------------------------
    sigma_y_ap = math.sqrt(emit_eq_y * tw.beta_y)  # D_y = 0 => pure betatron
    half_aperture = APERTURE_SIGMAS * sigma_y_ap
    xi = half_aperture**2 / (2.0 * sigma_y_ap**2)

    return Machine(
        circumference=lat.length,
        tune_x=qx,
        tune_y=qy,
        twiss=tw,
        alpha_c=momentum_compaction(lat_store_arc),
        ref_inject=ref_inject,
        ref_store=ref_store,
        n_ramp_turns=n_turns,
        emit_injected=EMIT_INJECT,
        emit_adiabatic=emit_adiabatic,
        emit_adiabatic_tracked_y=emit_tracked_y,
        emit_adiabatic_tracked_x=emit_tracked_x,
        u0=u0,
        phi_s=phi_s,
        eta=eta,
        qs=qs,
        emit_eq_x=emit_eq_x,
        emit_eq_y=emit_eq_y,
        sigma_delta=sigma_delta,
        sigma_z=sigma_z,
        bucket_height=rf_bucket_height(stationary_twin(lat)),
        tau_x=tau_x,
        tau_y=tau_y,
        tau_z=tau_z,
        f_rev=f_rev,
        sigma_x_ip=sigma_x_ip,
        sigma_y_ip=sigma_y_ip,
        lumi_geometric=lumi_geom,
        hourglass=hg,
        lumi=lumi_geom * hg,
        lumi_injected_beam=lumi_injected,
        sigma_y_aperture=sigma_y_ap,
        half_aperture=half_aperture,
        xi=xi,
        tau_quantum=quantum_lifetime(half_aperture, sigma_y_ap, tau_y),
    )


# --- narration ------------------------------------------------------------------
def _head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    m = build()
    inj, sto = m.ref_inject, m.ref_store

    _head("0. The lattice (Stage 1-2) -- the same at every energy")
    print(f"  circumference        C  = {m.circumference:.1f} m, {N_CELLS} FODO cells")
    print(f"  tunes                   = ({m.tune_x:.3f}, {m.tune_y:.3f})")
    print(f"  beta_x, beta_y          = {m.twiss.beta_x:.3f} m, {m.twiss.beta_y:.3f} m")
    print(f"  dispersion           D_x= {m.twiss.disp_x:.3f} m")
    print(
        f"  momentum compaction  a_c= {m.alpha_c:.5f}  ->  gamma_t = {1 / math.sqrt(m.alpha_c):.2f}"
    )
    print("  Magnets are geometric (k1l, bend angle), so this block is energy-independent:")
    print("  physically, the magnets ramp with the beam. Every number below that moves")
    print("  with energy is the beam's doing, not the lattice's.")

    _head("1. Inject and accelerate (Stage 5)")
    print(
        f"  inject at {inj.total_energy_eV / 1e9:.2f} GeV (gamma0 = {inj.gamma0:.0f}), "
        f"store at {sto.total_energy_eV / 1e9:.2f} GeV (gamma0 = {sto.gamma0:.0f})"
    )
    print(f"  gamma_t = {1 / math.sqrt(m.alpha_c):.2f}, so the whole ramp is ABOVE transition")
    print("  -- no transition crossing to negotiate.")
    print(
        f"  net gain {RAMP_GAIN_PER_TURN / 1e3:.0f} keV/turn -> {m.n_ramp_turns} turns "
        f"= {m.n_ramp_turns / (sto.beta0 * CLIGHT / m.circumference) * 1e3:.1f} ms"
    )
    print(f"  injected emittance      eps    = {m.emit_injected * 1e9:.2f} nm")
    print(
        f"  after adiabatic damping eps    = {m.emit_adiabatic * 1e9:.2f} nm "
        f"(x P0_inj/P0_store = 1/{sto.momentum_eV / inj.momentum_eV:.3f})"
    )
    print(
        f"  ...measured from the tracked ramp, vertical   = "
        f"{m.emit_adiabatic_tracked_y * 1e9:.2f} nm "
        f"({abs(m.emit_adiabatic_tracked_y / m.emit_adiabatic - 1) * 1e2:.2f}% off)"
    )
    print(
        f"  ...the same measurement,          horizontal = "
        f"{m.emit_adiabatic_tracked_x * 1e9:.2f} nm "
        f"({abs(m.emit_adiabatic_tracked_x / m.emit_adiabatic - 1) * 1e2:.2f}% off)"
    )
    print("  Adiabatic damping shrinks the GEOMETRIC emittance as 1/P0; the normalised")
    print("  emittance is what the ramp conserves. The vertical residual is the finite")
    print("  ramp rate and shrinks like 1/n_turns. The horizontal one does NOT -- it is")
    print("  synchro-betatron coupling: x -> zeta through the dispersive R51/R52, zeta")
    print("  -> delta in the cavity, delta -> x through D_x. That loop only exists once")
    print("  RF and dispersion are in the same ring, and D_y = 0 keeps the vertical")
    print("  plane out of it. It is the chain showing something no stage sees alone.")

    _head("2. Store (Stage 7 radiation + Stage 3/5 RF)")
    print(f"  energy loss/turn      U0 = {m.u0 / 1e3:.2f} keV")
    print(
        f"  RF replenishes it: V = {V_STORE / 1e6:.1f} MV, phi_s = {m.phi_s * 1e3:.2f} mrad, "
        f"gain/turn = {energy_gain_per_turn(store_ring()) / 1e3:.2f} keV"
    )
    print(
        f"  U0/|qV| = {abs(m.u0 / (sto.charge * V_STORE)):.4f}  -- the bucket is 'moving' by"
        " this much;"
    )
    print("  the stationary-bucket machinery would reject it, so the acceptance below")
    print("  is quoted from the stationary twin (see the module docstring).")
    print(f"  slip factor           eta = {m.eta:+.5f} (above transition)")
    print(f"  synchrotron tune       Qs = {m.qs:.5f}")
    print(
        f"  equilibrium emittance     = {m.emit_eq_x * 1e9:.2f} nm   (eps_y = "
        f"{COUPLING:.0%} of it, an INPUT -- no vertical-emittance model)"
    )
    print(f"  equilibrium spread  sig_d  = {m.sigma_delta:.3e}")
    print(f"  bunch length        sig_z  = {m.sigma_z * 1e3:.2f} mm   <- radiation x RF x lattice")
    print(
        f"  RF acceptance     delta_max= {m.bucket_height:.3e} = "
        f"{m.bucket_height / m.sigma_delta:.1f} sigma_delta"
    )
    print(
        f"  damping times   (x, y, z)  = ({m.tau_x * 1e3:.1f}, {m.tau_y * 1e3:.1f}, "
        f"{m.tau_z * 1e3:.1f}) ms"
    )
    print(
        f"  The beam arrives at {m.emit_adiabatic * 1e9:.1f} nm and damps to "
        f"{m.emit_eq_x * 1e9:.1f} nm -- a factor {m.emit_adiabatic / m.emit_eq_x:.2f},"
    )
    print(f"  taking a few tau_x, i.e. ~{3 * m.tau_x * 1e3:.0f} ms. (Closed-form, not tracked:")
    print("  accsim has no damped map -- see the module docstring.)")

    _head("3. Collide (Stage 6)")
    print(
        f"  design beta*            = {BETA_STAR * 1e3:.1f} mm, N = {BUNCH_POPULATION:.1e}/bunch, "
        f"f_rev = {m.f_rev / 1e6:.3f} MHz"
    )
    print(
        f"  IP beam size (sig_x, sig_y) = ({m.sigma_x_ip * 1e6:.2f}, {m.sigma_y_ip * 1e6:.2f}) um"
    )
    print(f"  geometric luminosity    = {m.lumi_geometric * 1e-4:.3e} cm^-2 s^-1")
    print(f"  hourglass (sig_z/beta* = {m.sigma_z / BETA_STAR:.2f}) = {m.hourglass:.4f}")
    print(f"  luminosity              = {m.lumi * 1e-4:.3e} cm^-2 s^-1")
    print(
        f"  Had the collider been handed the UNDAMPED injected beam instead: "
        f"{m.lumi_injected_beam * 1e-4:.3e} cm^-2 s^-1"
    )
    print(
        f"  -- a factor {m.lumi_geometric / m.lumi_injected_beam:.2f} lower, which is exactly the "
        "emittance ratio. Radiation"
    )
    print("  damping is a luminosity multiplier, and the hourglass factor is the one")
    print("  place the bunch length (radiation x RF) reaches into Stage 6.")

    _head("4. Account for the losses (Stage 4 + quantum lifetime)")
    print(
        f"  vertical size at the aperture sig_y = {m.sigma_y_aperture * 1e6:.2f} um "
        f"(D_y = 0, pure betatron)"
    )
    print(
        f"  half-aperture A = {APERTURE_SIGMAS:.1f} sigma = {m.half_aperture * 1e3:.3f} mm "
        f"-> xi = A^2/2sig^2 = {m.xi:.3f}"
    )
    print(f"  instantaneous Gaussian cut 1 - e^-xi = {1 - math.exp(-m.xi):.4f} survive")
    print(f"  quantum lifetime tau_q = tau_y e^xi/(2 xi) = {m.tau_quantum * 1e3:.1f} ms")
    print("  The SAME xi drives both: the aperture removes the amplitude tail, and")
    print("  tau_q is the rate at which radiation excitation refills it. That is why")
    print("  the lifetime is exponentially sensitive to the aperture:")
    for n_sig in (2.5, 4.0, 6.0, 8.0, 10.0):
        xi = n_sig**2 / 2.0
        tq = quantum_lifetime(n_sig * m.sigma_y_aperture, m.sigma_y_aperture, m.tau_y)
        print(f"    A = {n_sig:4.1f} sigma -> xi = {xi:5.1f} -> tau_q = {tq:9.3e} s")
    print("  Four extra sigma of aperture (4 -> 8) buys ten orders of magnitude.")


if __name__ == "__main__":
    main()
