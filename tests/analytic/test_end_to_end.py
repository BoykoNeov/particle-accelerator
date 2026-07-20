r"""D1 — the end-to-end chain: gates on the **seams**, not on the stages.

The machine itself lives in ``examples/build_a_machine.py`` (inject -> accelerate ->
store -> collide -> account), so the narration and these assertions cannot drift
apart. This file asserts only what *no single-stage test can*.

**The trap this file is written against.** Every stage quantity is a pure function
of one lattice, so ``equilibrium_emittance(ring)`` returns the same number here as it
does in ``test_radiation.py``. Re-asserting a stage's own invariant on the chained
run is green forever and tests nothing — it is the D4 tautology in a new costume.
The discriminating question for every assertion below is: *would this still pass if
the value were recomputed from a fresh standalone lattice?* If yes, it is not a seam
and it does not belong here.

The seams, in chain order:

1. **Stage 5 -> Stage 7 (emittance provenance).** Adiabatic damping shrinks the
   *geometric* emittance as ``1/P0`` while the radiation equilibrium grows as
   ``gamma^2``. The composite ``eps_adiabatic/eps_eq`` is therefore exactly
   ``propto 1/(beta0 gamma0^3)`` — a statement no single stage can make, and the
   ``1/P0`` half of it is read out of the **tracked** ramp, not predicted.
2. **Stage 7 -> Stage 3/5 (the RF setting).** ``U0`` (radiation) sets ``phi_s``
   (RF) on the **assembled** ring's own ``eta``. Both branches deliver the same
   energy gain; only one has a bucket, and the tracked synchrotron tune at the store
   point confirms it.
3. **Stage 7 + 3/5 -> Stage 6 (what reaches the luminosity).** The bunch length is
   not an input: ``sigma_z = sigma_delta |eta| C / (2 pi Qs)`` is radiation x RF x
   lattice, and it reaches Stage 6 through the hourglass factor. The luminosity must
   consume the *damped* emittance, never the injected one.
4. **Stage 7 -> Stage 4 + lifetime (the same ``xi``).** ``xi = A^2/2 sigma^2``
   governs *both* the aperture's amplitude cut (tracked, Stage 4) and the quantum
   lifetime ``tau_q = tau_y e^xi/(2 xi)``. Fed the Stage-7 equilibrium ``sigma`` and
   the Stage-7 damping time, the two must agree on it.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from accsim import (
    Aperture,
    Bunch,
    Lattice,
    Particle,
    RFCavity,
    Tracker,
    closed_twiss,
    damping_times,
    energy_gain_per_turn,
    energy_loss_per_turn,
    equilibrium_emittance,
    equilibrium_energy_spread,
    hourglass_reduction,
    luminosity,
    naff,
    quantum_lifetime,
    rf_bucket_height,
    slip_factor,
    synchronous_phase,
    synchrotron_tune,
)
from accsim.twiss import UnstableLatticeError


def _load_example():
    """Import ``examples/build_a_machine.py`` without making ``examples`` a package."""
    path = Path(__file__).resolve().parents[2] / "examples" / "build_a_machine.py"
    spec = importlib.util.spec_from_file_location("build_a_machine", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_a_machine"] = module
    spec.loader.exec_module(module)
    return module


bm = _load_example()


@pytest.fixture(scope="module")
def machine():
    """The chained run, built once (it tracks a 4667-turn ramp)."""
    return bm.build()


# --- seam 1: Stage 5 -> Stage 7, emittance provenance ----------------------------
def test_ramp_damps_the_tracked_emittance_as_one_over_p0(machine) -> None:
    """The *tracked* ramp reproduces ``eps -> eps * P0(0)/P0(n)`` in the free plane.

    Not a restatement of ``accelerate``'s own ``px *= r`` invariant: the quantity
    measured is the Courant-Snyder **action** of the tracked ``(y, py)`` at the end
    of the ramp, evaluated with the ring's Twiss, against a prediction made purely
    from the reference-momentum program. It is what licenses handing
    ``emit_adiabatic`` to the store step at all.

    Vertical, because ``D_y = 0``; the horizontal plane carries a synchro-betatron
    ripple that the next test isolates. The residual here is the **non-adiabaticity**
    of a finite ramp rate, not tolerance slack — the convergence test pins that.
    """
    rel = abs(machine.emit_adiabatic_tracked_y / machine.emit_adiabatic - 1.0)
    assert rel < 2.0e-3


def test_horizontal_action_carries_a_synchro_betatron_ripple(machine) -> None:
    r"""The horizontal action misses the ``1/P0`` law, and it is physics, not error.

    **A seam finding, asserted so it cannot silently go away.** Once RF and
    dispersion live in the same ring a loop closes that neither owns:
    ``x -> zeta`` through the dispersive one-turn entries ``R51 x + R52 px``,
    ``zeta -> delta`` in the cavity, ``delta -> x`` through ``D_x``. The horizontal
    CS action therefore does not damp cleanly as ``1/P0``, while the vertical
    (``D_y = 0``, no such path) does.

    Asserted as an inequality between the two planes rather than a value: if a future
    change made the horizontal plane *as* clean as the vertical, the coupling would
    have been lost and this should be looked at.
    """
    horizontal = abs(machine.emit_adiabatic_tracked_x / machine.emit_adiabatic - 1.0)
    vertical = abs(machine.emit_adiabatic_tracked_y / machine.emit_adiabatic - 1.0)
    assert horizontal > 5.0 * vertical
    assert horizontal < 0.05  # ...but still a ripple, not a blow-up


@pytest.mark.slow
def test_vertical_adiabatic_residual_converges_as_the_ramp_slows() -> None:
    """Halving the gain per turn (doubling the ramp length) roughly halves the miss.

    This is what makes the tolerance above a *measurement* rather than a choice: the
    miss is the finite ramp rate and it goes like ``1/n_turns``. Taken as the worst
    case over betatron launch phases, so a lucky final phase cannot flatter a coarse
    ramp.
    """
    predicted = (
        bm.EMIT_INJECT * bm.arc(bm.E_INJECT).ref.momentum_eV / bm.arc(bm.E_STORE).ref.momentum_eV
    )
    residuals = []
    for gain in (1.2e6, 6.0e5, 3.0e5):
        ramp = bm.ring(bm.E_INJECT, bm.V_RAMP, gain)
        n_turns = int(round((bm.E_STORE - bm.E_INJECT) / gain))
        residuals.append(
            max(
                abs(
                    bm.tracked_adiabatic_action(
                        ramp, bm.EMIT_INJECT, n_turns, plane="y", phase=phase
                    )
                    / predicted
                    - 1.0
                )
                # The action ripple has period pi in the launch phase; sample it.
                for phase in (0.0, math.pi / 8.0, math.pi / 4.0, 3.0 * math.pi / 8.0)
            )
        )
    for coarse, fine in zip(residuals, residuals[1:], strict=False):
        assert fine < 0.75 * coarse, f"residual did not shrink with the ramp rate: {residuals}"


@pytest.mark.parametrize("energies", [(1.0e9, 2.0e9, 3.0e9, 5.0e9)])
def test_adiabatic_over_equilibrium_scales_as_one_over_beta_gamma_cubed(energies) -> None:
    r"""``eps_adiabatic / eps_eq * beta0 * gamma0^3`` is store-energy independent.

    **The composite no stage owns.** The numerator is Stage 5 (adiabatic damping,
    ``propto 1/P0 = 1/(beta0 gamma0 m)``); the denominator is Stage 7 (quantum
    excitation, ``propto gamma0^2`` on fixed lattice geometry). Neither exponent is
    visible from inside its own stage's tests, and a sign or power error in either
    breaks the product. Exact, so it is asserted at machine precision.

    Physically it is the reason a ring is filled at low energy and stored high: the
    injected beam is emittance-dominated by ``16x`` at 1 GeV and by only ``2x`` at
    2 GeV, and above ~2.4 GeV the equilibrium is *larger* than what the injector
    delivered — damping would blow the beam up, not shrink it.
    """
    p_inject = bm.arc(bm.E_INJECT).ref.momentum_eV
    invariants = []
    for energy in energies:
        lat = bm.arc(energy)
        ref = lat.ref
        emit_adiabatic = bm.EMIT_INJECT * p_inject / ref.momentum_eV
        ratio = emit_adiabatic / equilibrium_emittance(lat)
        invariants.append(ratio * ref.beta0 * ref.gamma0**3)
    assert max(invariants) / min(invariants) - 1.0 < 1.0e-12


# --- seam 2: Stage 7 -> Stage 3/5, the RF setting --------------------------------
def test_store_rf_replenishes_exactly_the_radiated_energy(machine) -> None:
    """``energy_gain_per_turn`` (Stage 5) == ``energy_loss_per_turn`` (Stage 7).

    A provenance assertion, and it is labelled as one: ``phi_s`` was *inverted* from
    ``U0``, so this is a round trip through :func:`synchronous_phase`, not an
    independent derivation. It earns its place because it is the only line that
    fails if the store lattice is ever built from a stale or different ``U0``.
    """
    assert energy_gain_per_turn(bm.store_ring()) == pytest.approx(machine.u0, rel=1e-12)


def test_only_one_phi_s_branch_has_a_bucket_on_the_assembled_ring() -> None:
    r"""The lepton store ring's stable branch, checked on the ring that will be used.

    ``phi_s`` is set from the **assembled ring's** ``eta`` and the reference's
    **signed** charge. This ring has ``q V < 0`` (electron, positive voltage), so the
    stable root is ``asin(s)`` even though it is above transition — the opposite of
    the proton rule. Both branches give the same ``U0`` replenishment, so the energy
    bookkeeping cannot tell them apart; only the bucket can.

    Negative control for the whole store step: build the other branch and there is
    no stable longitudinal motion at all.
    """
    lat = bm.store_ring()
    eta = slip_factor(lat)
    assert eta > 0.0  # above transition
    assert lat.ref.charge * bm.V_STORE < 0.0  # ...and q V < 0, which flips the branch
    phi_s = next(e.phi_s for e in lat.elements if isinstance(e, RFCavity))
    assert -math.pi / 2 < phi_s < 0.0

    wrong = Lattice(
        [e for e in lat.elements if not isinstance(e, RFCavity)]
        + [RFCavity.from_harmonic(bm.V_STORE, bm.HARMONIC, lat.length, lat.ref, math.pi - phi_s)],
        ref=lat.ref,
    )
    assert energy_gain_per_turn(wrong) == pytest.approx(energy_gain_per_turn(lat), rel=1e-12)
    assert synchrotron_tune(lat) > 0.0
    with pytest.raises(UnstableLatticeError):
        synchrotron_tune(wrong)


def test_store_bucket_accepts_the_equilibrium_energy_spread(machine) -> None:
    """The RF acceptance must swallow the radiation-set energy spread.

    ``delta_max`` is Stage 3/5, ``sigma_delta`` is Stage 7; the ring is only a store
    ring if their ratio is comfortably large. Quoted from the **stationary twin**
    because the real store bucket is moving by ``sin phi_s = U0/(qV)`` — a limitation
    this chain surfaced, and the small parameter is asserted here so that the
    substitution stays honest as parameters change.
    """
    assert abs(machine.u0 / (machine.ref_store.charge * bm.V_STORE)) < 0.05
    assert machine.bucket_height / machine.sigma_delta > 10.0


@pytest.mark.slow
def test_tracked_synchrotron_tune_matches_the_formula_at_the_store_point() -> None:
    """Tracking the store bucket recovers ``synchrotron_tune``, converging as ``Qs^2``.

    The store lattice is assembled from a *radiation* quantity (``U0`` -> ``phi_s``);
    this tracks it and reads the tune back with NAFF. The residual is the known
    lumped-cavity approximation (``synchrotron_tune`` collapses the cavity into one
    thin kick), and it is shown **converging** as ``Qs -> 0`` rather than tolerated:
    dividing the relative gap by ``Qs^2`` gives a constant.

    ``sigma_z`` inherits exactly this error, since it is built from the formula's
    ``Qs`` — which is why it is measured here rather than assumed.
    """
    u0 = energy_loss_per_turn(bm.arc(bm.E_STORE))
    gaps = []
    for voltage in (1.25e6, 5.0e6, 20.0e6):
        lat = bm.ring(bm.E_STORE, voltage, u0)
        formula = synchrotron_tune(lat)
        history = Tracker(lat).track_turns(
            Particle.from_array([0.0, 0.0, 0.0, 0.0, 1.0e-5, 0.0]), 8000, nonlinear=True
        )
        # A real signal's FFT peak is ambiguous between Qs and 1 - Qs; fold it.
        tracked = naff(history[:, 4])
        tracked = min(tracked, 1.0 - tracked)
        gaps.append((formula, abs(tracked / formula - 1.0)))
    for formula, gap in gaps:
        assert gap < 0.02, f"tracked Qs missed the formula by {gap:.2e} at Qs={formula:.4f}"
    coefficients = [gap / formula**2 for formula, gap in gaps]
    assert max(coefficients) / min(coefficients) < 60.0  # gap ~ Qs^2 over a 16x V range
    assert gaps[0][1] < gaps[-1][1]  # ...and it shrinks with Qs


# --- seam 3: Stage 7 + 3/5 -> Stage 6, what reaches the luminosity ---------------
def test_luminosity_consumes_the_damped_emittance_not_the_injected_one(machine) -> None:
    """``L(equilibrium) / L(injected) == eps_adiabatic / eps_eq``, exactly.

    Provenance with teeth: ``L propto 1/(sigma_x sigma_y) propto 1/eps`` for a beam
    round in normalised terms, so the luminosity ratio *is* the emittance ratio. If
    the collider step were ever handed the injected emittance (or a hard-coded
    ``sigma``), this equality breaks — it is the only assertion that ties the Stage-6
    output back to the Stage-7 input.
    """
    expected = machine.emit_adiabatic / machine.emit_eq_x
    assert machine.lumi_geometric / machine.lumi_injected_beam == pytest.approx(expected, rel=1e-12)
    assert expected > 1.0  # the injected beam really is the bigger one here


def test_bunch_length_reaches_stage_6_through_the_hourglass() -> None:
    r"""Raising the RF voltage shortens the bunch and *recovers* luminosity.

    The three-stage number: ``sigma_z = sigma_delta |eta| C / (2 pi Qs)`` — radiation
    (Stage 7) x RF (Stage 3/5) x lattice (Stage 1-2) — with ``beta*`` chosen so that
    ``sigma_z ~ beta*`` and the hourglass actually bites.

    **Negative control against a frozen ``sigma_z``.** A pipeline that hard-coded the
    bunch length, or took it from anywhere but the chain, would give a hourglass
    factor flat in ``V``. Both the monotonicity and the *size* of the response are
    asserted, so flatness fails.
    """
    u0 = energy_loss_per_turn(bm.arc(bm.E_STORE))
    sigma_delta = equilibrium_energy_spread(bm.arc(bm.E_STORE))
    voltages = (1.25e6, 5.0e6, 20.0e6)
    lengths, factors = [], []
    for voltage in voltages:
        lat = bm.ring(bm.E_STORE, voltage, u0)
        sigma_z = bm.bunch_length(lat, sigma_delta)
        lengths.append(sigma_z)
        factors.append(hourglass_reduction(sigma_z, bm.BETA_STAR))

    assert lengths[0] > lengths[1] > lengths[2]  # more RF -> shorter bunch
    assert factors[0] < factors[1] < factors[2]  # shorter bunch -> less hourglass loss
    assert factors[-1] / factors[0] > 1.5  # not flat: a frozen sigma_z gives exactly 1
    # sigma_z propto 1/Qs, so the length ratio is the tune ratio (the chain's own route).
    tune_ratio = synchrotron_tune(bm.ring(bm.E_STORE, voltages[-1], u0)) / synchrotron_tune(
        bm.ring(bm.E_STORE, voltages[0], u0)
    )
    assert lengths[0] / lengths[-1] == pytest.approx(tune_ratio, rel=1e-12)
    # ...and the luminosity follows the hourglass one-for-one.
    lumi = [luminosity(1.0e10, 1.0e10, 1.0e-5, 1.0e-6, 1.0e6) * factor for factor in factors]
    assert lumi[-1] / lumi[0] == pytest.approx(factors[-1] / factors[0], rel=1e-12)


@pytest.mark.slow
def test_bunch_length_constant_is_pinned_by_tracking() -> None:
    r"""``sigma_z = sigma_delta |eta| C / (2 pi Qs)`` — the whole constant, from tracking.

    Everything else about ``sigma_z`` in this file is a *ratio*, and a ratio cannot
    see a wrong constant: drop the ``2 pi``, or slip a factor of two, and every
    hourglass response test stays green. The bunch length has no independent
    reference inside accsim, so this pins it against the tracker.

    A particle launched at ``(zeta, delta) = (0, sigma_delta)`` rotates on the
    matched ellipse; the ratio of its excursions, ``zeta_max/delta_max``, *is* the
    ellipse aspect ratio ``|eta| C / (2 pi Qs)``, measured by tracking the nonlinear
    map rather than by evaluating the formula. Both extrema are read from the same
    turn-sampled history, so the sampling bias largely cancels.

    The residual is the same lumped-cavity ``O(Qs^2)`` error as the tracked-tune
    test, and it is shown shrinking with ``Qs`` rather than tolerated.
    """
    sigma_delta = equilibrium_energy_spread(bm.arc(bm.E_STORE))
    u0 = energy_loss_per_turn(bm.arc(bm.E_STORE))
    misses = []
    for voltage in (1.25e6, 20.0e6):
        lat = bm.ring(bm.E_STORE, voltage, u0)
        predicted = bm.bunch_length(lat, sigma_delta)
        history = Tracker(lat).track_turns(
            Particle.from_array([0.0, 0.0, 0.0, 0.0, 0.0, sigma_delta]), 3000, nonlinear=True
        )
        aspect = np.abs(history[:, 4]).max() / np.abs(history[:, 5]).max()
        misses.append(abs(aspect * sigma_delta / predicted - 1.0))
    assert misses[0] < 0.03, f"bunch-length constant missed by {misses[0]:.3f}"
    assert misses[0] < misses[-1]  # ...and the miss grows with Qs, as the lumping does


def test_hourglass_is_actually_biting_at_this_design_point(machine) -> None:
    """``sigma_z ~ beta*``, so the Stage-6 seam is exercised rather than nominal.

    A guard on the *example*, not on the physics: if ``BETA_STAR`` or the RF drifted
    to a regime where ``H ~ 1``, the test above would still pass its monotonicity
    checks while measuring nothing. Fail loudly instead.
    """
    assert 0.3 < machine.sigma_z / bm.BETA_STAR < 5.0
    assert machine.hourglass < 0.95
    # Re-evaluated with *named* arguments: hourglass_reduction takes (sigma_z,
    # beta*), and sigma_z ~ beta* here means a positional swap is numerically
    # plausible and otherwise invisible. This is the line that catches it.
    assert machine.hourglass == pytest.approx(
        hourglass_reduction(sigma_z=machine.sigma_z, beta_x_star=bm.BETA_STAR), rel=1e-12
    )


# --- seam 4: Stage 7 -> Stage 4 + lifetime, the same xi --------------------------
def test_quantum_lifetime_and_aperture_cut_share_one_xi(machine) -> None:
    r"""``-ln(1 - T) == xi == ln(2 xi tau_q / tau_y)``, on the Stage-7 equilibrium.

    Closed-form half of the seam. A matched Gaussian's betatron **amplitude** obeys
    ``P(a > A) = e^{-xi}`` with ``xi = A^2/2 sigma^2``; the same ``xi`` sits in the
    exponent of ``tau_q = tau_d e^xi/(2 xi)``. The aperture removes the amplitude
    tail and ``tau_q`` is the rate at which radiation excitation refills it — they
    cannot disagree about ``xi`` unless ``sigma`` has different provenance on the two
    sides, which is exactly the failure this catches.
    """
    # Provenance first: the sigma both sides share must be the Stage-7 equilibrium.
    # Without this line the aperture could be sized off the *injected* beam and xi
    # would still come out at APERTURE_SIGMAS^2/2 — it is defined in sigmas, so it
    # is blind to which sigma. Every downstream equality would stay green.
    assert machine.sigma_y_aperture == pytest.approx(
        math.sqrt(machine.emit_eq_y * machine.twiss.beta_y), rel=1e-12
    )
    assert machine.emit_eq_y == pytest.approx(bm.COUPLING * machine.emit_eq_x, rel=1e-12)
    assert machine.sigma_x_ip == pytest.approx(
        math.sqrt(machine.emit_eq_x * bm.BETA_STAR), rel=1e-12
    )

    xi = machine.xi
    assert xi == pytest.approx(bm.APERTURE_SIGMAS**2 / 2.0, rel=1e-12)
    surviving = 1.0 - math.exp(-xi)
    assert -math.log(1.0 - surviving) == pytest.approx(xi, rel=1e-12)
    assert math.log(2.0 * xi * machine.tau_quantum / machine.tau_y) == pytest.approx(xi, rel=1e-12)


@pytest.mark.parametrize("energies", [(1.0e9, 1.5e9, 2.0e9, 3.0e9)])
def test_quantum_lifetime_composes_two_stage_7_scalings(energies) -> None:
    r"""``tau_q(gamma)`` built from ``sigma propto gamma`` and ``tau_y propto 1/gamma^3``.

    A **fixed physical** half-gap (not a fixed number of sigmas, which would make
    ``xi`` energy-independent and the test trivial), so
    ``xi(gamma) = xi_ref (gamma_ref/gamma)^2`` and

        tau_q(gamma) = tau_ref (gamma_ref/gamma)^3 * e^{xi(gamma)} / (2 xi(gamma)).

    Two independent Stage-7 energy scalings feed one Stage-4 quantity, and the
    exponential makes the composition brutally sensitive: over 1 -> 3 GeV the
    lifetime falls by nine orders of magnitude, so a wrong power anywhere is not
    subtle.
    """
    half_gap = 0.15e-3  # m
    beta_y = closed_twiss(bm.arc(bm.E_STORE)).beta_y  # optics is energy-independent
    reference = bm.arc(energies[0])
    sigma_ref = math.sqrt(bm.COUPLING * equilibrium_emittance(reference) * beta_y)
    xi_ref = half_gap**2 / (2.0 * sigma_ref**2)
    tau_ref = damping_times(reference)[1]
    gamma_ref = reference.ref.gamma0

    for energy in energies:
        lat = bm.arc(energy)
        sigma = math.sqrt(bm.COUPLING * equilibrium_emittance(lat) * beta_y)
        tau_y = damping_times(lat)[1]
        scale = gamma_ref / lat.ref.gamma0
        # tau = 2 E T0 / (J U0) with T0 = C/(beta0 c): gamma^-3 *and* a beta0 factor.
        beta_ratio = reference.ref.beta0 / lat.ref.beta0
        xi = xi_ref * scale**2
        expected = tau_ref * scale**3 * beta_ratio * math.exp(xi) / (2.0 * xi)
        assert quantum_lifetime(half_gap, sigma, tau_y) == pytest.approx(expected, rel=1e-9)

    span = quantum_lifetime(
        half_gap,
        math.sqrt(bm.COUPLING * equilibrium_emittance(bm.arc(energies[0])) * beta_y),
        damping_times(bm.arc(energies[0]))[1],
    ) / quantum_lifetime(
        half_gap,
        math.sqrt(bm.COUPLING * equilibrium_emittance(bm.arc(energies[-1])) * beta_y),
        damping_times(bm.arc(energies[-1]))[1],
    )
    assert span > 1.0e8  # the composition really is exponential, not a mild trend


@pytest.mark.slow
def test_tracked_aperture_cut_recovers_the_lifetime_xi(machine) -> None:
    r"""Tracking a Stage-7 equilibrium bunch onto a Stage-4 aperture measures ``xi``.

    The independent half of seam 4. A bunch matched to the **equilibrium** vertical
    size is tracked for enough turns that betatron phases decohere, so the surviving
    fraction stops being the *spatial* cut and becomes the **amplitude** cut,
    ``1 - e^{-xi}`` — the very quantity ``tau_q``'s exponent describes. Nothing here
    is told the answer: ``sigma`` comes from ``equilibrium_emittance``, the loss
    count comes from ``track_bunch_losses``.

    Vertical on purpose: ``D_y = 0``, so the size is pure betatron and the Gaussian
    amplitude law applies without a dispersive contribution.
    """
    sigma = machine.sigma_y_aperture
    aperture = Aperture("rectangular", 1.0, machine.half_aperture)
    lat = bm.ring(bm.E_STORE, bm.V_STORE, machine.u0, aperture=aperture)

    n_particles, n_turns = 20000, 80
    rng = np.random.default_rng(20260720)
    states = np.zeros((6, n_particles))
    states[2] = rng.normal(0.0, sigma, n_particles)  # alpha_y = 0 at the cell centre,
    states[3] = rng.normal(0.0, sigma / machine.twiss.beta_y, n_particles)  # so it is matched
    result = Tracker(lat).track_bunch_losses(Bunch(states), n_turns=n_turns)

    expected = 1.0 - math.exp(-machine.xi)
    binomial = math.sqrt(expected * (1.0 - expected) / n_particles)
    assert result.transmission == pytest.approx(expected, abs=3.0 * binomial + 2.0e-3)
    assert -math.log(1.0 - result.transmission) == pytest.approx(machine.xi, rel=0.05)


# --- the example itself ----------------------------------------------------------
def test_the_worked_example_runs(capsys) -> None:
    """``examples/build_a_machine.py`` executes and narrates the chain it asserts."""
    bm.main()
    out = capsys.readouterr().out
    assert out.isascii()  # the narration prints on a plain Windows console
    for heading in ("Inject and accelerate", "Store", "Collide", "Account for the losses"):
        assert heading in out


def test_the_store_ring_is_the_one_the_example_narrates(machine) -> None:
    """The reported numbers come from the assembled store lattice, not a twin.

    Cheap guard against the narration and the gates drifting onto different
    lattices — the failure mode that would quietly turn every seam above into a
    statement about a machine nobody builds.
    """
    lat = bm.store_ring()
    assert machine.circumference == pytest.approx(lat.length)
    assert machine.qs == pytest.approx(synchrotron_tune(lat), rel=1e-12)
    assert machine.eta == pytest.approx(slip_factor(lat), rel=1e-12)
    assert machine.sigma_delta == pytest.approx(equilibrium_energy_spread(lat), rel=1e-12)
    assert machine.bucket_height == pytest.approx(rf_bucket_height(bm.stationary_twin(lat)))
    # ...and phi_s really was inverted from U0 on that lattice.
    phi_s = next(e.phi_s for e in lat.elements if isinstance(e, RFCavity))
    assert phi_s == pytest.approx(
        synchronous_phase(
            bm.V_STORE,
            energy_loss_per_turn(bm.arc(bm.E_STORE)),
            above_transition=slip_factor(lat) > 0.0,
            charge=lat.ref.charge,
        )
    )
