# Roadmap

Each stage is a milestone defined by its **acceptance tests**. A stage is complete
only when those pass. **Validate, don't advance:** never start stage N+1 until
stage N passes its analytic benchmarks (and any applicable Xsuite cross-checks).

There are two sub-projects with a clean handoff at the interaction point:
*accelerator / beam dynamics* (gets beams to collision) and an optional *event
physics* phase (what comes out of a collision — orchestration, not rebuilding).

## Validation strategy (non-negotiable)

- **`tests/analytic/`** — every physics quantity has a closed-form check. Always
  run in CI.
- **`tests/reference/`** — `xtrack` (and optionally MAD-X) cross-checks, behind the
  `reference` pytest marker so they skip when the dep is absent. These catch the
  coefficient/convention errors that hand-derived analytic checks can share.
- **Long-term tracking sanity** — track a matched particle for 1e4–1e5 turns and
  confirm the action/emittance does not drift. This is the symplecticity smoke
  test (`pytest -m slow`).
- **Gate** — a stage's acceptance tests must pass before the next stage starts.

> The Stage 1+ Xsuite cross-checks depend on the `xtrack` JIT compiler. This was
> blocked on this machine and is now **resolved** (built via clang-cl) — see the
> toolchain notes in [`CONVENTIONS.md`](CONVENTIONS.md). The `zeta`-sign question
> is **settled**: drift, quad, and dipole 6×6 maps (incl. the dipole's
> longitudinal row) match xtrack's sign exactly, through Stage 1.

## Stage 0 — Scaffold ✅ COMPLETE

Repo, `pyproject.toml`, CI, the `Element`/`Lattice`/`Tracker`/`Particle`
skeletons, plotting, and the analytic test harness.

- **Acceptance:** a `Drift` propagates a particle to the analytically expected
  position; CI is green. ✅
- Delivered beyond the minimum: the full 6×6 drift map (incl. the longitudinal
  `R56 = L/γ₀²`, derived symbolically), a symplecticity check, and a
  gracefully-skipping xtrack cross-check scaffold.

## Stage 1 — Beam optics (linear transverse) ✅ COMPLETE

Transfer-matrix formalism; `Drift`, `Quadrupole` (thin + thick), `Dipole`;
one-turn map; Twiss propagation (β, α, dispersion, phase advance); tunes.

**Status:** all delivered and xtrack-validated. `Quadrupole` (thin + thick),
`Dipole` (pure sector bend), Courant-Snyder Twiss (matched β/α + continuous
phase + tunes), and matched/propagated dispersion. Every element's 6×6 agrees
with xtrack's R-matrix to ~1e-6 (drift/quad/dipole), the FODO Twiss matches
xtrack's 4D Twiss to ~1e-14, and the dispersion matches xtrack's `dx`/`dpx`
(same `δ` convention, ratio 1.0). Out of this stage by design: edge focusing,
combined-function gradients (Stage 2), momentum compaction (Stage 3).

> **Stage 1 prerequisites — all resolved (kept for the record):**
> - ✅ **Xsuite/xtrack cross-check live** (2026-06-29, via clang-cl). Every Stage
>   1 element now cross-checks against xtrack's R-matrix; see CONVENTIONS.md.
> - ✅ **`zeta` sign vs Xsuite — settled, no mismatch.** Every element's 6×6
>   (drift, quad, dipole) and the dipole's longitudinal `R51/R52/R56` matched
>   xtrack's sign exactly out of the box; no reconciliation was needed.
> - ✅ **Composition-order test added** with `Quadrupole`
>   (`test_quad_drift_composition_is_order_sensitive`): an asymmetric drift+quad
>   sequence that changes if the `M_last @ … @ M_first` order is reversed.

- **Acceptance:** for a single FODO cell, the phase advance per cell `μ` (from
  `cos μ = ½·Tr M`) and the β-functions match the **symbolically-derived**
  closed-form thin-lens result (derive it, don't trust a remembered coefficient).
  β should oscillate between a maximum at the focusing quad and a minimum at the
  defocusing quad. Cross-check a small ring against Xsuite Twiss to < 1e-6. ✅
  **MET** — `tests/analytic/test_fodo_cell.py` (symbolic `μ`, `β_max`, `β_min`,
  max-at-F/min-at-D oscillation) and `tests/reference/test_fodo_twiss_xtrack.py`
  (thick-quad FODO ring vs xtrack 4D Twiss, agreement ~1e-14 ≪ 1e-6).

## Stage 2 — Magnetic lenses ✅ COMPLETE

FODO lattices; thin vs thick lens; natural chromaticity; sextupoles for
chromaticity correction (linear effect); beam-envelope plots.

- **Acceptance:** the FODO cell's natural chromaticity matches the analytic
  estimate; the stability boundary (`|Tr M| < 2`) matches the analytic
  phase-advance limit. ✅ **MET** (chromaticity + stability boundary below); the
  beam-envelope deliverable closes the stage.

**Progress:**
- ✅ **Natural chromaticity** — `natural_chromaticity(lattice)` returns
  `(Q'_x, Q'_y) = dQ/dδ` from quad off-momentum weakening `k1 → k1/(1+δ)`, as the
  β-weighted integral `Q'_x = −(1/4π)∮β_x k1 ds` (opposite sign for `y`); thin
  quads exact, thick quads sub-sliced. Independently validated to machine
  precision by the symbolically-differentiated `δ`-dependent one-turn map
  (`tests/analytic/test_chromaticity.py`, **not** the circular sum-vs-sum), and
  cross-checked against xtrack's `dqx`/`dqy` real-particle tracking to `rel≈1e-4`
  with a convention guard (`tests/reference/test_chromaticity_xtrack.py`). See
  CONVENTIONS.md → *Natural chromaticity*.
- ✅ **Stability boundary `|Tr M| < 2` vs the analytic phase-advance limit** —
  for the symmetric thin FODO, `cos μ = 1 − L²/(2f²)`, so the one *reachable*
  boundary is the over-focusing edge `f_crit = L/2` where `cos μ = −1`, i.e. the
  phase advance per cell hits its analytic limit `μ = π`. `f_crit` is derived
  symbolically from `Tr M = −2` (no accsim — avoids the `is_stable`≡`½Tr`
  circularity), and the element chain reproduces it: `½Tr → −1` in both planes at
  `f_crit`, `is_stable` flips across it (`closed_twiss` raising just beyond), the
  `is_stable` region matches `sin(μ/2)=L/(2f)<1` over a focal-length sweep, and
  the independent `tunes()` atan2 path sends `Q → ½` as `f → f_crit⁺`
  (`tests/analytic/test_stability_boundary.py`). See CONVENTIONS.md → *Stability
  boundary*.
- ✅ **`Sextupole` element (chromaticity correction, linear effect)** — `Sextupole`
  (thick) + `ThinSextupole`, whose *linear* map is a drift (identity for thin), so
  they leave β/dispersion/tunes untouched. The Stage-2 effect is chromaticity
  **feed-down** at dispersion: `chromaticity(lattice)` = quad `natural_chromaticity`
  `+ (1/4π)∮β_x k2 D_x ds` (x) `− (1/4π)∮β_y k2 D_x ds` (y). Pinned to machine
  precision by a symbolic δ-dependent trace derivative and cross-checked against
  xtrack's real-tracking `Δdqx`/`Δdqy` via a with-minus-without-sextupole
  difference (so accsim's uncomputed dipole term cancels) to `rel≈2e-3`
  (`tests/analytic/test_sextupole.py`, `tests/reference/test_sextupole_xtrack.py`).
  See CONVENTIONS.md → *Sextupole*.
- ✅ **Beam-envelope plots** — `beam_sigma(twiss, emit_x, emit_y, sigma_delta)`
  returns the 1-σ envelopes `σ_u = √(ε_u β_u + (D_u σ_δ)²)` (betatron + dispersive
  offset added **in quadrature**), plotted by `plotting.plot_beam_envelope`; the
  `emittance=` branch of `plot_beta_functions` now delegates to the same helper
  (σ_δ=0), so there is a single σ formula. Physics gated by the exact
  decomposition `σ_x² − ε_x β_x == (D_x σ_δ)²` on a dispersive (dipole) arc cell
  (`tests/analytic/test_beam_envelope.py`); ε and σ_δ are inputs (no equilibrium
  emittance until Stages 3/5), and no xtrack test is warranted (pure algebra over
  β and D, both already Stage-1 validated). See CONVENTIONS.md → *Beam envelope*.

## Stage 3 — Synchrotron motion (longitudinal) ✅ COMPLETE

RF bucket, synchronous phase, momentum-compaction factor, synchrotron tune,
longitudinal phase-space tracking, separatrix.

- **Acceptance:** the small-amplitude synchrotron tune `Qs` matches the analytic
  formula; the bucket height matches; particles launched inside the separatrix
  stay bounded over ≥ 1e4 turns. ✅ **MET** — symbolic `Qs`
  (`tests/analytic/test_synchrotron_tune.py`) + xtrack `tw.qs`; symbolic bucket
  height `δ_max = 2Qs/(h|η|)` and inside-bounded / outside-runs-away 1e4-turn
  nonlinear tracking (`tests/analytic/test_rf_bucket.py`, `-m slow`).

**Progress:**
- ✅ **Momentum-compaction factor + slip factor** — `momentum_compaction(lattice)`
  computes the geometric `α_c = (1/C)∮ D_x h ds` (dispersion transported and
  integrated through thick dipoles; only bends contribute, so `α_c = 0` on a
  straight lattice), and `slip_factor(lattice)` returns `η = α_c − 1/γ₀²`
  (single-sourced `1/γ₀²`, matching xtrack's `slip_factor` sign). Pinned by the
  symplecticity identity `α_c = 1/γ₀² − (R51 D_x + R52 D_px + R56)/C` (independent
  matrix entries), a sympy proof that the integral and identity paths are
  algebraically identical, and an xtrack cross-check of `momentum_compaction_factor`
  /`slip_factor` (~1e-6). See CONVENTIONS.md → *Momentum compaction / slip factor*.
- ✅ **RF cavity + synchrotron tune `Qs`** — `RFCavity(voltage, frequency, phi_s)`,
  a thin longitudinal kick `Δδ = (qV/β₀²E₀)[sin(φs−k_rf·zeta)−sin φs]` whose phase
  convention matches xtrack's `Cavity` (`φ = φs − k_rf·zeta`, `k_rf = 2πf/β₀c`).
  `synchrotron_tune(lattice)` builds the reduced one-turn 2×2 from the **slip
  factor** (not the bare `R56` — flag A) and the cavity slope `R65`, giving
  `Qs = arccos(½Tr Ms)/2π`, which reproduces the symbolic closed form
  `Qs² = −(hηqV cosφs)/(2πβ₀²E₀)`. Stationary bucket only (`φs=0`/`π`
  below/above transition; wrong side raises). Pinned by a sympy derivation and an
  xtrack cross-check (accsim's own 6×6 eigen-tune matches `tw.qs` to ~1e-6; the
  lumped formula to the sub-percent synchro-betatron coupling order). See
  CONVENTIONS.md → *RF cavity / synchrotron tune*.
- ✅ **Nonlinear longitudinal tracking + RF bucket / separatrix** — the
  nonlinear-tracking seam (`Element.track`, `Tracker(..., nonlinear=True)`) with the
  RF `sin` kick as its first user; the synchrotron `longitudinal_hamiltonian`,
  `separatrix`, and `rf_bucket_height` (`δ_max = 2Qs/(h|η|) = √(2qV/(πh|η|β₀²E₀))`,
  derived symbolically). Inside-the-separatrix particles librate bounded and
  conserve `H` over 1e4 turns; outside, `zeta` runs away (rotation). See
  CONVENTIONS.md → *RF bucket / nonlinear longitudinal tracking*.

## Stage 4 — Beam losses ✅ COMPLETE

Geometric apertures + collimators with survival/loss accounting; simple lifetime
models (aperture and quantum lifetime). **Touschek and intrabeam scattering are
advanced/optional — stub, don't build, unless asked.**

- **Acceptance:** a particle outside the aperture is flagged at the correct
  longitudinal location; transmission through a known aperture matches a hand
  calculation; the loss map reproduces a simple analytic case. ✅ **MET** — all
  three gates in `tests/analytic/test_beam_losses.py`.

**Progress:**
- ✅ **`Aperture` / `Collimator` element** — optics-transparent (identity 6×6)
  geometric acceptance boundary (circular / elliptical / rectangular), with a
  vectorised `survives(states)` predicate and an inclusive on-boundary convention
  matching xtrack. `Collimator` is the finite-length jaw (entry/exit check only —
  the interior-peak miss is flagged). Predicate geometry pinned with hand-placed,
  off-knife-edge particles (`tests/analytic/test_aperture.py`). See CONVENTIONS.md
  → *Beam losses / apertures*.
- ✅ **Loss-aware tracking + `LossResult`** — `Tracker.track_bunch_losses(bunch,
  n_turns)` walks the lattice accumulating geometric `s`, tests survivors at each
  aperture, records `(loss_turn, loss_s, loss_element)` and freezes/skips lost
  particles; `LossResult` exposes `transmission` and `loss_map()`. Meets all three
  acceptance gates: loss flagged at correct geometric `s` (not `zeta`);
  round-Gaussian circular transmission `1 − exp(−R²/2σ²)` (sympy-proven) + the
  separable rectangular `erf`-product, both vs a binomial tolerance; two-aperture
  loss map reproduces the analytic per-location counts.
- ✅ **Quantum (aperture-limited) lifetime** — `quantum_lifetime(aperture, sigma,
  amplitude_damping_time)` = `τ_d·e^ξ/(2ξ)`, `ξ = A²/2σ²`, **derived** from the
  amplitude-diffusion Fokker–Planck MFPT (not a remembered constant): the exact
  `(τ_d/2)∫₀^ξ (e^w−1)/w dw` verified symbolically and matched by the closed form
  to `O(1/ξ)`. The amplitude-vs-emittance factor-of-2 damping-time convention is
  documented and pinned (`tests/analytic/test_quantum_lifetime.py`). See
  CONVENTIONS.md → *Quantum lifetime*.
- Out of scope by design (roadmap): Touschek / IBS (advanced — not built).

## Stage 5 — RF cavities ✅ COMPLETE

Standalone `RFCavity` (voltage, harmonic number, phase), multi-cavity support,
acceleration ramp, energy gain per turn. **Beam loading, higher-order modes, and
wakefields are out of scope** unless a milestone adds them.

- **Acceptance:** energy gain per turn equals `qV·sin(φs)`; the synchronous
  particle stays synchronous; behaviour is consistent with the Stage 3 model. ✅
  **MET** — all three gates in `tests/analytic/test_acceleration.py`.

**Progress:**
- ✅ **Harmonic-number interface + multi-cavity** — `RFCavity.from_harmonic(voltage,
  harmonic, circumference, ref, phi_s)` sets `frequency = h·β₀c/C` (so `k_rf·C = 2πh`
  exactly) and `harmonic_number()` inverts it. `energy_gain_per_turn(lattice)` sums
  `q·V·sin(φs)` over **all** cavities (they may differ in voltage/phase), so
  multi-cavity rings add contributions.
- ✅ **Acceleration ramp + energy gain per turn** — the Stage-3 cavity kick already
  carried the accelerating physics (its `−sin(φs)` term is the energy the reference
  absorbs, so `zeta=0` gets zero net kick). Stage 5 turns the ramp on:
  `accelerate(lattice, particle, n_turns)` tracks nonlinearly while the reference
  energy climbs `E₀(n) = E₀(0) + n·ΔE_s`, `ΔE_s = ΣqV·sin(φs)`, rebuilding a fresh
  `ReferenceParticle` each turn (the lattice's `ref` is never mutated). Returns a
  `RampResult` (states + energy program). **Energy gain per turn == qV·sin(φs)**
  (gate 1) is asserted both as the closed form and as the actual per-turn
  increment; **the synchronous particle stays synchronous** (gate 2) is asserted
  *together* with the ramp being real (origin→origin while E₀ climbs), below and
  above transition; **consistency with Stage 3** (gate 3): with `sin φs = 0` the
  ramp is a no-op and `accelerate` reproduces Stage-3 nonlinear tracking
  **bit-for-bit**.
- ✅ **Adiabatic damping (derived)** — re-referencing the normalised momenta to the
  ramped `P₀'` multiplies `(px, py, delta)` by `r = P₀/P₀'` once per turn (derived
  from the coordinate definitions in `docs/CONVENTIONS.md`, not a remembered
  factor). Pinned by the exact telescoped closed form `px[n] = px0·P₀(0)/P₀(n)` on
  a drift+cavity ring, and by an off-momentum neighbour executing a **damped**
  synchrotron oscillation whose amplitude shrinks while the adiabatic invariant
  (action `≈ δ_max²/Qs`) is conserved — the geometric amplitude shrinking is
  physics, **not** a symplecticity leak, so the invariant (not raw action) is the
  right thing to assert.
- ✅ **`synchronous_phase(voltage, energy_gain, above_transition)`** — inverts
  `ΔE_s = qV·sin(φs)` for the **stable** root (`η·cos φs < 0`): `φs ∈ (0, π/2)`
  below transition, `(π/2, π)` above, reducing to the Stage-3 stationary `0`/`π` at
  zero gain.
- ✅ **Moving-bucket guard** — the Stage-3 `rf_bucket_height`/`separatrix`/
  `longitudinal_hamiltonian` (which assume a *stationary* bucket symmetric about
  `zeta=0`) now raise `NotImplementedError` for `sin φs ≠ 0` rather than return a
  plausible-wrong curve; `φs ∈ {0, π}` (stationary) still works. The moving-bucket
  *acceptance* (bucket area vs. φs), beam loading, and transition crossing are out
  of scope.
- No xtrack cross-check is warranted: the deliverables are derived closed forms
  (`qV·sin φs`; the `P₀/P₀'` re-referencing) over already-validated Stage-1/3 maps —
  the same rationale as the Stage-2 beam-envelope. See CONVENTIONS.md →
  *Acceleration / energy ramp*.

## Stage 6 — Collider design ✅ COMPLETE

Two beams, interaction point(s), low-β insertion, luminosity from beam parameters,
crossing angle; weak-strong beam-beam kick and beam-beam tune shift.
**Strong-strong beam-beam, crab cavities, and dynamic-aperture studies are
research-grade and out of scope** unless explicitly requested.

- **Acceptance:** the luminosity formula reproduces a textbook worked example for
  a known machine; the beam-beam tune shift `ξ` matches the analytic expression;
  a head-on weak-strong kick conserves the expected invariants. ✅ **MET** — all
  three gates below (`tests/analytic/test_luminosity.py`, `test_beam_beam.py`,
  `test_beam_beam_tune_shift.py`).

**Progress:**
- ✅ **Luminosity (gate 1)** — `luminosity(N1, N2, σ_x, σ_y, f_rev, n_bunches, …)`
  = `f_rev·n_b·N1·N2/(4π σ_x σ_y)` [m⁻²s⁻¹] with the optional Piwinski crossing
  reduction `S = 1/√(1+(σ_z tan(φ/2)/σ_cross)²)` (`accsim.collider`). The `4π`
  (equal-beam) coefficient is **derived** from the Gaussian overlap integral
  (sympy), not remembered; the acceptance number is the **LHC nominal** worked
  example (LHC Design Report Vol I, Table 2.1): head-on `1.20e34 cm⁻²s⁻¹`, design
  `1.0e34` with the 285 µrad crossing (`tests/analytic/test_luminosity.py`). The
  cm/m 10⁴ trap and the normalized-vs-geometric-emittance stray-γ trap are pinned;
  hourglass was flagged out of scope here and landed later as **C2**.
  `ReferenceParticle.classical_radius_m`
  (`r0 = r_e·(m_e/m)·q²`) added for the beam-beam kick. See CONVENTIONS.md →
  *Luminosity*.
- ✅ **Weak-strong beam-beam kick (gate 3)** — `BeamBeam(n_particles, sigma,
  strong_charge)` (`accsim.elements.beambeam`): the thin head-on kick from a round
  Gaussian strong bunch, `Δpx = K x g(u)`, `Δpy = K y g(u)`, `K = (q2/q1) N r0/(γσ²)`,
  `g(u)=(1−e^{−u})/u` (axis-regular). The **sign is derived from the Lorentz force**
  (like charges defocus, opposite focus), and the kick conserves the expected
  invariants: **curl-free** `∂Δpx/∂y=∂Δpy/∂x` and **angular momentum** `L_z=x py−y px`
  (radial ⇒ no torque), both round-beam properties, plus a match to an independent
  bare-`1/r` closed form (`tests/analytic/test_beam_beam.py`). Elliptical
  Bassetti–Erskine was flagged out of scope here and landed later as **C1** (which
  keeps the round beam's `L_z` invariant but loses it for `σ_x ≠ σ_y`, as anticipated).
  See CONVENTIONS.md → *Weak-strong beam-beam kick*.
- ✅ **Beam-beam tune shift ξ (gate 2)** — `beam_beam_tune_shift(bb, ref, β_x, β_y)`
  returns the **signed** `ΔQ_u = −β_u K/(4π)`, the small-amplitude limit of the
  BeamBeam kick (`|ΔQ_u| = ξ_u = N r0 β_u*/(4πγσ²)`, round). The `β/(4π)`
  coefficient is **derived** symbolically from the one-turn trace (`½Tr = cos μ −
  k1l β sin μ/2`), and the shift is validated **through a real ring** — inserting
  the linearised element into a FODO and reading `tunes()` reproduces `−βK/(4π)`
  with an O(K²) residual (quadratic-convergence check). Sign follows the kick:
  pp defocus ⇒ `ΔQ < 0`; LHC nominal `ξ ≈ 0.0037` per IP. See CONVENTIONS.md →
  *Beam-beam tune shift ξ*.
- ✅ **Low-β insertion** needed no new code: the IP waist `β(s) = β* + s²/β*`,
  `α(s) = −s/β*` is exactly the Stage-1 drift Twiss propagation around a zero-`α`
  point — pinned (both planes, waist-symmetric, `β` minimum at the IP) by
  `tests/analytic/test_low_beta_insertion.py`. Hourglass was out of scope here and
  landed later as **C2**; strong-strong / crab cavities / dynamic aperture remain
  out of scope.

## Stage 7 — Synchrotron radiation & radiation damping ✅ COMPLETE

The radiation the beam emits on its curved orbit: energy loss, the damping it
produces (transverse and longitudinal), and the quantum excitation that balances it
into an equilibrium emittance and energy spread. Delivered as `src/accsim/radiation.py`
(baseline core physics — numpy only, **not** gated). This was expansion axis **B1**,
chosen 2026-07-11.

- **Acceptance:** Robinson's theorem `J_x + J_y + J_z = 4` holds exactly; the
  isomagnetic energy-loss / integral closed forms match; the equilibrium emittance and
  energy spread scale as `γ²` / `γ`; and the whole set cross-checks against xtrack's
  radiation twiss. ✅ **MET** — `tests/analytic/test_radiation.py` (11 gates) and
  `tests/reference/test_radiation_xtrack.py`.

**Progress:**
- ✅ **Radiation integrals `I1..I5`** — `radiation_integrals(lattice)`
  (`RadiationIntegrals` dataclass), reusing the thick-dipole dispersion sub-slicing of
  `momentum_compaction` and the β-transport of `natural_chromaticity`. Pure sector
  bends (no combined-function gradient, no pole-face edge — Stage-1 scope), so
  `I4 = ∮ D_x h³ ds` and `I5 = ∮ curlyH |h|³ ds` with the dispersion invariant
  `curlyH = γ_x D_x² + 2α_x D_x D_x' + β_x D_x'²`. `I1 == α_c·C` is the independent
  within-baseline check on the dispersion transport; slice-converged.
- ✅ **Energy loss + partition numbers + damping times** — `energy_loss_per_turn`
  `U0 = (C_γ/2π)E⁴I2`; `damping_partition_numbers` `(1−I4/I2, 1, 2+I4/I2)` (Robinson
  exact by construction); `damping_times` `τ_i = 2E·T0/(J_i U0)` (**amplitude**
  convention — retroactively completes Stage 4, whose `quantum_lifetime` took the
  damping time as an input and can now source it from the lattice). Constants
  `C_γ = 4π r0/(3(mc²)³)`, `C_q = (55/32√3)ħc/(mc²)` computed from the reference species
  (electron `8.846e-5 m/GeV³`, `3.832e-13 m`).
- ✅ **Equilibrium emittance + energy spread** — `equilibrium_emittance`
  `ε_x = C_q γ² I5/(J_x I2)` (geometric); `equilibrium_energy_spread`
  `σ_δ = √(C_q γ² I3/(J_z I2))`. `I5` (curly-H) has **no clean absolute closed form**,
  so its analytic gate is the energy **scaling** (`ε_x ∝ γ²`, `σ_δ ∝ γ`, machine
  precision, since the integrals are pure geometry) + the xtrack absolute — stated as
  the gate, not a loosened tolerance (mirrors the Phase-2 A_FB magnitude handling).
- ✅ **xtrack cross-check** — `U0` and the convention-invariant `τ_y` match to
  `1e-4`/`2e-3`; `α_c`(=I1) to `1e-7`. Partition numbers (~1%) and `ε_x` (~3-4%) differ
  because xtrack's `radiation_analysis` uses the **damped one-turn-map eigenanalysis**,
  not radiation integrals (it exposes none); the two methods differ at that level in this
  strong ring (`I4/I2≈0.38`). accsim's integrals are independently pinned within-baseline
  (`I4=h²α_c·C` to `1e-10`; `I5` vs a `propagate_twiss` integration to `1e-6`), so this is
  a method difference, not a bug. See CONVENTIONS.md → *Synchrotron radiation*.
- **Flat-lattice scope:** `J_y ≡ 1` and equilibrium `ε_y ≈ 0` (no vertical bending or
  betatron coupling). Combined-function damping partition, edge/coupling `ε_y`, and
  intra-beam effects remain out of scope.

## Phase 2 (optional) — Collision event physics — both clauses + Delphes + hadronic Drell-Yan + Collins-Soper A_FB done

> **Milestone status:** clause (a) is analytically **met** (toy), clause (b) is
> **demonstrated end-to-end** (real Pythia chain), the canonical **Delphes**
> fast-detector step is **added** (`pipelines/ee_mumu_delphes/`, ILD @ 250 GeV,
> truth-vs-reco), the **hadronic (LHAPDF) Drell-Yan** extension is **added**
> (`pipelines/pp_mumu_drellyan/`, CMS @ 13 TeV, real proton PDFs, truth-vs-reco Z peak),
> and its **Collins-Soper `A_FB(m)`** angular observable — with the `pp` dilution made
> explicit — is now **added** too (user-requested; previously out of scope). Every named
> Phase-2 deliverable plus the CS `A_FB` extension is now built; whether to mark this
> optional phase formally *closed* remains a **user decision** — not marked ✅ unilaterally.

**Do not rebuild event generators.** Orchestrate the established chain: event
generator (Pythia / MadGraph) → fast detector sim (Delphes) → analysis in the
scientific-Python / ROOT ecosystem. A from-scratch toy 2→2 generator (matrix
element + RAMBO + PDFs) is welcome **as a clearly-labelled learning module only**.

- **Acceptance:** the toy generator's total cross-section for a known process
  matches the analytic value within Monte-Carlo error; the orchestrated pipeline
  runs end-to-end and produces a labelled distribution.
  - ✅ **Toy generator (acceptance clause a) — MET.** `accsim.events`: a labelled
    learning module for `e+ e- → μ+ μ-` (tree-level QED). Matrix element ×
    RAMBO flat phase space × MC integration; the MC total cross-section matches the
    analytic `σ = 4πα²/(3s)` (≈ 0.87 nb at √s = 10 GeV) within its Monte-Carlo
    error. Three analytic gates ordered **phase-space volume → dσ/dΩ shape →
    total σ** so a wrong `|M|²` and a wrong measure can't cancel; the `1/(8π)`
    2-body volume and `4πα²/(3s)` σ are sympy-derived, not remembered
    (`tests/analytic/test_toy_generator.py`). Process chosen leptonic (**no PDFs**)
    to keep the analytic gate clean. See CONVENTIONS.md → *Toy event generator*.
  - ✅ **Real orchestration (acceptance clause b) — DEMONSTRATED via Pythia8 in Docker.**
    `pipelines/ee_mumu_pythia/` drives an **established** generator (Pythia8 8.3),
    not the toy: `run_pipeline.py` starts a `hepstore/rivet-pythia` container,
    compiles a small C++ generator (`generate_pythia.cc`, process
    `WeakSingleBoson:ffbar2ffbar(s:gmZ)`, `e+e- → γ*/Z → μ+μ-` at √s=10 GeV),
    copies the `cos θ` data out, and `analyze.py` renders the **labelled
    distribution** on the host. Runs end-to-end in one command; the μ⁻ angular
    spectrum tracks the toy's `1+cos²θ` law (qualitative cross-check — **not** a
    σ-equality: all-flavour σ ≈ 6.15 nb vs the toy's 0.87 nb, plus QED FSR / fixed
    √s. The γ-Z forward-backward asymmetry is *measured* `A_FB = −0.0022 ± 0.0074`
    on 18k events — consistent with zero, i.e. **not** resolved at 10 GeV, so it is
    not claimed as a distinguishing feature). Docker is used because Pythia/Delphes don't
    build natively on Win/Py3.14 and there is no Windows pip/conda `pythia8`; a
    bind mount is avoided (spaced path) via `docker cp`. See
    `pipelines/ee_mumu_pythia/README.md`.
  - ✅ **Delphes fast-detector step — ADDED (`pipelines/ee_mumu_delphes/`).** The
    canonical generator→**fast detector sim**→analysis chain: Pythia8
    (`e+e- → γ*/Z → μ+μ-` at **√s = 250 GeV**) → **HepMC3** → **Delphes 3.5.0** with
    the **ILD** card (`scailfin/delphes-python-centos`, IRIS-HEP) → a **truth-vs-reco**
    `cos θ` distribution. 250 GeV (not the clause-(b) 10 GeV) because standard Delphes
    e+e- cards are only valid ≥ 91 GeV. The plot *shows the detector*: reco ⊆ truth
    (acceptance × ε ≈ 0.91), reco vanishes beyond the ILD `|η| < 2.4` edge
    (`|cos θ| = 0.984`) while truth reaches ±1, and above the Z the μ⁻ is
    forward-peaked (`A_FB ≈ +0.53`, *measured* — contrast the 10 GeV `A_FB ≈ 0`). The
    signal μ⁻ is isolated by an angle-neutral `|p| > 100 GeV` cut (status 23 is lost
    through the HepMC round-trip). Gated addon (`ACCSIM_ENABLE_DELPHES`). See
    `pipelines/ee_mumu_delphes/README.md` and CONVENTIONS.md → *Delphes detector step*.
  - ✅ **Hadronic Drell-Yan extension — ADDED (`pipelines/pp_mumu_drellyan/`).** The
    same generator→**fast detector sim**→analysis chain, now **hadronic**: Pythia8
    `pp → γ*/Z → μ+μ-` at **√s = 13 TeV** with a **real LHAPDF6 proton PDF**
    (`NNPDF31_lo_as_0118`, LO to match Pythia's LO ME; downloaded at run time) → **HepMC3**
    → **Delphes 3.5.0** with the **CMS** card → a **truth-vs-reco** di-muon
    invariant-mass spectrum. The deliverable is the canonical Drell-Yan **Z peak** at
    `M_Z ≈ 91.19 GeV`: the truth peak *mode* recovers `M_Z` to ~1 GeV (with an FSR
    low-side tail — not a clean Breit-Wigner), and the detector leaves two marks —
    **reco ⊆ truth** (`acceptance × ε² ≈ 0.36`, both muons must be in CMS acceptance) and
    a **modest peak broadening** (CMS muon momentum resolution, reco RMS > truth). The
    honest cross-check is `σ(DY×BR, 60<m<120) ≈ 1.5 nb`, matching the measured LHC value
    (~1.9 nb NNLO, LO ÷ K≈1.25) — a *real* PDF doing physical work. The resonance is
    forced to `Z→μμ`, so no τ→μ contamination and no `|p|` cut (leading OS pair suffices).
    Gated addon (`ACCSIM_ENABLE_LHAPDF`). See `pipelines/pp_mumu_drellyan/README.md` and
    CONVENTIONS.md → *Drell-Yan hadronic step*.
  - ✅ **Collins-Soper `A_FB(m)` — ADDED (the second deliverable of the DY chain).** The
    forward-backward asymmetry in the **Collins-Soper frame**, computed from the same
    truth/reco four-vectors by **one tested** function `accsim.events.collins_soper_costheta`
    (analytic gate: closed form == independent boost-into-rest-frame construction over 3000
    random pairs, `tests/analytic/test_collins_soper.py`; the `2/(Q√(Q²+Q_T²))` coefficient
    derived, not remembered). The physics gate is the **sign** (no clean closed form for the
    magnitude): `A_FB < 0` below `M_Z`, `> 0` above — measured below `−0.056 ± 0.007`, above
    `+0.108 ± 0.010` at 13 TeV / 100k (`SIGN GUARD: PASS`). The **`pp` dilution** is made
    explicit: `generate_hepmc.cc` emits the *true* incoming-quark `p_z` sign, so the
    **undiluted** `A_FB` (`+0.289` above pole) is overlaid on the `sign(Q_z)`-proxy diluted
    one (`+0.108`, factor ≈ 0.37); reco tracks the proxy (detector effect ≪ dilution). This
    was previously listed out of scope; it is now **built** (user-requested). See
    CONVENTIONS.md → *Collins-Soper A_FB*.

## Future expansion axes (candidate milestones — not started)

Directions the project could grow next, each written as a *candidate milestone*:
defined, as always, by its **analytic gate** (a direction without a closed-form
check is not worth building here — see the working agreement). Several are now done
(A1–A3, B1, C1, C2, D2, E2 — marked inline); the rest are unstarted, and picking one
promotes it to an open milestone and, where it overlaps *Out of scope* below, pulls
that item into scope. Ordered by proximity to what is already built,
not by priority. Effort tags are rough: **S** ≈ a session, **M** ≈ a few, **L** ≈ a
sustained arc.

### A. Drell-Yan angular physics (extends the Collins-Soper A_FB, Phase 2)

- **A1 — DY angular coefficients A₀–A₇ + the Lam–Tung relation.** ✅ **DONE (2026-07-11)**
  — decomposes the full Collins-Soper angular distribution
  `dσ/dΩ ∝ (1+cos²θ) + A₀·½(1−3cos²θ) + A₁·sin2θ cosφ + A₂·½sin²θ cos2φ + A₃·sinθ cosφ + A₄·cosθ`.
  Delivered: the CS **azimuthal φ*** sibling `collins_soper_angles` and a moment-projection
  extractor `angular_coefficients` (A₀–A₇), both in `accsim.events` (always-on baseline);
  machinery pinned by `tests/analytic/test_angular_coefficients.py` (moment closure,
  round-trip, quark-flip parity, `A_FB = 3/8·A₄`). **Gate met — the Lam–Tung relation
  `A₀ = A₂`** (exact at O(α_s), violated at O(α_s²)) proven *both* ways
  (`tests/analytic/test_lam_tung.py`): a closed-form symbolic proof from explicit
  Dirac-γ hadronic tensors (k² divides the A₀−A₂ numerator, remainder = 0) for `qq̄→Vg`,
  plus exact Gauss-quadrature confirmation (`qq̄→Vg` and `qg→Vq`, ~1e-14). The Pythia demo
  (`--angular-only`, 200k events) shows measured `A₀(q_T)≈A₂(q_T)`. See CONVENTIONS.md →
  *DY angular coefficients A₀–A₇ & Lam–Tung*. Built on [*Collins-Soper A_FB*].
- **A2 — sin²θ_W extraction from A_FB(m).** ✅ **DONE (2026-07-20)** — fit the measured
  `A_FB(m)` for the effective weak mixing angle, how LEP/LHC actually measure it.
  Delivered: `src/accsim/events/electroweak.py` (always-on baseline; numpy/scipy only —
  only the Pythia *data-producing* step stays behind `ACCSIM_ENABLE_LHAPDF`) with
  `neutral_current_couplings`, `afb_parton`, `afb_hadronic` (parton-luminosity weighted
  flavour sum) and `fit_sin2_theta_w`. The γ*/Z angular structure is **derived
  symbolically** from explicit Dirac-γ matrices with symbolic couplings, giving
  `dσ/dcosθ ∝ S(1+cos²θ) + 2D cosθ` and **`A_FB = (3/4)·D/S`**, `A₄ = 2D/S` — so the
  existing `A_FB = (3/8)A₄` anchor is reproduced *by construction*, tying the new model
  to the independently-validated A1 extractor. `_s_and_d` sums mediator **pairs**
  literally rather than hand-expanding `γγ + 2Re(γZ) + ZZ`, so no interference term can
  be dropped or mis-signed.
  **Gate met** (`tests/analytic/test_electroweak_afb.py`, 29 tests, layered so a wrong
  model and a wrong fitter can't cancel): module `S`/`D` matched term-by-term against
  the symbolic expression to `1e-12`; the CONVENTIONS sign gate (`A_FB<0` below `M_Z`,
  `>0` above) reproduced independently by the model; and a round-trip — sample from the
  model's own distribution → measure with the *real* `forward_backward_asymmetry` → fit
  back — recovering three injected values.
  **The "within fit error" trap was taken seriously**, since that phrasing is trivially
  satisfiable by inflating the error: the gate additionally asserts a unit-width **pull
  distribution** over 25 pseudo-experiments, an absolute cap `σ < 2e-3`, **χ² curvature**
  (a `1e-3` shift costs χ²≫1), starting-point independence, and a wrong-truth control.
  **Two things worth keeping.** (i) The generator ambiguity was real and is now closed:
  Pythia separates on-shell `sin2thetaW` from **effective** `sin2thetaWbar` (the one
  `A_FB` actually responds to) and neither was being set, so `generate_hepmc.cc` now sets
  both explicitly (`--sin2-theta-w`) and **reads them back out of Pythia** into
  `meta.dat` — the analysis reads truth from there, never a remembered default.
  (ii) A genuine bug: `scipy.optimize.least_squares` reports `success=True` when it
  converges *onto a bound* — a far-off start returned the window edge `0.45` with
  `χ² ≈ 6e6` as though it were a measurement. The fit now raises instead.
  **Scope, stated honestly:** the model is **LO** and lets the single parameter float in
  the `γ/Z` normalisation `κ` as well as the couplings; it fits the **undiluted** curve
  (the `pp` dilution correction is A3's job, kept orthogonal). The end-to-end fit
  against *generated* Pythia data is wired but **not yet run** — it needs LHAPDF/Docker,
  and a residual LO-vs-Pythia bias should be quoted rather than absorbed. See
  CONVENTIONS.md → *sin²θ_W from A_FB(m)*.
- **A3 — dilution unfolding.** ✅ **DONE (2026-07-20)** — recover the parton-level
  `A_FB` from the `sign(Q_z)`-proxy measurement. Delivered:
  `src/accsim/events/dilution.py` (always-on baseline, numpy only) with `parton_x`,
  `afb_diluted`, `dilution_factor`, `pdf_dilution` and `unfold_afb`, built on A2's
  validated `_s_and_d` rather than a re-derivation.
  **The physics in one line:** a mis-oriented event enters with `cos θ → −cos θ`, which
  flips the antisymmetric term and leaves the symmetric one alone, so **dilution
  reweights the numerator only** —
  `A_FB^obs = (3/4)·Σ_q(L_q⁺−L_q⁻)D_q / Σ_q(L_q⁺+L_q⁻)S_q` against the undiluted
  `(3/4)·Σ(L⁺+L⁻)D_q / Σ(L⁺+L⁻)S_q`. The denominator is untouched because a
  mis-oriented event is still an event.
  **Gate met** (`tests/analytic/test_dilution.py`, 13 tests), with the undiluted
  reference being A2's `afb_hadronic` — a different code path from the unfolding, so the
  two sides can't cancel. Layered: two exact limits (`L⁻=0` → `afb_hadronic` to `1e-15`;
  `L⁻=L⁺` → exactly zero), the formula closure (unfold → `afb_hadronic`, `1e-14`), and a
  **sampled MC closure** pushing real four-vectors through the actual
  `collins_soper_costheta` proxy and `forward_backward_asymmetry`, asserted as a pull
  (unit width over 12 seeds, max `|pull| = 2.8`) so an inflated error can't buy the pass.
  **The trap that would have made this vacuous:** with a *single* flavour the naive
  scalar divide is exact and the method goes untested. The toy proton therefore carries
  up **and** down with different valence hardness *and* different `A_FB`, and the suite
  asserts the flavour-blind `pdf_dilution` unfolding is **wrong by > 1e-3** on the same
  input while the correct one closes to `1e-14`.
  **Two things worth keeping.** (i) `D_eff` is **not** a PDF-only quantity: it carries
  the per-flavour `D_q` and so depends on `sin²θ_W` — the very parameter A2 fits from
  the unfolded curve (measured: a `0.2250 → 0.2380` shift moves it by up to `5e-2`). It
  is a systematic, or the fit should be iterated. (ii) `D_eff → 0` at central rapidity
  destroys the asymmetry outright rather than making it noisy, so those bins are masked
  to `nan`, never divided by.
  **Scope, stated honestly:** the luminosities are an *input* (the module never touches
  a PDF set, matching `afb_hadronic`'s `flavour_weights`), so the analytic gate runs on
  a toy proton. Reproducing the dilution against the Drell-Yan pipeline's own proxy/true
  ratio needs Pythia + LHAPDF and **has not been run**; the pipeline is unchanged. See
  CONVENTIONS.md → *pp dilution & unfolding*.

### B. Synchrotron radiation & radiation damping — a real "Stage 7" (accelerator core)

- **B1 — radiation integrals, damping, equilibrium emittance.** ✅ **DONE (2026-07-11)** —
  delivered as **Stage 7** (`src/accsim/radiation.py`); see the Stage 7 section above.
  Robinson exact, isomagnetic/energy-loss closed forms, `ε_x ∝ γ²` / `σ_δ ∝ γ` scaling,
  and the xtrack radiation cross-check all met; it completes Stage 4's `quantum_lifetime`
  (now sources the amplitude damping time from the lattice).

### C. Collider / beam-beam deepening (items explicitly deferred in Stage 6)

- **C1 — Bassetti–Erskine elliptical beam-beam kick.** ✅ **DONE (2026-07-20)** —
  generalises the round head-on kick (Stage 6) to `σ_x ≠ σ_y` via the complex error
  function. Delivered: the **same** `BeamBeam` element, now
  `BeamBeam(n_particles, sigma, sigma_y=None, strong_charge=1.0)` (always-on baseline;
  `scipy.special.wofz`), plus per-plane `strengths(ref) → (K_x, K_y)`; `matrix()` and
  `beam_beam_tune_shift` follow, so a flat beam gets an unequal `(ΔQ_x, ΔQ_y)`.
  **The stated gate was met but is *not sufficient*, and that drove the design.**
  "Reduces to the round `g(u)`" is a *singular* limit (`1/√(2(σ_x²−σ_y²))` blows up
  exactly there), and the classic Bassetti–Erskine error — writing `S_x + i S_y` for
  `S_y + i S_x` — **survives both the round limit and the on-axis values**, corrupting
  only the off-axis angular structure. So the gate was layered
  (`tests/analytic/test_beam_beam_elliptical.py`, 19 tests):
  the field is **derived symbolically from Coulomb's law** (`1/r² = ∫₀^∞e^{−r²t}dt`
  makes the Gaussian convolution elementary; sympy reproduces the `q`-integral with
  symbolic difference **exactly `0`**), the shipped closed form matches that derived
  integral, and both match an **independent brute-force 2D Coulomb integral** that never
  calls `wofz` — which is what pins the component assignment *empirically*. The round
  branch is the same integral's `w = 1/(q+σ²)` collapse, so both shapes are one
  derivation rather than two formulas.
  **The gates were mutation-tested**, not assumed: 8 deliberate bugs (swapped
  components, wrong `√(π/d)` coefficient, dropped damping term, missing aspect ratio in
  `z₂`, no sign folding, no tall-bunch axis swap, single-plane gradient) — 7 caught. The
  8th (arithmetic vs geometric mean in the round fallback) is **semantically null**:
  below the threshold the two differ by `O(eps²) ~ 1e-16`, under double precision, so no
  test *could* separate them. Recorded as such rather than papered over.
  **Two things worth keeping.** (i) The near-round folklore is wrong — `wofz` does *not*
  degrade catastrophically as `σ_x→σ_y`; accuracy is limited by **radius**, not
  ellipticity. The `1e-8` fallback threshold is **measured** (the round approximation's
  error is cleanly linear, `1.076·eps`), and exists to remove the exact-equality
  division by zero. (ii) **Gauss's law** (`K_x + K_y` = central charge density) is an
  independent anchor on the normalisation that the round limit alone cannot provide — it
  would absorb a stray 2 or π.
  **Scope, stated honestly:** `L_z` conservation is **genuinely lost** (the field is not
  radial, so it exerts a torque) — physical, not a defect, and the suite asserts the
  breakage so Stage 6's invariant is not over-claimed; curl-free survives, which is what
  symplectic tracking needs. `strength(ref)` now **raises** for an elliptical bunch
  instead of returning a misleading scalar. Hourglass / crossing-angle geometry *inside
  the kick* remains out of scope. See CONVENTIONS.md → *Elliptical Bassetti–Erskine kick*.
- **C2 — hourglass effect on luminosity.** ✅ **DONE (2026-07-20)** — the finite-`β*`/
  bunch-length luminosity reduction. Delivered: `hourglass_reduction(sigma_z,
  beta_x_star, beta_y_star=None)` in `accsim.collider` (always-on baseline,
  numpy/scipy), exact closed form `H = √π·a·e^{a²}·erfc(a)` (`a = β*/σ_z`) for a round
  waist, quadrature for `β_x* ≠ β_y*`.
  **Gate met** (`tests/analytic/test_hourglass.py`, 6 tests), layered so a wrong
  integrand and a wrong closed form can't cancel: the integrand is **derived
  symbolically** from the `ρ₁ρ₂` overlap — both the `e^{−s²/σ_z²}` weight and the
  waist factor *fall out* rather than being asserted — and the same derivation
  reproduces Stage 6's `1/(4π σ_x σ_y)`, tying the new factor to validated ground.
  On top: closed form vs quadrature over five decades of `a`, an **independent 2D
  `(s,t)` overlap** that never uses the `σ_z/√2` collapse (so a wrong collision-point
  width would not cancel), limits/monotonicity, the unequal-`β*` bracket, and the LHC
  nominal `H = 0.9907`.
  **Two things worth keeping.** (i) The *collision points* have rms `σ_z/√2`, not
  `σ_z` — both bunches must be there. Many references get this wrong; the symbolic
  derivation is what makes it not a remembered fact. (ii) `e^{a²}erfc(a)` overflows as
  `inf·0` for a short bunch, so it is coded with `scipy.special.erfcx`.
  **Scope, stated honestly:** `H` is **head-on** and does *not* factorise with the
  Piwinski `S` — a crossing angle couples the two integrals through the same growing
  `σ_x(s)`. The exact combined factor is a genuinely 2D integral and was **not**
  attempted; `luminosity()` is left unchanged and the caller applies `H`, rather than
  shipping `S·H` as if it were exact. See CONVENTIONS.md → *Hourglass effect*.

### D. Integration, validation & teaching (no new physics, high leverage)

- **D1 — end-to-end "build a machine" worked example.** [S] One narrated script/notebook:
  inject → accelerate (Stage 5) → store with radiation damping (B1) → collide (Stage 6) →
  account losses (Stage 4). **Gate:** each stage's existing analytic invariant still holds
  in the chained run — surfaces any seam between stages. Best done after B1.
- **D2 — tracking-based tune measurement (FFT/NAFF).** ✅ **DONE (2026-07-16)** —
  `src/accsim/tune.py` (always-on baseline: numpy/scipy only). Measures the tune the way
  a real machine does — track a particle, read the betatron frequency of its
  turn-by-turn record — as an independent route to `twiss.tunes()`. Delivered: `naff`
  (Hann-windowed Laskar NAFF: windowed-FFT peak → Brent refinement → **derivative
  root-find polish**), `ellipse_from_trajectory` (Courant-Snyder β/α from the
  trajectory's own covariance via `det Σ = J²`), and `tracked_tunes`.
  **Gate met and then some** (`tests/analytic/test_tracked_tune.py`, layered so a wrong
  estimator and a wrong lattice can't cancel): a *synthetic* tone recovers to `~1e-16`
  (no optics in the test), a known CS ellipse recovers to `1e-12`, and the integration
  gate — tracked tune == `tunes()` **mod 1** — lands at **~4e-15** vs the 1e-5 asked
  (asserted at 1e-10). Two design points worth keeping: β/α are taken from the *tracked
  data*, never from `twiss.py`, so a `match_periodic` bug can't corrupt both sides and
  cancel; and the `z = U − i·PU` sign was pinned **empirically**, not remembered. The
  derivative polish is what buys the last 7 digits — maximising a modulus by comparing
  values is capped at `√eps` (~1e-9). **Scope, stated honestly:** with `nonlinear=False`
  the tracking uses the *same* one-turn matrix `tunes()` is built from, so this
  validates the **extraction method**, not the map. The **symplecticity smoke test**
  the original entry called for already existed
  (`tests/analytic/test_tracking_stability.py`, `slow`) and was left alone. See
  CONVENTIONS.md → *Tracking-based tune / NAFF*.
- **D3 — MAD-X as a second reference** alongside xtrack. ✅ **DONE (2026-07-20)** —
  driven via **cpymad** (`tests/reference/_madx.py` + four `test_*_madx.py`), behind the
  existing `reference` marker. cpymad bundles the MAD-X binary and runs it in a
  subprocess, so unlike the xtrack JIT it needs **no build toolchain** — cp314 Windows
  wheels exist and it launches fine from this repo's space-containing path.
  **The gate's real content is the coordinate frame.** MAD-X is canonical
  `(x, px, y, py, T, PT)`: `PT` is an **energy** deviation where accsim's `delta` is a
  **momentum** one, and `T` scales oppositely to `zeta`. The transverse 4×4 compares
  entrywise, but the longitudinal row/column need
  `R_accsim = M·R_madx·M⁻¹` with `M = diag(1,1,1,1,β₀,1/β₀)`.
  **Both scale and sign were pinned empirically, never remembered.** The scale comes
  from a drift (MAD-X's `L/(β₀²γ₀²)` vs accsim's `R56 = L/γ₀²` — ratio exactly `β₀²`);
  the sign *cannot* be read off a drift, since its only non-zero longitudinal entry is
  even under flipping both `T` and `PT`, so it is fixed by the **dipole**, whose
  `R51`/`R52`/`R16` are odd under that flip.
  **Gate met:** drift, quadrupole and dipole 6×6 agree to **~2e-16** (whole matrix,
  longitudinal block included), and a matched **FODO-with-bends** ring agrees on β, α,
  μ, tunes and dispersion at `1e-9`. The ring carries dipoles on purpose — the
  bend-free xtrack cell has `D_x = 0` and `alpha_c = 0`, so comparing those would be
  comparing two zeros.
  **The longitudinal block was never dropped.** Comparing only the transverse 4×4 would
  have made every test pass while silently abandoning the `R56 = L/γ₀²` convention —
  precisely the error this gate exists to catch. Negative controls confirm teeth: a
  flipped transform sign gives `max|Δ| ≈ 4e-1` *and* breaks symplecticity; omitting the
  transform stays symplectic but fails entrywise at `4e-3`.
  **One honest disagreement, localised not tolerated.** MAD-X's `alpha_c` is exact;
  `momentum_compaction()` trapezoids the `D_x/ρ` integral and lands 1.6e-6 off. Slicing
  showed MAD-X stable and *accsim* converging at O(1/n²) — i.e. known quadrature error
  (already documented in the analytic suite), not a convention bug. So the test compares
  the **exact** identity to MAD-X at `1e-10` and then shows the quadrature converging
  onto MAD-X's number, upgrading that convergence check from self-consistency to
  agreement with an independent code.
  **Scope, stated honestly:** xsuite deliberately follows MAD-X's coordinate
  *conventions*, so a convention error the two share **by design** would still not be
  caught. What D3 buys is an independent *implementation* — an accsim arithmetic/sign
  error, or an xtrack bug, must now be reproduced by a separate Fortran codebase to
  survive. Sextupole (linear R is drift-like; `k2` enters only at 2nd order) and the
  radiation / synchrotron-tune checks were deliberately not mirrored. See
  CONVENTIONS.md → *MAD-X reference frame*.

- **D4 — make `momentum_compaction()` exact by default.** [S] *Surfaced by D3, deliberately
  deferred out of it (one feature per change).* The function trapezoids `∮D_x/ρ ds` at
  `slices=64` and is ~1.6e-6 off, while the **exact** identity
  `alpha_c = 1/γ₀² − (R51·D_x + R52·D_px + R56)/C` needs only the one-turn matrix and the
  matched dispersion — both already computed inside it. MAD-X (D3) and the analytic suite
  both confirm the identity is the right answer. **Gate:** the default path reproduces the
  identity to machine precision, `slip_factor`/`synchrotron_tune` still pass, and the
  quadrature stays reachable (it is the independent second route that keeps the identity
  honest — do *not* delete it, or the two cross-checks collapse into one).

### E. Event-physics siblings (new processes on the established chain)

- **E1 — W production + the W-mass Jacobian peak.** [M] Sibling to the Z chain; the
  neutrino escapes, so the observable is the **transverse mass** `m_T`. **Gate:** the
  Jacobian edge at `M_W` (Sudakov-smeared). Reuses the LHAPDF/Delphes orchestration.
- **E2 — jets / QCD: b-tagging performance against the card.** ✅ **DONE (2026-07-20)** —
  `src/accsim/events/btag.py` (always-on baseline: numpy only) + the `pp -> ttbar`
  pipeline in `pipelines/pp_ttbar_btag/` (behind **both** `ACCSIM_ENABLE_LHAPDF` and
  `ACCSIM_ENABLE_DELPHES`, default OFF). The **b-tag** branch was taken; the
  ATLAS-vs-CMS card comparison was **considered and rejected** — two detector outputs
  side by side have nothing to be refuted against, which fails this project's
  analytic-gate rule.
  **The gate's shape.** Delphes does not simulate a tagging algorithm: `BTagging`
  evaluates a per-flavour efficiency formula at the jet's `(pt, eta)` and sets a bit
  with that probability. The card therefore **is** the closed form — every jet has a
  known right answer. The formulas are **parsed out of the very card file Delphes ran**
  (`CMS_PhaseII_0PU`, chosen over `delphes_card_CMS.tcl` because it configures *three*
  working points on bits 0/1/2, making "the card's working points" plural and the
  ordering claim falsifiable). Never transcribed: a retyped formula is a remembered
  constant in disguise, and a typo in it would be invisible because both sides of the
  comparison would share it.
  **Gate met** — full run, 20 000 t̄t events / 132 988 jets: **χ²/ndf = 0.89 over 58
  bins** (σ = √(2/58) ≈ 0.19, so 0.6σ from unity), all three working points ordered in
  *both* coordinates (ε 0.756 > 0.593 > 0.408, mistag 0.0803 > 0.0082 > 0.0009), and
  ε_b > ε_c > ε_light per working point.
  **Two independent authorities, because the shipped ones are circular.** (i) The
  evaluator was checked against **Delphes' own `DelphesFormula`** over all 9 formulas on
  a 252-point grid landing deliberately *on* the card's step edges — **exact,
  0.000e+00 over 2268 points**, frozen into `tests/analytic/data/` so it gates in CI
  without Docker. (ii) `BTagging` keys on the same `Jet.Flavor` that
  `JetFlavorAssociation` writes, so histogramming one against the other validates the
  *handling* of the label but never its *definition*; an **independent** ΔR-matched
  label built from Pythia's own event record (no HepMC round-trip) agrees **0.968**
  overall (b 0.995, c 0.948, light/gluon 0.959).
  **Three things that were wrong first and are worth not re-learning:** the expected
  efficiency in a bin is the **jet-wise mean** of the formula, not the formula at the
  bin centre (a steeply falling spectrum makes that a quiet ~0.07 absolute bias); the
  pull uses the **expected** binomial variance, since the observed one is exactly zero
  — an infinite pull — in the zero-tag bins a ~0.1% mistag routinely produces; and bin
  validity gates on the **variance** `N·p·(1−p) ≥ 10`, not on jet count, because a bin
  can hold thousands of jets and still expect ~1 tag, which is Poisson and inflates χ².
  That last one is what moved the first real run from 1.90 to 0.67 — diagnosed as a
  broken *statistic* (only the lowest-p working point misbehaved) rather than a broken
  formula, and **not** by nudging a threshold.
  **Scope, stated honestly:** this is a **round-trip / consistency gate**, not a
  symbolic derivation like Robinson's theorem or `σ = 4πα²/3s` — the reference is a fit
  parametrisation the card encodes, so what is proven is that the extraction, flavour
  handling, binning and estimator are right. It is the **weakest analytic gate in this
  repo** and is labelled as such. The ROC is an **operating-point** ROC, not a
  continuous discriminant sweep: Delphes stores a decision bit and never a discriminant
  value, so a continuous ROC is not obtainable from it. Gates:
  `tests/analytic/test_btag_efficiency.py` (24 tests, synthetic jets + hand-written
  cards, no Docker; mutation-tested in two rounds, 13/13 caught). See CONVENTIONS.md →
  *b-tagging efficiency & the Delphes card* and `pipelines/pp_ttbar_btag/README.md`.

## Out of scope (unless a milestone explicitly calls for it)

Beyond even the expansion axes above — research-grade unless a milestone explicitly
pulls it in: Touschek / IBS, strong-strong beam-beam, crab cavities, wakefields,
higher-order modes, beam loading, full GEANT4, dynamic-aperture / frequency-map
studies, PDF-uncertainty bands, and research-grade machine design.
