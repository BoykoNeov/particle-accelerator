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
  `longitudinal_hamiltonian` (which assumed a *stationary* bucket symmetric about
  `zeta=0`) raised `NotImplementedError` for `sin φs ≠ 0` rather than return a
  plausible-wrong curve. **Superseded by D5**, which models the moving bucket; the
  guard is gone. Beam loading and transition crossing remain out of scope.
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

## Future expansion axes (candidate milestones)

Directions the project could grow next, each written as a *candidate milestone*:
defined, as always, by its **analytic gate** (a direction without a closed-form
check is not worth building here — see the working agreement).

**As of 2026-08-10 the delivered candidates are** A1–A3, B1, C1, C2, D1–D5, E1, E2,
**F1**, **F2**, **G1 in full** (betatron-coupling optics — skew quad, coupled
normal-mode tunes, closest-tune-approach `ΔQ_min` — *and* its ε_y vertical-emittance
half, the eigen-mode sharing, whose pre-committed coefficient was corrected by xtrack),
**G2** (Edwards-Teng coupled Twiss), **H1** and **H2** (tune/chromaticity and
insertion matching), **I1** (closed orbit and its correction, which made the
element map affine), **I2** (sextupole feed-down on a distorted orbit — the deferral
I1 named, which J1 was sequenced ahead of so that its gate would not be circular),
**I3** (the optics evaluated on that orbit, closing the gap I2 asserted), and **J1**
(the sextupole's nonlinear kick as a real map).
Each is marked inline with what it delivered and what it deliberately did not.
**As of 2026-08-10 there is no open follow-up on any axis below.**
A new milestone means writing a *new* candidate — either extending an
axis below or opening one — and, where it overlaps *Out of scope* below, pulling that
item into scope. Ordered by proximity to what is already built, not by priority. Effort tags are rough: **S** ≈ a session, **M** ≈ a few, **L** ≈ a
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

- **D1 — end-to-end "build a machine" worked example.** ✅ **DONE (2026-07-20)** —
  `examples/build_a_machine.py` (always-on baseline: numpy/scipy only) owns the machine
  and the narration; `tests/analytic/test_end_to_end.py` owns the gates. A 192 m, 24-cell
  **electron** FODO ring: inject 0.6 GeV → ramp (Stage 5) → store 2.0 GeV with radiation
  damping (B1) → collide (Stage 6) → account losses (Stage 4 + quantum lifetime).
  **The gate as written in this entry was the trap, and it was deliberately not built.**
  "Each stage's existing analytic invariant still holds in the chained run" is a
  tautology: every stage quantity is a pure function of one lattice, so
  `equilibrium_emittance(ring)` returns the same number here as in `test_radiation.py`.
  Re-asserting them is green forever. So the 17 gates are **seams only** — statements
  about what one stage hands the next — each written against the question *would this
  still pass if the value were recomputed from a fresh standalone lattice?*
  **The four seams.** (i) *Stage 5 → 7:* adiabatic damping shrinks geometric ε as
  `1/P0` while the radiation equilibrium grows as `γ²`, so `ε_adia/ε_eq ∝ 1/(β₀γ₀³)`
  **exactly** (machine precision over 1–5 GeV) — the composite no stage owns, and the
  `1/P0` half is read off the **tracked** ramp. (ii) *Stage 7 → 3/5:* `U0` sets `φ_s` on
  the **assembled** ring; both branches give the same gain, only one has a bucket, and
  the tracked NAFF synchrotron tune confirms it. (iii) *Stage 7+3/5 → 6:* `σ_z` is not
  an input — it is `σ_δ|η|C/(2πQ_s)` (radiation × RF × lattice) and reaches Stage 6
  through the hourglass; `L(eq)/L(injected) == ε_adia/ε_eq` exactly pins the luminosity's
  provenance. (iv) *Stage 7 → 4:* the same `ξ = A²/2σ²` drives the tracked aperture
  amplitude cut `1 − e^{−ξ}` and the quantum lifetime's exponent.
  **Two limitations of existing stages, surfaced and handled differently.**
  `synchronous_phase` keyed its stable branch on `η` alone — the `qV > 0` (proton)
  special case — and rejected the natural positive-voltage lepton ring outright. That
  **blocked** D1, so it was fixed first, in its own commit, with four gates and a
  proton-unchanged negative control. `rf_bucket_height`/`separatrix`/
  `longitudinal_hamiltonian` model only the **stationary** bucket, and a store ring
  replenishing `U0` has `sin φ_s = U0/(qV) ≠ 0`; that is a documented **scope limit**,
  so the acceptance is quoted from a stationary twin with the small parameter (1.9%)
  asserted alongside it. Moving-bucket acceptance stays out of scope.
  **The physics finding.** The **horizontal** CS action does not damp cleanly as `1/P0`
  through the ramp, and it is not ramp error: once RF and dispersion share a ring a loop
  closes that neither owns — `x → ζ` via `R51 x + R52 px`, `ζ → δ` in the cavity,
  `δ → x` via `D_x`. The residual is percent-level and does **not** shrink as the ramp
  slows. `D_y = 0`, so the vertical plane is free of it and its residual *is* the finite
  ramp rate (shown converging ∝ `1/n_turns`). Adiabatic gates therefore use the vertical
  plane, and the horizontal ripple is asserted to still be there.
  **Ratios cannot see a constant, so `σ_z`'s is pinned by tracking.** Every hourglass
  check is a ratio; dropping the `2π` would leave them all green. A particle at
  `(ζ, δ) = (0, σ_δ)` has `ζ_max/δ_max = |η|C/(2πQ_s)` off the nonlinear tracker,
  agreeing to **0.9%** at low `Q_s` with the residual being the known lumped-cavity
  `O(Q_s²)` error, shown shrinking.
  **Mutation-tested in three rounds; two real holes were found and closed** — the
  luminosity's hourglass was called with **swapped positional arguments** (`σ_z ≈ β*`
  here, so the swap is numerically plausible and every ratio test still passed; it is
  now asserted with *keyword* arguments), and the aperture could be sized off the
  *injected* beam with `ξ` unchanged (it is defined in sigmas, hence blind to which
  sigma — provenance is now asserted directly). Nine mutations, all caught after the fix.
  **Scope, stated honestly:** radiation damping is **closed-form, never tracked** (accsim
  has no damped or stochastic map), so "store with damping" is a data-flow handoff, not a
  simulated `ε → ε_eq` convergence; `β*` is a design parameter, not a matched low-β
  insertion; there is no vertical-emittance model, so `ε_y` is a coupling-fraction input;
  and `accelerate` is radiation-free and single-particle. See CONVENTIONS.md →
  *End-to-end chain (D1)*.
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

- **D4 — make `momentum_compaction()` exact by default.** ✅ **DONE (2026-07-20)** —
  *Surfaced by D3, deliberately deferred out of it (one feature per change).* The function
  trapezoided `∮D_x/ρ ds` at `slices=64` and was ~1.6e-6 off, while the **exact** identity
  `alpha_c = 1/γ₀² − (R51·D_x + R52·D_px + R56)/C` needs only the one-turn matrix and the
  matched dispersion — both already computed inside it. Now
  `momentum_compaction(lattice, slices=64, method="identity")`: the default is the
  identity (exact to machine precision, `slices` inert), and the trapezoid stays reachable
  as `method="quadrature"`. `slip_factor` / `synchrotron_tune` simply consume the now-exact
  default — `method` was deliberately *not* threaded through them (scope).

  **The trap this milestone was really about, and it is not the arithmetic.** Flipping the
  default silently converts every `momentum_compaction(lat) == identity(lat)` assertion
  into a tautology: the same code on both sides, green forever, testing nothing. Five
  assertions were in that state and are now explicit about the integral arm
  (`method="quadrature"`) — four in `tests/analytic/test_momentum_compaction.py`, plus the
  MAD-X coarse/fine convergence demonstration in `test_fodo_twiss_madx.py`, which would
  otherwise have compared MAD-X to the same exact number twice and quietly stopped
  demonstrating convergence at all. `radiation_integrals`' `I1` runs the same trapezoid, so
  `I1 == α_c·C` now asserts *both* arms: round-off against the quadrature (pinning the
  shared dispersion transport) and ~1e-5 against the exact default (the physics check, and
  the only one of the two that could catch a bug living in the shared machinery).
  The standing rule, recorded in CONVENTIONS.md: **the quadrature is not vestigial** — it
  touches the dispersion-generating matrix entries where the identity touches only the
  longitudinal row, so it is the independent second route. Delete it, or compare the
  default against the identity, and the two cross-checks collapse into one.

- **D5 — moving-bucket RF acceptance.** ✅ **DONE (2026-07-20)** —
  *Surfaced by D1, deliberately deferred out of it (one feature per change).*
  `rf_bucket_height` / `separatrix` / `longitudinal_hamiltonian` now model the
  accelerating (`sin φs ≠ 0`) bucket; the `sin φs` `NotImplementedError` guard is gone
  (the double-RF one stays). Always-on baseline — numpy/scipy only, no feature switch.

  **Height vs. area was the open question, and area was refused.** The scope note said
  "bucket area vs. `φs`", but every entry describing the actual debt names the three
  functions above and calls the missing piece the *overvoltage factor* — a **height**.
  Area is a non-elementary integral whose folklore form `(1−sin φs)/(1+sin φs)` is
  *itself* an approximation, so there is nothing to gate it against exactly; building it
  "for completeness" would have shipped the repo's first ungated number.

  **The closed form, derived symbolically from accsim's own `H`, holds on all four
  branches:** `δmax(φs)²/δmax(stationary)² = cos ψ − (π/2 − ψ)·sin ψ`, `ψ = asin|sin φs|`.
  The above-transition case is **not** the same function of `φs` — it is this function of
  `π − φs`; assuming the below-transition form transfers would have been wrong.

  **The real find, and it refuted the plan this milestone started from.** The handoff
  asserted `k_rf ζu = 2φs − π` was "already the general" unstable fixed point and only
  `separatrix`'s `±ζu` mirror needed fixing. It is **not general**: the unstable family is
  `k_rf ζ = 2φs + π + 2πn`, and the bucket is bounded by whichever of the two members
  *adjacent* to `ζ=0` gives the **smaller positive `δmax²`**. For `qV < 0` — an electron
  ring where a positive energy gain forces `sin φs < 0` — that is the other member, and the
  hardcoded one returns a silently **too-large** `δmax`. Lifting the guard without this
  would have mis-sized the acceptance of **exactly the machine D1 builds**. Found by
  numerics, then proved symbolically on all four branches; three fixes, one atomic commit.

  **The far turning point is transcendental, and is found, not assumed.** `separatrix`
  spans `ζu` to the other root of `U(ζ) = U(ζu)`; `U` is periodic-plus-tilt so the roots are
  many, but the right one is bracketed between `ζ=0` and the *other* adjacent unstable point
  (`U` monotonic there ⇒ unique sign change) and located with `brentq`. The stationary
  degeneracy is detected **relative to the bucket depth**, not against `0.0`: near that
  double root the level set is quadratic, so a root-find reaches only `√eps` — which is
  precisely how the symmetry test caught it (`0.09999999864` vs `0.1`).

  **Gate met** (`tests/analytic/test_moving_bucket.py`, 26 tests over all four branches),
  layered so a wrong fixed point and a wrong height formula cannot cancel: the ratio is
  compared to an expression **re-derived from `U` inside the test**, never to another call
  of the same code (D4's lesson); a **negative control** asserts the naive `2φs − π` is
  measurably wrong (>5%) on both `qV < 0` branches and right on both `qV > 0` ones; the
  separatrix is asserted genuinely **asymmetric** and to collapse to `±ζu` at zero gain; the
  `δ² ≤ 0` `ValueError` fires on the unstable root **for both signs of `qV`** (the two roots
  are indistinguishable by energy gain — asserted — so only stability separates them).
  D1's `test_moving_bucket_functions_raise` became a positive cross-check rather than being
  deleted.

  **The tracking leg twice gave a meaningless pass, and now self-guards.** The
  bounded/unbounded test is the only closed-form-free evidence, and on the electron branches
  1e4 turns covered **0.27 synchrotron periods** — an outside particle "stayed bounded"
  purely by not being tracked long enough to leave. The test now asserts `Qs·turns > 20`
  and `δmax < 0.05` before trusting either verdict.

  **`stationary_twin` is retired as a workaround** and kept only as the thing the real
  bucket is measured against: `examples/build_a_machine.py` now quotes the true moving
  acceptance, 1.46% shorter, with the reduction asserted against the closed form in
  `test_end_to_end.py` rather than the 1.9% small parameter being waved at.

  **Scope, stated honestly:** height only — **no bucket-area API**, for the reason above.
  The smooth (per-turn) Hamiltonian is unchanged, so the usual lumped-cavity `O(Qs²)` error
  applies; beam loading and transition crossing stay out of scope. See CONVENTIONS.md →
  *Moving-bucket acceptance*.

### E. Event-physics siblings (new processes on the established chain)

- **E1 — W production + the W-mass Jacobian peak.** ✅ **DONE (2026-07-20)** —
  `accsim.events.transverse_mass` / `jacobian_peak_pdf` / `jacobian_edge` (always-on
  baseline: numpy only) + the `pp -> W -> mu nu` pipeline in `pipelines/pp_W_mt/`
  (behind `ACCSIM_ENABLE_LHAPDF`), reusing the DY chain's Pythia8+LHAPDF → HepMC3 →
  Delphes-CMS orchestration wholesale.
  **The observable exists because the neutrino escapes.** In the Z chain both decay
  products are measured, so `m(mumu)` is reconstructible and the signature is a
  *peak*. Here `p_z^nu` is not recoverable even in principle, so there is **no
  invariant mass to build** — only `m_T^2 = 2 p_T^l p_T^nu (1 - cos dphi)`, whose
  distribution has a **Jacobian edge at `M_W`**.
  **The density was derived in sympy, not remembered:** back-to-back massless
  daughters give `dphi = pi` exactly and `p_T = (M/2) sin θ`, hence `m_T = M sin θ`;
  an isotropic `cos θ` then yields `dN/dm_T = m_T/(M sqrt(M^2 - m_T^2))` with CDF
  `1 - sqrt(1 - m_T^2/M^2)`.
  **Analytic gate met** (`test_transverse_mass.py` + `test_jacobian_edge.py`, 25
  tests): hand-computable configurations; the two exact symmetries (azimuthal
  rotation, and **longitudinal-boost invariance** — the reason `m_T` survives the
  unknown `qqbar` boost); the endpoint shown to survive both a large transverse
  recoil and a `V-A` weight; the shape vs the analytic CDF with the isotropy
  assumption stated. Six mutants (`1-cos -> 1+cos`, factor `2 -> 1`, `p_T -> p_z`,
  dropping either `sqrt`, `M^2-x^2 -> M^2+x^2`) are all killed.
  **The pipeline gate is a position, never `m_T <= M_W`.** That analytic bound holds
  for a *fixed* parent mass; Pythia's **Breit-Wigner** `W` legitimately produces
  `m_T > M_W` (**measured at 6.6%** of truth events). Asserting the
  bound would have failed on correct physics — or passed only behind a mass window placed right where
  the edge lives, hiding the effect. So E1 uses **no mass window** (unlike DY's
  `60..120`, which dodges the photon pole the charged current does not have), and
  gates on: truth edge within 5 GeV of `M_W`; reco edge measurably **rounder** than
  truth; truth `p_T^mu` edge within 5 GeV of **`M_W/2`**; and a loose reco-position
  band that catches a flipped reco MET. The tolerance is the measured estimator bias
  (~1.5) + binning (~0.3) + ISR recoil (~1) — **justified, not tuned** — and `M_W` is
  read back **out of Pythia**, never a remembered PDG constant.
  **Estimator: half-maximum of the falling edge, not `argmax`** (which is
  binning-jittery and sits ~1.5 GeV *below* the mass — asserted head-to-head). Its
  `+1 GeV + 0.73 sigma` bias is **tabulated and pinned by test**, not hidden; what
  makes it usable is that the offset is constant, tracking the true mass to
  `+1.55 ± 0.04 GeV` across `M = 60..100 GeV`.
  **Two conventions pinned empirically, in the D3 spirit.** Delphes' `GenMissingET`
  could have pointed along or opposite the neutrino (a `pi` shift that flips
  `1 - cos dphi` between `~0` and `~2`), so the macro emits **both** it and the
  summed truth neutrino and the analysis measures the angle — **100% aligned**,
  `sign = +1`, refusing to run on any other answer. And **muons are inside Delphes'
  `MissingET`** (`MissingET <- eflow <- TrackMerger <- MuonMomentumSmearing`),
  checked in the card rather than assumed — had they been excluded, MET would track
  the hadronic recoil and every reco `m_T` would be meaningless.
  **Gates met** on a 60k-event chain run: truth edge **81.41** vs `M_W` 80.385
  (**+1.03**), reco falloff **10.99** vs truth **2.24**, `p_T^mu` edge 42.91 vs
  `M_W/2` = 40.19. **Negative controls:** flipping the `GenMissingET` sign drops
  median `m_T` from 62.9 to **7.0 GeV** (edge 25 GeV off); feeding `p_T^mu` to the
  edge gate lands **35.8 GeV** off; flipping the reco MET sign drops median `m_T`
  to **9.4 GeV**. All fail.
  **The run re-derives its own motivation.** The `m_T` edge sits **+1.03 GeV** from
  `M_W` while the `p_T^mu` edge sits **+2.72 GeV** from `M_W/2` — the `m_T` edge is
  **2.7x better determined on the same events**, which is exactly the first-order
  ISR-recoil insensitivity that makes `m_T`, and not `p_T^l`, the `W`-mass
  observable. That was an input assumption of the design and came back out as a
  measurement.
  **Scope, stated honestly:** this *locates an edge*; it is not a `W`-mass
  measurement (which needs template fits, recoil calibration and PDF/QED systematics
  under 10 MeV). Not attempted: `W` charge asymmetry, recoil calibration, the
  electron channel, pileup. See CONVENTIONS.md → *Transverse mass and the W Jacobian
  edge* and *Jacobian-edge locator & the E1 pipeline*.
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

### F. Combined-function magnets & pole-face edges (core, deferred from Stages 1–2)

- **F1 — combined-function `Dipole` + pole-face edge focusing.** ✅ **DONE (2026-07-20)**
  — the bending magnet refinements deferred out of Stage 1 (`Dipole` was a pure sector
  bend). Delivered as three one-feature commits, all off by default so a plain sector
  bend is byte-identical:
  - **Pole-face edge angles `e1`/`e2`** (hard-edge): `R21 = +h·tan(e)` (horizontal
    defocus), `R43 = −h·tan(e)` (vertical focus), sandwiching the body
    `Edge(e2) @ Body @ Edge(e1)`. Sign/plane pinned empirically — whole 6×6 matches
    MAD-X `sbend` (`fint=hgap=0`) to **2e-16** and xtrack `Bend` (linear edge) to ~1e-6.
    Strongest gate: the **rectangular-bend identity** `e1=e2=θ/2` collapses the
    horizontal block to a drift (`R21=0`, proven symbolically). Hard-edge only; the
    vertical fringe correction stays out of scope.
  - **Combined-function gradient `k1`**: body `exp(L·A)` with `K_x=h²+k1`, `K_y=−k1`;
    branch-smooth dispersion integrals handle the removable `K_x=0` singularity (verified
    vs `expm`). Reduces to sector (`k1=0`, byte-identical) and to `Quadrupole` (`h=0`);
    matches MAD-X `sbend(k1)` ~1e-9 and xtrack `Bend(k1)` ~1e-6, both signs.
  - **`I4` damping partition** now carries the general
    `∮ D_x h(h²+2k1) ds − Σ_faces D_x h² tan(e)`. The `2k1` coefficient is pinned by a
    **closed-form smooth constant-gradient ring** (`J_x = n/(1−n)` exactly) *and*
    externally by MAD-X `synch_4` (~1%, decisively excluding the wrong coefficient which
    is ~9% off); the edge term is pinned against MAD-X `synch_4` (self-consistent for the
    sector-with-edges case). A strong gradient drives **`J_x < 0`** (horizontal
    anti-damping), a signature a sector bend can't fake.
  See CONVENTIONS.md → *Dipole — combined-function gradient*, *Dipole — pole-face edge
  focusing*, and the `I4` update under *Synchrotron radiation*.
  - **Dipole chromaticity — now shipped as F2** (was deliberately deferred here). See
    the F2 entry below.

- **F2 — full dipole chromaticity.** ✅ **DONE (2026-07-20)** — closes the gap F1 left:
  `natural_chromaticity` (and hence `chromaticity`) now carries the **whole dipole
  contribution** — weak-focusing `h²`, combined-function gradient `k1` **with its
  curvature-sextupole feed-down**, the dispersion corrections, and pole-face edges — on
  top of the quadrupole term. Derived from the exact curvilinear Hamiltonian; the
  β-weighted form is
  `Q'_x = -(1/4π)∮β_x(k1+h²) + (1/4π)∮h(γ_x D_x - 2α_x D_px) + (1/4π)∮2hk1 β_x D_x + (1/4π)Σβ_x h tan(e)`
  (`+β_y k1`, `+γ_y h D_x`, `-hk1 β_y D_x`, `-β_y h tan(e)` for `y`). See CONVENTIONS.md →
  *Dipole chromaticity*. **Fully xtrack-validated on sector, edged, and
  combined-function rings** (the combined-function case also agrees with MAD-X).
  - **The F1 trap resolved.** The naive `h²` weak-focusing term is large and negative,
    but the dispersion term — the `(1 + h D_x δ)` metric factor on the *dispersed* closed
    orbit — nearly cancels it, so a pure sector bend contributes almost nothing. That is
    exactly why the reverted F1 gradient-only patch was *worse* than omitting bends: it
    kept the cancelling partner's other half. `tests/analytic/test_dipole_chromaticity.py`
    asserts the partial (`h²`-only) fix is further from truth than omitting the dipole.
  - **The combined-function bug that a first pass shipped, then fixed.** The initial F2
    used a vector potential `ψ = -hx -(k1+h²)/2 x² + k1/2 y²` that **violates `∇·B=0` in
    the curved frame** (by `h·k1·y`), and framed the resulting combined-function mismatch
    as "model ambiguity, ship accsim's own model (Option C)." An adversarial review caught
    it: MAD-X and xtrack-*exact* agree to three figures (`+0.616`/`+0.617`), so `+0.616` is
    physical and accsim's `-0.365` was simply wrong. Maxwell forces a 3rd-order
    curvature-sextupole `ψ₃ = c₁x³ + c₂xy²` (`6c₁+2c₂+hk1=0`); pinning the split by the
    *horizontal* match gives `c₁=-hk1/3, c₂=hk1/2`, and the **vertical** coefficient then
    follows with no further freedom and matches xtrack — a non-circular confirmation.
    Feed-down `+2hk1 β_x D_x` / `-hk1 β_y D_x`. It does **not** change the linear map, so
    F1 stays validated.
  - **xtrack-validated (all cases).** Sector weak-focusing + dispersion to **~1e-6**,
    pole-face edges to **~1e-8**, and the combined-function AG ring (`k1=0.3`) `dqx/dqy` to
    ~1e-3 (`tests/reference/test_chromaticity_xtrack.py`: sector, edged, combined).
  - **Analytic gate** (`tests/analytic/test_dipole_chromaticity.py`, 11 tests): symbolic
    re-derivation of the full integrand incl. the Maxwell `(c₁,c₂)` (coefficients derived,
    not remembered); β-form == γ-form ring-identity equivalence (sector + combined);
    off-momentum-map self-consistency (sector, edges, combined); straight-combined dipole ≡
    quadrupole; and the partial-fix-is-worse reduction. `momentum_compaction(
    method="quadrature")` still sub-slices dipoles without `k1` (default `identity` is exact
    — low impact, unchanged).

### G. Betatron coupling & vertical emittance (core accelerator — closes a Stage-7 gap)

- **G1 — linear betatron coupling: skew quad + normal-mode tunes + `ΔQ_min`.**
  ✅ **CORE DONE (2026-07-21)** — the x-y coupling milestone, chosen after every prior
  candidate was complete. Delivered as three baseline features (always-on; numpy/scipy),
  one per commit, all analytically gated and cross-checked:
  - **`SkewQuadrupole` / `ThinSkewQuadrupole`** (`src/accsim/elements/skew_quadrupole.py`)
    — the coupling source, the exact hard-edge 45° roll of a normal quad
    (`[[A,B],[B,A]]`, `A=(F+D)/2`, `B=(D−F)/2`), symplectic by construction, `k1s=0` ⇒
    drift. The roll identity is the analytic gate (built directly, so the test has teeth);
    **MAD-X reproduces the whole 4×4 to ~2e-16**, confirming it exact, while **xtrack's
    `Quadrupole(k1s)` is a first-order-in-`k1s` model** (drift diagonal) — a documented,
    localised disagreement (D3-style), against which only the coupling sign is pinned.
  - **`normal_mode_tunes(lattice)`** — the coupled eigen-tunes (eigenvalues of the 4×4,
    symplectic-norm orientation), reducing to `tunes() mod 1` exactly when uncoupled. The
    uncoupled Courant-Snyder path is now **guarded** (`CoupledLatticeError`) so a skew
    lattice can't silently return decoupled-but-wrong betas/tunes.
  - **`closest_tune_approach(lattice) = |C⁻|`** — the milestone's **analytic gate**: the
    minimum mode-tune split at the difference resonance, `|C⁻| = (1/2π)√(β_xβ_y)|k1s·l|`.
    The `1/2π` prefactor is **derived** (exact eigen-split of a single-kick model,
    re-derived symbolically in-test), then triple-pinned — vs the exact eigenvalue gap
    on-resonance with **O((k1s·l)²) convergence**, the off-resonance hyperbola
    `√(Δ²+|C⁻|²)`, and the thick path gated separately; xtrack's coupled 4D Twiss
    reproduces the mode tunes (~1e-4) and `|C⁻|` (~3e-2).
  - Gates: `tests/analytic/test_skew_quadrupole.py`, `test_betatron_coupling.py`;
    reference `tests/reference/test_betatron_coupling_{xtrack,madx}.py`. See
    CONVENTIONS.md → *Betatron coupling*.
  - **ε_y follow-up ✅ DONE (2026-07-21).** The vertical-emittance half, delivered as
    `equilibrium_emittances_coupled(lattice) → (ε_1, ε_2)` (baseline; `radiation.py`),
    the eigen-mode sharing `ε_1 = ε_x0(G+Δ)/2G`, `ε_2 = ε_x0(G−Δ)/2G`,
    `G=√(Δ²+|C⁻|²)`, with `ε_x0` the coupling-off `equilibrium_emittance`, `Δ` the
    decoupled difference-resonance detuning, `|C⁻|` = `closest_tune_approach`. Sum
    conserved exactly; `ε_2` (the y-like mode) is the vertical emittance. Closes the
    Stage-7 flat-lattice gap (`ε_y≈0`).
    - **The pre-committed formula was wrong — xtrack settled it.** This entry originally
      committed `ε_y/ε_x = |C⁻|²/(|C⁻|²+Δ²)`. xtrack's radiation-envelope eigen-emittances
      (`eq_gemitt_x`/`eq_gemitt_y`) follow the **eigen-mode** ratio
      `ε_2/ε_1 = (G−Δ)/(G+Δ) → |C⁻|²/(4Δ²)` far off resonance — a factor of **4** below the
      committed form (the committed one is the raw `|C⁻|²/Δ²`; even the *projected*-emittance
      `½sin²2φ → |C⁻|²/(2Δ²)`, the beam **size**, is 2× the eigen-emittance a code reports).
      Empirically the shipped form matches `eq_gemitt_{x,y}` to **~1–3%** (weak-bend
      near-resonance ring), absolute and in the convention-invariant ratio; the roadmap form
      is refuted (~3–4× too large) in the reference test. The `1/4` asymptote is pinned
      symbolically; the coefficient itself is xtrack-anchored, not algebra (as the entry
      foresaw — "circular without the envelope").
    - **Scope (leading-order, equal-damping).** Clean only on a **weak-bend** ring: both the
      `ε_x0` integral-vs-envelope method difference (~3% weak → ~3× on 3×-stronger bends) and
      the skew-induced **vertical dispersion** (a second, uncarried `ε_y` source) grow with
      bend strength. Gates: `tests/analytic/test_coupling_emittance.py`,
      `tests/reference/test_coupling_emittance_xtrack.py`. See CONVENTIONS → *Betatron
      coupling → Vertical emittance from coupling*.
    - The full **radiation-envelope (Σ-matrix / Lyapunov) eigen-emittance** (option B) —
      dropping both the equal-damping assumption and the vertical-dispersion blind spot —
      remains reserved.
  - Deliberately **out of scope** here: solenoid coupling (body rotation + edge — messier
    gate), coupled Twiss / Edwards–Teng parametrisation, and the ε_y envelope (option B)
    unless pulled in. **The Edwards–Teng parametrisation was subsequently pulled in as
    G2** (see below); solenoid coupling and option B remain reserved.

- **G2 — coupled Twiss: the Edwards-Teng normal-mode optics.** ✅ **DONE (2026-08-10)**
  — closes the capability hole G1 opened. G1's `_require_uncoupled` guard is right
  (refuse rather than return decoupled-but-wrong optics) but left a skew lattice with
  **no beta functions and no beam sizes at all**. Delivered as three commits (two
  baseline features + the reference gate), always-on, numpy/scipy only:
  - **`coupled_twiss` / `match_periodic_coupled`** — factorises the transverse one-turn
    map `M4 = V U V^-1` with `V = [[γ_c I, C], [-adj(C), γ_c I]]`, `U = diag(A, B)`, so
    each betatron **normal mode** carries ordinary Courant-Snyder `(β, α)`. Exposes
    `γ_c`, the coupling matrix `C`, and `coupling_angle φ = arccos γ_c ∈ [0, π/4]`.
  - **`propagate_coupled_twiss` / `coupled_beam_sigma`** — normal-mode optics at every
    element boundary (by **local re-match** of `M(s) = T M₀ T⁻¹`, exact, no transport
    rule for `C` needed), and the **projected** beam sizes a screen sees,
    `Σ = V diag(ε₁B₁, ε₂B₂) Vᵀ`, plus the x-y ellipse tilt.
  - **The closed form was derived, and the recalled one was wrong.** The decoupling
    condition is the matrix Riccati `n + mX - Xq - XpX = 0` (`X = C/γ_c`), whose root is
    `X = λH`, `H = n + adj(p)`, `λ = -sgn(Δ)/(|Δ| + R)`, `R = √(Δ² + det H)`,
    `γ_c² = ½ + |Δ|/(2R)`. The remembered textbook `C` omitted a **factor 2** and broke
    the symplectic constraint `γ_c² + det C = 1` by O(1) (0.58 on a test ring) — caught
    *before* implementing, by testing the constraint numerically first. `λ` is
    re-derived symbolically in-test.
  - **Ties to G1 with no new coefficient:** `det C = sin²φ` matches the
    difference-resonance geometry `(1 - Δ/G)/2`, `G = √(Δ² + |C⁻|²)`, with `|C⁻|` from
    `closest_tune_approach`; `γ_c = 1/√2` exactly on resonance for *any* skew strength.
  - **Projected vs eigen made explicit.** `σ_y > √(ε₂β₂)` under coupling — mode 1 leaks
    into the vertical plane — which is *not* what `equilibrium_emittances_coupled`
    returns. Both now exist and the difference is gated. The leak is linear in `k1s`
    only for `|C⁻| ≪ Δ`; the saturation is asserted separately so the linear claim
    states its regime.
  - **Coupled dispersion for free:** the matched dispersion is solved from the full 4×4,
    so a skew at nonzero `D_x` gives **vertical dispersion** — **magnitude
    xtrack-validated** (`D_x,D_y,D_px,D_py` to `<2e-5` around a ring where `D_y` reaches
    ~0.07 m, no `β₀` factor, same sign), not just its `k1s`-linearity. G2 exposes it but
    does not feed it back into the ε_y sharing model — that stays with the reserved
    radiation-envelope (option B) work.
  - **xtrack-validated on all four Ripken betas** (`betx1 = γ_c²β₁`, `betx2 = (C B₂ Cᵀ)₀₀`,
    …): mode betas ~**1.7e-6**, the coupling-only cross terms ~**8e-5** (these pin `C`
    itself), at every boundary around the ring; the residual is xtrack's first-order-`k1s`
    skew model and **scales as `k1s²`** (factor 16.07 for a 4× stronger skew), asserted as
    its own test. Gates: `tests/analytic/test_coupled_twiss.py` (38),
    `tests/reference/test_coupled_twiss_xtrack.py` (5). See CONVENTIONS.md → *Coupled
    Twiss — the Edwards-Teng normal-mode optics*.
  - Still **out of scope**: mode phase advance / coupled tune from propagation (use
    `normal_mode_tunes`), solenoid coupling, and the ε_y radiation envelope (option B).

### H. Matching — asking the lattice for the optics you want (core accelerator)

- **H1 — tune & chromaticity matching.** ✅ **DONE (2026-08-10)** — the missing verb
  in "build a machine": D1 could *build* a lattice and report its optics, but there
  was no way to **ask** for `(Q_x, Q_y)` or `(ξ_x, ξ_y)` and get the strengths that
  deliver them. Shipped as two commits, always-on baseline (numpy/scipy only), in
  `src/accsim/matching.py`:
  - **`match_tunes` — two quad families, Newton on the β-weighted Jacobian.**
    `tune_response_matrix` gives `dQ_x/dv = +(1/4π) Σ w_i ∮β_x ds`, `dQ_y/dv =
    −(1/4π) Σ w_i ∮β_y ds` — the *same* first-order perturbation integral as the
    natural chromaticity, with the opposite sign because there the perturbation is
    `dk1 = −k1·δ`. Sign, `4π` and the per-member weighting are pinned against a
    symbolic `dQ/dv` differentiated from the thin one-turn map, which knows about
    none of the three.
  - **The Jacobian is first-order; the residual is exact.** `tunes()` supplies the
    residual, so the *fixed point* is exact — the gate asserts the recovered
    **strengths** against a known lattice (to 1e-9), not merely a small residual.
    Targets are full tunes, integer part included.
  - **`match_chromaticity` — two sextupole families, one exact linear solve.** Not
    an iteration, because a sextupole's linear map is a drift: `ξ` is *strictly
    affine* in `k2`, so `v = v₀ + S⁻¹(target − ξ(v₀))` lands exactly, from any
    start including `k2 = 0`. Asserted directly: `S` is identical at two different
    baselines (~1e-16), `ξ(v) − ξ(0) = S·v` for large arbitrary `v`, and the
    post-solve residual is **1.1e-16** — machine precision, not a tolerance.
  - **Ordering is a consequence, not a convention.** Match tunes then chromaticity,
    because sextupoles provably cannot move the tunes; a gate asserts the tunes
    survive the second match to 1e-13. Both matchers refuse the other's knob type
    *with the physical reason*.
  - **Knobs use MAD-X expression semantics** (`strength_i = w_i·v`) — the only form
    that handles both a family split into half-quads (weights `0.5`) and a family
    starting from zero. Newton backtracks over the stability boundary (where
    `tunes()` raises), degenerate knobs are refused on the condition number rather
    than solved, and a failed match rolls every strength back. No bounds and no
    `least_squares`, which sidesteps the "converged onto a bound, reported success"
    trap. The backtracking gate **counts** unstable excursions rather than only
    asserting convergence — measured first, because most starts never trip it.
  - **xtrack-validated on the matched machine** (not against xtrack's matcher —
    that would compare two optimisers): tunes to **4.0e-10 / 1.1e-9** including the
    integer part, total chromaticity to **2.4e-3 / 3.6e-4** on a correction of
    ~1.8. Gates: `tests/analytic/test_matching.py` (18),
    `tests/analytic/test_matching_chromaticity.py` (18),
    `tests/reference/test_matching_xtrack.py` (3). See CONVENTIONS.md → *Tune
    matching* / *Chromaticity matching*.
  - Still **out of scope**: general N-knob / M-target weighted objectives, strength
    bounds, β\* or dispersion matching at an insertion, and matching a coupled
    (skew) lattice — H1 is deliberately 2 families → 2 targets, twice. *(The first
    three are exactly what **H2** below then delivered.)*

- **H2 — insertion matching: local optics at a point, N knobs → M targets.**
  ✅ **DONE (2026-08-10)** — H1 could ask for the *tunes*; it could not ask for
  `β*` at the interaction point, which is the constraint a collider is actually
  designed around. `match_insertion` in `src/accsim/matching.py` (always-on
  baseline, numpy/scipy only) matches `beta_x/y`, `alpha_x/y` and the four
  dispersion components at any boundary point, with any number of quadrupole
  knobs against any number of targets.
  - **The waist condition is a quadratic, and the gate is built on that.** For
    the canonical line (waist `β₀` → drift `d₁` → thin lens `u = 1/f` → drift
    `d₂`), `α = 0` at the exit gives
    `(d₁²d₂ + d₂β₀²)u² − (d₁² + 2d₁d₂ + β₀²)u + (d₁+d₂) = 0`, **derived in sympy
    from `B → M B Mᵀ`**, never recalled. Two consequences are the milestone's
    physics content: `β*` is *determined, not chosen* (one knob buys a waist **or**
    a `β*`, so a one-knob/two-target problem is over-determined by construction),
    and the waist is *not unique* — two roots, two different `β*` (`u = 0.1361 →
    β* = 6.1495` and `u = 0.5404 → β* = 0.6505` on the gate's geometry). The gate
    matches from two starts and pins each root's **strength** to 1e-12; a test
    asserting "the" focal length would have flaked on the branch.
  - **Finite-difference Jacobian, and it is pinned rather than trusted.** Unlike
    H1's tune integral there is no universal closed form for a *local* β response,
    so the Jacobian central-differences the exact `propagate_twiss`. It is pinned
    the H1 way against a symbolic `dβ/dv` differentiated from the closed solution
    of a thin FODO: **7.9e-11** relative. Approximate Jacobian / exact residual
    still holds — a dispersion match lands at **1.4e-17**, machine precision.
  - **Two branches.** `twiss0=None` is periodic (the closed solution is re-solved
    every evaluation, so a quad moves the optics upstream of itself too);
    passing a `Twiss` is the transfer line, which is how a real insertion is
    matched — from the arc cell's exit Twiss into the IP.
  - **N ≠ M handled honestly.** `lstsq` gives the minimum-norm step for N > M and
    the least-squares step for N < M, but **a least-squares floor is not
    success**: convergence is declared only at `tol`, otherwise it raises naming
    every target and its miss. Over-determined ≠ unreachable, and the gate has
    both cases. Default target weights `1/max(|value|, 1)` keep a `β ~ 100 m`
    target from swamping an `α* = 0` one. Sextupole knobs are refused with the
    physical reason (their linear map is a drift).
  - **xtrack-validated on both branches** (not against xtrack's matcher): the
    line against xtrack's *open* twiss from the same entrance Twiss, the ring
    against its closed twiss, plus the **untargeted** y plane, which has no target
    to hide behind. All at **machine precision** (`β*` 8.9e-16 relative, `α*`
    8.5e-16 absolute, ring `β` 7.8e-16 / 6.7e-16) — nothing here is a first-order
    formula, unlike H1's chromaticity half.
  - **The finite-difference fallbacks are gated directly**, because no other test
    in the suite can see them: with an exact residual the fixed point is right
    however wrong the Jacobian is, so a wrong one-sided denominator would only
    cost iterations. A knob parked `5e-7` inside the thin FODO's `trace = 2` limit
    drives one trial point across it; the weight sign selects which side dies.
    The column is asserted against the **exact** one-sided quotient — near the
    boundary β diverges and the truncation error runs to 58–86 %, *larger* than
    the ×2 a `2h` denominator would introduce, so a finer difference could not
    tell the bug from the truncation.
    Gates: `tests/analytic/test_matching_insertion.py` (32),
    `tests/reference/test_matching_insertion_xtrack.py` (4). See CONVENTIONS.md →
    *Insertion matching*.
  - Still **out of scope**: strength bounds, phase-advance targets, matching a
    coupled (skew) lattice, and targets spanning several lattices (a shared
    insertion matched from both sides).

### I. Closed orbit — where the beam actually is, and steering it back (core accelerator)

- **I1 — closed-orbit correction: dipole correctors, response matrix, SVD steering.**
  ✅ **DONE (2026-08-10)** — every optics quantity up to here (β, Q, ξ, D, and both
  matching milestones) describes motion *about* the design orbit and assumes the beam
  is on it. A real machine never is, and steering it back is the most common
  operational task in an accelerator. Shipped as two commits, always-on baseline
  (numpy only): `Corrector` in `src/accsim/elements/corrector.py` and
  `src/accsim/orbit.py` (`closed_orbit`, `propagate_orbit`, `orbit_response_matrix`,
  `correct_orbit`).
  - **It needed a new contract on `Element`, and that is the milestone's structural
    content.** A corrector's action is *not* a matrix: the same angle for every
    particle is inhomogeneous and cannot appear anywhere in a 6×6 acting on the
    state. `Element` gained a concrete `kick(ref)` (zero for every element but a
    corrector), `track()` became affine, and `Lattice.transfer_map()` composes
    `(M₂M₁, M₂k₁ + k₂)` — so a kick is transported by everything *downstream* of it.
    The composition gate uses **two** correctors at different places on purpose: with
    one kick only a single term exists, so a transposed composition still produces a
    perfectly plausible closed orbit.
  - **The solve is a reuse, not new machinery.** `(I − M₄)x_co = k₄` is literally
    `_matched_dispersion`'s `D = (I − M₄)⁻¹d` with the corrector kicks in place of the
    map's `δ` column — dispersion *is* the closed orbit of an off-momentum particle.
    The textbook `θ√(β_kβ)/(2 sin πQ)·cos(Δψ − πQ)` is a consequence, derived in sympy
    and used only as the independent reference; the singular `I − M₄` is the **integer
    resonance**, gated on a real focusing lattice walked to `Q_y = 0` exactly (where
    the orbit has already grown 3100×) rather than on the degenerate no-focusing case.
  - **Correction is one exact linear solve.** The closed orbit is strictly affine in
    the kicks, so the response matrix is exact at any amplitude (gated at 0.3 rad) and
    `correct_orbit` needs no iteration — the same structural fact that makes
    `match_chromaticity` an exact solve. Two correctors annul a steering error
    completely *outside* the resulting bump (2.2e-19 at 14 monitors against 2 knobs;
    over-determined ≠ unreachable), and a gate asserts the orbit inside the bump is
    deliberately **not** zero, because no corrector can undo what happened upstream.
  - **`rms_after` is measured, never predicted** — a fresh closed-orbit solve of the
    corrected lattice. This is load-bearing because `correct_orbit` accepts a supplied
    `response`: an operational machine *measures* its response matrix. Evaluating
    `x0 + R·dθ` would report a perfect correction for any invertible `R`, right or
    wrong. Gated by handing it `1.5R` and asserting the reported residual is
    `rms_before/3` while the unused prediction is `< 1e-16`.
  - **SVD truncation is gated non-vacuously.** With N = M the plain solve is exact and
    truncation is never exercised, so the suite carries N > M (minimum-norm, verified
    against a null-space alternative) and N < M, plus a pair of correctors split by a
    1 mm drift (σ₁/σ₂ = 3157) where the untruncated answer — which *is* the better fit
    — asks for **0.66 rad** of steering to buy 32 %, and `n_singular=1` asks
    **6.8e-5 rad**: a norm ratio of 9752.
  - **The sign lives only in the reference suite**, per G1: every analytic reference
    for it is one accsim also derives. Established by probe —
    `Corrector(kick_x=+k) ≡ xt.Multipole(knl=[−k])` but `kick_y=+k ≡ ksl=[+k]`, the
    MAD-X normal/skew asymmetry — with a test asserting the other choice is decisively
    wrong. xtrack's *iterative* closed-orbit search agrees with the closed-form solve
    to **1.9e-15 m** on a 1 mm orbit, and independently confirms the corrected machine
    is flat outside the bump and still bumped inside it.
    Gates: `tests/analytic/test_orbit.py` (27),
    `tests/analytic/test_orbit_correction.py` (22),
    `tests/reference/test_orbit_xtrack.py` (5). See CONVENTIONS.md →
    *Closed orbit & its correction*.
  - **The affine map made loss tracking faster, not slower.** `track_bunch_losses`
    walks the elements itself, so it needed the kick separately; hoisting the
    per-element map out of the turn loop at the same time took 2000 turns × 31
    elements × 200 particles from **1.416 s to 1.021 s**. A zero kick is skipped
    rather than broadcast, and a gate steers a bunch into a collimator it
    otherwise clears, so that path is not merely fast but tested.
  - Deferred to **I2** (below, now done): sextupole feed-down on a distorted orbit —
    what makes "correctors do not move the optics" a linear-order statement. Still
    **out of scope**: misalignments as element attributes rather than explicit
    correctors, coupled (x-y) correction, local orbit bumps as a first-class API,
    corrector strength limits, and BPM noise / calibration errors.

- **I2 — sextupole feed-down on a distorted orbit.** ✅ **DONE (2026-08-10)** — the
  deferral I1 named by name, and the claim it qualified. Expanding J1's kick about an
  orbit offset splits **one sextupole into four elements**, each equal to one accsim
  already validates: a `Corrector` `θ_x = −½k2l(x_co²−y_co²)`, `θ_y = +k2l·x_co·y_co`;
  a `ThinQuadrupole` `k1l_eff = +k2l·x_co`; a `ThinSkewQuadrupole`
  `k1sl_eff = +k2l·y_co`; and the sextupole, unchanged. Every coefficient is derived
  in sympy. New (baseline, numpy only): `closed_orbit_nonlinear`,
  `propagate_orbit_nonlinear`, `linearised_element_maps`, `linearised_one_turn_map`,
  `OrbitConvergenceError`, and a `nonlinear` flag on `correct_orbit`.
  - **The orbit stops being a solve.** `θ_x` depends on the orbit it displaces, so
    the closed orbit is the fixed point of a *nonlinear* map, not `(I−M₄)x = k₄`.
    Newton runs on the **4D** subspace, seeded from `closed_orbit`: without RF there
    is no longitudinal fixed point at all (`R56` leaves `zeta → zeta + const`, so
    `J−I` is exactly singular there), and the seed makes "linear lattice ⇒ I1's
    answer at round-off" a free gate.
  - **The lead gate is the dipole, because the tune shift would be a rerun.** J1
    already measured `k1l_eff` by finite-differencing `track()`, so `ΔQ_x =
    +β_x k2l x_co/(4π)` is demoted to a consistency check (made non-vacuous with four
    sextupoles at two `β_x` and alternating-sign `k2l`, `|Σ| < 0.6Σ|·|`). The headline
    is instead that the orbit *departure* equals I1's linear response to the derived
    `θ` — and the content is the **order**: over four steerer sizes the departure
    falls by **4** per halving (`O(x_co²)`) while the residual falls by **8**
    (`O(x_co³)`). J1's consistently-mis-scaled sextupole, which passes every
    structural check, is caught as a clean factor of 2.
  - **A discriminator J1 structurally could not have**: `θ_x/k1l_eff = −x_co/2` is
    pure geometry, with no `k2l` in it. A gradient alone fixes the product `k2l·x_co`
    and never the split between the two terms.
  - **The vertical orbit is the strongest non-rerun gate.** A *normal* sextupole at
    `y_co ≠ 0` is a **skew** quadrupole — betatron coupling, reaching G1/G2's
    machinery from a direction nothing in the package had taken. `y = py = 0` is an
    **exact** invariant subspace (asserted at zero, not to tolerance), so a purely
    vertical steerer moving the *horizontal* orbit via `θ_x = +½k2l·y_co²` — with the
    opposite sign, the `x²−y²` structure in the orbit — is unambiguous, and flatly
    impossible linearly (xtrack reports exactly `0.0` at `k2l = 0`).
  - **β at the source is solved exactly, not recalled** (the G2 trap): the test ring
    is a palindrome, so `α = 0` at the sextupole and the 2×2 gives
    `(β′/β)² = sin²μ/(1 − (cos μ − k1l β sin μ/2)²)`. Compared **squared** because
    `M12 = β sin μ` is negative in this ring, which keeps it exact instead of
    asserting a `sin μ` branch.
  - **The operational punchline: correction becomes a loop — a *linearly* convergent
    one.** One pass leaves `O(k2l·x_co²)` instead of I1's machine zero. The
    contraction factor is **constant at 4.951e-4** (identical to 4 digits over three
    passes) because `R` is rebuilt from the *linear* model every time and never learns
    the feed-down gradient — a stale-Jacobian iteration, where a true Newton would be
    quadratic. The test asserts the factor *repeats*, which is sharper than "it
    shrinks". `nonlinear=False` stays bit-for-bit I1, blind spot and all.
  - **Two honest non-claims, gated.** Feed-down is **self-limiting** (10⁵× `k2l`
    *shrinks* the orbit), so convergence is **not** evidence of a stable machine; and
    the fixed point is **not unique** — from 50 m out Newton finds a different,
    outer orbit. Both are asserted rather than assumed away.
  - Gates: `tests/analytic/test_feeddown.py` (24),
    `tests/reference/test_feeddown_xtrack.py` (6). xtrack's own iterative nonlinear
    closed-orbit search agrees to a **measured 1.6e-13 m** at all boundaries with both
    planes steered (tolerance `1e-12`, 6.4× headroom), and its `R_matrix` matches
    `linearised_one_turn_map` to a measured **2.1e-11** on entries up to 7.0 (normal
    *and* skew blocks; tolerance `1e-9`, 47× headroom) — a floor that belongs to
    xtrack's differencing, since varying accsim's step moves it by 0.1 %. I1's linear
    solve misses xtrack by `>1e4×` that, which is what makes the agreement mean
    something. See CONVENTIONS.md → *Sextupole feed-down on a distorted orbit*.
  - Still **out of scope**: the 6D (RF-coupled) closed orbit, feed-down from
    octupoles and higher multipoles, misalignments as element attributes,
    amplitude-dependent detuning / resonance driving terms, and dynamic aperture.
    Named explicitly because the milestone statement raised it: **`chromaticity()`
    is not feed-down-corrected** — it is a design-orbit quantity built on on-axis
    `elem.matrix()`, so a steered machine carries the β-beat error I2 measures.
    `linearised_element_maps` supplies what a corrected version needs; the Twiss
    propagation on top of it is not built here, and a test asserts the blind spot.
    **That is I3, below.**

- **I3 — the optics evaluated on the real (steered) orbit.** ✅ **DONE (2026-08-10)**
  — the gap I2 named and asserted. Every optics function walked each element's
  *on-axis* `matrix()`, so a steered machine was reported with unperturbed β,
  dispersion and chromaticity however far off-axis the beam was. I3 builds the Twiss
  propagation on top of I2's per-element maps. Baseline (numpy only): a `maps=`
  argument on `propagate_twiss` and `propagate_coupled_twiss`, then
  `closed_twiss_on_orbit`, `propagate_twiss_on_orbit`, `tunes_on_orbit`,
  `coupled_twiss_on_orbit`, `propagate_coupled_twiss_on_orbit`,
  `natural_chromaticity_on_orbit`, `chromaticity_on_orbit` and
  `orbit.linearised_lattice`; plus `delta=` threaded through `closed_orbit`,
  `closed_orbit_nonlinear`, `propagate_orbit_nonlinear` and the linearised-map pair —
  **one Newton solver, not two**, its linear seed carrying the dispersive column.
  - **The gate is β(s), not β(0).** I2 already gates the one-turn map, and "the
    propagated table multiplies back to it" is *vacuous* — that map **is** the
    product of the per-element maps. The new content is the `s`-dependence, gated on
    the single-gradient closed form `Δβ/β = −Δk1l·β(src)·cos(2|Δψ| − 2πQ)/(2 sin 2πQ)`,
    **derived in sympy** against accsim's own `ThinQuadrupole` *and* its own Twiss
    propagation rule, with the `sin μ` square-root branch avoided entirely (the ring
    runs where `sin 2πQ < 0`). Over four steerer sizes the beat falls by **2** per
    halving and the residual by **4**; doubling the predicted gradient is caught 20×.
    The sextupole sits **mid-ring** so the `|Δψ|` branch is actually exercised, and
    `Q_y` sits 0.059 from the half integer *on purpose*, making the vertical the
    demanding plane.
  - **Chromaticity takes the other route, and the package's own structure decides
    it.** accsim's linear maps carry no `δ` dependence, so linearising the tracked
    map about the off-momentum orbit measures the sextupole feed-down term and is
    **exactly blind** to the natural chromaticity (measured: `3.7e-8` against a true
    `−0.29`). Implementing this by tracking alone would silently drop that whole
    term, and a test asserts the reason. So the F2-validated integrals run on
    `linearised_lattice` — I2's derived split, with the sextupole kept — and the
    tracked route is retained as the **independent** gate on the half it can see,
    via the difference `chromaticity_on_orbit − natural_chromaticity_on_orbit`.
    Agreement `2.2e-8` on values of order 2, **flat** in the orbit offset. Zero
    steering is bit-for-bit identical to `chromaticity()`.
  - **Scope lines enforced, not merely documented.** Thick sextupoles raise
    `NotImplementedError` (a thick body's offset varies across it — the `O(L²)` error
    I2 avoided), though `propagate_twiss_on_orbit` handles them fine because it
    differentiates the real `track()`. A vertically steered machine is genuinely
    coupled, so the uncoupled entry points raise `CoupledLatticeError` and
    `coupled_twiss_on_orbit` is the path — against a design optics whose coupling is
    exactly zero. `tunes_on_orbit` returns the accumulated phase, integer included,
    which removes the `δ`-difference wrap hazard rather than guarding it.
  - **One modelling difference is measured rather than absorbed.** accsim's `Dipole`
    and `Quadrupole` are *exactly linear*, so an off-axis orbit changes nothing about
    them; xtrack's are exact nonlinear maps whose Jacobian at 1.25 mm is not the
    on-axis one. At `k2l = 0` accsim's on-orbit optics equals its design optics to
    `4e-11` and xtrack's β has still moved `6.4e-4`. First order in the orbit, owned
    by accsim's element models rather than by I3, isolated in its own test — and the
    reason the β cross-check is a **with-minus-without-sextupole difference**, where
    that term cancels. xtrack puts the sextupole-induced Δβ at `6.6e-3 m` (0.22 %);
    accsim reproduces it to `1.35e-3` of the effect while the design orbit answers
    **exactly zero, bit for bit**. Chromaticity needs no difference: `3.3e-5`/`1.3e-4`
    against `1.7e-3`/`2.6e-3`, i.e. 52× and 21×.
    Gates: `tests/analytic/test_orbit_optics.py` (20),
    `tests/reference/test_orbit_optics_xtrack.py` (4). See CONVENTIONS.md →
    *Optics on the real (steered) orbit*.
  - Still **out of scope**: off-axis feed-down from accsim's own linear elements
    (the bend/quad nonlinearity above — a candidate milestone of its own); coupled
    Edwards-Teng chromaticity on a vertically steered machine; thick-sextupole
    chromaticity on orbit; and everything I2 listed — the 6D closed orbit,
    octupoles and higher multipoles, misalignments as element attributes,
    amplitude-dependent detuning, dynamic aperture.

### J. Nonlinear single-particle dynamics (core accelerator)

- **J1 — the thin nonlinear kick becomes real.** ✅ **DONE (2026-08-10)** — every
  element up to here acted through an affine map, so the sextupole carried `k2` but
  had no map at all: its `matrix()` is a drift and the strength was felt only
  through feed-down chromaticity. J1 gives it the kick
  `Δpx = −½k2l(x²−y²)`, `Δpy = +k2l·x·y`, exactly in `ThinSextupole.track()` and by
  drift–kick–drift in `Sextupole.track()` (`n_slices`). Always-on baseline (numpy
  only). `matrix()` is unchanged in both — the kick has no linear part at the
  origin, which is the physics rather than an omission.
  - **The milestone's content is which gate pins the ½, not the kick itself.** The
    obvious structural checks are *all* blind to the coefficient, because a
    mis-scaled sextupole is still a sextupole: symplecticity at any amplitude (true
    for any gradient kick, any strength), the curl-free/Maxwell condition (the
    *same* statement for a thin kick, not an independent one), identity Jacobian at
    the origin, small-amplitude tracked tunes, and a sympy derivation from a
    potential reverse-engineered out of the kick. A table of them is in
    CONVENTIONS.md, and the suite carries a **consistently mis-scaled** sextupole
    (`1` for `½` in *both* components) that passes every one of them.
  - **What discriminates is feed-down.** Linearising the new nonlinear map about the
    off-momentum closed orbit (Newton on the tracked map, then its FD Jacobian) and
    reading `dQ/dδ` off the result recovers a quantity accsim already computes by a
    completely different route — `chromaticity − natural_chromaticity`, pinned
    symbolically and cross-checked against xtrack's real tracking at `rel ≈ 5e-4`.
    Since accsim's linear matrices carry no `δ` of their own, the whole
    `δ`-dependence of the tracked tunes *is* the feed-down. Agreement `rel 1e-5`;
    the mis-scaled sextupole is caught as a clean **factor of two**. Scaling only
    `Δpx` is the other class of bug — no longer a field at all — and symplecticity
    *does* catch that one.
  - **The coefficient is derived from the field, anchored on the quadrupole.** The
    `½` is the `1/n!` of `B_y + iB_x = (Bρ)Σₙ kₙ(x+iy)ⁿ/n!`, the same expansion
    whose `n = 1` term is the `Quadrupole` already validated against xtrack *and*
    MAD-X — so the derivation borrows its credibility from an independently
    validated convention instead of from its own algebra.
  - **The sign is a probe, and the thick element is compared by difference.**
    `ThinSextupole(k2l) ≡ xt.Multipole(knl=[0,0,+k2l])` **bit-for-bit** (exact
    because a thin kick has no length, so there is no drift model to disagree
    about); the opposite sign misses by exactly twice the kick. A raw thick
    comparison instead leaves `~1e-8` that is **present unchanged at `k2 = 0`** —
    it is `−L·px·δ`, xtrack's exact drift against accsim's linear one, and belongs
    to the drift. Toggling `k2` isolates the nonlinear content (`rel 1e-3`).
    ⚠️ `n_slices` converges accsim onto the *exact* map, **not** onto xtrack, whose
    thick sextupole is itself a single-kick split — so `n_slices = 1` is its closest
    match and a "raise slices until they agree" gate would have read a modelling
    choice as a bug.
  - **Approximations flagged.** Drift–kick–drift is symplectic exactly but only
    second order: `O(L³)` per slice at fixed `k2`, `O(1/n_slices²)` overall (both
    ratios *measured*, 0.125 and 0.25). At fixed *integrated* `k2l` the `k2²L³` term
    is `k2l²L`, so the thin-lens limit is approached only **linearly** — a short
    thick sextupole is not a thin one.
  - **Tracking-path fallout.** `Tracker.track`/`track_turns` still default to
    `nonlinear=False`, which now *silently drops* a sextupole's kick — asserted in
    the suite so it is documented rather than discovered.
    `track_bunch_losses` gained a `nonlinear` flag (its hoisted matrices would
    otherwise linearise a sextupole), and `RFCavity.track` was vectorised over a
    bunch to make that path work at all. `accsim.symplectic` gained `jacobian()` and
    `is_symplectic_map()`.
    Gates: `tests/analytic/test_sextupole_kick.py` (21),
    `tests/reference/test_sextupole_kick_xtrack.py` (7). See CONVENTIONS.md →
    *The sextupole's nonlinear map*.
  - Still **out of scope**: dynamic aperture and frequency maps (nonlinear tracking
    against apertures is now *possible*, and nothing gates it), amplitude-detuning
    and resonance-driving-term closed forms (sextupole detuning is second order in
    `k2`; no coefficient is claimed), octupoles and higher multipoles, and
    normal-form / Lie-map machinery.

The follow-up on this axis' physics was **I2** (above, under axis I) — feed-down
belongs to the closed-orbit axis, and J1 was sequenced ahead of it only so its gate
would not be circular. That sequencing paid: I2's gates are built on J1's kick and
none of them is a rerun of it. ✅ Done.

## Out of scope (unless a milestone explicitly calls for it)

Beyond even the expansion axes above — research-grade unless a milestone explicitly
pulls it in: Touschek / IBS, strong-strong beam-beam, crab cavities, wakefields,
higher-order modes, beam loading, full GEANT4, dynamic-aperture / frequency-map
studies, PDF-uncertainty bands, and research-grade machine design.
