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

**As of 2026-08-19 the delivered candidates are** A1–A3, **B1–B3** (the radiation axis
complete: the design-route integrals, classical radiation *in tracking*, and the quantum
excitation that holds the equilibrium open), C1, C2, D1–D5, E1, E2,
**F1**, **F2**, **G1 in full** (betatron-coupling optics — skew quad, coupled
normal-mode tunes, closest-tune-approach `ΔQ_min` — *and* its ε_y vertical-emittance
half, the eigen-mode sharing, whose pre-committed coefficient was corrected by xtrack),
**G2** (Edwards-Teng coupled Twiss), **H1** and **H2** (tune/chromaticity and
insertion matching), **I1** (closed orbit and its correction, which made the
element map affine), **I2** (sextupole feed-down on a distorted orbit — the deferral
I1 named, which J1 was sequenced ahead of so that its gate would not be circular),
**I3** (the optics evaluated on that orbit, closing the gap I2 asserted), **J1**
(the sextupole's nonlinear kick as a real map) and, as of 2026-08-17, **J2** (the
octupole and amplitude-dependent detuning — the first tune that belongs to the
particle rather than to the machine) and **J3** (octupole feed-down on a distorted
orbit, which needed a new `ThinSkewSextupole` element to be written without dropping
a term), **K1** and **K2** (misalignments — transverse offsets and the rolled bend) and
**L1** (the drift's exact nonlinear map: the first element whose `track` is no longer its
`matrix`).
Each is marked inline with what it delivered and what it deliberately did not.
J3 (octupole feed-down on a distorted orbit, the deferral J2 named) closed the last
follow-up on axes A–J the same day it was opened, leaving nothing outstanding there.
Axis K (misalignments: elements gain a position and an orientation of their own) was
opened 2026-08-17 as a *new* axis rather than an extension, since offsets belong to
the closed-orbit axis and rolls to the coupling axis; K1 and K2 both shipped the
same day. The candidate K2 left behind — **exact (nonlinear) maps for `Drift`,
`Quadrupole` and `Dipole`** — became **axis L**, opened 2026-08-17 as a new axis rather
than an extension, since the gap predates axis K entirely and is about core map fidelity
rather than about where a magnet sits. **L1 (the drift) shipped 2026-08-17**, closing the
vertical-dispersion-from-an-orbit-angle gap for the drift's share and re-baselining 29
analytic tests in the process; **L2 (the quadrupole) shipped 2026-08-17** too, taking
tracked natural chromaticity from 45% of the analytic value to **all of it on a
bend-free ring**. **L3 (the dipole) shipped 2026-08-17** as well, taking that to all of
it on a *bendy* ring and converting K2's
`test_the_model_gap_is_fully_accounted_for_and_not_a_mystery` from a hand reconstruction
into a cross-check of the package's own dispersion against xtrack, at `1.7e-8`.
**L4 (the curved quadrupole) shipped 2026-08-18**, giving the last element whose `track`
was its `matrix` the expanded (`mat-kick-mat`) map with F2's Maxwell term, matched to
xtrack at `1.0e-16` — and finding, in the process, that the expanded family drops the
curvilinear metric factor, which is a **new** and precisely-named gap (L5 below). Every
element in the package now has a real map.
With L4 shipped, **every written milestone on axes A-L was delivered and the only
candidate left standing was L5, which is deliberately deferred** (it would be the first
milestone on that axis with no reference for the *map*, only for a number it converges
to). The next direction, chosen 2026-08-18, therefore **reopened axis B**: **B2** puts
classical radiation into `track()` so the damping the design route computes is
*observed*, and **B3** adds quantum excitation and the equilibrium it holds.
**B2 shipped 2026-08-19** — a tracked particle's vertical damping time now lands on
`damping_times` to `3e-5`, and the per-element kick matches xtrack to `6.5e-9` once
xtrack's own eight-step integration is matched, with that residual split between its
pre-2019 charge constant and its ultra-relativistic approximations. **B3 shipped
2026-08-19** — a tracked bunch's equilibrium, held open by the graininess of photon
emission, now lands on Stage 7's `equilibrium_emittance` and `equilibrium_energy_spread`
to 0.11% in both planes, with the two departures from round-off owned by the finite
synchrotron tune and by B2's lumping. Axis B was complete as written on 2026-08-19; it is the
natural consumer of axis L - a faithful per-element map is what a per-element energy
loss needs - and its reference arm was verified to arbitrate the map, not merely the
observable, before the candidate was written.
**The direction chosen next, the same day, extended axis B once more: B4 and B5 below.
Both are now SHIPPED — B4 on 2026-08-19, B5 on 2026-08-25. Axis B is complete as scoped.**
**The new direction chosen after it, on 2026-08-25, is axis M — the optics *off-momentum*,
the derivative of the machine rather than the machine at one momentum. It was picked by
applying the project's usual filter mechanically: every quantity xtrack's `twiss()` reports
was diffed against what accsim reports, and the chromatic family was the one clear gap with
*two* arbiters already wired. M1 shipped the same day — validating the chromatic functions
element by element, and finding that accsim, xtrack and MAD-X give three different
second-order chromaticities on a ring with bends while agreeing on the tune to ten digits.
M1 concluded the split could not be accsim's maps; **M2 shipped 2026-08-26 and found that it
was**. The disagreement is the **drift model** — accsim's drift is exact, xtrack's default
and MAD-X's are paraxial — and accsim is the one that is right: on a five-element ring whose
`Q''` is derived from lab-frame geometry at sixty digits, accsim converges onto the exact
answer while xtrack's default converges onto the paraxial one, and switching xtrack to
`Drift(model="exact")` collapses the whole 5% gap to nine-digit agreement. M1's inference
was valid reasoning from a false premise: it had compared exactly one element (the dipole)
and generalised, and the drift went unchecked because L1 had shipped it exact — which
validated its *map*, not its agreement with a reference's *default configuration*. **M3
shipped the same day and closed the axis** — second-order dispersion, which *bounds* M2's
finding rather than inheriting it: the drift model reaches the closed orbit as `3ab²`,
where `a` is the on-momentum orbit angle, so it is invisible on any ring that closes on
the axis (all three codes then agree on `ddx` to `2e-7` where they split `Q''` by 5%, and
MAD-X is reconciled rather than named for the first time on this axis) and plainly visible
once a steerer is on. The orbit and the optics about it are reached at different powers.** B4 gave a radiating bunch
somewhere to die — a momentum acceptance — so that Stage 4's year-old `quantum_lifetime`
became something the tracking *produces* rather than something the design route quotes.
Its pre-committed headline turned out to be **wrong**, and the correction is the result:
a tracked decay does *not* land on the closed form (it is 37% above it on this ring),
because the closed form is a continuum limit and a tracked bunch is a discrete walk
sampled once per turn. What shipped instead is a chain of three separately gated links,
ending in a departure that is proportional to the step size and extrapolates to zero; B5 replaces B3's Gaussian graininess with real photons
drawn from the synchrotron spectrum. The two candidates were chosen over the alternatives
on the project's usual filter, **whether an independent code can arbitrate the answer**:
the sideways photon recoil that would give the vertical plane a real emittance floor was
ruled out on 2026-08-19 by reading xtrack's `synrad_spectrum.h`, which scales `px`/`py`
along the direction of motion and applies no transverse kick at all — so building it would
mean inventing a model with no reference, which is L5's reason and the one trade this
project's validation strategy does not make. B5 carried the axis's most exposed
pre-commitment, and it contradicted the intuition that motivated the direction — **and it
held**: the single hard photon does *not* knock particles out of an electron storage ring
(suppressed by `e^-341` on B4's own ring, `e^-636` on the 5 GeV example that produced the
`~640`), and B4's lifetime was unchanged by switching from the Gaussian to real photons
(0.930 of it, against a one-sigma band of 8.2%). What B5 bought instead is the tail itself,
cross-checked against xtrack pointwise to better than 1% out to one draw in a thousand,
where B3's Gaussian is 19.4% low.
**With axis M closed the same day it opened, the direction chosen next — also
2026-08-26 — is axis N: the particle's spin.** It is the last piece of single-particle
physics the package has no notion of at all, and it was picked on the same filter M was:
`xtrack` 0.106.4 already tracks spin through its thick magnets, reports the closed spin
solution from `twiss(spin=True)` and computes a linearised polarization analysis, while
independent closed forms exist for the headline numbers — so both arms of the validation
strategy are in place before a line is written. Unlike axis L it re-baselines nothing,
because spin does not act back on the orbit; the flip side is that no existing gate
constrains it either, which is why each milestone's weight is entirely in its gates.
**N1 shipped the same day**, and three of its four findings were not the map: a bug in
accsim's own `SkewQuadrupole` field that only a direction-reading consumer could see, an
alarming-looking xtrack disagreement that turned out to be its default bend *integrator*
moving the orbit rather than its spin being wrong, and a real sign typo in xtrack's
`direction_of_motion` — asserted with the exponent the mechanism predicts rather than
dodged.

**N2, N3 and N4 shipped the same day too, and the axis closed on the shape it had from
N1** — and then **N5 reopened it**, on the one gap N4's own write-up had named: every ring
on the axis had no RF cavity. See the end of the axis for what that turned out to cost.
**N2, N3 and N4 shipped the same day too, and the axis closed on the shape it had from
N1.** Every milestone on it has a headline number that is *degenerate on any ring the
package would casually build* — N1's spin tune, N2's `n_0 = (0,1,0)`, N3's
`P_inf = 8/(5 sqrt3)`, N4's `dn/ddelta = 0` — and in each case the milestone's real weight
moved onto a lattice built to break the degeneracy: N2's closed vertical bump, which N3
and then N4 reused because it is still the only construction in the package that tilts
`n_0` at all. The axis's open question through N1–N3, the intrinsic resonance
`nu_0 = k ± Q_y`, was answered by **N4**: it is not a property of `n_0`, which rides the
closed orbit and can only resonate at integers, but of the invariant spin *field* around
it — and it is the same object depolarization is computed from. With it the axis is
**complete**: the map (N1), the closed solution (N2), the polarization radiation builds
(N3), and the field and the resonance that destroy it (N4).

**N5 reopened it the same day, and it was chosen the way this project chooses: by taking
a milestone's own stated limitation seriously.** N4 had written that its six-column
equation and xtrack's five-column one are the same object only because nothing reads
`zeta`, "and that stops being true the moment an RF cavity enters." Putting one in turned
out to break something a level below the spin machinery — the *closed orbit* on a bunched
ring is not where the 4D solve puts it — and to open a second family of resonances, the
synchrotron sidebands, which are what actually limits polarization in a real electron
ring. It also carried this axis's most exposed pre-commitment, **and that one did not
hold**: xtrack's momentum column was predicted to come into line once RF removed the
singular inverse N4 blamed, and instead a larger, differently-caused disagreement appeared
— arbitrated, this time, by a test that runs inside the reference's own tracker.

**The direction chosen next, on 2026-08-27, went back to axis I and closed the one gap
three separate milestones had named and none had built: I4, the closed orbit in all six
coordinates.** I2 listed it as out of scope, I3 repeated it verbatim, N5 deferred it a
fourth time *with its gate already written in closed form* — and B4, needing it, had built
a private 25-line Newton solve inside its own test file. It shipped the same day. Its
weight is in two places. The gate is not a tolerance but an **order**: the design-route
radiation integral departs from the tracked loss at first order in `U_0/E` about the design
orbit and at second order about the closed one, because the fixed point is exactly where
the energy sag is centred (fitted 0.999 against 2.003, over a factor 512). And the
cross-check is a **prediction rather than a comparison**: two mechanisms localised on a
single magnet in B2 were asserted, unadjusted, on two codes' whole-ring fixed points, and
landed at `2.29997e-8` against a predicted `2.29997e-8`. The milestone's *other*
pre-commitment — that the 4D orbit's residual would be the whole energy bill, in `delta` —
was **refuted**, and the refutation is the better result: the miss is largest in arrival
time, and the momentum is short by only 0.689 of the bill, because the slip in `zeta`
already collects a third of it back from the RF before the turn ends. One shipped claim
was corrected on the way: `phi_s != 0` does **not** need a 6D solve in this package, only
tracked radiation does.

**With I4 shipped, every written milestone on axes A-N was delivered and the only
candidate left standing was again L5. The direction chosen next, on 2026-08-27, opened
axis O — normalised coordinates** — on the same mechanical filter that chose axis M: every
field `xtrack`'s `twiss()` returns, diffed against `accsim`'s exports. The matrix `W` that
turns the one-turn map into a plain rotation was the largest remaining gap with an arbiter
already wired, and it is the object every optics quantity on axes A-N is secretly a
parameter of. Two things were known before the candidate was written: the two structural
checks (`M = W R W^-1`, and `W` symplectic) are **blind** to the three per-plane phases
that are the whole content of the parameterisation, so the primary gate has to be the tie
to Stage 1's `beta`/`alpha`; and the 6D normal form does not reproduce the 4D optics at
all, departing from it quadratically in the synchrotron tune.

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

- **B2 — classical ("mean") radiation *in tracking*: the damping observed rather than
  computed.** ✅ **SHIPPED (2026-08-19)** — `src/accsim/radiation_kick.py`: a per-element
  energy loss on the particle's *own* trajectory, opt-in per tracking call and off by
  default, so damping is something the simulation exhibits. Full write-up at
  CONVENTIONS.md → *Radiation in tracking*. Gates:
  `tests/analytic/test_radiation_tracking.py` (24) and
  `tests/reference/test_radiation_tracking_xtrack.py` (6). The headline: a vertically
  displaced particle tracked for 1500 turns damps at **`tau_y` to 3e-5 of
  `damping_times`** — a closed form computed by a completely separate route and gated
  against xtrack and MAD-X a year before any tracking radiation existed.
  - **The discriminating detail was the `px, py` scaling, as predicted, and its wrong-map
    signature is now measured rather than assumed.** Photons leave along the direction of
    motion, so `(px, py, 1+delta)` take **one** common factor; `pz` then scales by the same
    factor *exactly*, leaving `x'` and `y'` invariant to the last bit. Dropping it splits
    into two true statements the candidate entry could only guess at: inside the element it
    **anti**-damps the angle at first order (`+eps px (1+delta)^2/pz^3`), and per turn it
    gives **exactly zero** transverse damping — a fitted `tau_y` 300,000× too long — because
    `py` is never touched and the RF restores `delta`. The longitudinal arm is completely
    blind to it, exactly as written.
  - **The reference arm needed its integration order matched before it meant anything, and
    that is the milestone's real trap.** xtrack sub-steps the loss *inside* the element;
    its default `integrator="adaptive"` is **eight** uniform steps for a plain bend, worth
    `(N-1)/N · U/E` — 3.8e-5 at 5 GeV, 2.4e-3 at 20 GeV — which looks exactly like a wrong
    coefficient in `C_gamma`. With `integrator="uniform", num_multipole_kicks=1` the two are
    the same map, and the `(N-1)/N` law is asserted on both sides.
  - **What is left is 6.5e-9 with two named owners, both xtrack's**: `1.064e-8` from its
    **pre-2019 CODATA charge** (`r0 ∝ e`, so it lands on `C_gamma`, energy-independent) and
    `2/gamma0^2` from its ultra-relativistic approximations. accsim keeps the exact on-shell
    forms, so the second term *dies with energy* — asserted across a factor 80 in energy,
    which is what makes it an owner rather than a tolerance.
  - **The `tau_z` window the entry promised to check exists comfortably**, and the entry's
    "three-sided squeeze" was the right shape: `tau ∝ 1/E^3` buys the damping time down to
    ~2100 turns at 3 GeV, the rings are **above transition** so the cavity needs
    `phi_s = pi` and `V > U0`, and `Q_s ≈ 0.08` puts 43–65 synchrotron periods inside
    `tau_z`. What the entry did **not** anticipate: the equilibrium orbit must be found by
    **Newton on the radiation-on one-turn map**, not by tracking to it — with `tau_x` in the
    thousands of turns a "converged" orbit is still drifting, and that drift alone made the
    first `tau_x` measurement read as *no damping at all*.
  - ⚠️ **The entry's prediction about the loss per turn had the sign backwards.** A tracked
    turn loses **less** than `U0`, not more: the particle radiates at a progressively lower
    energy as it goes round, while the closed form evaluates everything at `E0`. It is
    `U0(1 - c U0/E)` with `c = 1.26` on the test ring, constant to 1% over a factor 64 in
    `U0/E`, and that stability is the gate.
  - **A model boundary found on the way, and it is the same one Stage 7 recorded from the
    other side.** The tracked route's damping *partition* is the damped-map eigenanalysis,
    not the integral method, and the two part company as `I4/I2` grows: 0.2% at
    `I4/I2 = 0.38`, **11%** at `0.71`. The load-bearing half is that **one** number explains
    both planes — whatever `I4/I2` the tracked map implies reproduces `J_x = 1 - I4/I2`
    *and* `J_z = 2 + I4/I2` together — so it is a method difference, not a broken plane, and
    slicing does not close it (it makes it *larger*, while driving Robinson's measured
    `J_x + J_y + J_z` from 4.026 to 4.0004). The sharp partition gates therefore run on a
    normal arc and the departure itself is asserted as a monotone function of `I4/I2`.
  - **A sign error found in xtrack on the way out.** Its `direction_of_motion` computes
    `iis = sqrt(1 - iix*iix + iiy*iiy)`; the `+` on the vertical term is wrong, and accsim
    uses the correct `1 - ix^2 - iy^2`. The two part company **quartically** in `py`
    (`6.0e-7` at `py = 2e-2`, `2.3e-5` at `5e-2`), because the projections differ by
    `2 B_par^2 iy^2` and `B_par` is itself linear in `py`. It is inert at the `py <= 1e-3`
    every other cross-check on this axis uses, so it took a deliberate probe with a
    twenty-times lever arm to see — and it is gated from both sides: the growth law *and* a
    reconstruction of xtrack's sign that lands back on the usual `1.19e-8` residual at
    every amplitude.
  - **Blast radius zero.** No existing test changed: `matrix()`/`kick()` are untouched and
    radiation defaults off, so the 908 analytic tests stayed green and 23 were added. The
    one API change is additive — `Tracker.track_once` is now public and `radiation=` reaches
    `track`, `track_turns`, `track_bunch` and `track_bunch_losses`; asking for it without
    `nonlinear=True` **raises** rather than silently returning an undamped answer.
  - **Deliberately not built:** the equilibrium orbit with radiation as a package function
    (it lives in the test's measurement machinery), and radiation from thin elements — a
    zero-length magnet has no path to radiate over, which is scope, not approximation.

- **B3 — quantum excitation, and the equilibrium the damping settles into.**
  ✅ **SHIPPED (2026-08-19)** — `radiation="quantum"` in `src/accsim/radiation_kick.py`:
  the radiated energy drawn from a Gaussian with the mean B2 already ships and the
  variance `σ_U² = 2 C_q E γ² κ U`, written with the package's own `C_q`. Full write-up
  at CONVENTIONS.md → *Quantum excitation and the tracked equilibrium*. Gates:
  `tests/analytic/test_radiation_quantum.py` (34) and
  `tests/reference/test_radiation_quantum_xtrack.py` (7). The headline: a tracked bunch's
  equilibrium, obtained from the tracking's *own* noise, lands on `equilibrium_emittance`
  and `equilibrium_energy_spread` — closed forms written a year earlier by a route that
  shares nothing with it but the constant `C_q` — to **0.11% in both planes**, and both
  departures from round-off have named owners with measured laws.
  - **The variance is derived from the spectrum, and the spectrum is integrated rather
    than quoted.** `⟨u⟩ = 8/(15√3) u_c`, `⟨u²⟩ = 11/27 u_c²` and `⟨u³⟩` come out of
    `∫_x^∞ K_{5/3}` in the suite itself, and `σ_U² = n_γ⟨u²⟩ → 2 C_q E γ² κ U` is a
    symbolic identity from *those*, not from the collapsed form. The bridge gate
    `n_γ⟨u⟩ = U` is what says the photon picture and B2's `C_γ` describe one effect.
  - **The load-bearing coefficient was the synchrotron phase-averaging ½, exactly as the
    candidate could not have guessed.** Photons kick `δ` only, but `⟨δ²⟩ = ⟨a²⟩/2` for an
    oscillation; drop the ½ and the predicted spread is exactly `√2` too wide — an error
    independent of energy, geometry and lattice, so *no* scaling gate could see it. It is
    pinned symbolically as exactly 2.
  - **The sharp gate is not tracking.** Tracking to equilibrium is statistics-limited by
    construction, so the equilibrium is obtained by solving the discrete Lyapunov equation
    `Σ = M Σ Mᵀ + D` for the tracked map — exactly, with no statistics — using a stand-in
    generator that returns a chosen number of standard deviations on one nominated draw.
    That turns the stochastic map into a differentiable one while still exercising the
    shipped code path. The tracked settle-from-any-start gate the entry pre-committed is
    kept, but as the *confirmation*, against a stated budget (2.0% on a width from 300
    particles × 2 damping times).
  - ⚠️ **The entry promised agreement with the closed forms and got it only after two
    departures were separated — neither of which is a tolerance.** (1) The closed forms
    are the **smooth-ring** limit: they assume the synchrotron phase barely advances while
    the turn's photons are emitted, and the tracked answer departs as `1 + c(2πQ_s)²`.
    This is gated sharply rather than fitted: **two rings whose radiated power differs by
    256×, whose spread differs by 4× and whose emittance differs by 16× depart by the same
    amount to 4 parts in 100,000, because they share `Q_s`.** (2) B2's one-kick-per-element
    lumping owns a ~0.6% `ε_x` offset that slicing removes and to which `σ_δ` is blind
    (3e-5). Two owners, two signatures — one dies under slicing and is `Q_s`-independent,
    the other survives slicing and scales as `Q_s²` — so neither can be mistaken for the
    other or for a wrong `C_q`.
  - **A pre-commitment that held, after a measurement that first said otherwise.** The
    horizontal excitation is *dispersion*, not photon recoil: deleting the direct
    transverse kick from the injected noise moves `ε_x` by **4e-6**. The first attempt to
    measure this zeroed the *propagated* noise column instead of the injected vector and
    read 45% — a reminder that the blindness claim needs the right object, not a
    plausible one.
  - **`ε_y` is exactly 0.0, not small, and that has consequences.** No opening angle means
    no vertical excitation at all, so the equilibrium `Σ` is **singular** (rank 4,
    `cholesky` raises — found by sampling an equilibrium bunch) and a vertically displaced
    beam keeps damping through the equilibrium rather than stopping at it, because the
    noise on `py` is *multiplicative*. The real floor is the `1/γ` opening-angle limit,
    omitted by construction.
  - **The Gaussian is unclamped, deliberately, and that is measured against xtrack.** With
    `n_γ ~ 16` photons per magnet, `u < 0` — an energy gain — is a 2σ event happening in
    1–3% of draws, not a tail. Clamping would bias mean *and* variance by ~1%, five times
    the agreement the equilibrium gates achieve. xtrack (real photons) never gains energy
    and is skewed −0.91; accsim gains 2.6% of the time and is skewed +0.003.
  - **The reference arm compares two genuinely different stochastic processes**, which is
    what makes it worth having: xtrack samples real photons off `K_{5/3}`, accsim draws
    one Gaussian, and the standard deviation of one magnet's loss agrees to **0.18%**
    against a 0.16% statistical floor. And **xtrack's own skewness counts its photons** —
    a compound Poisson sum's third moment inverts to `n_γ`, landing on the textbook
    `(5/(2√3)) α γ θ`.
  - **Blast radius one line, and it was only found because the first "suite is green"
    check was not evidence.** A background run reported exit 0 with a *zero-byte* log, and
    that was taken for a pass; re-run in the foreground, one B2 gate had in fact broken.
    `test_radiation_without_the_nonlinear_path_raises_instead_of_being_ignored` used
    `radiation="quantum"` as a name that was *unknown* when B2 was written — B3 made it
    real, so the "must be one of" refusal now had to be provoked with a name that is
    still unknown. Nothing else moved: 932 pre-existing analytic tests stayed green and
    34 were added (966 total), and the radiation reference arm is 14 passing. `C_q`/`HBAR_C_EV_M` moved to `radiation_kick.py` and are re-exported
    from `radiation.py` (the direction the dependency already ran), `mean_radiation_kick`
    is an alias for `radiation_kick`, and `rng` is additive everywhere. Radiation still
    defaults off, and a stochastic model without an `rng` **raises**.
  - **Deliberately not built:** a photon-resolved sampler. The equilibrium depends on the
    emission process only through its first two moments, which the Gaussian matches
    exactly, so it would change no number this milestone gates. What it *would* change is
    the tail — the single hard photon that empties the RF bucket — which is Stage 4's
    `quantum_lifetime` and an axis of its own.

- **B4 (SHIPPED 2026-08-19) — the acceptance a radiating bunch dies at, and the quantum
  lifetime the tracking produces.** Stage 4 shipped `lifetime.quantum_lifetime` a year
  ago and B1 gave it a damping time computed from the lattice, but nothing in accsim had
  ever *lost* a particle to radiation: B3's bunch reached its equilibrium and sat in it
  for ever, because the excitation only refilled a distribution with no exit. This is the
  exit. **Shipped:** `MomentumAperture` (a `|delta − center|` acceptance) beside the
  geometric `Aperture`, both now subclassing `AcceptanceElement` so the loss-aware pass
  dispatches on the base class; `lifetime.quantum_lifetime_exact`, the mean-first-passage
  integral itself; and 30 gates across `test_quantum_lifetime.py` (16) and
  `test_quantum_lifetime_tracking.py` (14).
  - ⚠️ **The headline this entry pre-committed was wrong, and the correction is the
    milestone's main result.** The draft said a tracked decay would land on the closed
    form "with no fitted parameter in between". It does not: on this ring the tracked
    decay is **37% above** `tau/lambda_1`, reproducibly. The reason is physics, not a
    bug. `quantum_lifetime` describes a *continuum* diffusion of the oscillation
    amplitude; a tracked bunch is a *discrete* walk looked at once per turn, and one turn
    moves the normalised action by 0.23 out of `xi = 3`. So the deliverable became **three
    links, each gated separately**, in B3's shape:
    1. the map's noise and damping — B3's Lyapunov solve (tracked `sigma_delta` is the
       Lyapunov one to 0.05%, tracked `delta` is Gaussian to kurtosis 2.99);
    2. the first-passage physics given that map — accsim's tracked survival curve against
       **an independent twenty-line implementation of the same process** with no lattice,
       no elements and no radiation model in it, agreeing pointwise across a factor of
       four in survival;
    3. that discrete process against the closed forms — its departure is proportional to
       the step size and extrapolates to zero.
    Stated plainly, because it is the limit of the argument: the toy shares the
    *conceptual* model, so it cannot catch a wrong noise magnitude. That is link 1's job
    and B3 already did it.
  - **The two departures, separated and named.** A coordinate cut is not an amplitude
    cut — `|delta|` is sampled once per turn, so a particle whose amplitude has crossed
    the boundary survives until a sample lands near its extreme, worth a further 22%; and
    even a true amplitude cut runs 14% long because the steps are not infinitesimal. The
    **`Q_s` scan is what separates the two mechanisms**, and it is the reason the plain
    `delta` cut had to replace the separatrix (below): the coordinate excess is *flat to
    1.5% while `Q_s` moves by a factor of four*, so it is once-per-turn **sampling** and
    not phase-rotation delay, which would have scaled with the synchrotron period. The
    step-size scan alone could not have told them apart, because it moves both together.
    At `Q_s = 0.5` the two cuts collapse onto each other and both fall *below* the closed
    form — the half-integer synchrotron resonance, where every sample lands on the same
    pair of phases; gated as the edge of the flat region rather than dropped.
  - **The boundary is a plain `|delta − delta_co| <= delta_acc` cut, and the RF separatrix
    is explicitly *not* it.** The first draft said "a low RF voltage for the longitudinal
    boundary", and that design fights itself from both ends. The bucket half-height goes
    as `sqrt(V)` and so does `Q_s`, so with a separatrix boundary `xi ∝ V` and
    `Q_s ∝ sqrt(V)` — **the discriminating gate above becomes unrunnable**, not merely
    awkward. Worse, `Q_s(amplitude) → 0` at the separatrix and the motion there is
    maximally anharmonic, while the closed form is a *harmonic* first-passage result.
    The plain `delta` cut has neither problem: `sigma_delta` is voltage-independent
    (radiation integrals only), measured as `2.0228e-3` to five figures across 30–180 MV
    while `Q_s` moves `0.075 → 0.190`. Ring: 6.5 GeV, 20 FODO cells, 90 MV, `xi = 3`,
    which puts the acceptance at **0.15 of the bucket height** — harmonic by construction.
  - ⚠️ **The cut must be centred on the local closed orbit.** Radiation drains `delta`
    through the arcs and the cavity restores it in one lump, so the fixed point is *not*
    `delta = 0` at most elements: measured, `delta_co(s)` swings from `−0.966` to `+0.921`
    sigma, **1.887 sigma peak to peak**, and it is `U0/E = 3.8e-3` that sets that scale.
    A symmetric cut at the worst element would sit at `xi = 1.73` one side and `7.20` the
    other instead of `4.00` on both — a **6x shorter** lifetime, from a boundary that
    looks perfectly reasonable. Centring is also the correct physics: the closed form's
    amplitude is measured from the fixed point.
  - **The mean first-passage time is not the decay constant, and gating against the wrong
    one would have read as a bug.** What a survival curve measures is the slowest
    eigenvalue `lambda_1` of the generator with an absorbing wall, not the mean time for
    one particle to reach it. Measured by discretising the operator whose
    backward-equation residual the suite already verifies symbolically:
    `MFPT/(1/lambda_1)` = **1.134, 1.079, 1.005, 1.000** at `xi` = 3, 4, 8, 12. The 8% gap
    at `xi = 4` against a `1/sqrt(N_lost)` budget of ~2% would have failed by 4x. The
    eigenvalue route also has a **ceiling**, gated rather than discovered: the symmetrising
    weight is `e^-w`, so past `xi ~ 20` double precision runs out and at `xi = 30` the
    eigenvalue comes back *negative* — while `quantum_lifetime_exact`, an
    everywhere-positive series, stays exact there.
  - **`quantum_lifetime_exact` had to ship first.** At `xi = 4` the exact integral is
    **17.6674** against the asymptote's **13.6495**, a ratio of **1.29436** gated to five
    figures as a pure number. Pre-committed trap, with a test of its own: gating that
    departure against a truncated `O(1/xi)` expression **would itself be wrong** —
    `1 + 1/xi = 1.25` and `1 + 1/xi + 2/xi^2 = 1.375` *bracket* the truth. The departure
    is the law `xi (exact/asymptote − 1) → 1`, not "halves when `xi` doubles", which is
    the same claim only in the limit (measured 2.42 at `xi = 8 → 16`).
  - **The vertical plane has no quantum lifetime, and that is a gate rather than a gap.**
    One ring, one model, two planes that disagree in a pre-stated way: the momentum cut
    loses two thirds of the bunch while a vertical aperture at comparable tightness loses
    **zero**. The vertical noise is not absent so much as *multiplicative* — `py` is
    scaled by a random factor, so diffusion goes as `py^2` and the equilibrium is zero
    rather than the drive being zero — so the bunch is started *below* the aperture;
    started above, the damping transient would carry particles across it and the zero
    would read as a broken gate.
  - **The horizontal plane is deliberately *not* gated on a lifetime.**
    `x = x_beta + D_x delta`, so a coordinate cut where `D_x sigma_delta` is comparable to
    the betatron size is a **two-mode** first-passage problem with no single `xi`, and the
    two modes here differ 4x in damping time. A FODO ring has no dispersion-free location.
    The earlier draft's "the horizontal, excited through dispersion, must show the closed
    form" conflated two roles of dispersion: in the bends it *creates* the horizontal
    emittance (kept), at the aperture it *breaks* the one-dimensional form (avoided). That
    `xi` is well posed in the *longitudinal* plane is checked rather than assumed, by
    projecting `Sigma` onto the eigen-modes of `Sigma S`: **99.3%** of the momentum spread
    is the longitudinal mode. (At 180 MV that falls apart — the two modes hybridise, 0.05
    against 0.95 — which is the real reason the scan is capped, not the damping-time drift
    the draft cited.)
  - **The factor of 2 is sidestepped rather than converted.** `tau = −1/ln|lambda|` from
    the one-turn Jacobian **is** the amplitude damping time by definition, so nothing has
    to be halved anywhere: measured 219.6 turns against the radiation integrals' 220.2, an
    independent cross-check of two routes' conventions for free.
  - **Cost, measured and deliberate.** The whole file is 54 s, of which the tracked gate
    is 32 s. The cheaper 10-cell ring was tried and rejected: at `U0/E = 1.1%` per turn it
    does not confine a beam at all (tracking without an acceptance returns `NaN`). The RF
    bucket's anharmonicity was checked as an explanation for its disagreement and **ruled
    out** — a pendulum toy matched to the same `Q_s` and bucket height returns 521 turns
    where the linear one returns 523.
  - **Not built:** Touschek and IBS (out of scope, a different loss mechanism), dynamic
    aperture (out of scope), the separatrix predicate (a different boundary, which the
    closed form does not describe), the horizontal quantum lifetime (above), the vertical
    one (unavailable by construction until the opening angle exists), and the
    `xt.LongitudinalLimitRect` reference arm — the analytic chain closed without it, and
    a statistical two-code comparison would add cost rather than discrimination.

- **B5 (SHIPPED 2026-08-25) — the photon-resolved sampler, and how far the tail actually reaches.**
  B3 shipped the graininess as a Gaussian of the right mean and the right variance and said
  plainly what that leaves out: the tail. This builds the real thing — `radiation="photons"`,
  a compound-Poisson sum with the count drawn from the photon rate
  `n_gamma = 5/(2 sqrt3) alpha gamma kappa l` and each energy drawn from the normalised
  synchrotron spectrum `S(x) = (9 sqrt3 / 8 pi) x int_x^inf K_{5/3}` that the B3 suite
  already integrates for its moments.
  - **The gates are already written and mostly already passing for the Gaussian.** The
    sampler's first three moments must reproduce the suite's own Bessel quadratures
    (`<x> = 8/(15 sqrt3)`, `<x^2> = 11/27`, and `<x^3>`); `n_gamma <u> = U` must bridge back
    to `C_gamma`; and the compound-Poisson variance `n_gamma <u^2>` must be
    `photon_energy_variance` **identically**, not approximately — which means B3's whole
    equilibrium battery re-runs under the new model and must land on the same closed forms
    within the same budget. That is the strongest statement available here: a model that
    changes every draw and no aggregate.
  - **The sharp gate is deterministic, via the sampler's own inverse.** Sampling by inverse
    transform makes each draw an exactly predictable function of one uniform, so B3's
    stand-in-generator trick applies unchanged: feed a chosen quantile and the photon energy
    is a number with a closed form, including far out in the tail where the Gaussian is
    wrong by orders of magnitude and where no amount of sampling would show it. The
    exceedance `P(x > X)` is then gated against the spectrum's own quadrature at `X` where
    the Gaussian model predicts `e^-X^2` and the truth is `~e^-X`.
  - **Two signatures separate this model from B3's, and both are pre-committed.** The
    Gaussian draws an energy *gain* in 1-3% of draws (B3 measured it, deliberately unclamped,
    and matched xtrack's contrasting skew of `-0.91`); the photon model must draw one
    **never**, and its skewness must be positive and must invert to `n_gamma` by the same
    compound-Poisson identity B3 used to count xtrack's photons from the outside.
  - ⚠️ **The prediction this entry is most exposed on, stated before it is measured: B4's
    lifetime will not move.** The exit from the bucket is a many-photon random walk —
    `n_gamma` per turn is in the hundreds and a damping time is thousands of turns — so the
    central limit theorem makes the accumulated step Gaussian whatever the individual photon
    spectrum is, and the lifetime depends on the emission process only through the first two
    moments the Gaussian already matches exactly. If switching B4's ring from `"quantum"` to
    `"photons"` moves the fitted lifetime by more than its stated statistical budget,
    something in one of the two models is wrong, and the gate is written to say so rather
    than to celebrate a difference.
  - ⚠️ **And the single hard photon that empties the bucket in one go — the thing B3 named as
    this axis's remaining physics — is, on any ring this package can build, exponentially
    dead.** The channel needs a photon with `u > E delta_acc`, and the spectrum falls as
    `e^-u/u_c`: at 5 GeV in a 10 m bend, `u_c/E = 5.5e-6` against an acceptance of `3.5e-3`,
    so the required ratio is **~640** and the probability is `e^-640`. No tracking budget, and
    no ring with a sane energy and bending radius, reaches it. This is not a reason to skip
    the sampler — the sampler is what lets the claim be *measured* rather than asserted — but
    the honest deliverable is the **suppression law**, `rate ~ n_gamma e^-(E delta_acc/u_c)`,
    gated across a scan of `u_c/(E delta_acc)` in a regime where it is observable, with the
    extrapolation to the real ring stated. The claim "graininess is what knocks particles
    out" is, for electron storage rings, false, and this milestone is where the package
    proves it instead of inheriting it.
  - **Deliberately not built:** the opening angle (still no reference implements it — see
    L5's reason, which is the same reason; verified 2026-08-19 in xtrack's
    `synrad_spectrum.h`, which scales `px`/`py` along the direction of motion and applies no
    transverse recoil), and beamstrahlung, which is the one place the single-photon channel
    *does* dominate and which belongs to the beam-beam axis.
  - Effort **M**. Depends on B4 only for its lifetime gate; the sampler itself stands alone.
  - ✅ **What shipped, and how the pre-commitments landed.** All of them held, and the two
    marked ⚠️ above — the ones the entry was most exposed on — held with numbers:
    - **the lifetime did not move**: 1154 turns against 1240 from the *same frozen bunch*,
      a ratio of 0.930 where one standard deviation is 8.2%, i.e. 0.86 sigma. The statistic
      is a **fitted decay** rather than binomial marks, because a fit uses every turn: 400
      particles then give a 25% three-sigma band where three marks at 800 would give 37%,
      and 25% is narrower than the 37% departure from the continuum B4 itself measured;
    - **the hard photon is exponentially dead**, and the exponent depends on the ring. B4's
      6.5 GeV ring at `xi = 3` needs `X = E delta_acc/u_c = 337` critical energies, where
      the exceedance is `e^-341`; the 5 GeV / 10 m / `3.5e-3` configuration this entry
      quoted computes to `X = 631` and `e^-636`, so the `~640` above was right *for that
      machine* and is not a property of the axis. Both are gated. Restated as something
      holdable: the hardest of the `4.0e8` photons in a whole `400 x 1200` run is
      **17.0 u_c**, a factor of twenty short, and ten times the run buys `log 10 = 2.3` more;
    - **the shape signatures both appeared**: the loss can never be negative (against the
      Gaussian's deliberate 2.6% of energy *gains*), and its skewness inverts to the photon
      count — in `delta` it lands at **-0.92** against xtrack's measured **-0.91**;
    - **and no aggregate moved**: `n_gamma <u> = U` and `n_gamma <u^2> = photon_energy_variance`
      are gated to **1e-13 through the shipped code path** off-axis, so the diffusion matrix
      is the same entry-by-entry in all 6x6 and B3's equilibrium battery re-runs unchanged.
  - ✅ **The claim had to be narrowed, and that is the milestone's real content.** No single
    photon carries a particle *from the core across* the acceptance. Particles are of course
    lost *at* an emission — it is the only place `delta` ever falls — but the photon that
    finishes the job is an ordinary one arriving at a particle the random walk has already
    carried to the wall. The broad version, "graininess is what knocks particles out", is
    false for an electron storage ring, and this is where the package proves it rather than
    inheriting it.
  - ✅ **B3's reference arm could finally be written the other way round.** Its own docstring
    opens by saying it is the most useful arm on this axis *because the two codes do not do
    the same thing*. They now do, by unrelated numerical routes — accsim inverse-transforms a
    tabulated quadrature of `int_x^inf K_{5/3}`, xtrack rejection-samples `K_{5/3}` along an
    exponential free path — and the tails agree **pointwise to better than 1% out to one draw
    in a thousand**, where B3's Gaussian is 19.4% low. One in ten thousand is where 200000
    particles run out, and that is gated as its own statement rather than assumed.
  - ⚠️ **Two traps worth carrying forward, both found as failures.** `quad(K_{5/3}, X, inf)`
    in one piece *silently* returns zero for small `X` — it warns and does not raise, and a
    cumulative distribution built on it comes out at exactly 2/3 of the truth. And xtrack's
    emission is **not seeded** by this suite, so every cross-check gate is a two-sample
    comparison whose budget needs the `sqrt(2)`; without it a routine 2.9-sigma fluctuation
    reads as a 4.1-sigma failure. Both are recorded in `CONVENTIONS.md`.
  - Reaching for 1e-13 also surfaced a real cancellation in B2's shipped kick:
    `f*(1+delta) - 1` keeps six digits of an increment of size `1e-7`. Rewritten as
    `delta + (f-1)(1+delta)` with the rationalised `f - 1 = -shrink/(1+f)`. The 96 analytic
    gates of B2/B3/B4 are unmoved by it.

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
    **The natural half needs its own gate**, because the tracked gate constrains a
    *difference* that cancels it: on a dispersion-free ring it is the exact thin-lens
    sum `−(1/4π) Σ β_x k1l` over the quads *and* the feed-down gradients, with β from
    the independently-gated on-orbit table (agreement `5e-13`). It discriminates
    because the two contributions oppose: the sextupole's direct term is `−3.03e-3`
    against the beat's `+1.52e-3`, so dropping it flips the sign of the answer.
    Gates: `tests/analytic/test_orbit_optics.py` (21),
    `tests/reference/test_orbit_optics_xtrack.py` (4). See CONVENTIONS.md →
    *Optics on the real (steered) orbit*.
  - Still **out of scope**: off-axis feed-down from accsim's own linear elements
    (the bend/quad nonlinearity above — a candidate milestone of its own); coupled
    Edwards-Teng chromaticity on a vertically steered machine; thick-sextupole
    chromaticity on orbit; and everything I2 listed — the 6D closed orbit,
    octupoles and higher multipoles, misalignments as element attributes,
    amplitude-dependent detuning, dynamic aperture.

- **I4 — the closed orbit in all six coordinates: where the beam arrives, once it has to
  pay for the light it emits.** ✅ **SHIPPED (2026-08-27)** — `accsim.closed_orbit_6d`,
  plus `closed_orbit_delta` promoted to the top-level namespace alongside it. Effort **S**.
  The one item three separate milestones have named and none has built. I2 listed "the
  6D closed orbit" as out of scope, I3 repeated it verbatim, and **N5** deferred it a
  fourth time *with its gate already written in closed form*. Meanwhile B4 needed it and
  built a private 25-line Newton solve inside its own test file
  (`tests/analytic/test_quantum_lifetime_tracking.py::_equilibrium_orbit`), which is the
  strongest evidence a candidate on this project can have: the function exists, it is
  just not in the package.

  **What is missing, precisely.** Every closed orbit in `orbit.py` is a fixed point of
  the *transverse* map at a `delta` the caller chooses — `closed_orbit` solves
  `(I - M4) x = k4`, `closed_orbit_nonlinear` Newtons on the same 4D subspace, and N5's
  `closed_orbit_delta` adds the one scalar `delta_co` that makes `zeta` close on a
  bunched ring. All three hold `zeta = 0`. That is exact for a ring whose cavity has
  nothing to make up, and it is *wrong* the moment the beam radiates in tracking: a ring
  that loses `U_0` per turn must arrive off the zero crossing of the RF wave, far enough
  up it that the cavity hands back exactly `U_0`.

  **The gate, in closed form, from N5's own deferral note.** `zeta_co` is where the
  cavity's kick equals the turn's loss,

      `q V [sin(phi_s - k_rf zeta_co) - sin(phi_s)] = U_turn`,

  read **at the cavity**, not at the lattice start — the two differ by whatever fraction
  of the loss is accumulated upstream of it, so a ring with the cavity spliced mid-lattice
  gates the distinction on its own. Two independent routes supply `U_turn`, and the
  milestone asserts both:

  - the **tracked** loss, summed element by element around the converged orbit, which
    must satisfy the equation at round-off (this is an identity, and it is what makes the
    solve testable without a second model); and
  - the package's own `energy_loss_per_turn`, the design-route radiation integral, which
    is an *independent* number and must therefore land nearby but not exactly.

  **The discriminating content is the order of that second departure, and it is a
  statement about the fixed point rather than about the radiation.** On the *design*
  orbit the integral and the tracked sum differ at **first** order in `U_0/E`: a lumped
  per-element kick makes the particle poorer as it goes, so every element after the first
  radiates at less than the design energy. On the *closed* orbit that error is gone,
  because the fixed point is precisely where the sag is centred — the beam sits `U_0/2E`
  high at the cavity's exit and `U_0/2E` low at its entrance, and the linear-in-`delta`
  part of the loss averages away over the turn. The departure is therefore **quadratic**,
  and a gate that fits the exponent reads a wrong fixed point off directly where a
  tolerance would not. Over a factor 512 in `U_0/E`: **2.003 on the closed orbit against
  0.999 on the design orbit** (measured while scoping, so recorded as scoping rather than
  as a pre-commitment). The two routes never cross — the closed orbit is better
  everywhere, by between three and four orders of magnitude.

  **A shipped claim is wrong and this milestone corrects it.** `orbit.py`'s
  `closed_orbit_delta` docstring and N5's deferral paragraph both say a ring needs a 6D
  fixed point when "`phi_s != 0`, **or** radiation switched on in tracking". The first
  half does not hold in this package. `energy_kick_delta` is
  `sin(phi_s - k zeta) - sin(phi_s)`, which vanishes at `zeta = 0` for **every** `phi_s`,
  and the reference energy ramp that would make an accelerating bucket mean something
  lives entirely inside `accelerate` (which builds its own `ref` per turn) and never
  touches the tracking path. Verified at three synchronous phases: `zeta_co` and
  `delta_co` come back **exactly** `0.0`. Tracked radiation is the only thing in accsim
  that moves `zeta_co`, and the two docs are corrected to say so.

  **The pre-commitment, written before the reference arm was run.** With integration
  order matched the way B2's cross-check matches it (`integrator="uniform"`,
  `num_multipole_kicks=1` — xtrack's `twiss` finds its closed orbit by tracking the line,
  so the element settings that govern its tracker govern this too — verified in
  `twiss.py` before the number was written down), xtrack's 6D closed
  orbit should reconstruct the **same loss**, `q V [sin(phi_s - k zeta) - sin(phi_s)]`,
  as accsim's, differing by exactly B2's already-named residual — the `1.064e-8` charge-
  constant vintage plus the `2/gamma0^2` of its ultra-relativistic approximations, and
  nothing else. The comparison is deliberately made on the *loss* rather than on `zeta_co`
  itself, which carries an `arcsin` and would fold the same disagreement through a
  ring-dependent factor.

  **The pre-commitment HELD, to five digits.** Predicted `2.29997e-8`, measured
  `2.29997e-8` — agreement to `2e-6` of the residual itself, which leaves no room for a
  third owner. Two mechanisms localised on a *single magnet* (B2) predict, without
  adjustment, the disagreement between two codes' *whole-ring fixed points*. Both codes put
  the arrival time at `8.887901 cm` on a 40 m ring, the momentum at `1.862083e-3` and the
  horizontal orbit at `-5.47e-7 m`. Two details of the comparison are gated rather than
  assumed: the `arcsin` factor `tan(k zeta)/(k zeta) = 1.0270` that would have been folded
  in silently by comparing `zeta` instead of the loss, and — the third, independent route —
  feeding **xtrack's own** `zeta` through accsim's closed form, which reproduces
  `twiss.energy_loss` (xtrack's own radiation bookkeeping, computed without reference to any
  closed orbit) to `2e-9`. The closed form belongs to neither code.

  **The milestone's *other* pre-commitment was REFUTED, and that is the better finding.**
  Written down first: the 4D orbit's one-turn residual should be dominated by `delta` and
  equal the whole bill, `U_0/(beta0^2 E0) = 3.8e-3`. Neither half holds. The largest
  residual is **`zeta`** — `2.72 cm` of arrival time against `2.6e-3` of momentum — and the
  momentum is short by only **0.689** of the bill. One cause for both, and the
  pre-commitment ignored a feedback: losing `delta` through the arc slips the orbit in
  `zeta`, and by the time the particle reaches the cavity that slip is already large enough
  for the RF to hand back a third of what was lost. The residual is the bill *minus what the
  ring accidentally pays on the way*, reconstructed bit-for-bit as `collected - bill` — and
  only with the **tracked** bill, since the design-route one is `4.4e-3` away here, which is
  the first-order lumping error the exponent gate above is about.

  **Two fixed points, and the far one is pinned rather than avoided.** `sin(k zeta) = U/V`
  has two roots per RF period, and only the near one is an orbit a beam rides; the other is
  the **unstable** point on the far side of the bucket. Newton does not know the difference:
  started half a metre away it converges, cleanly and to round-off, onto
  `k zeta = -(pi + arcsin(U/V))`, the unstable point of the *previous* bucket. Asserted
  against that closed form. The contract therefore matches `closed_orbit_nonlinear`'s — the
  default seed is the 4D answer and lands on the stable point; a far guess makes no claim.

  **Scope.** Deterministic maps only: a stochastic radiation model has no fixed point to
  find and is refused rather than iterated. The singular case is refused too and is the
  fourth appearance of one degeneracy on this project — without an RF cavity `zeta` and
  `delta` are both eigenvalue-`1` directions of `J - I`, so there is nothing to converge
  to; N5 met it, N4 explained it, N3 first hit it. Retiring B4's and B3's private solvers
  in favour of the library one is a **separate** commit after these gates are green, so
  that the analytic count moves by this milestone's own tests and by nothing else.

  Gates: `tests/analytic/test_closed_orbit_6d.py` (16),
  `tests/reference/test_closed_orbit_6d_xtrack.py` (4). The full analytic suite is
  **1245 passed**, against 1229 after N5 — the whole difference is this milestone's own
  file, so nothing on axes A-N moved, including the two files I4 edited (`orbit.py`'s new
  solve and corrected `closed_orbit_delta` docstring, and `__init__.py`'s exports).

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

- **J2 — the octupole, and the tune that belongs to the particle.** ✅ **DONE
  (2026-08-17)** — the first quantity in the package where the tune depends on the
  *amplitude* of the particle rather than only on the machine. Baseline (numpy only):
  `Octupole`/`ThinOctupole` in `src/accsim/elements/octupole.py` with the kick
  `Δpx = −⅙k3l(x³−3xy²)`, `Δpy = +⅙k3l(3x²y−y³)`, and
  `accsim.twiss.amplitude_detuning`, the symmetric 2×2 anharmonicity
  `∂Qx/∂Jx = +k3l βx²/(16π)`, `∂Qx/∂Jy = ∂Qy/∂Jx = −k3l βxβy/(8π)`.
  - **Why the octupole and not the sextupole.** A sextupole detunes too, but only
    through *second*-order perturbation theory — quadratic in `k2`, and therefore
    **linear in the action**, i.e. indistinguishable from the octupole's term by an
    amplitude scan. The octupole's term is first order in `k3l` and falls out of a
    single phase average, which is what makes an analytic gate possible at all. The
    detuning ring therefore carries **no sextupoles**, and that background is
    *measured* rather than assumed away: `amplitude_detuning` returns exactly zero
    for a sextupole-only ring while tracking shows a real shift scaling as **k2²**.
  - **The `1/6` is anchored, not asserted.** It is the `1/3!` of the same
    normal-multipole expansion whose `n = 1` term is the xtrack- *and* MAD-X-validated
    `Quadrupole` and whose `n = 2` term is J1's sextupole. J1's lesson repeats
    verbatim — symplecticity, Maxwell/curl-free, identity Jacobian at the origin,
    `matrix()` == drift, and a potential reverse-engineered out of the kick are **all
    blind** to the coefficient, and a deliberately mis-scaled octupole (`1` for `1/6`)
    is carried through every one of them to prove it.
  - **The averaging machinery is itself anchored.** `ΔQ = (1/2π)∂⟨V⟩/∂J` is first run
    on `V = k1l x²/2`, where it must reproduce `β k1l/(4π)` — checked symbolically
    *and* against accsim's own matrix tunes on a real ring. Only then is it pointed at
    the octupole potential.
  - **The tracked gate is an order gate.** Tracking sees all orders; the closed form
    is first order in both `k3l` and the action, so a single tolerance at a single
    amplitude would swallow exactly the coefficient error the gate exists to catch.
    Over four halvings the measured detuning falls by **4** and the residual by
    **16**; measured signal ratios 4.10/4.02/4.005 against residual ratios
    17.83/16.39/16.09 (at 1024 turns, and unchanged at 2048). The mis-scaled octupole
    is caught as a clean **factor of 6**.
  - **Three traps, each closed by an assertion rather than by care.** The launch has
    `px = py = 0`, so the action is `(1+α²)u₀²/(2β)` — the ring is a palindrome and
    `α = 0` is asserted, because a silent `1+α²` rescales every slope. The working
    point (`Qx = 0.294`, `Qy = 0.637`) is chosen by scan to sit 0.137 from every
    resonance an octupole drives (`4Qx`, `4Qy`, `2Qx ± 2Qy`) and from the tunes NAFF
    reads badly. And the measurement is a **difference** against the same ring with
    the octupole removed, which cancels NAFF's own bias instead of leaving it in the
    answer at the level the detuning lives.
  - **Scope enforced, not documented.** `orbit.linearised_lattice` *raises* on a live
    octupole (feed-down is out of scope, and passing it through would report a drift),
    while `linearised_element_maps` handles it because it differentiates `track()`. A
    sympy gate derives that an octupole at dispersion has **no** first-order
    chromaticity — its `δ` term is a sextupole, not a gradient — so `chromaticity()`
    ignoring it is right and `Q″` is the honest blind spot.
    Gates: `tests/analytic/test_octupole_kick.py` (20),
    `tests/analytic/test_amplitude_detuning.py` (12),
    `tests/reference/test_octupole_xtrack.py` (7). The reference suite fits the
    anharmonicity matrix from **xtrack's own tracked particles** and agrees within
    1.1 % on the diagonal and 0.3 % on the cross terms; the sign is a probe
    (`ThinOctupole(k3l) ≡ xt.Multipole(knl=[0,0,0,+k3l])`, one ulp, opposite sign off
    by twice the kick). See CONVENTIONS.md → *The octupole and amplitude detuning*.
  - Still **out of scope**: sextupole (second-order) detuning and the octupole's own
    second-order term, octupole feed-down on a distorted orbit, resonance driving
    terms and normal-form / Lie-map machinery, dynamic aperture and frequency maps,
    decapoles and above, and octupoles as matching knobs.

- **J3 — octupole feed-down on a distorted orbit.** ✅ **DONE (2026-08-17)** —
  the deferral J2 named by name. J2's octupole was exact on axis and *refused* off it:
  `orbit.linearised_lattice` raised rather than report a drift. J3 derives what it
  refused. Expanding the cubic kick about an orbit offset `(x_co, y_co)` splits **one
  octupole into six elements** (derived in sympy, 2026-08-17 — not recalled):

      dipole       theta_x = -1/6 k3l x_co (x_co^2 - 3 y_co^2)
                   theta_y = +1/6 k3l y_co (3 x_co^2 - y_co^2)
      normal quad  k1l_eff  = +1/2 k3l (x_co^2 - y_co^2)
      skew quad    k1sl_eff = +k3l x_co y_co
      normal sext  k2l_eff  = +k3l x_co
      skew sext    k2sl_eff = +k3l y_co
      octupole     unchanged

  Five of the six are elements accsim already validates; the **skew sextupole is not
  one of them**, so J3 is two commits — the element first, then the feed-down.
  - **The gate is a three-power ladder, and no single tolerance can fake it.** The
    three feed-down orders reach three quantities the package already computes by
    independent routes, each with a *different* power of the orbit:
    chromaticity moves as `x_co` (through `k2l_eff` at dispersion — first order in
    `k3l`, where an on-axis octupole is exactly zero and `Q''` is J2's blind spot),
    the tunes as `x_co^2` (through `k1l_eff`), and the closed orbit itself as
    `x_co^3` (through the dipole). One coefficient set has to satisfy all three
    ratios at once. **Met:** measured 2.0 / 4.0 / 8.0 per halving of the steerer,
    residuals one order better in each case (8 / 16 / 32), and the three exponents
    fitted directly as 1, 2, 3 to within 2 %.
  - **The ladder is blind on its own, and the magnitude gate is blind on its own.**
    A uniformly mis-scaled octupole (`1` for `1/6`, which J2 already carries through
    every structural check) leaves all three *powers* untouched and is caught only as
    a factor **6** in magnitude; a single-quantity magnitude check is what J1 and J2
    both showed a wrong coefficient can survive. Both halves are required, and the
    test file says so.
  - **The skew sextupole's sign is not accsim's to derive.** Nothing this package
    computes reads a skew sextupole — `_sextupole_feeddown` takes `k2l` off *normal*
    sextupoles and only at `D_x` — so no analytic gate can pin it and the 30-degree
    roll identity pins the shape but not the sign. It is fixed **by probe** against
    xtrack, the J1/J2 rule: measured 2026-08-17,
    `ThinSkewSextupole(k2sl) == xt.Multipole(ksl=[0, 0, +k2sl])` with
    `Delta px = +k2sl x y`, `Delta py = +1/2 k2sl (x^2 - y^2)`, and the roll that
    reproduces it is **-30 deg** (+30 deg gives the opposite sign).
  - **The exact six-way identity has to be made non-blind.** For a *thin* octupole
    the expansion terminates, so "tracked octupole at an offset == the chain of six"
    is an algebraic identity that passes at round-off — and would keep passing with
    `k2sl_eff` sign-flipped if the vertical orbit happened to be zero. It is
    therefore run with **both** planes steered (both asserted nonzero) and each of
    the six coefficients is separately shown to break it (the H2 lesson: an exact
    residual makes a convergence gate blind).
  - **A sharp asymmetry against I2 that nothing else in the suite has.** For the
    octupole `x = px = 0` is an *exact* invariant subspace as well as `y = py = 0`
    (the kick is odd in both), where the sextupole's is only the latter — so a purely
    vertical bump moves the horizontal orbit through a sextupole and **not** through
    an octupole. Asserted at exact zero, not to tolerance.
  - **Scope.** Thin octupoles carry the quantitative gates; a **thick** one still
    raises in `linearised_lattice` for I2's `O(L^2)` reason (its offset varies across
    the body), while `linearised_element_maps` handles it by differentiating
    `track()`. Out of scope, unchanged from J2: the octupole's second-order detuning,
    `Q''`, resonance driving terms and normal-form machinery, decapoles and above,
    the 6D closed orbit, and misalignments as element attributes.
  - **Delivered in two commits, element first.** `ThinSkewSextupole`
    (`src/accsim/elements/sextupole.py`) with its own gates, then the feed-down
    branch of `linearised_lattice`. J2's two scope tests were **converted, not
    deleted** — the refusal now applies to a thick octupole only, and the whole
    on-orbit optics family (`chromaticity_on_orbit` included) answers for a thin one.
    The xtrack cross-check compares the **derived** equivalent lattice — no finite
    difference on accsim's side — against xtrack's `R_matrix`, and shows the residual
    is xtrack's own differencing step by three independent measurements. On the
    steered bendy ring accsim's design-orbit chromaticity is decisively wrong and the
    on-orbit answer closes the gap by more than an order of magnitude.
    Gates: `tests/analytic/test_skew_sextupole.py` (11),
    `tests/analytic/test_octupole_feeddown.py` (17),
    `tests/reference/test_skew_sextupole_xtrack.py` (4),
    `tests/reference/test_octupole_feeddown_xtrack.py` (6). See CONVENTIONS.md →
    *Octupole feed-down on a distorted orbit* and *The skew sextupole*.

The follow-up on J1's physics was **I2** (above, under axis I) — feed-down
belongs to the closed-orbit axis, and J1 was sequenced ahead of it only so its gate
would not be circular. That sequencing paid: I2's gates are built on J1's kick and
none of them is a rerun of it. ✅ Done.

### K. Misalignments — the magnet is not where the lattice says it is (core accelerator)

Every element up to here sits exactly where the lattice puts it. Real magnets are
displaced by tens of microns and rolled by fractions of a milliradian, and that —
not the design optics — is what sets a real machine's orbit, its coupling and its
vertical emittance. Axis K gives elements a **position and an orientation of their
own**, and derives what each does. It is deliberately *not* filed as I4: offsets
land on the closed-orbit axis, rolls land on the coupling axis, and the pair only
makes sense together.

**What this axis must not pretend is new.** The roll angle that converts a normal
`2(n+1)`-pole into a pure skew one is **already derived and recorded** —
`π/(2(n+1))`, solved in sympy during J3 and written up at CONVENTIONS.md →
*The skew sextupole*, non-uniqueness included. Extending it to the octupole
(`22.5°`) is a data point, not a gate, and K claims no credit for it.

- **K1 — transverse offsets, and the orbit statistics they produce.** ✅ **SHIPPED
  (2026-08-17)** — an `(dx, dy)` attribute on elements, plus the first
  quantity in the package that is **statistical** rather than deterministic. Full
  write-up at CONVENTIONS.md → *Misalignments — transverse offsets*. Gates:
  `tests/analytic/test_misalignment.py` (34),
  `tests/reference/test_misalignment_xtrack.py` (6). Five things the entry below did
  not anticipate, each recorded in CONVENTIONS:
  - **The whole linear effect is a constant kick, and `matrix()` needed no change at
    all.** A translation leaves the homogeneous matrix untouched, so the misalignment
    is entirely `(I − M) d` in `Element.kick()`. That is what makes β and the tunes
    *bit-for-bit* unchanged (now asserted) and hence the ensemble average legitimate.
  - **The extension point had to move** to `_kick_body` / `_track_body`, because
    `kick`/`track` became the template methods that apply the shift. The sextupole's
    and octupole's zero-strength `super().track(...)` shortcut would otherwise have
    shifted the state **twice** — caught by I1's existing affine-contract gate.
  - **A displaced *bending* dipole is a different model, and is refused.** A bend
    rotates the reference frame through itself, so entry and exit translations are not
    the same transformation; xtrack displaces the curved body as a rigid object (its
    misalignment header takes the straight branch only when `angle == 0`) and the two
    differ by `3.6e-5` where the aligned maps agree to `5.8e-9`. `Dipole` raises
    rather than approximating.
  - **The pole scan had to run the other way.** Weakening the quads toward `Q → 0`
    loses *stability* before reaching the integer (a FODO with no focusing is a drift
    ring); the scan strengthens them toward `Q → 1` instead. With the measured β-sum
    divided out, `p·|sin πQ|` is constant to 10 digits while `p·sin²` moves by 153×.
  - **The two halves are not independent, contrary to the plan below.** Because the
    module *solves* rather than evaluating the closed form, the magnitude comparison
    already pins the prefactor **and** the `sin` dependence, and the pole scan's divisor
    is that same formula's numerator — so the scan cannot pass while the magnitude gate
    fails, and "neither able to fake the other" is an overclaim. What survives is the
    one-directional half that matters: a uniformly mis-scaled kick is invisible to the
    scan, measured by building the broken machine and running it through the *same*
    scan (clean first-order divergence, constant across `Q`, at `1.0` not `0.5`).
  - **The offset half is a refactor, and its gate says so.** A displaced element is
    precisely the feed-down expansion this package has already pinned twice: a
    displaced sextupole is **I2**, a displaced octupole is **J3**. So the gate is
    *reproduce those numbers*, not *derive new ones* — an offset attribute must
    return what I2 and J3 already validated against xtrack, and no coefficient is at
    risk. Written as a consistency requirement, not as physics.
  - **The quadrupole case is exact, which the sextupole and octupole cases were
    not** (derived in sympy 2026-08-17 — not recalled). A quad's gradient is
    uniform, so a displaced quad is **exactly** a quad plus a dipole with **no
    higher terms at all** (remainder identically `0`), where I2 and J3 each split
    one element into a family. The kick is

        theta_x = +k1l*dx        theta_y = -k1l*dy

    — the **same** displacement sign giving **opposite** kick signs in the two
    planes, because accsim's thin quad is `px -> px - k1l x` but
    `py -> py + k1l y`. This asymmetry has bitten the package once already
    (`Corrector` needs `knl=[−k]` for `kick_x=+k` but `ksl=[+k]` for `kick_y=+k`,
    CONVENTIONS.md:564), so it is derived and asserted rather than trusted.
  - **Offsets alone cannot couple the planes.** Both cross-derivatives of a
    displaced quad's kick vanish identically (`∂Δpx/∂y = ∂Δpy/∂x = 0`), so no
    displacement of any unrolled quadrupole produces a skew term — only a **roll**
    can. Asserted at exact zero, not to tolerance, and it is what separates K1 from
    K2 cleanly.
  - **The new content is the statistical gate, and it has J3's two-halves shape**
    (derived in sympy 2026-08-17). For `N` uncorrelated zero-mean displacements,
    superposing I1's single-kick closed form and averaging over the displacements
    gives, **exactly**,

        <x_co^2>(s) = beta(s) * theta_rms^2 / (4 sin^2(pi Q))
                      * sum_i beta_i cos^2(psi_i - pi Q)          [EXACT]

    and only *then*, on the further assumption that the sources' betatron phases are
    spread, the textbook form

        <x_co^2>(s) = beta(s) * theta_rms^2 * sum_i beta_i / (8 sin^2(pi Q))
        x_rms(s)    = sqrt(beta(s) * d_rms^2 * k1l^2 * sum_i beta_i) / (2*sqrt(2)*|sin(pi Q)|)

    **The gate is built on the exact form, not the textbook one.** The cross terms
    die because the displacements are uncorrelated — that is a genuine ensemble
    average. The step `cos²(ψ_i − πQ) → ½` is **not** an ensemble average at all:
    the phases `ψ_i` are deterministic properties of the lattice, so averaging over
    displacement samples never touches them. The suite therefore computes the exact
    sum from the ring's own phases and **measures** the departure from `Σβ_i/8`
    rather than inheriting it. Two checks, **neither able to fake the other**:
    - the **pole** — the `1/|sin πQ|` divergence, by scanning `Q` toward an integer.
      Blind to the prefactor: any constant scales out of a power-law fit.
      ⚠️ **The scan is contaminated as naively stated**: `Q` can only be moved by
      retuning quadrupoles, which moves `β(s)` and `Σβ_i` at the same time, so
      `x_rms` does *not* scale as a pure `1/|sin πQ|`. The gate must divide out the
      **measured** `√(β(s)·Σβ_i cos²(ψ_i − πQ))` at each working point and fit the
      residual. The scan also runs into `closed_orbit`'s own singular-`(I − M4)`
      guard (`orbit.py:146`) — the *same* physics as the pole, so it is a
      consistency point, but the test must stop short of it.
    - the **magnitude** — the prefactor and the `β`-weighting, at one working point.
      Blind to the pole: a single `Q` says nothing about the scaling.

    A uniformly mis-scaled kick (`c·θ`) moves the magnitude by `c²` and leaves the
    pole **exactly** untouched — the J1/J2/J3 failure mode, closed the J3 way, by
    requiring both halves rather than one tolerance.
  - **The sign of the offset has no analytic gate, and must be probed.** `x_rms`
    goes as `d²`, so the statistical gate cannot tell whether `dx` means *the magnet
    moved right* or *the beam sits right of the magnet centre* — precisely the
    relative sign that flips silently. The I2/J3 correspondence does constrain it
    (feeding `dx = −x_co` must reproduce their numbers), but the convention is fixed
    **by probe** against `xt.Quadrupole(...).shift_x` / `.shift_y` (present in
    xtrack 0.106.4, verified 2026-08-17), the J1/J2/J3 rule that already had to be
    applied to `ThinSkewSextupole`. **Met, and better than planned:** the *thin*
    probe (`xt.Multipole(knl=…, shift_x=…)`) is bit-for-bit, so it pins the sign with
    no tolerance at all; the thick quad is the one that carries the pre-existing
    linear-matrix-vs-thick-map difference, and the displacement adds nothing to it.
  - The `1/sin πQ` pole is not a new claim — `orbit.py:29` already carries the
    single-kick form and CONVENTIONS.md:1434 records that it is a *consequence*
    there, never the definition. K1's novelty is the **ensemble**, not the pole.
  - Effort **M**. Out of scope for K1: rolls (K2), longitudinal displacement
    (`ds`), misalignment of a *thick* element's body as distinct from its ends,
    displaced **bending** dipoles (refused — see the shipped summary above), and
    correction strategies beyond I1's existing SVD steering.

- **K2 — the rolled dipole, and the first vertical dispersion *source*.**
  ✅ **SHIPPED (2026-08-17)** — the half of axis K that is genuinely new physics
  rather than a refactor with a good gate. `roll` on every element; the curved-body
  geometry K1 declined, done for the roll; `src/accsim/elements/alignment.py`.
  Full write-up at CONVENTIONS.md → *Misalignments — the roll*. Gates:
  `tests/analytic/test_roll.py` (30), `tests/reference/test_roll_xtrack.py` (10).
  Three things the plan below did not anticipate:
  - **The rolled bend's map matches xtrack exactly as well as the *aligned* one does**
    — `3.3063e-9` both, the same number to five figures, so the roll contributes
    nothing detectable of its own. Where the conjugation model misses by `5.9e-3`.
  - **The milestone's headline claim had to be narrowed, and the measurement that
    narrowed it was a surprise.** A rolled bend is the first element whose *matrix*
    carries a vertical `δ` column — that stands. But it is **not** the only way a
    machine gets `D_y`: in the exact maps, *any* vertical closed-orbit **angle**
    makes vertical dispersion, because `y += L p_y / p_z` where accsim's linear
    element has `y += L p_y`. Isolated by a **vertical steerer in an otherwise
    perfect ring**: accsim returns exactly `0`, xtrack returns `2.1e-4`, and the two
    closed orbits agree to *eight digits*. On the K2 test arc that route is the
    **larger** of the two (`−3.34e-4` against accsim's `−3.05e-5`).
    - **It is understood, not merely named.** Putting the two dropped terms back by
      hand — an extra source `p_y L (h ⟨D_x⟩ − 1)` at every element, on accsim's own
      closed orbit, where the `−1` is the drift's `1/p_z` and the `+h⟨D_x⟩` is the
      extra arc a dispersed particle travels on the outside of a bend — reproduces
      xtrack's `dy` *and* `dpy` to **0.2 %** on both the rolled and the steered ring.
    - It is **not** the same statement as J1's `−L·px·δ` note (a per-element `1e-8`
      map residual); it shares a root cause and has a ring-level consequence three
      orders larger. It predates axis K entirely — K2 is only where it becomes
      consequential — and it cannot be fixed inside a 6×6, because the terms are
      bilinear (`p_y·δ`). Representing them means **exact nonlinear maps for
      `Drift` / `Quadrupole` / `Dipole`**, which would re-baseline every gate in the
      suite: a future milestone, whose specification is
      `test_the_model_gap_is_fully_accounted_for_and_not_a_mystery`.
  - **Two type-walking helpers had to start refusing.** `closest_tune_approach` sums
    over skew-quadrupole *elements*, so a rolled element — which couples without
    being one — made it return `0.0` for a demonstrably coupled ring; it now raises
    and points at `normal_mode_tunes`. `linearised_lattice` refuses a rolled thin
    sextupole or octupole rather than emitting the unrolled feed-down split.
  - **Nothing in accsim bends vertically**, but that does **not** mean `D_y` is zero
    today — measured 2026-08-17, and the naive claim that it is was written into this
    entry and caught the same day. `_matched_dispersion` solves `D = (I − M4)⁻¹ d`
    over the **full 4D** map, so **G1's skew quadrupole at nonzero `D_x` already
    produces vertical dispersion** by rotating horizontal dispersion into `y`
    through the off-diagonal blocks of `(I − M4)⁻¹`: on an 8-cell FODO arc with one
    `ThinSkewQuadrupole(k1sl=0.05)`, `D_y = −0.912 m` against `D_x = +2.385 m`, and
    `coupled_twiss` (G2) reports it on `CoupledTwiss.disp_y`.
  - **So K2's claim is about the *source vector*, not about `D_y` being new — and
    that is the sharper statement and the better gate.** A skew quad only *rotates*
    dispersion that horizontal bends already made; it adds nothing to the source
    vector `d`. A rolled dipole adds a **new term to `d`** itself.
    ⚠️ **The control as first written does not separate them** (caught 2026-08-17,
    before implementing): *"kill the horizontal dispersion by removing the bends"*
    removes the **rolled bend** too, so it kills both routes and proves nothing. The
    statement that does discriminate is the other way round — a rolled bend in a ring
    with **no coupling element anywhere** gives `D_y ≠ 0`, where every previously
    known route to `D_y` needs one.
  - `beam_sigma` (uncoupled) and **`coupled_beam_sigma`** (`twiss.py:622`, the G2
    counterpart, whose docstring carries the `sigma_y = sqrt(Sigma_yy +
    (D_y sigma_delta)^2)` and `tilt` forms) both already read `D_y`, so a new source
    propagates into beam size with no new plumbing.
  - **A rolled dipole is *not* a simple rotation of the aligned one, and that is the
    whole milestone** (measured against xtrack 2026-08-17, before any code was
    written — the check the K1 entry's refusal said would be needed). There are two
    *different* things called a roll, and xtrack carries both as separate attributes:
    - `rot_s_rad` — a **design tilt**: the reference frame is rolled with the magnet.
      This one *is* the simple conjugation `R(−φ) · M · R(+φ)`, verified **bit-for-bit**
      against a hand-built `SRotation · Bend · SRotation` sandwich (`2.2e-18`). It has
      **exactly zero kick**: the reference particle still comes out on the design
      orbit, because the design orbit was rolled along with it. That is a lattice
      *design* feature (a genuinely vertical bend), not a misalignment, and is out of
      scope for axis K.
    - `rot_s_rad_no_frame` — the **misalignment roll** (MAD-X `EALIGN`'s `DPSI`): the
      magnet turns and the machine does not. This is K2's subject, and it is *not* a
      conjugation. The magnet's exit face is now somewhere else, so what comes back
      out is a **rigid motion** — a displacement, a pitch, a yaw and a *residual*
      roll — exactly the curved-body geometry K1 declined for offsets.
  - **The kick formula in the first draft of this entry was wrong, in two ways**
    (both falsified by the measurement above, on `Dipole(L=1, θ=0.3)` at `φ=0.02`):
    - it said the vertical kick is `θ·sin φ = 5.99960e-3`. The measured value is
      `5.910010e-3`, which is **`φ·sin θ = 5.910404e-3`** — the roll multiplies the
      *chord* of the bend, not its angle. The two agree only to first order in `θ`
      as well as in `φ`, and at `θ = 0.8` they differ by 10%.
    - it said a small roll is a **pure vertical bend**. It is not: there is also a
      vertical **offset**, `−φ·ρ(1−cos θ)`, measured `2.977421e-3` against the
      predicted `2.977567e-3` — the sagitta of the arc, tipped out of the plane. In
      accsim's own dispersion solve that offset term is the **dominant** contributor
      (`∂y/∂δ = +2.844e-3` against `∂p_y/∂δ = −2.64e-4`), so an implementation that
      kept only the angle would be wrong by an order of magnitude *and* still produce
      vertical dispersion.

    What survives from the first draft is the part that matters: the vertical effect
    is **first** order in the roll while the horizontal loss is only **second**
    (measured `k_x = 2.84e-5`, `k_px = 5.65e-5` at `φ = 0.02`), and the effect is
    momentum-dependent, which is why a rolled dipole produces vertical **dispersion**
    and not merely a vertical **orbit**.
  - **There is also real x–y coupling, which the first draft did not mention.** The
    entry rotation `+φ` and the exit rotation do **not** cancel: the exit roll comes
    back as `φ·cos θ`, leaving a net `φ(1−cos θ)` (measured `8.933e-4` at `φ = 0.02`,
    `θ = 0.3`) plus off-diagonal matrix entries (`M[x,p_y] = +5.94e-4`). So G1's
    `|C⁻|` machinery will see a rolled bend whether K2 wants it to or not, and that
    is asserted rather than discovered.
  - **The gate the first draft named cannot be built, and the reason is worth
    recording.** It wanted "a roll and a vertical steerer tuned to the same `D_y`,
    giving the same `ε_y`". Both halves fail:
    - a vertical steerer produces **exactly zero** `D_y` **in accsim** —
      `Corrector.matrix()` is the identity and its kick carries no `1/(1+δ)`, so it
      contributes nothing at all to the source vector. It cannot be *tuned* to any
      `D_y`. ⚠️ And it is **not** a physical control either, which is the second
      surprise of this milestone: measured against xtrack, the same steered ring has
      `D_y = 2.1e-4` — see the blind-spot bullet in the shipped summary below.
    - nothing in accsim turns `D_y` into `ε_y`. `equilibrium_emittances_coupled` is
      driven by `|C⁻|`, and `radiation_integrals`' `I5` is horizontal-only —
      `equilibrium_emittance`'s docstring says so in as many words. Vertical
      dispersion → vertical emittance is an unimplemented route, not a check.

    The borrowed verdict K2 uses instead is **xtrack's own `twiss()` `dy`** on a ring
    carrying a rolled bend: an independent implementation of the dispersion solve,
    validated long before K existed, that cannot be bent to agree.
  - **The roll's sign is a probe, as every sign in this package has had to be** — and
    the first draft named the **wrong attribute**. Pinning against `rot_s_rad` would
    pass on a straight element (where design tilt and misalignment roll coincide) and
    say nothing about the bend, which is the only place they differ. The probe is
    against **`rot_s_rad_no_frame`** (xtrack 0.106.4, verified 2026-08-17).
  - **The wrong model is measured, not argued.** The conjugation model misses xtrack
    by `5.9e-3` in the kick and `6.2e-3` in the matrix at `φ = 0.02, θ = 0.3` (and by
    `0.11` / `0.22` at `φ = 0.1, θ = 0.8`), where the *aligned* accsim map agrees with
    xtrack to `1.0e-9`. Six to eight orders — the K1-refusal shape, so the model
    choice cannot hide inside a tolerance.
  - **`matrix()` does change, unlike K1.** A translation leaves the homogeneous matrix
    alone; a roll does not — it rotates the transverse blocks *and* the `δ` column. So
    K1's "β and the tunes are bit-for-bit unchanged" does **not** carry over, and any
    test that assumes it must say so.
  - Effort **M**. Out of scope for K2: rolled quadrupoles as a *coupling* source
    beyond what G1's skew quad already covers, rolled higher multipoles (the angle
    rule is J3's, above), chromatic coupling (still the named blind spot at
    CONVENTIONS.md → *The skew sextupole*), and misalignment **correction** —
    K measures what misalignment does; steering it out is I1's existing job.

### L. Exact (nonlinear) element maps — the map a particle really follows (core)

Opened 2026-08-17 as a **new axis**, not an extension of K: the gap it closes
*predates* axis K entirely (K2 is only where it became consequential), and it is about
core map fidelity rather than about where a magnet sits. Every element's `matrix()` is
the Jacobian of some exact map at the origin; this axis makes the exact maps real, one
element at a time, and `track()` use them. `matrix()` is untouched throughout, so
design optics stays bit-for-bit and only the *tracked* — and hence on-orbit — quantities
move.

- **L1 — the drift's exact map, and dispersion from an orbit angle.** ✅ **SHIPPED
  (2026-08-17)** — `x += L·px/pz` in place of `x += L·px`, with its conjugate
  longitudinal partner. Full write-up at CONVENTIONS.md → *The drift's exact map*.
  Gates: `tests/analytic/test_drift.py` (13),
  `tests/analytic/test_exact_drift_dispersion.py` (5),
  `tests/analytic/test_symplectic_canonical.py` (14),
  `tests/reference/test_drift_xtrack.py` (5, the bend-free steered ring among them).
  - **Two candidate maps, and xtrack's default is the wrong one.** `xt.Drift()` is the
    *expanded* `px/(1+δ)`; the exact `px/pz` needs `model="exact"`. accsim implements
    the exact one — `test_drift.py`'s existing symbolic derivation had already committed
    to it — and matches xtrack to `4.4e-16`. The two differ by `1.5e-6` to `1.7e-4` at
    large angles, so the analytic gate discriminates them; every reference cross-check
    must set the model explicitly or chase a phantom `O(angle³)` bug.
  - **The dropped term has a canonical partner, discovered by measurement.** Per drift,
    `M[x,δ] = M[zeta,px] = −L·px`, equal in size. A map with one and not the other is
    **not symplectic** and wrong at *first* order — which is why the transverse and
    longitudinal halves cannot be split across two milestones. K2's write-up did not
    have this; its "two dropped terms" were both about transverse motion.
  - **`(zeta, δ)` is not canonical, so accsim's own symplecticity check rejects the
    correct map** (residual second order in amplitude) while passing the cruder linear
    one (three independent shears always pass). Needed a prior commit adding
    `is_symplectic_map_canonical`, which is exactly zero for the right map and catches
    the transverse-only half-fix at first order. **The more faithful map failing the
    existing gate is the trap of this milestone**, and asserting the rejection matters
    as much as asserting the acceptance.
  - **The clean gate is a bend-free ring**, which K2's write-up did not anticipate.
    Setting `h = 0` kills the `+h⟨D_x⟩` half of K2's formula *and* makes `D_x ≡ 0`, so
    the drift's term is the whole effect: `D_y` goes from exactly `0` to `0.2590571`,
    which is xtrack's own answer to seven figures, on a ring with no bend and no
    coupling element. K2's own arc has **no drifts**, so L1 moves none of its numbers.
  - **The blast radius was 29 analytic tests across 9 files** — the re-baseline the
    axis-K entry predicted. Every one is a claim restated with its measured *order*
    (ratios of 4, 8, 16, and derived coefficients like `1.5·L·py²`), not a loosened
    tolerance. Three were discriminating gates whose teeth were re-verified after
    restating.
  - **Five costs, recorded rather than worked around:** element-by-element tracking is
    no longer one `transfer_map` product; the drift is a **first-order chromatic
    element** (tracked chromaticity is now 45% of the analytic natural chromaticity, so
    I3's "tracking is blind to chromaticity" is only partly true and every
    tracked-vs-derived gate needs a baseline subtracted — *L2 raised that to 58% on the
    same arc and to 100% on a bend-free ring; the baseline subtraction stays, because
    the dipole is still linear*); a drift has real amplitude
    detuning; Newton's basin shrank, because the exact map returns `NaN` for a particle
    with no forward momentum instead of inventing a trajectory; and `linearised_lattice`
    cannot represent the new pair, so it omits it — legitimate, since the terms carry no
    gradient, but it means dispersion from that route is silently the old zero.
  - A **numerical** point worth carrying forward: evaluating the longitudinal term as
    `1 − E/(E₀·pz)` cancels two numbers of size 1 and would have broken the
    design-optics gates at `3.6e-8`. The rationalised form is `2.7e-13`. Expect the same
    trap in the quadrupole and dipole.

- **L2 — the quadrupole's momentum-dependent map.** ✅ **SHIPPED (2026-08-17)** —
  `k1 → k1/(1+δ)` and `x' = px/(1+δ)`, with the conjugate path-length term. Full
  write-up at CONVENTIONS.md → *The quadrupole's momentum-dependent map*. Gates:
  `tests/analytic/test_exact_quadrupole.py` (16),
  `tests/reference/test_quadrupole_xtrack.py` (4). It closes what L1 half-opened:
  tracking sees the drifts' *and* the quadrupoles' share of natural chromaticity, and
  only the dipole's is left.
  - **The milestone's own prescribed gate shape does not transfer, and the entry above
    was wrong to assume it would.** L1 discriminated at *large angles*; large angles are
    exactly where this map is deliberately wrong. There is **no closed form** for the
    exact quadrupole Hamiltonian — the square root and the quadratic potential do not
    commute — so a code either expands the root and solves exactly (MAD-X, xtrack's
    `mat-kick-mat`, this) or splits the exact one and integrates. accsim takes the
    closed form *because* `matrix()` must stay the exact origin-Jacobian of `track()`,
    which is what bounded L1 and bounds this; a sliced map's origin Jacobian is the
    sliced approximation to the cos/sin block, and every design-optics gate would move.
  - **The discriminating axis is large `δ`, and there is an exact identity along it.**
    Substituting `px = (1+δ)p̃` makes the ring at momentum `δ` *literally* the design
    ring with every `k1` rescaled by `1/(1+δ)`. Tracked tunes off momentum equal the
    design tunes of the rescaled lattice to `1e-15` out to `δ = 0.05`, which pins every
    order in `δ` at once — a wrong power, or the factor in one plane only, fails at
    `O(δ)` where the first-order chromaticity number alone would not separate them.
  - **The chromaticity residual has a named owner, not a tolerance.** Tracked `dQ/dδ`
    matches `natural_chromaticity` on a bend-free ring, and what is left is the
    *trapezoid error of `slices`* — falling by 4 per doubling, asserted as that order.
    Two controls attribute the rest: a **thin**-quad ring already had 100% after L1
    (a thin kick is momentum-independent), and swapping a zero-angle `Dipole` for an
    identical-matrix `Drift` moves the tracked answer from **48% to 100%**, so the
    remaining blindness is provably the dipole's map — L3's.
  - **The "removes L1's inconsistency" prediction was wrong.** `Quadrupole(L, 0).track`
    is the *expanded* drift, not the exact one: the gap goes from **first** order
    (`O(px·δ)`) to **third** (`O(px³)`), narrowed rather than closed. Short-circuiting
    `k1 == 0` would close it only by making the map discontinuous in `k1`, so the
    residual is asserted — cubic in the angle and independent of `k1` — instead.
  - **Blast radius five analytic tests and two reference, against L1's 29 and 5**,
    because at `δ = 0` the transverse map *is* the linear matrix (to 1 ulp) and only
    `ζ` moves. Two consequences carried forward: `SkewQuadrupole` took the same map
    through its 45° roll conjugation (the same magnet must not behave two ways
    depending on how it is spelled), and `is_symplectic_map` was found to **accept** a
    correct exact map at small amplitude — the `(ζ,δ)` residual is second order *and*
    `1/γ₀²`-suppressed, `8.4e-10` on a `γ₀=20` ring, under its own default `atol`.
    The wrong check does not only reject correct maps; it can pass one for no reason
    connected to symplecticity.
  - **The `ζ` cancellation L1 warned about was real**, and is avoided the same way:
    `L(1 − 1/rvv)` rationalised through `(1+δ) + E/E₀`, split from the path integral,
    which is itself written division-free so there is no `K = 0` branch.
- **L3 — the dipole's exact map, and the close of K2's account.** ✅ **SHIPPED
  (2026-08-17)** — a uniform field's flow is a **circle**, so the pure sector bend gets
  a map exact in the *angles as well as* `δ`. Full write-up at CONVENTIONS.md → *The
  dipole's exact map*. Gates: `tests/analytic/test_exact_dipole.py` (14),
  `tests/reference/test_dipole_xtrack.py` (3 new), and the converted
  `tests/reference/test_roll_xtrack.py`. K2's specification test now asserts the
  *package's* answer against xtrack: `1.7e-8` on the rolled ring and `3.5e-9` on the
  steered one, where the hand reconstruction it replaced managed `2e-3`.
  - **The gate shape is a third one again, and it is an identity.** L1 discriminated at
    large angles, L2 at large `δ`; a bend is exact in both, so the gate is both at once —
    the map re-derived from plane geometry (circle meets exit face, sharing no arithmetic
    with the implementation) agreeing to `1e-15` out to `1.5 rad` and `δ = 0.3`, where the
    linear matrix is wrong by `2.3e-2`. Plus the exact Hamiltonian as an invariant
    (`4.4e-16`), which needs no reference implementation at all. Against xtrack's
    `bend-kick-bend`: `1.9e-16`.
  - **The `k1` split is *forced*, and that is the difference from L2's refused
    discontinuity.** With `k1 = 0` the vertical equation is a quadrature because `p_y` is
    conserved — that is *why* a closed form exists; with `k1 ≠ 0` it becomes an ODE with
    an `s`-dependent coefficient. The geometric term and vertical focusing are mutually
    exclusive in closed form, so the combined-function bend keeps the affine map rather
    than taking MAD-X's expanded one, which drops the very term this milestone adds.
  - **The ROADMAP entry above was incomplete, and only an exact bend could show it.**
    K2's `Δd_y = p_y·L·(h⟨D_x⟩ − 1)` is the `δ` column alone. The exact bend *also*
    couples the planes — `M[y,x] = p_y sin θ`, `M[y,p_x] = p_y ρ(1−cos θ)`, which are
    `p_y` times the bend's own dispersion entries — and that path transports horizontal
    dispersion into the vertical. On a real arc it is the larger of the two. So an
    upright sector bend on a vertical orbit is a **coupling source**, new to the package,
    and `closed_twiss_on_orbit` now raises on such a ring where the design route does not.
  - **The planes are not mirror images.** `M[y,δ] = −p_y ρ sin θ` but
    `M[x,δ] = −p_x ρ sin θ cos θ`: `p_x` is not conserved, so its response feeds back
    through the bend's own focusing (`ξ'' + h²ξ = (3h/2)sin 2hs`). Symmetrising is 8%
    wrong and invisible to every design-optics gate.
  - **The numerical trap L1 predicted was the biggest yet.** xtrack's own form builds an
    answer of size `x` from a numerator of size `h`; its origin Jacobian is `3.2e-9` and
    *degrades* as the step shrinks. Rearranged it is `4.9e-15` — and with no division by
    `h` left, so `Dipole(L, 0)` *is* `Drift(L)` with no branch, closing an inconsistency
    L1 recorded and re-baselining L2's control in the process.
  - **One cost that is not the body's:** a *rolled* bend is now symplectic only to first
    order in the roll (`4.7e-8`), because `frame_change()` is the affine linearisation of
    the true frame change — "exact for accsim's linear elements", which the body no longer
    is. `matrix()`/`kick()` are untouched, so every K2 number stands.
  - **A second reference gap closed on the way.** `test_orbit_optics_xtrack.py` had to
    open by stating that accsim's linear elements do not feed down off axis, worth
    `6.4e-4` in β against xtrack, and hoped for "a number to improve on". It is now
    `5.4e-10`, with the old `6.4e-4` *moved* onto the design route where a bilinear term
    legitimately cannot live. Its downstream β-change gate went `1.35e-3 → 2.8e-7` of the
    effect and was tightened `5e-3 → 1e-6` rather than left loose.
  - **One dark code path found and closed:** nothing in the suite tracked an *edged*
    pure sector bend — `test_dipole_edges.py` only ever compared matrices — so the
    `Edge · body · Edge` composition's order and its `h` argument had no gate.
  - Blast radius **nine analytic tests and three reference**, against L1's 29 and L2's 5.

- **L4 — the curved quadrupole's expanded map, and a model boundary found on the way.**
  ✅ **SHIPPED (2026-08-18)** — the last element whose `track` was its `matrix` now has
  one: `mat(L/2) · kick(h k1 L) · mat(L/2)`, MAD-X's `track_thick_cfd` with F2's Maxwell
  curvature-sextupole term, which is *exactly* `xt.Bend(model="mat-kick-mat")` with one
  uniform kick — reproduced to **1.0e-16** on all six coordinates. Full write-up at
  CONVENTIONS.md → *The curved quadrupole's expanded map*. Gates:
  `tests/analytic/test_curved_quadrupole.py` (25) and
  `tests/reference/test_dipole_combined_xtrack.py` (4 new). Both waiting gates behaved as
  written: the 56% control closed to 100%, and
  `test_a_combined_function_bend_is_deliberately_left_on_the_linear_map` failed loudly.
  - **The headline is not "56% → 100%", and the entry above was wrong to assume it would
    be.** The expanded family solves `x' = px/(1+δ)` where the exact curvilinear equation
    is `x' = px(1+hx)/p_z`, keeping the `(1+hx)` metric factor **only in the path
    length**. On the dispersed orbit that factor *is* F2's `h(γ_x D_x − 2α_x D_px)` /
    `γ_y h D_x` group — the term that largely cancels the geometric `−β_x h²` focusing. So
    a **bending** combined-function magnet's tracked chromaticity converges to *F2 minus
    that group*, and slicing does not close it. A **straight** gradient magnet has `h = 0`,
    both the metric group and the Maxwell kick vanish identically, and *that* is the case
    the 56% control measures — 100% there is real and says nothing about the bending case.
  - **The gap is measured from both sides, which is what makes it a boundary rather than
    a defect.** On one AG arc: xtrack's two exact families and accsim's
    `natural_chromaticity` all give `(+0.114815, −0.015121)`; xtrack's converged
    `mat-kick-mat`, accsim's sliced tracking, and F2-minus-the-metric-group all give
    `(−0.13973, −0.13206)`. Two families, two numbers, accsim's two routes one each.
  - ⚠️ **One diagnostic regressed, and it is named rather than buried.** On a bendy
    combined-function ring the **converged** tracked chromaticity is now further from the
    truth than the pre-L4 blind map was, because that map contributed nothing where this
    one contributes an uncancelled `−β_x h²`. That is the F1 failure mode. (At *one* kick
    per magnet the splitting error happens to push the answer back across the true value,
    so the unsliced ring reads closer — by luck, and it reverses on any slicing; both are
    asserted so the unsliced number is never quoted as an improvement.) The **map** is
    strictly better everywhere (it was `δ`-blind, it is now exact in `δ`), and
    `natural_chromaticity` — the deliverable, and what the exact models agree with — is
    untouched.
  - **The invariant survived, and not for free.** Two half-length Hill solutions compose
    to the full one identically and a cubic potential's kick has zero Jacobian at the
    origin, so `matrix()` is still the *exact* origin Jacobian of `track()` — `2.2e-16`,
    improving with the step. That is what still rules out every slicing family.
  - **A bend is now discontinuous in `k1` at zero** (`1.8e-5`, not shrinking with `k1`),
    and the obvious guess about its order was wrong: it is **quadratic** in the
    coordinates, not L2's `O(angle³)`, because it is two things at once — the expanded
    square root, which for a bend enters `px' = h p_z − h` already at `O(p²)`, and the
    dropped metric factor, whose signature is a bilinear `h x px`. Both matched to their
    closed forms.
  - **Four tests re-based, across three files**, against L1's 29, L2's five and L3's
    nine. Two of them are the same fact: a combined-function bend **moved symplecticity
    groups** (its `track` is no longer linear in `δ`, so the non-canonical check now
    rejects it), and a *rolled* one inherits L3's first-order-in-roll cost by the same
    mechanism — `6.2e-8` against the pure bend's `4.7e-8`, with `matrix()`/`kick()`
    untouched.

- **L5 (candidate) — the curvilinear metric term for combined-function bends.** What L4
  found and deliberately did not build. The missing piece has a single generator,
  `H_m = h x (px² + py²)/(2q)`, which produces *both* dropped terms at once (the `h x px`
  in `x'` and the `−h p²/2q` in `px'`) and has **identity Jacobian at the origin**, so it
  would slot into the existing splitting without touching the invariant that bounds the
  axis. Its flow is a Riccati system with a closed form. The reason to defer rather than
  do it: **neither MAD-X nor xtrack implements it**, so accsim would be inventing a model
  and trading a bit-for-bit reference cross-check for self-consistency — which is the one
  trade this project's validation strategy does not make. The gate already exists and is
  precise: tracked chromaticity on a bendy combined-function ring must move from
  `(−0.13973, −0.13206)` to `natural_chromaticity`'s `(+0.114815, −0.015121)`, with
  `xt.Bend(model="bend-kick-bend")` as the arbiter. Effort **M**.


### M. The optics off-momentum — the derivative of the machine (core accelerator)

Every optics quantity on axes A-L describes the machine at **one** momentum. Axis M
adds the derivative: how `beta`, `alpha` and the tune move as `delta` does. Opened
2026-08-25 as a new axis rather than an extension because it is not a new *element*
or a new *effect* — it is the same lattice differentiated, and its natural home is
the Twiss module rather than any magnet.

The direction was chosen on the project's usual filter, applied mechanically: every
quantity `xtrack`'s `twiss()` reports was compared against what accsim reports, and
the chromatic family (`bx_chrom`, `ax_chrom`, `wx_chrom`, `ddqx`, `ddqy`, `ddx`) was
the one clear gap with **two** independent arbiters already wired (xtrack and, since
D3, MAD-X).

- **M1 — the chromatic functions and second-order chromaticity.** ✅ **SHIPPED
  (2026-08-25)** — `chromatic_functions()` returns `dbeta/ddelta`, `dalpha/ddelta`
  and the MAD8 normalised `a`/`b`/`w` at every element boundary;
  `second_order_chromaticity()` returns `(Q''_x, Q''_y) = d^2Q/ddelta^2`. Both are
  central differences of `propagate_twiss_on_orbit` / `tunes_on_orbit` in `delta`,
  which is deliberate: it is the method **both** references use, so a disagreement
  arbitrates the maps rather than the truncation order of two different expansions
  (B2's argument). That also made M1 a thin layer over machinery L1-L4 and I1-I3
  had already built — the off-momentum closed orbit and the per-element linearised
  maps — which is why the milestone's weight is entirely in its gates.

  What it delivered, and what it found:

  - **The chromatic functions are validated in full.** They match xtrack **element
    by element** around a ring with bends, to `2e-4` relative on a curve whose
    `b_x` swings through more than 0.5 — a per-element readout rather than a single
    scalar, because a one-turn number can be right by cancellation and a curve
    cannot (J1's lesson).
  - **`Q''` was exact where M1 could check it, and (M2) is exact where it could not.**
    On a **bend-free** ring it reproduces a sympy closed form, and xtrack **and**
    MAD-X agree with it — a genuine three-code agreement. The closed form exists
    because a thin quadrupole carries **no** `1/(1+delta)` at all (its kick changes
    every particle's momentum equally) and the exact `Drift`, linearised at the
    origin, is simply `Drift(L/(1+delta))`; so a thin-lens ring's entire momentum
    dependence is one substitution, and the tune can be differentiated twice
    symbolically. That gate is the **control**, not a warm-up: it proves the
    second-difference machinery, the quad map, the drift map and the phase
    accumulation are all right at second order.
  - **On a ring with bends the three codes split**: accsim `0.79307`, xtrack
    `0.75202`, MAD-X `0.70441`, while agreeing on `Q` to **ten** digits and on `Q'`
    to seven. MAD-X is *further* from xtrack than accsim is, which rules out the
    easy reading ("two references agree, so accsim is wrong").
  - ⚠️ **M1 concluded "the split is provably not accsim's maps". That conclusion was
    wrong, and M2 (below) retired it.** What M1 actually established stands: accsim's
    `Dipole` Jacobian equals `xt.Bend`'s to `1.2e-9` entry by entry **on the
    off-momentum closed orbit** — weak focusing, dispersion generation and path
    lengthening alike — and the two closed orbits agree to `1e-9`. What it inferred
    from that does not: it generalised **one element** to "identical maps", and the
    element it never checked off-momentum — the **drift** — differs by `1e-7`, a
    hundred times more. accsim's `Drift` is exact, xtrack's default is paraxial, and
    that is the entire gap. The scaling law M1 measured and attributed to "the
    longitudinal constraint" (**exactly zero without bends, quadratic in the bending
    angle**) is the signature of the drift model, and M2 derives it as such. Two of
    M1's supporting numbers were also below the resolution of the effect: the tune
    difference in question is `2e-8`, while "two tune routes agree to seven digits" is
    `1e-7` absolute, and the `5e-9` Jacobian threshold called "the finite-difference
    floor" has an actual floor near `7e-10`. **The milestone's real result is the one
    it did deliver — the validated chromatic functions — plus the scaling law that let
    M2 find the cause in one sweep.**
  - **A pre-committed expectation was wrong, and the correction is a result.** The
    sextupole's contribution to `Q''` was expected to be linear in `k2l`. It is
    **quadratic** (measured exponent `2.02`), and the reason is that the two
    chromaticities take the same element at two different powers: a sextupole at
    dispersion feeds down a gradient `k2l D_x delta`, which is first order in
    `delta` and therefore lands on `Q'` *linearly* (exactly so — `dQ'/k2l` is one
    number to nine digits) and cannot reach a second derivative at all by that
    route. `Q''` is reached only at second order in the perturbation. The gate is
    now the pair of **exponents**, which discriminates where no tolerance does.
  - **xtrack's own `ddqx` is noise-limited below `delta ~ 5e-5`** (a second
    difference amplifies closed-orbit noise as `1/delta^2`), running to `1.10` at
    `1e-5` against its converged `0.7520`. accsim's default step and the gates'
    steps both sit in the flat middle, and the analytic suite gates the
    **convergence order** — halving `delta` quarters the residual against the
    symbolic answer — rather than a value at one step (B4's argument, applied to a
    derivative).
  - **xtrack's nonlinear dipole fringe is invisible on-momentum** (it moves neither
    tune at `delta = 0`, to thirteen digits) and acts only at *second* order in
    `delta`, in the vertical plane alone. accsim's `Dipole` uses the linear
    hard-edge kick, which is the identity at `e1 = e2 = 0`, so `edge='suppressed'`
    is the apples-to-apples setting — and since `Q''_x` is identical under both
    settings, the edge model is **not** what explains the horizontal split.

- **M2 — which code is right, and why the other two are not.** ✅ **SHIPPED
  (2026-08-26)** — the deferral M1 named, and the first milestone on this axis that
  opened with a genuinely unknown answer. It closed by **overturning M1's own
  conclusion**, which is the result rather than an embarrassment attached to it.

  **The answer: accsim is right, and the split is the drift model.** accsim's
  `Drift` is exact (`x += L px / sqrt((1+delta)^2 - px^2 - py^2)`, shipped in L1);
  xtrack's default `Drift` and MAD-X's TWISS drift are paraxial
  (`x += L px / (1+delta)`). Setting `xt.Drift(model="exact")` collapses the whole
  5% disagreement to the two codes' own second-difference noise — `Q''_y` agrees to
  **nine digits**, the off-momentum closed orbits to `1.4e-15` (from `2.9e-11`), and
  the one-turn Jacobians to `5.9e-10` (from `5.2e-7`).

  - **M1's hypothesis was wrong, and so was M1's headline.** The pre-committed
    suspect — what each code holds fixed longitudinally when closing an off-momentum
    orbit — is not merely unsupported, it is impossible: without RF the transverse
    map cannot depend on `zeta` at all, in *either* code. The step that found the
    real cause was not a derivation but a **localisation**: multiply xtrack's own
    per-element Jacobians into a one-turn map and read its trace. That reproduced
    xtrack's own `Q''` (`0.75210` against its `twiss` value `0.75205`), which
    exonerated its tune extraction and proved the disagreement was in the maps —
    the exact opposite of what M1 had asserted. Sweeping every element then took one
    run: `Quadrupole` `5.3e-10`, `Dipole` `6.7e-10 .. 1.1e-9`, `Drift`
    `6.4e-08 .. 1.0e-07`, all off-momentum; on-momentum every element sits at
    `1e-10`.
  - **Why the drift went unchecked, and the transferable lesson.** L1 had shipped the
    drift *exact*, so it read as settled. But L1 validated the drift's **map**; it did
    not validate that map's agreement with xtrack's **default configuration**, and
    those are different claims. Any element whose reference offers more than one model
    carries the same trap. M1's reasoning was valid — identical maps about identical
    orbits cannot give different tunes — from a premise it had checked on exactly one
    element out of three.
  - **The scaling law was the answer all along.** M1 measured the gap as exactly zero
    without bends and quadratic in the bending angle, and read it as dispersion acting
    twice. It is: the exact and paraxial drifts differ by the relative factor
    `(px^2 + py^2)/2`, they are the *same map* when the closed orbit is straight, the
    orbit acquires `px ~ D_px delta` only when the ring bends, and `D_px` is
    proportional to the angle. So the difference is `O(angle^2 delta^2)` — invisible in
    `Q` and `Q'`, landing squarely on `Q''`. The analytic suite now reproduces that
    sweep **inside the arbiter**, with neither reference code in the room.
  - **The gate: a ring whose `Q''` is derived rather than compared.**
    `tests/_m2_minimal_ring.py` builds `ThinQuadrupole(+0.9) Drift(0.5)
    Dipole(1.0, 0.12) ThinQuadrupole(-0.9) Drift(0.5)` and produces its `Q''` from
    lab-frame geometry at **sixty** decimal digits, once per drift model:
    exact `0.3073788909 / 0.2985909737`, paraxial `0.2932235794 / 0.2938154492`.
    accsim converges onto the exact pair at second order in `delta`; xtrack's default
    reproduces the paraxial pair to `4e-6`; xtrack's `model="exact"` reproduces the
    exact pair to `3e-6`. Both models are separately *confirmed* rather than merely
    reconciled, which is a stronger statement than either code agreeing with the other.
  - **The pre-committed ring was wrong in two ways, and saying so is part of the
    result.** "A single thin quadrupole plus a single sector bend" cannot show the
    effect at all — the effect lives in the **drift**, which that ring does not have —
    and a sector bend focuses horizontally only, so one quadrupole leaves the vertical
    plane unstable. Both are now asserted rather than assumed: the ring's stability and
    distance from the half integer are gated before any number is measured on it.
  - **The bend had to be re-derived, not ported.** `exact_sector_bend_map` is
    rearranged so that no two numbers of size one are ever subtracted (a rationalised
    `pz - 1`, an `arcsinc` for a difference of arcsines, no `1/h`); transcribing it
    into the arbiter would have tested that rearrangement against itself. The
    independent construction — the circle of radius `p_perp/h` meeting the exit face —
    agrees with it to `2.9e-15` over random states, which is a genuine new
    cross-check of L3's map as well as the arbiter's licence to be called a derivation.
  - **MAD-X is named, not reconciled.** On the minimal ring MAD-X lands `7.0e-4` from
    the paraxial answer horizontally and `7.3e-4` vertically — **the same residual in
    both planes** — while the drift-model split is `1.42e-2` and `4.78e-3`. So the
    drift accounts for 95% of MAD-X's gap in `x` and 82% in `y`, the difference being
    the denominator rather than MAD-X behaving differently, and a leftover of one size
    in both planes is the signature of one property of its maps. That leftover is its
    second-order TWISS expansion; its TWISS has no exact-drift option, so agreement
    with MAD-X on a dispersive ring is unreachable by construction rather than a bug
    to chase — and the ring's thin quadrupoles rule out a thick quadrupole's expansion
    order as the source of what is left.
  - **The 5% xtrack disagreement is deliberately still asserted** in the reference
    suite on xtrack's *default* settings. It is a real difference between two
    documented models, and a future change that quietly removed it would mean accsim
    had stopped being exact.

- **M3 — second-order dispersion.** ✅ **SHIPPED (2026-08-26)** — `second_order_dispersion()`
  gives `ddisp_x`, `ddisp_px`, `ddisp_y`, `ddisp_py` (and the first-order pair, free from
  the same three orbits) at every element boundary: where an off-momentum particle sits
  once the straight-line term is used up. It is a second difference of the **tracked**
  closed orbit, so it is a cross-check of the exact maps rather than a re-reading of the
  linear ones — every `Element.matrix()` in this package is `delta`-independent, so a
  purely affine machine has *no* second-order dispersion at all.

  **The milestone's pre-committed warning was wrong, and overturning it — with the
  condition that bounds the correction — is the result.** M2's entry above told M3 that
  "a second-order orbit quantity on a dispersive ring is exactly what the drift model
  moves, so any `ddx` cross-check must set `xt.Drift(model="exact")`". On a ring whose
  on-momentum orbit runs down the axis it must not, and the reason is one power of
  `delta`. On a **steered** ring it must after all, and saying so is the other half.

  - **On an unsteered ring the drift model is invisible here.** Inside xtrack, on one
    line with only `xt.Drift`'s `model` changed, `ddx` moves in the **ninth** significant
    digit while `ddqx` moves by **5%**. The arbiter says the same with no reference code
    in the room: its exact-drift and paraxial-drift `ddx` agree to `1e-15` while its
    `Q''` differs by `1.4e-2`.
  - **Why, and exactly when.** The exact drift exceeds the paraxial one by
    `L px (px²+py²)/(2(1+δ)³)`. With `px = a + bδ` on the closed orbit — `a` the
    **on-momentum** orbit angle, `b` the dispersion angle — the `δ²` part of that
    difference is `3ab²`, which vanishes when *either* factor is zero. `a = 0` is every
    ring in this project's suites and in both reference suites; `b = 0` is a ring with no
    bend. `Q''` is split regardless, because it differentiates the **Jacobian** about the
    orbit and `d/dpx` of the same term is `O(b²δ²)` — one order lower, and free of `a`.
    The orbit and the optics about it are separate objects, and a map difference can
    reach them at **different powers**; never carry a finding about one to the other
    without checking the order.
  - **The bound is gated, not merely stated.** M2's arbiter now takes a steerer. A 10
    mrad kick splits `ddisp_x` between the two drift models by **6.8e-3** relative
    (against `1e-15` at zero kick on the same ring), and the two exponents are asserted:
    first order in `a` (ratio `2.001` per doubling of the kick) and second order in `b`
    (ratio → `4` per doubling of the bending angle), with the split returning to machine
    zero when the bend is removed. The **first-order** dispersion splits too, by a
    different power that survives with no bend at all, so the reference suites' `disp_x`
    comparisons are safe only because their rings are unsteered. Writing the headline
    unconditionally would have been M1's own recorded failure — generalising from the
    cases that happened to be checked — with the counterexample sitting three tests away
    in the same file.
  - **All three codes agree, where all three disagreed on `Q''`.** accsim matches xtrack
    element by element around a bendy ring to `2e-6` relative — on **either** of xtrack's
    drift models — and MAD-X to `2e-7` after the change of momentum variable. The same
    ring, the same two references, split `Q''` three ways by 5%.
  - **MAD-X is reconciled rather than named, for the first time on this axis.** Its `DDX`
    is the `pt^2` coefficient — half a second derivative *and* in the energy variable:
    `DDX = (d²x/dδ² − (dx/dδ)/γ₀²)/(2β₀²)`. Reading it as a plain `½ d²x/dδ²` is wrong by
    `4.6e-4` at `γ₀ = 20`, which passes for round-off, and by `7.6e-3` at `γ₀ = 5`. Both
    energies are run, because the error *moves with the beam energy* and a single-ring fit
    could not distinguish a convention from a coincidence. M1 and M2 both ended with MAD-X
    permanently out of reach on `Q''` (its TWISS drift is paraxial and cannot be changed);
    that obstacle simply does not exist here.
  - **A trap found on the way: MAD-X renormalises `PX` at non-zero `DELTAP`.** Sampling
    its own table at three `DELTAP` values — the trick M1 and M2 used to check its tunes
    without trusting a `DD` column — works for `X` and is silently wrong for `PX`, which
    comes back divided by the *shifted* reference momentum. The second difference returns
    `d²px/dδ² − 2 dpx/dδ`: `-0.3083` where the true derivative is `+0.4381`, the wrong
    **sign**. Asserted in the reference suite rather than merely avoided.
  - **A second derivation, with no bend anywhere in the ring.** Momentum enters a
    *paraxial* drift only as `L → L/(1+delta)`, and a thin quadrupole's kick and a
    corrector's kick carry no momentum dependence at all — so a thin-lens corrector ring's
    closed orbit is a **rational function of `delta`**, and sympy produces its `ddx` in
    exact arithmetic, no floating point and no iteration. accsim's exact drift departs
    from that closed form at the **third** power of the kick angle (measured ratio `27.0`
    per tripling), which is the discriminating gate: a uniformly mis-scaled second-order
    dispersion would show a residual growing like `theta` and would pass any tolerance
    chosen on one ring. That residual *is* the drift-model split under another name — the
    same `3ab²` law with both factors carried by the kick, since the ring has no bend —
    and at `theta = 0.02` it is `2e-3` of `ddisp_x`, six orders above what the same split
    is worth on an unsteered ring. A *symmetric* thin FODO with the kick at the entrance is
    degenerate — sympy returns a numerator of degree one, so its `ddx` is not small but
    **identically zero** — which is kept as a control that no spurious additive term can
    hide behind.
  - **The value gate reuses M2's arbiter rather than adding a claim.** `dispersion_orders()`
    differentiates the same sixty-digit fixed point M2's `Q''` came from; accsim converges
    onto it at second order in the step (halving `delta` quarters the residual, over two
    halvings). The arbiter's hand-written **drift** — unpinned by M2, which was safe while
    the *bend* carried the answer — is now pinned against accsim's `Drift` directly, since
    the drift is the element this milestone is about.
  - **It is defined where M1's object is not.** `chromatic_functions()` differentiates a
    Courant-Snyder `beta` and refuses an x-y coupled lattice; a closed orbit exists all the
    same, and a skew quadrupole standing at horizontal dispersion gives the orbit a
    *vertical* second-order dispersion. Routing through the tracked orbit rather than the
    on-orbit Twiss is what buys that, and it is asserted rather than left to accident.
  - **`tol` is tighter than the orbit solve's own default on purpose.** A second difference
    divides by `delta²`, so `closed_orbit_nonlinear`'s `1e-14` would land as `~6e-9` of
    noise at the default step — a third of the truncation error, for nothing. `1e-15` takes
    the orbit to `~1e-19` and the noise disappears underneath.

  **Axis M is now closed**: no written candidate remains on it.

### N. Spin — the particle's own magnet, and the polarization it builds up (core accelerator)

Opened 2026-08-26 as a **new axis**. Every quantity on axes A-L is a property of the
particle's *position and momentum*, and axis M is that machine differentiated; a charged
particle also carries a **spin**, and that spin is acted on by every field the lattice
contains. It is the last piece of single-particle physics this package has no notion of
at all, and in an electron ring it is not a curiosity: synchrotron radiation slowly
polarizes the beam (Sokolov-Ternov), the polarized beam is what makes resonant
depolarization possible, and resonant depolarization is how LEP measured its own energy
to a part in a million and how FCC-ee intends to.

Chosen on the project's usual filter — **can an independent code arbitrate the answer?**
— and it passes on both arms at once, which nothing else remaining does:

- `xtrack` 0.106.4 (already installed, already wired into the reference suite) tracks
  `spin_x/y/z` through its thick magnets, reports the closed spin solution at every
  element boundary from `twiss(spin=True)`, and computes a full linearised polarization
  analysis (`polarization_analysis=True`: spin tune, buildup time, depolarizing
  component, `dn/ddelta`).
- Independent **closed forms** exist for the headline numbers — the precession rate
  `nu_0 = G gamma`, the Sokolov-Ternov equilibrium `8/(5 sqrt3)` and its time constant —
  so the reference is not the only judge.

Two properties worth stating before the milestones, because they set the axis's shape:

- **Spin does not feed back into the orbit.** Unlike L1, this axis re-baselines
  *nothing*: not one existing number moves, `matrix()` and `kick()` are untouched, and
  the symplecticity invariant that bounds axis L is unaffected (a rotation of a unit
  vector is not a phase-space map). The flip side is that **no existing gate constrains
  it either**, so every check on this axis is new and the milestones' whole weight is in
  their gates — the same position M1 was in, for the same reason.
- **`xtrack`'s own default is "no anomalous moment".** `xt.Particles` leaves
  `anomalous_magnetic_moment` at `0`, which silently makes every spin rotation the
  *cyclotron* rotation and the spin tune exactly zero. That is M2's trap by another name
  — a reference whose *default configuration* is not the physics being checked — and it
  is named here in advance rather than after being tripped over.

- **N1 — the spin as a tracked quantity: the Thomas-BMT map, element by element.**
  ✅ **SHIPPED (2026-08-26)** — `accsim.spin`, a precession seam on `Element` shaped like
  the radiation one, `Element.track_with_spin` and `Tracker.track_once_with_spin`. The
  map that rotates a particle's spin as it crosses a magnet, and nothing else. The
  spin obeys `dS/ds = Omega x S` with

      `Omega = -(1/(1+delta)) [ (1 + G gamma) b_perp + (1 + G) b_par ]`      [rad/m]

  where `b = B/(B rho)_0` is the element's existing `normalized_field` (so the whole of
  what a magnet must provide for spin is what it already provides for radiation, and no
  element grows a second field model), `b_par` / `b_perp` are its components along and
  across the direction of motion, `G` is the species' anomalous magnetic moment and
  `gamma` the **particle's** own. A bend additionally rotates the *frame* the spin is
  expressed in, by the design bend angle about `y`.

  Deliberately **not** a new state vector: the 6D `(x, px, y, py, zeta, delta)` state is
  untouched and the spin rides alongside it, because it neither influences it nor is
  part of it.

  Gates. The value gate (`nu_0 = G gamma`) is the **control**, not the gate — it depends
  only on the bends summing to `2 pi` and on the beam energy, so an implementation whose
  transverse coefficient is mis-scaled, or whose quadrupole contribution is missing
  entirely, reproduces it exactly. That blindness is asserted, J1-style, rather than
  hoped against. What discriminates:

  - **`G = 0` locks the spin to the direction of motion.** With no anomalous moment the
    BMT rotation *is* the cyclotron rotation, so a spin started along `p` must stay along
    `p` through any sequence of elements, at any amplitude, on or off momentum, in either
    frame. It is exact, needs no reference, and is broken by a mis-scaled transverse
    coefficient, by a wrong sign, and by a missing or mis-signed bend frame rotation —
    the three things the control cannot see.
  - **A quadrupole at a vertical offset** rotates the spin about `x` by
    `(1 + G gamma) k1 int y ds`, with `int y ds` derived symbolically through the
    quadrupole's own exact map. This is the gate that pins the `(1 + G gamma)` factor
    itself, which the `G = 0` identity cannot (it is `1` there by construction).
  - **The `(1 + G)` parallel term is exercised even with no solenoid in the package.**
    `b_par` is the component of a purely *transverse* field along the direction of
    motion, which is non-zero as soon as the particle has an angle; the term therefore
    enters at `O(px b_x)` rather than being dead code awaiting a longitudinal field.
  - **Reference: element by element, and the predicted gate was the wrong shape.** N1
    expected an `O(L^2)` disagreement everywhere, since `xtrack` does not evaluate an
    analytic field at all — `magnet_estimate_field` back-derives `B` from the
    trajectory's curvature — while accsim samples its own field at the traversal
    mid-point. **Two different recipes, and yet: on a bend, and on a quadrupole with only
    one transverse plane populated, they agree to round-off at every slicing.** The
    reason is derivable rather than lucky. In a single plane `b . i = 0` (a purely
    horizontal field never meets a purely vertical angle), so `Omega` points along one
    fixed Cartesian axis for the whole traversal, every rotation commutes with every
    other, only the **scalar** `int b ds` survives — and both codes' quadratures of that
    scalar are the same number. Populate **both** planes and the axis turns, the
    rotations stop commuting, and the two converge to each other as `1/N^3` (a factor 8
    per doubling, gated as that order). **The gap is non-commutativity, not the field
    model**, and the single-plane exactness is what proves it.

  What it delivered beyond the map itself:

  - **A three-way agreement on the one case with a closed form.** On the design orbit a
    sector bend rotates the spin by exactly `-G gamma theta` about `y` — the BMT rotation
    `-(1 + G gamma) theta` plus the frame's own `+theta` — with no quadrature error,
    because the field is constant. accsim lands on it to nine digits, and so does
    `xtrack` once its bend is given an exact map. That is the number the spin tune is
    built from, derived from Thomas-BMT rather than read off either code.
  - **A bug in accsim that only this axis could see.** `SkewQuadrupole.normalized_field`
    rolled the **opposite way from its own map**, returning the exactly sign-flipped
    field *vector* with the correct magnitude. Nothing could catch it: `normalized_field`
    had exactly one consumer — the radiation kick — and that takes `|b_perp|`, which is
    roll-invariant. Spin is the first quantity in the package that reads a field's
    *direction*, and a rolled `Quadrupole` and a `SkewQuadrupole` promptly agreed on the
    orbit to the last bit while disagreeing on which way a spin turned. The fix is one
    line; **the missing gate is the point**, and it now exists: every straight magnet's
    field must agree with its own momentum kick, `(dpx, dpy)/L -> (-b_y, +b_x)`.
  - **An apparent 1.4e-5 disagreement with xtrack that turned out not to be about spin
    at all.** `xt.Bend`'s *default* integration is a fourth-order splitting whose
    one-kick design orbit is not quite the axis; the spin then correctly follows that
    slightly wrong momentum. Four measurements say so rather than one argument: the
    residual is `O(theta^5)` (×32 per doubling of the bend angle), `O(N^-4)` in the
    number of kicks (÷16 per doubling — the splitting's own order), **independent** of
    both the element length and the beam energy, and equal to `theta` times the *orbit*
    residual to three digits. M2's lesson applied unchanged: localise before deriving.
  - **A genuine defect in `xtrack` 0.106.4, asserted with its exponents.**
    `direction_of_motion` (`track_magnet_radiation.h:22`) computes
    `sqrt(1 - ix*ix + iy*iy)` — a `+` where a `-` belongs — so the vector it returns is
    not a unit vector, and it is used unnormalised for the spin precession *and* for
    `compute_b_perp_mod`, which is what B2's radiation kick integrates. The order follows
    from the mechanism: the error multiplies `b . i`, which for a bend is `b_y i_y` and
    already carries one power of `py`. One from the projection, two from the botched
    normalisation ⇒ the spin disagreement is **third order in `py`** (measured ratio 8.00
    per doubling) and **exactly zero in `px`** (`3e-16` at `px = 4e-3`), with the orbit
    untouched at `1e-16` throughout. Gated as both exponents rather than avoided by
    tracking at `py = 0`.
  - **A third silent switch on the reference side**, beyond the two named when the axis
    opened: `xtrack` compiles spin into its kernel only if `line.configure_spin(...)` was
    called, and without it `track()` returns the spin **bit-for-bit unchanged** — through
    a magnet that certainly precesses it, with no error raised. A comparison written
    without it measures nothing and reads as "accsim invented a precession xtrack does
    not have". Asserted, so it cannot be un-set silently.

  Scope, stated rather than discovered:

  - **Thin elements do not precess**, and unlike radiation this is an approximation
    rather than a limit. A thin magnet's radiated energy really does go to zero with its
    length (`U ~ kappa^2 L`), but its integrated field does not, so a thin quadrupole's
    true spin rotation is finite and dropping it is a real omission. It is dropped
    anyway, because **`xtrack`'s thin `Multipole` does not rotate spin either** — spin
    lives only in its `track_magnet` family — so there would be no arbiter, which is
    L5's reason. The cost is precise and is recorded: a thin-lens ring has no vertical
    spin dynamics at all, so every gate on this axis is built from **thick** magnets.
  - **`xtrack` 0.106.4's `direction_of_motion` has a sign typo** —
    `sqrt(1 - ix*ix + iy*iy)`, a `+` where a `-` belongs
    (`beam_elements/elements_src/track_magnet_radiation.h:22`). The vector it returns is
    therefore not a unit vector, wrong at `O(py^2)`, and it feeds **both** the spin
    precession and `compute_b_perp_mod` (hence B2's radiation kick). accsim writes it
    correctly; the reference gate asserts the disagreement **and its order in `py`**
    rather than avoiding it by setting `py = 0`, which is M2's "both models separately
    confirmed" standard rather than a reconciliation.

  Gates: `tests/analytic/test_spin.py` (38), `tests/reference/test_spin_xtrack.py` (18).
  The full analytic suite is **1143 passed**, and the arithmetic is the claim: it was
  1144 before N1, and the one difference is a parametrisation *removed* from N1's own
  file (the design-orbit case, where the Dirac residual is identically zero at every
  slicing and the order-gate would have passed vacuously). **Nothing on axes A-M moved**,
  which is the structural claim the axis opened with.

- **N2 — the closed spin solution and the spin tune.**
  ✅ **SHIPPED (2026-08-26)** — `accsim.spin.closed_spin_solution`, `spin_one_turn_matrix`,
  `propagate_spin_solution`, `spin_tune`, `spin_axis_and_tune`, `ClosedSpinSolution`,
  `SpinSolutionError`. The periodic direction `n_0(s)` a spin must lie along to come back
  to itself after a turn, and the rate `nu_0` it precesses about it — the spin analogue of
  the closed orbit and the tune, and reached the same way (I1's fixed point, but on a
  rotation rather than an affine map).

  One simplification makes it cheaper than I1: the one-turn spin map is a **rotation**, so
  it is linear in the spin, and carrying the three Cartesian basis vectors around once
  gives the whole `3x3` **exactly** — no Newton iteration, no differencing step. `n_0` is
  then the eigenvector of eigenvalue `1` and `nu_0` the rotation angle about it.

  **The milestone's stated expectation about the gate was wrong, and the correction is the
  first thing in it.** N2 was written expecting the discriminating condition to be the
  first-order resonance `nu_0 = k +- Q_y`. It is not, for this object. `n_0` lives on the
  **closed orbit**, so the perturbation it sees is one-turn periodic and has only
  **integer** harmonics; its resonant denominator is `1 - e^{-2 pi i nu_0}`, which vanishes
  only at integer `nu_0`. That is the *imperfection* resonance, and it is what N2 gates.
  `k +- Q_y` is the *intrinsic* resonance and a statement about a different object — the
  invariant spin field of a particle with vertical betatron **amplitude**, whose
  denominator is `lambda_i I - A` with `lambda_i = e^{2 pi i Q_y}` an orbital eigenvalue.
  It needs the spin-orbit coupling matrix, which is exactly what `xtrack`'s
  `spin_n_matrix` / `spin_eigenvectors` / `spin_dn_ddelta_*` carry — **none of which appear
  in N2's own arbiter list**, while N3's depolarization term is built from precisely them.
  Moved to N3 rather than quietly dropped. The milestone's *epistemics* survive intact: an
  integer-indexed family of locations, predicted separably from any coefficient.

  **The degeneracy the milestone predicted did recur, and is worse than expected.** On a
  flat, unsteered ring `n_0 = (0, 1, 0)` **bit for bit** — for any lattice, any energy, any
  quadrupole strength, a quadrupole field multiplied by seven, *and a sign-flipped
  precession vector*. Every field a spin meets is vertical, so `y` is a fixed point of each
  factor separately and nothing about the coefficients survives. This is N1's "the spin
  tune is a control" arriving one milestone later in a second quantity, and M3's degeneracy
  in a third guise, asserted rather than hoped against.

  **The ring that breaks it, and the two facts that force its shape.** A closed vertical
  bump (three correctors, solved from the elements' own vertical transfer matrices) holding
  exactly **one thick quadrupole**, inside a **bend-free** straight, with a thin-lens FODO
  arc whose bends sum to `2 pi`:

  - **a bend traversed at a vertical angle precesses the spin about `z`**, at
    `Omega_z = h i_y i_z G (gamma - 1)` — what survives when a vertical angle gives a
    horizontal-field magnet a component along the motion, i.e. the difference between the
    `(1 + G gamma)` perpendicular coefficient and the `(1 + G)` parallel one. It is
    **first order in `py`** and comparable in size to the quadrupole kick the gate is
    about, so vertical orbit leaking into the arc is a second, distributed, uncontrolled
    driving term — and the closed form would still *fit*, with a wrong coefficient. Found
    by measurement before the ring was designed, not after the gate misbehaved;
  - **thin elements do not precess** (N1's stated omission), so the arc's focusing is thin
    and contributes exactly nothing. The one thick quadrupole is then the entire
    perturbation. N1's cost has become N2's instrument.

  The ring reduces to one localized rotation `chi` about `x` composed with a uniform
  `-2 pi nu_0` about `y`, and the sympy-derived closed form is

      `n_0 = ( -(chi/2) cot(pi nu_0),  1,  -chi/2 )`   to first order in `chi`,
      `nu_0 -> nu_0 + chi^2 cot(pi nu_0) / (8 pi)`     — unmoved at first order.

  **Its two transverse components are two different gates, and that is the finding.**

  - the **`z` component is `-chi/2` with no resonance denominator at all**, so it measures
    the kick itself — and through it the `(1 + G gamma)` factor, now pinned inside a *ring*
    where N1 pinned it in a single element. Its residual is the midpoint rule's quadrature
    of `int y ds` and converges at second order in the slice length (ratio 4.00 per
    halving, `n = 2..64`);
  - the **`x` component carries `cot(pi nu_0)`**, which diverges at every integer spin tune
    and nowhere else. `nu_0 = G gamma`, so the resonance is crossed by *scanning the beam
    energy*: a 111-point sweep from 4.80 to 5.35 GeV finds its two tilt peaks at
    `G gamma = 11` and `12` and nowhere else, while the same sweep's `z` component, divided
    by `chi`, stays at `-1/2` to four digits straight through both crossings;
  - their **ratio** `n_0.x / n_0.z = cot(pi nu_0)` drops `chi` entirely — the *direction*
    the solution leans in measures the spin tune with nothing about the imperfection in it.

  And `nu_0` itself is unmoved at first order in the steering, gated as both the exponent
  and the derived `chi^2 cot(pi nu_0)/(8 pi)` coefficient: this axis's second instance of
  "the number everybody quotes is the one that cannot see the perturbation."

  Three more things it turned up:

  - **`n_0` needs the *tracked* closed orbit, and the default is the expensive one for that
    reason.** The spin is rotated by what `track()` does, so a spin carried around the
    *linear* closed orbit is carried around a trajectory that does not close, and its
    "one-turn" rotation is a rotation between two different points. Axis L's exact maps
    depart from their own Jacobians at `O(x_co^3)` — asserted as that exponent — which on
    the gate ring is a one-turn orbit residual of `1e-8` against the tracked orbit's
    `1e-18`.
  - **The same third order means the bump does not close exactly**, since its correctors
    are solved from matrices. The leak is `1.6e-9` at a 2 mm bump, cubic in the amplitude,
    and the arc's whole spin driving that follows is **bounded** against `chi` at `5e-5` at
    the largest amplitude any gate uses — below every tolerance any of them assert. Bounded
    in the suite, not assumed in prose.
  - **A fourth silent switch on the reference side, and it is M2's drift model.** N1 named
    three (`configure_spin`, `anomalous_magnetic_moment`, the bend `model`). xtrack's
    *default* `xt.Drift` is the **paraxial** one, and a paraxial ring closes this
    matrix-solved bump *exactly* where an exact-drift ring does not — so with the default
    the two codes' closed orbits differ by precisely accsim's own leak, and every spin
    comparison inherits it. Gated in its own right: the gap equals the leak the analytic
    suite measures from a completely separate line, to two digits. M2's lesson unchanged —
    localise before deriving.

  **Reference.** With `xt.Drift(model="exact")` set, and the orbit compared element by
  element *before* any spin component is looked at (N1's finding 2 in a ring instead of a
  magnet), `n_0(s)` matches `twiss(spin=True)`'s `spin_x/y/z` element by element and
  `nu_0` matches `polarization_analysis`'s `spin_tune_fractional` (folded to `[0, 0.5]` the
  way xtrack folds it — the sign and half-turn are not there to unfold). The tolerance is
  `1e-8`/`1e-9` rather than round-off, and **the residual is xtrack's**: accsim's one-turn
  matrix is exact, while xtrack finite-differences it at `ds = 1e-5` and finds `n_0` with a
  two-knob optimiser to `1e-12`.

  Gates: `tests/analytic/test_spin_solution.py` (26), `tests/reference/test_spin_solution_xtrack.py` (5).
  The full analytic suite is **1169 passed**, against 1143 after N1 — the whole difference
  is this milestone's own file, so nothing on axes A–M or in N1 moved.

- **N3 — Sokolov-Ternov: the polarization the radiation builds up.** ✅ **SHIPPED
  (2026-08-26)** — `polarization_integrals`, `sokolov_ternov_polarization`,
  `polarization_buildup_time`, `PolarizationIntegrals`, in `accsim.radiation` rather
  than `accsim.spin`: it is one more integral over the same curvature `I1..I5` average.
  The spin-flip channel of axis B's synchrotron radiation — a bending electron
  occasionally flips its own spin, the two flip directions are not equally likely, and a
  stored beam polarizes on its own to `8/(5 sqrt3) = 92.376%` with a time constant
  running from a second in a small strong-bending ring to hours at LEP. Two closed-orbit
  averages carry all of it (Chao; `xtrack`'s `spin_alpha_plus_co` / `spin_alpha_minus_co`):
  `alpha_plus = (1/C) ∮ kappa^3 (1 - (2/9)(n_0·v)^2) ds` sets the rate,
  `alpha_minus = (1/C) ∮ kappa^3 (n_0·b) ds` sets the direction.

  - **The milestone's own headline number is a control, and the gates say so.** The
    roadmap predicted `P_inf = 8/(5 sqrt3)` would be "blind by construction", and it is
    worse than predicted: on a flat ring `n_0` is parallel to the field everywhere the
    ring bends, so the *ratio* is `-1` before either integral is evaluated. It returns
    the same sixteen digits — `-0.9237604307034013` — across six rings differing in
    focusing, cell count, size, energy and slice count, which is asserted rather than
    remarked on. Same family as J1's blind structural gates, B5's three quiet arbiters,
    and N1's spin tune. **What replaced it as the gate** is N2's vertical-bump ring,
    where `n_0` tilts by `t` and the two integrals stop being each other's negative:
    their *sum* is `t^2 (1/2 - (2/9)<cos^2>)` — second order, opposite signs, different
    coefficients — so one assertion pins **both** weights and cannot pass on a
    normalization coincidence, which `alpha_plus * C == I3` on its own would. Both
    one-legged alternatives are asserted excluded. `<cos^2>` is integrated in sympy: the
    remembered `1/2` is 0.6% wrong here, the correction falling only as `1/(G gamma)`.

  - **`n_0`'s horizontal part counter-rotates against the bend**, at `-G gamma` per unit
    bend angle. Taking that sign the other way leaves the arc average 1.5% out — which
    reads exactly like a quadrature error, and was separated from one only by refining
    the step and watching the residual **refuse to fall**. M2's "localise before
    deriving" on a third axis.

  - **The quadrature has to resolve the spin phase, not the optics.** `G gamma theta` is
    4.4 radians across one gate-ring bend where the dispersion `radiation_integrals`
    sub-steps through `theta = 0.39`, so this integral uses **Simpson** where that one
    uses the trapezoid: fourth order, at the round-off floor by the shared default of 64
    slices, where the trapezoid is still 1.5% short. Convergence has to be measured on
    the `(n_0·v)^2` **term**, which is one part in `10^8` of `alpha_plus` — a convergence
    test on `alpha_plus` reports machine precision at every slice count and sees nothing.

  - **The "no bending" refusal is nearly unreachable, and that is a finding.** A lattice
    of drifts and on-axis quadrupoles never reaches it: nothing precesses, so N2's
    `SpinSolutionError` fires first — both integrals are weighted by an `n_0` that does
    not exist. Exactly one construction separates the two conditions, a **quadrupole
    traversed off-axis**: real field on the orbit, so `n_0` is unique; not a dipole, so
    nothing in scope radiates.

  - **The coefficient is the discriminating quantity, exactly as predicted, and almost
    nothing sees it.** `P_inf` provably cannot — the constant cancels out of a ratio.
    `gamma^5` and `rho^3` scaling catch a wrong *power* and are exact for a rate ten
    times too fast. The analytic suite bounds the eV-to-SI bridge
    (`hbar/m_0 = (hbar c) c/(mc^2)`, assembled from the package's own `HBAR_C_EV_M` and
    checked against `scipy.constants`, which never passes through eV) and anchors the
    machine scale on **LEP**: a bare ring with LEP's radius and circumference at
    45.6 GeV gives **5.65 hours** against a published ~5.5. A wrong *factor* surviving
    all of that is caught only by xtrack — behind the skippable `reference` marker, so a
    green analytic suite is weaker evidence here than anywhere else on this axis.

  - **A fifth silent switch on the reference side, and it is the charge.** `xt.Particles`
    defaults **`q0 = +1`**, so a line built with an electron's `mass0` and no `q0` is a
    positively charged particle. Everything axis N compared before N3 is *blind* to it: a
    lattice specified by normalized strengths bends the same way whatever the charge, and
    the BMT rotation reads the field through the same normalization, so the orbit, `n_0`,
    the spin tune and the one-turn matrix are bit-for-bit unchanged — which is exactly why
    N1's and N2's reference files agreed without ever setting it. The polarization
    *direction* is the first quantity on this axis that asks what the **physical** field is,
    and charge is what turns a curvature into a field. Left on the default, xtrack
    cheerfully reports an electron beam polarizing *along* its guide field; and because
    accsim would flip with it if it made the same mistake, the error never surfaces as a
    disagreement. Pinned as a test that builds the ring both ways rather than fixed quietly
    in the fixture: `alpha_plus` unchanged, `alpha_minus` and `P_inf` exactly negated,
    orbit and spin tune identical.

  - **The arbiter cannot see the flat ring at all.** xtrack's polarization analysis inverts
    `lambda_i I - A` per orbital eigenvector; `method="4d"` leaves an orbital eigenvalue at
    exactly `1`, and a flat ring's spin matrix is a rotation about `y`, so `I - A` has a
    zero row. Its own central-differenced `A` makes that *exactly* singular — a `y`
    component returned untouched gives `(ds-(-ds))/(2ds) = 1` exactly — so `np.linalg.inv`
    raises rather than returning something large. Every N3 cross-check therefore runs on the
    tilted ring, including the ones the tilt is irrelevant to. The axis's own degeneracy,
    arriving one last time, in the arbiter.

  - **What the comparison says:** both integrals agree in magnitude to `4e-15` — tighter
    than N2's finite-differenced `n_0` would suggest, because both are dominated by the
    `kappa^3` geometry the two codes share exactly — and the buildup time agrees with
    `spin_t_pol_component_s`, which is the milestone's only real check on the coefficient.

  - **Scope, stated and measured:** only dipoles radiate, matching `radiation_integrals`,
    because `alpha_plus * C == I3` is a gate the two accsim routes must agree on. xtrack
    also counts the bump's offset quadrupole; on the gate ring that is `3e-12` of
    `alpha_plus` — negligible there, but growing as the **cube** of the orbit offset where
    the tilt term grows as its square.

  Gates: `tests/analytic/test_polarization.py` (21),
  `tests/reference/test_polarization_xtrack.py` (6). The full analytic suite is
  **1190 passed**, against 1169 after N2 — the whole difference is this milestone's own
  file, so nothing on axes A–M or in N1/N2 moved. That total also answers the only way
  this milestone could have reached beyond itself: it added `coords` and
  `elements.element` imports to `radiation.py`, which already imports `twiss`, and an
  import cycle there would have surfaced in whichever test imported first rather than in
  N3's own.

- **N4 — the invariant spin field, the intrinsic resonance, and the depolarization it
  drives.** ✅ **SHIPPED (2026-08-26)** — `spin_orbit_coupling`,
  `propagate_spin_orbit_coupling`, `SpinOrbitCoupling`, `SpinResonanceError` in
  `accsim.spin`; `depolarization_integrals`, `derbenev_kondratenko_polarization`,
  `polarization_time`, `DepolarizationIntegrals` in `accsim.radiation`. What N3
  deliberately did not build: the `(11/18) ∮ kappa^3 |dn/ddelta|^2` term that fights the
  buildup, and with it the whole invariant spin field `n(x) = n_0 + N x`. The off-momentum
  closed spin solution came along the way — `closed_spin_solution(lattice, delta=...)`,
  because it is what checks the field without using the field's own machinery.

  - **The whole matrix at once, as a Sylvester equation, and that is the design decision
    the milestone turns on.** `n_0 + N x` must map to itself under one turn, so
    `A N - N R = -D` — eighteen linear equations in `N`, solved with
    `scipy.linalg.solve_sylvester` rather than mode by mode as `xtrack` does. Two
    reductions, both forced: `n` is a unit vector, so `n_0 . N = 0` *exactly* and the solve
    happens in the plane perpendicular to `n_0`; the row that drops out is the consistency
    condition `n_0 . D = 0`, which holds to the `1e-10` differencing accuracy for the same
    reason. `A` is N2's exact rotation; `R` and `D` are central differences of **one
    shared** tracked turn, so they cannot disagree about which orbit they were taken on.

  - **That reduction is why accsim can do a flat ring, and it turns N3's observation into
    a mechanism.** N3 recorded that xtrack cannot twiss an exactly flat ring: it inverts
    `lambda_i I - A` per orbital eigenvector and `4d` leaves an eigenvalue at exactly `1`.
    That is not a fact about flat rings — `A n_0 = n_0` by definition, so `I - A` is
    singular for **every** ring. A tilted ring survives only because xtrack's
    finite-differenced `A` misses the zero by round-off. Perpendicular to `n_0` the
    eigenvalue does not exist and there is nothing to invert.

  - **The resonances are the two spectra, and there is nothing else in the equation for
    them to be.** `A`'s reduced eigenvalues are `exp(∓2 pi i nu_0)`, `R`'s are the
    betatron/synchrotron ones plus `1`. A Sylvester equation is solvable exactly when the
    spectra are disjoint, so `N` diverges at `nu_0 = k` (integer — N2's imperfection
    resonance, via the eigenvalue `1`) and at `nu_0 = k ± Q_x, k ± Q_y, k ± Q_s` (the
    intrinsic ones). **The roadmap's long-standing `k ± Q_y` lands here, and the milestone
    gates it as a location**, which is the same shape N2's gate had, shifted by `Q_y`:
    `1/|N E_y|` is linear near the pole and extrapolates to `Q_y` within `2e-6`, a quarter
    of a unit from the nearest integer. The residue `|N E_y| · 2|sin(pi (nu_0 - Q_y))|` is
    constant to 1.5% while `|N E_y|` itself runs over a factor of thirty, and **both**
    alternative denominators — `sin(pi nu_0)` and `sin(pi (nu_0 + Q_y))` — are asserted
    excluded at a factor of twenty. The separation from N2 is asserted too: across that
    scan `n_0`'s tilt is *monotone* (a pole cannot be) and the spin tune tracks what it was
    set to.

  - **A fourth degeneracy, and it is exact.** A flat ring has no vertical orbit, so nothing
    on it ever makes a horizontal field, so every rotation is about `y` — and a `delta`
    perturbation only changes how *fast* a spin turns about the axis it already lies along.
    `dn/ddelta = 0` identically, both new integrals are `0.0`, and `P_eq == P_inf` bit for
    bit. Asserted with `==`, because a milestone whose gate ring were accidentally flat
    would otherwise pass everything by returning N3's answers.

  - **The one gate that does not use the solve, and it is the primary one.** Without RF,
    `delta` is exactly conserved, so `R`'s eigenvalue-`1` direction *is* the dispersion and
    a particle on it sits on the closed orbit of a different momentum. Hence
    `N (D, 0, 1) = d/ddelta [n_0 closed at delta]`, whose right-hand side is reached by
    closing an orbit at `±ddelta` — no `A`, no `D`, no Sylvester solve. accsim satisfies it
    to `5e-9`. It also refuses to be a tautology: `N[:, DELTA]` is the partial derivative at
    *fixed transverse coordinates* (a photon emission is instantaneous and moves `delta`
    alone), a different vector by more than a factor of two, and using one for the other
    would pass every shape gate here.

  - **The same identity then arbitrates the arbiter, which is new on this axis.** The two
    codes' `dn/ddelta` differ by `2e-6` *absolute* while every other column of `N` agrees to
    `1e-8` relative. The identity says whose: accsim `5e-9`, xtrack `1e-4` — four orders
    apart, on a quantity neither code's spin-field machinery computes. The cause is the
    `inv(I - A)` above: entries of order `1e11`, the unphysical `n_0` component subtracted
    afterwards, and `1e11 × 1e-16 ~ 1e-5` of cancellation debris left behind. Because the
    debris is absolute, the agreement is **best nearest the resonance**, which is the
    reverse of the usual arrangement and is why the integrals are compared there.

  - **A silent switch on the reference side that was ours, not xtrack's.** N3's `_build`
    hard-coded `p0c` at 5 GeV. Harmless there; silently fatal here, because N4's only knob
    *is* the beam energy — it compared a resonance-tuned accsim ring against a 5 GeV xtrack
    one, and the two then agreed to nine digits on everything except the quantity the
    milestone is about. `_build` now reads the energy off `lattice.ref`. Five silent
    switches on this axis, and the fifth was in the project's own fixture.

  - **The headline, and it is not a control.** `P_eq` falls from `-0.92` to `-0.02` as the
    spin tune closes to `1e-5` of `Q_y`, while N3's `P_inf` drifts only in its ninth digit
    (its own `1/(G gamma)` energy dependence, asserted at that measured size). The
    depolarization grows as the **inverse square** of the tune distance — gated on the
    order, and *measured close in*, because at `d = 1e-3` a non-resonant background is still
    worth 32% and the fitted exponent comes out `-1.89`. Reporting that as agreement would
    have been the easy mistake; the residue `d^2 × integral` is asserted flat as well.

  - **Costs and scope, stated.** Eight sub-slices per dipole suffice where N3 needed 64:
    `|dn/ddelta|^2` is a squared *modulus* of a vector rotating about `n_0`, and a modulus
    is blind to the rotation. The oscillating `dn/ddelta . b` term is the one
    quadrature-limited quantity here, and it is a hundred times smaller. `N[:, ZETA]` is
    **exactly** zero because nothing reads `zeta` — which is what makes accsim's six-column
    equation and xtrack's five-column one the same object, and stops being true the moment
    an RF cavity enters. N3's quadrature walk was factored into
    `radiation._quadrature_nodes` and is now shared, so the two routes' `alpha_*_co` agree
    bit for bit rather than to a tolerance.

  Gates: `tests/analytic/test_depolarization.py` (20),
  `tests/reference/test_depolarization_xtrack.py` (9). The full analytic suite is
  **1210 passed**, against 1190 after N3 — the whole difference is this milestone's own
  file, so nothing on axes A–M or in N1–N3 moved, including the two files N4 edited
  (`radiation.py`'s shared quadrature and `spin.py`'s new `delta` keyword).

- **N5 — the spin field when the beam is bunched: a closed orbit with a momentum of its
  own, and the sidebands the synchrotron motion opens.** ✅ **SHIPPED (2026-08-26)** —
  `accsim.orbit.closed_orbit_delta`, plus the `delta` keyword `spin_orbit_coupling` used to
  drop. Every ring on N1–N4 had **no RF cavity**, and N4's write-up named the consequence
  itself: `N[:, ZETA]` is exactly zero "because nothing reads `zeta` — which is what makes
  accsim's six-column equation and xtrack's five-column one the same object, and stops
  being true the moment an RF cavity enters." This milestone entered it. Effort **M**.

  - **The finding that made it an implementation and not only a test file, and it was not
    the expected one.** The N4 machinery is already six-column general and an RF cavity is
    thin, so it does not precess — the map needed nothing. What broke was *underneath*:
    `_closed_state` built the spin's carrier state from the **4D** closed orbit with
    `zeta = delta = 0`, and on a bunched ring that is **not a fixed point**. One turn moves
    `zeta` by `-8.3e-7 m` on N2's bump ring — the closed orbit is longer than the design
    circumference — and the cavity turns that into a `delta` kick of `-4.7e-9` per turn.
    The whole of N4's construction (an exact one-turn rotation `A`, a shared bundle for `R`
    and `D`) then describes a trajectory that does not close. The `zeta` slip is there on
    the RF-free rings too and is invisible only because nothing reads it.

  - **What the fixed point actually is, and it is one scalar.** `zeta_co = 0` exactly
    (accsim's cavity subtracts `sin(phi_s)`, so the synchronous particle sits at the zero
    crossing whatever the frequency), and the ring instead locks its revolution period by
    shifting the beam **off momentum** until the path length matches:
    `delta_co = -(Delta C / C) / alpha_c`. Measured `-4.778883e-8` against the closed
    form's `-4.778882e-8` — seven digits, with `alpha_c` from the package's own
    `momentum_compaction` — and xtrack's 6D `twiss` closed orbit confirms it to seven
    digits as well. The solve is a secant on the tracked `zeta` slip, deliberately *not*
    the closed form, so the two stay independent. **No 6D closed-orbit solver was built**
    — see the deferral below.

  - **It is not a rounding correction, and the slope inside it is not `G gamma`.**
    `delta_co` moves the spin tune by `5.4e-7`, five percent of the distance at the closest
    point of the resonance scan, and it biases the *whole* scan the same way — so the pole
    would still have looked like a clean straight line, just one aimed slightly off `Q_s`.
    The obvious coefficient is wrong too: `d nu_0/d delta` is **`0.7003 G gamma`**, 43%
    below the naive guess, and it is `0.7003` with the vertical bump on *or off* — a
    property of the arc (dispersion through the quadrupoles, path length through the
    dipoles), not of the distortion. The gate measures it rather than quoting `G gamma`.

  - **The two-sided gate is the bump, not the frequency, and the frequency guess was
    wrong.** Changing the RF frequency changes the bucket width and nothing else: accsim's
    cavity phase is `phi_s - k zeta` with no turn counter, so the RF is locked to the
    *reference* revolution and both `zeta_co` and `delta_co` are frequency-independent.
    What turns the effect off is the orbit distortion that lengthens the ring — bump at
    zero, `delta_co == 0.0` exactly, asserted with `==`. It scales as the bump amplitude
    **squared** (fitted exponent within `0.02` of 2), because a path length is even in the
    orbit angle.

  - **The headline held: the imperfection resonance acquires synchrotron sidebands.** With
    RF the orbital spectrum is `exp(+-2 pi i Q_x)`, `exp(+-2 pi i Q_y)`,
    `exp(+-2 pi i Q_s)` — the doubled eigenvalue `1` is gone — and the Sylvester equation's
    poles move to `nu_0 = k +- Q_s` as well. On N2's bump ring with `Q_s = 0.0505`:
    `1/|N E_s|` extrapolates to `Q_s` within `1e-6`, and across three decades `|N E_s|`
    runs over a factor of `1000` while the residue `|N E_s| . 2|sin(pi (nu_0 - Q_s))|` is
    constant to `0.2%`; all four alternative denominators are excluded by a factor of `20`.
    The lower sideband `k - Q_s` is a separate pole and is gated separately. At the far end
    (`1e-2`) a non-resonant background is still worth `19%` — named, and asserted at that
    size, rather than swallowed by a tolerance.

  - **The energy knob is no longer clean, and mode identification needed a new rule.**
    N4's scan rested on the beam energy moving `nu_0` and nothing else. With a cavity
    `Q_s^2 ~ 1/E` moves the *target*, and `Q_x` picks up a `+4.1e-3` synchro-betatron shift
    through `R56 . R65`, so the scan energy is solved **self-consistently**; `Q_y` is the
    one tune that does not move, and all three are asserted. Separately, N4's "which plane
    does the eigenvector live in" rule fails with three modes — on a dispersive ring the
    *horizontal* eigenvector's largest component is `zeta` — and the first version of this
    scan found no pole at all because of it. The rule that works: the synchrotron mode is
    the one with the largest `|delta|`.

  - **The primary gate is now tracking, and it is sharper than the one it replaces.** N4's
    identity `N (D, 0, 1) = d/ddelta [n_0 closed at delta]` does not exist here. What
    replaces it is the definition: launch at `x` with spin `n_0 + N x`, track, and require
    the spin to still be `n_0 + N x(turn)` forty turns later. Because that is a
    *first-order* statement, the **relative** residual falls linearly with the amplitude
    for a correct `N` (`5.6e-4, 5.6e-5, 5.6e-6`) and sits at a constant `f` for one wrong
    by a fraction `f`. It reads a wrong matrix off directly rather than through a
    tolerance. A second, weaker anchor is continuity onto the RF-free field: the new `zeta`
    column and the shift in `dn/ddelta` both vanish as `Q_s^2`, gated as a fitted exponent
    (`2.09, 2.03, 2.01`) because a linear-in-`Q_s` correction rides on the quadratic law.

  - **The pre-commitment was REFUTED, and the milestone is bigger for it.** Written down
    before measuring: with no eigenvalue `1` left, xtrack's momentum column should come
    into line at the `1e-8` its betatron columns reach. It does not. N4's `2e-6` gap is
    unchanged (it is reproduced here at `1.8e-6` on N4's own ring), and a **new**
    disagreement appears that is zero without RF, grows as `Q_s^2`, and reaches **14%** on
    the gate ring — the `zeta` and `delta` columns differing by a constant factor `1.1434`,
    the horizontal ones by `1.14` through the dispersion, the vertical ones not at all.

  - **Whose gap it is, decided outside both codes' spin-field machinery.** The invariance
    test above, run in **xtrack's own tracker**: xtrack's own matrix sits at `3.56%` at
    every amplitude — a first-order error — while accsim's falls with the amplitude.
    accsim's tracker returns the same verdict, so there is no configuration in which the
    reference's matrix is the invariant one. And the gap is **downstream of everything the
    two codes agree on**: differencing xtrack's own map gives `D` matching accsim's to
    `1.6e-9` and `R` to `1.2e-10`; the mode-by-mode construction transcribed from xtrack's
    own source reproduces accsim's Sylvester solve to `7.6e-11` (the two formulations are
    one equation, confirmed numerically rather than argued); and feeding **xtrack's** `D`
    and `R` through it returns *accsim's* matrix to `1.0e-7`, three orders inside the `14%`
    at issue. The
    error enters somewhere after that, in the stage where xtrack rescales its
    eigenvectors, tracks them at finite amplitude and reassembles
    `NN = EE_spin @ inv(EE_orb)` — *which* of those steps is not determined, and is not
    claimed. One mechanism is **excluded** by data
    already taken — cancellation would grow as the resonance is approached, and the
    discrepancy is flat in the tune distance — and no other is claimed.

  - **A sixth silent switch on the reference side, and the first that is a documented
    convention rather than a default.** xtrack's RF kernel uses
    `q = fabs(q0) * charge_ratio` (`track_rf.h`) — the **absolute** charge — while accsim's
    `RFCavity` multiplies by the signed `ref.charge`. For an electron the two cavities are
    exact negatives, so the correspondence is `phase = phi_s + pi`, not `phase = phi_s`.
    With the naive mapping the xtrack line is longitudinally **unstable** (eigenvalues
    `1.373` and `0.728`) and its 6D `twiss` dies inside the normal form with `Invalid n3` —
    loudly, which is the only reason this one did not become a quiet wrong number.
    `rfcavity.py` claimed the conventions simply matched; it now states the caveat, as does
    `docs/CONVENTIONS.md`, which also gains the correction that **which** stationary phase
    is stable depends on the sign of the charge (every ring on axis N is an electron ring
    above transition at `phi_s = 0`, the opposite of the proton rule the file stated).

  - **The same degeneracy, for the third time on this axis.** N3 found xtrack's `twiss`
    unable to do an exactly flat ring; N4 explained it as `inv(I - A)` being singular for
    every ring; and a genuine 6D Newton solve is singular on an *RF-free* ring for the
    third instance of it — `zeta` and `delta` are both eigenvalue-`1` directions of
    `I - J`. That is why `closed_orbit_delta` returns `0.0` on an unbunched ring by an
    explicit guard rather than by solving, and it is what keeps N1–N4 exactly where they
    were.

  - **Deferred, named, with its gate — and DELIVERED as I4 (2026-08-27).** A genuine
    **6D closed orbit** — the fixed point in all six coordinates, where `zeta_co != 0`
    because the cavity has to make up the turn's energy loss. Nothing in this milestone
    exercises it (`phi_s = 0`, and the polarization route integrates radiation analytically
    rather than tracking it), so building it here would have been a feature no gate touches.
    Its gate already existed in closed form: `zeta_co` is where
    `q V [sin(phi_s - k zeta) - sin(phi_s)] = U`, with xtrack's 6D `twiss` closed orbit as
    the arbiter — and I4 was written on exactly that. **One half of this paragraph was
    wrong**: it named "a ring with `phi_s != 0` **or** with radiation tracked" as needing
    the 6D solve. `phi_s != 0` does not, in this package — the cavity kick vanishes at
    `zeta = 0` for every `phi_s`, and the ramping reference lives inside `accelerate`, which
    never touches the tracking path. Tracked radiation is the only thing in accsim that
    moves `zeta_co`. I4 asserts that with `==` at three phases and corrects both documents.

  - **Scope, stated.** The Derbenev-Kondratenko `11/18` stays anchored where **N4**
    anchored it — on an unbunched ring, where the two codes' fields agree. N5's
    polarization comparison inherits the field disagreement (`-0.00569` against `-0.00747`)
    and is recorded as such rather than used as a physics check. The sideband pole here is
    a *first-order* one; the higher-order sideband hierarchy of the literature is not in
    scope.

  Gates: `tests/analytic/test_spin_sidebands.py` (19),
  `tests/reference/test_spin_sidebands_xtrack.py` (10). The full analytic suite is
  **1229 passed**, against 1210 after N4 — the whole difference is this milestone's own
  file, so nothing on axes A–M or in N1–N4 moved, including the three files N5 edited
  (`orbit.py`'s new solve, `spin.py`'s `delta = None` default, and `rfcavity.py`'s
  corrected docstring).

### O. Normalised coordinates — the machine as a rotation (core accelerator)

Every optics quantity on axes A-N describes the machine in **laboratory** coordinates:
`beta` and `alpha` per plane, Edwards-Teng's mixing per mode, dispersion per momentum.
Axis O adds the change of variables those quantities are really parameters *of*: the
symplectic matrix `W` for which the one-turn map is a **plain rotation**,

    M = W R W^-1,   R = diag(Rot(2 pi Q1), Rot(2 pi Q2), Rot(2 pi Q3)),

so a particle's position becomes a point on a circle whose radius does not change and
whose angle advances by the tune every turn. Opened 2026-08-27 as a new axis rather than
an extension because it is not a new element, a new effect, or a new derivative — it is
the whole 6x6 map re-expressed, and every existing optics function turns out to be one
entry of it.

The direction was chosen on the project's usual filter, applied mechanically: every field
`xtrack`'s `twiss()` returns was listed and diffed against `accsim`'s exports. `W_matrix`
and the normalised coordinates built from it were the largest gap with an arbiter already
wired (`tw.W_matrix`, element by element, plus `tw.get_normalized_coordinates`); the
Mais-Ripken cross-plane betas (`betx2`, `bety1`) and the crab dispersion (`dx_zeta`) are
the same gap seen from two other sides, and are sequenced behind it as O2. O3, written
2026-08-27 after its arbiter had been verified by running it, leaves the representation
alone: it is the *second-order* normal form, and the first entry on the axis that changes
a number rather than re-expressing one.

- **O1 — the normalising matrix `W`, the actions it makes invariant, and the
  order at which the 6D optics leaves the 4D optics.** ✅ **SHIPPED (2026-08-27)**.
  Effort **M**. `accsim.twiss` gains
  `normal_form(one_turn, method="6d"|"4d") -> NormalForm(W, W_inv, R, tunes)`, the pair
  `to_normalized`/`from_normalized`, and `actions(state)`, the invariants
  `J_i = (u_i^2 + p_i^2)/2` in normalised coordinates.

  **The parameterisation is the entire content, and two of the obvious gates cannot see
  it.** `M = W R W^-1` does **not** determine `W`: any `W D` with `D` commuting with `R`
  works too, so that identity is blind to both a per-plane phase and a per-plane scale.
  Requiring `W` symplectic pins the scale and leaves the phases — exactly three real
  numbers, one rotation angle per plane, entirely free. So the definition-plus-symplecticity
  pair is a **structural** check of the same family J1 found blind to its kick coefficient,
  and it is labelled as such rather than counted. What pins the three angles is the
  convention chosen here, in the shape xtrack's `linear_normal_form.py` uses: each
  eigenvector is phase-rotated until its own plane's *position* component is real and
  positive, which forces `W[0,1] = W[2,3] = W[4,5] = 0`; columns are then
  `[Re v1, Im v1, Re v2, Im v2, Re v3, Im v3]`, normalised by `Re . S . Im = 1`, with the
  modes ordered by which plane each eigenvector lives in.

  **The primary gate, and it is the reason the convention is a choice rather than a copy.**
  Under exactly that phase convention the 2x2 diagonal blocks of `W` must *be* the
  Courant-Snyder matrix `[[sqrt(beta), 0], [-alpha/sqrt(beta), 1/sqrt(beta)]]` built from
  the `beta`/`alpha` `closed_twiss` has reported since Stage 1 — a quantity derived on a
  completely different route (a matched 2x2 block, not an eigenvector). Measured while
  scoping on a FODO ring: `8.9e-16`, with the off-diagonal blocks *exactly* zero, and the
  mode tunes equal to `tunes()` to `1e-16`. That agreement is independent evidence the
  convention is the Courant-Snyder one; it is not obtainable by rotating until xtrack
  agrees, which is the move this project forbids.

  **The second tie pins the mixing, and it is exact.** On a coupled, dispersion-free ring
  (skew quadrupole, no bends) the 4D `W` must equal `V . diag(B1, B2)` — G2's Edwards-Teng
  decoupling transformation times each mode's Courant-Snyder block. Measured while scoping
  at two coupling strengths: `2.1e-15` and `2.7e-15`, with the residual per-mode rotation
  **zero to `1e-15`**, so the two parameterisations coincide rather than agreeing up to a
  block rotation. The mode tunes match `normal_mode_tunes` to `1e-16` on the same rings.
  Stated where dispersion vanishes on purpose: with bends present the betatron eigenvectors
  acquire longitudinal components and the upper-left 4x4 is no longer `V . diag(B)`.

  **The discriminating content is an exponent, not a tolerance — and it is a statement
  about what "the optics" means once the RF is on.** The 6D normal form does **not**
  reproduce the 4D optics. On the I4 ring its `beta_x` is 7.5% *lower* than
  `closed_twiss`'s, its horizontal tune is `6.5e-3` lower than `tunes()`, and the
  dispersion read off `W` by xtrack's own formula is 24% *higher* than the matched
  `disp_x`. None of that is an error in either: the 4D quantities are the response to a
  momentum held **fixed**, and the 6D ones are the response to a momentum **oscillating at
  the synchrotron tune** — the ring driven off-resonance rather than statically. The
  milestone's claim, pre-committed here: all three departures vanish **quadratically in
  `Q_s`**, so the 6D optics reduces to the 4D optics in the `Q_s -> 0` limit at second
  order and not first. Scoped over a factor 32 in `Q_s` (voltage `9e7` down to `8.8e4`):
  the successive ratios go `4.65, 4.14, 4.04, 4.01, 4.00` for the dispersion,
  `4.87, 4.19, 4.05, 4.01, 4.00` for `beta_x` and `4.79, 4.17, 4.04, 4.01, 4.00` for the
  tune, against the `4` a halved `Q_s` predicts. The gate fits the exponent and requires
  `2.00 +- 0.02`, which reads a wrong normalisation off directly where a tolerance on any
  one voltage would not. (Measured while scoping, so recorded as scoping rather than as a
  pre-commitment; the *pre-commitment* is that the fitted exponent is 2 and that the same
  three quantities are the ones that move.)

  **A refusal, and it is the fifth appearance of one degeneracy.** An RF-free ring has no
  6D normal form at all: `zeta` and `delta` are a repeated unit eigenvalue, the third
  mode's symplectic norm is exactly zero, and xtrack's own routine raises `Invalid n3` on
  it. N3 met this first (its `twiss` could not do a flat ring), N4 explained it
  (`inv(I - A)` singular for every ring), N5 hit it in the spin field and I4 in the 6D
  orbit; here it is a named error rather than a division by zero. Since most rings in the
  suite are RF-free, `method="4d"` is part of **this** milestone rather than deferred: it
  normalises the 4x4 block, which is where the two closed-form ties above live anyway.

  **Comparing against xtrack.** `tw.W_matrix[0]` entry by entry at the ring start.
  Pre-commitment, written before the reference arm is run: the two matrices agree to
  `1e-12` **absolute** on a weakly-coupled, non-radiating ring with the integration
  settings matched the way B2 and I4 match them. Absolute, not relative, because
  `linear_normal_form.py` ends with `W[abs(W) < 1e-14] = 0` and a relative comparison
  against an entry xtrack has zeroed is a spurious failure. Weakly coupled, because
  xtrack's `sort_modes` tie-breaks on `|v[5]|` then `|v[2]|` while accsim's existing
  `normal_mode_tunes` labels by total per-plane weight — rules that agree away from the
  difference resonance and need not agree on it. And one thing checked rather than
  assumed: xtrack's `W` is written in `(x, px, y, py, zeta, pzeta)` where accsim's map is
  in `delta`, but `dpzeta/ddelta = 1` **exactly** at `delta = 0` (from
  `pzeta = (E - E0)/(beta0^2 E0)` and `dE/ddelta = beta0 P0`), so the two linear maps
  coincide and there is no `beta0^2` to chase. This is asserted, not assumed.

  **Scope.** Linear normal form only: the higher-order (Deprit/Dragt-Finn) normal form
  that would give amplitude-dependent tunes *from the map* rather than from J2's tracking
  is not in it. Emittance-scaled normalised coordinates (xtrack's
  `get_normalized_coordinates`, which divides by `sqrt(emitt)`) are a one-line wrapper and
  are deliberately **not** the primary API — `W_inv x` is, so that the matrix under test is
  never entangled with an emittance convention.

  **SHIPPED (2026-08-27).** `accsim.normal_form`, `NormalForm`, `NormalFormError`,
  `to_normalized`, `from_normalized`, `actions` — in `twiss.py`, 259 lines appended, one
  new import (`scipy.optimize.linear_sum_assignment`, for a mode-to-plane
  assignment that is always a permutation).

  **Every claim above held, and the two that were framed as blind really are.**
  `test_definition_and_symplecticity_are_blind_to_the_phase` builds `W . diag(Rot(0.7),
  Rot(-1.3))`, shows it reconstructs the map to `1e-12` and is symplectic to `1e-12`, and
  shows its Courant-Snyder block is wrong — so the file *demonstrates* the blindness
  rather than asserting it in a comment. The primary gate lands at `8.9e-16` with the
  off-diagonal blocks exactly zero; the Edwards-Teng tie at `1e-13` with no residual
  rotation; the exponent fit at `2.00 +- 0.02` for all three quantities over a decade in
  `Q_s`; the RF-free refusal is a `NormalFormError`, with `method="4d"` working on the
  same ring.

  **The reference pre-commitment was half right, and the wrong half is the finding.**
  Promised: the two `W` matrices agree to `1e-12` absolute. The transverse block does, at
  `9e-16` — four orders inside. The longitudinal columns **do not**, flooring at
  `2.6e-11`, and the entire excess is **one entry of xtrack's own one-turn matrix**.
  xtrack obtains `R56` by symmetric finite difference of its *exact* drift map, whose
  `zeta(delta)` is both curved (an `h^2` truncation) and a difference of two nearly-equal
  path lengths (a cancellation round-off going as `1/h`); accsim's `R56` is that same
  function's exact derivative `L/gamma0^2`, derived symbolically in Stage 0. So the entry
  has a **U-shaped** error in the step size with a minimum near `ddelta = 1e-5`. The
  attribution is *gated*, not asserted: the minimum itself is a test (a model
  disagreement has no minimum in a numerical step), and the residual tracks
  `|R56_accsim - R56_xtrack|` one for one at every step, to 15%. Nothing on accsim's side
  is involved — which is why the tolerance is split in two named constants rather than
  loosened to one.

  **Three conventions checked rather than assumed, and all three held.** `dpzeta/ddelta`
  is `1` exactly at `delta = 0`, so xtrack's `pzeta` rows and accsim's `delta` rows are
  the same rows and no `beta0^2` enters. Both codes put each mode's position component on
  the positive real axis (`W[2i, 2i+1] = 0`), asserted on both matrices in the same test.
  And `tw.get_normalized_coordinates` for a real particle reproduces `W^-1 x` once its
  `sqrt(nemitt/(beta0 gamma0))` scaling is multiplied back in — a per-column check that a
  whole-matrix norm can hide.

  **One thing the milestone did not predict and now documents.** The 6D-versus-4D gap is
  *dispersive*: on a bend-free ring the two are equal to `1e-12` in both codes. That is
  what made the entry-by-entry comparison possible at all — a bendy ring would have been
  measuring the bend model (the residual axis L and B2 own) instead of `W`, so the tight
  comparison runs bend-free and the dispersion comparison, which needs bends to exist,
  runs on the one bendy ring whose reference line is already built to B2's settings.

  **Tightened after O2 (2026-08-27).** That dispersion comparison was pre-committed at
  `2e-3` on the same wrong reasoning O2 caught in itself — budgeting for a bend-model
  disagreement that B2 had already removed. It measures `5.7e-11` relative on `dx` and
  `2.9e-11` absolute on `dpx`, so the gate is now `DISPERSION_RTOL`/`DISPERSION_ATOL` at
  `1e-9`, seven orders tighter. The attribution is *gated, not asserted*, in the same test:
  scaling xtrack's `ddelta` by ten scales the disagreement by a hundred over two decades
  and in both components (`1.95e-4` -> `1.95e-6` -> `1.94e-8`), which is a symmetric finite
  difference's `h^2` truncation and nothing else — a model disagreement is flat in the
  reference's step. accsim's `dpx` is `-6.5e-16` at this symmetry point, so the whole `dpx`
  gap is xtrack's own departure from zero. No test was added; the file is still 9.

  Gates: `tests/analytic/test_normal_form.py` (21),
  `tests/reference/test_normal_form_xtrack.py` (9). The full analytic suite is
  **1266 passed**, against 1245 after I4 — the whole difference is this milestone's own
  file, so nothing on axes A-N moved, and the only package files touched are `twiss.py`
  (the new section, appended) and `__init__.py` (the exports).

- **O2 — `W` along the ring, and the two quantities that only exist there.** ✅ **SHIPPED (2026-08-27).**
  Effort **S**. `propagate_normal_form`, following `propagate_twiss`'s shape:
  `W(s) = M(0->s) W(0)`, re-phased to the same convention at each point. It is what makes
  the cross-plane Mais-Ripken betas (`betx2`, `bety1` — how much of mode 2 is carried in
  `x`) and the crab dispersion (`dx_zeta`, the transverse orbit's dependence on arrival
  time) computable at all; both are `twiss()` fields accsim has no analogue for. Gate: the
  re-phasing must reproduce `propagate_twiss`'s `beta`/`alpha` and the phase advance
  `mu(s)` on an uncoupled ring, and `propagate_coupled_twiss` on a coupled one; arbiter
  `tw.betx2`, `tw.bety1`, `tw.dx_zeta`, element by element.

  **Pre-commitment, written before the reference file was run.**

  - **The re-phasing is the whole milestone, and almost nothing can see it.** `betx2`,
    `alfx2`, `bety1` and `dx_zeta` are every one of them invariant under
    `W -> W diag(Rot, Rot, Rot)` — the phase cancels between the two factors of each
    product, and the crab dispersion is a ratio taken *inside* one eigenvector. Only two
    witnesses exist: the convention itself (`W[2p, 2p+1] = 0`) and `mu(s)`. So the `mu`
    gate is made **quantised** rather than tolerance-based: `tunes()` returns the *full*
    integer-plus-fractional tune, and a dropped `np.unwrap` branch is wrong by exactly
    `1`. Rings with an integer part are chosen deliberately for it, and a companion test
    asserts no element advances the phase by more than `pi` so the reason localises.
  - **Crab dispersion is not zero on an ordinary ring, and its two exponents are the
    physics.** The transverse response to a momentum oscillating at `Q_s` is driven
    off-resonance, so it *lags*: `c0 = v3[x]/v3[delta]` acquires an imaginary part first
    order in `Q_s`. `dx_zeta` is that lag times the longitudinal mode's momentum content,
    which is *also* first order, so `dx_zeta` is **second** order. Predicted: lag exponent
    `1.00`, `dx_zeta` exponent `2.00`, both to `+-0.02` over a decade in `Q_s`; and
    `dx_zeta` **exactly zero** on a bend-free ring, since the transverse rows never see
    `delta` there.
  - **Ties.** `mu`/`beta`/`alpha` reproduce `propagate_twiss` at `1e-13` on an uncoupled
    ring; `W(s) = V(s) . diag(B1(s), B2(s))` reproduces G2's `propagate_coupled_twiss` at
    `1e-12` at every point on a coupled one — the tie that gives `betx2`/`bety1`, which
    have no closed form of their own, something independently derived to stand on.
    `betx2` grows as the **square** of the skew strength.
  - **Arbiter tolerances.** Element by element against `twiss()`: `mux`/`muy`/`muzeta` and
    the transverse block of `W(s)` at `1e-12` **absolute** on the bend-free ring (O1's
    constant, now at every point); `betx2`/`bety1`/`alfx2`/`alfy1` at `1e-9` relative on a
    coupled bend-free ring; `dx_zeta`/`dpx_zeta` at `5e-3` relative on a bendy ring —
    looser than the rest and stated as such, because a bendy ring is also comparing the
    two codes' bend models, which is the residual axis L and B2 already own.
  - **One thing expected to be missed, named in advance.** O1 localised a `2.6e-11`
    residual in `W`'s longitudinal columns to xtrack's finite-differenced `R56`. That
    error is *transported* by `W(s) = M(0->s) W(0)`, so it must **grow** along the ring
    rather than stay put. The claim is that it stays confined to the longitudinal columns
    — the transverse block must not degrade at all — and a separate constant records what
    the longitudinal ones actually reach.

  **SHIPPED (2026-08-27).** `accsim.propagate_normal_form`, `closed_normal_form`,
  `NormalFormPoint` — in `twiss.py`, 252 lines appended, plus a refactor that moves O1's
  `NormalForm` onto the same private `_ripken_*` / `_dispersion_from_w` helpers so the
  entrance is not a special case. No new dependency.

  **Every claim about the blindness held, and it is worse here than in O1.** `betx2`,
  `alfx2`, `bety1`, `dx_zeta` and the dynamic dispersion are *all* invariant under
  `W -> W diag(Rot, Rot, Rot)` — the phase cancels between the two factors of each
  product, and the dispersions are ratios taken inside a single eigenvector.
  `test_the_new_quantities_are_blind_to_the_re_phasing` builds a `W(s)` mis-phased by
  `(0.7, -1.3, 2.1)` radians and shows all five come back unchanged to `1e-12`, and that
  it is still symplectic. So the re-phasing — which is the entire operation this milestone
  performs — has exactly two witnesses, and the gates are weighted accordingly.

  **The quantised tune gate did the work it was chosen for.** `mu(C)/2 pi` matches
  `tunes()`'s *full* integer-plus-fractional tune to `1e-12` on two rings picked for
  having an integer part (`1.235`, and the I4 ring's `1.630`/`1.282`) — the four-cell FODO
  used elsewhere in the file reaches only `0.206`, where the gate is vacuous and was
  caught being so on the first run. The companion localiser measures the worst
  per-element phase step at `0.26` rad, an order inside the `pi` the unwrap needs.

  **Crab dispersion is not zero on a ring with no crab cavity, and the two exponents are
  the finding.** The predicted lag exponent `1.00` came in at `1.0000000000011`; the
  predicted `dx_zeta` exponent `2.00` at `2.0011`, over a decade in `Q_s`. The second one
  had to be *corrected before* it was written: the first derivation gave `1`, missing that
  the longitudinal mode's momentum content `|v3[delta]|^2` is itself linear in `Q_s`
  (the ellipse elongates as the cavity weakens). The identity that multiplies the two,
  `dx_zeta = -gamma_3 Im(c0)/sigma_3`, is asserted alongside them so the pair is one
  statement rather than two coincidences. On a bend-free ring `dx_zeta` is exactly `0` in
  both codes.

  **The ties held at the tight end.** `mu`/`beta`/`alpha` reproduce `propagate_twiss` at
  `1e-13` at every point; `W(s) = V(s) . diag(B1(s), B2(s))` reproduces G2's
  `propagate_coupled_twiss` at `1e-12` at every point for `k1sl` in `{0.02, 0.1}`, with no
  residual per-mode rotation — which is what gives `betx2`/`bety1`, quantities with no
  closed form of their own, something independently derived to stand on. A scan of the
  local Edwards-Teng `Delta` around that ring found no sign change, so the mode-labelling
  hazard `propagate_coupled_twiss` documents does not bite here; it was checked rather
  than hoped.

  **`betx2`'s exponent is `2` asymptotically, and the window is stated.** Fitted `2.00` on
  `k1sl` in `[3.1e-4, 2.5e-3]`; the fourth-order term is still worth measuring higher up
  (`1.90` at `0.01`, `1.67` at `0.04`), so fitting where the coupling tests elsewhere sit
  would have needed a tolerance three times as wide. The first run failed at `1.956` for
  exactly that reason and the window moved, not the tolerance.

  **The reference pre-commitment was missed once, in the place O1 already named, and was
  five orders too *loose* four times.**

  - *The miss.* `mux`/`muy` land at `1.9e-16` and `3.3e-16` — four orders inside the
    promised `1e-12` — while `muzeta` floors at `1.8e-11`. Same owner as O1's `W`
    residual: the longitudinal phase advance is read off the columns of `W` that carry
    xtrack's finite-differenced `R56`. Three signatures gate the attribution rather than
    asserting it, and the first is the sharp one: **changing xtrack's momentum step over
    three decades moves `muzeta`'s residual by five orders of magnitude and moves
    `mux`/`muy` by not one bit.** A disagreement between two codes' physics does not care
    what step the reference differentiates with, and does not stop at a plane boundary.
    The residual also traces the same U with the same minimum at `ddelta = 1e-5`, and
    above the minimum its step-to-step ratio matches `R56`'s to `2%`. Two named constants,
    not one loosened one.
  - *The looseness.* `betx2`/`bety1` were promised `1e-9` relative and measure `3.9e-15`;
    `alfx2`/`alfy1` `2.3e-15` absolute; `dx_zeta` was promised `5e-3` relative on a signal
    of `+-0.077` and measures `1.1e-9` **absolute**; `dx` was promised `2e-3` and measures
    `1.4e-8`. The reasoning behind the loose pair was wrong in a specific, recordable way:
    it budgeted for the two codes' bend models disagreeing — the residual axis L and B2
    own — but **B2 had already removed it**, by setting `integrator="uniform"` and one
    multipole kick per element on the reference line for exactly this purpose. A gate at
    `5e-3` on a quantity agreeing at `1e-9` would sleep through any regression worth
    catching, so all four tolerances were tightened to roughly two orders above the
    measurement. O1's entrance-only `dx` comparison carried the same over-loose `2e-3` for
    the same reason, and was tightened to `1e-9` the same day — see O1's *Tightened after
    O2* note.
  - *What was predicted and did happen.* The `R56` residual is **transported** by
    `W(s) = M(0->s) W(0)`, so it grows along the ring — `2.6e-11` at the entrance to
    `1.35e-10` at the end. The gate is not that it stays small but that it stays
    **confined**: the transverse block is four orders cleaner at every point, which a
    transport bug could not manage.

  **One guard that is about the element set, not the algebra.** The 4D propagation is
  unambiguous only because `_transverse_4d(A B) = _transverse_4d(A) _transverse_4d(B)`,
  which needs no element to make the transverse coordinates depend on `zeta` and none to
  make `delta` depend on the transverse ones. Both hold for accsim today and are asserted
  to hold (residual exactly `0.0` on the bendy ring), so a future crab cavity — or a
  radiation map passed through `maps` — fails loudly instead of drifting.

  **One thing that shipped ungated and was caught before the commit.** `gammas` — the
  Ripken matrix read off the *momentum* row — was in the code, in the docs and in the
  blindness test, and had no check of its **formula** anywhere: the blindness test only
  asserts it is *invariant*, and the pre-commitment above listed the betas and alphas and
  not it. Writing row `2p` where row `2p+1` belongs would have made it equal `betas` and
  passed all 34 tests, ruff and both cross-checks. It now has two ties of its own: the
  Stage 1 one on an uncoupled ring (`gamma = (1 + alpha^2)/beta`) and, on a coupled one,
  the symplectic identity that binds all three matrices at once — `B G - A^2 = det^2`
  entry by entry, with each mode's determinants summing to `1` over the planes. Plus
  `tw.gamx1`/`gamx2`/`gamy1`/`gamy2`, which cost nothing on the fixture already built.
  The general lesson: **a pre-commitment is also a list of what will not be gated.**

  Gates: `tests/analytic/test_normal_form_along_ring.py` (26),
  `tests/reference/test_normal_form_along_ring_xtrack.py` (11). The full analytic suite is
  **1292 passed**, against 1266 after O1 — the whole difference is this milestone's own
  file, so nothing on axes A-N or O1 moved; `test_normal_form.py` is still 21/21 despite
  the shared-helper refactor. The reference suite is **240 passed**, against 229.

- **O3 — the second-order normal form, and the detuning `amplitude_detuning` declared
  out of scope.** ✅ **SHIPPED 2026-08-27.** Effort **M** (as estimated).

  O1 and O2 re-expressed the machine without changing a number. O3 is the first entry on
  this axis that changes an *answer*: `sextupole_detuning` (plus `total_detuning` for the
  sum with J2's octupole term) computes the `dQ/dJ` that sextupoles produce at **second**
  order in `k2`, which J2's own docstring named as a hole and said no closed form for was
  claimed anywhere in the package. That paragraph now points here. An octupole detunes at
  first order — one phase average of one potential. A sextupole's potential is odd, its
  first-order average is exactly zero, and the effect exists only where the ring's
  sextupoles act in **pairs**: hence a double sum over all ordered pairs (diagonal
  included — one sextupole detunes on its own), with `β^(3/2)` and `β^(1/2)β_y` on each
  generator and the four denominators `Q_x`, `3Q_x`, `Q_x ± 2Q_y`. Full statement,
  including both invariances, in `docs/CONVENTIONS.md` → *Sextupole amplitude detuning*.

  **The derivation was done, not recalled — and the check that made it safe was not the
  one the candidate expected.** Four conventions are pinned by a test before being used
  (the generator against `ThinSextupole.track` exactly; the resonance-basis bracket
  against the ordinary Poisson bracket; the `+μ` conjugation against the `−μ`; and the
  `+½` in the Lie composition *solved for*, then re-checked on a whole two-sextupole
  turn). Then the machinery is anchored twice before it is aimed at a sextupole: on the
  octupole, where it must return J2's shipped first-order matrix, and — the load-bearing
  one — on **two thin quadrupoles**, where its second-order answer must equal the exact
  expansion of `cos 2πQ = ½Tr(M)` of the real one-turn matrix. Written in unit-modulus
  symbols both sides are rational functions, so that anchor holds as a **symbolic
  identity**, in **both beam orderings** (which is what forces the `|μ_i − μ_j|`). All
  nine coefficients of the shipped formula are likewise verified as exact identities, not
  fitted to a tolerance.

  **Six findings, in descending order of how much they would have cost someone.**

  - **The pre-committed primary gate had the wrong shape, and the truth is stronger.**
    The candidate said the tolerance must be a *power*, not a number: PTC is all-orders,
    the formula is second order, so they must disagree at any fixed strength and the
    residual should fall as `k2²`. It does not fall at all. PTC's `anhx(1,0,0)` **is** the
    quartic coefficient of the normal form — the same object accsim computes — so once the
    kinematic baseline (below) is out, the two agree to round-off: ratio `1.0000000000` on
    all three independent entries at three working points, and `no = 4, 5, 6` return
    bit-identical values. The exponent scan was a hedge against a mismatch that does not
    exist. It is retired here and kept where all orders genuinely enter — the tracked gate.
    The flip side is recorded too: both sides now compute the same object the same way, so
    **PTC is not the milestone's independent leg**. Those are the two-quadrupole
    exact-trace anchor and tracking.
  - **The exact drift detunes with no magnets at all, in both codes.** PTC with
    `exact=true` reports `0.127` on the O3 fixture with every multipole off, against
    `0.542` for the sextupoles — a quarter of the signal. accsim's `Drift` is exact too
    (M2/M3's drift-model note) and does the same under tracking. `sextupole_detuning`
    returns exactly zero there, correctly. Every comparison against a tracked or PTC
    number is therefore a **difference** against the same ring at `k2 = 0`. Found by the
    residual *growing* as the strength shrank — the absolute gap was constant, which is
    the signature of a term with no `k2` in it.
  - **The tracked residual falls as an ODD power of the amplitude.** J2's octupole gate
    watches 16 per halving; this one gives **8**, which a tune cannot do. It is not the
    formula: the prediction is evaluated at the *Courant-Snyder* action of the launch
    point, and with a sextupole present that is not the particle's invariant action — it
    is wrong by a phase-dependent `O(k2 x³)`. Both halves measured: at fixed action the
    tracked detuning varies by ±2.1 % across launch phase at 2 mm and ±1.1 % at 1 mm (the
    spread is *linear* in amplitude, as that explanation requires and nothing else would
    be), and averaging over launch phase restores the expected order exactly — **16.02,
    16.00, 16.00**. Anyone measuring `dQ/dJ` by tracking here must average over phase.
  - **A thick sextupole approaches a thin one LINEARLY, not quadratically.** J2 gates the
    octupole's limit as `L²`; a test written by analogy fails, and for physics rather than
    a bug. The pair kernel carries `cos(|Δμ| − πQ)`, whose `|·|` has a **kink** at zero
    phase separation, so the mean `|Δμ|` between two slices of one body is first order in
    the length. Measured 2.08, 2.04, 2.02. The midpoint slicing itself converges at second
    order (4.05, 4.01, 4.00, 4.00), which is the check that the slice placement is right.
  - **The `3Q_x` term is not a small correction on a short ring.** The received picture is
    that it matters only near `Q_x = n/3`. On this 12 m four-cell fixture it is already
    **3.6×** the `Q_x` part at a generic working point, because the phase advances between
    sextupoles are a large fraction of the turn. The candidate's own text said "a small
    correction at a generic tune"; that was wrong here and the gate now measures the split
    instead of assuming it.
  - **Two fixture degeneracies, one caught in advance and one not.** The `Q_x = Q_y` trap
    the candidate named was real and was avoided (at equal tunes `Q_x ± 2Q_y` collapse onto
    `3Q_x` and `Q_x`, hiding a wrong cross-plane term). The one it did not name: the first
    version of the `β^(3/2)` gate compared positions whose `β_x` differed by **4 %**, where
    `β²` and `β³` are indistinguishable — a gate that would have passed with the wrong
    exponent. It now runs minimum-to-maximum `β_x` (41 % contrast, a factor 2.8 in the
    prediction) and explicitly excludes the neighbouring exponents. The same flaw showed up
    a third time, in the combined-magnet gate: at the strength first chosen the octupole
    term was `10^4` times the sextupole one, so the sum matches PTC whatever the sextupole
    formula says. Redone at comparable strengths, where the cross term very nearly
    *cancels* (`-2.617 + 2.346`) and the sum is a tenth the size of either piece — an
    amplifier rather than a blanket. It still agrees to nine digits, so **at this order
    there is no measurable sextupole-octupole cross term**, which is the prediction the
    unadjusted sum was making. **General lesson: a gate is only as sharp as the contrast or
    the cancellation it is run at, and choosing that is part of writing the gate rather
    than a detail of the fixture.**

  **What is not gated,** per O2's closing lesson that a pre-commitment is also a list of
  what will not be gated: the phase structure of the double sum is only partly observable,
  so two compensating terms could survive the three rings; the coupled denominators
  `Q_x ± 2Q_y` are reached only through the single cross entry `dQ_x/dJ_y` and through
  PTC; nothing here gates detuning *quadratic* in the action, `k2⁴`, the octupole's own
  second-order term, or the design-orbit assumption. `ResonantLatticeError` (a new sibling
  of `UnstableLatticeError`/`CoupledLatticeError`) refuses a lattice sitting on one of its
  own driven lines rather than dividing by zero; the divergence *approaching* one is
  physical and is returned — verified against PTC at `|Q_x - 1/3| = 4.0e-3` (`Q_x =
  0.329334`), where `dQx/dJx` reaches `19.573491` in both codes (ratio `1.0000000000`)
  against `0.542` at the generic point.

  **Sequenced behind it: the resonance driving terms (candidate O4).** Now shipped —
  see the entry below, which also decodes the `gnfu` indexing this paragraph called
  undecoded.

- **O4 — the resonance driving terms, and PTC's `gnfu` indexing decoded.**
  ✅ **SHIPPED 2026-08-27.** Effort **M** (as the candidate estimated).

  `resonance_driving_terms` returns the seven first-order terms a **normal sextupole**
  (`f3000`, `f2100`, `f1020`, `f1011`, `f1002`) and a **skew quadrupole** (`f1010`,
  `f1001`) drive, complex, at the lattice entrance. (O5 has since widened the same function
  to twenty terms, adding the octupole's and the skew sextupole's; these seven did not move
  — see the O5 entry below for why that is structural rather than lucky.) The candidate's own case for the
  milestone was that "the Lie/normal-form machinery now exists and is anchored, so an O4
  would be a read-out of the same calculation rather than a new one", and that is exactly
  what it turned out to be: O3 keeps the action-only part of the normal form and the RDT
  is the coefficient of the generator that removes each *non*-action monomial — the
  intermediate quantity `_homological` already produces and then discards. The O4 test
  file **imports** O3's machinery rather than restating it, so O3's four pinned
  conventions carry over without being re-argued, and all nine shipped coefficients are
  checked as exact symbolic identities against that machinery's own output. Full statement
  in `docs/CONVENTIONS.md` → *Resonance driving terms*.

  **The candidate's two doubts, one confirmed and one dissolved.**

  - *Confirmed.* "xtrack's is a first-order perturbation formula from twiss plus
    strengths, so accsim writing the same expression is closer to a reimplementation than
    a cross-check." Correct, and the reference file says so in its own docstring rather
    than banking the agreement as independence. The independent legs are **G1's shipped
    `|C⁻|`** and **tracking**.
  - *Dissolved.* "PTC's `gnfu` indexing is **still undecoded** … mapping that is a session
    of its own." It took twenty minutes, and the obstacle was never physics. Two things at
    once: `select_ptc_normal, gnfu=` takes **four** indices and `normal_results` carries an
    **`order4`** column the earlier probe did not read, so every key came back truncated;
    and PTC returns **empty rows whose keys were never requested and are not even cubic**
    (`(0,0,2,0)`, `(1,0,0,0)`, `(2,0,0,0)`), interleaved with the real ones. Truncated keys
    plus junk-keyed blanks is precisely the "five entries keyed `(1,0,0)`, `(1,0,1)`,
    `(1,0,2)`, `(2,1,0)`, `(3,0,0)` for three requests" that was recorded as inexplicable.
    With the fourth column read and blanks recognised by **value** rather than key, PTC's
    key is `(j,k,l,m)` — accsim's own indices — and the only difference is a factorial
    weight `j!k!l!m!`, established from the *pattern* (four distinct weights `6, 2, 2, 1`
    on five terms, which one wrong constant cannot fake) rather than from one ratio.

  **Six findings, in descending order of what they would cost someone.**

  - **First order in `k2` is EXACT here — the mirror image of O3.** The natural
    expectation after O3 (all-orders code vs first-order formula must disagree, and the
    gap should grow with strength) is wrong, and structurally so: the bracket of two cubic
    generators is *quartic*, so a sextupole's second-order contribution lands on the
    detuning O3 computes and never returns to these cubic coefficients. PTC agrees at
    `1e-14` at `0.3×`, `1×` and `3×` strength, and `no = 4` / `no = 6` are bit-identical.
    Keep the pair in mind: a sextupole's **detuning** does not exist at first order at all,
    while its **cubic driving terms** are complete at first order.
  - **An RDT is covariant, not invariant, and that is the sharpest gate in the file.** O3's
    detuning does not move when the observation point does; this does. Moving the start
    through `(d_x, d_y)` past a set of sources gives
    `f_new = e^{+i(m_x d_x + m_y d_y)} (f_old + F_crossed)` with `F_crossed` the sum of the
    **plain, undivided** coefficients of just the sources stepped over. Gated to `1e-11`
    with shifts crossing zero, one, two and five sources, which separates the rotation from
    the jump instead of measuring them together; a wrong conjugation or a missing phase
    passes every single-point comparison and fails here. `|f|` varies by more than `1.5×`
    around this ring, so quoting it at one point as "the ring's `f3000`" is wrong.
  - **At a thin source the two codes report opposite sides of the jump.** xtrack's row for
    a source element is **downstream** of its kick; rolling accsim's list so the source
    comes first observes **upstream**. Everywhere else they agree to round-off, so a
    comparison written without this fails at exactly three points out of twenty-two and
    reads like a phase error. It became a *gate* rather than a caveat because the step is
    predicted — crossing one source at zero phase advance adds precisely its plain `F` —
    and that is checked on all five terms against xtrack at `3e-11` on steps of order one.
  - **The basis is the whole content, and no magnitude gate can see it.** `h = û + i p̂`
    versus `û − i p̂` differ by complex conjugation and agree in modulus, so a package could
    ship the mirror image of what everyone else calls `f3000` and pass every `abs()` test
    ever written — O1's "the content is the PHASE" in a new costume. accsim ships the `+i`
    basis (xtrack's and MAD-X's), **measured** three ways: xtrack `1e-10`, PTC `1e-14`, and
    — the one that is not a convention argument at all — tracked sideband phases at `1e-4`
    with the conjugate excluded by a factor of a thousand.
  - **The residual against a coupled reference is CUBIC in the skew strength, not
    quadratic.** `f1001` is linear in `k1sl`, so the obvious guess is a first correction at
    the square. Measured: the *relative* gap falls by four per halving, i.e. the absolute
    gap is cubic — because what perturbs the answer is the optics, whose own shift from
    coupling is itself quadratic. Same exponent, same cause, in both the tracked gate and
    the xtrack comparison.
  - **The gate for a thick skew quadrupole's body map had to be replaced, because the
    obvious one is blind.** A sextupole's linear map is a drift; a skew quadrupole's is
    not — but the walk transports the *coupling-off* map, and deleting a skew
    quadrupole's off-blocks deletes all of its coupling, so whether what remains is a
    drift is a real question. It is not (a focusing term second order in `k1s`), and the
    first gate written for it — the `|C⁻|` tie with the body in place — **cannot tell**,
    because G1 slices such a body through the same decoupled path and would share any
    error. Replaced with a direct comparison against the same sources transported by
    plain drifts, which is two to four orders worse. Same shape as L4's and G1's own
    findings: a plausible-looking slicing path quietly using the wrong body map.
  - **A rolled sextupole was being summed as a normal one at full strength, and the
    guard that appeared to catch it was catching round-off.** A rolled sextupole *is* a
    skew sextupole (exactly, at −30°), so only `k2l cos(3·roll)` of it is normal — a
    type-walking sum reads a *wrong* number, not a missing one. It looked covered,
    because the inherited coupling guard did refuse such a ring; but a sextupole's linear
    map is a drift, so what that guard was firing on was a `1e-18` off-block left by the
    rotation, at an angle that happened not to cancel. An *offset* source, which feeds
    down, was not refused at all. Both now have an explicit check, and the difference
    between "it refuses" and "it refuses for the right reason" is the whole finding.
  - **A fixture guard caught a vacuous gate on its first run.** The helper now asserts that
    every requested source was actually *placed*, and it immediately failed: two of three
    skew quadrupoles in one comparison ring were landing inside quadrupoles and silently
    vanishing, so a three-source gate had been running with one. O3's contrast lesson,
    generalised — checking the fixture is part of writing the gate, not a detail of it.

  **The G1 tie, which is the only leg sharing no algebra with the derivation.**
  `closest_tune_approach` was derived in G1 from the exact eigen-tune split, a completely
  different route. The relation `|f1001| · 4|sin(π(Q_x − Q_y))| = π|C⁻|` was **measured
  before it was asserted** — on three rings with different tunes, positions and strengths
  (one of them four orders weaker), checked to agree with each other *first* — and comes
  out `π` to round-off at every strength, because both sides evaluate on the same
  unperturbed optics and neither approximates the other.

  **What is not gated,** per O2's rule that a pre-commitment is also a list of what will
  not be gated. `f1010` has no *closed-form* leg of its own — G1's `|C⁻|` is the
  **difference** resonance, so it reaches `f1001` only — and it is instead tracked, off
  the `−Q_x` sideband of `h_y` on the same trajectory that gives `f1001`; the two are
  separated by measuring that the ordering `|f1010|/|f1001|` **flips** between working
  points (`0.17` at one, `1.8` at the other), which a repeated number cannot do. Nothing
  here gates second-order RDTs, octupole or skew-sextupole terms, feed-down from a closed
  orbit or a misalignment, or an RDT returned as a table *along* the ring. And the tracked
  legs cover `f3000`, `f1001` and `f1010` — the remaining four (`f2100`, `f1020`, `f1011`,
  `f1002`) are reached by the symbolic identity, the covariance law and the two reference
  codes, all of which share the first-order formula, so a wrong coefficient common to the
  derivation and to both external codes would survive.

  Gates: `tests/analytic/test_resonance_driving_terms.py` (39),
  `tests/reference/test_resonance_driving_terms_xtrack.py` (7),
  `tests/reference/test_resonance_driving_terms_madx.py` (6). The full analytic suite is
  **1364 passed**, against 1325 before this milestone — the whole difference is this
  milestone's own file, so nothing on axes A–N or O1–O3 moved. The reference suite is
  **263 passed**, against 250 before — again the whole difference is this milestone's own
  two files.

- **O5 — the octupole's and the skew sextupole's driving terms.**
  ✅ **SHIPPED 2026-08-31.** Effort **M**.

  `resonance_driving_terms` grows from seven terms to twenty: the eight a **normal
  octupole** drives (`f4000`, `f3100`, `f2020`, `f2011`, `f2002`, `f1120`, `f0031`,
  `f0040`) and the five a **skew sextupole** drives (`f2010`, `f2001`, `f1110`, `f0021`,
  `f0030`), on top of O4's cubic pair. Chosen over the alternatives on the project's usual
  filter — *does an arbiter already exist* — and the filter was applied by **running** the
  two candidate arbiters before committing, not by reading their source: xtrack's routine
  is written generically over multipole order, and MAD-X PTC returns all eight quartic keys
  at `no = 5`. That probe is also what turned up the milestone's biggest scope surprise
  (third finding below). Full statement in `docs/CONVENTIONS.md` → *The octupole's and the
  skew sextupole's driving terms*.

  Like O4 this is a read-out of machinery that already existed rather than a new
  calculation: the O5 test file **imports** O3's Lie machinery, so O3's four pinned
  conventions and O4's fifth carry over without being re-argued, and all thirteen shipped
  coefficients are checked as exact symbolic identities against that machinery's own
  output.

  **Findings, in descending order of what they would cost someone.**

  - **An octupole moves the very line its own driving term sits on, and the tracked gate
    reads leakage if you do not notice.** The two halves of an octupole's generator
    interfere *in the measurement*: the action half is the amplitude detuning, which shifts
    `Q_x` off the lattice's linear tune, and the sideband the non-action half is read from
    moves with it — **three times as far**, since `f4000` sits at `−3 Q_x`. At a 1.5 mm
    launch the tune shift is only `2·10⁻⁴`, but a windowed projection at the *linear* tune
    then returns `4.0` against `193` — **a factor of 48 low**, which is far too large to
    read as a tolerance and far too small to look like a structural bug. (The raw sideband
    amplitude is `6·10⁻⁵` of its true value; the two numbers differ because the primary
    line is mismeasured as well and the read-out is a ratio, so the errors partly cancel
    and conspicuously do not cancel completely.) O4 never met this because a **sextupole
    has no first-order detuning at all**. The O5 tracked gates measure `Q_x` from the
    trajectory, and the trap itself is gated.

  - **Both reference codes disagreed, for two different reasons, and neither was a
    tolerance to loosen.** Both came back *close but not round-off* — the shape a real
    effect makes and a convention error does not.

    *MAD-X PTC is related to accsim by an identity, not by equality.* PTC runs with
    `exact=true`, so the **lattice itself** — exact drifts, and the quadrupoles' own
    kinematic terms — is nonlinear at quartic order and drives these lines with no octupole
    present. accsim, like `sextupole_detuning` before it, reports the *magnets'*
    contribution. So `PTC(with octupoles) = accsim(octupoles) + PTC(same ring, octupoles
    off)`, which holds to `1e-12` on all eight terms in both parts against `1e-4` for the
    raw comparison. The right-hand side is measured rather than fitted: the absolute gap is
    the **same number** at `0.3×`, `1×` and `3×` strength while the terms change by a
    factor of ten — which is what identifies it as the lattice's and not as a mis-scaled
    octupole term, since an error on accsim's side would be proportional to the octupoles.
    This is O3's "an exact drift detunes with no magnets in the ring" arriving one degree
    up, from an independent all-orders code.

    *xtrack leaks a nonlinear magnet into its **linear** tune.* Its `twiss` obtains the
    one-turn map by finite differences, so an octupole's cubic kick pollutes the tune it
    reports — by `8·10⁻¹⁰`, exactly proportional to `k3l`, and exactly zero with the
    octupoles removed (with no sources the two codes agree to `1e-16`). An octupole
    *cannot* shift the linear tune, since its shift is proportional to the action and the
    action is zero on the closed orbit, so this is differencing rather than physics. An RDT
    divides by `exp(−2πi(m_x Q_x + m_y Q_y)) − 1`, which converts a tune error into a term
    error roughly in proportion to the **charge**: from the measured gap alone that predicts
    `8.9·10⁻⁸` on `f4000` against `8.1·10⁻⁸` observed. Hence tolerances of `1e-6` here
    where O4's cubic ones were `1e-8`, and a lesson worth carrying: **a driving-term
    comparison against a finite-difference optics code cannot be tighter than the tune
    agreement times the charge.**

  - **PTC exposes no odd-vertical-charge term at all, so five of the thirteen ship with one
    reference leg.** Asked for all twenty cubic keys, PTC returns the same five — the normal
    sextupole's — *whatever the ring contains*. Every one has even `l − m`; every
    skew-sextupole term has odd `l − m`; none of the latter is ever returned. Established
    three ways rather than by one empty table, because "not listed" is precisely the
    inference O4 had to unlearn: the five come back **numerically zero** on a skew-only
    ring, they are **bit-identical** with and without skew sextupoles added to an octupole
    ring, and accsim's own answer for those five on that ring is exactly zero. Stated
    plainly in the shipped docs: the eight octupole terms have **two** independent reference
    codes, the five skew-sextupole terms have **one** (xtrack) plus tracking.

  - **The four source kinds land on four disjoint sets of monomials, and that is
    structural.** A monomial's degree is the multipole's order and its vertical charge
    parity separates normal from skew, so no term takes a contribution from two kinds, one
    flat table holds all twenty, and a ring carrying all four needs no cross term. Read the
    other way, it is why O4's seven numbers did not move when this landed. Gated as a
    property of the *derived generators* (no monomial appears under two kinds) as well as of
    the shipped function, so the table cannot satisfy it by construction.

  - **The one leg sharing no algebra with the derivation is a tie to a function shipped ten
    milestones ago.** An octupole's first-order generator splits into an action part and the
    rest; the rest is the eight RDTs and the action part is **exactly** J2's
    `amplitude_detuning`, which was derived by averaging the kick over the betatron phase
    rather than by solving a homological equation. Measured before asserted, as O4's `π|C⁻|`
    tie was. A **skew sextupole has no action part at all** — the same statement as "a skew
    sextupole does not shift the tune to first order" — so it gets no such tie, which is
    half the reason those five terms rest on fewer legs.

  - **The misalignment guard had to be widened, and for a sharper reason than O4's.** A
    rolled octupole is a mixture of a normal and a skew octupole, so a type-walking sum
    reads a *wrong* strength. An **offset** octupole is worse than unmodelled: it feeds down
    into a normal sextupole (`k2l = k3l x_co`) **and** a skew one (`k2sl = k3l y_co`), and
    both of those are source kinds in this same sum — so the error would land on terms the
    function returns rather than on lines outside its list, which is the one shape of
    wrongness a caller cannot detect from the output. The inherited coupling guard is not
    merely weak here but **blind**: an octupole's linear map is a drift, so offsetting or
    rolling it leaves no off-block at all to fire on. Measured and asserted, so that "it is
    refused" is not confused with "it is refused for the right reason" — O4's lesson,
    one degree up.

  - **`|f|` varies around the ring very unequally, and near-constancy is not invariance.**
    O4 recorded that `|f|` is not a ring invariant; O5 adds that the *size* of the variation
    differs enormously between terms on one and the same ring — `f0030` swings by `7.5×` and
    `f3100` by `3.0×`, while `f2001` moves by under one per cent. A term that happens to be
    nearly constant on one ring is not an invariant, it is a term whose sources sit at
    phases that nearly cancel the jump, so the gate is written on the terms that show the
    variation clearly rather than on the ones that hide it.

  - **The fixture guard earned itself twice more.** O4's assertion that every requested
    source was actually *placed* caught two separate mistakes in this milestone's own
    fixtures on their first runs — and MAD-X caught a third by aborting on a negative drift.
    The related trap it does *not* catch was found by a different gate: two positions 3 m
    apart in a 3 m cell share a beta to within 4%, so a "contrast" test can be run on two
    points that are effectively the same point.

  **What is not gated,** per O2's rule that a pre-commitment is also a list of what will not
  be gated. Nothing here covers **skew octupoles** (the remaining eight canonical quartic
  terms), decapoles and above, **second-order** RDTs, feed-down from a closed orbit or a
  misalignment, or an RDT returned as a table *along* the ring rather than at the entrance.
  Of the thirteen, three (`f4000`, `f2001`, `f2010`) have a tracked leg and the octupole
  block has the `amplitude_detuning` tie; the remaining terms are reached by the symbolic
  identity, the covariance law and the reference codes, all of which share the first-order
  formula — so a wrong coefficient common to the derivation and to both external codes
  would survive.

  Gates: `tests/analytic/test_octupole_driving_terms.py` (48),
  `tests/reference/test_octupole_driving_terms_xtrack.py` (8),
  `tests/reference/test_octupole_driving_terms_madx.py` (9).

## Out of scope (unless a milestone explicitly calls for it)

Beyond even the expansion axes above — research-grade unless a milestone explicitly
pulls it in: Touschek / IBS, strong-strong beam-beam, crab cavities, wakefields,
higher-order modes, beam loading, full GEANT4, dynamic-aperture / frequency-map
studies, PDF-uncertainty bands, and research-grade machine design.
