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
  hourglass is flagged out of scope. `ReferenceParticle.classical_radius_m`
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
  Bassetti–Erskine flagged out of scope. See CONVENTIONS.md → *Weak-strong
  beam-beam kick*.
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
  `tests/analytic/test_low_beta_insertion.py`. Hourglass / strong-strong / crab
  cavities / dynamic aperture remain out of scope.

## Phase 2 (optional) — Collision event physics

**Do not rebuild event generators.** Orchestrate the established chain: event
generator (Pythia / MadGraph) → fast detector sim (Delphes) → analysis in the
scientific-Python / ROOT ecosystem. A from-scratch toy 2→2 generator (matrix
element + RAMBO + PDFs) is welcome **as a clearly-labelled learning module only**.

- **Acceptance:** the toy generator's total cross-section for a known process
  matches the analytic value within Monte-Carlo error; the orchestrated pipeline
  runs end-to-end and produces a labelled distribution.

## Out of scope (unless a milestone explicitly calls for it)

Touschek / IBS, strong-strong beam-beam, crab cavities, wakefields, higher-order
modes, beam loading, full GEANT4, dynamic-aperture studies, research-grade
machine design.
