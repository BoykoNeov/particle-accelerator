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
> blocked on this machine and is now **resolved** (built via clang-cl; the Stage 0
> drift cross-check passes) — see the toolchain notes in
> [`CONVENTIONS.md`](CONVENTIONS.md). The `zeta`-sign reconciliation remains open
> for quads/dipoles/full-ring until Stage 1 exercises them.

## Stage 0 — Scaffold ✅ COMPLETE

Repo, `pyproject.toml`, CI, the `Element`/`Lattice`/`Tracker`/`Particle`
skeletons, plotting, and the analytic test harness.

- **Acceptance:** a `Drift` propagates a particle to the analytically expected
  position; CI is green. ✅
- Delivered beyond the minimum: the full 6×6 drift map (incl. the longitudinal
  `R56 = L/γ₀²`, derived symbolically), a symplecticity check, and a
  gracefully-skipping xtrack cross-check scaffold.

## Stage 1 — Beam optics (linear transverse) ⬅ IN PROGRESS

Transfer-matrix formalism; `Drift`, `Quadrupole` (thin + thick), `Dipole`;
one-turn map; Twiss propagation (β, α, dispersion, phase advance); tunes.

**Status:** the FODO acceptance gate below is **met** ✅ — `Quadrupole`
(thin + thick), Twiss matching/propagation, and tunes are done and
xtrack-validated. **Remaining:** the `Dipole` (sector bend) + dispersion to round
out the stage's element set before Stage 2.

> **Do these FIRST, before building Stage 1 physics:**
> - ✅ **Get the Xsuite/xtrack cross-check live.** DONE (2026-06-29). The JIT
>   compile, previously dark, now builds via clang-cl (`tests/reference/_xtrack_jit.py`);
>   `test_drift_xtrack.py` passes with the full 6×6 drift map matching xtrack to
>   ~1.5e-10. Full diagnosis in CONVENTIONS.md. This validates the Stage 0 drift
>   convention against the reference; the Stage 1 FODO Twiss `<1e-6` check is the
>   remaining acceptance gate.
> - **Expect a `zeta` sign mismatch vs Xsuite as the first "discrepancy."** That
>   is a convention reconciliation (adopt Xsuite's sign — it is part of "match
>   Xsuite ordering"), **not** a physics bug. Do not "fix" correct physics here.
> - **Add a composition-order test when `Quadrupole` lands.** The existing
>   `test_drifts_compose_additively` cannot distinguish matrix order (drifts
>   commute). Add an asymmetric drift+quad sequence whose result changes if the
>   `M_last @ … @ M_first` order is reversed.

- **Acceptance:** for a single FODO cell, the phase advance per cell `μ` (from
  `cos μ = ½·Tr M`) and the β-functions match the **symbolically-derived**
  closed-form thin-lens result (derive it, don't trust a remembered coefficient).
  β should oscillate between a maximum at the focusing quad and a minimum at the
  defocusing quad. Cross-check a small ring against Xsuite Twiss to < 1e-6. ✅
  **MET** — `tests/analytic/test_fodo_cell.py` (symbolic `μ`, `β_max`, `β_min`,
  max-at-F/min-at-D oscillation) and `tests/reference/test_fodo_twiss_xtrack.py`
  (thick-quad FODO ring vs xtrack 4D Twiss, agreement ~1e-14 ≪ 1e-6).

## Stage 2 — Magnetic lenses

FODO lattices; thin vs thick lens; natural chromaticity; sextupoles for
chromaticity correction (linear effect); beam-envelope plots.

- **Acceptance:** the FODO cell's natural chromaticity matches the analytic
  estimate; the stability boundary (`|Tr M| < 2`) matches the analytic
  phase-advance limit.

## Stage 3 — Synchrotron motion (longitudinal)

RF bucket, synchronous phase, momentum-compaction factor, synchrotron tune,
longitudinal phase-space tracking, separatrix.

- **Acceptance:** the small-amplitude synchrotron tune `Qs` matches the analytic
  formula; the bucket height matches; particles launched inside the separatrix
  stay bounded over ≥ 1e4 turns.

## Stage 4 — Beam losses

Geometric apertures + collimators with survival/loss accounting; simple lifetime
models (aperture and quantum lifetime). **Touschek and intrabeam scattering are
advanced/optional — stub, don't build, unless asked.**

- **Acceptance:** a particle outside the aperture is flagged at the correct
  longitudinal location; transmission through a known aperture matches a hand
  calculation; the loss map reproduces a simple analytic case.

## Stage 5 — RF cavities

Standalone `RFCavity` (voltage, harmonic number, phase), multi-cavity support,
acceleration ramp, energy gain per turn. **Beam loading, higher-order modes, and
wakefields are out of scope** unless a milestone adds them.

- **Acceptance:** energy gain per turn equals `qV·sin(φs)`; the synchronous
  particle stays synchronous; behaviour is consistent with the Stage 3 model.

## Stage 6 — Collider design

Two beams, interaction point(s), low-β insertion, luminosity from beam parameters,
crossing angle; weak-strong beam-beam kick and beam-beam tune shift.
**Strong-strong beam-beam, crab cavities, and dynamic-aperture studies are
research-grade and out of scope** unless explicitly requested.

- **Acceptance:** the luminosity formula reproduces a textbook worked example for
  a known machine; the beam-beam tune shift `ξ` matches the analytic expression;
  a head-on weak-strong kick conserves the expected invariants.

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
