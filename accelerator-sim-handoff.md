# Particle accelerator simulator — Claude Code project handoff

## 0. How to use this document

This is the foundational brief for the project. The fastest path:

- Drop the **working agreement (§8)**, **stack (§3)**, and a pointer to conventions into a root `CLAUDE.md` so Claude Code reads them at the start of every session. Keep that file tight — Claude Code's own guidance is that memory files over ~200 lines start to lose adherence.
- Keep the **roadmap (§6)** and **validation strategy (§7)** as a separate `docs/ROADMAP.md`, referenced from `CLAUDE.md` (e.g. `@docs/ROADMAP.md`), so the detail loads without bloating every prompt.
- Treat each stage in §6 as one milestone. Do not start a stage until the previous stage's acceptance tests pass.

---

## 1. What we're building

A modular particle accelerator simulator, grown in stages from linear beam optics up to (optionally) collider and collision-event physics. There are two distinct sub-projects with a clean handoff at the interaction point:

- **Accelerator / beam dynamics** — gets the beams *to* the collision point (optics, lenses, longitudinal motion, losses, RF, collider layout).
- **Event physics (optional Phase 2)** — what comes *out* of a collision. This sub-project orchestrates established tools rather than rebuilding them.

**Fidelity target:** physically correct at the analytic / toy level, validated against closed-form results and the Xsuite reference code. This is an educational-to-serious-hobby simulator, **not** a research-grade machine-design tool. Anything that would require research-grade scope is explicitly out of bounds unless a milestone calls for it (see §6).

---

## 2. The one thing that matters most

**Physics correctness is the bottleneck, not code volume.** Plausible-looking accelerator code is routinely, subtly wrong — a flipped sign, a missing factor of 2π, a relativistic γ in the wrong place — and it still runs and produces convincing plots. The entire project is structured around defending against this:

- **Test-first for physics.** Before implementing a stage, write the test with the known analytic answer. The code is "done" only when it reproduces that number.
- **Validate, don't advance.** Never start stage N+1 until stage N passes its analytic benchmarks (and, where applicable, agrees with Xsuite).
- **Disagreement is a physics bug, not a test to loosen.** When a result misses the analytic/reference value, localize it (convention? unit? 2π? sign?) — do not relax the tolerance.

---

## 3. Tech stack

- **Language:** Python 3.11+, `numpy`, `scipy`.
- **Plotting:** `matplotlib` for static figures; optionally `plotly` for interactive phase-space portraits.
- **Testing:** `pytest`, with a dedicated `tests/analytic/` suite of closed-form checks.
- **Validation-only deps (dev, not runtime):** `xsuite` (CERN) for cross-checking optics and tracking; optionally `cpymad` (MAD-X) for lattice cross-checks. Gate the reference tests behind a pytest marker so the suite still runs if these aren't installed.
- **Project hygiene:** `src/` layout, `pyproject.toml`, `ruff` + `black`, full type hints, CI that runs the analytic suite on every push.

---

## 4. Architecture

Core abstractions:

- **`Element`** — base class. Each lattice element exposes a transfer map: a 6×6 linear matrix at minimum, later a possibly-nonlinear map. Subclasses: `Drift`, `Quadrupole`, `Dipole`/`SBend`, `Sextupole`, `RFCavity`, `Aperture`, `Marker`.
- **`Lattice`** — an ordered sequence of elements. Computes the one-turn map, closed orbit, and accumulated optics.
- **`TwissResult`** — β, α, γ, dispersion, phase advance, tunes.
- **`Tracker`** — pushes particles / bunches through the lattice turn by turn.
- **`Particle` / `Bunch`** — 6D phase-space state.

Suggested layout:

```
src/accsim/
  elements/      # drift, quad, dipole, sextupole, rf, aperture
  lattice.py     # Lattice, one-turn map, closed orbit
  twiss.py       # Twiss propagation, tunes, chromaticity
  tracking.py    # Tracker, Particle, Bunch
  longitudinal.py
  losses.py
  collider.py
  plotting.py
tests/
  analytic/      # closed-form checks (always run)
  reference/     # Xsuite / MAD-X cross-checks (marked, skippable)
docs/
  ROADMAP.md
  CONVENTIONS.md
```

---

## 5. Conventions and pitfalls (the bug magnets)

Document every one of these in `docs/CONVENTIONS.md` and update it whenever a new choice is introduced.

- **State vector.** Use a 6D vector `(x, px, y, py, z, δ)` where momenta are normalized to the reference momentum and `δ = Δp/p₀`. Match Xsuite/MAD-X ordering so reference cross-checks are direct. Write the exact layout down once and never deviate.
- **Units.** Pick one internal convention (SI, or the accelerator mix of GeV / m / rad) and convert only at the boundary. State it explicitly.
- **Magnet strength sign.** Define `K1 > 0` as focusing in `x`, and keep the focusing/defocusing sign consistent across thin- and thick-lens forms.
- **Phase advance vs tune.** Tune `Q = μ_total / 2π`. Keep the 2π factors explicit and consistent.
- **Relativistic factors.** Be explicit about where β and γ enter longitudinal dynamics and the RF kick.
- **Symplecticity.** Thin-lens kicks composed with exact drifts are symplectic; thick-element matrices must be the exact closed-form maps, not truncated expansions. Flag any shortcut that breaks symplecticity — it will silently damp or blow up long-term tracking.

---

## 6. Roadmap

Each stage is a milestone defined by its acceptance tests. The stage is complete when those pass.

### Stage 0 — Scaffold
- Repo, `pyproject.toml`, CI, plus `Element`/`Lattice`/`Tracker`/`Particle` skeletons and the plotting + test harness.
- **Acceptance:** a `Drift` propagates a particle to the analytically expected position; CI is green.

### Stage 1 — Beam optics (linear transverse)
- Transfer-matrix formalism; `Drift`, `Quadrupole` (thin + thick), `Dipole`; one-turn map; Twiss propagation (β, α, dispersion, phase advance); tunes.
- **Acceptance:** for a single FODO cell, the phase advance per cell `μ` (from `cos μ = ½·Tr M` of the one-turn matrix) and the β-functions match the closed-form thin-lens result — **have Claude Code derive the closed form symbolically and verify it**, rather than trusting a remembered coefficient. β should oscillate between a maximum at the focusing quad and a minimum at the defocusing quad. Cross-check a small ring against Xsuite Twiss to < 1e-6.

### Stage 2 — Magnetic lenses
- FODO lattices; thin vs thick lens; natural chromaticity; sextupoles for chromaticity correction (linear effect); beam-envelope plots.
- **Acceptance:** the FODO cell's natural chromaticity matches the analytic estimate; the stability boundary (`|Tr M| < 2`) matches the analytic phase-advance limit.

### Stage 3 — Synchrotron motion (longitudinal)
- RF bucket, synchronous phase, momentum-compaction factor, synchrotron tune, longitudinal phase-space tracking, separatrix.
- **Acceptance:** the small-amplitude synchrotron tune `Qs` matches the analytic formula; the bucket height matches; particles launched inside the separatrix stay bounded over ≥ 1e4 turns.

### Stage 4 — Beam losses
- Geometric apertures + collimators with survival/loss accounting; simple lifetime models (start with aperture and quantum lifetime). **Touschek and intrabeam scattering are advanced/optional** — stub them, don't build them unless asked.
- **Acceptance:** a particle outside the aperture is flagged at the correct longitudinal location; transmission through a known aperture matches a hand calculation; the loss map reproduces a simple analytic case.

### Stage 5 — RF cavities
- Standalone `RFCavity` (voltage, harmonic number, phase), multi-cavity support, acceleration ramp, energy gain per turn. **Beam loading, higher-order modes, and wakefields are out of scope** unless a milestone adds them.
- **Acceptance:** energy gain per turn equals `qV·sin(φs)`; the synchronous particle stays synchronous; behavior is consistent with the Stage 3 longitudinal model.

### Stage 6 — Collider design
- Two beams, interaction point(s), low-β insertion, luminosity from beam parameters, crossing angle; weak-strong beam-beam kick and beam-beam tune shift. **Strong-strong beam-beam, crab cavities, and dynamic-aperture studies are research-grade and out of scope** unless explicitly requested.
- **Acceptance:** the luminosity formula reproduces a textbook worked example for a known machine's parameters; the beam-beam tune shift `ξ` matches the analytic expression; a head-on weak-strong kick conserves the expected invariants.

### Phase 2 (optional) — Collision event physics
- **Do not rebuild event generators.** Orchestrate the established chain: an event generator (Pythia or MadGraph) → fast detector simulation (Delphes) → analysis in the scientific-Python / ROOT ecosystem.
- A from-scratch toy 2→2 generator (matrix element + RAMBO phase-space sampling + parton distribution functions) is welcome **as a clearly-labeled learning module only**, never presented as physically precise.
- **Acceptance:** the toy generator's total cross-section for a known process matches the analytic value within Monte-Carlo error; the orchestrated pipeline runs end-to-end and produces a labeled distribution.

---

## 7. Validation strategy (non-negotiable)

- **`tests/analytic/`** — every physics quantity has a closed-form check. These always run in CI.
- **`tests/reference/`** — Xsuite (and optionally MAD-X) cross-checks for optics and tracking, behind a pytest marker so they're skippable when those deps are absent. These catch the coefficient/convention errors that hand-derived analytic checks can share.
- **Long-term tracking sanity** — track a matched particle for 1e4–1e5 turns and confirm the action/emittance does not drift. This is the symplecticity smoke test.
- **Gate.** A stage's acceptance tests (analytic + any applicable reference checks) must pass before the next stage is started.

---

## 8. Working agreement for Claude Code

(This section is the part to lift into `CLAUDE.md`.)

- Start each session by reading this agreement and the current open milestone.
- One element or feature per change. Write or update the analytic test **first**.
- When a result disagrees with the analytic or reference value, treat it as a physics bug to localize — check convention, units, 2π, sign — never loosen the tolerance.
- Maintain `docs/CONVENTIONS.md`; record every sign / unit / coordinate choice the moment it's made.
- Flag explicitly any approximation that breaks symplecticity or physical fidelity, and say what it costs.
- Do not pull in research-grade scope (Touschek/IBS, strong-strong beam-beam, wakefields, full GEANT4) unless the milestone calls for it.
- Prefer clarity over cleverness: this is a teaching codebase as much as a tool.

---

## 9. First task

Implement **Stage 0** plus **Stage 1 through Twiss propagation for a single FODO cell**, with the analytic β-function and phase-advance tests passing. Then **stop and report** the β-functions and tune for review before starting Stage 2.
