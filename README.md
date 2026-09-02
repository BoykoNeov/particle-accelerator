# accsim — a particle accelerator simulator

A modular, **physics-correct** particle accelerator simulator, grown in stages
from linear beam optics upward. It is an educational-to-serious-hobby tool:
physically correct at the analytic / toy level and validated against closed-form
results and the [Xsuite](https://xsuite.readthedocs.io/) reference code — **not**
a research-grade machine-design package.

> **The one thing that matters most:** physics correctness is the bottleneck,
> not code volume. Plausible-looking accelerator code is routinely, subtly wrong
> (a flipped sign, a missing 2π, a stray γ). Every physics quantity is pinned by
> a closed-form analytic test *before* it is implemented, and cross-checked
> against Xsuite where applicable. See [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)
> and [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Status

**Stages 0–4 complete, xtrack-validated.** Stage 0 (Scaffold): core abstractions
(`Element`, `Lattice`, `Tracker`, `Particle`/`Bunch`, `ReferenceParticle`), a
fully-derived `Drift`, the analytic test harness, and CI. Stage 1 (linear
transverse beam optics): thin/thick `Quadrupole`, `Dipole`, FODO Twiss (β, α, tune,
dispersion). Stage 2 (magnetic lenses): natural chromaticity, `Sextupole`
(chromaticity feed-down), stability boundary, and beam-envelope plots. Stage 3
(synchrotron motion): momentum compaction / slip factor, `RFCavity`, synchrotron
tune `Qs`, and the nonlinear RF bucket / separatrix. Stage 4 (beam losses):
`Aperture`/`Collimator` with loss-aware tracking (transmission + loss map) and the
aperture-limited quantum lifetime. Stage 5 (RF cavities): the energy ramp,
synchronous phase, and adiabatic damping. Stage 6 (collider design): luminosity,
the hourglass effect, and the (elliptical) beam-beam kick and tune shift. Stage 7
(synchrotron radiation): radiation integrals, damping times and partition numbers,
and the equilibrium emittance / energy spread. Beyond the stages, the roadmap's
**expansion axes A–P** are delivered through P1: betatron coupling, matching, the closed
orbit and its correction, nonlinear kicks and feed-down, misalignments, exact element
maps, the off-momentum optics, spin, normalised coordinates and driving terms, and — the
newest — the **transfer map to second order** (`accsim.taylor`: the `6×6×6` map per
element and per turn, its composition rule, and the symplectic identities on it). Each
stage's 6×6 maps are cross-checked against [Xsuite](https://xsuite.readthedocs.io/) and,
independently, [MAD-X](https://mad.web.cern.ch/mad/) via cpymad; the second-order map
adds MAD-X PTC as a third arbiter. A separate **Phase 2** builds
collision-event physics (Drell-Yan angular coefficients and the Lam-Tung relation,
sin²θ_W from A_FB, W-mass Jacobian edge, b-tagging) on Pythia8 + Delphes, behind
opt-in feature switches. See the roadmap for the staged plan and what's next.

## Quick start

```bash
# create an isolated environment and install with dev + test tooling
py -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on POSIX
pip install -e ".[dev]"

# run the always-on analytic test suite
pytest

# walk one machine end to end: inject -> accelerate -> store -> collide -> account
python examples/build_a_machine.py

# (optional) install the validation reference codes and run the cross-checks
pip install -e ".[reference]"
pytest -m reference   # xtrack >= 0.111 JIT-compiles; the reference conftest enables it
```

```python
import accsim as ac

ref = ac.ReferenceParticle.from_total_energy(ac.PROTON_MASS_EV, 10e9)  # 10 GeV proton
lattice = ac.Lattice([ac.Drift(2.0)], ref)
out = ac.Tracker(lattice).track(ac.Particle(x=1e-3, px=2e-4))
print(out)  # x advanced by L*px
```

## Coordinates (the convention everything depends on)

6D state vector `(x, px, y, py, zeta, delta)`, matching the Xsuite/MAD-X external
ordering. `px, py` are momenta normalised to the reference momentum `P0`;
`delta = (P − P0)/P0` is a **momentum** deviation; `zeta = s − β₀ct` (a particle
ahead of the reference has `zeta > 0`). The full rationale, sign choices, and the
symbolic drift derivation live in [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).

## Project layout

```
src/accsim/        # package: coords, reference particle, elements/, lattice, tracking, twiss, orbit,
                   #          taylor (the map beyond first order), symplectic, radiation, spin, plotting
examples/          # build_a_machine.py — one narrated end-to-end run of the whole stack
pipelines/         # opt-in Pythia8/Delphes event-physics chains (Phase 2)
tests/analytic/    # closed-form checks — always run in CI
tests/reference/   # Xsuite/MAD-X cross-checks — marked `reference`, skipped if the dep is absent
docs/              # ROADMAP.md, CONVENTIONS.md
```

## Development

- **Tooling:** `ruff` for both linting and formatting (`ruff check`, `ruff format`),
  `pytest` for tests, `sympy` to derive closed-form benchmarks. All configured in
  `pyproject.toml`.
- **Workflow:** test-first for physics. Write the analytic test with the known
  answer, then implement until it reproduces that number. A disagreement is a
  physics bug to localise (convention / unit / 2π / sign), never a tolerance to
  loosen.

## License

Boyko Non-Commercial License v1.0 (BNCL-1.0) — non-commercial use only; commercial
use requires a separate license from the copyright holder. See [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE).
