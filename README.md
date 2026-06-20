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

**Stage 0 (Scaffold) — complete.** Core abstractions (`Element`, `Lattice`,
`Tracker`, `Particle`/`Bunch`, `ReferenceParticle`), a fully-derived `Drift`
element, the analytic test harness, and CI are in place. See the roadmap for the
staged plan.

## Quick start

```bash
# create an isolated environment and install with dev + test tooling
py -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on POSIX
pip install -e ".[dev]"

# run the always-on analytic test suite
pytest

# (optional) install the validation reference code and run the cross-checks
pip install -e ".[reference]"
pytest -m reference
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
src/accsim/        # package: coords, reference particle, elements/, lattice, tracking, symplectic, plotting
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

MIT — see [`LICENSE`](LICENSE).
