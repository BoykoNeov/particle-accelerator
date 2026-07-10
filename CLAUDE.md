# accsim — Claude Code working agreement

A modular, physics-correct particle accelerator simulator, grown in stages. This
file is the contract for every session. Detail lives in `docs/` (read on demand —
not embedded here): the staged plan in `docs/ROADMAP.md`, and every sign / unit /
coordinate choice in `docs/CONVENTIONS.md`.

## The one rule that matters most

**Physics correctness is the bottleneck, not code volume.** Plausible accelerator
code is routinely, subtly wrong (a flipped sign, a missing 2π, a stray γ) and
still produces convincing plots. Everything below defends against that.

## Working agreement

- Start each session by reading the current open milestone in `docs/ROADMAP.md`.
- **Test-first for physics.** Before implementing, write the analytic test with
  the known closed-form answer. The code is done only when it reproduces that
  number. Where a coefficient is involved, **derive it symbolically** (sympy) —
  don't trust a remembered constant.
- **Disagreement is a physics bug, not a tolerance to loosen.** When a result
  misses the analytic/reference value, localise it — convention? unit? 2π? sign?
  Never relax the tolerance to make a test pass.
- **Validate, don't advance.** Do not start stage N+1 until stage N's acceptance
  tests pass (analytic + any applicable Xsuite cross-check).
- One element or feature per change.
- **Everything past the pure-Python baseline is an opt-in runtime switch.** The
  baseline (accelerator core + toy event generator; numpy/scipy/matplotlib only)
  is always on. Any addon / expansion / module / component that adds an external
  tool, a heavy dependency, or an optional extension must sit behind
  `accsim.features` (`require()` for in-package callers, the `ACCSIM_ENABLE_*` env
  var for standalone scripts/CI), **default OFF**. Don't scaffold a switch until
  real code lands behind it. See `docs/CONVENTIONS.md` → *Feature switches*.
- Maintain `docs/CONVENTIONS.md`: record every sign / unit / coordinate choice the
  moment it is made.
- Flag explicitly any approximation that breaks symplecticity or physical
  fidelity, and say what it costs.
- Do **not** pull in research-grade scope (Touschek/IBS, strong-strong beam-beam,
  wakefields, full GEANT4, dynamic aperture) unless the milestone calls for it.
- Prefer clarity over cleverness — this is a teaching codebase as much as a tool.

## Stack & tooling

- Python 3.11+ (developed on 3.14), `numpy`, `scipy`; `matplotlib` (+ optional
  `plotly`) for plots; `sympy` for symbolic derivations.
- `pytest` with `tests/analytic/` (closed-form, always run) and `tests/reference/`
  (`xtrack`/MAD-X cross-checks, behind the `reference` marker, skippable).
- `ruff` for **both** lint and format (`ruff check`, `ruff format`) — no separate
  black. Config in `pyproject.toml`.
- `src/` layout, full type hints, CI runs ruff + the analytic suite on every push.

## Environment quickstart

```bash
.venv/Scripts/python.exe -m pytest        # analytic suite (always green target)
.venv/Scripts/python.exe -m ruff check .   # lint
.venv/Scripts/python.exe -m ruff format .  # format
```

- Use the project `.venv` (Python 3.14). The reference dep is **`xtrack`**, not the
  `xsuite` umbrella (which fails to build on 3.14 — see `docs/CONVENTIONS.md`).
- **xtrack JIT on Windows (resolved 2026-06-29):** the C-kernel build now compiles
  via **clang-cl** (`winget install LLVM.LLVM`), wired in by the `_xtrack_jit`
  test fix-up; the reference cross-checks pass locally. It is a no-op off Windows
  and when clang-cl is absent, so reference tests still skip (not fail) elsewhere,
  and CI runs only ruff + the analytic suite. See `docs/CONVENTIONS.md`.

## Coordinates (full detail in `docs/CONVENTIONS.md`)

6D state `(x, px, y, py, zeta, delta)`, Xsuite/MAD-X ordering. `px,py` normalised
to `P0`; `delta = Δp/p₀` (momentum, not energy); `zeta = s − β₀ct` (ahead ⇒ > 0).

## Repo etiquette

- Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`). Each commit
  should compile and keep the analytic suite green. Commit per milestone/feature.
- At the end of a work batch (or when the user says "session end"): update
  `docs/` and memory, then commit and push.
