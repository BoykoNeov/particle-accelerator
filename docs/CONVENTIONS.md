# Conventions and pitfalls

The bug magnets. Every sign, unit, and coordinate choice is recorded here the
moment it is made. When a result disagrees with an analytic or reference value,
the cause is almost always a mismatch with something on this page — check it
before touching a tolerance.

## State vector

A single fixed 6D layout, matching the Xsuite / MAD-X external ordering so
reference cross-checks are direct (defined in `src/accsim/coords.py`):

| index | name    | meaning |
|-------|---------|---------|
| 0     | `x`     | horizontal position [m] |
| 1     | `px`    | horizontal momentum `Px / P0` (normalised, dimensionless) |
| 2     | `y`     | vertical position [m] |
| 3     | `py`    | vertical momentum `Py / P0` (normalised, dimensionless) |
| 4     | `zeta`  | longitudinal position `s − β₀·c·t` [m]; reference particle has `zeta = 0` |
| 5     | `delta` | relative **momentum** deviation `(P − P0) / P0` (dimensionless) |

- `zeta > 0` ⇒ the particle is **ahead** of the synchronous particle.
- `delta` is a **momentum** deviation, *not* an energy deviation. This choice
  changes the longitudinal transfer-matrix coefficients (see the drift below):
  with `delta`, the drift `R56 = L/γ₀²`; with the energy variable `ptau` it would
  be `L/(β₀²γ₀²)`. We use `delta` because it is the coordinate Xsuite exposes.

## Units

Internal storage: **eV** for energies and momenta (`p0·c` in eV), **metres** for
lengths, **radians**/dimensionless for the normalised momenta. Only the
dimensionless ratios `β₀`, `γ₀` enter the transfer matrices, so the eV choice is
a boundary convenience, not a physics commitment. Convert at the boundary only.

## Reference particle

`E0 = γ₀·m c²`, `β₀ = √(1 − 1/γ₀²)`, `(p0 c)² = E0² − (m c²)²`. Constructors
(`from_total_energy`, `from_kinetic_energy`, `from_momentum`, `from_gamma`) make
the energy specification explicit; the raw dataclass takes total energy.

## Drift transfer matrix (derived, not remembered)

The linear 6×6 drift map is **derived symbolically** from the exact map and pinned
by `tests/analytic/test_drift.py::test_drift_matrix_matches_symbolic_derivation`.

Exact drift of length `L` (independent variable = path length along the
reference), on the normalised coordinates:

```
pz   = √((1+δ)² − px² − py²)            # longitudinal momentum Ps/P0
x  → x + L·px/pz                         # paraxial → linear: x + L·px
y  → y + L·py/pz                         # paraxial → linear: y + L·py
zeta → zeta + L·(1 − β₀·(1+δ)/(pz·β_p))  # time-of-flight slip
```

Linearising about `(px, py, δ) = 0` gives the only non-trivial entries:

```
R12 = ∂x/∂px       = L
R34 = ∂y/∂py       = L
R56 = ∂zeta/∂δ     = L·m²/(P0² + m²) = L/γ₀²   (positive)
```

**Sign of R56:** a higher-momentum particle (`δ > 0`) is faster, arrives earlier,
so `zeta = s − β₀ct` increases ⇒ `R56 > 0`. **Limit:** as `γ₀ → ∞`, `R56 → 0` —
at ultrarelativistic energy all particles travel at ~c regardless of `δ`, so a
straight section produces no longitudinal slip.

> Common trap: the coefficient is `L/γ₀²` **for the momentum variable `δ`**. The
> often-quoted `L/(β₀²γ₀²)` is correct for the *energy* variable `ptau`. They
> agree only as `β₀ → 1`. Using the wrong one is a silent low-energy bug.

## Quadrupole strength sign (Stage 1 — implemented)

`k1 = (1/Bρ)(∂B_y/∂x)` [m⁻²], the MAD-X / Xsuite normalised gradient. The
linearised equations of motion are

```
x'' + k1·x = 0      y'' − k1·y = 0
```

so **`k1 > 0` focuses in `x` and defocuses in `y`** (R21 = −ω·sin ωL < 0 in the
focusing plane). Cross-checked against xtrack's `Quadrupole`
(`tests/reference/test_quadrupole_xtrack.py`): the full 6×6 agrees to ~1e-6,
the focusing/defocusing signs match, and the longitudinal slip
**`R56 = L/γ₀²` is carried *inside* the thick quad** (not sliced into adjacent
drifts). A pure quadrupole has no curvature ⇒ no dispersion.

- **Thick** (`Quadrupole(length, k1)`): closed-form trig block in the focusing
  plane, cosh/sinh in the defocusing plane, with `ω = √|k1|`. Written as one
  analytic family `_focusing_block(g, L)` so `k1 → 0` reduces *exactly* to a
  `Drift` and the sign of `k1` simply swaps the planes. Symplectic by
  construction: it is `exp(L·A)` of the Hamiltonian generator `A` (pinned
  symbolically in `tests/analytic/test_quadrupole.py`).
- **Thin** (`ThinQuadrupole(k1l)`): integrated strength `k1l = k1·L = 1/f`
  [m⁻¹], a zero-length kick `px → px − k1l·x`, `py → py + k1l·y`. No length ⇒
  no longitudinal slip (`R56 = 0`). It is the `L → 0` limit of the thick quad at
  fixed `k1l`; the leading correction to the thin kick is `+k1l²·L/6` (O(L)).

## Phase advance vs tune (Stage 1)

`Q = μ_total / 2π`. Keep all 2π factors explicit. Phase advance per cell `μ` from
`cos μ = ½·Tr(M)` of the one-turn matrix.

## Symplecticity

A linear map is symplectic iff `Mᵀ J M = J` (`accsim.symplectic`). Thin-lens kicks
composed with exact drifts are symplectic; thick-element matrices must be exact
closed-form maps, not truncated expansions. Any shortcut that breaks
symplecticity must be flagged — it silently damps or blows up long-term tracking.

Caveat: `(zeta, delta)` is canonically conjugate only in the constant-velocity
approximation used by the linear maps; the strictly-canonical longitudinal pair
is `(zeta, ptau)`. For the linear drift this does not break the `Mᵀ J M = J`
check, but it is flagged for the longitudinal stages (Stage 3+).

## Toolchain / environment notes

- **Python 3.14** is the development interpreter. `numpy`, `scipy`, `matplotlib`,
  `sympy`, `pytest`, `ruff` all work on it.
- **Reference code is `xtrack`, not the `xsuite` umbrella.** The `xsuite`
  meta-package fails to build on 3.14 because `xcoll` (collimation/FLUKA) hits a
  `pathlib` change (`UnsupportedOperation: cannot instantiate 'FsPath'`). The core
  tracker `xtrack` installs and imports fine, and is all the optics cross-checks
  need. The `reference` optional dependency is therefore `xtrack`.
- **xtrack JIT compilation — RESOLVED 2026-06-29 (now live via clang-cl).**
  `xtrack` compiles C kernels on first use via `cffi` → the platform C compiler.
  On Windows that path had three independent failure layers; all are now handled
  by the `tests/reference/_xtrack_jit.py` fix-up (applied from
  `tests/reference/conftest.py`). The diagnosis, kept for the record:
  1. Needs `setuptools` in the venv (stdlib `distutils` gone on 3.12+) — installed.
  2. **xobjects discards compiler flags on Windows.** In
     `xobjects/context_cpu.py::compile_kernel`, the `os.name == "nt"` branch sets
     `xtr_compile_args = []` (literal comment `# TODO: to be handled properly`),
     throwing away **both** the computed `-I<site-packages>` include flag (→
     `C1083: cannot open 'xtrack/multisetter/multisetter.h'`) **and** the
     `-DXO_CONTEXT_CPU` / `-DXO_CONTEXT_CPU_SERIAL` context defines (→ `C1189:
     Unknown context`). The spaced project path is **not** the cause — it is passed
     to the compiler as a single argv element correctly (corrects the earlier
     "spaced path" hypothesis).
  3. **xtrack's own C source is not MSVC-clean.** Past layers 1–2, MSVC `cl.exe`
     rejects xtrack source with `C2166: l-value specifies const object`
     (`track_misalignments.h`, the `S_SHIFT(part0, -mis_s)` macro on a negated
     `const`). GCC/Clang accept this; MSVC's stricter front-end is the outlier —
     xsuite is developed on Linux.
  - **Fix that worked:** compile with **clang-cl** instead of `cl.exe`. clang-cl is
    a cl-compatible front-end that reproduces the reference toolchain's GCC/Clang
    behaviour (clearing the `C2166`) while emitting MSVC-ABI objects the MSVC
    linker links. The `_xtrack_jit` fix-up monkeypatches the distutils MSVC
    compiler to: swap `self.cc → clang-cl`, re-add `site-packages` to the include
    path, restore the `XO_CONTEXT_CPU*` defines, and drop `/GL`+`/LTCG` (clang-cl
    bitcode is incompatible with the MSVC linker's LTCG). It is a **no-op** off
    Windows and when clang-cl is absent, so reference tests skip gracefully there.
    Requires `winget install LLVM.LLVM` (clang-cl 22.x verified); `xpart` must also
    be installed (xtrack's R-matrix/Twiss helpers import it).
  - **Dead ends checked:** `pip install xsuite-prebuilt-kernels` → no PyPI
    distribution. Relocating to a space-free path → would not help (layer 2 is
    path-independent).
  - **Status:** `tests/reference/test_drift_xtrack.py` now **passes** (not skips) —
    the full 6×6 drift map agrees with xtrack to ~1.5e-10 (`R56 = L/γ₀²`, the
    momentum-variable value `0.5` for `γ₀=2`, confirming it over the energy-variable
    `0.667`; sign `+`). This validates the **Stage 0** drift convention against the
    reference. It is **not** Stage 1 acceptance (the FODO Twiss `<1e-6` check is
    still ahead), and the `zeta`-sign reconciliation is settled **for the drift
    R56 only** — keep the flag live for quads/dipoles/full-ring in Stage 1.
  - **CI note:** CI runs ruff + the analytic suite only; the `reference` marker is
    not exercised in CI (and clang-cl is not installed there). This cross-check is
    therefore a **local Windows gate**, not a per-push CI regression catch.
- **Expect a `zeta` sign mismatch vs Xsuite when the cross-check first runs.**
  That is a convention reconciliation, not a physics bug: matching the reference's
  sign is part of "match Xsuite ordering," so adopt Xsuite's sign if it differs —
  do not change correct physics to chase it.
- Until the JIT works, the drift convention rests on the **symbolic derivation**
  (two independent routes agree), which is itself a gold-standard analytic check.
