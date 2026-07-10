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

## Dipole — sector bend (Stage 1 — implemented)

`Dipole(length, angle)`: a **pure sector** bend (no pole-face/edge angles, no
gradient `k1`), bending horizontally. Curvature `h = 1/ρ = θ/L`, `θ = angle`.
Edge focusing and combined-function gradients are **Stage 2**, not here. The 6×6
is `exp(L·A)` of the sector-bend Hamiltonian generator (symplectic by
construction); with `C = cos θ`, `S = sin θ`:

```
R11 = R22 = C          R12 = S/h = ρS         R21 = −hS = −S/ρ
R16 = (1−C)/h = ρ(1−C) R26 = S                (dispersion; R16 > 0 ⇒ outward)
R34 = L                                        (vertical = plain drift)
R51 = −S               R52 = (C−1)/h = −R16    (symplectic partners of dispersion)
R56 = ρS − L + L/γ₀²   = L/γ₀² − ρ(θ − S)
```

- **Dispersion sign:** a higher-momentum particle (`δ > 0`) bends less, so it is
  displaced **outward** ⇒ `R16 > 0`.
- **`R51`/`R52` are forced by symplecticity** from the dispersion:
  `R51 = R21·R16 − R11·R26`, `R52 = R22·R16 − R12·R26`. Deriving the map as
  `exp(L·A)` makes this automatic — a hand-built map that gets these wrong fails
  `is_symplectic`.
- **`R56`** is the drift slip `L/γ₀²` (same momentum-variable coefficient as the
  drift/quad) **minus** the extra arc the design orbit travels, `ρ(θ − S)`. The
  momentum-compaction interpretation of this term belongs to Stage 3 — not built
  here.
- **θ → 0 limit:** every curvature term vanishes and the map is exactly a
  `Drift(L)` (`R56 → L/γ₀²`).
- Cross-checked entrywise against xtrack's `Bend` configured as a pure sector
  (`edge_entry/exit_active = 0`, `k1 = 0`) to ~1e-6
  (`tests/reference/test_dipole_xtrack.py`).

## Dispersion in Twiss (Stage 1 — implemented)

The matched linear dispersion `D = (Dx, Dpx, Dy, Dpy) = d(x,px,y,py)/dδ` is the
first-order off-momentum closed orbit. Conventions:

- **Variable is `δ` (momentum):** `D = dx/dδ`. **xtrack's `twiss.dx` uses the
  same `δ` variable** — verified ratio `xtrack.dx / D = 1.0` at γ₀ = 5
  (β₀ ≈ 0.98), decisively **not** the MAD-X `pt`-based `DX = (1/β₀)·dx/dδ`
  (which would differ by ≈ 2% there). Tested at γ₀ = 5 deliberately, so a stray
  `1/β₀` would be an unmistakable 2% gap rather than a 0.1% one
  (`tests/reference/test_dispersion_xtrack.py`).
- **Matched:** `D = (I₄ − M₄)⁻¹·[R16, R26, R36, R46]ᵀ` from the one-turn 4×4
  transverse block `M₄` and its `δ`-column. For an uncoupled lattice with no
  vertical bending, `Dy = Dpy = 0` falls out (the vertical `δ`-column is zero).
- **Propagation is affine:** `D(s₊) = M₄ᵉˡᵉᵐ·D(s) + [R16, R26, R36, R46]ᵀ` —
  matrix transport plus the element's dispersive kick. This is **not** the
  quadratic `B = C·B·Cᵀ` rule used for `β`/`α`; dispersion is an orbit, not a
  second moment.
- A lattice with no bending magnet has `D ≡ 0` everywhere (the `Twiss`
  dispersion fields default to `0.0`).

## Twiss / phase advance / tune (Stage 1 — implemented)

Linear Courant-Snyder optics live in `src/accsim/twiss.py`. Conventions:

- **Matched (periodic) Twiss** comes from the 2×2 transverse blocks of the
  one-turn 6×6: `cos μ = ½·Tr(block)`; `β = M12/sin μ`; `α = (M11−M22)/(2 sin μ)`.
  The matched `β` is **positive by construction** — the sign of `sin μ` is fixed
  by `sign(M12)`, i.e. `sin μ = sign(M12)·√(1−cos²μ)`. Holds even when
  `μ ∈ (π, 2π)` makes `M12 < 0`.
- **Stability** of a plane requires `|½·Tr(block)| < 1` (`|Tr| < 2`). An unstable
  plane has no real matched `β`; `match_periodic`/`closed_twiss` raise
  `UnstableLatticeError` rather than returning a complex β.
- **Propagation** is `B₁ = C·B₀·Cᵀ` with `B = [[β, −α], [−α, γ]]`,
  `γ = (1+α²)/β`. This is exact and preserves the invariant `γβ − α² = 1` when
  `det C = 1` (verified symbolically).
- **`Q = μ_total / 2π`**, and the phase is **accumulated continuously** along the
  lattice — `Δμ = atan2(C12, β₀·C11 − α₀·C12)` per element, summed — **not** taken
  from `acos` of the one-turn matrix. `acos` yields only the *fractional* tune
  (it aliases `μ` into `[0, π]`) and loses the integer part; continuous
  accumulation recovers the full tune. Keep all 2π factors explicit.
- **Scope:** transverse `x`/`y` only (drifts + quads neither couple the planes
  nor disperse, so the 2×2 reduction is exact). Dispersion (coupling to `delta`)
  arrives with the `Dipole`.
- **Cross-check:** a thick-quad FODO ring matches xtrack's 4D Twiss
  (`β`, `α`, `μ/2π`, `Q` in both planes) to **machine precision** (~1e-14, gate
  is <1e-6) — `tests/reference/test_fodo_twiss_xtrack.py`.

### Thin-lens FODO closed form (acceptance gate)

For the symmetric cell `QF/2 − drift(L) − QD − drift(L) − QF/2` (full-quad focal
length `f`, half-cell drift `L`, F split into 2f halves at the ends), derived
symbolically (`tests/analytic/test_fodo_cell.py`):

```
cos μ = 1 − L²/(2f²)        ⇒  sin(μ/2) = L/(2f)
β_max = L_cell·(1 + sin(μ/2)) / sin μ      (at the F centre)
β_min = L_cell·(1 − sin(μ/2)) / sin μ      (at the D centre,  L_cell = 2L)
```

`β_x` peaks at the F quad and troughs at the D quad; `β_y` is the mirror image
(`β_y(F) = β_min`). Because the D quad is a single thin kick (not split), no
element boundary sits exactly at its centre: `β` is continuous across it while
`α` flips sign antisymmetrically, so `α ≠ 0` at the recorded D-centre boundary.

## Natural chromaticity (Stage 2 — implemented)

`natural_chromaticity(lattice)` returns `(Q'_x, Q'_y) = (dQ_x/dδ, dQ_y/dδ)`, the
tune's first-order momentum dependence from the off-momentum weakening of the
quadrupole gradient, `k1 → k1/(1+δ)`. Conventions:

- **Definition is the *un-normalised* derivative** `Q' = dQ/dδ` — **not** the
  normalised `ξ = Q'/Q`. This matches **xtrack's `twiss.dqx`/`dqy`**, pinned by a
  convention guard that finite-differences xtrack's *own* tunes at `δ = ±h` and
  recovers `tw.dqx` (`tests/reference/test_chromaticity_xtrack.py`). A stray `Q`
  or `2π` would show up there.
- **Per-plane signs are opposite** because a quad focuses `x` with `+k1` and `y`
  with `−k1`:
  ```
  Q'_x = −(1/4π) ∮ β_x(s) k1(s) ds
  Q'_y = +(1/4π) ∮ β_y(s) k1(s) ds
  ```
  Both come out **negative** for an ordinary FODO of pure quads (off-momentum
  particles are under-focused). For the FODO cell here `ξ/Q ≈ −1.0` per plane.
- **Thin vs thick.** Thin quads are exact single-point contributions — `β` is
  continuous across a thin kick, so `β·k1l` at the quad is exact. Thick quads are
  integrated by trapezoidal sub-slicing of `β` across the body (`slices=64`
  default): the β-at-the-quad point value is *not* exact when `β` varies over the
  magnet length. Keep the analytic closed-form on thin quads; the thick path is
  cross-checked against xtrack.
- **Scope: quadrupole gradients only.** Drifts contribute nothing; dipole
  weak-focusing / edge chromaticity is **not** computed (flagged — a lattice with
  bends carries an extra, uncomputed dipole term). The Stage 2 FODO acceptance
  lattice is quads + drifts, so this is exact there.
- **Independent validation.** The coefficient and per-plane sign are pinned to
  **machine precision** by differentiating the `δ`-dependent thin one-turn map
  symbolically (`cos μ(δ) = ½ Tr M(δ)`, `Q = μ/2π`, `dQ/dδ|₀`) — a check that
  never touches `β` or `4π`, so it is not circular with the β-sum
  (`tests/analytic/test_chromaticity.py`). The thick β-integration path matches a
  finite-difference tune derivative (always-on) and xtrack's real-particle
  tracking to `rel ≈ 1e-4`.

## Sextupole (Stage 2 — implemented)

A normal sextupole (`Sextupole(length, k2)`, thin `ThinSextupole(k2l)`) applies
the nonlinear kick

```
Δpx = −½ k2l (x² − y²),     Δpy = +k2l (x·y),
```

with `k2 = (1/Bρ)(∂²B_y/∂x²)` [m⁻³] (MAD-X / Xsuite convention) and integrated
strength `k2l = k2·L` [m⁻²]. Conventions:

- **Linear map is a drift.** The Jacobian of the kick at the closed orbit
  `(x, y) = 0` is the identity, so `Sextupole.matrix()` is a drift of length `L`
  (incl. the longitudinal slip `R56 = L/γ₀²`) and `ThinSextupole.matrix()` is the
  identity. A sextupole therefore leaves `β`, dispersion, and the tunes of the
  linear lattice **unchanged** (asserted to `rel 1e-14`). The full nonlinear kick
  (amplitude-dependent tune, dynamic aperture) is **out of Stage 2 scope** — no
  nonlinear tracking map is implemented.
- **Chromaticity feed-down** is the Stage-2 "linear effect." At dispersion
  `x = x_β + D_x·δ`, the quadratic kick yields a `δ`-dependent linear gradient
  `k1_eff = k2·D_x·δ`, shifting the chromaticity by
  ```
  Q'_x += +(1/4π) ∮ β_x k2 D_x ds
  Q'_y += −(1/4π) ∮ β_y k2 D_x ds
  ```
  The per-plane signs are **opposite to the quad** natural term (`+k2·D_x` vs
  `−k1`), which is exactly what lets a sextupole at `D_x > 0` push a negative
  natural chromaticity back toward zero. Vanishes on a dispersion-free (drift +
  quad) lattice.
- **`natural_chromaticity` vs `chromaticity`.** `natural_chromaticity` keeps its
  term-of-art meaning — the **bare quad-gradient** chromaticity (the negative
  number sextupoles correct); since a sextupole's map is a drift it contributes
  zero there, untouched. `chromaticity(lattice)` = `natural_chromaticity` + the
  sextupole feed-down. **Neither is a complete absolute total:** both omit the
  dipole's own weak-focusing / edge chromaticity (out of scope), and feed-down is
  nonzero only when bends are present — so an uncomputed dipole term always
  coexists with it. The validated deliverables are the *feed-down term itself*,
  the accsim-internal *correction* (feed-down cancels the quad natural term), and
  the *difference* cross-check below.
- **Thin vs thick.** Thin sextupoles are exact single-point contributions (`β` and
  `D_x` continuous across the zero-length kick); thick sextupoles integrate
  `β·D_x` by trapezoidal sub-slicing across the drift-like body (`slices=64`), which
  converges to the thin value quadratically in the length.
- **Independent validation.** The coefficient and per-plane sign are pinned to
  **machine precision** by the symbolic `δ`-dependent trace derivative — modelling
  the sextupole as the thin quad `k1l_eff = k2l·D_x·δ`, never touching `β` or `4π`
  (`tests/analytic/test_sextupole.py`). That check shares the feed-down *model*
  (sextupole ≡ extra quad) with the formula, so the **xtrack cross-check** is the
  one that validates the model itself: it tracks the real nonlinear kick and
  compares the **with-minus-without-sextupole difference** (toggling `k2` at fixed
  geometry, so `β`/dispersion/tunes — hence the shared dipole term — cancel
  exactly). accsim's feed-down matches xtrack's `Δdqx`/`Δdqy` to `rel ≈ 2e-3`
  (`tests/reference/test_sextupole_xtrack.py`).

## Stability boundary (Stage 2 — validated)

A transverse plane is stable iff its one-turn 2×2 block obeys `|½·Tr| < 1`
(`|Tr M| < 2`); an unstable plane has no real matched `β` and `match_periodic`/
`closed_twiss` raise `UnstableLatticeError` (see *Twiss* above). Stage 2's
acceptance ties this trace test to the analytic **phase-advance limit**:

- For the symmetric thin FODO (full-quad focal length `f`, half-cell drift `L`),
  `cos μ = 1 − L²/(2f²)`. The upper edge `cos μ = +1` is just the no-focusing
  `f → ∞` limit, so the *only reachable* instability is the over-focusing edge
  `cos μ = −1`, at `f_crit = L/2`, where the phase advance per cell reaches
  `μ = π`. A symmetric FODO therefore has **one** boundary, not two, and both
  planes hit it together (`μ_x = μ_y`).
- **Anti-circularity:** `is_stable` *is* `|½·Tr| < 1`, so `f_crit` is derived
  **symbolically** from `Tr M = −2` (hand-built thin matrices, no accsim) and the
  element chain must reproduce it: `½·Tr → −1` in both planes at `f_crit`,
  `is_stable` flips across it, the stable region matches the hand criterion
  `sin(μ/2) = L/(2f) < 1` over a focal-length sweep, and the **independent**
  `tunes()` atan2 accumulation sends `Q → ½` (μ → π) as `f → f_crit⁺`. Pinned by
  `tests/analytic/test_stability_boundary.py`.
- **Caveat (parametrising by target μ):** `f = L/(2 sin(μ/2))` maps `μ` and
  `2π − μ` to the *same* `f`, so it only reaches the stable range `(0, π)` — the
  unstable side is reached by lowering `f` below `f_crit`, never by pushing a
  target μ past π. Also `β_max ∝ 1/sin μ` diverges at the boundary, so μ-target
  checks stay off it (μ ≈ 0.9π).

## Beam envelope / beam size (Stage 2 — implemented)

The 1-σ transverse beam envelope adds the betatron width and the momentum-spread
offset **in quadrature** — they are statistically independent in a matched beam,
so there is no cross term and no coefficient to remember:

    σ_u(s) = √( ε_u · β_u(s) + (D_u(s) · σ_δ)² ),   u ∈ {x, y}.

- `ε_x`, `ε_y` are **geometric** (not normalised) emittances [m·rad]; `σ_δ` is the
  RMS relative *momentum* spread `σ(δ)` (dimensionless, same `δ` as the state
  vector). All three are **inputs**, not computed — there is no radiation/RF yet to
  set an equilibrium (that arrives in Stages 3/5). `σ_δ = 0` gives the pure
  betatron envelope `√(ε_u β_u)`.
- Each plane uses **its own** dispersion `D_u`, so vertical dispersion is included
  for free if a lattice ever produces it; a flat, uncoupled lattice has `D_y = 0`,
  so `σ_y` is betatron-only there.
- Units check: `D_u` [m], `σ_δ` dimensionless, `ε_u·β_u` [m·rad] ≈ [m] → `σ_u` [m].
- The physics lives in `accsim.beam_sigma` (testable); `plotting.plot_beam_envelope`
  and the `emittance=` branch of `plot_beta_functions` (betatron-only, `σ_δ = 0`)
  both call it — there is deliberately **one** σ formula in the codebase.
- **Validation:** the discriminating check needs dispersion, so it runs on an arc
  cell *with a dipole* (`D_x ≠ 0`) and asserts the exact decomposition
  `σ_x² − ε_x β_x == (D_x σ_δ)²` at every point, plus `σ → √(εβ)` when `σ_δ = 0`
  (`tests/analytic/test_beam_envelope.py`). **No xtrack cross-check** is warranted:
  the envelope is pure algebra over `β` and `D`, both already xtrack-validated in
  Stage 1; the analytic quadrature test covers the only new thing.

## Momentum compaction / slip factor (Stage 3 — implemented)

The momentum-compaction factor is the fractional circumference change per unit
momentum deviation — a purely **geometric** quantity (no `γ₀`):

    α_c = (1/C) ∮ D_x(s) · h(s) ds,    h = 1/ρ,   C = circumference.

- Only **bending magnets** contribute (`h = 0` in drifts, quads, sextupoles), so a
  straight / dispersion-free lattice has `α_c = 0`. Sign: outward dispersion in a
  normal focusing arc ⇒ the higher-momentum orbit is longer ⇒ `α_c > 0`.
- `accsim.momentum_compaction` computes the integral directly: it transports the
  matched dispersion along the lattice and, inside each thick dipole, integrates
  `D_x(s)` by trapezoidal sub-slicing of the sector sub-bend map (`h` constant
  across a body) — the same idiom as `natural_chromaticity`.
- **Phase-slip factor** `η = α_c − 1/γ₀²` (`accsim.slip_factor`). The `1/γ₀²` is
  taken from the reference particle — the *same single source* as the drift/dipole
  `R56 = L/γ₀²` (see [Drift](#drift-transfer-matrix-derived-not-remembered)); do
  **not** independently write `1/(β₀²γ₀²)`. `η` sets the sign of the longitudinal
  restoring force and vanishes at transition (`γ₀ = 1/√α_c`); Stage 3's synchrotron
  tune `Qs` is built on it. Sign convention matches xtrack's `slip_factor`.
- **Validation.** CI runs only the analytic suite, so it must catch a sign flip on
  its own. The load-bearing analytic net is the **symplecticity identity**

      α_c = 1/γ₀² − (R51·D_x + R52·D_px + R56) / C

  evaluated on the matched dispersion orbit from the **one-turn longitudinal row**
  (`R51/R52/R56`, Stage-1 xtrack-pinned) — a *different* set of matrix entries than
  the dispersion-generating ones the integral uses, so a sign error in the integral
  makes it fail (the RHS never touches the integral). The drift limit (`D=0`,
  `R56=C/γ₀²` ⇒ `α_c=0`) anchors the `1/γ₀²` term but can't test sign (both sides
  zero) — the bending cases do. A sympy re-derivation proves the integral path and
  the identity path are **algebraically identical** on a thick-dipole arc cell (so
  the `1/γ₀²` cancels, confirming `α_c` is γ₀-free), and — because the identity is a
  symplecticity *consequence*, not independent physics — the absolute value is
  anchored externally by an **xtrack cross-check** of both `momentum_compaction_factor`
  and `slip_factor` (~1e-6). See `tests/analytic/test_momentum_compaction.py` and
  `tests/reference/test_momentum_compaction_xtrack.py`.

## RF cavity / synchrotron tune (Stage 3 — implemented)

`RFCavity(voltage, frequency, phi_s)` is a **thin** longitudinal kick. In the
momentum variable `delta` the (nonlinear) kick is

    Δδ = (q V / (β₀² E₀)) · [ sin(φ_s − k_rf·zeta) − sin(φ_s) ],
    k_rf = 2π·frequency / (β₀ c)   [1/m],   φ_s  [rad].

- **Energy factor is `β₀² E₀`, not `E₀`.** With the *momentum* variable,
  `dE = β₀² E₀ · δ` at the reference, so `Δδ = ΔE/(β₀² E₀)` — the same `β₀²` that
  separates `R56 = L/γ₀²` (momentum) from `L/(β₀²γ₀²)` (energy). `V` in volts, `E₀`
  in eV, `q = ref.charge` (e-units) ⇒ `qV` in eV, ratio dimensionless.
- **Phase convention matches xtrack's `Cavity` exactly:** xtrack applies
  `energy_kick = qV·sin(lag_rad − (2πf/c)·zeta/β₀)`, i.e. `φ = φ_s − k_rf·zeta`
  with accsim's `φ_s` = xtrack's `lag` (xtrack in **degrees**, accsim in
  **radians** — pass `lag = degrees(φ_s)` when cross-checking). Verified: accsim's
  full 6×6 one-turn map equals xtrack's on the `(zeta, delta)` block, so the
  coupled synchrotron eigen-tune matches `tw.qs` to ~1e-6.
- **Linear map** (`RFCavity.matrix`) is the small-amplitude shear
  `R65 = ∂δ/∂zeta|₀ = −(q V k_rf cos φ_s)/(β₀² E₀)` (only `M[DELTA, ZETA]`); it is
  symplectic (a shear, det = 1). The full `sin` kick (`energy_kick_delta`) is the
  tracking map (the pendulum whose separatrix is the bucket) — Stage-3 nonlinear
  tracking. **Stationary bucket only**: `φ_s = 0` below transition, `φ_s = π` above;
  the accelerating `qV·sin(φ_s)` energy gain per turn is **Stage 5**.
- **Synchrotron tune** `synchrotron_tune(lattice)` builds the reduced one-turn 2×2
  `M_s = [[1,0],[R65_tot,1]] · [[1,−ηC],[0,1]]` and returns
  `Qs = arccos(½ Tr M_s)/2π`, reproducing the closed form
  `Qs² = −(h η qV cos φ_s)/(2π β₀² E₀)` (`k_rf C = 2π h`) — derived symbolically in
  `tests/analytic/test_synchrotron_tune.py`, no remembered constant.
- **The slip comes from `slip_factor()` (η), NOT the bare one-turn `R56`.** On a
  dispersive ring the raw `(zeta, delta)` block's `R56` entry is *not* `−ηC` — it
  omits the `R51 D_x + R52 D_px` dispersion coupling, and can even have the opposite
  sign (on the Stage-3 test ring the bare block is itself *unstable*). Sourcing the
  arc drift from `η` folds that coupling in; this is what makes `Qs` correct with
  bends present. Stability requires `Qs²>0` ⇒ `−η cos φ_s > 0`, which selects
  `φ_s = 0`/`π` below/above transition; the wrong side raises
  `UnstableLatticeError`.
- **Lumped ≠ exact.** The reduced-2×2 `Qs` is the textbook small-amplitude
  *formula*; it omits second-order synchro-betatron coupling that the full 6D map
  carries (sub-percent on the test ring). accsim's own 6×6 eigen-tune matches
  `tw.qs` to ~1e-6; the lumped value is validated against the symbolic closed form
  and cross-checked to xtrack at the coupling order
  (`tests/reference/test_synchrotron_tune_xtrack.py`).

## RF bucket / nonlinear longitudinal tracking (Stage 3 — implemented)

The synchrotron *tune* is linear, but the RF *bucket* is nonlinear (the cavity
keeps its full `sin`). The one-turn longitudinal map is the pendulum / standard
map — a kick-drift pair, each a symplectic shear:

    zeta  ← zeta − ηC·delta                               (arc slip, from η)
    delta ← delta + (qV/β₀²E₀)[sin(φ_s − k_rf·zeta) − sin φ_s]   (cavity kick)

- **Nonlinear tracking seam.** `Element.track(state, ref)` maps one 6D state;
  default is the linear `matrix(ref) @ state` (so element-by-element tracking of a
  linear lattice equals the one-turn matrix). `RFCavity.track` overrides it with
  the exact `sin` kick (`energy_kick_delta`). `Tracker.track` / `track_turns` take
  `nonlinear=True` to push element-by-element. The kick + linear drift is
  symplectic, so a bounded orbit conserves the Hamiltonian below (bounded ripple,
  **no** secular drift over ≥1e4 turns — the longitudinal symplecticity smoke test,
  the analogue of the transverse action-conservation run).
- **Synchrotron Hamiltonian** (`longitudinal_hamiltonian(lattice)` → callable),
  the smooth-approximation invariant:

      H(zeta, delta) = −½ηC·delta² + U(zeta),
      U(zeta) = −(qV/β₀²E₀)[(1/k_rf) cos(φ_s − k_rf·zeta) − zeta·sin φ_s],

  with `dzeta/dn = ∂H/∂delta`, `ddelta/dn = −∂H/∂zeta`. Stable fixed point at the
  synchronous particle `(0,0)`; unstable fixed point at `k_rf·zeta_u = 2φ_s − π`.
- **Separatrix** (`separatrix(lattice)`): the level set `H = H(zeta_u, 0)`. Inside
  ⇒ libration (bounded `zeta` **and** `delta`); outside ⇒ rotation — `delta` stays
  bounded but **`zeta` runs away without bound** (the discriminator for the
  ≥1e4-turn bounded test is unbounded `zeta`, *not* `delta`).
- **Bucket height** (`rf_bucket_height(lattice)`): max `|delta|` on the separatrix
  (at the centre `zeta=0`), `δ_max² = 2[U(0) − U(zeta_u)]/(ηC)`, which for a
  stationary bucket reduces to the closed forms

      δ_max = 2 Q_s / (h|η|) = √( 2qV / (π h |η| β₀² E₀) ).

  Both are **derived symbolically** from `H` (no remembered coefficient) and pinned
  in `tests/analytic/test_rf_bucket.py`.
- **Reduced ⇒ needs no dispersion.** `H`/separatrix/bucket use the *reduced*
  longitudinal dynamics (arc slip via `η`). They are exact when there is no
  dispersion coupling; the bounded/unbounded tracking test therefore runs on a
  **bend-free** ring (`α_c = 0`, `η = −1/γ₀²`, below transition, `φ_s = 0`) so the
  separatrix is crisp. With bends the reduced model is the standard leading-order
  approximation (the sub-percent synchro-betatron coupling seen in `Qs`).
- **Stationary bucket only** (`φ_s = 0`/`π` below/above transition). The
  accelerating moving bucket (`sin φ_s ≠ 0`) and the `qV sin φ_s` energy gain are
  **Stage 5**. `rf_bucket_height`/`separatrix` assume a single RF harmonic
  (cavities may share `frequency`/`φ_s`, summing voltage); double-RF raises.

## Acceleration / energy ramp (Stage 5 — implemented)

Turning the RF ramp on. The Stage-3 cavity kick was already the accelerating kick —
the ``- sin(phi_s)`` term is the energy the **reference** absorbs each turn, so a
synchronous particle (``zeta = 0``) gets zero net ``Delta delta`` and stays at
``delta = 0`` by construction. Stage 5 adds the reference-energy program and the
adiabatic damping that must accompany it (`accsim.accelerate`).

- **Energy gain per turn** ``Delta E_s = sum_cav q V sin(phi_s)`` [eV]
  (``accsim.energy_gain_per_turn``). ``q = ref.charge`` (e-units), ``V`` in volts ⇒
  ``qV`` in eV. Summed over **all** cavities (multi-cavity support), so a stationary
  bucket (``phi_s = 0``/``pi``) gives zero — recovering Stage 3. This is the Stage-5
  acceptance quantity; it is asserted both as this closed form *and* as the actual
  constant first difference of the reference-energy program.
- **The reference ramps; the lattice's ``ref`` does not mutate.** ``accelerate``
  builds a fresh immutable :class:`ReferenceParticle` each turn from
  ``E0(n) = E0(0) + n Delta E_s`` and tracks that turn's arc at the **turn-entry**
  reference. Because the beam energy is constant around the ring *except* across the
  cavity, this is exact when the cavity is the last element (the standard ring), and
  correct to ``O(Delta E_s/E0)`` per turn otherwise — negligible (keV on GeV).
- **Adiabatic damping factor is ``r = P0/P0'``, derived — not remembered.** With
  ``px = Px/P0`` and ``delta = (P - P0)/P0``: after the cavity (at fixed ``P0``) the
  particle is at ``P = P0(1 + delta_A)``; re-referencing to ``P0' = P0 + Delta P_s``
  gives ``delta' = (P0/P0')(1 + delta_A) - 1 = (P0/P0')·(delta + A[sin phi - sin
  phi_s])`` because ``A sin phi_s = Delta P_s/P0`` cancels the reference-gain terms.
  The **physical** ``Px, Py`` are untouched by the longitudinal kick, so
  ``px' = Px/P0' = (P0/P0') px`` (and ``py``). Hence one factor ``r = P0(n)/P0(n+1)``
  multiplies ``(px, py, delta)`` once per turn; ``r = 1`` at zero gain, so
  ``accelerate`` reduces to Stage-3 nonlinear tracking **bit-for-bit**.
- **The only approximation (flagged): the linearized energy→momentum conversion.**
  The cavity kick converts an energy gain into ``Delta delta`` via the linear
  coefficient ``A = qV/(beta0^2 E0)`` (i.e. ``delta P/P = A sin phi`` — first order in
  ``delta`` and in ``qV/E0``), **inherited unchanged from Stage 3**, not introduced by
  Stage 5. The ``-sin phi_s`` kick term and the ``(r - 1)`` in the re-referencing are
  the *same* reference-bookkeeping term done once: ``delta *= r`` is the **exact**
  partner of that kick (re-referencing the honestly-kicked coordinate
  ``delta + A sin phi``), **not** a second approximation. Consequences: the
  synchronous particle is exact to **all** orders (both the code and an
  exact-momentum bookkeeping give ``delta = 0``); off-momentum particles carry the
  ``O(delta^2, (qV/E0)^2)`` residual of the Stage-3 thin kick. (Correspondingly
  ``A sin phi_s = Delta P_s/P0`` holds only to first order in ``Delta E_s/E0`` — the
  code uses the *exact* ``r`` from ``from_total_energy`` with the *linear* ``A``, and
  that tiny mismatch is part of the same first-order residual.)
- **Position ``(x, y, zeta)`` is NOT rescaled** at the thin cavity — it is a spatial
  coordinate, not normalised by ``P0``. The betatron/synchrotron motion converts the
  momentum damping into overall amplitude damping over a period, conserving the
  **adiabatic invariant** ``P0·J`` (canonical action). For a drift+cavity ring the
  transverse momentum telescopes to the exact closed form ``px[n] = px0·P0(0)/P0(n)``
  (pinned to ``rel 1e-12``).
- **Assert the invariant, not the raw amplitude.** During the ramp the geometric
  action/emittance genuinely **shrinks** — this *is* adiabatic damping, **not** a
  symplecticity violation, so the Stage-3 raw-action smoke test does not carry over.
  A neighbour's synchrotron oscillation damps in amplitude while the action
  ``≈ delta_max^2 / Qs`` (area ~ amplitude²/frequency) is conserved (tested to a few
  % window ripple over a 40%-energy ramp).
- **Stable synchronous phase** (``accsim.synchronous_phase``): inverts
  ``Delta E_s = qV sin phi_s`` for the root satisfying **both** net gain
  (``sin phi_s > 0``) and small-amplitude stability ``Qs^2 = -(h eta qV cos
  phi_s)/(2 pi beta0^2 E0) > 0`` ⇒ ``eta cos phi_s < 0``. So ``phi_s ∈ (0, pi/2)``
  below transition, ``(pi/2, pi)`` above — derived from **accsim's own** kick
  convention (``phi = phi_s - k_rf zeta``), reducing to the Stage-3 stationary
  ``0``/``pi`` at zero gain. ``eta``'s sign is a lattice property independent of the
  cavity phase, so it can be evaluated before the cavities are built.
- **Harmonic-number interface** ``RFCavity.from_harmonic(voltage, harmonic,
  circumference, ref, phi_s)`` sets ``frequency = harmonic·beta0·c/C`` so
  ``k_rf·C = 2 pi h`` exactly; ``harmonic_number(ref, C)`` inverts it. ``frequency``
  remains the stored canonical field (it is what enters ``k_rf``); the harmonic ctor
  is the natural ring interface where ``h`` is the design integer.
- **Moving-bucket guard.** The Stage-3 ``rf_bucket_height``/``separatrix``/
  ``longitudinal_hamiltonian`` assume a *stationary* bucket (fixed points symmetric
  about ``zeta = 0``); for ``sin phi_s != 0`` they now **raise**
  ``NotImplementedError`` rather than return a plausible-wrong stationary curve. The
  guard keys on ``|sin phi_s| > 1e-9``, so ``phi_s ∈ {0, pi}`` (``sin ~ 0``) stays
  valid. The moving-bucket *acceptance* is out of scope.
- **Scope.** Constant magnetic optics (``k1``/bend angles held fixed = magnets ramp
  with the beam — the physical "tracking" ramp), so the transverse Twiss is
  energy-invariant. Beam loading, higher-order modes, wakefields, and transition
  crossing are **out of scope**. No xtrack cross-check is warranted (derived closed
  forms over Stage-1/3-validated maps — the Stage-2 beam-envelope rationale).

## Beam losses / apertures (Stage 4 — implemented)

Geometric transverse acceptance with survival/loss accounting.

- **`Aperture(shape, half_x, half_y=None, length=0.0)`** — an **optics-transparent**
  element: `matrix()` is the identity, so inserting one never perturbs Twiss,
  tunes, dispersion, or the one-turn map. Its physics is a *predicate*,
  `survives(states)`, on the transverse `(x, y)`:
  - `"circular"` (radius `R = half_x`): `x² + y² ≤ R²`;
  - `"elliptical"`: `(x/half_x)² + (y/half_y)² ≤ 1`;
  - `"rectangular"`: `|x| ≤ half_x` **and** `|y| ≤ half_y`.
  Centred on the reference orbit. **Boundary convention:** on-boundary **survives**
  (inclusive `≤`), matching xtrack `LimitRect`/`LimitEllipse`; tests stay off the
  knife-edge. `survives` is vectorised: `(6,)→bool`, `(6,N)→(N,)`.
- **`Collimator`** — the same geometric test with finite `length` (default 1 mm)
  and a label. **Approximation (flagged):** survival is checked at the element
  only, not continuously along the jaw, so a particle whose transverse excursion
  *peaks inside* a finite jaw and returns within the aperture at the exit is not
  caught. Negligible for pencil-thin collimators; costs accuracy only for long
  jaws with large local betatron slope.
- **Loss accounting is separate from the element.** `Tracker.track_bunch_losses(
  bunch, n_turns)` walks the lattice element-by-element (linear optics),
  accumulating the **geometric** `s`. At each aperture the surviving particles are
  tested; a failure is recorded and the particle is **frozen** (state stops
  advancing) and skipped on all later elements/turns. Keeping the aperture in the
  element sequence is what makes its `s` well-defined. Returns a `LossResult`:
  `alive` mask, `loss_turn`, `loss_s` (the aperture's geometric `s` in `[0, C)`,
  **not** the particle's `zeta`), `loss_element`, plus `transmission` and
  `loss_map()` (counts by location, summed over turns).
- **Transmission closed forms** (`tests/analytic/test_beam_losses.py`). Round
  Gaussian beam through a *circular* aperture radius `R`: survival
  `T = 1 − exp(−R²/2σ²)` (Rayleigh radial CDF, **sympy-proven**) — valid **only**
  for `σ_x = σ_y` + circular. Independent separable case (different shape):
  rectangular acceptance, `T = erf(a_x/√2σ_x)·erf(a_y/√2σ_y)`. Both compared to the
  empirical survival with a **binomial** tolerance `√(T(1−T)/N)`, not a tuned
  number.

## Quantum lifetime (Stage 4 — implemented)

Aperture-limited lifetime `quantum_lifetime(aperture, sigma, amplitude_damping_time)`.
**Derived, not remembered** (`tests/analytic/test_quantum_lifetime.py`): with the
normalized action `w = a²/2σ²` the radiation-damped, quantum-excited betatron
distribution has equilibrium `e^{-w}`; the amplitude-diffusion Fokker–Planck
mean-first-passage time from the core to an aperture at `w = ξ = A²/2σ²` is
exactly `τ_q = (τ_d/2)∫₀^ξ (e^w−1)/w dw`, whose `ξ≫1` asymptote is the standard

    τ_q = τ_d · e^ξ / (2ξ),    ξ = A²/2σ².

The MFPT solution is verified against its backward equation symbolically (residual
`= −1`) and the closed form matches the exact integral to `O(1/ξ)` (error halves as
`ξ` doubles). **Factor-of-2 convention:** `τ_d` is the **amplitude** damping time
(amplitude `∝ e^{−t/τ_d}`); the emittance damps twice as fast (`τ_ε = τ_d/2`), so if
you hold `τ_ε` pass `2·τ_ε`. `τ_d` was a caller input at Stage 4; **as of Stage 7 it
is computable from the lattice** — `radiation.damping_times(lattice)` returns exactly
this amplitude damping time (same convention, so they compose without a stray 2).
`ξ = A²/2σ²` shares its `·/2σ²` structure with the circular transmission formula
(same aperture-to-sigma ratio governs both).

## Synchrotron radiation / radiation damping (Stage 7 — implemented)

`src/accsim/radiation.py` (baseline core physics, **not** gated). Five lattice
integrals (Sands, SLAC-121) and the damping/equilibrium quantities they feed, in **SI**
(eV, m, s): so `C_γ` is in `m/eV³`, `U0` in eV, `C_q` in m.

- **Integrals** `radiation_integrals(lattice) → RadiationIntegrals(I1..I5)`:
  `I1 = ∮ D_x h ds` (`= α_c·C`), `I2 = ∮ h² ds`, `I3 = ∮ |h|³ ds`, `I4 = ∮ D_x h³ ds`,
  `I5 = ∮ curlyH |h|³ ds` with `curlyH = γ_x D_x² + 2α_x D_x D_x' + β_x D_x'²`. `h = 1/ρ`
  is signed; `I3`/`I5` use `|h|³` (excitation is bend-sign-blind), `I4` keeps `h³`'s
  sign. Reuses the thick-dipole dispersion sub-slicing of `momentum_compaction`; `I5`
  additionally **co-transports `β_x,α_x`** through the dipole body (the one bug-prone
  spot). Slice-converged (64 ≡ 1024 to 6 digits).
- **Pure-sector-bend `I4`/`I5`.** Dipoles are pure sector bends (no combined-function
  gradient, no pole-face edge — Stage-1 scope), so the `2k1` body term and the
  `−D_x h² tan(edge)` face term both vanish — the **textbook MAD-X/Sands** definition.
  `I4` is pinned within-baseline by the isomagnetic identity `I4 = h²·α_c·C`; `I5`
  (curly-H, on the new β/α co-transport) is pinned by an independent `propagate_twiss`
  integration to `1e-6`. So the integrals are validated. xtrack's `radiation_analysis`
  computes `ε_x`/partition numbers by a **different method** — the damped one-turn-map
  eigen/envelope analysis, **not** radiation integrals (it exposes none) — so it differs
  ~1% (partitions) / ~3-4% (`ε_x`) in this strong ring (`I4/I2≈0.38`, ~5× normal) while
  `I1`/`I2`/`U0` match to `1e-6`. Integral-formula vs exact-eigenanalysis, not a bug
  (`tests/reference/test_radiation_xtrack.py`).
- **Constants (species-general, from the reference particle):**
  `C_γ = 4π r0/(3(mc²)³)`, `C_q = (55/32√3)·ħc/(mc²)` with `ħc = 1.9732698045e-7 eV·m`.
  Electron: `8.846e-5 m/GeV³`, `3.832e-13 m` (pinned symbolic-rational + numeric).
- **Energy loss** `U0 = (C_γ/2π)E⁴ I2` [eV] (isomagnetic `= C_γ E⁴/ρ`, the 88.5 keV
  formula). **Partition numbers** `(J_x,J_y,J_z) = (1−I4/I2, 1, 2+I4/I2)`; Robinson
  `J_x+J_y+J_z = 4` is exact by construction — the structural gate. **Damping times**
  `τ_i = 2E·T0/(J_i U0)` [s], `T0 = C/(β0 c)` — the **amplitude** damping time (action/
  emittance damp at `τ_i/2`); matches Stage-4 `quantum_lifetime`'s input convention.
- **Equilibrium** `ε_x = C_q γ² I5/(J_x I2)` (**geometric** m·rad; ×β0γ0 for normalized)
  and `σ_δ = √(C_q γ² I3/(J_z I2))`. No clean absolute closed form for `ε_x` (curly-H),
  so its analytic gate is the **energy scaling** (`ε_x ∝ γ²`, `σ_δ ∝ γ` to machine
  precision — the integrals are pure geometry) + the xtrack absolute; stated as the
  gate, not a loosened tolerance (as with the Phase-2 A_FB magnitude).
- **Flat-lattice scope:** `J_y ≡ 1` and equilibrium `ε_y ≈ 0` (no vertical bending or
  betatron coupling — real rings set `ε_y` by coupling/vertical dispersion, out of
  scope).

## Luminosity (Stage 6 — implemented)

`luminosity(N1, N2, sigma_x, sigma_y, f_rev, n_bunches, crossing_angle=0,
sigma_z=0, crossing_plane="x")` returns the peak luminosity in **`m^-2 s^-1`**
(`accsim.collider`). Head-on, equal Gaussian beams:

    L = f_rev · n_bunches · N1 · N2 / (4 π σ_x σ_y).

- **The `4π` is *derived*, not remembered.** `L` = (bunch-collision rate) ×
  (transverse overlap `∮ ρ1 ρ2 d²r`); for two equal normalized 2D Gaussians the
  overlap is `1/(4π σ_x σ_y)` (sympy-proven in `test_luminosity.py`). The `4π`
  therefore **bakes in `σ_1 = σ_2`** per plane; the general two-size form replaces
  `σ_u → √((σ_{1u}² + σ_{2u}²)/2)` and reduces to `4π` when equal. Gaussian profile
  assumed.
- **Units traps (both pinned):**
  - *cm vs m.* `L` is `m^-2 s^-1` internally; textbooks quote `cm^-2 s^-1`
    (× `1e-4`). The classic 10⁴ error.
  - *geometric vs normalized emittance.* `σ_u* = √(ε_u β_u*)` needs the
    **geometric** ε; machines quote **normalized** `ε_n = β₀γ₀·ε` (the stray-γ
    trap — divide by `β₀γ₀`, not `γ₀`).
- **Crossing angle (Piwinski).** A full crossing angle `φ` reduces `L` by the
  multiplicative geometric factor
  `S = 1/√(1 + (σ_z·tan(φ/2)/σ_cross)²)` (`piwinski_reduction`), `σ_cross` the
  beam size in the crossing plane. **`tan(φ/2)`, not `tan φ`** — each beam tilts by
  half the full angle. `S → 1` head-on or for a point bunch. The **hourglass**
  effect (`β` varying across `σ_z` when `σ_z ≳ β*`) is a *separate* reduction and
  is **out of scope**.
- **Worked example (acceptance gate).** LHC nominal (LHC Design Report Vol I,
  Table 2.1: `N=1.15e11`, `n_b=2808`, `f_rev=11245 Hz`, `β*=0.55 m`,
  `ε_n=3.75 µm`, 7 TeV/beam) gives head-on **`1.20e34 cm^-2 s^-1`**, and with the
  nominal 285 µrad crossing / 7.55 cm bunch the Piwinski `S≈0.84` brings it to the
  design peak **`1.0e34 cm^-2 s^-1`** (`tests/analytic/test_luminosity.py`). No
  xtrack cross-check is warranted — a closed-form overlap integral, validated
  symbolically and against a published machine.
- **Low-β insertion / classical radius.** The IP low-β *optics* need no new code:
  the waist `β(s) = β* + s²/β*` is exactly what the Stage-1 drift Twiss
  propagation already produces around a zero-`α` point. The classical particle
  radius `r0 = r_e·(m_e c²/m c²)·q²` (`ReferenceParticle.classical_radius_m`,
  `r_e = ELECTRON_RADIUS_M`) is added for the Stage-6 beam-beam kick / tune shift.

## Weak-strong beam-beam kick (Stage 6 — implemented)

`BeamBeam(n_particles, sigma, strong_charge=1.0)` (`accsim.elements.beambeam`) is
a **thin** head-on kick from a **round** Gaussian strong bunch (weak-strong: the
strong bunch is rigid). Per plane, regularised on the axis:

    Delta px = K x g(u),   Delta py = K y g(u),
    u = (x^2+y^2)/(2 sigma^2),   g(u) = (1 - e^{-u})/u  (-> 1 as u -> 0),
    K = (q2/q1) N r0 / (gamma sigma^2)   [1/m].

- **`r0` is the *test* particle's classical radius** (`ref.classical_radius_m`),
  `gamma`/`q1` its Lorentz factor / charge; `N`/`q2` are the strong bunch's
  population / charge. `g(u)` is coded as `-expm1(-u)/u` so the axis is
  singularity-free (the `1/r^2` in the textbook form cancels).
- **Sign is *derived* from the Lorentz force, not remembered.** `E` and `B` add for
  counter-propagating beams (the `2N` — note `K` above already folds the `2` into
  the `1/(2 sigma^2)` of `u`, so the small-`u` slope is `K`, see below). Like
  charges (`q1 q2 > 0`, pp) **repel → defocus** (`K > 0`, `Delta px` has the sign of
  `x`); opposite charges (`e+ e-`, p-pbar) **attract → focus** (`K < 0`). The
  historical `-(2 N r0/gamma)(1/r)(...)` textbook form is the *opposite-charge*
  case; the signed `q2/q1` reproduces both.
- **Invariants (gate 3 — "conserves the expected invariants").** The kick derives
  from a potential ⇒ **curl-free** `∂Δpx/∂y = ∂Δpy/∂x` (the property that keeps
  long-term tracking symplectic; `is_symplectic` is **linear-only** so it is *not*
  the right check for the nonlinear kick — use the Jacobian or this curl identity).
  Being radial it exerts **no torque**, so the transverse angular momentum
  `L_z = x py - y px` is **exactly** conserved (positions untouched by the thin
  kick). Both hold **only for the round beam**.
- **Linear map** (`matrix`) is the `u → 0` limit `px → px + K x`, `py → py + K y` —
  a thin lens focusing **both** planes **equally** (round symmetry), unlike a
  quadrupole (opposite signs). Effective thin-quad strength `k1l = -K`, same in
  both planes. This `K` is what the Stage-6 beam-beam tune shift `ξ` is built on
  (its small-amplitude limit). Cross-checked against an independent bare-`1/r`
  closed form (`tests/analytic/test_beam_beam.py`).
- **Elliptical Bassetti–Erskine (`scipy.special.wofz`) is out of scope** — optional
  generality not needed for the gate, and it breaks the `L_z` conservation the
  round beam enjoys. Hourglass / crossing-angle geometry in the kick is likewise
  out of scope (the crossing angle enters *luminosity* only).

## Beam-beam tune shift ξ (Stage 6 — implemented)

`beam_beam_tune_shift(beambeam, ref, beta_x, beta_y=None)` (`accsim.collider`)
returns the **signed** small-amplitude tune shift `(ΔQx, ΔQy)` the head-on
beam-beam kick produces at an IP with beta functions `β_x, β_y` (`β_y` defaults to
`β_x`, round IP). It is the **small-amplitude limit of the [BeamBeam
kick](#weak-strong-beam-beam-kick-stage-6--implemented)**, not a standalone
remembered formula:

    ΔQ_u = -β_u K/(4π),   K = (q2/q1) N r0/(γ σ²)   ⇒   |ΔQ_u| = ξ_u.

- **Coefficient `β/(4π)` is derived, not remembered.** A thin lens
  `[[1,0],[−k1l,1]]` composed with a Courant-Snyder rotation `R(μ;β,α)` has
  `½Tr = cos μ − k1l·β·sin μ/2`, so `dμ/dk1l = β/2` (implicit differentiation, no
  `Abs`) and `dQ/dk1l = β/(4π)` (sympy, `test_beam_beam_tune_shift.py`). The
  beam-beam linear part is `k1l = −K`, giving `ΔQ = −βK/(4π)`.
- **Sign (follows the kick's Lorentz-force sign).** Like charges (pp) defocus ⇒
  `K > 0` ⇒ `ΔQ < 0` (defocusing lowers the tune); opposite charges (e+e-, p-pbar)
  focus ⇒ `ΔQ > 0`. The **magnitude** is the conventional beam-beam parameter
  `ξ_u = N r0 β_u*/(4π γ σ²)` (round beam; the general elliptic form is
  `N r0 β_u*/(2πγ σ_u(σx+σy))`). LHC nominal → `ξ ≈ 0.0037` per IP.
- **First-order only.** Validated *through a real ring*: inserting the linearised
  `BeamBeam` into a FODO and reading `tunes()` (independent `atan2` accumulation)
  reproduces `−βK/(4π)` as `K → 0`, with the residual scaling **quadratically** in
  `K` (the O(ξ²) amplitude-detuning term the full nonlinear kick carries, out of
  scope here). No xtrack cross-check is warranted — a closed form derived over the
  Stage-1-validated Twiss/tune machinery and pinned by the through-ring measurement.

## Toy event generator (Phase 2 — implemented, learning module)

`accsim.events` is the **clearly-labelled learning module** the roadmap permits for
Phase 2: a from-scratch Monte-Carlo generator for `e+ e- → μ+ μ-` (tree-level QED,
s-channel photon). *Orchestrate, don't rebuild* still governs physics-grade work —
the toy is the analytically-gated half (clause a). The **real** orchestration
(clause b) is met separately by `pipelines/ee_mumu_pythia/` — Pythia8 8.3 in the
`hepstore/rivet-pythia` Docker image, driven end-to-end (`run_pipeline.py`) to a
labelled `cos θ` distribution; see that dir's README. Docker is used because
Pythia/Delphes don't build natively on Win/Py3.14 (no Windows pip/conda `pythia8`;
native-Windows pip finds no wheel), and a bind mount is avoided (spaced path) via
`docker cp`. The two halves are complementary: the toy is pinned to `4πα²/(3s)`;
the Pythia μ⁻ spectrum is compared to `1+cos²θ` only qualitatively (all-flavour σ
≈ 6.15 nb vs the toy's 0.87 nb, plus QED FSR / fixed √s). At 10 GeV the process is
γ\*-dominated, so the γ-Z forward-backward asymmetry is unresolved — *measured*
`A_FB = −0.0022 ± 0.0074` on 18k events (consistent with zero), so it is not
claimed as a visible distinguishing feature.

- **Natural units, local to the module.** `accsim.events` works in `ħ = c = 1`,
  GeV — the universal cross-section convention — *unlike* the SI/eV beam-dynamics
  core ([Units](#units)). The single boundary crossing back to lab units is the
  cross-section: **`1 GeV⁻² = 0.3893793721 mb = (ħc)²`** (`GEV2_TO_MBARN`), kept as
  one tested constant so the `0.389` factor is never sprinkled inline.
- **Metric.** Mostly-minus `(+,−,−,−)`; four-vectors are `(E, px, py, pz)` numpy
  arrays with energy in index 0, so `p·p = m²`.
- **Process picked by the acceptance gate.** `e+ e- → μ+ μ-` has the cleanest
  closed form and **no PDFs** (leptonic initial state), so the analytic gate is
  unmuddied. Massless limit (`√s ≫ m_μ`): `dσ/dΩ = α²(1+cos²θ)/(4s)`,
  `σ = 4πα²/(3s)` (**≈ 0.87 nb at √s = 10 GeV**). Spin-averaged
  `⟨|M|²⟩ = 32π²α²(t²+u²)/s² = 16π²α²(1+cos²θ)`. Hadronic Drell-Yan (needs LHAPDF)
  is a deliberately-deferred extension, not the first cut.
- **RAMBO (Kleiss-Stirling-Ellis 1986), massless.** Flat Lorentz-invariant phase
  space with a **constant** weight = the total volume, so `∫f dΦ ≈ volume·⟨f⟩`.
  Volume formula `Φ_n = (π/2)^{n-1} s^{n-2} (2π)^{4-3n} / (Γ(n)Γ(n-1))`; for `n=2`
  it is `1/(8π)` (s-independent), for `n=3` it is `s/(256π³)`.
- **Cross-section master formula.** `σ = (1/2s)∫⟨|M|²⟩dΦ₂ ≈ (weight/2s)⟨|M|²⟩`, flux
  factor `F = 2s` (massless). Result in GeV⁻²; `gev2_to_barn` converts.
- **Gate ordering guards against cancellation (advisor).** The three analytic gates
  run **phase-space volume → dσ/dΩ shape → total σ** so a wrong `|M|²` and a wrong
  phase-space measure cannot cancel into a right-looking σ. Gate 1 is validated
  *independently of any matrix element*: the `1/(8π)` volume is derived from the β
  factor (sympy), the general formula is checked against an independently-derived
  three-body `s/(256π³)` (phase-space convolution), and the sampler is verified to
  conserve four-momentum, stay massless, and fill 2-body phase space isotropically
  (`cos θ` uniform, mean 0 / var ⅓). Gate 3 (MC σ vs analytic within MC error) is
  the roadmap's Phase 2 acceptance clause. See `tests/analytic/test_toy_generator.py`.
- **Out of scope (labelled):** running coupling, initial-state radiation, `Z`
  interference/resonance, masses/thresholds, hadronic PDFs, higher orders, and the
  real Pythia→Delphes orchestration.

## Delphes detector step (Phase 2 — detector extension)

`pipelines/ee_mumu_delphes/` adds the canonical **fast detector simulation**
(Delphes) after the generator, so the deliverable is a **generator-level (truth) vs
detector-level (reco)** `cos θ` comparison — *what the detector does to the truth*.
Two **established** tools, coupled through a **HepMC3** file (the standard
generator→detector interchange): Pythia8 (`hepstore/rivet-pythia`) writes HepMC3 via
`Pythia8Plugins/HepMC3.h`; Delphes 3.5.0 + ROOT (`scailfin/delphes-python-centos:3.5.0`,
IRIS-HEP) runs `DelphesHepMC3` with the **ILD** card. We decouple through HepMC3
(rather than `DelphesPythia8`) because no trustworthy single image ships both tools and
`DelphesPythia8` needs Delphes compiled against this Pythia. Gated addon
(`ACCSIM_ENABLE_DELPHES` / `features.require("delphes")`); see the dir's README.

- **√s = 250 GeV, not the clause-(b) 10 GeV — a *card-validity* choice, not a whim.**
  Standard Delphes e+e- cards (ILD/IDEA/CLIC) are parametrized for **≥ 91 GeV**; at
  10 GeV *no* card is physically valid. 250 GeV (ILC) is the ILD card's designed range.
  Bonus: above the Z, γ*-Z interference makes the μ⁻ **forward-peaked** — a *measured*
  `A_FB ≈ +0.53` (contrast the 10 GeV chain's `A_FB ≈ 0`). The symmetric `1 + cos²θ`
  toy law does **not** hold here (it is the far-below-Z limit), so no `1+cos²θ` overlay.
- **`cos θ` conventions.** Truth from the generator `Particle` branch: `cos θ = pz/|p|`
  (`|p| = √(px²+py²+pz²)`). Reco from the `Muon` branch: `cos θ = tanh(η)` (Delphes
  stores pseudorapidity; `η = artanh cos θ`, exact for the ultra-relativistic 125 GeV
  muons). Both are produced by the **same** ROOT macro (`extract_reco.C`) from the
  **same** Delphes file, so truth and reco are one population up to detector response.
- **Signal isolation by an *angle-neutral* `|p| > 100 GeV` cut.** The
  `ffbar2ffbar(s:gmZ)` process sums all outgoing flavours, so the sample also makes
  μ from τ→μ and b/c decays. Two facts: (1) Pythia's hard-outgoing **status 23 is not
  preserved through the HepMC round-trip** (FSR → status 51/52 copies + a status-1
  final), so it cannot tag the signal in the Delphes record; (2) the signal μ⁻ is
  **monochromatic at |p| ≈ 125 GeV at every polar angle**, secondaries are soft — the
  status-1 μ⁻ `|p|` spectrum is bimodal (~125 GeV spike + soft tail) with a wide empty
  valley (≈ 60–110 GeV). So both truth and reco cut `|p| > 100 GeV` (`|p| = pT·cosh η`
  for reco). **`|p|` not `pT`** is the crux: the signal is 125 GeV at *all* `cos θ`, so
  the cut **cannot manufacture a forward edge** — the only edge is the detector's.
- **Validation — the detector must *remove* muons, and the acceptance edge is the
  proof.** The ILD card reconstructs muons at 95% efficiency for **|η| < 2.4**, zero
  beyond. So: **reco ⊆ truth** (never adds muons; a bug where reco > truth from τ→μ
  contamination was fixed by this design); `reco/truth = acceptance × ε ≈ 0.91`; **reco
  vanishes beyond `|cos θ| = tanh(2.4) = 0.984`** while truth extends to ±1 — that edge
  is the live-detector signature. Cross-check: the `|p|` cut yields `truth N ≈ 1908`,
  matching the generator's independent status-23 primary-μ⁻ count (`≈ 1956`) to ~2.5%,
  confirming the cut selects the signal. `A_FB` is preserved truth↔reco (forward-back
  symmetric acceptance). No analytic pin (a fast-sim response is not a closed form); the
  gates are the four above. See `pipelines/ee_mumu_delphes/README.md`.
- **Out of scope (labelled):** hadronic/PDF (LHAPDF Drell-Yan) extension; pile-up,
  beam backgrounds, jet/b-tag performance, and full ILD reco (Delphes features left
  unused — the deliverable is the muon channel truth-vs-reco).

## Drell-Yan hadronic step (Phase 2 — hadronic extension)

`pipelines/pp_mumu_drellyan/` is the **hadronic** analogue of the leptonic Delphes
chain: the same Pythia8 → **HepMC3** → Delphes → analysis orchestration, but with a
**real proton PDF (LHAPDF6)** in the initial state, so the partonic √ŝ is a
*distribution* — the point of "with real PDFs". Process `WeakSingleBoson:ffbar2gmZ`
(`q q̄ → γ*/Z → μ+μ-`, textbook Drell-Yan) at **√s = 13 TeV**, run through the Delphes
**CMS** hadron-collider card. Gated addon (`ACCSIM_ENABLE_LHAPDF` /
`features.require("lhapdf")`); see the dir's README.

- **Why the 2→1 resonant process works here (it did *not* leptonically).** The
  leptonic chains had to use the 2→2 continuum `ffbar2ffbar(s:gmZ)` because the 2→1
  resonant `ffbar2gmZ` *underflows to zero* at a fixed partonic √s below the Z (its
  Breit-Wigner integrates over a δ-function `mHat`). With protons the **PDFs spread
  the partonic mHat across a continuum**, so `ffbar2gmZ` is exactly the right tool —
  this is the concrete physics difference the PDFs make.
- **Real LO PDF, downloaded at run time.** Default `NNPDF31_lo_as_0118`, member 0
  (recorded in `meta.dat`). **LO** to match Pythia's LO matrix element. The image ships
  LHAPDF *without* grids, so `run_pipeline.py` runs `lhapdf get <set>` first (clean
  error on no network). ISR/FSR stay **on**; we do **not** set `PDF:lepton = off` (a
  lepton-beam toggle, irrelevant to protons).
- **Clean dimuon sample by forced decay — no `|p|` cut.** Because this is a *resonance*
  process we force `23:onMode=off; 23:onIfMatch=13 -13` (`Z→μμ`), so the only prompt
  muons *are* the signal pair — no τ→μ / heavy-flavour contamination, hence no
  monochromatic-`|p|` trick (which the leptonic Delphes chain needed). Both truth and
  reco take the **leading opposite-sign muon pair** (robust to >2 muons from FSR).
- **Deliverables = the Z peak in `m(μμ)` *and* `A_FB(m)`, truth vs reco.** The
  container macro `extract_kinematics.C` dumps the **μ⁻/μ⁺ four-vectors** per event
  (truth from the `Particle` branch `(Px,Py,Pz,E)`; reco from the `Muon` branch via
  `SetPtEtaPhiM(PT,Eta,Phi,m_μ)`), both from the **same** Delphes file, so one
  population up to detector response. *All* physics — `m(μμ)` and `cos θ*_CS` — is
  then computed on the host by the **single tested** `accsim.events.collins_soper_costheta`
  (see *Collins-Soper A_FB* below), so no sign-error-prone frame transform is
  duplicated in untested C++. μ⁻ is **PID +13** (mu+ = −13), carried through exactly.
- **The truth peak is *not* a clean Breit-Wigner.** FSR pulls `m(μμ)` below the pole →
  a **low-side radiative tail**, so the truth peak *mode* recovers `M_Z ≈ 91.19` only
  to ~1 GeV (a bin). Interpret mode, not a δ — this is physics, do not tighten to force
  a sharp `M_Z`.
- **The detector leaves two marks (this is a mass spectrum, so no acceptance *edge*).**
  (1) **reco ⊆ truth** — both muons must be reconstructed inside CMS acceptance, so
  `reco/truth = acceptance × ε² ≈ 0.36` (a 13 TeV Z is longitudinally boosted by the
  PDF asymmetry, pushing one muon forward of `|η|<2.4`); a detector never *adds* muons.
  (2) **reco peak broader than truth** — CMS momentum-resolution smearing (reco RMS >
  truth RMS), but **modest** (excellent CMS muon resolution at `pT≈45 GeV` adds sub-GeV
  on top of `Γ_Z≈2.49 GeV`).
- **The honest cross-check is σ, not the (semi-circular) peak position.**
  `σ(DY×BR(Z→μμ), 60<m<120) ≈ 1.5 nb` at 13 TeV, matching the measured LHC value
  (~1.9 nb NNLO per flavour; LO ÷ K≈1.25) — a *real global-fit PDF* convolved with the
  LO ME doing physical work. The magnitude also settles a convention: `sigmaGen()` here
  is production σ **times** BR (the μ-channel σ in the window), not the full production
  σ. No analytic pin (a fast-sim response is not a closed form).
- **`A_FB(m)` in the Collins-Soper frame — now measured (see *Collins-Soper A_FB*
  below).** The second deliverable of this chain. Out of scope remains: pile-up,
  NLO/NNLO + K-factors, PDF-uncertainty bands, jet/b-tag. See
  `pipelines/pp_mumu_drellyan/README.md`.

## Collins-Soper A_FB (Phase 2 — Drell-Yan angular observable)

The forward-backward asymmetry `A_FB(m)` of the Drell-Yan chain, the classic
γ*/Z-interference signature, measured in the **Collins-Soper (CS) frame**. All the
frame physics lives in **one tested function**,
`accsim.events.collins_soper_costheta` (pure numpy, always-on baseline); the gated
pipeline and the container macro only feed it four-vectors.

- **The closed form (massless-lepton).** For `ℓ⁻` (particle 1) and `ℓ⁺` with beams
  along `±ẑ`, `cos θ*_CS = 2(p⁻_z E⁺ − E⁻ p⁺_z) / (m_ℓℓ √(m_ℓℓ² + Q_T²))`. This is
  the CS bisector-axis projection; the `2/(Q√(Q²+Q_T²))` coefficient is **derived,
  not memorised** — pinned by equality to an independent boost-into-rest-frame
  bisector construction over 3000 random pairs (`tests/analytic/test_collins_soper.py`),
  plus hand orientation configs (`cos θ* = ±1`). It is the standard **massless-lepton**
  form; at the real muon mass vs ~45 GeV Z-decay momentum it is off by ~1e-6 (`β_μ`),
  negligible, and is what every DY experiment uses.
- **`μ⁻` is PID +13** (μ⁺ = −13); carried through `generate_hepmc.cc`,
  `extract_kinematics.C`, and `analyze.py` identically — **one flip inverts `A_FB`**.
- **The `pp` quark-direction proxy (dilution).** `pp` does not fix the quark
  direction, so the CS axis is oriented by `sign(Q_z)` (the di-lepton boost — the
  valence quark statistically carries more momentum than the sea antiquark). This
  probabilistic assignment **dilutes** `A_FB` below parton level. The pipeline
  quantifies it: `generate_hepmc.cc` emits the **true** incoming-quark `p_z` sign
  (hard-process parton, status `-21`, id 1..6) per event, and `analyze.py` overlays
  the **undiluted** `A_FB` (true direction) on the **diluted** proxy. Measured at 13
  TeV, 100k events: above the pole undiluted `+0.289 ± 0.010` vs proxy `+0.108`, a
  **dilution factor ≈ 0.37** (proxy suppresses `A_FB`), worst near central rapidity.
  Reco (Delphes CMS, proxy only — an experiment never knows the true direction)
  tracks the proxy truth, so the **detector effect on `A_FB` ≪ the dilution**.
- **The physics gate is the sign, not a tolerance.** There is **no clean closed form**
  for the `A_FB` *magnitude* (γ*/Z interference within the bin × the `pp` dilution),
  so — unlike the beam-dynamics stages — the acceptance check is the **sign guard**:
  `A_FB < 0` below `M_Z`, `> 0` above (zero-crossing just under the pole). This is the
  analog of the xtrack sign cross-checks; the opposite sign means a flipped `μ⁻/μ⁺`
  or axis orientation. Measured: below `−0.056 ± 0.007`, above `+0.108 ± 0.010`
  (`SIGN GUARD: PASS`). The **integrated-over-60–120 `A_FB` is near zero** (`+0.018`)
  by below/above cancellation over the near-symmetric window — correct physics, *not*
  the headline; `A_FB(m)` binned is the deliverable.
- **Out of scope (labelled):** the theory dilution-correction unfolding (recovering
  parton-level `A_FB` from data without the generator truth) and `sin²θ_W` extraction.
  The Collins-Soper *azimuthal* `φ*` and angular coefficients `A_0..A_7` were previously
  out of scope; they are now **built** — see *DY angular coefficients A₀–A₇ & Lam–Tung*
  below.

## DY angular coefficients A₀–A₇ & Lam–Tung (Phase 2 — extends Collins-Soper A_FB)

The full Drell-Yan lepton angular distribution in the Collins-Soper frame,
decomposed into the eight coefficients `A₀..A₇`:

```
dσ/dΩ ∝ (1 + cos²θ) + A₀·½(1 − 3cos²θ) + A₁·sin2θ cosφ + A₂·½sin²θ cos2φ
        + A₃·sinθ cosφ + A₄·cosθ + A₅·sin²θ sin2φ + A₆·sin2θ sinφ + A₇·sinθ sinφ
```

All frame physics stays in **one tested module** — `accsim.events` (pure numpy,
always-on baseline); the gated pipeline only feeds it four-vectors.

- **The CS angles `(cosθ*, φ*)` — `collins_soper_angles`.** The sibling of
  `collins_soper_costheta`, adding the azimuth `φ*` by explicit frame construction:
  boost `ℓ⁻,ℓ⁺` into the di-lepton rest frame, then build the CS axes — `ẑ_CS`
  bisects beam1 and the reversed beam2 (the standard CS bisector), `ŷ_CS ∝ k̂₁ × k̂₂`
  (normal to the production plane), `x̂_CS = ŷ_CS × ẑ_CS`. Then `cosθ* = ẑ_CS·ℓ̂⁻`,
  `φ* = atan2(ŷ_CS·ℓ̂⁻, x̂_CS·ℓ̂⁻)`. Pinned to `collins_soper_costheta` to 2e-14 in the
  massless limit (`tests/analytic/test_angular_coefficients.py`).
- **Extraction by moment projection — `angular_coefficients`.** Each `Aᵢ = ⟨Pᵢ⟩`,
  the solid-angle average of an orthogonal weight polynomial `Pᵢ(θ,φ)`:
  `P₀ = 4 − 10cos²θ`, `P₁ = 5·sin2θ cosφ`, `P₂ = 10·sin²θ cos2φ`, `P₃ = 4·sinθ cosφ`,
  `P₄ = 4·cosθ`, `P₅ = 5·sin²θ sin2φ`, `P₆ = 5·sin2θ sinφ`, `P₇ = 4·sinθ sinφ`. The
  coefficients are **derived by symbolic closure** (⟨Pᵢ·(basis)⟩ = δ, norm 16π/3;
  `test_angular_coefficients.py`), not memorised. **Requires 4π acceptance** — it is a
  truth-level observable, so the pipeline analyses generator truth and **skips
  Delphes** (`--angular-only`). Consistency anchor: `A_FB = 3/8·A₄`.
- **Quark-flip parity.** Swapping the quark/antiquark direction sends
  `cosθ* → −cosθ*`, `φ* → −φ*`; so `{A₀,A₂,A₃,A₆}` are parity-**even** (immune to the
  `pp` sign(Q_z) dilution) and `{A₁,A₄,A₅,A₇}` are **odd** (diluted, like `A_FB`).
  `A₀,A₂` and hence Lam–Tung are therefore robust to the `pp` proxy. Pinned in
  `test_angular_coefficients.py`.
- **The physics gate — the Lam–Tung relation `A₀ = A₂`.** *Dynamical* (the DY analog
  of Callan–Gross `2xF₁ = F₂`): it follows from the spin-½ quark coupling, not from
  kinematics or current conservation. **Exact at O(α_s), violated only at O(α_s²)** —
  so it is a genuine closed-form gate. Proven in `tests/analytic/test_lam_tung.py`
  from **explicit Dirac-γ matrices** (Dirac basis, metric `diag(+,−,−,−)`), no
  remembered helicity constants:
  - Build the production hadronic tensor `Wᵘᵛ` for single-parton emission via the two
    Feynman diagrams (quark spin sums + gluon-polarisation sum `−g_αβ` as traces),
    for **both** `qq̄→Vg` and the crossed `qg→Vq`; contract with the leptonic tensor
    `Lᵘᵛ = Tr[l̸⁻γᵘl̸⁺γᵛ]` to get `dσ/dΩ`; project `A₀,A₂`.
  - **Closed-form symbolic proof (`qq̄→Vg`):** on the gluon on-shell surface `k²=0`,
    `A₀−A₂` vanishes because **`k²` divides the `A₀−A₂` numerator** (polynomial
    remainder in `Q` is exactly 0). The `sinθ` solid-angle Jacobian is **required**
    (dropping it gives an unphysical `A₀<0` — a bug caught during development).
  - **Both channels** also confirmed to **~1e-14** by exact Gauss-Legendre quadrature
    (the intensity is a bounded-degree trig polynomial → integrated exactly, no
    Monte-Carlo ratio bias).
  - Correctness anchors so a wrong `W` can't sneak through: `W` is real, symmetric and
    V-current-conserved (`q_μ Wᵘᵛ = 0`), and the extracted `A₀` is a nonzero physical
    (`0 ≤ A₀ ≤ 2`) number — so `A₀ = A₂` is not vacuous.
- **Runtime note (symbolic proof kept always-run).** The naive route — `sp.cancel` on
  the fully contracted rational intensity — takes **~2 h** (multivariate GCD) and
  would break the always-green analytic suite. Two factorisations fix it to **~12 s**,
  keeping the closed-form proof in the always-run tier: (i) `Wᵘᵛ` is θ,φ-independent,
  so integrate the small leptonic basis once and contract after (linearity); (ii) each
  `Wᵘᵛ` has the **known** common denominator `DA²·DB²`, so clear it to get pure
  polynomial numerators (`A₀−A₂ = (P₀−P₂)/Pₙ`) and prove divisibility by polynomial
  remainder — **no `cancel`/GCD**.
- **The pipeline demo (`--angular-only`).** `run_pipeline.py --angular-only` runs GEN
  only (Pythia8 + LHAPDF, gated `ACCSIM_ENABLE_LHAPDF`) and `analyze_angular.py` bins
  `A₀(q_T)`/`A₂(q_T)` in the Z window `80<m<100`. Measured (13 TeV, 200k events):
  `A₀` rises from ~0 at low `q_T` to `+0.225±0.029` at `q_T≈57` GeV, with `A₂`
  tracking it; the guard is low-`q_T` `⟨|A₀−A₂|⟩ = 0.023 ± 0.019`
  (`LAM-TUNG DEMO: PASS`). **The compelling evidence is the mid-`q_T` bins, not the
  low-`q_T` average**: as `q_T→0` the distribution → pure `(1+cos²θ)` so `A₀,A₂→0`
  *regardless* of the frame construction (a broken `φ*` would still pass a low-`q_T`
  guard). Where both coefficients are substantially nonzero they still agree —
  `q_T≈12.5`: `A₀=0.074`, `A₂=0.077`; `q_T≈37.5`: `A₀=0.165`, `A₂=0.166` — which is the
  real on-data confirmation. (Frame/extraction correctness is independently gated by
  the analytic machinery tests; this demo is the physical illustration.)

## Feature switches (optional addons — implemented)

**The rule:** the pure-Python **baseline** — the accelerator optics/tracking core
(Stages 0–6) and the toy event generator (`accsim.events`), all numpy/scipy/
matplotlib only — is always on and never gated. **Everything past that baseline**
— any addon / expansion / module / component that pulls an external tool
(Docker/Pythia/Delphes), a heavy dependency, or an optional extension — sits
behind an explicit **runtime switch, default OFF** (`accsim.features`). This is a
standing project contract, not a per-stage note.

- **One source of truth, two surfaces.** `accsim.features` holds a fixed set of
  known addon names (`KNOWN_ADDONS = {pythia, delphes, lhapdf}` — one per real
  gated pipeline) and a process-global override table. Both entry surfaces read it:
  - **In-package callers** guard the heavy entry point with
    `features.require("<name>")`, which raises `AddonDisabledError` (carrying the
    enable instruction) when off. Call it **before** importing the optional
    dependency, so "off" fails cleanly instead of crashing on a missing import.
    This is the switch that earns its keep on *future* in-package additions
    (a Delphes/LHAPDF step called from inside `accsim`).
  - **Standalone scripts / CI** flip the same flag via the env var
    `ACCSIM_ENABLE_<NAME>` (e.g. `ACCSIM_ENABLE_PYTHIA=1`). Running a pipeline
    script *is* the opt-in, so its gate is deliberately light — the Pythia
    `run_pipeline.py` `main()` calls `features.require("pythia")` right after
    arg-parsing and bails with the enable instruction when off.
- **Precedence** (single rule): a programmatic override
  (`enable`/`disable`/`enabled`) beats the env var; with no override the env var
  decides; absent both, OFF.
- **Context manager is the primary API.** `with features.enabled(name):` restores
  the prior override state — *including no override* — on exit, even on exception,
  so a flag never leaks past its block. The suite's autouse fixture
  (`tests/conftest.py`) calls `features.reset()` around every test for the same
  reason (the override table is process-global).
- **No empty scaffolding.** A name enters `KNOWN_ADDONS` only when real gated code
  lands behind it (one feature per change): `pythia` (leptonic chains), `delphes`
  (the ILD detector step), and `lhapdf` (the hadronic Drell-Yan chain) each front a
  live pipeline. An *unknown* name still raises `UnknownAddonError` (typo guard),
  not a silent pass. Gated behavior (defaults OFF, baseline green with everything
  off, `require` raises-off/passes-on, precedence) is pinned by
  `tests/analytic/test_features.py` — behavioral, not a physics derivation.

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
- **(Historical, resolved)** The `zeta` sign was expected to possibly mismatch
  Xsuite on first cross-check — a convention reconciliation, not a physics bug.
  **Outcome:** no mismatch. Drift, quad, and dipole 6×6 maps agree with xtrack with
  no sign flip (see the ROADMAP: the `zeta`-sign question is **settled**).
- **(Historical)** Before the JIT was fixed the drift convention rested solely on
  the **symbolic derivation** (two independent routes agree) — itself a gold-standard
  analytic check. That derivation still stands and is now *also* corroborated by the
  passing xtrack cross-check above.
