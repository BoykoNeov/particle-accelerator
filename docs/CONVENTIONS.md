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

## The drift's exact map (L1 — implemented 2026-08-17)

**The drift now has two maps, and the difference between them is physics.**
`matrix()` is the linear map above, which every optics function is built on;
`track()` is the exact map, which is what the linear one was always the origin-slope
*of*. The exact form is the one already written above and in the sympy derivation
that pins `matrix()` — this milestone made the code use it.

`zeta` is evaluated in a **rationalised** form rather than as `1 − E/(E₀·pz)`:

```
pz    = √((1+δ)² − px² − py²)
x    → x + L·px/pz
y    → y + L·py/pz
zeta → zeta + L·(δ(2+δ)/γ₀² − px² − py²) / (pz·(pz + E/E₀))
```

The two are algebraically identical (multiply through by `pz + E/E₀`, using
`(E/E₀)² = 1 + β₀²δ(2+δ)`), but `1 − E/(E₀·pz)` subtracts two numbers of size 1 to
get a small answer. That is invisible in the value and **not** in its derivative: the
finite-difference Jacobian `linearised_element_maps()` takes carried `~2e-9` per
drift, showing up as `3.6e-8` on a 16-drift ring where the design-optics gates ask
for `1e-10`. Rationalised it is `2.7e-13`. The form is also the clearer physics — a
momentum term that speeds the particle up against an angle term that lengthens its
path, with the `1/γ₀²` showing directly why the effect dies ultrarelativistically.

Verified against `xt.Drift(model="exact")` to **4.4e-16** on every coordinate.

### The two candidate "exact" drifts, and which one this is

xtrack carries both, and its **default is not the exact one**
(`beam_elements/elements_src/drift.h`: `model = adaptive → expanded`):

| model | transverse | matches |
|---|---|---|
| `expanded` (xtrack default) | `x += L·px/(1+δ)` | MAD-X, paraxial in the angles |
| `exact` (`model="exact"`) | `x += L·px/pz` | the Hamiltonian flow |

accsim implements the **exact** one. The choice was not free: `test_drift.py`'s
existing symbolic derivation already committed to `1/pz` as the physical truth, and
`1/pz` is what makes the map the flow of a Hamiltonian. The two differ by
`(px²+py²)/2` relatively — measured `1.5e-6` at `px = 1e-2` and `1.7e-4` at
`px = 5e-2`, against `4.4e-16` for the right choice, so the analytic gate at large
angles discriminates them and cannot be satisfied by the wrong one.

> **Reference cross-checks must say `model="exact"` explicitly.** Comparing accsim's
> exact drift against a default `xt.Drift()` produces an `O(angle³)` discrepancy that
> looks like a sign or convention bug and is not one.

### The dropped term has a conjugate partner — you cannot take one without the other

Expanding `L·px/pz` gives `L·px − L·px·δ + …`. The linear matrix keeps the first and
drops the second, which is **bilinear** (a product of two small quantities) and so
cannot live in any 6×6 at all — a structural loss, not a truncation to tighten.

**It does not come alone.** Per element, at a closed-orbit angle,

```
M[x, δ] = M[zeta, px] = −L·px          M[y, δ] = M[zeta, py] = −L·py
```

— equal magnitude, and they are canonical partners: the transverse position's
dependence on momentum and the path length's dependence on angle. A map carrying one
without the other is **not symplectic**, and wrong at *first* order in the amplitude
where the correct map is exactly right. That is why the transverse and longitudinal
halves had to land in the same change, and it is a fact K2's write-up did not have
(its "two dropped terms" were the `1/pz` and the bend's extra arc — both about
transverse motion; this is a third thing).

Measured on a steered FODO ring: the `δ` column and `ζ` row account for `5.4e-3` of
the one-turn map, everything else for `3.3e-6` — second order in the orbit angle,
falling by exactly `4.000` per halving.

### Symplecticity: `(zeta, δ)` rejects this correct map — use the canonical check

See *Symplecticity* below and `accsim/symplectic.py`'s module docstring. In short:
`(zeta, δ)` is not a canonically conjugate pair, the linear drift passes anyway
because three independent shears always do, and the exact drift **fails** —
residual in the two `(p, δ)` entries, second order in amplitude
(`7.7e-8` at `1e-3`, `7.7e-6` at `1e-2`). The same map in `(zeta, p_zeta)` gives
**exactly zero**. So `is_symplectic_map` cannot judge an exact map;
`is_symplectic_map_canonical` is the gate, and it catches the tempting half-fix
(transverse exact, `zeta` left linear) at *first* order — `2.0e-4` against `0`.

### What this bought: vertical dispersion from an orbit *angle*

The gap K2 measured and could not represent. On a ring of **drifts and thin
quadrupoles with a vertical steerer and no bend at all** — so `D_x ≡ 0` and the
drift's term is the whole effect:

```
design optics   D_y = 0 exactly     (built on matrix(); correctly blind)
on-orbit optics D_y = 0.2590571     (closed_twiss_on_orbit)
xtrack exact    dy  = 0.2590571     (seven figures)
```

The first-order closed form `−L·p_co` per drift gives `0.2591583`; the residual is
`3.9e-4` relative and **second order in the orbit angle** (ratios `4.000` over four
halvings), being the `(1+δ)/pz³` the closed form drops.

K2's own test arc has *no drifts*, so this milestone moves none of its numbers.

### On the design orbit nothing changes — which bounds the whole milestone

At `px = py = 0` the exact map's Jacobian **is** the linear matrix, entry for entry
(`∂(L·px/pz)/∂δ = 0` when `px = 0`, and likewise the conjugate). So everything
computed from `matrix()` is **bit-for-bit** unchanged and every design-optics
cross-check is untouched. `linearised_element_maps()` moves only by its
finite-difference floor, `2.7e-13`.

### What it cost — five consequences worth knowing

1. **Element-by-element tracking is no longer one `transfer_map` product**, even for
   a lattice of "linear" elements. They agree only on the design orbit.
2. **The exact drift is a first-order chromatic element.** Its effective length is
   `L/pz ≈ L(1−δ)`, so tracked chromaticity is no longer zero on a sextupole-free
   machine: `−0.1289` against an analytic natural chromaticity of `−0.2893`, i.e.
   **45%** of it. I3's "tracking is blind to natural chromaticity" is now only
   *partly* true — tracking sees the drifts' share and not the quadrupoles', because
   accsim's quadrupole map is still momentum-independent. The exact quadrupole closes
   it; until then `chromaticity_on_orbit` must still be built as a difference, and
   every tracked-vs-derived feed-down gate needs a sextupole-free **baseline
   subtracted**. It also produces real chromatic beta-beat (`1.8e-4`) where the
   package could previously produce none.
3. **A drift is weakly amplitude-dependent in its focusing**
   (`∂(L·px/pz)/∂px = L(1+3px²/2)` at `py=0`), so a drift-and-quad ring has genuine
   amplitude detuning with no nonlinear magnet in it — `3.8e-8`, four orders below
   J2's octupole detuning, and *quadratic in amplitude* like it, so the two separate
   only by size.
4. **Newton's basin of attraction shrank.** A particle with `px²+py² ≥ (1+δ)²` has no
   forward momentum, the map returns `NaN`, and `closed_orbit_nonlinear` reports
   `OrbitConvergenceError` naming the iterate rather than letting the `NaN` surface as
   `LinAlgError: SVD did not converge` out of `numpy.linalg.cond`. The old "outer
   fixed points" a far guess converged onto were artefacts of a linear map that
   propagates any state at all; they were never physical.
5. **`linearised_lattice()` cannot represent the new pair and returns anyway.** No
   accsim element carries a transverse `δ` column without also bending, so the entries
   are simply absent — the two "equivalent machine" routes differ by `5.4e-3`. That is
   an omission, not a wrong number: the dropped terms carry **no gradient**, so no
   chromaticity integral reads them, and refusing would break the I2/I3/J2/J3
   feed-down machinery over a term it never looks at (unlike K2's rolled-multipole
   branch, which would emit a wrong *split* and so does refuse). The real hazard is
   asking it for dispersion, which silently returns the old orbit-blind `D_y = 0`.

### Known inconsistency, deliberately left

A zero-strength thick `Sextupole`/`Octupole`/`Quadrupole` is documented as "identical
to a `Drift` of length `L`". That is still true of its **matrix** and no longer of its
`track`. Physically a zero-strength magnet *is* a drift, so this is a real
inconsistency and the price of doing one element at a time.

**Updated and re-verified 2026-08-17 (L2)** — the three elements are no longer in the
same position, and the paragraph above was describing a code path rather than checking
one:

- **`Quadrupole(L, 0).track`** is now the **expanded** drift `x += L·px/(1+δ)`, not the
  affine default and not the exact drift. The gap to `Drift` narrows from first order
  (`O(px·δ)`) to third (`O(px³)`); it is **not** closed, and the ROADMAP's prediction
  that L2 would close it was wrong. See *The quadrupole's momentum-dependent map* below.
- **`Sextupole(L, 0).track` and `Octupole(L, 0).track`** do keep the affine default —
  they short-circuit on `k2 == 0` / `k3 == 0` — so for those two the paragraph stands as
  written. Checked in `elements/sextupole.py` and `elements/octupole.py`, not inherited.
- A related fact the original paragraph did not cover: at **nonzero** strength those two
  slice with `_drift_matrix`, the **linear** drift, not with `Drift.track`. So the
  drift-kick-drift split has carried the pre-L1 drift since J1, and `n_slices → ∞`
  converges onto the *paraxial* thick multipole rather than onto the exact one. That is
  the same model boundary L2 records for the quadrupole, reached by a different route.

Gates: `tests/analytic/test_drift.py` (13), `tests/analytic/test_exact_drift_dispersion.py`
(5), `tests/analytic/test_symplectic_canonical.py` (14),
`tests/reference/test_drift_xtrack.py` (5 — including the bend-free steered ring, so the
`0.2590571` above is cross-checked in the suite and not only in a scratch measurement).

29 pre-existing analytic tests across 9 files and 5 reference tests were re-baselined;
each is a claim restated with its measured *order* — ratios of 4, 8, 16, and derived
coefficients like `1.5·L·py²` — rather than a loosened tolerance. The two that are
bounded rather than order-pinned by their primary assertion (the beta-weighted
chromaticity sum, and the coupling angle) carry a *second* assertion on the order, for
the same reason.

## The quadrupole's momentum-dependent map (L2 — implemented 2026-08-17)

**`k1` is normalised to the *reference* rigidity, so a particle of momentum `1+δ` is
focused by `k1/(1+δ)`.** accsim's thick quadrupole did not know that: its `track()` was
its `matrix()` at every momentum — a chromatically ideal magnet, which is not what a
magnet is. L1 left tracking seeing 45% of the natural chromaticity (the drifts'
share); this closes the quadrupoles' share, and only the dipole's is left (L3).

With `K = k1/(1+δ)`, `C = cos(√K L)`, `S = sin(√K L)/√K` (continued to `cosh`/`sinh`
for `K < 0`, and to `(1, L)` at `K = 0`), and the geometric angles `x' = px/(1+δ)`:

```
x    → x C  + x' S              px → (−K x S  + x' C ) (1+δ)
y    → y Ch + y' Sh             py → (+K y Sh + y' Ch) (1+δ)      [K_y = −K]
ζ    → ζ + L(1 − 1/rvv) − I/rvv                                   [rvv = β/β₀]
I    = ½ Σ_u [ K_u u² T − K_u u u' S_u² + u'² (L − T) ],  T = (L − C_u S_u)/2
```

Verified against `xt.Quadrupole(model="mat-kick-mat")` — xtrack's own default — to
**1.1e-16** on all six coordinates.

### What it is exact in, and what it is not

It is the flow of the **paraxial** Hamiltonian

```
H = p_ζ − (1+δ) + (px² + py²)/(2(1+δ)) + (k1/2)(x² − y²),
```

the expansion in the angles of the exact `H = p_ζ − √((1+δ)² − px² − py²) + …`, whose
flow has **no closed form at all**: the square root and the quadratic potential do not
commute. Every code picks one of two families — expand the root and solve exactly
(MAD-X, xtrack's `mat-kick-mat`, this), or split the exact one and integrate
numerically (xtrack's `drift-kick-drift-*`, PTC's `exact`). They differ by `8e-5` to
`1.5e-4` on the reference states, so the choice is a real one and is pinned as such.

So the map is **exact in `δ` to all orders** and **paraxial in the angles**, dropping
`O(angle³)` relative. It is nonetheless *exactly* symplectic — the exact flow of an
approximate Hamiltonian, not an approximate flow of the exact one — which is the
property worth having: a truncated-but-symplectic map is safe to iterate for a million
turns, a more accurate non-symplectic one is not.

**Why not the splitting family.** `matrix()` must remain the exact Jacobian of
`track()` at the origin, or design optics and the tracked machine describe different
rings. A sliced map's origin Jacobian is the *sliced approximation* to the cos/sin
block, not the block — so every "tracking agrees with the matrix on the design orbit"
gate in the package would have moved. That invariant is what bounded L1 and bounds this.

### The discriminating gate is large `δ`, not large angles — and it is an identity

L1's gate shape does **not** transfer: large angles are exactly where this map is
deliberately wrong. Substituting `px = (1+δ)p̃` turns the exact drift's `L/(1+δ)` back
into `L` and the quadrupole's block into a design quadrupole of strength `k1/(1+δ)`, so

> the machine a particle of momentum `δ` traverses **is** the design machine with every
> gradient rescaled by `1/(1+δ)`

on any bend-free ring — exactly, not to first order. Tracked tunes off momentum equal
the *design* tunes of the rescaled lattice to `1e-15` out to `δ = 0.05`. That pins every
order in `δ` at once: `1/(1+δ)²`, or the factor in the trigonometric argument only, or
applied in one plane only, all agree at `δ = 0` and all fail here at `O(δ)`.

### Chromaticity: 45% → 100%, and what is left

On a bend-free drift + thick-quad ring the tracked `dQ/dδ` now equals
`natural_chromaticity`'s `−(1/4π)∮β k1 ds`. The residual is **the trapezoid error of
`slices`**, not the map's: it falls by 4 per doubling (`9.8e-6` at 64 slices,
`2.4e-9` at 4096), and is asserted as that order rather than to a tolerance.

Two controls make that attributable rather than merely observed:

- a **thin**-quad + exact-drift ring already had 100% after L1 — a thin kick
  `Δpx = −k1l x` is momentum-independent, and the drift is what turns momentum into
  angle. It shares the integral with the thick-quad gate and none of the new code.
- swapping a **zero-angle `Dipole`** for a `Drift` of the same length — identical
  matrices, identical design optics, identical analytic chromaticity — moves the
  tracked answer from **48% to 100%**. So the remaining blindness is the dipole's map
  and nothing else. On the suite's bendy test arc the tracked route now reports
  `−0.1665` against `−0.2893` (58%, up from 45%).

### The `ζ` cancellation trap, avoided again

MAD-X and xtrack both evaluate `Δζ = L − path/rvv`: two numbers of size `L` differenced
to get a small one — invisible in the value, not in its derivative, which is what
`linearised_element_maps` takes. Splitting it as `L(1 − 1/rvv) − I/rvv` and
rationalising the first term through `(1+δ) + E/E₀` (using
`(E/E₀)² = 1 + β₀²δ(2+δ)`, the drift's own identity) removes the cancellation, leaving
two small quantities that are *added*. The first term is the drift's `ζ` map at zero
angle — a free cross-check on the algebra. The path integral `I` is likewise written
**division-free**: substituting `A = −K u` and `B = u'` into the textbook form cancels
every `1/K`, so it is *entire* in `K` and the drift limit needs no branch.

Measured floor on `linearised_element_maps`: `5.4e-14`, in the `(ζ, δ)` entry, and it
is the central difference's own `O(step²)` truncation (a hundredfold per decade of
step), three orders below the `1e-10` the design-optics gates ask for.

### The ROADMAP's prediction was wrong: `Quadrupole(L, 0)` is still not a `Drift`

L2 was written up as removing the inconsistency above. It does not. At `k1 = 0` this
map is the **expanded** drift `x += L px/(1+δ)`; `Drift` is the **exact**
`x += L px/pz`. What L2 does is take the gap from **first** order to **third**: the old
linear map differed at `O(px·δ)`, this one at `O(px³)` — the same gap that separates
xtrack's own two drift models, and the price of a closed form. Asserted in the shape
that identifies it: cubic in the angle (×125 for ×5 in `px`) and *independent of `k1`*
as `k1 → 0`. Short-circuiting `k1 == 0` to the exact drift would close it only by making
the map **discontinuous in `k1`**, so the residual is documented instead.

### Blast radius

Far smaller than L1's, and for a structural reason: **at `δ = 0` the transverse map is
the linear matrix** (to one unit in the last place — the two arithmetics associate the
multiplications differently), so an on-momentum particle of any amplitude tracks exactly
as before. Only `ζ` moves, plus everything genuinely off momentum. Five analytic tests
re-baselined and two reference ones, against L1's 29 and 5.

Two consequences worth carrying:

- **`SkewQuadrupole` got the same map**, through the 45° roll conjugation it is already
  defined by. Leaving it linear while `Quadrupole(roll=−45°)` became chromatic would
  have made the same magnet behave two ways depending on how it was spelled.
- **`Dipole(L, 0, k1)` and `Quadrupole(L, k1)` are the same magnet and now track
  differently** — identical matrices, and the bend's `track` is still its matrix. This
  is a *new* divergence created by L2, it is deliberate (L3 closes it), and it is
  **load-bearing**: `test_the_bend_is_the_only_thing_tracking_is_still_blind_to` uses
  exactly that pair as a controlled experiment, since swapping one for the other changes
  the tracked chromaticity and nothing else.
- **`is_symplectic_map` now *accepts* a correct exact map at small amplitude.** The
  `(ζ, δ)` residual is second order in the amplitude *and* suppressed by `1/γ₀²`; on a
  `γ₀ = 20` ring at amplitude `1e-3` it is `8.4e-10`, under the default `atol` of
  `1e-9`. So the wrong check does not merely reject correct maps — it can pass one for
  no reason connected to symplecticity. `test_roll.py`'s rolled quadrupole was clearing
  its `1e-8` by a factor of six and now uses `is_symplectic_map_canonical`.

Gates: `tests/analytic/test_exact_quadrupole.py` (16),
`tests/reference/test_quadrupole_xtrack.py` (4).

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

`Dipole(length, angle, k1=0, e1=0, e2=0)`: a bend, bending horizontally, with an
optional combined-function gradient and pole-face angles. Curvature `h = 1/ρ =
θ/L`, `θ = angle`. With the defaults `k1 = e1 = e2 = 0` it is a **pure sector**
bend and the map below is byte-identical to the original. The gradient and edge
focusing are documented in the two subsections that follow. The `k1 = 0` body
6×6 is `exp(L·A)` of the sector-bend Hamiltonian generator (symplectic by
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

## Dipole — combined-function gradient `k1` (implemented)

A body quadrupole gradient `k1` [m⁻²] (same normalisation as `Quadrupole`) turns
the sector bend into a **combined-function** magnet. Equations of motion:

```
x'' + (h² + k1) x = h·δ        y'' − k1 y = 0
```

so horizontal focusing is `K_x = h² + k1` — the geometric weak focusing `h²`
**plus** the gradient — and vertical is `K_y = −k1`. Thus `k1 > 0` focuses `x`
and defocuses `y`, exactly as in a quadrupole; the bend's `h²` is an extra
horizontal focusing a straight quad does not have.

- **Body map** is the closed form of `exp(L·A)` for the combined generator (which
  adds `k1(x²−y²)/2` to the sector Hamiltonian). Transverse blocks reuse
  `Quadrupole._focusing_block` with `K_x` (x) and `−k1` (y). Dispersion and the
  longitudinal slip pick up the gradient through `K_x`:
  `R16 = h·(1−cos ωₓL)/K_x`, `R26 = h·sin(ωₓL)/ωₓ`, `R51 = −R26`, `R52 = −R16`,
  `R56 = L/γ₀² + h²·(sin(ωₓL)/ωₓ − L)/K_x`, with `ωₓ = √K_x`.
- **Branch-smooth in `K_x`.** The dispersion/slip integrals are written via
  helpers (`_dispersion_integrals`) with the removable singularity at `K_x = 0`
  (the `h² = −k1` tune) handled by the leading Taylor terms — verified exact to
  machine precision against `scipy.linalg.expm` there. `K_x < 0` (net horizontal
  defocus) uses the cosh/sinh branch automatically.
- **Reductions** (free regressions): `k1 = 0` is **byte-identical** to the pure
  sector map; `h = 0` (angle 0) is a **pure `Quadrupole`** and dispersion
  vanishes with the curvature.
- **Signs/reductions pinned empirically:** the whole 6×6 matches MAD-X `sbend`
  with `k1` (~1e-9) and xtrack `Bend` with `k1` (~1e-6) for both signs of `k1`
  (`tests/reference/test_dipole_combined_{madx,xtrack}.py`); the symbolic
  `exp(L·A)` gate covers `K_x` >0, <0 and the singular =0
  (`tests/analytic/test_dipole_combined.py`).
- The gradient's contribution to the radiation damping partition (`I4`'s
  `2 k1 D_x h` body term) is handled in *Synchrotron radiation* below. Edge angles
  compose on top (`Edge @ combined-body @ Edge`) and are gated together.

## Dipole — pole-face (edge) focusing (implemented)

`e1` (entrance) and `e2` (exit) are pole-face rotation angles [rad]: `e = 0` is
the sector face, `e = θ/2` the symmetric rectangular face. Each face adds a thin
**hard-edge** quadrupole-like kick, and the full map sandwiches the body —
`M = Edge(e2) @ Body @ Edge(e1)`, the entrance edge acting first. The edge map is
the identity except:

```
R21 = +h·tan(e)     (px += h·tan(e)·x  — horizontal DEFOCUS for e > 0)
R43 = −h·tan(e)     (py -= h·tan(e)·y  — vertical FOCUS   for e > 0)
```

- **Sign/plane is empirical, not remembered.** A positive edge angle defocuses
  `x` and focuses `y`; the whole 6×6 matches MAD-X `sbend` (`fint = hgap = 0`) to
  **~2e-16** and xtrack `Bend` (linear edge model, fringe off) to ~1e-6
  (`tests/reference/test_dipole_edges_{madx,xtrack}.py`).
- **Hard edge only.** The fringe-field correction (`e → e − ψ` in the *vertical*
  plane, `ψ = h·g·fint·(1+sin²e)/cos e`) is **not** applied — this is the
  apples-to-apples match to MAD-X's default `fint = hgap = 0` and xtrack's
  fringe-off defaults. Fringe is a separate, opt-in refinement (not yet built).
- **Rectangular-bend identity (the strongest gate).** For `e1 = e2 = θ/2` the two
  edge kicks *exactly* cancel the body's horizontal weak focusing: the horizontal
  block collapses to a drift `[[1, ρ·sin θ], [0, 1]]` with `R21 = 0` to machine
  precision — **proven symbolically** (`sin θ·tan²(θ/2) − sin θ + 2 cos θ·tan(θ/2)
  = 0`), not asserted "small". Meanwhile the vertical plane, a pure drift in the
  body, gets *all* its focusing from the edges (`R43 ≈ −2h·tan(θ/2)`).
- Edges are **optics-active** (they change β, tune, chromaticity and dispersion
  through composition) but add **no length** and no direct longitudinal coupling
  (the edge map's longitudinal block is the identity). Symplectic by construction
  (each 2×2 kick block has unit determinant).
- Analytic gates in `tests/analytic/test_dipole_edges.py`; the effect on the
  radiation damping partition (`I4`'s `−D_x h² tan(e)` face term) is in
  *Synchrotron radiation* below.

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

## Tracking-based tune / NAFF (D2 — implemented)

`src/accsim/tune.py` measures the tune a second, independent way: track a particle
for many turns and find the frequency of its betatron oscillation (what a real
machine does with turn-by-turn BPM data). Baseline module — numpy/scipy only, so
**no feature switch**.

- **Only the fractional tune is observable.** Turn-by-turn data samples the phase
  once per turn, so an integer number of full rotations is invisible.
  `tracked_tunes` returns `Q mod 1`; `tunes()` returns the **full** integer+fractional
  tune. **Always compare modulo 1** — they are not the same quantity.
- **The signal must be complex.** A real signal (position only) has a symmetric
  spectrum and cannot separate `Q` from `1−Q`; it can only ever yield `min(Q, 1−Q)`.
  The phase-space pair gives a signed rotation direction. **In this codebase's phase
  convention the forward (+Q) combination is `z = U − i·PU`** in normalised
  coordinates — `U + i·PU` measures `1 − Q`. Pinned empirically by
  `test_signal_sign_gives_forward_tune`, not remembered.
- **β/α come from the tracked data, never from `twiss.py`.** Normalising with
  `closed_twiss` would import the very module the check exists to cross-check — a bug
  in `match_periodic` would corrupt both sides and cancel. Instead
  `ellipse_from_trajectory` recovers the ellipse from the trajectory's own
  covariance: over a non-resonant phase, `Σ = ⟨[[u², u·u'],[u·u', u'²]]⟩ = J·[[β, −α],
  [−α, γ]]`, and since `βγ − α² = 1` exactly, `det Σ = J²` fixes the scale without
  knowing `J`:

  ```
  J = √(det Σ),   β = Σ₁₁ / J,   α = −Σ₁₂ / J
  ```

  Normalised coordinates `U = u/√β`, `PU = (α·u + β·u')/√β` then turn the ellipse into
  a circle, so the motion is a pure rotation with a single spectral line.
- **Estimator = Hann-windowed NAFF (Laskar), with a derivative polish.** A windowed
  FFT locates the peak bin (`1/N` resolution), Brent refines within ±1 bin, and the
  result is then polished by **root-finding the derivative** of the projection
  modulus. The polish is not optional dressing: locating a maximum by *comparing
  values* is capped at `~√eps` in the argument (the modulus is quadratic at its peak,
  so `eps` in the value maps to `√eps` in `f`; scipy's `fminbound` even floors its
  tolerance at `√eps·|f|`). Measured: the tone gate stalls at **~1e-9** without the
  polish and reaches **~1e-16** with it. The derivative crosses zero *linearly*, which
  recovers the lost half of the digits.
- **The normalisation need not be perfect.** Finite-turn phase sampling leaves an
  `O(1/N)` error in the recovered β/α, so the normalised orbit is a slightly eccentric
  near-circle, which leaks a small conjugate line at `−Q`. That line sits `2Q` away in
  frequency, and Hann sidelobes fall off steeply, so it does **not** measurably shift
  the `+Q` peak — hence β/α accurate to `~1e-4` still gives a tune good to `~1e-15`.
- **Scope — do not oversell.** With `nonlinear=False` the tracking applies the *same*
  one-turn matrix that `tunes()` is built from, so agreement validates the **extraction
  method**, not the one-turn map itself. The map is pinned separately by the element
  tests and the xtrack Twiss cross-check.
- **Gate** (`tests/analytic/test_tracked_tune.py`, layered so a wrong estimator and a
  wrong lattice cannot cancel): (1) a *synthetic* tone of known frequency recovered to
  `<1e-12` — no optics in the test at all; (2) a known CS ellipse recovered from
  exactly-sampled synthetic points to `1e-12`; (3) integration — tracked tune ==
  `tunes() mod 1` to **1e-10** (ROADMAP D2 asks 1e-5; measured ~4e-15). The test ring
  (28-cell FODO arc, `Qx = 2.2434`, `Qy = 1.7946`) is chosen to dodge every
  degeneracy: non-zero and *differing* integer parts (so `frac(Q) ≠ Q` is exercised),
  fractional parts far apart (no plane swap), clear of 0/0.5/1 — and `frac(Q_y) =
  0.795 > 0.5`, which a real-signal estimator would alias to 0.205, so that plane
  passes only because the signal is complex.
- **Long-term symplecticity** (tracked motion neither damps nor blows up) is the
  sibling check in `tests/analytic/test_tracking_stability.py` (marked `slow`) — see
  *Symplecticity* below.

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
- **Scope (as of F2): quads *and* dipoles.** The quadrupole term above is the
  Stage-2 core; **F2** added the dipole weak-focusing, dispersion and pole-face
  edge terms (next section). Drifts still contribute nothing.
- **Independent validation.** The coefficient and per-plane sign are pinned to
  **machine precision** by differentiating the `δ`-dependent thin one-turn map
  symbolically (`cos μ(δ) = ½ Tr M(δ)`, `Q = μ/2π`, `dQ/dδ|₀`) — a check that
  never touches `β` or `4π`, so it is not circular with the β-sum
  (`tests/analytic/test_chromaticity.py`). The thick β-integration path matches a
  finite-difference tune derivative (always-on) and xtrack's real-particle
  tracking to `rel ≈ 1e-4`.

## Dipole chromaticity (F2 — implemented)

`natural_chromaticity` (above) now carries the **full dipole** contribution, not
just the quadrupole gradient. Derived from the exact curvilinear Hamiltonian
`H = -(1+hx)·√((1+δ)²-p_x²-p_y²) - ψ`, with `ψ = (1+hx)·a_s = -hx - (k1+h²)/2·x²
+ k1/2·y²` (the `-h²/2·x²` is the curvilinear metric correction — without it the
on-momentum `K_x` comes out `2h²+k1` instead of the validated `h²+k1`).

**The β-weighted form the module ships:**
```
Q'_x = -(1/4π) ∮ β_x (k1 + h²) ds + (1/4π) ∮ h (γ_x D_x - 2 α_x D_px) ds
       + (1/4π) ∮ 2 h k1 β_x D_x ds  + (1/4π) Σ_faces β_x h tan(e)
Q'_y = +(1/4π) ∮ β_y k1 ds        + (1/4π) ∮ γ_y h D_x ds
       - (1/4π) ∮   h k1 β_y D_x ds  - (1/4π) Σ_faces β_y h tan(e)
```
with `h = 1/ρ` the curvature, `k1` the (quad or combined-function) gradient, and
`γ_u = (1+α_u²)/β_u`. Four groups:

- **Gradient focusing** `-β_x k1` / `+β_y k1` — the classic quadrupole term,
  unchanged.
- **Dipole weak focusing + dispersion.** Naively the geometric `h²` focusing
  would dominate (`-∮β_x h²`, a large negative), but the dispersion term — the
  `(1 + h·D_x·δ)` factor in the metric evaluated on the *dispersed* closed orbit —
  largely cancels it. So a **pure sector bend contributes almost nothing**. This
  is why the reverted F1 "gradient-only" patch was *worse* than omitting bends:
  it kept a partial term whose cancelling partner was missing. Validated against
  **xtrack to ~1e-6** on bendy FODO rings.
- **Combined-function curvature-sextupole feed-down** `+2 h k1 β_x D_x` /
  `-h k1 β_y D_x`. A combined-function *sector* magnet cannot have exactly
  `B_y = h + k1·x, B_x = k1·y` — that field has `∇·B = h·k1·y ≠ 0` in the curved
  frame. Maxwell forces a 3rd-order correction `ψ₃ = c₁·x³ + c₂·x·y²` with
  `6c₁ + 2c₂ + h·k1 = 0`; pinning the split by the horizontal xtrack match gives
  `c₁ = -h·k1/3, c₂ = +h·k1/2`. This curvature term acts as a **sextupole** and
  feeds down to chromaticity at dispersion. Note the coefficients `2:−1` are **not**
  the symmetric ratio of an ordinary sextupole — because `ψ₃` isn't a pure
  sextupole. This term is what makes the combined-function result match **xtrack
  and MAD-X** (both give `dqx ≈ +0.62` on an AG `k1=0.3` ring; without it accsim
  gave `−0.72`). It does **not** touch the linear map, so F1's validated map is
  unchanged.
- **Pole-face edges** `±β h tan(e)` — a localised thin-kick contribution at each
  face. Validated against **xtrack to ~1e-8**.

**Derivation route.** Linearise the exact equations of motion about the dispersed
orbit `x_co = D_x·δ, p_x,co = D_px·δ`. With the Maxwell-corrected `ψ` the canonical
focusing entry `a21` gains a `δ`-dependent piece `∝ h k1 x_co` (the curvature
feed-down); the remaining chromaticity is the drift-term (`a12 = (1+h x_co)/(1+δ)`)
effect. The tune-shift-from-generator formula `Δμ = -α N11 - (β/2) N21 + (γ/2) N12`
gives the **γ-form** integrand; the β-form above is its integration-by-parts partner
(equal around a closed ring via `∮ γ ds = ∮ β K ds`, i.e. `∮ α' ds = 0`).

**Validation (`tests/analytic/test_dipole_chromaticity.py`, `tests/reference/
test_chromaticity_xtrack.py`).** The integrand is re-derived symbolically from the
Hamiltonian — `(c₁,c₂)` fixed by Maxwell + the *horizontal* match, and the
**vertical** coefficient then follows with no further freedom (the non-circular
confirmation). β-form == γ-form ring total; an independent off-momentum map
(`exp(A·ds)`) agrees to ~1e-5; and xtrack cross-checks sector, edged, **and**
combined-function rings (the last also agreeing with MAD-X). The `2h k1`/`−h k1`
split emerged as a **bug fix**, not a model choice: a first ship treated the
combined-function term as model-ambiguous, but the codes actually agree and accsim
was the outlier — the missing Maxwell term was the cause.

## Sextupole (Stage 2 — implemented; nonlinear map added by J1)

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
  linear lattice **unchanged** (asserted to `rel 1e-14`). Since **J1** this is a
  measured statement, not a modelling one: the FD Jacobian of the nonlinear
  `track()` at the origin *is* the identity matrix.
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
  the *difference* cross-check below. Both are also **design-orbit** quantities —
  they use on-axis `elem.matrix()` throughout, so a *steered* machine's β-beat from
  I2 feed-down is not included; see *Sextupole feed-down on a distorted orbit*.
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

## The sextupole's nonlinear map (J1 — implemented)

Until J1 the sextupole carried `k2` but no map: `matrix()` was a drift and the
strength was felt only through feed-down chromaticity. J1 makes the kick real.
`ThinSextupole.track()` applies it exactly; `Sextupole.track()` composes
drift–kick–drift. `matrix()` is **unchanged** in both — the kick has no linear
part at the origin, which is the physics, not an omission.

**Where the coefficients come from — the field, not a potential.** The `½` is the
`1/n!` of the MAD-X / Xsuite normal-multipole expansion

```
B_y + i B_x = (Bρ) Σ_n k_n (x + i y)^n / n!,     Δpx = −(q/p₀)∫B_y ds,  Δpy = +(q/p₀)∫B_x ds
```

whose `n = 1` term is the `Quadrupole` this package already validates against both
xtrack and MAD-X (`B_y = Bρ k1 x` ⇒ `Δpx = −k1 L x`, `x″ + k1 x = 0`). Deriving the
sextupole from a potential `V` *chosen to reproduce the kick* would have re-proved
the algebra, not the physics; anchoring the same expansion at `n = 1` does not.

**No `1/(1+δ)` scaling.** `px` is normalised to the *reference* momentum `p₀` and
the kick is an integrated field over that same `p₀`, so the particle's own momentum
deviation does not enter — the same reasoning as `Corrector`. Verified against
xtrack at `δ ≠ 0`.

### What gates the `½` — and the four checks that cannot

This is the milestone's methodological content. The obvious structural checks are
**all blind to the coefficient**, because a mis-scaled sextupole *is still a
sextupole*:

| check | blind to `½`? | why |
|---|---|---|
| symplecticity at any amplitude | **yes** | true for any gradient kick, any strength |
| `∂Δpx/∂y = ∂Δpy/∂x` (curl-free ⇒ Maxwell) | **yes** | the same statement as symplecticity for a thin kick |
| Jacobian at origin = identity ⇒ tunes/β unmoved | **yes** | true for any purely quadratic kick |
| sympy derivation from a potential `V` | **yes** | `V` was reverse-engineered from the kick |
| small-amplitude `tracked_tunes` = linear tunes | **yes** | detuning ~`k2²J` is below FFT noise at `x₀ = 1e-8` |

The gate that **does** discriminate: linearise the new nonlinear map about the
**off-momentum closed orbit** (Newton on the tracked map, then its FD Jacobian) and
read `dQ/dδ` straight off the result. Since accsim's linear element matrices carry
no `δ` dependence of their own, the entire `δ`-dependence of those tunes *is* the
sextupole feed-down — comparable against `chromaticity − natural_chromaticity`,
which is pinned symbolically and cross-checked against xtrack's real tracking at
`rel ≈ 5e-4`. Two routes, no shared code, and the tracked value scales linearly
with the kick coefficient. Agreement is `rel 1e-5`.

Proved to have teeth: a sextupole with `1` in place of `½` (**both** components
scaled, so it stays a genuine gradient kick) passes every structural check in the
table and is caught by the feed-down gate as a clean **factor of two**. Scaling
only `Δpx` is a different bug — that one is no longer a field at all, and
symplecticity does catch it. Both are in the suite, labelled as the two distinct
classes they are.

### Thick = drift–kick–drift, and what that costs

`Sextupole.track()` splits into `n_slices` × `drift(L/2n) · kick(k2l/n) ·
drift(L/2n)`. Each factor is symplectic, so the composition is symplectic
**exactly**. It is *not* exact in the map: the BCH remainder is `O(L³)` per slice at
fixed `k2` (terms in both `k2 L³` and `k2² L³`), i.e. `O(1/n_slices²)` — a
second-order integrator. Both scalings are measured, not asserted (`L³`: ratio
`0.125` per halving; slices: ratio `0.25` per doubling).

⚠️ **The thin-lens limit is slower than "second order" suggests.** At fixed
*integrated* strength `k2l` the `k2² L³` term is `k2l² L`, so shrinking `L` at fixed
`k2l` closes the gap only **linearly** (measured ratio `0.4999` per halving). A
genuinely thin magnet is `ThinSextupole`, not a short thick one. At `k2 = 0` the
composition collapses onto the linear drift map identically, for any `n_slices`.

### The sign, by probe (G1 rule)

`ThinSextupole(k2l)` ≡ `xt.Multipole(knl=[0, 0, +k2l])`, **bit-for-bit** — the
tightest cross-check in the package, and exact precisely because a thin kick has no
length, so the two codes share no drift model to disagree about. The opposite sign
misses by exactly twice the kick. Established empirically rather than argued: the
MAD-X normal/skew asymmetry already bit this package once (`Corrector`:
`kick_x = +k` is `knl=[−k]` but `kick_y = +k` is `ksl=[+k]`).

**The thick element is compared by difference, and why.** A raw `Sextupole` vs
`xt.Sextupole` comparison leaves a `~1e-8` residual that has nothing to do with the
sextupole: it is present **unchanged at `k2 = 0`** and equals `−L·px·δ`, the leading
term of xtrack's exact drift `x += L px / √((1+δ)² − px² − py²)` against accsim's
linear `x += L px`. Toggling `k2` at fixed geometry cancels it and isolates the
nonlinear content (`rel 1e-3`; the remainder is that same drift difference feeding
the kick a slightly different `x`). Two further consequences are gated rather than
tolerated: xtrack's `zeta` moves when `k2` turns on (path lengthening of a deflected
trajectory, `−(L/2)Δ(px²+py²)/2`, predicted to `2 %`) while accsim's linear drift has
no `px` dependence at all, so its `zeta` does not move.

⚠️ **`n_slices` does not converge accsim onto xtrack** — it converges it onto the
*exact* map. xtrack's thick sextupole is itself a single-kick split, so
`n_slices = 1` is its closest match and larger counts move away. That is not a bug
in either code, and a naive "increase slices until they agree" gate would have read
as one.

### Tracking-path consequences

- `Tracker.track` / `track_turns` default to `nonlinear=False`, which **silently
  drops** the sextupole kick — a lattice containing a sextupole is no longer one
  whose two tracking paths agree. Asserted explicitly in the suite so it is
  documented behaviour rather than a discovery.
- `Tracker.track_bunch_losses` gained a `nonlinear` flag for the same reason: its
  hoisted per-element matrices would otherwise linearise a sextupole in a loss-aware
  track. ⚠️ Nonlinear tracking against apertures is the machinery **dynamic
  aperture** is built from; DA remains out of scope and nothing gates
  amplitude-dependent survival.
- `RFCavity.track` was vectorised over a `(6, n)` bunch (it took `float(zeta)`, so
  the new bunch path would have raised on any lattice with a cavity).
- `accsim.symplectic` gained `jacobian(map_fn, state)` (central FD) and
  `is_symplectic_map(map_fn, state)`. Symplecticity of a *nonlinear* map is a
  statement about its Jacobian at **every** point, so a test must sample a nonzero
  amplitude — at the origin every thin kick passes vacuously.

Gates: `tests/analytic/test_sextupole_kick.py` (21),
`tests/reference/test_sextupole_kick_xtrack.py` (7).

**Follow-up.** J1's kick expanded about a *closed-orbit* offset rather than a
dispersive one is **I2** (*Sextupole feed-down on a distorted orbit*), which is
where the dipole and skew terms — the two J1 never produced — are gated, and where
the closed orbit becomes a fixed point. J1 was sequenced first only so its own gate
would not be circular. The *next* multipole is **J2**, below.

## The octupole and amplitude detuning (J2 — implemented)

The `n = 3` term of the same expansion, and the first quantity in the package where
the tune belongs to the **particle** rather than to the machine.

```
Δpx = −(1/6) k3l (x³ − 3xy²),      Δpy = +(1/6) k3l (3x²y − y³)
k3 = (1/Bρ)(∂³B_y/∂x³)  [m⁻⁴],     k3l = k3·L  [m⁻³]     (MAD-X / Xsuite)
```

`Octupole` (thick, drift–kick–drift with `n_slices`) and `ThinOctupole` live in
`src/accsim/elements/octupole.py`, always-on baseline (numpy only). `matrix()` is a
drift (thin: the identity) — an octupole has no linear part at all, so β, dispersion
and the linear tunes are **bit-for-bit** independent of `k3l`.

**The coefficient.** The `1/6` is the `1/3!` of the same normal-multipole expansion
above, anchored at `n = 1` (`Quadrupole`, xtrack- *and* MAD-X-validated) and `n = 2`
(`ThinSextupole`, pinned against xtrack in J1). Every structural check is blind to
it exactly as in J1 — symplecticity, curl-free/Maxwell, identity Jacobian at the
origin, `matrix()` == drift, and a sympy derivation from a potential
reverse-engineered out of the kick — and a deliberately mis-scaled octupole (`1` for
`1/6`) is carried through all of them to prove it.

**The sign, by probe (G1 rule):** `ThinOctupole(k3l) ≡ xt.Multipole(knl=[0,0,0,+k3l])`.
Agreement is **one ulp**, not bit-for-bit as for the sextupole: xtrack reaches the
cubic through its general `knl` recursion, so the arithmetic ordering differs in the
last bit. `−k3l` misses by exactly twice the kick.

### Amplitude detuning — the physics gate

`accsim.twiss.amplitude_detuning(lattice)` returns the symmetric 2×2 anharmonicity

```
∂Qx/∂Jx = + k3l βx²/(16π),   ∂Qy/∂Jy = + k3l βy²/(16π),
∂Qx/∂Jy = ∂Qy/∂Jx = − k3l βx βy/(8π)                      [m⁻¹, J in m·rad]
```

with `u_max = √(2 J_u β_u)`. Derived by averaging `V = k3l(x⁴ − 6x²y² + y⁴)/24` over
both betatron phases at fixed action and taking `ΔQ_u = (1/2π)·∂⟨V⟩/∂J_u`. Thin
octupoles contribute at a point; a thick one integrates β² across its body (β
transports as through a drift), and the gap to the thin element closes as **O(L²)**.

Two properties fall out rather than being imposed, and both are gated: the matrix is
**symmetric** (one averaged Hamiltonian), and for a single octupole the *ratio* of
cross term to diagonal is `−2βy/βx` — hence exactly `−2` at `βx = βy`, a pure number
carrying no `k3l` at all.

⚠️ The ratio is the right form; `−2√(∂Qx/∂Jx · ∂Qy/∂Jy)` is **not**. The product of
the two diagonals is positive for *either* sign of `k3l`, so that square root is
always negative, while the true cross term flips sign with `k3l`. A defocusing
octupole is entirely ordinary — Landau octupoles are run at both polarities — so both
signs are gated, along with the sign-free identity `A01² = 4·A00·A11`.

**The averaging machinery is anchored, not assumed.** `ΔQ = (1/2π)∂⟨V⟩/∂J` is first
run on `V = k1l x²/2`, where it must reproduce `β k1l/(4π)` — checked symbolically
*and* against accsim's own matrix tunes by adding a weak `ThinQuadrupole` to a real
ring. Only then is it pointed at the octupole.

**The tracked gate is an order gate, not a tolerance.** Tracking sees all orders; the
closed form is first order in both `k3l` and the action, so one tolerance at one
amplitude would swallow exactly the coefficient error the gate exists to catch. Over
four halvings of the launch amplitude the measured detuning falls by **4** (linear in
the action) while the residual falls by **16** (quadratic). Measured 2026-08-17:
signal ratios 4.095 / 4.021 / 4.005, residual ratios 17.83 / 16.39 / 16.09 (1024
turns; identical at 2048 — these are physics, not sampling noise). The mis-scaled
octupole is caught as a clean **factor of 6**.

Details that are load-bearing rather than incidental:

- **The ring carries no sextupoles.** A sextupole detunes too — at *second* order in
  `k2`, and therefore **linearly in the action**, indistinguishable from the
  octupole's term by an amplitude scan. That background does not vanish as `k3l → 0`
  and no closed form for it is claimed anywhere here. It is measured instead:
  `amplitude_detuning` returns exactly zero for a sextupole-only ring while tracking
  shows a real shift scaling as **k2²** (gated).
- **The working point matters.** `Qx = 0.294`, `Qy = 0.637`, chosen by scan to sit
  0.137 from every resonance an octupole itself drives (`4Qx`, `4Qy`, `2Qx ± 2Qy`)
  and from the tunes NAFF reads badly (0, ½, 1). Sitting near one reads as a
  coefficient error.
- **The action carries a `1 + α²`.** `tracked_tunes` launches with `px = py = 0`, so
  `J = (1 + α²)u₀²/(2β)`, not `u₀²/(2β)`. The ring is a palindrome so `α = 0` at the
  launch point — asserted, not assumed, because a silent `1 + α²` rescales every
  measured slope.
- **The measurement is a difference** against the same ring with the octupole
  removed, tracked at the same amplitude. NAFF has its own `O(1/n_turns)` bias;
  differencing two measurements made the same way cancels it, while comparing against
  an exact linear tune would leave it in the answer at the level the detuning lives.
- `tracked_tunes` refuses a zero launch amplitude in either plane, so one action can
  never be isolated: the four matrix entries come from a least-squares **plane fit**
  over a (Jx, Jy) grid, which makes the symmetry an experimental result.

**Scope, enforced rather than documented.**

- ⚠️ **Superseded by J3 (2026-08-17), recorded because the reasoning still holds.**
  At J2 `orbit.linearised_lattice` **raised** on any non-zero octupole — the split was
  not derived, and passing the element through would report a drift, i.e. claim the
  beam on a distorted orbit sees no gradient from it. J3 derived the split, so the
  line now sits at **thick vs thin**: a `ThinOctupole` is expanded, a thick `Octupole`
  still raises for the `O(L²)` reason. See *Octupole feed-down on a distorted orbit*.
- **So at J2 the I3 on-orbit family split in half — it no longer does.** The entry
  points that differentiate `track()` (`closed_twiss_on_orbit`,
  `propagate_twiss_on_orbit`, `tunes_on_orbit`, `coupled_twiss_on_orbit`) always
  worked with a live octupole; the ones that walk element *types* through
  `linearised_lattice` (`chromaticity_on_orbit`, `natural_chromaticity_on_orbit`)
  raised at J2 and **answer since J3**. The guard is still gated through the
  user-facing calls rather than only by calling `linearised_lattice` directly, so an
  edit that stopped routing through it could not drop the thick-element guard
  unnoticed.
- `chromaticity()` ignoring octupoles is **right at first order on the design
  orbit**: expanding `x = x_β + D_x δ` about `x_co = 0` gives an octupole a
  *sextupole* term linear in `δ` and a gradient only at `δ²`, so `Q′` is untouched and
  `Q″` is the honest blind spot (derived in sympy, gated). On a **distorted** orbit
  the same expansion about `x_co ≠ 0` produces a δ-linear gradient `k3l·x_co·D_x`,
  which is J3's first rung — genuine `Q′` of the real machine, not a piece of `Q″`
  arriving late. `Q″` stays uncomputed on **both** orbits (the `½k3l·D_x²` gradient,
  and the δ-dependence of `x_co` itself).
- `Tracker.track(nonlinear=False)` silently drops the kick, exactly as for the
  sextupole — asserted so it is documented rather than discovered. A detuning study
  run on the default path measures exactly zero, convincingly.
- Not claimed **at J2**: sextupole (second-order) detuning, the octupole's own
  second-order term, octupole feed-down on a distorted orbit (**delivered by J3**),
  resonance driving terms, normal-form machinery, dynamic aperture, and octupoles as
  matching knobs.

Gates: `tests/analytic/test_octupole_kick.py` (20),
`tests/analytic/test_amplitude_detuning.py` (12),
`tests/reference/test_octupole_xtrack.py` (7). The reference suite fits the
anharmonicity matrix from **xtrack's own tracked particles** (actions built from
xtrack's twiss; accsim supplies only the D2-validated NAFF step, applied identically
to both runs so its bias cancels) and agrees within **1.1 %** on the diagonal and
**0.3 %** on the cross terms — the residual being the second-order-in-action term the
first-order form does not carry.

## Octupole feed-down on a distorted orbit (J3 — implemented)

J2's octupole was exact on axis and *refused* off it. J3 derives what it refused.
Expanding the cubic kick about an orbit offset `(x_co, y_co)` splits **one octupole
into six elements** (derived in sympy, never recalled):

```
dipole       θx = −(1/6)k3l·x_co(x_co² − 3y_co²),  θy = +(1/6)k3l·y_co(3x_co² − y_co²)
normal quad  k1l_eff  = +(1/2)k3l(x_co² − y_co²)
skew quad    k1sl_eff = +k3l·x_co·y_co
normal sext  k2l_eff  = +k3l·x_co
skew sext    k2sl_eff = +k3l·y_co
octupole     unchanged
```

`orbit.linearised_lattice` now emits the four *elements* (the dipole does not appear:
it is what placed the orbit these are read at, and a `Corrector`'s `matrix()` is the
identity anyway) and keeps the octupole. A **thick** `Octupole` still raises, for the
thick sextupole's `O(L²)` reason; `linearised_element_maps` handles both, since it
differentiates `track()`.

**A cubic kick reaches two orders below itself**, where the sextupole's quadratic kick
reached one. That is the whole milestone: the same coefficient set drives three
quantities the package computes by three unrelated routes, at three different powers
of the orbit.

| rung | quantity | mechanism | power | residual |
|---|---|---|---|---|
| 1 | `Q′` (chromaticity integrals) | `k2l_eff` at dispersion | `x_co` | `x_co³` |
| 2 | tunes / β (linearised matrix) | `k1l_eff` | `x_co²` | `x_co⁴` |
| 3 | the closed orbit (Newton) | the dipole | `x_co³` | `x_co⁵` |

Measured over four halvings of the steerer: 2.0 / 4.0 / 8.0 with residuals 8 / 16 /
32, and the three exponents fitted directly as 1, 2, 3 to within 2 %.

⚠️ **Neither half of the gate works alone**, and this is J1/J2's lesson for the third
time. A uniformly mis-scaled octupole (`1` for `1/6`) leaves all three *powers*
untouched — it is caught only as a clean factor **6** in magnitude. A single
magnitude check at a single amplitude is exactly what a wrong coefficient survives.
Both are in `test_octupole_feeddown.py` and the file says so.

Details that are load-bearing:

- **Rung 1 starts from exactly zero.** On the design orbit an octupole contributes
  *nothing* to `Q′` (J2: its `δ` term is a sextupole, not a gradient), so
  `chromaticity_on_orbit == chromaticity` bit-for-bit on axis. Steering turns that
  into a first-order effect — a quantity the package previously said did not exist.
  The tracked route (`tunes_on_orbit` at `δ = ±h`) reaches the same number by a route
  with no integral in it: the gap between them falls by **4 per halving of `h`**
  (measured 4.09e-3 → 6.40e-5), i.e. it is the central difference's own `O(h²)` and
  extrapolates to zero. A wrong `k2l_eff` would leave an `h`-independent gap.
- **The six-way identity is exact, and therefore had to be made non-blind.** For a
  thin octupole the expansion terminates, so the chain of six reproduces the kick to
  round-off. It is run with **both** planes steered (both offsets asserted non-zero,
  or the skew pair vanishes and its signs are unconstrained) and each of the six
  coefficients is flipped in turn and shown to break it.
- **`θx/k1l_eff = −x_co/3` is pure geometry** — no `k3l` in it — where the sextupole's
  (I2) is `−x_co/2`. A gradient measurement alone fixes the product `k3l·x_co²` and
  never the split.
- **`x = px = 0` is an *exact* invariant subspace, as well as `y = py = 0`.** The
  octupole's kick is odd in both coordinates, so a purely vertical bump does **not**
  steer the beam horizontally — where through a sextupole (whose `Δpx` is *even* in
  `y`) it does. Asserted at exact zero, and the sharpest single distinction between
  the two elements' feed-down.
- **Coupling needs both planes here.** `k1sl_eff = k3l·x_co·y_co` is a *product*, so a
  vertical bump alone leaves `γ_c` exactly 1, where a sextupole's `k2l·y_co` would
  couple. The skew *sextupole* is live with a vertical bump alone but has no linear
  part, so it is invisible to `γ_c` — which is its own element's blind spot, restated.
- **The equivalent lattice is exact; the residual is the differencing.** Against
  accsim's own `linearised_one_turn_map` the gap falls as `step²` (1.66e-6 → 1.85e-9
  over steps 2.7e-6 → 1e-7). ⚠️ Steps must be a *constant* factor apart or the
  expected ratio alternates (9 vs 11.1) and reads as a broken scaling.

**xtrack cross-check** (`tests/reference/test_octupole_feeddown_xtrack.py`, 6):

- Nonlinear closed orbit vs xtrack's own iterative search, both planes steered:
  agreement `1e-12` m, while I1's linear solve misses by `>1e3×` that.
- **The derived split as a whole matrix.** `linearised_lattice(...).one_turn_map()` —
  built from the derived coefficients, *no finite difference on accsim's side* —
  against xtrack's `R_matrix`. The residual is **xtrack's own differencing**, proven
  three ways: it falls as xtrack's `steps_R_matrix` squared (1.17e-7 → 1.17e-9 for
  `dx` 1e-6 → 1e-7), it is strictly proportional to `k3l`, and at `k3l = 0` the two
  codes agree to 7e-13 on the same steered ring.
- **Chromaticity on a steered octupole ring**, the milestone itself: xtrack's `dqx`
  moves by far more than the bend-nonlinearity floor, accsim's design-orbit
  `chromaticity` reports the unsteered number bit-for-bit and is decisively wrong, and
  `chromaticity_on_orbit` closes the gap by more than an order of magnitude.

Gates: `tests/analytic/test_octupole_feeddown.py` (17),
`tests/reference/test_octupole_feeddown_xtrack.py` (6), plus the two J2 scope tests in
`test_octupole_kick.py` **converted** rather than deleted (thin handled, thick
refused; the whole on-orbit family now answers).

Not claimed, unchanged from J2: the octupole's second-order detuning, `Q″`, resonance
driving terms and normal-form machinery, decapoles and above, the 6D closed orbit, and
misalignments as element attributes.

## The skew sextupole (J3 part 1 — implemented)

The `n = 2` **skew** term of the same expansion, written with both families present:

```
B_y + iB_x = (Bρ)·Σₙ (kₙ + i·kₙₛ)(x + iy)ⁿ/n!
Δpx = +k2sl (x y),      Δpy = +(1/2) k2sl (x² − y²)
k2sl = k2s·L  [m⁻²]                                    (MAD-X / Xsuite)
```

`ThinSkewSextupole` lives in `src/accsim/elements/sextupole.py`, always-on baseline
(numpy only). `matrix()` is the identity. There is deliberately **no thick
`SkewSextupole`** — nothing needs one, and a thick sextupole is already refused by
`linearised_lattice` for its own `O(L²)` reason.

**Why it exists at all:** J3's octupole feed-down produces one. An octupole at a
*vertical* orbit offset is a skew sextupole of strength `k2sl = k3l·y_co`, and
dropping that term would be exactly the silent omission the octupole branch of
`linearised_lattice` exists to prevent.

⚠️ **This is the one element in the package whose sign no analytic gate can pin.**
Nothing accsim computes responds to it: `chromaticity` sums `k2l` over *normal*
sextupoles at `D_x`, `amplitude_detuning` walks octupoles. So the analytic suite is
structural (symplectic, curl-free, identity `matrix()`, identity Jacobian at the
origin) or shape-only, and **all of it is satisfied by the opposite sign**. The
convention is fixed by probe alone (G1 rule), measured 2026-08-17:
`ThinSkewSextupole(k2sl) ≡ xt.Multipole(ksl=[0,0,+k2sl])`, agreement a few ulp,
`−k2sl` missing by exactly twice the kick. The normal/skew asymmetry is not
cosmetic — `Corrector` already needs `knl=[−k]` for `kick_x=+k` but `ksl=[+k]` for
`kick_y=+k`.

**Two things the analytic suite *can* do**, and they are what make the coefficient
non-circular:

- The `kₙ + i·kₙₛ` series is written **once** and evaluated at three places. Its
  `n = 1` skew term must reproduce `ThinSkewQuadrupole` (xtrack-pinned in G1) and its
  `n = 2` normal term must reproduce `ThinSextupole` (xtrack-pinned in J1), so the
  `1/2` and the *relative* sign of the two skew components inherit two independent
  verdicts. The only difference between the families is the `i` on the strength.
- **The roll angle is solved for, not recalled.** A skew sextupole is a normal one
  rolled by **−30°** (`π/(2(n+1))` for a `2(n+1)`-pole). Sympy is asked which rolls
  satisfy `R(φ)ᵀ·k_normal(R(φ)r) = k_skew(r)`; the coefficient conditions collapse
  onto the *tripled* angle, so the answer is a family `−π/6 + 2πn/3` — the sextupole's
  three-fold symmetry. Recorded because it is **not unique** (`+π/2` is the same
  magnet) and because `+π/6` satisfies the same shape with the opposite sign.

**The chromatic effect a skew sextupole really has is not modelled.** At dispersion
it feeds down a `δ`-dependent *skew* gradient `k1sl = k2sl·D_x·δ` — chromatic
**coupling**, a quantity this package does not compute anywhere. `chromaticity()`
returning bit-identical values with and without one is asserted in the suite so the
blind spot is documented rather than discovered.

Gates: `tests/analytic/test_skew_sextupole.py` (11),
`tests/reference/test_skew_sextupole_xtrack.py` (4).

## Betatron coupling — skew quad, normal-mode tunes, ΔQ_min (G1 — implemented)

The linear x-y coupling milestone (expansion axis **G1**). Three baseline pieces
(always-on; numpy/scipy only), built one feature per commit.

**`SkewQuadrupole` / `ThinSkewQuadrupole` (the coupling source).** A skew
quadrupole is a normal `Quadrupole` **rolled 45° about the s-axis**: its field
mixes the planes, the canonical source of betatron coupling. `k1s` is the skew
gradient of the equivalent normal quad. The map is implemented **directly** from
the closed form of the roll conjugation `R(π/4)·Q_body(k1s)·R(−π/4)`; on the pairs
`(x,px)`,`(y,py)` it is the block matrix `[[A,B],[B,A]]` with `A=(F+D)/2`,
`B=(D−F)/2`, where `F=_focusing_block(k1s,L)` (cos/sin) and `D=_focusing_block(−k1s,L)`
(cosh/sinh) are the same blocks a normal quad uses. So `k1s=0 ⇒ F=D ⇒ B=0` (a
drift), and `k1s→−k1s` swaps `F,D` and flips the coupling — no special cases.
Longitudinal `R56=L/γ₀²` (the roll leaves `(zeta,delta)` alone). The thin kick is
`Δpx=k1s·l·y`, `Δpy=k1s·l·x` (symmetric, `R[px,y]=R[py,x]=k1s·l`).

- **Sign convention, pinned three ways.** The `k1s` sign is *not* derivable from the
  roll-identity gate alone (it is symmetric in construction), so it is pinned
  empirically: accsim `+k1s` agrees with **MAD-X** `quadrupole,k1s>0` and **xtrack**
  `Quadrupole(k1s>0)` on the coupling sign.
- **Exact roll vs xtrack's first-order model (honest disagreement, D3-style).**
  accsim's map is the **exact** hard-edge roll (`exp(L·A)`, symplectic), so its
  diagonal blocks carry the `(F+D)/2` focusing (order `k1s²`). **MAD-X reproduces the
  whole transverse 4×4 to ~2e-16** (`test_betatron_coupling_madx.py`) — it does the
  same exact tilt. **xtrack's `Quadrupole(k1s)` is first-order in `k1s`**: its diagonal
  is a pure drift and only the linear coupling `R[px,y]=R[py,x]=k1s·L` is kept, so
  against xtrack we pin *that* (the sign anchor) and document the model gap rather than
  loosen a tolerance (`test_betatron_coupling_xtrack.py`). The analytic gate is the
  roll identity with teeth: the element is built directly, and the test asserts it
  equals `R(π/4)·Quadrupole·R(−π/4)`, plus `exp(L·A)` of the rolled generator, plus
  symplecticity, the `k1s→0` drift limit, and the sign flip (`test_skew_quadrupole.py`).

**`normal_mode_tunes` (coupled eigen-tunes).** A coupled lattice no longer separates
into x/y betatron oscillations; the invariant description is the eigenvectors of the
transverse 4×4 one-turn matrix `M4`. A stable symplectic `M4` has four eigenvalues on
the unit circle in two conjugate pairs `e^{±i2πQ1}, e^{±i2πQ2}`; the mode tunes are
the phases /2π, returned **fractional in [0,1)** (eigenvalues lose the integer part).
Each mode's rotation sense is fixed by the sign of its eigenvector's symplectic norm
`Im(v* J v)` — the standard convention that maps a conjugate pair to a single tune in
`[0,1)` rather than the ambiguous `[0,½]` `acos` value. Modes are labelled by dominant
plane, so in the uncoupled limit `(Q1,Q2)=tunes() mod 1` **exactly** (analytic gate).
Raises `UnstableLatticeError` if any eigenvalue leaves the unit circle (a coupled
instability the per-plane `is_stable` cannot see). **The uncoupled Courant-Snyder path
is now guarded**: `match_periodic`/`tunes`/`natural_chromaticity`/… raise
`CoupledLatticeError` on a lattice with nonzero off-block terms rather than return
decoupled-but-wrong betas/tunes (`_require_uncoupled`, `atol=1e-9`; a no-op for the
exactly block-diagonal drift/quad/dipole/sextupole lattices).

**`closest_tune_approach` = `|C⁻|` (the analytic gate).** As a ring is tuned toward
the difference resonance `Qx=Qy`, the two mode tunes **repel** — they cannot cross —
and their minimum gap is the modulus of the difference coupling coefficient

    C⁻ = (1/2π) Σ_j (k1s·l)_j √(β_x β_y)_j exp(i(μ_x−μ_y))_j,

summed over skew sources `j`, with `β`/phase from the **unperturbed** (coupling-off,
`_decoupled`) optics at each source; a thick `SkewQuadrupole` is trapezoid-sliced.
**The `1/2π` prefactor and the geometric mean `√(β_xβ_y)` are derived, not recalled**:
the exact eigen-tune split of a single-skew-kick model gives cos(mode)=cos μ ±
½√(β_xβ_y)k sin μ, hence a tune gap `√(β_xβ_y)k/(2π)` (re-derived symbolically inside
`test_betatron_coupling.py::test_cminus_prefactor_derived_symbolically`). Validation is
**triple-pinned** and non-circular: the closed form vs the **exact** `normal_mode_tunes`
eigenvalue gap on a symmetric FODO (on-resonance, where the gap *equals* `|C⁻|`),
converging with an **O((k1s·l)²) relative residual** as coupling→0 (the beam-beam
tune-shift-style quadratic check); the off-resonance **hyperbola** `gap=√(Δ²+|C⁻|²)`;
and the thick path gated separately against the eigen-gap. xtrack's coupled 4D Twiss
`qx,qy` reproduces both the mode tunes (~1e-4) and `|C⁻|` (~3e-2) on the on-resonance
ring (`test_betatron_coupling_xtrack.py`).

**Scope, stated honestly.** Multi-source `|C⁻|` is *implemented* (the phasor sum) but
only the **single-source** case is analytically gated. The thick-skew unperturbed
optics keep the `(F+D)/2` diagonal (a higher-order choice, not the `k1s→0` drift),
validated only for **short** magnets.

### Vertical emittance from coupling — the eigen-mode sharing (G1 ε_y — implemented)

`equilibrium_emittances_coupled(lattice) → (ε_1, ε_2)` (in `radiation.py`) returns the
two **eigen-mode** equilibrium geometric emittances under linear betatron coupling. The
horizontal quantum excitation that alone sets `ε_x` on a flat ring is shared between the
coupled normal modes; diagonalising excitation/damping in the mode basis, with the mixing
fixed by the difference-resonance geometry `tan 2φ = |C⁻|/Δ`, gives

    G = √(Δ² + |C⁻|²),
    ε_1 = ε_x0·(G+Δ)/(2G) = ε_x0 cos²φ   (x-like mode),
    ε_2 = ε_x0·(G−Δ)/(2G) = ε_x0 sin²φ   (y-like mode — the vertical emittance),

with **`ε_x0`** the coupling-off `equilibrium_emittance` (skews → their `k1s=0` drift
limit via `_coupling_off_lattice` — exact for a thin skew), **`Δ`** the distance of the
decoupled `Q_x−Q_y` to the nearest integer (`tunes` of the coupling-off lattice), and
**`|C⁻|` = `closest_tune_approach`**. The sum is conserved exactly (`ε_1+ε_2 = ε_x0`).

**The coefficient is a correction to the roadmap's pre-committed form, resolved by
xtrack.** The G1 ε_y entry pre-committed `ε_y/ε_x = |C⁻|²/(|C⁻|²+Δ²)`. That is **wrong**:
xtrack's radiation-envelope eigen-emittances (`eq_gemitt_x`/`eq_gemitt_y`, a Σ-matrix
eigenanalysis) follow the **eigen-mode** ratio `ε_2/ε_1 = (G−Δ)/(G+Δ) → |C⁻|²/(4Δ²)` far
off resonance — a factor of **4** below the pre-committed form (and a factor of 2 below
the *projected*-emittance `½sin²2φ → |C⁻|²/(2Δ²)`, which is the beam **size**, not the
eigen-emittance a code reports). Empirically the shipped form matches xtrack's
`eq_gemitt_{x,y}` to **~1–3%** across coupling strengths on the weak-bend near-resonance
ring, both absolute and (convention-invariantly) in the ratio; the roadmap form is
~3–4× too large and is *refuted* in the reference test. The `1/4` asymptote is pinned
symbolically (series), and CONVENTIONS' own claim that this had "no clean symbolic gate"
stands for the *coefficient* — it is xtrack, not algebra, that fixes the `4`.

**Scope, stated honestly.** This is the **leading-order, equal-damping** two-mode result.
It is clean only on a **weak-bend** ring, because two effects both grow with bend
strength: (i) `ε_x0` is an integral-formula emittance while xtrack's is a damped-map
envelope — they diverge from ~3% (weak) to ~3× (3×-stronger bends, verified uncoupled);
(ii) a skew at finite `D_x` also generates **vertical dispersion**, a second `ε_y` source
the sharing model does not carry. Both are ≲ few % on the gated ring (`h≈0.011 m⁻¹`,
`J_x≈0.997`). The full **radiation-envelope (Σ-matrix / Lyapunov) eigen-emittance**
(roadmap option B) — which would drop both the equal-damping assumption and the
vertical-dispersion blind spot — remains the rigorous alternative, reserved. Gates:
`tests/analytic/test_coupling_emittance.py`, `tests/reference/test_coupling_emittance_xtrack.py`.

### Coupled Twiss — the Edwards-Teng normal-mode optics (G2 — implemented)

G1 left a hole it created on purpose: `_require_uncoupled` makes `match_periodic` /
`closed_twiss` / `tunes` raise `CoupledLatticeError` on a skew lattice, which is right
(better than decoupled-but-wrong betas) but left such a ring with **no beta functions
and no beam sizes at all**. G2 fills it with the Edwards-Teng decomposition.

**The factorisation.** The transverse one-turn map is written as

    M4 = V U V^-1,   V = [[gamma_c I, C], [-C+, gamma_c I]],   U = diag(A, B),

with `C+ = adj(C) = -J C^T J` the symplectic conjugate (for 2x2, the adjugate, so
`C C+ = det(C) I`). `V` is symplectic **iff** `gamma_c^2 + det C = 1`. `A` and `B` are
then ordinary 2x2 Courant-Snyder blocks, one per betatron **normal mode**, and
`match_periodic_coupled` reads `(beta_1, alpha_1)`, `(beta_2, alpha_2)` off them with
the same `_matched_block` the uncoupled path uses.

**The closed form is derived, and the remembered one was wrong.** Requiring the
off-diagonal block of `V^-1 M4 V` to vanish gives, in blocks `M4 = [[m, n], [p, q]]`
and with `X = C/gamma_c`, the matrix **Riccati equation**

    n + m X - X q - X p X = 0.

Its root is proportional to `H = n + adj(p)` — `X = lambda H` — and with
`Delta = (tr m - tr q)/2`, `R = sqrt(Delta^2 + det H)`:

    lambda    = -sgn(Delta) / (|Delta| + R)
    gamma_c^2 = 1 / (1 + det X) = 1/2 + |Delta| / (2 R)
    C         = gamma_c X = -sgn(Delta) H / (2 gamma_c R)

The textbook form recalled at the start of this milestone had `C = -sgn(Delta) H /
(gamma_c R)` — missing the **factor 2** — which breaks `gamma_c^2 + det C = 1` by
O(1) (`0.58` on the first test ring). It was caught *before* implementing, by checking
the symplectic constraint numerically first. `lambda` is re-derived symbolically inside
`test_coupled_twiss.py::test_riccati_root_derived_symbolically` (sympy solves the
Riccati), so the shipped constant is gated, not trusted.

**Conventions pinned here.**
- Taking `|Delta|` (not `Delta`) selects `gamma_c >= 1/sqrt(2)`, i.e. `V` is the
  *smaller* of the two possible rotations. That makes **mode 1 the x-like mode**, the
  same labelling `normal_mode_tunes` uses (dominant plane), so the two agree.
- `coupling_angle` `phi = arccos(gamma_c)` lies in `[0, pi/4]`; `det C = sin^2 phi`.
  `phi = pi/4` (`gamma_c = 1/sqrt(2)`) is full mixing and is reached **only** on the
  difference resonance — a symmetric FODO gives it exactly, for *any* skew strength,
  which is the resonance's signature.
- Exactly on resonance (`Delta = 0`) the branch `sgn(0) = +1` is arbitrary and the 1/2
  labels may swap relative to the eigenvector route; the *pair* is still well defined.
- **How an unstable coupled lattice actually fails:** via `_matched_block` on a
  normal-mode block (`|½Tr(A)| ≥ 1`). That is what fires on G1's known coupled-instability
  ring, where the discriminant stays *positive* (gated, including the assertion that the
  discriminant branch does **not** fire). The `Delta^2 + det H < 0` raise is a
  **defensive** guard: a scan over FODO rings (symmetric and split-tune, thin skews
  `0.05`–`2`, including near the sum resonance) found no lattice reachable with the
  current element set that triggers it, so it is documented as unexercised rather than
  claimed as the instability path.

**Propagation is by local re-match, not transport.** `propagate_coupled_twiss` rebuilds
the **local** one-turn map `M(s) = T(s) M(0) T(s)^-1` at each element boundary and
re-decomposes. That is exact and needs no transport rule for `C`. **Scope:** no mode
*phase* is accumulated, so it yields no tune (use `normal_mode_tunes`); and mode labels
are per-point, so a ring sitting exactly on the difference resonance can swap them
between points. Off resonance the labelling is stable (gated: mode 1 stays within 0.5%
of the uncoupled `beta_x` around the whole ring at weak coupling).

**Projected vs eigen — the distinction that bites.** `coupled_beam_sigma` returns what a
screen measures: `Sigma = V diag(e1 B1, e2 B2) V^T`, dispersion added in quadrature,
plus the ellipse `tilt = 1/2 atan2(2<xy>, <x^2>-<y^2>)`. With coupling,
`sigma_y > sqrt(e2 beta_2)` — mode 1 leaks into the vertical plane. This is *not* what
`equilibrium_emittances_coupled` returns (eigen-emittances), exactly as the G1 ε_y entry
warned; both are now available and the difference is gated. With `e2 = 0` the leaked
`sigma_y` is linear in `k1s` **far from resonance only** (verified to 1e-4 at
`|C^-|/Delta <= 0.11`); as `|C^-| -> Delta` the mixing saturates and the growth falls
below linear (1.75x per doubling at `|C^-|/Delta = 0.79`) — the test asserts both, so
the linear claim states its own regime.

**Coupled dispersion comes for free.** The matched dispersion is solved from the full
coupled 4x4 (`D = (I - M4)^-1 d`), so a skew quad at nonzero `D_x` produces **vertical
dispersion**. Its **magnitude is xtrack-validated**, not merely its scaling: on a
dipole ring where `D_y` reaches ~0.07 m, `D_x, D_y, D_px, D_py` agree with xtrack's
`dx/dy/dpx/dpy` to `< 2e-5` around the whole ring, with no `β₀` factor and the same
sign (the analytic suite alone could only show `D_y = 0` at `k1s = 0` and linear growth,
which a wrong overall factor would survive). That is the second `ε_y` source the G1
sharing model does not carry; G2 exposes it but does **not** feed it back into
`equilibrium_emittances_coupled` — that stays reserved for the radiation-envelope
(option B) work.

**xtrack cross-check (`tests/reference/test_coupled_twiss_xtrack.py`).** xtrack reports
**Ripken** betas; the dictionary from the ET parameters is
`betx1 = gamma_c^2 beta_1`, `bety2 = gamma_c^2 beta_2`, `betx2 = (C B2 C^T)_00`,
`bety1 = (adj(C) B1 adj(C)^T)_00`. On an off-resonance FODO with a thick skew
(`k1s = 0.02`): mode betas **1.7e-6**, cross terms (which pin `C` itself) **8e-5 / 3e-5**,
mode tunes `<1e-5` — and the same at *every* boundary around the ring (worst 1.75e-6),
so the propagation is pinned too. The residual is xtrack's documented first-order-in-`k1s`
skew model, and it **scales as `k1s^2`** (1.07e-7 -> 1.72e-6 for a 4x stronger skew, a
factor 16.07), which is asserted as its own test — that scaling is what separates a known
model gap from a sign/prefactor error.

Gates: `tests/analytic/test_coupled_twiss.py` (38 tests),
`tests/reference/test_coupled_twiss_xtrack.py` (5).

## Tune matching (H1 — implemented)

`accsim.matching.match_tunes` drives two quadrupole families to a target
`(Q_x, Q_y)`. The one piece of physics is the **tune response matrix**
(`tune_response_matrix`): a gradient perturbation `dk1` shifts the tunes by the
β-weighted first-order integral

    dQ_x = +(1/4π) ∮ β_x dk1 ds,      dQ_y = −(1/4π) ∮ β_y dk1 ds,

i.e. **more focusing in x raises `Q_x` and lowers `Q_y`**. This is the same
perturbation integral as the natural chromaticity above, where the perturbation
is the off-momentum weakening `dk1 = −k1·δ` — which is exactly why
`natural_chromaticity` carries the opposite sign and an extra factor `k1`.

Sign, coefficient and weighting are **not** taken from a remembered formula: the
gate differentiates `Q(v) = acos(½·Tr M(v))/2π` symbolically from the thin
one-turn map and matches the β-form to ~1e-12. That derivative knows nothing
about β, about `4π`, or about the family weights, so agreement pins all three at
once.

**Approximate Jacobian, exact residual.** The response matrix is first order (β
itself moves as the strengths move), but the Newton residual comes from the exact
`tunes()`, so the *fixed point* is exact — the matcher converges to strengths
that hit the target to ~1e-12 in tune units, not to first-order strengths. The
Jacobian is recomputed every iteration. The acceptance test therefore asserts the
recovered **strengths** against a known lattice, not merely a small residual.

**Targets are full tunes, integer part included.** `tunes()` accumulates phase
advance through `propagate_twiss` rather than reading the one-turn trace, so
`Q_x = 6.28` is expressible and distinct from `0.28`.

**Knob semantics: `strength_i = w_i · v`** (MAD-X expression semantics, weights
default `1.0`). This is the only form that handles both cases a matcher meets: a
family split into half-quads at the ends of a cell (weights `0.5` — a purely
*additive* knob would desynchronise them) and a family starting from **zero**
strength (which a purely *multiplicative* knob could never move). `k1` [m⁻²] and
`k1l` [m⁻¹] are different units, so one knob may never span thick and thin
members — refused, not silently coerced.

**Newton must backtrack.** A first-order step can overshoot the stability
boundary, where `tunes()` *raises* rather than returning a wrong number. Each
step is halved until it is both stable and residual-reducing. The gate **counts**
the unstable excursions (weak quads `f = 3` driven to `Q = 0.35` overshoot exactly
once) rather than only asserting convergence — measured first: most starts,
including ones deliberately placed near the boundary, never trip the branch at
all, so "it converged" alone would not have tested it.

**A knob's `value` is only meaningful while the family is ganged.** `value` reads
back from the first member, so if a caller sets another member's strength
directly after construction, the knob silently misreports where the lattice is.
Both matchers therefore re-run `check_ganged()` before reading `value`, not just
the constructor. **The two matchers fail differently without it, measured by
disabling the re-check:**

- `match_tunes` is the real hazard: the residual is evaluated on the lattice the
  matcher itself produced, so Newton converges happily to the target and
  *silently overwrites* the desynced member (`k1l = 0.4 → 0.679` on the FODO
  gate), reporting success. Nothing downstream catches it.
- `match_chromaticity` mostly catches it already — but names the wrong cause. A
  wrong baseline breaks the affine prediction, so the post-solve residual check
  fires with "the chromaticity is not affine in these knobs", a *physics* claim
  about what is really a bookkeeping error. And it only catches the desyncs big
  enough to break the prediction: a desync small enough that the post-solve
  residual still passes `match_chromaticity`'s `tol`, yet big enough to fail
  `check_ganged`'s `abs_tol`, is overwritten silently (with the shipped defaults
  `tol = 1e-9` and `abs_tol = 1e-12`, measured: `1e-8` refused, `1e-9` not). So
  here the re-check buys the correct diagnosis plus that window — not the
  difference between caught and uncaught.

The gates assert on the *message* (`"not consistent"`) for that reason, and use a
desync small enough that the start is still stable — otherwise the refusal would
come from the stability guard and the test would prove nothing.

**Mutation and rollback.** `Lattice.__init__` copies the element *list* but shares
the element *objects*, so copying a `Lattice` does not protect `k1` — matching
necessarily mutates in place. Every entry point snapshots the raw per-element
strengths and restores them if it raises, so a failed match leaves the lattice
byte-identical.

**Degeneracy is refused, not solved.** Two knobs at equivalent optics (e.g. the
two half-quads of a symmetric cell) give proportional Jacobian columns; the
condition number is checked (`> 1e10` raises `MatchingError`) rather than letting
a bare 2×2 solve return a huge meaningless step. No bounds and no `least_squares`
— plain Newton sidesteps the "converged onto a bound, reported success" trap.

### Chromaticity matching — an exact solve, not an iteration (H1)

`match_chromaticity` drives two **sextupole** families to a target `(ξ_x, ξ_y)`.
The two halves of H1 are deliberately **not** the same algorithm, and the reason
is physics, not taste:

> A sextupole's *linear* map is a drift, so changing `k2` moves neither `β`, nor
> the dispersion, nor the tunes.

The total chromaticity is therefore **strictly affine** in the sextupole
strengths, `ξ(v) = ξ(v₀) + S·(v − v₀)`, with `S` a genuine constant rather than a
local linearisation — so the answer is one solve, `v = v₀ + S⁻¹(target − ξ(v₀))`,
exact from any starting point including `k2 = 0`. Newton here would be machinery
pretending the problem is harder than it is.

    dξ_x/dv_j = +(1/4π) Σ_i w_i ∮ β_x D_x ds,
    dξ_y/dv_j = −(1/4π) Σ_i w_i ∮ β_y D_x ds

(the same `+`/`−` split as the feed-down above — the `x² − y²` structure of the
kick, which is exactly what lets two families pull two negative natural
chromaticities toward zero independently). Targets are the **total** ξ (natural +
feed-down), the quantity `chromaticity()` returns and a real machine measures;
the natural part is the `k2`-independent constant of the affine relation.

**The affineness is asserted, not assumed.** The gate checks `S` computed at two
different `k2` baselines is identical to ~1e-16, that `ξ(v) − ξ(0) = S·v` for a
large arbitrary `v`, and that the post-solve residual is at machine precision
(measured **1.1e-16**) rather than merely inside a tolerance. A miss is raised as
an error, not iterated over — it would mean affineness is broken, which for
sextupoles cannot happen. `chromaticity_response_matrix` mirrors
`chromaticity()`'s own quadrature term for term (same trapezoid, same drift
transport of `D_x` through a thick body) precisely so that residual stays at
machine precision instead of at the `slices` discretisation error.

**Ordering is a consequence, not a convention:** match the tunes first and the
chromaticity second, because the second step provably cannot disturb the first.
A gate asserts the tunes survive the chromaticity match to 1e-13.

Cross-refusals carry the reason: sextupole knobs are rejected by `match_tunes`
("a sextupole's linear map is a drift"), quadrupole knobs by
`match_chromaticity` ("moving a quadrupole moves β, the dispersion and the tunes
as well, so the response would not be linear"). A sextupole at `D_x = 0` has no
response at all and is caught by the same condition-number check.

Gates: `tests/analytic/test_matching.py` (18),
`tests/analytic/test_matching_chromaticity.py` (18),
`tests/reference/test_matching_xtrack.py` (3). The reference gate hands the
*matched* strengths to xtrack and asks it, independently, what optics they
produce — deliberately **not** a comparison against xtrack's own matcher, which
would compare two optimisers and say nothing about the physics. Measured
2026-08-10: tunes agree to **4.0e-10 / 1.1e-9** (integer part included), total
chromaticity to **2.4e-3 / 3.6e-4** on a correction of ~1.8, i.e. ~1.3e-3
relative — accsim's first-order feed-down vs. xtrack's real nonlinear kick, the
same model difference `test_sextupole_xtrack` documents.

### Insertion matching — local optics, N knobs → M targets (H2 — implemented)

`match_insertion` matches the Twiss functions **at a point**: `β*`, a waist
(`α* = 0`), or the dispersion — the thing H1 could not ask for. A `Target` names
one of `beta_x, alpha_x, beta_y, alpha_y, disp_x, disp_px, disp_y, disp_py`, a
value, and a boundary index `at`.

**`at` indexes `propagate_twiss` boundary points, `0 .. len(lattice)`.** Elements
carry no names in this codebase, so the index *is* the identifier: `0` is the
lattice entrance, `k` the exit of element `k−1`. Out-of-range is refused with the
valid span in the message; if the point you want is not a natural boundary, insert
a zero-length element there.

**Two branches, selected by `twiss0`.** `twiss0=None` (default) is the **periodic**
case: `closed_twiss` is **re-solved at every evaluation**, so a quadrupole moves
the optics everywhere including upstream of itself. Passing a `Twiss` is the
**transfer-line** case — propagate from a fixed entrance, impose no periodicity,
and the lattice need not even be stable. That is the real insertion problem: match
from the arc cell's exit Twiss into the IP.

**The waist is a quadratic, and therefore not unique.** For the canonical
one-lens line (waist `β₀` → drift `d₁` → thin lens `u = 1/f` → drift `d₂`),
demanding `α = 0` at the exit gives — *derived in sympy from `B → M B Mᵀ`*, not
recalled —

    (d₁²d₂ + d₂β₀²)·u² − (d₁² + 2d₁d₂ + β₀²)·u + (d₁ + d₂) = 0.

Two consequences shape the whole API and gate:

- **`β*` is determined, not chosen.** One knob buys a waist *or* a value of `β*`,
  never both — a one-knob/two-target problem is over-determined **by
  construction**. Measured on the gate's geometry (`β₀=5, d₁=3, d₂=2`): the roots
  are `u = 0.1361` → `β* = 6.1495` and `u = 0.5404` → `β* = 0.6505`.
- **Newton lands on whichever root it starts nearest.** There is no "the" focal
  length to assert; the gate matches from two starts and pins each root
  separately, to 1e-12 in the strength.

**Finite-difference Jacobian — and why, when H1's is closed-form.** The tune
response is one universal integral valid for every lattice. The response of a
*local* β or dispersion is not: it depends on the quantity, on where the knob sits
relative to the observation point, and (periodic branch) on the re-solved closed
solution. Central differencing the exact `propagate_twiss` covers all of it
uniformly and works identically for a ring and a line. **Approximate Jacobian,
exact residual** still holds, so the fixed point is exact — measured: a dispersion
match lands at `1.4e-17`, machine precision, not at the differencing error. The
Jacobian is *pinned* the H1 way, against a symbolic `dβ/dv` differentiated from
the closed solution of a thin FODO — measured agreement **7.9e-11** relative
(gated at `1e-9`), the central-difference truncation floor at `h ≈ 1e-6`.

**The FD step is `h = fd_step · max(|v|, 1)`, and the floor is load-bearing.** H1
deliberately supports knobs starting at `v = 0`; a purely *relative* step would
give such a knob a zero column and the conditioning check would report a
degenerate knob on a perfectly well-posed problem. If one side of the central
difference falls outside the stability boundary the column falls back to one-sided,
**over `h`, not `2h`** — and that denominator needs its own gate, because nothing
else in the suite can see it. With an exact residual the fixed point is right
however wrong the Jacobian is, so a halved column costs iterations and nothing
else; and the gate that pins the Jacobian numerically runs on a comfortably stable
lattice where both trial points succeed. So the one-sided branches are gated
directly, by driving a trial point across the thin FODO's horizontal limit
(`trace = 2` at exactly `vd = 1`, asserted rather than assumed) with the knob
sitting `5e-7` inside it — less than one `h = 1e-6` step. The knob's *weight sign*
selects which side dies, so both branches are covered.

That gate asserts the column against the **exact one-sided quotient**, not against
a finer central difference, and the reason is measured: right at the boundary β
diverges, and the one-sided truncation error runs to **58–86 %** — *larger* than
the factor of two a `2h` bug introduces, so a comparison against a finer difference
could not tell the bug from the truncation. The exact quotient agrees bit for bit
(relative difference `0.0`), and injecting `2h` fails it by exactly ×2.

When *both* trial points are unstable it raises. The message says the stable window
is narrower than the step, and deliberately does **not** claim the knob sits on the
boundary — the gate reaches this branch with a huge `fd_step` on a knob that is
nowhere near the limit, so that conclusion would not follow from what the code sees.

**One ULP of the boundary raises `LinAlgError`, not `UnstableLatticeError`.** At
`trace = 2` exactly, `I − M₄` is singular and `_matched_dispersion`'s solve fails.
Measured width of that band: **1.11e-16** in `vd` — the knife edge only, reachable
by deliberate bisection and not by a matcher's trial points, every one of which
lands in the clean `UnstableLatticeError` region above. Left alone rather than
papered over; the `finally` restore below is what keeps it harmless.

**Default weight `1/max(|value|, 1)` — an unweighted 2-norm is meaningless here.**
β is metres and can be ~100, α is dimensionless and ~1: unweighted, the matcher
satisfies whichever target carries the biggest number first. The default is
relative for large targets and absolute for small ones, so it stays finite for the
ubiquitous `α* = 0`. Pass `weight` to prioritise.

**N ≠ M is allowed, and a least-squares floor is not success.** The step is
`lstsq` on the weighted Jacobian: minimum-norm for N > M (move the strengths as
little as the target allows, rather than picking an arbitrary point of the
solution family), least-squares for N < M. Convergence is declared **only** when
the weighted residual reaches `tol` (default `1e-12`); otherwise it raises, naming
each target and its miss. Note over-determined ≠ unreachable — the gate contains
both: `(α*=0, β*=6.1495)` is two targets for one knob and solvable exactly at the
second root, while `(α*=0, β*=3.40)` (between the two roots) is provably
unreachable and must raise.

**Sextupole knobs are refused** — a sextupole's linear map is a drift, so *no*
setting can move a local β, α or D. Same physical fact as H1's ordering argument,
used for the opposite purpose.

**`insertion_response_matrix` is the only response matrix in the package that
mutates the lattice.** `tune_response_matrix` and `chromaticity_response_matrix`
are pure integrals over the current optics; differencing has to *move* the knobs.
Each knob is restored from a raw per-element **snapshot in a `finally`** — not by
re-applying `v`, which would round-trip through `w_i·(strength_i/w_0)` and can
land an ULP away for awkward weights. The `finally` is load-bearing because the
function is public: `UnstableLatticeError` is caught inside the loop, but a
`CoupledLatticeError` (periodic branch, skew lattice) is not, and without it the
knob escapes at `v ± h` with no outer rollback to save it. Measured on the FODO
gate: `0.33333383` left behind instead of `0.33333333`.

Backtracking, ganging re-checks, degeneracy refusal and rollback are inherited
from H1 unchanged; the periodic branch additionally catches `closed_twiss` raising
mid-step, which a long first step routinely triggers (gate **counts** the unstable
excursions before asserting convergence, as in H1).

Gates: `tests/analytic/test_matching_insertion.py` (32), which covers the
dispersion in **both** branches — the periodic one is different code (the
re-solved `_matched_dispersion` of the one-turn map, not affine transport from a
fixed entrance) and is where a real dispersion target is used — and all three
finite-difference fallbacks at the stability boundary (both one-sided branches
and the both-unstable refusal);
`tests/reference/test_matching_insertion_xtrack.py` (4). The reference gate covers
**both** branches — xtrack's *open* twiss from the same entrance Twiss for the
line, its closed twiss for the ring — and includes the **untargeted** y plane,
which has no target to hide behind. Measured 2026-08-10 (xtrack 0.106.4): line
`β*` **8.9e-16** relative, `α*` **8.5e-16** absolute, untargeted y **6.7e-16** /
**0.0**; ring `β_x` **7.8e-16**, `β_y` **6.7e-16**. Machine precision, not a
tolerance — unlike H1's chromaticity gate (2.4e-3) nothing here is a first-order
formula: both codes evaluate the same thick-element linear optics.

## Closed orbit & its correction (I1 — implemented)

`src/accsim/orbit.py`, always-on baseline (numpy only). Everything else in the
package describes motion *about* the design orbit; this is where the beam
actually is.

**The element map is affine, and that is a new contract.** `Element` gained a
concrete `kick(ref) → (6,)`, zero for every element but a `Corrector`, and
`track()` is now `matrix(ref) @ state + kick(ref)`. A constant deflection —
the same angle whatever the coordinates — is *inhomogeneous* and cannot appear
anywhere in a 6×6 acting on `(x, px, …)`, so it needs its own slot. `Corrector`
is thin (`length = 0`), `kick_x`/`kick_y` are angles [rad], and its `matrix()` is
the **identity** — that is physics, not a placeholder: a dipole kick moves the
closed orbit and leaves the map *about* it alone, so β, the tunes, chromaticity
and dispersion are untouched. Steering and optics stay separate handles.

`closed_orbit` checks the conditioning **before** the zero-kick shortcut, so the
contract does not depend on the machine being imperfect: on an integer tune a
kick-free lattice does have zero as *a* fixed point but not the only one, and it
keeps `orbit_response_matrix`'s zeroed baseline on the same code path as its
unit-kick columns.

**`track_bunch_losses` hoists the per-element map out of the turn loop**, and
skips the kick add entirely when the kick is zero (every element but a
corrector). That inner loop runs `n_turns × len(lattice)` times — 10⁵ and up in
the long-term gates — and the hoist made it *faster* than before the affine map
existed: measured 2000 turns × 31 elements × 200 particles, **1.416 s → 1.021 s**
(28 %) with no corrector, 1.508 s → 1.073 s with one.

**Composition transports the kick.** `Lattice.transfer_map() → (M, k)` follows
the same right-to-left rule as `transfer_matrix`:

    x → M₂(M₁x + k₁) + k₂ = (M₂M₁)x + (M₂k₁ + k₂),

so a kick is carried by everything *downstream* of it, and the same kick at two
places closes into two different orbits. The linear `Tracker` paths
(`track`, `track_bunch`, `track_turns`) now go through this; before, they would
have silently dropped every corrector kick while the element-by-element path
kept it, breaking the promise that the two agree.

**The closed orbit is the same solve as the dispersion.** `closed_orbit` solves
the fixed point `(I − M₄)x_co = k₄` on the 4D transverse subspace at `δ = 0` —
literally `_matched_dispersion`'s `D = (I − M₄)⁻¹d` with the corrector kicks in
place of the map's `δ` column. Sharing that algebra is the physics: dispersion
*is* the closed orbit of an off-momentum particle, and by linearity a particle at
`δ` rides `x_co + D·δ`. `I − M₄` is singular exactly on an **integer tune**,
where a kick repeats in phase every turn; `closed_orbit` raises
`ClosedOrbitError` above a condition number of `1e12` rather than return a huge
meaningless orbit. The textbook single-kick form
`θ√(β_kβ(s))/(2 sin πQ)·cos(Δψ − πQ)` is a *consequence* here, never the
implementation — it is derived in sympy in the gate and the exact solve checked
against it (the G2 lesson: verify a recalled closed form against its exact
invariant, don't implement it).

**The response matrix is exact, not a finite difference.** The closed orbit is
strictly *affine* in the corrector kicks, so `orbit(θ) = orbit(0) + Rθ` holds at
any amplitude with no truncation error (gated at θ = 0.3 rad). Column *j* is
taken by setting corrector *j* to one radian and subtracting the baseline;
`orbit_response_matrix` is therefore the **second** response matrix in the
package that mutates the lattice (after `insertion_response_matrix`), restoring
every corrector from a snapshot in a `finally`. Kicks it does not own — a
steering *error* being corrected against — cancel in the baseline subtraction to
round-off, not algebraically: `(col + base) − base` costs the last bit or two.

That exactness makes `correct_orbit` **one linear solve**, not an iteration —
the same structural fact that makes `match_chromaticity` an exact solve.

**`rms_after` is measured, never predicted.** It comes from re-solving the closed
orbit of the corrected lattice. This matters because `correct_orbit` accepts a
supplied `response`: a real machine *measures* its response matrix, and that
measurement disagrees with the model. Evaluating `x0 + R·dθ` would report a
perfect correction for **any** invertible `R`, right or wrong. Gated by handing
the solver `1.5R` and asserting the reported residual is `rms_before/3`, while
the prediction it did not use is `< 1e-16`.

**Two correctors annul a steering error — outside the bump.** The closed orbit is
fixed by two numbers per plane, so two independent correctors can zero it
completely (measured 2.2e-19 at 14 monitors against 2 knobs — over-determined is
not unreachable, the H2 lesson). Between the error and the last corrector the
beam is genuinely off axis: that arc *is* the closed bump, and a gate asserts it
stays there (7.5e-4) rather than claiming "orbit zero everywhere".

**SVD truncation, made non-vacuous.** With as many correctors as monitors the
plain solve is exact and truncation is never exercised, so the gate carries N > M
and N < M *and* a near-degenerate pair — two correctors split by a 1 mm drift,
hence nearly the same betatron phase, giving σ₁/σ₂ = 3157. Untruncated, the
least-squares answer is *mathematically better* and asks those steerers for
**0.66 rad** (38°, which no corrector magnet delivers) to buy a 32 % improvement;
`n_singular=1` asks **6.8e-5 rad** and still improves the orbit — a norm ratio of
9752. Both facts are asserted, since only their combination is the point. Two
correctors either side of a *thin quad* (which advances no phase) are exactly
rank 1; the round-off cutoff drops that direction by itself.

**Corrector sign — pinned in the reference suite only.** Every analytic
reference for the sign is one accsim also derives, so a sign gate there would be
self-confirming (G1's pre-committed coefficient was wrong and xtrack fixed it).
Established empirically 2026-08-10:

    accsim Corrector(kick_x = +k)  ==  xt.Multipole(knl = [−k])
    accsim Corrector(kick_y = +k)  ==  xt.Multipole(ksl = [+k])

The asymmetry is the MAD-X multipole convention — `knl[0]` is the *normal* dipole
component and carries the bend sign (`px −= knl[0]`), `ksl[0]` the skew one
(`py += ksl[0]`). A test asserts the other horizontal choice is decisively wrong
(the exactly negated orbit, a ~2 mm error), so the gate is not vacuous.

**Scope, stated plainly.** Linear, `δ = 0` orbit theory. Sextupole feed-down was
out of scope here and is now **I2** (below), which is where the qualification
"correctors do not move the optics is a linear-order, on-axis-sextupole statement"
is discharged. Misalignments are not modelled as such — a quadrupole displaced by
`dx` gives a kick `−k1·L·dx`; place an explicit `Corrector` of that angle.
Correction is per plane (`plane='x'` / `'y'`); a coupled lattice is out of scope.

Gates: `tests/analytic/test_orbit.py` (27), `tests/analytic/test_orbit_correction.py`
(22), `tests/reference/test_orbit_xtrack.py` (5). The affine path through
`track_bunch_losses` is gated separately (a corrector steers a bunch into a
collimator it otherwise clears), since that loop walks the elements itself rather
than going through `Element.track`. xtrack's *iterative* closed-orbit
search agrees with the closed-form solve to **1.9e-15 m** on a 1 mm orbit (1.6e-12
relative) — the floor is xtrack's iteration, not accsim, whose own residual is
exact — and confirms the corrected machine is flat outside the bump and still
bumped inside it.

## Sextupole feed-down on a distorted orbit (I2 — implemented)

The deferral I1 named by name. Expanding J1's kick about an orbit offset
`(x_co, y_co)` — `x = x_co + X`, `y = y_co + Y` — splits **one sextupole into four
elements**, every coefficient derived symbolically (`test_feeddown_expansion_is_derived`),
never recalled:

| term | strength | equals the element |
|---|---|---|
| dipole | `θ_x = −½·k2l·(x_co² − y_co²)`, `θ_y = +k2l·x_co·y_co` | `Corrector` |
| normal quad | `k1l_eff = +k2l·x_co` | `ThinQuadrupole` |
| skew quad | `k1sl_eff = +k2l·y_co` | `ThinSkewQuadrupole` |
| sextupole | unchanged | itself |

Those three elements are separately validated (two of them against xtrack), so the
decomposition borrows their credibility instead of asserting its own.

**Sign trap — the expansion's `θ` is not the equivalent element's kick.** The table
above is the Taylor series in the *betatron deviation* `X = x − x_co`. A
`ThinQuadrupole` placed in a lattice acts on the **laboratory** `x` and so already
delivers `−k1l_eff·x_co` at the orbit; cancelling that back out means the
*equivalent lattice* needs `Corrector(+½·k2l·(x_co² − y_co²), −k2l·x_co·y_co)` —
the same coefficients with the **opposite sign**. Both are right; they answer
different questions. Getting this wrong builds a plausible equivalent lattice that
is silently wrong, so `_equivalent_lattice` spells it out.

**The orbit stops being a solve.** `θ_x` depends on the very orbit it displaces, so
the closed orbit is the fixed point of a *nonlinear* map, not `(I − M4)x = k4`.
`closed_orbit_nonlinear` Newtons on the tracked map. Three conventions:

- **4D, `zeta = delta = 0`**, mirroring `closed_orbit`. Not a liftable restriction:
  without RF there *is* no longitudinal fixed point (`R56` leaves `zeta → zeta +
  const`), so `J − I` is exactly singular in that block and 6D Newton has nothing
  to converge to. J1's test dodged this by pinning `zeta`/`delta` inside its turn
  map; I2 iterates on the transverse subspace instead.
- **Seeded from `closed_orbit`** — the correct first-order guess, so a linear
  lattice reproduces I1's answer at round-off for free.
- `OrbitConvergenceError` **subclasses** `ClosedOrbitError`, so I1-era callers that
  roll back on "no orbit" keep working, but it says something different: out of
  budget, not eigenvalue 1.

**Two honest non-claims, both gated.** Feed-down is **self-limiting** — raising
`k2l` by 10⁵ *shrinks* the orbit, because the same gradient stiffens the `(I − M4)`
being inverted; convergence is therefore **not** evidence of a stable machine
(closure needs `(I − M4)` invertible, not stable). And the fixed point is **not
unique**: started 50 m out, Newton converges onto a genuinely different orbit
(0.074 m — an outer, unstable one). The docstring claims nothing about which one a
far guess finds, and a test asserts both really are fixed points.

**`linearised_element_maps` is the optics primitive, not a one-turn Jacobian.**
`propagate_twiss` calls each element's *on-axis* `matrix()` and would miss feed-down
entirely, so the per-element Jacobian about the propagated orbit is what optics are
read from. Their product in beam order *is* the one-turn Jacobian by the chain rule
(gated to bit-equality against `linearised_one_turn_map`, and to `1e-8` against a
whole-turn finite difference). Every linear element comes back as its own matrix, so
the sextupole is the only place the linearisation differs.

### What gates the coefficient — and why the tune shift is not the lead

`ΔQ_x = +β_x·k2l·x_co/(4π)` (mirror `−β_y·…` in y) is **demoted to a consistency
check**: J1 already measured `k1l_eff` by finite-differencing `track()` about a
dispersive offset, and redoing it about a corrector-induced one is the same
measurement. It is made non-vacuous anyway — **four** sextupoles at two different
`β_x` with alternating-sign `k2l`, so the sum has real cancellation (asserted:
`|Σ| < 0.6·Σ|·|`) — and its residual is checked to fall **quadratically** with `k2l`.

**The lead gate is the dipole term, because it has no J1 analogue.** The departure
`x_nl − x_lin` must equal I1's *linear* response to a `Corrector` of the derived
`θ`, computed from the **linear** orbit. The content is the **order**, not the
magnitude: measured over four steerer sizes, the departure falls by **4** per
halving (`O(x_co²)`) while the residual falls by **8** (`O(x_co³)`). A mis-scaled
kick cannot satisfy both — a consistently doubled sextupole (J1's `_Misscaled`,
which passes every structural check) is caught as a clean factor of 2.

**And a ratio J1 structurally could not have.** `θ_x / k1l_eff = −x_co/2` is **pure
geometry** — no `k2l` in it — measured off the tracked map as (constant part of the
kick)/(slope of the kick). J1 only ever saw the gradient, and a gradient alone fixes
the product `k2l·x_co`, never the split between the two terms.

**β at the source, exactly, not a recalled β-beat form** (the G2 trap). The test
ring is a **palindrome**, so `α = 0` where the sextupole sits and the perturbed map
there is just `M0·Q`. The 2×2 is solved in sympy with no expansion:
`(β′/β)² = sin²μ / (1 − (cos μ − k1l·β·sin μ/2)²)`. It is compared **squared** on
purpose — `M12 = β·sin μ` is *negative* in this ring (fractional tune > ½), and
squaring keeps the statement exact instead of asserting a `sin μ` branch. The gate's
content is *localisation*: feeding it the other plane's `β_k` is rejected.

### The vertical orbit — the strongest non-rerun gate

J1 only ever exercised feed-down in the horizontal, dispersive plane. A vertical
orbit makes a *normal* sextupole a **skew** quadrupole, i.e. a coupling source,
reaching G1/G2's machinery from a direction nothing in the package had taken. Three
consequences, all gated:

- **`y = py = 0` is an exact invariant subspace** of the kick (at `y = 0` it is
  `Δpx = −½k2l·x²`, `Δpy = 0`), so a horizontal bump keeps the orbit planar at
  **exactly zero**, not to tolerance. That is what makes the vertical results
  attributable to the vertical bump rather than to leakage.
- **A purely vertical steerer moves the horizontal orbit**, through
  `θ_x = +½·k2l·y_co²` — flatly impossible linearly (xtrack reports *exactly* `0.0`
  at `k2l = 0`). Its sign is **opposite** to the horizontal case: the `x² − y²`
  structure showing up in the orbit rather than in the tune.
- **`θ_y = +k2l·x_co·y_co` needs both planes**; no single bump switches it on.

### Orbit correction becomes a loop — and it converges *linearly*

The operational punchline. `orbit_response_matrix` stays the affine **model**
response (its "exact, not a finite difference" claim is now explicitly scoped to
the linear lattice); `correct_orbit` gained `nonlinear=False`, which when `True`
measures `x0` and `rms_after` from the nonlinear orbit. One pass then leaves an
`O(k2l·x_co²)` residual (measured: falls by **4** per halving of the error kick)
instead of I1's machine zero, because the model `R` knows nothing about the dipole
the steering itself created.

**The convergence is linear, not quadratic, and that distinction is physics.** `R`
is rebuilt from the *linear* model every pass, so it never learns the feed-down
gradient: the loop is a stale-Jacobian fixed-point iteration whose contraction
factor is **constant** rather than shrinking — measured at **4.951e-4**, identical
to 4 digits over three consecutive passes (2.9e-4 → 1.5e-7 → 7.5e-11 → 3.7e-14). A
true Newton, relinearising each pass, is what would be quadratic. The test asserts
the factor *repeats*, which is far sharper than "it gets smaller". Control: the same
machine at `k2l = 0` still lands on machine zero in one pass, with a square
3-corrector/3-monitor problem so "did not reach zero" cannot be least-squares
over-determination in disguise.

`nonlinear=False` remains bit-for-bit I1 — including, deliberately, reporting
machine zero on a machine whose real orbit is not zero. That blind spot is asserted
rather than left to be discovered.

**Reference cross-checks.** Element equivalences reused from I1/J1, not re-probed
(`Corrector(kick_x=+k) ≡ knl=[−k]`, `kick_y=+k ≡ ksl=[+k]`,
`ThinSextupole(k2l) ≡ knl=[0,0,+k2l]`). All figures **measured 2026-08-10**, with
the tolerance and its headroom stated rather than chosen:

| quantity | measured | tolerance | headroom |
|---|---|---|---|
| closed orbit, both planes steered, 15 boundaries | `1.6e-13 m` on a 1.1 mm orbit (`1.4e-10` rel) | `1e-12` | 6.4× |
| `R_matrix` vs `linearised_one_turn_map`, 4×4 | `2.1e-11` on entries up to 7.0 | `1e-9` | 47× |
| accsim's own FD floor (`k2l = 0` vs exact analytic 6×6) | `7.2e-12` | `1e-10` | 14× |

The matrix floor belongs to **xtrack's** differencing, not accsim's: varying
accsim's `step` over `1e-6`/`1e-7` moves the discrepancy by `0.1 %` (`2.149e-11` →
`2.147e-11`), i.e. it is insensitive to accsim's choice, and xtrack's own
`steps_R_matrix` is `dx = 1e-6`, `dpx = 1e-7`. Tune shift `2e-3` rel, β `2e-5` rel,
and `tw.c_minus` confirms the vertical bump couples (`> 1e-4`) while the horizontal
one gives `0.0`. **The gate that makes the rest meaningful**: I1's linear solve
misses xtrack by `> 1e4 ×` the tolerance the nonlinear one meets — these are not
measuring round-off.

**Still out of scope:** the 6D (RF-coupled) closed orbit; feed-down from octupoles
and higher multipoles; misalignments as element attributes; amplitude-dependent
detuning and resonance driving terms; dynamic aperture. And explicitly —
**`chromaticity()` is not corrected for feed-down**. It is a *design-orbit*
quantity: `propagate_twiss` walks on-axis `elem.matrix()`, so a steered machine is
evaluated with unperturbed β and dispersion, wrong at the β-beat level I2 measures
(~0.4 % in `β_x` for a 0.3 mm orbit at `k2l = 20`). `linearised_element_maps`
supplies everything a corrected version would need; I2 does not build the Twiss
propagation on top of it. `test_chromaticity_is_a_design_orbit_quantity` asserts the
blind spot so it is documented rather than discovered — the same treatment
`correct_orbit(nonlinear=False)` gets. **I3 closes exactly this gap** (see *Optics
on the real (steered) orbit* below); `chromaticity()` itself stays a design-orbit
quantity, and the blind-spot test stays with it.

Gates: `tests/analytic/test_feeddown.py` (24),
`tests/reference/test_feeddown_xtrack.py` (6). The quantitative gates all use
`ThinSextupole` deliberately — a thick body's orbit varies across the magnet, so a
thin-lens sum would carry an `O(L²)` error that would read as a loosened tolerance —
but a thick `Sextupole(n_slices=4)` is separately gated through the same machinery
(Newton converges, the fixed point holds, the departure is still `O(x_co²)`).

## Optics on the real (steered) orbit (I3 — implemented)

I2 shipped the per-element maps about the distorted orbit and asserted the gap it
left: every optics function still walked on-axis `elem.matrix()`, so a steered
machine was reported with unperturbed β, dispersion and chromaticity. I3 builds the
Twiss propagation on top of those maps.

**API.** `propagate_twiss` and `propagate_coupled_twiss` gain `maps=` — one 6×6 per
element, in beam order, substituting the transport and nothing else (the lattice is
still needed for `s`). On top of that: `closed_twiss_on_orbit`,
`propagate_twiss_on_orbit`, `tunes_on_orbit`, `coupled_twiss_on_orbit`,
`propagate_coupled_twiss_on_orbit`, `natural_chromaticity_on_orbit`,
`chromaticity_on_orbit`, and `orbit.linearised_lattice`. `delta=` is threaded
through `closed_orbit`, `closed_orbit_nonlinear`, `propagate_orbit_nonlinear`,
`linearised_element_maps` and `linearised_one_turn_map` — **one Newton solver, not
two**, with the linear seed carrying the dispersive column
(`rhs = k4 + M[trans, DELTA]·δ`; without it the seed starts a whole dispersion
orbit away from the answer).

Naming: `_on_orbit`, not `_nonlinear`. The Twiss is linear; what is nonlinear is
the *orbit it is taken about*.

### The gate is β(s), not β(0)

I2 already gates `match_periodic(linearised_one_turn_map(...))`, so the value at the
ring start is a rerun, and "the propagated table multiplies back to the one-turn
map" is **vacuous** — `linearised_one_turn_map` *is* the product of
`linearised_element_maps`. The new content is the `s`-dependence, gated on the
single-gradient closed form

```
Δβ(s)/β(s) = −Δk1l · β(s_src) · cos(2|Δψ| − 2πQ) / (2 sin 2πQ)     (+ in y)
```

with `Δk1l = k2l·x_co`. **Derived, not recalled** (`test_beta_beat_closed_form_is_
derived`): the Courant-Snyder transfer parameterisation is first verified against
accsim's *own* propagation rule (`B1 = C B Cᵀ`, phase from
`atan2(C12, βC11 − αC12)`) and shown to multiply back to the one-turn form; only
then is accsim's own `ThinQuadrupole` (`px → px − k1l·x`) inserted. The `sin μ`
square-root branch never appears, because `β = M12/sin μ` is differentiated via
`d(sin)/dk = −cos·d(cos)/dk / sin` — rational in `sin μ`, `cos μ`. That matters:
the test ring runs at `Q_x = 0.690`, where `sin 2πQ < 0`.

Two deliberate choices in the test ring, both load-bearing:

- The sextupole sits **mid-ring**, so sample points exist on *both* sides of it and
  the `|Δψ|` branch is actually exercised. At `s_src = 0` it never is.
- `Q_y = 0.559` sits 0.059 from the **half integer**, so the `1/sin 2πQ` amplifies
  the vertical beat ~2.6× — the vertical is the more demanding plane on purpose.
  Retuning the ring away from the half integer would silently weaken the gate.

Content is the **order**, as in I2: over four steerer sizes the beat falls by **2**
per halving (first order in the offset) while the residual against the first-order
form falls by **4**. Doubling the predicted gradient is caught 20× over.

### Chromaticity takes the other route, and the package's structure decides it

accsim's linear element maps carry **no `δ` dependence of their own** — `track()`
through a quadrupole is its `matrix()` at every momentum. So linearising the tracked
map about the off-momentum orbit measures the **sextupole feed-down term and nothing
else**: it is exactly blind to the natural chromaticity, which accsim supplies
analytically (F2). Measured: on a sextupole-free steered ring the tracked `dQ/dδ` is
`3.7e-8` (the FD floor of an identically-zero quantity) while the true natural
chromaticity is `−0.29`. **Implementing `chromaticity_on_orbit` by tracking alone
would silently drop that entire term**, and a test asserts the reason.

So the existing F2-validated integrals are run on `linearised_lattice` — each thin
sextupole joined by I2's derived split `ThinQuadrupole(k2l·x_co)` +
`ThinSkewQuadrupole(k2l·y_co)`, the sextupole itself **kept** (the split is the
*static* feed-down; the sextupole still feeds down a further `δ`-dependent
`k2·D_x·δ` at dispersion — different terms, both physical). The dipole part of the
split is absent from both routes: a Jacobian is the linear part only, and a
`Corrector`'s `matrix()` is the identity anyway, so it is invisible to every
matrix-based optics function.

`linearised_lattice` (derived coefficients) and `linearised_element_maps` (finite
differences) describe the same machine by independent routes; their agreement is
gated at `1e-10`.

**The independent gate is the pair's difference.** `chromaticity_on_orbit −
natural_chromaticity_on_orbit` is the feed-down term at the *beaten* β and `D_x` —
an analytic integral over `matrix()`. The tracked route reaches the same number
through Newton plus a finite-difference Jacobian, with no integral anywhere. J1
gated these two against each other on the design orbit; I3 extends it to a steered
machine. Measured agreement `2.2e-8` absolute on values of order 2, **flat** in the
orbit offset rather than growing with it — so the concern that the sextupole's
*dipole* feed-down (whose δ-derivative is a term of the same order) might be missing
from the equivalent lattice is settled empirically as well as by the argument that
it *is* the feed-down gradient acting on the dispersion.

Zero steering is **bit-for-bit**: at `x_co = 0` every added gradient is exactly
zero, so `chromaticity_on_orbit(lat) == chromaticity(lat)` as the same float.

**The natural half needs its own gate, because the difference cancels it.** The
tracked gate constrains `chromaticity_on_orbit − natural_chromaticity_on_orbit`, so
the natural term drops out of it entirely. On a **dispersion-free** ring it has an
exact closed form of its own — there is no sextupole feed-down chromaticity at all,
so the whole answer is the thin-lens sum `ξ_x = −(1/4π) Σ β_x(e)·k1l(e)` over the
real quadrupoles *and* each sextupole's `k1l_eff = k2l·x_co`, with `β` read off
`propagate_twiss_on_orbit` (finite-differenced from `track()`, and separately gated
against the derived beat form) rather than off the equivalent lattice. Agreement
`5e-13`.

That gate discriminates because **the two contributions oppose each other**: of a
total shift `−1.51e-3`, the sextupole's own direct term is `−3.03e-3` and the beat
acting on the pre-existing quadrupoles supplies `+1.52e-3`. Dropping the direct term
does not shrink the answer, it **flips its sign** — an implementation that beat `β`
correctly but forgot that an off-axis sextupole *is itself a quadrupole* cannot
pass. Mutation-checked (2026-08-10) against xtrack as well: doubling, sign-flipping
or omitting the feed-down gradient moves `dqx` to `−1.71e-3` / `+3.50e-3` /
`+1.70e-3`, all beyond the `5e-4` the reference test allows, against `−3.3e-5` for
the correct one.

### Scope lines, enforced rather than documented

- **Thick sextupoles raise `NotImplementedError` in `linearised_lattice`** (and so
  in the chromaticity pair): the offset varies across the body, so one entrance-orbit
  gradient would carry an `O(L²)` error — the very error I2 avoided by using thin
  sextupoles. `propagate_twiss_on_orbit` has *no* such restriction, because it
  differentiates the thick element's real `track()`.
- **A vertically steered machine is genuinely coupled** (a normal sextupole at
  `y_co ≠ 0` is a skew quadrupole), so `closed_twiss_on_orbit` /
  `propagate_twiss_on_orbit` / `chromaticity_on_orbit` raise `CoupledLatticeError`
  there rather than returning a plausible 2×2 answer; `coupled_twiss_on_orbit` is
  the path. Design coupling is exactly zero on the same lattice — a contrast
  impossible in the linear theory at any kick. Horizontal steering alone leaves the
  on-orbit map **exactly** block-diagonal (off-block norm `0.0`), so no `atol`
  loosening was needed.
- `tunes_on_orbit` returns the **accumulated** phase advance, integer part included.
  That is not cosmetic: the chromaticity gate central-differences it in `δ`, and a
  fractional-only tune from `acos` would be wrong by an integer whenever the two
  sample points straddled a half integer. The hazard is removed, not guarded.

### Finite-difference floors, measured

| quantity | measured (2026-08-10) | used as |
|---|---|---|
| `linearised_element_maps` vs exact matrix, per element | `1.9e-13` | floor of the "no sextupole ⇒ design optics" gate |
| ...their product, one turn | `2.4e-12` | ditto |
| on-orbit β vs design β, no sextupole | `<1e-11` rel | ditto |
| on-orbit β at `δ = 0` vs `δ = 1e-3`, linear lattice | `1.2e-10` rel | not bit-for-bit: the Jacobians are differenced at a dispersion-shifted state |

### xtrack cross-check — and one modelling difference stated, not absorbed

accsim's `Dipole` and `Quadrupole` are **exactly linear** maps, so an off-axis orbit
changes nothing about them. xtrack's `Bend`/`Quadrupole` are exact *nonlinear* maps
whose Jacobian at a 1.25 mm offset is not the on-axis one. That difference is **first
order in the orbit** and belongs to accsim's element models, not to I3: at `k2l = 0`
on a ring steered 1.25 mm, accsim's on-orbit optics equals its design optics to
`4e-11` while xtrack's β has moved `6.4e-4` relative.
`test_accsims_linear_elements_do_not_feed_down_off_axis` isolates and measures it, so
a future milestone giving the bends their real off-axis map has a number to improve
on.

That is why the β cross-check is a **with-minus-without-sextupole difference** (J1's
device): the bend nonlinearity is common to both terms and cancels. Measured on a
bendy 8-cell ring with one sextupole per cell, steered 1.25 mm:

| quantity | xtrack | accsim on-orbit | accsim design orbit |
|---|---|---|---|
| sextupole-induced Δβ_x | `6.6e-3 m` (0.22 % of β) | reproduced to `1.35e-3` of the effect | **exactly `0.0`** |
| `dqx` | `+2.6529347` | `+2.6529020` (`3.3e-5`) | `+2.6546373` (`1.7e-3`) |
| `dqy` | `−4.5204588` | `−4.5205851` (`1.3e-4`) | `−4.5178598` (`2.6e-3`) |

The β row is the strong form, not the weak one: the design route answers **exactly
zero, bit for bit** (a sextupole's `matrix()` is a drift, so adding one changes no
matrix anywhere), which makes this an effect that was previously invisible rather
than an improved estimate. Chromaticity needs no difference — 52× closer in `x`,
21× in `y`. Control: the *unsteered* ring agrees between the two codes to `9.3e-10`
in β, so the steered disagreement is the orbit and not the ring description.

Gates: `tests/analytic/test_orbit_optics.py` (21),
`tests/reference/test_orbit_optics_xtrack.py` (4, four cached `xt.Line` builds).

**Still out of scope:** off-axis feed-down from accsim's *linear* elements (the
bend/quad nonlinearity above); coupled (Edwards-Teng) chromaticity on a vertically
steered machine; thick-sextupole chromaticity on orbit; and everything I2 already
listed — the 6D closed orbit, octupoles, amplitude-dependent detuning, dynamic
aperture. Misalignments as element attributes were on that list until **K1**, below.

## Misalignments — transverse offsets (K1 — implemented)

Every element carries `(dx, dy)` [m]: **where the magnet actually is**, relative to
where the lattice puts it. `dx > 0` means the magnet has moved towards positive `x`,
so a particle on the design orbit passes through it at body coordinate `−dx`. The map
is the element's own map **conjugated by that translation**:

    track(state) = d + body(state − d),      d = (dx, 0, dy, 0, 0, 0)

— step into the magnet's frame, apply the magnet, step back out.

### A translation does not touch the matrix; the whole linear effect is a kick

Expanding the conjugation for a linear body,

    M (state − d) + k = M state + (k + (I − M) d),

so `matrix()` is returned **unchanged** and the entire linear content of a
misalignment is the constant term `(I − M) d`, added in `Element.kick()`. Three
consequences, all asserted at exact zero rather than to a tolerance:

- **A displacement moves no optics.** β, the tunes, dispersion and the coupling of a
  misaligned ring are *bit-for-bit* those of the aligned one. This is not a nicety:
  it is what allows K1's ensemble average over displacements to be taken at fixed
  optics.
- **A `Drift` and a `Corrector` are translation-invariant** — `(I − M) d` is
  identically zero for both (a drift moves `x` only through `px`; a corrector's `M`
  is the identity). They still *accept* `dx`/`dy`, so a lattice can be misaligned
  wholesale and the invariance asserted instead of assumed.
- **Offsets cannot couple the planes.** Both cross-derivatives of a displaced quad's
  kick vanish identically (`∂Δpx/∂y = ∂Δpy/∂x = 0`), so no displacement of an
  unrolled quadrupole produces a skew term. Only a **roll** can (K2) — this is what
  separates the two halves of axis K cleanly.

### The quadrupole case is exact, and the two planes' signs differ

A quad's gradient is uniform, so a displaced quad is **exactly** a quad plus a dipole
with no higher terms at all (remainder identically `0`), where I2 and J3 each split
one element into a family:

    theta_x = +k1l·dx        theta_y = −k1l·dy

— the **same** displacement sign giving **opposite** kick signs, because accsim's
thin quad is `px → px − k1l x` but `py → py + k1l y`. That asymmetry had already
bitten this package once (`Corrector` needs `knl=[−k]` for `kick_x=+k` but
`ksl=[+k]` for `kick_y=+k`, above), so it is derived in sympy and asserted.

For a **thick** quad the kick is the full `(I − M) d` (which also displaces `x` and
`y`); it collapses onto `(+k1 L dx, −k1 L dy)` as `L → 0`, so the thin form is a
limit, not an identity.

### The sign, by probe — `dx` is xtrack's `shift_x`

`x_rms` goes as `d²`, so **no statistical gate can see the sign of an offset**: it
cannot tell "the magnet moved right" from "the beam sits right of the magnet centre",
which is exactly the relative sign that flips silently. Fixed by probe (measured
2026-08-17), the J1/J2/J3 rule:

    ThinQuadrupole(k1l, dx=d, dy=d')  ==  xt.Multipole(knl=[0, k1l], shift_x=d, shift_y=d')
    ThinSextupole(k2l, dx=d, dy=d')   ==  xt.Multipole(knl=[0, 0, k2l], shift_x=d, shift_y=d')

**bit-for-bit** at the probe state — both codes translate, apply the same polynomial
and translate back. The probe is run as a *delta*: the aligned pair is pinned first
(also bit-identical) and the shift then added to both sides, so it isolates the one
sign K1 owns instead of re-litigating the `knl` convention. The opposite sign misses
by exactly twice the displacement kick. Across a scan of amplitudes the agreement is
a few ulp rather than bit-exact, because xtrack reaches the same polynomial through
its general `knl` recursion.

The thick quad agrees to `1.3e-7` shifted against `1.6e-7` aligned — i.e. the
misalignment adds **no** error of its own, the residual being the pre-existing
linear-matrix-vs-thick-map difference. A flipped sign misses by `3.9e-4`.

### The extension point moved: `_kick_body` / `_track_body`

`Element.kick()` and `Element.track()` are now the template methods that add the
misalignment. Subclasses override **`_kick_body`** (constant part in the element's own
frame; only `Corrector` does) and **`_track_body`** (the nonlinear map; `Sextupole`,
`ThinSextupole`, `ThinSkewSextupole`, `Octupole`, `ThinOctupole`, `RFCavity`,
`BeamBeam`). Overriding the public pair would apply the shift twice or drop it.

The trap this created, and how it was caught: the sextupole's and octupole's
zero-strength shortcut used to read `return super().track(state, ref)`, which after
K1 re-enters the wrapper and **shifts the state twice**. It is now
`super()._track_body(...)`. The gate that caught it already existed —
`test_thick_track_respects_the_affine_contract_at_zero_strength`, whose fictitious
constant-kick subclass exists precisely to make a dropped or double-counted constant
part observable.

### Nonlinear elements: the linear orbit is blind, and says so

A thin sextupole's `matrix()` is the identity, so `(I − M) d` is **exactly zero** and
`closed_orbit` returns exactly zero for a machine whose only imperfection is a
displaced sextupole — while the real map has an `O(d)` gradient and an `O(d²)` dipole
kick `−½k2l(dx² − dy²)`. This is not new blindness (it is the same statement as
"`matrix()` is the Jacobian **at the origin**", the reason `closed_orbit_nonlinear`
exists), but before K1 that zero was *right* and after K1 it is *wrong*, so it is
recorded as a verdict rather than a footnote:

- `closed_orbit` on a displaced-sextupole ring returns exactly `0`;
- `closed_orbit_nonlinear` finds the real orbit and lands on the orbit of the
  hand-assembled I2 family to `1e-14` — the two lattices have literally the same map;
- `orbit_statistics` **refuses** a displaced sextupole or octupole as a source rather
  than returning a reassuring zero, pointing the caller at the nonlinear solve.

### A displaced element *is* I2 / J3 — exactly, not to a tolerance

The consistency requirement that makes K1 a refactor rather than new physics: a
magnet displaced by `d` and a beam displaced by `−d` are the same physics, so

    ThinSextupole(k2l, dx, dy)  ==  I2's four-term family at (x_co, y_co) = (−dx, −dy)
    ThinOctupole(k3l, dx, dy)   ==  J3's six-term family at (x_co, y_co) = (−dx, −dy)

and because both splits are exact rearrangements of a polynomial (not truncations),
the agreement is **exact**. A tolerance here would hide a wrong coefficient in the
third digit. No coefficient was at risk, and none moved.

`linearised_lattice` evaluates the feed-down split at the offset **in the magnet's own
frame**, `x_co − dx`, not at the lab orbit. Without that, every misalignment would be
invisible to the chromaticity integrals, which walk element *types* rather than maps.
Both halves matter: on the ring used in the gate, dropping `−dx` gives `1e-6` instead
of `9e-4` (a factor of 900) and dropping `x_co` ignores the 0.3 µm orbit the
displacement itself created.

### A bending dipole refuses to be displaced — and the refusal is measured

K1's conjugation is a **straight**-element statement. A bend rotates the reference
frame through itself, so translating in at the entry and out at the exit are not the
same transformation: displacing a bend is a **rigid-body** motion of a curved body,
with an angular and a path-length consequence the straight formula does not have.
xtrack implements exactly that distinction — its misalignment header
(`track_misalignments.h`) takes the straight branch only when `angle == 0`, and
otherwise conjugates the displacement by the frame transport to the anchor.

Measured 2026-08-17 on `Dipole(1.0, 0.12)` with `dx = 3e-4`: the straight formula
misses xtrack by **`3.6e-5`**, where the *aligned* maps agree to **`5.8e-9`** — four
thousand times the model difference, which is not a tolerance question. So
`Dipole.kick()` / `Dipole._track_body()` raise `NotImplementedError` when `angle != 0`
and an offset is set, on the statistical path too (where the offset is set after
construction and a silent wrong model would be averaged over hundreds of machines with
nobody looking). A **straight** dipole (`angle = 0`, i.e. a pure gradient magnet) is
displaced like anything else. Represent a bend's steering error with an explicit
`Corrector`, or displace the quadrupoles — which is where a real machine's orbit comes
from anyway.

### The statistical orbit: exact ensemble average, textbook form *measured*

`orbit_statistics(lattice, dx_rms=…, dy_rms=…)` is the first quantity in the package
that is **statistical** rather than deterministic: the rms closed orbit over an
*ensemble* of machines built to a given alignment tolerance. Nothing is sampled — the
closed orbit is exactly linear in the displacements, so

    <x_co(s)²> = Σ_i (∂x_co(s)/∂d_i)² <d_i²>

with the derivatives from `misalignment_response` (one exact solve per source per
plane; `misalign(lattice, rng, …)` draws one machine from the ensemble). The cross
terms die because the displacements are uncorrelated — **that** is the ensemble
average, and it is the only assumption. Exact linearity is asserted from 1 µm to
**1 m**, six orders either side of the tolerance.

Written out with I1's single-kick closed form, the result is, still exactly,

    <x_co²>(s) = beta(s)·theta_rms²/(4 sin²(pi Q)) · Σ_i beta_i cos²(dpsi_i − pi Q)   [EXACT]

reproduced by the solve to **`4.5e-16`** relative at every boundary in both planes.
The textbook `Σ_i beta_i/(8 sin²(pi Q))` needs one further step, `cos² → ½`, and
**that step is not an ensemble average**: the phases are deterministic properties of
the lattice, and averaging over displacement samples never touches them. On the 6-cell
thin FODO used here the textbook form is **12% high** (ratio `1.120…1.123`), so the
suite computes the exact sum from the ring's own phases and measures the departure
instead of inheriting it. Distribution-free: a uniform distribution of the same rms
reproduces the same prediction, since only the second moment enters. A 400-machine
Monte-Carlo agrees within its own `~4%` standard error (mean ratio within `3%`).

**The load-bearing gate is the magnitude**, and the pole scan is *not* an independent
half — the honest statement, arrived at by noticing the algebra:

- **Magnitude** — the solve against the exact form above, prefactor and β-weighting
  included, at every boundary in both planes (`1e-12`).
- **The `Q` scan** — the same identity stressed to 0.2 % from the integer, where
  `1/|sin πQ|` has grown 150-fold and the fixed-point solve is near singular, plus the
  *exponent* of the divergence. Its divisor is the magnitude formula's own numerator,
  so `p·|sin πQ| = 0.5` follows algebraically from the magnitude identity holding at
  those working points: **the scan cannot pass while the magnitude gate fails**. Saying
  "two halves, neither able to fake the other" would be an overclaim.

The one-directional statement runs the other way, and it is what makes the magnitude
comparison load-bearing: a uniformly mis-scaled kick (the J1/J2/J3 failure mode) is
**invisible** to the scan. Built and measured rather than argued — the broken machine is
constructed and run through the *same* scan, which still gives a clean first-order
divergence, constant across `Q`, at `1.0` instead of `0.5`.

The pole scan has two traps, both walked into and recorded:

1. **Direction.** Weakening the quadrupoles to drive `Q → 0` does *not* work: a FODO
   with no focusing left is a drift ring, and it loses stability (`|½Tr| → 1`, raised
   as `UnstableLatticeError`) **before** the integer is reached. The scan therefore
   *strengthens* the quads toward `Q_total → 1`, crossing it between `scale = 1.45`
   and `1.46`.
2. **Contamination.** Retuning quadrupoles moves β and the source strengths at the
   same time as `Q`, so the raw rms is *not* a pure `1/|sin pi Q|` (it grows by only
   17× while `1/|sin|` grows by 150×). The divisor
   `sqrt(beta(0)·Σ_i k1l_i² beta_i cos²(dpsi_i − pi Q))·d_rms` is built from
   `propagate_twiss`'s **measured** numbers and contains no `sin`, so what is left is
   the pole and nothing else: `p·|sin pi Q| = 0.5` constant to **10 digits** across the
   scan, while `p·sin²` moves by a factor **153**. The exponent is pinned; the constant
   is not (by construction).

`1/|sin pi Q| → ∞` and `(I − M4)` going singular are one statement, so the statistical
entry point inherits I1's `ClosedOrbitError` at the integer rather than returning a
huge but meaningless number — checked at `scale = 0.2`, I1's own exactly-on-resonance
working point (`Q_y = 0`), so the two suites agree about where the resonance is.

The new map is asserted **symplectic** as well, as J1/J2/J3 each are. It passes by
construction (a translation is a symplectomorphism, so conjugating a symplectic map by
one stays symplectic) and that is the reason to run it: it is the check that would catch
`d` being applied *asymmetrically* in the wrapper, which no amount of orbit statistics
would reveal.

Gates: `tests/analytic/test_misalignment.py` (34, ~43 s),
`tests/reference/test_misalignment_xtrack.py` (6, six `xt.Line` builds, ~57 s).

**Out of scope for K1:** rolls (K2, below), longitudinal displacement (`ds`),
misalignment of a thick element's *body* as distinct from its ends, displaced bending
dipoles (refused, above), misalignment **correction** beyond I1's existing SVD
steering, and statistics of anything but the closed orbit.

## Misalignments — the roll (K2 — implemented)

`roll` [rad] turns an element about the beam axis `s` **while the machine stays where
it is**: MAD-X `EALIGN`'s `DPSI`, xtrack's `rot_s_rad_no_frame`. Every element carries
it, defaulting to `0`.

### There are two things called a roll, and accsim implements the error

xtrack carries both as *separate attributes*, and the distinction is the whole
milestone (measured 2026-08-17, before any code was written):

| | frame | accsim | map | kick |
|---|---|---|---|---|
| **design tilt** — MAD-X `TILT`, xtrack `rot_s_rad` | rolls **with** the magnet | not offered | conjugation `R(−φ)·M·R(+φ)` | **exactly zero** |
| **roll error** — MAD-X `DPSI`, xtrack `rot_s_rad_no_frame` | stays put | `roll` | rigid motion (below) | first order in `φ` |

The design tilt is a lattice *design* choice (it is how you build a vertical bend),
not a misalignment, and it is out of scope for axis K. It really is the plain
conjugation: xtrack's `rot_s_rad` reproduces a hand-built `Rotation · Bend · Rotation`
sandwich to a few ulp, with **no kick at all** — the reference particle still comes out
on the design orbit, because the design orbit was rolled along with it. Assuming "a
simple rotation will do" for a *misalignment* would therefore have shipped a model that
predicts **no vertical closed orbit whatever**.

### Straight elements: a conjugation, and already-known physics

For every element that does not bend, the two rolls coincide and the map is

    track(state) = R(−roll) · body( R(+roll) · state ),

with `R` the passive frame rotation (`alignment.s_rotation`, byte-identical to
xtrack's `SRotation`/`Rotation(rot_s_rad=…)`). Two consequences are asserted against
machinery that predates axis K rather than being taken on trust — J3's angle rule
`π/(2(n+1))`, with the **sign** as the content:

    ThinQuadrupole(k1l, roll=−π/4)  ==  ThinSkewQuadrupole(k1l)     (G1)
    ThinSextupole(k2l, roll=−π/6)   ==  ThinSkewSextupole(k2l)      (J3)

both to machine precision. A *positive* roll of `π/4` therefore gives the
**negative**-strength skew quadrupole. A `Drift` is exactly roll-invariant; a
`Corrector`'s kick simply rotates.

### A bending dipole is the exception, and it is the curved geometry K1 declined

A bend carries the reference frame around with it, so rolling the magnet by `φ` about
its **entrance** puts the exit face somewhere the lattice does not expect. The exit
transformation is not `R(−φ)` but the rigid motion

    T = A⁻¹ · R_s(φ) · A,        A = the design arc's own rigid motion

(`alignment.arc_motion`), and conjugating by `A` turns the rotation axis through the
bend angle, so `T` comes out as a **displacement, a pitch, a yaw and only part of the
roll**. `alignment.frame_change` converts a rigid motion into accsim's affine `(M, k)`;
`Dipole._alignment_exit` builds `T`. Derived in sympy, the exact kick is

    k_py = −sin(φ)·sin(θ)                                            [EXACT]
    k_y  = −ρ(1 − cos θ)·sin(φ) / (sin²θ·cos φ + cos²θ)
    k_px = +(1 − cos φ)·sin(θ)·cos(θ)                                [EXACT]
    k_x  = +ρ(1 − cos φ)(1 − cos θ)·cos θ / (sin²θ·cos φ + cos²θ)
    k_ζ  = −ρ(1 − cos φ)(1 − cos θ)·sin θ / (sin²θ·cos φ + cos²θ)

Three facts in there, each of which the milestone's opening plan got wrong:

- **The vertical kick is `φ sin θ`, not `θ sin φ`.** The roll acts on the bend's
  *chord*, not on its angle. At `θ = 0.3` the two differ by `sin θ/θ = 1.5 %`
  (`5.910010e-3` measured against `5.99960e-3` claimed), and at `θ = 0.8` by 10 %.
- **A small roll is not a pure vertical bend.** There is also a vertical **offset**
  `−φ·ρ(1 − cos θ)` — the sagitta of the arc, tipped out of the plane. It is not a
  refinement: deleting it from the ring's source vector gets `D_y` **wrong in sign**
  (deleting the *angle* half instead costs 5 %).
- **The vertical effect is first order in `φ` and the horizontal loss only second**
  (`1 − cos φ`), which is what makes a roll a *vertical* error at leading order.

**Unlike K1, a roll changes `matrix()`.** It mixes the transverse blocks and the `δ`
column, so β, the tunes, the dispersion *and* the coupling all move. K1's
"displacements leave the optics bit-for-bit alone" is a statement about translations
only. Because of that, `matrix()` became a **template method** like `kick()` and
`track()`, and subclasses now override **`_matrix_body`** — the third extension point
to move in this axis. `Element._alignment_entry` / `_alignment_exit` return the affine
maps either side of the body, and `Dipole` overrides the exit; that is what makes the
straight and curved cases one code path.

### The wrong model is measured, not argued

The conjugation model misses xtrack by `5.9e-3` in the kick and `6.2e-3` in the matrix
at `φ = 0.02, θ = 0.3` (`0.11` / `0.22` at `φ = 0.1, θ = 0.8`). The rigid-motion model
matches xtrack to `3.3063e-9` — **the same number to five figures** as the *aligned*
bend does, i.e. the residual is entirely the pre-existing linear-map-vs-exact-map
difference and the roll adds nothing of its own. The sign is pinned against
`rot_s_rad_no_frame`; pinning it against `rot_s_rad` would have passed on a straight
element and said nothing about the bend, which is the only place they differ.

### A rolled bend also couples, and two type-walking helpers had to start refusing

Entry rolls by `φ`; the exit gives back only `arctan(cos θ · tan φ)`, leaving
`φ(1 − cos θ)` of roll. The largest transverse off-block entry is `2φ(1 − cos θ)`
(measured across two decades of bend angle), so the coupling is first order in the roll
and **second in the bend angle** — which is why nobody notices it on a weak arc.

- `closest_tune_approach` sums over skew-quadrupole *elements*, so it cannot see this
  at all and would return `0.0` for a demonstrably coupled ring. It now raises
  `CoupledLatticeError` — measured, not by type: any element whose own matrix has a
  nonzero transverse off-block and is not a skew quadrupole. `normal_mode_tunes`
  diagonalises the map and does see it.
- `linearised_lattice` refuses a rolled `ThinSextupole`/`ThinOctupole` rather than
  emitting the *unrolled* feed-down split. Rolled higher multipoles are out of scope.

### Orbit-driven vertical dispersion — a blind spot K2 made consequential

**The headline claim is narrower than "the first source of vertical dispersion", and
the measurement that narrowed it was a surprise.** What is true: a rolled bend is the
first element in this package whose **matrix** carries a vertical `δ` column, so it
produces `D_y` in a ring with **no coupling element anywhere** — G1's skew quadrupole
only *rotates* dispersion the horizontal bends already made (its own `δ` column is
identically zero), and a `Corrector`'s matrix is the identity.

What is **not** true is that this is the only route. accsim's linear elements drop the
`1/(1+δ)` on angles — a drift is `y += L·py`, where the exact map is `y += L·py/pz` —
so in the exact maps *any* vertical closed-orbit **angle** makes vertical dispersion,
and accsim cannot see it. Isolated by a **vertical steerer in an otherwise perfect
ring** (no roll, no coupling, nothing K2 touched):

    accsim  D_y = 0 exactly        xtrack  dy = 2.1e-4        orbits agree to 8 digits

and on K2's own rolled test arc that route is the **larger** of the two
(`−3.34e-4` against accsim's `−3.05e-5`).

**It is understood, not merely named.** Putting the two dropped terms back by hand
reproduces xtrack's `dy` *and* `dpy` to **0.2 %** on both rings. Per element of length
`L`, the exact vertical motion is `dy/ds = py(1 + h·x)/pz`, so at the closed orbit the
missing source is

    Δd_y = p_y·L·(h·⟨D_x⟩ − 1)          (and the same with p_x horizontally)

— the `−1` from `1/pz`, the `+h⟨D_x⟩` from the extra arc a dispersed particle travels
on the outside of a bend. On this arc those two nearly cancel (`h⟨D_x⟩ ≈ 0.39`), which
is why a `−L·py·δ`-only account overshoots by 1.5×.

This shares a root cause with J1's `−L·px·δ` note (*the thick sextupole is compared by
difference*) but is **not** the same statement: that one is a per-element map residual
worth `1e-8`, this is a ring-level physics consequence three orders larger. It predates
axis K entirely — K2 is only where it becomes consequential — and it **cannot be fixed
inside a 6×6**, because the terms are bilinear (`p_y·δ`). Representing them means exact
nonlinear maps for `Drift`/`Quadrupole`/`Dipole`, which would re-baseline every gate in
the suite: a future milestone, specified by
`test_the_model_gap_is_fully_accounted_for_and_not_a_mystery`.

> **Update (L1, 2026-08-17) — the drift's half is done, and there was a third dropped
> term.** `Drift.track()` is now the exact map, so a ring whose drifts sit at a vertical
> orbit angle reports `D_y` (see *The drift's exact map*). Two additions to the account
> above:
>
> - **The two terms named here are not the whole list.** `Δd_y = p_y·L·(h⟨D_x⟩ − 1)` is
>   entirely about transverse motion's `δ`-dependence. The `−1` has a **canonical
>   partner** in the `ζ` row, `M[zeta, py] = −L·p_y`, of the same size; a map carrying
>   one without the other is not symplectic, and wrong at first order. That is a third
>   thing, not a restatement of either.
> - **K2's own test arc has no drifts** (thin quadrupoles and thick dipoles), so L1
>   moves none of its numbers. The `+h⟨D_x⟩` half still needs the exact `Dipole`, and
>   `test_the_model_gap_is_fully_accounted_for_and_not_a_mystery` remains unwritten —
>   it belongs to the end of the L axis, not to its first step. The clean gate L1 could
>   build instead is a **bend-free** ring, where `h = 0` removes that half by
>   construction.

Gates: `tests/analytic/test_roll.py` (30),
`tests/reference/test_roll_xtrack.py` (10, two `xt.Line` builds — every map-level probe
lives in one line and is tracked through by index).

**Out of scope for K2:** the design tilt (`rot_s_rad`), rolled bends *and* offsets
together (the offset of a curved body is still refused), rolled higher multipoles as a
feed-down source (refused, above), pitch and yaw (`rot_x_rad`/`rot_y_rad`),
longitudinal displacement (`ds`), statistics of rolls, and misalignment **correction**.

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
- `accsim.momentum_compaction(lattice, slices=64, method="identity")` offers **two
  routes to the same number** (D4, 2026-07-20):
  - `method="identity"` (**default**) — the exact symplecticity identity
    `α_c = 1/γ₀² − (R51·D_x + R52·D_px + R56)/C`, read off the one-turn
    longitudinal row on the matched dispersion orbit. Both ingredients are
    closed-form, so this is exact to machine precision and **`slices` is ignored**.
  - `method="quadrature"` — the path integral evaluated directly: transport the
    matched dispersion along the lattice and, inside each thick dipole, integrate
    `D_x(s)` by trapezoidal sub-slicing of the sector sub-bend map (`h` constant
    across a body) — the same idiom as `natural_chromaticity`. Converges onto the
    identity at `O((h·ds)²)`: ~1.6e-6 at `slices=64`.

  The quadrature is **kept deliberately**, not vestigial: it touches the
  dispersion-generating matrix entries while the identity touches only the
  longitudinal row, so it is the independent second route that keeps the default
  honest. Delete it and the two cross-checks collapse into one. Consequently
  **every test comparing `α_c` against the identity must pass
  `method="quadrature"` explicitly** — on the default that comparison is a
  tautology that stays green while testing nothing.
- `radiation_integrals`' `I1 = ∮ D_x h ds` still runs the trapezoid, so
  `I1 == α_c·C` holds to round-off only against `method="quadrature"`, and to
  ~1e-5 (the trapezoid's own error) against the exact default. Both are asserted.
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
  `I1 = ∮ D_x h ds` (`= α_c·C`), `I2 = ∮ h² ds`, `I3 = ∮ |h|³ ds`,
  `I4 = ∮ D_x h (h² + 2k1) ds − Σ_faces D_x h² tan(e)`,
  `I5 = ∮ curlyH |h|³ ds` with `curlyH = γ_x D_x² + 2α_x D_x D_x' + β_x D_x'²`. `h = 1/ρ`
  is signed; `I3`/`I5` use `|h|³` (excitation is bend-sign-blind), `I4` keeps `h³`'s
  sign. Reuses the thick-dipole dispersion sub-slicing of `momentum_compaction`; `I5`
  additionally **co-transports `β_x,α_x`** through the dipole body (the one bug-prone
  spot). Slice-converged (64 ≡ 1024 to 6 digits).
- **Combined-function + edge `I4`.** `I4` carries the general form above: the `2k1`
  body term (quadrupole gradient) and the `−D_x h² tan(e)` pole-face term, reducing to
  the pure-sector `∮ D_x h³ ds` when `k1 = e1 = e2 = 0` (byte-identical transport, so the
  Stage-7 gates are unmoved). Inside a dipole the dispersion/β are co-transported through
  the **actual** combined-function body (sub-slices carry `k1`) and the thin edge kicks
  (applied to `D_x',α_x`; `D_x,β_x` continuous). `I2`/`I3` are pure geometry, unchanged.
  - **The `2k1` coefficient is pinned by a closed form**, not a remembered constant: a
    **smooth constant-gradient ring** has the *exact* fixed-point dispersion `D_x = h/K_x`,
    `D_x' = 0` (`K_x = h²+k1`; `R21 D_x + R26 = 0` identically — machine precision for any
    segment count), giving `I4 = h²(h²+2k1)C/K_x` and `J_x = 1 − I4/I2 = −k1/K_x = n/(1−n)`
    with field index `n = −k1/h²`. The wrong coefficient `h²+k1` (1 instead of 2) would
    force `I4 ≡ I2` and `J_x ≡ 0` for every `n` — refuted by `I4 ≠ I2`
    (`tests/analytic/test_radiation_combined.py`). A strong gradient can drive **`J_x < 0`
    (horizontal anti-damping)**, a signature a sector bend cannot fake.
  - **The edge term is pinned against MAD-X**, whose integral-method `synch_4` *is*
    trustworthy for a pure-sector-with-edges ring — there MAD-X `synch_1 == alfa·C` to
    `1e-15` (self-consistent), and accsim's `I4` matches `synch_4` to `~1e-3`
    (`tests/reference/test_radiation_edges_madx.py`).
  - **Why not MAD-X for the `2k1` term.** For a *combined-function* ring MAD-X's TWISS
    `synch_*` disagree with its **own** `alfa` at the ~0.8% level (its synch integrals
    treat the combined-function dispersion differently than `alfa` does) — the same order
    as the effect, so MAD-X is not a clean anchor there. accsim is self-consistent
    instead: `I1 == α_c·C` (exact identity) holds through combined-function + edge
    transport to `1e-5`. Likewise xtrack's `radiation_analysis` uses damped-map
    eigenanalysis (**not** integrals, it exposes none), differing ~1% (partitions) /
    ~3-4% (`ε_x`) in a strong ring while `I1`/`I2`/`U0` match to `1e-6` — integral-formula
    vs eigenanalysis, not a bug (`tests/reference/test_radiation_xtrack.py`).
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
  effect (`β` varying across `σ_z` when `σ_z ≳ β*`) is a *separate* reduction —
  see *Hourglass effect* below — and the two do **not** factorise.
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

## Hourglass effect (C2 — implemented)

`hourglass_reduction(sigma_z, beta_x_star, beta_y_star=None)` (`accsim.collider`,
always-on baseline: numpy/scipy only) returns the multiplicative luminosity
reduction `H` from the finite bunch length. Collisions are spread over the
crossing, and `β(s) = β*(1 + s²/β*²)` grows away from the waist, so the beams are
fatter than `σ*` almost everywhere:

    H = 1/(√π σ_z) ∫ ds e^{−s²/σ_z²} / √((1 + s²/β_x*²)(1 + s²/β_y*²))

- **The integrand is derived, not remembered** (`tests/analytic/test_hourglass.py`,
  6 tests): doing the `x`, `y`, `t` Gaussian integrals of `ρ₁ρ₂` in sympy makes
  *both* pieces fall out on their own — the `e^{−s²/σ_z²}` weight and the waist
  factor. The same derivation, integrated over `s`, reproduces Stage 6's
  `1/(4π σ_x σ_y)`, so the new factor rides on the already-validated overlap.
- **The collision points have rms `σ_z/√2`, not `σ_z`.** Both bunches must be
  present, so the two longitudinal Gaussians multiply — that is the `e^{−s²/σ_z²}`
  (variance `σ_z²/2`) above. Plenty of references write `σ_z` here; it is the
  classic hourglass trap and a factor-√2 error in the *shape*. `σ_z` is the
  **per-bunch** rms, the same meaning `piwinski_reduction` gives it.
- **Round waist is exact.** `H = √π·a·e^{a²}·erfc(a)` with `a = β*/σ_z`, from
  `∫e^{−u²}/(u²+a²)du = (π/a)e^{a²}erfc(a)` (sympy). Coded with **`scipy.special.erfcx`**
  (`= e^{a²}erfc(a)`) so a short bunch (`a` large) does not overflow to `inf·0`.
  Unequal `β_x* ≠ β_y*` has no such closed form and is quadratured; it is bracketed
  by the two round cases.
- **Limits.** `H → 1 − σ_z²/(2β*²)` for a short bunch; `H → √π β*/σ_z → 0` for a
  long one. LHC nominal (`β* = 0.55 m`, `σ_z = 7.55 cm`) gives `H = 0.9907` — under
  a percent, which is why Stage 6 could ignore it; squeezing `β*` to `0.15 m` at
  the same bunch length costs ~10%, the reason a `β*` squeeze alone does not buy
  the luminosity it appears to.
- **`H` does NOT factorise with the Piwinski `S`.** A crossing angle couples the
  transverse and longitudinal integrals through the same growing `σ_x(s)`, so
  `L₀·S·H` is an *approximation* good for a short bunch or a small angle. The
  exact combined factor is a genuinely 2D integral and is **not implemented**;
  `luminosity()` is therefore left unchanged and `H` is applied by the caller,
  deliberately, rather than silently multiplied in.

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
- **Elliptical Bassetti–Erskine** was out of scope at Stage 6 and landed later as
  **C1** — see [the next section](#elliptical-bassettierskine-kick-c1--implemented).
  It does break the `L_z` conservation the round beam enjoys, as anticipated here.
  Hourglass / crossing-angle geometry in the kick remains out of scope (the crossing
  angle enters *luminosity* only; the hourglass factor is [C2](#hourglass-effect-c2--implemented)).

## Elliptical Bassetti–Erskine kick (C1 — implemented)

`BeamBeam(n_particles, sigma, sigma_y=None, strong_charge=1.0)` — the **same element**
now covers `σ_x ≠ σ_y`. `sigma` is `σ_x`; `sigma_y=None` means round. Either ordering
is allowed (`σ_y > σ_x` is a tall bunch). Both shapes share one prefactor and differ
only in the field shape `S = 2πε₀E` [1/m]:

    Δp_perp = (q2/q1) (2 N r0/γ) · S(x, y).

- **The shape is *derived* from Coulomb's law, not transcribed.** Writing
  `1/r² = ∫₀^∞ e^{−r²t} dt` turns the convolution of the 2D point field with the
  Gaussian charge into an elementary Gaussian integral; sympy returns **exactly**
  (`tests/analytic/test_beam_beam_elliptical.py`, symbolic difference `0`)

      S_x = ½∫₀^∞ dq · x e^{−x²/2A − y²/2B} / (A^{3/2} B^{1/2}),   A = q + σ_x²
      S_y = ½∫₀^∞ dq · y e^{−x²/2A − y²/2B} / (A^{1/2} B^{3/2}),   B = q + σ_y²

  The round case is the `w = 1/(q+σ²)` collapse of this same integral back to Stage 6's
  `g(u)`, so the two branches are one derivation, not two formulas.
- **`S_y + i S_x`, *not* `S_x + i S_y`.** With `d = 2(σ_x²−σ_y²)` and `w` the Faddeeva
  function (`scipy.special.wofz`):

      S_y + i S_x = √(π/d) [ w((x+iy)/√d) − e^{−x²/2σ_x² − y²/2σ_y²} w((xσ_y/σ_x + iyσ_x/σ_y)/√d) ]

  This transposition is *the* classic Bassetti–Erskine error, and it is insidious: it
  survives both the round limit and the on-axis values, breaking only the **off-axis
  angular structure**. The stated milestone gate (reduces to round `g(u)`) therefore
  **cannot** catch it. It is pinned instead against a brute-force 2D Coulomb integral
  sharing no code with `wofz` — and mutation testing confirms that gate fails when the
  components are swapped.
- **`σ_y > σ_x` swaps axes internally** (the closed form assumes `σ_x > σ_y`), and the
  kick is evaluated at `(|x|,|y|)` with the signs restored afterwards. The charge is
  symmetric in both planes, so this is exact — and it keeps `w(z)` off the lower half
  plane, where it grows like `2e^{−z²}` and would overflow.
- **Near-round fallback.** Below `|σ_x−σ_y|/(σ_x+σ_y) < 1e-8` the round branch is used,
  removing the `1/√(σ_x²−σ_y²)` division by zero at exact equality. The threshold is
  **measured, not guessed**: the round approximation's error is cleanly linear
  (`1.076·eps`), so at the threshold it is `~1e-8` — at or below what the `wofz`
  difference itself achieves near the axis. The seam is asserted continuous.
  Contrary to folklore, `wofz` does **not** degrade catastrophically as `σ_x→σ_y`: the
  accuracy limit is set by *radius* (relative error `~1e-8` at `r/σ ~ 1e-4`, on a
  vanishing quantity), not by ellipticity.
- **Linear limit is now per plane** — `strengths(ref)` returns `(K_x, K_y)`:

      K_u = (q2/q1)(2 N r0/γ) / (σ_u (σ_x + σ_y))

  reducing to `K = (q2/q1) N r0/(γσ²)` when round. **The narrow plane is focused
  harder.** `strength(ref)` (scalar) now **raises** for an elliptical bunch rather than
  returning a misleading single number. `matrix()` and `beam_beam_tune_shift` use the
  pair, so a flat beam gets an unequal `(ΔQ_x, ΔQ_y)`.
- **Gauss's law fixes the normalisation independently.** `K_x + K_y = amp/(σ_xσ_y)`,
  i.e. the central charge density — a constraint the round limit alone cannot supply,
  since it would absorb a stray factor of 2 or π. Held exactly on both branches (the
  round fallback uses the **geometric** mean `√(σ_xσ_y)` for this reason, though at the
  threshold the choice is immaterial to `O(eps²)`).
- **The honest cost: `L_z` is no longer conserved.** The elliptical field is not radial,
  so it exerts a torque. This is **physical, not a defect**, and the suite asserts the
  *breakage* (alongside the round beam's exact conservation) so the Stage-6 invariant
  is not silently over-claimed. **Curl-free survives** — that is the property that
  matters for symplectic tracking.

## Beam-beam tune shift ξ (Stage 6 — implemented)

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
  parton-level `A_FB` from data without the generator truth) — milestone A3.
  The Collins-Soper *azimuthal* `φ*` and angular coefficients `A_0..A_7` were previously
  out of scope; they are now **built** — see *DY angular coefficients A₀–A₇ & Lam–Tung*
  below. `sin²θ_W` extraction was likewise out of scope and is now **built** (A2) —
  see *sin²θ_W from A_FB(m)* below.

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

## sin²θ_W from A_FB(m) (A2 — implemented)

Extracting the weak mixing angle by fitting the binned forward-backward asymmetry —
how LEP and the LHC actually measure it. `src/accsim/events/electroweak.py`
(**always-on baseline**: numpy/scipy only; the *data-producing* Pythia step stays
behind `ACCSIM_ENABLE_LHAPDF` as before).

- **Where the sensitivity comes from.** `g_A^f = T³_f` carries **no** `sin²θ_W`
  dependence at all; the entire response flows through
  `g_V^f = T³_f − 2Q_f sin²θ_W`. For a charged lepton `g_V^ℓ = −½ + 2sin²θ_W ≈ −0.038`
  — near its zero at `sin²θ_W = ¼`, so a small absolute shift in the angle is a large
  *relative* shift in `g_V^ℓ`, and `A_FB` inherits that amplification. This is the
  whole reason the measurement is sharp, and it is pinned by a test.
- **Angular structure — derived, not remembered** (`tests/analytic/test_electroweak_afb.py`,
  explicit Dirac-γ matrices, metric `diag(+,−,−,−)`, massless fermions, symbolic
  couplings). For a mediator pair `(V,V')` the spin-summed squared amplitude is
  `|M|²_{VV'} ∝ 4s²[(1+cos²θ)·SYM + 2cosθ·ASYM]` with
  `SYM = (v_ℓv_ℓ' + a_ℓa_ℓ')(v_qv_q' + a_qa_q')` and
  `ASYM = (a_ℓv_ℓ' + a_ℓ'v_ℓ)(a_qv_q' + a_q'v_q)`. Hence, summing mediator pairs with
  complex propagators `P_V`:
  `S = Σ Re[P_V P_V'^*]·SYM`, `D = Σ Re[P_V P_V'^*]·ASYM`,
  `dσ/dcosθ ∝ S(1+cos²θ) + 2D·cosθ`, and **`A_FB = (3/4)·D/S`**, `A₄ = 2D/S`.
  The second identity **reproduces the `A_FB = (3/8)A₄` anchor by construction**, tying
  this model to the independently-validated extractor of the previous section.
- **Mediators** (common `e²` stripped — it cancels in `D/S`): photon `v = Q_f, a = 0`,
  `P_γ = 1/s`; Z `v = g_V^f, a = g_A^f`, `P_Z = κ/(s − M_Z² + i M_ZΓ_Z)` with
  `κ = 1/(4 sin²θ_W cos²θ_W)` and `cos²θ_W = 1 − sin²θ_W`. `_s_and_d` implements the
  **literal double sum over mediator pairs** — deliberately *not* hand-expanded into
  `γγ + 2Re(γZ) + ZZ`, so an interference term cannot be dropped or mis-signed.
- **Which angle is recovered: the *effective* one.** Pythia separates
  `StandardModel:sin2thetaW` (on-shell, fixes the W/Z mass relation) from
  `StandardModel:sin2thetaWbar` (**effective**, enters the fermion vector coupling).
  `A_FB` responds to the *effective* angle. Leaving both at their defaults would have
  made "recover the value Pythia was configured with" ambiguous, so `generate_hepmc.cc`
  now **sets both explicitly** (`--sin2-theta-w`, default `0.2312`, via `DY_SIN2THETAW`)
  and **reads them back out of Pythia** into `meta.dat` as `sin2thetaw=` /
  `sin2thetawbar=`. The analysis must read the truth from `meta.dat` — **never hardcode
  a remembered default.**
- **The flavour sum is not a detail.** The hadronic observable is a parton-luminosity
  weighted sum over initial states; up- and down-type quarks have different asymmetries
  and their mix shifts with `m` through the PDFs. Weights combine at the level of `S`
  and `D`, **not** by averaging per-flavour `A_FB` values (`A_FB` is a *ratio* — averaging
  ratios is wrong):
  `A_FB(m) = (3/4)·Σ_q L_q(m)D_q(m) / Σ_q L_q(m)S_q(m)`. Only relative weights matter.
- **Fit the undiluted `A_FB`.** The model is parton-level: it assumes the quark
  direction is known. The `pp` `sign(Q_z)` proxy dilutes by ≈0.37 (previous section);
  correcting for that is **A3**, deliberately kept out of this model so the two
  milestones stay orthogonal.
- **Gate (layered, so a wrong model and a wrong fitter cannot cancel).** Symbolic
  derivation of the angular decomposition; the module's `S`/`D` matched term-by-term
  against that symbolic expression to `1e-12`; the sign gate (`A_FB<0` below `M_Z`,
  `>0` above, with a bisected zero-crossing under the pole) reproduced *independently*
  by the model; and a **round-trip** — sample events from the model's
  own distribution, measure with the *real* `forward_backward_asymmetry`, fit the angle
  back — at three injected values.
- **Which checks are actually external (important — most are not).** The round-trip
  runs the *same* `S`/`D` formula on both the generating and fitting side, so it cannot
  catch a wrong coupling or `κ`. And **`A_FB = (3/8)A₄` is a tautology** here, since `A₄`
  is defined from the same `S`/`D` — it is retained as a consistency tie to the A1
  extractor, *not* as evidence. The genuine external anchors are two, and they are
  **complementary by construction**:
  - **Pure-Z limit** `A_FB = (3/4)·A_ℓ·A_q` with `A_f = 2v_f a_f/(v_f²+a_f²)` — the
    standard LEP combination, written out independently and matched *both* symbolically
    (photon dropped from the bilinear; also asserted `s`-independent) *and* numerically
    on the pole through the production path, to 5%. This pins the **coupling**
    normalisation.
  - **`κ` derived**, not assumed: `(g_Z/2)²/e² = 1/(4sin²θ_W cos²θ_W)` from
    `g_Z = g/cosθ_W` and `e = g sinθ_W`, checked symbolically.
- **Why `κ` needs its own anchor (measured, not assumed).** The on-pole check is
  **blind to `κ`**: there the Z dominates and `κ` cancels from the ratio `D/S`. Probed
  directly — a **factor-2 error in `κ` shifts the on-pole value by only 0.06%**, and
  *toward* the pure-Z limit (more Z dominance ⇒ purer limit), so a wrong `κ` would look
  *better*. Its real effect is off-pole via interference, where it is large:
  `A_FB(m=75)` moves `−0.660 → −0.445` under `κ×2`. Since the off-pole shape is exactly
  where the `A_FB(m)` fit draws its sensitivity, an unverified `κ` would bias the
  extracted angle — hence the separate derivation plus a guard that the off-pole curve
  sits far from the pure-Z limit (i.e. interference is genuinely load-bearing).
- **Guarding the "within fit error" trap.** "Recovered within error" is vacuous if the
  error is inflated or the χ² is flat, so the gate also asserts: a **pull distribution**
  over 25 pseudo-experiments with unit width (an inflated error collapses it), an
  absolute cap `σ < 2e-3`, **χ² curvature** (a 1e-3 shift in `sin²θ_W` must cost χ²≫1),
  **starting-point independence**, and a **wrong-truth control** (data generated at
  `0.2450` must not be pulled toward a `0.2312` starting guess).
- **Bug found and fixed during development (worth remembering).**
  `scipy.optimize.least_squares` reports `success=True` when it converges **onto a
  bound** — for `initial=0.40` it returned the window edge `0.45` with `χ² ≈ 6e6`,
  dressed up as a measurement. `fit_sin2_theta_w` now **raises** on a bound-pinned
  solution rather than returning it. A converged-on-bound fit is a failed fit.
- **Known limitation, stated honestly.** The single fitted parameter floats in `κ` as
  well as in the couplings, which is a tree-level simplification (a real extraction
  fixes the `γ/Z` normalisation from `G_F M_Z²` independently of the fitted effective
  angle). The sensitivity is overwhelmingly through `g_V^ℓ`; `κ` only reweights `γ`
  vs `Z`. The model is also **LO** — Pythia's sample carries ISR and higher-order
  effects the model does not, so a residual bias against generated data is expected and
  should be quoted, not absorbed into a loosened error.

## pp dilution & unfolding (A3 — implemented)

`src/accsim/events/dilution.py` — always-on baseline (numpy only). Recovers the
parton-level `A_FB(m)` from the `sign(Q_z)`-proxy measurement. Reuses A2's
`_s_and_d`, so the angular strengths are not re-derived here.

- **Orientation split, not a beam split.** The luminosity of each flavour is split by
  whether the quark travels **along** the proxy direction (`lum_aligned`, `L⁺`) or
  against it (`lum_reversed`, `L⁻`). At LO the proxy `sign(Q_z)` equals `sign(x₁−x₂)`
  — a *deterministic* function of the configuration, not a random draw — so for
  `y > 0`, `L⁺ = q(x₁)q̄(x₂)` and `L⁻ = q̄(x₁)q(x₂)`; for `y < 0` the two swap. Stating
  the split by orientation makes it rapidity-sign agnostic.
- **The master formula.** A wrong orientation sends `cos θ → −cos θ`, flipping the
  antisymmetric term and leaving the symmetric one alone:

  ```
  A_FB^obs (m) = (3/4) · Σ_q (L_q⁺ − L_q⁻) D_q / Σ_q (L_q⁺ + L_q⁻) S_q
  A_FB^true(m) = (3/4) · Σ_q (L_q⁺ + L_q⁻) D_q / Σ_q (L_q⁺ + L_q⁻) S_q
  ```

  **Dilution reweights the numerator only** — the denominator (the rate) is untouched,
  because a mis-oriented event is still an event. That one difference is all of A3.
- **`D_eff` is not a PDF-only quantity.** `D_eff = Σ(L⁺−L⁻)D_q / Σ(L⁺+L⁻)D_q` carries
  the per-flavour `D_q` and therefore **depends on `sin²θ_W`** — the parameter A2 fits
  from the unfolded curve. It collapses to the clean PDF ratio `(L⁺−L⁻)/(L⁺+L⁻)` only
  for a *single* flavour. `dilution_factor` takes `sin2_theta_w` for this reason;
  `pdf_dilution` provides the flavour-blind ratio the literature usually plots, marked
  as an approximation. Measured size of the coupling on the toy: shifting `sin²θ_W`
  from `0.2250` to `0.2380` moves `D_eff` by up to `~5e-2` — weak, but not negligible
  beside a per-mille `A_FB`, so it belongs in the systematic budget or the fit should
  be iterated.
- **Degenerate region.** At central rapidity `x₁ → x₂`, so `L⁺ → L⁻` and `D_eff → 0`:
  the proxy is a coin flip and the asymmetry is *destroyed*, not merely noisy — no
  statistics recover it. `dilution_factor`/`unfold_afb` mask `|D_eff| < min_dilution`
  (default `1e-3`) to `nan` rather than returning a large number that reads as a
  measurement; the `nan` then fails `fit_sin2_theta_w`'s `σ > 0` filter, so such bins
  drop out downstream. Same failure mode as the `tracked_tunes` `Q ≈ 0, 0.5, 1` note.
- **Error propagation.** `unfold_afb` divides the error by `|D_eff|` as well — the
  honest statement that dilution destroys information rather than rescaling it.
  `D_eff` is treated as an exact model input; its PDF and `sin²θ_W` uncertainties are
  separate systematics, deliberately not folded in.
- **Gate met** (`tests/analytic/test_dilution.py`, 13 tests). The undiluted reference
  is A2's `afb_hadronic`, so the two sides of the closure are different code paths.
  Layered: the two exact limits (`L⁻ = 0` reproduces `afb_hadronic` to `1e-15`;
  `L⁻ = L⁺` gives exactly zero); the **formula closure** — unfold the diluted curve,
  recover `afb_hadronic` to `1e-14`; and a **sampled MC closure** driving real
  four-vectors through the actual `collins_soper_costheta` proxy and
  `forward_backward_asymmetry`, asserted as a **pull** (unit-width over 12 seeds,
  max `|pull| = 2.8`) so a wrong error can't hide.
- **What stops the gate being vacuous.** With a *single* flavour the naive scalar
  divide is exact and the whole physics content goes untested, so the toy proton
  carries up **and** down with different valence hardness *and* different `A_FB`, and
  the suite asserts the naive `pdf_dilution` unfolding is **wrong by > 1e-3** on the
  same input while the correct one closes to `1e-14`. On the toy the dilution is
  severe (`D_eff ≈ 0.13–0.19`); the raw proxy measurement sits 12–50σ from truth.
- **Scope, stated honestly.** The luminosities are an *input* — the module never
  touches a PDF set, exactly as `afb_hadronic` takes `flavour_weights`. The analytic
  gate therefore runs on a toy proton, not a real PDF. Reproducing the dilution
  against the Drell-Yan pipeline's own proxy/true ratio (`truth_gen.dat`) needs
  Pythia + LHAPDF and **has not been run**; the pipeline is unchanged by A3.

## b-tagging efficiency & the Delphes card (E2 — implemented)

`src/accsim/events/btag.py` (always-on **baseline**: numpy only — no Docker, no
ROOT); the data-producing chain is `pipelines/pp_ttbar_btag/`, gated on
`lhapdf` **and** `delphes`.

- **The card is the closed form.** Delphes does not simulate a tagging algorithm.
  Its `BTagging` module is a *parametrisation*: it picks a per-flavour efficiency
  formula, evaluates it at the jet's `(pt, eta)`, and sets a bit with that
  probability. So every jet has a known right answer, written in the card.
- **Formulas are parsed, never transcribed.** They are read out of the very card
  file handed to `DelphesHepMC3` (the pipeline copies it back to the host). A
  retyped formula is a remembered constant in disguise — it drifts silently when
  the card changes, and a typo in it is invisible because both sides of the
  comparison then share it.
- **`Jet.BTag` is a bitmask, not a boolean.** A multi-working-point card packs
  Loose/Medium/Tight into bits 0/1/2 of one integer, so `BTag == 1` means "loose
  but *not* medium". Decoded as `(bits >> bit_number) & 1`, with the bit number
  coming from the parsed card — the card decides which bit means what.
- **`Jet.Flavor` for a light jet is `1`/`2`/`3`/`21`, not `0`.** Delphes writes the
  |PDG| of the hardest parton in the cone; only `4` and `5` have their own
  formula, and everything else falls to the card's default (`{0}`), which *is*
  the mistag rate. Selection is therefore "has no dedicated formula", not
  "flavour == 0"; comparing raw codes against a 0-means-light truth label scores
  every light jet as a mismatch.

**TCL/Delphes expression semantics** (evaluated by an `ast` walk with a node
whitelist — card text is never `eval`-ed):

- a comparison yields the *number* `1`/`0`, which is what makes the step-function
  cards pure arithmetic. Bare bools would make `(a)+(b)` numpy's logical OR.
- `&&`/`||` bind **looser** than the comparisons around them, so they map to
  Python's `and`/`or` (same loose precedence), evaluated element-wise — **not**
  to `&`/`|`, which bind *tighter* than comparison and silently reassociate
  `pt > 30 && pt <= 100` into the chained `pt > (30 & pt) <= 100`.
- `^` is **exponentiation** (Delphes' `TFormula`-based parser), not stock TCL's
  bitwise xor.

**Two statistical choices that are physics, not style:**

- **The expected efficiency in a p_T bin is the jet-wise mean of the formula, not
  the formula at the bin centre.** The jet spectrum falls steeply, so a bin is not
  populated at its centre while the efficiency still varies across it. The
  bin-centre value is a quiet ~0.07 absolute bias that survives any "looks about
  right" plot inspection; the suite asserts it is >10σ wrong on a falling spectrum
  where the jet-wise mean closes. It also makes smooth and step-function cards
  work through one code path — edges inside a bin average correctly.
- **The pull uses the *expected* binomial variance**, `sqrt(p_exp(1-p_exp)/N)`,
  not the observed one, which is exactly zero (infinite pull) in the zero-tag bins
  a ~0.1% mistag routinely produces. Relatedly, a bin counts toward the χ² only
  when `N·p·(1−p) ≥ 10` — a floor on the **variance**, not on the jet count. The
  two come apart exactly where the tight working points live: thousands of jets
  with ~1 expected tag is Poisson, its achievable pulls are discrete, and folding
  it in inflates the χ² and invites a threshold nudge instead of a fix.

**The gate, and its honest kind.** This is a **round-trip / consistency** gate,
not a symbolic derivation like Robinson's theorem or `σ = 4πα²/3s` — the weakest
analytic gate in this repo, labelled as such. There is no independent physics
closed form; the reference is a fit parametrisation the card encodes (the CMS
card cites arXiv:1211.4462). What is proven is that the extraction, the flavour
handling, the binning and the estimator are right.

**Two independent authorities are used, because the card alone is a closed loop:**

1. **`DelphesFormula` — the evaluator authority.** accsim's evaluator is checked
   against *Delphes' own* (`DelphesFormula`, the `TFormula` subclass the
   `BTagging` module uses) over all 9 CMS_PhaseII_0PU formulas × a 252-point
   `(pt, eta)` grid that lands deliberately **on** the card's step edges
   (pt 20/30/100/1000, |η| 1.8/2.4/3.4). **Agreement is exact — 0.000e+00 over
   2268 points** — and asserted as exact, since both sides do the same IEEE
   double arithmetic. The reference is frozen into
   `tests/analytic/data/delphes_formula_reference.json` so the gate runs in CI
   without Docker. *Regenerate* with `pipelines/pp_ttbar_btag/eval_formulas.C`
   inside the Delphes image. **Gotcha:** `DelphesFormula`'s
   `(name, expression)` constructor does **not** leave the formula ready to
   execute — `Eval()` returns `nan` and logs *"Formula is invalid"*. Default-
   construct and call `Compile()`, as Delphes' own modules do.
2. **A ΔR-matched generator label — the flavour authority.** Delphes' `BTagging`
   keys on exactly the `Jet.Flavor` that `JetFlavorAssociation` writes, so
   histogramming that field against the tag bit validates the *handling* of the
   label but never its *definition*. The generator therefore dumps its own heavy
   quarks straight from Pythia's record (no HepMC round-trip) and the analysis
   builds an independent label by ΔR matching. The parton selection is
   deliberately **status-code-free** — the last quark of each flavour chain —
   because Pythia status codes do not survive the HepMC3 round-trip (see the
   *Delphes detector step* section).

**Scope, stated honestly.** Only the discrete **operating points** a card offers,
not a continuous discriminant ROC — Delphes stores a decision bit and never a
discriminant value, so a continuous ROC is not obtainable from it. Not attempted:
jet-energy-scale/resolution performance, τ-tagging, pileup. The
**ATLAS-vs-CMS card comparison** was considered for E2 and **rejected**: two
detector outputs side by side have nothing to be refuted against, which fails the
working agreement's analytic-gate rule.

## Transverse mass and the W Jacobian edge (E1 — implemented)

`accsim.events.transverse_mass` (baseline: numpy only). The **W-mass** observable
at a hadron collider, where the neutrino escapes down the beam pipe.

**Definition.**

    m_T² = 2 · p_T^ℓ · p_T^ν · (1 − cos Δφ)

Angles in **radians**. The `(1 − cos Δφ)` form is periodic, so `Δφ` is **never
wrapped** — wrapping would be a no-op at best and a sign trap at worst. The
product is clipped at zero before the `sqrt`: it is non-negative analytically, but
a collinear pair can round to ~−1e−17 and NaN the root.

**Only transverse information is used, by construction.**
`transverse_mass_from_vectors` takes four-vectors but ignores `E` and `p_z` —
the missing-momentum estimator *has* no `p_z`, so leaking one in would be
unphysical. The analytic suite asserts this by scrambling the neutrino's `E` and
`p_z` and demanding a bit-identical result.

**The neutrino proxy.** Truth uses the real neutrino four-vector; **reco uses
MET** (Delphes `MissingET`). That substitution is the truth-vs-reco seam of the
E1 pipeline and the dominant source of edge smearing. There is **no** full
invariant mass to build on the reco side — do not attempt one.

**The edge is at `M_W`, the lepton-`p_T` peak is at `M_W/2`.** Both are Jacobian
peaks and confusing them is *the* error this observable invites. `m_T` is the
`W`-mass observable specifically because its edge is insensitive to the `W`'s
recoil `p_T` at first order, while the `p_T^ℓ` peak is smeared by it at first
order. The analytic suite asserts both endpoints in one test to keep the
distinction pinned.

**Idealised density (derived in sympy, not remembered).** For an on-shell,
zero-`p_T`, **isotropic** two-body decay, the daughters are back-to-back in the
rest frame, so `Δφ = π` exactly and both carry `p_T = (M/2) sin θ` — hence
`m_T = M sin θ`. Pushing `cos θ ~ U(−1,1)` through that gives

    dN/dm_T = m_T / (M √(M² − m_T²)),   0 ≤ m_T ≤ M,
    CDF     = 1 − √(1 − m_T²/M²)

(`jacobian_peak_pdf`, normalised to 1). The `1/√(M²−m_T²)` **integrable
singularity** at the endpoint *is* the Jacobian edge: `dm_T/dcos θ → 0` at
`θ = 90°`, so a broad swathe of decay angles piles into a narrow `m_T` interval.

**Scope, stated honestly.** The **endpoint location** is exact and
convention-independent; it survives a transverse boost (asserted at `β = 0.4`,
far beyond real ISR) and a `V−A` angular weight. The **shape** does not: the
finite width `Γ_W`, the `W`'s recoil `p_T` (Sudakov-suppressed at low `p_T`), and
the MET resolution all round the edge, and `V−A` reweights it. So the shape test
states its isotropy assumption explicitly, and the pipeline gates on the **edge
location**, never on a delta-function or on the idealised shape.

**Quadrature note.** Normalising the pdf uses the substitution `m_T = M sin a`,
which removes the singularity analytically (the integrand is just `sin a`). In
*factored* form the exact endpoint is `∞ · 0`, so the test integrates by the
**midpoint** rule, which never samples `a = π/2`. That is a quadrature artifact,
not a physics one.

## Jacobian-edge locator & the E1 pipeline (E1 — implemented)

`accsim.events.jacobian_edge` (baseline: numpy only) + the `pp -> W -> mu nu`
pipeline in `pipelines/pp_W_mt/` (behind `ACCSIM_ENABLE_LHAPDF`). Extends
*Transverse mass and the W Jacobian edge* above with the **measuring device** and
what the pipeline does with it.

**Estimator: half-maximum of the falling side**, not `argmax`. The shape is a
divergence piled against a cliff, so its binned `argmax` is binning-dependent and
sits *below* the endpoint; a cliff convolved with a roughly symmetric kernel passes
through half its height essentially *at* the cliff. Measured head-to-head on the
same 600k sample (`sigma = 2`): half-max gives `81.84–82.13 GeV` across
`bins = 30..120`, `argmax` gives `78.3–79.2 GeV` — the latter both ~1.5 GeV low and
jittering. Asserted, not asserted-in-prose (`test_jacobian_edge.py`).

**It is biased high, and the bias is recorded rather than hidden:** roughly
`+1 GeV + 0.73 sigma` (full table in the docstring, pinned by a parametrised test).
What makes it usable is that the offset is **constant at fixed smearing** — at
`sigma = 2` the recovered edge tracks the true mass to `+1.55 ± 0.04 GeV` across
`M = 60..100 GeV`, so it measures the *mass*, not an artifact of the shape.

**`falloff_width`** (peak-centre to half-max point) is a crude monotone measure of
edge roundness — the truth-vs-reco contrast rests on it and nothing else.

### The pipeline gate is a position, never `m_T <= M_W`

The analytic gate's `m_T <= M` holds for a **fixed** parent mass. Pythia gives the
`W` a **Breit-Wigner** mass, so off-shell events legitimately give `m_T > M_W` —
**measured at 6.6%** of truth events. A `max(m_T) <= M_W` assertion would either
fail on correct physics or pass only because a generation mass window had been
imposed near the edge, hiding the effect being measured. Hence **no mass window** in
the E1 generator (unlike the DY chain's `60..120 GeV`, which exists to dodge the
photon pole — the charged current has no such pole).

Three gates: truth edge within 5 GeV of `M_W`; reco edge measurably **rounder** than
truth; and the truth `p_T^mu` edge within 5 GeV of `M_W/2`. The tolerance is set by
the measured bias (~1.5) + binning (~0.3) + ISR recoil (~1), and sits far below the
~35–40 GeV a `p_T`-for-`m_T` mix-up produces — **justified, not tuned**.

**The gate reads `M_W` back out of Pythia** (`meta.dat`'s `m_w_gev`), never a
remembered PDG constant, or it would compare two remembered numbers.

### Two conventions pinned empirically, not remembered

- **`GenMissingET` points ALONG the neutrino** (`sign = +1`). Delphes' `Merger`
  negates its input sum, but `GenMissingET`'s input is the **neutrino list** itself,
  so the result could have pointed either way — a `pi` shift in `Δφ`, flipping
  `(1 - cos Δφ)` between `~0` and `~2`. The macro emits **both** `GenMissingET` and
  the directly summed truth neutrino; `analyze.py` measures the angle
  (**median |Δφ| = 0.0000, 100% aligned**) and **refuses to run** if it matches
  neither convention.
- **Muons are inside Delphes' `MissingET`** — `MissingET <- EFlowMerger/eflow <-
  HCal/eflowTracks <- TrackMerger`, which takes `MuonMomentumSmearing/muons`
  (`delphes_card_CMS.tcl` ~line 201). Checked in the card. Had muons been excluded,
  MET would track the hadronic recoil and every reco `m_T` would be meaningless.

**Measured (60k-event chain run):** truth edge **81.41** vs `M_W` 80.385, falloff
**2.24**; reco edge **85.16**, falloff **10.99**; `p_T^mu` edge **42.91** vs
`M_W/2` = 40.19.

**Negative controls (same run):** flipping the `GenMissingET` sign drops median
`m_T` from 62.9 to **7.0 GeV** (edge 25 GeV off); feeding `p_T^mu` to gate 1 lands
**35.8 GeV** off; flipping the reco MET sign drops median `m_T` to **9.4 GeV**. All
three fail the gates.

**The run re-derives its own motivation.** On the same events the `m_T` edge lands
**+1.03 GeV** from `M_W` while the `p_T^mu` edge lands **+2.72 GeV** from `M_W/2` —
the `m_T` edge is **2.7x better determined**. That gap *is* the first-order
ISR-recoil insensitivity that makes `m_T` the `W`-mass observable: it entered as a
design assumption and came back out as a measurement.

**Scope.** This locates an edge; it is **not** a W-mass measurement (which needs
template fits, recoil calibration, and PDF/QED systematics under 10 MeV). Not
attempted: `W` charge asymmetry, recoil calibration, the electron channel, pileup.

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
is `(zeta, p_zeta)`. For the linear drift this does not break the `Mᵀ J M = J`
check, but it is flagged for the longitudinal stages (Stage 3+).

**That caveat acquired teeth in L1 (2026-08-17), and the consequence is a trap.** The
moment a map is *exact* in `delta`, `(zeta, delta)` starts rejecting maps that are
genuinely symplectic:

- every **linear** element passes, because `matrix()` is three independent shear blocks
  and a shear is symplectic in whatever pair it acts on;
- the **exact drift** — the flow of a Hamiltonian, so symplectic by construction —
  **fails**, with the residual confined to the two `(px, delta)` / `(py, delta)` entries
  and second order in amplitude (`8.0e-14` at `1e-6`, `7.7e-8` at `1e-3`, `7.7e-6` at
  `1e-2`, a clean square). The same map in `(zeta, p_zeta)` gives **exactly zero**.

So in `(zeta, delta)` the more faithful map fails the check the cruder one passes, and
`is_symplectic_map` **cannot be used to judge an exact map**. Use
`is_symplectic_map_canonical(map_fn, state, ref)`, which conjugates by the coordinate
change first. Its real job is catching the tempting half-fix — transverse motion exact,
`zeta` left linear — which is wrong at **first** order in the amplitude (`2.0e-4` where
the correct map is `0`) and which the `(zeta, delta)` check misses because it rejects
both. The danger is "repairing" an exact map until the old check goes green: that lands
on the half-fix.

The coordinate change is `p_zeta = (E − E₀)/(β₀²E₀)` (xtrack's `pzeta`, its
`ptau/β₀`), with `pzeta_from_delta` / `delta_from_pzeta`. Two things the gates caught
rather than assumed:

- `delta_from_pzeta` must carry `dE = β₀²E₀·p_zeta` **directly** and never form `E`
  itself; writing `E = E₀(1 + β₀²p_zeta)` and subtracting `E₀` again rounds the small
  part away, losing `delta` to a relative `1e-4` by `p_zeta ~ 1e-12`.
- `p_zeta = delta` to **first** order at *every* energy, not only ultrarelativistically
  (`dE/E = β²·dp/p` makes the leading coefficient exactly 1). The real relation is
  `p_zeta = delta + delta²/(2γ₀²)` — second order *and* suppressed by `1/γ₀²`, which is
  why the distinction hides so well. A first draft of the test asserted the two differ
  at `γ₀ = 5` and was simply wrong.

## MAD-X reference frame (D3 — implemented)

The second reference code, driven via **cpymad** (`tests/reference/_madx.py`,
behind the `reference` marker). cpymad bundles the MAD-X binary and runs it in a
subprocess, so unlike the xtrack JIT it needs **no build toolchain**; cp314
Windows wheels exist and the subprocess launches fine from this repo's
space-containing path.

**Coordinates.** MAD-X is canonical `(x, px, y, py, T, PT)`, not accsim's
`(x, px, y, py, zeta, delta)`:

| | accsim | MAD-X | relation |
|---|---|---|---|
| longitudinal position | `zeta = s − β₀ct` | `T` | `zeta = β₀·T` |
| longitudinal momentum | `delta = Δp/p₀` (**momentum**) | `PT = ΔE/(p₀c)` (**energy**) | `PT = β₀·delta` |

The transverse 4×4 block shares ordering *and* normalisation, so it compares
entrywise with no transform. The longitudinal row/column need the diagonal
similarity transform

    R_accsim = M · R_madx · M⁻¹,   M = diag(1, 1, 1, 1, β₀, 1/β₀)

**Pinned empirically, not remembered.** The *scale* comes from a drift: MAD-X
reports `dT/dPT = L/(β₀²γ₀²)` where accsim carries `R56 = L/γ₀²` — a ratio of
exactly `β₀²`. The *sign* cannot be read off a drift (its only non-zero
longitudinal entry is even under flipping both `T` and `PT`); it is fixed by the
**dipole**, whose `R51`/`R52` (path lengthening) and `R16`/`R26` (dispersion) are
odd under that flip. With the sign above the dipole agrees entrywise at **2e-16**.
Negative controls confirm the check has teeth: a flipped sign shows up as
`max|Δ| ≈ 4e-1` *and* breaks symplecticity; omitting the transform entirely stays
symplectic but fails entrywise at `4e-3`.

**Twiss-table conventions**, consistent with the same `β₀`:
- `DX`/`DPX` are derivatives w.r.t. `PT`, so `D_accsim = β₀ · DX_madx`.
- `MUX`/`MUY` are in **turns**, not radians (accsim's `mu_x` is radians).
- The twiss table appends a zero-length `$end` marker row duplicating the final
  `s`; drop it before comparing s-grids point-for-point.

**What D3 does and does not buy.** xsuite deliberately follows MAD-X's coordinate
*conventions*, so a convention error the two share **by design** — and that accsim
copied — would not be caught by adding MAD-X. What the second reference genuinely
adds is an **independent numerical implementation**: an accsim arithmetic or sign
error, or an xtrack bug, now has to be reproduced by a separate Fortran codebase
to survive. The docs state that claim and no more.

**`alpha_c`: MAD-X is exact, and since D4 so is accsim's default.** MAD-X evaluates
`(1/C)∮D_x/ρ ds` in closed form per element. At D3 `momentum_compaction()`
trapezoided it (`slices=64`), giving ~1.6e-6 relative error on a 1 m-sector-bend
ring — a *known, documented* limitation, not a newly found bug. Rather than loosen
a tolerance, the D3 test compared the **exact** identity
`alpha_c = 1/γ₀² − (R51·D_x + R52·D_px + R56)/C` to MAD-X at `1e-10`, then showed the
quadrature *converging onto MAD-X's number* — which upgraded the existing
convergence test from self-consistency to agreement with an independent code.

**D4 then made that identity the default**, so the `1e-10` arm is now also the
shipped default's MAD-X check. The convergence arm asks for `method="quadrature"`
explicitly; without that it would compare MAD-X to the same exact number twice and
the convergence demonstration would silently evaporate while staying green.

**Scope.** Drift, quadrupole and dipole R-matrices plus one matched FODO-with-bends
ring (β, α, μ, tunes, dispersion, `alpha_c`). Deliberately **not** mirrored:
sextupole (its linear R-matrix is drift-like — `k2` enters only at second order,
so a MAD-X `RE` comparison would add nothing over the drift check) and the
radiation / synchrotron-tune checks (RF and radiation setup in MAD-X is a
different beast for little marginal confidence). The FODO ring carries dipoles on
purpose: the bend-free xtrack cell has `D_x = 0` and `alpha_c = 0`, so comparing
those would be comparing two zeros.

## Synchronous phase branch — keyed on `sign(η·q·V)` (fix, surfaced by D1)

`synchronous_phase(voltage, energy_gain, above_transition, charge)` inverts
`ΔE_s = q V sin φ_s` and must pick the **stable** of the two roots. Stability is

```
Qs² = -(h η q V cos φ_s) / (2π β₀² E0) > 0   ⟺   sign(cos φ_s) = -sign(η · q · V)
```

so the branch depends on **`η · q · V`**, not on `η` alone:

| `η`          | `q V` | stable branch     |
|--------------|-------|-------------------|
| < 0 (below)  | > 0   | `asin(s)`         |
| > 0 (above)  | > 0   | `π − asin(s)`     |
| < 0 (below)  | < 0   | `π − asin(s)`     |
| > 0 (above)  | < 0   | `asin(s)`         |

The first two rows are the familiar proton rule and are unchanged bit for bit —
the fix is a pure extension. The last two matter for **leptons**: an electron
(`q = −1`) driven by the usual positive voltage has `qV < 0`, so an electron
storage ring **above** transition sits at `φ_s = asin(s)`, just *below* zero when
the RF replenishes a radiation loss `U0`. The old rule handed back the unstable
root there and `synchrotron_tune` refused the lattice.

**Only stability distinguishes the roots.** `ΔE_s = q V sin φ_s` is identical on
both branches, so no energy-bookkeeping check can catch a wrong branch — which is
why the gate is `synchrotron_tune` raising `UnstableLatticeError` on the other
one. Both branches were pinned **empirically** (build the lattice, ask for the
tune), not from a remembered table.

Zero gain returns the branch's stationary phase: `0` when `sign(cos φ_s) > 0`,
`π` when negative. The Stage-3 mnemonic "0 below transition, π above" is the
`qV > 0` special case; a lepton ring above transition is stationary at `0`.

## Moving-bucket acceptance (D5 — implemented)

*Supersedes the former "the store bucket is moving, and `rf_bucket_height` models
only stationary ones" scope limit, which D1 surfaced and this closed.*

`rf_bucket_height` / `separatrix` / `longitudinal_hamiltonian` now model the
**moving** bucket (`sin φ_s ≠ 0`) on all four branches. The `sin φ_s` guard is
gone; the double-RF / multi-harmonic guard stays.

**Height.** The separatrix peaks where `dU/dζ = 0`, i.e. at `ζ = 0` for a moving
bucket too, so `δ_max² = 2[U(0) − U(ζ_u)]/(ηC)` is unchanged in form. Against the
same ring's stationary bucket it obeys, **exactly**,

    δ_max(φ_s)² / δ_max(stationary)² = cos ψ − (π/2 − ψ)·sin ψ,   ψ = asin|sin φ_s|

`= 1` at `ψ = 0` and `= 0` at `ψ = π/2`. **The same function of `ψ` on all four
branches** (proton/electron × below/above transition) — the above-transition case
is *not* the same function of `φ_s`; it is this function of `π − φ_s`. Derived
symbolically from accsim's own `U` (`tests/analytic/test_moving_bucket.py`), never
from a remembered constant.

**The bounding unstable fixed point cannot be hardcoded.** `dU/dζ = 0` has the
unstable family `k_rf ζ = 2φ_s + π + 2πn`. Stage 3 hardcoded the `n = −1` member,
`k_rf ζ_u = 2φ_s − π`. That is right only for `qV > 0`: the bucket is bounded by
whichever of the **two adjacent** unstable points (those straddling `ζ = 0`, taken
mod `2π`) gives the **smaller positive `δ_max²`**, and for `qV < 0` — an electron
ring driven by a positive voltage, where a positive energy gain forces
`sin φ_s < 0` — that is the *other* one. Keeping the hardcoded member there
returns a silently **too-large** `δ_max`. This is the same `sign(η·q·V)` keying as
the synchronous-phase branch above, and it is exactly the machine D1 builds.

**The bucket is asymmetric.** `separatrix()` spans from `ζ_u` to the **far turning
point** — the other root of `U(ζ) = U(ζ_u)`, on the opposite side of `ζ = 0`. The
potential is periodic-plus-tilt so this root is transcendental and there are many;
the right one is bracketed between `ζ = 0` and the *other* adjacent unstable point
(`U` is monotonic there, so the sign change is unique) and found with `brentq`. No
`±ζ_u` mirror. In the stationary limit the two barriers are degenerate; that is
detected **relative to the bucket depth**, not against `0.0`, and the far tip is
set to `−ζ_u` exactly — near that double root the level set is quadratic, so a
root-find would only reach `√eps`.

**Bucket *area* is deliberately not provided.** It is a non-elementary
(elliptic-type) integral, and the folklore `(1 − sin φ_s)/(1 + sin φ_s)` is itself
an approximation, so there is no exact reference to gate it against. The scope note
in `acceleration.py` that said "bucket area vs. `φ_s`" was loose wording for this
height factor; it has been corrected.

**Unstable branch.** With the guard gone, the `δ² ≤ 0` check is load-bearing: a
`φ_s` on the unstable branch (same `sin φ_s`, hence the *same energy gain*, opposite
`sign(cos φ_s)`) raises `ValueError` — asserted for both signs of `qV`, since only
stability distinguishes the two roots.

## End-to-end chain (D1 — implemented)

`examples/build_a_machine.py` owns the machine (a 192 m, 24-cell electron FODO
ring: inject 0.6 GeV → ramp → store 2.0 GeV → collide → account) and the
narration; `tests/analytic/test_end_to_end.py` owns the gates. **The gates are
seams only.** Every stage quantity is a pure function of one lattice, so
re-asserting a stage's own invariant on the chained run is green forever and
tests nothing — the same tautology D4 is an essay about. The discriminating
question for each assertion: *would it still pass if the value were recomputed
from a fresh standalone lattice?*

Conventions the chain fixed:

- **Magnets are geometric** (`k1l`, bend angle), so the optics is
  energy-independent — physically, the magnets ramp with the beam. Every energy
  dependence in the chain is the beam's.
- **Radiation damping is closed-form, never tracked.** accsim has no damped or
  stochastic map, so "store with damping" is a *data-flow handoff* (the store
  energy's `eps_eq`, `sigma_delta`), not a tracked `eps → eps_eq` convergence.
  The damping *times* say how long it would take.
- **`beta*` is a design parameter, not a matched insertion.** Stage 6's
  `luminosity`/`hourglass_reduction` are closed forms in `(eps, beta*, sigma_z)`.
- **There is no vertical-emittance model** — `equilibrium_emittance` is the
  horizontal one, and a flat uncoupled lattice has `eps_y = 0`. `eps_y` is an
  input (a coupling fraction), stated as such.

**The finding: the horizontal action is not cleanly adiabatic, and that is
physics.** Once RF and dispersion share a ring a loop closes that neither stage
owns — `x → ζ` through the dispersive one-turn entries `R51 x + R52 px`,
`ζ → δ` in the cavity, `δ → x` through `D_x`. The horizontal Courant-Snyder
action therefore carries a percent-level synchro-betatron ripple through the ramp
that does **not** shrink as the ramp slows. `D_y = 0`, so the vertical plane has
no such path and shows the `1/P0` law with a residual that *is* the finite ramp
rate (`∝ 1/n_turns`, demonstrated converging). Adiabatic-damping checks therefore
use the **vertical** plane; the horizontal ripple is asserted to still be there,
as an inequality between the planes.

**`sigma_z` has no independent reference in accsim, so its constant is pinned by
tracking.** `sigma_z = sigma_delta·|eta|·C/(2π Qs)` is the chain's three-stage
number (radiation × RF × lattice) and it reaches Stage 6 through the hourglass
factor — but every hourglass check is a *ratio*, and a ratio cannot see a wrong
constant. A particle launched at `(ζ, δ) = (0, σ_δ)` has
`ζ_max/δ_max = |η|C/(2π Qs)` by construction of the matched ellipse, measured off
the nonlinear tracker; that pins the constant (2π included) to <1% at low `Qs`,
with the residual being the same lumped-cavity `O(Qs²)` error as the tracked-tune
check and shown shrinking with `Qs`.

**`hourglass_reduction(sigma_z, beta*)` is asserted with keyword arguments**, on
purpose: at this design point `sigma_z ≈ beta*`, so a positional swap is
numerically plausible and otherwise invisible. It was made, and caught, during D1.

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

## Test-suite cost (2026-08-10)

Where the runtime goes, measured, and what was done about it. Recorded because the
dominant term is **not** in accsim's code and is easy to misattribute.

- **The reference suite is a compiler benchmark, not a physics benchmark.** Every
  `xt.Line` tracker build JIT-compiles a fresh C kernel through clang-cl. There is
  **no cache of any kind**: `xobjects/context_cpu.py::build_kernels` does
  `module_name = module_name or str(uuid.uuid4().hex)`, so each build gets a
  globally fresh name and can never hit a previous one. Measured on one test —
  `tests/reference/test_drift_xtrack.py` — **1 test → exactly 1 new `.pyd`**, and
  that compile *is* the entire cost of the test. The structural claim (no reuse is
  possible) is load-independent; the **12.2 s** that build took is an **upper
  bound** measured while the machine was heavily contended, not a pinned number.
- **Headline, both measured on a quiet box: 603.87 s → 343.12 s, a 43 % cut.**
  Full suite before (562 tests, 4 warnings) vs the new `-m 'not reference'` default
  (517 passed, 45 deselected). Reference is therefore ~261 s ≈ 43 % of the old run,
  ~5.8 s per test / ~7 s per actual compile. This is the only before/after pair in
  this section where both halves were taken under comparable load — quote this one.
- **Reference is ~40 % of the suite, not "the bulk"** — this corrects an earlier
  claim here that extrapolated 12.2 s × 45 ≈ 540 s and concluded reference dominated.
  It does not. Measured **back-to-back in one contention window**, which is what makes
  the two comparable: `-m reference` → **548.88 s** (45 tests), the new default
  `-m 'not reference'` → **819.52 s** (517 tests). That is a load-normalised split of
  **40 % reference / 60 % analytic**. The naive 12.2 s × 45 = 549 s "matched" only
  because *both* figures were contended; quiet, the per-compile cost is nearer 5 s.
  Applying the 40/60 split to the quiet 603.87 s full-suite baseline puts reference at
  ~240 s and analytic at ~360 s. (The ratio is measured; the split of 604 s follows
  from it only under the assumption that contention inflates both suites alike — do
  not quote the two component numbers as if they were measured directly.)
- **Compiles per reference test: 37 `.pyd` for 45 tests (~0.82), from a clean start.**
  Not the 1.0 that the single-file sample suggested — some tests reuse a tracker or
  never build a Line. The `.pyd` delta is load-independent, so it is the honest metric
  for any future attempt to cut compiles; wall-clock on this box is not.
- **A prebuilt-kernel mechanism *does* exist — it is just gated off here.**
  (Corrects an earlier claim in this section that there was none; that came from a
  grep run against the wrong path and was wrong.) `xtrack/tracker.py` takes
  `use_prebuilt_kernels=True` by **default**, and when a suitable kernel is found it
  skips compilation entirely and loads via `kernels_from_file`. But the lookup is
  guarded by `from xsuite import get_suitable_kernel, XSK_PREBUILT_KERNELS_LOCATION`,
  and on `ImportError` it sets `kernel_info = None` and **falls silently through to a
  full compile**. We install `xtrack`, not the `xsuite` umbrella (see the toolchain
  note above), so that import always fails and every Line build compiles. So the
  cause is *"the gate package is absent"*, not *"no mechanism exists"*.
- **Below that gate there is genuinely no reuse.** `xobjects/context_cpu.py::build_kernels`
  calls `compile_kernel(...)` **unconditionally** whenever `compile=True` — no
  existence check, no import-first path. Passing an explicit `module_name` only
  stabilises the filename and sets `clean_up_so = False`; it does *not* skip a
  compile, and `add_kernels` does not even expose the parameter to forward one. So
  once the prebuilt path is bypassed, `module_name = module_name or uuid4().hex`
  guarantees a fresh compile every time.
- **Installing `xsuite` to open that gate is an UNVERIFIED lead — not a recommendation.**
  Recorded so the next person starts from the mechanism rather than the grep. Two
  reasons it was not pursued: (i) `pip install xsuite` failed twice on transient
  `IncompleteRead` network errors, and with `--no-build-isolation` pip backtracked to
  **xsuite 0.6.0** (vs 0.58.0 current), which predates the prebuilt-kernel API — so
  the install that *succeeds* is the one that does not help; (ii) more fundamentally,
  `get_suitable_kernel` matches a **fixed set of element classes + config**, while the
  reference tests deliberately build many *different* Lines, so most lookups would
  return `None` and compile anyway. Expected payoff is partial at best. Anyone trying
  it should measure the `.pyd` delta (below) before and after — that is the honest
  metric, not wall-clock.
- **The artifacts leak, on Windows specifically.** `containing_dir` defaults to
  `"."` — the CWD, i.e. the repo root — and the cleanup at the end of
  `build_kernels` is guarded by `(os.name != "nt" or so_file.suffix != ".pyd")`, so
  on Windows the `.pyd` is deliberately **not** unlinked (a loaded DLL cannot be).
  These accumulate forever: 1056 `.pyd` in the repo root plus a 3 GB `Release/` of
  clang-cl intermediates had built up, 4.3 GB total, reclaimed 2026-08-10. All of it
  is already gitignored (`.gitignore` — `Release/`, `*.pyd/obj/exp/lib`, and the
  32-hex-char `/[0-9a-f]*.c` source spills); re-clean periodically.
- **Default selection is now `-m "not reference"`** via pyproject `addopts` — 517 of
  562 tests. This makes bare-`pytest` mean what CLAUDE.md always claimed it meant
  (the analytic suite) instead of silently pulling in the kernel compiles. Expect it
  to cut roughly 40 % off a full run — real, but *not* the order-of-magnitude the
  first pass here assumed; the remaining ~60 % is sympy in `tests/analytic` and is now
  the larger term.
  A command-line `-m` **overrides** an `addopts` `-m` (last-wins), so
  `pytest -m reference` still runs the cross-checks deliberately. CI is unaffected:
  it installs `.[dev]` only, so those tests already skipped for want of the dep.
- **`scripts/nicepytest.py` is the entry point, not bare `pytest`.** It drops the
  process to `BelowNormal` (POSIX: `nice +10`) *before* importing pytest/numpy, and
  child processes inherit the priority class — so one call covers the whole xdist
  worker pool. Motivation is concrete: this machine routinely runs several agent
  sessions each driving `pytest -n auto`, and a normal-priority accsim run both
  starves them and is starved by them (a measurement leg once sat 37 min producing
  nothing under ~60 competing python processes). Windows trap encoded there:
  `GetCurrentProcess()` returns pseudo-handle `(HANDLE)-1`, and without
  `restype = wintypes.HANDLE` ctypes zero-extends it to an invalid handle so
  `SetPriorityClass` fails **silently** — the restype and the return check are both
  load-bearing.
- **Parallelism helps `tests/reference` and *hurts* `tests/analytic`.** These are
  opposite workloads and must not share an `-n`:
  - `tests/reference` — **parallelises well.** 45 independent clang-cl compiles,
    CPU-bound, modest memory per worker, and no output collision is possible since
    every module name is a fresh `uuid4().hex`. It is ~40 % of the total (above), and
    it is the half that parallelises, so this is where `-n` pays. Run it as
    `nicepytest.py -m reference -n 8`.
  - `tests/analytic` — **keep serial, or `-n 4` at most.** The expensive tests here
    are *sympy derivations* (`test_riccati_root_derived_symbolically`,
    `test_skew_matrix_matches_symbolic_exponential`,
    `test_overlap_integrand_is_derived_from_rho1_rho2`, `test_mfpt_derivation_symbolic`)
    whose peak expression-tree footprint multiplies per worker. Measured 2026-08-10:
    `-n 8` alongside other sessions on this box took **883 s versus a 604 s serial
    baseline** *and* failed 4 tests with `MemoryError` — including numpy being unable
    to allocate **3 MiB**. That is the Windows commit limit, not physical RAM
    (63 GB total). CI is likewise serial. Priority and worker count remain
    complementary — lowered priority makes the suite *yield* a core it already holds,
    a smaller `-n` never takes it — but neither substitutes for the other.
  - Worker isolation itself is not the problem: the autouse `_reset_feature_switches`
    fixture is per-process state, so xdist workers cannot interfere. Memory is.
- **Do not target optimisation work off durations measured under contention.** In
  that `-n 8` run `test_tracked_aperture_cut_recovers_the_lifetime_xi` reported
  187.78 s against 8.81 s (+5.81 s setup) in a clean `-m slow` leg — roughly 20×
  inflation. Only compare timings taken back-to-back on a quiet box.
- **Worse: individual sympy test durations are not reproducible at all**, so per-test
  targeting is unsound even between two clean runs. Same test, same code, serial both
  times:

  | test | contended | quiet |
  | --- | --- | --- |
  | `test_moment_weights_close_symbolically` | 128.52 s | **15.79 s** |
  | `test_overlap_integrand_is_derived_from_rho1_rho2` | 42.47 s | **115.95 s** |

  The second ran ~3× *slower* on the quieter box — contention cannot do that. sympy's
  global cache and hash-ordered heuristics pick different solution paths per process,
  and cost swings by an order of magnitude either way. Only **aggregate totals** over
  the whole suite are meaningful here. Concretely: an earlier plan to rewrite
  `test_moment_weights_close_symbolically` (unexpanded `sp.integrate` + `sp.simplify`
  → the `sp.integrate(sp.expand_trig(sp.expand(...)))` form already used in
  `test_lam_tung.py`) was dropped on exactly this basis — at 15.79 s quiet it was
  never a hot spot, and the 128 s that motivated it was noise. Before touching a
  physics test for speed, measure it in isolation several times.
- **Turn counts are off-limits as a speed lever.** `N_TURNS = 10_000` and the other
  tracking gates need many synchrotron periods to be non-vacuous; shrinking them
  makes the tests pass without testing anything. Parallelism is the lever.
