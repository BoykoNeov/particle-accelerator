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

**Updated again 2026-08-17 (L3)** — for one element the inconsistency is now *gone*
rather than narrowed. The exact sector-bend map has **no division by the curvature left
in it**, so `Dipole(L, 0).track` is not a special case but the `h → 0` limit of the same
formula, and it agrees with `Drift(L).track` to `6.5e-19` (a few ulp — one map by two
arithmetic routes). A zero-angle dipole is a drift in `matrix` *and* in `track`. This is
the exception that shows the rule: it happened because a closed form existed, not because
the inconsistency was fixed, and `Quadrupole`, `Sextupole` and `Octupole` are unchanged.

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

  > **Superseded by L3 (2026-08-17).** Both arms of that control now read 100%: the
  > exact bend map has no division by the curvature left in it, so a zero-angle `Dipole`
  > *is* a `Drift`. The controlled experiment moved to `Dipole(L, 0, k1)` against
  > `Quadrupole(L, k1)` — **56% against 100%** — which isolates the only map still
  > missing, the curved quadrupole's. The bendy arc now reads `−0.28934` against
  > `−0.28934`. See *The dipole's exact map* below.

  > **Closed by L4 (2026-08-18).** The `Dipole(L, 0, k1)` control reads **100% against
  > 100%**, and the two `track` outputs agree to `1e-17` by independently written
  > arithmetic. Note the `angle = 0`: that magnet has no curvature, so it carries neither
  > the Maxwell curvature-sextupole kick nor the curvilinear-metric group, and 100% there
  > says nothing about a *bending* combined-function magnet — which does **not** reach
  > 100%, for a reason named in closed form. See *The curved quadrupole's expanded map*.

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
  is a *new* divergence created by L2, it is deliberate, and it is **load-bearing**:
  `test_the_gradient_bend_is_the_only_thing_tracking_is_still_blind_to` uses exactly that
  pair as a controlled experiment, since swapping one for the other changes the tracked
  chromaticity and nothing else. L3 did *not* close it — a curved quadrupole has no
  closed-form flow either — so this pair is now the sharpest statement of what is left.

  > **Closed by L4 (2026-08-18):** the gradient bend now takes the expanded
  > (`mat-kick-mat`) map, which at `h = 0` *is* L2's quadrupole map, and the two agree to
  > `1e-17` in `track` as they already did in `matrix`. The divergence L2 created lasted
  > two milestones and is gone; what replaced it as "the sharpest statement of what is
  > left" is the **bending** gradient magnet, where the expanded family drops the
  > curvilinear metric factor.
- **`is_symplectic_map` now *accepts* a correct exact map at small amplitude.** The
  `(ζ, δ)` residual is second order in the amplitude *and* suppressed by `1/γ₀²`; on a
  `γ₀ = 20` ring at amplitude `1e-3` it is `8.4e-10`, under the default `atol` of
  `1e-9`. So the wrong check does not merely reject correct maps — it can pass one for
  no reason connected to symplecticity. `test_roll.py`'s rolled quadrupole was clearing
  its `1e-8` by a factor of six and now uses `is_symplectic_map_canonical`.

Gates: `tests/analytic/test_exact_quadrupole.py` (16),
`tests/reference/test_quadrupole_xtrack.py` (4).

## The dipole's exact map (L3 — implemented 2026-08-17)

**A uniform field has a closed-form flow, and it is a circle.** A particle of momentum
`1+δ` moves, in projection onto the bend plane, on a circle of radius
`r = p⊥/h` with `p⊥ = √((1+δ)² − p_y²)`, and the map is that circle meeting the exit
face. So unlike L2's quadrupole, the **pure sector** bend's map is exact in the *angles
as well as in* `δ`, and unlike L1's drift it is exact in both at once.

Verified against `xt.Bend(model="bend-kick-bend")` to **1.9e-16**, and against an
independent plane-geometry construction (circle-meets-plane, sharing no arithmetic with
the implementation) to **1e-15** at bend angles up to `1.5 rad` and `δ` up to `0.3`. The
exact Hamiltonian `H = −(1+hx)√((1+δ)²−p_x²−p_y²) + h(x + hx²/2)` is `s`-independent and
is conserved by the map to `4.4e-16` — a check needing no reference implementation.

### The split at `k1` is *forced*, not chosen — and this is not L2's refused discontinuity

L2 declined to short-circuit `k1 == 0` because that would have made a map discontinuous
in `k1` for no reason but convenience. This looks like the same thing and is not:

- with `k1 = 0`, `p_y` is conserved, so `y' = p_y(1+hx)/p_z` is a **quadrature** over a
  known `x(s)` — which is *why* a closed form exists;
- with `k1 ≠ 0`, `p_y' = −k1 y` turns that into a second-order ODE with an `s`-dependent
  coefficient. **The geometric term and vertical focusing are mutually exclusive in
  closed form.**

So the sub-case has a strictly better map that the general family provably cannot
express. The remaining option for a curved quadrupole is MAD-X's expanded map (xtrack's
`mat-kick-mat`, `track_thick_cfd.h`), which is paraxial in the angles and therefore
**drops the very term L3 exists to add**. A combined-function bend is left on the affine
map, which keeps `matrix()` the exact origin Jacobian — the invariant that bounds the
whole axis and rules out every slicing family.

Measured separation between the two model families, so the boundary is a number:
`1.4e-5` on the reference states, against `1.9e-16` for the right one.

> **Taken up by L4 (2026-08-18).** The combined-function bend now has that expanded map,
> matched to `xt.Bend(model="mat-kick-mat")` at `1.0e-16`, and the invariant survived
> after all: two half-length Hill solutions compose exactly and the Maxwell kick is
> quadratic, so `matrix()` is still the *exact* origin Jacobian. What L3 called "paraxial
> in the angles" turned out to understate the cost — the family also drops the
> curvilinear metric factor `(1+hx)`, which is a first-order-in-`x` term and not a
> third-order-in-angle one. See *The curved quadrupole's expanded map*.

### The `+h⟨D_x⟩` half of K2's account, and the half K2's formula did not have

Per bend, to first order in the orbit, the Jacobian gains

```
M[y, δ] = M[ζ, p_y] = −p_y·ρ·sin θ                     ← K2's specified source
M[y, x] = +p_y·sin θ       M[y, p_x] = +p_y·ρ(1−cos θ)  ← plane coupling
M[x, δ] = −p_x·ρ·sin θ·cos θ      M[ζ, p_x] = −p_x·ρ·sin θ
```

with the partners `M[p_x, p_y] = −M[y, x]` and `M[x, p_y] = −M[y, p_x]`. Two facts here
are not guessable:

- **the coupling pair is `p_y` times the bend's own dispersion entries `R26` and
  `R16`.** An upright sector bend on a vertical orbit is therefore a **coupling source** —
  new in this package, which previously needed a skew quadrupole (G1), a rolled magnet
  (K2) or a misaligned multipole. It also transports horizontal dispersion into the
  vertical, and on a real arc **that path is the larger one**: K2's
  `Δd_y = p_y L (h⟨D_x⟩ − 1)` is the `δ` column alone and misses it, giving `3.3e-4`
  where the answer is `8.6e-5` on the L3 test ring. K2's 0.2 % agreement on its own
  rings stands; what is corrected is the formula's *scope*.
- **the horizontal response is not the plane swap of the vertical one.** `p_x` is not
  conserved, so the response feeds back through the bend's own focusing:
  `ξ'' + h²ξ = (3h/2)·sin 2hs`, `ξ(0)=0`, `ξ'(0)=−1`, giving `ξ(L) = −ρ sin θ cos θ`.
  Symmetrising the planes is 8 % wrong at `θ = 0.39` and no design-optics gate would
  notice.

Consequence: `closed_twiss_on_orbit` **raises** `CoupledLatticeError` on a bendy ring
with a vertical orbit, where the design `closed_twiss` does not. Use
`coupled_twiss_on_orbit`. The two are describing different maps and both are right.

### What it closes — K2's specification, from 0.2 % to 1.7e-8

`test_the_model_gap_is_fully_accounted_for_and_not_a_mystery` was written by K2 as a
*specification*: it could only put the dropped terms back by hand and note that doing it
for real meant exact maps for `Drift`, `Quadrupole` and `Dipole`. L1–L3 are those maps.

```
                       accsim design   accsim on-orbit   xtrack
rolled ring   D_y      −3.05e-5        −3.34250898e-4    −3.34250903e-4   (1.7e-8)
steered ring  D_y       0.0             2.12984605e-4     2.12984604e-4   (3.5e-9)
```

The design optics still reports the old answers and is still **right** to: the terms are
bilinear in `(p, δ)` and no 6×6 can hold them.

### Chromaticity: 58% → 100%, with bends

On a ring of thin quadrupoles and thick sector bends — *no drifts*, so the bend is the
only element with length — the tracked `dQ/dδ` equals `natural_chromaticity` including
F2's full dipole terms. Before L3 that ring's tracked chromaticity was **zero**: a thin
kick is momentum-independent and the linear bend was chromatically ideal. The control
runs that statement rather than remembering it (a `Dipole` subclass whose `track` is its
`matrix`). On the suite's bendy arc: `−0.28934` against `−0.28934`, up from `−0.1665`.

On a *steered* ring a residual appears that is **linear in the steerer** (`2.05e-5` at
`4e-4`, exactly zero without it): the analytic integral is taken over the design optics
while tracking sees the machine the beam is in. Its order is pinned, which is what
separates it from a missing share of the map.

### Numerics: the trap L1 predicted, and it was the biggest one yet

Transcribed as xtrack writes it, `x = (pz_out·h − dpx/ds − k)/(h·k)` builds an answer of
size `x` out of a numerator of size `h`. The origin Jacobian then comes out at `3.2e-9`
against `matrix()` and **degrades as the finite-difference step shrinks** — the signature
of cancellation, not truncation — which would have broken every design-optics gate.

Rearranged so nothing of size one is ever subtracted:

```
u = pz − 1 = (δ(2+δ) − p_x² − p_y²)/(pz + 1)          C = u − h x
px_out = px cos θ + C sin θ
Q      = (px − px_out)/h = px·h·½L²sinc²(θ/2) − C·L·sinc θ
x_out  = x cos θ + px·L·sinc θ + Q(px+px_out)/(pz_out+pz) + u·h·½L²sinc²(θ/2)
D/h    = arcsinc(w)·S·Q/p⊥,   w = (a−b)S,  S = Σ/2 + (a+b)²/(2Σ),  Σ = √(1−a²)+√(1−b²)
ζ     → ζ + L(1 − 1/rvv) − (δL/(1+δ) + D/h)·E/E₀
```

`4.9e-15`, improving with the step. Three points worth carrying:

- **`asin(a) − asin(b)` is folded into a single `asin`** via
  `a√(1−b²) − b√(1−a²) = (a−b)[Σ/2 + (a+b)²/(2Σ)]`, which is the standard identity
  rearranged so its *own* inner subtraction cancels too. `Σ → 2`, never zero.
- **There is no division by `h` left anywhere**, so the straight limit needs no branch:
  `Dipole(L, 0)` *is* `Drift(L)`, agreeing to `6.5e-19` (a few ulp, two arithmetic routes
  for one map). A weak bend degrades gracefully — `2.6e-13` at `h = 1e-4`, where the
  transcribed form is `1.4e-5`.
- `Δζ = L − Λ·E/E₀` is split as `L(1 − 1/rvv) − (path − L)/rvv` **and** the path excess is
  kept as `δL/(1+δ) + D/h`, never as `(1+δ)Λ − L`: the latter differences two `O(δ)`
  numbers whose leading parts cancel.

### Blast radius, and one cost that is not L3's body

Nine analytic tests across six files, and two reference tests — against L1's 29 and L2's
five. Each was restated with its new content rather than renumbered:

- **L2's 48%→100% control lost its teeth** and was moved, not widened: its two arms
  (a zero-angle `Dipole` and a `Drift`) are now the same map. The surviving controlled
  experiment is `Dipole(L, 0, k1)` against `Quadrupole(L, k1)` — byte-identical matrices,
  **56% against 100%** — which is exactly what the curved-quadrupole map will close.
- **A rolled bend is now symplectic only to first order in the roll** (`4.7e-8` at
  `φ = 0.02`, halving with `φ`). This is **not** the body: the aligned bend is symplectic
  to `3.7e-13`, both frame-change matrices are symplectic to `3.3e-16`, and a rolled
  *straight* dipole — plain rotation instead of the curved rigid motion — is symplectic
  to `2e-13`. The cause is that `frame_change()` returns the **affine linearisation** of
  the true frame change, which its own docstring notes "is exact for accsim's linear
  elements". It was; the body is no longer linear. `matrix()` and `kick()` are unaffected,
  so every K2 number stands. Making the frame change nonlinear inside `track` would close
  it and is a separate milestone.
- **A particle can now leave the model in a bend.** `test_moving_bucket`'s
  outside-the-bucket particle reaches `δ = 0.7` and `x = 1.4 m` and the exact map returns
  `NaN` rather than inventing a trajectory (L1's rule). It has already escaped by 1900
  bucket widths by then, so the escape is asserted on the turns before the loss.
- **The multipole-free share of "linear vs nonlinear tracking" has doubled once per exact
  map**: `0.9%` → `1.3%` → `2.6%` of the sextupole signal, and the chromatic beta-beat
  separation is now a factor of 12 rather than 100. Both are stated as measured ratios,
  because the honest reading is that each exact map narrows them again.

### A second reference gap closed, and where its old number went

`test_orbit_optics_xtrack.py` opened with a modelling difference it had to state before
anything else: accsim's elements were exactly linear, so an off-axis orbit changed
nothing about them, while xtrack's exact bends moved `β` by `6.4e-4` relative. Its test
closed with the hope that "a future milestone giving the bends their real off-axis map
has a number to improve on". That number is now **`5.4e-10`**.

What is worth recording is *where the `6.4e-4` went*: it did not shrink, it **moved to
the design route**, which now disagrees with xtrack by exactly what the on-orbit route
used to, and is first order in the orbit (asserted as that order). That is the correct
home for it — a 6×6 cannot hold a bilinear term, so linear optics is blind there by
construction, not by defect.

Downstream, the sextupole-induced β-change cross-check went from `1.35e-3` of the effect
to `2.8e-7`, and its tolerance was **tightened** `5e-3 → 1e-6` rather than left: a bound
sized for a model gap that no longer exists would hide any future regression. The
with-minus-without-sextupole *difference* construction is also no longer load-bearing —
the undifferenced β tables now agree to `1.1e-9` on their own, which is asserted
alongside it.

### The edged bend was a dark code path

`_track_body` composes `Edge(e2) · body · Edge(e1)`, and `tests/analytic/test_dipole_edges.py`
**never calls `track()`** — it compares matrices. So the composition had no gate: its
*order*, and the `h` passed to `_edge_matrix` (an edge kick is `h·tan e`), could both
have been wrong while every other L3 gate passed. Now pinned by rebuilding the
composition by hand, by asserting the reversed order is a *different* map (`>1e-6`), and
by re-checking the Jacobian identity and canonical symplecticity with the edges on. A
rectangular bend (`e1 = e2 = θ/2`) is used as the structural check: its edges cancel the
body's horizontal weak focusing exactly, so `R21 = 0` in the tracked Jacobian, which a
swapped or dropped edge would destroy.

Gates: `tests/analytic/test_exact_dipole.py` (15),
`tests/reference/test_dipole_xtrack.py` (3 new), and the converted
`tests/reference/test_roll_xtrack.py` and `tests/reference/test_orbit_optics_xtrack.py`.

## The curved quadrupole's expanded map (L4 — implemented 2026-08-18)

**The last element whose `track` was its `matrix`.** L3 proved the split at `k1`: a pure
bend's flow is a circle and has a closed form, a *curved* quadrupole's does not. So the
combined-function bend gets the **expanded** map — MAD-X's `track_thick_cfd`, xtrack's
`mat-kick-mat` — plus F2's Maxwell curvature-sextupole term as one centred thin kick:

```
mat(L/2) . kick(h k1 L) . mat(L/2)
```

which is *exactly* `xt.Bend(model="mat-kick-mat")` with `num_multipole_kicks=1` and the
`uniform` integrator, reproduced to **1.0e-16** on all six coordinates. `k1 = 0` still
takes L3's exact circle; nothing about the pure bend changed.

### The map, and where each piece comes from

With `q = 1+δ`, `x' = px/q`, `y' = py/q`:

```
K_x = (h² + k1)/q     K_y = -k1/q      G = h - k0 = h δ/q
A = -K_x x + G        B = x'           c1 = (1-C_x)/K_x

x  → x C_x + x' S_x + G c1              px → (A S_x + B C_x) q
y  → y C_y + y' S_y                     py → (-K_y y S_y + y' C_y) q
ζ  → ζ + L(1 - 1/rvv) - (Λ - L)/rvv,    Λ - L = h ∫x ds + ∫(x'²+y'²)/2 ds
∫x ds = x S_x + x' c1 - G c2            ∫u'²/2 ds = (A² t1 + A B S² + B²(L - K t1))/2
```

`h` is **not** divided by `q` — it is the geometry of the reference orbit, not a field
strength — and that asymmetry is precisely what makes `G` nonzero. `G` *is* dispersion:
the design particle feels no net drive, a stiffer one is under-bent and drifts outward.

The Maxwell kick is `ψ₃ = -(h k1/3)x³ + (h k1/2)x y²` (F2, above), so
`Δpx = h k1 L(-x² + y²/2)`, `Δpy = h k1 L x y`. It carries **no** `1/(1+δ)`, for the same
reason a `ThinQuadrupole` does not: a field changes every particle's *momentum* equally.

### `matrix()` is still the exact origin Jacobian — and that is not automatic here

Two half-length Hill solutions compose to the full one **identically** (same
inhomogeneous equation, and the path integrals add), and a cubic potential's kick has
**zero** Jacobian at the origin. So the composition's origin Jacobian is
`_combined_function_body` entry for entry — measured at `2.2e-16` with a `1e-8` step,
*improving* as the step shrinks. This is the invariant that bounds the whole L axis and
is what rules out every slicing family: a sliced bend's Jacobian is the product of the
slices' matrices, which is not `exp(L A)`.

### Numerics: three integrals with removable poles, and none of them branched badly

`c1 = (1-C)/K` is evaluated as `2 S(K, L/2)²` — the half-angle identity
`1 - cos u = 2 sin²(u/2)` — so the *transverse* map never divides by `K` at all, at any
`K`, with no branch. `c2 = (S-L)/K` and `t1 = (L - CS)/(2K)` cannot be written that way and
switch to their Taylor series at `|K L²| = 1e-2`:

```
c2 = -L³ Σ (-K L²)ᵐ/(2m+3)!            t1 = 2L³ Σ (-4 K L²)ᵐ/(2m+3)!
```

Five terms truncate at `1e-19` relative there, and the closed form has lost only `~1e-14`
to cancellation — so **neither side of the switch is the inaccurate one**, which is not
true of the scalar `_dispersion_integrals` (threshold `1e-9`, two terms) that `matrix()`
uses. `K_x = 0` exactly (the `k1 = -h²` tune) is verified against an ODE integration at
`4.4e-16`.

`ζ` is split as `L(1 - 1/rvv) - (Λ-L)/rvv` and the first term rationalised through
`(1+δ) + E/E₀`, exactly as L1, L2 and L3 had to: xtrack's own `length - length_/rvv`
differences two numbers of size `L`.

### ⚠️ What the expanded family drops — and it is **not** only `O(angle³)`

It solves `x' = px/(1+δ)` where the exact curvilinear equation is `x' = px(1+hx)/p_z`,
keeping the `(1+hx)` metric factor **only in the path length**. Evaluated on the dispersed
orbit `x = D_x δ`, that factor *is* F2's group

```
+ (1/4π) ∮ h (γ_x D_x - 2 α_x D_px) ds        + (1/4π) ∮ γ_y h D_x ds
```

and that group is what largely **cancels** the geometric `-β_x h²` focusing. So a
*bending* combined-function magnet's **tracked** chromaticity converges to
**F2 minus that group**, not to F2, and slicing does not close it.

Measured on one AG arc of eight bending gradient magnets (`k1 = ±0.35`, `θ = 2π/16`):

```
                                        Q'_x        Q'_y
accsim natural_chromaticity (F2)      +0.114815   -0.015121
xtrack rot-kick-rot   (exact)         +0.114815   -0.015121
xtrack bend-kick-bend (exact)         +0.114815   -0.015121
--------------------------------------------------------------
accsim tracked, 16 slices             -0.139529   -0.131975
xtrack mat-kick-mat, converged        -0.139733   -0.132062
F2 minus the metric group             -0.139733   -0.132062
```

Two families, two numbers, and accsim's two routes land on one each. **The gap is not
accsim's**: it is the expanded family's, it is the same size in both codes, and it is
named in closed form. `natural_chromaticity` is untouched by L4, is what the exact models
agree with, and remains the function to use.

⚠️ The blunt consequence, stated because it is a regression in one diagnostic: on a ring
of *bending* gradient magnets the **converged** tracked chromaticity is now further from
the truth than the pre-L4 blind map was (`0.254` against `0.234` on the arc above),
because that map contributed nothing at all where this one contributes an uncancelled
`-β_x h²`. That is the F1 failure mode (see *Dipole chromaticity*), and it is the price of
the only family with a closed-form linear part. The **map** is nonetheless strictly better
everywhere: it was `δ`-blind and is now exact in `δ`.

"Converged" is load-bearing, and was found by asserting it rather than by reasoning: at
**one** kick per magnet the *splitting* error happens to push the answer back across the
true value, so an unsliced ring reads closer than the blind map — by luck, and it reverses
as soon as the magnet is sliced at all. Both are pinned in
`test_the_gradient_bend_is_no_longer_chromatically_ideal`, so the unsliced number can
never be quoted as an improvement.

Recovering the metric term is a candidate milestone (**L5**), not this one: its
Hamiltonian `H_m = h x (px² + py²)/(2q)` generates both missing pieces at once and has
identity Jacobian at the origin, so it would fit the splitting — but neither MAD-X nor
xtrack implements it, so it would cost the bit-for-bit reference this milestone is
validated by.

### A *straight* gradient magnet has none of this

`h = 0` kills the metric factor and the Maxwell kick identically, and a drift-and-quad
ring has no dispersion for either to feed on. So L2's re-based control —
`Dipole(L, 0, k1)` against `Quadrupole(L, k1)`, byte-identical matrices — goes
**56% → 100%**, and their `track` outputs agree to `1e-17` by two independently written
arithmetic routes. Reading that control as covering the *bending* case is the one mistake
this milestone invites; both test files say so explicitly.

### A bend is now discontinuous in `k1` at zero, and the jump is **quadratic**

`Dipole(L, θ, 0)` takes the exact circle, `Dipole(L, θ, ε)` takes this map however small
`ε` is. The jump converges to `1.8e-5` at millimetre amplitudes and does **not** shrink
with `ε`. L2 refused exactly this kind of discontinuity; L3's argument is what justifies
it here (with `k1 = 0`, `p_y` is conserved and the vertical equation is a quadrature, so
the sub-case admits a strictly better map that the general family provably cannot express)
— but the size still has to be measured, and the obvious guess about its **order is
wrong**:

- it is `O(2)` in the coordinates, not L2's `O(angle³)` — a factor of two in amplitude
  gives a factor of four;
- **the expanded square root**, which for a bend enters `px' = h p_z - h` already at
  `O(p²)`: probed at `x = 0`, `Δpx = +h px² L/2` (`1.22e-5` against `1.26e-5`);
- **the dropped metric factor**, whose signature is a *bilinear* `x px` term that neither
  coordinate produces alone: the mixed second difference is `-h x px L`
  (`-8.47e-6` against `-9.42e-6`).

Two residuals, one of them this milestone's own model boundary.

### Blast radius

**Four** analytic tests across three files, against L1's 29, L2's five and L3's nine —
the axis is converging. Each was restated with its new content:

- L3's `test_a_combined_function_bend_is_deliberately_left_on_the_linear_map` was written
  to **fail loudly** when this landed. It did, and is now
  `test_the_split_at_k1_is_still_two_maps_and_no_longer_one_of_them_linear`.
- L2's 56% control closed to 100%, with the ⚠️ above written into it.
- **A combined-function bend moved symplecticity groups.** Its `track` was its `matrix`,
  so plain `is_symplectic_map` (in accsim's non-canonical `(ζ, δ)`) passed; the map is now
  exact in `δ` and that check **rejects** it. It joins the quadrupole and the pure bend in
  the canonical group.
- A **rolled** combined-function bend inherits L3's cost by the same mechanism —
  `frame_change()` is the affine linearisation of the true curved frame change, and the
  body is no longer linear: `6.2e-8` at `φ = 0.02`, first order in `φ`, against the pure
  bend's `4.7e-8`. Aligned it is symplectic to `1.6e-12`, and a rolled *straight* gradient
  magnet to `3.1e-12`, so it is the curved frame change and nothing else. `matrix()` and
  `kick()` are untouched, so every K1/K2 number stands.

Gates: `tests/analytic/test_curved_quadrupole.py` (25) and
`tests/reference/test_dipole_combined_xtrack.py` (4 new).

### What gates the Maxwell coefficients — and the four checks that cannot

J1's lesson again, and sharper. Blind to the `2:-1` split: **symplecticity** (any `(c₁,c₂)`
is a gradient kick — run as a control with the coefficient doubled, and it still passes),
**Maxwell** alone (`6c₁ + 2c₂ + hk1 = 0` is one equation for two unknowns), the
**origin-Jacobian identity** (the kick is quadratic there), and the **56% control**
(`h = 0`, so the kick is absent). What discriminates is **feed-down**: linearising `track`
about an orbit must reproduce F2's derived generator,

```
a21 gains -2 h k1 x₀       a43 gains + h k1 x₀       a23, a41 gain + h k1 y₀
```

— four numbers from two coefficients, so a uniform mis-scale shows in the size and a wrong
split in the ratio; the vertical-orbit pair needs a bend on a vertical orbit, which is the
coupling source L3 found. That, plus the `1.0e-16` xtrack match, is the whole gate on the
Maxwell half.

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
driving terms and normal-form machinery, decapoles and above, the 6D closed orbit
(**delivered by I4**) and misalignments as element attributes.

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

**Still out of scope** (as of I2): the 6D (RF-coupled) closed orbit — **delivered by
I4**, below; feed-down from octupoles
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
listed — the 6D closed orbit (**delivered by I4**, below), octupoles,
amplitude-dependent detuning, dynamic aperture. Misalignments as element attributes were on that list until **K1**, below.

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
- **Phase convention matches xtrack's `Cavity` — for a *positive* charge only
  (corrected 2026-08-26, N5).** xtrack applies
  `energy_kick = q·V·sin(phase + lag_rad − (2πf/c)·zeta/β₀)`, i.e. the same
  `φ = φ_s − k_rf·zeta` — **but its `q` is `fabs(q0)·charge_ratio`**
  (`beam_elements/elements_src/track_rf.h`), the *absolute* charge, where accsim uses
  the signed `ref.charge`. For an electron the two cavities are therefore exact
  negatives of each other and the correspondence is
  **`phase = φ_s + π`** (equivalently `lag = degrees(φ_s) + 180`), not `phase = φ_s`.
  Neither is wrong: accsim's kick is the physical `q E·v`, xtrack's makes `lag` mean
  the same thing for every species. Verified: accsim's full 6×6 one-turn map equals
  xtrack's on the `(zeta, delta)` block, so the coupled synchrotron eigen-tune
  matches `tw.qs` to ~1e-6 (positive charge, `test_synchrotron_tune_xtrack.py`) and
  all three 6D eigen-tunes match to 1e-9 on an electron ring with the extra `π`
  (`test_spin_sidebands_xtrack.py`). With the naive mapping the xtrack line comes out
  longitudinally **unstable** (eigenvalues `1.373`/`0.728`) and its 6D `twiss` dies
  inside the normal form with `Invalid n3` — loudly, not quietly.
  Prefer `phase` (radians): xtrack's `lag` is deprecated, and if both are set the
  effect is their sum.
- **Linear map** (`RFCavity.matrix`) is the small-amplitude shear
  `R65 = ∂δ/∂zeta|₀ = −(q V k_rf cos φ_s)/(β₀² E₀)` (only `M[DELTA, ZETA]`); it is
  symplectic (a shear, det = 1). The full `sin` kick (`energy_kick_delta`) is the
  tracking map (the pendulum whose separatrix is the bucket) — Stage-3 nonlinear
  tracking. **Stationary bucket only**, and *which* stationary phase depends on the
  **sign of the charge**: stability needs `Qs² = −(h η q V cos φ_s)/(2π β₀² E₀) > 0`,
  i.e. `η q cos φ_s < 0`. For a **proton** (`q > 0`) that is `φ_s = 0` below
  transition and `φ_s = π` above; for an **electron** (`q < 0`) it is the other way
  round, so every ring in axis N — electrons, well above transition — uses
  `φ_s = 0`. The accelerating `qV·sin(φ_s)` energy gain per turn is **Stage 5**.
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

## Beam losses / apertures (Stage 4 — implemented; momentum acceptance B4)

Acceptance boundaries with survival/loss accounting. All of them subclass
**`AcceptanceElement`**: an optics-transparent element (`matrix()` is the identity)
whose physics is the predicate `survives(states)`, with loss *accounting* done by the
tracking pass rather than the element. `track_bunch_losses` dispatches on that base
class, so a new boundary only has to implement the predicate.

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
- **`MomentumAperture(half_delta, center=0.0, length=0.0)`** (B4) — the longitudinal
  counterpart: `|delta − center| ≤ half_delta`. Only `delta` is consulted, so it is a
  *momentum* acceptance and not a full longitudinal one; a `zeta` boundary is a different
  object (the RF separatrix is not a rectangle in `(zeta, delta)`) and is deliberately not
  this element. Same inclusive-`≤` convention as `Aperture`.
  - ⚠️ **`center` must be the local closed-orbit `delta` on any ring with radiation.**
    Radiation drains `delta` through the arcs and the cavity restores it in one lump, so
    the periodic fixed point is *not* `delta = 0` at most elements; the swing is of order
    `U0/E`. Measured on the B4 ring (6.5 GeV, `U0/E = 3.8e-3`, `sigma_delta = 2.0e-3`),
    `delta_co(s)` runs from `−0.966 sigma` to `+0.921 sigma` — **1.887 sigma peak to
    peak**. Because the quantum lifetime goes as `e^xi`, a symmetric cut at the worst
    element is `xi = 1.73` one side and `7.20` the other where both should read `4.00`:
    an order of magnitude in the lifetime from a boundary that looks reasonable. Centring
    is also the correct physics — the closed form's amplitude is measured from the fixed
    point. Get the number by propagating the radiation fixed point (Newton on
    `track_once(s) == s` with `radiation="mean"`) and reading `delta` at the element's own
    position.
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

**The exact integral ships too (B4).** `quantum_lifetime_exact(A, sigma, tau_d)` returns
the MFPT integral itself, evaluated as the everywhere-positive series
`Sum_{n>=1} xi^n/(n n!)`. The equivalent `Ei(xi) - gamma - ln xi` is *not* what ships:
it is a difference of large near-equal terms as `xi -> 0`. Use the exact form whenever
`xi` is not large — at the `xi = 4` of a deliberately tight acceptance the asymptote is
wrong by **29%** (17.6674 against 13.6495, ratio 1.29436), and the asymptotic *series*
does not rescue it: `1 + 1/xi = 1.25` and `1 + 1/xi + 2/xi^2 = 1.375` **bracket** the
truth. The departure is the law `xi (exact/asymptote - 1) -> 1`, not "halves when xi
doubles" (measured 2.42 at `xi = 8 -> 16`).

**Mean first-passage time is not the decay constant.** `quantum_lifetime` /
`quantum_lifetime_exact` are the mean time for *one* particle at the core to reach the
aperture. What a survival curve measures is the slowest eigenvalue `lambda_1` of the same
generator with an absorbing boundary, and the two agree only as `xi -> infinity`:
`MFPT/(1/lambda_1)` = 1.135 (`xi=3`), 1.080 (`xi=4`), 1.005 (`xi=8`), 1.0004 (`xi=12`).
At a real ring's `xi` of tens they are the same number; at a gate-sized `xi` they are
not, and a fitted lifetime must be compared to `lambda_1`. The `lambda_1` route has a
**ceiling**: its symmetrising weight is `e^-w`, so past `xi ~ 20` double precision runs
out and at `xi = 30` the eigenvalue comes back *negative*. `quantum_lifetime_exact` is
exact there; use it, not the eigenvalue, at a real ring's `xi`.

**⚠️ A tracked decay is not the closed form, and the gap is physics (B4).** The closed
form is a *continuum* diffusion of the oscillation amplitude; a tracked bunch is a
*discrete* walk looked at once per turn. On the B4 ring (`tau = 220` turns, `xi = 3`) one
turn moves the normalised action by 0.23, and the tracked decay lands **37% above**
`tau/lambda_1`. Two separable owners, both measured in
`tests/analytic/test_quantum_lifetime_tracking.py`:

- **a coordinate cut is not an amplitude cut** (+22%). `|delta|` is sampled once per
  turn, so a particle whose amplitude has crossed the boundary survives until a sample
  lands near its extreme. It is **flat in `Q_s` to 1.5% across a factor of four**, which
  is what identifies it as *sampling* rather than phase-rotation delay — a rotation delay
  would scale with the synchrotron period. At `Q_s = 0.5` the two cuts collapse onto each
  other and both drop *below* the closed form (the half-integer synchrotron resonance:
  every sample lands on the same pair of phases).
- **finite steps** (+14% on top). Even a true amplitude cut runs long. The excess is
  proportional to the step `sqrt(2 xi) sqrt(2/tau)` and extrapolates to zero.

So gate a tracked lifetime against an **independent implementation of the same discrete
process**, and gate *that* against the closed form as the step vanishes. Gating tracking
directly against `quantum_lifetime` is a 37% failure that is not a bug.

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

## Radiation in tracking (B2 — implemented)

Stage 7 / B1 is entirely a **design-route** module: the radiation integrals ride the
Twiss functions and `damping_times` / `equilibrium_*` are closed forms on them. B2 is the
other half — `src/accsim/radiation_kick.py`, a per-element energy loss applied to a
*tracked* particle, so damping is something the simulation exhibits rather than asserts.
Opt-in per tracking call (`radiation="off" | "mean" | "mean_delta_only"`, default off);
`matrix()` and `kick()` are untouched, so every optics quantity is bit-for-bit unchanged.

### The map

Per element, once, with the field sampled at the **mid-point** of the traversal:

```
kappa  = |B_perp| / (B rho)_0 / (1 + delta)      # B_perp: field component ⊥ to motion
l_path = rvv * (L - Delta zeta)                  # the element's own path length
U      = (C_gamma / 2 pi) * E^4 * kappa^2 * l_path
f      = sqrt(1 - U (2E - U) / (E^2 - m^2))      # on-shell, rationalised
(1 + delta, px, py) *= f
```

- **One factor on all three momentum components** is the whole of the transverse damping.
  Photons leave along the direction of motion, so the momentum *vector* shrinks with its
  direction fixed. `pz` then scales by the same `f` **exactly**, and `x' = px/pz`, `y'`
  are invariant to the last bit — an exact statement, not a leading-order one.
- **`f` is the on-shell momentum ratio**, `sqrt((E-U)^2 - m^2)/sqrt(E^2 - m^2)`, written
  rationalised so it never subtracts two numbers of size `E` (the trap L1 recorded for the
  drift and L3 for the bend). It is `1 - U/(beta^2 E)` to first order and **exactly**
  `1 - U/E` in the massless limit — there is no second-order term to argue about.
- **The field comes from the element**, via `Element.normalized_field(x, y)` → `(bx, by)`
  normalised to `(B rho)_0`: `(k1 y, h + k1 x)` for a dipole, `(k1 y, k1 x)` for a
  quadrupole, the same through its 45° roll for a skew one, and zero by default. Sampling
  the gradient at the particle's **own** `x` is what makes a combined-function magnet's
  `J_x` differ from 1 in tracking.
- **Applied in the element's body frame**, so a misaligned magnet radiates according to
  where it really is: a particle on a *shifted* quadrupole's own axis radiates nothing.
- **Thin elements do not radiate** (no length, no path). Scope, not approximation.

### The wrong map, and why only one gate can see it

Reducing `delta` alone and leaving `px, py` is the natural-looking mistake. It gets the
**longitudinal** damping exactly right; inside the element it *anti*-damps the angle at
first order, `d(x') = +eps px (1+delta)^2 / pz^3`; and per turn it produces **exactly
zero** transverse damping, because `py` is never touched and the RF restores `delta`
(measured: a fitted `tau_y` 300,000× too long). Available as `"mean_delta_only"` purely so
the analytic suite can assert this.

### Costs, all deliberate

- **Not symplectic.** The first map in the package that must *fail* both
  `is_symplectic_map` and `is_symplectic_map_canonical`; the suite asserts the rejection.
- **`matrix()` is no longer the origin Jacobian of `track()`** with radiation on (the
  reference particle radiates too). That is exactly why it is a per-call mode.
- **One kick per element** evaluates the loss at the element's *entry* energy. Slicing
  converges it as `dE(N) = U (1 - (N-1)/N · U/E)` — asserted as that law.
- **The linear tracking path refuses it.** `radiation=` without `nonlinear=True` raises:
  there is no element to radiate in, and silently returning an undamped answer would be
  the worst outcome.

### Two numbers that are *not* errors

- **A tracked turn loses `U0 (1 - c U0/E)`, slightly less than the closed form**, because
  the particle radiates at a progressively lower energy as it goes round while `U0`
  evaluates everything at `E0`. On the 8-cell test ring `c = 1.26`, constant to 1% across
  a factor 64 in `U0/E` (its leading part is the `(N-1)/N = 15/16` over 16 bends; the rest
  is the orbit's own response). Gated as a stable coefficient, never as a tolerance.
- **The damping *partition* is the damped-map eigenanalysis, not the integral method.**
  The two are different methods and part company as `I4/I2` grows: 0.2% at `I4/I2 = 0.38`
  (a normal arc), 11% at `0.71` (a very strong one). Stage 7 already recorded this against
  xtrack at the ~1% level; B2 measures it from inside, and the load-bearing half is that
  **one** number explains both planes — whatever `I4/I2` the tracked map implies reproduces
  `J_x = 1 - I4/I2` *and* `J_z = 2 + I4/I2` together. So the sharp partition gates run on a
  normal arc, and Robinson's `J_x + J_y + J_z = 4` from the *measured* rates converges to
  4.000 as the lattice is sliced (4.026 → 4.0004).

### Measuring damping at all: the three-sided squeeze

`tau` in turns is `2E/(J U0)` — **144,000** on Stage 7's own 1 GeV ring, unrunnable. It
falls as `1/E^3`, so the test rings are deliberately fast: 8 FODO cells at 3 GeV
(`tau_y ≈ 2130` turns) and 20 cells at 5 GeV (`tau_y ≈ 1150`). Both are **above
transition**, so the cavity needs `phi_s = pi`, and `V > U0` so the RF can replace the
loss at all — the beam then settles at the `zeta` where it does. `tau_z` is read off a
synchrotron oscillation, so the ring must also satisfy `T_s << tau_z`: `Q_s ≈ 0.08` gives
43–65 periods per damping time. The equilibrium orbit is found by **Newton on the
radiation-on one-turn map**, not by tracking to it — with `tau_x` in the thousands of
turns a "converged" orbit is still drifting, and that drift contaminates every rate
measured against it.

### The xtrack cross-check needs its integration order matched first

`xt.Line.configure_radiation(model="mean")` radiates from a **thick** `xt.Bend` with no
slicing, so the comparison is per-element, like L1–L4. But xtrack sub-steps the loss
*inside* the element, and its default `integrator="adaptive"` resolves to **eight** uniform
steps for a plain bend — a 3.8e-5 disagreement at 5 GeV rising to 2.4e-3 at 20 GeV that
looks exactly like a wrong coefficient. Set `integrator="uniform"`,
`num_multipole_kicks=1` and the two are the same map. (`adaptive` picks for itself only
while `num_multipole_kicks` is at its constructed `0`; set it to 8 and `adaptive` gives
the *two*-step answer.)

What remains is **6.5e-9, with two named owners, both xtrack's**:

- `1.064e-8` from its **pre-2019 CODATA** elementary charge (`QELEM = 1.60217662e-19`
  against today's exact `1.602176634e-19`). `r0 = e/(4 pi eps0 m c^2)` is linear in the
  charge, so it lands on `C_gamma` and is energy-independent — a constants vintage.
- `2/gamma0^2` from its **ultra-relativistic approximations** (`gamma = gamma0(1+delta)`,
  `l/c` for `l/(beta c)`, `U/E` for the on-shell `U/(beta^2 E)`). accsim keeps the exact
  forms, so this term *dies with energy*, and the reference suite asserts that it does
  across a factor 80 in energy — which is what makes it a named owner and not a fitted
  tolerance.

### A sign error in xtrack's perpendicular projection (found by B2)

`track_magnet_radiation.h::direction_of_motion` computes
`iis = sqrt(1 - iix*iix + iiy*iiy)`. The `+` on the vertical term is wrong — the direction
cosines of a unit vector need `1 - ix^2 - iy^2` — and accsim uses the correct form. The
two therefore part company at large **vertical** angles, and the growth is **quartic** in
`py`, not quadratic: the projections differ by `2 B_par^2 iy^2` and `B_par = bx ix + by iy`
is itself linear in `py`. Measured on a combined-function bend at 20 GeV: `1.2e-8`
(i.e. nothing, just the usual residual) at `py = 1e-3`, `6.0e-7` at `2e-2`, `2.3e-5` at
`5e-2`. It is attributed from both sides — the quartic growth *and* the fact that
substituting xtrack's sign into accsim's own kick reproduces xtrack to the same `1.19e-8`
at every amplitude — so no tolerance is absorbing it. It is inert at the `py <= 1e-3` of
every other cross-check here, which is why it went unnoticed until a deliberate probe.

Gates: `tests/analytic/test_radiation_tracking.py` (24),
`tests/reference/test_radiation_tracking_xtrack.py` (6).

## Quantum excitation and the tracked equilibrium (B3 — implemented)

B2's mean kick is, taken alone, a lie about the physics: it damps every amplitude to
zero, and a real beam does not shrink to a point. Light comes in photons, and the random
walk that graininess produces is what holds the beam open. B3 adds
`radiation="quantum"` — the same map as `"mean"` with the radiated energy drawn from a
Gaussian of the right mean and the right variance — and gates the equilibrium it settles
into against Stage 7's closed forms, which were written a year earlier by a completely
separate route.

### The variance, and the one constant the two routes share

Emission in one element is a compound Poisson process:

- `n_γ = (5/(2√3)) α γ |κ l_path|` photons — the textbook `(5/(2√3)) α γ` per radian,
- each drawn from the synchrotron spectrum, whose moments in units of the critical
  energy `u_c = (3/2) ħc γ³ κ` are `⟨u⟩ = 8/(15√3) u_c` and `⟨u²⟩ = 11/27 u_c²`.

The mean of that sum is **exactly** B2's `U = (C_γ/2π) E⁴ κ² l_path` — the bridge
between the `α, ħc` system and the `C_γ, r_e` system, `r_e mc² = α ħc`, and the gate that
says the two halves of axis B are describing one effect. Its variance is

```
σ_U² = n_γ ⟨u²⟩ = (55/(24√3)) u_c U = 2 C_q E γ² κ U
```

written with the package's own `quantum_constant_cq`, so the tracked route and the design
route cannot carry two copies of the constant that sets the size of the whole effect.
(`C_q` and `C_γ` both now live in `radiation_kick.py` and are re-exported by
`radiation.py`, which is the direction the module dependency already ran.)

**The moments are integrated, not quoted.** `⟨u⟩`, `⟨u²⟩` and `⟨u³⟩` come out of
`∫_x^∞ K_{5/3}` in the analytic suite (swapping the order of integration once collapses
the double integral to a single quadrature), so the symbolic gate on `σ_U²` is derived
*from* the spectrum rather than from itself.

**The synchrotron phase-averaging ½ is load-bearing.** The photons kick `δ` only, but
`δ` is one coordinate of an oscillation: a kick at random phase adds `⟨Δδ²⟩` to the
*invariant* `a²`, and `⟨δ²⟩ = ⟨a²⟩/2`. Balancing `2⟨δ²⟩/τ_z = ½ Σ⟨u²⟩/E²` against
`1/τ_z = J_z U0/(2E)` gives `σ_δ² = C_q γ² I3/(J_z I2)` **exactly**. Drop the ½ and the
answer is exactly 2× — an energy-, geometry- and lattice-independent error that no
scaling gate could see, so it is pinned symbolically as exactly 2.

### Measuring the equilibrium: solve, don't track

Tracking to equilibrium is statistics-limited by construction, so it is not the sharp
gate. The sharp gate is the **discrete Lyapunov equation** `Σ = M Σ Mᵀ + D` — the fixed
point of "diffusion in, damping out" for the tracked map itself — solved exactly, with

- `M` the one-turn Jacobian at the radiation-shifted fixed point, computed with
  `radiation="mean"`. **Both** must use the mean map: Newton on a stochastic map does not
  converge, and finite-differencing a noisy map returns garbage that does not fail loudly.
- `D = Σ_i c_i c_iᵀ`, each `c_i` the noise element `i` injects propagated to the end of
  the turn. Built with a **stand-in generator** that returns a chosen number of standard
  deviations on one nominated draw and zero on the rest, which turns the stochastic map
  into a differentiable one — exercising the shipped code path, including the variance
  formula, with no statistics anywhere. Per-element Jacobians are accumulated backwards,
  which is `O(n)` rather than `O(n²)` and is what makes a sliced ring affordable.

Emittances come from the eigenvalues of `Σ S`, never from `σ_x²/β_x`: on these rings the
dispersive term `(D_x σ_δ)²` is a third of `σ_x²`, so dividing by `β_x` reports an
emittance ~2× too large.

**Summing each element's variance is wrong by 24% on the test ring.** A kick injected
early in the turn is partly rotated into `zeta` before the turn ends; propagating each
element's noise to the observation point is the whole content of `D`.

### Two departures from the closed forms, both with named owners

The two routes do **not** agree to round-off, and the reasons were separated rather than
absorbed:

1. **The finite synchrotron tune.** `σ_δ² = C_q γ² I3/(J_z I2)` and
   `ε_x = C_q γ² I5/(J_x I2)` are the **smooth-ring** result — they assume the
   synchrotron phase barely advances while the turn's photons are emitted. Solving the
   discrete map assumes nothing of the sort, so the two part company as
   `1 + c (2π Q_s)²`. The claim that `Q_s` is the *whole* story is gated sharply:
   **1.25 GeV at 30 MV and 5 GeV at 120 MV have the same `Q_s` and their departures agree
   to 4 parts in 100,000**, while `U0 ∝ E⁴` differs by 256×, the equilibrium spread by 4×
   and the emittance by 16×. Nothing but `Q_s` could do that. On a fixed geometry
   `c ≈ 0.098` for `σ_δ²`, constant to 0.5% over a factor 2.8 in `Q_s`; `c` itself is
   geometry-dependent, so it is the *order* that is asserted, not the number.
2. **B2's one-kick-per-element lumping**, which lands on **one plane only**: it is a
   ~0.6% offset in `ε_x` that slicing removes, while `σ_δ` moves by 3e-5 and is blind to
   it. Two owners with two different signatures — this one dies under slicing and is
   `Q_s`-independent, the other survives slicing and scales as `Q_s²` — so neither can be
   mistaken for the other or for a wrong `C_q`.

With both controlled (`Q_s = 0.024`, 8 slices) the two independent routes land on
**0.11%** of each other in *both* `σ_δ` and `ε_x`, in the same direction.

### What the emittance gate is blind to, stated up front

**The horizontal excitation is dispersion, not photon recoil.** A photon does carry away
transverse momentum, but the injected noise vector is exactly `(0, px, 0, py, 0, 1+δ)` —
one common factor, as in B2 — so its transverse part is smaller than its longitudinal
part by `px/(1+δ) ~ 2e-4`, and its effect on a *variance* is that squared. What excites
the horizontal plane is the energy kick meeting the dispersion: the off-momentum closed
orbit moves and the betatron amplitude jumps by `D_x Δδ`. That is the curly-H in `I5`.
Deleting the direct recoil from the injected noise entirely moves `ε_x` by **4e-6**.

So `ε_x` is a gate on `I5` and `C_q`, and it is blind to the kick's transverse arm to six
figures. The gate that is *not* blind to it is B2's vertical damping time, which is the
whole reason that one exists.

### No vertical excitation at all — exactly zero, not small

The photons leave along the direction of motion, so the model gives them no opening
angle. On a flat lattice the injected noise has an identically zero vertical component
and `ε_y` is **exactly** `0.0`, not merely small. Consequences worth knowing:

- The equilibrium `Σ` is **singular** and has rank 4; `np.linalg.cholesky` on it raises.
  Sampling an equilibrium bunch needs an eigen square root.
- From a *nonzero* vertical start the beam does not stop at a floor — it keeps damping.
  The noise on `py` is **multiplicative** (`py (f − ⟨f⟩)`, proportional to `py` itself),
  so it perturbs the damping *rate* and leaves the fixed point at zero.
- The real floor is the `1/γ` photon opening angle,
  `ε_y = (13/55) C_q ⟨β_y/|ρ|³⟩ / (J_y I2)`, omitted by construction — the same
  flat-lattice boundary Stage 7 records from the design side and G1's
  `equilibrium_emittances_coupled` fills from coupling.

### The Gaussian is unclamped, deliberately

With `n_γ ~ 16–24` photons per magnet the relative fluctuation is `√(4.30/n_γ) ≈ 0.4–0.5`,
so `u < 0` — an energy *gain* — sits at about 2 σ and happens in **1–3% of draws**. It is
not a tail event. Clamping at zero would bias the mean **and** the variance by ~1%, which
is five times the agreement the equilibrium gates achieve; an unclamped Gaussian keeps
both exact, and the on-shell factor handles `f > 1` without a branch. It is asserted as a
measured boundary, against xtrack, rather than left as a caveat.

### The xtrack cross-check compares two genuinely different processes

`configure_radiation(model="quantum")` in xtrack is a real compound Poisson process:
exponential free paths, each photon's energy rejection-sampled off `K_{5/3}`, subtracted
one at a time. accsim draws one Gaussian and never counts a photon. So everything that
agrees is a statement that **only the first two moments matter** — the justification for
the Gaussian, checked against the thing it approximates.

- **Standard deviation of one magnet's loss: 0.18%**, against a statistical floor of
  0.16% on 200,000 particles. Mean: 0.05%. Setup carries over from B2 unchanged
  (`integrator="uniform"`, `num_multipole_kicks=1`).
- **The shape is where they part company, and it is not subtle**: xtrack's skewness is
  −0.91 and it *never* gains energy; accsim's is +0.003 and it gains 2.6% of the time.
- **xtrack's own skewness counts its photons.** For a compound Poisson sum the skewness
  is `⟨u³⟩/(√n_γ ⟨u²⟩^{3/2})`, so inverting it recovers `n_γ` — and using accsim's
  spectrum moments it lands within 5% of `(5/(2√3)) α γ θ`, the rate xtrack computes
  independently from `α`. The number the Gaussian throws away, measured from the tracking
  that shows it is being thrown away.
- xtrack's photon-record API (`start_internal_logging_for_elements_of_type`) returned no
  photons on this build, so the spectrum is checked through the skewness route above
  instead of directly.
- **`ħc` is not a third named owner.** xtrack hardcodes `1.973269804593025e-7` and the
  package rounds it; they agree to 5e-11. B2's two owners (the pre-2019 elementary charge
  and the ultra-relativistic approximations) are unchanged. For the *variance* xtrack
  additionally uses `β0 γ0` in the photon rate and `γ² γ0` in the critical energy, so its
  diffusion carries `(1+δ)⁴` where the exact result carries `(1+δ)⁷` — a `3δ` effect that
  averages to `O(δ²) ~ 1e-6`, far below any stochastic floor.

### Deliberately not built — *superseded by B5 (2026-08-25)*

A **photon-resolved sampler** (compound Poisson off the true spectrum) was left out here,
on the grounds that the equilibrium depends on the emission process *only* through its
first two moments, which the Gaussian matches exactly. That reasoning held: B5 built the
sampler and **every number this milestone gates is unchanged**. What B5 added is the
tail, and what it found there is that the hard-photon loss channel the argument above
gestured at does not exist on any ring this package can build — see *Photon-resolved
emission (B5)* below.

### API

`rng: np.random.Generator` is plumbed through `Element.track` and every `Tracker` entry
point (`track`, `track_once`, `track_turns`, `track_bunch`, `track_bunch_losses`) and is
**required** for any model in `radiation_kick.STOCHASTIC_MODELS`. Asking for `"quantum"`
without one **raises**: the package never seeds a global generator, because an unseeded
stochastic track is not reproducible. Same convention as `orbit.misalign`.
`mean_radiation_kick` is retained as an alias for the now-more-honestly-named
`radiation_kick`.

One existing gate changed: B2's
`test_radiation_without_the_nonlinear_path_raises_instead_of_being_ignored` used
`radiation="quantum"` as a name that was unknown then and is a real model now, so its
"unknown model is refused" check needed a name that is still unknown.

Gates: `tests/analytic/test_radiation_quantum.py` (34),
`tests/reference/test_radiation_quantum_xtrack.py` (7). The settling gate tracks 600
particles for five damping times and costs ~43 s — the dominant cost in the file (61 s
total), and the roadmap's pre-committed gate, so it stays in the default suite.

## Photon-resolved emission (B5 — implemented)

`radiation="photons"` replaces B3's Gaussian with the thing it stands in for: a Poisson
count of photons per element, each energy drawn from the real synchrotron spectrum. The
milestone's result is that **almost nothing changes**, and the value is entirely in the
one thing that does.

### Which spectrum — a factor of 4.297, not a factor of 1.3

The textbook figure is the **power** spectrum `F(x) = x ∫_x^∞ K_{5/3}`; photons are
counted off the **number** spectrum `F(x)/x`. In units of the critical energy
`u_c = (3/2) ħc γ³ κ`, the sampler draws from

    p(x) = (3 / 5π) ∫_x^∞ K_{5/3}(t) dt,   ∫_0^∞ p = 1.

Normalising `F` as a density instead makes every photon `<x²>/<x>² = 4.297` times too
energetic — a pure number nothing dimensional would catch, and *not* the 1.3 that the
ratio of the two means looks like at a glance (that was written down wrong first, and the
gate caught it). 4.297 is the same constant B3 already uses in the other direction, to
count xtrack's photons out of a relative fluctuation.

### Every constant derived, none quoted

sympy integrates `K_{5/3}` outright, so the normalisation and all three moments are exact:

| quantity | value | used by |
|---|---|---|
| `∫_0^∞ K_{5/3}(t) t dt` | `5π/3` | the normalisation |
| `<x>` | `8/(15√3)` | B2's mean loss `U` |
| `<x²>` | `11/27` | B3's `photon_energy_variance` |
| `<x³>` | `224/(135√3)` | the skewness, which counts photons |

B3 obtained the first two by quadrature; the third is new and is what turns a loss
distribution's shape back into a photon count.

### The two quadrature traps, both recorded as gates

- **`quad(K_{5/3}, X, ∞)` in one piece silently returns zero for small `X`.** It emits an
  `IntegrationWarning` and no exception, and a cumulative distribution built on it comes
  out at exactly **2/3** of the truth — because the missing `x ∫_x^∞ K` half is exactly
  half of the `∫_0^x K t` half that survives. Every integral in `photon_spectrum` splits
  at `t = 1`: below it in `log t` (turning the `t^(-5/3)` singularity into a decaying
  exponential), above it in the exponentially-scaled `kve`.
- **The exceedance collapses to one quadrature.** Swapping the order of integration gives
  `P(x > X) = (3/5π) ∫_X^∞ K_{5/3}(t)(t − X) dt`, which in scaled form is `e^-X` times a
  well-conditioned integral. So `photon_log_survival` is exact at `X = 640`, where the
  answer is `e^-636` and no histogram could hold anything.

`photon_number_cdf` is relatively accurate only down to `x ~ 1e-20`; below that `quad`
runs out of dynamic range. That is a quadrature floor, not physics, and it is documented.

### The sampler is an inverse, which is what makes it gate-able

Inverse-transform sampling makes each photon energy a deterministic function of one
uniform, so a test feeds a chosen quantile and checks the answer against the quadrature —
including at an exceedance of `1e-16` (a 33 `u_c` photon), where sampling would need
`1e18` draws to place one. The inverse is tabulated once per process (~2.5 s of
quadrature) in the two variables the distribution's two shapes are straight lines in:
`log x` against `log q` below the median (`x ∝ q³`, coefficient
`(27/10π) 2^(2/3) Γ(5/3) = 1.2316`) and `x` against `log P(x > X)` above it. Accurate to
**~1e-9** relative across the whole range, measured rather than assumed.

### The loss replaces the classical one; it does not perturb it

`"quantum"` adds a zero-mean Gaussian draw to `u`. `"photons"` **is** `u`. Adding it by
analogy with the Gaussian would double the mean loss — the natural copy-paste slip.

### What must not change, and does not

`n_γ <u> = U` and `n_γ <u²> = photon_energy_variance` are identities, gated to **1e-13
through the shipped code path** on off-axis trajectories with non-zero `ζ` — where a `κ`
or an `l_path` computed differently between the mean route and the photon route would
show and a symbolic identity could not. Both dimensional numbers are captured *out of*
the running kick by a stand-in generator rather than recomputed in the test.

Consequently the diffusion matrix `D` is the same entry-by-entry in all 6×6, and so is
the equilibrium beam — same momentum spread, same horizontal emittance, and the vertical
still **exactly** zero (these photons leave along the direction of motion; the `1/γ`
opening-angle floor is the thing neither model has).

### What does change — three signatures, all pre-committed

1. The loss can **never** be negative, against the Gaussian's deliberate 2.6% of energy
   *gains*.
2. It is skewed, and the skewness inverts to the photon count by
   `<u³>/(√n_γ <u²>^(3/2))`. In `delta` it lands at **−0.92** against xtrack's measured
   **−0.91**.
3. And **none of that survives a turn.** Crossing `N` magnets suppresses the skewness as
   `1/√N`, gated over a factor of 100 in `N`. One turn of the B3/B4 ring is already down
   to `−0.129` with a Gaussian kurtosis. (That is 11% short of `−0.92/√40`, because a
   turn is a *weighted* sum of its photons — each magnet's loss is fed through the
   remaining `R56` into `ζ` and then through the cavity — not a plain one.)

### The hard photon does not exist

The milestone's most exposed pre-commitment, and both halves land.

**The lifetime does not move**: 1154 turns against 1240 from the same frozen bunch, a
ratio of 0.930 where one standard deviation is 8.2% — 0.86 σ. The statistic is a *fitted*
decay rather than binomial marks, because a fit uses every turn: 400 particles then give
a 25% three-sigma band where three marks at 800 would give 37%, and 25% is narrower than
the 37% departure from the continuum that B4 already measured.

**And the reason is not that the tail is unimportant — it is that the tail does not
reach.** Emptying the bucket in one photon needs `X = E δ_acc / u_c` critical energies:

| ring | `u_c/E` | `δ_acc` | `X` | `log P(x > X)` |
|---|---|---|---|---|
| B4's 6.5 GeV ring at `ξ = 3` | `1.47e-5` | `4.96e-3` | **337** | **−341** |
| the roadmap's 5 GeV / 10 m example | `5.55e-6` | `3.5e-3` | **631** | **−636** |

Restated as something holdable: the hardest of the `4.0e8` photons emitted in a whole
400 × 1200 tracking run is **17.0 `u_c`**, a factor of twenty short, and ten times the run
buys only `log 10 = 2.3` more, because the largest of `N` draws off an exponential tail
grows as `log N`.

**The claim has to be the narrow one.** No single photon carries a particle *from the core
across* the acceptance. Particles are of course lost *at* an emission — that is the only
place `delta` ever falls — but the photon that finishes the job is an ordinary one
arriving at a particle the random walk has already carried to the wall. "Graininess is
what knocks particles out" is true only in that trivial sense.

### The cross-check B3 said it could not write

B3's reference arm is useful *because* the two codes did different things. They no longer
do, so `tests/reference/test_radiation_photons_xtrack.py` compares two genuine
compound-Poisson processes sampled by unrelated numerical routes (accsim: inverse
transform off a quadrature table; xtrack: rejection against `K_{5/3}` along an exponential
free path). The tail agrees **pointwise to better than 1% out to one draw in a thousand**
(−0.44%, −0.12%, +0.66%) where B3's Gaussian is **19.4% low**; the third and fourth
moments match; both count the same photons; and the median separates all three for free
(below the mean for a skewed distribution, exactly on it for a Gaussian).

Two lessons that first appeared as failures, both now in that file's prose:

- **xtrack's emission is not seeded by this suite.** Its sample is redrawn every run, so
  every gate is a *two-sample* comparison and its budget needs the `√2`. Without it a
  routine 2.9-σ fluctuation reads as a 4.1-σ failure. (B3's arm carries the same exposure
  with a `3.0 × floor` budget and has not tripped.)
- **An extreme *value* has no bounded variance on an exponential tail.** Gating xtrack's
  maximum failed on an ordinary redraw. The gate counts traversals past the Gaussian's
  `4.5 σ` ceiling instead — Poisson, so `1/√n` — where it expects 0.7 and both photon
  codes deliver ~70.

### A precision fix in B2's kick, surfaced by reaching for 1e-13

`out[delta] = f*(1 + delta) - 1` subtracts two numbers of size 1 to produce one of size
`1e-7`, keeping six digits of the increment. Rewritten as
`delta + (f - 1)(1 + delta)` with the rationalised `f - 1 = -shrink/(1 + f)`, it keeps
full relative precision — the same trap this module's docstring already warned about for
`E`, one level down. The 96 analytic gates of B2/B3/B4 are unmoved by it.

### API and cost

`RADIATION_MODELS` gains `"photons"`; `STOCHASTIC_MODELS` gains it too, so it **requires**
an explicit `rng` exactly as `"quantum"` does. New public helpers in `radiation_kick`:
`critical_photon_energy`, `photon_rate`, `fine_structure_constant` (computed from the
species' own `r_0 mc²/ħc`, the one bridge between the `α, ħc` and `C_γ, r_0` systems).
`accsim.photon_spectrum` is the dimensionless spectrum and its sampler, and knows nothing
about rings.

`sample_photon_sum` draws off `rng` in a fixed order — the Poisson counts for the whole
input at once, then one uniform per photon — so a bunch and a single particle consume the
generator differently and the bunch-vs-particle gate is **distributional**, not
draw-by-draw.

Cost: ~16 uniforms per particle per magnet, a few times dearer than one Gaussian.
Gates: `tests/analytic/test_photon_spectrum.py` (42, ~5 s),
`test_radiation_photons.py` (23, ~25 s), `test_photon_equilibrium.py` (3, **~155 s** — the
largest file in the analytic suite, and it says so in its own docstring),
`test_photon_lifetime.py` (6, 73–128 s), `tests/reference/test_radiation_photons_xtrack.py`
(12). The two expensive files are the roadmap's pre-committed tracking gates; parts 1 and 2
buy their sharpness from determinism instead, which is why the price is paid twice and not
everywhere.


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


## Chromatic functions (M1 — implemented)

`chromatic_functions()` reports how the linear optics move with momentum, in the
**MAD8 physics manual §6.3** normalisation that both MAD-X and xtrack use:

```
b_u = (dbeta_u/ddelta) / beta_u
a_u = dalpha_u/ddelta - (dbeta_u/ddelta) * alpha_u / beta_u
w_u = sqrt(a_u^2 + b_u^2)
```

Note the asymmetry, which is the place a remembered formula goes wrong: `b` **is**
divided by `beta`, `a` is **not**, and `a` carries the `- dbeta * alpha / beta`
correction rather than being a bare `dalpha`. The raw derivatives `dbeta_u` [m] and
`dalpha_u` [1] are reported alongside, so a caller never has to un-normalise.

**The derivative is with respect to `delta`, not `pzeta`.** The two differ by
`beta0` factors, so on an ultra-relativistic ring they agree to round-off and the
choice looks like a naming preference; it is not, and the reference suite pins it
by rebuilding xtrack's reported `bx_chrom`/`ax_chrom` from a finite difference of
xtrack's *own* `betx`/`alfx` in `delta`. xtrack's source is itself ambiguous on this
point — its finite-difference site divides by a variable named `ddelta_local`, while
its non-periodic branch names the same quantity `dbetx_dpzeta`.

**Both quantities are central differences, deliberately.** The references compute
them the same way, so a disagreement arbitrates the *maps* rather than the
truncation order of two different expansions — the argument B2 established. The
consequence is that `delta` is a **step size**, not a tolerance: the error is
`O(delta^2)` from truncation and `O(orbit noise / delta)` (or `/delta^2` for a
second difference) from the closed-orbit solve, so it is bounded at both ends and
the default `1e-3` sits in the flat middle for the rings this package builds. Gates
are written on the **convergence order**, never on a value at one step.

## The drift model is what splits `Q''` on a bendy ring (M1 measured it, M2 named it)

`second_order_chromaticity()` returns the plain second difference

```
Q'' = (Q(+d) - 2 Q(0) + Q(-d)) / d^2
```

so it is `d^2Q/ddelta^2` and **not** the coefficient of `delta^2` in
`Q = Q0 + Q' delta + Q'' delta^2 / 2`, which is half of it. It differences
`tunes_on_orbit`, which carries the integer part of the tune — a second difference
of *fractional* tunes is wrong by an integer whenever two of the three sample points
straddle a half integer.

**M1 found three codes giving three answers on a ring with bends** — accsim
`0.79307`, xtrack `0.75202`, MAD-X `0.70441` — while all three agreed on `Q` to ten
digits and on `Q'` to seven, and agreed with each other and with a sympy closed form
on a **bend-free** ring. It shipped the number as an unarbitrated boundary.

**M2 settled it. The cause is the drift model, and accsim is right.**

| code | drift map | `Q''_x` (M1's arc) |
|------|-----------|--------------------|
| accsim | exact, `x += L px / pz` | `0.79307` |
| xtrack, `Drift(model="exact")` | exact | `0.79309` |
| xtrack, default | paraxial, `x += L px / (1+delta)` | `0.75205` |
| MAD-X TWISS | paraxial | `0.70441` |

`pz = sqrt((1+delta)^2 - px^2 - py^2)`, so the exact map is the paraxial one times
`1 + (px^2 + py^2)/(2 (1+delta)^2) + ...`. Three consequences, and they are the whole
of the phenomenon:

- **At `px = py = 0` the two maps are identical at every momentum.** A ring whose
  closed orbit is straight cannot tell them apart however far off-momentum it is
  asked, which is why M1's bend-free control was a genuine three-code agreement and
  not a lucky one.
- **With bends the closed orbit carries `px ~ D_px delta`**, so the difference is
  `O(delta^2)` — it leaves `Q` and `Q'` untouched and lands squarely on `Q''`.
- **`D_px` is proportional to the bending angle**, so the gap is proportional to its
  **square**. That is exactly the scaling law M1 measured against MAD-X
  (`gap/angle^2` = `8.91`, `8.22` at `0.03` and `0.06` rad) and attributed to "the
  longitudinal constraint". The analytic suite now reproduces the same law inside a
  sixty-digit arbiter with neither reference code present.

### How it was found, and what M1 got wrong

M1's inference was **valid reasoning from a false premise**. It established that
accsim's `Dipole` Jacobian equals `xt.Bend`'s off-momentum, and generalised that to
"identical maps"; from identical maps about identical orbits, different tunes are
impossible, so the spread had to be in tune extraction. The premise was false because
only *one element* was ever compared. Walking the closed orbit element by element
gives:

```
Quadrupole   6.2e-11 on-momentum    5.3e-10 off-momentum
Dipole       6.0e-10 on-momentum    6.7e-10 .. 1.1e-9 off-momentum
Drift        1.0e-10 on-momentum    6.4e-08 .. 1.0e-07 off-momentum
```

The drift is a hundred times every other element, and **only** off-momentum. The
reason it went unchecked is worth keeping: L1 had shipped the drift *exact*, so it
read as settled — but L1 validated the drift's **map**, not its agreement with
xtrack's **default configuration**, and those are different claims. The same trap is
live for any element whose reference has more than one model.

Two further numbers M1 quoted were below the resolution of the effect, and neither
discriminated: the one-turn tune difference being explained is `2e-8`, while
"accsim's two tune routes agree to seven digits" is `1e-7` absolute, and the `5e-9`
dipole-Jacobian threshold was called "the finite-difference floor" when the actual
floor is `~7e-10`.

### The arbiter: a ring whose `Q''` is derived, not compared

`tests/_m2_minimal_ring.py` builds
`ThinQuadrupole(+0.9) Drift(0.5) Dipole(1.0, 0.12) ThinQuadrupole(-0.9) Drift(0.5)`
and derives its `Q''` from lab-frame geometry — the bend as a circle of radius
`p_perp/h` meeting the exit face — at **sixty** decimal digits, once per drift model:

```
exact drift     Q''_x = 0.3073788909    Q''_y = 0.2985909737
paraxial drift  Q''_x = 0.2932235794    Q''_y = 0.2938154492
```

accsim converges onto the exact pair at second order in `delta` (residual `4.1e-5`,
`1.0e-5`, `2.6e-6` as `delta` halves from `1e-2`); xtrack's default reproduces the
paraxial pair to `4e-6`; xtrack's `model="exact"` reproduces the exact pair to
`3e-6`; MAD-X lands `7.0e-4` (horizontal) and `7.3e-4` (vertical) from the paraxial
pair — the same residual in both planes, against a drift-model split of `1.42e-2` and
`4.78e-3` — so the drift explains 95% of MAD-X's gap in `x` and 82% in `y`, and the
leftover is one property of its second-order TWISS maps rather than two unrelated
discrepancies.

Three points about the design of that ring, each of which had to be got right:

- **It needs a drift.** The roadmap's pre-committed "one thin quadrupole plus one
  sector bend" cannot show the effect at all, because the effect lives in the drift.
- **It needs two quadrupoles.** A sector bend focuses horizontally only, so a single
  quadrupole leaves one plane unstable.
- **The bend must be derived, not transcribed.** `exact_sector_bend_map` is heavily
  rearranged for numerical stability (a rationalised `pz - 1`, an `arcsinc`, no
  `1/h`); porting that arrangement into the arbiter would test it against itself. The
  independent geometric construction agrees with it to `2.9e-15`.
- **mpmath, not sympy.** A second difference at `delta = 1e-12` carried at sixty
  digits has `O(delta^2)` truncation near `1e-24` and round-off near `1e-36`. The
  sympy route (a third-order Taylor series in five variables about the closed orbit)
  was written and reaches the same place, but is far too slow for the analytic suite.

### What is *not* claimed

MAD-X's TWISS has **no** exact-drift option, so agreement with MAD-X on a dispersive
ring is unreachable by construction, not a bug to chase. accsim's ring parameters
appear in `tests/reference/test_chromatic_optics_xtrack.py`; the two codes' `Q''_x`
still differ by `~5%` there **on xtrack's default settings**, and that assertion is
deliberately kept — it is a real difference between two documented models, and a
future change that quietly removed it would mean accsim had stopped being exact.

### Two facts found on the way, both worth keeping

- **The sextupole reaches `Q'` and `Q''` at different powers of `k2l`.** A sextupole
  at dispersion sits at `D_x delta` and feeds down a gradient `k2l D_x delta`. That
  is first order in `delta`, so it lands on `Q'` **linearly** — exactly so, `dQ'/k2l`
  is one number to nine digits — and by that route it cannot contribute to a second
  derivative at all. `Q''` is reached only at second order in the perturbation and is
  therefore **quadratic** in `k2l` (measured exponent `2.02`, approached from above
  because a cubic term is also present). A pre-committed expectation of "linear" was
  wrong here; the gate is now the pair of exponents.
- **xtrack's nonlinear dipole fringe (`edge='full'`) is invisible on-momentum.** It
  moves neither tune at `delta = 0` to thirteen digits, and acts only at *second*
  order in `delta`, in the **vertical** plane alone. accsim's `Dipole` uses the
  linear hard-edge kick of `_edge_matrix`, which is the identity at `e1 = e2 = 0`,
  so `edge='suppressed'` is the apples-to-apples xtrack setting. Since `Q''_x` is
  identical under both settings, the edge model is **not** what explains the
  horizontal split above.

### `natural_chromaticity`'s slicing is coarser than it looks

`natural_chromaticity(lattice, slices=64)` integrates the beta-weighted gradient by
trapezoidal sub-slicing, so its error falls as `1/slices^2`. On a modest arc
(3 cells, `Dipole(1.0, 0.12)`) the default leaves **1.5e-5 relative** — larger than
the agreement a reference cross-check wants to assert. Measured residual against the
tracked derivative: `6.9e-5`, `4.4e-6`, `3.0e-7`, `4.7e-8` at 16, 64, 256 and 1024
slices. This is convergence at the trapezoid's own order, **not** a physics gap
between the analytic and tracked routes; raise `slices` when comparing against a
reference at better than `1e-4`.

## Second-order dispersion (M3 — implemented)

`second_order_dispersion()` reports the off-momentum closed orbit's Taylor expansion
to second order at every element boundary:

```
x_co(delta) = disp_x * delta + 1/2 * ddisp_x * delta^2 + ...
```

`disp_*` are `d(x, px, y, py)/ddelta` — the same quantity as `Twiss.disp_*`, but
measured on the **tracked** orbit rather than solved for from the linear maps.
`ddisp_*` are the **full** second derivatives `d^2(x, px, y, py)/ddelta^2`, matching
`xtrack`'s `ddx` / `ddpx` / `ddy` / `ddpy` — *not* half of them.

### Two codes, two conventions, and neither is guessable

- **xtrack's `ddx` is the full second derivative.** Pinned by twice-differencing
  xtrack's *own* `x` at three momenta and reading the ratio to its reported column:
  `1.0`, decisively not `0.5`.
- **MAD-X's `DDX` is the `pt^2` coefficient**, i.e. half a second derivative *and* in
  the energy variable rather than the momentum one. With
  `pt = beta0*delta + beta0*delta^2/(2*gamma0^2)` and `x = DX*pt + DDX*pt^2`:

  ```
  DDX = (d^2x/ddelta^2 - (dx/ddelta)/gamma0^2) / (2*beta0^2)
  ```

  Reading `DDX` as a plain `1/2 d^2x/ddelta^2` is wrong by
  `|1 - (dx/ddelta)/(d^2x/ddelta^2)| / (beta0*gamma0)^2` — **4.6e-4** at `gamma0 = 20`,
  small enough to pass for round-off, and **7.6e-3** at `gamma0 = 5`. The reference
  suite runs both energies for exactly that reason: the error moves with the beam
  energy, so a single-ring fit cannot masquerade as agreement. After the transform,
  MAD-X and accsim agree to **2e-7**. This is consistent with the first-order
  convention already recorded above (`DX = (1/beta0) dx/ddelta`) — both orders live in
  the same momentum variable.

### MAD-X renormalises `PX` at non-zero `DELTAP`

Sampling MAD-X's own table at three `DELTAP` values — the trick M1 and M2 used to
check its tunes without trusting a `DD` column — works for `X` and is **silently
wrong for `PX`**. MAD-X divides the transverse momentum by the shifted reference
momentum, so the second difference returns

```
d^2/ddelta^2 [ px/(1+delta) ] = d^2px/ddelta^2 - 2 dpx/ddelta
```

which on the M1 arc is `-0.3083` where the true derivative is `+0.4381` — the wrong
**sign**, not merely the wrong size. Asserted in the reference suite rather than
merely avoided.

### The drift model reaches this quantity only on a steered ring

M2 established that accsim's exact `Drift` and xtrack's/MAD-X's paraxial one give
`Q''` values 5% apart on a ring that bends. The roadmap pre-committed that a `ddx`
cross-check would therefore have to force `xt.Drift(model="exact")`. **On a ring whose
on-momentum orbit runs down the axis it does not**, and the algebra says exactly when.

The exact drift exceeds the paraxial one by `L*px*(px^2+py^2)/(2*(1+delta)^3)`. Write
`px = a + b*delta` on the closed orbit (`a` the **on-momentum** orbit angle, `b` the
dispersion angle `D_px`). In the flat case the `delta^2` coefficient of that difference
is

```
3*a*b^2
```

so it vanishes identically when **either** factor is zero — `a = 0` for any unsteered
ring, `b = 0` for any ring with no bend. Every ring in this project's analytic and
reference suites closes on the axis, which is why `ddx` looked drift-model-independent:
on M2's minimal ring the two models agree to `1e-15`, and inside xtrack, switching drift
models moves `ddx` in the ninth significant digit while moving `ddqx` by 5%.

`Q''` is split regardless, because it differentiates the **Jacobian** about the orbit,
and `d/dpx` of the same term is `O(b^2 delta^2)` — one order lower, and free of `a`.
**The orbit and the optics about it are separate objects, and a map difference can reach
them at different powers.** Never carry a finding about one to the other without
checking the order.

**Steered, the split comes back**, and the two exponents are gated: first order in `a`
(measured ratio `2.001` per doubling of the steerer) and second order in `b` (ratio →
`4` per doubling of the bending angle). A 10 mrad steerer on M2's minimal ring splits
`ddisp_x` by **6.8e-3** relative. The **first-order** dispersion is split too, by a
different power, and survives even with the bend removed — `1.9e-4` relative there — so
the reference suites' `disp_x` comparisons are safe only because their rings are
unsteered. On a machine with a real error orbit (K1 misalignments, uncorrected
steering), accsim and a paraxial reference are measuring different things, exactly as
they are for `Q''`.

The `theta^3` law in the corrector-ring gate is the same formula with both factors
carried by the kick: that ring has no bend, so `a` and `b` are both proportional to the
kick angle.

### A linear-matrix machine has no second-order dispersion at all

Every `Element.matrix()` in this package is `delta`-independent, so the *affine*
closed orbit is exactly `D*delta` and its second derivative is identically zero. What
`second_order_dispersion()` returns is, in full, the difference between the map a
particle follows and the matrix used to describe it — which makes it a cross-check of
the exact maps (L1–L4) rather than a re-reading of the linear ones.

### `tol` is tighter than the orbit solve's own default, and it is measurable

A second difference divides by `delta^2`, so `closed_orbit_nonlinear`'s default
`tol = 1e-14` lands as **~6e-9** of noise in `ddisp_x` at the default step — a third
of the truncation error, for nothing. At `tol = 1e-15` Newton's last step takes the
orbit to `~1e-19` and the noise disappears under the truncation. Truncation is
`(d^4x/ddelta^4)/12 * delta^2`; on M2's minimal ring the two cross near
`delta ~ 7e-5`, and the default `1e-3` sits in the truncation-dominated region where
the analytic suite's convergence-order gate is meaningful.

### It is defined where the chromatic functions are not

`chromatic_functions()` differentiates a Courant-Snyder `beta`, which an x-y coupled
lattice does not have, so it raises `CoupledLatticeError`. A closed orbit exists all
the same: `second_order_dispersion()` routes through `propagate_orbit_nonlinear`, not
through the on-orbit Twiss, and a skew quadrupole standing at horizontal dispersion
gives the orbit a **vertical** second-order dispersion at both orders.

### What drives it: the sextupole exponent is one, where `Q''`'s is two

A sextupole at dispersion `D` sees `D*delta` and gives back a dipole kick
`-1/2 k2l (D delta)^2` — second order in `delta`, first order in `k2l`. So it lands on
`ddisp_x` **linearly** (measured ratio `2.0000` per doubling of `k2l`). The same
feed-down reaches `Q''` only as a *gradient*, which is first order in `delta` and so
needs the perturbation twice: `Q''` goes as `k2l^2` (M1 measured `2.02`). One element,
two quantities, two different powers — the pair is the gate, because a uniformly
mis-scaled sextupole kick would be invisible to a tolerance on either.

## Spin precession (N1 — implemented)

Axis N's first milestone: a particle's **spin**, rotated by every field the lattice
contains. It is carried *alongside* the 6D state, never appended to it, because a spin
neither influences the orbit nor is part of it — which is why N1 re-baselines nothing.

### The map

A spin is a unit 3-vector `(S_x, S_y, S_z)` in the local curvilinear frame — the same
axes `(x, y, s)` the 6D state uses. Per unit of **path length**, in the element's own
frame,

```
dS/ds = Omega x S
Omega = -(1/(1 + delta)) [ (1 + G gamma) b_perp + (1 + G) b_par ]      # rad/m
```

- `b = B/(B rho)_0` is `Element.normalized_field(x, y)` — the *same* thing the radiation
  kick asks a magnet for. No element grows a second field model.
- `b_par` / `b_perp` split `b` along and across the **direction of motion**
  `i = (px, py, p_s)/(1 + delta)`.
- `G = (g - 2)/2` is `ReferenceParticle.anomalous_moment`.
- `gamma` is the **particle's** own `E(delta)/m`, not `gamma0`. Using `gamma0` would be a
  silent 1% error at `delta = 1e-2` that no on-momentum gate can see.

**The charge cancels.** The textbook `Omega` divides by the rigidity `B rho = p/q` and so
carries `q`; the package's `b` already *is* `B q / p_0` (that is what makes `k1` mean what
it means), and the two factors cancel. An electron ring and a proton ring are the same
expression, and a stray `ref.charge` would be visible on the very first cross-species
check.

**The frame rotation is half the physics of a bend.** `Omega` is written in the
curvilinear frame, and inside a bend that frame *turns*. `Element.frame_rotation_angle`
is `0` for every straight element and `Dipole.angle` for a bend, applied in **two halves**
either side of the BMT rotation (the midpoint rule, and what xtrack does). On the design
orbit the field is constant, so the composition is exact:

```
BMT      -(1 + G gamma) theta   about y
frame    +theta                 about y
net      -G gamma theta         about y      <- exactly, no quadrature error
```

Summed over a flat ring's `2 pi` of bending, that is the spin tune `nu_0 = G gamma`.

**The sign of the frame rotation is not a convention to be chosen.** With `G = 0` a spin
must come out of a sector bend still pointing along the design orbit, and only `+theta`
does that. Every sign in this module is pinned that way rather than argued.

### The spin tune is a control, not a gate

`nu_0 = G gamma` depends only on the ring's bends summing to `2 pi` and on the beam
energy. On the design orbit a quadrupole has **no field at all**, so an implementation
whose quadrupole precession were mis-scaled by a factor of seven — or missing entirely —
reproduces it *exactly*, at every energy. That is asserted rather than hoped against
(`test_the_spin_tune_is_blind_to_a_mis_scaled_quadrupole_precession` scales the field by 7
and checks the tune is unchanged **bit for bit**). It is J1's shape again: the famous
number is the one that cannot see the coefficient it appears to test.

What carries the milestone instead:

- **The Dirac identity.** With `G = 0` the BMT rotation *is* the cyclotron rotation, so a
  spin started along `p` stays along `p` — at any amplitude, on or off momentum, in
  either frame. It needs no reference code. It is *exact* where the field is constant
  along the path (a drift, a sector bend on the design orbit) and converges at **second
  order in the slice length** where it is not. Its teeth are asserted three ways: scaling
  `Omega` by 1.01, flipping its sign, and dropping the bend frame rotation each leave a
  residual that **does not converge at all**.
- **A quadrupole at a vertical offset**, which pins the `(1 + G gamma)` factor itself
  against a sympy-derived `int y ds = y0 sinh(sqrt(k) L)/sqrt(k)`. The Dirac identity
  cannot: the factor is `1` there by construction. At 5 GeV the three candidate readings
  are an order of magnitude apart — `(1 + G gamma) = 12.35`, `G gamma = 11.35`,
  `(1 + G) = 1.0012` — so no tolerance is needed to tell them apart.
- **The `(1 + G)` parallel term**, read off exactly as `Omega . i = -(1 + G)(b . i)/(1 +
  delta)`, since `b_perp . i` is zero by construction. It is *not* dead code awaiting a
  solenoid: `b_par` is the component of a purely **transverse** field along the direction
  of motion, non-zero as soon as the particle has an angle. It is identically zero on the
  design orbit, though — the same degeneracy M3 found for second-order dispersion — so
  the gate is its **first order in `px`**, not its value.

### Scope, and one real approximation

- **Thin elements do not precess**, and unlike radiation this is an approximation rather
  than a limit. A thin magnet's radiated energy genuinely vanishes with its length
  (`U ~ kappa^2 L`); its *integrated* field does not, so a thin quadrupole's true spin
  rotation is finite and dropping it is a real omission. It is dropped anyway because
  **xtrack's thin `Multipole` does not rotate spin either** — spin lives only in its
  `track_magnet` family — so building it would mean inventing a model with no arbiter,
  which is L5's reason and the one trade this project's validation strategy does not
  make. The cost is precise: **a thin-lens ring has no spin dynamics at all**, so every
  gate on this axis is built from thick magnets.
- **The field is integrated by the midpoint rule**, exactly as `radiation_kick`
  integrates it, over the same `l_path = rvv (L - Delta zeta)`. Exact for a sector bend
  on the design orbit; second-order accurate otherwise.
- **A rolled *bend* raises.** A roll of a straight element is a plain conjugation and the
  spin rides through it (`rotate_about_s` in, its inverse out) — a rolled quadrupole
  precesses a spin exactly like the skew quadrupole it is. For a *bending* magnet K2's
  rigid-body geometry moves the exit face, so the frame turn is no longer a rotation
  about `y` in the lattice frame; `spin_precession` refuses rather than applying a
  plausible wrong one.
- **`anomalous_moment` has no numeric default.** `ReferenceParticle.anomalous_moment` is
  `None` until set, and `accsim.spin.anomalous_moment` raises on it. An explicit `0.0` is
  accepted — it is the Dirac limit the sharpest gate runs in. This is deliberate: see the
  next section.

### Two traps on the reference side

**`xt.Particles` defaults `anomalous_magnetic_moment` to `0`.** That is not "spin physics
off": it is the cyclotron rotation, a spin tune of exactly zero, and a plausible-looking
tracked spin answering a different question. It is M2's trap by another name — a
reference whose *default configuration* is not the physics being checked — and it is why
accsim refuses rather than defaulting.

**xtrack 0.106.4's `direction_of_motion` has a sign typo.** In
`beam_elements/elements_src/track_magnet_radiation.h:22`:

```c
double iis = sqrt(1 - iix * iix + iiy * iiy);     // a '+' where a '-' belongs
```

The vector it returns is therefore **not a unit vector**, long by `~py^2/(1+delta)^2`, and
it is used unnormalised for both the spin precession *and* `compute_b_perp_mod`, which is
what B2's radiation kick integrates. accsim writes it correctly (`accsim.spin.direction_of_motion`).

### A bug this axis found in accsim itself

`SkewQuadrupole.normalized_field` rolled the **opposite way from its own map**: the map is
`s_rotation(+45) . Q . s_rotation(-45)`, the field was built with the rotations the other
way round. The result was the exactly sign-flipped field *vector* with the correct
magnitude.

Nothing could see it. `normalized_field` had exactly one consumer — `radiation_kick` —
and that consumer takes `|b_perp|`, which is roll-invariant. Spin is the first quantity in
the package that reads a field's **direction**, and it found the mismatch on the first
comparison: a rolled `Quadrupole` and a `SkewQuadrupole` agreed on the orbit to the last
bit and disagreed on which way a spin turned.

**Which of the two was wrong is settled externally, not by reading the source.** The
map's roll sense is pinned by both references and was already: MAD-X reproduces
`SkewQuadrupole`'s whole transverse 4x4 to `1e-13` including the off-diagonal coupling
blocks (`test_betatron_coupling_madx.py`), and xtrack pins the sign of `R[px, y]`
(`test_betatron_coupling_xtrack.py`) — a flipped roll would flip exactly those. So the
map was right, the field had to be brought to it, and the direction of the fix is not a
judgement call. Worth stating because the full analytic suite **cannot** distinguish the
fix from the bug: radiation takes `|b_perp|`, which is roll-invariant, so the pre-fix and
post-fix code radiate bit-identically and 1144 passing tests are no evidence either way.

The fix is one line; the gate is the point. `test_a_straight_magnets_field_agrees_with_its_own_momentum_kick`
now asserts, for every straight magnet, that `(dpx, dpy)/L -> (-b_y, +b_x)` as `L -> 0`.
It is written against `_track_body`, not `track`, because that is where the invariant
lives: **`normalized_field` is the element's field in its own frame**, and both consumers
evaluate it on *body-frame* coordinates. Bends are excluded — there the curvilinear
frame's own turn cancels the design field, so a sector bend's kick is zero while its field
is `h`.

### Comparing against xtrack: three switches, all silent

`tests/reference/test_spin_xtrack.py`. Each of these returns a plausible wrong answer if
it is not set, and none of them errors:

1. **`line.configure_spin("auto")`.** Without it the kernel is compiled with spin *off*
   and `track()` returns the spin **bit-for-bit unchanged** — through a magnet that
   certainly precesses it. A comparison written without this measures nothing, and reads
   as "accsim invented a precession xtrack does not have".
2. **`anomalous_magnetic_moment` on the particle.** `xt.Particles` defaults it to `0`.
3. **`model="bend-kick-bend"`** on the bend, with `integrator="uniform"` and
   `num_multipole_kicks=1` (B2's argument, unchanged).

### What the comparison then says, and it is sharper than N1 predicted

N1 was written expecting an `O(L^2)` gap *everywhere*, because the two codes build the
field by genuinely different recipes: accsim samples its **analytic** `normalized_field`
at the traversal mid-point, xtrack's `magnet_estimate_field` **back-derives** `B` from the
trajectory's curvature. What is actually there:

- On a **bend**, and on a quadrupole with only **one** transverse plane populated, the
  two agree to **round-off at every slicing**. The reason is derivable rather than lucky:
  in a single plane `b . i = 0` (a purely horizontal field never meets a purely vertical
  angle), so `Omega` points along one fixed Cartesian axis for the whole traversal, every
  rotation commutes with every other, only the **scalar** `int b ds` survives — and both
  codes' quadratures of that scalar are the same number.
- With **both** planes populated the axis *turns*, the rotations stop commuting, and the
  two lumped maps converge to each other as `1/N^3` — a factor 8 per doubling, gated as
  that order.

So the gap is **non-commutativity, not the field model**, and the single-plane exactness
is what proves it.

### xtrack's default bend integration moves the *orbit*, and its spin honestly follows

An apparent `1.4e-5` spin disagreement on a bend looks alarming and is not a spin
disagreement at all. With `G = 0` a sector bend must leave a design-orbit spin exactly
alone; `bend-kick-bend` does. The **default** integration does not — but the residual is
the orbit's, not the spin's, and four measurements say so:

| what was varied | what the spin residual did |
| --- | --- |
| bend angle `theta` | `x32` per doubling → `O(theta^5)` |
| `num_multipole_kicks` `N` | `/16` per doubling → `O(N^-4)`, a fourth-order splitting |
| element length | **nothing** |
| beam energy (1 → 20 GeV) | **nothing** |

and the spin residual equals `theta` times the **orbit** residual to three digits. The
default splitting leaves the design particle itself off axis by `O(theta^5/N^4)`, and the
spin correctly follows that slightly wrong momentum. M2's lesson, again: localise before
deriving — no tolerance on the spin alone could have separated "xtrack's spin is wrong"
from "xtrack's orbit is approximate and its spin is right".

### xtrack 0.106.4's `direction_of_motion` is not a unit vector

`beam_elements/elements_src/track_magnet_radiation.h:22`:

```c
double iis = sqrt(1 - iix * iix + iiy * iiy);     // a '+' where a '-' belongs
```

The vector is long by `~iy^2` and is used **unnormalised** for both the spin precession
and `compute_b_perp_mod`, which is what B2's radiation kick integrates. accsim writes it
correctly (`accsim.spin.direction_of_motion`).

The order is derivable and is what the gate asserts: the error multiplies `b . i`, which
for a bend is `b_y i_y` and already carries one power of `py`. One power from the
projection, two from the botched normalisation ⇒ the spin disagreement is **third order in
`py`** (measured ratio 8.00 per doubling) and **exactly zero in `px`** (`3e-16` at
`px = 4e-3`). The orbit is untouched at `1e-16` throughout, so it is a pure spin
statement. Asserted with both exponents rather than dodged by tracking at `py = 0`: a
disagreement with a derived exponent is a finding; one hidden by a choice of test point is
a gap.

## Closed spin solution and spin tune (N2 — implemented)

`accsim.spin.closed_spin_solution` returns `n_0`, the spin direction that comes back to
itself after one turn, and `nu_0`, the rate a spin *not* along it winds around it. They
are the spin analogues of I1's closed orbit and the betatron tune, and are reached the
same way — as the fixed point of the one-turn map — with the simplification that the
one-turn *spin* map is a rotation, hence **linear in the spin**: carrying the three
Cartesian basis vectors around once gives the whole 3×3 exactly, with no Newton iteration
and no differencing step (`spin_one_turn_matrix`).

### Two sign conventions, and why neither was free

- **`n_0` is oriented upward: `n_0 · ŷ > 0`.** This is xtrack's convention, forced rather
  than adopted for taste — its fixed-point search sets `s_y = +sqrt(1 - s_x² - s_z²)`, so
  it can only ever return an upward solution, and matching it is what makes the two
  comparable at all. Where `n_0` is exactly horizontal the rule falls back to `n_0 · x̂ > 0`
  and then `n_0 · ẑ > 0`.
- **`nu_0` is the fraction in `[0, 1)` defined by `R = R(n_0, −2π nu_0)`** — the spin turns
  by `2π nu_0` about `n_0` in the *negative* sense. The minus sign is not decoration. A
  flat ring's net spin rotation is `−(1 + Gγ) θ` from Thomas-BMT plus `+θ` from the frame,
  i.e. `−Gγ θ` about `+ŷ`; writing it as `+2π nu_0` would give `nu_0 = −Gγ`, and every
  textbook — and the whole of N1 — quotes `nu_0 = +Gγ`.
- Only the **fraction** is knowable. A rotation matrix has no memory of how many whole
  turns produced it, exactly as a one-turn transfer matrix has none of the integer betatron
  tune.
- **xtrack's `spin_tune_fractional` is folded to `[0, 0.5]`**: it takes
  `max(angle(eigvals(A)))/2π` and `np.angle` returns `(−π, π]`, so the sign and the
  half-turn are thrown away. The comparison folds accsim's (`min(ν, 1−ν)`) rather than
  unfolding xtrack's — the information is not there to unfold.

### The degeneracy is the whole milestone

On a flat, unsteered ring every field a spin meets is vertical, every rotation is about
`ŷ`, and **`n_0 = (0, 1, 0)` bit for bit** — for any lattice, any energy, any quadrupole
strength, a quadrupole field multiplied by seven, and even a *sign-flipped* precession
vector. Nothing about the transverse coefficient can be read off such a ring. This is N1's
"the spin tune is a control" arriving one milestone later in a second quantity, and M3's
degeneracy in a third guise. `n_0` becomes informative only once the closed orbit has a
**vertical** excursion through a field.

### The gate ring, and the closed form it was built for

A closed vertical bump (three correctors, solved from the elements' own vertical transfer
matrices) holding exactly **one thick quadrupole**, inside a **bend-free** straight, with a
thin-lens FODO arc whose bends sum to `2π`. Two facts make that the only honest
construction:

- **A bend traversed with a vertical angle precesses the spin about `ẑ`**, at
  `Ω_z = h i_y i_z G(γ−1)` — the difference between the `(1 + Gγ)` perpendicular
  coefficient and the `(1 + G)` parallel one, which is what survives when a vertical angle
  gives a horizontal-field magnet a component along the motion. It is **first order in
  `py`** and scales with `Gγ`, so it is comparable to the quadrupole kick the gate is
  about. Vertical orbit leaking into the arc is therefore a second, distributed,
  uncontrolled driving term — and the closed form would still *fit*, with a wrong
  coefficient.
- **Thin elements do not precess** (N1's stated omission), so the arc's focusing can be
  thin and contributes nothing at all. The one thick quadrupole is then the entire spin
  perturbation.

The ring reduces to one localized rotation `chi` about `x̂` composed with a uniform
`−2π nu_0` about `ŷ`, and (derived with sympy, observed at the entrance with the kick at
the top of the turn)

    n_0 = ( −(chi/2) cot(π nu_0),  1,  −chi/2 )      to first order in chi
    nu_0 → nu_0 + chi² cot(π nu_0) / (8π)            i.e. unmoved at first order

The two transverse components are **two different gates**:

- the `z` component is `−chi/2` with **no resonance denominator at all**, so it measures
  the kick itself, and through it the `(1 + Gγ)` factor — inside a ring, where N1 pinned it
  in a single element. Its residual is the midpoint rule's quadrature of `∫ y ds` and
  converges at second order in the slice length (ratio 4.00 per halving, measured over
  n = 2…64).
- the `x` component carries `cot(π nu_0)`, which diverges at every **integer** spin tune
  and nowhere else. Since `nu_0 = Gγ`, the resonance is crossed by **scanning the beam
  energy**, which is where a polarized ring's energy calibration comes from.
- their **ratio** `n_0·x̂ / n_0·ẑ = cot(π nu_0)` drops `chi` entirely, so the *direction* the
  solution leans in measures the spin tune with nothing about the imperfection left in it.

`nu_0` is unmoved at first order in the steering — this axis's version of "the number
everybody quotes is the one that cannot see the perturbation." Use the tilt, not the tune.

### The resonance is at an **integer**, not at `nu_0 = k ± Q_y`

The roadmap's N2 entry named `nu_0 = k ± Q_y` as the discriminating condition. That is
wrong for this object, and the correction is recorded rather than quietly applied. `n_0`
lives on the **closed orbit**, so the perturbation it sees is one-turn periodic and has
only **integer** harmonics; the resonant denominator is `1 − e^{−2πi nu_0}`, which vanishes
only at integer `nu_0`. That is the *imperfection* resonance.

`k ± Q_y` is the **intrinsic** resonance, and it is a statement about a different object:
the invariant spin field of a particle with vertical betatron *amplitude*, whose
denominator is `λ_i I − A` with `λ_i = e^{2πi Q_y}` an orbital eigenvalue (this is literally
xtrack's `EE_spin[:, i] = inv(λ_i I − A) @ DD @ EE_orb[:, i]`). It needs the spin–orbit
coupling matrix `∂n/∂(x, px, y, py, ζ, δ)`, which is exactly what xtrack's `spin_n_matrix`,
`spin_eigenvectors` and `spin_dn_ddelta_*` carry and what N3's depolarization term is built
from — and none of those appear in N2's own arbiter list. Deferred to N3, deliberately.

### `n_0` needs the **tracked** closed orbit, not the linear one

`closed_spin_solution` defaults `orbit0` to `closed_orbit_nonlinear`, and the expense is
the point: the spin is rotated by what `track()` does, so a spin carried around the
*linear* closed orbit is carried around a trajectory that does not quite close, and its
"one-turn" rotation is really a rotation between two different points. With the exact maps
of axis L the gap is `O(x_co³)` — measured as a factor 8 per doubling of the bump — and on
the gate ring it is a one-turn orbit residual of `1e-8` where the tracked orbit gives
`1e-18`.

The same third-order departure means the bump, whose correctors are solved from the
elements' **matrices**, does not close *exactly* under the exact maps. The leak is
`1.6e-9` at a 2 mm bump, cubic in the amplitude; the arc's whole spin driving that follows
from it is bounded against `chi` at `5e-5` at the largest amplitude any gate uses, below
every tolerance any of them assert. Bounded, not assumed.

### `SpinSolutionError`: the spin twin of an integer betatron tune

At integer `nu_0` the one-turn rotation is the identity, **every** direction is periodic,
and none is *the* periodic one. `spin_axis_and_tune` raises rather than returning one of
them — the same contract, and the same reason, as `closed_orbit` raising on an integer
tune where `I − M4` is singular. The test is the second-smallest singular value of `R − I`,
which is `2|sin(π nu_0)|`, so it is literally "is the spin tune an integer?".

### Comparing against xtrack: a fourth silent switch, and it is the drift model

N1 recorded three switches that silently return a plausible wrong answer
(`configure_spin`, `anomalous_magnetic_moment`, the bend `model`). N2 adds a fourth, and it
is M2's finding arriving on a new axis: **xtrack's default `xt.Drift` is the paraxial one**.
A ring of paraxial drifts closes this bump *exactly* — the bump is solved from matrices —
while a ring of exact drifts does not. With the default, the two codes' closed orbits
therefore disagree by exactly accsim's own leak, and every spin comparison inherits it.
`xt.Drift(model="exact")` removes it. The disagreement is gated in its own right
(`test_the_paraxial_drift_closes_the_bump_that_the_exact_one_does_not`): the gap equals the
leak the analytic suite measures from a completely separate line, to two digits.

With that set, and the orbit compared element by element **before** any spin component is
looked at (N1's finding 2 in a ring instead of a magnet), `n_0(s)` agrees with
`twiss(spin=True)`'s `spin_x/y/z` element by element and `nu_0` with
`polarization_analysis`'s `spin_tune_fractional`. The tolerance is `1e-8`/`1e-9` rather
than round-off, and **the residual is xtrack's, not accsim's**: accsim's one-turn matrix is
exact, while xtrack finite-differences it at `ds = 1e-5` and finds `n_0` with a two-knob
optimiser to `1e-12`.

### One more blind structural check, kept and labelled

The one-turn matrix is orthogonal to `1e-14`. That is *not* a check on the physics — a
product of rotations is orthogonal whatever fields it was built from — and the proof is
that it survives multiplying every quadrupole field by seven. It catches a broken
Rodrigues formula and nothing else.

## Sokolov-Ternov polarization (N3 — implemented)

The spin-flip channel of the synchrotron radiation axis B already models. A bending
electron radiates, a tiny fraction of that radiation flips its spin, the two flip
directions are not equally likely, and a stored beam therefore polarizes on its own.
`accsim.radiation` gets `polarization_integrals`, `sokolov_ternov_polarization`,
`polarization_buildup_time` and `PolarizationIntegrals`. Its natural home is the
radiation module and not the spin one: it is one more integral over the same curvature.

### The two integrals, and where each sign lives

Following Chao (the same pair `xtrack` reports as `spin_alpha_plus_co` /
`spin_alpha_minus_co`), as arc-length averages over the ring:

    alpha_plus  = (1/C) ∮ kappa^3 (1 - (2/9) (n_0 · v)^2) ds     — sets the rate
    alpha_minus = (1/C) ∮ kappa^3 (n_0 · b) ds                   — sets the direction

    P_inf  = (8 / (5 sqrt3)) alpha_minus / alpha_plus
    1/tau  = (5 sqrt3 / 8) r_0 (hbar / m_0) gamma^5 alpha_plus

`kappa` is the local orbit curvature, `b` the unit vector along the **physical** magnetic
field, `v` the unit vector along the motion, `n_0` N2's periodic spin direction.

**`alpha_minus` is signed and `alpha_plus` is not.** The size of the bend lives in
`kappa^3`, the sense of the bend lives in `b`. A reverse bend therefore subtracts from
`alpha_minus` and adds to `alpha_plus`, which is the physically right behaviour and would
not survive putting the sign in `kappa^3` instead.

### `normalized_field` is not the field direction — the charge's sign is missing

`Element.normalized_field` returns `B / (B rho)_0`, and `(B rho)_0 = p/q` **carries the
charge's sign**. Recovering the physical field direction means multiplying that sign back
out (`accsim.radiation._polarization_integrand`). This is the one place in N3 where a
mistake produces every magnitude in the milestone correctly and the direction backwards,
and no magnitude check anywhere can see it.

Consequence, and it is the textbook one: on an ordinary electron ring `n_0` is `+y` while
the guide field points `-y`, so **`P_inf` is negative** — the beam polarizes *antiparallel*
to the field. That is physics, not a convention this package was free to choose. It is
anchored on two independent knobs, each of which flips it alone: swap the charge to a
positron, or reverse every bend.

### `P_inf = 8/(5 sqrt3)` is a control, not a gate

It is a *ratio* of the two integrals. On any flat, unsteered ring `n_0` is parallel to the
field everywhere the ring bends, so the ratio is `-1` before either integral is evaluated,
and any uniform mis-scale of the pair — a wrong power of `kappa`, a wrong circumference, a
stray factor in the accumulation — cancels out of it exactly. `tests/analytic/`
`test_polarization.py` asserts this rather than saying it: the same sixteen digits,
`-0.9237604307034013`, come back across six rings differing in focusing, cell count,
size, energy and slice count. Same family as J1's blind structural gates and B5's three
quiet arbiters.

### What does gate it: the two weights pulling apart on a tilted ring

N2's vertical-bump ring tilts `n_0` away from the field by `t`, and the two integrals then
stop being each other's negative:

    |alpha_minus| C / I3 = n_y = 1 - t^2/2 + O(t^4)          — the (n_0 · b) weight
     alpha_plus   C / I3 = 1 - (2/9) t^2 <cos^2>             — the (n_0 · v)^2 weight

so their **sum** is `t^2 (1/2 - (2/9) <cos^2>)`: one number carrying both weights, with
different coefficients and *opposite signs*. That is the milestone's gate, and it is the
form to assert — each integral checked separately against a `kappa^3` integral would pass
on a normalization coincidence with the weights wrong. Both one-legged alternatives are
asserted to be excluded, not merely different. `<cos^2>` is integrated in sympy; the
remembered "average of `cos^2` is `1/2`" is 0.6% wrong here, because the correction falls
only as `1/(G gamma)`.

### `n_0`'s horizontal part counter-rotates against the bend

Inside a bend the horizontal projection of `n_0` turns through **`-G gamma`** per unit bend
angle relative to the direction of motion — the opposite way from the trajectory. Taking
that sign the other way leaves the arc average of `cos^2` **1.5%** out, which reads exactly
like a quadrature error and is not one; it was found by watching the residual refuse to
converge under refinement while a genuine quadrature error would have fallen as
`slices^-2`. **Localise before deriving**, M2's lesson, on a third axis.

### The quadrature must resolve the *spin* phase, not the optics

Across one bend of angle `theta` the spin phase moves by `G gamma theta` — 4.4 radians on
N2's 5 GeV gate ring — where the dispersion `radiation_integrals` sub-steps moves by
`theta = 0.39`. The two integrals therefore use **different rules on purpose**:
`radiation_integrals` trapezoids, `polarization_integrals` uses **Simpson**, which
converges as `slices^-4` and reaches the round-off floor of the `(n_0 · v)^2` term at the
shared default of 64 slices where the trapezoid is still 1.5% short of it. At higher
energies `G gamma theta` grows and 64 stops being obviously enough.

Measure convergence on the `(n_0 · v)^2` **term**, never on `alpha_plus`: the term is one
part in `10^8` of it, so a convergence test on `alpha_plus` reports machine precision at
every slice count and sees nothing.

### Scope: only dipoles radiate, matching the radiation integrals

`polarization_integrals` counts only `Dipole`, exactly as `radiation_integrals` does, and
deliberately the same restriction: `alpha_plus * C == I3` is a gate the two routes have to
agree on, so they must agree about what radiates. A quadrupole traversed off-axis really
does curve the orbit and really does radiate, and `xtrack` counts it (it reads `kappa` from
the closed orbit element by element). On the gate ring that omission is `3e-12` of
`alpha_plus`, negligible there against both `alpha_plus` and the `1e-8` tilt term — but it
grows as the **cube** of the orbit offset where the tilt term grows as its square, so the
margin closes on a badly steered ring. Lifting the restriction means lifting it in both
places, which moves axis B's numbers, so it is a separate change.

### The coefficient is the discriminating quantity, and almost nothing sees it

`P_inf` provably cannot: the constant cancels out of a ratio. `gamma^5` and `rho^3` scaling
catch a wrong *power* and are exact for a rate ten times too fast. What the analytic suite
can do is bound the eV-to-SI bridge — `hbar / m_0 = (hbar c) c / (m c^2)` in `m^2/s`,
assembled from the package's own `HBAR_C_EV_M` and rest energy, checked against
`scipy.constants`, which never passes through eV — and anchor the machine-scale answer on
LEP, where a bare ring with LEP's radius and circumference at 45.6 GeV gives **5.65 hours**
against a published ~5.5. A wrong *factor* surviving all of that is caught only by
`xtrack`, behind the skippable `reference` marker. **A green analytic suite is weaker
evidence on this milestone than anywhere else on the axis**, and the test module says so.

### The "no bending" refusal is nearly unreachable, and that is the finding

`sokolov_ternov_polarization` and `polarization_buildup_time` refuse when `alpha_plus` is
exactly zero, rather than reporting `0/0` as `8/(5 sqrt3)`. The obvious lattice for that —
drifts and on-axis quadrupoles — never reaches the refusal: with no field anywhere on the
orbit nothing precesses, the one-turn spin rotation is the identity, and N2's
`SpinSolutionError` fires first, because both integrals are weighted by an `n_0` that does
not exist. Exactly one construction separates the two conditions: a **quadrupole traversed
off-axis**, whose field on the orbit is real (so `n_0` is unique) and which is not a dipole
(so nothing in scope radiates). Both branches are asserted.

### Comparing against xtrack: a fifth silent switch, and it is the **charge**

N1 catalogued three silent switches on the reference side, N2 found a fourth (the drift
model). This is the fifth, and the quietest yet: **`xt.Particles` defaults `q0 = +1`**, so
a line built with `mass0=ELECTRON_MASS_EV` and no `q0` is a positively charged particle of
electron mass.

Everything axis N compared before N3 is **blind to it**. A lattice specified by normalized
strengths (`k0`, `k1`) bends the same way whatever the charge, and the Thomas-BMT rotation
reads the field through the same normalization — so the closed orbit, `n_0`, the spin tune
and the one-turn rotation are all bit-for-bit unchanged by `q0`. That is exactly why N1's
and N2's reference files agreed without ever setting it, and it is asserted in N3's file.

The polarization **direction** is the first quantity on this axis that asks what the
*physical* field is, and charge is what turns a curvature into a field. Run with the
default and xtrack cheerfully reports an electron beam polarizing *along* its guide field —
and because both codes would flip together if accsim made the same mistake, the error never
surfaces as a disagreement. With `q0 = -1.0` set, `alpha_plus` is unchanged and
`alpha_minus` and `P_inf` are exactly negated.

### xtrack cannot run its polarization analysis on an exactly flat ring in `4d`

`_get_spin_polarization` inverts `lambda_i I - A` per orbital eigenvector. With
`method="4d"` one orbital eigenvalue is exactly `1`, and a flat ring's spin matrix `A` is a
rotation about `y` — so `I - A` has a zero row and `np.linalg.inv` raises
`LinAlgError: Singular matrix`. xtrack's own `A` is built by central-differencing tracked
spins at `±ds`, and a `y` component that comes back untouched gives `(ds-(-ds))/(2ds) = 1`
*exactly*, so its `I - A` is exactly singular rather than merely ill-conditioned. A tilted
ring survives only because `A`'s middle row is no longer exactly `(0,1,0)`.

The quantities N3 compares are not computed through that inverse — it feeds the
`dn/ddelta` depolarization term deferred to N4 — but it aborts the whole `twiss`, so **the
flat ring is simply unavailable as a reference comparison**. Every N3 cross-check therefore
runs on the tilted ring, including the ones the tilt is irrelevant to. accsim's own exact
matrix has off-diagonal `y` terms that are exactly zero and a diagonal `9e-16` from one:
the difference between "cannot be inverted" and "raises".

### What the comparison then says

`alpha_plus` and `alpha_minus` agree to ~`4e-15` in magnitude — better than the `1e-9`
N2's finite-differenced `n_0` would suggest, because both integrals are dominated by the
`kappa^3` geometry the two codes share exactly. The **buildup time** agrees with
`spin_t_pol_component_s`, which is the milestone's only real check on the coefficient.
Compare against `spin_t_pol_component_s` and `spin_polarization_inf_no_depol`, **never**
`spin_t_pol_buildup_s` / `spin_polarization_eq`: those carry the `(11/18) ∮ kappa^3
|dn/ddelta|^2` depolarization term accsim defers to N4, and would show up as a plausible
few-percent miss rather than as a disagreement. **N4 reverses this instruction** for its
own comparisons — see *Invariant spin field & depolarization* below.

## Invariant spin field & depolarization (N4 — implemented)

What N3 deliberately left out: the term that *fights* the Sokolov-Ternov buildup. A
particle off the closed orbit has its own periodic spin direction — the **invariant spin
field** `n(x) = n_0 + N x` — and a photon emission jumps its `delta`, and with it its `n`.
Lives in `accsim.spin` (the field) and `accsim.radiation` (the integrals), matching the
N2/N3 split.

### The equation: a Sylvester equation, not a mode-by-mode inverse

A particle displaced by `x` carries the spin `n_0 + N x`. One turn later its deviation is
`R x` and its spin `A (n_0 + N x) + D x`. For `n` to be a *field* those must agree, so

    A N - N R = -D                       (`accsim.spin.spin_orbit_coupling`)

with `A` = N2's exact one-turn spin rotation, `R` = the 6×6 one-turn Jacobian about the
closed orbit, `D = d(spin out)/d(orbit in)` with the spin started along `n_0`. `R` and `D`
are central differences of **one shared tracked turn** (step `1e-6`, flat between `1e-7`
and `1e-5`); `A` is exact. Solved with `scipy.linalg.solve_sylvester`.

**Reduce to the plane perpendicular to `n_0`.** `n` is a unit vector, so `n_0 · N = 0`
exactly — the parallel component is meaningless, not small. The corresponding row of the
full equation is the consistency condition `n_0 · D = 0`, which holds to the differencing
accuracy (`1e-10`).

That reduction is **why accsim can do a flat ring and xtrack cannot**. Solved mode by
mode, the eigenvalue-`1` orbital mode (`delta`, in a lattice with no RF) needs
`inv(I - A)` — and `I - A` is singular for *every* ring, because `A n_0 = n_0`. N3 recorded
this as a fact about flat rings; it is a fact about all of them, and a tilted ring survives
in xtrack only because its finite-differenced `A` misses the zero by round-off.

### Sign / index conventions

- `N` is `(3, 6)`, columns ordered `(x, px, y, py, zeta, delta)` — `xtrack`'s
  `spin_n_matrix` on the same convention, comparable entry for entry.
- `dn_ddelta = N[:, DELTA]` is the **partial** derivative at fixed transverse coordinates,
  *not* the derivative along the dispersion orbit. That is the physics, not a choice: a
  photon emission is instantaneous, so it moves `delta` and nothing else. The two differ by
  `N[:, :4] D` and on N4's gate ring they differ by more than a factor of two.
- `N[:, ZETA]` is **exactly** `0.0` for any lattice with no RF, because nothing reads
  `zeta`. That is what makes accsim's six-column equation and xtrack's five-column
  (`zeta`-deleted) formulation the same object. **It stops being true when an RF cavity
  enters.**

### The resonances are the spectra, and `k ± Q_y` finally lands here

In the reduced plane `A`'s eigenvalues are `exp(∓2πi ν_0)`; `R`'s are `exp(±2πi Q_x)`,
`exp(±2πi Q_y)`, `exp(±2πi Q_s)`, plus `1` twice with no RF. A Sylvester equation is
solvable exactly when the two spectra are disjoint, so `N` diverges at

    ν_0 = k                 — integer: N2's *imperfection* resonance (via the eigenvalue 1),
    ν_0 = k ± Q_x, k ± Q_y, k ± Q_s   — the **intrinsic** resonances,

and at nothing else. `ν_0 = k ± Q_y` is what N2 was written expecting and did not find:
`n_0` rides the closed orbit and sees only one-turn-periodic drive, so it can only resonate
at integers. `SpinResonanceError` (a subclass of N2's `SpinSolutionError`) is raised within
`1e-8` in tune of one. `n_0` is still perfectly well defined there — it is the field
*around* `n_0` that does not close.

**Gating a resonance is gating a location.** `1/|N E_y|` is linear in `ν_0` near the pole
and extrapolates to `Q_y` within `2e-6`, a quarter of a unit from the nearest integer; and
the residue `|N E_y| · 2|sin(π(ν_0 − Q_y))|` is constant to 1.5% while `|N E_y|` itself
varies thirtyfold. Both alternative denominators (`sin(π ν_0)`, `sin(π(ν_0 + Q_y))`) vary
by a factor of 20+ and are asserted excluded. The energy is the only knob — `ν_0 = G γ`,
and normalized strengths keep `Q_y` frozen to `1e-12` across the scan.

Identify the orbital modes by **eigenvector content**, never by position in
`numpy.linalg.eig`'s output.

### The Derbenev-Kondratenko integrals

    alpha_plus  = alpha_plus_co  + (11/18)(1/C) ∮ kappa^3 |dn/ddelta|^2 ds
    alpha_minus = alpha_minus_co -        (1/C) ∮ kappa^3 (dn/ddelta · b) ds

`derbenev_kondratenko_polarization` = `8/(5√3) alpha_minus/alpha_plus` (xtrack's
`spin_polarization_eq`); `polarization_time` is `polarization_buildup_time`'s expression
with the corrected `alpha_plus` (xtrack's `spin_t_pol_buildup_s`). The first correction is
an average of a **square**, so it can only ever lower the polarization and shorten the
time — a fast-polarizing ring is not a well-polarized one.

Computed in **one walk** sharing N3's quadrature (`radiation._quadrature_nodes`), carrying
the `(6, 13)` differencing bundle launched *on* the field instead of a single closed-orbit
particle: `N(s)` then falls out of its central differences at every node. `alpha_plus_co` /
`alpha_minus_co` come back **bit-for-bit** equal to `polarization_integrals`'.

**Eight sub-slices suffice, where N3 needed 64.** `|dn/ddelta|^2` is the squared *modulus*
of a vector rotating about `n_0`, and a modulus is blind to the rotation — converged to
twelve digits at 8 slices. The oscillating `dn/ddelta · b` term is quadrature-limited but a
hundred times smaller.

### The flat ring is degenerate here too — exactly

No vertical orbit ⇒ no horizontal field anywhere on it ⇒ every rotation is about `y` ⇒ a
`delta` perturbation only changes how fast a spin turns about the axis it already lies
along. `dn/ddelta = 0` identically, both new integrals are `0.0`, and `P_eq == P_inf` **to
the last bit**. The axis's degeneracy for the fourth time (after `n_0`, `P_inf`, and the
arbiter itself).

### The collapse, and the one scaling law

`|dn/ddelta|^2 ~ 1/(ν_0 − Q_y)^2`, so `P_eq` falls from `-0.92` to `-0.02` as the spin tune
closes to `1e-5` of `Q_y`, while N3's `P_inf` drifts only in its ninth digit (its own
`1/(G γ)` energy dependence). Fit the power **close in**: at `d = 1e-3` a non-resonant
background is still worth 32% and the fitted exponent comes out `-1.89`; by `1e-4` it is
worth 3% and the residue `d^2 × integral` is flat to 1%.

### Reference comparison — and the field names are the reverse of N3's

Compare against `spin_polarization_eq` / `spin_t_pol_buildup_s` / `spin_n_matrix` /
`spin_dn_ddelta_*`, **never** the `_co` / `_component` / `no_depol` ones N3 uses. On the
resonant ring the two differ by a factor of **46**, so the trap is unmissable here where it
was a quiet few percent in N3 — but it is asserted rather than left to memory.

**`_build` must read the energy off `lattice.ref`.** N3's version hard-coded `P0C` at
5 GeV, which is harmless there and silently fatal here: N4's only knob *is* the beam
energy, so a hard-coded `p0c` compares a resonance-tuned accsim ring against a 5 GeV xtrack
one — agreeing to nine digits on everything except the quantity the milestone is about.
Fixed in `tests/reference/test_polarization_xtrack.py::_build`; the fifth silent switch on
the reference side and the first that is ours rather than xtrack's.

**One real disagreement, measured and attributed.** The two `dn/ddelta` differ by `2e-6`
*absolute* while every other column of `N` agrees to `1e-8` relative. It is xtrack's, and
the tie is broken by a third quantity neither code's spin-field machinery computes: without
RF, `N (D, 0, 1)` must equal the momentum derivative of the **off-momentum closed spin
solution** (`closed_spin_solution(lattice, delta=…)`, threaded through in N4). accsim
satisfies that identity to `5e-9`; xtrack misses it by `1e-4`. The cause is the
`inv(I − A)` above: entries of order `1e11`, the unphysical `n_0` component subtracted
afterwards, and `1e11 × 1e-16 ~ 1e-5` of cancellation debris left in what survives.

Because the debris is **absolute**, the agreement is *best* nearest the resonance, where
`|dn/ddelta| ~ 9`: the two codes' equilibrium polarizations agree to `7e-5` there, and the
residual is xtrack's element-granularity rectangle rule, not the spin field.

## The bunched ring: `closed_orbit_delta` and the synchrotron sidebands (N5 — implemented)

Axes N1–N4 all ran on rings with **no RF cavity**. Adding one changes nothing about the
spin *map* — a cavity is thin, and thin elements do not precess — but it changes three
things underneath, and this section records all three.

### The closed orbit acquires a momentum, and the 4D solve cannot see it

`accsim.orbit.closed_orbit(_nonlinear)` finds the fixed point of the **transverse** map at
a `delta` the caller chooses; `zeta` is neither solved for nor looked at. That is exactly
right without RF, where `delta` is conserved and nothing reads `zeta`. With a cavity it is
not: the cavity reads `zeta`, so a state whose `zeta` does not return to itself is not
periodic in `delta` either.

On N2's bump ring one turn moves `zeta` by `−8.3e-7 m` — the closed orbit is **longer**
than the design circumference, because an orbit distortion adds path length at second
order in its angles. The RF is locked to the reference revolution, so the beam can only
arrive at the same phase every turn by shifting in momentum until the path length matches:

    delta_co = −(ΔC / C) / alpha_c            [ΔC = the one-turn zeta slip at delta = 0]

`accsim.orbit.closed_orbit_delta(lattice)` returns that scalar. It is solved as the **root
of the tracked slip** by secant, *not* from the formula above — the formula is the analytic
gate (`tests/analytic/test_spin_sidebands.py`), and the two must stay independent. They
agree to seven digits (`−4.778883e-8` against `−4.778882e-8`), and xtrack's 6D `twiss`
closed orbit confirms it to seven digits as well.

- **`zeta_co = 0` exactly**, so only the one scalar is needed. accsim's cavity kick is
  `sin(φ_s − k zeta) − sin(φ_s)`, so the synchronous particle sits at the zero crossing
  whatever the RF frequency is. Changing the frequency changes the bucket width and
  nothing else — there is no turn counter in the element, so the RF cannot drift in phase
  against the reference.
- **`closed_orbit_delta` returns `0.0` exactly when the one-turn `M[DELTA, ZETA]` is
  zero**, i.e. on any unbunched ring. That is a deliberate boundary, not a shortcut:
  without RF *every* momentum closes, the one that also closes `zeta` is still well
  defined, and N1–N4 deliberately did not use it. Adopting it silently would re-baseline
  four milestones of gates for no physics.
- It is also the **third appearance of one degeneracy** on this axis. Without RF, `zeta`
  and `delta` are both eigenvalue-`1` directions of the one-turn map, so a genuine 6D
  Newton solve is singular there — the same eigenvalue `1` that stops xtrack twissing a
  flat ring (N3) and that makes `inv(I − A)` singular for every ring (N4).
- `accsim.spin._closed_state(lattice, orbit0, delta=None)` now defaults to `None` = "the
  ring's own closed momentum". An **explicit** `delta` still means "close the orbit at
  this momentum", which is N4's off-momentum question and is only meaningful without RF.
  `spin_orbit_coupling` used to drop the keyword entirely; it now carries it, and
  `SpinOrbitCoupling` gained a `delta` field.

### Why it is not a rounding correction

`nu_0 = G gamma (1 + delta)` to the accuracy that matters, so `delta_co` moves the spin
tune by `G gamma delta_co ≈ 5.4e-7` — five percent of the distance at the closest point of
N5's resonance scan. Dropping it biases the *whole* scan the same way, so the pole would
still look like a clean straight line, just one aimed slightly off `Q_s`.

**And the slope is not `G gamma`.** The obvious guess — `nu_0 = G gamma`, and `gamma`
scales with the momentum — is **43% too big**. Measured on N2's arc,
`d nu_0/d delta = 0.7003 · G gamma`, with the vertical bump on *or off*, so it is a
property of the arc rather than of the distortion: an off-momentum closed orbit rides the
dispersion through the thin quadrupoles and takes a different path length through the
dipoles. Use the measured slope (an off-momentum `closed_spin_solution`), never `G gamma`.

### The resonance spectrum changes: sidebands at `nu_0 = k ± Q_s`

With RF the orbital spectrum is `exp(±2πi Q_x)`, `exp(±2πi Q_y)`, `exp(±2πi Q_s)` — the
**doubled eigenvalue `1` is gone**. The Sylvester equation `A N − N R = −D` is singular
exactly when the two spectra meet, so `N` now diverges at the *synchrotron sidebands*
`nu_0 = k ± Q_s` as well as the betatron ones. `N[:, ZETA]` is no longer zero.

- Gated as N4 gated its vertical sideband: `1/|N E_s|` is linear near the pole and
  extrapolates to `Q_s` within `1e-6`, and the residue
  `|N E_s| · 2|sin(π (nu_0 − Q_s))|` is constant to `2%` over three decades while
  `|N E_s|` runs over a factor of `1000`. Four alternative denominators
  (`nu_0 + Q_s`, `nu_0`, `nu_0 − Q_y`, `nu_0 − Q_x`) are excluded by a factor of `20`.
- **The far end of the scan carries a real background**: at `1e-2` the residue is still
  `19%` below its limit. Named, not swallowed by a tolerance.
- **The energy knob is no longer clean.** N4 could set `nu_0 = G gamma` by the beam energy
  while every optical tune stayed put. With a cavity, `Q_s² ~ 1/E` moves the *target*, and
  `Q_x` picks up a synchro-betatron shift (`+4.1e-3` at the gate voltage) through
  `R56·R65`. The scan energy must be solved **self-consistently**. `Q_y` is the one tune
  that does not move.
- **Mode identification needs a new rule.** N4 identified an orbital mode by which plane
  its eigenvector lived mostly in. That fails with three modes: on a dispersive ring the
  *horizontal* eigenvector's largest component is `zeta` (a betatron oscillation changes
  the path length), so "most content in `(zeta, delta)`" picks the wrong mode — and the
  first version of N5's scan found no pole at all because of it. The rule that works: the
  synchrotron mode is the one with the largest `|delta|`.
- **The sideband reaches the horizontal plane through the dispersion.** Across the scan
  `N`'s vertical columns move by `3%` while its horizontal ones grow by `900` alongside
  `|N E_s|` — because the resonant part of `N` lives along the synchrotron eigenvector,
  and on a dispersive ring that eigenvector has horizontal content.

### The primary gate is now tracking, because N4's identity does not survive

N4's sharpest check was `N (D, 0, 1) = d/ddelta [n_0 closed at delta]`, which leaned on
`delta` being conserved. With a cavity there is no off-momentum closed orbit to
differentiate. What replaces it is the **definition**: launch a particle at `x` with spin
`n_0 + N x`, track it, and require its spin to still be `n_0 + N x(turn)` many turns later.

That is a *first-order* statement, which is what makes it discriminating: the true residual
is `O(x²)`, so the **relative** residual (measured against `|N x|`) falls linearly with the
amplitude — measured `5.6e-4, 5.6e-5, 5.6e-6` — while a matrix wrong by a fraction `f`
leaves a relative residual stuck at `f` at every amplitude. It reads a wrong `N` off
directly rather than through a tolerance.

A second, weaker anchor: continuity onto the RF-free field, which N4's identity did pin.
Both the new `zeta` column and the shift in `dn/ddelta` vanish as `Q_s²` as the voltage is
taken down — gated as a **fitted exponent** (`2.09, 2.03, 2.01`), because the ratios
`|·|/Q_s²` are not flat: a linear-in-`Q_s` correction rides on the quadratic law.

### Comparing against xtrack: a pre-commitment that was refuted, and a real disagreement

N4 attributed the two codes' `2e-6` gap in `dn/ddelta` to xtrack's mode-by-mode
`inv(λ I − A)` on the `delta` mode, whose orbital eigenvalue is exactly `1`. Since RF
removes that eigenvalue, N5 pre-committed — in `docs/ROADMAP.md`, before measuring — that
xtrack's momentum column would come into line at `1e-8`.

**It does not.** A new and larger disagreement appears instead: zero without RF, growing as
`Q_s²`, and reaching **14%** on the gate ring (the `zeta` and `delta` columns differ by a
constant factor `1.1434`, the horizontal ones by `1.14` through the dispersion, the
vertical ones not at all).

The gap is **xtrack's**, decided without either code's spin-field machinery:

- The invariance test above, run in **xtrack's own tracker**: xtrack's own matrix sits at
  `3.56%` at every amplitude (a first-order error), accsim's falls with the amplitude.
  accsim's tracker gives the same verdict. There is no configuration in which the
  reference's matrix is the invariant one.
- And it is **downstream of everything both codes agree on**: differencing xtrack's own map
  gives `D` matching accsim's to `1.6e-9` and `R` to `1.2e-10`; the mode-by-mode
  construction transcribed from xtrack's own source reproduces accsim's Sylvester solve to
  `7.6e-11` (the two formulations are the same equation, confirmed numerically); and
  feeding **xtrack's** `D` and `R` through it returns *accsim's* matrix to `1.0e-7` — three
  orders inside the `14%` at issue. The error
  enters somewhere after that, in the stage where xtrack rescales its eigenvectors, tracks
  them at finite amplitude and reassembles `NN = EE_spin @ inv(EE_orb)` — *which* of those
  steps is not determined.
- One tempting mechanism is **excluded**: cancellation in reading tiny tracked deviations
  against a finite closed orbit would grow as the resonance is approached, and the
  discrepancy is flat in the tune distance (`1.1435` at `1e-2`, `1.1434` at `1e-3`). No
  further mechanism is claimed.

Consequence for scope: the Derbenev-Kondratenko `11/18` stays anchored where **N4** anchored
it — on an unbunched ring, where the two codes' fields agree. N5's polarization comparison
inherits the field disagreement (`−0.00569` against `−0.00747`) and is recorded as such
rather than used as a physics check.

## The 6D closed orbit: where a radiating ring actually closes (I4 — implemented)

`accsim.closed_orbit_6d(lattice, guess=None, *, radiation="off", ...)` — Newton on the
**full** tracked turn, `x ← x − (J − I)⁻¹(T(x) − x)`, with `J` by central differences.
Every other closed orbit in `orbit.py` pins `zeta = 0` and solves the transverse subspace
at a chosen `delta`; this one does not.

**What it changes, and it is exactly one thing.** With `radiation="off"` the answer *is*
the 4D one: `zeta_co = 0` and `delta_co = closed_orbit_delta`. Switch radiation on **in
tracking** and the ring has to pay for the light it emits, so it closes where the cavity
hands back exactly a turn's loss:

    q V [sin(φ_s − k_rf ζ_co) − sin(φ_s)] = U ,

read **at the cavity**. At any other point ζ differs by the share of the loss accumulated
in between — 10% of it on a ring with the cavity spliced mid-lattice. With the cavity last
(a thin element, so the turn *ends* at its entrance) the lattice start and the cavity
coincide exactly, which is why the mid-lattice ring exists in the test file.

**A correction to what N5 recorded.** `closed_orbit_delta`'s docstring and ROADMAP N5 both
said a 6D fixed point is needed when `φ_s ≠ 0` **or** radiation is tracked. Only the second
half holds in this package: the kick `sin(φ_s − kζ) − sin(φ_s)` vanishes at `ζ = 0` for
*every* `φ_s`, and the ramping reference that gives an accelerating bucket its meaning
lives inside `accelerate()`, which builds its own `ReferenceParticle` per turn and never
touches the tracking path. Asserted with `==` at three synchronous phases. **Tracked
radiation is the only thing in accsim that moves `ζ_co`.**

**The two arms, and why the second one's *order* is the gate.** `U` above is supplied
independently by (a) the tracked loss summed element by element along the converged orbit —
an identity, which holds to the tolerance the solve stopped at (`1.7e-5 eV` against a
`tol`-implied budget of `6.5e-5 eV`), and (b) `energy_loss_per_turn`, a design-route
radiation integral. (b) lands `1.7e-7` away, and that number is *not* a tolerance:

| evaluated on | departure of `energy_loss_per_turn` from the tracked loss | fitted exponent in `U₀/E` |
|---|---|---|
| the design orbit | `4.4e-3` | **0.999** |
| the 6D closed orbit | `1.7e-7` | **2.003** |

A lumped per-element kick makes the particle poorer as it goes, so every element after the
first radiates below the design energy — first order. On the closed orbit that error is
gone, because the fixed point is *where the sag is centred*: the beam sits high at the
cavity's exit and low at its entrance, and the linear-in-`delta` part of the loss averages
away over the turn. An orbit wrong by any fraction of the sag puts the first-order term
back and the exponent falls to 1. Fitted over a factor 512 in `U₀/E`.

**The 4D orbit's residual — a refuted pre-commitment, kept because the refutation is the
physics.** Predicted: dominated by `delta`, equal to the whole bill `U₀/(β₀²E₀) = 3.8e-3`.
Measured: dominated by **`zeta`** (`2.72e-2 m`, ten times the momentum miss), with the
momentum short by only `0.689` of the bill. Same cause for both — losing `delta` through
the arc slips the orbit in `zeta`, and by the cavity that slip already collects a third of
the loss back. Reconstructed bit-for-bit as `collected − bill` (those two are the only
things in the ring that change `delta`), and only with the *tracked* bill: the design-route
one is `4.4e-3` away here, which is the first-order lumping error above.

**Two fixed points, and the far one is named rather than avoided.** `sin(kζ) = U/V` has two
roots per RF period. The default seed (the 4D answer) lands on the stable one; a guess half
a metre away converges cleanly onto `k ζ = −(π + arcsin(U/V))`, the **unstable** point of
the previous bucket. Same contract as `closed_orbit_nonlinear`: a far guess makes no claim
about which fixed point comes back.

**Refused rather than iterated.** A ring with **no RF cavity** raises `ClosedOrbitError` —
the fourth appearance of one degeneracy on this project (N3 hit it, N4 explained it, N5
guarded it): with nothing reading `zeta`, both `zeta` and `delta` are eigenvalue-1
directions of `J − I`. A **stochastic** radiation model (`"quantum"`, `"photons"`) raises
`ValueError`: a random map has no fixed point, and Newton would converge onto whichever
photons it happened to draw.

**xtrack cross-check — a prediction, not a tolerance.** xtrack finds its 6D closed orbit by
*tracking* the line, so B2's rule applies unchanged (`integrator="uniform"`,
`num_multipole_kicks=1`, or the two codes integrate different maps). Written into the
ROADMAP before the file was run: the whole disagreement should be B2's already-named
residual — the CODATA-2014 charge in xtrack's `r0` (`1.0639e-8`) plus `2/γ₀²` — and nothing
else. **Predicted `2.29997e-8`, measured `2.29997e-8`**, agreeing to `2e-6` of the residual
itself. Both codes put the arrival time at `8.887901 cm` on a 40 m ring.

Two details of that comparison are gated rather than assumed. It is made on the **loss**,
not on `ζ_co`: the arcsine maps one relative error onto the other through
`tan(kζ)/(kζ)`, which is `1.0270` here and is ring-dependent, so comparing `ζ` directly
would fold in a factor nobody would think to divide out (the factor is asserted). And
feeding **xtrack's own** `ζ` through accsim's closed form reproduces `twiss.energy_loss` —
xtrack's own radiation bookkeeping, computed without reference to any closed orbit — to
`2e-9`, so the closed form belongs to neither code.

Gates: `tests/analytic/test_closed_orbit_6d.py` (16),
`tests/reference/test_closed_orbit_6d_xtrack.py` (4, one cached `xt.Line` build).

**Axis B's three private copies of this solve are retired into it** (`test_radiation_
tracking.py`, `test_radiation_quantum.py`, `test_quantum_lifetime_tracking.py` each carried
the same 25-line Newton), and one overlap is deliberate rather than a leftover: I4's ring is
B4's ring, so `test_the_momentum_swings_by_exactly_the_turns_loss_across_the_cavity` (I4)
and `test_the_radiating_closed_orbit_swings_by_two_sigma_in_momentum` (B4) make the same
statement about the same machine. **They are not copies.** I4 owns the *identity* — the
cavity restores exactly what the arcs took, exact by construction — and B4 owns the *shape*,
the element-by-element sag pinned at `−0.966`/`+0.921`/`1.887` sigma, which I4 never
measures and which is now a regression gate on `closed_orbit_6d` from outside I4's own file.
Neither assertion became circular in the swap: `closed_orbit_6d` imports nothing from
`radiation.py`, so B4 still puts a tracked orbit on one side and a design-route integral on
the other.

**Still out of scope:** a 6D orbit on a ring whose reference energy actually ramps (that is
`accelerate`'s per-turn reference, not a fixed point at all); and everything I3 listed
apart from this.

## Normalised coordinates: the linear normal form (O1 — implemented)

`accsim.normal_form(one_turn, *, method="6d"|"4d")` returns a `NormalForm` carrying `W`,
`W⁻¹`, the block-rotation `R` and the fractional mode tunes, with
`to_normalized` / `from_normalized` / `actions` beside it. The defining identity is

    M = W R W⁻¹ ,   R = diag(Rot(2π Q₁), Rot(2π Q₂), Rot(2π Q₃)) ,
    Rot(μ) = [[cos μ, sin μ], [−sin μ, cos μ]] .

### The parameterisation is a choice, and here is the choice

`M = W R W⁻¹` does **not** determine `W`. Right multiplication by anything commuting with
`R` preserves it — a per-plane rescaling *and* a per-plane rotation. Requiring `W`
symplectic removes the rescaling and leaves **three real numbers**, one rotation angle per
plane, completely free. So the two checks that look definitive are not:

| check | pins the scale | pins the phase |
|---|---|---|
| `M = W R W⁻¹` | no | no |
| `W` symplectic | yes | no |
| the Courant-Snyder tie below | yes | **yes** |

`tests/analytic/test_normal_form.py::test_definition_and_symplecticity_are_blind_to_the_phase`
builds a deliberately mis-phased `W` and shows it passes the first two. This is the J1
lesson (structural gates blind to the coefficient) in a new place.

**The convention chosen.** Each eigenvector is multiplied by `exp(−i arg(v[2p]))` — a
phase rotation until its **own plane's position component** is real and positive. Columns
are then `[Re v₁, Im v₁, Re v₂, Im v₂, Re v₃, Im v₃]`, scaled so `Re(v)·S·Im(v) = 1` with
`S` the block-diagonal `[[0,1],[−1,0]]`. Consequences, all of them testable:

- `W[0,1] = W[2,3] = W[4,5] = 0` and `W[2p,2p] > 0`;
- the 2×2 diagonal blocks *are* `[[√β, 0], [−α/√β, 1/√β]]`;
- one turn advances the normalised angle by `2π Q` in each plane, in the −atan2 sense
  set by `Rot`.

**Why it is a choice and not a copy.** It is also xtrack's convention
(`xtrack/linear_normal_form.py`), but that is not the justification. Under this phase and
no other, the diagonal blocks equal the Courant-Snyder matrix built from `closed_twiss`'s
`β`/`α` — a Stage-1 quantity obtained by matching a 2×2 block, with no eigenvector
anywhere in its derivation. Measured: `8.9e-16`, off-diagonal blocks exactly `0`.

**Mode labelling.** Each eigenvector is assigned to the plane where its weight
`|v[2p]|² + |v[2p+1]|²` is largest, by a maximum-weight assignment (`scipy`'s
`linear_sum_assignment`), so the labelling is always a permutation — never two modes on one
plane. This matches `normal_mode_tunes`'s rule. xtrack instead tie-breaks on `|v[5]|` then
`|v[2]|`; the two agree away from a coupling resonance and need not agree on one, which is
why the entry-by-entry cross-check is run off resonance.

**Rotation sense.** Within each conjugate pair the representative is the eigenvector with
**positive** symplectic norm `Re(v)·S·Im(v)`. That is what puts each tune in `[0, 1)`
rather than in the `arccos`-ambiguous `[0, 0.5]` — the same convention `normal_mode_tunes`
already used.

### The 6D normal form is not the 4D optics, and the difference is physics

`method="6d"` and `method="4d"` on the same ring give **different** `β`, different tunes
and a different dispersion. On the I4 ring: `β_x` 7.5% lower, `Q_x` `6.5e-3` lower, the
dispersion 24% higher. Neither is wrong:

- the **4D** quantities answer *a momentum held fixed* — the matched dispersion solves
  `(I − M₄)D = k₄` at constant `δ`;
- the **6D** quantities answer *a momentum oscillating at `Q_s`* — with RF on, `δ` is not
  a parameter but a coordinate, and the ring is being driven off-resonance rather than
  statically.

The two agree in the `Q_s → 0` limit and the departure is **quadratic** in `Q_s`, not
linear: fitted exponent `2.00` for all three quantities over a decade
(`test_6d_departs_from_4d_quadratically_in_the_synchrotron_tune`). `NormalForm.dispersion`
is therefore documented as the **dynamic** dispersion and is *not* interchangeable with
`Twiss.disp_x`; its formula is the one xtrack reports as `dx`, the mode-3 columns with the
`ζ` direction projected out.

### An RF-free ring has no 6D normal form — the fifth appearance of one degeneracy

Without a cavity, `ζ` and `δ` are both eigenvalue-`1` directions: the longitudinal mode's
symplectic norm is exactly zero and there is no plane to rotate in. `normal_form` raises
`NormalFormError` rather than dividing by it; xtrack's own routine raises `Invalid n3` on
the same matrix. This is the same degeneracy N3 met (`twiss` unable to do a flat ring), N4
explained (`inv(I − A)` singular for every ring), N5 hit in the spin field and I4 refused
in the 6D orbit. `method="4d"` is the answer for such a ring, and it is the mode in which
both closed-form ties (Courant-Snyder, Edwards-Teng) live anyway.

### `δ` versus `p_ζ`: nothing to correct at linear order

xtrack writes `W` in `(x, px, y, py, ζ, p_ζ)`; accsim's one-turn matrix is in `δ`. From
`p_ζ = (E − E₀)/(β₀²E₀)` and `dE/dδ = β₀P₀`, `dp_ζ/dδ = P₀/(β₀E₀) = 1` **exactly** at
`δ = 0`, so the two linear maps coincide and there is no `β₀²` anywhere in the comparison.
Asserted, not assumed (`test_pzeta_and_delta_are_the_same_variable_at_linear_order`).

### Emittance is deliberately not in the matrix

`to_normalized` is `W⁻¹x` and nothing else. xtrack's `get_normalized_coordinates` divides
by `√(ε_n/(β₀γ₀))` per mode; that scaling is a caller's concern, kept out so that the
object under test is never entangled with an emittance convention. The cross-check
multiplies it back in.

### The reference residual, and why it is not ours

The entry-by-entry xtrack comparison holds at `9e-16` on the transverse block — four
orders inside the pre-committed `1e-12` — and floors at `2.6e-11` on the longitudinal
columns. The whole excess is **one entry of xtrack's one-turn matrix**, `R56`, which it
obtains by symmetric finite difference of its exact drift map. That map's `ζ(δ)` is curved
(an `h²` truncation) *and* a difference of two nearly-equal path lengths (a cancellation
round-off going as `1/h`), so the entry has a U-shaped error with a minimum near
`ddelta = 1e-5`, where accsim's exact `L/γ₀²` and xtrack's finite difference agree to
`3e-11`. The attribution is gated, not asserted: the residual's minimum in the step size,
and its one-for-one tracking of `|R56_accsim − R56_xtrack|`, are both tests. A model
disagreement would have neither. Compare **absolutely**, never relatively:
`linear_normal_form.py` ends with `W[abs(W) < 1e-14] = 0`.

**Out of scope for O1:** the higher-order (Dragt-Finn / Deprit) normal form, which would
give amplitude-dependent tunes from the map rather than from J2's tracking; and `W` along
the ring, with the Mais-Ripken cross-plane betas and the crab dispersion that only exist
there — **shipped as O2**, see the next section.

## The normal form along the ring (O2 — implemented)

`accsim.propagate_normal_form(lattice, form0, *, maps=None)` returns one
`NormalFormPoint` per element boundary — `len(lattice) + 1` of them, the entrance then
each exit, the same shape and alignment as `propagate_twiss`. The rule is

    W(s) = M(0 -> s) W(0) . D(s),

with `D(s) = diag(Rot(theta_1), Rot(theta_2), Rot(theta_3))` the per-mode rotation that
puts the result back into O1's phase convention. `D` commutes with `R`, so every point
normalises **its own** local one-turn map `M(0->s) M M(0->s)^-1` to the *same* `R`: the
tunes belong to the ring, not to the point. `closed_normal_form(lattice, method=...)` is
the `closed_twiss` counterpart that supplies `form0`.

### The re-phasing angle, and the one quantity that can see it

Writing `(a, b) = (A[2p, 2p], A[2p, 2p+1])` for `A = M(0->s) W(0)`, the unique rotation
that zeroes the second entry and leaves the first positive is `theta_p = atan2(-b, a)`,
after which `W[2p, 2p] = +sqrt(a^2 + b^2)`. The angle removed, `phi_p = -theta_p`
accumulated continuously, **is** the mode's phase advance:

    mu_p(s) = unwrap( atan2(W_raw[2p, 2p+1], W_raw[2p, 2p]) ),  in radians.

Radians, to match `propagate_twiss`; xtrack reports the same thing in units of `2 pi` and
shifted to zero at the start.

**Almost nothing else in the milestone can see the re-phasing.** The Mais-Ripken tables and
both dispersions are invariant under `W -> W D` for any such `D`: in `betx2 = |v2[x]|^2`
and `alfx2 = -Re(v2[x] conj(v2[px]))` the phase cancels between the two factors, and the
dispersions are ratios taken *inside* one eigenvector. So the two witnesses are the
convention itself (`W[2p, 2p+1] = 0`, `W[2p, 2p] > 0`, asserted at every point) and `mu`.
This is O1's blindness lesson one level worse, and
`tests/analytic/test_normal_form_along_ring.py::test_the_new_quantities_are_blind_to_the_re_phasing`
demonstrates it with a `W(s)` mis-phased by `(0.7, -1.3, 2.1)` radians.

Because `mu` carries the whole weight, its gate is **quantised rather than a tolerance**:
`tunes()` returns the *full* integer-plus-fractional tune, so a dropped `np.unwrap` branch
is wrong by exactly `1` and nothing can absorb it. That needs a ring with an integer part
— the four-cell FODO used elsewhere reaches only `0.206` — and it needs no element to
advance the phase by more than `pi`, which is asserted separately as the localiser
(measured worst step `0.26` rad).

### Mais-Ripken: `betas[plane, mode]`

`NormalFormPoint.betas`, `.alphas`, `.gammas` are `(n_modes, n_modes)` matrices,

    B[p, m] = W[2p, 2m]^2 + W[2p, 2m+1]^2,
    A[p, m] = -(W[2p, 2m] W[2p+1, 2m] + W[2p, 2m+1] W[2p+1, 2m+1]),
    G[p, m] = W[2p+1, 2m]^2 + W[2p+1, 2m+1]^2,

so `B[0,0]`, `B[0,1]`, `B[1,0]`, `B[1,1]` are xtrack's `betx1`, `betx2`, `bety1`, `bety2`.
`mode_beta`/`mode_alpha` are the diagonals, and O1's `NormalForm` was refactored onto the
same helpers so the entrance is not a special case.

**`gammas` needed its own gates, and the reason generalises.** `betas` is tied to
`propagate_twiss`, to Edwards-Teng and to xtrack; `alphas` likewise. `gammas` is read off
the *momentum* row `2p+1` and had none of those: it appeared in the code, in this
document and in the blindness test, where the only assertion is that it is **invariant**
under the re-phasing. A wrong row index would have made it equal `betas` and passed every
test in both files. It now has the Stage 1 tie `gamma = (1 + alpha^2)/beta` on an
uncoupled ring, and on a coupled one the symplectic identity that binds all three matrices
at once:

    B[p,m] G[p,m] - A[p,m]^2 = det(block)^2   (Lagrange),
    sum over planes p of det(block)           = 1   for each mode m (W symplectic),

with `block = [[W[2p,2m], W[2p,2m+1]], [W[2p+1,2m], W[2p+1,2m+1]]]`. Note the per-plane
`beta gamma - alpha^2 = 1` is the **uncoupled** special case and is false entry by entry
once the planes mix; the sum is what survives. The general lesson worth carrying: **a
pre-commitment is also a list of what will not be gated** — anything shipped that is not
on it needs a tie found for it before the commit.

The **off-diagonal** entries are the point: they say how much of mode 2 is carried in `x`,
are exactly zero without coupling, and have no closed form of their own. What ties them to
something independent is G2: on a dispersion-free coupled ring

    W(s) = V(s) . diag(B_1(s), B_2(s))

at **every** point, with `V` the Edwards-Teng decoupling transform that
`propagate_coupled_twiss` obtains by re-matching the *local* one-turn map and transporting
nothing. Measured `1e-12`, no residual per-mode rotation. The two conventions are
compatible by construction: Edwards-Teng's `V[0,1] = V[2,3] = 0`, which is exactly what
O1's phase convention demands.

**A labelling hazard that was checked, not hoped.** `propagate_coupled_twiss` labels modes
per point and its own docstring warns they can swap where the local
`Delta = (Tr m - Tr q)/2` passes through zero; `propagate_normal_form` labels once at
`s = 0` and transports. A scan of `Delta(s)` around the coupled test ring found it constant
at `-0.30` with no sign change, so the tie can be written as an ordered comparison. On a
ring where it does flip, compare the unordered pair.

### Crab dispersion: an ordinary ring has it, and it is the dispersion's phase lag

`NormalFormPoint.crab_dispersion` is `(dx_zeta, dpx_zeta, dy_zeta, dpy_zeta)`, xtrack's
formula — the same construction as `dispersion` with the roles of `zeta` and `delta`
exchanged. Physically it is the transverse excursion in phase with **arrival time** rather
than with momentum: the head and the tail of a bunch sitting at different `x`.

A crab cavity produces it by construction. The interesting case is that an ordinary ring
produces it too, and the mechanism is worth stating because it also fixes the order.
Writing `c0 = v3[x] / v3[delta]` for the transverse response to the oscillating momentum,

    dx_zeta = - gamma_3 . Im(c0) / sigma_3,
    gamma_3 = |v3[delta]|^2,   sigma_3 = Im( v3[delta] conj(v3[zeta]) ).

`sigma_3` is the longitudinal share of the unit symplectic norm, order one. The other two
each carry one power of `Q_s`:

- `Im(c0)` — the **lag**. `c0 = [(lambda_3 I - M_4)^-1 m_delta]_x` with
  `lambda_3 = exp(2 pi i Q_s)`. At `Q_s -> 0` that is the real 4D matched dispersion; at
  finite `Q_s` the ring is driven off-resonance and the response acquires a phase, first
  order. Fitted exponent **1.0000000000011**.
- `gamma_3` — the mode's **momentum content**. As the cavity weakens the longitudinal
  ellipse elongates, so the momentum amplitude falls linearly in `Q_s` at fixed norm.

So `dx_zeta` is **second** order in `Q_s`, fitted **2.0011**. Both exponents and the
identity that multiplies them are gated, which makes them one statement instead of two
coincidences. (The first derivation of this gave `1`, having tracked only the lag; the
numerics said `2.0011` and the missing factor was `gamma_3`. Recorded because the lag
exponent alone is a plausible and wrong answer that a single loose gate would accept.)

On a **bend-free** ring `dx_zeta` is exactly `0` — `m_delta` has no transverse rows there,
so `c0 = 0`. Both codes return literal zero, which is the free gate that the quantity is
dispersive rather than numerical noise.

### No renormalisation, deliberately

xtrack renormalises the eigenvectors at every point (`_renormalize_eigenvectors`), which it
needs because it also propagates through radiation maps, where the symplectic norm
genuinely decays. accsim does not: with symplectic element maps `M(0->s) W(0)` stays
normalised exactly, so `W(s)^T S W(s) = S` at every point is a **measurement** rather than
an assumption, and it is gated as one. If radiation maps are ever passed through `maps=`,
that gate is where it will show up — which is the intended behaviour.

### The 4D propagation is unambiguous because of the element set, not the algebra

`method="4d"` transports the transverse block of the running 6x6 transfer matrix. That
equals the product of the per-element transverse blocks only when
`A[0:4,4] B[4,0:4] + A[0:4,5] B[5,0:4] = 0`, which holds because **no accsim element makes
the transverse coordinates depend on `zeta`** (there is no crab cavity) and **none makes
`delta` depend on the transverse ones** (the cavity kicks `delta` from `zeta` alone). Both
are properties of today's element set. The residual is asserted to be exactly `0.0` on the
bendy ring, so a future crab cavity — or a radiation map through `maps=`, which does give
`M[5, 0:4] != 0` — fails loudly rather than drifting.

### The reference residual, part two: it is transported, and it is confined

O1 pinned a `2.6e-11` residual in `W`'s longitudinal columns on xtrack's
finite-differenced `R56`. `W(s) = M(0->s) W(0)` **transports** that error, so along the
ring it grows — to `1.35e-10` by the end. The claim gated is not that it stays small but
that it stays **confined**: the transverse block holds at `9e-16` to `2e-15` at every
point, four orders cleaner, which a transport bug could not manage.

The same owner reaches `muzeta`, which is read off those columns: `mux`/`muy` land at
`1.9e-16` and `3.3e-16` while `muzeta` floors at `1.8e-11`, so the phase-advance
comparison carries **two** named constants rather than one loosened one. The sharpest of
the three signatures that pin it: **changing xtrack's momentum differentiation step over
three decades moves `muzeta`'s residual by five orders of magnitude and moves `mux`/`muy`
by not one bit.** A disagreement between two codes' physics does not care what step the
reference differentiates with, and does not stop at a plane boundary. (The other two: the
same U with the same minimum at `ddelta = 1e-5`, and step-to-step ratios matching `R56`'s
to `2%` above the minimum.)

### A tolerance that was too loose, and why

The bendy-ring comparisons were pre-committed at `5e-3` (`dx_zeta`) and `2e-3` (`dx`)
relative, on the reasoning that a ring with bends is also comparing the two codes' *bend
models* — the residual axis L and B2 own. They measure `1.1e-9` and `1.4e-8` **absolute**.
The reasoning was wrong in a specific way worth recording: **B2 had already removed that
residual**, by setting `integrator="uniform"` and one multipole kick per element on the
reference line for exactly this purpose. A gate at `5e-3` on a quantity agreeing at `1e-9`
would sleep through any regression worth catching, so the tolerances follow the
measurement (roughly two orders above it) rather than the pre-commitment. O1's
entrance-only `dx` comparison still carries the same over-loose `2e-3` for the same
reason.

Compare these **absolutely**, not relatively: `dx_zeta` and the alphas pass through zero
around the ring, where a relative comparison is meaningless, and
`xtrack/linear_normal_form.py` zeroes `W` entries below `1e-14` outright.

**Out of scope for O2:** the higher-order normal form (still O1's note), `W` at
sub-element resolution, and the Ripken-parameterised beam envelope — `coupled_beam_sigma`
already covers the 4D coupled case through `V`.

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

- **The analytic suite's one expensive tracking gate is B4's (2026-08-19).**
  `tests/analytic/test_quantum_lifetime_tracking.py` costs **54 s** of which the tracked
  survival gate is **32 s** (1500 particles x 1200 turns x 51 elements, `nonlinear=True`
  forced by radiation). That is the price of the milestone's headline and it was sized
  deliberately, not discovered: the budget is binomial on the survival fraction (1.9-4.4%
  over the three marks, gated at 3 sigma), which has to stay tight against a 37% effect.
  Two cheaper designs were measured and rejected — a fitted decay constant instead of
  pointwise survival is ~2x noisier for the same cost, and a 10-cell ring (half the
  elements, a third of the turns) does not confine a beam at all. **`slow` does not
  deselect**: `addopts` is `-m 'not reference'`, so a `slow` mark changes nothing about
  the default run; do not add `not slow` to make room, it would silently drop the
  existing long-term tracking gates from the green target.

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
