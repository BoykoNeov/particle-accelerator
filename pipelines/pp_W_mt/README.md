# E1 — `pp → W → μν` and the W-mass Jacobian edge (LHAPDF → Delphes CMS)

The **charged-current sibling** of `../pp_mumu_drellyan/`. Same orchestration
(Pythia8 + LHAPDF6 → HepMC3 → Delphes CMS), one decisive physics difference:
**the neutrino escapes.**

## Why this needs a different observable

In `pp → γ*/Z → μ⁺μ⁻` both decay products are measured, so `m(μμ)` is
reconstructible and the observable is a resonance **peak**. Here one product is a
neutrino — no detector signal at all, and its `p_z` is not recoverable even in
principle (the longitudinal boost of the `qq̄` system is unknown). So there is **no
invariant mass to build**. The observable is the **transverse mass**

```
m_T² = 2 · p_T^μ · p_T^ν · (1 − cos Δφ)
```

whose distribution has a **Jacobian edge at `M_W`**.

### The `M_W/2` trap, stated up front

The lepton-`p_T` spectrum *also* has a Jacobian peak — at **`M_W/2`**, not `M_W`.
Confusing the two is the classic error this observable invites. `m_T` is *the*
W-mass observable precisely because its edge is insensitive to the W's recoil `p_T`
at first order, while the `p_T^μ` peak is smeared by it at first order. The
pipeline asserts **both** edges (gates 1 and 3 below) so the distinction can't rot.

## The chain

```
Pythia8 + LHAPDF (gen) ──HepMC3──► Delphes CMS ──► *_kin.dat (μ + missing pT) ──► plots
 rivet-pythia                      scailfin/delphes-python-centos                host .venv
```

| File | Runs where | Role |
|------|-----------|------|
| `generate_hepmc.cc` | **rivet-pythia container** | Pythia8 `pp → W → μν` at 13 TeV, LHAPDF6 proton set, forced `W→μν` (**both charges**), **no mass window** → HepMC3 + `meta.dat` (σ, PDF set, **and the generator's own `M_W`/`Γ_W`**) |
| `extract_kinematics.C` | **delphes container** (ROOT) | dumps the μ four-vector + missing-`p_T` vector for **both** truth (`GenMissingET`) and reco (`MissingET`) of the same events — raw vectors only |
| `analyze.py` | **host** (`.venv`) | computes `m_T` via the tested `accsim.events.transverse_mass`, locates the edge via `jacobian_edge`, plots truth-vs-reco, enforces the gates |
| `run_pipeline.py` | **host** | orchestrates the chain (incl. the runtime `lhapdf get`) |

All physics is computed on the host by **tested** implementations (analytic gates:
`tests/analytic/test_transverse_mass.py`, `tests/analytic/test_jacobian_edge.py`).
The container macro only extracts, so no `(1 − cos Δφ)` is ever duplicated in
untested C++ — the same rule the DY chain follows for the Collins-Soper transform.

## Two design choices that differ from the Z chain

**No mass window.** The Z chain sets `PhaseSpace:mHatMin/Max = 60..120` to dodge the
divergent low-mass photon pole. The charged current has no photon-exchange piece, so
there is no pole to dodge — and a window would be **actively harmful**, imposing a
hard cutoff exactly where the Jacobian edge lives and manufacturing an artificial
one.

**Both W charges.** The edge sits at `M_W` for `W⁺` and `W⁻` alike, and `pp`
produces more `W⁺` (the proton's valence `uud` favours `u d̄ → W⁺`). Splitting by
charge would only cost statistics on a charge-independent observable.

## The gate is the edge *location*, never `m_T ≤ M_W`

The analytic suite proves `m_T ≤ M` for a **fixed** parent mass. Pythia gives the W
a **Breit-Wigner** mass, so off-shell events with `m(μν) > M_W` are real physics and
genuinely produce `m_T > M_W` — **measured at 6.2% of truth events** in a 3k run. An
assertion like "truth `max(m_T) ≤ M_W`" would therefore either fail on correct
physics, or pass only because a mass window had been imposed near the edge, hiding
the very effect being measured. Hence three gates on *positions and shapes*:

| Gate | Statement | Why it has teeth |
|---|---|---|
| **1** | truth edge within 5 GeV of the generator's `M_W` | catches a wrong `m_T`, a flipped MET sign, or a `p_T^μ` mix-up |
| **2** | reco edge measurably **rounder** than truth (falloff width) | pins the MET-resolution seam; a tight reco *position* gate is impossible (resolution), a loose one vacuous |
| **3** | truth `p_T^μ` edge within 5 GeV of `M_W/2` | the `M_W/2` trap, asserted rather than commented |

The 5 GeV tolerance is **justified, not tuned**: the half-maximum estimator's
measured bias at `Γ_W`-scale smearing is ~+1.5 GeV (tabulated in `jacobian_edge`'s
docstring), binning adds ~0.3 GeV, ISR recoil ~1 GeV — and 5 GeV is far below the
~35–40 GeV a `p_T`-for-`m_T` mix-up would produce.

The gate compares against **`meta.dat`'s `m_w_gev`, read back out of Pythia**, not a
remembered PDG number — otherwise the gate would be comparing two remembered
constants.

## The `GenMissingET` sign is pinned by data, not memory

Delphes' `Merger` negates its vector sum to form a "missing" momentum, but
`GenMissingET`'s input is the **neutrino list itself** (`PdgCodeFilter` on
`|pid| ∈ {12,14,16}`), not the visible particles. So the output could point *along*
the neutrino or *opposite* it — a `π` shift in `Δφ`, which flips `(1 − cos Δφ)`
between `~0` and `~2` and wrecks `m_T`.

Rather than trust a convention, the macro emits **both** `GenMissingET` and the
directly summed truth neutrino, and `analyze.py` measures the angle between them:

```
GenMissingET vs summed neutrino: median |dphi| = 0.0000 rad, aligned 100.0%, opposed 0.0%
  -> pinned: GenMissingET points ALONG the neutrino (sign = +1)
```

It **refuses to run** if the two match neither convention.

**Muons are inside Delphes' `MissingET` — verified in the card, not assumed.**
`MissingET ← EFlowMerger/eflow ← HCal/eflowTracks ← TrackMerger`, and `TrackMerger`
takes `MuonMomentumSmearing/muons` (`delphes_card_CMS.tcl` line ~201). Had the muon
been excluded, MET would track the hadronic recoil instead of the neutrino and every
reco `m_T` would be meaningless.

## Negative controls (measured, 3k events)

| Mutation | Result |
|---|---|
| `GenMissingET` sign flipped | median `m_T` 62.3 → **6.9 GeV**, edge 30.2 GeV off → **FAIL** |
| `p_T^μ` fed to gate 1 | edge 44.8, **35.6 GeV off** → **FAIL** |
| reco `MissingET` sign flipped | median `m_T` 64.0 → **9.4 GeV** → **FAIL** |

## Run it

```bash
# one command; artifacts land in <out-dir> (default a temp dir), NOT committed.
ACCSIM_ENABLE_LHAPDF=1 \
.venv/Scripts/python.exe pipelines/pp_W_mt/run_pipeline.py --n 3000 \
    --out-dir M:/claud_projects/temp/e1_wmass/run_smoke
```

> **Disk warning — this one is worse than the other chains.** The HepMC3
> interchange file is verbose ASCII and carries the *full* event (ISR/FSR +
> underlying event), and W production has a large cross section, so the file grows
> fast: **60 000 events is ≈ 12.5 GB** (measured), against ≈ 5.7 GB for 20 000
> `t̄t` events in `../pp_ttbar_btag/`. It is written to `--out-dir`, `docker cp`-ed
> into the Delphes container, and copied back — so budget ~30 GB free and two
> copies of that size. **`--n 3000` (≈ 0.6 GB) fires every gate with margin** and is
> the recommended default; larger samples only smooth the histogram, since the
> gates are on edge *positions* and those are already well determined at 3k.

This is a **gated addon** (`ACCSIM_ENABLE_LHAPDF=1`) — see
`docs/CONVENTIONS.md` → *Feature switches*. Outputs `meta.dat`, `events.hepmc`,
`truth_kin.dat`, `reco_kin.dat` and `w_transverse_mass.png`. All regenerable, so
**not committed**. The run exits non-zero if any gate fails.

## Scope, stated honestly

This locates an edge; it is **not** a W-mass measurement. A real one fits templates
over the whole spectrum, calibrates the hadronic recoil, and controls PDF and
QED-radiation systematics to well under 10 MeV. `jacobian_edge` returns a number
good to roughly the smearing scale — which is what the E1 gate needs and no more.
Not attempted: the `W` charge asymmetry, `p_T^W` resolution/recoil calibration, the
electron channel, or pileup.
