# Phase 2 hadronic extension — Drell-Yan `p p → γ*/Z → μ+ μ-` (LHAPDF → Delphes CMS)

This is the **hadronic** extension of Phase 2: the leptonic chains
(`../ee_mumu_pythia/`, `../ee_mumu_delphes/`) had a *fixed* partonic √s; this one puts
**real proton PDFs (LHAPDF6)** in the initial state, so the partonic √ŝ is a
*distribution* — the essence of "with real PDFs". It drives the same two
**established** tools as the leptonic detector chain — Pythia8 and Delphes — coupled
through a **HepMC3** file, honouring the roadmap's *orchestrate, don't rebuild* rule.

Three deliverables, from the **same** generator/reco di-muon four-vectors:
1. the canonical **di-muon invariant-mass spectrum** with the **Z resonance peak** at
   `M_Z ≈ 91.19 GeV`,
2. the **forward-backward asymmetry `A_FB(m)`** in the **Collins-Soper frame** — the
   γ*/Z-interference signature (negative below the Z pole, positive above), with the
   `pp` **dilution** made explicit, and
3. the **DY angular coefficients `A₀(q_T)`, `A₂(q_T)`** and the **Lam–Tung relation
   `A₀ = A₂`** (see the Lam–Tung section below) — a truth-level `4π` observable, so
   this one **skips Delphes** (`--angular-only`).

The first two are shown **truth (generator) vs reco (after Delphes CMS)**; the third is
generator-truth only.

## The chain (two images, decoupled via HepMC3)

```
Pythia8 + LHAPDF (gen)  ──HepMC3 file──►  Delphes CMS (fast sim)  ──►  *_kin.dat (4-vectors)  ──►  plots
 rivet-pythia                             scailfin/delphes-python-centos                          host .venv
```

| File | Runs where | Role |
|------|-----------|------|
| `generate_hepmc.cc` | **rivet-pythia container** | Pythia8 `pp → γ*/Z → μ+μ-` at 13 TeV with an LHAPDF6 proton set, forced `Z→μμ`, `60<m<120 GeV` → a **HepMC3** file + `meta.dat` (σ, PDF set, params) **+ `truth_gen.dat`** (the *true* incoming-quark `p_z` sign per event, for the `A_FB` dilution reference) |
| `extract_kinematics.C` | **delphes container** (ROOT) | reads the Delphes ROOT file, writes the **μ⁻/μ⁺ four-vectors** for **both** truth (`Particle` branch) and reco (`Muon` branch) of the *same* events — raw kinematics only; all physics is computed on the host |
| `analyze.py`        | **host** (`.venv`) | from the four-vectors computes `m(μμ)` **and** `cos θ*_CS` via the tested `accsim.events.collins_soper_costheta`; overlays truth-vs-reco for the **mass spectrum** and **`A_FB(m)`** (with the undiluted true-direction overlay); enforces the `A_FB` **sign guard** |
| `analyze_angular.py`| **host** (`.venv`) | from `truth_gen.dat` computes `(cos θ*, φ*)` via `accsim.events.collins_soper_angles`, bins **`A₀(q_T)`, `A₂(q_T)`** via `angular_coefficients`, plots them, and enforces the **Lam–Tung demo guard** (`A₀ ≈ A₂` at low `q_T`). Truth-level `4π`, so **no Delphes** |
| `run_pipeline.py`   | **host** | orchestrates the chain (incl. the runtime `lhapdf get`); `--angular-only` runs GEN + `analyze_angular.py` only |

### Two established images

- **Generator** — `hepstore/rivet-pythia` (Pythia8 8.3 + HepMC3 + **LHAPDF 6.5.5**). Writes
  HepMC3 via the `Pythia8Plugins/HepMC3.h` plugin (ascii3).
- **Detector** — `scailfin/delphes-python-centos:3.5.0` (IRIS-HEP; **Delphes 3.5.0** + ROOT),
  run with the **CMS** hadron-collider card (`delphes_card_CMS.tcl`, no pile-up variant to
  keep the `Muon` branch clean). We decouple through the standard **HepMC3**
  generator→detector interchange (not `DelphesPythia8`, which would need Delphes compiled
  against this Pythia), as in the leptonic detector chain.

Bind mounts are avoided (the project path has a space, which breaks Docker `-v`); we
`docker cp` sources in and data out.

## Why 13 TeV + the CMS card (not the leptonic chain's 250 GeV / ILD)

This is a **hadron**-collider process, so it needs a proton PDF *and* a hadron-collider
detector card. 13 TeV is the LHC Run-2/3 energy and the CMS card is Delphes' canonical LHC
card. (The leptonic chains ran at ILC energies with the ILD e+e- card — a different machine.)

## Why an LHAPDF **LO** set

Pythia's Drell-Yan matrix element is **leading order**, so an **LO** PDF set is the
consistent partner (default `NNPDF31_lo_as_0118`, member 0 — recorded in `meta.dat`). The
image ships LHAPDF *without* grids, so the pipeline runs `lhapdf get <set>` first (needs
network in the container — a failure there is reported cleanly, not as a cryptic Pythia
init crash). ISR/FSR stay **on** (physical for hadronic DY); we deliberately do **not**
copy the leptonic chain's `PDF:lepton = off` (that was a lepton-beam ISR toggle, irrelevant
to protons).

## Run it

```bash
# one command; artifacts land in <out-dir> (default a temp dir), NOT committed.
# ~100k events keeps the off-peak A_FB bins statistically meaningful.
ACCSIM_ENABLE_LHAPDF=1 \
.venv/Scripts/python.exe pipelines/pp_mumu_drellyan/run_pipeline.py --n 100000 \
    --out-dir M:/claud_projects/temp/phase2_drellyan
```

(This is a gated addon — see below.) Outputs `meta.dat`, `events.hepmc`, `truth_kin.dat`,
`reco_kin.dat`, `truth_gen.dat`, and the labelled `drellyan_truth_vs_reco.png` (mass) and
`drellyan_afb_vs_mass.png` (`A_FB`). All regenerable, so **not committed**. The run exits
non-zero if the `A_FB` sign guard fails.

## The detector signature (why the plot proves the sim is live)

Unlike the leptonic `cosθ` chain — whose detector proof was the `|η| < 2.4` acceptance
**edge** — the Drell-Yan deliverable is a **mass spectrum**, and the detector leaves **two**
visible marks on it:

1. **reco ⊆ truth — the detector *removes* muon pairs.** Requiring *both* muons inside CMS
   acceptance and reconstructed gives `reco/truth = acceptance × ε² ≈ 0.36` (a Z at 13 TeV
   carries a large PDF-driven longitudinal boost, so one muon is often forward of
   `|η| < 2.4`). A detector never *adds* muons; that suppression is the first live-sim mark.
2. **The reco Z peak is broader than truth — momentum-resolution smearing.** The CMS card
   smears each muon's momentum, so the reco peak RMS exceeds the truth peak RMS. The effect
   is **modest** (CMS muon resolution at `pT ≈ 45 GeV` is excellent, ~1%, so the added width
   is sub-GeV on top of the natural `Γ_Z ≈ 2.49 GeV`) — but it is present and in the right
   direction (reco > truth).

The **truth** peak is itself **not** a clean Breit-Wigner: final-state radiation pulls
`m(μμ)` below the pole, giving a **low-side radiative tail**, so the truth peak *mode*
recovers `M_Z` only to ~1 GeV (a bin) — that is physics, not a defect.

### The honest cross-check: σ against a known LHC number

The Z-peak-at-`M_Z` check is semi-circular (the generator's `M_Z` in → out). The genuine
"the PDFs are doing something physical" evidence is the **cross section**: Pythia reports
`σ(DY × BR(Z→μμ), 60<m<120) ≈ 1.5 nb` at 13 TeV, which lands right on the measured LHC value
(σ(pp→Z→ℓℓ, 66–116 GeV) ≈ 1.9 nb NNLO per flavour; LO undershoots by the known K-factor
≈ 1.25). That agreement — from a *real global-fit PDF* convolved with the LO ME — is the
physical anchor. (It also settles a convention: the ~1.5 nb magnitude confirms Pythia's
`sigmaGen()` here is production σ **times** BR(Z→μμ), i.e. the μ-channel σ in the window,
not the full production σ.)

### Signal selection — simpler than the leptonic chain

Because this is a *resonance* process, we **force the boson decay** `23 → μ+μ-` in the
generator, so the only prompt muons **are** the signal pair — there is no τ→μ /
heavy-flavour contamination to reject, and hence **no** monochromatic-`|p|` cut (which the
leptonic Delphes chain needed). Both truth and reco simply take the **leading opposite-sign
muon pair** (robust when FSR occasionally yields more than two muons).

## A_FB(m) in the Collins-Soper frame — the second deliverable

Hadronic Drell-Yan `A_FB` needs the **Collins-Soper (CS)** frame — the di-muon rest frame
whose polar axis bisects beam-1 and reversed-beam-2 (minimising `Q_T` sensitivity). The
frame transform is **one tested function**, `accsim.events.collins_soper_costheta` (pure
numpy, analytic gate `tests/analytic/test_collins_soper.py`: the closed form equals an
independent boost-into-rest-frame construction over 3000 random pairs). The container macro
only dumps four-vectors, so no sign-error-prone frame math is duplicated in untested C++.

**The signature (and the physics gate).** `A_FB(m)` is **negative below** `M_Z`, crosses
**zero just under the pole**, and is **positive above** — the classic γ*/Z interference. The
magnitude has no clean closed form (interference × the `pp` dilution), so the acceptance
check is the **sign guard**: `A_FB < 0` below, `> 0` above. Measured (100k events): below
`−0.056 ± 0.007`, above `+0.108 ± 0.010` → `SIGN GUARD: PASS`. (The integrated-over-60–120
`A_FB` is ~0 by below/above cancellation — correct physics, not the headline.)

**The `pp` dilution, made explicit.** `pp` does not fix the quark direction, so the CS axis
is oriented by the `sign(Q_z)` proxy (the di-muon boost — the valence quark usually carries
more momentum). This probabilistic guess **dilutes** `A_FB` below parton level. We quantify
it: `generate_hepmc.cc` emits the **true** incoming-quark `p_z` sign, and `analyze.py`
overlays the **undiluted** `A_FB` (true direction) on the diluted proxy. Above the pole:
undiluted `+0.289 ± 0.010` vs proxy `+0.108` → **dilution factor ≈ 0.37**. The **reco** curve
(proxy only — a detector never knows the true quark direction) tracks the proxy truth, so the
**detector effect on `A_FB` ≪ the dilution**. (The leptonic 250 GeV chain measures `A_FB`
undiluted, because there the incoming `e-` direction *is* known — the contrast is the point.)

## A₀(q_T)/A₂(q_T) and Lam–Tung — the third deliverable

The full CS lepton angular distribution decomposes into coefficients `A₀..A₇`; the
headline is the **Lam–Tung relation `A₀ = A₂`** — the Drell-Yan analog of Callan–Gross,
**exact at O(α_s)** and violated only at O(α_s²). Because the moment inversion needs
**`4π` acceptance**, this is a **truth-level** observable: `run_pipeline.py --angular-only`
runs GEN only and `analyze_angular.py` extracts `A₀(q_T)`, `A₂(q_T)` in the Z window
(`80<m<100`) from the same `accsim.events` frame code (`collins_soper_angles`,
`angular_coefficients` — both analytic-gated on the host, `tests/analytic/`).

**The signature (and the gate).** Both coefficients **vanish as `q_T → 0`** and grow with
`q_T`; Lam–Tung holds across `q_T` within statistics. Measured (13 TeV, 200k events):
`A₀` rises from ~0 to `+0.225 ± 0.029` at `q_T ≈ 57` GeV, `A₂` tracking it; low-`q_T`
`⟨|A₀−A₂|⟩ = 0.023 ± 0.019` → `LAM-TUNG DEMO: PASS`. The *closed-form* O(α_s) proof of
`A₀ = A₂` (from explicit Dirac-γ hadronic tensors) is the always-run analytic gate
`tests/analytic/test_lam_tung.py`; this pipeline is the on-data demonstration of it. See
CONVENTIONS.md → *DY angular coefficients A₀–A₇ & Lam–Tung*.

## Gated addon

Like the other Phase-2 chains, this whole pipeline is an **opt-in runtime switch** (project
rule: everything past the pure-Python baseline is OFF by default). Running the script is the
opt-in; it gates on `accsim.features.require("lhapdf")` — set `ACCSIM_ENABLE_LHAPDF=1` (or
`accsim.features.enable("lhapdf")` in-process). The switch is named for the new tool this
chain introduces (**LHAPDF**), consistent with the `pythia` / `delphes` switches.

## Out of scope (deliberately)

Pile-up (the `_PileUp` CMS card), NLO/NNLO matrix elements + K-factors, the full γ*/Z/W
Drell-Yan family, PDF-uncertainty bands (the error-set members), and jet/b-tag performance.
For `A_FB`: the theory **dilution-correction unfolding** (recovering parton-level `A_FB` from
data *without* the generator truth) and the `sin²θ_W` extraction. (The CS **azimuthal** `φ*`
and angular coefficients `A_0..A_7` were previously out of scope; they are now the **third
deliverable** above.) The deliverables are the truth-vs-reco **Z-peak** and **`A_FB(m)`**, plus
the truth-level **`A₀(q_T)`/`A₂(q_T)`** Lam–Tung demo, in the muon channel with a real PDF.
