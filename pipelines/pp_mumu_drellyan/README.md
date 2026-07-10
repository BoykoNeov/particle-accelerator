# Phase 2 hadronic extension — Drell-Yan `p p → γ*/Z → μ+ μ-` (LHAPDF → Delphes CMS)

This is the **hadronic** extension of Phase 2: the leptonic chains
(`../ee_mumu_pythia/`, `../ee_mumu_delphes/`) had a *fixed* partonic √s; this one puts
**real proton PDFs (LHAPDF6)** in the initial state, so the partonic √ŝ is a
*distribution* — the essence of "with real PDFs". It drives the same two
**established** tools as the leptonic detector chain — Pythia8 and Delphes — coupled
through a **HepMC3** file, honouring the roadmap's *orchestrate, don't rebuild* rule.

The deliverable is the canonical Drell-Yan signature: the **di-muon invariant-mass
spectrum** with the **Z resonance peak** at `M_Z ≈ 91.19 GeV`, shown **truth (generator)
vs reco (after Delphes CMS)**.

## The chain (two images, decoupled via HepMC3)

```
Pythia8 + LHAPDF (gen)  ──HepMC3 file──►  Delphes CMS (fast sim)  ──►  truth.dat + reco.dat  ──►  plot
 rivet-pythia                             scailfin/delphes-python-centos                          host .venv
```

| File | Runs where | Role |
|------|-----------|------|
| `generate_hepmc.cc` | **rivet-pythia container** | Pythia8 `pp → γ*/Z → μ+μ-` at 13 TeV with an LHAPDF6 proton set, forced `Z→μμ`, `60<m<120 GeV` → a **HepMC3** file + `meta.dat` (σ, PDF set, params) |
| `extract_mass.C`    | **delphes container** (ROOT) | reads the Delphes ROOT file, writes **both** truth (`Particle` branch) and reco (`Muon` branch) `m(μμ)` for the *same* events |
| `analyze.py`        | **host** (`.venv`) | overlays truth vs reco `m(μμ)` + the `M_Z` marker; quantifies the peak mode + resolution broadening |
| `run_pipeline.py`   | **host** | orchestrates the whole two-container chain (incl. the runtime `lhapdf get`) |

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
# one command; artifacts land in <out-dir> (default a temp dir), NOT committed
ACCSIM_ENABLE_LHAPDF=1 \
.venv/Scripts/python.exe pipelines/pp_mumu_drellyan/run_pipeline.py --n 20000 \
    --out-dir M:/claud_projects/temp/phase2_drellyan
```

(This is a gated addon — see below.) Outputs `meta.dat`, `events.hepmc`, `truth_mass.dat`,
`reco_mass.dat`, and the labelled `drellyan_truth_vs_reco.png`. All regenerable, so **not
committed**.

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

## A_FB is deliberately *not* measured here

Hadronic Drell-Yan `A_FB` requires the **Collins-Soper** frame and a **quark-direction
dilution** correction (the `pp` initial state does not fix the quark direction) — that is
research-grade and out of scope per the roadmap. The mass spectrum is the clean, correct
deliverable. (The leptonic 250 GeV chain *does* measure `A_FB`, because there the incoming
`e-` direction is known.)

## Gated addon

Like the other Phase-2 chains, this whole pipeline is an **opt-in runtime switch** (project
rule: everything past the pure-Python baseline is OFF by default). Running the script is the
opt-in; it gates on `accsim.features.require("lhapdf")` — set `ACCSIM_ENABLE_LHAPDF=1` (or
`accsim.features.enable("lhapdf")` in-process). The switch is named for the new tool this
chain introduces (**LHAPDF**), consistent with the `pythia` / `delphes` switches.

## Out of scope (deliberately)

Pile-up (the `_PileUp` CMS card), NLO/NNLO matrix elements + K-factors, the full γ*/Z/W
Drell-Yan family, PDF-uncertainty bands (the error-set members), jet/b-tag performance, and
`A_FB` in the Collins-Soper frame. The deliverable is the truth-vs-reco Z-peak in the muon
channel with a real PDF.
