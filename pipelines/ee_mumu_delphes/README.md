# Phase 2 detector step — `e+ e- → μ+ μ-` through Delphes (ILD) at √s = 250 GeV

This is the **detector extension** of Phase 2: it adds the canonical *fast detector
simulation* step (**Delphes**) after the generator, so the deliverable is a
**generator-level (truth) vs detector-level (reco)** comparison — *what the detector
does to the truth*. It reuses the "orchestrate, don't rebuild" spirit of the
clause-(b) chain (`../ee_mumu_pythia/`) and drives two **established** tools —
Pythia8 and Delphes — coupled through a **HepMC3** file.

## Why √s = 250 GeV (not the 10 GeV clause-(b) chain)

Every standard Delphes e+e- card (**ILD**, IDEA, CLIC, …) is parametrized for
**≥ 91 GeV**; at the clause-(b) chain's 10 GeV there is *no* physically valid card.
So this chain runs at **250 GeV (ILC)**, where the ILD card operates in its designed
range. A bonus of going above the Z: γ*-Z interference makes the μ⁻ strongly
**forward-peaked** — a *measured* forward-backward asymmetry `A_FB ≈ +0.53` (vs the
10 GeV chain's `A_FB` consistent with **zero**). Not the symmetric `1 + cos²θ` of the
toy; that law only holds far below the Z.

## The chain (two images, decoupled via HepMC3)

```
Pythia8 (gen)  ──HepMC3 file──►  Delphes ILD (fast sim)  ──►  truth.dat + reco.dat  ──►  plot
 rivet-pythia                     scailfin/delphes-python-centos                          host .venv
```

| File | Runs where | Role |
|------|-----------|------|
| `generate_hepmc.cc` | **rivet-pythia container** | Pythia8 `e+e- → γ*/Z → μ+μ-` at 250 GeV → a **HepMC3** event file + a `meta.dat` (Pythia's σ + params) |
| `extract_reco.C`    | **delphes container** (ROOT) | reads the Delphes ROOT file, writes **both** truth (`Particle` branch) and reco (`Muon` branch) `cosθ` for the *same* events |
| `analyze.py`        | **host** (`.venv`) | overlays truth vs reco + the ILD acceptance edge; measures `A_FB` |
| `run_pipeline.py`   | **host** | orchestrates the whole two-container chain |

### Two established images

- **Generator** — `hepstore/rivet-pythia` (Pythia8 8.3 + HepMC3, as in the clause-(b)
  chain). Writes HepMC3 via the `Pythia8Plugins/HepMC3.h` plugin (ascii3).
- **Detector** — `scailfin/delphes-python-centos:3.5.0` (IRIS-HEP; **Delphes 3.5.0** +
  ROOT, ships `DelphesHepMC3` and the ILD/IDEA/CLIC cards). No single trustworthy
  image ships *both* Pythia and Delphes, and coupling them (`DelphesPythia8`) would
  need Delphes compiled against this Pythia — so we decouple through the standard
  **HepMC3** generator→detector interchange file instead.

Bind mounts are avoided (the project path has a space, which breaks Docker `-v`); we
`docker cp` sources in and data out, as in the clause-(b) chain.

## Run it

```bash
# one command; artifacts land in <out-dir> (default a temp dir), NOT committed
ACCSIM_ENABLE_DELPHES=1 \
.venv/Scripts/python.exe pipelines/ee_mumu_delphes/run_pipeline.py --n 20000 \
    --out-dir M:/claud_projects/temp/phase2_delphes
```

(This is a gated addon — see below.) Outputs `meta.dat`, `events.hepmc`,
`truth_costheta.dat`, `reco_costheta.dat`, and the labelled
`eemumu_truth_vs_reco.png`. All regenerable, so **not committed**.

## The detector signature (why the plot proves the sim is live)

The ILD card reconstructs muons at **95% efficiency for |η| < 2.4**, zero beyond, and
smears their *momentum* (angular resolution stays excellent). So on the `cosθ`
distribution:

- **reco ⊆ truth** — the detector *removes* muons, never adds. reco/truth =
  acceptance × efficiency ≈ **0.91** (≈ 0.95 eff × geometric acceptance).
- **reco vanishes beyond |cosθ| = tanh(2.4) = 0.984** — the |η| < 2.4 edge — while
  truth extends to ±1. That edge is the proof the detector step is doing something;
  if reco tracked truth to the beampipe, the card would be inert.
- `A_FB` is preserved (`+0.53` truth vs reco) — the acceptance is forward-backward
  symmetric, so it scales the yield without biasing the asymmetry.

### Isolating the signal muon (an honest subtlety)

The `ffbar2ffbar(s:gmZ)` process sums **all** outgoing fermion flavours, so the sample
also makes muons from τ→μ and heavy-flavour (b/c) decays. Two facts drive the handling:

1. Pythia's hard-outgoing **status 23 is not preserved through the HepMC round-trip**
   (FSR replaces it with status 51/52 copies + a status-1 final), so it cannot tag the
   signal in the Delphes record.
2. The signal μ⁻ is **monochromatic at |p| ≈ 125 GeV at every polar angle**, while the
   secondaries are soft — the status-1 μ⁻ `|p|` spectrum is bimodal (a ~125 GeV spike +
   a soft tail) with a wide empty valley.

So both truth and reco isolate the signal by an **angle-neutral `|p| > 100 GeV` cut**
(`|p| = pT·cosh η` for reco). `|p|` (not `pT`) is the crux: the signal is 125 GeV at
*all* `cosθ`, so the cut **cannot manufacture a forward edge** — the only edge left is
the detector's. Cross-check: the cut yields `truth N ≈ 1908`, matching the generator's
independent primary-μ⁻ count (`n_primary_mu ≈ 1956`, status 23 straight from Pythia) to
~2.5% — i.e. the cut really is selecting the signal.

## Gated addon

Like the clause-(b) chain, this whole pipeline is an **opt-in runtime switch** (project
rule: everything past the pure-Python baseline is OFF by default). Running the script is
the opt-in; it gates on `accsim.features.require("delphes")` — set
`ACCSIM_ENABLE_DELPHES=1` (or `accsim.features.enable("delphes")` in-process).

## Out of scope (deliberately)

Hadronic / PDF (LHAPDF Drell-Yan) is the remaining natural extension. Pile-up,
beam-induced backgrounds, jet/b-tag performance, and a full ILD reco are Delphes
features left unused — the deliverable is truth-vs-reco for the muon channel.
