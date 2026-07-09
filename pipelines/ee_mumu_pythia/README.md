# Phase 2 real chain — `e+ e- → μ+ μ-` with Pythia8 (via Docker)

This is the **orchestrated** half of Phase 2 (roadmap acceptance clause **b**:
*"the orchestrated pipeline runs end-to-end and produces a labelled
distribution"*). It drives an **established generator — Pythia8** — rather than the
from-scratch toy in `accsim.events`, honouring the roadmap's *orchestrate, don't
rebuild* rule.

The toy generator (`accsim.events`, acceptance clause **a**) and this real chain
are complementary: the toy is analytically gated against `σ = 4πα²/(3s)`; this
chain shows a production generator running end-to-end, and its μ⁻ angular
distribution is qualitatively compared to the toy's `1 + cos²θ` law.

## Why Docker

Pythia8/MadGraph/Delphes do **not** build natively on this Windows / Python 3.14
host, and there is no `pythia8` wheel on native-Windows pip nor a Windows
conda-forge build (conda-forge `pythia8` is Linux/macOS only). Docker is installed
and its daemon runs on the Linux (WSL2) backend, so we use the prebuilt
**`hepstore/rivet-pythia`** image (Pythia8 8.3 as a C++ library).

Two deliberate choices:

- **No bind mount.** The project path contains a space
  (`M:\claud_projects\particle accelerator`), which breaks Docker `-v`. We
  `docker cp` the source in and the data out instead.
- **C++, not Python bindings.** This image ships Pythia8 as a C++ library only (no
  `pythia8` Python module), so the generator is a small `.cc` compiled in-container
  with `pythia8-config` flags — which is the native Pythia interface anyway.

## Files

| File | Runs where | Role |
|------|-----------|------|
| `generate_pythia.cc` | **in container** | Pythia8 generator: `e+e- → γ*/Z → μ+μ-`, writes `cos θ` of the primary μ⁻ |
| `analyze.py`        | **host** (`.venv`) | reads the data, renders the labelled `cos θ` histogram + `1+cos²θ` overlay |
| `run_pipeline.py`   | **host** | orchestrates the whole chain (start container → cp → compile → run → cp out → analyse) |

## Run it

```bash
# one command; artifacts land in <tempdir>/phase2_pythia by default
.venv/Scripts/python.exe pipelines/ee_mumu_pythia/run_pipeline.py --n 40000 \
    --out-dir M:/claud_projects/temp/phase2_pythia
```

Outputs `eemumu_costheta.dat` (raw angles + a header with Pythia's cross-section)
and `eemumu_costheta.png` (the labelled distribution). Both are regenerable, so
they are **not** committed — they go to a temp directory.

## Physics notes (honesty about the cross-check)

- **Process.** The 2→2 `WeakSingleBoson:ffbar2ffbar(s:gmZ)`. The 2→1 resonance
  `ffbar2gmZ` *underflows to zero* at a fixed √s far below the Z (its Breit-Wigner
  integrates over a delta-function `mHat`), so the 2→2 continuum process is the
  correct tool here.
- **Muon selection.** The process sums all outgoing fermion flavours; we pick the
  μ⁺μ⁻ subset by the **primary** pair (Pythia hard-process status code 23), which
  avoids counting muons produced in τ decays. The reported `σ ≈ 6.15 nb` at
  √s = 10 GeV is therefore the **all-flavour** total, not the μ⁺μ⁻ partial.
- **No cross-section equality with the toy.** `σ` is summed over all flavours
  (~6.15 nb at 10 GeV vs the toy's 0.87 nb for μ⁺μ⁻ alone), Pythia adds QED FSR /
  running α, and `PDF:lepton = off` fixes √s (no ISR). The cross-check is therefore
  **qualitative** — the μ⁻ angular *shape* tracks `1 + cos²θ` — never a numerical
  match to `4πα²/(3s)`.
- **Forward-backward asymmetry is NOT resolved here (measured, not assumed).** At
  10 GeV the process is γ\*-dominated (the Z is far off), so the γ-Z interference
  A_FB is only a few percent. Measured on an 18k-event μ⁺μ⁻ sample it is
  `A_FB = −0.0022 ± 0.0074` — consistent with zero. So the honest statement is that
  the Pythia and toy angular *shapes agree* at this energy; a visible A_FB is *not*
  a distinguishing feature at this statistics.

## Out of scope (deliberately)

The **detector** step (Delphes) and a **hadronic / PDF** extension (Drell-Yan with
LHAPDF) are not included — clause (b) needs only generation → labelled
distribution. See `docs/ROADMAP.md` (Phase 2) and `docs/CONVENTIONS.md` →
*Toy event generator* for the toy half.
