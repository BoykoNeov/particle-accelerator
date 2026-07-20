# `pp -> ttbar` — b-tagging performance against the Delphes card (milestone E2)

The generator → **fast detector sim** → analysis chain, pointed at *detector
performance* rather than a physics observable: how well does the simulated
detector tag b jets, and how often does it mistag a light jet?

```
Pythia8 (LHAPDF, pp -> ttbar, 13 TeV)  ->  HepMC3  ->  Delphes 3.5.0
        (CMS_PhaseII_0PU card: jets, flavour association, 3 b-tag working points)
                                       ->  analyze.py  (host .venv)
```

Run it:

```bash
ACCSIM_ENABLE_LHAPDF=1 ACCSIM_ENABLE_DELPHES=1 \
  .venv/Scripts/python.exe pipelines/pp_ttbar_btag/run_pipeline.py --n 20000
```

Both switches are required because the chain genuinely uses both tools — the
proton PDF and the detector. Everything past the pure-Python baseline is an
opt-in runtime switch, default OFF (see `docs/CONVENTIONS.md` → *Feature
switches*).

> **Disk warning.** t̄t events are large — full hadronic final states with ISR/FSR
> and the underlying event — and the HepMC3 interchange file is verbose ASCII.
> **20 000 events is ≈ 5.7 GB**, and it is written to `--out-dir`, copied into the
> Delphes container, and copied back. Budget ~15 GB of free space and a
> `docker cp` of that size in each direction. `--n 2000` (≈ 0.6 GB) is enough to
> see every guard fire; the larger sample only buys precision on the *tight*
> mistag rate, which is ~0.1% and therefore the statistics-hungriest number here.

## The gate: the card is the closed form

Delphes does **not** simulate a b-tagging algorithm. Its `BTagging` module is a
*parametrisation*: for each jet it picks an efficiency formula by the jet's
associated parton flavour, evaluates it at that jet's kinematics, and sets a bit
with that probability. So there is a **known right answer** for the tagging rate
of every jet, written down in the card, and a correct analysis must recover it.

The formulas are **parsed out of the card file itself** — the same file copied
out of the Delphes image and handed to `DelphesHepMC3` — by the tested host-side
module `accsim.events.btag`. They are never retyped into Python. A transcribed
formula is a remembered constant in disguise: it drifts silently when the card
changes, and a typo in it is invisible, because both sides of the comparison
would then share it.

## Why `ttbar`

The gate needs jets of *known* flavour, and needs all three classes the card
parametrises, in one sample:

| Source | Flavour | Card formula |
|---|---|---|
| `t -> W b` (two per event, always) | b | `EfficiencyFormula {5}` |
| `W -> c s` | c | `EfficiencyFormula {4}` |
| `W -> u d`, QCD radiation | light, gluon | `EfficiencyFormula {0}` (default = mistag) |

This is why t̄t is the canonical b-tagging calibration sample. The sibling
Drell-Yan chain (`../pp_mumu_drellyan/`) forces `Z -> mu mu` and contains no
signal jets at all, so E2 could not be a re-plot of it.

Decays are left **inclusive** — nothing is forced. Forcing a W channel would
bias the light/c mixture the mistag measurement is made of. Both production
channels (`gg2ttbar`, dominant, and `qqbar2ttbar`, ~10%) are on, since using one
alone would distort the gluon-jet content that is part of the mistag population.

## Why `CMS_PhaseII_0PU` and not the Drell-Yan chain's `delphes_card_CMS.tcl`

`delphes_card_CMS.tcl` configures a **single** `BTagging` module (`BitNumber 0`).
One bit is one operating point: no efficiency-vs-mistag curve can be drawn from
it, and "the card's working points" would be a singular. `CMS_PhaseII_0PU` runs
**three** — Loose / Medium / Tight on bits 0 / 1 / 2 — which makes the claim
plural and falsifiable, and throws in a free anchor (a tighter working point must
buy purity with efficiency, in *both* coordinates). The 0-pileup variant is used
because pileup is irrelevant to this gate and expensive to simulate.

Delphes is invoked with the card's own directory as the working directory,
because that card `source`s its per-working-point formula files (`btagLoose.tcl`,
`btagMedium.tcl`, `btagTight.tcl`) by **relative** path. The whole directory is
then copied back to the host so the analysis parses exactly what ran.

## The closed loop, and how it is broken

Delphes' `BTagging` keys on exactly the `Jet.Flavor` that its
`JetFlavorAssociation` module writes. So histogramming that field against the tag
bit and recovering the card is, on its own, a **closed loop**: it validates the
*handling* of the flavour label but cannot validate the label's *definition*. A
wrong definition would be reproduced faithfully by both sides.

So `generate_hepmc.cc` also dumps the generator-level heavy quarks straight from
Pythia's own event record — no HepMC/Delphes round-trip — and `analyze.py`
builds an **independent** truth flavour by ΔR-matching jets to them, then reports
how often the two labellings agree. That agreement is the part of the chain
Delphes cannot mark its own homework on.

The parton selection is deliberately **status-code-free**: the last quark of each
flavour chain (a `|PID|` 4 or 5 particle with no daughter of the same flavour),
not "status 23". This project already found that Pythia status codes do not
survive the HepMC3 round-trip (see `../ee_mumu_delphes/README.md`); keeping the
definition status-free makes it robust and applicable on either side.

## Two things the analysis gets right that are easy to get wrong

**`Jet.BTag` is a bitmask, not a boolean.** Three working points pack into one
integer, so `BTag == 1` means "loose but *not* medium" — not a working point at
all. The macro emits the mask untouched and the host decodes it with a bit number
that came from parsing the card, so the card decides which bit means what.

**The expected efficiency in a p_T bin is the jet-wise mean of the formula, not
the formula at the bin centre.** The jet spectrum falls steeply, so a bin is not
populated at its centre while the efficiency still varies across it. Using the
bin centre is a quiet, plot-invisible bias of order 0.07 in absolute efficiency —
the analytic suite asserts it is >10σ wrong on a falling spectrum where the
jet-wise mean closes. It is also what lets the same code handle a smooth `tanh`
card and a piecewise step-function card with no special-casing.

Relatedly, the pull uses the **expected** binomial variance rather than the
observed one, which is exactly zero — an infinite pull — in the zero-tag bins
that a ~0.1% mistag rate routinely produces.

## Output

`analyze.py` writes a two-panel figure and a table of per-bin pulls, plus four
guards:

- **CARD GATE** — χ²/ndf of (measured − configured)/σ over every populated bin.
- **WP ORDERING** — efficiency *and* mistag both fall Loose → Medium → Tight.
- **FLAVOUR ORDER** — ε_b > ε_c > ε_light; a tagger failing this is not a b-tagger.
- **FLAVOUR LABEL** — the independent ΔR-matched label agrees with `Jet.Flavor`.

Left panel: measured efficiency (points, with binomial errors) over the card's
configured curve (lines) vs jet p_T, for all three flavours × three working
points. Right panel: the three **operating points** in the (mistag, efficiency)
plane. That is an operating-point ROC, *not* a continuous discriminant sweep —
Delphes stores a decision bit and never a discriminant value, so a continuous ROC
is not obtainable from it, and the plot says so.

## Honest scope

- This is a **round-trip / consistency gate**, not a symbolic derivation like
  Robinson's theorem or `σ = 4πα²/3s`. There is no independent physics closed
  form here; the reference is a fit parametrisation the card encodes (the CMS
  card's cites arXiv:1211.4462). What is proven is that the extraction, the
  flavour handling, the binning and the estimator are right. It is the weakest
  analytic gate in this repo and is labelled as such.
- The analytic suite (`tests/analytic/test_btag_efficiency.py`) runs on
  **synthetic** jets and hand-written cards — no Docker, always in CI. This
  pipeline is the empirical demonstration on real Pythia/Delphes output.
- Not attempted: a continuous discriminant ROC (Delphes does not expose one),
  jet-energy-scale or resolution performance, τ-tagging, pileup, and the
  ATLAS-vs-CMS card comparison — that last one was considered for E2 and
  **rejected** because two detector outputs side by side have nothing to be
  refuted against, which fails this project's analytic-gate rule.
