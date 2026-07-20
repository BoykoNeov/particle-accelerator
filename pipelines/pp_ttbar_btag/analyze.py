#!/usr/bin/env python3
r"""b-tagging performance against the Delphes card's working points (milestone E2).

Runs on the HOST in the project ``.venv``. Deliverable of the ``pp -> ttbar``
chain: the **efficiency-vs-mistag operating points** of the CMS_PhaseII card's
three b-tag working points (Loose / Medium / Tight), and the demonstration that
each **reproduces the efficiency the card configured**.

What the gate is
----------------
Delphes does not simulate a tagging algorithm; its ``BTagging`` module evaluates a
per-flavour efficiency formula and sets a bit with that probability. So the card
*is* the closed form: for every jet there is a known right answer, and a correct
analysis must recover it. The physics is done by the tested host-side module
:mod:`accsim.events.btag`, whose formulas are **parsed out of the very card file
Delphes was run with** -- never transcribed -- so the reference and the
simulation read from one source.

The closed loop, and how it is broken
-------------------------------------
Delphes' ``BTagging`` keys on exactly the ``Jet.Flavor`` that its
``JetFlavorAssociation`` module writes. Histogramming that field against the tag
bit therefore validates the *handling* of the flavour label but cannot validate
the label's *definition* -- it is a closed loop.

So this script also builds an **independent** truth flavour: it dR-matches each
jet to the generator-level heavy quarks that Pythia dumped directly from its own
event record (no HepMC/Delphes round-trip), and reports how often the two
labellings agree. That agreement is the part of the chain Delphes cannot mark its
own homework on.

Usage:  python analyze.py <jets.dat> <gen_partons.dat> <card.tcl> <out.png> [meta.dat]
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from accsim.events.btag import (
    BOTTOM_FLAVOR,
    CHARM_FLAVOR,
    LIGHT_FLAVOR,
    efficiency_vs_pt,
    parse_btagging_working_points,
    roc_points,
)

# The card's JetFlavorAssociation cone. Matching the jet to generator partons
# with the SAME dR the card uses keeps the independent label comparable to
# Delphes' own; a different cone would make a disagreement uninterpretable.
MATCH_DR = 0.5

# Jet pT bins for the efficiency curves. The lowest edge sits at the card's jet
# finder threshold (15 GeV); the top bin is wide because the spectrum has fallen
# by orders of magnitude there -- and the jet-wise expectation handles the
# resulting within-bin variation correctly (see accsim.events.btag).
PT_EDGES = [20.0, 30.0, 45.0, 65.0, 95.0, 140.0, 220.0, 400.0]

FLAVOUR_LABELS = {
    BOTTOM_FLAVOR: ("b jets", "tab:red"),
    CHARM_FLAVOR: ("c jets", "tab:orange"),
    LIGHT_FLAVOR: ("light / gluon jets", "tab:blue"),
}


def load_jets(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Columns: event, pt, eta, phi, mass, flavor, btag."""
    raw = np.loadtxt(path, comments="#")
    raw = np.atleast_2d(raw)
    return {
        "event": raw[:, 0].astype(np.int64),
        "pt": raw[:, 1],
        "eta": raw[:, 2],
        "phi": raw[:, 3],
        "mass": raw[:, 4],
        "flavor": raw[:, 5].astype(np.int64),
        "btag": raw[:, 6].astype(np.int64),
    }


def load_partons(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Columns: event, pid, pt, eta, phi."""
    raw = np.atleast_2d(np.loadtxt(path, comments="#"))
    return {
        "event": raw[:, 0].astype(np.int64),
        "pid": raw[:, 1].astype(np.int64),
        "pt": raw[:, 2],
        "eta": raw[:, 3],
        "phi": raw[:, 4],
    }


def independent_flavour(jets: dict[str, np.ndarray], partons: dict[str, np.ndarray]) -> np.ndarray:
    """Truth flavour from a dR match to the generator-level heavy quarks.

    Derived from Pythia's own record, so it shares no code and no intermediate
    file with the ``Jet.Flavor`` that Delphes' tagger keys on. The b-over-c
    priority mirrors the card's stated rule (the highest PDG code inside the cone
    wins), so a jet containing both is a b jet on both sides.
    """
    by_event: dict[int, list[int]] = {}
    for idx, ev in enumerate(partons["event"]):
        by_event.setdefault(int(ev), []).append(idx)

    out = np.zeros(jets["pt"].size, dtype=np.int64)
    for j in range(out.size):
        rows = by_event.get(int(jets["event"][j]))
        if not rows:
            continue
        rows_arr = np.asarray(rows)
        deta = partons["eta"][rows_arr] - jets["eta"][j]
        dphi = np.abs(partons["phi"][rows_arr] - jets["phi"][j])
        dphi = np.minimum(dphi, 2.0 * np.pi - dphi)  # wrap into [0, pi]
        near = rows_arr[np.hypot(deta, dphi) < MATCH_DR]
        if near.size == 0:
            continue
        pids = np.abs(partons["pid"][near])
        out[j] = BOTTOM_FLAVOR if np.any(pids == 5) else CHARM_FLAVOR
    return out


def flavour_class(codes: np.ndarray) -> np.ndarray:
    """Collapse a parton PDG code to the class the card parametrises.

    Delphes writes the |PDG| of the hardest parton in the cone, so a light jet
    carries 1/2/3 and a gluon jet carries 21 -- **not** 0. Only b (5) and c (4)
    have their own formula; everything else falls to the card's default, which is
    the mistag rate. Comparing raw codes against a truth label that uses 0 for
    "light" would score every light jet as a disagreement.
    """
    out = np.zeros(codes.size, dtype=np.int64)
    a = np.abs(codes)
    out[a == BOTTOM_FLAVOR] = BOTTOM_FLAVOR
    out[a == CHARM_FLAVOR] = CHARM_FLAVOR
    return out


def report_flavour_agreement(delphes: np.ndarray, independent: np.ndarray) -> float:
    """Compare Delphes' label against the dR-matched generator one, by class."""
    d = flavour_class(delphes)
    i = flavour_class(independent)
    agree = float(np.mean(d == i))
    print("\nINDEPENDENT FLAVOUR CROSS-CHECK (breaks the Delphes closed loop)")
    print(f"  overall agreement            : {agree:6.3f}")
    for flav in (BOTTOM_FLAVOR, CHARM_FLAVOR, LIGHT_FLAVOR):
        sel = i == flav
        if sel.sum():
            name = FLAVOUR_LABELS[flav][0]
            print(
                f"  {name:22s}: gen-matched {int(sel.sum()):7d}"
                f"   labelled the same by Delphes {np.mean(d[sel] == flav):6.3f}"
            )
    return agree


def main() -> int:
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    jets_path, partons_path, card_path, out_path = (pathlib.Path(p) for p in sys.argv[1:5])
    meta = sys.argv[5] if len(sys.argv) > 5 else None

    jets = load_jets(jets_path)
    partons = load_partons(partons_path)
    working_points = parse_btagging_working_points(card_path)

    print(f"jets: {jets['pt'].size}   generator heavy quarks: {partons['pt'].size}")
    print(f"card: {card_path.name}")
    print("working points: " + ", ".join(f"{w.name} (bit {w.bit_number})" for w in working_points))
    if meta:
        print("meta: " + pathlib.Path(meta).read_text(encoding="utf-8").strip())

    agreement = report_flavour_agreement(jets["flavor"], independent_flavour(jets, partons))

    # ---- efficiency vs pT, per working point and flavour -------------------
    pulls: list[float] = []
    curves: dict[tuple[str, int], list] = {}
    for wp in working_points:
        print(f"\nWORKING POINT {wp.name}  (bit {wp.bit_number})")
        print(
            f"  {'flavour':<20s}{'pT bin':>16s}{'N':>8s}{'measured':>11s}{'card':>10s}{'pull':>8s}"
        )
        for flav, (label, _) in FLAVOUR_LABELS.items():
            points = efficiency_vs_pt(
                wp,
                flavor=jets["flavor"],
                pt=jets["pt"],
                eta=jets["eta"],
                btag_bits=jets["btag"],
                pt_edges=PT_EDGES,
                select_flavor=flav,
            )
            curves[(wp.name, flav)] = points
            for p in points:
                # Gate on the binomial VARIANCE, not the jet count: a bin with
                # thousands of jets but ~1 expected tag is Poisson, and its pull
                # is not Gaussian. See EfficiencyPoint.gaussian_valid.
                if not p.gaussian_valid:
                    continue
                pulls.append(p.pull)
                print(
                    f"  {label:<20s}{f'{p.pt_low:.0f}-{p.pt_high:.0f}':>16s}"
                    f"{p.n_jets:>8d}{p.measured:>11.4f}{p.expected:>10.4f}{p.pull:>8.2f}"
                )

    # ---- guards ------------------------------------------------------------
    pull_arr = np.array([p for p in pulls if np.isfinite(p)])
    chi2_ndf = float(np.mean(pull_arr**2)) if pull_arr.size else float("nan")
    roc = roc_points(
        working_points,
        flavor=jets["flavor"],
        pt=jets["pt"],
        eta=jets["eta"],
        btag_bits=jets["btag"],
    )
    effs = [sig.measured for _, _, sig in roc]
    mistags = [bkg.measured for _, bkg, _ in roc]

    print("\nGUARDS")
    card_ok = np.isfinite(chi2_ndf) and chi2_ndf < 2.0
    print(
        f"  CARD GATE      : chi2/ndf = {chi2_ndf:.2f} over {pull_arr.size} bins"
        f"  -> {'PASS' if card_ok else 'FAIL'}"
    )
    wp_ok = all(a > b for a, b in zip(effs, effs[1:], strict=False)) and all(
        a > b for a, b in zip(mistags, mistags[1:], strict=False)
    )
    print(
        f"  WP ORDERING    : eff {' > '.join(f'{e:.3f}' for e in effs)}"
        f" | mistag {' > '.join(f'{m:.4f}' for m in mistags)}"
        f"  -> {'PASS' if wp_ok else 'FAIL'}"
    )
    # Judged INCLUSIVELY (all pT in one bin), not as a mean of the pT bins: c jets
    # are the rarest class in ttbar, so at modest statistics no c bin clears the
    # per-bin population floor and the mean would be nan -- a statistics artifact
    # reported as a physics failure.
    flav_ok = True
    for wp in working_points:
        vals = []
        for flav in (BOTTOM_FLAVOR, CHARM_FLAVOR, LIGHT_FLAVOR):
            vals.append(
                efficiency_vs_pt(
                    wp,
                    flavor=jets["flavor"],
                    pt=jets["pt"],
                    eta=jets["eta"],
                    btag_bits=jets["btag"],
                    pt_edges=[PT_EDGES[0], np.inf],
                    select_flavor=flav,
                )[0].measured
            )
        flav_ok &= bool(vals[0] > vals[1] > vals[2])
        print(
            f"  FLAVOUR ORDER  : {wp.name:16s} "
            f"eps_b {vals[0]:.3f} > eps_c {vals[1]:.3f} > eps_light {vals[2]:.4f}"
            f"  -> {'PASS' if vals[0] > vals[1] > vals[2] else 'FAIL'}"
        )
    match_ok = agreement > 0.85
    print(
        f"  FLAVOUR LABEL  : independent dR match agrees {agreement:.3f}"
        f"  -> {'PASS' if match_ok else 'FAIL'}"
    )

    _plot(curves, working_points, roc, chi2_ndf, out_path)
    print(f"\nwrote {out_path}")
    return 0 if (card_ok and wp_ok and flav_ok and match_ok) else 1


def _plot(curves, working_points, roc, chi2_ndf: float, out_path: pathlib.Path) -> None:
    fig, (ax_eff, ax_roc) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    centres = 0.5 * (np.array(PT_EDGES[:-1]) + np.array(PT_EDGES[1:]))
    styles = {0: "-", 1: "--", 2: ":"}

    for wp in working_points:
        ls = styles.get(wp.bit_number, "-")
        for flav, (label, colour) in FLAVOUR_LABELS.items():
            pts = curves[(wp.name, flav)]
            ok = np.array([p.gaussian_valid for p in pts])
            if not ok.any():
                continue
            meas = np.array([p.measured for p in pts])
            err = np.array([p.error for p in pts])
            exp = np.array([p.expected for p in pts])
            # The card's configured value: the line. The measurement: the points.
            ax_eff.plot(centres[ok], exp[ok], ls, color=colour, lw=1.2, alpha=0.75)
            ax_eff.errorbar(
                centres[ok],
                meas[ok],
                yerr=err[ok],
                fmt="o",
                ms=3.5,
                color=colour,
                lw=1.0,
                label=f"{label}, {wp.name}" if wp.bit_number == 0 else None,
            )

    ax_eff.set_xscale("log")
    ax_eff.set_yscale("log")
    ax_eff.set_xlabel(r"jet $p_T$  [GeV]")
    ax_eff.set_ylabel("tagging efficiency")
    ax_eff.set_title(
        f"measured (points) vs card (lines)\n"
        rf"$\chi^2/\mathrm{{ndf}} = {chi2_ndf:.2f}$;  line style = Loose / Medium / Tight"
    )
    ax_eff.legend(loc="lower right", fontsize=8)
    ax_eff.grid(alpha=0.25, which="both")

    mistags = [bkg.measured for _, bkg, _ in roc]
    effs = [sig.measured for _, _, sig in roc]
    ax_roc.plot(mistags, effs, "o-", color="k", ms=7, lw=1.2)
    for (name, _bkg, _sig), m, e in zip(roc, mistags, effs, strict=True):
        ax_roc.annotate(
            f"{name}\n({m:.4f}, {e:.3f})",
            (m, e),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=8,
        )
    ax_roc.set_xscale("log")
    ax_roc.set_xlabel("light/gluon mistag rate")
    ax_roc.set_ylabel("b-tagging efficiency")
    ax_roc.set_title(
        "operating points of the card\n(discrete working points, not a discriminant sweep)"
    )
    ax_roc.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
