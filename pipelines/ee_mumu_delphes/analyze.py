#!/usr/bin/env python3
"""Analysis stage — runs on the HOST in the project ``.venv``.

Overlays the **generator-level (truth)** and **detector-level (reco, after
Delphes ILD)** ``cos(theta_mu-)`` distributions for the *same* Pythia sample, the
literal deliverable of the Phase 2 *detector* extension. The point of a fast
detector sim is what it does to the truth, so the two histograms sit on one axis;
the visible detector signature is the **acceptance edge** at |cos theta| = tanh(2.4)
= 0.984 (the ILD card's |eta| < 2.4 muon acceptance): reco muons vanish beyond it,
and sit at ~95% of truth inside it (the card's muon efficiency). That edge is the
proof the detector step is live -- if reco tracked truth with no edge, the card
would be doing nothing.

At sqrt(s) = 250 GeV the process is well above the Z, so gamma-Z interference makes
the mu- **forward-peaked** (a sizeable A_FB) -- unlike the symmetric 1 + cos^2 theta
of the 10 GeV clause-(b) chain. A_FB is therefore *measured* here (truth and reco),
not assumed, and printed on the plot.

Truth and reco angles both come from the Delphes ROOT macro (extract_reco.C), same
events, signal isolated by an angle-neutral |p| cut; the generator's meta.dat only
supplies Pythia's cross-section and an independent primary-mu- count.

Usage:  python analyze.py <meta.dat> <truth.dat> <reco.dat> <out.png>
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# ILD card muon acceptance: |eta| < 2.4  ->  |cos theta| < tanh(2.4).
ETA_MAX = 2.4
COS_EDGE = float(np.tanh(ETA_MAX))


def parse(path: str) -> tuple[dict[str, str], np.ndarray]:
    header: dict[str, str] = {}
    values: list[float] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                for tok in line.lstrip("#").split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        header[k] = v
                continue
            values.append(float(line))
    return header, np.asarray(values)


def a_fb(cos_theta: np.ndarray) -> tuple[float, float]:
    """Forward-backward asymmetry (N_F - N_B)/(N_F + N_B) + binomial error."""
    n_f = int((cos_theta > 0).sum())
    n_b = int((cos_theta < 0).sum())
    n = n_f + n_b
    if n == 0:
        return float("nan"), float("nan")
    afb = (n_f - n_b) / n
    err = np.sqrt((1.0 - afb**2) / n)  # binomial variance of the asymmetry
    return afb, err


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(
            "usage: python analyze.py <meta.dat> <truth.dat> <reco.dat> <out.png>",
            file=sys.stderr,
        )
        return 2
    meta_path, truth_path, reco_path, out_path = argv[1], argv[2], argv[3], argv[4]
    meta, _ = parse(meta_path)
    _, truth = parse(truth_path)
    _, reco = parse(reco_path)

    n_bins = 25
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    t_counts, _ = np.histogram(truth, bins=edges)
    r_counts, _ = np.histogram(reco, bins=edges)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(
        centers,
        t_counts,
        width=widths,
        align="center",
        color="0.75",
        edgecolor="0.5",
        label=f"truth (generator, N={truth.size})",
    )
    ax.step(
        edges[:-1],
        r_counts,
        where="post",
        color="C3",
        lw=2,
        label=f"reco (Delphes ILD, N={reco.size})",
    )
    # Draw the reco step's closing edge so the last bin is not left open.
    ax.plot([edges[-2], edges[-1]], [r_counts[-1], r_counts[-1]], color="C3", lw=2)

    # ILD |eta| < 2.4 acceptance edge -- the detector signature.
    for x in (-COS_EDGE, COS_EDGE):
        ax.axvline(x, color="C0", ls="--", lw=1.3)
    ax.axvspan(COS_EDGE, 1.0, color="C0", alpha=0.07)
    ax.axvspan(-1.0, -COS_EDGE, color="C0", alpha=0.07)
    ax.text(
        COS_EDGE,
        ax.get_ylim()[1] * 0.97,
        r"  ILD $|\eta|<2.4$",
        color="C0",
        va="top",
        ha="left",
        fontsize=8,
    )

    afb_t, afb_t_e = a_fb(truth)
    afb_r, afb_r_e = a_fb(reco)
    transmission = reco.size / truth.size if truth.size else float("nan")

    sqrt_s = meta.get("sqrt_s_GeV", "?")
    sigma_nb = float(meta.get("sigma_mb", "nan")) * 1e6
    n_primary_gen = meta.get("n_primary_mu", "?")
    ax.set_xlabel(r"$\cos\theta_{\mu^-}$")
    ax.set_ylabel("events / bin")
    ax.set_xlim(-1.0, 1.0)
    ax.set_title(
        rf"$e^+e^-\to\gamma^*/Z\to\mu^+\mu^-$ at $\sqrt{{s}}={sqrt_s}$ GeV"
        rf"  —  truth vs Delphes ILD reco"
    )
    txt = (
        rf"$A_{{FB}}$ truth $= {afb_t:+.3f}\pm{afb_t_e:.3f}$" + "\n"
        rf"$A_{{FB}}$ reco  $= {afb_r:+.3f}\pm{afb_r_e:.3f}$" + "\n"
        rf"acceptance $\times\varepsilon = {transmission:.2f}$" + "\n"
        rf"truth $N={truth.size}$ (gen primary $={n_primary_gen}$)" + "\n"
        rf"$\sigma_{{\rm all\,ff}}={sigma_nb:.3g}$ nb"
    )
    ax.text(
        0.02,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    reco_max = float(np.abs(reco).max()) if reco.size else float("nan")
    print(
        f"saved truth-vs-reco distribution -> {out_path}\n"
        f"  truth N={truth.size} (gen primary={n_primary_gen})  reco N={reco.size}"
        f"  accept*eff={transmission:.3f}\n"
        f"  A_FB truth={afb_t:+.3f}+/-{afb_t_e:.3f}  reco={afb_r:+.3f}+/-{afb_r_e:.3f}\n"
        f"  reco max |cos| = {reco_max:.3f}  (ILD edge = {COS_EDGE:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
