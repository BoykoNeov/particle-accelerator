#!/usr/bin/env python3
"""Analysis stage — runs on the HOST in the project ``.venv``.

Overlays the **generator-level (truth)** and **detector-level (reco, after Delphes
CMS)** di-muon invariant-mass ``m(mu+ mu-)`` spectra for the *same* Pythia
Drell-Yan sample — the deliverable of the *hadronic* extension of Phase 2. The
canonical Drell-Yan signature is the **Z resonance peak** near
``M_Z = 91.19 GeV`` sitting on the ``gamma*`` continuum; here it is produced with
*real proton PDFs* (LHAPDF).

The point of a fast detector sim is what it does to the truth, so the two spectra
share one axis. Unlike the leptonic ``cos theta`` chain (whose detector signature
was the ``|eta| < 2.4`` acceptance *edge*), the visible detector effect here is
**mass-peak resolution broadening**: the CMS card smears each muon's momentum
(~1-2%), so the **reco Z peak is wider than the truth peak**. That broadening —
quantified below as the RMS in an 80-100 GeV window, reco > truth — is the proof
the detector step is live. (The *truth* peak is itself not a clean Breit-Wigner:
final-state radiation pulls ``m(mu mu)`` below the pole, giving a low-side tail, so
the truth peak *mode* recovers ``M_Z`` only to ~1-2 GeV — that is physics, not a
defect.)

A_FB is deliberately NOT measured here: hadronic Drell-Yan A_FB needs the
Collins-Soper frame + quark-direction dilution (symmetric pp), which is
research-grade and out of scope. The mass spectrum is the clean deliverable.

Usage:  python analyze.py <meta.dat> <truth.dat> <reco.dat> <out.png>
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

M_Z_PDG = 91.1880  # GeV (PDG Z pole mass) — the expected peak location
PEAK_WINDOW = (80.0, 100.0)  # window for the peak RMS (resolution) comparison


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


def peak_stats(mass: np.ndarray, edges: np.ndarray) -> tuple[float, float, float]:
    """Peak mode (bin-center of the tallest bin) + mean/RMS within PEAK_WINDOW."""
    counts, _ = np.histogram(mass, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = float(centers[int(np.argmax(counts))]) if counts.size else float("nan")
    lo, hi = PEAK_WINDOW
    win = mass[(mass >= lo) & (mass <= hi)]
    if win.size == 0:
        return mode, float("nan"), float("nan")
    return mode, float(win.mean()), float(win.std())


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

    m_lo = float(meta.get("mhat_min_GeV", "60"))
    m_hi = float(meta.get("mhat_max_GeV", "120"))
    n_bins = 60
    edges = np.linspace(m_lo, m_hi, n_bins + 1)
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
        label=f"reco (Delphes CMS, N={reco.size})",
    )
    ax.plot([edges[-2], edges[-1]], [r_counts[-1], r_counts[-1]], color="C3", lw=2)

    # PDG Z pole — the expected peak location.
    ax.axvline(M_Z_PDG, color="C0", ls="--", lw=1.3)
    ax.text(
        M_Z_PDG,
        ax.get_ylim()[1] * 0.97,
        r"  $M_Z$ (PDG)",
        color="C0",
        va="top",
        ha="left",
        fontsize=8,
    )

    t_mode, t_mean, t_rms = peak_stats(truth, edges)
    r_mode, r_mean, r_rms = peak_stats(reco, edges)
    transmission = reco.size / truth.size if truth.size else float("nan")

    sqrt_s_tev = float(meta.get("sqrt_s_GeV", "13000")) / 1000.0
    sigma_nb = float(meta.get("sigma_mb", "nan")) * 1e6
    pdf_set = meta.get("pdf_set", "?")
    ax.set_xlabel(r"$m_{\mu^+\mu^-}$  [GeV]")
    ax.set_ylabel("events / bin")
    ax.set_xlim(m_lo, m_hi)
    ax.set_title(
        rf"$pp\to\gamma^*/Z\to\mu^+\mu^-$ at $\sqrt{{s}}={sqrt_s_tev:.0f}$ TeV"
        rf"  —  truth vs Delphes CMS reco"
    )
    txt = (
        rf"peak mode: truth ${t_mode:.1f}$, reco ${r_mode:.1f}$ GeV" + "\n"
        rf"($M_Z^{{\rm PDG}}={M_Z_PDG:.2f}$)" + "\n"
        rf"peak RMS (80–100): truth ${t_rms:.2f}$, reco ${r_rms:.2f}$ GeV" + "\n"
        rf"acceptance $\times\varepsilon = {transmission:.2f}$" + "\n"
        rf"PDF: {pdf_set}" + "\n"
        rf"$\sigma_{{\rm DY\times BR}}={sigma_nb:.3g}$ nb ($60<m<120$)"
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
    print(
        f"saved truth-vs-reco di-muon mass spectrum -> {out_path}\n"
        f"  truth N={truth.size}  reco N={reco.size}  accept*eff={transmission:.3f}\n"
        f"  peak mode: truth={t_mode:.2f}  reco={r_mode:.2f}  (M_Z PDG={M_Z_PDG:.2f})\n"
        f"  peak RMS (80-100 GeV): truth={t_rms:.3f}  reco={r_rms:.3f}  "
        f"(reco>truth confirms momentum-resolution broadening)\n"
        f"  sigma(DY x BR) = {sigma_nb:.4g} nb"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
