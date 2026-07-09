#!/usr/bin/env python3
"""Analysis stage — runs on the HOST in the project ``.venv``.

Reads the ``cos(theta)`` data produced inside the container by
``generate_pythia.py`` and renders a **labelled distribution** — the literal
Phase 2 clause-(b) deliverable for the real (Pythia) chain. Overlays the toy
model's ``1 + cos^2 theta`` law so the two sit side by side; they agree in *shape*
where gamma*/Z interference is small, which is the qualitative cross-check the
advisor recommended (no numerical cross-section equality is asserted — Pythia's
physics is richer than the toy's tree-level QED).

Usage:  python analyze.py <costheta.dat> <out.png>
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


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


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python analyze.py <costheta.dat> <out.png>", file=sys.stderr)
        return 2
    in_path, out_path = argv[1], argv[2]
    header, cos_theta = parse(in_path)

    n_bins = 20
    counts, edges = np.histogram(cos_theta, bins=n_bins, range=(-1.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(centers, counts, width=widths, alpha=0.7, label="Pythia8 events", align="center")

    # Area-matched 1 + cos^2(theta) reference (integral over [-1,1] is 8/3).
    total = counts.sum()
    bin_width = float(widths.mean())
    ref = (1.0 + centers**2) * total * bin_width / (8.0 / 3.0)
    ax.plot(centers, ref, "r-", lw=2, label=r"toy $1+\cos^2\theta$ (QED tree)")

    sqrt_s = header.get("sqrt_s_GeV", "?")
    sigma_nb = float(header.get("sigma_mb", "nan")) * 1e6
    ax.set_xlabel(r"$\cos\theta_{\mu^-}$")
    ax.set_ylabel("events / bin")
    ax.set_title(
        rf"Pythia8: $e^+e^-\to\gamma^*/Z\to\mu^+\mu^-$ at "
        rf"$\sqrt{{s}}={sqrt_s}$ GeV  ($\sigma={sigma_nb:.3g}$ nb)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved labelled distribution -> {out_path}  ({len(cos_theta)} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
