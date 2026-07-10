#!/usr/bin/env python3
r"""Analysis stage — runs on the HOST in the project ``.venv``.

Two deliverables from the SAME truth/reco di-muon four-vectors of a Pythia
Drell-Yan sample (real proton PDFs, LHAPDF), each shown **truth (generator) vs reco
(after Delphes CMS)**:

1. **Di-muon invariant-mass spectrum** — the canonical **Z resonance peak** near
   ``M_Z = 91.19 GeV``. The visible detector effect is **mass-peak resolution
   broadening** (CMS smears each muon's momentum, so the reco peak RMS > truth), on
   top of **reco ⊆ truth** acceptance loss. (The *truth* peak is itself not a clean
   Breit-Wigner: FSR pulls ``m(mu mu)`` below the pole, a low-side tail.)
2. **Forward-backward asymmetry ``A_FB(m)``** in the **Collins-Soper frame** — the
   classic γ*/Z-interference signature: ``A_FB < 0`` below the Z pole, ``≈ 0`` at
   the pole, ``> 0`` above it. This is the **sign guard** (a flipped μ⁻/μ⁺ or axis
   sign would invert it); the *magnitude* has no clean closed form (interference +
   the ``pp`` dilution), so the sign is the physics gate, not a tolerance.

All physics is computed here from the raw four-vectors by the *single tested*
implementation ``accsim.events.collins_soper_costheta`` (analytic gate:
``tests/analytic/test_collins_soper.py``) — the container macro only extracts
four-vectors, so the CS frame transform is never duplicated in untested C++.

**The ``pp`` dilution.** ``A_FB`` uses the standard ``sign(Q_z)`` quark-direction
proxy (the di-lepton boost direction), which *dilutes* the measured asymmetry below
the parton level. If the optional generator-truth file (``truth_gen.dat``, carrying
the *true* incoming-quark ``p_z`` sign from the Pythia record) is passed, a third
curve — the **undiluted** ``A_FB`` from the true quark direction — is overlaid, and
the proxy/true ratio (the dilution factor, ``<1``) is reported. The reco curve stays
proxy-only, as an experiment never knows the true quark direction.

Usage:  python analyze.py <meta.dat> <truth_kin.dat> <reco_kin.dat> <mass.png>
        <afb.png> [truth_gen.dat]
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from accsim.events import (
    collins_soper_costheta,
    forward_backward_asymmetry,
    invariant_mass_squared,
)

M_Z_PDG = 91.1880  # GeV (PDG Z pole mass) — the expected peak location
PEAK_WINDOW = (80.0, 100.0)  # window for the peak RMS (resolution) comparison
# Two off-pole regions for the A_FB sign guard (avoid the ~0 pole itself).
BELOW_POLE = (60.0, 88.0)
ABOVE_POLE = (94.0, 120.0)


def parse(path: str) -> tuple[dict[str, str], np.ndarray]:
    """Read a ``# key=val`` header + numeric rows. Returns (header, ``(N, ncol)``)."""
    header: dict[str, str] = {}
    rows: list[list[float]] = []
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
            rows.append([float(x) for x in line.split()])
    return header, np.asarray(rows, dtype=np.float64)


def kinematics(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From ``(N, 8)`` (μ⁻ then μ⁺ four-vectors) return ``(m_mumu, cos θ*_CS)``."""
    if data.size == 0:
        return np.empty(0), np.empty(0)
    p_minus = data[:, 0:4]
    p_plus = data[:, 4:8]
    m = np.sqrt(np.abs(invariant_mass_squared(p_minus + p_plus)))
    cos = collins_soper_costheta(p_minus, p_plus)  # default sign(Q_z) proxy
    return m, cos


def kinematics_gen(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From ``(N, 9)`` generator truth (``qsign`` + μ⁻,μ⁺ four-vectors) return
    ``(m, cos θ* proxy, cos θ* true-quark-direction)`` — the dilution pair."""
    if data.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    qsign = data[:, 0]
    p_minus = data[:, 1:5]
    p_plus = data[:, 5:9]
    m = np.sqrt(np.abs(invariant_mass_squared(p_minus + p_plus)))
    cos_proxy = collins_soper_costheta(p_minus, p_plus)  # sign(Q_z)
    cos_true = collins_soper_costheta(p_minus, p_plus, quark_direction=qsign)
    return m, cos_proxy, cos_true


def peak_stats(mass: np.ndarray, edges: np.ndarray) -> tuple[float, float, float]:
    """Peak mode (tallest-bin centre) + mean/RMS within PEAK_WINDOW."""
    counts, _ = np.histogram(mass, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = float(centers[int(np.argmax(counts))]) if counts.size else float("nan")
    lo, hi = PEAK_WINDOW
    win = mass[(mass >= lo) & (mass <= hi)]
    if win.size == 0:
        return mode, float("nan"), float("nan")
    return mode, float(win.mean()), float(win.std())


def afb_in_window(mass: np.ndarray, cos: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """``A_FB`` (+ binomial error) among pairs with ``lo <= m < hi``."""
    sel = (mass >= lo) & (mass < hi)
    return forward_backward_asymmetry(cos[sel])


def afb_profile(
    mass: np.ndarray, cos: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin ``A_FB`` and its error across ``edges``."""
    afb = np.full(edges.size - 1, np.nan)
    err = np.full(edges.size - 1, np.nan)
    for i in range(edges.size - 1):
        afb[i], err[i] = afb_in_window(mass, cos, edges[i], edges[i + 1])
    return afb, err


def mass_plot(
    meta: dict[str, str],
    truth_m: np.ndarray,
    reco_m: np.ndarray,
    m_lo: float,
    m_hi: float,
    out_path: str,
) -> None:
    n_bins = 60
    edges = np.linspace(m_lo, m_hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    t_counts, _ = np.histogram(truth_m, bins=edges)
    r_counts, _ = np.histogram(reco_m, bins=edges)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(
        centers,
        t_counts,
        width=widths,
        align="center",
        color="0.75",
        edgecolor="0.5",
        label=f"truth (generator, N={truth_m.size})",
    )
    ax.step(
        edges[:-1],
        r_counts,
        where="post",
        color="C3",
        lw=2,
        label=f"reco (Delphes CMS, N={reco_m.size})",
    )
    ax.plot([edges[-2], edges[-1]], [r_counts[-1], r_counts[-1]], color="C3", lw=2)

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

    t_mode, _, t_rms = peak_stats(truth_m, edges)
    r_mode, _, r_rms = peak_stats(reco_m, edges)
    transmission = reco_m.size / truth_m.size if truth_m.size else float("nan")
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
        rf"acceptance $\times\varepsilon^2 = {transmission:.2f}$" + "\n"
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
    plt.close(fig)


def afb_plot(
    truth_m: np.ndarray,
    truth_c: np.ndarray,
    reco_m: np.ndarray,
    reco_c: np.ndarray,
    m_lo: float,
    m_hi: float,
    sqrt_s_tev: float,
    out_path: str,
    gen: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[float, float, float, float]:
    """Plot ``A_FB(m)``; if ``gen=(m, cos_proxy, cos_true)`` is given, add the
    undiluted (true-quark-direction) truth curve to show the ``pp`` dilution."""
    n_bins = 12  # coarse-ish so the off-peak bins keep statistics
    edges = np.linspace(m_lo, m_hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    t_afb, t_err = afb_profile(truth_m, truth_c, edges)
    r_afb, r_err = afb_profile(reco_m, reco_c, edges)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(M_Z_PDG, color="C0", ls="--", lw=1.3)

    dilution_txt = ""
    if gen is not None:
        gm, _gc_proxy, gc_true = gen
        u_afb, u_err = afb_profile(gm, gc_true, edges)
        ax.errorbar(
            centers,
            u_afb,
            yerr=u_err,
            fmt="^-",
            color="C2",
            ms=4,
            lw=1.5,
            capsize=2,
            label=r"truth, true quark dir (undiluted)",
        )
        u_above, u_above_e = afb_in_window(gm, gc_true, *ABOVE_POLE)
        p_above, _ = afb_in_window(gm, _gc_proxy, *ABOVE_POLE)
        d_factor = p_above / u_above if u_above else float("nan")
        dilution_txt = (
            "\n"
            rf"dilution (above pole): proxy/true $={d_factor:.2f}$"
            "\n"
            rf"  undiluted $A_{{\rm FB}}={u_above:+.3f}\pm{u_above_e:.3f}$"
        )

    ax.errorbar(
        centers,
        t_afb,
        yerr=t_err,
        fmt="o-",
        color="0.35",
        ms=4,
        lw=1.5,
        capsize=2,
        label=r"truth, $\mathrm{sign}(Q_z)$ proxy (diluted)",
    )
    ax.errorbar(
        centers,
        r_afb,
        yerr=r_err,
        fmt="s--",
        color="C3",
        ms=4,
        lw=1.5,
        capsize=2,
        label=r"reco (Delphes CMS), proxy",
    )

    # Off-pole integrated A_FB — the physics sign guard (on the diluted proxy truth).
    t_below, t_below_e = afb_in_window(truth_m, truth_c, *BELOW_POLE)
    t_above, t_above_e = afb_in_window(truth_m, truth_c, *ABOVE_POLE)

    ax.text(
        M_Z_PDG, ax.get_ylim()[0] * 0.92, r"$M_Z$", color="C0", va="bottom", ha="center", fontsize=9
    )
    ax.set_xlabel(r"$m_{\mu^+\mu^-}$  [GeV]")
    ax.set_ylabel(r"$A_{\rm FB}$  (Collins–Soper frame)")
    ax.set_xlim(m_lo, m_hi)
    ax.set_title(
        rf"$A_{{\rm FB}}(m)$ in the Collins–Soper frame  —  "
        rf"$pp\to\gamma^*/Z\to\mu^+\mu^-$ at $\sqrt{{s}}={sqrt_s_tev:.0f}$ TeV"
    )
    txt = (
        "sign guard (proxy truth):\n"
        rf"  below ${BELOW_POLE[0]:.0f}$–${BELOW_POLE[1]:.0f}$: "
        rf"$A_{{\rm FB}}={t_below:+.3f}\pm{t_below_e:.3f}$ (expect $<0$)" + "\n"
        rf"  above ${ABOVE_POLE[0]:.0f}$–${ABOVE_POLE[1]:.0f}$: "
        rf"$A_{{\rm FB}}={t_above:+.3f}\pm{t_above_e:.3f}$ (expect $>0$)"
    ) + dilution_txt
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
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return t_below, t_below_e, t_above, t_above_e


def main(argv: list[str]) -> int:
    if len(argv) not in (6, 7):
        print(
            "usage: python analyze.py <meta.dat> <truth_kin.dat> <reco_kin.dat> "
            "<mass.png> <afb.png> [truth_gen.dat]",
            file=sys.stderr,
        )
        return 2
    meta_path, truth_path, reco_path, mass_png, afb_png = argv[1:6]
    truth_gen_path = argv[6] if len(argv) == 7 else None
    meta, _ = parse(meta_path)
    _, truth_data = parse(truth_path)
    _, reco_data = parse(reco_path)

    truth_m, truth_c = kinematics(truth_data)
    reco_m, reco_c = kinematics(reco_data)

    # Optional generator-truth file with the TRUE quark direction → dilution overlay.
    gen = None
    if truth_gen_path:
        try:
            _, gen_data = parse(truth_gen_path)
            if gen_data.size:
                gen = kinematics_gen(gen_data)
        except FileNotFoundError:
            print(f"[note] {truth_gen_path} absent — dilution overlay skipped", file=sys.stderr)

    m_lo = float(meta.get("mhat_min_GeV", "60"))
    m_hi = float(meta.get("mhat_max_GeV", "120"))
    sqrt_s_tev = float(meta.get("sqrt_s_GeV", "13000")) / 1000.0

    mass_plot(meta, truth_m, reco_m, m_lo, m_hi, mass_png)
    t_below, t_below_e, t_above, t_above_e = afb_plot(
        truth_m, truth_c, reco_m, reco_c, m_lo, m_hi, sqrt_s_tev, afb_png, gen=gen
    )

    t_mode, _, t_rms = peak_stats(truth_m, np.linspace(m_lo, m_hi, 61))
    r_mode, _, r_rms = peak_stats(reco_m, np.linspace(m_lo, m_hi, 61))
    transmission = reco_m.size / truth_m.size if truth_m.size else float("nan")
    afb_all, afb_all_e = forward_backward_asymmetry(truth_c)
    guard_ok = (t_below < 0.0) and (t_above > 0.0)

    dilution_line = ""
    if gen is not None:
        gm, gc_proxy, gc_true = gen
        u_above, u_above_e = afb_in_window(gm, gc_true, *ABOVE_POLE)
        p_above, _ = afb_in_window(gm, gc_proxy, *ABOVE_POLE)
        d = p_above / u_above if u_above else float("nan")
        dilution_line = (
            f"  DILUTION (gen truth, above pole {ABOVE_POLE}):\n"
            f"    undiluted (true quark dir): {u_above:+.4f} +/- {u_above_e:.4f}\n"
            f"    proxy (sign Qz):            {p_above:+.4f}\n"
            f"    dilution factor proxy/true = {d:.3f}  (<1 => proxy suppresses A_FB)\n"
        )

    print(
        f"saved di-muon mass spectrum        -> {mass_png}\n"
        f"saved A_FB(m) Collins-Soper plot   -> {afb_png}\n"
        f"  truth N={truth_m.size}  reco N={reco_m.size}  accept*eff^2={transmission:.3f}\n"
        f"  mass peak mode: truth={t_mode:.2f}  reco={r_mode:.2f}  (M_Z PDG={M_Z_PDG:.2f})\n"
        f"  peak RMS (80-100 GeV): truth={t_rms:.3f}  reco={r_rms:.3f}  "
        f"(reco>truth confirms momentum-resolution broadening)\n"
        f"  A_FB (truth, Collins-Soper sign(Qz) proxy):\n"
        f"    below pole {BELOW_POLE}: {t_below:+.4f} +/- {t_below_e:.4f}  (expect < 0)\n"
        f"    above pole {ABOVE_POLE}: {t_above:+.4f} +/- {t_above_e:.4f}  (expect > 0)\n"
        f"    integrated 60-120 (near-zero by below/above cancellation): "
        f"{afb_all:+.4f} +/- {afb_all_e:.4f}\n"
        f"{dilution_line}"
        f"  SIGN GUARD (A_FB<0 below pole AND >0 above): "
        f"{'PASS' if guard_ok else 'FAIL'}"
    )
    return 0 if guard_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
