#!/usr/bin/env python3
r"""Angular-coefficient analysis of the Drell-Yan sample — the Lam-Tung demo.

Runs on the HOST in the project ``.venv``. Third deliverable of the ``pp -> Z ->
mumu`` chain (alongside the Z-peak mass spectrum and ``A_FB(m)``): the **Drell-Yan
angular coefficients ``A_0(q_T)`` and ``A_2(q_T)``** in the Collins-Soper frame, and
the **Lam-Tung relation ``A_0 = A_2``** — the Drell-Yan analog of the Callan-Gross
relation, exact at O(alpha_s) and violated only at O(alpha_s^2).

**What this is (and is not).** The *closed-form* Lam-Tung gate is the always-run
symbolic O(alpha_s) proof ``A_0 - A_2 == 0`` in
``tests/analytic/test_lam_tung.py``. This script is the empirical **demonstration**
on real Pythia four-vectors — the analog of the ``A_FB`` sign guard, not a tight
tolerance. Pythia's parton shower is LL-resummed (effectively beyond fixed
O(alpha_s)) and adds hadronization, so ``A_0`` and ``A_2`` both rise with ``q_T``
from ~0 and **track each other**, with residual ``|A_0 - A_2|`` from the
higher-order / non-perturbative content. The headline is that they move together.

**Truth level only (full 4-pi acceptance).** The moment-projection inversion
``A_i = <P_i>`` is unbiased *only* over the full solid angle; detector acceptance
cuts bias it, and unfolding is out of scope. So this runs on the generator-truth
four-vectors (``truth_gen.dat``, all muons, no ``|eta|`` edge). ``A_0`` and ``A_2``
are also **immune to the pp quark-direction dilution** (both are even under the
quark flip), so the ``sign(Q_z)`` proxy gives the correct values with no true-quark
information needed.

Usage:  python analyze_angular.py <truth_gen.dat> <out.png> [meta.dat]
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from accsim.events import (
    angular_coefficients,
    collins_soper_angles,
    invariant_mass_squared,
)
from accsim.events.kinematics import _moment_weights  # projection polynomials (tested)

# Z-peak mass window: restrict to on-shell Z so the angular structure is the
# Drell-Yan one (the coefficients are ~mass-independent across the peak, and this
# keeps the interpretation clean). q_T bin edges up to a moderate reach.
Z_WINDOW = (80.0, 100.0)
QT_EDGES = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 70.0])
LOW_QT = 15.0  # "low q_T" region where Lam-Tung is expected tightest


def parse(path: str) -> tuple[dict[str, str], np.ndarray]:
    """Read a ``# key=val`` header + numeric rows -> (header, ``(N, ncol)``)."""
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


def load_fourvectors(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(p_minus, p_plus)`` from a truth_gen (9-col) or kin (8-col) array."""
    if data.size == 0:
        return np.empty((0, 4)), np.empty((0, 4))
    if data.shape[1] == 9:  # truth_gen.dat: qsign + mu- + mu+
        return data[:, 1:5], data[:, 5:9]
    return data[:, 0:4], data[:, 4:8]  # kin.dat: mu- + mu+


def coefficient_error(cos_theta: np.ndarray, phi: np.ndarray, key: str) -> float:
    """Statistical error on ``A_i`` = std(P_i)/sqrt(N) (the moment is a sample mean)."""
    poly = _moment_weights(cos_theta, phi)[key]
    n = poly.size
    return float(np.std(poly, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def ai_vs_qt(
    p_minus: np.ndarray, p_plus: np.ndarray, edges: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-``q_T``-bin ``A_0, A_2`` (+ errors) in the Z window. Returns {key:(A,err)}."""
    q = p_minus + p_plus
    m = np.sqrt(np.abs(invariant_mass_squared(q)))
    qt = np.sqrt(q[:, 1] ** 2 + q[:, 2] ** 2)
    cos_theta, phi = collins_soper_angles(p_minus, p_plus)  # proxy (A0,A2 dilution-immune)

    lo, hi = Z_WINDOW
    inwin = (m >= lo) & (m < hi)
    out = {
        k: (np.full(edges.size - 1, np.nan), np.full(edges.size - 1, np.nan)) for k in ("A0", "A2")
    }
    for i in range(edges.size - 1):
        sel = inwin & (qt >= edges[i]) & (qt < edges[i + 1])
        if np.count_nonzero(sel) < 200:  # need statistics for a stable moment
            continue
        coeffs = angular_coefficients(cos_theta[sel], phi[sel])
        for k in ("A0", "A2"):
            out[k][0][i] = coeffs[k]
            out[k][1][i] = coefficient_error(cos_theta[sel], phi[sel], k)
    return out


def plot(
    res: dict[str, tuple[np.ndarray, np.ndarray]], edges: np.ndarray, n: int, out_path: str
) -> tuple[float, float, float]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    a0, e0 = res["A0"]
    a2, e2 = res["A2"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.errorbar(
        centers,
        a0,
        yerr=e0,
        fmt="o-",
        color="C0",
        ms=5,
        lw=1.6,
        capsize=3,
        label=r"$A_0$  ($4-\langle10\cos^2\theta\rangle$)",
    )
    ax.errorbar(
        centers,
        a2,
        yerr=e2,
        fmt="s--",
        color="C1",
        ms=5,
        lw=1.6,
        capsize=3,
        label=r"$A_2$  ($\langle10\sin^2\theta\cos2\varphi\rangle$)",
    )
    ax.axvline(LOW_QT, color="0.7", ls=":", lw=1.0)

    # Lam-Tung difference in the low-q_T region (the tightest expectation).
    lowbins = centers < LOW_QT
    ok = lowbins & np.isfinite(a0) & np.isfinite(a2)
    diff = float(np.nanmean(np.abs(a0[ok] - a2[ok]))) if ok.any() else float("nan")
    differr = float(np.sqrt(np.nansum(e0[ok] ** 2 + e2[ok] ** 2)) / max(1, ok.sum()))

    ax.set_xlabel(r"$q_T$  (di-muon transverse momentum)  [GeV]")
    ax.set_ylabel(r"angular coefficient")
    ax.set_title(
        r"Lam-Tung $A_0=A_2$ in Drell-Yan  —  $pp\to\gamma^*/Z\to\mu^+\mu^-$"
        + f"\ntruth level, {Z_WINDOW[0]:.0f}$<m<${Z_WINDOW[1]:.0f} GeV, N={n}"
    )
    txt = (
        r"Lam-Tung: $A_0=A_2$ (exact at $O(\alpha_s)$)"
        + "\n"
        + rf"low $q_T<{LOW_QT:.0f}$: $\langle|A_0-A_2|\rangle={diff:.3f}\pm{differr:.3f}$"
    )
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return diff, differr, float(centers[ok][-1]) if ok.any() else float("nan")


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(
            "usage: python analyze_angular.py <truth_gen.dat> <out.png> [meta.dat]", file=sys.stderr
        )
        return 2
    data_path, out_png = argv[1], argv[2]
    _, data = parse(data_path)
    p_minus, p_plus = load_fourvectors(data)
    if p_minus.shape[0] == 0:
        print("no events", file=sys.stderr)
        return 1

    res = ai_vs_qt(p_minus, p_plus, QT_EDGES)
    diff, differr, _ = plot(res, QT_EDGES, p_minus.shape[0], out_png)

    a0, e0 = res["A0"]
    a2, e2 = res["A2"]
    print(f"saved Lam-Tung A0/A2 vs q_T plot -> {out_png}")
    print(f"  N(events)={p_minus.shape[0]}  Z window={Z_WINDOW} GeV")
    centers = 0.5 * (QT_EDGES[:-1] + QT_EDGES[1:])
    for i in range(centers.size):
        if np.isfinite(a0[i]):
            print(
                f"  q_T~{centers[i]:5.1f}: A0={a0[i]:+.3f}+/-{e0[i]:.3f}  "
                f"A2={a2[i]:+.3f}+/-{e2[i]:.3f}  |A0-A2|={abs(a0[i] - a2[i]):.3f}"
            )
    # Guard: A0 and A2 agree within ~2 sigma at low q_T (the empirical Lam-Tung check).
    guard = np.isfinite(diff) and diff < max(0.06, 2.5 * differr)
    print(f"  LOW-q_T <|A0-A2|> = {diff:.3f} +/- {differr:.3f}")
    print(f"  LAM-TUNG DEMO (A0~=A2 at low q_T): {'PASS' if guard else 'CHECK'}")
    return 0 if guard else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
