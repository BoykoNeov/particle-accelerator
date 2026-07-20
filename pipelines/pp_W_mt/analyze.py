#!/usr/bin/env python3
r"""Analysis stage for E1 — runs on the HOST in the project ``.venv``.

From the same Pythia ``pp -> W -> mu nu`` sample, shown **truth (generator) vs reco
(after Delphes CMS)**: the **transverse-mass spectrum** and its **Jacobian edge at
``M_W``**.

All physics comes from the single *tested* implementations
``accsim.events.transverse_mass`` and ``accsim.events.jacobian_edge`` (analytic
gates: ``tests/analytic/test_transverse_mass.py``,
``tests/analytic/test_jacobian_edge.py``). The container macro only extracts raw
vectors, so no ``(1 - cos dphi)`` is ever duplicated in untested C++.

WHAT IS GATED, AND WHY IT IS NOT ``m_T <= M_W``
----------------------------------------------
The analytic suite proves ``m_T <= M`` for a **fixed** parent mass. Pythia gives the
``W`` a **Breit-Wigner** mass, so off-shell events with ``m(mu nu) > M_W`` are real
physics and genuinely produce ``m_T > M_W``. A "truth ``max(m_T) <= M_W``" assertion
would therefore either fail on correct physics, or pass only because a generation
mass window had been imposed near the edge — hiding the very effect being measured.
So the gate is on the **edge location**, never a hard cutoff:

1. **Truth edge near ``M_W``** — the sharp gate. ``Gamma_W ~ 2.09 GeV`` smears the
   cliff before any detector effect, and the half-maximum estimator carries a known
   ``~ +1 GeV + 0.73 sigma`` bias (measured, tabulated in ``jacobian_edge``'s
   docstring), so the tolerance is a few GeV and is justified by those two numbers
   rather than tuned until it passed.
2. **The reco edge is measurably rounder than truth** — the MET-resolution seam.
   Reco MET resolution is ``O(15 GeV)``, so a *tight* reco edge gate is impossible
   and a loose one would be vacuous; the falloff width is what carries the teeth.
3. **``m_T`` edges at ``M_W`` while ``p_T^mu`` edges at ``M_W/2``** — the pipeline
   -level statement of the classic trap. If the extraction had leaked ``p_T^mu`` in
   place of ``m_T``, gate 1 would land near 40 GeV and fail.
4. **The ``GenMissingET`` direction convention, pinned from the data** — see below.

THE ``GenMissingET`` SIGN
-------------------------
Delphes' ``Merger`` negates its vector sum to build a "missing" momentum, but
``GenMissingET``'s input is the **neutrino** list itself, not the visible particles.
So the output may point *along* the neutrino or *opposite* to it, and that choice
shifts ``dphi`` by ``pi`` — flipping ``(1 - cos dphi)`` between ``~0`` and ``~2``
and wrecking ``m_T``. The macro emits both ``GenMissingET`` and the directly summed
truth neutrino; this script measures the angle between them and **pins the
convention empirically**, refusing to proceed if it matches neither ``0`` nor ``pi``.

Usage:  python analyze.py <meta.dat> <truth_kin.dat> <reco_kin.dat> <mt.png>
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from accsim.events import jacobian_edge, transverse_mass

# Histogram/gate settings. The window is deliberately wide enough to contain the
# reco edge (which sits well above M_W once MET resolution rounds it) without
# letting the low-m_T rise become the histogram peak.
MT_WINDOW = (40.0, 140.0)
MT_BINS = 60
# Tolerance on the truth edge: the estimator's measured bias at Gamma_W-scale
# smearing is ~ +1.5 GeV, binning adds ~0.3 GeV, and ISR recoil contributes a
# further ~1 GeV. 5 GeV is comfortably above that sum and far below the ~40 GeV
# that would be needed to accommodate a p_T^mu-for-m_T mix-up (gate 3).
TRUTH_EDGE_TOL = 5.0
# The reco edge must be at least this much rounder than truth (in falloff width).
MIN_ROUNDING = 1.5
# Loose sanity band on the reco edge POSITION. MET resolution rounds and shifts the
# reco edge, so a tight gate here is impossible — but a band this wide is still not
# vacuous: a flipped reco MET sign puts the edge near 45 GeV (measured), i.e. ~35
# GeV off, which fails. This exists because gates 1-3 all key off the TRUTH sample,
# leaving a flipped reco MET otherwise uncaught.
RECO_EDGE_TOL = 20.0


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
            rows.append([float(t) for t in line.split()])
    return header, np.array(rows, dtype=np.float64) if rows else np.empty((0, 9))


def mt_and_ptmu(data: np.ndarray, met_sign: float) -> tuple[np.ndarray, np.ndarray]:
    """``(m_T, p_T^mu)`` from a kinematics table.

    ``met_sign`` is ``+1`` if the stored missing-momentum vector points along the
    neutrino and ``-1`` if it points opposite; it is determined empirically by
    :func:`pin_genmet_sign`, never assumed.
    """
    # Column 0 (E) and column 3 (p_z) are deliberately unused: m_T is a purely
    # transverse observable and the missing-momentum vector has no p_z at all.
    px, py = data[:, 1], data[:, 2]
    met_pt, met_phi = data[:, 4], data[:, 5]
    pt_mu = np.hypot(px, py)
    phi_mu = np.arctan2(py, px)
    # a sign flip on the vector is a pi shift in phi
    phi_met = met_phi if met_sign > 0 else met_phi + np.pi
    return transverse_mass(pt_mu, phi_mu, met_pt, phi_met), pt_mu


def pin_genmet_sign(truth: np.ndarray) -> float:
    """Determine the ``GenMissingET`` direction convention from the data itself.

    Compares ``GenMissingET``'s azimuth against the directly summed truth-neutrino
    azimuth over events where both are well defined. Returns ``+1`` (points along
    the neutrino) or ``-1`` (points opposite). Raises if the two agree with
    neither convention, which would mean the branch is not what it is taken to be.
    """
    met_pt, met_phi, nu_pt, nu_phi = (
        truth[:, 4],
        truth[:, 5],
        truth[:, 6],
        truth[:, 7],
    )
    ok = (met_pt > 5.0) & (nu_pt > 5.0)
    if np.count_nonzero(ok) < 50:
        raise SystemExit("cannot pin the GenMissingET sign: too few usable events")
    dphi = np.abs(np.arctan2(np.sin(met_phi[ok] - nu_phi[ok]), np.cos(met_phi[ok] - nu_phi[ok])))
    median = float(np.median(dphi))
    frac_aligned = float(np.mean(dphi < np.pi / 4))
    frac_opposed = float(np.mean(dphi > 3 * np.pi / 4))
    print(
        f"GenMissingET vs summed neutrino: median |dphi| = {median:.4f} rad, "
        f"aligned {frac_aligned:.1%}, opposed {frac_opposed:.1%}"
    )
    if frac_aligned > 0.9:
        print("  -> pinned: GenMissingET points ALONG the neutrino (sign = +1)")
        return +1.0
    if frac_opposed > 0.9:
        print("  -> pinned: GenMissingET points OPPOSITE the neutrino (sign = -1)")
        return -1.0
    raise SystemExit(
        "GenMissingET matches neither convention "
        f"(aligned {frac_aligned:.1%}, opposed {frac_opposed:.1%}) — refusing to guess"
    )


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    meta_path, truth_path, reco_path, plot_path = sys.argv[1:5]

    meta, _ = parse(meta_path)
    truth_hdr, truth = parse(truth_path)
    reco_hdr, reco = parse(reco_path)
    if truth.size == 0 or reco.size == 0:
        print("empty truth or reco table — nothing to analyse")
        return 1

    # The generator's OWN W mass/width, not a remembered PDG number: the gate must
    # compare against what Pythia actually ran with.
    m_w = float(meta.get("m_w_gev", "nan"))
    width_w = float(meta.get("width_w_gev", "nan"))
    if not np.isfinite(m_w):
        print("meta.dat carries no m_w_gev — cannot gate without the generator's mass")
        return 1
    print(f"generator: M_W = {m_w:.4f} GeV, Gamma_W = {width_w:.4f} GeV")

    # --- pin the missing-momentum direction convention before using it ---------
    met_sign = pin_genmet_sign(truth)
    # Reco MissingET is a genuine "missing" vector built from VISIBLE particles, so
    # it points along the neutrino by construction; the pinned sign applies to the
    # GenMissingET branch only.
    mt_truth, pt_truth = mt_and_ptmu(truth, met_sign)
    mt_reco, pt_reco = mt_and_ptmu(reco, +1.0)

    # --- the gates -------------------------------------------------------------
    edge_t, width_t = jacobian_edge(mt_truth, bins=MT_BINS, window=MT_WINDOW)
    edge_r, width_r = jacobian_edge(mt_reco, bins=MT_BINS, window=MT_WINDOW)
    # p_T^mu edge, on the SAME events — the M_W/2 statement
    edge_pt, _ = jacobian_edge(pt_truth, bins=MT_BINS, window=(15.0, 80.0))

    print(
        f"\ntruth: N = {mt_truth.size}, edge = {edge_t:.2f} GeV "
        f"(falloff {width_t:.2f}), median m_T = {np.median(mt_truth):.2f}"
    )
    print(
        f"reco : N = {mt_reco.size}, edge = {edge_r:.2f} GeV "
        f"(falloff {width_r:.2f}), median m_T = {np.median(mt_reco):.2f}"
    )
    print(f"truth p_T^mu edge = {edge_pt:.2f} GeV (expect ~M_W/2 = {m_w / 2:.2f})")
    frac_above = float(np.mean(mt_truth > m_w))
    print(
        f"truth events with m_T > M_W: {frac_above:.2%} "
        "(nonzero by construction — the Breit-Wigner tail, not a bug)"
    )

    gate1 = abs(edge_t - m_w) < TRUTH_EDGE_TOL
    gate2 = (width_r - width_t) > MIN_ROUNDING
    gate3 = abs(edge_pt - m_w / 2.0) < TRUTH_EDGE_TOL
    gate4 = abs(edge_r - m_w) < RECO_EDGE_TOL
    print(
        f"\ngate 1 truth edge at M_W        : |{edge_t:.2f} - {m_w:.2f}| = "
        f"{abs(edge_t - m_w):.2f} < {TRUTH_EDGE_TOL}  {'PASS' if gate1 else 'FAIL'}"
    )
    print(
        f"gate 2 reco edge rounder        : {width_r:.2f} - {width_t:.2f} = "
        f"{width_r - width_t:.2f} > {MIN_ROUNDING}  {'PASS' if gate2 else 'FAIL'}"
    )
    print(
        f"gate 3 p_T^mu edge at M_W/2     : |{edge_pt:.2f} - {m_w / 2:.2f}| = "
        f"{abs(edge_pt - m_w / 2):.2f} < {TRUTH_EDGE_TOL}  {'PASS' if gate3 else 'FAIL'}"
    )
    print(
        f"gate 4 reco edge sane (MET sign): |{edge_r:.2f} - {m_w:.2f}| = "
        f"{abs(edge_r - m_w):.2f} < {RECO_EDGE_TOL}  {'PASS' if gate4 else 'FAIL'}"
    )

    # --- plot ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    bins = np.linspace(*MT_WINDOW, MT_BINS + 1)
    ax.hist(
        mt_truth,
        bins=bins,
        histtype="step",
        density=True,
        lw=2.0,
        color="#1f77b4",
        label=f"truth (gen), N={mt_truth.size}",
    )
    ax.hist(
        mt_reco,
        bins=bins,
        histtype="step",
        density=True,
        lw=2.0,
        color="#d62728",
        label=f"reco (Delphes CMS MET), N={mt_reco.size}",
    )
    ax.axvline(m_w, color="k", ls="--", lw=1.2, label=f"$M_W$ = {m_w:.2f} GeV")
    ax.axvline(edge_t, color="#1f77b4", ls=":", lw=1.4, label=f"truth edge {edge_t:.1f}")
    ax.axvline(edge_r, color="#d62728", ls=":", lw=1.4, label=f"reco edge {edge_r:.1f}")
    ax.set_xlabel(r"$m_T(\mu,\ p_T^{\rm miss})$  [GeV]")
    ax.set_ylabel("normalised events / bin")
    ax.set_title(
        r"E1: $pp \to W \to \mu\nu$ transverse mass — the Jacobian edge at $M_W$"
        f"\n{meta.get('pdf_set', '?')}, $\\sqrt{{s}}$ = "
        f"{float(meta.get('sqrt_s_GeV', 0)) / 1000:.0f} TeV"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    print(f"\nwrote {plot_path}")

    all_ok = gate1 and gate2 and gate3 and gate4
    print(f"\nE1 gates: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
