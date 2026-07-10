#!/usr/bin/env python3
"""Orchestrate the Phase 2 *hadronic* chain: Pythia8 (LHAPDF) -> HepMC3 -> Delphes.

End-to-end driver for the hadronic Drell-Yan extension of Phase 2: an *established*
generator (Pythia8) with a *real proton PDF* (LHAPDF6) feeds an *established* fast
detector simulation (Delphes, CMS card), honouring the roadmap's "orchestrate,
don't rebuild" rule. Cross-platform (pure ``subprocess`` + Docker): runs from
Windows PowerShell, Git Bash, or Linux.

This is the hadronic analogue of ``../ee_mumu_delphes/`` (leptonic). The two
established tools live in two images, decoupled via a HepMC3 file (the standard
generator->detector interchange), because no single trustworthy image ships both:

  1. GEN   (``hepstore/rivet-pythia``): ``lhapdf get`` a real proton PDF set, then
           compile + run ``generate_hepmc.cc`` (pp -> gamma*/Z -> mu+mu- Drell-Yan,
           forced Z->mumu, 60<m<120 GeV) -> a HepMC3 event file + ``meta.dat``.
  2. SIM   (``scailfin/delphes-python-centos``, Delphes 3.5.0 + ROOT):
           ``DelphesHepMC3 delphes_card_CMS.tcl out.root events.hepmc`` -> detector
           response; then ``extract_mass.C`` dumps BOTH truth (generator "Particle")
           and reco ("Muon") di-muon invariant mass for the *same* events.
  3. PLOT  (host ``.venv``): ``analyze.py`` overlays the truth vs reco Z-peak.

Why sqrt(s) = 13 TeV + the CMS card (not the leptonic chain's 250 GeV / ILD): this
is a *hadron* collider process, so it needs a proton PDF and a hadron-collider
detector card. The CMS card is Delphes' canonical LHC card (no pile-up variant,
to keep the Muon branch clean).

Why an LO PDF set (default ``NNPDF31_lo_as_0118``): Pythia's Drell-Yan matrix
element is leading-order, so an LO PDF is the consistent partner. The set is
downloaded at run time (``lhapdf get``) because the image ships LHAPDF *without*
grids; a network failure there is reported cleanly rather than as a Pythia crash.

Bind mounts are avoided (the project path contains a space, which breaks Docker
``-v``); we ``docker cp`` sources in and data out, as in the leptonic chains.

Usage:
    python run_pipeline.py [--sqrt-s 13000] [--n 20000] [--card CMS]
                           [--pdf-set NNPDF31_lo_as_0118] [--m-min 60] [--m-max 120]
                           [--out-dir DIR] [--seed 20260710]
                           [--gen-image ...] [--delphes-image ...]
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
GEN_SRC = HERE / "generate_hepmc.cc"
MASS_MACRO = HERE / "extract_mass.C"
ANALYZE = HERE / "analyze.py"

GEN_IMAGE = "hepstore/rivet-pythia"
DELPHES_IMAGE = "scailfin/delphes-python-centos:3.5.0"
DELPHES_INCLUDE = "/usr/local/venv/include"  # ROOT_INCLUDE_PATH for the macro
CARDS_DIR = "/usr/local/venv/cards"


def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, **kw)  # type: ignore[arg-type]


def generate(cid: str, args: argparse.Namespace, meta: pathlib.Path, hepmc: pathlib.Path) -> None:
    """Stage 1 (GEN): fetch the PDF, compile + run the Pythia generator in ``cid``."""
    run(["docker", "cp", str(GEN_SRC), f"{cid}:/tmp/generate_hepmc.cc"])
    # Download the LHAPDF grid first, with a clean error if the network is down --
    # otherwise Pythia's init would crash cryptically on the missing set.
    run(
        [
            "docker",
            "exec",
            cid,
            "bash",
            "-lc",
            f"lhapdf get {args.pdf_set} || {{ echo "
            f"'ERROR: could not download LHAPDF set {args.pdf_set} "
            f"(needs network in the container)'; exit 3; }}",
        ]
    )
    run(
        [
            "docker",
            "exec",
            cid,
            "bash",
            "-lc",
            "g++ /tmp/generate_hepmc.cc -o /tmp/gen $(pythia8-config --cxxflags --libs) -lHepMC3",
        ]
    )
    run(
        [
            "docker",
            "exec",
            "-e",
            f"DY_SQRT_S={args.sqrt_s}",
            "-e",
            f"DY_N={args.n}",
            "-e",
            f"DY_SEED={args.seed}",
            "-e",
            f"DY_MHAT_MIN={args.m_min}",
            "-e",
            f"DY_MHAT_MAX={args.m_max}",
            "-e",
            f"DY_PDF_SET={args.pdf_set}",
            "-e",
            "DY_META=/tmp/meta.dat",
            "-e",
            "DY_HEPMC=/tmp/events.hepmc",
            cid,
            "/tmp/gen",
        ]
    )
    run(["docker", "cp", f"{cid}:/tmp/meta.dat", str(meta)])
    run(["docker", "cp", f"{cid}:/tmp/events.hepmc", str(hepmc)])


def simulate(
    cid: str,
    args: argparse.Namespace,
    hepmc: pathlib.Path,
    truth: pathlib.Path,
    reco: pathlib.Path,
) -> None:
    """Stage 2 (SIM): Delphes detector sim + truth/reco mass extraction in ``cid``."""
    card = f"{CARDS_DIR}/delphes_card_{args.card}.tcl"
    run(["docker", "cp", str(hepmc), f"{cid}:/tmp/events.hepmc"])
    run(["docker", "cp", str(MASS_MACRO), f"{cid}:/tmp/extract_mass.C"])
    # DelphesHepMC3 refuses to overwrite; ensure a clean out.root.
    run(
        [
            "docker",
            "exec",
            cid,
            "bash",
            "-lc",
            f"cd /tmp && rm -f out.root && DelphesHepMC3 {card} /tmp/out.root /tmp/events.hepmc",
        ]
    )
    run(
        [
            "docker",
            "exec",
            "-e",
            f"ROOT_INCLUDE_PATH={DELPHES_INCLUDE}",
            cid,
            "bash",
            "-lc",
            "cd /tmp && root -l -b -q "
            '\'extract_mass.C("/tmp/out.root","/tmp/truth.dat","/tmp/reco.dat")\'',
        ]
    )
    run(["docker", "cp", f"{cid}:/tmp/truth.dat", str(truth)])
    run(["docker", "cp", f"{cid}:/tmp/reco.dat", str(reco)])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sqrt-s", type=float, default=13000.0, help="pp CM energy [GeV]")
    ap.add_argument("--n", type=int, default=20000, help="events to generate")
    ap.add_argument("--card", default="CMS", help="Delphes hadron card (CMS, ATLAS, ...)")
    ap.add_argument(
        "--pdf-set",
        default="NNPDF31_lo_as_0118",
        help="LHAPDF6 proton set (LO, to match Pythia's LO ME); member 0 is used",
    )
    ap.add_argument("--m-min", type=float, default=60.0, help="min m_hat [GeV] (Z window)")
    ap.add_argument("--m-max", type=float, default=120.0, help="max m_hat [GeV] (Z window)")
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--gen-image", default=GEN_IMAGE)
    ap.add_argument("--delphes-image", default=DELPHES_IMAGE)
    ap.add_argument(
        "--out-dir",
        default=str(pathlib.Path(tempfile.gettempdir()) / "phase2_drellyan"),
        help="host directory for the .dat/.hepmc/.png artifacts",
    )
    args = ap.parse_args(argv)

    if args.m_min >= args.m_max:
        raise SystemExit(f"--m-min {args.m_min} must be < --m-max {args.m_max}")

    # Optional addon (project rule: everything past the pure-Python baseline is an
    # opt-in runtime switch, default OFF). Running the script is the opt-in; gate on
    # the same registry the in-package API reads -- ACCSIM_ENABLE_LHAPDF=1.
    from accsim import features

    try:
        features.require("lhapdf")
    except features.AddonDisabledError as exc:
        print(f"[gated addon] {exc}", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = out_dir / "meta.dat"
    hepmc = out_dir / "events.hepmc"
    truth = out_dir / "truth_mass.dat"
    reco = out_dir / "reco_mass.dat"
    png = out_dir / "drellyan_truth_vs_reco.png"

    # Stage 1: generation container.
    cid = run(
        ["docker", "run", "-d", args.gen_image, "sleep", "infinity"],
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(f"[GEN] container {cid[:12]}")
    try:
        generate(cid, args, meta, hepmc)
    finally:
        subprocess.run(["docker", "rm", "-f", cid], stdout=subprocess.DEVNULL)

    # Stage 2: Delphes container (override entrypoint so it idles).
    cid = run(
        [
            "docker",
            "run",
            "-d",
            "--entrypoint",
            "/bin/bash",
            args.delphes_image,
            "-c",
            "sleep infinity",
        ],
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(f"[SIM] container {cid[:12]}")
    try:
        simulate(cid, args, hepmc, truth, reco)
    finally:
        subprocess.run(["docker", "rm", "-f", cid], stdout=subprocess.DEVNULL)

    # Stage 3: host-side analysis.
    run([sys.executable, str(ANALYZE), str(meta), str(truth), str(reco), str(png)])
    print(
        f"\nPhase 2 (hadronic Drell-Yan chain) done:\n"
        f"  meta:  {meta}\n  truth: {truth}\n  reco:  {reco}\n  plot:  {png}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
