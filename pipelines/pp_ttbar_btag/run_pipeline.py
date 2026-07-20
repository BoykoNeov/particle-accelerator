#!/usr/bin/env python3
"""Orchestrate the milestone-E2 chain: Pythia8 (LHAPDF) -> HepMC3 -> Delphes -> b-tag.

End-to-end driver for the b-tagging performance study: an *established* generator
(Pythia8) with a *real proton PDF* (LHAPDF6) produces ``pp -> ttbar``, an
*established* fast detector simulation (Delphes) clusters and tags the jets, and
the analysis recovers the tagging efficiency the Delphes card **configured**.
Cross-platform (pure ``subprocess`` + Docker): runs from Windows PowerShell, Git
Bash, or Linux.

Structurally the sibling of ``../pp_mumu_drellyan/`` -- same two images, same
HepMC3 hand-off -- with a different physics target:

  1. GEN   (``hepstore/rivet-pythia``): ``lhapdf get`` a real proton PDF set, then
           compile + run ``generate_hepmc.cc`` (pp -> ttbar, inclusive decays)
           -> a HepMC3 event file, ``meta.dat``, and a generator-level heavy-quark
           dump for the independent flavour cross-check.
  2. SIM   (``scailfin/delphes-python-centos``, Delphes 3.5.0 + ROOT):
           ``DelphesHepMC3 CMS_PhaseII_0PU.tcl`` -> jets, flavour association and
           three b-tag working points; then ``extract_jets.C`` dumps every jet's
           kinematics, flavour and raw ``BTag`` bitmask.
  3. PLOT  (host ``.venv``): ``analyze.py`` reproduces the card's efficiencies.

Why the CMS_PhaseII_0PU card (not the ``delphes_card_CMS.tcl`` the Drell-Yan
chain uses): ``delphes_card_CMS.tcl`` configures a **single** ``BTagging`` module
(``BitNumber 0``), so it offers exactly one operating point and no
efficiency-vs-mistag curve can be drawn from it. CMS_PhaseII_0PU runs three --
Loose / Medium / Tight on bits 0 / 1 / 2 -- which is what makes "reproduces the
card's configured working points" a plural, falsifiable statement. The 0-pileup
variant is chosen over the 140PU one because pileup is irrelevant to this gate
and costs a great deal of simulation time.

**The card is copied back out to the host** and handed to ``analyze.py``, so the
reference formulas are parsed from the *exact file Delphes ran with*, never
transcribed. The whole ``CMS_PhaseII`` directory comes along because the card
``source``s its per-working-point formula files (``btagLoose.tcl`` &c.), and
Delphes is invoked with that directory as the working directory so those relative
includes resolve.

Bind mounts are avoided (the project path contains a space, which breaks Docker
``-v``); we ``docker cp`` sources in and data out, as in the sibling chains.

Usage:
    python run_pipeline.py [--sqrt-s 13000] [--n 20000]
                           [--card CMS_PhaseII/CMS_PhaseII_0PU]
                           [--pdf-set NNPDF31_lo_as_0118]
                           [--out-dir DIR] [--seed 20260720]
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
JET_MACRO = HERE / "extract_jets.C"
ANALYZE = HERE / "analyze.py"

GEN_IMAGE = "hepstore/rivet-pythia"
DELPHES_IMAGE = "scailfin/delphes-python-centos:3.5.0"
DELPHES_INCLUDE = "/usr/local/venv/include"  # ROOT_INCLUDE_PATH for the macro
CARDS_DIR = "/usr/local/venv/cards"


def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, **kw)  # type: ignore[arg-type]


def generate(
    cid: str,
    args: argparse.Namespace,
    meta: pathlib.Path,
    hepmc: pathlib.Path,
    partons: pathlib.Path,
) -> None:
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
            f"TT_SQRT_S={args.sqrt_s}",
            "-e",
            f"TT_N={args.n}",
            "-e",
            f"TT_SEED={args.seed}",
            "-e",
            f"TT_PDF_SET={args.pdf_set}",
            "-e",
            "TT_META=/tmp/meta.dat",
            "-e",
            "TT_HEPMC=/tmp/events.hepmc",
            "-e",
            "TT_PARTONS=/tmp/gen_partons.dat",
            cid,
            "/tmp/gen",
        ]
    )
    run(["docker", "cp", f"{cid}:/tmp/meta.dat", str(meta)])
    run(["docker", "cp", f"{cid}:/tmp/events.hepmc", str(hepmc)])
    run(["docker", "cp", f"{cid}:/tmp/gen_partons.dat", str(partons)])


def simulate(
    cid: str,
    args: argparse.Namespace,
    hepmc: pathlib.Path,
    jets: pathlib.Path,
    cards_out: pathlib.Path,
) -> pathlib.Path:
    """Stage 2 (SIM): Delphes + jet/flavour/bitmask extraction; returns the host card."""
    card_rel = f"{args.card}.tcl"
    card_abs = f"{CARDS_DIR}/{card_rel}"
    card_dir = str(pathlib.PurePosixPath(card_abs).parent)

    run(["docker", "cp", str(hepmc), f"{cid}:/tmp/events.hepmc"])
    run(["docker", "cp", str(JET_MACRO), f"{cid}:/tmp/extract_jets.C"])
    # Run from the card's own directory so its relative `source` includes resolve.
    # DelphesHepMC3 refuses to overwrite; ensure a clean out.root.
    run(
        [
            "docker",
            "exec",
            cid,
            "bash",
            "-lc",
            f"cd {card_dir} && rm -f /tmp/out.root && "
            f"DelphesHepMC3 {card_abs} /tmp/out.root /tmp/events.hepmc",
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
            'cd /tmp && root -l -b -q \'extract_jets.C("/tmp/out.root","/tmp/jets.dat")\'',
        ]
    )
    run(["docker", "cp", f"{cid}:/tmp/jets.dat", str(jets)])

    # Bring the card (and everything it sources) back to the host, so the
    # analysis parses the EXACT formulas Delphes just applied.
    cards_out.parent.mkdir(parents=True, exist_ok=True)
    run(["docker", "cp", f"{cid}:{card_dir}", str(cards_out)])
    host_card = cards_out / pathlib.PurePosixPath(card_dir).name / pathlib.Path(card_rel).name
    if not host_card.is_file():  # `docker cp` flattens when the dest exists
        host_card = cards_out / pathlib.Path(card_rel).name
    return host_card


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sqrt-s", type=float, default=13000.0, help="pp CM energy [GeV]")
    ap.add_argument("--n", type=int, default=20000, help="ttbar events to generate")
    ap.add_argument(
        "--card",
        default="CMS_PhaseII/CMS_PhaseII_0PU",
        help="Delphes card path under the image's cards/ dir, without .tcl. Must "
        "define at least one `module BTagging`; the default defines three "
        "(Loose/Medium/Tight) which is what gives an efficiency-vs-mistag curve",
    )
    ap.add_argument(
        "--pdf-set",
        default="NNPDF31_lo_as_0118",
        help="LHAPDF6 proton set (LO, to match Pythia's LO ME); member 0 is used",
    )
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--gen-image", default=GEN_IMAGE)
    ap.add_argument("--delphes-image", default=DELPHES_IMAGE)
    ap.add_argument(
        "--out-dir",
        default=str(pathlib.Path(tempfile.gettempdir()) / "e2_ttbar_btag"),
        help="host directory for the .dat/.hepmc/.png artifacts",
    )
    args = ap.parse_args(argv)

    # Optional addon (project rule: everything past the pure-Python baseline is an
    # opt-in runtime switch, default OFF). Running the script is the opt-in; gate
    # on the same registry the in-package API reads. BOTH switches are required
    # because this chain genuinely uses both tools -- the PDF and the detector.
    from accsim import features

    for addon in ("lhapdf", "delphes"):
        try:
            features.require(addon)
        except features.AddonDisabledError as exc:
            print(f"[gated addon] {exc}", file=sys.stderr)
            return 2

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = out_dir / "meta.dat"
    hepmc = out_dir / "events.hepmc"
    partons = out_dir / "gen_partons.dat"
    jets = out_dir / "jets.dat"
    cards_out = out_dir / "cards"
    png = out_dir / "ttbar_btag_performance.png"

    # Stage 1: generation container.
    cid = run(
        ["docker", "run", "-d", args.gen_image, "sleep", "infinity"],
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(f"[GEN] container {cid[:12]}")
    try:
        generate(cid, args, meta, hepmc, partons)
    finally:
        subprocess.run(["docker", "rm", "-f", cid], stdout=subprocess.DEVNULL)

    # Stage 2: Delphes container.
    cid = run(
        ["docker", "run", "-d", args.delphes_image, "sleep", "infinity"],
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(f"[SIM] container {cid[:12]}")
    try:
        host_card = simulate(cid, args, hepmc, jets, cards_out)
    finally:
        subprocess.run(["docker", "rm", "-f", cid], stdout=subprocess.DEVNULL)

    # Stage 3: analysis on the host, against the card Delphes actually ran.
    proc = subprocess.run(
        [
            sys.executable,
            str(ANALYZE),
            str(jets),
            str(partons),
            str(host_card),
            str(png),
            str(meta),
        ],
        check=False,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
