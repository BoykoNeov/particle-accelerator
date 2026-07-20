#!/usr/bin/env python3
"""Orchestrate the E1 chain: Pythia8 (LHAPDF) ``pp -> W -> mu nu`` -> HepMC3 -> Delphes.

End-to-end driver for milestone E1, the **W-mass Jacobian edge**. It is the
charged-current sibling of ``../pp_mumu_drellyan/`` (neutral current) and reuses
that chain's orchestration wholesale: an *established* generator (Pythia8) with a
*real proton PDF* (LHAPDF6) feeding an *established* fast detector simulation
(Delphes, CMS card), honouring the roadmap's "orchestrate, don't rebuild" rule.
Cross-platform (pure ``subprocess`` + Docker).

  1. GEN   (``hepstore/rivet-pythia``): ``lhapdf get`` a real proton PDF set, then
           compile + run ``generate_hepmc.cc`` (pp -> W -> mu nu, both charges,
           **no mass window**) -> a HepMC3 event file + ``meta.dat`` carrying the
           generator's OWN ``M_W``/``Gamma_W``.
  2. SIM   (``scailfin/delphes-python-centos``, Delphes 3.5.0 + ROOT):
           ``DelphesHepMC3 delphes_card_CMS.tcl`` -> detector response; then
           ``extract_kinematics.C`` dumps the muon four-vector + the missing
           transverse momentum for BOTH truth (``GenMissingET``) and reco
           (``MissingET``) of the *same* events.
  3. PLOT  (host ``.venv``): ``analyze.py`` computes ``m_T`` with the tested
           ``accsim.events.transverse_mass``, locates the Jacobian edge, and
           enforces the E1 gates.

WHY NO MASS WINDOW (the one real departure from the DY chain). The Z chain set
``PhaseSpace:mHatMin/Max = 60..120`` to dodge the divergent low-mass photon pole.
The charged current has no photon-exchange piece, so there is no pole to dodge --
and a window would be actively harmful here, imposing a hard cutoff right where the
Jacobian edge lives and manufacturing an artificial one. See ``generate_hepmc.cc``.

Bind mounts are avoided (the project path contains a space, which breaks Docker
``-v``); we ``docker cp`` sources in and data out, as in the other chains.

Usage:
    python run_pipeline.py [--sqrt-s 13000] [--n 20000] [--card CMS]
                           [--pdf-set NNPDF31_lo_as_0118] [--out-dir DIR]
                           [--seed 20260720] [--gen-image ...] [--delphes-image ...]
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
GEN_SRC = HERE / "generate_hepmc.cc"
KIN_MACRO = HERE / "extract_kinematics.C"
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
            f"W_SQRT_S={args.sqrt_s}",
            "-e",
            f"W_N={args.n}",
            "-e",
            f"W_SEED={args.seed}",
            "-e",
            f"W_PDF_SET={args.pdf_set}",
            "-e",
            "W_META=/tmp/meta.dat",
            "-e",
            "W_HEPMC=/tmp/events.hepmc",
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
    """Stage 2 (SIM): Delphes detector sim + truth/reco kinematics extraction in ``cid``."""
    card = f"{CARDS_DIR}/delphes_card_{args.card}.tcl"
    run(["docker", "cp", str(hepmc), f"{cid}:/tmp/events.hepmc"])
    run(["docker", "cp", str(KIN_MACRO), f"{cid}:/tmp/extract_kinematics.C"])
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
            '\'extract_kinematics.C("/tmp/out.root","/tmp/truth_kin.dat",'
            '"/tmp/reco_kin.dat")\'',
        ]
    )
    run(["docker", "cp", f"{cid}:/tmp/truth_kin.dat", str(truth)])
    run(["docker", "cp", f"{cid}:/tmp/reco_kin.dat", str(reco)])


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
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--gen-image", default=GEN_IMAGE)
    ap.add_argument("--delphes-image", default=DELPHES_IMAGE)
    ap.add_argument(
        "--out-dir",
        default=str(pathlib.Path(tempfile.gettempdir()) / "e1_w_mt"),
        help="host directory for the .dat/.hepmc/.png artifacts",
    )
    args = ap.parse_args(argv)

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
    truth = out_dir / "truth_kin.dat"
    reco = out_dir / "reco_kin.dat"
    mt_png = out_dir / "w_transverse_mass.png"

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

    # Stage 3: host-side analysis (m_T spectrum + the Jacobian-edge gates).
    proc = subprocess.run(
        [sys.executable, str(ANALYZE), str(meta), str(truth), str(reco), str(mt_png)],
        text=True,
    )
    print(
        f"\nE1 (W transverse mass) done:\n"
        f"  meta:  {meta}\n  truth: {truth}\n  reco:  {reco}\n  m_T plot: {mt_png}"
    )
    # analyze.py returns non-zero if any E1 gate fails — surface it.
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
