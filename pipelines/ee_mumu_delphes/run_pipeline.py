#!/usr/bin/env python3
"""Orchestrate the Phase 2 *detector* chain: Pythia8 -> HepMC3 -> Delphes -> host.

End-to-end driver for the detector extension of Phase 2: an *established*
generator (Pythia8) feeds an *established* fast detector simulation (Delphes),
honouring the roadmap's "orchestrate, don't rebuild" rule. Cross-platform (pure
``subprocess`` + Docker), so it runs from Windows PowerShell, Git Bash, or Linux.

Two established tools live in two images (decoupled via a HepMC3 file, the standard
generator->detector interchange format), because no single trustworthy image ships
both and coupling them (``DelphesPythia8``) would need Delphes compiled against this
Pythia:

  1. GEN   (``hepstore/rivet-pythia``): compile ``generate_hepmc.cc``, run it ->
           a HepMC3 event file + a ``meta.dat`` (Pythia's cross-section + params).
  2. SIM   (``scailfin/delphes-python-centos``, IRIS-HEP; Delphes 3.5.0 + ROOT):
           ``DelphesHepMC3 <card> out.root events.hepmc`` -> detector response;
           then a ROOT macro (``extract_reco.C``) dumps BOTH the truth and reco
           ``cos(theta_mu-)`` (generator "Particle" + reco "Muon" branch, same events).
  3. PLOT  (host ``.venv``): ``analyze.py`` overlays truth vs reco + the ILD edge.

Why sqrt(s) = 250 GeV (not the clause-(b) chain's 10 GeV): the standard Delphes
e+e- cards (ILD/IDEA/CLIC) are parametrized for >= 91 GeV, so a physically
meaningful detector response needs an ILC-scale energy. See the README.

Bind mounts are deliberately avoided (the project path contains a space, which
breaks Docker ``-v``); we ``docker cp`` sources in and data out, as in the
clause-(b) pipeline.

Usage:
    python run_pipeline.py [--sqrt-s 250.0] [--n 20000] [--card ILD]
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
RECO_MACRO = HERE / "extract_reco.C"
ANALYZE = HERE / "analyze.py"

GEN_IMAGE = "hepstore/rivet-pythia"
DELPHES_IMAGE = "scailfin/delphes-python-centos:3.5.0"
DELPHES_INCLUDE = "/usr/local/venv/include"  # ROOT_INCLUDE_PATH for the macro
CARDS_DIR = "/usr/local/venv/cards"


def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, **kw)  # type: ignore[arg-type]


def generate(cid: str, args: argparse.Namespace, meta: pathlib.Path, hepmc: pathlib.Path) -> None:
    """Stage 1 (GEN): compile + run the Pythia generator inside ``cid``."""
    run(["docker", "cp", str(GEN_SRC), f"{cid}:/tmp/generate_hepmc.cc"])
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
            f"EEMUMU_SQRT_S={args.sqrt_s}",
            "-e",
            f"EEMUMU_N={args.n}",
            "-e",
            f"EEMUMU_SEED={args.seed}",
            "-e",
            "EEMUMU_META=/tmp/meta.dat",
            "-e",
            "EEMUMU_HEPMC=/tmp/events.hepmc",
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
    """Stage 2 (SIM): Delphes detector sim + truth/reco extraction inside ``cid``."""
    card = f"{CARDS_DIR}/delphes_card_{args.card}.tcl"
    run(["docker", "cp", str(hepmc), f"{cid}:/tmp/events.hepmc"])
    run(["docker", "cp", str(RECO_MACRO), f"{cid}:/tmp/extract_reco.C"])
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
            f'\'extract_reco.C("/tmp/out.root","/tmp/truth.dat","/tmp/reco.dat",{args.p_min})\'',
        ]
    )
    run(["docker", "cp", f"{cid}:/tmp/truth.dat", str(truth)])
    run(["docker", "cp", f"{cid}:/tmp/reco.dat", str(reco)])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sqrt-s", type=float, default=250.0, help="CM energy [GeV]")
    ap.add_argument("--n", type=int, default=20000, help="events to generate")
    ap.add_argument("--card", default="ILD", help="Delphes e+e- card (ILD, IDEA, ...)")
    ap.add_argument(
        "--p-min",
        type=float,
        default=None,
        help="signal |p| cut [GeV] isolating the ~sqrt(s)/2 mu- (valley of the "
        "bimodal spectrum; angle-neutral, so it does not bias cos theta). Default "
        "0.8*sqrt(s)/2, which scales with energy so off-250-GeV runs still cut in "
        "the valley rather than above the signal.",
    )
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--gen-image", default=GEN_IMAGE)
    ap.add_argument("--delphes-image", default=DELPHES_IMAGE)
    ap.add_argument(
        "--out-dir",
        default=str(pathlib.Path(tempfile.gettempdir()) / "phase2_delphes"),
        help="host directory for the .dat/.hepmc/.png artifacts",
    )
    args = ap.parse_args(argv)

    # The signal mu- is monochromatic at |p| = sqrt(s)/2. Scale the cut with energy
    # (default 0.8*that, in the spectrum's valley) so a run at another energy does
    # not silently cut *above* the signal and produce an empty plot. Guard the
    # explicit case too: a cut at/above sqrt(s)/2 removes all signal.
    p_signal = args.sqrt_s / 2.0
    if args.p_min is None:
        args.p_min = 0.8 * p_signal
    elif args.p_min >= p_signal:
        raise SystemExit(
            f"--p-min {args.p_min} >= signal |p| = sqrt(s)/2 = {p_signal}: "
            f"this cut removes all signal muons (empty distribution)."
        )

    # Optional addon (project rule: everything past the pure-Python toy baseline is
    # an opt-in runtime switch, default OFF). Running the script is the opt-in; gate
    # on the same registry the in-package API reads -- ACCSIM_ENABLE_DELPHES=1.
    from accsim import features

    try:
        features.require("delphes")
    except features.AddonDisabledError as exc:
        print(f"[gated addon] {exc}", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = out_dir / "meta.dat"
    hepmc = out_dir / "events.hepmc"
    truth = out_dir / "truth_costheta.dat"
    reco = out_dir / "reco_costheta.dat"
    png = out_dir / "eemumu_truth_vs_reco.png"

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
        f"\nPhase 2 (detector chain) done:\n"
        f"  meta:  {meta}\n  truth: {truth}\n  reco:  {reco}\n  plot:  {png}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
