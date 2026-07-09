#!/usr/bin/env python3
"""Orchestrate the real Phase 2 chain: Pythia8 (in Docker) -> host analysis.

End-to-end driver for acceptance clause (b) — "the orchestrated pipeline runs
end-to-end and produces a labelled distribution" — using an *established*
generator (Pythia8), not the from-scratch toy. Cross-platform (pure ``subprocess``
+ Docker), so it runs from Windows PowerShell, Git Bash, or Linux alike.

Why Docker (see the pipeline README): Pythia8/Delphes do not build natively on the
Windows/Python 3.14 host, and there is no ``pythia8`` pip/conda-Windows package;
the ``hepstore/rivet-pythia`` image ships Pythia8 8.3 prebuilt. We deliberately
**avoid a bind mount** (the project path contains a space, which breaks Docker
``-v``) and instead ``docker cp`` the source in and the data out.

Stages:
  1. start a throwaway container from the image
  2. copy ``generate_pythia.cc`` in, compile it with ``pythia8-config`` flags
  3. run it -> ``cos(theta)`` data for the mu+ mu- subset
  4. copy the data out, remove the container
  5. run ``analyze.py`` on the host (project ``.venv``) -> labelled PNG

Usage:
    python run_pipeline.py [--sqrt-s 10.0] [--n 40000] [--out-dir DIR]
                           [--image hepstore/rivet-pythia] [--seed 20260710]
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "generate_pythia.cc"
ANALYZE = HERE / "analyze.py"


def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, **kw)  # type: ignore[arg-type]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sqrt-s", type=float, default=10.0, help="CM energy [GeV]")
    ap.add_argument("--n", type=int, default=40000, help="events to generate")
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--image", default="hepstore/rivet-pythia")
    ap.add_argument(
        "--out-dir",
        default=str(pathlib.Path(tempfile.gettempdir()) / "phase2_pythia"),
        help="host directory for the .dat and .png artifacts",
    )
    args = ap.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dat = out_dir / "eemumu_costheta.dat"
    png = out_dir / "eemumu_costheta.png"

    # 1) throwaway container (idle; we exec into it).
    cid = run(
        ["docker", "run", "-d", args.image, "sleep", "infinity"],
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(f"container {cid[:12]}")
    try:
        # 2) copy source in + compile.
        run(["docker", "cp", str(SRC), f"{cid}:/tmp/generate_pythia.cc"])
        run(
            [
                "docker",
                "exec",
                cid,
                "bash",
                "-lc",
                "g++ /tmp/generate_pythia.cc -o /tmp/gen $(pythia8-config --cxxflags --libs)",
            ]
        )
        # 3) generate.
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
                "EEMUMU_OUT=/tmp/out.dat",
                cid,
                "/tmp/gen",
            ]
        )
        # 4) copy data out.
        run(["docker", "cp", f"{cid}:/tmp/out.dat", str(dat)])
    finally:
        subprocess.run(["docker", "rm", "-f", cid], stdout=subprocess.DEVNULL)

    # 5) host-side analysis.
    run([sys.executable, str(ANALYZE), str(dat), str(png)])
    print(f"\nPhase 2 (real chain) done:\n  data: {dat}\n  plot: {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
