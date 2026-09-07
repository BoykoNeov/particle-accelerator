"""Regenerate ``editor/presets.js`` — the scenarios bundled with the editor.

The presets are stored fully expanded (a 24-cell ring is 216 element records) so the
file is plain scenario JSON that ``accsim.scenario`` and the cross-check test can
read without any expansion rule of their own (they take the JSON after
``ACCSIM_PRESETS = `` up to the closing ``];``). This script is where the cells are
written once and repeated. Run it after editing::

    .venv/Scripts/python.exe scripts/make_presets.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "editor" / "presets.js"

TWO_PI = 2.0 * math.pi
FMT = "accsim-scenario/1"


def fodo_cell() -> list[dict]:
    """One symmetric cell of a 10 GeV proton synchrotron, thick quads, 24 cells to a ring."""
    theta = TWO_PI / 48
    return [
        {"type": "Quadrupole", "name": "QF.h", "length": 0.25, "k1": 0.55},
        {"type": "Drift", "name": "D1", "length": 1.0},
        {"type": "Dipole", "name": "B1", "length": 2.0, "angle": theta},
        {"type": "Drift", "name": "D2", "length": 1.0},
        {"type": "Quadrupole", "name": "QD", "length": 0.5, "k1": -0.55},
        {"type": "Drift", "name": "D3", "length": 1.0},
        {"type": "Dipole", "name": "B2", "length": 2.0, "angle": theta},
        {"type": "Drift", "name": "D4", "length": 1.0},
        {"type": "Quadrupole", "name": "QF.h", "length": 0.25, "k1": 0.55},
    ]


def electron_cell() -> list[dict]:
    """The cell of ``examples/build_a_machine.py``: thin quads, f = 3 m, two 2 m bends."""
    theta = TWO_PI / 48
    return [
        {"type": "ThinQuadrupole", "name": "QF.h", "k1l": 0.5 / 3.0},
        {"type": "Drift", "name": "D", "length": 1.0},
        {"type": "Dipole", "name": "B", "length": 2.0, "angle": theta},
        {"type": "Drift", "name": "D", "length": 1.0},
        {"type": "ThinQuadrupole", "name": "QD", "k1l": -1.0 / 3.0},
        {"type": "Drift", "name": "D", "length": 1.0},
        {"type": "Dipole", "name": "B", "length": 2.0, "angle": theta},
        {"type": "Drift", "name": "D", "length": 1.0},
        {"type": "ThinQuadrupole", "name": "QF.h", "k1l": 0.5 / 3.0},
    ]


def dba_cell() -> list[dict]:
    """A 3 GeV double-bend achromat; the central quad closes the dispersion in the straight."""
    theta = TWO_PI / 32
    half = [
        {"type": "Drift", "name": "straight", "length": 1.5},
        {"type": "Quadrupole", "name": "QD1", "length": 0.3, "k1": -2.85},
        {"type": "Drift", "name": "D1", "length": 0.3},
        {"type": "Quadrupole", "name": "QF1", "length": 0.3, "k1": 3.75},
        {"type": "Drift", "name": "D2", "length": 0.4},
        {"type": "Dipole", "name": "B1", "length": 1.0, "angle": theta},
        {"type": "Drift", "name": "D3", "length": 0.2},
        {"type": "ThinSextupole", "name": "SD", "k2l": -3.0},
        {"type": "Drift", "name": "D4", "length": 0.15},
    ]
    centre = [{"type": "Quadrupole", "name": "QFc", "length": 0.3, "k1": 7.389543282640958}]
    mirror = [dict(e) for e in reversed(half)]
    mirror[1] = {"type": "ThinSextupole", "name": "SF", "k2l": 6.0}
    mirror[3]["name"] = "B2"
    return half + centre + mirror


def triplet_line() -> list[dict]:
    return [
        {"type": "Drift", "name": "D0", "length": 3.0},
        {"type": "Corrector", "name": "CH", "kick_x": 2e-4, "kick_y": 0.0},
        {"type": "Drift", "name": "D1", "length": 1.0},
        {"type": "Quadrupole", "name": "Q1", "length": 0.5, "k1": 0.9},
        {"type": "Drift", "name": "D2", "length": 0.4},
        {"type": "Quadrupole", "name": "Q2", "length": 0.5, "k1": -1.6},
        {"type": "Drift", "name": "D3", "length": 0.4},
        {"type": "Quadrupole", "name": "Q3", "length": 0.5, "k1": 0.9},
        {"type": "Drift", "name": "D4", "length": 4.0},
        {"type": "Aperture", "name": "AP", "shape": "circular", "half_x": 0.02, "length": 0.0},
    ]


def presets() -> list[dict]:
    electron = {"species": "electron", "energy_mode": "total", "energy_eV": 2e9}
    e_beam = {"emit_x": 5e-8, "emit_y": 5e-10, "sigma_delta": 7e-4}
    ring = electron_cell() * 24
    kicked = [dict(e) for e in electron_cell() * 24]
    kicked[0] = dict(kicked[0], dx=5e-4)
    kicked.insert(27, {"type": "Corrector", "name": "CH", "kick_x": -6e-5, "kick_y": 0.0})
    return [
        {
            "format": FMT,
            "id": "fodo-cell",
            "name": "FODO cell",
            "description": "One symmetric cell of a 10 GeV proton synchrotron: half a focusing "
            "quadrupole, a sector bend, a defocusing quadrupole, a second bend, and the other "
            "half of the focusing quadrupole. 24 of these close a ring.",
            "periodic": True,
            "reference": {"species": "proton", "energy_mode": "total", "energy_eV": 10e9},
            "beam": {"emit_x": 2e-6, "emit_y": 2e-6, "sigma_delta": 5e-4},
            "elements": fodo_cell(),
        },
        {
            "format": FMT,
            "id": "electron-ring",
            "name": "Electron storage ring",
            "description": "The 192 m, 24-cell ring of examples/build_a_machine.py at its 2 GeV "
            "store energy: thin quadrupoles, sector bends totalling 2 pi, and one 5 MV cavity "
            "on harmonic 400.",
            "periodic": True,
            "reference": electron,
            "beam": e_beam,
            "elements": ring
            + [{"type": "RFCavity", "name": "RF", "voltage": 5e6, "harmonic": 400, "phi_s": 0.0}],
        },
        {
            "format": FMT,
            "id": "dba-cell",
            "name": "Double-bend achromat",
            "description": "One cell of a 3 GeV light source: two sector bends with a strong "
            "quadrupole between them, tuned so the dispersion closes to zero in the straight "
            "where an insertion device would go. Two thin sextupoles at dispersion pull the "
            "chromaticity back towards zero.",
            "periodic": True,
            "reference": {"species": "electron", "energy_mode": "total", "energy_eV": 3e9},
            "beam": {"emit_x": 5e-9, "emit_y": 5e-11, "sigma_delta": 1e-3},
            "elements": dba_cell(),
        },
        {
            "format": FMT,
            "id": "triplet-line",
            "name": "Quadrupole triplet line",
            "description": "A transfer line, not a ring: the beam enters with a given Twiss, a "
            "quadrupole triplet refocuses it, and a small corrector kick shows how a trajectory "
            "error grows downstream.",
            "periodic": False,
            "reference": {"species": "proton", "energy_mode": "kinetic", "energy_eV": 1e9},
            "initial_twiss": {
                "beta_x": 12.0,
                "alpha_x": 0.0,
                "beta_y": 12.0,
                "alpha_y": 0.0,
                "disp_x": 0.0,
                "disp_px": 0.0,
            },
            "beam": {"emit_x": 1e-6, "emit_y": 1e-6, "sigma_delta": 1e-3},
            "elements": triplet_line(),
        },
        {
            "format": FMT,
            "id": "wiggler-ring",
            "name": "Ring with a wiggler",
            "description": "The electron storage ring with a 2 m, 1.5 T wiggler in an added "
            "straight. A wiggler focuses vertically and nowhere else: watch beta_y through it.",
            "periodic": True,
            "reference": electron,
            "beam": e_beam,
            "elements": ring
            + [
                {"type": "Drift", "name": "DW", "length": 1.0},
                {"type": "Wiggler", "name": "W", "period": 0.1, "h0": 0.2248, "periods": 20},
                {"type": "Drift", "name": "DW", "length": 1.0},
                {"type": "RFCavity", "name": "RF", "voltage": 5e6, "harmonic": 408, "phi_s": 0.0},
            ],
        },
        {
            "format": FMT,
            "id": "kicked-ring",
            "name": "Displaced quadrupole",
            "description": "The electron storage ring with its first focusing quadrupole pushed "
            "0.5 mm sideways and one horizontal corrector three cells downstream. The closed "
            "orbit leaves the design axis; tune the corrector to flatten it.",
            "periodic": True,
            "reference": electron,
            "beam": e_beam,
            "elements": kicked,
        },
    ]


def main() -> None:
    body = json.dumps(presets(), indent=1)
    text = (
        "/*\n"
        " * Bundled scenarios for the accsim editor. GENERATED by scripts/make_presets.py —\n"
        " * edit that script, not this file. Plain JSON after the assignment, so\n"
        " * accsim.scenario (Python) and tests/analytic/test_scenario.py read it too: they\n"
        " * take the text after 'ACCSIM_PRESETS = ' up to the closing '];'.\n"
        " */\n"
        f"var ACCSIM_PRESETS = {body};\n"
        'if (typeof module === "object" && module.exports) module.exports = ACCSIM_PRESETS;\n'
    )
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(presets())} presets, {len(text)} bytes)")


if __name__ == "__main__":
    main()
