"""Cross-check the synchrotron tune ``Qs`` against xtrack on a dispersive ring.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

This is the external anchor for the RF-cavity phase convention and the ``Qs``
sign/coefficient (flag C). It pins two distinct things:

1. **The map, to machine precision.** accsim's full 6x6 one-turn matrix and
   xtrack's agree entrywise (verified here on the ``(zeta, delta)`` block), so the
   longitudinal *mode* eigen-tune extracted from accsim's own map matches xtrack's
   ``tw.qs`` to ~1e-6. This validates the RF phase convention
   (``phi = phi_s - k_rf zeta`` == xtrack's ``lag_rad - (2 pi f/c) zeta/beta0``),
   the ``beta0^2 E0`` energy factor, and the sign — a flipped convention would be
   an O(1) gap or outright instability, not 1e-6.

2. **The lumped analytic formula, to sub-percent.** :func:`synchrotron_tune`
   returns the textbook reduced-2x2 value built from the slip factor; it omits the
   second-order synchro-betatron coupling that the full 6D eigen-tune carries, so
   it agrees with ``tw.qs`` only at the coupling order (~0.4% here).

The bare one-turn ``(zeta, delta)`` block is itself *unstable* on this dispersive
ring (its half-trace exceeds 1) — the stable synchrotron mode exists only once the
dispersion coupling is folded in, exactly the point of sourcing ``Qs`` from the
slip factor (flag A).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Dipole,
    Lattice,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    synchrotron_tune,
)
from accsim.reference import CLIGHT

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6
GAMMA0 = 5.0
F_FOCAL = 2.5
L_BEND = 1.0
ANGLE = 0.15
HARMONIC = 8
VOLTAGE = 5.0e5
PHI_S = math.pi  # above transition => stable phase is pi


def _circumference() -> float:
    return 2.0 * L_BEND  # two thick bends, thin quads


def _frequency(ref: ReferenceParticle) -> float:
    return HARMONIC * ref.beta0 * CLIGHT / _circumference()


def _accsim_lattice() -> Lattice:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    arc = [
        ThinQuadrupole(0.5 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(-1.0 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]
    return Lattice([*arc, RFCavity(VOLTAGE, _frequency(ref), PHI_S)], ref)


def _mode_tune(matrix: np.ndarray) -> float:
    """Smallest nonzero eigen-tune of a symplectic 6x6 (the synchrotron mode here)."""
    tunes = np.sort(np.abs(np.angle(np.linalg.eigvals(matrix)))) / (2.0 * math.pi)
    return float(tunes[tunes > 1e-9][0])


def _xtrack_twiss():
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)

    def quad(k1l: float):
        return xt.Multipole(knl=[0.0, k1l], length=0.0)

    def bend():
        b = xt.Bend(length=L_BEND, angle=ANGLE, k1=0.0)
        b.edge_entry_active = 0
        b.edge_exit_active = 0
        return b

    cav = xt.Cavity(voltage=VOLTAGE, frequency=_frequency(ref), phase=PHI_S)
    line = xt.Line(
        elements=[
            quad(0.5 / F_FOCAL),
            bend(),
            quad(-1.0 / F_FOCAL),
            bend(),
            quad(0.5 / F_FOCAL),
            cav,
        ]
    )
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        tw = line.twiss()
        return tw, line.get_R_matrix(particle_on_co=tw.particle_on_co)["R_matrix"]
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


def test_one_turn_map_and_mode_tune_match_xtrack() -> None:
    tw, rmat = _xtrack_twiss()
    lat = _accsim_lattice()
    accsim_map = lat.one_turn_matrix()

    # The maps agree entrywise on the longitudinal block (RF R65 + arc R56).
    np.testing.assert_allclose(accsim_map[4:6, 4:6], rmat[4:6, 4:6], rtol=0, atol=1e-8)

    # ... so the coupled synchrotron eigen-tune of accsim's own map matches tw.qs.
    assert _mode_tune(accsim_map) == pytest.approx(abs(tw.qs), rel=1e-6)


def test_lumped_formula_matches_xtrack_to_coupling_order() -> None:
    tw, _ = _xtrack_twiss()
    # The lumped reduced-2x2 Qs omits synchro-betatron coupling, so it lands within
    # sub-percent of the exact eigen-tune -- close enough to reject a beta0^2 (~4%)
    # or 2pi error, honest about the coupling residual.
    assert synchrotron_tune(_accsim_lattice()) == pytest.approx(abs(tw.qs), rel=1e-2)
