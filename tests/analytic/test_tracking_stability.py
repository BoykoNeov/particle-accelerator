"""Long-term tracking stability — the symplecticity smoke test (ROADMAP pillar).

Track a matched particle for 1e4 turns of a dipole-containing ring and confirm
the Courant-Snyder action (the betatron emittance of a single particle) does not
drift and the motion stays bounded. For the *linear* maps of Stage 1 this is
largely implied by per-element ``is_symplectic`` plus stability (``|Tr| < 2``),
but it exercises the composed one-turn map over many turns end-to-end — the check
that first bites once thick/nonlinear exact maps arrive (Stage 3+).

Marked ``slow``: run with ``pytest -m slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    Dipole,
    Lattice,
    Particle,
    ReferenceParticle,
    ThinQuadrupole,
    Tracker,
    closed_twiss,
)

pytestmark = pytest.mark.slow

N_TURNS = 10_000


def _arc_ring(ref: ReferenceParticle, n_cells: int = 3) -> Lattice:
    cell = [
        ThinQuadrupole(0.5 / 2.5),
        Dipole(1.0, 0.15),
        ThinQuadrupole(-1.0 / 2.5),
        Dipole(1.0, 0.15),
        ThinQuadrupole(0.5 / 2.5),
    ]
    return Lattice(cell * n_cells, ref)


def _cs_action(traj: np.ndarray, beta: float, alpha: float, i_pos: int, i_mom: int) -> np.ndarray:
    """Courant-Snyder action J = gamma u^2 + 2 alpha u u' + beta u'^2 along a trajectory."""
    gamma = (1.0 + alpha**2) / beta
    u, up = traj[:, i_pos], traj[:, i_mom]
    return gamma * u**2 + 2.0 * alpha * u * up + beta * up**2


def test_courant_snyder_action_does_not_drift() -> None:
    ref = ReferenceParticle.from_gamma(938.27208816e6, 5.0)
    lat = _arc_ring(ref)
    tw = closed_twiss(lat)

    # On-momentum betatron motion (delta = 0 so dispersion does not shift the
    # closed orbit), launched at a moderate amplitude.
    p = Particle(x=1.0e-3, px=2.0e-4, y=-5.0e-4, py=1.0e-4)
    traj = Tracker(lat).track_turns(p, N_TURNS)

    jx = _cs_action(traj, tw.beta_x, tw.alpha_x, 0, 1)
    jy = _cs_action(traj, tw.beta_y, tw.alpha_y, 2, 3)

    # The action is an exact invariant of the symplectic linear map; over 1e4
    # turns only float64 round-off accumulates (~1e-12 relative). A non-symplectic
    # element would show orders-of-magnitude more drift, so 1e-10 catches real
    # damping/blow-up while tolerating round-off.
    assert np.max(np.abs(jx - jx[0])) / jx[0] < 1e-10
    assert np.max(np.abs(jy - jy[0])) / jy[0] < 1e-10

    # And the motion stays bounded — no secular growth in amplitude.
    assert np.max(np.abs(traj[:, 0])) < 5.0 * abs(p.x)
    assert np.max(np.abs(traj[:, 2])) < 50.0 * abs(p.y)


def test_zero_amplitude_particle_stays_on_axis() -> None:
    # The reference particle (all zeros) must remain exactly on-axis every turn.
    ref = ReferenceParticle.from_gamma(938.27208816e6, 5.0)
    traj = Tracker(_arc_ring(ref)).track_turns(Particle(), N_TURNS)
    assert np.max(np.abs(traj)) == 0.0
