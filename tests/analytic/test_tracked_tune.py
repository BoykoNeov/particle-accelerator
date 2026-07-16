"""Tracking-based tune (NAFF) — D2's analytic gates.

Layered deliberately (same discipline as the toy generator's three gates) so a
wrong estimator and a wrong lattice cannot cancel:

1. **The estimator alone** — recover a *synthetic* tone of known frequency, with no
   optics anywhere in the test. This is the primary gate.
2. **The normalisation alone** — recover a known Courant-Snyder ellipse from
   exactly-sampled synthetic phase-space points.
3. **Integration** — the tracked tune of a real lattice equals ``twiss.tunes()``
   modulo 1.

Gate 3 is the milestone requirement (ROADMAP D2: "tracked tune == analytic
one-turn-map tune to ~1e-5"). It is asserted far tighter than that here because
the estimator is exact to round-off; a loose gate would not catch a regression.

Long-term symplecticity is the sibling check in ``test_tracking_stability.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import Dipole, Lattice, ReferenceParticle, ThinQuadrupole, tunes
from accsim.tune import ellipse_from_trajectory, naff, tracked_tunes

PROTON_MASS_EV = 938.27208816e6


def _arc_ring(n_cells: int = 28, fq: float = 2.5) -> Lattice:
    """A FODO arc ring: 28 cells of QF/2 - bend - QD - bend - QF/2.

    Chosen so the tunes dodge every degeneracy the measurement could hide behind:
    ``Qx = 2.2434``, ``Qy = 1.7946``.

    - The **integer parts differ and are non-zero** (2 vs 1), so ``frac(Q) != Q`` and
      the mod-1 semantics are actually exercised rather than trivially true.
    - The **fractional parts are far apart** (0.243 vs 0.795), so swapping the planes
      could not pass.
    - Both sit **well clear of 0, 0.5 and 1**, where the +Q and -Q lines merge or alias.
    - ``frac(Qy) = 0.795 > 0.5``: a position-only (real) signal would report
      ``min(Q, 1-Q) = 0.205`` here, so this plane only passes because the estimator
      uses the signed complex signal.
    """
    ref = ReferenceParticle.from_gamma(PROTON_MASS_EV, 5.0)
    cell = [
        ThinQuadrupole(0.5 / fq),
        Dipole(1.0, 0.15),
        ThinQuadrupole(-1.0 / fq),
        Dipole(1.0, 0.15),
        ThinQuadrupole(0.5 / fq),
    ]
    return Lattice(cell * n_cells, ref)


# --- Gate 1: the estimator, isolated from all optics ------------------------


@pytest.mark.parametrize("f_true", [0.137, 0.618, 1.0 / math.pi, 0.9012, 0.0431])
def test_naff_recovers_synthetic_tone(f_true: float) -> None:
    """A pure tone of known frequency is recovered to round-off — the primary gate.

    No lattice, no Twiss: if this passes, the frequency estimator is right on its
    own terms. Amplitude and phase are non-trivial so neither can hide a bug.
    """
    k = np.arange(1024)
    z = 2.5 * np.exp(2.0j * np.pi * f_true * k + 0.7j)
    assert abs(naff(z) - f_true) < 1.0e-12


def test_naff_beats_the_fft_bin_resolution() -> None:
    """The refinement must resolve *within* a bin, not just pick the nearest one.

    Sitting a tone halfway between two FFT bins is the case a bare ``argmax`` gets
    maximally wrong (by ~1/(2N)); NAFF must do orders better.
    """
    n = 256
    f_true = 10.5 / n  # exactly between bins 10 and 11
    z = np.exp(2.0j * np.pi * f_true * np.arange(n))
    err = abs(naff(z) - f_true)
    assert err < 1.0e-12
    assert err < 1.0e-6 * (1.0 / (2.0 * n))  # crushes the bare-bin error


def test_naff_resolves_sign_of_frequency() -> None:
    """A negative frequency aliases to ``1 - f``, not back to ``f``.

    This is the property that lets the tune be read over the full [0, 1) rather
    than folded into [0, 0.5] — the reason the signal must be complex.
    """
    f = 0.137
    z = np.exp(-2.0j * np.pi * f * np.arange(1024))
    assert abs(naff(z) - (1.0 - f)) < 1.0e-12


def test_naff_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="at least 8 samples"):
        naff(np.ones(4, dtype=complex))
    with pytest.raises(ValueError, match="1-D"):
        naff(np.ones((8, 2), dtype=complex))


# --- Gate 2: the ellipse normalisation, isolated from tracking ---------------


@pytest.mark.parametrize(("beta", "alpha"), [(6.5, 0.0), (2.0, -1.3), (12.0, 0.75)])
def test_ellipse_from_trajectory_recovers_known_ellipse(beta: float, alpha: float) -> None:
    r"""``(beta, alpha)`` come back from phase-space points alone — no Twiss involved.

    Sampled at exactly-uniform phases so the averages are exact by orthogonality;
    this gates the covariance *algebra* (``beta = Sigma_11/sqrt(det Sigma)``,
    ``det Sigma = J^2`` via ``beta*gamma - alpha^2 = 1``) rather than the phase
    sampling. The action is deliberately not 1 to prove the scale is inferred, not
    assumed.
    """
    action = 3.7e-9
    phi = 2.0 * np.pi * np.arange(64) / 64.0
    u = np.sqrt(2.0 * action * beta) * np.cos(phi)
    up = -np.sqrt(2.0 * action / beta) * (np.sin(phi) + alpha * np.cos(phi))

    got_beta, got_alpha = ellipse_from_trajectory(u, up)
    assert got_beta == pytest.approx(beta, rel=1e-12)
    assert got_alpha == pytest.approx(alpha, abs=1e-12)


def test_ellipse_from_trajectory_rejects_degenerate_motion() -> None:
    """A particle on a line (or at rest) has no ellipse — say so rather than return junk."""
    with pytest.raises(ValueError, match="degenerate"):
        ellipse_from_trajectory(np.zeros(64), np.zeros(64))
    line = np.linspace(-1.0, 1.0, 64)
    with pytest.raises(ValueError, match="degenerate"):
        ellipse_from_trajectory(line, 2.0 * line)


# --- Gate 3: integration — tracked tune vs the matrix tune ------------------


def test_tracked_tune_matches_matrix_tune() -> None:
    """ROADMAP D2 gate: tracked tune == analytic one-turn-map tune (mod 1).

    Two independent routes to the same number: ``tunes()`` accumulates the
    Courant-Snyder phase advance element-by-element, while this tracks a particle
    and measures the frequency of its betatron oscillation. The roadmap asks for
    1e-5; the estimator is exact to round-off, so hold it to 1e-10.
    """
    lat = _arc_ring()
    qx_matrix, qy_matrix = tunes(lat)
    qx_tracked, qy_tracked = tracked_tunes(lat, n_turns=1024)

    assert qx_tracked == pytest.approx(qx_matrix % 1.0, abs=1e-10)
    assert qy_tracked == pytest.approx(qy_matrix % 1.0, abs=1e-10)


def test_tracked_tune_is_fractional_only() -> None:
    """Turn-by-turn sampling cannot see the integer part — assert that explicitly.

    The ring is chosen with ``Qx > 2`` and ``Qy > 1``, so ``frac(Q) != Q`` and this
    would fail if the comparison in the gate above were accidentally trivial.
    """
    lat = _arc_ring()
    qx_matrix, qy_matrix = tunes(lat)
    assert qx_matrix > 2.0 and qy_matrix > 1.0  # a real integer part exists
    assert int(qx_matrix) != int(qy_matrix)  # ...and the planes differ in it

    qx_tracked, qy_tracked = tracked_tunes(lat, n_turns=1024)
    assert 0.0 <= qx_tracked < 1.0
    assert 0.0 <= qy_tracked < 1.0
    # The integer part is genuinely absent, not silently recovered.
    assert abs(qx_tracked - qx_matrix) > 1.9


def test_planes_are_not_swapped() -> None:
    """The distinct fractional tunes (0.243 vs 0.795) pin x to x and y to y."""
    lat = _arc_ring()
    qx_matrix, qy_matrix = tunes(lat)
    qx_tracked, qy_tracked = tracked_tunes(lat, n_turns=1024)
    assert abs(qx_tracked - qy_tracked) > 0.15  # the planes are resolved apart
    assert abs(qx_tracked - qy_matrix % 1.0) > 0.15  # x is not reporting y's tune
    assert abs(qy_tracked - qx_matrix % 1.0) > 0.15


def test_signal_sign_gives_forward_tune() -> None:
    """Pin the ``z = U - i*PU`` convention named in ``accsim.tune``'s docstring.

    The opposite combination measures ``1 - Q``. Both are 'a tune'; only one matches
    this codebase's phase convention, so the choice is empirical, not remembered.
    A tune far from 0.5 makes the two unmistakable.
    """
    from accsim import Particle, Tracker

    lat = _arc_ring()
    qx_matrix, _ = tunes(lat)
    n_turns = 1024
    traj = Tracker(lat).track_turns(Particle(x=1e-6, y=1e-6), n_turns)[:n_turns]

    u, up = traj[:, 0], traj[:, 1]
    beta, alpha = ellipse_from_trajectory(u, up)
    un = (u - u.mean()) / math.sqrt(beta)
    pun = (alpha * (u - u.mean()) + beta * (up - up.mean())) / math.sqrt(beta)

    forward = naff(un - 1.0j * pun)
    backward = naff(un + 1.0j * pun)
    assert forward == pytest.approx(qx_matrix % 1.0, abs=1e-10)
    assert backward == pytest.approx(1.0 - qx_matrix % 1.0, abs=1e-6)


def test_tracked_tune_is_amplitude_independent_in_a_linear_lattice() -> None:
    """A linear map has no amplitude-dependent detuning — the tune must not move.

    Guards the normalisation: the recovered ellipse rescales with amplitude, so a
    scale error here would show up as an amplitude-dependent tune.
    """
    lat = _arc_ring()
    small = tracked_tunes(lat, n_turns=1024, x0=1e-9, y0=1e-9)
    large = tracked_tunes(lat, n_turns=1024, x0=1e-4, y0=1e-4)
    assert small[0] == pytest.approx(large[0], abs=1e-10)
    assert small[1] == pytest.approx(large[1], abs=1e-10)


def test_tracked_tune_rejects_a_plane_at_rest() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        tracked_tunes(_arc_ring(), n_turns=256, x0=0.0)
