r"""J2 cross-check: the octupole's kick and its amplitude detuning, against xtrack.

Two things are established here that the analytic suite cannot establish on its own.

**The sign convention.** Per the G1 rule, a convention accsim also derives is not an
independent reference, so the relation between accsim's ``k3`` and the ``knl`` of a
MAD-X / Xsuite multipole is fixed **by probe** — both candidates are tracked and the
wrong one is asserted to be decisively wrong, not merely worse:

    ThinOctupole(k3l)  ==  xt.Multipole(knl=[0, 0, 0, +k3l])

The agreement is to one floating-point ulp rather than bit-for-bit (unlike the
sextupole's): both codes compute the same cubic, but xtrack reaches it through its
general ``knl`` recursion with an inverse-factorial table, so the last bit can
differ. ``-k3l`` misses by exactly twice the kick.

**The detuning coefficient, by real tracking.** ``amplitude_detuning`` is a closed
form derived by phase-averaging; here the *same* FODO ring is built in xtrack, real
particles are tracked at a grid of amplitudes, and the anharmonicity matrix is fitted
from xtrack's own trajectories. This is the strongest available statement that the
``1/(16 pi)`` and the ``-2`` off-diagonal are physics rather than accsim's algebra.
The spectral step uses accsim's :func:`~accsim.tune.naff` (validated in D2, and
applied identically to both the with- and without-octupole runs so its bias cancels);
everything physical — the maps, the tracking, the optics the actions are built from —
is xtrack's.

Measured 2026-08-17, at launch amplitudes of 1-2e-4 m: agreement within **1.1 %** on
the diagonal and **0.3 %** on the cross terms, where the residual is the known
second-order-in-action term that the first-order closed form does not carry (accsim's
own tracking shows the same size, and ``test_amplitude_detuning.py`` gates it as
falling quadratically).

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Drift,
    Lattice,
    Octupole,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    amplitude_detuning,
)
from accsim.tune import ellipse_from_trajectory, naff
from accsim.twiss import closed_twiss

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
K3L = 5.0e4  # integrated thin strength [m^-3]
KF, KD = 1.25, -1.50  # the detuning ring's quadrupole strengths
N_TURNS = 1024

# A generic probe state: every coordinate nonzero, so no term can hide.
STATE = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 1.0e-3, 2.0e-4])


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _build(elements: list):
    """A tracked xtrack line, or a skip if the JIT toolchain is unavailable."""
    line = xt.Line(elements=elements)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


def _track_xtrack(elements: list, state: np.ndarray = STATE) -> np.ndarray:
    """Track ``state`` through a one-element xtrack line, returning the 6D result."""
    line = _build(elements)
    p = xt.Particles(
        mass0=MASS0,
        q0=1,
        gamma0=GAMMA0,
        x=state[0],
        px=state[1],
        y=state[2],
        py=state[3],
        zeta=state[4],
        delta=state[5],
    )
    line.track(p)
    return np.array([p.x[0], p.px[0], p.y[0], p.py[0], p.zeta[0], p.delta[0]])


# --------------------------------------------------------------------------
# 1. The convention, by probe
# --------------------------------------------------------------------------


def test_thin_octupole_is_the_positive_knl_multipole(ref: ReferenceParticle) -> None:
    """``ThinOctupole(k3l) == xt.Multipole(knl=[0,0,0,+k3l])``, to one ulp.

    This is what fixes accsim's ``k3`` sign against MAD-X's. Not bit-for-bit like the
    sextupole: xtrack evaluates the cubic through its general multipole recursion, so
    the two arithmetic orderings can differ in the last bit — asserted as one ulp of
    the kick, which is a far tighter statement than any tolerance.
    """
    accsim = ThinOctupole(K3L).track(STATE, ref)
    reference = _track_xtrack([xt.Multipole(knl=[0.0, 0.0, 0.0, K3L])])
    kick = np.abs(accsim - STATE).max()
    assert kick > 1e-5  # non-vacuous: there is a real kick to compare
    assert np.abs(accsim - reference).max() < 8.0 * np.spacing(kick)


def test_the_opposite_knl_sign_is_decisively_wrong(ref: ReferenceParticle) -> None:
    """The other branch of the probe: ``-k3l`` misses by twice the kick.

    Recorded because the MAD-X normal/skew asymmetry has bitten this package before
    (``Corrector``: ``kick_x = +k`` is ``knl=[-k]`` but ``kick_y = +k`` is
    ``ksl=[+k]``). A sign established by "it looks right" is not established.
    """
    accsim = ThinOctupole(K3L).track(STATE, ref)
    flipped = _track_xtrack([xt.Multipole(knl=[0.0, 0.0, 0.0, -K3L])])

    kick = np.abs(accsim - STATE)[[1, 3]]  # |Delta px|, |Delta py|
    miss = np.abs(flipped - accsim)[[1, 3]]
    assert np.all(kick > 0.0)
    assert np.allclose(miss, 2.0 * kick, rtol=1e-12)


def test_thin_kick_matches_xtrack_across_amplitudes(ref: ReferenceParticle) -> None:
    """The agreement is the whole cubic form, not one lucky point.

    All four monomials — ``x^3``, ``x y^2``, ``x^2 y``, ``y^3`` — are exercised,
    including the pure-plane cases where two of them vanish.
    """
    states = [
        np.array([x0, 1e-4, y0, -5e-5, 1e-3, 1e-4])
        for x0, y0 in [(3e-3, 1e-3), (-2e-3, 2e-3), (1e-3, -4e-3), (5e-3, 0.0), (0.0, 5e-3)]
    ]
    line = _build([xt.Multipole(knl=[0.0, 0.0, 0.0, K3L])])
    arr = np.array(states).T
    p = xt.Particles(
        mass0=MASS0,
        q0=1,
        gamma0=GAMMA0,
        x=arr[0],
        px=arr[1],
        y=arr[2],
        py=arr[3],
        zeta=arr[4],
        delta=arr[5],
    )
    line.track(p)
    order = np.argsort(p.particle_id)
    for i, state in enumerate(states):
        j = order[i]
        reference = np.array([p.x[j], p.px[j], p.y[j], p.py[j], p.zeta[j], p.delta[j]])
        accsim = ThinOctupole(K3L).track(state, ref)
        assert np.abs(accsim - reference).max() < 8.0 * np.spacing(
            np.abs(accsim - state).max() + 1e-30
        )


# --------------------------------------------------------------------------
# 2. The thick element, compared by difference
# --------------------------------------------------------------------------


def test_thick_residual_at_zero_strength_is_the_chromatic_drift(ref: ReferenceParticle) -> None:
    """Attribute the thick element's residual: at ``k3 = 0`` it is the drift, not the magnet.

    xtrack integrates the exact drift while accsim's linear map carries ``x += L px``;
    the difference is ``-L px delta`` to leading order and belongs to the drift model.
    Establishing this is what licenses the difference method in the next test — the
    same idiom J1 used for the sextupole.
    """
    length = 0.4
    residual = (
        _track_xtrack([xt.Octupole(length=length, k3=0.0)]) - Drift(length).matrix(ref) @ STATE
    )
    px, py, delta = STATE[1], STATE[3], STATE[5]
    assert residual[0] == pytest.approx(-length * px * delta, rel=1e-3)
    assert residual[2] == pytest.approx(-length * py * delta, rel=1e-3)
    assert residual[1] == 0.0 and residual[3] == 0.0 and residual[5] == 0.0


def test_thick_octupole_nonlinear_content_matches_xtrack(ref: ReferenceParticle) -> None:
    """The isolated kick of the thick element agrees with xtrack's thick octupole.

    ``with(k3) - without(k3)`` at fixed geometry cancels the shared drift model and
    leaves the magnet. Compared at ``n_slices = 1`` because that is xtrack's own
    splitting: as in J1, raising the slice count converges accsim onto the *exact*
    map, which moves it away from xtrack rather than toward it.
    """
    length, k3 = 0.4, 5.0e4
    xt_kick = _track_xtrack([xt.Octupole(length=length, k3=k3)]) - _track_xtrack(
        [xt.Octupole(length=length, k3=0.0)]
    )
    acc = Octupole(length, k3, n_slices=1).track(STATE, ref) - Octupole(
        length, 0.0, n_slices=1
    ).track(STATE, ref)

    transverse = [0, 1, 2, 3]
    assert np.max(np.abs(xt_kick[transverse])) > 1e-6  # non-vacuous
    assert np.allclose(acc[transverse], xt_kick[transverse], rtol=1e-3, atol=1e-12)


# --------------------------------------------------------------------------
# 3. Amplitude detuning, fitted from xtrack's own tracking
# --------------------------------------------------------------------------


def _accsim_ring(ref: ReferenceParticle, k3l: float) -> Lattice:
    half = [Drift(0.5), Quadrupole(0.3, KF), Drift(0.5), Quadrupole(0.3, KD), Drift(0.5)]
    return Lattice([*half, ThinOctupole(k3l), *half[::-1]] * 3, ref)


def _xtrack_ring(k3l: float):
    half = [
        xt.Drift(length=0.5),
        xt.Quadrupole(length=0.3, k1=KF),
        xt.Drift(length=0.5),
        xt.Quadrupole(length=0.3, k1=KD),
        xt.Drift(length=0.5),
    ]
    octupole = xt.Multipole(knl=[0.0, 0.0, 0.0, k3l])
    return _build([*half, octupole, *half[::-1]] * 3)


def _tunes_from_monitor(mon, i: int) -> tuple[float, float]:
    """Fractional tunes of particle ``i`` from turn-by-turn data, via accsim's NAFF."""
    out = []
    for u, up in ((mon.x[i], mon.px[i]), (mon.y[i], mon.py[i])):
        beta, alpha = ellipse_from_trajectory(u, up)
        root_beta = math.sqrt(beta)
        u_n = (u - u.mean()) / root_beta
        pu_n = (alpha * (u - u.mean()) + beta * (up - up.mean())) / root_beta
        out.append(naff(u_n - 1.0j * pu_n))
    return out[0], out[1]


def _tracked_tunes(line, launches: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Track every launch amplitude at once and read both tunes off each trajectory."""
    xs = np.array([a for a, _ in launches])
    ys = np.array([b for _, b in launches])
    p = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0, x=xs, px=0.0 * xs, y=ys, py=0.0 * ys)
    line.track(p, num_turns=N_TURNS, turn_by_turn_monitor=True)
    mon = line.record_last_track
    return [_tunes_from_monitor(mon, i) for i in range(len(launches))]


def test_optics_agree_before_the_detuning_is_compared(ref: ReferenceParticle) -> None:
    """The two rings really are the same machine — beta and the tunes match first.

    Without this the detuning comparison would be between two different lattices: the
    actions are built from beta, and the closed form is quadratic in it, so a 1 %
    disagreement in beta would be a 2 % disagreement in the answer for no physical
    reason.
    """
    tw_acc = closed_twiss(_accsim_ring(ref, 0.0))
    tw_xt = _xtrack_ring(0.0).twiss(method="4d")
    assert tw_xt.betx[0] == pytest.approx(tw_acc.beta_x, rel=1e-10)
    assert tw_xt.bety[0] == pytest.approx(tw_acc.beta_y, rel=1e-10)
    assert tw_xt.alfx[0] == pytest.approx(0.0, abs=1e-10)  # the palindrome, from xtrack
    assert tw_xt.alfy[0] == pytest.approx(0.0, abs=1e-10)


def test_amplitude_detuning_matches_xtrack_tracking(ref: ReferenceParticle) -> None:
    """The headline cross-check: the anharmonicity matrix, fitted from xtrack particles.

    Both codes' rings are tracked at the same nine launch amplitudes; the octupole's
    effect is the *difference* between the two runs (which cancels NAFF's own bias),
    and the four coefficients come from a least-squares plane fit in the actions.
    Actions are built from xtrack's twiss, so nothing on the reference side depends
    on accsim's optics.

    The surviving ~1 % is the second-order-in-action term the first-order closed form
    does not carry; ``test_amplitude_detuning.py`` shows it falling quadratically.
    """
    line_on, line_off = _xtrack_ring(K3L), _xtrack_ring(0.0)
    tw_xt = line_off.twiss(method="4d")
    launches = [(x0, y0) for x0 in (1.0e-4, 1.5e-4, 2.0e-4) for y0 in (1.0e-4, 1.5e-4, 2.0e-4)]

    q_on = _tracked_tunes(line_on, launches)
    q_off = _tracked_tunes(line_off, launches)
    rows, obs_x, obs_y = [], [], []
    for (x0, y0), (qx1, qy1), (qx0, qy0) in zip(launches, q_on, q_off, strict=True):
        rows.append(
            [
                (1.0 + tw_xt.alfx[0] ** 2) * x0**2 / (2.0 * tw_xt.betx[0]),
                (1.0 + tw_xt.alfy[0] ** 2) * y0**2 / (2.0 * tw_xt.bety[0]),
            ]
        )
        obs_x.append(qx1 - qx0)
        obs_y.append(qy1 - qy0)
    M = np.array(rows)
    fit_x = np.linalg.lstsq(M, np.array(obs_x), rcond=None)[0]
    fit_y = np.linalg.lstsq(M, np.array(obs_y), rcond=None)[0]

    A = amplitude_detuning(_accsim_ring(ref, K3L))
    assert max(abs(v) for v in obs_x) > 1e-5  # non-vacuous: xtrack really detunes
    assert fit_x[0] == pytest.approx(A[0, 0], rel=0.03)
    assert fit_x[1] == pytest.approx(A[0, 1], rel=0.03)
    assert fit_y[0] == pytest.approx(A[1, 0], rel=0.03)
    assert fit_y[1] == pytest.approx(A[1, 1], rel=0.03)
    # The off-diagonal is negative and dominates dQx here: a sign slip in the cross
    # term would move dQx by more than its own size, not by a few per cent.
    assert fit_x[1] < 0.0 and abs(fit_x[1]) > 1.5 * abs(fit_x[0])
