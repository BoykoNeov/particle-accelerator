"""Cross-check the Solenoid's **tracked** map against xtrack (S1).

Marked ``reference``: skipped when xtrack is absent, and skipped (not failed) when xtrack's
JIT C-kernel compilation is unavailable (see ``conftest.py`` / ``_xtrack_jit``).

**Why this leg tracks rather than compares matrices.** MAD-X already pins the 6x6 entrywise
at round-off (``test_solenoid_madx.py``); differencing xtrack's tracker to recover a matrix
would only measure the finite-difference truncation, which on the filter probe was ``1.9e-10``
— *above* the effects this file exists to see. So every comparison here is a tracked 6D
state, where the two codes' maps meet with nothing in between.

**The one honest disagreement, and why it is not an error.** accsim's solenoid is exact in
``delta`` and **paraxial** in the angles; xtrack's ``UniformSolenoid`` is not paraxial. The
residual is therefore P2 (iv)'s kinematic remainder, the expansion of
``sqrt((1+delta)^2 - px^2 - py^2)``, and it says so by its **scaling**: halving the
transverse amplitude divides it by exactly ``8``. That is asserted here as its own test.
``Solenoid`` deliberately does not offer ``kinematic_slices`` — in a solenoid that remainder
is a function of the *mechanical* momenta, not the canonical ones, so it is not the
momentum-only drift the quadrupole's slicing integrates — and the gap is recorded rather
than papered over.

**What the milestone is, measured against that floor.** A ``delta``-independent solenoid map
misses xtrack by ``6.2e-7`` at ``delta = 1e-3`` where the shipped one misses by ``1.1e-10``:
the momentum dependence is three orders above the paraxial gap, not lost in it.

**Cost.** An ``xt.Line`` build measured 30-70 s on this machine (see ``CONVENTIONS.md`` ->
*Test-suite cost*), so every line here is **module-scoped** and built exactly once.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    Solenoid,
    normal_mode_tunes,
)
from accsim.coords import DELTA, PX, PY, X, Y

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 0.51099895069e6  # electron, eV
GAMMA0 = 20.0
LENGTH = 0.9
KS = 0.6

#: A generic probe state: every coordinate nonzero, so no term can hide behind a zero.
STATE = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 1.0e-3, 2.0e-4])
#: The transverse amplitude the paraxial floor below was measured at.
AMPLITUDE = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5])

# The ring: a FODO with one solenoid in it, the smallest coupled machine this element makes.
# The trailing weak quadrupole splits the tunes off the difference resonance, where the bare
# 4-cell FODO sits exactly (``Q_x = Q_y``) and the two modes are degenerate — which would make
# matching mode to mode between the codes meaningless.
KQ, LQ, LD, N_CELLS, L_SOL = 1.2, 0.3, 0.7, 4, 0.4
L_SPLIT, K_SPLIT = 0.01, 5.0


@pytest.fixture(scope="module")
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0, charge=-1.0)


def _build(line: xt.Line) -> xt.Line:
    line.particle_ref = xt.Particles(mass0=MASS0, q0=-1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


@pytest.fixture(scope="module")
def solenoid_line() -> xt.Line:
    """One `xt.UniformSolenoid`, built once for the whole module."""
    return _build(xt.Line(elements=[xt.UniformSolenoid(ks=KS, length=LENGTH)]))


@pytest.fixture(scope="module")
def ring_line() -> xt.Line:
    """The FODO-plus-solenoid ring, built once for the whole module."""
    cell = [
        xt.Quadrupole(length=LQ, k1=KQ),
        xt.Drift(length=LD),
        xt.Quadrupole(length=LQ, k1=-KQ),
        xt.Drift(length=LD),
    ]
    elements = cell * N_CELLS
    return _build(
        xt.Line(
            elements=elements[:4]
            + [xt.UniformSolenoid(ks=KS, length=L_SOL)]
            + elements[4:]
            + [xt.Quadrupole(length=L_SPLIT, k1=K_SPLIT)]
        )
    )


def _accsim_ring(ks: float, ref: ReferenceParticle) -> Lattice:
    cell = [Quadrupole(LQ, KQ), Drift(LD), Quadrupole(LQ, -KQ), Drift(LD)]
    elements = cell * N_CELLS
    return Lattice(
        elements[:4] + [Solenoid(L_SOL, ks)] + elements[4:] + [Quadrupole(L_SPLIT, K_SPLIT)],
        ref,
    )


def _track(line: xt.Line, states: np.ndarray) -> np.ndarray:
    """Track a ``(6,)`` state or a ``(6, n)`` bunch through ``line``."""
    st = np.atleast_2d(np.asarray(states, dtype=float).T).T
    p = xt.Particles(
        mass0=MASS0,
        q0=-1,
        gamma0=GAMMA0,
        x=st[X],
        px=st[PX],
        y=st[Y],
        py=st[PY],
        zeta=st[4],
        delta=st[DELTA],
    )
    line.track(p)
    return np.array([p.x, p.px, p.y, p.py, p.zeta, p.delta])


def test_tracked_state_matches_xtrack(solenoid_line, ref: ReferenceParticle) -> None:
    """A generic 6D state, tracked through both codes, agrees at the paraxial floor.

    ``1.5e-9`` is the residual the filter probe measured (``3.9e-10`` over a set of four
    probe states) with a factor of ~4 of headroom — the difference is a *known physical
    term*, not round-off, so the gate is deliberately close to it rather than generous.
    """
    states = np.stack(
        [
            STATE,
            np.array([-1.0e-3, -2.0e-4, 3.0e-3, 1.0e-4, -5.0e-4, -3.0e-3]),
            np.array([5.0e-4, 0.0, 0.0, -1.0e-4, 0.0, 1.0e-2]),
            np.array([0.0, 1.0e-3, 0.0, 0.0, 0.0, 0.0]),
        ],
        axis=1,
    )
    got = Solenoid(LENGTH, KS).track(states, ref)
    assert np.abs(got - _track(solenoid_line, states)).max() < 1.5e-9


def test_the_residual_is_the_paraxial_term_and_says_so_by_its_scaling(
    solenoid_line, ref: ReferenceParticle
) -> None:
    """Halving the transverse amplitude divides the disagreement by exactly ``8``.

    The whole content of "the residual is P2 (iv)'s kinematic remainder". A cubic law is what
    the expansion of ``sqrt((1+delta)^2 - p^2)`` predicts, and a coefficient error anywhere in
    the solenoid's map would not obey it — it would scale linearly with the amplitude, or with
    ``ks``, or not at all. This is why the gap can be left open honestly.
    """
    lams = [1.0, 0.5, 0.25, 0.125]
    states = np.stack([np.concatenate([lam * AMPLITUDE, [0.0, 0.0]]) for lam in lams], axis=1)
    resid = np.abs(Solenoid(LENGTH, KS).track(states, ref) - _track(solenoid_line, states)).max(
        axis=0
    )
    assert resid[0] > 1e-11, "the residual must be measurable at all for the ratio to mean anything"
    for coarse, fine in zip(resid[:-1], resid[1:], strict=True):
        assert coarse / fine == pytest.approx(8.0, rel=0.02)


@pytest.mark.parametrize("delta", [1e-4, 1e-3, 1e-2])
def test_the_momentum_dependence_is_three_orders_above_that_floor(
    delta: float, solenoid_line, ref: ReferenceParticle
) -> None:
    """A frozen-``K`` solenoid fails against xtrack by orders, not by a tolerance.

    ``ks`` is normalised to the *reference* rigidity, so the focusing a particle feels is
    ``ks/(1+delta)``. Freezing it — which is what ``matrix()`` does, correctly, and what a
    map must not — is the milestone's central error, and this measures how visible it is: a
    factor of ``5.6e2`` at ``delta = 1e-4`` rising to ``5.7e4`` at ``delta = 1e-2``.
    """
    state = np.concatenate([AMPLITUDE, [0.0, delta]])
    want = _track(solenoid_line, state)[:, 0]
    shipped = np.abs(Solenoid(LENGTH, KS).track(state, ref) - want).max()

    # the frozen-K map: the linear matrix applied to the state, momentum ignored
    frozen = Solenoid(LENGTH, KS).matrix(ref) @ state
    frozen_gap = np.abs(frozen[[X, PX, Y, PY]] - want[[X, PX, Y, PY]]).max()

    assert shipped < 1.5e-9
    assert frozen_gap / shipped > 100.0


def test_the_ring_normal_mode_tunes_match_xtrack(ring_line, ref: ReferenceParticle) -> None:
    """A ring with a solenoid in it: both codes agree on the coupled normal-mode tunes.

    The integration gate. accsim reaches these through G1's eigen route
    (:func:`~accsim.twiss.normal_mode_tunes`), which diagonalises the one-turn map and needed
    no change at all for this element; xtrack reaches them through its own coupled Twiss. The
    solenoid is what makes the two modes *not* the two planes, and the test asserts that too:
    with the solenoid switched off, the plane tunes come back.

    The tolerance is the measured residual (**`9.1e-14`**) with a decade of headroom, not a
    physics-sized number. It is that small — far below the single-element tracking floor
    above — because a Twiss is built on the *linear* map, where the two codes' solenoids agree
    exactly; the paraxial gap this file's other tests measure is a nonlinear effect and does
    not reach the tunes at all.
    """
    tw = ring_line.twiss(method="4d")
    q1, q2 = normal_mode_tunes(_accsim_ring(KS, ref))
    got = sorted(v % 1.0 for v in (q1, q2))
    want = sorted(v % 1.0 for v in (tw.qx, tw.qy))
    assert got[0] == pytest.approx(want[0], abs=1e-12)
    assert got[1] == pytest.approx(want[1], abs=1e-12)

    # ... and the solenoid is what separates them from the uncoupled ring's plane tunes
    off = sorted(v % 1.0 for v in normal_mode_tunes(_accsim_ring(0.0, ref)))
    assert max(abs(a - b) for a, b in zip(got, off, strict=True)) > 1e-3
