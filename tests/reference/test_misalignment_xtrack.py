r"""K1 cross-check: the *sign* of ``(dx, dy)``, against xtrack's ``shift_x`` / ``shift_y``.

An offset's sign has no analytic gate. The rms orbit K1 predicts goes as ``d^2``, so
no statistical check can tell whether ``dx`` means *the magnet moved right* or *the
beam sits right of the magnet centre* — precisely the relative sign that flips
silently. The convention is therefore fixed here by probe, the rule J1, J2, J3 and
:class:`~accsim.elements.sextupole.ThinSkewSextupole` each ended up needing:

    ThinQuadrupole(k1l, dx=d)  ==  xt.Multipole(knl=[0, k1l], shift_x=d)

measured 2026-08-17, and **bit-for-bit** at the probe state — not to a tolerance. Both
codes translate, apply the same polynomial kick and translate back, so there is nothing
left to differ in. (Across a scan of amplitudes the last bit can still differ, because
xtrack reaches the same polynomial through its general ``knl`` recursion; that check
uses the few-ulp bound the skew-sextupole probe already uses.)

**The probe is a delta, deliberately.** Comparing a shifted accsim element against a
shifted xtrack element would also re-litigate the ``knl`` sign convention, which is
already pinned (and which CONVENTIONS.md records as having bitten this package once:
``Corrector`` needs ``knl=[-k]`` for ``kick_x=+k``). So each test pins the *aligned*
equivalence first and then adds the shift to both sides — the only thing left free is
the sign K1 owns.

**One model difference, stated rather than absorbed.** A displaced *bending* dipole is
not this conjugation at all: a bend rotates the reference frame through itself, so the
exit translation is not the entry one, and xtrack models the displacement as a rigid
body motion of the curved body (its misalignment header takes the straight branch only
when ``angle == 0``). The two differ by ``3.6e-5`` where the aligned maps differ by
``5.8e-9``, so accsim refuses instead of approximating —
:func:`test_a_displaced_bend_is_a_different_model_and_is_refused` is that measurement.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import Dipole, Quadrupole, ReferenceParticle, ThinQuadrupole, ThinSextupole

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0

K1L = 0.5  # thin integrated gradient [m^-1]
K1 = 1.7  # thick normalised gradient [m^-2]
K2L = 7.0  # thin integrated sextupole strength [m^-2]
LENGTH = 0.4  # thick quadrupole length [m]
DX = 3.0e-4  # a 0.3 mm horizontal misalignment
DY = -5.0e-4  # ...and a 0.5 mm vertical one, opposite in sign

# A generic probe state: every coordinate nonzero, so no term can hide.
STATE = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 1.0e-3, 2.0e-4])


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _track_xtrack(elements: list, state: np.ndarray = STATE) -> np.ndarray:
    """Track ``state`` through a one-element xtrack line, returning the 6D result."""
    line = xt.Line(elements=elements)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
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


def test_the_thin_quadrupole_shift_is_bit_for_bit_xtracks(ref: ReferenceParticle) -> None:
    """``ThinQuadrupole(k1l, dx, dy) == xt.Multipole(knl=[0, k1l], shift_x, shift_y)``.

    The gate that fixes accsim's offset sign everywhere. The *aligned* pair is checked
    first and is itself bit-identical, so the second comparison isolates the shift and
    nothing else: had accsim taken ``dx`` to mean the beam offset rather than the magnet
    position, this would miss by ``2 k1l dx``.
    """
    aligned = ThinQuadrupole(K1L).track(STATE, ref)
    assert np.array_equal(aligned, _track_xtrack([xt.Multipole(knl=[0.0, K1L])]))

    shifted = ThinQuadrupole(K1L, dx=DX, dy=DY).track(STATE, ref)
    reference = _track_xtrack([xt.Multipole(knl=[0.0, K1L], shift_x=DX, shift_y=DY)])
    assert np.abs(shifted - aligned).max() > 1e-5  # the shift really does something
    assert np.array_equal(shifted, reference)


def test_the_opposite_shift_sign_is_decisively_wrong(ref: ReferenceParticle) -> None:
    """The other branch of the probe: ``-d`` misses by exactly twice the displacement kick.

    Both planes, because ``theta_x = +k1l dx`` and ``theta_y = -k1l dy`` carry *opposite*
    signs for the same displacement — the asymmetry this package has been bitten by
    before — and a single-plane probe would leave it open.
    """
    shifted = ThinQuadrupole(K1L, dx=DX, dy=DY).track(STATE, ref)
    flipped = _track_xtrack([xt.Multipole(knl=[0.0, K1L], shift_x=-DX, shift_y=-DY)])
    expected_kick = np.array([0.0, K1L * DX, 0.0, -K1L * DY, 0.0, 0.0])
    assert np.allclose(flipped - shifted, -2.0 * expected_kick, rtol=1e-12, atol=1e-18)


def test_the_thin_sextupole_shift_agrees_too(ref: ReferenceParticle) -> None:
    """The I2 correspondence, pinned against a real tracker rather than against algebra.

    A displaced sextupole is I2's feed-down family (dipole + normal quad + skew quad +
    the sextupole itself), so this single bit-for-bit comparison checks the sign of
    every one of those terms at once against xtrack's own multipole map.
    """
    aligned = ThinSextupole(K2L).track(STATE, ref)
    assert np.array_equal(aligned, _track_xtrack([xt.Multipole(knl=[0.0, 0.0, K2L])]))

    shifted = ThinSextupole(K2L, dx=DX, dy=DY).track(STATE, ref)
    reference = _track_xtrack([xt.Multipole(knl=[0.0, 0.0, K2L], shift_x=DX, shift_y=DY)])
    assert np.abs(shifted - aligned).max() > 1e-6
    assert np.array_equal(shifted, reference)


def test_the_thick_quadrupole_shift_adds_no_error_of_its_own(ref: ReferenceParticle) -> None:
    """A displaced *thick* quad agrees as well as the aligned one does — no better, no worse.

    accsim's thick quad is a linear matrix and xtrack's is its own thick map, so the two
    already differ by ~1.6e-7 when perfectly aligned (the tolerance
    ``test_quadrupole_xtrack.py`` works at). The point here is that displacing it does not
    *increase* that: the misalignment itself is exact, and the residual is the
    pre-existing model difference. A flipped sign, by contrast, misses by 3.9e-4 — three
    thousand times larger.
    """
    aligned_dev = np.abs(
        Quadrupole(LENGTH, K1).track(STATE, ref)
        - _track_xtrack([xt.Quadrupole(length=LENGTH, k1=K1)])
    ).max()

    for dx, dy in ((DX, 0.0), (0.0, DY), (DX, DY)):
        got = Quadrupole(LENGTH, K1, dx=dx, dy=dy).track(STATE, ref)
        want = _track_xtrack([xt.Quadrupole(length=LENGTH, k1=K1, shift_x=dx, shift_y=dy)])
        assert np.abs(got - want).max() <= 2.0 * aligned_dev

    flipped = _track_xtrack([xt.Quadrupole(length=LENGTH, k1=K1, shift_x=-DX)])
    got = Quadrupole(LENGTH, K1, dx=DX).track(STATE, ref)
    assert np.abs(got - flipped).max() > 100.0 * aligned_dev


def test_a_displaced_bend_is_a_different_model_and_is_refused(ref: ReferenceParticle) -> None:
    """Where the conjugation meets curvature, accsim refuses — and here is the number.

    A bend rotates the reference frame through itself, so translating in at the entry and
    out at the exit are not the same transformation; xtrack displaces the curved body as
    a rigid object instead (its misalignment header takes the straight branch only when
    ``angle == 0``). The straight formula applied to a bend — computed here by hand,
    since the element itself will not do it — misses xtrack by ``3.6e-5`` where the
    *aligned* maps agree to ``5.8e-9``. Four thousand times the model difference is not
    a tolerance question, so :class:`~accsim.elements.dipole.Dipole` raises.
    """
    length, angle = 1.0, 0.12
    aligned_dev = np.abs(
        Dipole(length, angle).track(STATE, ref)
        - _track_xtrack([xt.Bend(length=length, angle=angle, k1=0.0)])
    ).max()

    # The straight-element conjugation, built by hand: d + M (state - d).
    bend = Dipole(length, angle)
    d = np.zeros(6)
    d[0] = DX
    straight_model = bend.matrix(ref) @ (STATE - d) + d
    xtrack_model = _track_xtrack([xt.Bend(length=length, angle=angle, k1=0.0, shift_x=DX)])
    assert np.abs(straight_model - xtrack_model).max() > 1000.0 * aligned_dev

    # ...so the element refuses rather than shipping the wrong one.
    with pytest.raises(NotImplementedError, match="cannot displace the bending Dipole"):
        Dipole(length, angle, dx=DX).track(STATE, ref)
    with pytest.raises(NotImplementedError, match="cannot displace the bending Dipole"):
        Dipole(length, angle, dy=DY).kick(ref)
    # A *straight* dipole (a pure gradient magnet) is displaced like any other element.
    assert Dipole(length, 0.0, k1=K1, dx=DX).kick(ref)[1] != 0.0


def test_the_shift_agrees_across_amplitudes_and_signs(ref: ReferenceParticle) -> None:
    """Ten states over two decades, both signs of ``x`` and ``y``, one line build.

    A single positive-quadrant probe cannot see a term that changes sign with the
    coordinate, which for the sextupole's ``x y`` structure is exactly where a wrong
    relative sign would hide.

    Here the agreement is to a few ulp rather than bit-for-bit, as it is for the
    unshifted skew sextupole (``test_skew_sextupole_xtrack.py``): both codes evaluate
    the same quadratic, but xtrack reaches it through its general ``knl`` recursion
    with an inverse-factorial table, so the last bit can differ at some amplitudes. A
    sign error would be a factor of two, not a bit.
    """
    rng = np.random.default_rng(20260817)
    line = xt.Line(elements=[xt.Multipole(knl=[0.0, 0.0, K2L], shift_x=DX, shift_y=DY)])
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")

    elem = ThinSextupole(K2L, dx=DX, dy=DY)
    for _ in range(10):
        amp = 10.0 ** rng.uniform(-4.0, -2.0)
        state = np.array(
            [
                amp * rng.uniform(-1.0, 1.0),
                1e-5 * rng.uniform(-1.0, 1.0),
                amp * rng.uniform(-1.0, 1.0),
                1e-5 * rng.uniform(-1.0, 1.0),
                1e-3 * rng.uniform(-1.0, 1.0),
                1e-4 * rng.uniform(-1.0, 1.0),
            ]
        )
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
        reference = np.array([p.x[0], p.px[0], p.y[0], p.py[0], p.zeta[0], p.delta[0]])
        got = elem.track(state, ref)
        kick = np.abs(got - state).max()
        assert kick > 1e-9  # non-vacuous: there is a real kick at this amplitude
        assert np.abs(got - reference).max() < 8.0 * np.spacing(kick)
