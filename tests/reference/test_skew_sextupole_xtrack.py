r"""J3 (part 1) cross-check: the skew sextupole's sign, against xtrack.

This file carries the **whole** sign statement for
:class:`~accsim.elements.sextupole.ThinSkewSextupole`, and that is unusual enough
to say plainly. Every other element in the package has some accsim-computed quantity
that responds to it — a tune, a chromaticity, a coupling coefficient — so its
analytic suite can pin more than shape. A skew sextupole moves *nothing* accsim
computes: :func:`~accsim.twiss.chromaticity` sums ``k2l`` over **normal** sextupoles
at ``D_x``, and :func:`~accsim.twiss.amplitude_detuning` walks octupoles. The
analytic gates are therefore structural (symplecticity, curl-free, identity
:meth:`matrix`) or shape-only (the -30 degree roll, which ``+30`` degrees satisfies
with the opposite sign), and the convention is fixed here by probe:

    ThinSkewSextupole(k2sl)  ==  xt.Multipole(ksl=[0, 0, +k2sl])

i.e. ``Delta px = +k2sl x y``, ``Delta py = +1/2 k2sl (x^2 - y^2)``. Measured
2026-08-17. As for the octupole the agreement is to a few ulp rather than
bit-for-bit — both codes compute the same quadratic, but xtrack reaches it through
its general ``ksl`` recursion with an inverse-factorial table, so the last bit can
differ. ``-k2sl`` misses by exactly twice the kick.

The probe matters because the MAD-X normal/skew asymmetry has bitten this package
before: :class:`~accsim.elements.corrector.Corrector` needs ``knl=[-k]`` for
``kick_x = +k`` but ``ksl=[+k]`` for ``kick_y = +k``. A sign that "looks right" by
analogy with the normal sextupole is not established.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import ReferenceParticle, ThinSextupole, ThinSkewSextupole

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
K2SL = 7.0  # integrated skew strength [m^-2]

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


def test_thin_skew_sextupole_is_the_positive_ksl_multipole(ref: ReferenceParticle) -> None:
    """``ThinSkewSextupole(k2sl) == xt.Multipole(ksl=[0, 0, +k2sl])``, to a few ulp.

    The one gate that fixes this element's sign anywhere in the package.
    """
    accsim = ThinSkewSextupole(K2SL).track(STATE, ref)
    reference = _track_xtrack([xt.Multipole(ksl=[0.0, 0.0, K2SL])])
    kick = np.abs(accsim - STATE).max()
    assert kick > 1e-6  # non-vacuous: there is a real kick to compare
    assert np.abs(accsim - reference).max() < 8.0 * np.spacing(kick)


def test_the_opposite_ksl_sign_is_decisively_wrong(ref: ReferenceParticle) -> None:
    """The other branch of the probe: ``-k2sl`` misses by exactly twice the kick.

    Both momentum components are checked, because the two skew components carry
    *different* functions of ``(x, y)`` and a single-component check would leave the
    relative sign — the thing the analytic roll gate also cannot see alone — open.
    """
    accsim = ThinSkewSextupole(K2SL).track(STATE, ref)
    flipped = _track_xtrack([xt.Multipole(ksl=[0.0, 0.0, -K2SL])])

    kick = np.abs(accsim - STATE)[[1, 3]]  # |Delta px|, |Delta py|
    miss = np.abs(flipped - accsim)[[1, 3]]
    assert np.all(kick > 0.0)
    assert np.allclose(miss, 2.0 * kick, rtol=1e-12)


def test_it_is_not_the_normal_sextupole_in_disguise(ref: ReferenceParticle) -> None:
    """The skew element is distinguishable from the normal one at the same strength.

    Guards the failure mode a shape-only gate cannot: an implementation that quietly
    applied the *normal* kick would still be symplectic, still curl-free, still have
    an identity :meth:`matrix`, and would still satisfy a roll identity — just the
    wrong one. xtrack's ``knl`` and ``ksl`` multipoles are the independent witnesses
    that these are two different magnets.
    """
    skew = ThinSkewSextupole(K2SL).track(STATE, ref)
    normal = ThinSextupole(K2SL).track(STATE, ref)
    assert np.abs(skew - normal).max() > 1e-6

    assert np.allclose(
        skew, _track_xtrack([xt.Multipole(ksl=[0.0, 0.0, K2SL])]), atol=1e-18, rtol=1e-12
    )
    assert np.allclose(
        normal, _track_xtrack([xt.Multipole(knl=[0.0, 0.0, K2SL])]), atol=1e-18, rtol=1e-12
    )


def test_the_kick_matches_across_amplitudes(ref: ReferenceParticle) -> None:
    """The agreement is the whole quadratic form, not one lucky point.

    Ten states spanning two decades in amplitude and both signs of ``x`` and ``y``,
    so the ``x y`` and ``x^2 - y^2`` structures are both exercised where they change
    sign — which a single positive-quadrant probe would never see.
    """
    rng = np.random.default_rng(20260817)
    sx = ThinSkewSextupole(K2SL)
    for _ in range(10):
        amp = 10.0 ** rng.uniform(-4.0, -2.0)
        state = np.array(
            [
                amp * rng.uniform(-1.0, 1.0),
                1e-5 * rng.uniform(-1.0, 1.0),
                amp * rng.uniform(-1.0, 1.0),
                1e-5 * rng.uniform(-1.0, 1.0),
                1e-3,
                2e-4,
            ]
        )
        accsim = sx.track(state, ref)
        reference = _track_xtrack([xt.Multipole(ksl=[0.0, 0.0, K2SL])], state)
        kick = np.abs(accsim - state).max()
        assert np.abs(accsim - reference).max() < max(16.0 * np.spacing(kick), 1e-18)
