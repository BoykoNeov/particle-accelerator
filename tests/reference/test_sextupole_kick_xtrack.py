r"""J1 cross-check: the sextupole's nonlinear kick against xtrack's tracking.

The analytic suite pins the kick's coefficient through feed-down chromaticity
(``tests/analytic/test_sextupole_kick.py``). What it cannot pin on its own is the
**sign convention** — the relation between accsim's ``k2`` and the ``knl`` of a
MAD-X / Xsuite multipole. Per the G1 rule, a convention that accsim also derives is
not an independent reference, so the sign is established here **by probe**: both
candidates are tracked and the wrong one is asserted to be decisively wrong, not
merely worse.

Probe result (this file's premise):

    ThinSextupole(k2l)  ==  xt.Multipole(knl=[0, 0, +k2l])

and the match is **bit-for-bit**, not approximate — the thin kick touches only
``(px, py)``, so there is no drift model for the two codes to disagree about. The
opposite sign is off by twice the kick.

**Why the thick element is compared by difference.** A raw comparison of
``Sextupole`` against ``xt.Sextupole`` leaves a residual of order ``1e-8`` that has
nothing to do with the sextupole: it is present **unchanged at ``k2 = 0``** and
equals the first-order chromatic drift term ``-L px delta`` — xtrack integrates the
exact drift ``x += L px / sqrt((1+delta)^2 - px^2 - py^2)`` while accsim's linear
map carries ``x += L px``, an omission that belongs to the linear drift and is
documented there, not here. Switching ``k2`` on and off at fixed geometry cancels it
term-for-term and isolates exactly the nonlinear content, the same difference idiom
``test_sextupole_xtrack.py`` uses for feed-down. That the residual does *not* shrink
as ``n_slices`` grows is also expected: xtrack's thick sextupole is itself a
single-kick drift-kick-drift split, so accsim's ``n_slices = 1`` is its closest
match and larger slice counts converge on the exact map, away from xtrack's default.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import Drift, ReferenceParticle, Sextupole, ThinSextupole

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
K2L = 3.0  # integrated thin strength [m^-2]

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


def test_thin_sextupole_is_exactly_the_positive_knl_multipole(ref: ReferenceParticle) -> None:
    """The convention, by probe: ``ThinSextupole(k2l) == xt.Multipole(knl=[0,0,+k2l])``.

    Exact to the last bit — a thin kick has no length, so the two codes share no
    drift model that could disagree. This is the tightest cross-check in the
    package and it is what fixes accsim's ``k2`` sign against MAD-X's.
    """
    accsim = ThinSextupole(K2L).track(STATE, ref)
    reference = _track_xtrack([xt.Multipole(knl=[0.0, 0.0, K2L])])
    assert np.allclose(accsim, reference, atol=0.0, rtol=0.0)


def test_the_opposite_knl_sign_is_decisively_wrong(ref: ReferenceParticle) -> None:
    """The other branch of the probe: ``-k2l`` misses by twice the kick, not by a whisker.

    Recorded because the MAD-X normal/skew asymmetry has bitten this package before
    (``Corrector``: ``kick_x = +k`` is ``knl=[-k]`` but ``kick_y = +k`` is
    ``ksl=[+k]``). A sign established by "it looks right" is not established.
    """
    accsim = ThinSextupole(K2L).track(STATE, ref)
    flipped = _track_xtrack([xt.Multipole(knl=[0.0, 0.0, -K2L])])

    kick = np.abs(accsim - STATE)[[1, 3]]  # |Delta px|, |Delta py|
    miss = np.abs(flipped - accsim)[[1, 3]]
    assert np.all(kick > 0.0)
    assert np.allclose(miss, 2.0 * kick, rtol=1e-12)


def test_thin_kick_matches_xtrack_across_amplitudes(ref: ReferenceParticle) -> None:
    """The agreement is the whole quadratic form, not one lucky point."""
    for x0, y0 in [(3e-3, 1e-3), (-2e-3, 2e-3), (1e-3, -4e-3), (5e-3, 0.0), (0.0, 5e-3)]:
        state = np.array([x0, 1e-4, y0, -5e-5, 1e-3, 1e-4])
        accsim = ThinSextupole(K2L).track(state, ref)
        reference = _track_xtrack([xt.Multipole(knl=[0.0, 0.0, K2L])], state)
        assert np.allclose(accsim, reference, atol=0.0, rtol=0.0)


def test_thick_residual_at_zero_strength_is_the_chromatic_drift(ref: ReferenceParticle) -> None:
    """Attribute the thick element's ~1e-8 residual: it is the drift, not the sextupole.

    At ``k2 = 0`` there is no sextupole left, yet the full residual is still there —
    and it matches the leading term of xtrack's exact drift minus accsim's linear
    one, ``Delta x = -L px delta``. Establishing this is what licenses the
    difference method in the next test.
    """
    length = 0.5
    residual = (
        _track_xtrack([xt.Sextupole(length=length, k2=0.0)]) - Drift(length).matrix(ref) @ STATE
    )

    px, py, delta = STATE[1], STATE[3], STATE[5]
    assert residual[0] == pytest.approx(-length * px * delta, rel=1e-3)
    assert residual[2] == pytest.approx(-length * py * delta, rel=1e-3)
    # ...and it lives entirely in the positions: no momentum is changed by a drift.
    assert residual[1] == 0.0 and residual[3] == 0.0 and residual[5] == 0.0


def test_thick_sextupole_nonlinear_content_matches_xtrack(ref: ReferenceParticle) -> None:
    """The isolated kick of the thick element agrees with xtrack's thick sextupole.

    ``with(k2) - without(k2)`` at fixed geometry cancels the shared drift model
    (previous test) and leaves the nonlinear content of the magnet. Compared at
    ``n_slices = 1`` because that is xtrack's own splitting.

    The surviving ``2e-4`` relative residual is the *second-order* shadow of the
    same drift difference: the two codes' half-drifts deliver the particle to the
    kick at slightly different ``x``, so the kick itself differs slightly, and that
    difference is then drifted again. It is not a disagreement about the sextupole —
    the kick alone is bit-exact (see the thin tests above).
    """
    length, k2 = 0.5, 12.0
    xt_kick = _track_xtrack([xt.Sextupole(length=length, k2=k2)]) - _track_xtrack(
        [xt.Sextupole(length=length, k2=0.0)]
    )
    acc = Sextupole(length, k2, n_slices=1).track(STATE, ref) - Sextupole(
        length, 0.0, n_slices=1
    ).track(STATE, ref)

    transverse = [0, 1, 2, 3]
    assert np.max(np.abs(xt_kick[transverse])) > 1e-6  # non-vacuous: a kick to compare
    assert np.allclose(acc[transverse], xt_kick[transverse], rtol=1e-3, atol=1e-12)


def test_thick_sextupole_omits_the_path_lengthening_of_a_kicked_trajectory(
    ref: ReferenceParticle,
) -> None:
    r"""The ``zeta`` difference is accsim's linear drift, quantified — not left as slop.

    Deflecting a particle lengthens its path: the exact drift advances
    ``zeta`` by ``-(L/2)(px^2 + py^2)`` to leading order, so turning ``k2`` on shifts
    ``zeta`` in xtrack. accsim's linear drift carries only ``R56 delta``, with no
    ``px`` dependence at all, so its ``zeta`` does not move. The gap is therefore
    predictable, and predicting it is what turns an unexplained ``3e-10`` into a
    known omission of the linear map (flagged in ``docs/CONVENTIONS.md``; it is the
    same order the transverse residual above comes from).
    """
    length, k2 = 0.5, 12.0
    xt_on = _track_xtrack([xt.Sextupole(length=length, k2=k2)])
    xt_off = _track_xtrack([xt.Sextupole(length=length, k2=0.0)])
    acc_on = Sextupole(length, k2, n_slices=1).track(STATE, ref)
    acc_off = Sextupole(length, 0.0, n_slices=1).track(STATE, ref)

    # accsim: zeta is blind to the kick.
    assert acc_on[4] - acc_off[4] == pytest.approx(0.0, abs=1e-18)

    # xtrack: the second half-drift sees the kicked momenta, so zeta moves by
    # -(L/2) * [ (px^2 + py^2)_after - (px^2 + py^2)_before ] / 2.
    before = STATE[1] ** 2 + STATE[3] ** 2
    after = acc_on[1] ** 2 + acc_on[3] ** 2
    predicted = -(length / 2.0) * (after - before) / 2.0
    assert xt_on[4] - xt_off[4] == pytest.approx(predicted, rel=2e-2)


def test_thick_sextupole_agrees_with_a_multipole_sandwich(ref: ReferenceParticle) -> None:
    """accsim's drift-kick-drift is xtrack's too, element for element.

    Building the split explicitly in xtrack — ``Drift(L/2) . Multipole . Drift(L/2)``
    — reproduces accsim's single-slice thick map up to the drift model alone, which
    is the sharpest available statement about the *composition* (as opposed to the
    kick, which is pinned exactly above).

    The momenta are not bit-exact even though the kick is: xtrack's exact half-drift
    hands the multipole a slightly different ``x``, and the kick is a function of
    ``x``. The miss is ``~k2l * x * Delta_x`` with ``Delta_x ~ (L/2) px delta``, i.e.
    parts in ``1e6`` here.
    """
    length, k2 = 0.5, 12.0
    sandwich = _track_xtrack(
        [
            xt.Drift(length=length / 2),
            xt.Multipole(knl=[0.0, 0.0, k2 * length]),
            xt.Drift(length=length / 2),
        ]
    )
    accsim = Sextupole(length, k2, n_slices=1).track(STATE, ref)

    assert np.allclose(accsim[[1, 3]], sandwich[[1, 3]], rtol=5e-6, atol=0.0)
    assert accsim[5] == sandwich[5]  # delta: untouched by both
    # Positions: the chromatic drift term, ~1e-8 at this amplitude.
    assert np.allclose(accsim[[0, 2]], sandwich[[0, 2]], atol=2e-8)
