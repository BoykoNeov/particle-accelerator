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

**The thick element used to need comparing by difference; P2 (ii) ended that.** A raw
comparison of ``Sextupole`` against ``xt.Sextupole`` used to leave ``1e-8`` that had
nothing to do with the sextupole — present unchanged at ``k2 = 0``, and equal to the
chromatic drift term ``-L px delta``, because the gaps between accsim's slices were the
linear drift matrix while xtrack integrated a real drift. The gaps are ``Drift.track``
now and the raw residual is ``2.7e-13``.

**What is left of it is a drift model, and it is the third-order term this time.**
``xt.Sextupole`` drifts with xtrack's *expanded* (paraxial) model, ``x += L px/(1+delta)``
— confirmed here, since the same sandwich built from explicit ``xt.Drift()`` elements
reproduces it exactly while ``xt.Drift(model="exact")`` agrees with accsim to ``3e-17``.
The difference between the two models is ``L px (px^2 + py^2)/2``, cubic in the state,
and that closed form is what the residual is gated against below rather than a
tolerance. It is the same three-code split ``docs/CONVENTIONS.md`` records for M2.

The comparison is still made at ``n_slices = 1``, because that is xtrack's own splitting:
raising the slice count converges accsim onto the exact map and therefore *away* from
xtrack, which does not slice further.

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
    """The ``1e-8`` that used to be here, and the ``3e-13`` that replaced it.

    At ``k2 = 0`` there is no sextupole left, so whatever is measured is the drift model.
    Two statements, and the first is the one P2 (ii) did not make disappear:

    1. accsim's linear *matrix* still misses ``Delta x = -L px delta`` against xtrack —
       that term is bilinear, no 6x6 can carry it, and ``matrix()`` is still what every
       optics function is built on. This is the ``1e-8``, unchanged.
    2. accsim's **tracked** thick body no longer misses it. What is left is
       ``L px (px^2 + py^2)/2``, the *cubic* angle term xtrack's expanded drift drops and
       accsim's exact one keeps — five orders smaller, and the opposite sign in the sense
       that accsim is now the one carrying more.
    """
    length = 0.5
    theirs = _track_xtrack([xt.Sextupole(length=length, k2=0.0)])
    px, py, delta = STATE[1], STATE[3], STATE[5]

    matrix_residual = theirs - Drift(length).matrix(ref) @ STATE
    assert matrix_residual[0] == pytest.approx(-length * px * delta, rel=1e-3)
    assert matrix_residual[2] == pytest.approx(-length * py * delta, rel=1e-3)
    # ...and it lives entirely in the positions: no momentum is changed by a drift.
    assert matrix_residual[1] == 0.0 and matrix_residual[3] == 0.0 and matrix_residual[5] == 0.0

    tracked_residual = Sextupole(length, 0.0).track(STATE, ref) - theirs
    angle_sq = px * px + py * py
    assert tracked_residual[0] == pytest.approx(length * px * angle_sq / 2, rel=1e-2)
    assert tracked_residual[2] == pytest.approx(length * py * angle_sq / 2, rel=1e-2)
    assert np.max(np.abs(tracked_residual)) < 1e-3 * np.max(np.abs(matrix_residual))


def test_thick_sextupole_nonlinear_content_matches_xtrack(ref: ReferenceParticle) -> None:
    """The whole thick element, compared **raw** — no difference idiom needed after P2 (ii).

    ``with(k2) - without(k2)`` was how this had to be asked while the two codes disagreed
    about the drift; it cancelled the shared geometry and left the magnet. Both maps are
    now the same composition of the same two exact factors, so the raw states are
    compared, on every coordinate including ``zeta``, and the residual is the ``2.7e-13``
    of xtrack's expanded drift rather than the ``1e-8`` of accsim's linear one.

    The difference form is kept alongside it, tightened by five orders, so that a
    regression in the drift and one in the kick would not be able to cancel.
    """
    length, k2 = 0.5, 12.0
    theirs = _track_xtrack([xt.Sextupole(length=length, k2=k2)])
    ours = Sextupole(length, k2, n_slices=1).track(STATE, ref)

    assert np.max(np.abs(ours - Drift(length).track(STATE, ref))) > 1e-6  # a real kick
    assert np.allclose(ours, theirs, rtol=0.0, atol=1e-12)

    xt_kick = theirs - _track_xtrack([xt.Sextupole(length=length, k2=0.0)])
    acc_kick = ours - Sextupole(length, 0.0, n_slices=1).track(STATE, ref)
    assert np.allclose(acc_kick, xt_kick, rtol=0.0, atol=1e-12)


def test_thick_sextupole_lengthens_the_path_of_a_kicked_trajectory_like_xtrack(
    ref: ReferenceParticle,
) -> None:
    r"""The ``zeta`` row: what accsim used to be blind to, and now measures with xtrack.

    Deflecting a particle lengthens its path, so the drift after the kick takes longer to
    cross and ``zeta`` moves when ``k2`` is switched on. accsim's linear gaps carried only
    ``R56 delta``, with no ``px`` dependence at all, so this number was exactly zero on
    one side of the comparison and ``3e-10`` on the other — an *explained* gap, but a gap.

    Both codes now report it, they agree to seven figures, and the closed form

        Delta zeta = -(L/4) [ (px^2 + py^2)_after - (px^2 + py^2)_before ]

    predicts it — the ``L/2`` of the second half-drift over the ``2`` of ``pz + E/E0``.
    """
    length, k2 = 0.5, 12.0
    xt_on = _track_xtrack([xt.Sextupole(length=length, k2=k2)])
    xt_off = _track_xtrack([xt.Sextupole(length=length, k2=0.0)])
    acc_on = Sextupole(length, k2, n_slices=1).track(STATE, ref)
    acc_off = Sextupole(length, 0.0, n_slices=1).track(STATE, ref)

    before = STATE[1] ** 2 + STATE[3] ** 2
    after = acc_on[1] ** 2 + acc_on[3] ** 2
    predicted = -(length / 4.0) * (after - before)

    assert abs(predicted) > 1e-12  # non-vacuous: the kick really does deflect
    assert acc_on[4] - acc_off[4] == pytest.approx(predicted, rel=1e-2)
    assert acc_on[4] - acc_off[4] == pytest.approx(xt_on[4] - xt_off[4], rel=1e-5)


def test_thick_sextupole_agrees_with_a_multipole_sandwich(ref: ReferenceParticle) -> None:
    """accsim's drift-kick-drift is xtrack's too, element for element — and which drift.

    Building the split explicitly in xtrack — ``Drift(L/2) . Multipole . Drift(L/2)`` —
    reproduces accsim's single-slice thick map, and *which* ``xt.Drift`` is used decides
    at what order. With ``model="exact"`` the two agree to ``3e-17``, i.e. to the last
    bits of a double on every coordinate: same kick, same drift, same composition.

    With xtrack's **default** drift the same sandwich lands on ``2.7e-13`` — and, more
    usefully, lands on exactly what ``xt.Sextupole`` gives. That identifies the model
    inside xtrack's thick multipole as the expanded one, which is the whole content of
    the residual gated in the tests above.
    """
    length, k2 = 0.5, 12.0
    accsim = Sextupole(length, k2, n_slices=1).track(STATE, ref)

    exact = _track_xtrack(
        [
            xt.Drift(length=length / 2, model="exact"),
            xt.Multipole(knl=[0.0, 0.0, k2 * length]),
            xt.Drift(length=length / 2, model="exact"),
        ]
    )
    assert np.allclose(accsim, exact, rtol=0.0, atol=1e-15)

    expanded = _track_xtrack(
        [
            xt.Drift(length=length / 2),
            xt.Multipole(knl=[0.0, 0.0, k2 * length]),
            xt.Drift(length=length / 2),
        ]
    )
    thick = _track_xtrack([xt.Sextupole(length=length, k2=k2)])
    assert np.array_equal(expanded, thick)  # xt.Sextupole drifts with the default model
    assert np.max(np.abs(accsim - expanded)) > 1e-14  # and that model is not the exact one
