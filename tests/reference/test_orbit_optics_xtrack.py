"""I3 cross-check: optics on a steered orbit, against xtrack's own twiss.

xtrack twisses about the closed orbit it finds by its own iterative tracking, so
``tw.betx`` and ``tw.dqx`` on a steered machine are exactly the quantity I3 adds —
computed by a code that shares none of accsim's machinery. The analytic suite
derives the beta-beat closed form and gates the convergence order; what it cannot
do is confirm that the *whole* answer, on a bendy ring with eight sextupoles, is
what an independent tracker reports.

**A modelling difference used to have to be stated first, and L3 closed it.** When
this file was written accsim's :class:`~accsim.elements.dipole.Dipole` and
:class:`~accsim.elements.quadrupole.Quadrupole` were *exactly* linear maps, so an
off-axis orbit changed nothing about them, while xtrack's are exact nonlinear maps
whose Jacobian at a 1.25 mm offset is genuinely not the on-axis one. That difference
was first order in the orbit and worth 6.4e-4 in beta. L1, L2 and L3 gave those
elements their exact maps, and
:func:`test_the_bends_off_axis_feed_down_now_matches_xtrack` now measures the same
quantity at **5.4e-10** — with the old 6.4e-4 having *moved*, undiminished, onto
accsim's design-orbit route, which is where a linear-optics blindness belongs.

The beta gate is still a **with-minus-without-sextupole difference**, the same device
J1 used for its chromaticity cross-check, and it is a strong form of the comparison
rather than a weak one: accsim's design-orbit route predicts **exactly zero** change
there (a sextupole's ``matrix()`` is a drift, bit for bit), so the gate asks accsim to
reproduce an effect its previous answer said did not exist. What has changed is that
the difference no longer *cancels* anything — the undifferenced tables agree on their
own to 1.1e-9, asserted alongside — so it is now the sharper question rather than a
necessary one, and its bound has been tightened by three and a half orders to match.

Element equivalences are the ones I1 and J1 established, reused rather than
re-probed::

    accsim Corrector(kick_x=+k)  ==  xt.Multipole(knl=[-k])
    accsim ThinSextupole(k2l)    ==  xt.Multipole(knl=[0, 0, +k2l])

Marked ``reference``: skips when xtrack or its JIT is absent. Four ``xt.Line``
builds, cached across the file — each costs ~12 s of C-kernel compilation and
nothing is ever cached by xobjects itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Corrector,
    Dipole,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinSextupole,
    chromaticity,
    chromaticity_on_orbit,
    closed_twiss,
    propagate_orbit_nonlinear,
    propagate_twiss,
    propagate_twiss_on_orbit,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
LQ = 0.3  # quad length [m]
K1 = 1.2  # quad gradient [m^-2]
BEND_L = 1.0  # dipole length [m]
N_CELLS = 8
BEND_ANG = 2.0 * math.pi / (2 * N_CELLS)  # two bends per cell -> a closed ring
K2L = 0.6  # integrated sextupole strength [m^-2], one per cell at dispersion
KICK = 6.0e-4  # steerer angle [rad] -> a ~1.25 mm orbit

# Measured 2026-08-10 on the unsteered ring: the two codes agree on beta to 9.3e-10
# relative and on dqx to 1e-7. That is the floor everything below is read against —
# it says the disagreements that follow are the orbit, not the ring description.
FLAT_BETA_RTOL = 1e-8
FLAT_DQ_ATOL = 1e-4

# Measured 2026-08-17 (L3): accsim reproduces xtrack's sextupole-induced beta change to
# 2.8e-7 of that change (1.8e-9 m on a 6.6e-3 m effect). It was 1.35e-3 (8.9e-6 m) while
# the leftover was the bend nonlinearity; with the bends' exact map in place that term
# is gone from both sides and what remains is the differencing floor.
BETA_CHANGE_RTOL = 1e-6  # was 5e-3, sized for a bend-model gap L3 closed (measured 2.8e-7)

_LINES: dict[tuple[float, float], object] = {}


def _accsim(k2l: float, kick: float) -> Lattice:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    els: list = [Corrector(kick_x=kick, name="steerer")]
    for i in range(N_CELLS):
        els += [
            Quadrupole(LQ, K1),
            ThinSextupole(k2l, name=f"sx_{i}"),
            Dipole(BEND_L, BEND_ANG),
            Quadrupole(LQ, -K1),
            Dipole(BEND_L, BEND_ANG),
        ]
    return Lattice(els, ref)


def _twiss(k2l: float, kick: float):
    """xtrack's 4d twiss of the same ring, built once per (k2l, kick) and cached."""
    key = (k2l, kick)
    if key in _LINES:
        return _LINES[key]
    bend = {
        "length": BEND_L,
        "angle": BEND_ANG,
        "k0": BEND_ANG / BEND_L,
        "k1": 0.0,
        "model": "rot-kick-rot",
    }
    els = [xt.Multipole(knl=[-kick])]
    for _ in range(N_CELLS):
        els += [
            xt.Quadrupole(length=LQ, k1=K1),
            xt.Multipole(knl=[0.0, 0.0, k2l]),
            xt.Bend(**bend),
            xt.Quadrupole(length=LQ, k1=-K1),
            xt.Bend(**bend),
        ]
    line = xt.Line(elements=els)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        tw = line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    _LINES[key] = tw
    return tw


def _betx_on_orbit(k2l: float, kick: float) -> np.ndarray:
    return np.array([t.beta_x for t in propagate_twiss_on_orbit(_accsim(k2l, kick))])


def _betx_design(k2l: float, kick: float) -> np.ndarray:
    lat = _accsim(k2l, kick)
    return np.array([t.beta_x for t in propagate_twiss(lat, closed_twiss(lat))])


def test_the_unsteered_ring_is_the_control() -> None:
    """Both codes describe this machine identically until the beam moves off axis.

    Without this the comparisons below would be unreadable: a disagreement on the
    steered ring could be the orbit physics I3 adds or simply two codes modelling a
    bend differently. Here the orbit is exactly zero, the sextupole is live but on
    axis, and the agreement is at round-off — so everything that follows is the
    orbit.
    """
    tw = _twiss(K2L, 0.0)
    lat = _accsim(K2L, 0.0)
    assert np.abs(propagate_orbit_nonlinear(lat)).max() == 0.0

    betx = np.array(tw.betx)
    assert _betx_on_orbit(K2L, 0.0) == pytest.approx(betx, rel=FLAT_BETA_RTOL)
    # ...and on axis the new function is the old one, so the design route agrees too.
    assert _betx_design(K2L, 0.0) == pytest.approx(betx, rel=FLAT_BETA_RTOL)

    for computed, reference in ((chromaticity_on_orbit(lat), tw.dqx), (chromaticity(lat), tw.dqx)):
        assert computed[0] == pytest.approx(reference, abs=FLAT_DQ_ATOL)


def test_the_bends_off_axis_feed_down_now_matches_xtrack() -> None:
    """The modelling difference this test was written to record, **closed by L3**.

    It used to say: with ``k2l = 0`` and the ring steered by 1.25 mm, accsim has
    *nothing* to feed down — its bends and quadrupoles were exactly linear maps, so
    the on-orbit optics equalled the design optics to 4e-11, while xtrack's exact
    bends moved beta by 6.4e-4 relative. It closed with the hope that "a future
    milestone giving the bends their real off-axis map has a number to improve on".

    L1, L2 and L3 are that milestone, and the number moved from **6.4e-4 to 5.4e-10**
    — six orders. What is worth noticing is *where the 6.4e-4 went*: it did not shrink,
    it **moved to the design route**, which now disagrees with xtrack by exactly the
    amount the on-orbit route used to. That is the correct place for it. The design
    optics is built on ``matrix()``, the feed-down terms are bilinear, and no 6x6 can
    hold them; being blind there is a property of linear optics, not a defect.

    So the assertion is inverted from "accsim has no feed-down" to "accsim's feed-down
    is xtrack's", and the design route is kept in the test as the measured contrast.
    """
    lat = _accsim(0.0, KICK)
    assert abs(propagate_orbit_nonlinear(lat)[0][0]) > 1e-4  # genuinely steered

    on_orbit, design = _betx_on_orbit(0.0, KICK), _betx_design(0.0, KICK)
    xt_betx = np.array(_twiss(0.0, KICK).betx)

    # The on-orbit route is now xtrack's own answer.
    assert np.abs(on_orbit / xt_betx - 1.0).max() < 1e-8

    # ...and the old disagreement has migrated to the design route, undiminished.
    assert np.abs(design / xt_betx - 1.0).max() == pytest.approx(6.365e-4, rel=1e-2)
    # Non-vacuous: the two accsim routes really have parted company off axis.
    assert np.abs(on_orbit - design).max() > 1e-3

    # First order in the orbit, which is what says it is feed-down and not a constant
    # modelling offset: halving the steerer halves the design route's error.
    halved = np.abs(_betx_design(0.0, KICK / 2) / np.array(_twiss(0.0, KICK / 2).betx) - 1.0).max()
    assert np.abs(design / xt_betx - 1.0).max() / halved == pytest.approx(2.0, rel=2e-2)


def test_the_sextupole_induced_beta_change_matches_xtrack() -> None:
    """**The headline cross-check.** The beat accsim now reports is the one xtrack sees.

    Both codes are asked the same question — *what does adding the sextupoles do to*
    ``beta(s)`` *on this steered ring?* — and the difference removes the bend
    nonlinearity isolated above, which is identical in both terms.

    The design-orbit route answers **exactly zero**, bit for bit, because a
    sextupole's ``matrix()`` is a drift: adding one changes no matrix anywhere. So
    this is not a 4x improvement on an existing estimate, it is an effect that was
    previously invisible. xtrack puts it at 6.6e-3 m (0.22 % of beta) and accsim's
    on-orbit route reproduces it to **1.8e-9 m, 2.8e-7 of the effect** — it was
    8.9e-6 m, 1.35e-3 of the effect, before L1-L3 gave the elements their exact maps.

    **The difference construction is no longer load-bearing, and that is the news.**
    It existed to cancel a bend nonlinearity present in xtrack's terms and absent from
    accsim's; with that gap closed, the *undifferenced* beta tables agree to 1.1e-9
    relative, which is asserted below as well. The difference is kept because it is
    still the sharper question — it isolates what the sextupoles do — but it is now a
    convenience rather than a necessity, and the bound has been tightened by three and
    a half orders to match. Leaving it at ``5e-3`` would have hidden any future
    regression inside a tolerance sized for a model gap that no longer exists.
    """
    acc_change = _betx_on_orbit(K2L, KICK) - _betx_on_orbit(0.0, KICK)
    des_change = _betx_design(K2L, KICK) - _betx_design(0.0, KICK)
    xt_change = np.array(_twiss(K2L, KICK).betx) - np.array(_twiss(0.0, KICK).betx)

    # The two tables line up boundary for boundary only because the element lists
    # are in lockstep and xt.Multipole is zero-length like Corrector/ThinSextupole.
    # Asserted rather than assumed: adding an element to one list and not the other
    # would compare different positions and read as a physics failure.
    assert len(acc_change) == 1 + len(_accsim(K2L, KICK).elements)
    assert len(xt_change) == len(acc_change)

    scale = np.abs(xt_change).max()
    assert scale > 1e-3  # non-vacuous: the sextupoles really do move beta

    # The design orbit says the change is identically zero -- not small, zero.
    assert np.abs(des_change).max() == 0.0

    assert np.abs(acc_change - xt_change).max() < BETA_CHANGE_RTOL * scale

    # And the tables the difference was built from now agree on their own, which is
    # what says the cancellation above is no longer doing any work.
    for k2l in (0.0, K2L):
        direct = _betx_on_orbit(k2l, KICK) / np.array(_twiss(k2l, KICK).betx) - 1.0
        assert np.abs(direct).max() < 1e-8


def test_chromaticity_on_orbit_tracks_xtrack_where_the_design_orbit_drifts() -> None:
    """The reported chromaticity follows the machine, in both planes.

    Unlike beta, chromaticity does not need a difference: the design-orbit answer is
    wrong by far more than the bend-nonlinearity floor, so the raw comparison is
    already decisive. Measured 2026-08-10 on the steered ring:

        dqx: xtrack +2.6529347, on-orbit +2.6529020 (3.3e-5), design +2.6546373 (1.7e-3)
        dqy: xtrack -4.5204588, on-orbit -4.5205851 (1.3e-4), design -4.5178598 (2.6e-3)

    i.e. 52x closer in ``x`` and 21x in ``y``. The remaining error is the same
    first-order bend nonlinearity — ``k2l = 0`` on this ring leaves 5.4e-5 in
    ``dqx`` by itself.
    """
    tw = _twiss(K2L, KICK)
    lat = _accsim(K2L, KICK)
    on_orbit, design = chromaticity_on_orbit(lat), chromaticity(lat)

    for i, reference in ((0, tw.dqx), (1, tw.dqy)):
        design_err = abs(design[i] - reference)
        on_orbit_err = abs(on_orbit[i] - reference)
        assert design_err > 1e-3  # the design orbit is decisively wrong...
        assert on_orbit_err < design_err / 10.0  # ...and this is an order better
        assert on_orbit_err < 5e-4
