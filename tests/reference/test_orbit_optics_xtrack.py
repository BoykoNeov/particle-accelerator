"""I3 cross-check: optics on a steered orbit, against xtrack's own twiss.

xtrack twisses about the closed orbit it finds by its own iterative tracking, so
``tw.betx`` and ``tw.dqx`` on a steered machine are exactly the quantity I3 adds —
computed by a code that shares none of accsim's machinery. The analytic suite
derives the beta-beat closed form and gates the convergence order; what it cannot
do is confirm that the *whole* answer, on a bendy ring with eight sextupoles, is
what an independent tracker reports.

**One modelling difference has to be stated first, because it sets what these
comparisons can say.** accsim's :class:`~accsim.elements.dipole.Dipole` and
:class:`~accsim.elements.quadrupole.Quadrupole` are *exactly* linear maps, so an
off-axis orbit changes nothing about them. xtrack's ``Bend`` and ``Quadrupole``
are exact nonlinear maps, whose Jacobian at a 1.25 mm offset is genuinely not the
on-axis one. That difference is **first order in the orbit** and belongs to
accsim's element models, not to I3:
:func:`test_accsims_linear_elements_do_not_feed_down_off_axis` isolates it at
``k2l = 0``, where accsim has nothing to feed down at all and yet xtrack's beta
still moves by 6.4e-4 relative.

So the beta gate is a **with-minus-without-sextupole difference**, the same device
J1 used for its chromaticity cross-check: the bend nonlinearity is common to both
terms and cancels, leaving the sextupole feed-down alone. It is a strong form of
the comparison rather than a weak one — accsim's design-orbit route predicts
**exactly zero** change there (a sextupole's ``matrix()`` is a drift, bit for bit),
so the gate asks accsim to reproduce an effect its previous answer said did not
exist.

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

# Measured 2026-08-10: accsim reproduces xtrack's sextupole-induced beta change to
# 1.35e-3 of that change (8.9e-6 m on a 6.6e-3 m effect). The leftover is the bend
# nonlinearity above, which cancels only to the extent that the orbit is the same
# with and without the sextupole -- the feed-down dipole moves it slightly.
BETA_CHANGE_RTOL = 5e-3

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


def test_accsims_linear_elements_do_not_feed_down_off_axis() -> None:
    """The modelling difference, isolated and measured rather than absorbed.

    With ``k2l = 0`` and the ring steered by 1.25 mm, accsim has **nothing** to feed
    down: its bends and quadrupoles are exactly linear maps, so the on-orbit optics
    is the design optics to 4e-11. xtrack's bends are exact nonlinear maps, so its
    beta moves by 6.4e-4 relative — an effect that is first order in the orbit and
    belongs to accsim's element models, not to anything I3 does.

    Recorded here so the 1.35e-3 residual of the beta gate below is understood
    rather than mistaken for a feed-down error, and so that a future milestone
    giving the bends their real off-axis map has a number to improve on.
    """
    lat = _accsim(0.0, KICK)
    assert abs(propagate_orbit_nonlinear(lat)[0][0]) > 1e-4  # genuinely steered

    on_orbit, design = _betx_on_orbit(0.0, KICK), _betx_design(0.0, KICK)
    assert np.abs(on_orbit - design).max() < 1e-9  # accsim: no feed-down at all

    beat_vs_xtrack = np.abs(on_orbit / np.array(_twiss(0.0, KICK).betx) - 1.0).max()
    assert 1e-4 < beat_vs_xtrack < 3e-3  # xtrack's own off-axis nonlinearity


def test_the_sextupole_induced_beta_change_matches_xtrack() -> None:
    """**The headline cross-check.** The beat accsim now reports is the one xtrack sees.

    Both codes are asked the same question — *what does adding the sextupoles do to*
    ``beta(s)`` *on this steered ring?* — and the difference removes the bend
    nonlinearity isolated above, which is identical in both terms.

    The design-orbit route answers **exactly zero**, bit for bit, because a
    sextupole's ``matrix()`` is a drift: adding one changes no matrix anywhere. So
    this is not a 4x improvement on an existing estimate, it is an effect that was
    previously invisible. xtrack puts it at 6.6e-3 m (0.22 % of beta) and accsim's
    on-orbit route reproduces it to 8.9e-6 m, 1.35e-3 of the effect.
    """
    acc_change = _betx_on_orbit(K2L, KICK) - _betx_on_orbit(0.0, KICK)
    des_change = _betx_design(K2L, KICK) - _betx_design(0.0, KICK)
    xt_change = np.array(_twiss(K2L, KICK).betx) - np.array(_twiss(0.0, KICK).betx)

    scale = np.abs(xt_change).max()
    assert scale > 1e-3  # non-vacuous: the sextupoles really do move beta

    # The design orbit says the change is identically zero -- not small, zero.
    assert np.abs(des_change).max() == 0.0

    assert np.abs(acc_change - xt_change).max() < BETA_CHANGE_RTOL * scale


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
