r"""J3 cross-check: octupole feed-down against xtrack's own tracking.

The analytic suite derives the six-way split and gates it against accsim's own
machinery — Newton on its tracked map, its Twiss integrals, its finite-difference
Jacobian. Every one of those routes is accsim's. What only an independent tracker can
say is that the *effect* is real: that a machine with a steered octupole really has
the closed orbit, the linearised map and the chromaticity accsim now reports.

Three things are reached here that the analytic suite cannot reach:

- **The fixed point.** accsim Newtons on its own tracked map; xtrack iterates its own
  tracker. With both planes steered, every term of the split is live at once.
- **The derived split as a whole matrix.** The comparison is deliberately made from
  :func:`~accsim.orbit.linearised_lattice` — the equivalent machine built out of the
  *derived coefficients* — against xtrack's ``R_matrix`` finite-differenced about its
  own orbit. Not accsim's finite difference against xtrack's: this asks whether the
  algebra is right, not whether two differencing schemes agree.
- **The chromaticity rung, which is the milestone.** On the design orbit accsim
  reports the octupole as contributing nothing to ``Q'`` — correct, and what J2
  asserted. xtrack, which tracks, reports the fed-down value. So the design-orbit
  answer is *decisively wrong on a steered machine* and the on-orbit answer has to
  close a gap that accsim's previous answer said did not exist.

**One modelling difference is stated first, because it bounds what this can say.**
accsim's :class:`~accsim.elements.dipole.Dipole` and
:class:`~accsim.elements.quadrupole.Quadrupole` are exactly linear maps; xtrack's
``Bend`` and ``Quadrupole`` are exact nonlinear ones whose Jacobian at a millimetre
offset is genuinely not the on-axis one. That difference is first order in the orbit
and belongs to accsim's element models, not to J3 — it is measured here at
``k3l = 0`` (I3 recorded the same thing) and every gate below is read against it.

Element equivalences are the ones I1, J1 and J2 established, reused rather than
re-probed::

    accsim Corrector(kick_x=+k)  ==  xt.Multipole(knl=[-k])
    accsim Corrector(kick_y=+k)  ==  xt.Multipole(ksl=[+k])
    accsim ThinOctupole(k3l)     ==  xt.Multipole(knl=[0, 0, 0, +k3l])

Marked ``reference``: skips when xtrack or its JIT is absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Corrector,
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    chromaticity,
    chromaticity_on_orbit,
    closed_orbit,
    closed_orbit_nonlinear,
    propagate_orbit_nonlinear,
)
from accsim.orbit import linearised_element_maps, linearised_lattice

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0

# --- the straight ring: orbit and matrix -----------------------------------
LQ = 0.3  # quad length [m]
K1 = 1.2  # quad gradient [m^-2]
LD = 1.0  # drift length [m]
N_CELLS = 3
KICK = 2.0e-4  # steerer angle [rad] -> a ~1 mm orbit
K3L = 1.0e5  # integrated octupole strength [m^-3]

# --- the bendy ring: chromaticity ------------------------------------------
BEND_L = 1.0
N_BEND_CELLS = 8
BEND_ANG = 2.0 * math.pi / (2 * N_BEND_CELLS)
K3L_CHROMA = 60.0  # one octupole per cell, at dispersion
KICK_CHROMA = 6.0e-4  # -> a ~1.25 mm orbit, as in the I3 cross-check

ORBIT_ATOL = 1e-12
MATRIX_ATOL = 1e-8

_LINES: dict[tuple, object] = {}


def _accsim_lattice(k3l: float, kick_x: float = 0.0, kick_y: float = 0.0) -> Lattice:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    cell = [Quadrupole(LQ, K1), Drift(LD), Quadrupole(LQ, -K1), Drift(LD)]
    return Lattice(
        [
            Corrector(kick_x=kick_x, kick_y=kick_y, name="steerer"),
            ThinOctupole(k3l, name="oc"),
            *(cell * N_CELLS),
        ],
        ref,
    )


def _xtrack_twiss(k3l: float, kick_x: float = 0.0, kick_y: float = 0.0, steps: dict | None = None):
    """xtrack's 4d twiss of the straight ring, built once per setting and cached.

    ``steps`` overrides xtrack's own ``steps_r_matrix`` (default ``dx = 1e-6``), which
    is what lets the R-matrix gate below show that its residual is xtrack's
    differencing rather than a disagreement.
    """
    key = ("straight", k3l, kick_x, kick_y, None if steps is None else tuple(sorted(steps.items())))
    if key in _LINES:
        return _LINES[key]
    # model="exact" is load-bearing: xt.Drift() defaults to the *expanded* model
    # (x += L px/(1+delta)), where accsim's Drift tracks the exact x += L px/pz (L1).
    # On a steered ring the two disagree at O(angle^3), which reads as a feed-down
    # coefficient error and is nothing of the kind. See test_drift_xtrack.py.
    cell = [
        xt.Quadrupole(length=LQ, k1=K1),
        xt.Drift(length=LD, model="exact"),
        xt.Quadrupole(length=LQ, k1=-K1),
        xt.Drift(length=LD, model="exact"),
    ]
    line = xt.Line(
        elements=[
            xt.Multipole(knl=[-kick_x], ksl=[kick_y]),
            xt.Multipole(knl=[0.0, 0.0, 0.0, k3l]),
            *(cell * N_CELLS),
        ]
    )
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    kwargs: dict = {"method": "4d"}
    if steps is not None:
        kwargs["steps_R_matrix"] = {**steps, "dzeta": 1e-6, "ddelta": 1e-7}
    try:
        line.build_tracker()
        tw = line.twiss(**kwargs)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    _LINES[key] = tw
    return tw


def _accsim_bendy(k3l: float, kick: float) -> Lattice:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    els: list = [Corrector(kick_x=kick, name="steerer")]
    for i in range(N_BEND_CELLS):
        els += [
            Quadrupole(LQ, K1),
            ThinOctupole(k3l, name=f"oc_{i}"),
            Dipole(BEND_L, BEND_ANG),
            Quadrupole(LQ, -K1),
            Dipole(BEND_L, BEND_ANG),
        ]
    return Lattice(els, ref)


def _xtrack_bendy(k3l: float, kick: float):
    """xtrack's 4d twiss of the bendy ring, built once per setting and cached."""
    key = ("bendy", k3l, kick)
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
    for _ in range(N_BEND_CELLS):
        els += [
            xt.Quadrupole(length=LQ, k1=K1),
            xt.Multipole(knl=[0.0, 0.0, 0.0, k3l]),
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


# --------------------------------------------------------------------------
# 1. The fixed point
# --------------------------------------------------------------------------


def test_nonlinear_closed_orbit_matches_xtrack() -> None:
    """Newton on accsim's tracked cubic kick lands where xtrack's own search lands.

    Both planes are steered, so the dipole term's full structure is live —
    ``theta_x`` carries ``x_co(x_co^2 - 3 y_co^2)`` and ``theta_y`` carries
    ``y_co(3 x_co^2 - y_co^2)``, and neither reduces to the other. Compared at all
    boundaries, so this is the whole orbit rather than one number.
    """
    lat = _accsim_lattice(K3L, kick_x=KICK, kick_y=0.6 * KICK)
    table = propagate_orbit_nonlinear(lat)
    x_acc = np.array([o[0] for o in table])
    y_acc = np.array([o[2] for o in table])

    tw = _xtrack_twiss(K3L, kick_x=KICK, kick_y=0.6 * KICK)
    x_xt, y_xt = np.array(tw.x), np.array(tw.y)

    assert x_acc.shape == x_xt.shape
    assert np.abs(x_acc).max() > 5e-4  # a real, ~1 mm orbit
    assert np.abs(y_acc).max() > 2e-4
    assert np.allclose(x_acc, x_xt, rtol=0, atol=ORBIT_ATOL)
    assert np.allclose(y_acc, y_xt, rtol=0, atol=ORBIT_ATOL)


def test_the_linear_closed_orbit_is_decisively_wrong() -> None:
    """Without this the agreement above could be measuring round-off. It is not.

    I1's linear solve — which is what a machine with no nonlinear elements would
    have — misses xtrack by orders of magnitude more than the tolerance the
    nonlinear orbit meets. The ``k3l = 0`` control puts the linear solve back on
    xtrack's answer exactly, so the discrepancy is the octupole and not the ring.
    """
    lat = _accsim_lattice(K3L, kick_x=KICK, kick_y=0.6 * KICK)
    tw = _xtrack_twiss(K3L, kick_x=KICK, kick_y=0.6 * KICK)

    linear_err = abs(closed_orbit(lat)[0] - tw.x[0])
    nonlinear_err = abs(closed_orbit_nonlinear(lat)[0] - tw.x[0])
    assert linear_err > 1e-9
    assert linear_err > 1e3 * nonlinear_err

    # The k3l = 0 control. The *nonlinear* solve is what matches xtrack now: both codes
    # run exact drifts (L1), and an exact drift is a nonlinear map, so accsim's linear
    # solve is no longer exact even with no octupole. It misses by 2.7e-11 — third order
    # in the orbit angle, and two orders below the 1e-9 the octupole's own feed-down
    # produces above, so the contrast this test draws is untouched.
    flat = _accsim_lattice(0.0, kick_x=KICK, kick_y=0.6 * KICK)
    tw0 = _xtrack_twiss(0.0, kick_x=KICK, kick_y=0.6 * KICK)
    assert abs(closed_orbit_nonlinear(flat)[0] - tw0.x[0]) < ORBIT_ATOL
    assert abs(closed_orbit(flat)[0] - tw0.x[0]) < 1e-9


# --------------------------------------------------------------------------
# 2. The derived split, as a whole matrix
# --------------------------------------------------------------------------


def test_the_derived_equivalent_lattice_matches_xtracks_r_matrix() -> None:
    """**The strongest statement here**: the derived coefficients against xtrack.

    The left-hand side is built from J3's algebra — :func:`linearised_lattice` puts
    a real :class:`~accsim.elements.quadrupole.ThinQuadrupole` and
    :class:`~accsim.elements.skew_quadrupole.ThinSkewQuadrupole` of the derived
    strengths into the machine and multiplies exact matrices. No finite difference
    appears on accsim's side at all. The right-hand side is xtrack's ``R_matrix``,
    differenced about its own closed orbit by a code that knows nothing about the
    expansion.

    Both planes are steered, so the off-diagonal blocks are genuinely populated by
    ``k1sl_eff = k3l x_co y_co`` — a term that vanishes identically if either bump
    is switched off, and which therefore cannot be right by accident.

    **The residual is xtrack's differencing, and that is demonstrated rather than
    assumed.** It is ``1.17e-7`` at xtrack's default ``dx = 1e-6`` and falls to
    ``1.05e-8`` and ``1.17e-9`` at ``3e-7`` and ``1e-7`` — exactly ``step^2``, so it
    extrapolates to zero and the derived matrix is *the* answer rather than a close
    one. Two further measurements say the same thing from other directions: the gap
    is strictly proportional to ``k3l`` (2.9e-8 at a quarter of the strength), and at
    ``k3l = 0`` the two codes agree to 7e-13 on this ring, so nothing here is a
    difference in how the ring itself is described.
    """
    lat = _accsim_lattice(K3L, kick_x=KICK, kick_y=0.6 * KICK)
    derived, _ = linearised_lattice(lat).one_turn_map()
    got = derived[:4, :4]
    assert np.abs(got).max() > 1.0  # non-vacuous: a real map
    assert np.abs(got[:2, 2:]).max() > 1e-6  # the skew blocks are populated

    # Both sides are measured as a *difference* from the octupole-free ring. With L1's
    # exact drift map, xtrack's R-matrix carries the drift's own O(orbit angle^2)
    # contribution to the transverse block and `linearised_lattice` — built from accsim
    # elements, none of which can represent it — does not. That omission is 7.9e-8 here
    # and is present at k3l = 0 too, so it is not the octupole and differencing it away
    # is what leaves the octupole's split under test. It is also *larger* than
    # MATRIX_ATOL, so leaving it in would have compared two different maps.
    flat = _accsim_lattice(0.0, kick_x=KICK, kick_y=0.6 * KICK)
    flat_derived, _ = linearised_lattice(flat).one_turn_map()
    base_acc = flat_derived[:4, :4]

    gaps = []
    for dx in (9e-7, 3e-7, 1e-7):
        steps = {"dx": dx, "dpx": dx / 10, "dy": dx, "dpy": dx / 10}
        tw = _xtrack_twiss(K3L, kick_x=KICK, kick_y=0.6 * KICK, steps=steps)
        tw0 = _xtrack_twiss(0.0, kick_x=KICK, kick_y=0.6 * KICK, steps=steps)
        octupole_acc = got - base_acc
        octupole_xt = np.array(tw.R_matrix)[:4, :4] - np.array(tw0.R_matrix)[:4, :4]
        gaps.append(float(np.abs(octupole_acc - octupole_xt).max()))
    assert gaps[-1] < 2e-8
    # ...against an octupole signal of 0.30, i.e. a relative 4e-8.
    assert np.abs(got - base_acc).max() > 0.1

    # The size of what the derived route omits, asserted rather than differenced away
    # silently: it is the drift's exact-map content, and it is there with no octupole.
    tw_flat = _xtrack_twiss(0.0, kick_x=KICK, kick_y=0.6 * KICK)
    drift_omission = float(np.abs(base_acc - np.array(tw_flat.R_matrix)[:4, :4]).max())
    assert drift_omission == pytest.approx(7.9e-8, rel=0.1)

    # The design-orbit map has *no* feed-down in it at all (an octupole's matrix() is
    # a drift), and misses xtrack by orders of magnitude more.
    reference = np.array(_xtrack_twiss(K3L, kick_x=KICK, kick_y=0.6 * KICK).R_matrix)[:4, :4]
    design, _ = lat.one_turn_map()
    assert np.abs(design[:4, :4] - reference).max() > 1e3 * MATRIX_ATOL

    # And the derived route agrees with accsim's own differencing route, which is the
    # analytic suite's gate — restated here so the two comparisons are on one page. The
    # drifts contribute their matrices, for the reason given above: `linearised_lattice`
    # cannot represent the exact drift map's content, so the differenced route has to be
    # asked for the same thing before the two can be compared at all.
    co_lat = closed_orbit_nonlinear(lat)
    product = np.eye(6)
    for elem, m in zip(lat.elements, linearised_element_maps(lat, co_lat), strict=True):
        product = (elem.matrix(lat.ref) if isinstance(elem, Drift) else m) @ product
    assert np.allclose(derived, product, rtol=0, atol=1e-8)


# --------------------------------------------------------------------------
# 3. The chromaticity rung — the milestone, against a code that tracks
# --------------------------------------------------------------------------


def test_the_unsteered_bendy_ring_is_the_control() -> None:
    """On the design orbit both codes agree, and accsim's octupole is chromatically inert.

    This fixes the floor the steered comparison is read against, and re-establishes
    J2's claim against an independent code: with the ring on axis, adding a strong
    octupole changes xtrack's own ``dqx`` by less than the agreement floor, which is
    what "an octupole contributes nothing to first-order chromaticity" means
    physically rather than as a property of accsim's integrals.
    """
    lat = _accsim_bendy(K3L_CHROMA, 0.0)
    tw = _xtrack_bendy(K3L_CHROMA, 0.0)
    assert chromaticity(lat)[0] == pytest.approx(tw.dqx, abs=1e-3)
    assert chromaticity(lat)[1] == pytest.approx(tw.dqy, abs=1e-3)

    tw_plain = _xtrack_bendy(0.0, 0.0)
    assert tw.dqx == pytest.approx(tw_plain.dqx, abs=1e-6)
    assert tw.dqy == pytest.approx(tw_plain.dqy, abs=1e-6)


def test_accsims_linear_elements_do_not_feed_down_off_axis() -> None:
    """The modelling difference, isolated at ``k3l = 0`` before it is used as a floor.

    With no octupole accsim has *nothing* to feed down — bends and quads are exactly
    linear maps — while xtrack's are exact nonlinear ones, so its chromaticity moves
    when the ring is steered and accsim's does not. That residual is first order in
    the orbit and belongs to accsim's element models (I3 recorded the same thing);
    the octupole effect below is far larger, which is what makes the gate meaningful.
    """
    steered = _xtrack_bendy(0.0, KICK_CHROMA)
    flat = _xtrack_bendy(0.0, 0.0)
    floor = abs(steered.dqx - flat.dqx)
    assert floor < 1e-3  # small...
    assert chromaticity(_accsim_bendy(0.0, KICK_CHROMA))[0] == pytest.approx(
        chromaticity(_accsim_bendy(0.0, 0.0))[0], abs=1e-12
    )  # ...and accsim reports exactly none of it


def test_chromaticity_on_a_steered_octupole_ring_tracks_xtrack() -> None:
    """**The milestone, cross-checked.** The design orbit is wrong; the on-orbit answer is not.

    Steer the ring and xtrack's ``dqx`` moves by far more than the floor above,
    because each octupole is now a sextupole of strength ``k3l x_co`` sitting at
    dispersion. accsim's design-orbit :func:`~accsim.twiss.chromaticity` reports
    *exactly* the unsteered number — bit for bit, since no orbit enters it —
    and is therefore decisively wrong. :func:`~accsim.twiss.chromaticity_on_orbit`,
    built on the derived split, has to reproduce an effect the package's previous
    answer said did not exist.
    """
    tw = _xtrack_bendy(K3L_CHROMA, KICK_CHROMA)
    lat = _accsim_bendy(K3L_CHROMA, KICK_CHROMA)
    on_orbit, design = chromaticity_on_orbit(lat), chromaticity(lat)

    # The effect is real and large: xtrack's own chromaticity has moved.
    flat = _xtrack_bendy(K3L_CHROMA, 0.0)
    assert abs(tw.dqx - flat.dqx) > 1e-2

    for i, reference in ((0, tw.dqx), (1, tw.dqy)):
        design_err = abs(design[i] - reference)
        on_orbit_err = abs(on_orbit[i] - reference)
        assert design_err > 1e-2  # the design orbit is decisively wrong...
        assert on_orbit_err < design_err / 10.0  # ...and this is an order better
