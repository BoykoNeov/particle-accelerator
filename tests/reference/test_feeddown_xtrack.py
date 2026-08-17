"""I2 cross-check: feed-down against xtrack's own nonlinear closed-orbit search.

The analytic suite derives the feed-down expansion and gates it against accsim's
own linear machinery. What it cannot do is confirm that the *fixed point* accsim's
Newton converges to is the one a real tracker finds, because both routes are
accsim's. xtrack finds its closed orbit by iterating its own tracking of the same
nonlinear line, and linearises about it by finite differences — two genuinely
independent algorithms for the two things I2 adds.

What this reaches that the analytic suite cannot:

- **An independent nonlinear closed-orbit search.** accsim runs Newton on its own
  tracked map; xtrack iterates its own tracker. They must land on the same orbit.
- **The linearised optics, as a whole matrix.** ``TwissTable.R_matrix`` is xtrack's
  6x6 finite-differenced about its own closed orbit, compared entry by entry with
  :func:`accsim.linearised_one_turn_map` — normal *and* skew feed-down at once, in
  one object, with both planes steered.
- **The claim that I1's linear solve is now insufficient**, which is the reason the
  milestone exists. :func:`test_the_linear_closed_orbit_is_decisively_wrong` asserts
  that ``closed_orbit`` misses xtrack by the feed-down scale — so these gates are
  not measuring round-off.

Element equivalences are the ones already established by I1 and J1, reused rather
than re-probed:

    accsim Corrector(kick_x=+k)  ==  xt.Multipole(knl=[-k])
    accsim Corrector(kick_y=+k)  ==  xt.Multipole(ksl=[+k])
    accsim ThinSextupole(k2l)    ==  xt.Multipole(knl=[0, 0, +k2l])

Thick quadrupoles again, so the maps being composed around the sextupole are the
non-trivial ones. Marked ``reference``: skips when xtrack or its JIT is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    Corrector,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinSextupole,
    closed_orbit,
    closed_orbit_nonlinear,
    linearised_one_turn_map,
    match_periodic,
    propagate_orbit_nonlinear,
)
from accsim.coords import DELTA, X

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
LQ = 0.3  # quad length [m]
K1 = 1.2  # quad gradient [m^-2]
LD = 1.0  # drift length [m]
N_CELLS = 3
KICK = 2.0e-4  # steerer angle [rad]
K2L = 20.0  # integrated sextupole strength [m^-2]

# Measured 2026-08-10: the two codes' closed orbits agree to 1.6e-13 m absolute
# (1.4e-10 relative) on a ~1.1 mm orbit with both planes steered. The floor is the
# two *iterative* searches, not accsim's Newton (whose own residual is driven to
# 1e-14 by construction). Tolerance below with 6.4x headroom.
ORBIT_ATOL = 1e-12

# Measured 2026-08-10: 2.1e-11 max absolute on 4x4 entries up to 7.0, i.e. 47x
# headroom below the tolerance. The comparison is finite-difference against
# finite-difference, and the floor is **xtrack's**, not accsim's: sweeping accsim's
# `step` over 1e-6/1e-7 moves the discrepancy by 0.1% (2.149e-11 -> 2.147e-11), so
# it is insensitive to accsim's choice, while xtrack's own `steps_R_matrix` is
# dx=1e-6, dpx=1e-7. accsim's own differencing floor is a separate, smaller number
# — 7.2e-12, gated by the k2l = 0 branch below against the exact analytic matrix.
MATRIX_ATOL = 1e-9
ACCSIM_FD_ATOL = 1e-10


def _fractional_tunes(M: np.ndarray) -> tuple[float, float]:
    """Fractional tunes from the 2x2 diagonal blocks of an uncoupled 6x6 map.

    ``match_periodic`` cannot supply these: its ``mu`` is the phase *accumulated
    from the lattice start*, which is zero at the periodic solution's own point.
    """
    out = []
    for a, b in ((0, 1), (2, 3)):
        half_trace = 0.5 * (M[a, a] + M[b, b])
        assert abs(half_trace) < 1.0, f"unstable linearised map: |tr/2| = {abs(half_trace)}"
        q = np.arccos(half_trace) / (2.0 * np.pi)
        out.append(float(q if M[a, b] >= 0.0 else 1.0 - q))
    return out[0], out[1]


def _accsim_cell() -> list:
    return [Quadrupole(LQ, K1), Drift(LD), Quadrupole(LQ, -K1), Drift(LD)]


def _linearised_with_matrix_drifts(lat: Lattice, co: np.ndarray) -> np.ndarray:
    """``linearised_one_turn_map``, but each drift contributes its ``matrix`` instead.

    Isolates the *sextupole's* feed-down from the drift's own exact-map content (L1). A
    drift at a closed-orbit angle departs from its matrix in three places — the ``delta``
    column, the conjugate ``zeta`` row, and the transverse block at ``O(angle^2)`` — and
    a composed *matrix* product carries none of them. Comparing the two directly would
    be comparing two different maps, at 7.9e-8 on this ring.
    """
    from accsim.orbit import linearised_element_maps

    product = np.eye(6)
    for elem, m in zip(lat.elements, linearised_element_maps(lat, co), strict=True):
        product = (elem.matrix(lat.ref) if isinstance(elem, Drift) else m) @ product
    return product


def _xtrack_cell() -> list:
    """The accsim cell, in xtrack. ``model="exact"`` on the drifts is load-bearing.

    ``xt.Drift()`` defaults to ``"adaptive"``, which resolves to the **expanded** model
    ``x += L px / (1 + delta)``. accsim's :class:`~accsim.elements.drift.Drift` tracks the
    exact ``x += L px / pz`` (L1), so a default drift here disagrees at ``O(angle^3)`` —
    which on the steered rings in this file is a real, orbit-dependent discrepancy and
    looks exactly like a feed-down coefficient error. It is not one. See
    ``test_drift_xtrack.py::test_xtracks_default_drift_is_the_expanded_model_not_the_exact_one``.
    """
    return [
        xt.Quadrupole(length=LQ, k1=K1),
        xt.Drift(length=LD, model="exact"),
        xt.Quadrupole(length=LQ, k1=-K1),
        xt.Drift(length=LD, model="exact"),
    ]


def _accsim_lattice(k2l: float, kick_x: float = 0.0, kick_y: float = 0.0) -> Lattice:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    return Lattice(
        [
            Corrector(kick_x=kick_x, kick_y=kick_y, name="steerer"),
            ThinSextupole(k2l, name="sx"),
            *(_accsim_cell() * N_CELLS),
        ],
        ref,
    )


def _xtrack_line(k2l: float, kick_x: float = 0.0, kick_y: float = 0.0):
    line = xt.Line(
        elements=[
            xt.Multipole(knl=[-kick_x], ksl=[kick_y]),
            xt.Multipole(knl=[0.0, 0.0, k2l]),
            *(_xtrack_cell() * N_CELLS),
        ]
    )
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        return line, line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


def test_nonlinear_closed_orbit_matches_xtrack() -> None:
    """Newton on accsim's tracked map lands where xtrack's own search lands.

    Both planes are steered so that every feed-down term is live at once — the
    ``x^2`` dipole, the ``y^2`` dipole of opposite sign, and the ``x y`` cross term
    that only exists when both bumps are on. Compared at all 15 boundaries, so this
    is the whole orbit and not one number.
    """
    lat = _accsim_lattice(K2L, kick_x=KICK, kick_y=0.6 * KICK)
    table = propagate_orbit_nonlinear(lat)
    x_acc = np.array([o[0] for o in table])
    y_acc = np.array([o[2] for o in table])

    _, tw = _xtrack_line(K2L, kick_x=KICK, kick_y=0.6 * KICK)
    x_xt, y_xt = np.array(tw.x), np.array(tw.y)

    assert x_acc.shape == x_xt.shape
    assert np.abs(x_acc).max() > 5e-4  # a real, ~1 mm orbit
    assert np.abs(y_acc).max() > 2e-4
    assert np.allclose(x_acc, x_xt, rtol=0, atol=ORBIT_ATOL)
    assert np.allclose(y_acc, y_xt, rtol=0, atol=ORBIT_ATOL)


def test_the_linear_closed_orbit_is_decisively_wrong() -> None:
    """**Why I2 exists**: I1's linear solve misses xtrack by the feed-down scale.

    Without this the agreement above would be uninformative — it would be possible
    for feed-down to be too small to matter here and for every gate to be measuring
    round-off. It is not: the linear answer is wrong by four orders of magnitude
    more than the tolerance the nonlinear one meets. The same lattice with
    ``k2l = 0`` is the control, where the linear solve is exact again.
    """
    lat = _accsim_lattice(K2L, kick_x=KICK, kick_y=0.6 * KICK)
    _, tw = _xtrack_line(K2L, kick_x=KICK, kick_y=0.6 * KICK)

    linear_err = abs(closed_orbit(lat)[0] - tw.x[0])
    nonlinear_err = abs(closed_orbit_nonlinear(lat)[0] - tw.x[0])
    assert linear_err > 1e-8
    assert linear_err > 1e4 * nonlinear_err

    # The k2l = 0 control. The *nonlinear* solve is what matches xtrack now: both codes
    # run exact drifts (L1), and an exact drift is itself a nonlinear map, so accsim's
    # linear solve is no longer the exact answer even with no sextupole. It misses by
    # 2.7e-11 — third order in the orbit angle, four orders below the 1e-8 feed-down
    # error above, so the contrast this test draws is untouched.
    flat = _accsim_lattice(0.0, kick_x=KICK, kick_y=0.6 * KICK)
    _, tw0 = _xtrack_line(0.0, kick_x=KICK, kick_y=0.6 * KICK)
    assert abs(closed_orbit_nonlinear(flat)[0] - tw0.x[0]) < ORBIT_ATOL
    assert abs(closed_orbit(flat)[0] - tw0.x[0]) < 1e-9


def test_linearised_one_turn_map_matches_xtrack() -> None:
    """The whole linearised 4x4, against xtrack's R-matrix about its own orbit.

    This is the single strongest statement in the file: normal feed-down (the
    diagonal blocks) and skew feed-down (the off-diagonal blocks) checked together,
    entry by entry, against a code that knows nothing about the expansion accsim
    derived. Both planes are steered, so the off-blocks are genuinely populated.

    The ``k2l = 0`` branch measures accsim's *own* differencing floor against its
    exact analytic matrix (7.2e-12), which is smaller than the 2.1e-11 the two codes
    reach — consistent with the floor being xtrack's differencing, as the constants
    above record.
    """
    flat = _accsim_lattice(0.0, kick_x=KICK, kick_y=0.6 * KICK)
    co0 = closed_orbit_nonlinear(flat)
    exact, _ = flat.one_turn_map()
    # Drifts contribute their matrices here, so this is still accsim's own differencing
    # floor and not the exact drift map's content — see `_linearised_with_matrix_drifts`.
    assert np.allclose(
        _linearised_with_matrix_drifts(flat, co0), exact, rtol=0, atol=ACCSIM_FD_ATOL
    )
    # ...and the content that was excluded really is there, so this is not vacuous: the
    # full differenced map carries the drift's conjugate pair at 7e-4.
    assert abs(linearised_one_turn_map(flat, co0)[X, DELTA]) > 1e-4

    lat = _accsim_lattice(K2L, kick_x=KICK, kick_y=0.6 * KICK)
    co = closed_orbit_nonlinear(lat)
    M_acc = linearised_one_turn_map(lat, co)[:4, :4]

    _, tw = _xtrack_line(K2L, kick_x=KICK, kick_y=0.6 * KICK)
    M_xt = np.asarray(tw.R_matrix)[:4, :4]

    # Non-vacuous: feed-down really has changed the map, in both blocks.
    M_on_axis = flat.one_turn_map()[0][:4, :4]
    assert np.abs(M_acc - M_on_axis).max() > 1e-3
    assert np.abs(M_acc[:2, 2:]).max() > 1e-4  # the skew block is populated

    assert np.allclose(M_acc, M_xt, rtol=0, atol=MATRIX_ATOL)


def test_feeddown_tune_shift_and_beta_match_xtrack() -> None:
    """The optics move, and they move by the same amount in both codes.

    ``correctors do not move the optics`` was I1's claim; here an outside code
    confirms it fails once the sextupole is off axis. The comparison is on the
    *shift* from the on-axis machine, so no integer-tune bookkeeping enters.
    """
    flat = _accsim_lattice(K2L)  # sextupole present, on axis
    lat = _accsim_lattice(K2L, kick_x=KICK)

    M_flat = linearised_one_turn_map(flat, closed_orbit_nonlinear(flat))
    M_lat = linearised_one_turn_map(lat, closed_orbit_nonlinear(lat))
    tw_lat = match_periodic(M_lat)

    _, xt_flat = _xtrack_line(K2L)
    _, xt_lat = _xtrack_line(K2L, kick_x=KICK)

    q_flat, q_lat = _fractional_tunes(M_flat), _fractional_tunes(M_lat)
    dq_acc = (q_lat[0] - q_flat[0], q_lat[1] - q_flat[1])
    dq_xt = (xt_lat.qx - xt_flat.qx, xt_lat.qy - xt_flat.qy)

    assert abs(dq_xt[0]) > 1e-5 and abs(dq_xt[1]) > 1e-5  # non-vacuous
    assert dq_acc[0] == pytest.approx(dq_xt[0], rel=2e-3)
    assert dq_acc[1] == pytest.approx(dq_xt[1], rel=2e-3)

    # ...and beta at the lattice start, which the tune shift alone does not pin.
    assert tw_lat.beta_x == pytest.approx(xt_lat.betx[0], rel=2e-5)
    assert tw_lat.beta_y == pytest.approx(xt_lat.bety[0], rel=2e-5)
    assert abs(tw_lat.beta_x - xt_flat.betx[0]) > 1e-3  # beta really moved


def test_a_purely_vertical_steerer_moves_the_horizontal_orbit_in_xtrack_too() -> None:
    r"""``theta_x = +1/2 k2l y_co^2``, confirmed by an outside code.

    The sharpest sign statement available: with ``kick_x = 0`` a linear machine has
    *identically* zero horizontal orbit — xtrack reports exactly ``0.0`` when
    ``k2l = 0``, not a small number — so every part of the horizontal orbit that
    appears is the sextupole's ``+y^2`` term. Its **sign is opposite** to the
    horizontal case, which is the ``x^2 - y^2`` structure showing up in the orbit.
    """
    _, tw0 = _xtrack_line(0.0, kick_y=KICK)
    # Nothing drives x linearly, so this is zero to well below any orbit scale.
    # (xtrack in fact returns exact 0.0 here, but asserting == 0.0 on a third-party
    # iterative solver's output would be a flake waiting on a version bump.)
    assert np.abs(np.array(tw0.x)).max() < 1e-18

    lat = _accsim_lattice(K2L, kick_y=KICK)
    x_acc = np.array([o[0] for o in propagate_orbit_nonlinear(lat)])
    _, tw = _xtrack_line(K2L, kick_y=KICK)
    x_xt = np.array(tw.x)

    assert np.abs(x_xt).max() > 1e-5  # feed-down alone put it there
    assert np.allclose(x_acc, x_xt, rtol=0, atol=ORBIT_ATOL)

    # Opposite sign to the horizontal-bump case, per x^2 - y^2.
    lat_x = _accsim_lattice(K2L, kick_x=KICK)
    dep_x = closed_orbit_nonlinear(lat_x)[0] - closed_orbit(_accsim_lattice(0.0, kick_x=KICK))[0]
    assert dep_x * x_acc[0] < 0.0


def test_the_vertical_bump_couples_the_planes_in_xtrack_too() -> None:
    """A normal sextupole at ``y_co != 0`` is a skew quad — xtrack's ``c_minus`` agrees.

    Reached here from the direction J1 never took. ``c_minus`` is xtrack's closest
    tune approach, an eigenvalue property of the coupled one-turn map, so it is
    computed from a different object than the R-matrix compared above. The control
    is the horizontal bump, where xtrack reports exactly zero coupling.
    """
    _, tw_h = _xtrack_line(K2L, kick_x=KICK)
    assert tw_h.c_minus < 1e-12  # a horizontal bump keeps the planes separate

    _, tw_v = _xtrack_line(K2L, kick_y=KICK)
    assert tw_v.c_minus > 1e-4  # ...a vertical one does not

    # accsim's own view of the same machine: the skew block of the linearised map.
    lat = _accsim_lattice(K2L, kick_y=KICK)
    M = linearised_one_turn_map(lat, closed_orbit_nonlinear(lat))[:4, :4]
    assert np.abs(M[:2, 2:]).max() > 1e-4

    M_h = linearised_one_turn_map(
        _accsim_lattice(K2L, kick_x=KICK),
        closed_orbit_nonlinear(_accsim_lattice(K2L, kick_x=KICK)),
    )[:4, :4]
    assert np.abs(M_h[:2, 2:]).max() < 1e-12  # and accsim agrees about the control
