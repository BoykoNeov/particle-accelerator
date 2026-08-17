r"""Vertical dispersion from a vertical orbit *angle* — what the exact drift buys.

The gap this closes was measured and written up by K2 (``docs/CONVENTIONS.md`` ->
*Orbit-driven vertical dispersion*) but could not be represented then: accsim's linear
drift moves ``y`` by ``L py`` where the exact map moves it by ``L py / pz``, and the
missing ``-L py delta`` is bilinear, so no 6x6 can carry it. The consequence was that a
ring with a vertical steerer had **exactly zero** vertical dispersion in accsim and
``2.1e-4`` in xtrack, with the two closed orbits agreeing to eight digits.

**The test machine here has no bending magnet at all**, which is what makes it a clean
gate rather than a partial one. K2's account of the missing source is

    Delta d_y = p_y L (h <D_x> - 1)

— a ``-1`` from the drift's ``1/pz`` and a ``+h <D_x>`` from the extra arc a dispersed
particle travels on the outside of a bend. Setting ``h = 0`` everywhere removes the
second term *and* makes ``D_x`` identically zero, so the drift's own contribution is the
whole effect and can be checked against a closed form with nothing else mixed in. On
K2's own arc (thin quadrupoles and thick dipoles, no drifts) the drift map changes
nothing whatever, and the bend's half of that formula is a later milestone.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    Corrector,
    Drift,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    closed_twiss,
)
from accsim.coords import DELTA, DIM, PX, PY, ZETA, X, Y
from accsim.orbit import (
    closed_orbit_nonlinear,
    linearised_element_maps,
    linearised_lattice,
    linearised_one_turn_map,
    propagate_orbit_nonlinear,
)
from accsim.twiss import _matched_dispersion, closed_twiss_on_orbit

L_D = 2.0
F_FOCAL = 2.5
N_CELLS = 8
STEER = 1.0e-3  # vertical steerer angle [rad]


@pytest.fixture
def ref() -> ReferenceParticle:
    """gamma0 = 5: modest enough that 1/gamma0^2 terms are not lost in the noise."""
    from accsim import PROTON_MASS_EV

    return ReferenceParticle.from_gamma(PROTON_MASS_EV, 5.0)


def _ring(ref: ReferenceParticle, steer: float = STEER) -> Lattice:
    """A FODO ring of drifts and thin quadrupoles. **No bends, so D_x == 0 exactly.**"""
    els: list = []
    for _ in range(N_CELLS):
        els += [
            ThinQuadrupole(0.5 / F_FOCAL),
            Drift(L_D),
            ThinQuadrupole(-1.0 / F_FOCAL),
            Drift(L_D),
            ThinQuadrupole(0.5 / F_FOCAL),
        ]
    if steer != 0.0:
        els.insert(1, Corrector(kick_y=steer))
    return Lattice(els, ref)


def _first_order_dispersion(lat: Lattice, ref: ReferenceParticle) -> np.ndarray:
    r"""The closed form, to first order in the orbit angle, built independently.

    Each drift's exact ``delta`` column is ``-L p_co (1+delta)/pz^3``; dropped to first
    order in the orbit angle that is just ``-L p_co``. This inserts that into the
    *linear* matrices and solves ``D = (I - M4)^-1 d`` — sharing no arithmetic with
    :func:`~accsim.orbit.linearised_element_maps`, which differentiates ``track()``.

    The conjugate ``zeta``-row entries are deliberately **not** added: dispersion is a
    4D quantity and ``_matched_dispersion`` reads only the transverse block and the
    ``delta`` column, so they cannot affect the answer. Their absence here is what lets
    the symplecticity of the pair be a separate statement from its dispersion.
    """
    orbit = propagate_orbit_nonlinear(lat)
    M = np.eye(DIM)
    for elem, o in zip(lat.elements, orbit, strict=False):
        m = elem.matrix(ref).copy()
        if isinstance(elem, Drift):
            m[X, DELTA] = -elem.length * o[PX]
            m[Y, DELTA] = -elem.length * o[PY]
        M = m @ M
    return _matched_dispersion(M)


# --------------------------------------------------------------------------
# 1. The headline: a steerer alone now makes vertical dispersion
# --------------------------------------------------------------------------


def test_a_vertical_steerer_alone_produces_vertical_dispersion(ref: ReferenceParticle) -> None:
    r"""The gap K2 measured, closed: ``D_y`` goes from exactly ``0`` to ``0.259 m``.

    Every previously known route to vertical dispersion in this package needs either a
    coupling element (G1's skew quadrupole, which only *rotates* dispersion the
    horizontal bends already made) or a rolled bend (K2, the first element whose matrix
    carries a vertical ``delta`` column). This ring has neither — no skew anything, no
    bend at all — and a plain vertical steerer, whose ``matrix`` is the identity and
    whose kick carries no momentum dependence whatever.

    The design optics still says ``0``, and *correctly*: it is built on ``matrix()``,
    where the term genuinely cannot live. The on-orbit optics says ``0.2590571``, which
    is xtrack's own answer to seven figures (``tests/reference``). The two are not in
    conflict — they are the linear and the exact map, and the difference between them
    is this milestone.
    """
    lat = _ring(ref)
    co = closed_orbit_nonlinear(lat)
    assert abs(co[Y]) > 1.0e-3  # the vertical orbit is genuinely there
    assert co[PY] == pytest.approx(-0.5 * STEER, rel=1e-9)  # symmetric ring: -kick/2

    # The design optics cannot see it, and says so at exact zero rather than "small".
    assert closed_twiss(lat).disp_y == 0.0
    assert closed_twiss(lat).disp_x == 0.0

    tw = closed_twiss_on_orbit(lat)
    assert tw.disp_y == pytest.approx(0.2590571, rel=1e-6)

    # No bends anywhere, so horizontal dispersion stays exactly zero — the vertical
    # signal is not leakage from a horizontal one, which is what makes this ring the
    # clean gate. Asserted at exact zero: px is zero all the way round, so the
    # drift's horizontal delta column is identically absent, not merely small.
    assert tw.disp_x == 0.0
    assert tw.disp_px == 0.0

    # D_py stays at the numerical floor: a drift's delta column touches y, never py,
    # so the *angle* of the dispersion function has no first-order source here.
    assert abs(tw.disp_py) < 1.0e-9


def test_the_dispersion_matches_the_closed_form_and_the_remainder_is_second_order(
    ref: ReferenceParticle,
) -> None:
    r"""The *value*, against a closed form, with the residual's **order** pinned.

    ``-L p_co`` per drift is the exact ``delta`` column dropped to first order in the
    orbit angle, and it reproduces the tracked answer to ``3.9e-4`` relative. That
    residual is the ``(1+delta)/pz^3`` the closed form threw away, so it must be
    *second* order in the orbit angle — and it is, to three digits: halving the steerer
    divides it by ``4.000`` four times over.

    Asserting the order rather than a tolerance is what makes this a check on the map.
    A uniform mis-scaling of the drift's new term — the classic way to get a plausible
    wrong coefficient — would move the residual to a *fixed* relative size and leave
    the ``4.000`` ratios untouched, so the two assertions below are checking different
    things and both are needed.
    """
    lat = _ring(ref)
    got = closed_twiss_on_orbit(lat).disp_y
    want = _first_order_dispersion(lat, ref)[2]

    assert want == pytest.approx(0.2591583, rel=1e-6)
    assert got == pytest.approx(want, rel=5e-4)

    def relative_residual(steer: float) -> float:
        lt = _ring(ref, steer=steer)
        d_exact = closed_twiss_on_orbit(lt).disp_y
        d_first = _first_order_dispersion(lt, ref)[2]
        return abs(d_exact - d_first) / abs(d_first)

    residuals = [relative_residual(STEER / 2**k) for k in range(4)]
    assert residuals[0] == pytest.approx(3.9066e-4, rel=1e-3)
    for big, small in zip(residuals[:-1], residuals[1:], strict=True):
        assert big / small == pytest.approx(4.0, rel=1e-2)

    # And the dispersion itself is first order in the steerer, which is what makes the
    # ratio above a statement about the residual rather than about the whole answer.
    halved = closed_twiss_on_orbit(_ring(ref, steer=STEER / 2)).disp_y
    assert got / halved == pytest.approx(2.0, rel=1e-3)


def test_an_unsteered_ring_reports_exactly_the_design_optics(ref: ReferenceParticle) -> None:
    r"""The control, and the bound on the whole change: no orbit, no new physics.

    The exact map's new terms are all proportional to the orbit *angle*, so on a
    machine whose closed orbit is exactly zero it must be indistinguishable from the
    linear one. The design optics is **bit-for-bit** unchanged because it is built on
    ``matrix()``, which this milestone did not touch; the on-orbit route agrees to its
    finite-difference floor rather than exactly, and the honest statement is the size of
    that floor — ``2.7e-13`` on the one-turn map, measured, against the ``1e-10`` the
    existing design-optics gates ask for.

    That floor is three orders better than it would have been: evaluating the
    longitudinal term as ``1 - E/(E0 pz)`` cancels two numbers of size 1 and would have
    put it at ``3.6e-8``, breaking those gates. See :class:`~accsim.elements.drift.Drift`.
    """
    lat = _ring(ref, steer=0.0)
    assert np.abs(closed_orbit_nonlinear(lat)).max() == 0.0

    assert closed_twiss(lat).disp_y == 0.0
    assert closed_twiss_on_orbit(lat).disp_y == 0.0  # exactly, not "small"

    residual = float(np.abs(linearised_one_turn_map(lat) - lat.one_turn_matrix()).max())
    assert residual < 1.0e-11
    assert residual > 1.0e-16  # it is a floor, not an identity — say so

    # Every element's on-orbit map is its design matrix to that floor, drifts included.
    for m, elem in zip(linearised_element_maps(lat), lat.elements, strict=True):
        assert np.abs(m - elem.matrix(ref)).max() < 1.0e-13


# --------------------------------------------------------------------------
# 2. The element-level content: a conjugate pair, and what it does to the routes
# --------------------------------------------------------------------------


def test_the_new_entries_are_a_conjugate_pair_of_the_derived_size(
    ref: ReferenceParticle,
) -> None:
    r"""Per drift: ``M[x,delta] = M[zeta,px] = -L px``, and likewise for ``y``.

    The element level is where the closed form is exact, so this is the sharpest form
    of the statement. Two things are asserted separately because they can fail
    separately:

    - the **size**, ``-L p_co``, to first order in the orbit angle;
    - the **pairing**, ``M[y,delta] == M[zeta,py]``, which holds to a relative
      ``1.9e-9`` — the finite-difference floor of the Jacobian these come from, not a
      physical difference. This is the symplectic condition in disguise, and it is the
      half a plausible implementation omits: adding the ``delta`` column alone gives the
      right dispersion and a map that is *not* symplectic (wrong at first order —
      ``test_symplectic_canonical.py``). A dispersion gate alone would pass that map,
      so the pairing needs its own check.

    Everything outside the pair is **second** order in the orbit angle, with a
    coefficient that is also derivable: ``d(L py/pz)/d(py) = L/pz + L py^2/pz^3``, which
    at ``px = 0`` is ``L (1 + 3 py^2 / 2)``, so ``M[y,py] - L = 1.5 L py^2``. Pinning
    that 1.5 is what distinguishes "the second-order terms are right" from "they are
    small".
    """
    lat = _ring(ref)
    maps = linearised_element_maps(lat)
    orbit = propagate_orbit_nonlinear(lat)

    checked = 0
    for m, elem, o in zip(maps, lat.elements, orbit, strict=False):
        if not isinstance(elem, Drift):
            continue
        L, py_co = elem.length, float(o[PY])
        assert float(o[PX]) == 0.0  # vertical steering only: nothing horizontal to see
        D = m - elem.matrix(ref)

        # Size, to first order in the orbit angle.
        assert D[Y, DELTA] == pytest.approx(-L * py_co, rel=1e-4)
        assert D[ZETA, PY] == pytest.approx(-L * py_co, rel=1e-4)
        # Pairing: the two are the *same number*, to the differencing floor — three
        # orders tighter than the 1e-4 either one is known to on its own.
        assert D[Y, DELTA] == pytest.approx(D[ZETA, PY], rel=1e-8)
        # The horizontal half is exactly absent, because px is exactly zero.
        assert D[X, DELTA] == 0.0
        assert D[ZETA, PX] == 0.0

        # Second order, coefficient pinned.
        assert D[Y, PY] == pytest.approx(1.5 * L * py_co**2, rel=2e-2)
        checked += 1

    assert checked == 2 * N_CELLS  # every drift was examined, not just the first


def test_the_two_equivalent_lattice_routes_now_differ_and_the_difference_is_named(
    ref: ReferenceParticle,
) -> None:
    r"""``linearised_lattice`` cannot carry the new entries, and returns anyway.

    :func:`~accsim.orbit.linearised_lattice` builds an equivalent machine out of real
    elements. No accsim element has a transverse ``delta`` column without also bending,
    so the drift's new pair has nothing to be built out of and is simply absent — the
    two routes to "the machine the beam really sees" now disagree, by ``0.13`` in the
    one-turn map of this ring.

    That is an omission, not a wrong number, and the difference decides the behaviour.
    The dropped terms carry **no gradient**, so no chromaticity integral reads them and
    everything that function exists to feed is unaffected; refusing would break the
    established feed-down machinery over a term it never looks at. The failure mode
    that *is* real is asking it for dispersion, which silently returns the old
    orbit-blind zero — asserted here so it is documented rather than discovered.
    """
    lat = _ring(ref)
    equivalent = linearised_lattice(lat)

    # Nothing was substituted: no sextupole, no octupole, so it is the same elements.
    assert len(equivalent.elements) == len(lat.elements)

    gap = float(np.abs(equivalent.one_turn_matrix() - linearised_one_turn_map(lat)).max())
    assert gap > 1.0e-2  # the routes really have parted company

    # The gap is dominated by the delta column and the zeta row — the conjugate pair,
    # transported. What is left outside them is the exact map's *second*-order content
    # in the orbit angle, and it is asserted as that order: quartering the steerer
    # divides it by sixteen, where the pair itself only quarters. That is what says the
    # derived route is still right about the gradients it exists to supply, rather than
    # merely close.
    def outside_the_pair(steer: float) -> float:
        lt = _ring(ref, steer=steer)
        D = linearised_one_turn_map(lt) - linearised_lattice(lt).one_turn_matrix()
        D[:, DELTA] = 0.0
        D[ZETA, :] = 0.0
        return float(np.abs(D).max())

    assert outside_the_pair(STEER) == pytest.approx(3.1122e-4, rel=1e-3)
    assert outside_the_pair(STEER) / outside_the_pair(STEER / 4) == pytest.approx(16.0, rel=2e-2)

    # And the failure mode, stated: dispersion from the derived route is the old zero.
    assert closed_twiss(equivalent).disp_y == 0.0
    assert closed_twiss_on_orbit(lat).disp_y > 0.2
