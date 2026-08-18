r"""Cross-check the tracked synchrotron-radiation kick (B2) against xtrack.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**The comparison has to be set up before it means anything**, and getting that wrong is
what a first attempt at this file would do. xtrack integrates the radiation loss *inside*
the element: with ``integrator='uniform'`` and ``N`` kicks it applies the loss ``N``
times, each at the energy the particle has reached, which gives

    ``dE(N) = U (1 - (N-1)/N * U/E)``

and its default ``integrator='adaptive'`` resolves to **eight** uniform steps for a plain
bend. Compared against accsim's one lumped kick per element that is a 3.8e-5 disagreement
at 5 GeV rising to 2.4e-3 at 20 GeV — entirely an integration-order difference, and it
would look exactly like a wrong coefficient. Set ``num_multipole_kicks=1`` and the two
maps are the same map.

What is left after that is **6.5e-9, and it has two named owners**, both on xtrack's side:

- ``1.064e-8`` from its **pre-2019 CODATA** elementary charge (``QELEM = 1.60217662e-19``
  against today's exact ``1.602176634e-19``). The classical radius is linear in the
  charge, ``r0 = e/(4 pi eps0 m c^2)``, so this lands on ``C_gamma`` and hence on the
  loss, unchanged with energy. It is a constants vintage, not physics.
- ``2/gamma0^2`` from its **ultra-relativistic approximations** — ``gamma = gamma0(1+delta)``
  for the momentum/energy identification, ``l/c`` rather than ``l/(beta c)``, and a
  momentum reduction ``U/E`` rather than the on-shell ``U/(beta^2 E)``. accsim keeps the
  exact forms, so this term *shrinks with energy* and the test asserts that it does, which
  is what makes it a named owner rather than a fitted tolerance.

The composed prediction is checked across a factor 80 in energy, on axis and off it, on a
pure bend and on a combined-function one where the field is sampled at the particle's own
``x`` — the case the whole damping-partition story rests on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import Dipole, ReferenceParticle
from accsim.radiation_kick import mean_radiation_kick

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 0.51099895069e6  # electron, eV
LENGTH = 1.0
ANGLE = 2.0 * math.pi / 40
CURVATURE = ANGLE / LENGTH

#: xtrack's ``r0`` is built from ``QELEM = 1.60217662e-19`` (CODATA 2014) where accsim
#: uses today's exact ``1.602176634e-19``; ``r0 ∝ e`` so the whole loss carries it.
CONSTANTS_VINTAGE = 1.0639e-8


def _line(k1: float = 0.0):
    """A one-magnet xtrack line whose integration matches accsim's one lumped kick."""
    model = "mat-kick-mat" if k1 else "bend-kick-bend"
    bend = xt.Bend(length=LENGTH, angle=ANGLE, k0=CURVATURE, k1=k1, model=model)
    line = xt.Line(elements=[bend], element_names=["b"])
    line.particle_ref = xt.Particles(mass0=MASS0, p0c=1.0e10)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    line["b"].integrator = "uniform"
    line["b"].num_multipole_kicks = 1  # one kick, like accsim -- see the module docstring
    return line


def _accsim_state(energy: float, state: np.ndarray, k1: float) -> np.ndarray:
    ref = ReferenceParticle.from_total_energy(MASS0, energy)
    elem = Dipole(LENGTH, ANGLE, k1=k1)
    plain = elem.track(state.copy(), ref)
    return mean_radiation_kick(elem, state, plain, ref)


def _xtrack_state(line, energy: float, state: np.ndarray) -> np.ndarray:
    p = xt.Particles(
        mass0=MASS0,
        p0c=math.sqrt(energy**2 - MASS0**2),
        x=state[0],
        px=state[1],
        y=state[2],
        py=state[3],
        zeta=state[4],
        delta=state[5],
    )
    line.track(p)
    return np.array(
        [
            float(p.x[0]),
            float(p.px[0]),
            float(p.y[0]),
            float(p.py[0]),
            float(p.zeta[0]),
            float(p.delta[0]),
        ]
    )


@pytest.fixture(scope="module")
def pure_bend_line():
    line = _line()
    line.configure_radiation(model="mean")
    return line


@pytest.fixture(scope="module")
def combined_line():
    line = _line(k1=0.6)
    line.configure_radiation(model="mean")
    return line


def test_the_lumped_kick_matches_xtrack_once_the_integration_order_is_matched(
    pure_bend_line,
) -> None:
    """On the design orbit at 20 GeV, where the loss is a full 0.28% of the beam energy."""
    energy = 20.0e9
    state = np.zeros(6)
    mine = _accsim_state(energy, state, 0.0)
    theirs = _xtrack_state(pure_bend_line, energy, state)
    predicted = CONSTANTS_VINTAGE + 2.0 * (MASS0 / energy) ** 2
    assert mine[5] / theirs[5] - 1.0 == pytest.approx(predicted, rel=0.02)


def test_the_residual_is_a_constant_plus_a_term_that_dies_as_one_over_gamma_squared(
    pure_bend_line,
) -> None:
    """Across a factor 80 in energy: the constants vintage stays, the approximation goes."""
    state = np.zeros(6)
    for energy in (1.0e9, 5.0e9, 20.0e9, 80.0e9):
        mine = _accsim_state(energy, state, 0.0)
        theirs = _xtrack_state(pure_bend_line, energy, state)
        predicted = CONSTANTS_VINTAGE + 2.0 * (MASS0 / energy) ** 2
        assert mine[5] / theirs[5] - 1.0 == pytest.approx(predicted, rel=0.03)


def test_the_kick_matches_off_axis_on_a_combined_function_magnet(combined_line) -> None:
    """The gradient sampled at the particle's own ``x`` — the term ``J_x`` rests on.

    accsim's curved quadrupole *is* ``xt.Bend(model='mat-kick-mat')`` with one uniform
    kick (L4, to 1.0e-16), so the maps agree exactly and the only thing being compared
    here is the radiation: the field at the mid-point, its perpendicular projection, and
    the path length.
    """
    energy = 20.0e9
    predicted = CONSTANTS_VINTAGE + 2.0 * (MASS0 / energy) ** 2
    cases = [
        np.array([2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 2e-3, 0.0, 0.0, 0.0]),
        np.array([0.0, 1e-3, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1e-3, 0.0, 0.0]),
        np.array([1e-3, 5e-4, 1e-3, 5e-4, 0.0, 1e-3]),
    ]
    for state in cases:
        mine = _accsim_state(energy, state, 0.6)
        theirs = _xtrack_state(combined_line, energy, state)
        loss_mine, loss_theirs = mine[5] - state[5], theirs[5] - state[5]
        assert loss_mine / loss_theirs - 1.0 == pytest.approx(predicted, rel=0.05)
        # and the transverse momenta, which is where the damping actually comes from
        for i in (1, 3):
            assert mine[i] == pytest.approx(theirs[i], rel=2e-8, abs=1e-16)


def test_xtracks_own_sub_stepping_follows_the_predicted_law(pure_bend_line) -> None:
    r"""``dE(N) = U (1 - (N-1)/N * U/E)`` — the reason the comparison needs ``N = 1``.

    Asserted here rather than assumed, because it is the whole explanation for a 2.4e-3
    apparent disagreement, and because it is the same law accsim's own analytic suite
    asserts for slicing a lattice.
    """
    energy = 20.0e9
    state = np.zeros(6)
    u = -_accsim_state(energy, state, 0.0)[5]  # accsim's single lumped kick, in delta
    for n in (1, 2, 4, 8):
        pure_bend_line["b"].num_multipole_kicks = n
        loss = -_xtrack_state(pure_bend_line, energy, state)[5]
        predicted = u * (1.0 - (n - 1) / n * u) / (1.0 + CONSTANTS_VINTAGE)
        assert loss == pytest.approx(predicted, rel=2e-4)
    pure_bend_line["b"].num_multipole_kicks = 1


def test_xtracks_default_adaptive_integrator_is_eight_uniform_steps_here(
    pure_bend_line,
) -> None:
    """The concrete trap: the default is *not* one kick, so a naive comparison is out by
    ``7/8 * U/E`` and looks like a wrong coefficient in the radiation constant.

    ``adaptive`` chooses for itself only when ``num_multipole_kicks`` is left at its
    constructed ``0``; set to anything else it honours that instead (and 8 is *not* 8 —
    it groups them). So the reset below is part of the statement, not tidying.
    """
    energy = 20.0e9
    state = np.zeros(6)
    pure_bend_line["b"].integrator = "uniform"
    pure_bend_line["b"].num_multipole_kicks = 8
    eight = _xtrack_state(pure_bend_line, energy, state)[5]
    pure_bend_line["b"].num_multipole_kicks = 1
    one = _xtrack_state(pure_bend_line, energy, state)[5]
    pure_bend_line["b"].num_multipole_kicks = 0
    pure_bend_line["b"].integrator = "adaptive"
    default = _xtrack_state(pure_bend_line, energy, state)[5]
    assert default == pytest.approx(eight, rel=1e-12)
    # delta is negative, so the (N-1)/N deficit in the LOSS is a positive shift here
    assert default / one - 1.0 == pytest.approx((7 / 8) * one, rel=2e-3)
    pure_bend_line["b"].integrator = "uniform"
    pure_bend_line["b"].num_multipole_kicks = 1


def test_xtracks_perpendicular_projection_has_a_sign_error_that_only_bites_at_large_py(
    combined_line, monkeypatch
) -> None:
    r"""``xtrack/beam_elements/elements_src/track_magnet_radiation.h::direction_of_motion``
    computes ``iis = sqrt(1 - iix*iix + iiy*iiy)``. The ``+`` on the vertical term is a
    sign error: the direction cosines of a unit vector need ``1 - ix^2 - iy^2``. accsim
    uses the correct form, so the two part company at ``O(py^2)``.

    Gated from both sides, which is what makes it an attribution rather than an excuse:
    the disagreement grows as ``py^4`` (invisible at ``1e-3``, ``6e-7`` at ``2e-2``,
    ``2.3e-5`` at ``5e-2``) — quartic, not quadratic, because the two projections differ
    by ``2 B_par^2 iy^2`` and ``B_par = bx ix + by iy`` is itself linear in ``py`` —
    **and** substituting xtrack's sign into accsim's own kick
    reproduces xtrack to the same ``1.19e-8`` residual it shows on axis, at every
    amplitude. Nothing else in the kick is involved.
    """
    energy = 20.0e9
    predicted = CONSTANTS_VINTAGE + 2.0 * (MASS0 / energy) ** 2
    departures = []
    for py in (1e-3, 5e-3, 2e-2, 5e-2):
        state = np.array([1e-3, 5e-4, 1e-3, py, 0.0, 0.0])
        theirs = _xtrack_state(combined_line, energy, state)
        departures.append(_accsim_state(energy, state, 0.6)[5] / theirs[5] - 1.0)
    # accsim's own answer drifts away from xtrack's, quadratically in py
    assert departures[0] == pytest.approx(predicted, rel=0.05)
    assert abs(departures[-1]) > 1e-5
    growth = (departures[3] - predicted) / (departures[2] - predicted)
    assert growth == pytest.approx((5e-2 / 2e-2) ** 4, rel=0.08)

    # ... and with xtrack's sign put back, every amplitude lands on the same residual
    from accsim import radiation_kick as rk

    original = rk._perpendicular_field

    def with_xtracks_sign(bx, by, px, py, delta):  # type: ignore[no-untyped-def]
        ix = np.asarray(px) / (1.0 + np.asarray(delta))
        iy = np.asarray(py) / (1.0 + np.asarray(delta))
        iz = np.sqrt(np.maximum(1.0 - ix * ix + iy * iy, 0.0))  # xtrack's '+'
        b_par = bx * ix + by * iy
        ex, ey, ez = bx - b_par * ix, by - b_par * iy, -b_par * iz
        return np.sqrt(ex * ex + ey * ey + ez * ez)

    monkeypatch.setattr(rk, "_perpendicular_field", with_xtracks_sign)
    assert rk._perpendicular_field is not original
    for py in (1e-3, 5e-3, 2e-2, 5e-2):
        state = np.array([1e-3, 5e-4, 1e-3, py, 0.0, 0.0])
        theirs = _xtrack_state(combined_line, energy, state)
        assert _accsim_state(energy, state, 0.6)[5] / theirs[5] - 1.0 == pytest.approx(
            predicted, rel=0.05
        )
