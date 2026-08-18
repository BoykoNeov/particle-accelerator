"""Cross-check the combined-function Dipole (bend + gradient) against xtrack.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

The xtrack ``Bend`` takes a body gradient ``k1`` directly; edges are disabled
(``edge_*_active = 0``) so the R-matrix is the pure combined-function body,
apples-to-apples with ``Dipole(L, theta, k1=...)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    PX,
    PY,
    Dipole,
    Drift,
    Lattice,
    ReferenceParticle,
    X,
    Y,
    natural_chromaticity,
    tunes_on_orbit,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 5.0
LENGTH = 2.0
ANGLE = 0.2


def _xtrack_combined_rmatrix(k1: float) -> np.ndarray:
    ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    bend = xt.Bend(length=LENGTH, angle=ANGLE, k1=k1)
    bend.edge_entry_active = 0
    bend.edge_exit_active = 0
    line = xt.Line(elements=[bend])
    line.particle_ref = ref
    try:
        line.build_tracker()
        res = line.get_R_matrix(particle_on_co=ref.copy())
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return np.asarray(res["R_matrix"])


@pytest.mark.parametrize("k1", [0.3, -0.25])
def test_combined_dipole_matches_xtrack(k1: float) -> None:
    R_xt = _xtrack_combined_rmatrix(k1)
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    R_us = Dipole(LENGTH, ANGLE, k1=k1).matrix(ref)

    np.testing.assert_allclose(R_us, R_xt, rtol=1e-6, atol=1e-9)
    assert np.sign(R_us[PX, X]) == np.sign(R_xt[PX, X])  # K_x = h^2 + k1
    assert np.sign(R_us[PY, Y]) == np.sign(R_xt[PY, Y])  # K_y = -k1


# --------------------------------------------------------------------------
# The expanded map (L4) — and the model family a curved quadrupole belongs to
# --------------------------------------------------------------------------

L_EXACT = 1.0
ANGLE_EXACT = 0.3
K1_EXACT = 0.6

# Large angle **and** large delta, plus a state with nothing but delta. The map is
# exact in delta and paraxial in the angles, so both doors have to be closed: a
# mishandled rigidity would agree at delta = 0, a mis-scaled gradient at x = 0.
_STATES = [
    np.array([3.0e-3, 8.0e-3, -2.0e-3, 5.0e-3, 1.0e-3, 2.0e-3]),
    np.array([2.0e-3, -5.0e-3, 1.5e-3, 4.0e-3, 0.0, 5.0e-2]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, -3.0e-2]),
    np.array([1.0e-2, 4.0e-3, -8.0e-3, 2.0e-3, 5.0e-4, -1.0e-2]),
]


def _xtrack_tracked_cf(model: str, num_kicks: int | None = None, integrator: str | None = None):
    """Track every state in ``_STATES`` through one combined-function ``xt.Bend``."""
    bend = xt.Bend(length=L_EXACT, angle=ANGLE_EXACT, k1=K1_EXACT, model=model)
    bend.edge_entry_active = 0  # accsim's default is a bare body, no edges
    bend.edge_exit_active = 0
    if num_kicks is not None:
        bend.num_multipole_kicks = num_kicks
    if integrator is not None:
        bend.integrator = integrator
    line = xt.Line(elements=[bend])
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    out = []
    for st in _STATES:
        p = xt.Particles(
            mass0=MASS0,
            q0=1,
            gamma0=GAMMA0,
            x=st[0],
            px=st[1],
            y=st[2],
            py=st[3],
            zeta=st[4],
            delta=st[5],
        )
        line.track(p)
        out.append(np.array([p.x[0], p.px[0], p.y[0], p.py[0], p.zeta[0], p.delta[0]]))
    return out


def test_the_curved_quadrupole_map_is_xtracks_mat_kick_mat_to_the_last_bit() -> None:
    r"""accsim's combined-function ``Dipole.track`` **is** ``mat-kick-mat``, to 1e-16.

    ``mat-kick-mat`` with ``num_multipole_kicks = 1`` and the ``uniform`` integrator is
    exactly ``mat(L/2) . kick(L) . mat(L/2)``, which is exactly what
    :meth:`~accsim.elements.dipole.Dipole._track_body` composes — so this is a
    like-for-like comparison of two independent implementations of one model, not an
    agreement up to somebody's splitting.

    It is the gate that pins the **Maxwell curvature-sextupole coefficient** against a
    reference rather than against accsim's own derivation: the kick's ``h k1 L`` and its
    ``2:-1`` split between the planes are the only free things here, and they are the
    ones nothing structural can see. Reading the coefficient out of xtrack's header
    would not have been enough — ``track_magnet_kick.h`` *adds* ``knl[1]`` to
    ``k1_h_correction * length`` in the same expression, so a ``Bend`` that populated
    both would double it; this agreement is the proof that it does not.

    ``zeta`` is included and is the sharp part again: accsim evaluates it as
    ``L(1 - 1/rvv) - (Lambda - L)/rvv`` so that nothing of size ``L`` is subtracted from
    anything else, where xtrack writes the cancelling ``length - length_/rvv``.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    want = _xtrack_tracked_cf("mat-kick-mat", 1, "uniform")
    for st, w in zip(_STATES, want, strict=True):
        got = Dipole(L_EXACT, ANGLE_EXACT, k1=K1_EXACT).track(st, ref)
        np.testing.assert_allclose(got, w, rtol=0.0, atol=1e-15)


def test_the_exact_families_are_a_different_map_and_the_gap_is_the_metric_factor() -> None:
    r"""**The model boundary, asserted rather than discovered.**

    xtrack carries two families for a combined-function bend, and they are not the same
    physics:

    - ``bend-kick-bend`` / the default ``rot-kick-rot`` keep the bend geometry exactly —
      including the curvilinear metric factor ``(1 + h x)`` in ``x' = px(1+hx)/pz`` — and
      put the gradient in as kicks;
    - ``mat-kick-mat`` (MAD-X's ``track_thick_cfd``) solves ``x' = px/(1+delta)``, keeping
      that factor only in the path length. accsim implements this one, because it is the
      only family whose linear part is a *closed form*, and a closed form is what keeps
      ``matrix()`` the exact origin Jacobian of ``track()``.

    They differ here by ``1e-5``, which is far above the previous test's ``1e-15`` and
    reads like a bug if the model is not named. So any reference comparison involving an
    off-axis combined-function bend has to say which family it means, and this test is
    where that is written down.

    The chromatic consequence is measured on the accsim side in
    ``tests/analytic/test_curved_quadrupole.py``; what is added here is the
    confirmation from xtrack's own numbers, in the ring test below.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    expanded = _xtrack_tracked_cf("mat-kick-mat", 1, "uniform")
    exact = _xtrack_tracked_cf("bend-kick-bend")

    worst = max(float(np.abs(a - b).max()) for a, b in zip(exact, expanded, strict=True))
    assert worst > 1.0e-6, "the two model families must be distinguishable at these amplitudes"

    # accsim is on the expanded side of that split, deliberately and by nine orders.
    for st, w in zip(_STATES, expanded, strict=True):
        got = Dipole(L_EXACT, ANGLE_EXACT, k1=K1_EXACT).track(st, ref)
        assert np.abs(got - w).max() < 1.0e-15


# --------------------------------------------------------------------------
# The chromaticity of the two families, from xtrack's side
# --------------------------------------------------------------------------

RING_GAMMA0 = 20.0
N_BENDS, LB, ANGB, KF, LD = 8, 1.0, 2.0 * np.pi / 16.0, 0.35, 0.4


def _accsim_ring() -> Lattice:
    ref = ReferenceParticle.from_gamma(MASS0, RING_GAMMA0)
    els: list = []
    for i in range(N_BENDS):
        els += [Dipole(LB, ANGB, k1=+KF if i % 2 == 0 else -KF), Drift(LD)]
    return Lattice(els, ref)


def _xtrack_ring_chromaticity(model: str, num_kicks: int | None, integrator: str | None):
    els: list = []
    for i in range(N_BENDS):
        b = xt.Bend(length=LB, angle=ANGB, k1=+KF if i % 2 == 0 else -KF, model=model)
        b.edge_entry_active = 0
        b.edge_exit_active = 0
        if num_kicks is not None:
            b.num_multipole_kicks = num_kicks
        if integrator is not None:
            b.integrator = integrator
        els += [b, xt.Drift(length=LD)]
    line = xt.Line(elements=els)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=RING_GAMMA0)
    try:
        line.build_tracker()
        tw = line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return float(tw.dqx), float(tw.dqy)


def test_the_two_families_chromaticities_bracket_accsims_two_answers() -> None:
    r"""**The whole of L4's model boundary, confirmed from xtrack's side.**

    On one alternating-gradient arc of eight bending combined-function magnets, xtrack's
    two families give two different chromaticities, and accsim's two routes land on one
    each:

    - xtrack's **exact** families (``bend-kick-bend``, ``rot-kick-rot``) agree with
      accsim's :func:`~accsim.natural_chromaticity` — F2's integral, including the
      curvilinear-metric group and the curvature-sextupole feed-down. That is the
      deliverable, and this re-confirms it to ~1e-6 relative;
    - xtrack's **expanded** family (``mat-kick-mat``, converged over its own default
      slicing) agrees with what accsim's *tracking* gives when the same magnet is sliced,
      which is F2 minus the metric group.

    So the gap between accsim's tracked and analytic chromaticity on such a ring is not
    accsim's: it is the expanded family's, it is the same size in both codes, and it is
    named in closed form. Measuring it from both sides is what turns "tracking disagrees"
    into a model boundary. ``natural_chromaticity`` is untouched by L4 and is the
    function to use.
    """
    lat = _accsim_ring()
    analytic = natural_chromaticity(lat, slices=1024)

    for model in ("bend-kick-bend", "rot-kick-rot"):
        dqx, dqy = _xtrack_ring_chromaticity(model, None, None)
        assert dqx == pytest.approx(analytic[0], rel=1e-5), model
        assert dqy == pytest.approx(analytic[1], rel=1e-4), model

    # The expanded family, converged: a different number, and the one accsim tracks.
    exp_x, exp_y = _xtrack_ring_chromaticity("mat-kick-mat", None, None)
    assert abs(exp_x - analytic[0]) > 0.2  # the families really do disagree

    ref = lat.ref
    sliced: list = []
    for i in range(N_BENDS):
        k1 = +KF if i % 2 == 0 else -KF
        sliced += [Dipole(LB / 16, ANGB / 16, k1=k1) for _ in range(16)]
        sliced.append(Drift(LD))
    lat_sliced = Lattice(sliced, ref)
    qx_p, qy_p = tunes_on_orbit(lat_sliced, delta=+1.0e-5)
    qx_m, qy_m = tunes_on_orbit(lat_sliced, delta=-1.0e-5)
    tracked = ((qx_p - qx_m) / 2.0e-5, (qy_p - qy_m) / 2.0e-5)

    assert tracked[0] == pytest.approx(exp_x, rel=2e-3)
    assert tracked[1] == pytest.approx(exp_y, rel=2e-3)


def test_a_straight_gradient_ring_has_no_family_split_in_the_chromaticity() -> None:
    r"""``h = 0`` collapses the model boundary, which is why the 56% control could close.

    The metric factor is ``(1 + h x)`` and the Maxwell kick is ``h k1 L``, so both vanish
    identically for a zero-angle gradient magnet — and with no bends there is no
    dispersion for either to feed on. The consequence is the one worth asserting: on a
    ring of *straight* gradient magnets, xtrack's expanded and exact families give the
    **same** chromaticity, accsim's analytic integral gives it too, and so does accsim's
    tracking. Four routes, one number.

    Contrast the bending ring above, where the same four routes split into two pairs.
    That contrast is the whole of L4's model boundary, and it is why reading
    ``tests/analytic/test_exact_quadrupole.py``'s 100% as covering the bending case is a
    mistake: that control's magnet has ``angle = 0``.

    ``bend-kick-bend`` needs one caveat and it is worth keeping rather than tuning away:
    for a *straight* magnet that model is a drift-kick-drift decomposition, so its
    gradient lives entirely in thin kicks and its answer carries its own **splitting**
    error — ``0.6%`` on this ring at its default kick count. That is an integrator
    residual, not a model difference, and the test says so by driving the kick count up
    and watching it converge onto the same number as everyone else. The contrast with the
    bending ring is exactly the point: there, no amount of slicing brings the two families
    together, because what separates them is a term neither integrator can supply.
    """
    ref = ReferenceParticle.from_gamma(MASS0, RING_GAMMA0)
    els: list = []
    for i in range(N_BENDS):
        els += [Dipole(LB, 0.0, k1=+KF if i % 2 == 0 else -KF), Drift(LD)]
    lat = Lattice(els, ref)
    analytic = natural_chromaticity(lat, slices=1024)

    qx_p, qy_p = tunes_on_orbit(lat, delta=+1.0e-5)
    qx_m, qy_m = tunes_on_orbit(lat, delta=-1.0e-5)
    tracked = ((qx_p - qx_m) / 2.0e-5, (qy_p - qy_m) / 2.0e-5)
    assert tracked[0] == pytest.approx(analytic[0], rel=1e-5)
    assert tracked[1] == pytest.approx(analytic[1], rel=1e-5)

    def xt_ring(model: str, num_kicks: int | None = None):
        line_els: list = []
        for j in range(N_BENDS):
            b = xt.Bend(length=LB, angle=0.0, k1=+KF if j % 2 == 0 else -KF, model=model)
            b.edge_entry_active = 0
            b.edge_exit_active = 0
            if num_kicks is not None:
                b.num_multipole_kicks = num_kicks
            line_els += [b, xt.Drift(length=LD)]
        line = xt.Line(elements=line_els)
        line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=RING_GAMMA0)
        try:
            line.build_tracker()
            tw = line.twiss(method="4d")
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
        return float(tw.dqx), float(tw.dqy)

    # The expanded family is a thick closed form here, so it lands on the number directly.
    dqx, dqy = xt_ring("mat-kick-mat")
    assert dqx == pytest.approx(analytic[0], rel=1e-4)
    assert dqy == pytest.approx(analytic[1], rel=1e-4)

    # The exact family is drift-kick-drift for a straight magnet: its residual is its own
    # integrator's, and it converges away, which is what says it is not a model gap.
    coarse = xt_ring("bend-kick-bend")
    fine = xt_ring("bend-kick-bend", 64)
    assert abs(coarse[0] - analytic[0]) > 5e-4  # there is something to converge
    assert abs(fine[0] - analytic[0]) < 0.1 * abs(coarse[0] - analytic[0])
    assert fine[0] == pytest.approx(analytic[0], rel=1e-4)
    assert fine[1] == pytest.approx(analytic[1], rel=1e-4)
