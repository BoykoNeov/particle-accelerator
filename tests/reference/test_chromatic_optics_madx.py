"""M1 second reference: ``Q''`` against MAD-X, and the scaling law that names the gap.

D3's argument for a second reference is that xsuite deliberately follows MAD-X's
conventions, so a convention error the two share by design would survive an
xtrack-only check. On this milestone that argument pays out directly: the two
references do **not** agree with each other.

On a **bend-free** ring accsim and MAD-X agree on ``Q''`` (and so does xtrack — see
``test_chromatic_optics_xtrack.py``). Once bends are present they separate, while
still agreeing on the tune to nine digits and on first-order chromaticity. The
separation is **exactly zero at zero bending angle and grows as the square of it**,
which is what the scaling test below pins.

**M2 resolved it, and MAD-X's share is mostly the same cause as xtrack's.** The
drift is the element the codes model differently: accsim's is exact
(``x += L px/pz``), MAD-X's and xtrack's defaults are paraxial
(``x += L px/(1+delta)``), and the two coincide exactly whenever the closed orbit is
straight — which is why the bend-free control agreed. On M2's five-element minimal
ring, where the ``Q''`` of *each model* is derived independently at sixty digits,
MAD-X lands ``7.0e-4`` from the paraxial number in ``x`` and ``7.3e-4`` in ``y`` — a
residual of its own that is the **same size in both planes** — while the drift-model
split itself is ``1.42e-2`` in ``x`` and ``4.78e-3`` in ``y``. So the drift accounts
for 95% of MAD-X's gap horizontally and 82% of it vertically; the difference between
those two percentages is the *denominator* changing, not MAD-X behaving differently.

The leftover is MAD-X's own — its TWISS transfer maps are second-order expansions
rather than the exact sector-bend flow — and it **cannot** be removed, because MAD-X's
TWISS offers no exact-drift option. So MAD-X is named here rather than reconciled:
agreement with it is unreachable by construction, and its number is not the one to
believe.

MAD-X's ``DDQ1`` column is deliberately **not** read. Second-difference conventions
differ between codes by exactly the factor a milestone like this exists to catch, so
MAD-X's own ``Q1`` is sampled at three ``DELTAP`` values and differenced here — the
same trick that cleared xtrack's ``ddqx`` formula of suspicion.

MAD-X is cheap compared with xtrack (no JIT compile), which is why the angle-scaling
sweep lives in this file rather than next door.

Marked ``reference``: skips (not fails) when cpymad is unavailable.
"""

from __future__ import annotations

import pytest
from _madx import import_madx, madx_session

from accsim import (
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    chromaticity,
    second_order_chromaticity,
    tunes,
)

pytestmark = pytest.mark.reference

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
LQ, K1, LD, LB, ANG, N_CELLS = 0.3, 1.2, 0.5, 1.0, 0.12, 3
DELTA = 1e-3

_CELL = ["qf", "dl", "dl", "bb", "qd", "bb", "dl"]


def _lattice(angle: float) -> Lattice:
    """The arc at a given bending angle; ``angle = 0`` replaces the bends by drifts."""
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    els: list = []
    for _ in range(N_CELLS):
        els += [
            Quadrupole(LQ, K1),
            Drift(LD),
            Drift(LD),
            Dipole(LB, angle) if angle else Drift(LB),
            Quadrupole(LQ, -K1),
            Dipole(LB, angle) if angle else Drift(LB),
            Drift(LD),
        ]
    return Lattice(els, ref)


def _build_madx(madx, *, angle: float, gamma: float = GAMMA0) -> None:
    """The same arc in MAD-X. ``angle = 0`` gives a drift, matching :func:`_lattice`."""
    madx.input(f"qf: quadrupole, l={LQ}, k1={K1};")
    madx.input(f"qd: quadrupole, l={LQ}, k1={-K1};")
    madx.input(f"bb: sbend, l={LB}, angle={angle};" if angle else f"bb: drift, l={LB};")
    madx.input(f"dl: drift, l={LD};")
    madx.input("ring: line=({});".format(", ".join(_CELL * N_CELLS)))
    madx.input(f"beam, particle=proton, gamma={gamma};")
    madx.input("use, sequence=ring;")


def _madx_tunes(madx, deltap: float) -> tuple[float, float]:
    madx.input(f"twiss, deltap={deltap:.12g};")
    return float(madx.table.summ.q1[0]), float(madx.table.summ.q2[0])


def _madx_second_difference(madx, delta: float = DELTA) -> tuple[float, float]:
    """``(Q''_x, Q''_y)`` from MAD-X's *own* ``Q1``/``Q2`` at three ``DELTAP`` values."""
    qp = _madx_tunes(madx, +delta)
    q0 = _madx_tunes(madx, 0.0)
    qm = _madx_tunes(madx, -delta)
    return (
        (qp[0] - 2.0 * q0[0] + qm[0]) / delta**2,
        (qp[1] - 2.0 * q0[1] + qm[1]) / delta**2,
    )


def test_madx_agrees_on_the_tune_and_first_order_chromaticity_with_bends() -> None:
    """The shared baseline: ``Q`` to nine digits, ``Q'`` to five, on the bendy ring.

    Asserted first and deliberately: without it, the second-order disagreement below
    could be dismissed as two codes building different machines. They are building
    the same machine, and they agree about it everywhere except the curvature of the
    tune-versus-momentum curve.
    """
    import_madx()
    lattice = _lattice(ANG)
    with madx_session() as madx:
        _build_madx(madx, angle=ANG)
        q_madx = _madx_tunes(madx, 0.0)
        qp = _madx_tunes(madx, +DELTA)
        qm = _madx_tunes(madx, -DELTA)

    q_ours = tunes(lattice)
    assert q_ours[0] == pytest.approx(q_madx[0], abs=1e-9)
    assert q_ours[1] == pytest.approx(q_madx[1], abs=1e-9)

    # ``slices`` is raised well above its default here, and the reason is worth
    # recording: ``natural_chromaticity`` integrates the beta-weighted gradient by
    # trapezoidal sub-slicing, whose error falls as ``1/slices^2``. On this arc the
    # default 64 leaves 1.5e-5 relative — larger than the agreement being asserted,
    # and *not* a physics gap: the residual against the tracked derivative measures
    # 6.9e-5, 4.4e-6, 3.0e-7, 4.7e-8 at 16, 64, 256 and 1024 slices, i.e. it
    # converges on the tracked answer at exactly the trapezoid's order.
    dq_madx = (qp[0] - qm[0]) / (2.0 * DELTA)
    assert chromaticity(lattice, 1024)[0] == pytest.approx(dq_madx, rel=1e-5)


def test_second_order_chromaticity_matches_madx_without_bends() -> None:
    """``Q''`` agrees with the second reference where there are no bends.

    Together with the xtrack control this makes the bend-free result a genuine
    three-code agreement, which is what lets the disagreement below be attributed to
    the bends rather than to accsim's second-difference machinery.
    """
    import_madx()
    with madx_session() as madx:
        _build_madx(madx, angle=0.0)
        theirs = _madx_second_difference(madx)

    ours = second_order_chromaticity(_lattice(0.0), delta=DELTA)
    assert ours[0] == pytest.approx(theirs[0], rel=1e-4)
    assert ours[1] == pytest.approx(theirs[1], rel=1e-4)


def test_the_gap_is_zero_without_bends_and_quadratic_in_the_bending_angle() -> None:
    r"""The disagreement's **scaling law** — the sharpest thing known about it.

    Sweeping the bending angle and differencing accsim against MAD-X at each:

    - at ``angle = 0`` the two agree to round-off, so the gap is not a constant
      offset somewhere in the machinery;
    - as the angle grows the gap grows from zero **as its square** — ``gap/angle^2``
      measures ``8.91`` and ``8.22`` at ``0.03`` and ``0.06`` rad, tending to a
      constant as the angle shrinks, with higher-order terms taking over by ``0.24``.

    **M2 explains the law rather than merely pinning it, and the explanation is the
    drift.** The exact and paraxial drift maps differ by the relative factor
    ``(px^2 + py^2)/2``, so they are the same map whenever the closed orbit is
    straight. With bends the orbit picks up ``px ~ D_px delta``, and ``D_px`` is
    proportional to the bending angle — so the difference is ``O(angle^2 delta^2)``:
    zero without bends, quadratic in the angle, and landing on the second derivative
    of the tune while leaving ``Q`` and ``Q'`` untouched. The analytic suite
    (``test_chromatic_arbiter.py``) reproduces this same sweep inside a sixty-digit
    arbiter with **neither** reference code involved, which is what turns the
    coincidence of shape into a cause.

    A quadratic law that vanishes with the bending angle, while ``Q`` and ``Q'`` stay
    in agreement, points at something proportional to the ring's dispersion acting
    twice. The leading suspect is what each code holds fixed **longitudinally** when
    it closes an off-momentum orbit: accsim's ``closed_orbit_nonlinear`` fixes
    ``zeta = 0`` and ``delta`` at the entrance, and path length through a bend
    depends on ``delta`` where through a drift it does not. That is M2's hypothesis
    and M2's gate.

    Gating the **order** rather than any single value is what makes this survive a
    version bump in either code: a changed model would move the values, but only a
    changed *mechanism* would move the exponent.
    """
    import_madx()
    angles = (0.03, 0.06)

    gaps = []
    for angle in (0.0, *angles):
        with madx_session() as madx:
            _build_madx(madx, angle=angle)
            theirs = _madx_second_difference(madx)[0]
        ours = second_order_chromaticity(_lattice(angle), delta=DELTA)[0]
        gaps.append(ours - theirs)

    # No bends, no gap.
    assert abs(gaps[0]) < 1e-5

    # With bends, a gap quadratic in the angle: doubling the angle quadruples it,
    # and the per-angle-squared coefficient is stable across the pair.
    assert gaps[1] > 1e-3  # genuinely present already at the smaller angle
    assert gaps[2] / gaps[1] == pytest.approx(4.0, rel=0.15)
    coefficients = [g / a**2 for g, a in zip(gaps[1:], angles, strict=True)]
    assert coefficients[1] == pytest.approx(coefficients[0], rel=0.15)


def test_the_madx_split_is_not_an_energy_variable_convention() -> None:
    r"""MAD-X's offset does not move with ``gamma0``, so it is not ``deltap`` vs ``pt``.

    The obvious suspect for a second-order-only disagreement is the energy variable:
    MAD-X works internally in ``PT`` (an *energy* deviation) while ``DELTAP`` is a
    momentum one, and a nonlinear conversion between them would inject a spurious
    curvature into a second difference. It would also scale with ``beta0``, and this
    shows it does not — the same ``Q''`` comes back from a barely-relativistic ring
    and an ultra-relativistic one.

    Ruling this out is what makes the split a statement about the *models* rather
    than about bookkeeping — and M2 confirmed that reading: the models differ in the
    **drift**, which no change of energy variable could ever produce.
    """
    import_madx()
    values = []
    for gamma in (20.0, 2000.0):
        with madx_session() as madx:
            _build_madx(madx, angle=ANG, gamma=gamma)
            values.append(_madx_second_difference(madx)[0])

    assert values[0] == pytest.approx(values[1], rel=1e-6)


def test_madx_sits_on_the_paraxial_drift_answer_with_a_small_residual_of_its_own() -> None:
    r"""M2's placement of MAD-X: 95% the drift model, 5% MAD-X's own maps.

    On the five-element minimal ring of ``tests/_m2_minimal_ring.py`` the ``Q''`` of
    each drift model is derived from lab-frame geometry at sixty digits, with no code
    in the room:

        exact drift     Q''_x = 0.3073788909
        paraxial drift  Q''_x = 0.2932235794

    MAD-X returns ``0.2925214`` in ``x`` and ``0.2945490`` in ``y``. Those are
    ``1.49e-2`` and ``4.04e-3`` from the exact numbers, and ``7.0e-4`` and ``7.3e-4``
    from the paraxial ones. The residual is **the same size in both planes** while the
    drift-model split is three times larger horizontally than vertically — which is
    why the drift accounts for 95% of the horizontal gap and 82% of the vertical one
    without MAD-X doing anything different in the two. The overwhelming majority of
    MAD-X's disagreement with accsim is the same paraxial drift that explains
    xtrack's, and what is left over is small, its own, and not the drift.

    Two things make this a decisive placement rather than a coincidence of magnitude.
    The ring uses **thin** quadrupoles, so a thick quadrupole's momentum-dependent
    focusing — another place codes differ in their expansion order — is absent, and
    the residual has only MAD-X's sector-bend map left to come from. And the
    assertion is two-sided: MAD-X must be far closer to the paraxial answer than to
    the exact one, *and* must not sit exactly on it, because a MAD-X that reproduced
    the paraxial arbiter to round-off would mean its bend map was exact, which it is
    not.

    The residual is *not* asserted to shrink under any option, because there is none:
    MAD-X's TWISS has no exact-drift setting.
    """
    from _m2_minimal_ring import ANG as M_ANG
    from _m2_minimal_ring import KF
    from _m2_minimal_ring import LB as M_LB
    from _m2_minimal_ring import LD as M_LD
    from _m2_minimal_ring import second_order_chromaticity as arbiter

    step = 1.25e-3

    def build(madx) -> None:
        madx.input(f"qf: multipole, knl={{0, {KF}}};")
        madx.input(f"qd: multipole, knl={{0, {-KF}}};")
        madx.input(f"bb: sbend, l={M_LB}, angle={M_ANG};")
        madx.input(f"dl: drift, l={M_LD};")
        madx.input("ring: line=(qf, dl, bb, qd, dl);")
        madx.input(f"beam, particle=proton, gamma={GAMMA0};")
        madx.input("use, sequence=ring;")

    import_madx()
    with madx_session() as madx:
        build(madx)
        theirs = _madx_second_difference(madx, step)

    exact, paraxial = arbiter(exact_drift=True), arbiter(exact_drift=False)
    for index, plane in ((0, "x"), (1, "y")):
        to_exact = abs(theirs[index] - exact[plane])
        to_paraxial = abs(theirs[index] - paraxial[plane])
        assert to_paraxial < 0.25 * to_exact  # overwhelmingly the paraxial answer
        assert to_paraxial > 1e-5  # but not exactly it: MAD-X's own bend map remains

    # ...and that leftover is the same size in both planes, which is what identifies it
    # as one property of MAD-X's maps rather than two unrelated discrepancies.
    residuals = [abs(theirs[i] - paraxial[p]) for i, p in ((0, "x"), (1, "y"))]
    assert residuals[0] == pytest.approx(residuals[1], rel=0.1)
