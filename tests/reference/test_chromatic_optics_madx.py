"""M1 second reference: ``Q''`` against MAD-X, and the scaling law that names the gap.

D3's argument for a second reference is that xsuite deliberately follows MAD-X's
conventions, so a convention error the two share by design would survive an
xtrack-only check. On this milestone that argument pays out directly: the two
references do **not** agree with each other.

On a **bend-free** ring accsim and MAD-X agree on ``Q''`` (and so does xtrack — see
``test_chromatic_optics_xtrack.py``). Once bends are present they separate, while
still agreeing on the tune to nine digits and on first-order chromaticity. The
separation is not a mystery term: it is **exactly zero at zero bending angle and
grows as the square of it**, which is what the scaling test below pins and what M2
is written to resolve.

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
    than about bookkeeping, and it is why M2 looks at the longitudinal constraint
    instead.
    """
    import_madx()
    values = []
    for gamma in (20.0, 2000.0):
        with madx_session() as madx:
            _build_madx(madx, angle=ANG, gamma=gamma)
            values.append(_madx_second_difference(madx)[0])

    assert values[0] == pytest.approx(values[1], rel=1e-6)
