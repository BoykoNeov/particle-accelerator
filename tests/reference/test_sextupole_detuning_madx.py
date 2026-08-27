r"""O3 against MAD-X PTC: the anharmonicity a second, all-orders code computes.

``ptc_normal`` builds the nonlinear normal form of the one-turn map in a completely
separate Fortran codebase and reports the anharmonicity as ``anhx``/``anhy``. That is
the only external check available for this milestone — xtrack's
``get_amplitude_detuning_coefficients`` is tracking plus NAFF and needs ``nafflib``,
which is not installed, and its ``rdt_first_order_perturbation`` is a first-order
formula from twiss plus strengths, i.e. closer to a reimplementation than a check.

Three things about this comparison had to be measured before it meant anything, and
each is a test below rather than a constant in the source.

**1. PTC reports ``dQ/d(2J)``, half of accsim's ``dQ/dJ``.** Not recalled — calibrated
here against the one quantity both codes already agree on, the octupole's
``k3l beta_x^2/(16 pi)``. A comparison written without the factor is wrong by exactly
two and reads like a missing ``1/2`` in the derivation.

**2. PTC's ring detunes with the sextupoles switched off.** ``exact=true`` gives PTC an
exact drift, whose ``x += L px/pz`` is nonlinear all by itself; on the fixture below it
contributes ``0.127`` where the sextupoles contribute ``0.54``. accsim's
:class:`~accsim.elements.drift.Drift` is exact too and does the same thing under
tracking. :func:`~accsim.twiss.sextupole_detuning` reports exactly zero there, correctly
— it is the sextupoles' contribution, not the ring's total anharmonicity — so every
comparison here is a **difference** against the same ring with ``k2 = 0``, the shape
J2's tracked gate uses.

**3. The agreement is exact, not asymptotic, and that retires a gate that was
pre-committed.** The roadmap's O3 entry expected the primary gate to be a scan: PTC
all-orders against a second-order formula must disagree at any fixed strength, so the
residual should fall as ``k2^2``. It does not, because ``anhx(1,0,0)`` *is* the quartic
coefficient of the normal form — the same object this package computes — so once the
kinematic baseline is out, the two agree to round-off and ``no = 4, 5, 6`` return
bit-identical values. The exponent scan was a hedge against a mismatch that does not
exist; the exact comparison is strictly stronger, and the scan is kept where all orders
genuinely do enter, which is the *tracked* gate in ``tests/analytic/``.

Because both sides now compute the same object, this file is deliberately **not** the
milestone's only evidence. The independent legs are the two-quadrupole exact-trace
anchor and the tracked measurement, both in ``tests/analytic/test_sextupole_detuning.py``.
What PTC adds that neither can: it reaches the ``3 Q_x`` denominator, by agreeing while
the answer diverges by four orders of magnitude next to the third-integer resonance.

Marked ``reference``: skips (not fails) when cpymad is unavailable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _madx import madx_session

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    ThinSextupole,
    amplitude_detuning,
    sextupole_detuning,
    total_detuning,
    tunes,
)

pytestmark = pytest.mark.reference

MASS0, GAMMA0 = 938.27208816e6, 20.0
ENERGY_GEV = 10.0

#: The working point of the generic fixture.  Qx != Qy so the Qx +- 2Qy lines do not
#: collapse onto Qx and 3Qx; all four driven lines sit >= 0.11 from an integer.
KF, KD = 0.80, -0.90

#: ``s`` -> weight for the three sextupoles.  Different positions, so beta_x and beta_y
#: differ between them and between the planes; unequal weights, including a sign change,
#: so no accidental symmetry can make a wrong cross term invisible.
SEXTS = {5.5: 1.0, 8.4: -0.7, 11.6: 0.45}

#: Where the octupole goes for the calibration and the combined-magnet prediction.
OCTU_AT = 2.4


def _madx_ring(k2_scale: float = 1.0, k3l: float = 0.0, kf: float = KF, kd: float = KD) -> str:
    """The fixture as a MAD-X sequence: ``Q(kf) D Q(kd) D`` four times, 12 m, no bends.

    Bend-free on purpose — no dispersion at the sextupoles, so no feed-down moving the
    linear tune underneath the measurement.
    """
    elems = [(0.25 + 3.0 * c, f"qf{c}: qf") for c in range(4)]
    elems += [(1.75 + 3.0 * c, f"qd{c}: qd") for c in range(4)]
    elems += [
        (pos, f"sx{i}: multipole, knl={{0,0,{w * k2_scale:.12g}}}")
        for i, (pos, w) in enumerate(SEXTS.items())
    ]
    if k3l:
        elems.append((OCTU_AT, f"oc: multipole, knl={{0,0,0,{k3l:.12g}}}"))
    body = "\n".join(f"      {d}, at={pos};" for pos, d in sorted(elems))
    return f"""
    beam, particle=proton, energy={ENERGY_GEV}, sequence=ring;
    qf: quadrupole, l=0.5, k1= {kf};
    qd: quadrupole, l=0.5, k1= {kd};
    ring: sequence, l=12.0, refer=centre;
{body}
    endsequence;
    """


def _accsim_ring(
    ref: ReferenceParticle,
    k2_scale: float = 1.0,
    k3l: float = 0.0,
    kf: float = KF,
    kd: float = KD,
) -> Lattice:
    """The identical ring in accsim, element for element."""
    thin = {round(p, 9): w * k2_scale for p, w in SEXTS.items()}
    els: list = []
    s = 0.0
    for _ in range(4):
        for k in (kf, kd):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            inside = sorted(p for p in thin if s < p < s + 1.0)
            inside += [OCTU_AT] if (k3l and s < OCTU_AT < s + 1.0) else []
            if inside:
                at = 0.0
                for p in sorted(inside):
                    els.append(Drift(p - s - at))
                    els.append(ThinOctupole(k3l) if p == OCTU_AT else ThinSextupole(thin[p]))
                    at = p - s
                els.append(Drift(1.0 - at))
            else:
                els.append(Drift(1.0))
            s += 1.0
    return Lattice(els, ref)


def _ptc_anharmonicity(sequence: str, order: int = 4) -> np.ndarray:
    """``[[dQx/d2Jx, dQx/d2Jy], [dQy/d2Jx, dQy/d2Jy]]`` from ``ptc_normal``.

    Read live out of the ``normal_results`` table; nothing is transcribed. The rows are
    selected *before* the call, which is how ``ptc_normal`` decides what to compute.
    """
    with madx_session() as madx:
        madx.input(sequence)
        madx.use(sequence="ring")
        madx.twiss(sequence="ring")
        madx.input(f"""
        ptc_create_universe;
        ptc_create_layout, model=2, method=6, nst=5, exact=true;
          select_ptc_normal, q1=0, q2=0;
          select_ptc_normal, anhx=1,0,0;
          select_ptc_normal, anhy=0,1,0;
          select_ptc_normal, anhx=0,1,0;
        ptc_normal, closed_orbit, normal, icase=4, no={order};
        ptc_end;
        """)
        t = madx.table.normal_results
        rows = {
            (str(n), int(o1), int(o2)): float(v)
            for n, o1, o2, v in zip(t.name, t.order1, t.order2, t.value, strict=True)
        }
    xy = rows[("anhx", 0, 1)]
    return np.array([[rows[("anhx", 1, 0)], xy], [xy, rows[("anhy", 0, 1)]]])


def _ptc_sextupole_part(k2_scale: float = 1.0, k3l: float = 0.0, **kw) -> np.ndarray:
    """PTC's anharmonicity with the magnets on, minus the same ring with them off.

    In accsim's ``dQ/dJ`` units. The subtraction removes the exact drift's own kinematic
    detuning, which no magnet is responsible for and which this package does not claim.
    """
    on = _ptc_anharmonicity(_madx_ring(k2_scale, k3l, **kw))
    off = _ptc_anharmonicity(_madx_ring(0.0, 0.0, **kw))
    return PTC_TO_DQDJ * (on - off)


#: PTC's anharmonicity is ``dQ/d(2J)``; accsim's is ``dQ/dJ``.  Measured, not recalled —
#: :func:`test_ptc_reports_dq_d2j_and_the_factor_is_measured_on_the_octupole` derives this
#: number from the octupole formula the two codes already share.
PTC_TO_DQDJ = 2.0


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def test_the_two_rings_are_the_same_machine(ref: ReferenceParticle) -> None:
    """Before anything is compared: accsim and MAD-X must agree on the linear optics.

    A tune mismatch here would show up downstream as a detuning mismatch, and the
    resonance denominators make ``dQ/dJ`` far more tune-sensitive than the tune itself —
    so this control has to come first.
    """
    with madx_session() as madx:
        madx.input(_madx_ring())
        madx.use(sequence="ring")
        madx.twiss(sequence="ring")
        qx, qy = madx.table.summ.q1[0], madx.table.summ.q2[0]
    ours = tunes(_accsim_ring(ref))
    assert ours == pytest.approx((qx, qy), rel=0, abs=1e-9)


def test_ptc_reports_dq_d2j_and_the_factor_is_measured_on_the_octupole(
    ref: ReferenceParticle,
) -> None:
    """The factor of two, pinned against the quantity both codes independently agree on.

    An octupole detunes at first order, ``dQ_x/dJ_x = k3l beta_x^2/(16 pi)``, and that
    formula is derived from scratch in ``tests/analytic/test_amplitude_detuning.py``. So
    the ratio between PTC's number and accsim's is a *measurement* of PTC's action
    convention, not an assumption about it — and the residual is real higher order in
    ``k3l``, which is why the tolerance is loose while the conclusion (exactly one half)
    is not.
    """
    k3l = 50.0
    ours = amplitude_detuning(_accsim_ring(ref, k2_scale=0.0, k3l=k3l))
    theirs = _ptc_anharmonicity(_madx_ring(0.0, k3l)) - _ptc_anharmonicity(_madx_ring(0.0, 0.0))
    for i, j in ((0, 0), (0, 1), (1, 1)):
        assert theirs[i, j] / ours[i, j] == pytest.approx(0.5, rel=2e-3)
    assert PTC_TO_DQDJ == 2.0


def test_ptc_detunes_with_no_sextupoles_and_accsim_correctly_reports_zero(
    ref: ReferenceParticle,
) -> None:
    """The trap: an exact drift is a nonlinear map, so the empty ring is not quiet.

    This is the reason every other comparison in this file is a difference. The bare
    number is not small next to the sextupole term — it is about a quarter of it — so a
    comparison that forgot the subtraction would look like a coefficient error of the
    right size to be believable.
    """
    bare = _ptc_anharmonicity(_madx_ring(0.0, 0.0))
    assert abs(bare[0, 0]) > 0.05
    assert np.array_equal(sextupole_detuning(_accsim_ring(ref, k2_scale=0.0)), np.zeros((2, 2)))
    sext = PTC_TO_DQDJ * (_ptc_anharmonicity(_madx_ring(1.0)) - bare)
    assert abs(bare[0, 0] / sext[0, 0]) > 0.1  # not a rounding-level contaminant


def test_sextupole_detuning_matches_ptc_on_the_generic_ring(ref: ReferenceParticle) -> None:
    """The headline: all three independent entries, to round-off rather than to a tolerance.

    ``rel=1e-9`` is not a physics tolerance — it is the width of MAD-X's own printed
    precision. Measured 2026-08-27: ratios ``1.0000000000`` on ``dQx/dJx``, ``dQx/dJy``
    and ``dQy/dJy``.
    """
    ours = sextupole_detuning(_accsim_ring(ref))
    theirs = _ptc_sextupole_part()
    assert ours[0, 1] == ours[1, 0]
    for i, j in ((0, 0), (0, 1), (1, 1)):
        assert ours[i, j] == pytest.approx(theirs[i, j], rel=1e-9)


def test_agreement_survives_a_hundredfold_change_of_strength(ref: ReferenceParticle) -> None:
    """Quadratic in ``k2`` on both sides, and still exact at each strength.

    The pre-committed gate was that the residual should fall as ``k2^2``. It does not
    fall at all, because there is nothing to fall: both codes compute the same quartic
    coefficient. What the scan does establish is that the agreement is not an accident of
    one strength, and that both sides really are second order — a tenfold change in
    ``k2`` moves the answer by a hundred.
    """
    prev = None
    for scale in (2.0, 0.5, 0.2):
        ours = sextupole_detuning(_accsim_ring(ref, k2_scale=scale))
        theirs = _ptc_sextupole_part(k2_scale=scale)
        assert ours[0, 0] == pytest.approx(theirs[0, 0], rel=1e-8)
        if prev is not None:
            assert ours[0, 0] / prev[1] == pytest.approx((scale / prev[0]) ** 2, rel=1e-9)
        prev = (scale, ours[0, 0])


def test_ptc_answer_does_not_move_between_normal_form_orders(ref: ReferenceParticle) -> None:
    """``no = 4, 5, 6`` give the same number — which is *why* the comparison is exact.

    ``anhx(1,0,0)`` is the coefficient of the term linear in the action, i.e. the quartic
    part of the normal form. Raising ``no`` adds higher powers of the action
    (``anhx(2,0,0)`` and friends); it does not correct this one. Recording that here is
    what turns "the two agree suspiciously well" into a statement about what is being
    compared.
    """
    vals = [_ptc_sextupole_part()[0, 0]]
    for order in (5, 6):
        on = _ptc_anharmonicity(_madx_ring(1.0), order=order)
        off = _ptc_anharmonicity(_madx_ring(0.0), order=order)
        vals.append(PTC_TO_DQDJ * (on - off)[0, 0])
    assert vals[1] == pytest.approx(vals[0], rel=1e-12)
    assert vals[2] == pytest.approx(vals[0], rel=1e-12)


#: A second working point, reached by different quadrupoles, so nothing about the first
#: one's phase pattern is load-bearing.  Qx = 0.19782, Qy = 0.31849.
KF2, KD2 = 0.60, -0.65

#: A third, with a much larger vertical tune again.  Qx = 0.27155, Qy = 0.57897.
KF4, KD4 = 0.95, -1.15

#: And one deliberately close to the third integer, where ``C(3,0)`` stops being a small
#: correction and becomes the whole answer.  Qx = 0.329334, i.e. 4.0e-3 below ``1/3``.
#: Not closer than that on purpose: the answer goes as ``1/sin(3 pi Q_x)``, so the
#: comparison inherits ``dQ / |Q_x - 1/3|`` from however well the two codes agree on the
#: *tune*, and at 1e-9 (the control above) 4e-3 leaves about 2.5e-7 of headroom.
KF3, KD3 = 0.690, -0.645


@pytest.mark.parametrize(("kf", "kd"), [(KF2, KD2), (KF4, KD4)])
def test_agreement_holds_at_other_working_points(
    ref: ReferenceParticle, kf: float, kd: float
) -> None:
    """Two more phase patterns, so nothing about the first ring's layout is load-bearing."""
    ours = sextupole_detuning(_accsim_ring(ref, kf=kf, kd=kd))
    theirs = _ptc_sextupole_part(kf=kf, kd=kd)
    for i, j in ((0, 0), (0, 1), (1, 1)):
        assert ours[i, j] == pytest.approx(theirs[i, j], rel=1e-8)


def test_near_the_third_integer_the_answer_diverges_and_ptc_follows_it(
    ref: ReferenceParticle,
) -> None:
    """The one denominator the analytic anchors cannot reach, gated where it dominates.

    A quadrupole drives no ``3 Q_x`` line, so the exact-trace anchor in
    ``tests/analytic/`` says nothing about the ``C(3,0)`` term; at a generic tune that
    term is a small correction and a wrong coefficient there costs a few percent. Next to
    ``Q_x = 1/3`` it is the entire answer, and it is enormous. PTC agreeing *there* is
    what pins it.

    The tolerance is set by the *tune*, not by the formula: ``1/sin(3 pi Q_x)`` amplifies
    any tune disagreement by ``1/|Q_x - 1/3|``, so the ``1e-9`` the two codes share on
    ``Q_x`` becomes about ``2.5e-7`` here. Going closer to the resonance would make the
    divergence more dramatic and the comparison less meaningful, which is why the working
    point is fixed at ``4e-3`` rather than pushed until the number looks impressive.
    """
    generic = abs(sextupole_detuning(_accsim_ring(ref))[0, 0])
    ours = sextupole_detuning(_accsim_ring(ref, kf=KF3, kd=KD3))
    qx, _ = tunes(_accsim_ring(ref, kf=KF3, kd=KD3))
    assert abs(qx - 1.0 / 3.0) == pytest.approx(4.0e-3, rel=0.05), qx
    assert abs(ours[0, 0]) > 20.0 * generic, "the third-integer term should dominate here"
    assert abs(math.sin(3 * math.pi * qx)) < 0.04
    theirs = _ptc_sextupole_part(kf=KF3, kd=KD3)
    assert ours[0, 0] == pytest.approx(theirs[0, 0], rel=1e-5)


def test_sextupoles_and_octupoles_together_are_predicted_not_fitted(
    ref: ReferenceParticle,
) -> None:
    """``total_detuning`` is two independently derived formulas added with nothing between.

    accsim reports the octupole's first-order term plus the sextupoles' second-order one;
    PTC reports the true anharmonicity of the ring that has both, including whatever
    cross term exists between them. The sum is asserted **unadjusted** — the shape I4 used
    on B2's two mechanisms — so any interaction the two miss shows up here as a
    disagreement rather than being budgeted for in advance. It does not: the two agree to
    nine digits, so at this order there is no measurable sextupole-octupole cross term.

    ``k3l`` is chosen so the two contributions are **comparable**, which is the whole
    point. The first version of this test used a strength where the octupole term was
    ``10^4`` times the sextupole one; the sum then agrees with PTC whatever the sextupole
    formula says, and the gate tests nothing. Here the two are within a factor of four in
    ``dQx/dJx`` and they very nearly **cancel** in the cross term (``-2.617 + 2.346``),
    leaving a sum a tenth the size of either piece — so that entry amplifies an error in
    either formula rather than hiding it.
    """
    k3l = 2.0
    lat = _accsim_ring(ref, k3l=k3l)
    octu, sext = amplitude_detuning(lat), sextupole_detuning(lat)
    assert 0.2 < abs(octu[0, 0] / sext[0, 0]) < 5.0, "the two terms must be comparable"
    assert abs(octu[0, 1] + sext[0, 1]) < 0.2 * abs(octu[0, 1]), "the cross term cancels"
    ours = total_detuning(lat)
    theirs = _ptc_sextupole_part(k3l=k3l)
    for i, j in ((0, 0), (0, 1), (1, 1)):
        assert ours[i, j] == pytest.approx(theirs[i, j], rel=1e-8)
