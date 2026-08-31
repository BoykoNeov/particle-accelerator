r"""O6 against ``xtrack``'s ``rdt_first_order_perturbation`` with ``feed_down=True``.

O4 and O5 compared the design-orbit sum against this routine with ``feed_down=False``
passed **explicitly**, because it defaults to ``True`` and accsim had no model for it: a
default that silently fed down would have made those comparisons a different one from the
one being claimed. This file is that flag turned on, on both sides.

**Why this arbiter is worth having here in particular.** For O5 xtrack was the only leg
on five of the thirteen terms. Here it is one of two — MAD-X PTC's normal form is built
about the closed orbit and therefore contains feed-down whether or not it is asked to,
which is why feed-down was chosen as the milestone over skew octupoles and second-order
RDTs (each of which has one leg). It is also the *broader* of the two: xtrack's routine
consumes ``shift_x``, ``shift_y`` and ``rot_s_rad`` from the strengths table as well as
the orbit, so it arbitrates the **misalignment** half, which PTC's closed-orbit normal
form does not separate out.

**What it can and cannot settle.** As in O4 and O5, both sides evaluate a first-order
perturbation sum, so this is a strong check on conventions, indices, phases and factors
of two and a weak one on the physics. What makes it more than a re-run of O5 is that the
quantity being compared is *created* by the thing under test: with the orbit on axis every
cubic term on this fixture is exactly zero in both codes, so any agreement below is
agreement about feed-down and nothing else.

**Tolerance: ``1e-6``, O5's, unchanged — but it took two findings to keep it, and both
are gated below rather than described here.**

*First, the two codes had to be made into one machine.* On a **steered** orbit accsim and
xtrack disagree about the drift: accsim's is exact, xtrack's is the expanded (paraxial)
one by default. That is the same split axis M localised for ``Q''``, and here it shows up
as an ``8.7e-8`` tune difference on a ring containing **no nonlinear magnet at all** —
which would have been charged to feed-down had it not been isolated. Setting
``XTRACK_USE_EXACT_DRIFTS`` collapses it to ``1.5e-12``, and only then does a comparison
of driving terms mean anything. Matching the model, not widening a tolerance, is the fix.

*Second, the residual that remains is the arbiter's and it is second order.* With the
models matched, the leftover disagreement grows as the **square** of the octupole
strength at fixed orbit (measured ``x4`` per doubling over a factor of eight in ``k3l``),
while *both* codes evaluate formulas that are **first** order in it. So it is content
neither first-order formula claims — xtrack's ``twiss`` obtains the map by finite
differences, and a source whose effective sextupole strength here is order ``1 m^-2``
leaks into the linear tune it reports far more than O5's on-axis fixture did. The bump
amplitude is therefore chosen so that this sits below ``1e-6``, and the scaling that
justifies the choice is measured in its own test rather than asserted.

**The fixture is matched deliberately, and the probes that chose this milestone are not
evidence for a number.** The two probe rings run on 2026-08-31 were steered differently on
purpose (a MAD-X ``hkicker`` mid-cell against a thin ``Multipole`` at the end), so the
9-18% spread between their raw magnitudes measured the two fixtures and nothing else.
Matching the fixture is part of this file's job, not something the probes established.
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
    ThinOctupole,
    ThinSextupole,
    ThinSkewQuadrupole,
    ThinSkewSextupole,
    closed_twiss,
    resonance_driving_terms,
    resonance_driving_terms_on_orbit,
)
from accsim.orbit import (
    closed_orbit_nonlinear,
    linearised_one_turn_map,
    propagate_orbit_nonlinear,
)
from accsim.twiss import (
    _RDT_TERMS,
    _coupling_norm,
    _rdt_sites_on_orbit,
    closed_twiss_on_orbit,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0, GAMMA0 = 938.27208816e6, 20.0
KF, KD = 0.80, -0.90

#: Octupoles and **no sextupole**, the probe's own fixture: every cubic term here is
#: purely fed down, so the comparison is not a small correction sitting on top of a
#: direct contribution this milestone does not change.
OCTS = {5.5: 400.0, 8.4: -280.0, 11.6: 180.0}

CUBIC = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "sext")
SKEWQ = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "skew")
SKEWSEXT = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "skewsext")
QUARTIC = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "oct")
KEYS = CUBIC + SKEWQ + SKEWSEXT + QUARTIC


def _ring(kick_x: float = 0.0, kick_y: float = 0.0, shift_x: float = 0.0, shift_y: float = 0.0):
    """The same ``Q D Q D`` ring in both codes, steered and/or misaligned identically.

    The steerer sits at the lattice entrance so that "the orbit at the start" is
    unambiguous in both codes, and the octupoles sit in drift space at three different
    ``beta`` and three different phases with a sign change among the weights — O4's
    fixture rule, kept, so no accidental symmetry can hide a wrong term.
    """
    els: list = [Corrector(kick_x=kick_x, kick_y=kick_y)]
    # xtrack: a thin Multipole's knl[0] deflects px by -knl[0], ksl[0] deflects py by
    # +ksl[0]. The signs are *probed* below rather than trusted from this comment.
    xels: list = [xt.Multipole(knl=[-kick_x], ksl=[kick_y], length=0.0)]
    names: list[str] = ["kick"]
    s = 0.0
    for _ in range(4):
        for k in (KF, KD):
            els.append(Quadrupole(0.5, k))
            xels.append(xt.Quadrupole(length=0.5, k1=k))
            names.append(f"q{len(names)}")
            s += 0.5
            done = 0.0
            for p in sorted(q for q in OCTS if s < q < s + 1.0):
                els.append(Drift(p - s - done))
                xels.append(xt.Drift(length=p - s - done))
                names.append(f"d{len(names)}")
                els.append(ThinOctupole(OCTS[p], dx=shift_x, dy=shift_y))
                m = xt.Multipole(knl=[0.0, 0.0, 0.0, OCTS[p]], length=0.0)
                m.shift_x, m.shift_y = shift_x, shift_y
                xels.append(m)
                names.append(f"o{len(names)}")
                done = p - s
            els.append(Drift(1.0 - done))
            xels.append(xt.Drift(length=1.0 - done))
            names.append(f"d{len(names)}")
            s += 1.0
    placed = sum(isinstance(e, ThinOctupole) for e in els)
    assert placed == len(OCTS), "an octupole landed inside a quadrupole rather than in drift space"
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    line = xt.Line(elements=xels, element_names=names)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    # The two codes must describe ONE machine before their optics can be compared, and on
    # a steered orbit the drift is where they part company by default: accsim's is exact,
    # xtrack's is the expanded (paraxial) one unless asked otherwise. That is the same
    # split axis M localised for Q'', arriving here as an 8.7e-8 tune difference on a ring
    # with no nonlinear magnet in it at all -- measured in
    # `test_the_two_codes_are_made_to_describe_one_machine_before_anything_is_compared`.
    line.configure_drift_model("exact")
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT unavailable: {exc}")
    return Lattice(els, ref), line


def _bare_ring(kick_x: float = 0.0, kick_y: float = 0.0, exact_drifts: bool = True):
    """The same ring with the octupoles removed: a machine with no nonlinearity at all.

    The control for every optics comparison here. Any difference between the two codes on
    this ring is a difference of *element models*, because there is nothing to feed down
    and nothing for a finite-difference twiss to trip over.
    """
    saved = dict(OCTS)
    OCTS.clear()
    try:
        lat, line = _ring(kick_x=kick_x, kick_y=kick_y)
    finally:
        OCTS.update(saved)
    if not exact_drifts:
        line.configure_drift_model("expanded")
    return lat, line


def _xtrack_rdts(line, feed_down: bool = True, keys=KEYS) -> dict[str, complex]:
    """xtrack's RDTs at the start of the line, on its own closed orbit.

    ``twiss4d()`` linearises about the closed orbit, so the table it returns already
    carries the on-orbit ``betx``/``mux`` *and* the ``x``/``y`` the feed-down step reads.
    That the optics half arrives without being asked for is exactly the roadmap's point:
    neither reference code announces it is doing it.
    """
    tw = line.twiss4d()
    out = xt.rdt_first_order_perturbation(
        list(keys), twiss=tw, strengths=line.get_table(attr=True), feed_down=feed_down
    )
    return {k: complex(out[k][0]) for k in keys}


# ==========================================================================
# 1. The elements first: a comparison of sums is worthless if a kick disagrees
# ==========================================================================


def test_the_steerer_and_the_displaced_octupole_are_probed_not_assumed() -> None:
    """Both new pieces of fixture, tracked through a single element in each code.

    O5's rule, and it matters more here: the steerer *is* the independent variable of
    every gate in this file, so a sign error in it would steer the two codes' rings in
    opposite directions and show up as a factor no tolerance could absorb — or worse,
    cancel in a term that is even in the orbit. The displaced octupole is probed for the
    same reason: ``shift_x`` is the one convention this file cannot arbitrate from the
    RDTs, because both codes would have to share it for the comparison to mean anything.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    state = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 0.0, 0.0])
    displaced = xt.Multipole(knl=[0.0, 0.0, 0.0, 400.0], length=0.0)
    displaced.shift_x, displaced.shift_y = 1.0e-3, -0.5e-3
    for mine, theirs in (
        (Corrector(kick_x=3.0e-4, kick_y=-2.0e-4), xt.Multipole(knl=[-3.0e-4], ksl=[-2.0e-4])),
        (ThinOctupole(400.0, dx=1.0e-3, dy=-0.5e-3), displaced),
    ):
        got = mine.track(state.copy(), ref)
        p = xt.Particles(
            mass0=MASS0, q0=1, gamma0=GAMMA0, x=state[0], px=state[1], y=state[2], py=state[3]
        )
        line = xt.Line(elements=[theirs], element_names=["m"])
        line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
        try:
            line.build_tracker()
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"xtrack JIT unavailable: {exc}")
        line.track(p)
        assert float(p.px[0]) == pytest.approx(got[1], rel=1e-11, abs=1e-16)
        assert float(p.py[0]) == pytest.approx(got[3], rel=1e-11, abs=1e-16)


def test_the_two_rings_settle_on_the_same_closed_orbit() -> None:
    """The independent variable itself, compared before anything computed from it.

    Every gate below is "the same orbit produces the same terms". If the two rings do not
    actually sit on the same orbit, an agreement in the terms would be a coincidence and a
    disagreement would be uninterpretable — which is the failure the probes explicitly
    refused to read a number from.
    """
    lat, line = _ring(kick_x=2.0e-4, kick_y=1.0e-4)
    tw = line.twiss4d()
    mine = propagate_orbit_nonlinear(lat, closed_orbit_nonlinear(lat))[:-1]
    assert float(tw.x[0]) == pytest.approx(mine[0][0], rel=1e-9, abs=1e-14)
    assert float(tw.y[0]) == pytest.approx(mine[0][2], rel=1e-9, abs=1e-14)
    assert max(abs(o[0]) for o in mine) == pytest.approx(float(np.max(np.abs(tw.x))), rel=1e-9)
    assert max(abs(o[2]) for o in mine) == pytest.approx(float(np.max(np.abs(tw.y))), rel=1e-9)


# ==========================================================================
# 2. The arbiter agrees these terms exist only because of feed-down
# ==========================================================================


def test_both_codes_put_the_cubic_terms_at_exactly_zero_on_the_design_orbit() -> None:
    """The reference point, in both codes at once.

    An octupole reaches the sextupole lines only by feeding down, so on axis these five
    terms are identically zero — not small. That is what makes every number below a
    measurement of feed-down rather than of a correction to something else.
    """
    lat, line = _ring()
    theirs = _xtrack_rdts(line, feed_down=True)
    mine = resonance_driving_terms_on_orbit(lat)
    for key in CUBIC + SKEWQ + SKEWSEXT:
        assert mine[key] == 0.0
        assert abs(theirs[key]) < 1e-12


def test_turning_the_arbiters_own_flag_off_removes_exactly_what_o6_adds() -> None:
    """``feed_down=False`` on a *bumped* ring returns zero on the five cubic terms.

    The cleanest statement of what this milestone is: the quantity being compared is
    created entirely by the flag, on xtrack's side as well as accsim's. It also rules out
    the reading in which the on-orbit optics alone (which xtrack's ``twiss4d`` supplies
    either way) could produce these terms — they cannot, because with the flag off the
    strengths are the design ones and an octupole drives no cubic line.
    """
    _, line = _ring(kick_x=2.0e-4, kick_y=1.0e-4)
    off = _xtrack_rdts(line, feed_down=False)
    on = _xtrack_rdts(line, feed_down=True)
    for key in CUBIC:
        assert abs(off[key]) < 1e-12
        assert abs(on[key]) > 1e-3
    for key in QUARTIC:
        assert abs(off[key]) > 1e-3, "an octupole drives the quartic terms directly"


# ==========================================================================
# 3. The comparison itself
# ==========================================================================


@pytest.mark.parametrize(
    ("kick_x", "kick_y"),
    [(2.0e-4, 0.0), (0.0, 2.0e-4), (2.0e-4, 1.0e-4)],
)
def test_a_steered_ring_agrees_with_xtrack_on_all_twenty_terms(
    kick_x: float, kick_y: float
) -> None:
    """Three orbits, because one would not separate the two halves of the expansion.

    A horizontal orbit reaches the normal-sextupole lines, a vertical one the
    skew-sextupole lines, and only both together the skew-quadrupole lines — so a model
    that fed down correctly in ``x`` and wrongly in ``y`` would pass the first case
    outright. The quartic terms are carried along in every case as a control: they are
    driven directly, so they must agree here *and* barely move with the orbit.
    """
    lat, line = _ring(kick_x=kick_x, kick_y=kick_y)
    mine = resonance_driving_terms_on_orbit(lat)
    theirs = _xtrack_rdts(line)
    checked = 0
    for key in KEYS:
        if abs(theirs[key]) < 1e-9:
            assert abs(mine[key]) < 1e-9, f"{key}: accsim drives a line xtrack does not"
            continue
        assert mine[key] == pytest.approx(theirs[key], rel=1e-6), key
        checked += 1
    assert checked >= 13, f"only {checked} terms were live on this orbit"


@pytest.mark.parametrize(("shift_x", "shift_y"), [(1.0e-3, 0.0), (0.0, 1.0e-3)])
def test_a_displaced_octupole_on_a_flat_orbit_agrees_too(shift_x: float, shift_y: float) -> None:
    """The misalignment half, which only this arbiter has — one plane at a time.

    PTC's normal form is built about the closed orbit and so contains the *orbit* half
    whether asked or not, but it does not separate a magnet that has moved from a beam
    that has: only xtrack's routine consumes ``shift_x``/``shift_y`` explicitly. This is
    therefore the one external check on ``z_0 = z_co - d``, and it is why the internal
    displaced-magnet identity in the analytic file exists as well — a sign shared by both
    codes could not be caught here.

    **One plane at a time on purpose.** A displacement in *both* planes makes a genuine
    skew quadrupole (``k1sl_eff = k3l dx dy``) and therefore a genuinely coupled ring,
    which this sum's decoupling premise then approximates. That case is not avoided — it
    is measured on its own below — but it does not belong inside a check of the
    displacement convention, where it would masquerade as a convention error.
    """
    lat, line = _ring(shift_x=shift_x, shift_y=shift_y)
    mine = resonance_driving_terms_on_orbit(lat)
    theirs = _xtrack_rdts(line)
    live = 0
    for key in CUBIC + SKEWSEXT:
        if abs(theirs[key]) < 1e-9:
            continue
        assert mine[key] == pytest.approx(theirs[key], rel=1e-6), key
        live += 1
    assert live >= 4, f"only {live} fed-down terms were live under this displacement"


def test_the_decoupling_premise_is_what_a_two_plane_displacement_costs() -> None:
    r"""The approximation, priced against an external code rather than against itself.

    Displaced in **both** planes, an octupole feeds down to a real skew quadrupole
    (``k1sl_eff = k3l dx dy``), so the ring is genuinely coupled — while first-order
    perturbation theory asks for the *unperturbed* optics and this walk therefore
    decouples every element map. The analytic file prices that internally, by showing the
    coupling is second order in the orbit where the terms it buys are first. Here the same
    claim is checked from the outside, against a code that does not make the
    approximation at all.

    The measured statement is the ordering, not a magnitude: the disagreement scales as
    the **square** of the ring's own coupling — fitted, over a displacement range in which
    the coupling itself varies by an order of magnitude. So what the premise costs
    vanishes faster than the physics it buys, and a *first*-power scaling here would
    instead mean the decoupling was corrupting the terms directly.
    """
    couplings, worsts = [], []
    for d in (0.6e-3, 0.9e-3, 1.35e-3):
        lat, line = _ring(shift_x=d, shift_y=-0.6 * d)
        mine, theirs = resonance_driving_terms_on_orbit(lat), _xtrack_rdts(line)
        couplings.append(_coupling_norm(linearised_one_turn_map(lat)))
        worsts.append(
            max(abs(mine[k] - theirs[k]) / abs(theirs[k]) for k in KEYS if abs(theirs[k]) > 1e-9)
        )
    assert couplings[-1] / couplings[0] > 4.0, "the scan does not move the coupling enough"
    slope = float(np.polyfit(np.log(couplings), np.log(worsts), 1)[0])
    assert slope == pytest.approx(2.0, abs=0.2), (
        f"the disagreement scales as coupling^{slope:.3g}; at first power the decoupling "
        "would be corrupting the driving terms rather than costing a second-order residual"
    )


# 4. The headline, gated against the arbiter rather than against itself
# ==========================================================================


def _strengths_only(lat: Lattice) -> dict[str, complex]:
    """The rival model: feed the strengths down, but keep the **design** optics.

    Built by handing the design-orbit function an equivalent lattice in which every
    octupole is joined by the thin sources it feeds down into at its own orbit offset.
    That is the obvious reading of the milestone, and the point of this construction is
    that it is *not enough* — a thin multipole does not change the linear optics, so this
    lattice's ``beta`` and phases are the blueprint's, not the steered machine's.
    """
    orbit = propagate_orbit_nonlinear(lat, closed_orbit_nonlinear(lat))[:-1]
    els: list = []
    for elem, o in zip(lat.elements, orbit, strict=True):
        els.append(elem)
        if isinstance(elem, ThinOctupole):
            x0, y0 = float(o[0]), float(o[2])
            els.append(ThinSextupole(elem.k3l * x0))
            els.append(ThinSkewSextupole(elem.k3l * y0))
            els.append(ThinSkewQuadrupole(elem.k3l * x0 * y0))
    return resonance_driving_terms(Lattice(els, lat.ref))


def test_the_optics_half_is_needed_to_match_the_arbiter_at_all() -> None:
    r"""The milestone's headline, decided by the reference code rather than by assertion.

    Recomputing each source's effective strength at its orbit offset is the leading
    effect and creates these terms from nothing. It is **not the whole of it**: feed-down
    from a quartic source reaches the quadrupole order too (``k1l_eff = k3l x_0^2 / 2``),
    so the optics the first-order formula is evaluated on move as well. Here that rival
    model — right strengths, blueprint optics — is built explicitly and *fails* against
    xtrack by far more than the tolerance the shipped one passes at, on a fixture where
    both are otherwise identical. So the correction is not a refinement that could be
    deferred; without it the milestone cannot match either arbiter.
    """
    lat, line = _ring(kick_x=2.0e-4, kick_y=1.0e-4)
    theirs = _xtrack_rdts(line)
    shipped = resonance_driving_terms_on_orbit(lat)
    rival = _strengths_only(lat)
    worst_shipped = worst_rival = 0.0
    for key in CUBIC:
        scale = abs(theirs[key])
        assert scale > 1e-3
        worst_shipped = max(worst_shipped, abs(shipped[key] - theirs[key]) / scale)
        worst_rival = max(worst_rival, abs(rival[key] - theirs[key]) / scale)
    assert worst_shipped < 1e-6
    assert worst_rival > 100 * worst_shipped, (
        f"the design-optics rival misses by {worst_rival:.3g} against the shipped "
        f"{worst_shipped:.3g}; if these were comparable the headline would be wrong"
    )


def test_the_optics_the_two_codes_walk_are_the_same_moved_optics() -> None:
    """Localises the above: it is the ``beta`` and the tunes that move, and both codes agree.

    Stated separately from the terms because a disagreement in the sum is ambiguous
    between the strengths and the optics, and this milestone changes both. ``xtrack``'s
    ``twiss4d`` linearises about the closed orbit, which is the same thing
    :func:`closed_twiss_on_orbit` does, so the beat itself is checkable directly.
    """
    # Purely horizontal: a vertical orbit through an octupole makes a skew quadrupole, and
    # closed_twiss_on_orbit refuses a coupled ring outright rather than reporting plane
    # betas for it. The optics half being compared here is horizontal anyway.
    lat, line = _ring(kick_x=3.0e-4)
    tw = line.twiss4d()
    mine_design, mine_orbit = closed_twiss(lat), closed_twiss_on_orbit(lat)
    beat_mine = mine_orbit.beta_x / mine_design.beta_x - 1.0
    assert abs(beat_mine) > 100 * 1e-6, "the fixture does not exercise the optics half"
    assert float(tw.betx[0]) == pytest.approx(mine_orbit.beta_x, rel=1e-6)
    assert float(tw.betx[0]) != pytest.approx(mine_design.beta_x, rel=1e-6)


# ==========================================================================
# 5. Why the tolerance is what it is, localised rather than chosen
# ==========================================================================


def test_the_two_codes_are_made_to_describe_one_machine_before_anything_is_compared() -> None:
    r"""The drift model, which parts the two codes the moment the orbit is not on axis.

    On a ring with **no nonlinear magnet in it at all**, a steered orbit still moves the
    optics — because an exact drift is nonlinear, so its Jacobian at a nonzero orbit
    *angle* is not the paraxial one. accsim's drift is exact; xtrack's is the expanded
    one unless ``XTRACK_USE_EXACT_DRIFTS`` is set. The gap is ``~1e-7`` in the tune, which
    is the same order as everything this milestone computes, and it has **nothing** to do
    with feed-down: there is nothing here to feed down.

    This is axis M's finding arriving in a new place. M2 localised a three-code split in
    ``Q''`` to exactly this — accsim exact, xtrack default and MAD-X paraxial — and the
    lesson carried forward is that the fix is to match the model, never to widen a
    tolerance around it.
    """
    _, expanded = _bare_ring(kick_x=2.0e-4, kick_y=1.0e-4, exact_drifts=False)
    lat, exact = _bare_ring(kick_x=2.0e-4, kick_y=1.0e-4, exact_drifts=True)
    _, qx, qy = _rdt_sites_on_orbit(lat, 32, None, 0.0, 1e-7)
    tw_e, tw_x = expanded.twiss4d(), exact.twiss4d()
    # Same machine on axis either way: the two drift models coincide at zero angle.
    lat0, flat = _bare_ring()
    _, qx0, qy0 = _rdt_sites_on_orbit(lat0, 32, None, 0.0, 1e-7)
    assert qx0 == pytest.approx(float(flat.twiss4d().qx), abs=1e-12)
    # Steered, the expanded drift is a different machine...
    assert abs(qx - float(tw_e.qx)) > 1e-8
    # ...and the exact one is the same machine, to four orders better.
    assert qx == pytest.approx(float(tw_x.qx), abs=1e-11)
    assert qy == pytest.approx(float(tw_x.qy), abs=1e-11)


def test_the_remaining_gap_is_second_order_in_the_source_strength() -> None:
    r"""Which side owns the residual, decided by a scaling rather than by assertion.

    Both codes evaluate a driving-term formula that is **first** order in the multipole
    strength — O4 established that first order in ``k2`` is exact for these terms, and O5
    re-ran it. A disagreement that is *first* order in the strength would therefore be a
    genuine error in one of them. What is measured instead is a clean **second** order:
    quadrupling when the octupoles double. That is outside what either formula claims, so
    it is numerical, and the mechanism is O5's — xtrack's ``twiss`` obtains the one-turn
    map by finite differences, so a nonlinear magnet leaks into the linear tune it
    reports, and an RDT divides by a resonance denominator that converts a tune error
    into a term error roughly in proportion to the charge.

    It is worse here than in O5 for a reason that is itself the milestone: a displaced
    octupole *is* a strong sextupole (``k2l_eff = k3l x_co``), so the fixture that makes
    feed-down visible is also the fixture that makes the leak visible.
    """
    saved = dict(OCTS)
    worst = []
    try:
        for scale in (0.25, 0.5, 1.0):
            OCTS.clear()
            OCTS.update({k: v * scale for k, v in saved.items()})
            lat, line = _ring(kick_x=2.0e-4, kick_y=1.0e-4)
            theirs = _xtrack_rdts(line)
            mine = resonance_driving_terms_on_orbit(lat)
            worst.append(
                max(
                    abs(mine[k] - theirs[k]) / abs(theirs[k]) for k in KEYS if abs(theirs[k]) > 1e-9
                )
            )
    finally:
        OCTS.clear()
        OCTS.update(saved)
    for small, large in zip(worst, worst[1:], strict=False):
        assert large / small == pytest.approx(4.0, rel=0.35), (
            f"the residual scales as {worst}, which is not second order in k3l; a FIRST "
            "order scaling would mean one of the two formulas is actually wrong"
        )
    assert worst[-1] < 1e-6
