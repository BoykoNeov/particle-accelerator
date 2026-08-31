r"""O5 against ``xtrack``'s ``rdt_first_order_perturbation``.

**What this file is and is not.** xtrack's routine is an analytic first-order perturbation
sum built from a twiss table and a strengths table -- the same expression accsim
evaluates, written by different people in a different code. So this is a strong check on
conventions, indices, phases and factors of two, and a **weak** one on the physics: a
first-order formula that both codes got wrong the same way would agree here. The legs
that do not share the algebra are the shipped ``amplitude_detuning`` tie and the tracked
sidebands, both in ``tests/analytic/test_octupole_driving_terms.py``.

It is worth having anyway for one reason O4 established: xtrack's routine is written
generically over multipole order, so it reaches the quartic terms and the skew-cubic ones
with no new code on its side -- and MAD-X PTC, the other reference, turns out to expose
**no** odd-vertical-charge terms at all (see ``test_octupole_driving_terms_madx.py``). For
the five skew-sextupole terms this file is therefore the only reference leg there is.

**Why the tolerances here are ``1e-6`` where O4's were ``1e-8``.** Not a concession: the
limit was localised and is on xtrack's side, and it is gated in its own test below.
xtrack's ``twiss`` obtains the one-turn map by finite differences, so a *nonlinear* magnet
leaks into what it reports as the **linear** tune -- by ``8e-10`` here, exactly
proportional to ``k3l`` and exactly zero with the octupoles removed. An RDT divides by
``exp(-2 pi i (m_x Q_x + m_y Q_y)) - 1``, which amplifies a tune error roughly in
proportion to the charge, so the quartic terms inherit it worst: the observed deviation on
``f4000`` (charge ``4``) is ``8.1e-8`` against ``8.9e-8`` predicted from the tune gap
alone. O4's cubic fixture has no such leak -- a sextupole is nonlinear too, but its own
fixture's tune gap is ``5e-12`` -- so this is a fact about *quartic* comparisons against a
finite-difference optics code, not about accsim.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    ThinSkewSextupole,
    resonance_driving_terms,
    tunes,
)
from accsim.twiss import _RDT_TERMS, _rdt_sites

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0, GAMMA0 = 938.27208816e6, 20.0
KF, KD = 0.80, -0.90

#: Three octupoles and three skew sextupoles, all in drift space, at different beta and
#: different phase, with a sign change among the weights.
OCTS = {5.5: 400.0, 8.4: -280.0, 11.6: 180.0}
SKEWSEXTS = {2.4: 3.0, 6.8: -2.1, 9.9: 4.5}

OCT_KEYS = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "oct")
SKEWSEXT_KEYS = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "skewsext")
KEYS = OCT_KEYS + SKEWSEXT_KEYS


def _rings(
    octs: dict[float, float] | None = None,
    skewsexts: dict[float, float] | None = None,
    kf: float = KF,
    kd: float = KD,
    build: bool = True,
):
    """The identical ring in both codes, element for element.

    ``ThinOctupole(k3l) == xt.Multipole(knl=[0,0,0,k3l])`` and
    ``ThinSkewSextupole(k2sl) == xt.Multipole(ksl=[0,0,k2sl])`` are the two relations this
    whole file rests on, and both are re-probed below rather than taken on trust.

    ``build=False`` skips the ``xt.Line`` for callers that only need accsim's half: every
    line build JIT-compiles a fresh C kernel at around twelve seconds and one leaked
    ``.pyd`` apiece (``docs/CONVENTIONS.md`` -> *Test-suite cost*).
    """
    marks: dict[float, tuple[str, float]] = {}
    marks.update({p: ("o", w) for p, w in (octs or {}).items()})
    marks.update({p: ("t", w) for p, w in (skewsexts or {}).items()})
    els: list = []
    xels: list = []
    names: list[str] = []
    s = 0.0
    for _ in range(4):
        for k in (kf, kd):
            els.append(Quadrupole(0.5, k))
            xels.append(xt.Quadrupole(length=0.5, k1=k))
            names.append(f"q{len(names)}")
            s += 0.5
            done = 0.0
            for p in sorted(q for q in marks if s < q < s + 1.0):
                kind, w = marks[p]
                els.append(Drift(p - s - done))
                xels.append(xt.Drift(length=p - s - done))
                names.append(f"d{len(names)}")
                if kind == "o":
                    els.append(ThinOctupole(w))
                    xels.append(xt.Multipole(knl=[0.0, 0.0, 0.0, w], length=0.0))
                else:
                    els.append(ThinSkewSextupole(w))
                    xels.append(xt.Multipole(ksl=[0.0, 0.0, w], length=0.0))
                names.append(f"m{len(names)}")
                done = p - s
            els.append(Drift(1.0 - done))
            xels.append(xt.Drift(length=1.0 - done))
            names.append(f"d{len(names)}")
            s += 1.0
    placed = sum(isinstance(e, (ThinOctupole, ThinSkewSextupole)) for e in els)
    assert placed == len(marks), "a source landed inside a quadrupole, not in drift space"
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    if not build:
        return Lattice(els, ref), None
    line = xt.Line(elements=xels, element_names=names)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT unavailable: {exc}")
    return Lattice(els, ref), line


def _xtrack_rdts(line, keys=KEYS) -> dict[str, complex]:
    """xtrack's RDTs at the start of the line.

    ``feed_down=False`` explicitly: it defaults to ``True``, and while these rings are
    bend-free and closed on axis -- so it changes nothing here -- accsim does not model
    feed-down at all, and a default that silently did would make the comparison a
    different one from the one being claimed.
    """
    tw = line.twiss4d()
    out = xt.rdt_first_order_perturbation(
        list(keys), twiss=tw, strengths=line.get_table(attr=True), feed_down=False
    )
    return {k: complex(out[k][0]) for k in keys}


def test_the_two_multipole_relations_are_probed_not_assumed() -> None:
    """Both new sources, tracked through a single kick in each code.

    A sign flip or a factorial slip on either source would move every term it drives by a
    constant and leave every *ratio* between this file's comparisons untouched, so the
    element relation has to be established before the RDTs are compared and not inferred
    from them agreeing.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    state = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 0.0, 0.0])
    for mine, theirs in (
        (ThinOctupole(400.0), xt.Multipole(knl=[0.0, 0.0, 0.0, 400.0], length=0.0)),
        (ThinSkewSextupole(3.0), xt.Multipole(ksl=[0.0, 0.0, 3.0], length=0.0)),
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
        assert float(p.px[0]) == pytest.approx(got[1], rel=1e-12)
        assert float(p.py[0]) == pytest.approx(got[3], rel=1e-12)


def test_the_comparison_is_limited_by_xtracks_finite_difference_optics() -> None:
    r"""Why this file's tolerances are ``1e-6``, established rather than assumed.

    The disagreement with xtrack is ``1e-8``-ish rather than round-off, which is too good
    for a convention error and too poor for arithmetic. Localised, in three measurements
    that between them leave one explanation standing:

    - With **no sources at all** the two codes' tunes agree to ``1e-16``. So nothing about
      the quadrupoles, the drifts or the fixture's construction is responsible.
    - With the octupoles in, xtrack's reported **linear** tune moves by ``8e-10`` -- and it
      is *exactly proportional to the octupole strength* (a tenth of the strength gives a
      tenth of the gap). An octupole cannot shift the linear tune: its tune shift is
      proportional to the action, which is zero on the closed orbit. So this is xtrack's
      ``twiss`` finite-differencing its one-turn map through a cubic kick, not physics.
    - An RDT divides by ``exp(-2 pi i (m_x Q_x + m_y Q_y)) - 1``, which turns a tune error
      into a term error roughly in proportion to the charge. Computed from the measured
      tune gap alone, that predicts ``8.9e-8`` on ``f4000``; the observed deviation is
      ``8.1e-8``.

    The consequence for a reader: a driving-term comparison against a finite-difference
    optics code cannot be tighter than the tune agreement times the charge, and the charge
    is what makes this milestone's terms harder to compare than O4's. accsim's own tune
    comes from an analytic one-turn matrix, so it does not have the error being measured
    here.
    """
    import math

    bare_lat, bare_line = _rings()
    assert float(bare_line.twiss4d().qx) == pytest.approx(tunes(bare_lat)[0], abs=1e-14)

    gaps = {}
    for scale in (1.0, 0.1):
        lat, line = _rings({p: scale * w for p, w in OCTS.items()})
        gaps[scale] = float(line.twiss4d().qx) - tunes(lat)[0]
    assert abs(gaps[1.0]) > 1e-10  # the leak is real and resolved
    assert gaps[0.1] == pytest.approx(0.1 * gaps[1.0], rel=1e-3)  # and linear in k3l

    lat, line = _rings(OCTS)
    qx_x, qy_x = float(line.twiss4d().qx), float(line.twiss4d().qy)
    qx_a, qy_a = tunes(lat)
    got, want = resonance_driving_terms(lat), _xtrack_rdts(line, OCT_KEYS)
    mx, my = _RDT_TERMS["f4000"][0], _RDT_TERMS["f4000"][1]
    den = np.exp(-2j * math.pi * (mx * qx_a + my * qy_a)) - 1.0
    shifted = np.exp(-2j * math.pi * (mx * qx_x + my * qy_x)) - 1.0
    predicted = abs(shifted - den) / abs(den)
    observed = abs(got["f4000"] - want["f4000"]) / abs(want["f4000"])
    assert observed == pytest.approx(predicted, rel=0.3), (observed, predicted)


def test_octupole_terms_agree_to_round_off_including_the_phase() -> None:
    """All eight, real and imaginary part, on a ring of three octupoles.

    An octupole introduces no linear coupling and no linear focusing, so unlike O4's skew
    quadrupole there is no "which optics is the formula evaluated on" question here: both
    codes use the same unperturbed twiss and the agreement is round-off rather than a
    convergent approximation.
    """
    lat, line = _rings(OCTS)
    got, want = resonance_driving_terms(lat), _xtrack_rdts(line, OCT_KEYS)
    for key in OCT_KEYS:
        assert got[key] == pytest.approx(want[key], rel=1e-6), (key, got[key], want[key])
        assert abs(got[key]) > 1.0  # and the terms are not all near zero


def test_skew_sextupole_terms_agree_to_round_off_including_the_phase() -> None:
    """All five -- and this is the only reference leg they have.

    MAD-X PTC returns no rows for odd-vertical-charge terms whatever the ring contains
    (measured in ``test_octupole_driving_terms_madx.py``), so where the octupole half of
    the milestone has two external codes, these five have one. A skew sextupole's linear
    map is the identity, so again there is no coupled-optics gap to converge away: the
    agreement is round-off at full strength, not a limit approached as the strength falls.
    """
    lat, line = _rings(None, SKEWSEXTS)
    got, want = resonance_driving_terms(lat), _xtrack_rdts(line, SKEWSEXT_KEYS)
    for key in SKEWSEXT_KEYS:
        assert got[key] == pytest.approx(want[key], rel=1e-6), (key, got[key], want[key])
        assert abs(got[key]) > 1e-3


def test_the_strength_scaling_agrees_as_well_as_the_value() -> None:
    """Both codes are first order in the source, and they are first order the same way.

    A shared constant error would survive every value comparison in this file. It would
    not survive a strength scan only if the two codes disagreed about the *order*, which
    is what this checks: tripling the octupoles triples both answers, so neither code is
    quietly carrying a second-order piece the other lacks.
    """
    for scale in (0.3, 3.0):
        lat, line = _rings({p: scale * w for p, w in OCTS.items()})
        base_lat, _ = _rings(OCTS, build=False)
        got, want = resonance_driving_terms(lat), _xtrack_rdts(line, OCT_KEYS)
        base = resonance_driving_terms(base_lat)
        for key in OCT_KEYS:
            assert got[key] == pytest.approx(want[key], rel=1e-6), (scale, key)
            assert got[key] == pytest.approx(scale * base[key], rel=1e-12), (scale, key)


def test_both_magnets_at_once_needs_no_cross_term() -> None:
    """A ring carrying both kinds equals the two rings carrying one kind each.

    The disjointness the shipped table asserts, checked against a code that computes each
    term from the full strengths table rather than from a per-kind list. If accsim were
    routing a source into the wrong block, this is where the two codes would part company
    while every single-kind comparison above still passed.
    """
    lat, line = _rings(OCTS, SKEWSEXTS)
    got, want = resonance_driving_terms(lat), _xtrack_rdts(line)
    for key in KEYS:
        assert got[key] == pytest.approx(want[key], rel=1e-6), key
    oct_alone, _ = _rings(OCTS, build=False)
    ss_alone, _ = _rings(None, SKEWSEXTS, build=False)
    a, b = resonance_driving_terms(oct_alone), resonance_driving_terms(ss_alone)
    for key in OCT_KEYS:
        assert got[key] == pytest.approx(a[key], rel=1e-12), key
    for key in SKEWSEXT_KEYS:
        assert got[key] == pytest.approx(b[key], rel=1e-12), key


def test_the_reference_point_moves_the_same_way_in_both_codes() -> None:
    r"""xtrack reports every term along the ring; accsim reports it at the entrance.

    O4's covariance gate, re-run against the external code on the quartic charges. Rolling
    accsim's element list to start at element ``i`` must reproduce xtrack's row ``i``,
    which is a far stronger statement than agreeing at one point: it checks the phase
    advance, the jump at each source and the charge of every term simultaneously, at
    twenty-two points around the ring.

    The one place the two codes differ is **at** a thin source, and it is a bookkeeping
    difference rather than a disagreement: xtrack's row for a source element is
    *downstream* of its kick, and rolling accsim's list so the source comes first observes
    *upstream*. Those points are compared against the predicted step instead.
    """
    lat, line = _rings(OCTS, SKEWSEXTS)
    tw = line.twiss4d()
    out = xt.rdt_first_order_perturbation(
        list(KEYS), twiss=tw, strengths=line.get_table(attr=True), feed_down=False
    )
    els = lat.elements
    compared = 0
    for i, elem in enumerate(els):
        if isinstance(elem, (ThinOctupole, ThinSkewSextupole)):
            continue  # the two codes sit on opposite sides of the jump here
        rolled = Lattice(els[i:] + els[:i], lat.ref)
        mine = resonance_driving_terms(rolled)
        for key in KEYS:
            assert mine[key] == pytest.approx(complex(out[key][i]), rel=1e-5, abs=1e-5), (i, key)
        compared += 1
    assert compared >= 15  # the gate is the whole ring, not one lucky point


def test_at_a_source_the_two_codes_report_opposite_sides_of_the_jump() -> None:
    r"""The exception above, as a *predicted* step rather than a tolerated mismatch.

    Crossing one source at zero phase advance adds exactly that source's plain, undivided
    coefficient. So the difference between xtrack's row at a source and accsim's rolled
    value there is not merely "different" -- it is a number this milestone can compute,
    and checking it is what turns O4's caveat into a gate on the new kinds.
    """
    lat, line = _rings(OCTS, SKEWSEXTS)
    tw = line.twiss4d()
    out = xt.rdt_first_order_perturbation(
        list(KEYS), twiss=tw, strengths=line.get_table(attr=True), feed_down=False
    )
    els = lat.elements
    checked = 0
    for i, elem in enumerate(els):
        if not isinstance(elem, (ThinOctupole, ThinSkewSextupole)):
            continue
        upstream = Lattice(els[i:] + els[:i], lat.ref)
        mine = resonance_driving_terms(upstream)
        sites, _, _ = _rdt_sites(upstream, 32)
        for key in KEYS:
            mx, my, px, py, coef, kind = _RDT_TERMS[key]
            strength, bx, by, mux, muy = sites[kind]
            if strength.size == 0 or mux[0] > 1e-12:
                step = 0.0j
            else:  # the source now sitting at the start, at zero phase advance
                step = coef * strength[0] * bx[0] ** px * by[0] ** py
            assert complex(out[key][i]) == pytest.approx(mine[key] + step, rel=1e-5, abs=1e-5), (
                i,
                key,
            )
        checked += 1
    assert checked == len(OCTS) + len(SKEWSEXTS)
