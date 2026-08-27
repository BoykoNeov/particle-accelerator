r"""O4 against xtrack: a second implementation of the same first-order formula.

**What this file is and is not.** ``xt.rdt_first_order_perturbation`` is analytic --
first-order perturbation theory from a twiss table plus a strengths table -- so it is
the *same family of expression* accsim derives, computed by different people in
different code. That makes it a strong **convention arbiter** and a weak independence
claim, and this file says so rather than letting the milestone quietly bank it as a
cross-check. The independent legs live in ``tests/analytic``: the shipped ``|C^-|`` of
G1 (a different derivation entirely, from the exact eigen-tune split) and **tracking**.

The roadmap's O4 candidate predicted exactly this ("closer to a reimplementation than a
cross-check"), and the prediction survives: the agreement below is to round-off, not
asymptotic, on a sextupole-only ring.

**What it does establish, which nothing else can.**

1. **The basis, hence the sign of every phase.** A resonance driving term is fixed only
   once you say whether ``h_u`` is ``u_hat + i p_hat_u`` or ``u_hat - i p_hat_u``; the
   two differ by complex conjugation, agree in modulus, and are therefore invisible to
   any magnitude comparison. accsim ships the first, which is what xtrack and MAD-X use
   -- **measured here**, and with the conjugate asserted to be decisively wrong rather
   than merely worse. (Tracking pins the same thing from physics, in
   ``tests/analytic/test_resonance_driving_terms.py``; this says the label matches the
   rest of the field.)
2. **That xtrack's ``fjklm`` is the *normalised* term**, divided by its resonance
   denominator, and not the bare sum. Determined by probe, not by reading the name: the
   ring is walked toward ``Q_x = 1/3`` and only the normalised reading diverges. xtrack
   also exposes the undivided one as ``fjklm_open``, and that column is checked to stay
   put across the same scan.
3. **That the coupled case converges, and at what order.** With skew quadrupoles in the
   ring the two codes stop agreeing exactly -- accsim evaluates first-order theory on
   the *unperturbed* optics, xtrack on the twiss it is handed, which is the coupled one.
   The gap is measured and falls as the **cube** of the skew strength.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinSextupole,
    ThinSkewQuadrupole,
    resonance_driving_terms,
    tunes,
)
from accsim.twiss import _RDT_TERMS, _rdt_sites

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0, GAMMA0 = 938.27208816e6, 20.0
KF, KD = 0.80, -0.90

#: Three sextupoles, three skew quadrupoles, all in drift space, at different beta and
#: different phase, with a sign change among the weights.
SEXTS = {5.5: 1.0, 8.4: -0.7, 11.6: 0.45}
SKEWS = {2.4: 0.02, 6.8: -0.014, 9.9: 0.031}

SEXT_KEYS = ("f3000", "f2100", "f1020", "f1011", "f1002")
SKEW_KEYS = ("f1001", "f1010")
KEYS = SEXT_KEYS + SKEW_KEYS


def _accsim_ring(
    sexts: dict[float, float] | None = None,
    skews: dict[float, float] | None = None,
    kf: float = KF,
    kd: float = KD,
) -> Lattice:
    """Just accsim's half of the fixture -- for the tune search, which needs no line.

    Kept separate on purpose: every ``xt.Line`` build JIT-compiles a fresh C kernel at
    around twelve seconds and one leaked ``.pyd`` apiece (see ``docs/CONVENTIONS.md`` ->
    *Test-suite cost*), so a root-find that built one per iteration would cost minutes
    for nothing.
    """
    return _rings(sexts, skews, kf, kd, build=False)[0]


def _rings(
    sexts: dict[float, float] | None = None,
    skews: dict[float, float] | None = None,
    kf: float = KF,
    kd: float = KD,
    build: bool = True,
):
    """The identical ring in both codes, element for element.

    ``ThinSextupole(k2l) == xt.Multipole(knl=[0, 0, k2l])`` and
    ``ThinSkewQuadrupole(k1sl) == xt.Multipole(ksl=[0, k1sl])`` are the relations pinned
    by ``test_sextupole_kick_xtrack`` and ``test_normal_form_along_ring_xtrack``; the
    first is re-probed below rather than taken on trust.
    """
    marks: dict[float, tuple[str, float]] = {}
    marks.update({p: ("s", w) for p, w in (sexts or {}).items()})
    marks.update({p: ("k", w) for p, w in (skews or {}).items()})
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
                if kind == "s":
                    els.append(ThinSextupole(w))
                    xels.append(xt.Multipole(knl=[0.0, 0.0, w], length=0.0))
                else:
                    els.append(ThinSkewQuadrupole(w))
                    xels.append(xt.Multipole(ksl=[0.0, w], length=0.0))
                names.append(f"m{len(names)}")
                done = p - s
            els.append(Drift(1.0 - done))
            xels.append(xt.Drift(length=1.0 - done))
            names.append(f"d{len(names)}")
            s += 1.0
    placed = sum(isinstance(e, (ThinSextupole, ThinSkewQuadrupole)) for e in els)
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


def _xtrack_rdts(line, keys=KEYS, column: str = "") -> dict[str, complex]:
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
    return {k: complex(out[k + column][0]) for k in keys}


def test_the_multipole_relation_is_probed_not_assumed() -> None:
    """``ThinSextupole(k2l) == xt.Multipole(knl=[0,0,k2l])``, and the sign matters here.

    A sign flip on the source would flip every RDT's phase by pi and leave every modulus
    alone -- the same blindness the basis convention has. It is cheap to exclude, so it
    is excluded: both candidates are tracked and the wrong one misses by twice the kick.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    state = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 1.0e-3, 2.0e-4])
    mine = ThinSextupole(1.3).track(state.copy(), ref)
    part = xt.Particles(
        mass0=MASS0,
        q0=1,
        gamma0=GAMMA0,
        **dict(zip(("x", "px", "y", "py", "zeta", "delta"), state, strict=True)),
    )
    xt.Multipole(knl=[0.0, 0.0, 1.3], length=0.0).track(part)
    assert float(part.px[0]) == pytest.approx(mine[1], rel=0.0, abs=1e-18)
    assert float(part.py[0]) == pytest.approx(mine[3], rel=0.0, abs=1e-18)


def test_sextupole_terms_agree_to_round_off_including_the_phase() -> None:
    """All five, complex, on a sextupole-only ring -- and the conjugate is excluded.

    This is the convention statement, and the reason it is worth a test rather than a
    comment: modulus agreement is *guaranteed* under either basis, so a file that
    compared ``abs()`` would pass with accsim shipping the mirror image of what every
    other code calls ``f3000``.
    """
    lat, line = _rings(SEXTS)
    got, want = resonance_driving_terms(lat), _xtrack_rdts(line, SEXT_KEYS)
    for key in SEXT_KEYS:
        # measured 0.9-2.4e-10 relative on the five, which is xtrack's numeric twiss and
        # not a physics gap; gated two orders above that rather than at a round number
        assert got[key] == pytest.approx(want[key], rel=1e-8, abs=0.0), key
        # the mirror basis agrees in modulus and is decisively wrong in phase
        assert abs(got[key]) == pytest.approx(abs(np.conj(want[key])), rel=1e-8)
        assert abs(got[key] - np.conj(want[key])) / abs(want[key]) > 0.1, key


def test_xtrack_reports_the_normalised_term_and_that_is_established_by_probe() -> None:
    """Which of the two objects is behind the name, decided by the resonance denominator.

    ``f3000`` carries the ``3 Q_x`` line. Walk the ring toward ``Q_x = 1/3`` and the
    *normalised* term must blow up while the bare sum over sources barely moves -- a
    fingerprint no naming convention can disguise. Both codes' ``f3000`` diverge together
    and xtrack's ``f3000_open`` does not, which identifies all three columns at once.
    """
    from scipy.optimize import brentq

    def kf_for(qx_target: float) -> float:
        return brentq(lambda kf: tunes(_accsim_ring(kf=kf))[0] - qx_target, 0.76, 1.19, xtol=1e-14)

    far_lat, far_line = _rings(SEXTS, kf=kf_for(0.28))
    near_lat, near_line = _rings(SEXTS, kf=kf_for(1.0 / 3.0 - 1.0e-3))
    grow_mine = abs(resonance_driving_terms(near_lat)["f3000"]) / abs(
        resonance_driving_terms(far_lat)["f3000"]
    )
    grow_theirs = abs(_xtrack_rdts(near_line, ("f3000",))["f3000"]) / abs(
        _xtrack_rdts(far_line, ("f3000",))["f3000"]
    )
    assert grow_mine > 15.0
    assert grow_theirs == pytest.approx(grow_mine, rel=1e-6)
    open_ratio = abs(_xtrack_rdts(near_line, ("f3000",), "_open")["f3000"]) / abs(
        _xtrack_rdts(far_line, ("f3000",), "_open")["f3000"]
    )
    assert open_ratio < 2.0, "the _open column carries no resonance denominator"


def test_skew_terms_agree_and_the_gap_is_cubic_in_the_strength() -> None:
    """Where the two codes genuinely differ, and by how much -- measured, not assumed.

    accsim evaluates first-order theory on the **unperturbed** optics; xtrack evaluates
    it on the twiss table it is given, which for a coupled ring already carries the
    coupling. So this is not a disagreement about the formula, it is a disagreement about
    what to substitute into it, and it must vanish with the coupling.

    The order was measured. ``f1001`` is *linear* in the skew strength, so the obvious
    guess is a first correction at the square. It is not: the relative gap falls by four
    for every halving, i.e. the absolute gap is **cubic** -- because what perturbs the
    optics is itself quadratic in the coupling. The same exponent, from the same cause,
    appears in the tracked gate in ``tests/analytic``.
    """
    rel_prev, ratios = None, []
    for scale in (1.0, 0.5, 0.25, 0.125):
        lat, line = _rings(None, {p: scale * w for p, w in SKEWS.items()})
        got, want = resonance_driving_terms(lat), _xtrack_rdts(line, SKEW_KEYS)
        rel = max(abs(got[k] - want[k]) / abs(want[k]) for k in SKEW_KEYS)
        if scale == 1.0:
            assert rel < 6.0e-2  # measured 5.3e-2 on this ring at full strength
            for k in SKEW_KEYS:
                assert np.angle(got[k]) == pytest.approx(np.angle(want[k]), abs=6e-2), k
                assert abs(got[k] - np.conj(want[k])) / abs(want[k]) > 0.1, k
        if rel_prev is not None:
            ratios.append(rel_prev / rel)
        rel_prev = rel
    assert all(r == pytest.approx(4.0, rel=0.1) for r in ratios), ratios


def test_both_magnets_at_once_needs_no_cross_term() -> None:
    """A ring carrying sextupoles *and* skew quadrupoles: the two sums simply add.

    At first order there is nothing to fit between them -- each magnet contributes to its
    own monomials and to no others -- and this is the statement that accsim's unadjusted
    sum is what a second code gets. The residual is the same coupled-optics gap as above,
    not a missing term, so it is gated at the size that gap has on this ring.
    """
    lat, line = _rings(SEXTS, SKEWS)
    got, want = resonance_driving_terms(lat), _xtrack_rdts(line)
    for key in KEYS:
        assert got[key] == pytest.approx(want[key], rel=6e-2, abs=0.0), key
    # the sextupole terms do not move when skew quadrupoles are added and vice versa
    sext_alone = resonance_driving_terms(_accsim_ring(SEXTS))
    skew_alone = resonance_driving_terms(_accsim_ring(None, SKEWS))
    for key in SEXT_KEYS:
        assert got[key] == pytest.approx(sext_alone[key], rel=1e-12, abs=0.0), key
    for key in SKEW_KEYS:
        assert got[key] == pytest.approx(skew_alone[key], rel=1e-12, abs=0.0), key


def test_the_reference_point_moves_the_same_way_in_both_codes() -> None:
    r"""xtrack reports every RDT along the ring; accsim reports it at the entrance.

    Rolling accsim's element list to each of xtrack's observation points must reproduce
    xtrack's column -- which tests the covariance law (``f`` rotates between sources and
    jumps at them) against a second implementation of it, rather than only against the
    algebra that produced it. Both codes' rows sit at the element **entrance**, which is
    itself established here: a quadrupole and a drift have length, so entrance and exit
    differ by a real phase advance, and agreeing at either one excludes the other.
    """
    lat, line = _rings(SEXTS)
    tw = line.twiss4d()
    out = xt.rdt_first_order_perturbation(
        list(SEXT_KEYS), twiss=tw, strengths=line.get_table(attr=True), feed_down=False
    )
    els = list(lat.elements)
    checked = 0
    for i, elem in enumerate(els):
        if isinstance(elem, ThinSextupole):
            continue  # the discontinuous points -- their own test, below
        rolled = Lattice(els[i:] + els[:i], lat.ref)
        here = resonance_driving_terms(rolled)
        for key in SEXT_KEYS:
            assert here[key] == pytest.approx(complex(out[key][i]), rel=1e-8, abs=0.0), (i, key)
        checked += 1
    assert checked >= 15
    # and it really does move: the entrance value is not the value everywhere
    assert abs(complex(out["f3000"][0]) - complex(out["f3000"][len(els) // 2])) > 1e-3


def test_at_a_source_the_two_codes_report_opposite_sides_of_the_jump() -> None:
    r"""And the size of the step is the source's own undivided contribution, exactly.

    Found rather than expected. An RDT is discontinuous at a thin source, so "the value
    at element ``i``" needs a side, and the two codes pick different ones: rolling
    accsim's list so the sextupole is first observes **upstream** of its kick, while
    xtrack's row for that element is **downstream** of it. Neither is wrong and the
    difference is invisible everywhere else in the ring, which is exactly what makes it
    worth pinning: a comparison written without it fails at three points out of
    twenty-two and looks like a phase error.

    What turns that into a gate rather than a caveat is that the step is *predicted*. The
    covariance law says crossing one source at zero phase advance adds precisely its
    plain ``F`` -- coefficient, strength, beta powers, no resonance denominator -- and
    that is checked here, on all five terms, against a second code: agreement to ``3e-11``
    absolute on steps of order one.
    """
    lat, line = _rings(SEXTS)
    tw = line.twiss4d()
    out = xt.rdt_first_order_perturbation(
        list(SEXT_KEYS), twiss=tw, strengths=line.get_table(attr=True), feed_down=False
    )
    els = list(lat.elements)
    sources = [i for i, e in enumerate(els) if isinstance(e, ThinSextupole)]
    assert len(sources) == len(SEXTS)
    for i in sources:
        upstream = Lattice(els[i:] + els[:i], lat.ref)
        before = resonance_driving_terms(upstream)
        after = resonance_driving_terms(Lattice(els[i + 1 :] + els[: i + 1], lat.ref))
        # the source sits at zero phase in the rolled frame, so its plain F has no phase
        sites, _, _ = _rdt_sites(upstream, 32)
        strength, bx, by, _, _ = sites["sext"]
        for key in SEXT_KEYS:
            mx, my, px, py, coef, _kind = _RDT_TERMS[key]
            step = coef * strength[0] * bx[0] ** px * by[0] ** py
            assert after[key] - before[key] == pytest.approx(step, rel=1e-11, abs=0.0), key
            assert complex(out[key][i]) == pytest.approx(after[key], rel=1e-8, abs=0.0), key
            assert abs(complex(out[key][i]) - before[key]) > 0.1 * abs(step), key
