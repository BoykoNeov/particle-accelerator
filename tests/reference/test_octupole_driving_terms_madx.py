r"""O5 against MAD-X PTC's ``gnfu`` -- and the half of the milestone PTC cannot see.

O4 decoded PTC's generating-function output: ``select_ptc_normal, gnfu=`` takes **four**
indices, ``normal_results`` carries ``order1`` through ``order4``, blank rows arrive under
keys that were never requested, and the returned value is ``j! k! l! m!`` times the RDT.
This file re-runs that decoding one degree up, where it is a **stronger** test than it was
at cubic order: the eight quartic terms carry four distinct factorial weights
(``24, 6, 4, 2``) against the cubic five's three, so a wrong normalisation would have to
be four different numbers at once.

**And it establishes a scope fact that cost the milestone a reference leg.** PTC's
``gnfu`` exposes only terms whose vertical charge ``l - m`` is **even** -- the terms a
mid-plane-symmetric machine drives. Every skew-sextupole term has odd vertical charge, and
PTC returns no row for any of them: asked for all twenty cubic keys it returns the same
five normal-sextupole keys whatever the ring contains, and on a ring whose only sources
are skew sextupoles those five come back **exactly zero**. That is measured here
(``test_ptc_exposes_no_odd_vertical_charge_term``) rather than inferred, because "the
table does not list it" is precisely the reasoning O4 had to unlearn. The consequence is
recorded plainly: the eight octupole terms have two independent reference codes, the five
skew-sextupole terms have one (``xtrack``) plus tracking.

**The two codes are related by an identity, not by equality.** Raw, they agree to about
four digits -- too good for a convention error, too poor for round-off. The cause is not
arithmetic on either side: PTC runs here with ``exact=true``, so the *lattice itself* is
nonlinear at quartic order and drives these lines with no octupole present at all, while
accsim (like ``sextupole_detuning`` before it) reports the **magnets'** contribution and
returns exactly zero for a ring of drifts. The relation that does hold, to ``1e-12`` on
all eight terms in both parts, is

    PTC(ring with octupoles) = accsim(octupoles) + PTC(same ring, octupoles off)

and its right-hand side is measured, not fitted: the gap is the *same number* at 0.3x, 1x
and 3x strength while the terms themselves change by a factor of ten. This is O3's "an
exact drift detunes with no magnets in the ring" arriving one degree up, from an
independent all-orders code.
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
    ThinSkewSextupole,
    resonance_driving_terms,
    tunes,
)
from accsim.twiss import _RDT_TERMS

pytestmark = pytest.mark.reference

MASS0, GAMMA0 = 938.27208816e6, 20.0
ENERGY_GEV = 10.0

#: A generic working point: ``Q_x != Q_y``, and every driven line well clear.
KF, KD = 0.8225, -0.90

#: Three octupoles at different beta and different phase, one of them negative.
OCTS = {5.5: 400.0, 8.4: -280.0, 11.6: 180.0}

#: Two skew sextupoles, for the ring that establishes what PTC will not report.
SKEWSEXTS = {4.2: 3.0, 9.7: -2.1}

KEYS = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "oct")
SKEWSEXT_KEYS = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "skewsext")


def _indices(key: str) -> tuple[int, int, int, int]:
    return tuple(int(c) for c in key[1:])  # type: ignore[return-value]


def _factorial_weight(key: str) -> int:
    return math.prod(math.factorial(i) for i in _indices(key))


def _madx_ring(scale: float = 1.0, skewsexts: dict[float, float] | None = None) -> str:
    """The fixture as a MAD-X sequence: ``Q(kf) D Q(kd) D`` four times, 12 m, no bends.

    Bend-free on purpose -- no dispersion at the sources, so no feed-down, which accsim
    does not model here. The free stretches are ``(0.5, 1.5)``, ``(2.0, 3.0)``, ...; a
    source placed inside a quadrupole makes MAD-X abort on a negative drift rather than
    quietly vanish, which is how the positions below were checked.
    """
    elems = [(0.25 + 3.0 * c, f"qf{c}: qf") for c in range(4)]
    elems += [(1.75 + 3.0 * c, f"qd{c}: qd") for c in range(4)]
    elems += [
        (pos, f"oc{i}: multipole, knl={{0,0,0,{w * scale:.12g}}}")
        for i, (pos, w) in enumerate(OCTS.items())
    ]
    elems += [
        (pos, f"ss{i}: multipole, ksl={{0,0,{w:.12g}}}")
        for i, (pos, w) in enumerate((skewsexts or {}).items())
    ]
    body = "\n".join(f"      {d}, at={pos};" for pos, d in sorted(elems))
    return f"""
    beam, particle=proton, energy={ENERGY_GEV}, sequence=ring;
    qf: quadrupole, l=0.5, k1= {KF};
    qd: quadrupole, l=0.5, k1= {KD};
    ring: sequence, l=12.0, refer=centre;
{body}
    endsequence;
    """


def _accsim_ring(scale: float = 1.0, skewsexts: dict[float, float] | None = None) -> Lattice:
    """The identical ring in accsim, element for element."""
    marks: dict[float, tuple[str, float]] = {p: ("o", w * scale) for p, w in OCTS.items()}
    marks.update({p: ("t", w) for p, w in (skewsexts or {}).items()})
    els: list = []
    s = 0.0
    for _ in range(4):
        for k in (KF, KD):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            done = 0.0
            for p in sorted(q for q in marks if s < q < s + 1.0):
                kind, w = marks[p]
                els.append(Drift(p - s - done))
                els.append(ThinOctupole(w) if kind == "o" else ThinSkewSextupole(w))
                done = p - s
            els.append(Drift(1.0 - done))
            s += 1.0
    placed = sum(isinstance(e, (ThinOctupole, ThinSkewSextupole)) for e in els)
    assert placed == len(marks), "a source landed inside a quadrupole, not in drift space"
    return Lattice(els, ReferenceParticle.from_gamma(MASS0, GAMMA0))


def _all_keys(degree: int) -> list[tuple[int, int, int, int]]:
    """Every four-index key of the given total degree, so nothing is pre-selected."""
    n = degree + 1
    return [
        (a, b, c, d)
        for a in range(n)
        for b in range(n)
        for c in range(n)
        for d in range(n)
        if a + b + c + d == degree
    ]


def _ptc_gnf(sequence: str, degree: int = 4, order: int = 5) -> dict:
    """``{"gnfa"|"gnfc"|"gnfs"|"q1"|"q2": {(j,k,l,m): value}}`` from ``ptc_normal``.

    Read live out of ``normal_results``; nothing is transcribed, and **every** key of the
    requested degree is asked for rather than the ones expected to be nonzero, so a term
    landing somewhere unexpected shows up instead of being missed.

    ``no=5`` rather than O4's ``no=4``: a quartic generating-function term needs the map
    to one order beyond it. ``no=6`` returns the same numbers bit for bit, which is gated
    below.
    """
    sel = "\n".join(
        f"      select_ptc_normal, gnfu={a},{b},{c},{d};" for a, b, c, d in _all_keys(degree)
    )
    with madx_session() as madx:
        madx.input(sequence)
        madx.use(sequence="ring")
        madx.twiss(sequence="ring")
        madx.input(f"""
        ptc_create_universe;
        ptc_create_layout, model=2, method=6, nst=5, exact=true;
          select_ptc_normal, q1=0, q2=0;
{sel}
        ptc_normal, closed_orbit, normal, icase=4, no={order};
        ptc_end;
        """)
        t = madx.table.normal_results
        out: dict[str, dict[tuple[int, ...], float]] = {}
        for n, o1, o2, o3, o4, v in zip(
            t.name, t.order1, t.order2, t.order3, t.order4, t.value, strict=True
        ):
            out.setdefault(str(n).strip(), {})[(int(o1), int(o2), int(o3), int(o4))] = float(v)
    return out


def _ptc_complex(gnf, key: str) -> complex:
    """PTC's term as a complex number, with its factorial weight divided out."""
    idx = _indices(key)
    return complex(gnf["gnfc"][idx], gnf["gnfs"][idx]) / _factorial_weight(key)


@pytest.fixture(scope="module")
def paired():
    """``(PTC with the octupoles, PTC with them switched off, accsim)``.

    The bare run is part of the comparison rather than a control, for the reason set out
    in ``test_ptc_carries_the_lattices_own_nonlinearity_and_accsim_does_not``: PTC reports
    the ring's *total* quartic content and accsim reports its magnets' contribution, so
    the two are related by an identity rather than by equality.
    """
    return (
        _ptc_gnf(_madx_ring()),
        _ptc_gnf(_madx_ring(scale=0.0)),
        resonance_driving_terms(_accsim_ring()),
    )


def test_the_two_rings_are_the_same_machine() -> None:
    """Before anything is compared: the linear optics must agree.

    A tune mismatch would move every resonance denominator underneath the comparison and
    show up as a driving-term disagreement that is really an optics one.
    """
    qx, qy = tunes(_accsim_ring())
    gnf = _ptc_gnf(_madx_ring())
    assert gnf["q1"][(0, 0, 0, 0)] == pytest.approx(qx, rel=1e-9)
    assert gnf["q2"][(0, 0, 0, 0)] == pytest.approx(qy, rel=1e-9)
    assert abs(qx - qy) > 0.1  # the fixture is not on a degenerate working point


def test_the_quartic_keys_are_j_k_l_m_and_blank_rows_are_recognised_by_value(paired) -> None:
    """O4's decoding, re-established one degree up rather than inherited.

    All thirty-five quartic keys are requested; PTC returns a far smaller set, of which
    exactly eight carry content and are keyed precisely ``(j, k, l, m)`` -- accsim's own
    eight names, one for one. The rest are numerically zero and carry keys that were never
    requested and are not even quartic.

    The two rules O4 had to learn are re-asserted here because they are what makes the
    next section's negative result trustworthy: **absence from the table is not evidence a
    term is zero**, and **a row must be recognised by its value, not by its key**.
    """
    gnf, _bare, _mine = paired
    wanted = {_indices(k) for k in KEYS}
    rows = gnf["gnfa"]
    assert wanted <= set(rows), sorted(set(rows))
    for idx in wanted:
        assert abs(rows[idx]) > 1e-9, idx
    for idx, value in rows.items():
        if idx not in wanted:
            assert abs(value) < 1e-9, (idx, value)
    assert len(rows) < len(_all_keys(4)), "PTC returns a subset -- absence proves nothing"
    assert any(sum(idx) != 4 for idx in rows), "the empty rows carry non-quartic keys"


def test_ptc_carries_the_lattices_own_nonlinearity_and_accsim_does_not(paired) -> None:
    r"""The two codes are related by an **identity**, not by equality -- and it is exact.

    Raw, PTC and accsim agree to about four digits and no further, which is far too good
    to be a convention error and far too poor to be round-off. Localised rather than
    tolerated, and the cause is not on either side's arithmetic: MAD-X runs PTC here with
    ``exact=true``, so the *lattice itself* -- exact drifts, and the quadrupoles' own
    kinematic terms -- is nonlinear at quartic order and contributes to these coefficients
    with no octupole present at all. PTC reports the ring's total quartic content; accsim,
    like :func:`sextupole_detuning` before it, reports the **magnets'** contribution and
    returns exactly zero for a ring of drifts.

    So the relation to check is

        PTC(ring with octupoles) = accsim(octupoles) + PTC(same ring, octupoles off)

    which holds to ``1e-12`` on all eight terms in both real and imaginary part -- three
    orders better than the raw comparison and, unlike it, an equality rather than an
    agreement. This is O3's "an exact drift detunes with no magnets in the ring" arriving
    one order up and from an independent code; the same quantity is seen from the tracking
    side in ``tests/analytic/test_octupole_driving_terms.py``.
    """
    full, bare, mine = paired
    for key in KEYS:
        theirs, floor = _ptc_complex(full, key), _ptc_complex(bare, key)
        assert theirs == pytest.approx(mine[key] + floor, rel=0.0, abs=1e-11), key
        assert abs(floor) > 0.0, key  # the bare lattice really does drive these lines
        assert abs(floor) / abs(mine[key]) < 1e-3, key  # and it is a small correction


def test_the_lattices_own_contribution_does_not_scale_with_the_octupoles(paired) -> None:
    """Which is what identifies it as the lattice's and not a mis-scaled octupole term.

    A wrong strength or a wrong coefficient on accsim's side would be *proportional* to
    the octupoles and would masquerade as this offset at any single strength. It does not
    scale: the absolute gap ``PTC - accsim`` is the same number at ``0.3x``, ``1x`` and
    ``3x``, to round-off, while the terms themselves change by a factor of ten.
    """
    _full, bare, _mine = paired
    gaps = {}
    for scale in (0.3, 1.0, 3.0):
        gnf = _ptc_gnf(_madx_ring(scale))
        mine = resonance_driving_terms(_accsim_ring(scale))
        gaps[scale] = {k: _ptc_complex(gnf, k) - mine[k] for k in KEYS}
        for key in KEYS:  # the terms themselves are strictly proportional
            base = resonance_driving_terms(_accsim_ring(1.0))
            assert mine[key] == pytest.approx(scale * base[key], rel=1e-12, abs=0.0), key
    for key in KEYS:
        for scale in (0.3, 3.0):
            assert gaps[scale][key] == pytest.approx(gaps[1.0][key], rel=0.0, abs=1e-11), (
                scale,
                key,
            )
        assert gaps[1.0][key] == pytest.approx(_ptc_complex(bare, key), rel=0.0, abs=1e-11), key


def test_the_factorial_weight_is_measured_not_recalled(paired) -> None:
    """``gnf = j! k! l! m! f_jklm`` -- established from the pattern, not from one number.

    Four distinct weights on eight terms (``24`` for ``f4000`` and ``f0040``, ``6`` for
    ``f3100`` and ``f0031``, ``4`` for ``f2020`` and ``f2002``, ``2`` for ``f2011`` and
    ``f1120``), so this cannot be a single constant absorbing an error: a wrong overall
    normalisation would have to be four different numbers at once, and a wrong *per-term*
    coefficient would have to conspire with the weight it is paired against. The ratios
    are measured first and only then asserted to be the factorial.
    """
    full, bare, mine = paired
    measured = {
        key: complex(full["gnfc"][_indices(key)], full["gnfs"][_indices(key)])
        / (mine[key] + _ptc_complex(bare, key))
        for key in KEYS
    }
    for key, ratio in measured.items():
        assert ratio.imag == pytest.approx(0.0, abs=1e-10 * abs(ratio)), key
        assert ratio.real == pytest.approx(_factorial_weight(key), rel=1e-10), key
    assert {round(r.real) for r in measured.values()} == {24, 6, 4, 2}, "the weights must differ"


def test_all_eight_terms_agree_in_real_and_imaginary_part(paired) -> None:
    """The comparison itself: the complex number, not the modulus.

    A modulus-only test would pass with accsim shipping the mirror basis, which is the one
    convention O4 had to choose and this milestone inherits. Both codes' phases agree, and
    the conjugate is asserted to be decisively wrong rather than merely worse.

    The tolerance is ``1e-4`` here and ``1e-12`` in the identity above, and the difference
    between those two numbers is the whole content of this section: what is left at ``1e-4``
    is not error, it is the lattice's own quartic content, which the identity accounts for
    and this raw comparison does not.
    """
    full, bare, mine = paired
    for key in KEYS:
        theirs = _ptc_complex(full, key)
        assert mine[key] == pytest.approx(theirs, rel=1e-3, abs=0.0), key
        assert abs(mine[key] - np.conj(theirs)) / abs(theirs) > 0.1, key
        assert mine[key] + _ptc_complex(bare, key) == pytest.approx(theirs, rel=0.0, abs=1e-11)


def test_ptc_exposes_no_odd_vertical_charge_term() -> None:
    r"""The measured scope fact: the skew-sextupole half of this milestone has no PTC leg.

    Asked for **all twenty** cubic keys, PTC returns the same five -- the normal
    sextupole's ``(3,0,0,0)``, ``(2,1,0,0)``, ``(1,0,2,0)``, ``(1,0,1,1)``, ``(1,0,0,2)``
    -- whatever the ring holds. Every one of those has even vertical charge ``l - m``;
    every skew-sextupole term has odd vertical charge; and none of the latter is returned.

    The negative is established three ways rather than by one empty table, because "not
    listed" is exactly the inference O4 had to unlearn: the five come back **numerically
    zero** on a ring whose only sources are skew sextupoles, they are **unchanged bit for
    bit** when skew sextupoles are added to a ring that also has octupoles, and accsim's
    own answer for those five terms on the skew-only ring is likewise exactly zero -- so
    the two codes agree about what is there, and simply have nothing to say to each other
    about what PTC will not report.
    """
    skew_only = _ptc_gnf(_madx_ring(scale=0.0, skewsexts=SKEWSEXTS), degree=3, order=5)
    both = _ptc_gnf(_madx_ring(skewsexts=SKEWSEXTS), degree=3, order=5)
    octs_only = _ptc_gnf(_madx_ring(), degree=3, order=5)

    cubic = {idx for idx in skew_only["gnfa"] if sum(idx) == 3}
    assert cubic, "PTC returned no cubic rows at all"
    assert all((ell - m) % 2 == 0 for (_j, _k, ell, m) in cubic), sorted(cubic)
    assert not ({_indices(k) for k in SKEWSEXT_KEYS} & cubic), sorted(cubic)

    for idx in cubic:  # nothing there, on a ring that demonstrably drives cubic terms
        assert abs(skew_only["gnfa"][idx]) < 1e-12, idx
    for name in ("gnfc", "gnfs"):
        for idx in cubic:
            assert both[name][idx] == octs_only[name][idx], (name, idx)

    mine = resonance_driving_terms(_accsim_ring(scale=0.0, skewsexts=SKEWSEXTS))
    assert all(abs(mine[k]) > 1e-6 for k in SKEWSEXT_KEYS)  # accsim does see them
    assert all(mine[k] == 0.0 for k in ("f3000", "f2100", "f1020", "f1011", "f1002"))


def test_first_order_in_k3_is_exact_here(paired) -> None:
    """First order in ``k3`` is **exact** for these terms, as it was in ``k2`` for O4's.

    The structural reason is the same one, one degree up: the bracket of two quartic
    generators is of degree six, so an octupole's second-order contribution cannot return
    to these quartic coefficients. The evidence is the previous test read the other way --
    once the lattice's own contribution is subtracted, what is left tracks accsim's
    strictly linear sum to ``1e-11`` across a factor of ten in strength, so neither code
    is carrying a second-order piece the other lacks.

    Worth keeping the pair in mind: an octupole's **detuning** is first order and large
    enough to move the tune (it is what ``amplitude_detuning`` returns), while its quartic
    **driving terms** have no second-order correction at all.
    """
    _full, bare, _mine = paired
    for scale in (0.3, 3.0):
        gnf = _ptc_gnf(_madx_ring(scale))
        mine = resonance_driving_terms(_accsim_ring(scale))
        for key in KEYS:
            residual = _ptc_complex(gnf, key) - _ptc_complex(bare, key)
            assert residual == pytest.approx(mine[key], rel=0.0, abs=1e-11), (scale, key)


def test_increasing_ptcs_order_returns_the_same_numbers() -> None:
    """``no = 5`` and ``no = 6`` agree bit-for-bit, which is PTC's own version of the above.

    Nothing at higher order feeds back into these coefficients, so raising the order the
    map is computed to changes nothing at all. If it did, "first order is exact here" would
    be false and the previous test's agreement would be a coincidence of strength.
    """
    a = _ptc_gnf(_madx_ring(), order=5)
    b = _ptc_gnf(_madx_ring(), order=6)
    for key in KEYS:
        for name in ("gnfc", "gnfs"):
            assert a[name][_indices(key)] == b[name][_indices(key)], (name, key)
