r"""O6 against MAD-X PTC's ``gnfu`` normal form, built about the **closed orbit**.

The second of the two independent arbiters, and the reason feed-down was chosen as the
milestone at all: PTC does not have a feed-down *option*. ``ptc_normal, closed_orbit``
constructs the normal form about the orbit the machine actually sits on, so a displaced
source's feed-down is in its answer whether or not anyone asks — which is precisely the
roadmap's point that neither reference code announces that it is doing this.

**What this leg covers that the xtrack one does not.** PTC obtains the map by composing
exact element maps rather than by finite differences, so it is not subject to the
nonlinear-tune leak that sets the ``1e-6`` floor next door. It agrees here to ``3e-7``.

**And what it does not cover.** O5 measured, one degree up, that PTC exposes **no**
odd-vertical-charge generating-function row, and the same is true here: of the twenty
terms, only the five with even vertical charge — exactly the normal-sextupole lines a
*horizontal* orbit feeds down onto — come back. The five skew-sextupole lines a
*vertical* orbit produces, and the misalignment half, have xtrack as their only arbiter.
That the two legs cover different halves is why the milestone needed both, and it is why
neither file's coverage is stated as "the reference legs agree".

**The comparison is an identity, not an equality — O4's structural fact, unchanged.**
PTC reports the ring's **total** cubic content and accsim reports its *magnets'*
contribution. On a bumped ring the lattice's own exact drifts carry a few ``1e-5`` of
cubic content of their own, which is a hundred times the tolerance here, so the octupoles
are switched off in a second PTC run and the difference is what accsim is compared with.
Without that subtraction the agreement would be limited to ``3e-4`` — measured, and gated
below so the subtraction cannot be mistaken for a fitting parameter.

**Convention.** O4 decoded the indexing and O5 re-ran it; nothing is re-argued here:

    gnfc(j,k,l,m) + i gnfs(j,k,l,m)  ==  j! k! l! m!  *  f_jklm,

with no conjugation, and ``no = 4`` — one order beyond the cubic terms being read.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _madx import madx_session

from accsim import (
    Corrector,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    resonance_driving_terms,
    resonance_driving_terms_on_orbit,
)
from accsim.orbit import closed_orbit_nonlinear
from accsim.twiss import _RDT_TERMS, _rdt_sites_on_orbit

pytestmark = pytest.mark.reference

MASS0, GAMMA0 = 938.27208816e6, 20.0
ENERGY_GEV = 10.0

#: O5's working point, so the two milestones describe one machine.
KF, KD = 0.8225, -0.90

#: Octupoles and **no sextupole**: every cubic term here is purely fed down.
OCTS = {5.5: 400.0, 8.4: -280.0, 11.6: 180.0}

#: The horizontal steerer that creates the whole effect under test.
KICK = 2.0e-4

#: The five normal-sextupole lines, taken from the shipped table by source kind so that a
#: later milestone adding a source cannot quietly widen or narrow this gate.
KEYS = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "sext")


def _indices(key: str) -> tuple[int, int, int, int]:
    return tuple(int(c) for c in key[1:])  # type: ignore[return-value]


def _factorial_weight(key: str) -> int:
    return math.prod(math.factorial(i) for i in _indices(key))


def _madx_ring(kick: float = KICK, scale: float = 1.0) -> str:
    """``Q(kf) D Q(kd) D`` four times, 12 m, bend-free, with a horizontal steerer at ``s=0``.

    ``scale`` multiplies every octupole, so ``scale=0`` is the same ring on the same bump
    with nothing to feed down — the control the identity above needs.
    """
    elems = [(0.25 + 3.0 * c, f"qf{c}: qf") for c in range(4)]
    elems += [(1.75 + 3.0 * c, f"qd{c}: qd") for c in range(4)]
    elems += [
        (pos, f"oc{i}: multipole, knl={{0,0,0,{w * scale:.12g}}}")
        for i, (pos, w) in enumerate(OCTS.items())
    ]
    elems += [(0.0, f"kck: hkicker, kick={kick:.12g}")]
    body = "\n".join(f"      {d}, at={pos};" for pos, d in sorted(elems))
    return f"""
    beam, particle=proton, energy={ENERGY_GEV}, sequence=ring;
    qf: quadrupole, l=0.5, k1= {KF};
    qd: quadrupole, l=0.5, k1= {KD};
    ring: sequence, l=12.0, refer=centre;
{body}
    endsequence;
    """


def _accsim_ring(kick: float = KICK, scale: float = 1.0) -> Lattice:
    """The identical ring in accsim, element for element."""
    els: list = [Corrector(kick_x=kick)]
    s = 0.0
    for _ in range(4):
        for k in (KF, KD):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            done = 0.0
            for p in sorted(q for q in OCTS if s < q < s + 1.0):
                els.append(Drift(p - s - done))
                els.append(ThinOctupole(OCTS[p] * scale))
                done = p - s
            els.append(Drift(1.0 - done))
            s += 1.0
    placed = sum(isinstance(e, ThinOctupole) for e in els)
    assert placed == len(OCTS), (
        "an octupole landed inside a quadrupole rather than in drift space; the free "
        "stretches of this fixture are (0.5, 1.5), (2.0, 3.0), ... and a silently "
        "unplaced source would make every gate here vacuous"
    )
    return Lattice(els, ReferenceParticle.from_gamma(MASS0, GAMMA0))


def _cubic_keys() -> list[tuple[int, int, int, int]]:
    """Every four-index key of total degree three, so nothing is pre-selected.

    Asking for the ones expected to be nonzero would hide a term landing somewhere
    unexpected — which is exactly how O5 discovered that PTC returns no odd-vertical-charge
    row at all.
    """
    return [
        (a, b, c, d)
        for a in range(4)
        for b in range(4)
        for c in range(4)
        for d in range(4)
        if a + b + c + d == 3
    ]


def _ptc_gnf(sequence: str, order: int = 4) -> dict:
    """``{"gnfa"|"gnfc"|"gnfs"|"q1"|"q2": {(j,k,l,m): value}}`` read live from PTC.

    ``closed_orbit`` is the whole point: it is what makes the normal form a property of
    the steered machine rather than of the blueprint, and therefore what puts feed-down
    into PTC's answer without a flag.
    """
    sel = "\n".join(
        f"      select_ptc_normal, gnfu={a},{b},{c},{d};" for a, b, c, d in _cubic_keys()
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


def _fed_down(bumped, bare, key: str) -> complex:
    """PTC's *magnet* contribution: its total, less the same ring with no octupoles."""
    return _ptc_complex(bumped, key) - _ptc_complex(bare, key)


@pytest.fixture(scope="module")
def paired():
    """``(PTC bumped, PTC bumped with octupoles off, PTC flat, accsim on-orbit)``."""
    return (
        _ptc_gnf(_madx_ring()),
        _ptc_gnf(_madx_ring(scale=0.0)),
        _ptc_gnf(_madx_ring(kick=0.0)),
        resonance_driving_terms_on_orbit(_accsim_ring()),
    )


# ==========================================================================
# 1. One machine, and one steered orbit, before anything is compared
# ==========================================================================


def test_the_two_rings_are_the_same_machine_on_the_same_bump(paired) -> None:
    """Tunes and closed orbit first: a driving-term comparison presumes both.

    PTC's ``q1``/``q2`` are compared with the walk's **own** on-orbit tunes rather than
    with the design lattice's, because those are what the sum divides by. The closed orbit
    is compared because it is the independent variable of the entire milestone: if the two
    codes sit on different orbits, an agreement in the terms would be a coincidence.
    """
    bumped, _, _, _ = paired
    lat = _accsim_ring()
    _, qx, qy = _rdt_sites_on_orbit(lat, 32, None, 0.0, 1e-7)
    assert bumped["q1"][(0, 0, 0, 0)] == pytest.approx(qx, rel=1e-7)
    assert bumped["q2"][(0, 0, 0, 0)] == pytest.approx(qy, rel=1e-7)
    with madx_session() as madx:
        madx.input(_madx_ring())
        madx.use(sequence="ring")
        tw = madx.twiss(sequence="ring")
        theirs = float(tw.x[0])
    assert closed_orbit_nonlinear(lat)[0] == pytest.approx(theirs, rel=1e-7)


def test_on_the_design_orbit_neither_code_drives_a_cubic_line(paired) -> None:
    """The reference point, and in PTC it is exact rather than small.

    With the steerer off, PTC returns **identically zero** on all five rows — an octupole
    reaches the sextupole lines only by feeding down, and there is nothing to feed down
    through. accsim's design-orbit function agrees, also exactly. Everything below is
    therefore a measurement of feed-down and of nothing else.
    """
    _, _, flat, _ = paired
    design = resonance_driving_terms(_accsim_ring(kick=0.0))
    for key in KEYS:
        assert _ptc_complex(flat, key) == 0.0
        assert design[key] == 0.0


# ==========================================================================
# 2. The identity, and why a subtraction belongs in it
# ==========================================================================


def test_ptc_carries_the_lattices_own_cubic_content_and_accsim_does_not(paired) -> None:
    r"""Why the comparison subtracts a control run instead of comparing totals.

    PTC reports the **ring's** total cubic content; accsim sums over its four source kinds
    and reports the **magnets'**. On a steered orbit those differ by the exact drifts'
    own contribution, which O3 already recorded in another guise ("an exact drift detunes
    with no magnets in the ring"). Here it is a few ``1e-5`` — a hundred times this
    file's tolerance, so it is not optional — and it is *independent of the octupoles*,
    which is what makes subtracting it an identity rather than a fitted offset.
    """
    bumped, bare, _, mine = paired
    for key in KEYS:
        own = _ptc_complex(bare, key)
        assert abs(own) > 1e-6, f"{key}: the control is empty, so subtracting it proves nothing"
        assert abs(own) < 1e-3 * abs(mine[key]), f"{key}: the control is not a small correction"
    # ...and comparing raw totals would be limited by exactly that, an order or two
    # above the agreement the subtraction achieves.
    raw = max(abs(_ptc_complex(bumped, k) - mine[k]) / abs(mine[k]) for k in KEYS)
    net = max(abs(_fed_down(bumped, bare, k) - mine[k]) / abs(mine[k]) for k in KEYS)
    assert raw > 30 * net


def test_the_control_run_really_is_the_bare_lattice() -> None:
    """A zero-strength multipole must be *inert*, not merely weak.

    The subtraction is only an identity if the ``scale=0`` run is the ring without its
    octupoles rather than the ring with three extra thin elements in it. PTC inserts a
    marker for every element in the sequence, and a thin multipole carrying
    ``knl={0,0,0,0}`` could in principle split a drift or add an integration boundary.
    This compares it against a sequence in which the octupoles are **absent from the
    lattice altogether**, which is the thing the control is supposed to stand for.
    """
    elems = [(0.25 + 3.0 * c, f"qf{c}: qf") for c in range(4)]
    elems += [(1.75 + 3.0 * c, f"qd{c}: qd") for c in range(4)]
    elems += [(0.0, f"kck: hkicker, kick={KICK:.12g}")]
    body = chr(10).join(f"      {d}, at={pos};" for pos, d in sorted(elems))
    absent = f"""
    beam, particle=proton, energy={ENERGY_GEV}, sequence=ring;
    qf: quadrupole, l=0.5, k1= {KF};
    qd: quadrupole, l=0.5, k1= {KD};
    ring: sequence, l=12.0, refer=centre;
{body}
    endsequence;
    """
    zeroed, removed = _ptc_gnf(_madx_ring(scale=0.0)), _ptc_gnf(absent)
    for key in KEYS:
        assert _ptc_complex(zeroed, key) == pytest.approx(_ptc_complex(removed, key), abs=1e-12)


def test_all_five_fed_down_terms_agree_in_real_and_imaginary_part(paired) -> None:
    """The comparison itself: both parts, both codes, one identity.

    Real and imaginary separately rather than a magnitude, because O1's lesson is that a
    magnitude-only comparison passes with the conjugate convention — and the conjugate is
    asserted to be decisively wrong rather than merely worse. Measured agreement is
    ``3e-7``; gated an order above that, which is still three times tighter than the
    xtrack leg can be.
    """
    bumped, bare, _, mine = paired
    for key in KEYS:
        theirs = _fed_down(bumped, bare, key)
        assert abs(mine[key]) > 1e-2, f"{key} is vacuously small on this fixture"
        assert mine[key].real == pytest.approx(theirs.real, rel=3e-6), key
        assert mine[key].imag == pytest.approx(theirs.imag, rel=3e-6), key
        assert abs(mine[key] - np.conj(theirs)) / abs(theirs) > 0.1, key


# ==========================================================================
# 3. What the agreement is and is not evidence for
# ==========================================================================


def test_the_departure_from_linearity_in_k3l_is_the_optics_half(paired) -> None:
    r"""Quartering the octupoles does **not** quarter the terms, and the miss is the point.

    The fed-down *strengths* are exactly linear in ``k3l`` — ``k2l_eff = k3l x_co``. If
    that were the whole model the terms would be linear too. They are not: feed-down also
    reaches the quadrupole order (``k1l_eff = k3l x_co^2 / 2``), so the ``beta`` and
    phases the sum is evaluated at move with the strength as well, and the closed orbit
    shifts a little too. The departure measured here is ``~0.1%`` — small, but it is this
    milestone's headline correction seen from a new direction, and it is a thousand times
    the tolerance the cross-code comparison passes at.

    So the gate is two-sided: near-linear, but **provably not linear**. And PTC, whose
    normal form is all-orders and which knows nothing of this decomposition, tracks the
    non-linear answer rather than the linear one — which is the actual result.
    """
    _, _, _, full = paired
    scale = 0.25
    bumped = _ptc_gnf(_madx_ring(scale=scale))
    bare = _ptc_gnf(_madx_ring(scale=0.0))
    mine = resonance_driving_terms_on_orbit(_accsim_ring(scale=scale))
    departures = []
    for key in KEYS:
        linear = scale * full[key]
        departures.append(abs(mine[key] - linear) / abs(linear))
        assert mine[key] == pytest.approx(linear, rel=0.02), f"{key}: not even nearly linear"
        # ...and PTC agrees with the actual answer, not with the linear extrapolation.
        assert mine[key] == pytest.approx(_fed_down(bumped, bare, key), rel=3e-6), key
    assert max(departures) > 1e-4, (
        f"the largest departure from linearity is {max(departures):.3g}; if the terms "
        "really were linear in k3l the optics half would not exist and the milestone's "
        "headline would be wrong"
    )


def test_increasing_ptcs_order_returns_the_same_numbers(paired) -> None:
    """``no = 4`` is converged: the answer is not an artefact of where PTC truncates.

    A cubic generating-function term needs the map to one order beyond it, and going
    further must change nothing. O4 and O5 each ran this check at their own degree; it is
    repeated rather than inherited because ``closed_orbit`` changes what PTC is expanding
    about, which is exactly the sort of thing that could reintroduce an order dependence.
    """
    bumped, bare, _, _ = paired
    hi_b, hi_r = _ptc_gnf(_madx_ring(), order=5), _ptc_gnf(_madx_ring(scale=0.0), order=5)
    for key in KEYS:
        assert _fed_down(hi_b, hi_r, key) == pytest.approx(_fed_down(bumped, bare, key), rel=1e-9)


def test_ptc_exposes_no_odd_vertical_charge_cubic_row(paired) -> None:
    r"""The half of the milestone this arbiter cannot see, measured rather than assumed.

    A *vertical* orbit through an octupole feeds down to a **skew** sextupole, driving the
    five odd-vertical-charge lines ``f2010 f2001 f1110 f0021 f0030``. PTC returns no such
    row — the same scope fact O5 measured one degree up, re-measured here at cubic order
    on a ring that genuinely drives them. So those five, and the misaligned-magnet half,
    have ``xtrack`` as their only arbiter, and this file must not be read as covering the
    milestone on its own.
    """
    bumped, _, _, _ = paired
    returned = set(bumped["gnfc"])
    odd = [k for k in _cubic_keys() if (k[2] - k[3]) % 2 == 1]
    assert odd, "the fixture for this check is empty"
    for key in odd:
        assert key not in returned or bumped["gnfc"][key] == 0.0, (
            f"PTC returned an odd-vertical-charge row {key}; if that is now supported, "
            "the skew half of this milestone gains a second arbiter and O5's scope note "
            "needs revisiting"
        )
