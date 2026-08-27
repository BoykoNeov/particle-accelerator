r"""O4 against MAD-X PTC -- and the ``gnfu`` indexing the roadmap left undecoded.

``ptc_normal`` builds the nonlinear normal form of the one-turn map in a separate Fortran
codebase and reports the generating-function terms as ``gnfa``/``gnfc``/``gnfs``
(amplitude, cosine, sine). The O3 session found those terms and could not say **which
RDT each one was**: selecting three-index keys returned rows whose keys did not line up
with the request, and the roadmap's O4 candidate recorded "PTC's ``gnfu`` indexing is
still undecoded -- mapping that is a session of its own."

**It is decoded here, and the difficulty was two things at once, neither of them
physics.** ``select_ptc_normal, gnfu=...`` takes **four** indices and ``normal_results``
carries ``order1`` through **``order4``** -- reading only the first three drops the last
index, which is what made the returned keys look unrelated to the request. And PTC
returns a handful of **empty rows whose keys were never asked for and are not even cubic**
(here ``(0,0,2,0)``, ``(1,0,0,0)``, ``(2,0,0,0)``), sitting in the table next to the real
ones. Truncated keys plus junk-keyed blanks is exactly the "five entries keyed
``(1,0,0)``, ``(1,0,1)``, ``(1,0,2)``, ``(2,1,0)``, ``(3,0,0)`` for three requests" the
earlier session recorded.

With the fourth column read and the blanks recognised by their **value** rather than
their key, the mapping is trivial: PTC's key is ``(j, k, l, m)``, the same four indices
accsim uses.

The remaining difference is a **factorial**, and it is measured rather than recalled:

    gnfc(j,k,l,m) + i gnfs(j,k,l,m)  ==  j! k! l! m!  *  f_jklm ,

with ``gnfa`` its modulus. It is established below by ratio, on all five terms at once,
where the four distinct factors ``6, 2, 2, 1`` appear on four different terms -- a single
wrong overall constant cannot reproduce that pattern.

**What this establishes, and what it does not.** Unlike xtrack's, PTC's normal form is
computed all-orders from the map itself rather than from a twiss table and a strengths
list, so this is a genuinely different route to the number, and it fixes the same two
things that route can fix: the basis (accsim's phases, not their conjugates) and the
overall normalisation. It is *not* independent evidence about the tracked physics -- for
that see the tracked sidebands in ``tests/analytic``.

**And one thing that is not the same as O3.** O3's sextupole detuning does not exist at
first order at all; it is a second-order effect and the whole milestone was about that
second order. These RDTs are the opposite: the agreement with an all-orders code stays at
round-off when the sextupole strength is **tripled**, because a sextupole's second-order
contribution lands on *quartic* monomials and never returns to the cubic ones. So "first
order in ``k2``" is exact here rather than a leading approximation -- which is why this
file gates a ratio of one and not a convergence.

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
    ThinSextupole,
    resonance_driving_terms,
    tunes,
)

pytestmark = pytest.mark.reference

MASS0, GAMMA0 = 938.27208816e6, 20.0
ENERGY_GEV = 10.0

#: A generic working point: ``Q_x != Q_y``, and all four driven lines well clear.
KF, KD = 0.8225, -0.90

#: Three sextupoles at different beta and different phase, one of them negative.
SEXTS = {5.5: 1.0, 8.4: -0.7, 11.6: 0.45}

KEYS = ("f3000", "f2100", "f1020", "f1011", "f1002")


def _indices(key: str) -> tuple[int, int, int, int]:
    return tuple(int(c) for c in key[1:])  # type: ignore[return-value]


def _factorial_weight(key: str) -> int:
    return math.prod(math.factorial(i) for i in _indices(key))


def _madx_ring(k2_scale: float = 1.0, kf: float = KF) -> str:
    """The fixture as a MAD-X sequence: ``Q(kf) D Q(kd) D`` four times, 12 m, no bends.

    Bend-free on purpose -- no dispersion at the sextupoles, so no feed-down, which
    accsim does not model here.
    """
    elems = [(0.25 + 3.0 * c, f"qf{c}: qf") for c in range(4)]
    elems += [(1.75 + 3.0 * c, f"qd{c}: qd") for c in range(4)]
    elems += [
        (pos, f"sx{i}: multipole, knl={{0,0,{w * k2_scale:.12g}}}")
        for i, (pos, w) in enumerate(SEXTS.items())
    ]
    body = "\n".join(f"      {d}, at={pos};" for pos, d in sorted(elems))
    return f"""
    beam, particle=proton, energy={ENERGY_GEV}, sequence=ring;
    qf: quadrupole, l=0.5, k1= {kf};
    qd: quadrupole, l=0.5, k1= {KD};
    ring: sequence, l=12.0, refer=centre;
{body}
    endsequence;
    """


def _accsim_ring(k2_scale: float = 1.0, kf: float = KF) -> Lattice:
    """The identical ring in accsim, element for element."""
    where = {p: w * k2_scale for p, w in SEXTS.items()}
    els: list = []
    s = 0.0
    for _ in range(4):
        for k in (kf, KD):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            done = 0.0
            for p in sorted(q for q in where if s < q < s + 1.0):
                els += [Drift(p - s - done), ThinSextupole(where[p])]
                done = p - s
            els.append(Drift(1.0 - done))
            s += 1.0
    assert sum(isinstance(e, ThinSextupole) for e in els) == len(SEXTS)
    return Lattice(els, ReferenceParticle.from_gamma(MASS0, GAMMA0))


#: Every cubic four-index key, so nothing is pre-selected by what the answer should be.
_ALL_CUBIC = [
    (a, b, c, d)
    for a in range(4)
    for b in range(4)
    for c in range(4)
    for d in range(4)
    if a + b + c + d == 3
]


def _ptc_gnf(sequence: str, order: int = 4) -> dict[str, dict[tuple[int, ...], float]]:
    """``{"gnfa"|"gnfc"|"gnfs": {(j,k,l,m): value}}`` from ``ptc_normal``.

    Read live out of ``normal_results``; nothing is transcribed, and **every** cubic key
    is requested rather than the five that are expected to be nonzero, so a term landing
    somewhere unexpected shows up instead of being missed.
    """
    sel = "\n".join(f"      select_ptc_normal, gnfu={a},{b},{c},{d};" for a, b, c, d in _ALL_CUBIC)
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
    return _ptc_gnf(_madx_ring()), resonance_driving_terms(_accsim_ring())


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


def test_the_gnfu_key_has_four_indices_and_they_are_j_k_l_m(paired) -> None:
    """The decoding -- and the exact reason the three-index reading looked impossible.

    All twenty cubic keys are requested; PTC returns **eight** rows. Five carry content
    and are keyed precisely ``(j, k, l, m)``, matching accsim's five names one for one.
    The other three are numerically **zero** and carry keys that were never requested and
    are not even cubic -- ``(0,0,2,0)``, ``(1,0,0,0)``, ``(2,0,0,0)`` on this ring. That
    is the whole of the earlier session's "five entries keyed differently from the
    request": junk-keyed empty rows sitting next to correctly-keyed real ones, with the
    fourth index dropped so the two kinds could not be told apart.

    Two consequences are pinned here because they will bite the next person:
    **absence from the table is not evidence a term is zero** (PTC returns far fewer rows
    than were asked for), and **a row must be recognised by its value, not by trusting
    its key** -- every row whose key is not one of the five is asserted to be empty.
    """
    gnf, _ = paired
    wanted = {_indices(k) for k in KEYS}
    rows = gnf["gnfa"]
    assert wanted <= set(rows), sorted(set(rows))
    for idx in wanted:
        assert abs(rows[idx]) > 1e-9, idx
    for idx, value in rows.items():
        if idx not in wanted:
            assert abs(value) < 1e-9, (idx, value)
    assert len(rows) < len(_ALL_CUBIC), "PTC returns a subset -- absence proves nothing"
    assert any(sum(idx) != 3 for idx in rows), "the empty rows carry non-cubic keys"


def test_the_factorial_weight_is_measured_not_recalled(paired) -> None:
    """``gnf = j! k! l! m! f_jklm`` -- established from the pattern, not from one number.

    The five terms carry four distinct weights (``6, 2, 2, 1``), so this is not a single
    constant that could absorb any error: a wrong overall normalisation would have to be
    four different numbers at once. The ratios are measured first and the factorial is
    only then asserted to be what they are.
    """
    gnf, mine = paired
    measured = {
        key: complex(gnf["gnfc"][_indices(key)], gnf["gnfs"][_indices(key)]) / mine[key]
        for key in KEYS
    }
    for key, ratio in measured.items():
        assert ratio.imag == pytest.approx(0.0, abs=1e-10 * abs(ratio)), key
        assert ratio.real == pytest.approx(_factorial_weight(key), rel=1e-10), key
    assert {round(r.real) for r in measured.values()} == {6, 2, 1}, "the weights must differ"


def test_all_five_terms_agree_in_real_and_imaginary_part(paired) -> None:
    """The comparison itself: round-off on the complex number, not on the modulus.

    A modulus-only test would pass with accsim shipping the mirror basis, which is the
    one convention this milestone had to choose. Both codes' phases agree, so accsim's
    ``h = u_hat + i p_hat`` is the field's convention and not merely a self-consistent
    one; and the conjugate is asserted to be decisively wrong rather than merely worse.
    """
    gnf, mine = paired
    for key in KEYS:
        theirs = _ptc_complex(gnf, key)
        assert mine[key] == pytest.approx(theirs, rel=1e-10, abs=0.0), key
        assert abs(mine[key] - np.conj(theirs)) / abs(theirs) > 0.1, key


def test_the_agreement_does_not_degrade_when_the_sextupoles_are_tripled() -> None:
    """First order in ``k2`` is **exact** for these terms, which O3's detuning is not.

    The natural expectation after O3 -- an all-orders code against a first-order formula
    must disagree somewhere, and the gap should grow with strength -- is wrong here, and
    the reason is structural rather than numerical: the bracket of two cubic generators
    is *quartic*, so a sextupole's second-order contribution lands on the amplitude
    detuning O3 computes and never returns to these cubic coefficients.

    Gated by measurement across a factor of ten in strength: accsim is exactly linear (it
    is a single sum over sources) and PTC tracks it to round-off at every point, so a
    silent second-order term in either code would show up as a drift here.
    """
    devs = []
    for scale in (0.3, 1.0, 3.0):
        gnf = _ptc_gnf(_madx_ring(scale))
        mine = resonance_driving_terms(_accsim_ring(scale))
        base = resonance_driving_terms(_accsim_ring(1.0))
        for key in KEYS:
            assert mine[key] == pytest.approx(scale * base[key], rel=1e-12, abs=0.0), key
            theirs = _ptc_complex(gnf, key)
            devs.append(abs(theirs / mine[key] - 1.0))
    assert max(devs) < 1e-10, max(devs)


def test_increasing_ptcs_order_returns_the_same_numbers() -> None:
    """``no = 4`` and ``no = 6`` agree bit-for-bit, which is the same statement from PTC.

    O3 found this for the detuning and it holds here for a different reason: there, the
    quartic coefficient was complete at ``no = 4``; here the cubic ones are complete at
    any order at all, because nothing at higher order feeds back into them.
    """
    a = _ptc_gnf(_madx_ring(), order=4)
    b = _ptc_gnf(_madx_ring(), order=6)
    for key in KEYS:
        for name in ("gnfc", "gnfs"):
            assert a[name][_indices(key)] == b[name][_indices(key)], (name, key)
