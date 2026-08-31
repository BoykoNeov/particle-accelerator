r"""O5: the resonance driving terms of a **normal octupole** and a **skew sextupole**.

The quartic sibling of ``test_resonance_driving_terms.py``. O4 shipped the seven
first-order terms a normal sextupole and a skew quadrupole drive; this file adds the
eight an octupole drives and the five a skew sextupole drives, by exactly the same route
-- the coefficient of the Lie generator that removes each non-action monomial, read off
the machinery ``test_sextupole_detuning.py`` (O3) builds and then discards.

**What this file inherits rather than re-argues.** O3 pinned four conventions (the
generator's sign, the resonance basis, the homological solution and the direction of the
rotation) and O4 pinned the fifth (``h = u_hat + i p_hat``, measured three ways). This
file imports O3's machinery directly, so a drift between the two would show up here as a
failed identity rather than as a silent disagreement. Nothing about the basis is
re-decided; what is genuinely new is which monomials a quartic and a *y-odd* cubic
generator reach.

**The one leg no reference code supplies.** An octupole's first-order generator splits
into an action-only part and the rest. The rest is what this file gates. The action-only
part is :func:`~accsim.twiss.amplitude_detuning`, shipped in J2 and derived there by a
route that shares no algebra with this one, so the two shipped functions must between
them account for the whole generator with nothing left over -- and that identity is the
milestone's equivalent of O4's ``|C-|`` tie. A skew sextupole has **no** action-only part
at all, so it gets no such leg, which is the same statement as "a skew sextupole does not
shift the tune to first order".

**The trap the cubic milestone did not have.** An octupole's two halves interfere in the
*measurement*: the action half is the amplitude detuning, which moves ``Q_x`` off the
lattice's linear tune, and the sideband the non-action half is read from moves with it --
three times as far, since that line sits at ``-3 Q_x``. Projecting at the linear tune
returns an answer a factor of ``48`` too small, which is why the tracked gates here
measure the tune from the trajectory. A sextupole has no first-order detuning,
which is exactly why O4 could use ``tunes(lat)`` and this file cannot.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Drift,
    Lattice,
    Octupole,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    ThinSextupole,
    ThinSkewQuadrupole,
    ThinSkewSextupole,
    closed_twiss,
    resonance_driving_terms,
    tunes,
)
from accsim.tracking import Particle, Tracker
from accsim.tune import naff
from accsim.twiss import (
    _RDT_TERMS,
    CoupledLatticeError,
    ResonantLatticeError,
    _blocks,
    _decoupled,
    _propagate_block,
    _rdt_sites,
    amplitude_detuning,
    match_periodic,
)

sys.path.insert(0, os.path.dirname(__file__))

import test_sextupole_detuning as o3  # noqa: E402

MASS0, GAMMA0 = 938.27208816e6, 20.0

#: The generic working point, inherited from O4 so the two files describe one machine.
KF, KD = 0.80, -0.90

#: Three octupoles at different beta and different phase, with a sign change among the
#: weights so no accidental symmetry can hide a wrong term.
OCTS = {5.5: 400.0, 8.4: -280.0, 11.6: 180.0}

#: Three skew sextupoles, likewise.
SKEWSEXTS = {2.4: 3.0, 6.8: -2.1, 9.9: 4.5}

#: The thirteen this file is about, selected by source kind for the same reason O4's list
#: is: a later milestone adding a source must not quietly widen or narrow these gates.
OCT_KEYS = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "oct")
SKEWSEXT_KEYS = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "skewsext")
KEYS = OCT_KEYS + SKEWSEXT_KEYS


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _fodo(
    ref: ReferenceParticle,
    octs: dict[float, float] | None = None,
    skewsexts: dict[float, float] | None = None,
    kf: float = KF,
    kd: float = KD,
    sexts: dict[float, float] | None = None,
    skews: dict[float, float] | None = None,
) -> Lattice:
    """``Q(kf) D Q(kd) D`` four times, 12 m, with thin sources dropped in at ``s``.

    Bend-free on purpose: no dispersion at the sources, so no feed-down, which this
    milestone does not model and which would otherwise contaminate every comparison.
    ``sexts``/``skews`` are here only so that one test can put all four source kinds in
    one ring and check they do not reach each other's terms.

    The placement assertion is O4's, kept because it earned itself there on its first
    run: the free stretches of this cell are ``(0.5, 1.5)``, ``(2.0, 3.0)``, ... and a
    source landing inside a quadrupole vanishes silently, which makes every gate reading
    it vacuous rather than wrong.
    """
    marks: dict[float, tuple[str, float]] = {}
    for tag, given in (("o", octs), ("t", skewsexts), ("s", sexts), ("k", skews)):
        marks.update({p: (tag, w) for p, w in (given or {}).items()})
    requested = sum(len(g or {}) for g in (octs, skewsexts, sexts, skews))
    assert len(marks) == requested, "two sources share an s"
    build = {
        "o": ThinOctupole,
        "t": ThinSkewSextupole,
        "s": ThinSextupole,
        "k": ThinSkewQuadrupole,
    }
    els: list = []
    s = 0.0
    for _ in range(4):
        for k in (kf, kd):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            done = 0.0
            for p in sorted(q for q in marks if s < q < s + 1.0):
                tag, w = marks[p]
                els.append(Drift(p - s - done))
                els.append(build[tag](w))
                done = p - s
            els.append(Drift(1.0 - done))
            s += 1.0
    placed = sum(isinstance(e, tuple(build.values())) for e in els)
    assert placed == requested, (
        f"{requested - placed} source(s) fell in a quadrupole rather than a drift: "
        "the free stretches of this fixture are (0.5, 1.5), (2.0, 3.0), ... "
        "-- a silently unplaced source would make every gate here vacuous"
    )
    return Lattice(els, ref)


# ==========================================================================
# 1. The two conventions O3 and O4 did not already pin: the new generators
# ==========================================================================


def test_octupole_generator_reproduces_the_shipped_kick(ref: ReferenceParticle) -> None:
    r"""``exp(:f:)`` with ``f = -k3l (x^4 - 6 x^2 y^2 + y^4)/24`` *is* ``ThinOctupole``.

    The Lie series terminates -- ``f`` depends on no momentum, so the second bracket
    vanishes -- which makes this an identity rather than a truncation, exactly as O3's
    sextupole check and O4's skew-quadrupole check are. Without it, every coefficient
    below would be the correct normal form of the wrong magnet.
    """
    k3l = sp.Rational(9, 5)
    f = -k3l * (o3._X**4 - 6 * o3._X**2 * o3._Y**2 + o3._Y**4) / 24
    got = o3._lie_map(f * o3._EPS, terms=4, order=1)
    want = [
        o3._X,
        o3._PX - k3l * (o3._X**3 - 3 * o3._X * o3._Y**2) / 6,
        o3._Y,
        o3._PY + k3l * (3 * o3._X**2 * o3._Y - o3._Y**3) / 6,
    ]
    assert o3._worst([e.subs(o3._EPS, 1) for e in got], want) < 1e-30

    state = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 1.0e-3, 2.0e-4])
    tracked = ThinOctupole(1.8).track(state.copy(), ref)
    sub = dict(zip(o3._Z, state[:4], strict=True))
    lie = [float(sp.N(e.subs(o3._EPS, 1).subs(sub))) for e in got]
    assert lie == pytest.approx(list(tracked[:4]), rel=0.0, abs=1e-18)


def test_skew_sextupole_generator_reproduces_the_shipped_kick(ref: ReferenceParticle) -> None:
    r"""``exp(:f:)`` with ``f = +k2sl (3 x^2 y - y^3)/6`` *is* ``ThinSkewSextupole``.

    The sign is the whole content: the normal sextupole's generator carries
    ``-Re[(x + iy)^3]`` and the skew one ``+Im[(x + iy)^3]``, and a package that got the
    relative sign wrong would still produce terms of the right magnitude on the right
    lines.
    """
    k2sl = sp.Rational(7, 4)
    f = k2sl * (3 * o3._X**2 * o3._Y - o3._Y**3) / 6
    got = o3._lie_map(f * o3._EPS, terms=4, order=1)
    want = [
        o3._X,
        o3._PX + k2sl * o3._X * o3._Y,
        o3._Y,
        o3._PY + k2sl * (o3._X**2 - o3._Y**2) / 2,
    ]
    assert o3._worst([e.subs(o3._EPS, 1) for e in got], want) < 1e-30

    state = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 1.0e-3, 2.0e-4])
    tracked = ThinSkewSextupole(1.75).track(state.copy(), ref)
    sub = dict(zip(o3._Z, state[:4], strict=True))
    lie = [float(sp.N(e.subs(o3._EPS, 1).subs(sub))) for e in got]
    assert lie == pytest.approx(list(tracked[:4]), rel=0.0, abs=1e-18)


# ==========================================================================
# 2. The derivation: read the generator O3 already builds, coefficient by coefficient
# ==========================================================================

_O, _T = sp.symbols("O T")  # k3l (normal octupole), k2sl (skew sextupole)
_BX, _BY = sp.symbols("beta_x beta_y", positive=True)
_A, _B = sp.symbols("A B")  # A = exp(2 pi i Q_x), B = exp(2 pi i Q_y)
_a, _b = sp.symbols("a b")  # a = exp(i mu_x), b = exp(i mu_y)


def _source_generators() -> dict[str, dict]:
    """The two sources' generators, referred to the reference point, in O3's basis."""
    xh, yh = o3._coord(_a, "x"), o3._coord(_b, "y")
    x2, y2 = o3._ppow(xh, 2), o3._ppow(yh, 2)
    oct_gen = o3._padd(
        o3._pscale(o3._ppow(xh, 4), -_O * _BX**2 / 24),
        o3._pscale(o3._pmul(x2, y2), +_O * _BX * _BY / 4),
        o3._pscale(o3._ppow(yh, 4), -_O * _BY**2 / 24),
    )
    skewsext = o3._padd(
        o3._pscale(o3._pmul(x2, yh), +_T * _BX * sp.sqrt(_BY) / 2),
        o3._pscale(o3._ppow(yh, 3), -_T * _BY ** sp.Rational(3, 2) / 6),
    )
    return {"oct": oct_gen, "skewsext": skewsext}


@pytest.fixture(scope="module")
def derived() -> dict[tuple[int, int, int, int], dict[str, sp.Expr]]:
    """``{monomial: {"oct"|"skewsext": coefficient}}`` of the normalising generator ``G``."""
    out: dict[tuple[int, int, int, int], dict[str, sp.Expr]] = {}
    for kind, F in _source_generators().items():
        G = o3._homological(o3._nonavg(o3._rot(F, 1 / _A, 1 / _B)), _A, _B)
        for mono, coeff in G.items():
            out.setdefault(mono, {})[kind] = sp.cancel(coeff)
    return out


def test_the_generator_is_the_one_the_normal_form_actually_solves() -> None:
    """``G`` from the fixture is not a lookalike: it is ``_normal_form``'s own first step.

    Feeding the same generator through O3's pipeline must leave nothing non-action at
    first order, which is the assertion ``_normal_form`` makes internally.
    """
    for F in _source_generators().values():
        first, _ = o3._normal_form([F], _A, _B, orders=1)
        assert all(m[0] == m[1] and m[2] == m[3] for m in first)


@pytest.mark.parametrize("key", KEYS)
def test_shipped_coefficient_is_an_exact_identity(key: str, derived: dict) -> None:
    """Each of the thirteen, as ``cancel(shipped - derived) == 0``. No tolerance anywhere.

    The shipped basis is ``h = u_hat + i p_hat``, O3's is ``h = u_hat - i p_hat``, so the
    two differ by complex conjugation. On unit-modulus symbols conjugation is inversion,
    so the comparison stays a rational-function identity.
    """
    mx, my, px, py, coef, kind = _RDT_TERMS[key]
    j, k, ell, m = (int(c) for c in key[1:])
    assert (j - k, ell - m) == (mx, my)  # the key's own indices fix the charge
    assert j + k + ell + m == (4 if kind == "oct" else 3)  # and its degree fixes the magnet

    strength = _O if kind == "oct" else _T
    shipped = (
        coef
        * strength
        * _BX ** sp.Rational(px)
        * _BY ** sp.Rational(py)
        * _a ** (-mx)
        * _b ** (-my)
        / (_A ** (-mx) * _B ** (-my) - 1)
    )
    conj = shipped.subs({_a: 1 / _a, _b: 1 / _b, _A: 1 / _A, _B: 1 / _B}, simultaneous=True)
    assert sp.cancel(sp.together(conj - derived[(j, k, ell, m)][kind])) == 0


def test_the_thirteen_are_every_first_order_term_these_two_magnets_drive(derived: dict) -> None:
    """Nothing is silently dropped: the monomials are these thirteen and their conjugates.

    ``F`` is real, so ``G_{kjml} = conj(G_{jklm})`` and half the monomials are redundant.
    If a magnet drove a fourteenth independent line it would appear here as a monomial
    whose partner is not among the keys.
    """
    shipped = {tuple(int(c) for c in k[1:]) for k in KEYS}
    for (j, k, ell, m), by_kind in derived.items():
        assert (j, k, ell, m) in shipped or (k, j, m, ell) in shipped, (j, k, ell, m, by_kind)
    for j, k, ell, m in shipped:
        assert (k, j, m, ell) not in shipped or (j, k, ell, m) == (k, j, m, ell)


def test_the_four_source_kinds_reach_four_disjoint_sets_of_monomials(derived: dict) -> None:
    """Degree separates the orders, vertical charge separates normal from skew.

    This is why one flat table can hold all twenty terms and why a ring with all four
    magnets needs no cross term. Stated as a property of the *derived* generators, not of
    the shipped table, so the table cannot satisfy it by construction.
    """
    for mono, by_kind in derived.items():
        assert len(by_kind) == 1, (mono, sorted(by_kind))  # no monomial takes both kinds
    for key, (_mx, my, _px, _py, _c, kind) in _RDT_TERMS.items():
        normal = kind in ("sext", "oct")
        assert (my % 2 == 0) is normal, (key, kind, my)


# ==========================================================================
# 3. The tie: what the octupole's normal form keeps, J2 already shipped
# ==========================================================================


def test_the_octupole_action_part_is_the_shipped_amplitude_detuning() -> None:
    r"""The generator's action-only half is :func:`amplitude_detuning`, exactly.

    The milestone's independent leg, and the only one here that shares no algebra with
    the derivation: J2 obtained ``dQx/dJx = +k3l bx^2/(16 pi)`` by averaging the kick over
    the betatron phase, not by solving a homological equation. Between them the two
    shipped functions account for the whole first-order generator with nothing left over
    -- the RDTs are the non-action part and the detuning is the action part.

    The relation is **measured before it is asserted**: the three second derivatives are
    read out of O3's normal form and only then compared with J2's closed forms.
    """
    F = _source_generators()["oct"]
    N1, _ = o3._normal_form([F], _A, _B, orders=1)
    Jx, Jy = sp.symbols("J_x J_y", positive=True)
    e = o3._to_actions(N1, Jx, Jy)
    got = {
        "xx": sp.cancel(-sp.diff(e, Jx, 2) / (2 * sp.pi)),
        "xy": sp.cancel(-sp.diff(sp.diff(e, Jx), Jy) / (2 * sp.pi)),
        "yy": sp.cancel(-sp.diff(e, Jy, 2) / (2 * sp.pi)),
    }
    want = {
        "xx": +_O * _BX**2 / (16 * sp.pi),
        "xy": -_O * _BX * _BY / (8 * sp.pi),
        "yy": +_O * _BY**2 / (16 * sp.pi),
    }
    for name in ("xx", "xy", "yy"):
        assert sp.simplify(got[name] - want[name]) == 0, (name, got[name])


def test_the_tie_holds_numerically_on_a_whole_ring(ref: ReferenceParticle) -> None:
    """The same statement on a real lattice, against the shipped J2 function.

    The symbolic identity above is per source at symbolic optics; this is the summed,
    numeric version, which is what would catch a walk that visited the octupoles
    differently in the two functions.
    """
    lat = _fodo(ref, OCTS)
    tw = closed_twiss(lat)
    sites, _, _ = _rdt_sites(lat, 32)
    k3l, bx, by, _mux, _muy = sites["oct"]
    assert k3l.size == len(OCTS)
    want = amplitude_detuning(lat)
    got = np.array(
        [
            [(k3l * bx**2).sum() / (16 * math.pi), -(k3l * bx * by).sum() / (8 * math.pi)],
            [-(k3l * bx * by).sum() / (8 * math.pi), (k3l * by**2).sum() / (16 * math.pi)],
        ]
    )
    assert got == pytest.approx(want, rel=1e-12)
    assert abs(tw.beta_x) > 0  # the ring is a real one, not a degenerate fixture


def test_a_skew_sextupole_has_no_action_part_and_so_no_such_tie() -> None:
    """It shifts no tune at first order, which is why it gets one reference leg fewer.

    Recorded as a gate rather than a remark because it is the reason the skew-sextupole
    half of this milestone rests on xtrack plus tracking, with neither MAD-X PTC (which
    does not expose odd-vertical-charge terms at all) nor a shipped detuning to check it.
    """
    F = _source_generators()["skewsext"]
    N1, _ = o3._normal_form([F], _A, _B, orders=1)
    assert not {m: c for m, c in N1.items() if sp.cancel(c) != 0}


# ==========================================================================
# 4. From one source to a ring: the shipped function against a re-derivation
# ==========================================================================


def _numeric_rdts(lat: Lattice) -> dict[str, complex]:
    """Re-run the symbolic pipeline numerically on a whole ring, source by source.

    This never sees ``_RDT_TERMS``: it builds each source's generator from the *element*,
    sums them, solves the homological equation and conjugates into the shipped basis.
    """
    sites, qx, qy = _rdt_sites(lat, 32)
    A, B = np.exp(2j * math.pi * qx), np.exp(2j * math.pi * qy)
    total: dict[tuple[int, int, int, int], complex] = {}
    for kind in ("oct", "skewsext"):
        strength, bx, by, mux, muy = sites[kind]
        for S, b_x, b_y, m_x, m_y in zip(strength, bx, by, mux, muy, strict=True):
            F = _source_generators()[kind]
            subs = {
                _O if kind == "oct" else _T: float(S),
                _BX: float(b_x),
                _BY: float(b_y),
                _a: complex(np.exp(1j * m_x)),
                _b: complex(np.exp(1j * m_y)),
            }
            for mono, coeff in F.items():
                total[mono] = total.get(mono, 0j) + complex(sp.N(coeff.subs(subs)))
    out: dict[str, complex] = {}
    for key in KEYS:
        j, k, ell, m = (int(c) for c in key[1:])
        if (j, k, ell, m) not in total:
            out[key] = 0j
            continue
        rot = total[(j, k, ell, m)] * A ** (-(j - k)) * B ** (-(ell - m))
        out[key] = complex(np.conj(rot / (1 - A ** (-(j - k)) * B ** (-(ell - m)))))
    return out


def test_shipped_function_is_the_normal_form_for_many_sources(ref: ReferenceParticle) -> None:
    """The generalisation step, checked for source counts up to eight of each kind."""
    rng = np.random.default_rng(20260831)
    where = [0.6, 1.2, 2.4, 5.5, 6.8, 8.4, 9.9, 11.6]
    for n in (1, 2, 3, 5, 8):
        oc = {p: float(w) for p, w in zip(where[:n], rng.uniform(-500, 500, n), strict=True)}
        ss = {p + 0.15: float(w) for p, w in zip(where[:n], rng.uniform(-5, 5, n), strict=True)}
        lat = _fodo(ref, oc, ss)
        got, want = resonance_driving_terms(lat), _numeric_rdts(lat)
        for key in KEYS:
            assert got[key] == pytest.approx(want[key], rel=1e-10), (n, key)


def test_a_ring_with_no_sources_drives_nothing(ref: ReferenceParticle) -> None:
    """And each magnet reaches only its own eight or five."""
    empty = resonance_driving_terms(_fodo(ref))
    assert all(empty[key] == 0.0 for key in KEYS)
    oct_only = resonance_driving_terms(_fodo(ref, OCTS))
    assert all(abs(oct_only[key]) > 0.0 for key in OCT_KEYS)
    assert all(oct_only[key] == 0.0 for key in SKEWSEXT_KEYS)
    ss_only = resonance_driving_terms(_fodo(ref, None, SKEWSEXTS))
    assert all(abs(ss_only[key]) > 0.0 for key in SKEWSEXT_KEYS)
    assert all(ss_only[key] == 0.0 for key in OCT_KEYS)


def test_all_four_source_kinds_in_one_ring_do_not_reach_each_other(ref: ReferenceParticle) -> None:
    """The disjointness of section 2, as a property of the shipped function.

    A ring carrying every source kind at once must return, for each of the twenty terms,
    exactly what the ring carrying only that term's own magnet returns. This is the gate
    that would fail if a source kind were summed into the wrong list.
    """
    sexts = {0.7: 1.0, 6.6: -0.7}
    skews = {1.1: 0.02, 4.1: -0.014}
    full = resonance_driving_terms(_fodo(ref, OCTS, SKEWSEXTS, sexts=sexts, skews=skews))
    alone = {
        "oct": resonance_driving_terms(_fodo(ref, OCTS)),
        "skewsext": resonance_driving_terms(_fodo(ref, None, SKEWSEXTS)),
        "sext": resonance_driving_terms(_fodo(ref, sexts=sexts)),
        "skew": resonance_driving_terms(_fodo(ref, skews=skews)),
    }
    for key, (_mx, _my, _px, _py, _c, kind) in _RDT_TERMS.items():
        assert full[key] == pytest.approx(alone[kind][key], rel=1e-12), key
        assert abs(full[key]) > 0.0, key  # and every one of the twenty is actually driven


def test_terms_are_linear_in_the_strength(ref: ReferenceParticle) -> None:
    """First order means first order: doubling every source doubles every term."""
    base = resonance_driving_terms(_fodo(ref, OCTS, SKEWSEXTS))
    for scale in (0.5, 2.0, -3.0):
        scaled = resonance_driving_terms(
            _fodo(
                ref,
                {p: scale * w for p, w in OCTS.items()},
                {p: scale * w for p, w in SKEWSEXTS.items()},
            )
        )
        for key in KEYS:
            assert scaled[key] == pytest.approx(scale * base[key], rel=1e-12), (scale, key)


def test_beta_exponents_are_measured_at_real_contrast(ref: ReferenceParticle) -> None:
    r"""The ``(p_x, p_y)`` powers, read off a single source moved between real betas.

    One source, one ring, several positions with different ``beta_x``/``beta_y``: the
    ratio of a term between two positions must be ``(bx2/bx1)^px (by2/by1)^py``, with the
    betas taken from the walk rather than from the table. The quartic block carries three
    distinct exponent pairs (``(2,0)``, ``(1,1)``, ``(0,2)``) and the skew-cubic block two
    (``(1, 1/2)``, ``(0, 3/2)``), so a term filed under a neighbouring pair cannot pass --
    which a magnitude check at a single position would never catch.
    """
    for kind, keys, strength in (("oct", OCT_KEYS, 400.0), ("skewsext", SKEWSEXT_KEYS, 3.0)):

        def ring(pos: float, _kind: str = kind, _s: float = strength) -> Lattice:
            return _fodo(ref, {pos: _s}) if _kind == "oct" else _fodo(ref, None, {pos: _s})

        lat_a = ring(5.5)  # the cell is 3 m, so positions 3 m apart share a beta: 2.4 and
        # 5.5 differ by only 4%, which is not contrast, and neither is 9.9. These four sit
        # where beta_x actually spans 6.26 to 8.83 against the base point's 7.47.
        sa, _, _ = _rdt_sites(lat_a, 32)
        bxa, bya = float(sa[kind][1][0]), float(sa[kind][2][0])
        fa = resonance_driving_terms(lat_a)
        seen: set[tuple[float, float]] = set()
        for pos in (0.6, 1.2, 1.4, 2.2):
            lat_b = ring(pos)
            sb, _, _ = _rdt_sites(lat_b, 32)
            bxb, byb = float(sb[kind][1][0]), float(sb[kind][2][0])
            fb = resonance_driving_terms(lat_b)
            assert abs(bxb / bxa - 1.0) > 0.05 or abs(byb / bya - 1.0) > 0.05, pos
            for key in keys:
                _mx, _my, px, py, _c, _k = _RDT_TERMS[key]
                want = (bxb / bxa) ** px * (byb / bya) ** py
                assert abs(fb[key]) / abs(fa[key]) == pytest.approx(want, rel=1e-10), (key, pos)
                seen.add((px, py))
        assert len(seen) >= 2, seen  # the contrast is between pairs, not within one


# ==========================================================================
# 5. Covariance: an RDT belongs to the ring *and* the point it is read at
# ==========================================================================


def _entrance_optics(lat: Lattice) -> list[tuple[float, float, float, float]]:
    """``(beta_x, beta_y, mu_x, mu_y)`` at the entrance of every element, coupling off."""
    dec = [_decoupled(e.matrix(lat.ref)) for e in lat.elements]
    one_turn = np.eye(6)
    for M in dec:
        one_turn = M @ one_turn
    tw0 = match_periodic(one_turn)
    bx, ax, mux = tw0.beta_x, tw0.alpha_x, 0.0
    by, ay, muy = tw0.beta_y, tw0.alpha_y, 0.0
    rows = []
    for M in dec:
        rows.append((bx, by, mux, muy))
        cx, cy = _blocks(M)
        bx, ax, dmux = _propagate_block(cx, bx, ax)
        by, ay, dmuy = _propagate_block(cy, by, ay)
        mux, muy = mux + dmux, muy + dmuy
    return rows


def test_moving_the_start_obeys_the_covariance_law(ref: ReferenceParticle) -> None:
    r"""``f_new = e^{+i (m_x d_x + m_y d_y)} (f_old + F_crossed)``, on the quartic terms.

    O4's sharpest gate, re-run on the new charges -- and sharper here, because this table
    carries charges up to ``4`` in a plane where the cubic one reached ``3``, so a wrong
    conjugation or a dropped phase is amplified rather than hidden. The shifts cross zero,
    one, two and more sources, which separates the rotation from the jump instead of
    measuring the two together.

    It is also the gate that would catch the two new kinds being routed to the wrong
    ``F_crossed``: an octupole stepped over must add its plain quartic coefficient to the
    octupole terms and nothing at all to the skew-sextupole ones.
    """
    lat = _fodo(ref, OCTS, SKEWSEXTS)
    rows = _entrance_optics(lat)
    src = []
    for i, e in enumerate(lat.elements):
        if isinstance(e, ThinOctupole) and e.k3l:
            src.append((i, "oct", e.k3l, *rows[i]))
        elif isinstance(e, ThinSkewSextupole) and e.k2sl:
            src.append((i, "skewsext", e.k2sl, *rows[i]))
    assert len(src) == len(OCTS) + len(SKEWSEXTS)
    base = resonance_driving_terms(lat)
    crossings = set()
    for shift in (3, 7, 14, len(lat.elements) // 2, len(lat.elements) - 1):
        rolled = Lattice(lat.elements[shift:] + lat.elements[:shift], ref)
        new = resonance_driving_terms(rolled)
        _, _, dx, dy = rows[shift]
        crossings.add(sum(1 for s in src if s[0] < shift))
        for key in KEYS:
            mx, my, px, py, coef, kind = _RDT_TERMS[key]
            crossed = sum(
                coef * S * b_x**px * b_y**py * np.exp(-1j * (mx * m_x + my * m_y))
                for (i, k, S, b_x, b_y, m_x, m_y) in src
                if k == kind and i < shift
            )
            want = np.exp(1j * (mx * dx + my * dy)) * (base[key] + crossed)
            assert new[key] == pytest.approx(want, rel=1e-11, abs=1e-13), (shift, key)
    assert {0, 1, 2}.issubset(crossings) and max(crossings) >= 4


def test_the_magnitude_alone_is_not_a_ring_invariant(ref: ReferenceParticle) -> None:
    """``|f|`` varies around the ring, so quoting one number as "the ring's f3100" is wrong.

    O4 measured this for the cubic terms and the quartic ones inherit it -- but the size
    of the variation is very unequal, which is worth recording because it is a trap. On
    this fixture ``f0030`` swings by ``7.5x`` and ``f3100`` by ``3.0x`` around the ring,
    while ``f2001`` moves by under one per cent. A term that happens to be nearly constant
    on one ring is *not* an invariant; it is a term whose sources happen to sit at phases
    that nearly cancel the jump, and the gate is written on the terms that show it clearly
    rather than on the ones that hide it.
    """
    lat = _fodo(ref, OCTS, SKEWSEXTS)
    els = lat.elements
    spread = {}
    for key in ("f4000", "f3100", "f2020", "f0040", "f2010", "f2001", "f1110", "f0030"):
        mags = [
            abs(resonance_driving_terms(Lattice(els[i:] + els[:i], lat.ref))[key])
            for i in range(0, len(els), 2)
        ]
        spread[key] = max(mags) / min(mags)
    assert spread["f0030"] > 3.0, spread
    assert spread["f3100"] > 2.0, spread
    assert all(v > 1.0 for v in spread.values()), spread


# 6. The resonances these terms are divided by
# ==========================================================================


def _tuned_ring(ref: ReferenceParticle, qx_target: float, octs=None, skewsexts=None) -> Lattice:
    """The fixture retuned so ``Q_x`` hits a requested value, sources unchanged."""
    from scipy.optimize import brentq

    def miss(kf: float) -> float:
        return tunes(_fodo(ref, octs, skewsexts, kf))[0] - qx_target

    # The bracket starts at 0.76, not lower: this cell goes unstable below kf ~ 0.75
    # (measured -- kf = 0.70 has no real matched beta), so a wider bracket makes brentq
    # raise UnstableLatticeError out of the fixture rather than fail the gate.
    kf = brentq(miss, 0.76, 1.10, xtol=1e-14)
    return _fodo(ref, octs, skewsexts, kf)


def test_only_the_4qx_term_blows_up_at_the_quarter_integer(ref: ReferenceParticle) -> None:
    """Approaching ``4 Q_x`` diverges ``f4000`` and leaves the ``2 Q_x`` terms finite.

    The charge is the resonance, so the divergence pattern is a direct read of whether
    each term is filed under the right line. ``f3100`` and ``f2011`` sit on ``2 Q_x`` and
    must *not* move much; ``f4000`` must grow like ``1/dQ``.
    """
    far = resonance_driving_terms(_tuned_ring(ref, 0.30, OCTS))
    grow = {}
    for dq in (1e-2, 1e-3):
        near = resonance_driving_terms(_tuned_ring(ref, 0.25 - dq / 4.0, OCTS))
        grow[dq] = {key: abs(near[key]) / abs(far[key]) for key in OCT_KEYS}
    assert grow[1e-3]["f4000"] / grow[1e-2]["f4000"] == pytest.approx(10.0, rel=0.15)
    assert grow[1e-3]["f4000"] > 30.0
    for key in ("f3100", "f2011"):
        assert grow[1e-3][key] / grow[1e-2][key] == pytest.approx(1.0, rel=0.5), key


def test_sitting_on_a_driven_line_raises_rather_than_inventing_a_number(
    ref: ReferenceParticle,
) -> None:
    """``Q_x = 1/4`` exactly is refused, and the message names the line."""
    lat = _tuned_ring(ref, 0.25, OCTS)
    with pytest.raises(ResonantLatticeError, match="Qx"):
        resonance_driving_terms(lat)


# ==========================================================================
# 7. What the sum refuses to guess at
# ==========================================================================


def _one_source_ring(ref: ReferenceParticle, source) -> Lattice:
    els = [Quadrupole(0.5, KF), Drift(0.5), source, Drift(0.5), Quadrupole(0.5, KD), Drift(1.0)]
    return Lattice(els * 4, ref)


@pytest.mark.parametrize(
    "cls,args",
    [(ThinOctupole, (400.0,)), (ThinSkewSextupole, (3.0,)), (Octupole, (0.2, 2000.0))],
)
@pytest.mark.parametrize(
    "misalignment", [{"dx": 1e-3}, {"dy": 1e-3}, {"roll": 0.05}], ids=["dx", "dy", "roll"]
)
def test_a_misaligned_source_is_refused_by_its_own_check(
    ref: ReferenceParticle, cls, args, misalignment
) -> None:
    """Refused, and for the right reason -- which is a different reason than O4's.

    A rolled octupole is a mixture of a normal and a skew octupole, so a type-walking sum
    reads a *wrong* strength rather than a missing one. An **offset** octupole is worse
    than that: it feeds down to a normal sextupole (``k2l = k3l x_co``) and to a skew one
    (``k2sl = k3l y_co``), and both of those are source kinds in this same sum, so the
    error would land on terms the function returns rather than on lines outside its list.
    The inherited coupling guard cannot see any of this: an octupole's linear map is a
    drift, so rolling or shifting it leaves no off-block to fire on.
    """
    with pytest.raises(CoupledLatticeError, match="driving-term source"):
        resonance_driving_terms(_one_source_ring(ref, cls(*args, **misalignment)))
    assert resonance_driving_terms(_one_source_ring(ref, cls(*args)))["f4000"] is not None


def test_the_coupling_guard_would_not_have_caught_an_offset_octupole(
    ref: ReferenceParticle,
) -> None:
    """Measured, so that "it is refused" is not confused with "it is refused correctly".

    O4 found the inherited guard firing on a ``1e-18`` off-block left by a rotation. Here
    the point is sharper still: an offset octupole leaves the linear map a *drift*, so
    there is no off-block at all and the guard is not merely weak but blind.
    """
    off = ThinOctupole(400.0, dx=1e-3)
    M = off.matrix(ref)
    from accsim.twiss import PX, PY, X, Y

    assert not M[np.ix_([X, PX], [Y, PY])].any()
    assert not M[np.ix_([Y, PY], [X, PX])].any()


# ==========================================================================
# 8. Thick bodies
# ==========================================================================


def _thick_ring(ref: ReferenceParticle, length: float, k3l: float) -> Lattice:
    """The fixture with the octupole at ``s = 5.5`` given a body of ``length``."""
    els: list = []
    s = 0.0
    for _ in range(4):
        for k in (KF, KD):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            if s < 5.5 < s + 1.0:
                lead = 5.5 - s - 0.5 * length
                els += [Drift(lead), Octupole(length, k3l / length), Drift(1.0 - lead - length)]
            else:
                els.append(Drift(1.0))
            s += 1.0
    return Lattice(els, ref)


def test_thick_sources_converge_on_the_thin_limit(ref: ReferenceParticle) -> None:
    """A body shrunk at fixed integrated strength reproduces the thin answer.

    An octupole's own linear map is a drift, exactly as a sextupole's is, so the slicing
    walk carries the optics across the body with half-drifts. That shared structure is the
    reason this test is short: what it checks is that the *new* kind is routed through the
    same path, not that the path is right.
    """
    thin = resonance_driving_terms(_fodo(ref, {5.5: 400.0}))["f4000"]
    errs = [
        abs(resonance_driving_terms(_thick_ring(ref, L, 400.0))["f4000"] - thin) / abs(thin)
        for L in (0.4, 0.2, 0.1)
    ]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-3


def test_thick_slicing_converges_at_second_order(ref: ReferenceParticle) -> None:
    """The midpoint rule's ``O(1/n^2)``, measured on the octupole body."""
    lat = _thick_ring(ref, 0.4, 400.0)
    fine = resonance_driving_terms(lat, slices=4096)["f4000"]
    errs = [abs(resonance_driving_terms(lat, slices=n)["f4000"] - fine) for n in (8, 16, 32, 64)]
    for a, b in zip(errs, errs[1:], strict=False):
        assert a / b == pytest.approx(4.0, rel=0.35), errs


# ==========================================================================
# 9. Tracking: the leg that shares no algebra with any of the above
# ==========================================================================

N_TURNS = 8192


def _turns(lat: Lattice, x: float, y: float = 1e-9, n: int = N_TURNS) -> np.ndarray:
    return Tracker(lat).track_turns(Particle(x=x, y=y), n, nonlinear=True)[:n]


def _line(h: np.ndarray, nu: float) -> complex:
    """Complex amplitude of the ``exp(-2 pi i nu n)`` component, Hann-windowed."""
    n = np.arange(h.size)
    w = 0.5 - 0.5 * np.cos(2.0 * math.pi * (n + 0.5) / h.size)
    return complex((w * h * np.exp(2j * math.pi * nu * n)).sum() / w.sum())


def _h(traj: np.ndarray, tw, plane: str) -> np.ndarray:
    """``h_u = u_hat + i p_hat_u`` from a tracked trajectory, in the shipped basis."""
    i, beta, alpha = (0, tw.beta_x, tw.alpha_x) if plane == "x" else (2, tw.beta_y, tw.alpha_y)
    u, pu = traj[:, i], traj[:, i + 1]
    return u / math.sqrt(beta) + 1j * (alpha * u / math.sqrt(beta) + math.sqrt(beta) * pu)


def _measured_qx(hx) -> float:
    """``Q_x`` of the tracked trajectory itself, not of the linear lattice.

    ``naff`` returns the dominant line of ``h_x``, which rotates as ``exp(-2 pi i Q_x n)``
    in the shipped basis, so the frequency it reports is ``1 - Q_x``. Why this is needed
    at all is the subject of ``test_the_line_must_be_read_at_the_amplitude_shifted_tune``.
    """
    return 1.0 - naff(hx)


def _measure_f4000(lat, tw, amp: float, n: int = N_TURNS) -> tuple[complex, float]:
    r"""``f4000`` off the ``-3 Q_x`` sideband of ``h_x``, and the tune it was read at.

    The same derivation as O4's ``f3000``, one power further. First-order perturbation
    puts ``-2i dG/dconj(zeta_x)`` into ``h_x``; the only monomial contributing
    ``conj(zeta_x)^3`` is ``(0,4,0,0)``, whose multiplier is its own exponent ``4`` where
    ``f3000``'s was ``3``. So where O4 divides by ``6i`` this divides by ``8i``:

        A(-3 Q_x) / conj(A(Q_x))^3 = 8 i conj(f4000).

    A ratio, so neither the action nor the launch phase survives into the comparison.
    """
    h = _h(_turns(lat, amp, n=n), tw, "x")
    qx = _measured_qx(h)
    got = np.conj(_line(h, -3.0 * qx) / (8j * np.conj(_line(h, qx)) ** 3))
    return complex(got), qx


@pytest.fixture(scope="module")
def tracked_f4000():
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    lat = _fodo(ref, OCTS)
    return lat, closed_twiss(lat), tunes(lat)[0], resonance_driving_terms(lat)["f4000"]


def test_the_line_must_be_read_at_the_amplitude_shifted_tune(tracked_f4000) -> None:
    r"""The trap this milestone has and the cubic one did not, stated as a gate.

    An octupole's **action** part and its **non-action** part come off the same generator,
    and they interfere in the measurement: the action part is the amplitude detuning,
    which moves ``Q_x`` away from the lattice's linear tune, and the sideband the
    non-action part sits on moves with it. At the launch used here the shift is only about
    ``2 * 10^-4`` in tune -- but the ``-3 Q_x`` line is *three times* as far off, and a
    Hann-windowed projection at the linear tune then reads leakage rather than the line.

    Measured: the raw ``-3 Q_x`` line comes out ``6 * 10^-5`` of its true amplitude, and
    the answer built from it lands at ``4.0`` against ``193`` -- **a factor of 48 low**.
    The two numbers differ because the *primary* line is mismeasured at the linear tune as
    well, and the formula is a ratio; the errors partly cancel and conspicuously do not
    cancel completely, so the surviving discrepancy is far too large to read as a
    tolerance and far too small to look like a structural bug. A sextupole has no
    first-order detuning, which is exactly why O4's tracked gate could use ``tunes(lat)``
    and this one cannot.
    """
    lat, tw, qx_linear, pred = tracked_f4000
    h = _h(_turns(lat, 1.5e-3), tw, "x")
    qx_measured = _measured_qx(h)
    assert abs(qx_measured - qx_linear) > 1e-4  # the detuning is real and resolved

    def read_at(q: float) -> complex:
        return complex(np.conj(_line(h, -3.0 * q) / (8j * np.conj(_line(h, q)) ** 3)))

    assert abs(read_at(qx_measured) - pred) / abs(pred) < 0.05
    assert abs(read_at(qx_linear)) / abs(pred) < 0.05  # measured: 0.021, a factor of 48


def test_tracked_f4000_matches_in_magnitude_and_phase(tracked_f4000) -> None:
    """The primary physics gate: a real trajectory, and the phase is half the content.

    O1's lesson, restated at quartic order -- a magnitude-only comparison passes with the
    wrong conjugation, and the sign of a measured sideband's phase is not a naming choice.
    Measured at a launch of ``1 mm``: ``1.0%`` in magnitude and ``5 * 10^-4`` radians in
    phase, with the conjugate basis wrong by ``74%``.
    """
    lat, tw, _qx_linear, pred = tracked_f4000
    got, _ = _measure_f4000(lat, tw, 1.0e-3)
    assert abs(got - pred) / abs(pred) < 0.02
    assert np.angle(got) == pytest.approx(np.angle(pred), abs=2e-3)
    assert abs(got - np.conj(pred)) / abs(pred) > 0.5


def test_the_tracked_residual_is_the_next_order_in_amplitude(tracked_f4000) -> None:
    """It falls as the action, which is what "first order" has to mean here.

    Measured, not predicted: halving the launch amplitude quarters the relative error
    (``3.8%``, ``0.98%``, ``0.25%`` at ``2``, ``1`` and ``0.5 mm``), i.e. the first
    correction is one power of action down. Turn count is scanned as a control -- the
    numbers are unchanged between ``8192`` and ``16384`` turns, so what is being measured
    is the physical next order and not spectral leakage.
    """
    lat, tw, _qx, pred = tracked_f4000
    errs = [abs(_measure_f4000(lat, tw, a)[0] - pred) / abs(pred) for a in (2e-3, 1e-3, 5e-4)]
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.2), errs
    assert errs[1] / errs[2] == pytest.approx(4.0, rel=0.2), errs
    longer = abs(_measure_f4000(lat, tw, 1e-3, 16384)[0] - pred) / abs(pred)
    assert longer == pytest.approx(errs[1], rel=0.05), (longer, errs[1])


def test_the_exact_drift_drives_this_line_too_but_far_below(ref: ReferenceParticle) -> None:
    r"""The background O4 did not have -- present, measurable, and five orders down.

    accsim's drift is exact, so its map is nonlinear at *quartic* order,
    ``-L (px^2 + py^2)^2 / 8``, and ``px^4`` reaches the very ``4 Q_x`` line ``f4000``
    sits on. O3 met the same fact as a detuning that exists with no magnets in the ring.
    It is **not** a defect in the reported number: like :func:`sextupole_detuning`, this
    function reports the magnets' contribution and correctly returns exactly zero here.

    What had to be measured is how big the floor is, because the tracked gate above would
    silently be reading the lattice if it were not small. Put through the same formula, a
    ring of drifts and quadrupoles yields an apparent ``|f4000|`` of ``1.6 * 10^-3``
    against the octupoles' ``193`` -- a part in ``10^5``.
    """
    bare = _fodo(ref)
    tw = closed_twiss(bare)
    assert resonance_driving_terms(bare)["f4000"] == 0.0
    floor, _ = _measure_f4000(bare, tw, 1.5e-3)
    lat = _fodo(ref, OCTS)
    real = abs(resonance_driving_terms(lat)["f4000"])
    assert abs(floor) / real < 1e-4, (abs(floor), real)
    assert abs(floor) > 0.0  # present, not absent -- the drift really does drive it


def test_tracked_skew_sextupole_terms_match_on_the_two_2qx_lines(ref: ReferenceParticle) -> None:
    r"""``f2001`` and ``f2010`` off the ``+-2 Q_x`` lines of ``h_y``, phase included.

    The vertical mirror of O4's ``f1001``/``f1010`` pair, one order up: a skew sextupole
    puts vertical motion at **twice** the horizontal tune, where a skew quadrupole puts it
    at the horizontal tune itself. The monomial ``(2,0,0,1)`` is the only one placing
    ``zeta_x^2`` into ``h_y`` and ``(0,2,0,1)`` the only one placing ``conj(zeta_x)^2``
    there, so

        f2001 = A_y(2 Q_x) / (2 i A_x(Q_x)^2),
        f2010 = conj[ A_y(-2 Q_x) / (2 i conj(A_x(Q_x))^2) ] .

    Reading both off the *same* trajectory is what makes the pair a real test of the sign
    of ``m_y``: the two differ only in that sign, and swapping them moves each answer onto
    the other's line. The launch is purely horizontal -- the vertical motion being read is
    entirely what the skew sextupoles put there.

    The residual is **cubic** in the skew strength, the same exponent and the same cause
    as O4's coupling terms: the relative error falls by four per halving (``0.52%``,
    ``0.13%``, ``0.033%``), because what perturbs the answer is the optics the formula is
    evaluated on, whose own shift is quadratic.
    """
    prev: dict[str, float] = {}
    ratios: dict[str, list[float]] = {"f2001": [], "f2010": []}
    for scale in (1.0, 0.5, 0.25):
        lat = _fodo(ref, None, {p: scale * w for p, w in SKEWSEXTS.items()})
        tw = closed_twiss(lat)
        traj = _turns(lat, 1.0e-3)
        hx, hy = _h(traj, tw, "x"), _h(traj, tw, "y")
        qx = _measured_qx(hx)
        ax = _line(hx, qx)
        got = {
            "f2001": _line(hy, 2.0 * qx) / (2j * ax**2),
            "f2010": np.conj(_line(hy, -2.0 * qx) / (2j * np.conj(ax) ** 2)),
        }
        pred = resonance_driving_terms(lat)
        for key in ("f2001", "f2010"):
            rel = abs(got[key] - pred[key]) / abs(pred[key])
            if scale == 1.0:
                assert rel < 1e-2, (key, rel)
                assert np.angle(got[key]) == pytest.approx(np.angle(pred[key]), abs=1e-3), key
                assert abs(got[key] - np.conj(pred[key])) / abs(pred[key]) > 0.1, key
            if key in prev:
                ratios[key].append(prev[key] / rel)
            prev[key] = rel
    for key, seen in ratios.items():
        assert all(r == pytest.approx(4.0, rel=0.15) for r in seen), (key, seen)


def test_the_two_skew_sextupole_terms_are_not_the_same_number(ref: ReferenceParticle) -> None:
    """``2 Q_x + Q_y`` and ``2 Q_x - Q_y`` differ only in the sign of ``m_y``.

    O4's separation gate, re-run: a formula with that sign wrong in both places
    consistently would pass every comparison above. The separation is not asserted from a
    guess about which is larger but from the *ordering flipping* between working points --
    ``|f2010|/|f2001|`` measures ``0.027`` low in the cell and ``1.85`` at the fixture's
    own point -- which one number computed twice cannot do.
    """
    low = resonance_driving_terms(_fodo(ref, None, SKEWSEXTS, 0.70, -0.80))
    high = resonance_driving_terms(_fodo(ref, None, SKEWSEXTS))
    lo = abs(low["f2010"]) / abs(low["f2001"])
    hi = abs(high["f2010"]) / abs(high["f2001"])
    assert lo < 0.5 and hi > 1.5, (lo, hi)
