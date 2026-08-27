r"""O4: the first-order resonance driving terms, derived from O3's own machinery.

O3 built the normal form of the one-turn map and kept the part that depends only on the
actions -- the tune shift. This milestone keeps what that step *throws away*. Removing a
non-action monomial from the map costs one Lie generator, and the coefficient of that
generator is the resonance driving term. So O4 is not a new calculation: it is the
intermediate quantity ``_homological`` already computes inside
``test_sextupole_detuning``, read out instead of discarded. This file imports that
machinery rather than restating it, which is what makes the conventions it pinned --
the generator, the ``+mu`` conjugation, the resonance basis, the bracket -- carry over
for free.

**What is new here, and therefore pinned before it is used.**

1. **The skew-quadrupole generator.** ``f = k1sl x y``, checked against
   ``ThinSkewQuadrupole``'s own map exactly, the way O3 checked the sextupole's.
2. **The basis.** O3 works in ``h_u = u_hat - i p_hat_u``; the shipped RDTs are quoted
   in the opposite basis ``h_u = u_hat + i p_hat_u``, which is xtrack's and MAD-X's, so
   every term is the complex conjugate of O3's generator coefficient. That is a naming
   choice, not physics, and it is not settled by fiat: **tracking** decides it, because
   the sign of a measured sideband's phase is not a convention.

**The four legs, in the order they were built.**

- *Symbolic.* Each of the seven shipped coefficients is checked as an **exact identity**
  in unit-modulus symbols against ``_homological`` applied to the derived generator --
  ``cancel(...) == 0``, not a tolerance.
- *G1.* ``closest_tune_approach`` (``|C^-|``, shipped since G1, derived from the exact
  eigen-tune split -- a completely different route that shares no algebra with this one)
  fixes ``f1001``'s magnitude. The tie is not asserted from a remembered prefactor: the
  ratio is **measured** on two structurally different rings, found to be the same
  constant, and only then pinned.
- *Tracking.* Each RDT is a named sideband of the turn-by-turn spectrum. ``f3000`` is
  read off the ``-2 Q_x`` line of ``h_x``, ``f1001`` off the ``Q_x`` line of ``h_y``, in
  both cases **magnitude and phase**, and in both cases through a ratio that cancels the
  launch phase and the action so nothing about the launch can be tuned to fit.
- *Structure.* An RDT is covariant, not invariant: rolling the lattice must move it by
  the exact law ``f_new = e^{+i m.d} (f_old + F_crossed)``, which no wrong conjugation
  or missing phase survives.
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
    Quadrupole,
    ReferenceParticle,
    Sextupole,
    SkewQuadrupole,
    ThinSextupole,
    ThinSkewQuadrupole,
    closest_tune_approach,
    resonance_driving_terms,
    tunes,
)
from accsim.tracking import Particle, Tracker
from accsim.twiss import (
    _RDT_TERMS,
    CoupledLatticeError,
    ResonantLatticeError,
    _blocks,
    _decoupled,
    _propagate_block,
    _rdt_sites,
    closed_twiss,
    match_periodic,
)

sys.path.insert(0, os.path.dirname(__file__))

import test_sextupole_detuning as o3  # noqa: E402

MASS0, GAMMA0 = 938.27208816e6, 20.0

#: The generic working point. ``Q_x != Q_y`` so ``Q_x +- 2 Q_y`` does not collapse onto
#: ``3 Q_x``/``Q_x``, and every driven line sits well away from an integer.
KF, KD = 0.80, -0.90

#: Three sextupoles at different beta and different phase, with a sign change among the
#: weights so no accidental symmetry can hide a wrong term.
SEXTS = {5.5: 1.0, 8.4: -0.7, 11.6: 0.45}

#: Three skew quadrupoles, likewise.
SKEWS = {2.4: 0.02, 6.8: -0.014, 9.9: 0.031}

KEYS = tuple(_RDT_TERMS)


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _fodo(
    ref: ReferenceParticle,
    sexts: dict[float, float] | None = None,
    skews: dict[float, float] | None = None,
    kf: float = KF,
    kd: float = KD,
) -> Lattice:
    """``Q(kf) D Q(kd) D`` four times, 12 m, with thin sources dropped in at ``s``.

    Bend-free on purpose: no dispersion at the sources, so no feed-down, which this
    milestone does not model and which would otherwise contaminate every comparison.
    """
    marks: dict[float, tuple[str, float]] = {}
    marks.update({p: ("s", w) for p, w in (sexts or {}).items()})
    marks.update({p: ("k", w) for p, w in (skews or {}).items()})
    els: list = []
    s = 0.0
    for _ in range(4):
        for k in (kf, kd):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            done = 0.0
            for p in sorted(q for q in marks if s < q < s + 1.0):
                kind, w = marks[p]
                els.append(Drift(p - s - done))
                els.append(ThinSextupole(w) if kind == "s" else ThinSkewQuadrupole(w))
                done = p - s
            els.append(Drift(1.0 - done))
            s += 1.0
    placed = sum(isinstance(e, (ThinSextupole, ThinSkewQuadrupole)) for e in els)
    assert placed == len(marks), (
        f"{len(marks) - placed} source(s) fell in a quadrupole rather than a drift: "
        "the free stretches of this fixture are (0.5, 1.5), (2.0, 3.0), ... "
        "-- a silently unplaced source would make every gate here vacuous"
    )
    return Lattice(els, ref)


def _bare(lat: Lattice) -> Lattice:
    """The same ring with the skew quadrupoles removed.

    A thin skew quadrupole is the only coupling source in these fixtures, so dropping it
    leaves the *decoupled* linear optics untouched -- which is what first-order
    perturbation theory uses, and what ``closed_twiss``/``tunes`` are able to see.
    """
    return Lattice([e for e in lat.elements if not isinstance(e, ThinSkewQuadrupole)], lat.ref)


# ==========================================================================
# 1. The one convention O3 did not already pin: the skew-quadrupole generator
# ==========================================================================


def test_skew_quadrupole_generator_reproduces_the_shipped_kick(ref: ReferenceParticle) -> None:
    """``exp(:f:)`` with ``f = k1sl x y`` *is* ``ThinSkewQuadrupole``'s map, exactly.

    The Lie series terminates -- ``f`` depends on no momentum, so the second bracket
    vanishes -- which makes this an identity rather than a truncation, exactly as O3's
    sextupole check is. The opposite sign is decisively wrong, not merely worse.
    """
    k1sl = sp.Rational(7, 5)
    f = k1sl * o3._X * o3._Y
    got = o3._lie_map(f * o3._EPS, terms=4, order=1)
    want = [o3._X, o3._PX + k1sl * o3._Y, o3._Y, o3._PY + k1sl * o3._X]
    assert o3._worst([e.subs(o3._EPS, 1) for e in got], want) < 1e-30

    # and numerically against the element itself
    state = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 1.0e-3, 2.0e-4])
    tracked = ThinSkewQuadrupole(1.4).track(state.copy(), ref)
    sub = dict(zip(o3._Z, state[:4], strict=True))
    lie = [float(sp.N(e.subs(o3._EPS, 1).subs(sub))) for e in got]
    assert lie == pytest.approx(list(tracked[:4]), rel=0.0, abs=1e-18)

    wrong = o3._lie_map(-f * o3._EPS, terms=4, order=1)
    bad = [float(sp.N(e.subs(o3._EPS, 1).subs(sub))) for e in wrong]
    assert abs(bad[1] - tracked[1]) > 1e-4  # twice the kick, not a rounding difference


# ==========================================================================
# 2. The derivation: read the generator O3 already builds, coefficient by coefficient
# ==========================================================================

_S, _K = sp.symbols("S K")
_BX, _BY = sp.symbols("beta_x beta_y", positive=True)
_A, _B = sp.symbols("A B")  # A = exp(2 pi i Q_x), B = exp(2 pi i Q_y)
_a, _b = sp.symbols("a b")  # a = exp(i mu_x), b = exp(i mu_y)


def _source_generators() -> dict[str, dict]:
    """The two sources' generators, referred to the reference point, in O3's basis."""
    xh, yh = o3._coord(_a, "x"), o3._coord(_b, "y")
    sext = o3._padd(
        o3._pscale(o3._ppow(xh, 3), -_S * _BX ** sp.Rational(3, 2) / 6),
        o3._pscale(o3._pmul(xh, o3._ppow(yh, 2)), +_S * sp.sqrt(_BX) * _BY / 2),
    )
    skew = o3._pscale(o3._pmul(xh, yh), _K * sp.sqrt(_BX * _BY))
    return {"sext": sext, "skew": skew}


@pytest.fixture(scope="module")
def derived() -> dict[tuple[int, int, int, int], dict[str, sp.Expr]]:
    """``{monomial: {"sext"|"skew": coefficient}}`` of the normalising generator ``G``.

    ``G`` is exactly what ``_normal_form`` computes on its way to the detuning and then
    never returns: the solution of the homological equation that kills the non-action
    part at first order. Its coefficients *are* the RDTs.
    """
    out: dict[tuple[int, int, int, int], dict[str, sp.Expr]] = {}
    for kind, F in _source_generators().items():
        G = o3._homological(o3._nonavg(o3._rot(F, 1 / _A, 1 / _B)), _A, _B)
        for mono, coeff in G.items():
            out.setdefault(mono, {})[kind] = sp.cancel(coeff)
    return out


def test_the_generator_is_the_one_the_normal_form_actually_solves() -> None:
    """``G`` from the fixture is not a lookalike: it is ``_normal_form``'s own first step.

    Feeding the same generator through O3's pipeline must leave nothing non-action at
    first order -- which is the assertion ``_normal_form`` makes internally. If the two
    had drifted apart, every coefficient below would be checking a different object from
    the one the shipped detuning is built on.
    """
    F = _source_generators()["sext"]
    first, _ = o3._normal_form([F], _A, _B, orders=1)
    assert all(m[0] == m[1] and m[2] == m[3] for m in first)


@pytest.mark.parametrize("key", KEYS)
def test_shipped_coefficient_is_an_exact_identity(key: str, derived: dict) -> None:
    """Each of the seven, as ``cancel(shipped - derived) == 0``. No tolerance anywhere.

    The shipped basis is ``h = u_hat + i p_hat``, O3's is ``h = u_hat - i p_hat``, so the
    two differ by complex conjugation. On unit-modulus symbols conjugation is inversion
    (``conj(a) = 1/a``), so the comparison is still a rational-function identity: the
    shipped closed form, with every phase symbol inverted, must equal O3's coefficient.
    """
    mx, my, px, py, coef, kind = _RDT_TERMS[key]
    j, k, ell, m = (int(c) for c in key[1:])
    assert (j - k, ell - m) == (mx, my)  # the key's own indices fix the charge

    strength = _S if kind == "sext" else _K
    # the shipped formula, written in symbols:  coef * S * bx^px * by^py * E / (D - 1)
    shipped = (
        coef
        * strength
        * _BX ** sp.Rational(px)
        * _BY ** sp.Rational(py)
        * _a ** (-mx)
        * _b ** (-my)
        / (_A ** (-mx) * _B ** (-my) - 1)
    )
    # conjugate it (invert every unit-modulus symbol) and compare with O3's G
    conj = shipped.subs({_a: 1 / _a, _b: 1 / _b, _A: 1 / _A, _B: 1 / _B}, simultaneous=True)
    assert sp.cancel(sp.together(conj - derived[(j, k, ell, m)][kind])) == 0


def test_the_seven_are_every_first_order_term_these_two_magnets_drive(derived: dict) -> None:
    """Nothing is silently dropped: the ten + four monomials are these seven, conjugated.

    ``F`` is a real function, so ``G_{kjml} = conj(G_{jklm})`` and half the monomials are
    redundant. This asserts that the redundancy is *exactly* the shipped list -- if a
    magnet drove an eighth independent line, it would show up here as a monomial whose
    partner is not among the keys.
    """
    shipped = {tuple(int(c) for c in k[1:]) for k in KEYS}
    for (j, k, ell, m), by_kind in derived.items():
        partner = (k, j, m, ell)
        assert (j, k, ell, m) in shipped or partner in shipped, (j, k, ell, m, by_kind)
    # and no key is a conjugate of another key -- the list carries no duplicates
    for j, k, ell, m in shipped:
        assert (k, j, m, ell) not in shipped or (j, k, ell, m) == (k, j, m, ell)


# ==========================================================================
# 3. From one source to a ring: the shipped function against a numeric re-derivation
# ==========================================================================


def _numeric_rdts(lat: Lattice) -> dict[str, complex]:
    """Re-run the symbolic pipeline numerically on a whole ring, source by source.

    This never sees ``_RDT_TERMS``: it builds each source's generator from the *element*,
    sums them, solves the homological equation and conjugates into the shipped basis. It
    is the check that the per-source closed form generalises to a sum, which the
    derivation asserts rather than proves.
    """
    sites, qx, qy = _rdt_sites(lat, 32)
    A, B = np.exp(2j * math.pi * qx), np.exp(2j * math.pi * qy)
    total: dict[tuple[int, int, int, int], complex] = {}
    for kind, (strength, bx, by, mux, muy) in sites.items():
        for S, b_x, b_y, m_x, m_y in zip(strength, bx, by, mux, muy, strict=True):
            F = _source_generators()[kind]
            subs = {
                _S if kind == "sext" else _K: float(S),
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
        mono = (j, k, ell, m)
        if mono not in total:
            out[key] = 0j
            continue
        rot = total[mono] * A ** (-(j - k)) * B ** (-(ell - m))
        g = rot / (1 - A ** (-(j - k)) * B ** (-(ell - m)))
        out[key] = complex(np.conj(g))  # into the shipped basis
    return out


def test_shipped_function_is_the_normal_form_for_many_sources(ref: ReferenceParticle) -> None:
    """The generalisation step, checked for source counts up to eight of each kind."""
    rng = np.random.default_rng(20260827)
    where = [0.6, 1.2, 2.4, 5.5, 6.8, 8.4, 9.9, 11.6]
    for n in (1, 2, 3, 5, 8):
        sx = {p: float(w) for p, w in zip(where[:n], rng.uniform(-1.5, 1.5, n), strict=True)}
        sk = {
            p + 0.15: float(w) for p, w in zip(where[:n], rng.uniform(-0.03, 0.03, n), strict=True)
        }
        lat = _fodo(ref, sx, sk)
        got, want = resonance_driving_terms(lat), _numeric_rdts(lat)
        for key in KEYS:
            assert got[key] == pytest.approx(want[key], rel=1e-11, abs=0.0), key


def test_a_ring_with_no_sources_drives_nothing(ref: ReferenceParticle) -> None:
    """And each magnet drives only its own lines -- a cross-contamination check."""
    empty = resonance_driving_terms(_fodo(ref))
    assert all(v == 0.0 for v in empty.values())
    sext_only = resonance_driving_terms(_fodo(ref, SEXTS))
    assert sext_only["f1001"] == 0.0 and sext_only["f1010"] == 0.0
    assert all(sext_only[k] != 0.0 for k in ("f3000", "f2100", "f1020", "f1011", "f1002"))
    skew_only = resonance_driving_terms(_fodo(ref, None, SKEWS))
    assert all(skew_only[k] == 0.0 for k in ("f3000", "f2100", "f1020", "f1011", "f1002"))
    assert skew_only["f1001"] != 0.0 and skew_only["f1010"] != 0.0


def test_terms_are_linear_in_the_strength(ref: ReferenceParticle) -> None:
    """First order means first order: doubling every source doubles every term, exactly.

    This is the cheapest possible separation from O3's detuning, which is *quadratic* in
    ``k2`` -- a formula that had picked up a second-order piece would fail here at once.
    """
    base = resonance_driving_terms(_fodo(ref, SEXTS, SKEWS))
    for factor in (2.0, 0.5, -3.0):
        scaled = resonance_driving_terms(
            _fodo(
                ref,
                {p: factor * w for p, w in SEXTS.items()},
                {p: factor * w for p, w in SKEWS.items()},
            )
        )
        for key in KEYS:
            assert scaled[key] == pytest.approx(factor * base[key], rel=1e-12, abs=0.0), key


def test_beta_exponents_are_measured_at_real_contrast(ref: ReferenceParticle) -> None:
    """``beta_x^(3/2)`` and ``beta_x^(1/2) beta_y`` -- with the neighbours excluded.

    O3's lesson, applied before it could bite again: a scaling gate is only as sharp as
    the contrast it is run at. Its first attempt compared two positions whose ``beta_x``
    differed by 4 %, where ``beta^2`` and ``beta^3`` are indistinguishable. Here a single
    sextupole is moved between the *minimum* and the *maximum* ``beta_x`` the ring
    offers, and the gate asserts both that the measured ratio matches ``beta_x^(3/2)``
    and that it does **not** match ``beta_x^1`` or ``beta_x^2``.
    """
    where = [0.6, 0.9, 1.2, 1.4, 2.1, 2.5, 2.9, 3.6, 4.0, 4.4]
    rows = []
    for pos in where:
        lat = _fodo(ref, {pos: 1.0})
        sites, _, _ = _rdt_sites(lat, 32)
        _, bx, by, _, _ = sites["sext"]
        f = resonance_driving_terms(lat)
        rows.append((float(bx[0]), float(by[0]), abs(f["f3000"]), abs(f["f1011"])))
    lo = min(rows, key=lambda r: r[0])
    hi = max(rows, key=lambda r: r[0])
    contrast = hi[0] / lo[0]
    assert contrast > 1.3, f"the fixture must offer real beta contrast, got {contrast}"

    ratio = hi[2] / lo[2]
    assert ratio == pytest.approx(contrast**1.5, rel=1e-9)
    for wrong in (1.0, 2.0):
        assert abs(ratio / contrast**wrong - 1.0) > 0.1, wrong

    # the cross-plane weighting is beta_x^(1/2) beta_y, and beta_y is really in there
    cross = hi[3] / lo[3]
    assert cross == pytest.approx(math.sqrt(contrast) * (hi[1] / lo[1]), rel=1e-9)
    assert abs(cross / math.sqrt(contrast) - 1.0) > 0.1


# ==========================================================================
# 4. G1: an independent derivation already in the package fixes f1001
# ==========================================================================


def test_f1001_against_the_shipped_closest_tune_approach(ref: ReferenceParticle) -> None:
    r"""``|f1001| * 4 |sin(pi (Q_x - Q_y))| == pi |C^-|`` -- constant first, then pinned.

    ``closest_tune_approach`` has been in the package since G1 and was derived from the
    exact eigen-tune split of a single skew kick, sharing no algebra with the Lie
    machinery above. It knows only the *modulus*, which makes it a magnitude leg and not
    a phase one -- but the magnitude is where an overall factor of two or of ``2 pi``
    would hide, and that is exactly what the basis choice risks.

    The prefactor is **measured before it is asserted**: the ratio is computed on two
    rings with different tunes, different skew positions and different strengths, and
    the assertion is first that those agree with each other. Only then is the common
    value pinned -- and it comes out ``pi`` to round-off, at every strength, because both
    sides use the same unperturbed optics and neither is an approximation of the other.
    """
    measured = []
    for kf, kd, skews in (
        (KF, KD, SKEWS),
        (1.05, -1.15, {2.9: 0.03, 6.8: 0.011, 11.6: -0.02}),
        (KF, KD, {5.5: 1e-4}),
    ):
        lat = _fodo(ref, None, skews, kf, kd)
        qx, qy = tunes(_bare(lat))
        f1001 = resonance_driving_terms(lat)["f1001"]
        gap = closest_tune_approach(lat)
        measured.append(abs(f1001) * 4.0 * abs(math.sin(math.pi * (qx - qy))) / gap)
    assert measured[1] == pytest.approx(measured[0], rel=1e-12)
    assert measured[2] == pytest.approx(measured[0], rel=1e-12)
    assert measured[0] == pytest.approx(math.pi, rel=1e-12)


# ==========================================================================
# 5. Structure: an RDT is covariant, not invariant
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
    r"""``f_new = e^{+i (m_x d_x + m_y d_y)} (f_old + F_crossed)``, to round-off.

    This is the sharp structural gate, and it is sharp because it is *not* an
    invariance. O3's detuning is a property of the ring and does not move at all when the
    start does; an RDT is a property of the ring **and the observation point**. Each term
    rotates by its own charge times the phase advance stepped over, and jumps by the
    plain (undivided) contribution of every source crossed. A wrong conjugation, a wrong
    denominator or a missing phase all reproduce a single-point comparison and all fail
    here -- and the shifts below are chosen to cross zero, one, two and five sources, so
    the rotation and the jump are separated rather than measured together.
    """
    lat = _fodo(ref, SEXTS, SKEWS)
    rows = _entrance_optics(lat)
    src = []
    for i, e in enumerate(lat.elements):
        if isinstance(e, ThinSextupole) and e.k2l:
            src.append((i, "sext", e.k2l, *rows[i]))
        elif isinstance(e, ThinSkewQuadrupole) and e.k1sl:
            src.append((i, "skew", e.k1sl, *rows[i]))
    base = resonance_driving_terms(lat)
    crossings = set()
    for shift in (3, 7, 14, len(lat.elements) // 2, len(lat.elements) - 1):
        rolled = Lattice(lat.elements[shift:] + lat.elements[:shift], ref)
        new = resonance_driving_terms(rolled)
        _, _, dx, dy = rows[shift]
        crossings.add(sum(1 for s in src if s[0] < shift))
        for key, (mx, my, px, py, coef, kind) in _RDT_TERMS.items():
            crossed = sum(
                coef * S * b_x**px * b_y**py * np.exp(-1j * (mx * m_x + my * m_y))
                for (i, k, S, b_x, b_y, m_x, m_y) in src
                if k == kind and i < shift
            )
            want = np.exp(1j * (mx * dx + my * dy)) * (base[key] + crossed)
            assert new[key] == pytest.approx(want, rel=1e-11, abs=1e-13), (shift, key)
    assert {0, 1, 2}.issubset(crossings) and max(crossings) >= 5


def test_the_magnitude_alone_is_not_a_ring_invariant(ref: ReferenceParticle) -> None:
    """The flip side, stated as a gate rather than left implicit in the docstring.

    Between sources ``|f|`` is constant and only the phase turns; across a source it
    jumps. Reporting ``|f3000|`` at one point and calling it "the ring's ``f3000``" is
    therefore wrong, and this pins the size of that error on the fixture -- large.
    """
    lat = _fodo(ref, SEXTS)
    els = lat.elements
    mags = []
    for shift in range(0, len(els), 3):
        rolled = Lattice(els[shift:] + els[:shift], ref)
        mags.append(abs(resonance_driving_terms(rolled)["f3000"]))
    assert max(mags) / min(mags) > 1.5


# ==========================================================================
# 6. The denominators, identified by which term diverges
# ==========================================================================


def _tuned_ring(ref: ReferenceParticle, qx_target: float, sexts, skews=None) -> Lattice:
    """A fixture whose horizontal tune is driven to ``qx_target`` by scanning ``kf``."""
    from scipy.optimize import brentq

    def miss(kf: float) -> float:
        return tunes(_fodo(ref, None, None, kf, KD))[0] - qx_target

    kf = brentq(miss, 0.76, 1.19, xtol=1e-14)  # the stable window of this fixture
    return _fodo(ref, sexts, skews, kf, KD)


def test_only_the_3qx_term_blows_up_at_the_third_integer(ref: ReferenceParticle) -> None:
    """The denominator is a per-term fingerprint, and it is checked as one.

    Approaching ``Q_x = 1/3``, ``f3000`` must diverge like ``1/|sin(3 pi Q_x)|`` and the
    other sextupole terms -- which carry ``Q_x``, not ``3 Q_x`` -- must barely move. That
    is the check that each coefficient was divided by *its own* line and not by a shared
    one, which no single-ring comparison can see.
    """
    far = resonance_driving_terms(_tuned_ring(ref, 0.28, SEXTS))
    grew = {}
    for dq in (4.0e-3, 1.0e-3, 2.5e-4):
        near = resonance_driving_terms(_tuned_ring(ref, 1.0 / 3.0 - dq, SEXTS))
        grew[dq] = abs(near["f3000"]) / abs(far["f3000"])
        for key in ("f2100", "f1011"):
            assert abs(near[key]) / abs(far[key]) < 3.0, key
    assert grew[4.0e-3] > 5.0
    # a factor-four step in the distance is a factor-four step in the divergence
    assert grew[2.5e-4] / grew[1.0e-3] == pytest.approx(4.0, rel=0.05)
    assert grew[1.0e-3] / grew[4.0e-3] == pytest.approx(4.0, rel=0.05)


def test_sitting_on_a_driven_line_raises_rather_than_inventing_a_number(
    ref: ReferenceParticle,
) -> None:
    """``ResonantLatticeError``, with the offending line named in the message."""
    lat = _tuned_ring(ref, 1.0 / 3.0, SEXTS)
    qx, _ = tunes(lat)
    assert abs(qx - 1.0 / 3.0) < 1e-13
    with pytest.raises(ResonantLatticeError, match="3 Qx"):
        resonance_driving_terms(lat)


def test_a_coupling_source_this_sum_cannot_see_is_refused(ref: ReferenceParticle) -> None:
    """G1's guard, inherited: a rolled quadrupole would silently zero f1001/f1010.

    It is a coupling source that is not a skew quadrupole, so a type-walking sum reports
    ``0`` for a ring that is demonstrably coupled. Measured, not asserted by type.
    """
    lat = _fodo(ref, SEXTS)
    rolled = Lattice(
        [Quadrupole(0.5, 0.4, roll=0.2)] + list(lat.elements),
        ref,
    )
    with pytest.raises(CoupledLatticeError, match="couples x and y"):
        resonance_driving_terms(rolled)


# ==========================================================================
# 7. Thick bodies
# ==========================================================================


def _thick_ring(ref: ReferenceParticle, length: float, k2l: float) -> Lattice:
    """The fixture with one sextupole given a body of ``length``, centred where the thin
    one sat, so the thin limit is approached at fixed integrated strength."""
    els: list = []
    s = 0.0
    at = 5.5
    for _ in range(4):
        for k in (KF, KD):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            if s < at < s + 1.0:
                a = at - s - 0.5 * length
                els += [Drift(a), Sextupole(length, k2l / length), Drift(1.0 - a - length)]
            else:
                els.append(Drift(1.0))
            s += 1.0
    return Lattice(els, ref)


def test_thick_sources_converge_on_the_thin_limit(ref: ReferenceParticle) -> None:
    """A shortening body approaches the thin kick, and the midpoint slicing converges.

    Two different limits, deliberately separated. Shrinking the *body* is the physics
    limit; adding *slices* at fixed body is the numerics one, and it is second order
    because the midpoint rule is.
    """
    thin = resonance_driving_terms(_fodo(ref, {5.5: 1.0}))["f3000"]
    errs = [
        abs(resonance_driving_terms(_thick_ring(ref, L, 1.0))["f3000"] - thin)
        for L in (0.4, 0.2, 0.1, 0.05)
    ]
    orders = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(o > 1.8 for o in orders), orders  # converges, at least first order

    lat = _thick_ring(ref, 0.4, 1.0)
    fine = resonance_driving_terms(lat, slices=2048)["f3000"]
    slice_errs = [
        abs(resonance_driving_terms(lat, slices=n)["f3000"] - fine) for n in (8, 16, 32, 64)
    ]
    ratios = [slice_errs[i] / slice_errs[i + 1] for i in range(len(slice_errs) - 1)]
    assert all(r == pytest.approx(4.0, rel=0.15) for r in ratios), ratios


def test_a_thick_skew_quadrupole_is_sliced_on_its_own_map(ref: ReferenceParticle) -> None:
    """Not a drift: a skew quadrupole focuses, and the walker uses the element's matrix.

    The gate is G1's, which handles a thick skew body by its own independent route: the
    ``pi |C^-|`` tie must survive with the body in place, to the same round-off.
    """
    els: list = []
    s = 0.0
    for _ in range(4):
        for k in (KF, KD):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            if abs(s - 2.0) < 1e-12:
                els += [Drift(0.2), SkewQuadrupole(0.3, 0.09), Drift(0.5)]
            else:
                els.append(Drift(1.0))
            s += 1.0
    lat = Lattice(els, ref)
    # the same ring with the body replaced *in place* by a drift, so the unperturbed
    # optics -- which is what both sides evaluate on -- is untouched
    bare = Lattice([Drift(e.length) if isinstance(e, SkewQuadrupole) else e for e in els], ref)
    qx, qy = tunes(bare)
    assert closest_tune_approach(lat) > 0.0  # the body really is in the ring
    f1001 = resonance_driving_terms(lat, slices=256)["f1001"]
    ratio = abs(f1001) * 4.0 * abs(math.sin(math.pi * (qx - qy))) / closest_tune_approach(lat)
    assert ratio == pytest.approx(math.pi, rel=2e-4)


# ==========================================================================
# 8. Tracking: the leg that shares no algebra with any of the above
# ==========================================================================

N_TURNS = 4096


def _turns(lat: Lattice, x: float, n: int = N_TURNS) -> np.ndarray:
    return Tracker(lat).track_turns(Particle(x=x, y=1e-9), n, nonlinear=True)[:n]


def _line(h: np.ndarray, nu: float) -> complex:
    """Complex amplitude of the ``exp(-2 pi i nu n)`` component, Hann-windowed.

    The window is what keeps the huge primary line from leaking onto the small sideband;
    its gain divides out because the same normalisation is applied to both.
    """
    n = np.arange(h.size)
    w = 0.5 - 0.5 * np.cos(2.0 * math.pi * (n + 0.5) / h.size)
    return complex((w * h * np.exp(2j * math.pi * nu * n)).sum() / w.sum())


def _h(traj: np.ndarray, tw, plane: str) -> np.ndarray:
    """``h_u = u_hat + i p_hat_u`` from a tracked trajectory, in the shipped basis."""
    i, beta, alpha = (0, tw.beta_x, tw.alpha_x) if plane == "x" else (2, tw.beta_y, tw.alpha_y)
    u, pu = traj[:, i], traj[:, i + 1]
    return u / math.sqrt(beta) + 1j * (alpha * u / math.sqrt(beta) + math.sqrt(beta) * pu)


@pytest.fixture(scope="module")
def tracked_f3000():
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    lat = _fodo(ref, SEXTS)
    return lat, closed_twiss(lat), tunes(lat)[0], resonance_driving_terms(lat)["f3000"]


def _measure_f3000(lat, tw, qx, amp: float, n: int = N_TURNS) -> complex:
    r"""``f3000`` read off the ``-2 Q_x`` sideband of ``h_x``, launch phase divided out.

    Derived, not quoted. First-order perturbation gives ``h_x = zeta - 2i dG/dconj(zeta)``
    in O3's basis; the only monomial contributing a ``conj(zeta_x)^2`` is ``(0,3,0,0)``,
    whose coefficient is ``conj`` of ``f3000``'s. Conjugating into the shipped basis, the
    two lines of ``h_x`` are

        A(Q_x)    = sqrt(2 J_x) e^{-i phi_0},
        A(-2 Q_x) = 12 i J_x conj(f3000) e^{+2 i phi_0},

    so ``A(-2 Q_x) / conj(A(Q_x))^2 = 6 i conj(f3000)`` -- free of both the action and the
    launch phase, which is what stops the comparison being tunable.
    """
    h = _h(_turns(lat, amp, n), tw, "x")
    return complex(np.conj(_line(h, -2.0 * qx) / (6j * np.conj(_line(h, qx)) ** 2)))


def test_tracked_f3000_matches_in_magnitude_and_phase(tracked_f3000) -> None:
    """The primary physics gate: a real trajectory, and the phase is half the content.

    O1's lesson applied -- a magnitude-only comparison would pass with the wrong
    conjugation, which is exactly the convention this milestone had to choose. Tracking
    is what decides it, because the sign of a measured sideband's phase is not a naming
    choice.
    """
    lat, tw, qx, pred = tracked_f3000
    got = _measure_f3000(lat, tw, qx, 2.0e-4)
    assert abs(got - pred) / abs(pred) < 1e-4
    assert abs(got) == pytest.approx(abs(pred), rel=1e-4)
    assert np.angle(got) == pytest.approx(np.angle(pred), abs=1e-4)
    # the conjugate convention is decisively excluded, not merely worse
    assert abs(got - np.conj(pred)) / abs(pred) > 1.0


def test_the_tracked_residual_is_the_next_order_in_amplitude(tracked_f3000) -> None:
    """It falls as the action, which is what "first order" has to mean here.

    Two things could make the tracked number miss: the second-order RDT this milestone
    does not compute (which is one power of amplitude down) and spectral leakage from
    the primary line (which is not). They are separated by scanning each: at a long turn
    count the residual is quadratic in the launch amplitude, and at fixed amplitude it
    falls with turn count until it reaches that physical floor.
    """
    lat, tw, qx, pred = tracked_f3000
    long = 16384
    errs = [
        abs(_measure_f3000(lat, tw, qx, a, long) - pred) / abs(pred) for a in (4e-4, 2e-4, 1e-4)
    ]
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.25), errs
    assert errs[1] / errs[2] > 2.5, errs
    short = [abs(_measure_f3000(lat, tw, qx, 2e-4, n) - pred) / abs(pred) for n in (1024, 2048)]
    assert short[0] > short[1] > errs[1]  # leakage, and it is removed by more turns


def test_tracking_a_ring_without_sextupoles_has_no_such_sideband(ref: ReferenceParticle) -> None:
    """The exact drift is nonlinear and detunes (O3), but it drives no ``3 Q_x`` line.

    Its own nonlinearity is even in the momenta, so it reaches ``2 Q_x`` and ``4 Q_x`` and
    not the third integer. Without this the tracked gate above could be reading the
    lattice rather than the sextupoles.
    """
    lat = _fodo(ref)
    tw, qx = closed_twiss(lat), tunes(lat)[0]
    h = _h(_turns(lat, 2.0e-4), tw, "x")
    assert abs(_line(h, -2.0 * qx)) / abs(_line(h, qx)) < 1e-6


def test_tracked_f1001_matches_and_the_residual_is_higher_order(ref: ReferenceParticle) -> None:
    r"""``f1001`` off the ``Q_x`` line of ``h_y``, and the error is **cubic** in ``k1sl``.

    Same derivation as ``f3000``'s: the only monomial putting a bare ``zeta_x`` into
    ``h_y`` is ``(1,0,0,1)``, so ``A_y(Q_x) / A_x(Q_x) = 2 i f1001`` -- again independent
    of the action and the launch phase.

    The residual's order was **measured, not predicted**, and it is one power better than
    the obvious guess. ``f1001`` is linear in the skew strength, so a naive reading
    expects the first correction at the square. It is not: the relative error falls by
    four for every halving of the strength, i.e. the absolute error is **cubic**. That is
    consistent with what actually perturbs the answer -- the closed optics this formula
    is evaluated on, whose shift from coupling is itself quadratic -- and it is why the
    comparison against a fully coupled reference converges faster than it looks like it
    should.
    """
    prev, ratios = None, []
    for scale in (1.0, 0.5, 0.25, 0.125):
        lat = _fodo(ref, None, {p: scale * w for p, w in SKEWS.items()})
        bare = _bare(lat)
        tw, qx = closed_twiss(bare), tunes(bare)[0]
        traj = _turns(lat, 1.0e-4)
        got = _line(_h(traj, tw, "y"), qx) / (2j * _line(_h(traj, tw, "x"), qx))
        pred = resonance_driving_terms(lat)["f1001"]
        rel = abs(got - pred) / abs(pred)
        if scale == 1.0:
            assert rel < 1e-2  # already close at full strength
            assert np.angle(got) == pytest.approx(np.angle(pred), abs=1e-2)
        if prev is not None:
            ratios.append(prev / rel)
        prev = rel
    assert all(r == pytest.approx(4.0, rel=0.1) for r in ratios), ratios
