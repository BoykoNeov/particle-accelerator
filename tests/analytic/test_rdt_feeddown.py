r"""O6: feed-down into the driving terms — the symbolic leg.

O4 and O5 built twenty first-order resonance driving terms and evaluated all twenty on
the **design** orbit with every source perfectly placed. Both restrictions are enforced
by a refusal in :func:`~accsim.twiss._rdt_sites` rather than modelled. O6 replaces the
first refusal with a model, and this file is the derivation that model rests on: what a
multipole's strength *becomes* when the beam does not pass through its centre.

**The one object.** Write every transverse multipole kick in the complex form the field
expansion already uses in :mod:`accsim.elements.sextupole` and
:mod:`accsim.elements.octupole`,

    B_y + i B_x = (B rho) sum_n k_n (x + i y)^n / n!,
    Delta px = -(q/p0) int B_y ds,   Delta py = +(q/p0) int B_x ds,

which collapses to a single statement about one complex strength per order,

    Delta px - i Delta py  =  - sum_n K_n z^n / n!,
    K_n = k_nl + i k_nsl,   z = x + i y.

That the *shipped* kicks obey it — normal and skew, quadrupole through octupole — is
measured in section 1 rather than assumed, because the whole milestone is carried on it.

**Why this form and not the real one.** J3 already expanded the octupole kick about an
orbit offset and got six real terms; I2 did the sextupole and got three. In the complex
form both are the *same* one-line statement, because shifting the argument of a
polynomial is a binomial expansion:

    K_m_eff = sum_{n >= m} K_n z_0^(n - m) / (n - m)!,     z_0 = z_co - d,

with ``z_co`` the closed orbit at the magnet and ``d = dx + i dy`` where the magnet
actually sits. Section 2 proves that identity symbolically and then checks it reproduces
J3's and I2's shipped tables entry by entry — two anchors already in the tree, which pin
the *offset* convention completely and for free.

**What those anchors cannot pin, and so is measured.** J3 and I2 are both pure offsets,
so neither says anything about a **roll**. Carrying the frame rotation through the same
algebra gives ``K_n -> K_n exp(-i (n + 1) psi)`` — with a sign this project has no
licence to transcribe, the working agreement's rule and G2's and O3's lesson. Section 3
*measures* the exponent, integer and sign, off the shipped rolled elements' own
``track()``.

**The measurement primitive.** The kick is a polynomial in ``z`` with no dependence on
``conj(z)`` — that analyticity is the reason one complex strength per order suffices —
so sampling it on a circle about the orbit and taking a discrete Fourier transform
returns every ``K_m_eff`` exactly, to round-off, for a polynomial of degree below the
sample count. :func:`_measure_strengths` does that against ``track()`` and nothing else,
so every identity in this file is established against the shipped map rather than
against another formula.
"""

from __future__ import annotations

import cmath
import math
import os
import sys

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Corrector,
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
    resonance_driving_terms_on_orbit,
    tunes,
)
from accsim.coords import PX, PY, X, Y
from accsim.orbit import (
    closed_orbit_nonlinear,
    linearised_one_turn_map,
    propagate_orbit_nonlinear,
)
from accsim.tracking import Particle, Tracker
from accsim.tune import naff
from accsim.twiss import (
    _RDT_TERMS,
    CoupledLatticeError,
    _rdt_sites_on_orbit,
    closed_twiss_on_orbit,
    match_periodic_coupled,
)

sys.path.insert(0, os.path.dirname(__file__))

import test_octupole_driving_terms as o5  # noqa: E402

MASS0, GAMMA0 = 938.27208816e6, 20.0

#: The three groups of terms the orbit scan separates, taken from the shipped table by
#: source kind rather than listed by hand: a later milestone adding a source must not
#: quietly widen or narrow these gates. O4's rule, kept.
_CUBIC = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "sext")
_SKEWQ = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "skew")
_SKEWSEXT = tuple(k for k, v in _RDT_TERMS.items() if v[5] == "skewsext")

#: Highest multipole order the package has a source for. ``n = 1`` quadrupole,
#: ``n = 2`` sextupole, ``n = 3`` octupole; ``n = 0`` is the dipole the feed-down
#: *produces* and no source kind here is.
MAX_ORDER = 3


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


# ==========================================================================
# The measurement primitive: read a magnet's effective strengths off track()
# ==========================================================================


def _complex_kick(elem, ref: ReferenceParticle, z: complex) -> complex:
    """``Delta px - i Delta py`` of ``elem`` at transverse position ``z = x + i y``.

    Read off :meth:`~accsim.elements.element.Element.track` — the shipped map, with
    whatever alignment the element carries — entering at rest transversely so the
    outgoing momenta *are* the kick.
    """
    state = np.zeros(6)
    state[X] = z.real
    state[Y] = z.imag
    out = elem.track(state, ref)
    return complex(out[PX], -out[PY])


def _measure_strengths(
    elem,
    ref: ReferenceParticle,
    order: int = MAX_ORDER,
    *,
    z_co: complex = 0.0,
    radius: float = 1e-3,
) -> list[complex]:
    """``[K_0_eff, ..., K_order_eff]`` measured off ``elem.track()`` about ``z_co``.

    The kick is ``-sum_m K_m_eff z^m / m!`` in the betatron coordinate ``z`` measured
    from ``z_co``, a polynomial in ``z`` alone. Sampling it at ``N > order`` points on a
    circle of radius ``r`` about ``z_co`` and taking the DFT therefore inverts it
    **exactly** rather than approximately: the ``m``-th Fourier coefficient is
    ``-K_m_eff r^m / m!``, with every other order landing on a different harmonic.

    This is deliberately independent of any formula in ``accsim`` — it differentiates
    nothing and expands nothing, it just samples the shipped map — so an identity gated
    against it is gated against the code that actually tracks particles.
    """
    n = 4 * (order + 1)
    samples = np.array(
        [
            _complex_kick(elem, ref, z_co + radius * cmath.exp(2j * math.pi * j / n))
            for j in range(n)
        ]
    )
    coeffs = np.fft.fft(samples) / n
    return [complex(-coeffs[m] * math.factorial(m) / radius**m) for m in range(order + 1)]


# ==========================================================================
# 1. The complex form is the shipped kick, for every source the package has
# ==========================================================================


def _thin_quad(k1l: float, **kw):
    """A normal quadrupole in the thin limit — the package ships no ``ThinQuadrupole`` alias here.

    Length ``L`` with ``k1 = k1l / L`` at small ``L``: its kick is ``-k1l x`` up to the
    ``O(L)`` drift the body also does, which the transverse kick measurement does not see.
    """
    L = 1e-9
    return Quadrupole(L, k1l / L, **kw)


#: ``(name, builder, order, K)`` — one entry per shipped multipole, normal and skew.
#: ``K = k_nl + i k_nsl`` is what the complex form claims the magnet's strength is, and
#: the *skew* rows are the half of the convention a normal-only check would leave free.
_SOURCES = [
    ("normal quad", _thin_quad, 1, lambda s: complex(s)),
    ("skew quad", ThinSkewQuadrupole, 1, lambda s: 1j * s),
    ("normal sext", ThinSextupole, 2, lambda s: complex(s)),
    ("skew sext", ThinSkewSextupole, 2, lambda s: 1j * s),
    ("normal oct", ThinOctupole, 3, lambda s: complex(s)),
]


@pytest.mark.parametrize(("name", "build", "order", "expected"), _SOURCES)
def test_the_shipped_kick_is_the_complex_multipole_form(
    ref: ReferenceParticle, name: str, build, order: int, expected
) -> None:
    """``Delta px - i Delta py = -K_n z^n / n!`` on every thin source in the package.

    The single assumption the rest of the milestone is carried on, measured on the
    shipped ``track()`` rather than read off a docstring. It also fixes ``K``'s
    *imaginary* part as the skew strength with a ``+`` sign, which is the half of the
    convention a normal-magnet-only check would leave free.
    """
    strength = 0.7
    K = _measure_strengths(build(strength), ref, order)
    assert K[order] == pytest.approx(expected(strength), rel=1e-9, abs=1e-12)
    for m in range(order):
        assert abs(K[m]) < 1e-9, f"{name}: order {m} should be empty on a centred magnet"


# ==========================================================================
# 2. The expansion, derived — and checked against J3's and I2's shipped tables
# ==========================================================================


@pytest.fixture(scope="module")
def expansion() -> dict[int, sp.Expr]:
    """``{m: K_m_eff}`` derived in sympy from the shift ``z -> z_0 + z``, not quoted."""
    z, z0 = sp.symbols("z z0")
    K = sp.symbols(f"K0:{MAX_ORDER + 1}")
    shifted = sp.expand(sum(K[n] * (z0 + z) ** n / sp.factorial(n) for n in range(MAX_ORDER + 1)))
    poly = sp.Poly(shifted, z)
    return {m: sp.expand(poly.coeff_monomial(z**m) * sp.factorial(m)) for m in range(MAX_ORDER + 1)}


def test_the_effective_strength_is_the_binomial_shift(expansion: dict[int, sp.Expr]) -> None:
    """``K_m_eff = sum_{n >= m} K_n z_0^(n-m) / (n-m)!``, as an exact symbolic identity.

    The whole of the feed-down model in one line. Everything below is this line read at
    a particular ``m``, or measured against ``track()``.
    """
    z0 = sp.Symbol("z0")
    K = sp.symbols(f"K0:{MAX_ORDER + 1}")
    for m in range(MAX_ORDER + 1):
        claim = sum(K[n] * z0 ** (n - m) / sp.factorial(n - m) for n in range(m, MAX_ORDER + 1))
        assert sp.simplify(expansion[m] - claim) == 0


def _split(expr: sp.Expr, x0: sp.Symbol, y0: sp.Symbol) -> tuple[sp.Expr, sp.Expr]:
    """``(normal, skew)`` — the real and imaginary parts of one effective strength.

    The strength substituted into ``expr`` is taken **real** (a normal source) and
    ``z_0 = x_0 + i y_0`` complex, which is the case both shipped tables were derived for.
    """
    sub = sp.expand(expr.subs(sp.Symbol("z0"), x0 + sp.I * y0))
    return sp.simplify(sp.re(sub)), sp.simplify(sp.im(sub))


def test_the_expansion_reproduces_j3s_shipped_octupole_feeddown(
    expansion: dict[int, sp.Expr],
) -> None:
    """Anchor one: J3's six real terms, already shipped in ``linearised_lattice``.

    ``tests/analytic/test_octupole_feeddown.py`` derived these independently, in real
    coordinates, one milestone ago. They are not re-derived here — they are the *check*
    that the complex one-liner above is the same physics, which is what makes the offset
    convention (including ``z_0``'s sign) pinned rather than chosen.
    """
    x0, y0, k3l = sp.symbols("x0 y0 k3l", real=True)
    K = sp.symbols(f"K0:{MAX_ORDER + 1}")
    only_oct = {K[0]: 0, K[1]: 0, K[2]: 0, K[3]: k3l}

    def eff(m: int) -> tuple[sp.Expr, sp.Expr]:
        return _split(expansion[m].subs(only_oct), x0, y0)

    # normal sext k2l_eff = +k3l x_co ; skew sext k2sl_eff = +k3l y_co
    assert sp.simplify(eff(2)[0] - k3l * x0) == 0
    assert sp.simplify(eff(2)[1] - k3l * y0) == 0
    # normal quad k1l_eff = +1/2 k3l (x_co^2 - y_co^2) ; skew quad k1sl_eff = +k3l x_co y_co
    assert sp.simplify(eff(1)[0] - k3l * (x0**2 - y0**2) / 2) == 0
    assert sp.simplify(eff(1)[1] - k3l * x0 * y0) == 0
    # the dipole part: Delta px = -Re(K_0_eff), Delta py = +Im(K_0_eff)
    assert sp.simplify(-eff(0)[0] + k3l * x0 * (x0**2 - 3 * y0**2) / 6) == 0
    assert sp.simplify(+eff(0)[1] - k3l * y0 * (3 * x0**2 - y0**2) / 6) == 0


def test_the_expansion_reproduces_i2s_shipped_sextupole_feeddown(
    expansion: dict[int, sp.Expr],
) -> None:
    """Anchor two: I2's split, shipped in ``linearised_lattice`` one order down.

    The same statement at ``n = 2``. Two anchors rather than one because a single one
    could be reproduced by a wrong formula that happens to agree at one order — the
    quadratic and cubic cases have different binomial coefficients, so agreeing with
    both fixes the ``1/(n-m)!`` as well as the sign.
    """
    x0, y0, k2l = sp.symbols("x0 y0 k2l", real=True)
    K = sp.symbols(f"K0:{MAX_ORDER + 1}")
    only_sext = {K[0]: 0, K[1]: 0, K[2]: k2l, K[3]: 0}

    def eff(m: int) -> tuple[sp.Expr, sp.Expr]:
        return _split(expansion[m].subs(only_sext), x0, y0)

    # normal quad k1l_eff = +k2l x_co ; skew quad k1sl_eff = +k2l y_co
    assert sp.simplify(eff(1)[0] - k2l * x0) == 0
    assert sp.simplify(eff(1)[1] - k2l * y0) == 0
    # the dipole part, I2's theta: Delta px = -1/2 k2l (x_co^2 - y_co^2), Delta py = +k2l x_co y_co
    assert sp.simplify(-eff(0)[0] + k2l * (x0**2 - y0**2) / 2) == 0
    assert sp.simplify(+eff(0)[1] - k2l * x0 * y0) == 0


@pytest.mark.parametrize(
    ("name", "build", "order"),
    [("sext", ThinSextupole, 2), ("oct", ThinOctupole, 3), ("skewsext", ThinSkewSextupole, 2)],
)
def test_the_expansion_is_what_the_shipped_map_actually_does(
    ref: ReferenceParticle, name: str, build, order: int
) -> None:
    """The formula, evaluated numerically, against the same magnet's own ``track()``.

    Sections 2's identities are symbolic; this closes the loop back onto the shipped
    code at a generic complex offset, including a *skew* source, which neither anchor
    covers (both J3 and I2 expanded a normal magnet).
    """
    strength = 0.9
    K_true = _measure_strengths(build(strength), ref, order)[order]
    z0 = 1.7e-3 - 2.3e-3j
    measured = _measure_strengths(build(strength), ref, order, z_co=z0)
    for m in range(order + 1):
        claim = K_true * z0 ** (order - m) / math.factorial(order - m)
        assert measured[m] == pytest.approx(claim, rel=1e-7, abs=1e-12)


# ==========================================================================
# 3. The roll phase — measured, never transcribed
# ==========================================================================


def _rolled(build, strength: float, psi: float):
    """The same magnet ``build`` makes, carrying roll ``psi``."""
    elem = build(strength)
    elem.roll = float(psi)
    return elem


@pytest.mark.parametrize(
    ("name", "build", "order"),
    [("quad", _thin_quad, 1), ("sext", ThinSextupole, 2), ("oct", ThinOctupole, 3)],
)
def test_a_roll_multiplies_the_strength_by_a_measured_phase(
    ref: ReferenceParticle, name: str, build, order: int
) -> None:
    """``K_n -> K_n exp(i c psi)``, with the integer ``c`` **fitted** off ``track()``.

    The two shipped anchors are pure offsets, so neither constrains this at all, and the
    package's own hint — a rolled sextupole is a skew sextupole "at -30 degrees" — is
    exactly the kind of remembered constant the working agreement forbids trusting. So
    the exponent is obtained by regression over a spread of roll angles and *then*
    compared with ``-(n + 1)``; a transcription error of either sign or magnitude fails
    here rather than surviving into the sum.
    """
    strength = 0.6
    psis = np.linspace(0.05, 0.35, 7)
    phases = []
    for psi in psis:
        K = _measure_strengths(build(strength), ref, order)[order]
        K_rolled = _measure_strengths(_rolled(build, strength, psi), ref, order)[order]
        phases.append(cmath.phase(K_rolled / K))
    slope = float(np.polyfit(psis, np.unwrap(phases), 1)[0])
    assert slope == pytest.approx(-(order + 1), rel=1e-6), (
        f"{name}: measured roll exponent {slope:.9g}, not -(n+1) = {-(order + 1)}"
    )


def test_a_rolled_sextupole_is_a_skew_sextupole_at_the_measured_angle(
    ref: ReferenceParticle,
) -> None:
    """The package's own claim, checked rather than repeated.

    ``K_2 -> k2l exp(-3 i psi)`` is pure imaginary — a pure skew sextupole — at
    ``psi = -30 deg``, and the sign of ``k2sl`` that comes out is the thing a remembered
    "+30 degrees" would get backwards.
    """
    k2l = 0.6
    rolled = _rolled(ThinSextupole, k2l, -math.pi / 6)
    K = _measure_strengths(rolled, ref, 2)[2]
    assert K == pytest.approx(1j * k2l, rel=1e-9, abs=1e-12)
    skew = _measure_strengths(ThinSkewSextupole(k2l), ref, 2)[2]
    assert K == pytest.approx(skew, rel=1e-9, abs=1e-12)


def test_a_rolled_quadrupole_is_a_skew_quadrupole_at_the_measured_angle(
    ref: ReferenceParticle,
) -> None:
    """The same statement at ``n = 1``, where G1 already shipped the skew element.

    ``K_1 -> k1l exp(-2 i psi)`` is pure imaginary at ``psi = -45 deg``. The two angles
    ``-30`` and ``-45`` differing is itself the content: they are ``-90 deg / (n + 1)``,
    which is the exponent this section fitted.
    """
    k1l = 0.6
    quad = _rolled(_thin_quad, k1l, -math.pi / 4)
    K = _measure_strengths(quad, ref, 1)[1]
    assert K == pytest.approx(1j * k1l, rel=1e-6, abs=1e-9)
    skew = _measure_strengths(ThinSkewQuadrupole(k1l), ref, 1)[1]
    assert K == pytest.approx(skew, rel=1e-6, abs=1e-9)


# ==========================================================================
# 4. Displacement and orbit are the same variable — with the sign written first
# ==========================================================================


def test_a_displaced_magnet_equals_a_centred_magnet_at_the_opposite_orbit(
    ref: ReferenceParticle,
) -> None:
    """``z_0 = z_co - d``: a magnet moved by ``+d`` is a magnet the beam passes at ``-d``.

    Purely internal and sharp, and it catches a displacement-sign error that **both**
    external reference codes would share and therefore could not arbitrate. The sign is
    written down from the substitution before it is run: the element enters its body at
    ``z_lab - d`` (:meth:`Element._alignment_entry`), so an orbit ``z_co`` through a
    magnet at ``d`` presents the body with ``z_co - d`` — the *difference*, which is why
    only one variable reaches the feed-down and not two.

    Note what cannot be done here: the orbit cannot be held fixed while the magnet moves,
    because a displaced nonlinear source is itself a dipole kick and moves the closed
    orbit it would be evaluated on. The identity is a statement about the pair, so it is
    checked on the pair.
    """
    k3l, d, z_co = 0.9, 1.1e-3 - 0.4e-3j, 2.5e-3 + 1.3e-3j
    displaced = ThinOctupole(k3l, dx=d.real, dy=d.imag)
    centred = ThinOctupole(k3l)
    moved = _measure_strengths(displaced, ref, 3, z_co=z_co)
    equiv = _measure_strengths(centred, ref, 3, z_co=z_co - d)
    for m in range(4):
        assert moved[m] == pytest.approx(equiv[m], rel=1e-7, abs=1e-12)


def test_the_equivalent_element_identity_is_measured_before_it_is_asserted(
    ref: ReferenceParticle,
) -> None:
    """An octupole at offset ``x_0`` contributes what a sextupole of ``k3l x_0`` does.

    The obvious statement of feed-down — and this project has been caught on precisely
    this identity before: I2 recorded that the expansion's coefficient is *not* the
    equivalent element's kick and that signs flip between them. So it is established by
    measuring both magnets' shipped maps, never by quoting the expansion at itself.
    """
    k3l, x0 = 0.9, 3.0e-3
    oct_off = _measure_strengths(ThinOctupole(k3l), ref, 3, z_co=complex(x0))
    sext = _measure_strengths(ThinSextupole(k3l * x0), ref, 2)
    assert oct_off[2] == pytest.approx(sext[2], rel=1e-7, abs=1e-12)
    # ...and the two are *not* equal below that order: the octupole also carries a
    # gradient and a dipole the equivalent sextupole does not have, which is the half of
    # the identity a one-order check would miss.
    assert abs(oct_off[1]) > 1e-6
    assert abs(sext[1]) < 1e-12


# ==========================================================================
# 5. The sum on the orbit: the sibling, and the design path it must not disturb
# ==========================================================================


@pytest.fixture
def octs_only(ref: ReferenceParticle):
    """The probe's own fixture: **octupoles and no sextupole**.

    Deliberately not a mixed ring. Every cubic term on this lattice is *purely* fed
    down — an octupole drives none of them directly — so an orbit scan measures the
    feed-down alone rather than a small correction sitting on top of a direct
    contribution the milestone does not change. On a mixed ring the direct part would
    not be constant either (the optics move, which is this milestone's own headline),
    and a power fit would carry that as a residual.
    """

    def build(kick_x: float = 0.0, kick_y: float = 0.0):
        base = o5._fodo(ref, octs=o5.OCTS)
        return Lattice([Corrector(kick_x=kick_x, kick_y=kick_y)] + list(base.elements), ref)

    return build


def test_on_a_flat_orbit_the_two_functions_are_the_same_function(ref: ReferenceParticle) -> None:
    """The cheapest and sharpest gate in the file, and it runs before any power fit.

    A machine whose closed orbit is on axis has no feed-down, so the on-orbit sibling
    must reproduce :func:`resonance_driving_terms` on every one of the twenty terms and
    all four source kinds at once. It agrees to ``1e-10`` relative rather than exactly,
    and the reason is named: the sibling takes its optics from
    :func:`~accsim.orbit.linearised_element_maps`, which differentiates ``track()`` by
    finite differences, so it carries that primitive's own ``~1e-12`` round-off. That is
    precisely why the design path was left alone rather than re-pointed at the same
    primitive — O4 pins its agreement with MAD-X PTC at ``1e-14``, four orders inside
    this floor.
    """
    lat = o5._fodo(
        ref,
        octs=o5.OCTS,
        skewsexts=o5.SKEWSEXTS,
        sexts={2.7: 0.5, 7.1: -0.4},
        skews={5.9: 0.02},
    )
    design = resonance_driving_terms(lat)
    on_orbit = resonance_driving_terms_on_orbit(lat)
    assert set(design) == set(on_orbit)
    for key, value in design.items():
        assert abs(value) > 1e-3, f"{key} is vacuously small on this fixture"
        assert on_orbit[key] == pytest.approx(value, rel=1e-10)


def test_the_design_path_is_untouched_by_the_sibling(ref: ReferenceParticle) -> None:
    """A misaligned source still raises from :func:`resonance_driving_terms` itself.

    O6 adds a model; it does not lift the old function's refusal. Keeping that explicit
    is what makes "nothing on axes A-N or O1-O5 moved" checkable rather than asserted.
    """
    els = list(o5._fodo(ref, octs=o5.OCTS).elements)
    for elem in els:
        if isinstance(elem, ThinOctupole):
            elem.dx = 1e-3
    with pytest.raises(CoupledLatticeError, match="misaligned"):
        resonance_driving_terms(Lattice(els, ref))


# ==========================================================================
# 6. The primary gate: a power of the orbit, fitted term by term
# ==========================================================================


def _fit_power(xs: list[float], ys: list[float]) -> float:
    """``d log y / d log x`` — the exponent, by regression rather than by ratio."""
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def _orbit_extent(lat: Lattice, plane: int) -> float:
    """Peak ``|x|`` or ``|y|`` of the closed orbit — the variable the fit is against.

    Measured from the ring rather than taken as the corrector strength: an octupole is
    nonlinear, so the orbit it settles on is not exactly proportional to the kick that
    made it, and fitting against the kick would smear the exponent it is meant to
    resolve.
    """
    return max(abs(o[plane]) for o in propagate_orbit_nonlinear(lat, closed_orbit_nonlinear(lat)))


def test_on_a_flat_orbit_an_octupole_drives_no_cubic_term_at_all(octs_only) -> None:
    """The reference point the whole scan is measured against, and it is *exact*.

    Not "small": an octupole reaches the sextupole lines only through feed-down, so with
    the orbit on axis these five terms are identically zero rather than negligible. That
    is what makes the scan below a measurement of feed-down and nothing else.
    """
    f = resonance_driving_terms_on_orbit(octs_only())
    for key in _CUBIC + _SKEWQ + _SKEWSEXT:
        assert f[key] == 0.0


@pytest.mark.parametrize("key", _CUBIC)
def test_a_horizontal_orbit_reaches_the_normal_sextupole_lines_at_first_power(
    octs_only, key: str
) -> None:
    """``k2l_eff = k3l x_co``: five terms, each fitted **separately**.

    Never in aggregate, and that is J3's lesson rather than a stylistic choice: a cubic
    kick lands on three quantities at three different powers of the orbit, and a uniform
    mis-scale is invisible to all three at once. An aggregate fit passes with a wrong
    common factor; five separate fits do not.
    """
    scan = [(t, octs_only(kick_x=1e-4 * t)) for t in (1.0, 2.0, 4.0)]
    xs = [_orbit_extent(lat, X) for _, lat in scan]
    ys = [abs(resonance_driving_terms_on_orbit(lat)[key]) for _, lat in scan]
    assert _fit_power(xs, ys) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("key", _SKEWSEXT)
def test_a_vertical_orbit_reaches_the_skew_sextupole_lines_at_first_power(
    octs_only, key: str
) -> None:
    """``k2sl_eff = k3l y_co`` — the *other* half of the same complex coefficient.

    The two halves are what make the complex strength the right object: one real
    displacement reaches the normal lines and one vertical displacement the skew ones,
    from a single ``K_3 z_0``. A model that fed down only in ``x`` would pass the test
    above and fail this one.
    """
    scan = [(t, octs_only(kick_y=1e-4 * t)) for t in (1.0, 2.0, 4.0)]
    ys_orbit = [_orbit_extent(lat, Y) for _, lat in scan]
    ys = [abs(resonance_driving_terms_on_orbit(lat)[key]) for _, lat in scan]
    assert _fit_power(ys_orbit, ys) == pytest.approx(1.0, abs=0.02)


def test_the_skew_quadrupole_lines_need_both_planes_at_once(octs_only) -> None:
    """``k1sl_eff = k3l x_co y_co`` — a *product*, not a sum, and exactly zero otherwise.

    The sharpest single check that the feed-down is the complex expansion rather than
    two independent real ones: a purely horizontal orbit and a purely vertical orbit each
    leave ``f1001``/``f1010`` identically zero, and only both together switch them on --
    then linearly in whichever plane is scanned.
    """
    for key in _SKEWQ:
        assert resonance_driving_terms_on_orbit(octs_only(kick_x=2e-4))[key] == 0.0
        assert resonance_driving_terms_on_orbit(octs_only(kick_y=2e-4))[key] == 0.0
    both = resonance_driving_terms_on_orbit(octs_only(kick_x=1e-4, kick_y=1e-4))
    twice_x = resonance_driving_terms_on_orbit(octs_only(kick_x=2e-4, kick_y=1e-4))
    twice_y = resonance_driving_terms_on_orbit(octs_only(kick_x=1e-4, kick_y=2e-4))
    for key in _SKEWQ:
        assert abs(both[key]) > 1e-6
        assert abs(twice_x[key]) == pytest.approx(2.0 * abs(both[key]), rel=0.02)
        assert abs(twice_y[key]) == pytest.approx(2.0 * abs(both[key]), rel=0.02)


# ==========================================================================
# 7. The half that is not the strengths: the optics move too
# ==========================================================================


def test_the_on_orbit_optics_move_by_far_more_than_the_reference_tolerance(octs_only) -> None:
    """The milestone's headline, and the reason it is a *correction* rather than a rival.

    Feed-down from a quartic source reaches the **quadrupole** order as well
    (``k1l_eff = k3l x_co^2 / 2``), so the ``beta`` and phases the first-order formula is
    evaluated at move. The effect is small against the strengths — which go from nothing
    to everything — but it is not small against the tolerance the reference legs are
    gated at: measured here at ``0.01%`` to ``0.2%`` of ``beta_x``, i.e. hundreds of
    times ``1e-6``. Neither arbiter announces it (``xtrack``'s ``twiss`` linearises about
    the closed orbit and PTC's normal form is built about it), so a model that fed down
    only into the strengths would miss both by a margin they could not explain.
    """
    beats = []
    for kick in (1e-4, 4e-4):
        lat = octs_only(kick_x=kick)
        beats.append(abs(closed_twiss_on_orbit(lat).beta_x / closed_twiss(lat).beta_x - 1.0))
    assert beats[0] > 1e-4, "the beta beat is too small on this fixture to be the point"
    assert beats[1] > 100 * 1e-6, "below the reference tolerance, so the headline is wrong"
    # ...and it is genuinely quadratic in the orbit, which is what k3l x_co^2 / 2 says.
    assert beats[1] / beats[0] == pytest.approx(16.0, rel=0.15)


def test_the_sum_divides_by_the_steered_machines_tunes_not_the_blueprints(octs_only) -> None:
    """The tunes the sum divides by are the *steered* machine's, not the design lattice's.

    An RDT is divided by ``exp(-2 pi i [m_x Q_x + m_y Q_y]) - 1``, so a tune taken from
    the design lattice would put a wrong resonance denominator under every one of the
    twenty terms, the error growing with how close the working point sits to a driven
    line. The shift is tiny in absolute terms and that is exactly why it needs a gate:
    it is the kind of difference a comparison against another code would absorb into a
    loosened tolerance rather than localise.
    """
    lat = octs_only(kick_x=4e-4)
    _, qx, qy = _rdt_sites_on_orbit(lat, 32, None, 0.0, 1e-7)
    design_qx, design_qy = tunes(lat)
    assert qx != pytest.approx(design_qx, abs=1e-9)
    assert qy != pytest.approx(design_qy, abs=1e-9)
    assert (qx, qy) == pytest.approx((design_qx, design_qy), abs=1e-3)


def test_the_effective_strengths_are_the_derived_expansion(octs_only) -> None:
    """The intermediate, gated directly rather than through the twenty-term sum.

    Three of this milestone's pre-committed checks are statements about ``z_0`` and
    ``K_m_eff``, not about the sum; asserting them on the sum would mean inverting twenty
    superposed contributions to localise a sign. :func:`_rdt_sites_on_orbit` exposes the
    per-source strengths it filed, so they are checked where they are made — and against
    the orbit read independently off
    :func:`~accsim.orbit.propagate_orbit_nonlinear`, not off the walk itself.
    """
    lat = octs_only(kick_x=3e-4, kick_y=2e-4)
    sites, _, _ = _rdt_sites_on_orbit(lat, 32, None, 0.0, 1e-7)
    # propagate_orbit_nonlinear returns len(elements) + 1 boundary points; the orbit a
    # thin source sees is the one at its ENTRANCE, so drop the ring's closing point.
    orbit = propagate_orbit_nonlinear(lat, closed_orbit_nonlinear(lat))[:-1]
    at_octs = [
        complex(o[X], o[Y])
        for o, e in zip(orbit, lat.elements, strict=True)
        if isinstance(e, ThinOctupole)
    ]
    k3ls = [e.k3l for e in lat.elements if isinstance(e, ThinOctupole)]
    assert len(at_octs) == len(o5.OCTS)
    rows = zip(at_octs, k3ls, sites["sext"][0], sites["skewsext"][0], strict=True)
    for z0, k3l, sext, skewsext in rows:
        assert sext == pytest.approx(k3l * z0.real, rel=1e-6)
        assert skewsext == pytest.approx(k3l * z0.imag, rel=1e-6)


# ==========================================================================
# 8. Thick bodies, where the body is no longer a drift
# ==========================================================================


def test_a_thick_source_on_a_steered_orbit_converges_at_second_order(
    ref: ReferenceParticle,
) -> None:
    """``slices`` is midpoint quadrature, and it converges as ``1/slices^2``.

    The gate that a thick body's own feed-down is carried rather than dropped. On a
    steered orbit the half-slice block is *not* a plain drift any more — the gradient
    ``k1l_eff = K_2 z_0`` acts inside the body — and a walk that kept the design path's
    drift block would still converge, just to a different number. What pins it is the
    **order**: a first-order error would show gap ratios of 2 rather than 4.
    """
    base = list(o5._fodo(ref).elements)
    base.insert(1, Octupole(0.4, 400.0 / 0.4))
    lat = Lattice([Corrector(kick_x=3e-4)] + base, ref)
    values = [
        abs(resonance_driving_terms_on_orbit(lat, slices=n)["f3000"]) for n in (16, 32, 64, 128)
    ]
    gaps = [abs(values[i] - values[i + 1]) for i in range(3)]
    for a, b in zip(gaps, gaps[1:], strict=False):
        assert a / b == pytest.approx(4.0, rel=0.1), f"gaps {gaps} are not second order"


# ==========================================================================
# 9. What the model covers, and what stays refused
# ==========================================================================


def test_an_offset_source_is_modelled_where_the_design_path_refuses_it(
    ref: ReferenceParticle,
) -> None:
    """The refusal O6 exists to remove — and it must produce physics, not merely not raise."""
    els = list(o5._fodo(ref, octs=o5.OCTS).elements)
    for elem in els:
        if isinstance(elem, ThinOctupole):
            elem.dx = 1.0e-3
    f = resonance_driving_terms_on_orbit(Lattice(els, ref))
    for key in _CUBIC:
        assert abs(f[key]) > 1e-3, f"{key} should be driven by a displaced octupole"


def test_a_rolled_octupole_stays_refused_and_says_why(ref: ReferenceParticle) -> None:
    """The one recognised kind whose rolled half has no model.

    ``Im(K_3 exp(-4 i psi)) = -k3l sin 4 psi`` is a **skew octupole**, which drives eight
    terms this table has no row for. Letting it through would smuggle in exactly the
    scope O6 was chosen *over*: the probes measured that MAD-X PTC exposes no
    odd-vertical-charge quartic row at all, so a skew-octupole block would have one
    reference leg where feed-down has two. An *offset* octupole is modelled; only the
    roll is refused, and the message says which.
    """
    els = list(o5._fodo(ref, octs=o5.OCTS).elements)
    for elem in els:
        if isinstance(elem, ThinOctupole):
            elem.roll = 0.1
    with pytest.raises(CoupledLatticeError, match="skew octupole"):
        resonance_driving_terms_on_orbit(Lattice(els, ref))


def test_a_rolled_quadrupole_stays_refused_by_the_measured_coupling_guard(
    ref: ReferenceParticle,
) -> None:
    """The refusal that must **survive** the surgery, and the reason over-lifting is easy.

    Both refusals live in the same loop, so widening the misalignment one is the natural
    way to delete this one by accident. A rolled quadrupole corrupts ``f1001``/``f1010``
    and O6 gives it no model, so it stays refused — by the *measured* guard on the
    element's own matrix, exactly as :func:`closest_tune_approach` does it.
    """
    els = list(o5._fodo(ref, octs=o5.OCTS).elements)
    for elem in els:
        if isinstance(elem, Quadrupole):
            elem.roll = 0.05
            break
    with pytest.raises(CoupledLatticeError, match="couples x and y"):
        resonance_driving_terms_on_orbit(Lattice(els, ref))


def test_a_rolled_sextupole_comes_into_scope_rather_than_tripping_the_guard(
    ref: ReferenceParticle,
) -> None:
    """The other side of that surgery, and it is not symmetrical with the quadrupole.

    A rolled sextupole *is* a skew sextupole — the complex strength says so exactly, and
    section 3 measured the angle — and both parities are rows in the table, so it is
    modelled rather than refused. It also demonstrates why the coupling guard could not
    simply be re-pointed at these elements: a sextupole's map is a **drift**, so rolling
    it leaves off-blocks of ``1e-18`` round-off, and that guard tests exact nonzero. It
    would fire on arithmetic noise rather than on physics.
    """
    sexts = {2.4: 3.0, 6.8: -2.1}
    els = list(o5._fodo(ref, sexts=sexts).elements)
    for elem in els:
        if isinstance(elem, ThinSextupole):
            elem.roll = -math.pi / 6
    got = resonance_driving_terms_on_orbit(Lattice(els, ref))
    want = resonance_driving_terms_on_orbit(
        Lattice(
            [
                ThinSkewSextupole(e.k2l) if isinstance(e, ThinSextupole) else e
                for e in o5._fodo(ref, sexts=sexts).elements
            ],
            ref,
        )
    )
    for key in _SKEWSEXT:
        assert abs(want[key]) > 1e-3
        assert got[key] == pytest.approx(want[key], rel=1e-8)
    for key in _CUBIC:
        assert abs(got[key]) < 1e-9


# ==========================================================================
# 10. The approximation that stops being free, measured and named
# ==========================================================================


def test_the_decoupling_premise_costs_one_order_more_than_the_terms_it_buys(
    ref: ReferenceParticle,
) -> None:
    """A vertical orbit through a sextupole **is** a skew quadrupole, so the ring couples.

    First-order perturbation theory asks for the unperturbed optics, so this walk
    decouples every element map — as :func:`resonance_driving_terms` and
    :func:`closest_tune_approach` do. On a vertical orbit that stops being a convenience
    and becomes an approximation with a size, so the size is measured rather than left
    unowned: the fed-down skew strength the sum actually uses grows **linearly** in
    ``y_co``, while the coupling it genuinely produces grows **quadratically** (read as
    ``1 - gamma_c``, Edwards-Teng's mixing). The neglected coupling is therefore one
    order down on the terms it is neglected in favour of, and it vanishes with the orbit
    rather than sitting at a fixed size.
    """
    strengths, couplings, orbits = [], [], []
    for kick in (1e-4, 2e-4, 4e-4):
        base = o5._fodo(ref, sexts={2.4: 3.0, 6.8: -2.1, 9.9: 4.5})
        lat = Lattice([Corrector(kick_y=kick)] + list(base.elements), ref)
        sites, _, _ = _rdt_sites_on_orbit(lat, 32, None, 0.0, 1e-7)
        strengths.append(float(np.abs(sites["skew"][0]).sum()))
        couplings.append(abs(1.0 - match_periodic_coupled(linearised_one_turn_map(lat)).gamma_c))
        orbits.append(_orbit_extent(lat, Y))
    assert _fit_power(orbits, strengths) == pytest.approx(1.0, abs=0.02)
    assert _fit_power(orbits, couplings) == pytest.approx(2.0, abs=0.05)


# ==========================================================================
# 11. Tracking: the leg that shares no algebra with the expansion or the sum
# ==========================================================================

N_TURNS = 8192


def _betatron_turns(lat: Lattice, amp: float, n: int = N_TURNS) -> tuple[np.ndarray, np.ndarray]:
    """Turn-by-turn deviation **from the closed orbit**, and the orbit it was taken about.

    The whole difference from O4's and O5's tracked legs, and it is the milestone: there
    the closed orbit was the axis, so a trajectory *was* its own betatron motion. Here the
    beam circulates about a bump, so the launch has to be the closed orbit plus an
    amplitude and the record has to have that same orbit taken back off — otherwise the
    "betatron" coordinate carries a DC offset, which lands on every sideband at once.
    """
    co = closed_orbit_nonlinear(lat)
    p = Particle(x=co[0] + amp, px=co[1], y=co[2], py=co[3])
    traj = Tracker(lat).track_turns(p, n, nonlinear=True)[:n]
    out = np.array(traj, dtype=float, copy=True)
    out[:, :4] -= co
    return out, co


def _hx(traj: np.ndarray, tw) -> np.ndarray:
    """``h_x = x_hat + i p_hat_x`` in the shipped basis, from the **on-orbit** optics."""
    x, px = traj[:, 0], traj[:, 1]
    return x / math.sqrt(tw.beta_x) + 1j * (
        tw.alpha_x * x / math.sqrt(tw.beta_x) + math.sqrt(tw.beta_x) * px
    )


def _line(h: np.ndarray, nu: float) -> complex:
    """Complex amplitude of the ``exp(-2 pi i nu n)`` component, Hann-windowed."""
    n = np.arange(h.size)
    w = 0.5 - 0.5 * np.cos(2.0 * math.pi * (n + 0.5) / h.size)
    return complex((w * h * np.exp(2j * math.pi * nu * n)).sum() / w.sum())


def _tracked_f3000(lat: Lattice, tw, amp: float, n: int = N_TURNS) -> tuple[complex, float]:
    r"""``f3000`` off the ``-2 Q_x`` sideband, with ``Q_x`` taken from the trajectory.

    O4's read-out, unchanged — ``A(-2 Q_x) / conj(A(Q_x))^2 = 6 i conj(f3000)``, free of
    both the action and the launch phase — applied to a ring in which **no sextupole
    exists**. Every bit of what it measures is fed down from the octupoles by the orbit.

    The tune is measured from the trajectory rather than taken from the lattice, which is
    O5's trap carried forward and is doubly required here: the octupoles detune with
    amplitude (O5's reason) *and* the bump moves the linear tune (this milestone's own
    headline). Reading the sideband at the design tune would put it in the wrong place by
    both errors at once.
    """
    traj, _ = _betatron_turns(lat, amp, n)
    h = _hx(traj, tw)
    qx = 1.0 - naff(h)
    return complex(np.conj(_line(h, -2.0 * qx) / (6j * np.conj(_line(h, qx)) ** 2))), qx


@pytest.fixture(scope="module")
def tracked(ref_module: ReferenceParticle):
    """A steered octupole ring, its on-orbit optics, and the term the sum predicts."""
    base = o5._fodo(ref_module, octs=o5.OCTS)
    lat = Lattice([Corrector(kick_x=2.0e-4)] + list(base.elements), ref_module)
    return lat, closed_twiss_on_orbit(lat), resonance_driving_terms_on_orbit(lat)["f3000"]


@pytest.fixture(scope="module")
def ref_module() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def test_tracking_sees_a_term_the_design_orbit_function_says_does_not_exist(tracked) -> None:
    """The sharpest discriminator available for this milestone, and it needs no tolerance.

    This ring holds no sextupole, so :func:`resonance_driving_terms` returns ``f3000``
    **exactly zero** — not small. A real trajectory nevertheless carries a ``-2 Q_x``
    sideband whose amplitude is the fed-down term, so the two functions are separated
    here by everything rather than by a fraction of a percent. No conjugation, scale
    factor or convention can make a zero agree with a measurement.
    """
    lat, tw, pred = tracked
    assert resonance_driving_terms(lat)["f3000"] == 0.0
    assert abs(pred) > 1e-2
    got, _ = _tracked_f3000(lat, tw, 2.0e-4)
    assert abs(got) > 1e-2
    # Measured 1.07e-4 at this amplitude and turn count; gated an order above it, not at
    # the round number a loose gate would take, so the assertion can actually fail.
    assert abs(got - pred) / abs(pred) < 1e-3


def test_the_tracked_term_matches_in_magnitude_and_phase(tracked) -> None:
    """Both halves, because the phase is where a wrong conjugation would hide.

    O1's lesson, which O4 and O5 both had to apply: a magnitude-only comparison passes
    with the conjugate convention. The sign of a measured sideband's phase is not a naming
    choice, so tracking is what decides it — and here it decides it for a term whose whole
    existence comes from the feed-down expansion's complex arithmetic.
    """
    lat, tw, pred = tracked
    got, _ = _tracked_f3000(lat, tw, 1.0e-4, n=16384)
    # Measured 2.6e-5 in magnitude and 4.6e-5 in phase; both gated an order above.
    assert abs(got) == pytest.approx(abs(pred), rel=3e-4)
    assert np.angle(got) == pytest.approx(np.angle(pred), abs=3e-4)
    assert abs(got - np.conj(pred)) / abs(pred) > 1.0


def test_the_tracked_residual_falls_with_the_launch_amplitude(tracked) -> None:
    """What "first order" has to mean: the miss is the next order, not a fixed offset.

    A wrong constant in the read-out, or a wrong coefficient in the feed-down expansion,
    would be a pure **scale factor** — a fixed relative offset no launch amplitude
    removes. What is required instead is a residual that shrinks as the action does,
    which is the signature of the second-order content this milestone does not compute.
    """
    lat, tw, pred = tracked
    errs = [
        abs(_tracked_f3000(lat, tw, a, 16384)[0] - pred) / abs(pred) for a in (4e-4, 2e-4, 1e-4)
    ]
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[0] / errs[2] > 3.0, errs


def test_the_tune_the_window_uses_is_measured_not_taken_from_the_lattice(tracked) -> None:
    """O5's trap, and on a steered ring it has two independent causes rather than one.

    The tracked tune differs from the design lattice's because the octupoles detune with
    amplitude **and** because the bump moves the linear tune through feed-down. Both are
    checked to matter: reading the sideband at the design tune degrades the answer badly,
    which is what makes measuring it a requirement rather than a refinement.
    """
    lat, tw, pred = tracked
    got, qx_measured = _tracked_f3000(lat, tw, 2.0e-4)
    qx_design = tunes(lat)[0]
    assert qx_measured != pytest.approx(qx_design, abs=1e-6)
    traj, _ = _betatron_turns(lat, 2.0e-4)
    h = _hx(traj, tw)
    at_design = complex(
        np.conj(_line(h, -2.0 * qx_design) / (6j * np.conj(_line(h, qx_design)) ** 2))
    )
    assert abs(got - pred) / abs(pred) < abs(at_design - pred) / abs(pred)
