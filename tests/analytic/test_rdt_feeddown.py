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

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    ThinSextupole,
    ThinSkewQuadrupole,
    ThinSkewSextupole,
)
from accsim.coords import PX, PY, X, Y

MASS0, GAMMA0 = 938.27208816e6, 20.0

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
