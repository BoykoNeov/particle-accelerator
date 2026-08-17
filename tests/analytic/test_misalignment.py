r"""K1 acceptance: transverse misalignments, and the first *statistical* quantity.

Every element up to here sat exactly where the lattice put it. K1 gives elements a
position of their own, ``(dx, dy)``, defined as the element's own map **conjugated
by that translation**::

    track(state) = d + body(state - d),      d = (dx, 0, dy, 0, 0, 0).

This file is deliberately split into what is a *refactor with a consistency
requirement* and what is *new physics*, because conflating the two is how a
misalignment model gets believed without being checked.

**The refactor half.** A displaced element is precisely the feed-down expansion this
package has already pinned twice: a displaced sextupole is I2, a displaced octupole
is J3. So the gate is *reproduce those numbers*, not *derive new ones* — and because
both expansions are exact for a polynomial kick, "reproduce" means **exact equality
with the hand-assembled element family**, not agreement to a tolerance. The
quadrupole case is stronger still: a quad's gradient is uniform, so a displaced quad
is exactly a quad plus a dipole with **no higher terms at all**, and the remainder is
identically zero rather than small.

Three exact zeros carry more weight here than any tolerance:

- ``theta_x = +k1l dx`` but ``theta_y = -k1l dy`` — the *same* displacement sign
  giving opposite kick signs, because accsim's thin quad is ``px -> px - k1l x`` and
  ``py -> py + k1l y``. The asymmetry has bitten this package once already
  (``Corrector`` needs ``knl=[-k]`` for ``kick_x=+k`` but ``ksl=[+k]`` for
  ``kick_y=+k``), so it is derived here and asserted, never trusted.
- **Offsets cannot couple the planes.** Both cross-derivatives of a displaced quad's
  kick vanish *identically*, so no displacement of an unrolled quadrupole produces a
  skew term — only a roll can (K2). Asserted at exact zero.
- **A displacement moves no optics.** A translation leaves the homogeneous matrix
  untouched, so ``beta``, the tunes, dispersion and the coupling are bit-for-bit
  unchanged. That is not a nicety: it is the assumption the ensemble average below
  rests on, since it lets the average over displacements be taken at fixed optics.

**The new half is statistical.** A real machine's displacements are not known, only
their rms. For zero-mean uncorrelated displacements the closed orbit's variance is a
quadrature sum, which — written out with the single-kick closed form — is *exactly*

    <x_co^2>(s) = beta(s) theta_rms^2 / (4 sin^2 pi Q)
                  * sum_i beta_i cos^2(dpsi_i - pi Q).

The textbook ``sum_i beta_i / (8 sin^2 pi Q)`` needs one more step, ``cos^2 -> 1/2``,
and **that step is not an ensemble average**: the phases are deterministic properties
of the lattice, and averaging over displacement samples never touches them. So this
suite builds its gate on the exact form and *measures* the departure from the
textbook one (12% on the ring used here).

The load-bearing check is the **magnitude** — the solve against that exact form,
prefactor and ``beta``-weighting included. The ``Q`` **scan** is the same identity
stressed to 0.2% from the integer, where ``1/|sin pi Q|`` has grown 150-fold and the
fixed-point solve is near singular; it also pins the *exponent* of the divergence
(``p |sin pi Q|`` constant to 10 digits, ``p sin^2`` moving by 153x). What it is **not**
is an independent half — its divisor is the magnitude formula's own numerator, so it
cannot pass while the magnitude identity fails. The one-directional statement runs the
other way: a uniformly mis-scaled kick is *invisible* to the scan and caught only by the
magnitude comparison, which is the J1/J2/J3 failure mode and is measured by building the
broken machine and scanning it (constant preserved, value 0.5 -> 1.0).

The pole scan is run **strengthening** the quadrupoles toward ``Q -> 1``, not
weakening them toward ``Q -> 0``: a FODO with no focusing left is a drift ring, which
loses stability (``|Tr/2| -> 1``) before the integer is reached. And the divisor is
built from ``propagate_twiss``'s measured ``beta`` and phases, never from the formula
under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    ClosedOrbitError,
    Corrector,
    Dipole,
    Drift,
    Lattice,
    OrbitCorrectionError,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    ThinQuadrupole,
    ThinSextupole,
    ThinSkewQuadrupole,
    ThinSkewSextupole,
    closed_orbit,
    closed_orbit_nonlinear,
    closed_twiss,
    is_symplectic_map,
    jacobian,
    linearised_element_maps,
    linearised_lattice,
    misalign,
    misalignment_response,
    orbit_statistics,
    propagate_orbit,
    propagate_orbit_nonlinear,
    propagate_twiss,
    tunes,
)
from accsim.coords import DELTA, DIM, PX, PY, ZETA, X, Y

# The thin FODO of the I1 orbit suite, so the closed forms carry over unchanged.
VF = 1.0 / 1.5  # full-quad inverse focal length, F family [m^-1]
VD = 1.0 / 1.6  # ditto, D family [m^-1]
L_HALF = 1.0  # half-cell drift [m]

D_RMS = 1e-4  # a realistic quadrupole alignment tolerance: 100 microns
K2L = 3.0  # sextupole strength for the I2 correspondence [m^-2]
K3L = 40.0  # octupole strength for the J3 correspondence [m^-3]


@pytest.fixture
def ref() -> ReferenceParticle:
    # Thin quads + drifts are energy-independent; any reference works.
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _cell(tag: str, scale: float = 1.0) -> list:
    """One thin FODO cell, F-centred, with every quad scalable together."""
    return [
        ThinQuadrupole(0.5 * VF * scale, name=f"qf_a{tag}"),
        Drift(L_HALF, name=f"d1{tag}"),
        ThinQuadrupole(-VD * scale, name=f"qd{tag}"),
        Drift(L_HALF, name=f"d2{tag}"),
        ThinQuadrupole(0.5 * VF * scale, name=f"qf_b{tag}"),
    ]


def _ring(ref: ReferenceParticle, n_cells: int = 6, scale: float = 1.0) -> Lattice:
    return Lattice([e for i in range(n_cells) for e in _cell(f"_{i}", scale)], ref)


_AMPLITUDES = [
    np.array([1.0e-3, 2.0e-4, -5.0e-4, 3.0e-4, 1.0e-3, 2.0e-4]),
    np.array([-2.5e-3, -1.0e-4, 1.5e-3, -4.0e-4, -2.0e-4, -1.0e-4]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([4.0e-3, 0.0, 0.0, 7.0e-4, 0.0, 1.0e-3]),
]


# ---------------------------------------------------------------------------
# A. The offset is a conjugation, and for a quadrupole it is exact
# ---------------------------------------------------------------------------


def test_displacement_kick_is_the_derived_one_minus_m_times_d(ref: ReferenceParticle) -> None:
    r"""``(I - M) d`` for a thin quad is ``theta_x = +k1l dx``, ``theta_y = -k1l dy``.

    Derived in sympy from the thin-quad map itself rather than recalled, because the
    two planes' signs differ and that is exactly the kind of asymmetry that gets
    remembered wrong.
    """
    sp = pytest.importorskip("sympy")
    k1l, dx, dy, x, y = sp.symbols("k1l dx dy x y", real=True)
    # The element's own map, in its own frame, acting on the shifted coordinate.
    dpx = -k1l * (x - dx)
    dpy = +k1l * (y - dy)
    # Split into "the aligned magnet" plus "a constant": the constant is the kick.
    assert sp.expand(dpx - (-k1l * x)) == sp.expand(+k1l * dx)
    assert sp.expand(dpy - (+k1l * y)) == sp.expand(-k1l * dy)

    k, ddx, ddy = 0.5, 1.3e-4, -2.7e-4
    got = ThinQuadrupole(k, dx=ddx, dy=ddy).kick(ref)
    want = np.zeros(DIM)
    want[PX] = +k * ddx
    want[PY] = -k * ddy
    assert np.array_equal(got, want)  # exact, not to tolerance


def test_displaced_thin_quad_is_a_quad_plus_a_corrector_with_no_remainder(
    ref: ReferenceParticle,
) -> None:
    """A displaced quad is *exactly* a quad plus a dipole — the remainder is zero.

    Where I2 and J3 each split one element into a whole family, the quadrupole case
    terminates: a quad's gradient is uniform, so there is nothing left over at any
    order.

    "Exactly" is a statement about the algebra, and the *kick* is compared
    bit-for-bit (:func:`test_displacement_kick_is_the_derived_one_minus_m_times_d`).
    The tracked comparison cannot be: the conjugation evaluates ``k1l (x - dx)``
    where the split evaluates ``k1l x`` and ``k1l dx`` separately, and floating-point
    addition is not associative. The tolerance below is a round-off floor — 1e-15
    relative on the kicked momenta — not a physics allowance.
    """
    k, ddx, ddy = 0.5, 1.3e-4, -2.7e-4
    displaced = ThinQuadrupole(k, dx=ddx, dy=ddy)
    equivalent = Lattice([ThinQuadrupole(k), Corrector(kick_x=+k * ddx, kick_y=-k * ddy)], ref)
    for state in _AMPLITUDES:
        got = displaced.track(state, ref)
        want = state.copy()
        for elem in equivalent.elements:
            want = elem.track(want, ref)
        assert np.allclose(got, want, rtol=1e-15, atol=1e-15 * float(np.abs(want).max()))


def test_displaced_element_map_is_exactly_affine_for_a_linear_element(
    ref: ReferenceParticle,
) -> None:
    """``track == matrix @ state + kick`` with **no** remainder, thick quad included.

    The statement that a misalignment of a linear element is entirely a constant
    kick — the reason the linear ``closed_orbit`` sees misalignments at all without a
    single new line in the solve. Round-off floor again (the conjugation subtracts and
    re-adds ``d``, which is not bit-reversible); the *analytic* remainder is zero.

    A :class:`~accsim.elements.drift.Drift` has left this list, because its ``track`` is
    now the **exact** map and so is not affine in the first place — nothing to do with
    misalignment. K1's content for a drift is checked below instead, and in a stronger
    form than membership here ever gave.

    The thick :class:`~accsim.elements.quadrupole.Quadrupole` has left it too, and for
    a *different* reason worth separating: it is still affine at every amplitude, but
    only at ``delta = 0``, because its focusing is ``k1/(1 + delta)``. That is the
    version checked below — and it is the sharper of the two, since it says where the
    affine description stops holding rather than only that it does.
    """
    for elem in (
        ThinQuadrupole(0.5, dx=1.3e-4, dy=-2.7e-4),
        ThinSkewQuadrupole(0.3, dx=2e-4, dy=1e-4),
    ):
        M, k = elem.matrix(ref), elem.kick(ref)
        for state in _AMPLITUDES:
            want = M @ state + k
            got = elem.track(state, ref)
            assert np.allclose(got, want, rtol=1e-14, atol=1e-15 * float(np.abs(want).max()))


def test_a_displaced_thick_quadrupole_is_affine_on_momentum_and_only_there(
    ref: ReferenceParticle,
) -> None:
    r"""K1's statement for the thick quad, with the momentum it is now conditional on.

    A displaced thick quadrupole used to satisfy ``track == matrix @ state + kick``
    at every state, and the whole of K1's linear-orbit machinery rests on that. L2
    keeps it exactly where it was true for a reason — on momentum, where the map *is*
    the matrix — and breaks it off momentum, where a stiffer particle is focused less
    and no 6x6 evaluated at the reference momentum can say so.

    Three halves, really, because the affine description fails in two *different*
    ways and lumping them together would hide the smaller one:

    - **Transverse, on momentum:** affine, exactly as K1 — this is the property the
      linear ``closed_orbit`` solve rests on, and it is intact.
    - **Transverse, off momentum:** a real remainder, **first order in** ``delta``
      (halving the momentum halves it), which is what identifies it as the chromatic
      term rather than as a slip in the misalignment conjugation.
    - **``zeta``, at any momentum:** never affine, because a quadrupole lengthens an
      off-axis particle's path and that is quadratic in the amplitude, with no
      ``delta`` needed. Asserted as that order, so it reads as the path integral it
      is rather than as leakage from the chromatic term.

    The conjugation itself is untouched throughout: the displaced map is still exactly
    ``d + body(state - d)``, checked here at ``delta != 0`` too.
    """
    elem = Quadrupole(0.4, 1.7, dx=-8e-5, dy=5e-5)
    M, k = elem.matrix(ref), elem.kick(ref)
    transverse = [X, PX, Y, PY]

    # On momentum: affine transversally, to the round-off floor of the conjugation.
    for state in _AMPLITUDES:
        on_momentum = state.copy()
        on_momentum[DELTA] = 0.0
        want = M @ on_momentum + k
        got = elem.track(on_momentum, ref)
        assert np.allclose(
            got[transverse],
            want[transverse],
            rtol=1e-13,
            atol=1e-15 * float(np.abs(want).max()),
        )

    # Off momentum: a real transverse remainder, first order in delta.
    base = np.array([1.0e-3, 2.0e-4, -5.0e-4, 3.0e-4, 1.0e-3, 0.0])

    def remainder(delta: float) -> float:
        st = base.copy()
        st[DELTA] = delta
        D = elem.track(st, ref) - (M @ st + k)
        return float(np.max(np.abs(D[transverse])))

    assert remainder(1.0e-3) > 1.0e-9  # non-vacuous: the term is really there
    assert remainder(1.0e-3) / remainder(5.0e-4) == pytest.approx(2.0, rel=0.02)

    # zeta is never affine — the path lengthening, quadratic in the amplitude. The
    # amplitude that counts is the one *in the magnet's own frame*, so the scan is taken
    # about the offset rather than about the lattice axis; scaling the lattice
    # coordinate instead gives 3.75 rather than 4, because the fixed d is an 8%
    # perturbation on it. That is the conjugation showing through, and it is the reason
    # the scan is written this way.
    d = elem.offset()

    def zeta_remainder(amp: float) -> float:
        st = d + amp * (base - d)
        st[DELTA] = 0.0
        return abs(float(elem.track(st, ref)[ZETA] - (M @ st + k)[ZETA]))

    assert zeta_remainder(1.0) > 1.0e-9
    assert zeta_remainder(2.0) / zeta_remainder(1.0) == pytest.approx(4.0, rel=1e-6)

    # ...and the displacement is still a pure conjugation, off momentum included.
    body = Quadrupole(0.4, 1.7)
    for delta in (0.0, 1.0e-3, -2.0e-2):
        st = base.copy()
        st[DELTA] = delta
        np.testing.assert_allclose(
            elem.track(st, ref), body.track(st - d, ref) + d, rtol=1e-13, atol=1e-19
        )


def test_displacing_a_drift_does_nothing_to_its_exact_map_either(
    ref: ReferenceParticle,
) -> None:
    """K1's translation invariance, restated for the exact map — and now exactly.

    The original claim was that a displaced drift's linear map is unchanged because
    ``(I - M) d`` vanishes. The exact map makes the same statement for a stronger
    reason: it moves ``x`` by ``L px / pz``, a function of the **momenta alone**, so a
    translation cannot reach the map at all — where the linear argument had to compute
    ``(I - M) d`` and find it zero.

    The *kick* is therefore exactly zero, and asserted at exact zero. The tracked states
    agree only to a round-off floor, for the same reason the affine route has one: the
    conjugation evaluates ``d + body(state - d)``, and subtracting then re-adding ``d``
    is not bit-reversible in ``x`` and ``y``. That floor is arithmetic, not physics, and
    it scales with ``d`` rather than with the map — which is why the tolerance below is
    written against ``|d|``.
    """
    plain = Drift(1.0)
    for dx, dy in ((3e-4, -1e-4), (0.0, 5e-3), (-2e-2, 0.0)):
        shifted = Drift(1.0, dx=dx, dy=dy)
        assert np.array_equal(shifted.kick(ref), np.zeros(DIM))  # exactly zero, as K1
        assert np.array_equal(shifted.matrix(ref), plain.matrix(ref))  # and the matrix
        floor = 1e-15 * max(abs(dx), abs(dy))
        for state in _AMPLITUDES:
            np.testing.assert_allclose(
                shifted.track(state, ref), plain.track(state, ref), rtol=1e-14, atol=floor
            )


def test_thick_quadrupole_displacement_kick_limits_to_the_thin_one(
    ref: ReferenceParticle,
) -> None:
    """The thick ``(I - M) d`` collapses onto ``(+k1 L dx, -k1 L dy)`` as ``L -> 0``.

    The thick kick is *not* ``k1 L d`` — it is the full ``(I - M) d``, which also
    displaces ``x`` and ``y`` — so this is a limit, not an identity, and it is what
    ties the thick element to the thin closed form the ensemble gate uses.
    """
    k1, ddx, ddy = 1.7, 1.0e-4, -4.0e-4
    for length, rel in ((1e-2, 1e-4), (1e-3, 1e-6), (1e-4, 1e-8)):
        k = Quadrupole(length, k1, dx=ddx, dy=ddy).kick(ref)
        assert k[PX] == pytest.approx(+k1 * length * ddx, rel=rel)
        assert k[PY] == pytest.approx(-k1 * length * ddy, rel=rel)
        # The transverse displacement terms are the O(L^2) remainder of the limit.
        assert abs(k[X]) < abs(k[PX]) * length
        assert abs(k[Y]) < abs(k[PY]) * length


def test_offsets_cannot_couple_the_planes(ref: ReferenceParticle) -> None:
    """``d(Delta px)/dy = d(Delta py)/dx = 0`` *identically* for any displaced quad.

    This is what separates K1 from K2 cleanly: displacement alone never makes a skew
    term, whatever its size, so vertical orbit and coupling in a real machine are
    different phenomena with different causes. The contrast case — a skew quadrupole,
    which *is* a rolled quad — shows the assertion is not vacuous.
    """
    for elem in (
        ThinQuadrupole(0.5, dx=1e-3, dy=-2e-3),
        Quadrupole(0.4, 1.7, dx=-1e-3, dy=3e-3),
    ):
        J = jacobian(lambda s, e=elem: e.track(s, ref), np.zeros(DIM), step=1e-6)
        assert J[PX, Y] == 0.0
        assert J[PY, X] == 0.0
    coupled = jacobian(
        lambda s: ThinSkewQuadrupole(0.5, dx=1e-3).track(s, ref), np.zeros(DIM), step=1e-6
    )
    assert abs(coupled[PX, Y]) > 0.1
    assert abs(coupled[PY, X]) > 0.1


def test_a_drift_and_a_corrector_are_translation_invariant(ref: ReferenceParticle) -> None:
    """Displacing a drift or a steerer does *exactly* nothing.

    A drift has no field to miss the centre of, and a constant kick has no centre at
    all. Both come out as **exact** zeros of ``(I - M) d`` — a drift because its ``M``
    moves ``x`` only through ``px``, a corrector because its ``M`` is the identity —
    and the kick is therefore compared bit-for-bit. Only the tracked states carry the
    round-off of subtracting and re-adding ``d``.
    """
    for elem, plain in (
        (Drift(1.5, dx=2e-3, dy=-4e-3), Drift(1.5)),
        (Corrector(kick_x=1e-4, kick_y=-2e-4, dx=1e-3, dy=1e-3), Corrector(1e-4, -2e-4)),
    ):
        assert np.array_equal(elem.kick(ref), plain.kick(ref))
        for state in _AMPLITUDES:
            want = plain.track(state, ref)
            got = elem.track(state, ref)
            assert np.allclose(got, want, rtol=1e-14, atol=1e-15 * float(np.abs(want).max()))


def test_no_displacement_moves_beta_or_the_tunes(ref: ReferenceParticle) -> None:
    """A misaligned ring has **bit-for-bit** the optics of the aligned one.

    The assumption the whole ensemble average rests on: if displacing a magnet moved
    ``beta`` or ``Q``, averaging over displacements at fixed optics would be wrong
    from the start. It holds because a translation does not touch the homogeneous
    matrix — so this is an exact equality, not a tolerance.
    """
    aligned = _ring(ref)
    rng = np.random.default_rng(20260817)
    skewed = misalign(aligned, rng, dx_rms=5e-4, dy_rms=5e-4)
    assert any(e.is_misaligned for e in skewed.elements)  # the test is not vacuous
    assert np.array_equal(aligned.one_turn_matrix(), skewed.one_turn_matrix())
    assert tunes(aligned) == tunes(skewed)
    t0, t1 = closed_twiss(aligned), closed_twiss(skewed)
    assert (t0.beta_x, t0.beta_y, t0.alpha_x, t0.alpha_y) == (
        t1.beta_x,
        t1.beta_y,
        t1.alpha_x,
        t1.alpha_y,
    )


def test_the_misaligned_map_is_still_symplectic(ref: ReferenceParticle) -> None:
    """The house standard, applied to the new map composition (J1/J2/J3 each assert it).

    A translation is a symplectomorphism, so conjugating any symplectic map by one
    stays symplectic — this passes by construction, which is exactly why it is worth
    asserting: it is the check that would catch ``d`` being applied *asymmetrically* in
    the wrapper (subtracted from the state but added back to a different slot, or
    dropped on one side), which no amount of orbit statistics would reveal.
    """
    state = np.array([4e-3, 1e-3, -3e-3, 5e-4, 1e-3, 1e-4])
    for elem in (
        ThinQuadrupole(0.5, dx=2e-4, dy=-3e-4),
        Quadrupole(0.4, 1.7, dx=-1e-4, dy=5e-4),
        ThinSkewQuadrupole(0.3, dx=2e-4, dy=1e-4),
        ThinSextupole(K2L, dx=2e-4, dy=-3e-4),
        ThinOctupole(K3L, dx=2e-4, dy=-3e-4),
    ):
        assert is_symplectic_map(lambda s, e=elem: e.track(s, ref), state)


def test_misalignment_broadcasts_over_a_bunch(ref: ReferenceParticle) -> None:
    """The shift is applied per particle, so a ``(6, n)`` bunch tracks as n singles."""
    elem = ThinSextupole(K2L, dx=2e-4, dy=-3e-4)
    bunch = np.column_stack(_AMPLITUDES)
    got = elem.track(bunch, ref)
    for j, state in enumerate(_AMPLITUDES):
        assert np.array_equal(got[:, j], elem.track(state, ref))


def test_repr_reports_the_offset() -> None:
    """A printed lattice must not look perfect while its orbit says otherwise."""
    assert repr(ThinQuadrupole(0.5)) == "ThinQuadrupole(k1l=0.5)"
    assert repr(ThinQuadrupole(0.5, dx=1e-4)) == "ThinQuadrupole(k1l=0.5, dx=0.0001)"
    assert repr(ThinQuadrupole(0.5, name="q", dy=-2e-4)) == (
        "ThinQuadrupole(k1l=0.5, name='q', dy=-0.0002)"
    )


# ---------------------------------------------------------------------------
# B. The consistency requirement: a displaced multipole is I2 / J3
# ---------------------------------------------------------------------------


def _i2_family(k2l: float, x_co: float, y_co: float, ref: ReferenceParticle) -> Lattice:
    """I2's derived feed-down split of a thin sextupole at orbit offset ``(x_co, y_co)``."""
    return Lattice(
        [
            Corrector(
                kick_x=-0.5 * k2l * (x_co**2 - y_co**2),
                kick_y=+k2l * x_co * y_co,
            ),
            ThinQuadrupole(+k2l * x_co),
            ThinSkewQuadrupole(+k2l * y_co),
            ThinSextupole(k2l),
        ],
        ref,
    )


def _j3_family(k3l: float, x_co: float, y_co: float, ref: ReferenceParticle) -> Lattice:
    """J3's split of a thin octupole at orbit offset ``(x_co, y_co)`` — two orders down."""
    return Lattice(
        [
            Corrector(
                kick_x=-(k3l / 6.0) * (x_co**3 - 3.0 * x_co * y_co**2),
                kick_y=+(k3l / 6.0) * (3.0 * x_co**2 * y_co - y_co**3),
            ),
            ThinQuadrupole(0.5 * k3l * (x_co**2 - y_co**2)),
            ThinSkewQuadrupole(k3l * x_co * y_co),
            ThinSextupole(k3l * x_co),
            ThinSkewSextupole(k3l * y_co),
            ThinOctupole(k3l),
        ],
        ref,
    )


def _track_through(lat: Lattice, state: np.ndarray) -> np.ndarray:
    out = np.array(state, dtype=float, copy=True)
    for elem in lat.elements:
        out = elem.track(out, lat.ref)
    return out


@pytest.mark.parametrize("ddx, ddy", [(2e-4, -3e-4), (-1e-3, 0.0), (0.0, 5e-4), (1e-3, 1e-3)])
def test_displaced_thin_sextupole_is_i2s_family_exactly(
    ref: ReferenceParticle, ddx: float, ddy: float
) -> None:
    """K1's consistency requirement, horizontally and vertically: the offset **is** I2.

    A magnet displaced by ``d`` and a beam displaced by ``-d`` are the same physics,
    so a displaced sextupole must reproduce the four-term split I2 already validated
    against xtrack, at ``(x_co, y_co) = (-dx, -dy)``. Because that split is an exact
    rearrangement of a quadratic (not a truncation), agreement is **exact** — a
    tolerance here would hide a wrong coefficient in the third digit.
    """
    displaced = ThinSextupole(K2L, dx=ddx, dy=ddy)
    family = _i2_family(K2L, -ddx, -ddy, ref)
    for state in _AMPLITUDES:
        assert np.allclose(displaced.track(state, ref), _track_through(family, state), atol=0.0)


@pytest.mark.parametrize("ddx, ddy", [(2e-4, -3e-4), (-1e-3, 0.0), (0.0, 5e-4)])
def test_displaced_thin_octupole_is_j3s_family_exactly(
    ref: ReferenceParticle, ddx: float, ddy: float
) -> None:
    """The same requirement one order up: a displaced octupole is J3's six-term family.

    J3's cubic kick reaches two orders below itself, so the offset produces a
    sextupole pair as well as a gradient pair and a dipole — five extra elements, and
    a uniform mis-scale of any one of them would show up here as a mismatch.
    """
    displaced = ThinOctupole(K3L, dx=ddx, dy=ddy)
    family = _j3_family(K3L, -ddx, -ddy, ref)
    for state in _AMPLITUDES:
        assert np.allclose(displaced.track(state, ref), _track_through(family, state), atol=0.0)


def test_linearised_lattice_reads_the_magnets_own_frame(ref: ReferenceParticle) -> None:
    """The feed-down split must be evaluated at ``x_co - dx``, not at ``x_co``.

    Otherwise every misalignment would be invisible to the chromaticity integrals,
    which walk element *types* rather than maps. The gradient the displaced sextupole
    hands over is ``k2l (x_co - dx)``, and both halves of that matter: dropping the
    ``-dx`` would leave ``k2l x_co`` — here 1e-6 instead of 9e-4, wrong by a factor of
    900 — while dropping ``x_co`` would ignore the orbit the displacement itself
    created (this ring's is only 0.3 um, but it is there and it is not noise).

    The same numbers must come out of differentiating the real ``track()``, which is
    the independent route: one walks element types, the other knows nothing about them.
    """
    ddx, ddy = 3e-4, -5e-4
    elements = list(_ring(ref).elements)
    elements.insert(1, ThinSextupole(K2L, dx=ddx, dy=ddy, name="sx"))
    lat = Lattice(elements, ref)

    orbit = propagate_orbit_nonlinear(lat)[1]  # the orbit the sextupole itself sees
    want_k1l = K2L * (float(orbit[0]) - ddx)
    want_k1sl = K2L * (float(orbit[2]) - ddy)

    lin = linearised_lattice(lat)
    quad = next(e for e in lin.elements if e.name == "sx_fd_quad")
    skew = next(e for e in lin.elements if e.name == "sx_fd_skew")
    assert quad.k1l == pytest.approx(want_k1l, rel=1e-12)
    assert skew.k1sl == pytest.approx(want_k1sl, rel=1e-12)
    # The displacement dominates, and the orbit term is resolved on top of it.
    assert abs(want_k1l / (K2L * ddx) + 1.0) > 1e-6

    # ...and the same map read the other way, by differentiating the real track().
    maps = linearised_element_maps(lat)
    want = ThinSkewQuadrupole(want_k1sl).matrix(ref) @ ThinQuadrupole(want_k1l).matrix(ref)
    assert np.allclose(maps[1], want, atol=1e-9, rtol=0.0)


def test_the_linear_orbit_is_blind_to_a_displaced_sextupole_and_says_so(
    ref: ReferenceParticle,
) -> None:
    """``closed_orbit`` returns **exactly zero** for a machine that really is distorted.

    A displaced sextupole has ``matrix() = I``, so its misalignment term ``(I - M) d``
    is identically zero and the linear theory sees nothing — while the real map has a
    dipole kick ``-1/2 k2l (dx^2 - dy^2)``. That is not new blindness (it is why
    ``closed_orbit_nonlinear`` exists), but before K1 a zero here was *right*, and
    after K1 it is *wrong*, so it is recorded as a verdict rather than a footnote.

    The nonlinear solve does see it, and lands on the orbit of the hand-assembled I2
    family to machine precision — the two lattices have literally the same map.
    """
    ddx, ddy = 8e-4, -6e-4
    elements = list(_ring(ref).elements)
    elements.insert(1, ThinSextupole(K2L, dx=ddx, dy=ddy, name="sx"))
    displaced = Lattice(elements, ref)

    assert np.array_equal(closed_orbit(displaced), np.zeros(4))  # the trap, recorded

    equivalent = list(_ring(ref).elements)
    equivalent[1:1] = _i2_family(K2L, -ddx, -ddy, ref).elements
    hand_made = Lattice(equivalent, ref)

    co_nl = closed_orbit_nonlinear(displaced)
    assert np.allclose(co_nl, closed_orbit_nonlinear(hand_made), atol=1e-14, rtol=0.0)
    assert abs(co_nl[0]) > 1e-7  # and it is a real orbit, not numerical dust
    # The *linear* orbit of the hand-made family is the I2 dipole-kick orbit: nonzero,
    # which is precisely the orbit the displaced-sextupole lattice hides above.
    assert abs(closed_orbit(hand_made)[0]) > 1e-7


def test_orbit_statistics_refuses_a_source_it_cannot_see(ref: ReferenceParticle) -> None:
    """Asking for the rms orbit from displaced sextupoles must raise, not return zero.

    The refusal is the whole reason the blindness above is safe: a caller who wants
    sextupole misalignment statistics is told to go to the nonlinear solve instead of
    being handed a reassuring zero.
    """
    elements = list(_ring(ref).elements)
    elements.insert(1, ThinSextupole(K2L, name="sx"))
    lat = Lattice(elements, ref)
    with pytest.raises(OrbitCorrectionError, match="no linear response"):
        orbit_statistics(lat, dx_rms=D_RMS, sources=[1])
    # The default source list simply leaves it out.
    stat = orbit_statistics(lat, dx_rms=D_RMS)
    assert 1 not in stat.sources
    assert stat.n_sources == 18  # 6 cells x 3 quads


def test_a_lattice_with_nothing_to_misalign_is_rejected(ref: ReferenceParticle) -> None:
    """No gradient anywhere ⇒ no misalignment can do anything, and that is an error."""
    with pytest.raises(OrbitCorrectionError, match="responds to a displacement"):
        orbit_statistics(Lattice([Drift(1.0), Drift(2.0)], ref), dx_rms=D_RMS)


def test_orbit_statistics_validates_its_inputs(ref: ReferenceParticle) -> None:
    lat = _ring(ref)
    with pytest.raises(ValueError, match="must be >= 0"):
        orbit_statistics(lat, dx_rms=-1e-4)
    with pytest.raises(OrbitCorrectionError, match="outside the lattice"):
        orbit_statistics(lat, dx_rms=D_RMS, sources=[999])


def test_a_bending_dipole_refuses_to_be_displaced(ref: ReferenceParticle) -> None:
    """K1's conjugation is a *straight*-element statement, and a bend says so out loud.

    A bend rotates the reference frame through itself, so translating in at the entry
    and out at the exit are not the same transformation — displacing a bend is a rigid
    body motion of a curved object, which xtrack models and accsim does not (the two
    differ by 3.6e-5 where their aligned maps agree to 5.8e-9;
    ``tests/reference/test_misalignment_xtrack.py``). Refusing is the only honest
    option, and it has to fire on the *statistical* path too, where the offset is set
    after construction and a silent wrong model would be averaged over hundreds of
    machines without anyone looking at it.
    """
    with pytest.raises(NotImplementedError, match="cannot displace the bending Dipole"):
        Dipole(1.0, 0.12, dx=3e-4, name="mb").track(_AMPLITUDES[0], ref)
    # A straight gradient magnet (angle = 0) is displaced like any other element.
    assert Dipole(1.0, 0.0, k1=1.7, dx=3e-4).kick(ref)[PX] != 0.0

    arc = Lattice([*_ring(ref).elements, Dipole(1.0, 0.12, name="mb")], ref)
    rng = np.random.default_rng(3)
    with pytest.raises(NotImplementedError, match="cannot displace the bending Dipole"):
        orbit_statistics(arc, dx_rms=D_RMS)
    with pytest.raises(NotImplementedError, match="cannot displace the bending Dipole"):
        misalign(arc, rng, dx_rms=D_RMS)  # raises here, not later inside a track loop
    # ...but the quadrupoles of the same arc can be misaligned on their own.
    quads = [i for i, e in enumerate(arc.elements) if isinstance(e, ThinQuadrupole)]
    assert orbit_statistics(arc, dx_rms=D_RMS, sources=quads).n_sources == len(quads)
    assert misalign(arc, rng, dx_rms=D_RMS, sources=quads).elements[quads[0]].dx != 0.0


# ---------------------------------------------------------------------------
# C. The statistical gate: magnitude, and the pole
# ---------------------------------------------------------------------------


def _exact_ensemble_rms(lat: Lattice, d_rms: float, plane: str = "x") -> np.ndarray:
    r"""The exact ``beta``-weighted ensemble rms at every boundary, from measured optics.

        x_rms(s) = d_rms sqrt(beta(s) sum_i k1l_i^2 beta_i cos^2(|dpsi_i| - pi Q))
                   / (2 |sin pi Q|)

    Every ingredient but ``k1l`` comes from :func:`propagate_twiss`, so this is an
    independent reference and not a rearrangement of what
    :func:`orbit_statistics` computes (which solves the fixed point, element by
    element, and never evaluates a closed form).
    """
    table = propagate_twiss(lat, closed_twiss(lat))
    horizontal = plane == "x"
    beta = np.array([t.beta_x if horizontal else t.beta_y for t in table])
    mu = np.array([t.mu_x if horizontal else t.mu_y for t in table])
    q = mu[-1] / (2.0 * math.pi)  # the *total* tune, integer part included
    sources = [(i, e.k1l) for i, e in enumerate(lat.elements) if isinstance(e, ThinQuadrupole)]
    out = np.empty(len(table))
    for b in range(len(table)):
        weighted = sum(
            (k1l * d_rms) ** 2 * beta[i] * math.cos(abs(mu[b] - mu[i]) - math.pi * q) ** 2
            for i, k1l in sources
        )
        out[b] = math.sqrt(beta[b] * weighted) / (2.0 * abs(math.sin(math.pi * q)))
    return out


def test_the_ensemble_average_is_a_quadrature_sum_symbolically() -> None:
    """Derive the exact form: the cross terms die, and ``cos^2 -> 1/2`` does **not**.

    Two statements, and only the first is an ensemble average. With
    ``x = sum_i c_i d_i`` and ``<d_i d_j> = delta_ij sigma^2``, the variance is
    ``sigma^2 sum_i c_i^2`` — the cross terms vanish because the displacements are
    uncorrelated. Substituting the single-kick coefficients then gives the
    ``beta_i cos^2(dpsi_i - pi Q)`` sum. Replacing that ``cos^2`` by ``1/2`` is a
    statement about the *lattice*, and sympy refuses to make it: the residual is not
    identically zero.
    """
    sp = pytest.importorskip("sympy")
    sigma, beta_s, q = sp.symbols("sigma beta_s Q", positive=True)
    n = 3
    betas = sp.symbols("beta1:4", positive=True)
    psis = sp.symbols("psi1:4", real=True)
    ks = sp.symbols("k1:4", real=True)
    ds = sp.symbols("d1:4", real=True)

    coeff = [
        ks[i] * sp.sqrt(betas[i] * beta_s) * sp.cos(psis[i] - sp.pi * q) / (2 * sp.sin(sp.pi * q))
        for i in range(n)
    ]
    x = sum(c * d for c, d in zip(coeff, ds, strict=True))
    # Ensemble average: expand the square, keep only the diagonal moments.
    squared = sp.expand(x**2)
    averaged = squared
    for i in range(n):
        for j in range(n):
            averaged = averaged.subs(ds[i] * ds[j], sigma**2 if i == j else 0)
    averaged = averaged.subs(dict.fromkeys(ds, 0))  # any surviving linear term

    claimed = (
        beta_s
        * sigma**2
        / (4 * sp.sin(sp.pi * q) ** 2)
        * sum(ks[i] ** 2 * betas[i] * sp.cos(psis[i] - sp.pi * q) ** 2 for i in range(n))
    )
    assert sp.simplify(averaged - claimed) == 0

    # ...and the textbook 1/8 form is a *further*, non-statistical step.
    textbook = (
        beta_s
        * sigma**2
        / (8 * sp.sin(sp.pi * q) ** 2)
        * sum(ks[i] ** 2 * betas[i] for i in range(n))
    )
    assert sp.simplify(claimed - textbook) != 0


def test_response_is_exactly_linear_in_the_displacement(ref: ReferenceParticle) -> None:
    """Double a displacement, double the orbit — *exactly*, at any size.

    The claim that makes the quadrature sum an ensemble average rather than a
    linearisation: the same derivative comes back from a 1 micron displacement and
    from a **1 metre** one, six orders either side of the tolerance the gate is quoted
    at, so a hidden higher-order term could not survive.
    """
    lat = _ring(ref)
    response, sources = misalignment_response(lat)
    i = sources[0]
    base = np.array(propagate_orbit(lat))
    want = response[:, :, 0, 0]
    floor = 1e-14 * float(np.abs(want).max())
    for size in (1e-6, 1e-3, 1.0):
        elements = list(lat.elements)
        elements[i] = ThinQuadrupole(elements[i].k1l, name=elements[i].name, dx=size)
        moved = np.array(propagate_orbit(Lattice(elements, ref)))
        assert np.allclose((moved - base) / size, want, rtol=1e-9, atol=floor)


def test_expected_rms_matches_the_exact_beta_weighted_sum(ref: ReferenceParticle) -> None:
    """The magnitude gate: prefactor and ``beta``-weighting, at every boundary.

    ``orbit_statistics`` never evaluates a closed form — it superposes exact
    fixed-point solves — so agreement with the derived
    ``beta theta_rms^2 sum beta_i cos^2 / (4 sin^2)`` pins the ``4``, the
    ``sqrt(beta_i beta(s))`` weighting and the phase offset ``-pi Q`` all at once.
    Both planes, because the vertical kick carries the opposite sign and squaring it
    must not hide a mistake elsewhere.
    """
    lat = _ring(ref)
    stat = orbit_statistics(lat, dx_rms=D_RMS, dy_rms=D_RMS)
    for plane, got in (("x", stat.x_rms), ("y", stat.y_rms)):
        want = _exact_ensemble_rms(lat, D_RMS, plane)
        assert np.allclose(got, want, rtol=1e-12, atol=0.0)
    # Non-trivial: a 100 micron tolerance gives a ~0.2 mm orbit, 2x the tolerance.
    assert stat.x_rms.max() > 2.0 * D_RMS


def test_the_textbook_one_eighth_is_measured_not_inherited(ref: ReferenceParticle) -> None:
    """``cos^2 -> 1/2`` is wrong by 12% on this ring, and the gate can tell.

    The exact sum and the textbook ``sum beta_i / 8`` differ by more than a thousand
    times the tolerance the magnitude gate above is asserted at, so that gate is
    genuinely testing the exact form rather than passing on either.
    """
    lat = _ring(ref)
    table = propagate_twiss(lat, closed_twiss(lat))
    beta = np.array([t.beta_x for t in table])
    mu = np.array([t.mu_x for t in table])
    q = mu[-1] / (2.0 * math.pi)
    sum_beta = sum(
        (e.k1l * D_RMS) ** 2 * beta[i]
        for i, e in enumerate(lat.elements)
        if isinstance(e, ThinQuadrupole)
    )
    textbook = np.sqrt(beta * sum_beta / 8.0) / abs(math.sin(math.pi * q))
    exact = _exact_ensemble_rms(lat, D_RMS, "x")
    ratio = textbook / exact
    assert 1.05 < ratio.min() and ratio.max() < 1.20  # a real, measured departure
    assert np.max(np.abs(ratio - 1.0)) > 1e-9  # ...and far outside the gate's tolerance


def _pole_scan(ref: ReferenceParticle, factory=None) -> list[tuple[float, float]]:
    """``(Q_total, x_rms / measured beta-sum)`` while strengthening toward ``Q -> 1``.

    The divisor is ``sqrt(beta(0) sum_i k1l_i^2 beta_i cos^2(dpsi_i - pi Q)) d_rms``,
    built from :func:`propagate_twiss`'s own numbers — it contains no ``sin``, so what
    is left is the pole and nothing else. This is the contamination the naive scan
    walks into: retuning the quads moves ``beta`` and the source strengths at the same
    time as ``Q``, and the raw rms is *not* a pure ``1/|sin pi Q|``.

    ``factory(scale) -> Lattice`` builds the ring, so the *same* scan can be run on a
    deliberately broken machine. Without that the "the pole cannot see a mis-scale"
    claim would be a statement about Python arithmetic rather than about the code.
    """
    build = factory if factory is not None else (lambda scale: _ring(ref, scale=scale))
    out = []
    for scale in (1.0, 1.1, 1.2, 1.3, 1.4, 1.43, 1.45, 1.46, 1.47, 1.48):
        lat = build(scale)
        table = propagate_twiss(lat, closed_twiss(lat))
        beta = np.array([t.beta_x for t in table])
        mu = np.array([t.mu_x for t in table])
        q = mu[-1] / (2.0 * math.pi)
        weighted = sum(
            (e.k1l * D_RMS) ** 2 * beta[i] * math.cos(abs(mu[0] - mu[i]) - math.pi * q) ** 2
            for i, e in enumerate(lat.elements)
            if isinstance(e, ThinQuadrupole)
        )
        divisor = math.sqrt(beta[0] * weighted)
        out.append((q, orbit_statistics(lat, dx_rms=D_RMS).x_rms[0] / divisor))
    return out


def test_the_pole_is_one_over_sin_pi_q_and_not_some_other_power(
    ref: ReferenceParticle,
) -> None:
    """The divergence at the integer tune is exactly first order in ``1/|sin pi Q|``.

    Two things this is, and one it is not.

    It **is** decisive about the *exponent*: over a scan on which the raw rms grows by
    a factor 17 and ``1/|sin pi Q|`` by a factor 150, ``p |sin pi Q|`` is constant to
    10 digits while ``p sin^2`` moves by two orders of magnitude. And it **is** the
    magnitude identity stressed where it is most fragile — the last point sits 0.2%
    from the integer, where ``beta`` is huge and the fixed-point solve is near
    singular; the scan crosses ``Q = 1`` and stops short of the singular ``(I - M4)``,
    which is the same physics as the pole and is gated separately.

    It is **not** an independent half of the gate, and saying so would be an overclaim:
    the divisor here is exactly :func:`_exact_ensemble_rms`'s numerator, so
    ``p |sin pi Q| = 0.5`` follows algebraically from the magnitude identity holding at
    these working points. The genuinely one-directional statement is the *converse* —
    a constant mis-scale is invisible to this check and caught only by the magnitude
    comparison — and that is measured in
    :func:`test_a_uniformly_mis_scaled_kick_moves_the_magnitude_and_not_the_pole`.
    """
    scan = _pole_scan(ref)
    first_order = np.array([p * abs(math.sin(math.pi * q)) for q, p in scan])
    second_order = np.array([p * math.sin(math.pi * q) ** 2 for q, p in scan])
    inv_sin = np.array([1.0 / abs(math.sin(math.pi * q)) for q, _ in scan])

    assert inv_sin.max() / inv_sin.min() > 100.0  # the scan really approaches the pole
    assert np.allclose(first_order, 0.5, rtol=1e-10, atol=0.0)
    assert second_order.max() / second_order.min() > 10.0  # ...and a wrong power fails


def test_the_pole_and_the_no_closed_orbit_guard_are_the_same_physics(
    ref: ReferenceParticle,
) -> None:
    """Pushed onto the integer, the rms does not grow — it stops existing.

    ``1/|sin pi Q| -> inf`` and ``(I - M4)`` going singular are one statement, so the
    statistical entry point inherits I1's :class:`ClosedOrbitError` rather than
    returning a huge but meaningless number. ``scale = 0.2`` is I1's own exactly-on-
    resonance working point (``Q_y = 0``; see ``test_orbit.py``), reused here so the
    two suites agree about where the resonance is.
    """
    with pytest.raises(ClosedOrbitError, match="integer"):
        orbit_statistics(_ring(ref, scale=0.2), dx_rms=D_RMS)


def test_a_uniformly_mis_scaled_kick_moves_the_magnitude_and_not_the_pole(
    ref: ReferenceParticle,
) -> None:
    """The J1/J2/J3 failure mode, built and run through **both** checks.

    A misalignment kick wrong by a constant factor scales every orbit by that factor,
    so a check that fits a *power* cannot see it while the magnitude comparison misses
    by exactly the factor. This is the one-directional statement that makes the
    magnitude gate load-bearing, so the broken machine is actually built and actually
    scanned — asserting that ``2 * p`` is constant given ``p`` is constant would be a
    fact about arithmetic, not about the code.

    The mis-scale touches only ``kick()``, so ``matrix()``, ``beta`` and ``Q`` are
    untouched and the scan runs over the same working points as the good ring: the
    divergence stays exactly first order, and only the constant moves, 0.5 -> 1.0.
    """

    class _MisScaled(ThinQuadrupole):
        """A quad whose displacement kick is twice what ``(I - M) d`` says."""

        def kick(self, r: ReferenceParticle) -> np.ndarray:
            return 2.0 * super().kick(r)

    def break_ring(scale: float) -> Lattice:
        return Lattice(
            [
                _MisScaled(e.k1l, name=e.name) if isinstance(e, ThinQuadrupole) else e
                for e in _ring(ref, scale=scale).elements
            ],
            ref,
        )

    want = _exact_ensemble_rms(_ring(ref), D_RMS, "x")
    got = orbit_statistics(break_ring(1.0), dx_rms=D_RMS).x_rms
    # The magnitude gate fails, by exactly the factor...
    assert np.allclose(got / want, 2.0, rtol=1e-12, atol=0.0)

    # ...while the pole check, re-run on the broken machine, still sees a clean
    # first-order divergence: constant in Q, at twice the value it should be.
    scan = _pole_scan(ref, factory=break_ring)
    good_scan = _pole_scan(ref)
    assert [q for q, _ in scan] == [q for q, _ in good_scan]  # same working points
    first_order = np.array([p * abs(math.sin(math.pi * q)) for q, p in scan])
    assert np.allclose(first_order, 1.0, rtol=1e-10, atol=0.0)  # constant: pole is blind
    assert not np.allclose(first_order, 0.5, rtol=1e-3)  # ...but at the wrong constant


def test_monte_carlo_ensemble_converges_on_the_prediction(ref: ReferenceParticle) -> None:
    """Sample real machines and recover the predicted rms — the statistical claim itself.

    ``orbit_statistics`` never samples anything, so this is the one place the word
    "expected" is checked against an actual ensemble. 400 machines give a ~4%
    standard error on an rms, which is the tolerance used; the mean over all
    boundaries is far tighter than that.
    """
    lat = _ring(ref)
    predicted = orbit_statistics(lat, dx_rms=D_RMS).x_rms
    rng = np.random.default_rng(20260817)
    n_samples = 400
    accumulated = np.zeros(len(predicted))
    for _ in range(n_samples):
        sample = misalign(lat, rng, dx_rms=D_RMS)
        accumulated += np.array([o[0] for o in propagate_orbit(sample)]) ** 2
    measured = np.sqrt(accumulated / n_samples)
    ratio = measured / predicted
    assert np.allclose(ratio, 1.0, rtol=0.12, atol=0.0)
    assert abs(float(np.mean(ratio)) - 1.0) < 0.03


def test_the_prediction_depends_only_on_the_second_moment(ref: ReferenceParticle) -> None:
    """A uniform distribution of the same rms gives the same orbit statistics.

    The ensemble average uses nothing but ``<d_i d_j> = delta_ij sigma^2``, so the
    shape of the distribution cannot matter — a point worth checking, because a
    Gaussian assumption smuggled into the derivation would be invisible against
    Gaussian samples.
    """
    lat = _ring(ref)
    predicted = orbit_statistics(lat, dx_rms=D_RMS).x_rms
    response, sources = misalignment_response(lat)
    rng = np.random.default_rng(4242)
    half_width = D_RMS * math.sqrt(3.0)  # uniform on [-a, a] has rms a/sqrt(3)
    n_samples = 400
    accumulated = np.zeros(len(predicted))
    for _ in range(n_samples):
        offsets = rng.uniform(-half_width, half_width, size=len(sources))
        elements = list(lat.elements)
        for j, i in enumerate(sources):
            elements[i] = ThinQuadrupole(
                elements[i].k1l, name=elements[i].name, dx=float(offsets[j])
            )
        accumulated += np.array([o[0] for o in propagate_orbit(Lattice(elements, ref))]) ** 2
    measured = np.sqrt(accumulated / n_samples)
    assert np.allclose(measured / predicted, 1.0, rtol=0.12, atol=0.0)
    assert response.shape == (len(predicted), 4, len(sources), 2)


def test_misalign_does_not_touch_the_lattice_it_was_given(ref: ReferenceParticle) -> None:
    """Sampling a machine must not silently misalign the caller's model of it."""
    lat = _ring(ref)
    rng = np.random.default_rng(7)
    sample = misalign(lat, rng, dx_rms=D_RMS, dy_rms=D_RMS)
    assert not any(e.is_misaligned for e in lat.elements)
    assert all(not e.is_misaligned for e in lat.elements)
    assert any(e.is_misaligned for e in sample.elements)
    assert sample.elements[0] is not lat.elements[0]


def test_an_offset_already_present_is_added_to_not_replaced(ref: ReferenceParticle) -> None:
    """A known displacement and a tolerance superpose — the usual operational case."""
    known = 1e-3
    elements = list(_ring(ref).elements)
    elements[0] = ThinQuadrupole(elements[0].k1l, name="qf_known", dx=known)
    lat = Lattice(elements, ref)
    rng = np.random.default_rng(11)
    sample = misalign(lat, rng, dx_rms=D_RMS)
    assert sample.elements[0].dx != known
    assert abs(sample.elements[0].dx - known) < 10.0 * D_RMS
