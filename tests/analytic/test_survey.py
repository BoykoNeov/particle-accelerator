r"""R1 — the survey: where the machine actually sits, in laboratory coordinates.

Every optics number in this package is a displacement from a reference curve that has, until
now, simply been assumed to exist and to close. This module walks that curve through the
laboratory, and these are its gates.

**Closure has two halves and only one of them is nearly vacuous.** The *direction* closes
when the bend angles sum to ``2 pi``, which is a property of the fixture and of nothing else.
The *position* closing is a second condition — the step vectors must cancel too — and it is
**not** implied by the first: the obvious "no symmetry at all" fixture, six unequal bends
summing to ``2 pi`` with six unequal drifts, leaves the machine 2.1 m from where it started.
That is why the ring here has three-fold symmetry, which supplies position closure exactly
while leaving every row inside a cell distinct. Even so, both halves are satisfied by a
walker that replaces every arc with its chord and so sits wrong by the sagitta everywhere —
that control is
:func:`test_a_chord_walker_also_closes_which_is_why_closure_is_not_the_gate`, and it is why
the sharp tests are the per-element chord, the sagitta, and an element-by-element comparison
inside the cell.

**The independent leg is arithmetic, not a second copy.** The reference walk in
:func:`_complex_walk` is done in the complex plane — one multiplication per element, no
rotation matrices at all — so it shares no code path with the module's ``W @ dv``. On a
regular polygon that would prove little; inside this fixture's cell it pins every row.

**The blindness gates are the discriminating ones.** Since Q2 a bend's field ``k0`` and its
geometry ``angle/L`` are different numbers, so a survey that read the field would rescale the
entire ring **while still closing exactly** — the most plausible bug available here, and
``test_survey_is_blind_to_the_taper`` is the cheapest sharp gate against it. Both reference
codes were measured to be blind the same way before any of this was written.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_closed_orbit_6d import ring  # noqa: E402  (sibling fixture, as Q1's tests do)

from accsim.elements.dipole import Dipole
from accsim.elements.drift import Drift
from accsim.elements.quadrupole import Quadrupole, ThinQuadrupole
from accsim.geometry import survey
from accsim.lattice import Lattice
from accsim.reference import ReferenceParticle
from accsim.tapering import taper

ELECTRON_MASS_EV = 0.51099895069e6

#: A ring with just enough symmetry to close, and none inside the cell. Three cells, each
#: three bends of three *different* angles (halves, thirds and fifths of the cell's turn)
#: separated by three drifts of three different lengths.
#:
#: **The symmetry is not decoration — a ring does not close without it.** Angles summing to
#: ``2 pi`` close the *direction* and say nothing about the *position*: that needs the vector
#: sum of the steps to vanish as well, which is a condition on the lengths. Three-fold
#: symmetry supplies it exactly (``v + Rv + R^2 v = 0`` for a 120-degree rotation), while
#: leaving every row inside a cell at a different place — which is what a regular polygon
#: cannot offer.
PERIODS = 3
CELL_SHARES = (0.5, 0.3, 0.2)
BEND_ANGLES = tuple(2.0 * math.pi / PERIODS * f for f in CELL_SHARES) * PERIODS
BEND_LENGTHS = (1.0, 0.8, 1.3) * PERIODS
DRIFT_LENGTHS = (0.4, 0.9, 0.3) * PERIODS


def _ref() -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 1.0e9)


def uneven_ring() -> Lattice:
    """The three-cell ring described above: closed, and non-uniform inside the cell."""
    elements: list = []
    for angle, lb, ld in zip(BEND_ANGLES, BEND_LENGTHS, DRIFT_LENGTHS, strict=True):
        elements.append(Dipole(lb, angle))
        elements.append(Drift(ld))
    return Lattice(elements, ref=_ref())


def _complex_walk(lattice: Lattice) -> tuple[np.ndarray, np.ndarray]:
    r"""The same walk in the complex plane ``u = Z + i X`` — the independent leg.

    A straight element advances ``u`` by ``e * L``; a bend of angle ``a`` advances it by
    ``e * rho (sin a + i (cos a - 1))`` and rotates the direction by ``e *= exp(-i a)``.
    There is no rotation matrix anywhere in it, and it divides by the angle exactly where the
    module under test refuses to — so the two agree only if the geometry is right, not
    because they share an expression.
    """
    u, e = 0.0 + 0.0j, 1.0 + 0.0j
    Z, X = [0.0], [0.0]
    for elem in lattice.elements:
        length = float(elem.length)
        angle = float(getattr(elem, "angle", 0.0))
        if angle == 0.0:
            u += e * length
        else:
            rho = length / angle
            u += e * rho * (math.sin(angle) + 1j * (math.cos(angle) - 1.0))
            e *= complex(math.cos(angle), -math.sin(angle))
        Z.append(u.real)
        X.append(u.imag)
    return np.asarray(X), np.asarray(Z)


# ---------------------------------------------------------------------------
# Gate 1 — the per-element step is the arc's chord, and the sagitta with it.


@pytest.mark.parametrize("angle", [math.pi / 2, 1.0, 0.3, -0.7])
def test_one_bend_lands_on_the_closed_form(angle: float) -> None:
    """``rho (cos a - 1, 0, sin a)`` with ``rho = L/a``, and ``theta -> -a``.

    Only moderate angles are compared this way, and that is a statement about the *reference*
    expression rather than about the module: ``rho (cos a - 1)`` cancels catastrophically as
    the bend weakens, and at ``a = 1e-8`` it returns exactly zero for a displacement that is
    really ``-6.5e-9`` m. The weak-bend cases are gated against the series instead, below.
    """
    length = 1.3
    table = survey(Lattice([Dipole(length, angle)], ref=_ref()))
    rho = length / angle
    assert table.X[-1] == pytest.approx(rho * (math.cos(angle) - 1.0), rel=0.0, abs=1e-15)
    assert table.Z[-1] == pytest.approx(rho * math.sin(angle), rel=0.0, abs=1e-15)
    assert table.theta[-1] == pytest.approx(-angle, rel=0.0, abs=1e-16)
    assert table.s[-1] == pytest.approx(length, rel=0.0, abs=1e-16)


def test_the_sagitta_is_what_a_straight_walker_drops() -> None:
    """The transverse offset ``rho (1 - cos a)`` — the quantity closure is blind to.

    Gated on its own because the deliberate break below (every arc replaced by its chord)
    reproduces the *arc length* and the *closure* exactly and differs only here.
    """
    length, angle = 1.0, math.pi / 2
    table = survey(Lattice([Dipole(length, angle)], ref=_ref()))
    rho = length / angle
    assert -table.X[-1] == pytest.approx(rho * (1.0 - math.cos(angle)), rel=0.0, abs=1e-15)
    # Not a small correction: on a 90-degree bend the sagitta is the chord itself.
    assert -table.X[-1] == pytest.approx(0.6366197723675813, rel=0.0, abs=1e-15)


def test_a_weak_bend_does_not_cancel() -> None:
    """At ``a = 1e-8`` the chord is written without dividing by the curvature.

    ``rho (cos a - 1)`` evaluated literally loses everything to cancellation here — the two
    terms differ by ``5e-17`` — so the module's ``-L (a/2) sinc(a/2)^2`` form is checked
    against the series ``-L a/2 (1 - a^2/12)`` instead, which is exact to double precision.
    """
    length, angle = 2.0, 1e-8
    table = survey(Lattice([Dipole(length, angle)], ref=_ref()))
    expected = -length * angle / 2.0 * (1.0 - angle**2 / 12.0)
    assert table.X[-1] == pytest.approx(expected, rel=1e-14)
    assert table.Z[-1] == pytest.approx(length * (1.0 - angle**2 / 6.0), rel=1e-15)


def test_a_straight_element_advances_by_its_length_exactly() -> None:
    """Bit-exact, not approximately: the ``a = 0`` branch is the same expression."""
    lattice = Lattice(
        [Drift(1.0), Quadrupole(2.0, 0.3), ThinQuadrupole(0.4), Drift(4.0)], ref=_ref()
    )
    table = survey(lattice)
    assert list(table.Z) == [0.0, 1.0, 3.0, 3.0, 7.0]
    assert not np.any(table.X)
    assert not np.any(table.theta)


# ---------------------------------------------------------------------------
# Gate 2 — the composition gate, on the fixture that can see it.


def test_uneven_ring_matches_the_independent_complex_walk() -> None:
    """Element by element on the non-uniform ring, against a walk that shares no code."""
    table = survey(uneven_ring())
    X, Z = _complex_walk(uneven_ring())
    assert np.allclose(table.X, X, rtol=0.0, atol=1e-14)
    assert np.allclose(table.Z, Z, rtol=0.0, atol=1e-14)
    # And the fixture really is spread out — this is not agreement on a point.
    assert np.ptp(table.X) > 2.0
    assert np.ptp(table.Z) > 2.0


def _outgoing_frame_walk(lattice: Lattice) -> np.ndarray:
    """Deliberate break: the step expressed in the frame the element **leaves** in."""
    from accsim.geometry import _step, _yaw

    pos, theta = np.zeros(3), 0.0
    out = [pos.copy()]
    for elem in lattice.elements:
        angle = float(getattr(elem, "angle", 0.0))
        theta -= angle
        pos = pos + _yaw(theta) @ _step(float(elem.length), angle)
        out.append(pos.copy())
    return np.asarray(out)


def test_the_outgoing_frame_walker_closes_the_ring_and_is_wrong_from_the_first_row() -> None:
    """The step taken in the frame the element **leaves** in rather than the one it enters.

    It closes the ring — both the four-fold square and the three-fold fixture, to round-off —
    because a walker that repeats a cell under a rotation always closes, whatever the cell is.
    So closure does not see this at all. The rows do: it is 0.95 m out at the very first
    boundary and 1.9 m out at worst.
    """
    square = Lattice(
        [e for _ in range(4) for e in (Dipole(1.0, math.pi / 2), Drift(0.5))], ref=_ref()
    )
    assert np.linalg.norm(_outgoing_frame_walk(square)[-1]) < 1e-14

    uneven = uneven_ring()
    good, bad = survey(uneven).positions, _outgoing_frame_walk(uneven)
    assert np.linalg.norm(bad[-1] - bad[0]) < 1e-14  # it closes this ring too
    assert np.linalg.norm(bad[1] - good[1]) > 0.9  # and is wrong at the first boundary
    assert np.max(np.linalg.norm(bad - good, axis=1)) > 1.5


def _transposed_frame_walk(lattice: Lattice) -> np.ndarray:
    """Deliberate break: ``W`` transposed, i.e. the sign of ``theta`` flipped."""
    from accsim.geometry import _step, _yaw

    pos, theta = np.zeros(3), 0.0
    out = [pos.copy()]
    for elem in lattice.elements:
        angle = float(getattr(elem, "angle", 0.0))
        pos = pos + _yaw(theta).T @ _step(float(elem.length), angle)
        theta -= angle
        out.append(pos.copy())
    return np.asarray(out)


def test_the_transposed_frame_walker_closes_the_ring_and_is_wrong_from_the_second_row() -> None:
    """The other break of the same shape, and it is invisible one row longer.

    A transposed rotation is the identity at the start, so the **first** boundary is right;
    everything after it is a machine traversed the wrong way round, up to 8.4 m out. It, too,
    closes both rings to round-off — which is three separate walkers now that closure cannot
    tell apart, and the reason the entry calls that gate nearly vacuous.
    """
    square = Lattice(
        [e for _ in range(4) for e in (Dipole(1.0, math.pi / 2), Drift(0.5))], ref=_ref()
    )
    assert np.linalg.norm(_transposed_frame_walk(square)[-1]) < 1e-14

    uneven = uneven_ring()
    good, bad = survey(uneven).positions, _transposed_frame_walk(uneven)
    assert np.linalg.norm(bad[-1] - bad[0]) < 1e-14  # closes
    assert np.linalg.norm(bad[1] - good[1]) < 1e-15  # row 1 agrees: no turn composed yet
    assert np.linalg.norm(bad[2] - good[2]) > 0.5  # row 2 is where it shows
    assert np.max(np.linalg.norm(bad - good, axis=1)) > 5.0


# ---------------------------------------------------------------------------
# Gate 3 — closure, and the control that shows it is nearly vacuous.


def test_the_uneven_ring_closes() -> None:
    """Position and direction both return, to round-off."""
    table = survey(uneven_ring())
    assert table.closure_gap < 1e-14
    assert table.closure_angle < 1e-15
    assert table.total_angle == pytest.approx(2.0 * math.pi, rel=0.0, abs=1e-15)


def test_a_chord_walker_also_closes_which_is_why_closure_is_not_the_gate() -> None:
    r"""Replace every arc by a straight chord of the **same length ``L``** — still closes.

    This walker has no sagitta at all: it turns by the right angles in the right order, so
    its positions come back to the origin, while sitting up to 26 cm away from the true curve
    in between. Closure cannot tell the two apart; gate 1 can.
    """
    from accsim.geometry import _yaw

    lattice = uneven_ring()
    pos, theta = np.zeros(3), 0.0
    out = [pos.copy()]
    for elem in lattice.elements:
        angle = float(getattr(elem, "angle", 0.0))
        # a straight step of length L in the *mean* direction of the arc, then the full turn
        pos = pos + _yaw(theta - 0.5 * angle) @ np.array([0.0, 0.0, float(elem.length)])
        theta -= angle
        out.append(pos.copy())
    chord = np.asarray(out)

    assert np.linalg.norm(chord[-1] - chord[0]) < 1e-14  # closes to round-off
    gap = np.max(np.linalg.norm(chord - survey(lattice).positions, axis=1))
    assert gap > 0.05  # and is wrong by centimetres everywhere in between


# ---------------------------------------------------------------------------
# Gates 6 and 7 — planar by construction, and theta unwrapped.


def test_the_survey_is_exactly_planar() -> None:
    """``Y``, ``phi`` and ``psi`` are identically zero — asserted absolutely, not by ``rel=``.

    Nothing in ``accsim`` can bend out of the horizontal plane: ``roll`` is a misalignment
    (the magnet turns, the frame does not), not a design tilt. **This test fails the day
    accsim gains a design tilt**, which is the point of writing it — the refusal is recorded
    here, not only in the docs.
    """
    table = survey(uneven_ring())
    assert not np.any(table.Y)
    assert not np.any(table.phi)
    assert not np.any(table.psi)
    assert not any(hasattr(e, "tilt") for e in uneven_ring().elements)


def test_theta_is_not_wrapped() -> None:
    """A full turn reports ``-2 pi``, which is what both reference codes report."""
    table = survey(uneven_ring())
    assert table.theta[-1] == pytest.approx(-2.0 * math.pi, rel=0.0, abs=1e-15)
    assert np.all(np.diff(table.theta) <= 1e-16)  # monotone: it never jumps back


def test_the_frames_are_rotations_carrying_local_s_into_the_direction_of_travel() -> None:
    """``W`` is orthogonal with unit determinant, and its third column is the tangent."""
    table = survey(uneven_ring())
    for i in range(len(table)):
        W = table.W[i]
        assert np.allclose(W @ W.T, np.eye(3), rtol=0.0, atol=1e-15)
        assert np.linalg.det(W) == pytest.approx(1.0, rel=0.0, abs=1e-15)
        tangent = np.array([math.sin(table.theta[i]), 0.0, math.cos(table.theta[i])])
        assert np.allclose(W[:, 2], tangent, rtol=0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Gates 4 and 5 — what the survey must not see.


def test_survey_is_blind_to_the_taper() -> None:
    """``survey(taper(lattice)) == survey(lattice)`` on I4's ring, to the last bit.

    Q2 made a bend's field ``k0`` a different number from its geometry ``angle/L``, and a
    tapered ring differs from the design ring in ``k0`` **in every magnet**. A survey that
    read the field instead of the geometry would rescale the whole machine — and would still
    close exactly, which is why closure cannot stand in for this.
    """
    lattice, _ = ring()
    tapered = taper(lattice)

    # The fixture is doing what it claims: the fields really did move.
    moved = [
        abs(t.k0 - d.k0)
        for t, d in zip(tapered.elements, lattice.elements, strict=True)
        if isinstance(d, Dipole)
    ]
    assert len(moved) > 0 and max(moved) > 1e-6

    before, after = survey(lattice), survey(tapered)
    assert np.array_equal(before.X, after.X)
    assert np.array_equal(before.Z, after.Z)
    assert np.array_equal(before.theta, after.theta)


def test_survey_is_blind_to_the_pole_faces() -> None:
    """``e1``/``e2`` bound the field, not the reference curve — bit-identical tables."""
    plain = Lattice([Dipole(1.0, 0.4), Drift(0.5), Dipole(1.0, 0.4)], ref=_ref())
    faced = Lattice(
        [Dipole(1.0, 0.4, e1=0.3, e2=-0.2), Drift(0.5), Dipole(1.0, 0.4, e1=0.1, e2=0.25)],
        ref=_ref(),
    )
    a, b = survey(plain), survey(faced)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.theta, b.theta)


def test_survey_is_blind_to_misalignments() -> None:
    """A misalignment moves the magnet, not the curve the lattice is built around.

    A bending :class:`Dipole` refuses to be displaced (K2), so its misalignment leg here is
    ``roll``; the transverse offsets go on the quadrupole, where they are legal.
    """
    aligned = Lattice([Dipole(1.0, 0.4), Quadrupole(0.5, 0.3), Drift(0.2)], ref=_ref())
    misaligned = Lattice(
        [
            Dipole(1.0, 0.4, roll=0.05),
            Quadrupole(0.5, 0.3, dx=1e-3, dy=-2e-3, roll=0.1),
            Drift(0.2),
        ],
        ref=_ref(),
    )
    a, b = survey(aligned), survey(misaligned)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.theta, b.theta)


# ---------------------------------------------------------------------------
# The table's own shape — the one design decision this milestone makes.


def test_rows_are_boundaries_so_both_reference_conventions_are_slices() -> None:
    """``N + 1`` rows: ``[:-1]`` is xtrack's per-element table, ``[1:]`` is MAD-X's.

    The two codes report opposite ends of every magnet — measured, not assumed — and this is
    the object both are projections of.
    """
    lattice = uneven_ring()
    table = survey(lattice)
    n = len(lattice.elements)
    assert len(table) == n + 1
    assert len(table.names) == n
    assert table.s[0] == 0.0 and table.X[0] == 0.0 and table.Z[0] == 0.0
    assert table.s[-1] == pytest.approx(lattice.length, rel=0.0, abs=1e-14)
    # xtrack's row for element i is entrance i; MAD-X's is exit i. Both live here.
    assert table.Z[:-1].size == n and table.Z[1:].size == n
