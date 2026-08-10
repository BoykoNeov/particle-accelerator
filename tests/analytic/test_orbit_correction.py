"""I1 acceptance (part 2): the orbit response matrix and SVD steering.

The closed orbit is **exactly affine** in the corrector kicks (part 1 gates that
to 1e-18), so ``orbit(theta) = orbit(0) + R theta`` holds for any kick with no
truncation error. Two consequences drive everything here:

- the response matrix ``R`` is *exact*, not a finite difference, and correction
  is **one linear solve** rather than an iteration — the same structural fact
  that makes ``match_chromaticity`` an exact solve;
- and a wrong ``R`` therefore produces a wrong *answer*, not merely slow
  convergence — provided the reported result is **measured**. A residual
  evaluated as ``x0 + R dtheta`` would be perfectly zero for any invertible
  ``R``, right or wrong, so :func:`correct_orbit` re-solves the closed orbit of
  the corrected lattice. That is gated directly by handing it a deliberately
  wrong response matrix.

Two traps are pre-committed against:

1. **The response matrix must be pinned independently.** ``R``'s columns are
   checked against the single-kick closed form, a different code path from the
   ``(I - M4)^-1`` solve that produces them.
2. **SVD truncation must not be vacuous.** With as many correctors as monitors
   the plain solve is exact and truncation never does anything. So both ``N > M``
   and ``N < M`` appear here, and a *near-degenerate* corrector pair — split by a
   0.1 mm drift, hence nearly the same betatron phase — where the untruncated
   answer asks for 0.66 rad of steering and the truncated one asks for 6.8e-5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Corrector,
    Drift,
    Lattice,
    OrbitCorrectionError,
    Quadrupole,
    ReferenceParticle,
    ThinQuadrupole,
    closed_twiss,
    correct_orbit,
    orbit_response_matrix,
    propagate_orbit,
    propagate_twiss,
    tunes,
)

VF = 1.0 / 1.5
VD = 1.0 / 1.6
L_HALF = 1.0
ERR_KICK = 3e-4  # the steering error every correction test works against [rad]


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _ring(n_cells: int = 6) -> list:
    """``n_cells`` thin FODO cells — the same lattice as the closed-orbit suite."""
    return [
        e
        for i in range(n_cells)
        for e in (
            ThinQuadrupole(0.5 * VF, name=f"qfa_{i}"),
            Drift(L_HALF, name=f"d1_{i}"),
            ThinQuadrupole(-VD, name=f"qd_{i}"),
            Drift(L_HALF, name=f"d2_{i}"),
            ThinQuadrupole(0.5 * VF, name=f"qfb_{i}"),
        )
    ]


def _index_of(lat: Lattice, elem) -> int:
    """Element index of ``elem`` in ``lat`` (by identity)."""
    return next(i for i, e in enumerate(lat.elements) if e is elem)


# ---------------------------------------------------------------------------
# The response matrix
# ---------------------------------------------------------------------------


def test_response_columns_match_the_single_kick_closed_form(ref: ReferenceParticle) -> None:
    """Every entry of ``R`` equals ``sqrt(beta_j beta_i)/(2 sin pi Q) cos(dpsi - pi Q)``.

    The independent pin the milestone rests on. ``R`` is built from the
    ``(I - M4)^-1`` fixed point; the reference is Twiss parameters and phase
    advances, which that solve never sees. Both correctors and all 33 boundaries
    are compared, so the ``sqrt(beta_j beta_i)`` product, the shared denominator
    and the ``dpsi`` wrap-around (for monitors *upstream* of the corrector, where
    the phase must be taken the long way round the ring) are pinned at once.
    """
    elems = _ring()
    ca, cb = Corrector(name="ca"), Corrector(name="cb")
    elems.insert(5, ca)
    elems.insert(16, cb)
    lat = Lattice(elems, ref)
    monitors = list(range(len(lat) + 1))

    R = orbit_response_matrix(lat, [ca, cb], monitors, "x")
    table = propagate_twiss(lat, closed_twiss(lat))
    qx, _ = tunes(lat)
    two_pi_q = 2.0 * math.pi * qx

    for j, c in enumerate((ca, cb)):
        k = _index_of(lat, c) + 1  # the kick enters at the corrector's exit boundary
        beta_k, psi_k = table[k].beta_x, table[k].mu_x
        for i in monitors:
            dpsi = table[i].mu_x - psi_k
            if dpsi < 0.0:
                dpsi += two_pi_q  # a monitor upstream is reached the long way round
            want = (
                math.sqrt(beta_k * table[i].beta_x)
                / (2.0 * math.sin(math.pi * qx))
                * math.cos(dpsi - math.pi * qx)
            )
            assert R[i, j] == pytest.approx(want, abs=1e-14)
    assert np.abs(R).max() > 1.0  # the responses are O(1) m/rad, not round-off


def test_response_is_exact_not_a_finite_difference(ref: ReferenceParticle) -> None:
    """``orbit(theta) = orbit(0) + R theta`` at a kick a thousand times any real one.

    A finite-difference Jacobian would drift at large amplitude; this one cannot,
    because the map really is affine. That is what licenses the single solve.
    """
    elems = _ring()
    ca, cb = Corrector(name="ca"), Corrector(name="cb")
    elems.insert(5, ca)
    elems.insert(16, cb)
    lat = Lattice(elems, ref)
    monitors = [0, 7, 13, 24, 31]
    R = orbit_response_matrix(lat, [ca, cb], monitors, "x")
    base = np.array([o[0] for o in propagate_orbit(lat)])[monitors]

    theta = np.array([0.3, -0.17])  # radians — absurd for a steerer, exact anyway
    ca.kick_x, cb.kick_x = theta
    got = np.array([o[0] for o in propagate_orbit(lat)])[monitors]
    assert np.allclose(got, base + R @ theta, rtol=1e-12, atol=0)


def test_response_matrix_restores_the_lattice(ref: ReferenceParticle) -> None:
    """It mutates while it works and puts every corrector back, even on the way out."""
    elems = _ring()
    ca, cb = Corrector(kick_x=1.1e-4, name="ca"), Corrector(kick_x=-2.2e-4, name="cb")
    elems.insert(5, ca)
    elems.insert(16, cb)
    lat = Lattice(elems, ref)
    orbit_response_matrix(lat, [ca, cb], None, "x")
    assert (ca.kick_x, cb.kick_x) == (1.1e-4, -2.2e-4)


def test_response_does_not_depend_on_the_present_orbit(ref: ReferenceParticle) -> None:
    """``R`` is a property of the machine, not of where the beam currently is.

    Computed with the correctors at zero, then again with them set, and again with
    a different uncorrected error present. The listed correctors are zeroed before
    measuring, so the first two agree *bit for bit*. An error source is not
    touched, so it survives only through the baseline subtraction
    ``(col + base) - base``, which costs the last bit or two — the agreement there
    is to round-off (measured ~1e-16 relative), not exact.
    """
    elems = _ring()
    err = Corrector(kick_x=ERR_KICK, name="err")
    ca, cb = Corrector(name="ca"), Corrector(name="cb")
    elems.insert(2, err)
    elems.insert(9, ca)
    elems.insert(21, cb)
    lat = Lattice(elems, ref)

    r0 = orbit_response_matrix(lat, [ca, cb], None, "x")
    ca.kick_x, cb.kick_x = 5e-4, -8e-4
    r1 = orbit_response_matrix(lat, [ca, cb], None, "x")
    err.kick_x = -1e-3
    r2 = orbit_response_matrix(lat, [ca, cb], None, "x")
    assert np.array_equal(r0, r1)
    assert np.allclose(r0, r2, rtol=1e-13, atol=0)
    assert not np.array_equal(r0, r2)  # ... and it really is the last-bit effect


def test_vertical_response_is_the_vertical_plane(ref: ReferenceParticle) -> None:
    """``plane='y'`` drives ``kick_y`` and reads ``y``; the two planes differ."""
    elems = _ring()
    c = Corrector(name="c")
    elems.insert(5, c)
    lat = Lattice(elems, ref)
    rx = orbit_response_matrix(lat, [c], None, "x")
    ry = orbit_response_matrix(lat, [c], None, "y")
    assert rx.shape == ry.shape == (len(lat) + 1, 1)
    assert not np.allclose(rx, ry)  # different beta and tune in the two planes
    assert c.kick_x == 0.0 and c.kick_y == 0.0


# ---------------------------------------------------------------------------
# Correction: the single solve, and what it can and cannot reach
# ---------------------------------------------------------------------------


def _bumped_ring(ref: ReferenceParticle) -> tuple[Lattice, Corrector, Corrector, Corrector]:
    """A ring with a steering error and two correctors downstream of it.

    Returns ``(lattice, error, ca, cb)`` with the error at element 2 and the
    correctors at 10 and 22 — the geometry every correction test below uses.
    """
    elems = _ring()
    err = Corrector(kick_x=ERR_KICK, name="err")
    ca, cb = Corrector(name="ca"), Corrector(name="cb")
    elems.insert(2, err)
    elems.insert(10, ca)
    elems.insert(22, cb)
    return Lattice(elems, ref), err, ca, cb


def _outside_inside(lat: Lattice, err, cb) -> tuple[list[int], list[int]]:
    """Monitor boundaries outside and inside the error-to-last-corrector arc."""
    ie, ib, n = _index_of(lat, err), _index_of(lat, cb), len(lat)
    return [*range(ib + 1, n + 1), *range(ie + 1)], list(range(ie + 1, ib + 1))


def test_two_correctors_zero_the_orbit_outside_the_bump_exactly(
    ref: ReferenceParticle,
) -> None:
    """One solve, machine precision — and 14 monitors against 2 knobs.

    The physics: the closed orbit at any point is fixed by two numbers per plane,
    so **two** correctors can annul a steering error completely — outside the arc
    between the error and the last corrector. Over-determined is not the same as
    unreachable (the H2 lesson): 14 targets, 2 knobs, residual 2.2e-19.
    """
    lat, err, ca, cb = _bumped_ring(ref)
    outside, _ = _outside_inside(lat, err, cb)
    assert len(outside) == 14

    res = correct_orbit(lat, [ca, cb], outside, "x")
    assert res.rms_before > 2e-4
    assert res.rms_after < 1e-15
    # Independently measured from the corrected lattice, not from the return value.
    table = propagate_orbit(lat)
    assert max(abs(table[i][0]) for i in outside) < 1e-15


def test_the_orbit_inside_the_bump_is_deliberately_not_zero(ref: ReferenceParticle) -> None:
    """A corrector cannot fix what happened upstream of it.

    The counterpart of the previous test, and the reason it restricts its
    monitors: between the error and the last corrector the beam is genuinely off
    axis — that arc *is* the closed bump. Asserting "orbit zero everywhere" would
    be asserting something false.
    """
    lat, err, ca, cb = _bumped_ring(ref)
    outside, inside = _outside_inside(lat, err, cb)
    correct_orbit(lat, [ca, cb], outside, "x")
    table = propagate_orbit(lat)
    assert max(abs(table[i][0]) for i in inside) > 5e-4  # measured 7.5e-4


def test_least_squares_over_all_monitors_improves_but_cannot_be_exact(
    ref: ReferenceParticle,
) -> None:
    """N < M with an unreachable target: the residual is reported, not hidden.

    Monitoring inside the bump too, two correctors cannot zero 34 monitors. The
    solve returns the least-squares compromise and says how big it is; nothing
    claims success.
    """
    lat, err, ca, cb = _bumped_ring(ref)
    res = correct_orbit(lat, [ca, cb], None, "x")
    assert res.rms_after < res.rms_before  # it did help ...
    assert res.rms_after > 1e-4  # ... but nowhere near zero (measured 2.7e-4)


def test_more_correctors_than_monitors_gives_the_minimum_norm_solution(
    ref: ReferenceParticle,
) -> None:
    """N > M: exact at the monitors, and the smallest kicks that get there.

    Verified against an explicit alternative built by adding a null-space
    direction — it hits the same monitors and needs more steering.
    """
    elems = _ring()
    err = Corrector(kick_x=ERR_KICK, name="err")
    cs = [Corrector(name=f"c{i}") for i in range(4)]
    elems.insert(1, err)
    for j, c in enumerate(cs):
        elems.insert(6 + 6 * j, c)
    lat = Lattice(elems, ref)
    monitors = [12, 28]

    R = orbit_response_matrix(lat, cs, monitors, "x")
    res = correct_orbit(lat, cs, monitors, "x")
    assert res.rms_after < 1e-15  # exact at the two monitors

    null = np.linalg.svd(R)[2][len(monitors) :]  # the 2D null space of R
    alt = res.kicks + 3e-4 * null[0]
    assert np.allclose(R @ (alt - res.initial_kicks), R @ (res.kicks - res.initial_kicks))
    assert np.linalg.norm(res.kicks) < np.linalg.norm(alt)  # measured 3.13e-4 vs 4.34e-4


def test_correction_never_makes_the_monitored_orbit_worse(ref: ReferenceParticle) -> None:
    """``rms_after <= rms_before`` always — zero kicks are in the solve's span."""
    lat, err, ca, cb = _bumped_ring(ref)
    for monitors in (None, [0, 5, 9], [3], list(range(0, 30, 3))):
        ca.kick_x = cb.kick_x = 0.0
        res = correct_orbit(lat, [ca, cb], monitors, "x")
        assert res.rms_after <= res.rms_before


def test_perfect_machine_needs_no_correction(ref: ReferenceParticle) -> None:
    """With no error the orbit is already zero and the kicks stay zero."""
    elems = _ring()
    ca, cb = Corrector(name="ca"), Corrector(name="cb")
    elems.insert(10, ca)
    elems.insert(22, cb)
    lat = Lattice(elems, ref)
    res = correct_orbit(lat, [ca, cb], None, "x")
    assert res.rms_before == 0.0
    assert res.rms_after == 0.0
    assert np.array_equal(res.kicks, np.zeros(2))


# ---------------------------------------------------------------------------
# The result is measured, not predicted
# ---------------------------------------------------------------------------


def test_a_wrong_response_matrix_is_exposed_by_the_reported_residual(
    ref: ReferenceParticle,
) -> None:
    """The gate on the one thing that could make every test above vacuous.

    A real machine's response matrix is *measured*, and disagrees with the model;
    ``correct_orbit`` therefore accepts one. Hand it ``1.5 R``: the solve then
    applies two thirds of the right kick, so the true residual is a third of the
    original orbit — while the *prediction* ``x0 + (1.5R) dtheta`` is exactly
    zero. The reported ``rms_after`` must be the former. If it were the latter, a
    wrong response matrix would report a perfect correction, and no other test in
    this file could tell the difference.
    """
    lat, err, ca, cb = _bumped_ring(ref)
    outside, _ = _outside_inside(lat, err, cb)
    R = orbit_response_matrix(lat, [ca, cb], outside, "x")
    x0 = np.array([o[0] for o in propagate_orbit(lat)])[outside]

    res = correct_orbit(lat, [ca, cb], outside, "x", response=1.5 * R)
    dtheta = res.kicks - res.initial_kicks

    # What a prediction-based implementation would have reported: exactly zero.
    predicted = x0 + (1.5 * R) @ dtheta
    assert np.sqrt(np.mean(predicted**2)) < 1e-16
    # What the machine actually has: a third of the original orbit.
    assert res.rms_after == pytest.approx(res.rms_before / 3.0, rel=1e-9)
    # ... confirmed from the lattice itself, independently of the return value.
    table = propagate_orbit(lat)
    truth = np.array([table[i][0] for i in outside])
    assert res.rms_after == pytest.approx(float(np.sqrt(np.mean(truth**2))), rel=1e-12)


def test_a_correct_response_makes_prediction_and_measurement_agree(
    ref: ReferenceParticle,
) -> None:
    """The control for the previous test: with the right ``R`` the two coincide.

    Which is exactly why the discriminating test has to corrupt it.
    """
    lat, err, ca, cb = _bumped_ring(ref)
    outside, _ = _outside_inside(lat, err, cb)
    R = orbit_response_matrix(lat, [ca, cb], outside, "x")
    res = correct_orbit(lat, [ca, cb], outside, "x", response=R)
    assert res.rms_after < 1e-15


# ---------------------------------------------------------------------------
# SVD truncation, made non-vacuous
# ---------------------------------------------------------------------------


def _degenerate_ring(ref: ReferenceParticle, gap: float = 1e-3) -> tuple[Lattice, list]:
    """Two correctors split by a ``gap``-metre drift: nearly the same phase.

    Their response columns are then nearly parallel, and the second singular
    value falls as the gap does (measured: ratio 3157 at 1 mm, 31573 at 0.1 mm).
    """
    elems = _ring()
    elems.insert(1, Corrector(kick_x=ERR_KICK, name="err"))
    a, b = Corrector(name="a"), Corrector(name="b")
    elems[9:9] = [a, Drift(gap, name="tiny"), b]
    return Lattice(elems, ref), [a, b]


def test_the_degenerate_pair_really_is_ill_conditioned(ref: ReferenceParticle) -> None:
    """Measure the spectrum before asserting anything about truncating it."""
    lat, pair = _degenerate_ring(ref)
    s = np.linalg.svd(orbit_response_matrix(lat, pair, None, "x"), compute_uv=False)
    assert s[0] / s[1] > 1000.0  # measured 3157


def test_truncation_trades_a_little_orbit_for_a_realisable_correction(
    ref: ReferenceParticle,
) -> None:
    """The reason ``n_singular`` exists, in numbers.

    Untruncated, the least-squares answer is *mathematically* better and asks the
    two nearly-parallel steerers for 0.66 rad — 38 degrees, which no corrector
    magnet can deliver — to buy a 32 % improvement. Keeping one singular value
    asks for 6.8e-5 rad, four orders of magnitude smaller, and still improves the
    orbit. Both facts are asserted, because only their combination is the point.
    """
    lat, pair = _degenerate_ring(ref)
    full = correct_orbit(lat, pair, None, "x")
    k_full = full.kicks.copy()
    for c in pair:
        c.kick_x = 0.0
    trunc = correct_orbit(lat, pair, None, "x", n_singular=1)

    assert full.n_used == 2 and trunc.n_used == 1
    assert full.rms_after < trunc.rms_after  # the full solve *is* the better fit ...
    assert np.linalg.norm(k_full) > 0.1  # ... and it is unusable: measured 0.66 rad
    assert np.linalg.norm(trunc.kicks) < 1e-3  # measured 6.8e-5 rad
    ratio = np.linalg.norm(k_full) / np.linalg.norm(trunc.kicks)
    assert ratio > 100.0  # measured 9752
    assert trunc.rms_after < trunc.rms_before  # still an improvement, not a no-op


def test_an_exactly_parallel_pair_drops_the_dead_direction_by_itself(
    ref: ReferenceParticle,
) -> None:
    """Two correctors at the *same* phase: the null direction is discarded silently.

    A thin quadrupole advances no phase, so a corrector on each side of one has an
    identical response — the matrix is exactly rank 1. Without the round-off
    cutoff the solve would divide by a 1e-15 singular value and return a
    meaningless pair of enormous, cancelling kicks.
    """
    elems = _ring()
    elems.insert(1, Corrector(kick_x=ERR_KICK, name="err"))
    a, b = Corrector(name="a"), Corrector(name="b")
    elems[9:9] = [a, ThinQuadrupole(0.3, name="thin"), b]
    lat = Lattice(elems, ref)
    s = np.linalg.svd(orbit_response_matrix(lat, [a, b], None, "x"), compute_uv=False)
    assert s[1] / s[0] < 1e-14  # measured ~1e-16: exactly rank 1

    res = correct_orbit(lat, [a, b], None, "x")
    assert res.n_used == 1
    assert np.linalg.norm(res.kicks) < 1e-3
    assert res.rms_after < res.rms_before


# ---------------------------------------------------------------------------
# Refusals — each with the physical reason
# ---------------------------------------------------------------------------


def test_a_quadrupole_cannot_steer(ref: ReferenceParticle) -> None:
    """Quadrupoles move the optics; only a dipole moves the orbit."""
    elems = _ring()
    q = Quadrupole(0.3, 0.5, name="q")
    elems.insert(3, q)
    lat = Lattice(elems, ref)
    with pytest.raises(OrbitCorrectionError, match="Quadrupole"):
        orbit_response_matrix(lat, [q], None, "x")  # type: ignore[list-item]


def test_a_corrector_outside_the_lattice_is_refused(ref: ReferenceParticle) -> None:
    """An element the lattice has never seen has no response at all."""
    lat = Lattice(_ring(), ref)
    with pytest.raises(OrbitCorrectionError, match="not in this lattice"):
        orbit_response_matrix(lat, [Corrector()], None, "x")


def test_a_corrector_listed_twice_is_refused(ref: ReferenceParticle) -> None:
    """Its column would be ambiguous — the same rule ``Knob`` applies."""
    elems = _ring()
    c = Corrector(name="c")
    elems.insert(3, c)
    lat = Lattice(elems, ref)
    with pytest.raises(OrbitCorrectionError, match="twice"):
        orbit_response_matrix(lat, [c, c], None, "x")


def test_the_same_corrector_placed_twice_is_allowed_and_summed(
    ref: ReferenceParticle,
) -> None:
    """Repeated *placements* are legitimate: one power supply, two magnets."""
    elems = _ring()
    c = Corrector(name="shared")
    elems.insert(3, c)
    elems.insert(17, c)
    lat = Lattice(elems, ref)
    r_both = orbit_response_matrix(lat, [c], [0], "x")

    one = _ring()
    solo = Corrector(name="solo")
    one.insert(3, solo)
    r_one = orbit_response_matrix(Lattice(one, ref), [solo], [0], "x")
    assert abs(r_both[0, 0]) > abs(r_one[0, 0])  # two placements respond, not one


def test_bad_plane_monitor_and_singular_count_are_refused(ref: ReferenceParticle) -> None:
    """The three remaining ways to ask an ill-posed question."""
    elems = _ring()
    c = Corrector(name="c")
    elems.insert(3, c)
    lat = Lattice(elems, ref)
    with pytest.raises(OrbitCorrectionError, match="plane must be"):
        orbit_response_matrix(lat, [c], None, "z")
    with pytest.raises(OrbitCorrectionError, match="outside"):
        orbit_response_matrix(lat, [c], [999], "x")
    with pytest.raises(OrbitCorrectionError, match="n_singular"):
        correct_orbit(lat, [c], [0, 5], "x", n_singular=3)
    with pytest.raises(OrbitCorrectionError, match="no correctors"):
        orbit_response_matrix(lat, [], None, "x")


def test_a_mis_shaped_supplied_response_is_refused(ref: ReferenceParticle) -> None:
    """A measured response matrix that does not match the problem is caught."""
    elems = _ring()
    ca, cb = Corrector(name="ca"), Corrector(name="cb")
    elems.insert(5, ca)
    elems.insert(16, cb)
    lat = Lattice(elems, ref)
    with pytest.raises(OrbitCorrectionError, match="response has shape"):
        correct_orbit(lat, [ca, cb], [0, 5, 9], "x", response=np.ones((3, 5)))
