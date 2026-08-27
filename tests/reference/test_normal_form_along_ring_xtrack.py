r"""Cross-check the normal form along the ring (O2) against xtrack's ``twiss``.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

O1 compared one matrix at one point. This file compares ``W(s)`` at **every** element
boundary, plus the three tables that only exist once ``W`` is a function of ``s``: the
phase advances ``mux``/``muy``/``muzeta``, the Mais-Ripken cross-plane betas
``betx2``/``bety1``, and the crab dispersion ``dx_zeta``. None of those had an accsim
analogue before this milestone, so the arbiter is doing real work rather than confirming.

**Why the phase advance is the load-bearing comparison.** Every other quantity here is
invariant under the re-phasing that is O2's whole operation — ``betx2 = |v2[x]|^2`` and
``alfx2 = -Re(v2[x] conj(v2[px]))`` have the phase cancel between their factors, and
``dx_zeta`` is a ratio taken inside one eigenvector. So a re-phasing bug would ship
silently through the Ripken and crab tables and show up **only** in ``mu``. The analytic
file gates ``mu`` against the *full* integer-plus-fractional tune, which is quantised; this
file gates it element by element against an independent implementation.

**Three rings, because one cannot do the job.**

  * *Bend-free with a cavity* — the only ring on which the two codes' one-turn matrices
    agree to round-off, so it is where ``W(s)`` is compared entry by entry. It also
    happens to make ``betx2``, ``bety1`` and ``dx_zeta`` **exactly zero** in both codes,
    which is a free gate on all three being dispersive/coupling-driven rather than
    numerical noise.
  * *Bend-free, coupled, with a tune split* — where ``betx2``/``bety1`` are nonzero. The
    split is deliberate: the plain FODO ring has ``qx = qy`` exactly, sitting on the
    difference resonance, which is the one place the two codes' mode-labelling rules need
    not agree (O1's file says why).
  * *The I4 ring* — bends, so ``dx_zeta`` exists. Its tolerance is stated as looser
    because a bendy ring is also comparing the two codes' bend models, which is the
    residual axis L and B2 already own.

**The pre-commitment, written into ``docs/ROADMAP.md`` -> O2 before this file was run,**
including the part that was expected to be missed: O1 localised a ``2.6e-11`` residual in
``W``'s longitudinal columns to xtrack's finite-differenced ``R56``, and
``W(s) = M(0->s) W(0)`` *transports* that error, so it must grow along the ring. The claim
gated here is not that it stays small but that it stays **confined**: the transverse block
must not degrade at all.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from accsim import (  # noqa: E402
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    ThinSkewQuadrupole,
    closed_normal_form,
    propagate_normal_form,
)

MASS0 = 0.51099895069e6  # electron, eV
GAMMA0 = 5.0
KQ, N_CELLS, LQ, LDRIFT = 1.2, 4, 0.3, 0.7
VOLTAGE, HARMONIC = 1.0e5, 8

#: O1's constant, reused unchanged: the two ``W`` matrices agree to this **absolutely** on
#: the transverse block. Absolute, not relative — ``xtrack/linear_normal_form.py`` ends
#: with ``W[abs(W) < 1e-14] = 0``, so a relative test against a zeroed entry is spurious.
W_ATOL = 1e-12

#: What the *longitudinal* columns reach once transported around the whole ring. O1
#: measured ``2.6e-11`` at the entrance and pinned it on xtrack's finite-differenced
#: ``R56``; ``W(s) = M(0->s) W(0)`` carries that error forward, so it grows. This constant
#: records the growth, and :func:`test_the_reference_residual_stays_in_the_longitudinal_columns`
#: gates the thing that actually matters — that it does not leak sideways.
ALONG_RING_LONGITUDINAL_ATOL = 5e-10

#: The transverse phase advances ``mux``/``muy``, absolutely. Measured ``3.3e-16`` — four
#: orders inside — and, more to the point, **bit-identical** however xtrack's momentum
#: differentiation step is set. That is what separates them from ``muzeta`` below.
MU_ATOL = 1e-12

#: What ``muzeta`` reaches, and it is not accsim's number. The longitudinal phase advance
#: is read off the same columns of ``W`` where O1 localised xtrack's finite-differenced
#: ``R56``, so it inherits that error: it traces the same U in the step size with the same
#: minimum. Measured ``1.8e-11`` at that minimum.
#: :func:`test_the_longitudinal_phase_residual_is_the_references_own_r56` gates the
#: attribution rather than asserting it.
MUZETA_ATOL = 5e-11

#: xtrack's ``R56`` step, at the minimum of the U-curve O1 measured. See that file.
_BEST_STEPS = {
    "dx": 1e-6,
    "dpx": 1e-7,
    "dy": 1e-6,
    "dpy": 1e-7,
    "dzeta": 1e-6,
    "ddelta": 1e-5,
}

#: The Mais-Ripken tables, on the coupled bend-free ring. The pre-commitment was ``1e-9``
#: and the measurement is ``3.9e-15`` relative on the betas and ``2.3e-15`` absolute on the
#: alphas — the same round-off as the transverse block of ``W`` they are built from, which
#: is what one should expect once one notices that a Ripken beta is two entries of ``W``
#: squared. Set roughly two orders above the measurement, not at the pre-commitment.
RIPKEN_RTOL = 1e-12
RIPKEN_ATOL = 1e-12

#: The crab dispersion, on the bendy ring. Signal ``+-0.077``, measured agreement
#: ``1.1e-9`` **absolute** — absolute because ``dx_zeta`` passes through zero twice per
#: cell, where a relative comparison means nothing.
#:
#: **The pre-committed ``5e-3`` was five orders too loose, and the reason is worth
#: recording.** It budgeted for the two codes' bend models disagreeing — the residual axis
#: L and B2 own — but B2 had already removed it: ``_line()`` sets ``integrator="uniform"``
#: and one multipole kick per element precisely so that xtrack integrates the bend the way
#: accsim does. A gate at ``5e-3`` on a quantity that agrees at ``1e-9`` would sleep
#: through any regression worth catching, so the tolerance follows the measurement.
CRAB_ATOL = 1e-7

#: The dynamic dispersion along the same ring. Signal ``1.9`` to ``3.3``, measured
#: ``1.4e-8`` absolute. O1 compared this at the entrance only and stated ``2e-3`` for the
#: same (mistaken) reason as above.
DISPERSION_ATOL = 1e-6

#: The tune split that moves the coupled ring off ``qx = qy``, where the two codes'
#: mode-labelling rules need not agree.
SPLIT, K1SL = 0.05, 0.1


def _cells():
    """The same bend-free FODO cells in both codes."""
    acc, xtk = [], []
    for _ in range(N_CELLS):
        acc += [Quadrupole(LQ, KQ), Drift(LDRIFT), Quadrupole(LQ, -KQ), Drift(LDRIFT)]
        xtk += [
            xt.Quadrupole(length=LQ, k1=KQ),
            xt.Drift(length=LDRIFT),
            xt.Quadrupole(length=LQ, k1=-KQ),
            xt.Drift(length=LDRIFT),
        ]
    return acc, xtk


def _build(line: xt.Line) -> xt.Line:
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


@pytest.fixture(scope="module")
def pair():
    """``(accsim lattice, xtrack line)`` for the bend-free ring with a cavity.

    Module-scoped: every ``xt.Line`` JIT-compiles a fresh C kernel at ~12 s
    (CONVENTIONS.md -> *Test-suite cost*), so the three tests that use this ring share one
    build.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    acc, xtk = _cells()
    plain = Lattice(acc, ref)
    cavity = RFCavity.from_harmonic(VOLTAGE, HARMONIC, plain.length, ref, phi_s=0.0)
    lattice = Lattice([*acc, cavity], ref)
    line = xt.Line(elements=[*xtk, xt.Cavity(voltage=VOLTAGE, frequency=cavity.frequency)])
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1.0, gamma0=GAMMA0)
    return lattice, _build(line)


@pytest.fixture(scope="module")
def coupled_pair():
    """``(accsim lattice, xtrack line)`` for a coupled, bend-free, tune-split ring.

    One thin skew quadrupole makes ``betx2``/``bety1`` nonzero; one thin normal quadrupole
    splits the tunes away from ``qx = qy``. No bends and no cavity, so this is the ``4d``
    comparison and nothing dispersive contaminates it.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    acc, xtk = _cells()
    lattice = Lattice(acc[:4] + [ThinSkewQuadrupole(K1SL)] + acc[4:] + [ThinQuadrupole(SPLIT)], ref)
    line = xt.Line(
        elements=[
            *xtk[:4],
            xt.Multipole(knl=[0.0, 0.0], ksl=[0.0, K1SL], length=0.0),
            *xtk[4:],
            xt.Multipole(knl=[0.0, SPLIT], length=0.0),
        ]
    )
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1.0, gamma0=GAMMA0)
    return lattice, _build(line)


@pytest.fixture(scope="module")
def bendy_pair():
    """``(accsim lattice, xtrack line)`` for the I4 ring — the one with dispersion.

    ``_line()`` builds its cavity with a placeholder ``frequency=1.0``, so it must be
    given the real one or the ring has effectively no RF: ``Q_s`` collapses, and with it
    both quantities this ring exists to check. (Left unpatched, ``tw.dx`` comes back as
    the *matched* dispersion — 24% away from the dynamic one — and ``dx_zeta`` as roughly
    zero, since both departures are second order in ``Q_s``.)
    """
    import test_closed_orbit_6d as i4
    import test_closed_orbit_6d_xtrack as i4x

    lattice = i4.ring()[0]
    line = i4x._line()
    line["cav"].frequency = lattice.elements[-1].frequency
    return lattice, line


def _rows(tw, *names):
    """xtrack columns as plain arrays, with the ``_end_point`` row kept.

    ``twiss`` returns one row per element entrance plus a final ``_end_point``, which is
    exactly :func:`propagate_normal_form`'s entrance-then-each-exit shape. It also NaNs
    out rows inside thin groups of elements, so those are asserted absent rather than
    assumed absent.
    """
    out = [np.asarray(tw[n], dtype=float) for n in names]
    for name, col in zip(names, out, strict=True):
        assert not np.isnan(col).any(), f"xtrack returned NaN in {name} (a thin group?)"
    return out


# --------------------------------------------------------------------------------------
# W(s) and the phase advance, on the ring where the two codes' maps agree to round-off
# --------------------------------------------------------------------------------------


def test_w_matches_xtrack_at_every_point(pair) -> None:
    """The transverse block of ``W(s)``, entry by entry, all the way round.

    O1's ``1e-12`` at the entrance, now claimed at every element boundary. This is the
    gate that would catch a wrong *transport* — as opposed to a wrong re-phasing, which
    it cannot see at all.
    """
    lattice, line = pair
    tw = line.twiss(steps_R_matrix=_BEST_STEPS)
    points = propagate_normal_form(lattice, closed_normal_form(lattice, method="6d"))
    theirs = np.asarray(tw.W_matrix)
    assert len(points) == len(theirs)
    for point, s, w in zip(points, _rows(tw, "s")[0], theirs, strict=True):
        assert point.s == pytest.approx(float(s), abs=1e-12)
        assert np.allclose(point.w[:4, :4], w[:4, :4], atol=W_ATOL, rtol=0.0)


def test_the_reference_residual_stays_in_the_longitudinal_columns(pair) -> None:
    """The expected miss, and the claim that actually matters about it.

    O1 pinned a ``2.6e-11`` residual on xtrack's finite-differenced ``R56``.
    ``W(s) = M(0->s) W(0)`` transports that error, so along the ring it **grows** — by a
    factor of a few here. That growth is not a new finding and is not gated tightly. What
    is gated is that it stays where it started: the transverse block must not degrade at
    all, because a transport bug would contaminate everything, not one corner.
    """
    lattice, line = pair
    tw = line.twiss(steps_R_matrix=_BEST_STEPS)
    points = propagate_normal_form(lattice, closed_normal_form(lattice, method="6d"))
    theirs = np.asarray(tw.W_matrix)
    transverse = [
        np.abs(p.w[:4, :4] - w[:4, :4]).max() for p, w in zip(points, theirs, strict=True)
    ]
    whole = [np.abs(p.w - w).max() for p, w in zip(points, theirs, strict=True)]
    assert max(transverse) < W_ATOL
    assert max(whole) < ALONG_RING_LONGITUDINAL_ATOL
    # It grew: this is transport of one entry, not a floor either code could remove.
    assert whole[-1] > 2.0 * whole[0]
    # And it did not leak: the transverse block is four orders cleaner at every point.
    assert max(transverse) < min(whole) / 1000.0


def test_the_phase_advances_match_xtrack_element_by_element(pair) -> None:
    """``mux``, ``muy``, ``muzeta`` -- the only table here that can see the re-phasing.

    xtrack reports these in units of ``2 pi`` and shifted to zero at the start; accsim
    keeps radians, to match :func:`~accsim.propagate_twiss`.

    **The pre-committed ``1e-12`` held transversely and was missed longitudinally, by the
    same owner O1 already named.** ``mux``/``muy`` land at ``3.3e-16``; ``muzeta`` floors
    at ``1.8e-11``, because it is read off the columns of ``W`` that carry xtrack's
    finite-differenced ``R56``. Hence two constants rather than one loosened one — and
    :func:`test_the_longitudinal_phase_residual_is_the_references_own_r56` next door,
    which gates the attribution.
    """
    lattice, line = pair
    tw = line.twiss(steps_R_matrix=_BEST_STEPS)
    points = propagate_normal_form(lattice, closed_normal_form(lattice, method="6d"))
    mux, muy, muzeta = _rows(tw, "mux", "muy", "muzeta")
    for point, qx, qy, qs in zip(points, mux, muy, muzeta, strict=True):
        assert point.mu[0] / (2.0 * math.pi) == pytest.approx(qx, abs=MU_ATOL)
        assert point.mu[1] / (2.0 * math.pi) == pytest.approx(qy, abs=MU_ATOL)
        assert point.mu[2] / (2.0 * math.pi) == pytest.approx(qs, abs=MUZETA_ATOL)


def test_the_longitudinal_phase_residual_is_the_references_own_r56(pair) -> None:
    """Why ``muzeta`` gets its own constant, gated rather than asserted.

    Three independent signatures, none of which a genuine model disagreement has:

      * **It is confined to one plane while responding to the other's step.** Changing
        xtrack's momentum differentiation step over three decades moves ``muzeta``'s
        residual by five orders of magnitude and moves ``mux``/``muy`` by **not one bit**.
        A physics disagreement between two codes does not care what step the reference
        differentiates with, and does not stop at a plane boundary.
      * **It has a minimum.** The residual traces a U — an ``h^2`` truncation above,
        cancellation round-off below — with its floor at the same ``ddelta = 1e-5`` where
        O1 found ``R56``'s.
      * **Above the minimum it tracks ``R56`` one for one.** In the truncation-dominated
        regime both are the same ``h^2``, so their ratios between two steps must agree;
        they do, to a few percent. Stated as a *ratio* so that nothing here depends on
        this particular ring's transport factor.
    """
    lattice, line = pair
    m6 = lattice.one_turn_matrix()
    points = propagate_normal_form(lattice, closed_normal_form(lattice, method="6d"))
    mine = np.array([[p.mu[i] / (2.0 * math.pi) for i in range(3)] for p in points])
    err, gap = {}, {}
    for ddelta in (1e-6, 1e-5, 1e-4, 1e-3):
        tw = line.twiss(steps_R_matrix=dict(_BEST_STEPS, ddelta=ddelta))
        theirs = np.column_stack([np.asarray(tw[n], dtype=float) for n in ("mux", "muy", "muzeta")])
        err[ddelta] = np.abs(mine - theirs).max(axis=0)
        gap[ddelta] = abs(float(np.asarray(tw.R_matrix)[4, 5]) - m6[4, 5])

    transverse = np.array([err[d][:2] for d in err])
    assert np.array_equal(transverse, np.tile(transverse[0], (len(err), 1))), (
        f"the transverse residual moved with the reference's momentum step: {transverse}"
    )
    assert transverse.max() < MU_ATOL

    longitudinal = {d: err[d][2] for d in err}
    assert longitudinal[1e-5] < longitudinal[1e-6]
    assert longitudinal[1e-5] < longitudinal[1e-4] / 10.0
    assert longitudinal[1e-5] < MUZETA_ATOL
    assert longitudinal[1e-3] / longitudinal[1e-4] == pytest.approx(gap[1e-3] / gap[1e-4], rel=0.05)


def test_both_codes_make_the_new_quantities_vanish_on_a_bend_free_ring(pair) -> None:
    """A free gate on all three: without coupling or bends they are identically zero.

    Worth asserting rather than skipping, because "zero in both codes" is the only place
    the cross-plane betas and the crab dispersion can be compared without a model
    residual standing between them.
    """
    lattice, line = pair
    tw = line.twiss(steps_R_matrix=_BEST_STEPS)
    points = propagate_normal_form(lattice, closed_normal_form(lattice, method="6d"))
    betx2, bety1, dx_zeta = _rows(tw, "betx2", "bety1", "dx_zeta")
    assert np.abs(betx2).max() == 0.0
    assert np.abs(bety1).max() == 0.0
    assert np.abs(dx_zeta).max() == 0.0
    assert max(abs(p.betas[0, 1]) for p in points) < 1e-28
    assert max(abs(p.betas[1, 0]) for p in points) < 1e-28
    assert max(abs(p.crab_dispersion[0]) for p in points) < 1e-25


# --------------------------------------------------------------------------------------
# The cross-plane betas, on a ring that has them
# --------------------------------------------------------------------------------------


def test_the_cross_plane_betas_match_xtrack(coupled_pair) -> None:
    """``betx2`` and ``bety1`` element by element -- quantities accsim could not compute
    before this milestone, so the arbiter is doing real work here.

    The ring is split off ``qx = qy`` on purpose: on the difference resonance the two
    codes' mode-labelling rules (xtrack tie-breaks on ``|v[5]|`` then ``|v[2]|``; accsim
    maximises the total per-plane weight) need not agree, and a labelling swap would read
    as a physics failure.
    """
    lattice, line = coupled_pair
    tw = line.twiss(method="4d")
    points = propagate_normal_form(lattice, closed_normal_form(lattice, method="4d"))
    betx1, betx2, bety1, bety2 = _rows(tw, "betx1", "betx2", "bety1", "bety2")
    assert max(betx2) > 1e-3, "a gate on the cross-plane beta needs a cross-plane beta"
    for point, b11, b12, b21, b22 in zip(points, betx1, betx2, bety1, bety2, strict=True):
        assert point.betas[0, 0] == pytest.approx(b11, rel=RIPKEN_RTOL)
        assert point.betas[0, 1] == pytest.approx(b12, rel=RIPKEN_RTOL)
        assert point.betas[1, 0] == pytest.approx(b21, rel=RIPKEN_RTOL)
        assert point.betas[1, 1] == pytest.approx(b22, rel=RIPKEN_RTOL)


def test_the_cross_plane_alphas_match_xtrack(coupled_pair) -> None:
    """``alfx2`` and ``alfy1`` -- the same matrix read off with the momentum row.

    Compared **absolutely**: unlike the betas these change sign around the ring, so they
    pass through zero and a relative test there is meaningless.
    """
    lattice, line = coupled_pair
    tw = line.twiss(method="4d")
    points = propagate_normal_form(lattice, closed_normal_form(lattice, method="4d"))
    alfx1, alfx2, alfy1, alfy2 = _rows(tw, "alfx1", "alfx2", "alfy1", "alfy2")
    assert np.abs(alfx2).max() > 1e-3
    for point, a11, a12, a21, a22 in zip(points, alfx1, alfx2, alfy1, alfy2, strict=True):
        assert point.alphas[0, 0] == pytest.approx(a11, abs=RIPKEN_ATOL)
        assert point.alphas[0, 1] == pytest.approx(a12, abs=RIPKEN_ATOL)
        assert point.alphas[1, 0] == pytest.approx(a21, abs=RIPKEN_ATOL)
        assert point.alphas[1, 1] == pytest.approx(a22, abs=RIPKEN_ATOL)


def test_the_cross_plane_gammas_match_xtrack(coupled_pair) -> None:
    """``gamx1``/``gamx2``/``gamy1``/``gamy2`` -- the one Ripken matrix nothing else pins.

    ``betas`` is tied to ``propagate_twiss`` and to Edwards-Teng; ``alphas`` likewise.
    ``gammas`` is read off the *momentum* row and has neither tie, so without this test
    (and its analytic partner against ``(1 + alpha^2)/beta``) a wrong row index would
    ship: it would still be invariant under the re-phasing, still symplectic, and still
    pass every other assertion in both files.
    """
    lattice, line = coupled_pair
    tw = line.twiss(method="4d")
    points = propagate_normal_form(lattice, closed_normal_form(lattice, method="4d"))
    gamx1, gamx2, gamy1, gamy2 = _rows(tw, "gamx1", "gamx2", "gamy1", "gamy2")
    assert min(gamx2) > 1e-6, "a gate on the cross-plane gamma needs a cross-plane gamma"
    for point, g11, g12, g21, g22 in zip(points, gamx1, gamx2, gamy1, gamy2, strict=True):
        assert point.gammas[0, 0] == pytest.approx(g11, rel=RIPKEN_RTOL)
        assert point.gammas[0, 1] == pytest.approx(g12, rel=RIPKEN_RTOL)
        assert point.gammas[1, 0] == pytest.approx(g21, rel=RIPKEN_RTOL)
        assert point.gammas[1, 1] == pytest.approx(g22, rel=RIPKEN_RTOL)


def test_the_phase_advances_match_on_a_coupled_ring(coupled_pair) -> None:
    """The re-phasing witness again, this time where the modes are genuinely mixed."""
    lattice, line = coupled_pair
    tw = line.twiss(method="4d")
    points = propagate_normal_form(lattice, closed_normal_form(lattice, method="4d"))
    mux, muy = _rows(tw, "mux", "muy")
    for point, qx, qy in zip(points, mux, muy, strict=True):
        assert point.mu[0] / (2.0 * math.pi) == pytest.approx(qx, abs=1e-11)
        assert point.mu[1] / (2.0 * math.pi) == pytest.approx(qy, abs=1e-11)


# --------------------------------------------------------------------------------------
# The crab dispersion, on a ring that has bends
# --------------------------------------------------------------------------------------


def test_the_crab_dispersion_matches_xtrack_on_a_bendy_ring(bendy_pair) -> None:
    """``dx_zeta``/``dpx_zeta`` element by element, at a stated and looser tolerance.

    Looser because a bendy ring is also comparing the two codes' *bend* models — the
    residual axis L and B2 already own — and because ``dx_zeta`` is second order in
    ``Q_s``, so it is a small number sitting on top of the dispersion. The reason it is
    nonzero at all is the interesting part: this ring has no crab cavity, and the effect
    is the dispersion's phase lag.
    """
    lattice, line = bendy_pair
    tw = line.twiss()
    points = propagate_normal_form(lattice, closed_normal_form(lattice, method="6d"))
    dx_zeta, dpx_zeta = _rows(tw, "dx_zeta", "dpx_zeta")
    assert len(points) == len(dx_zeta)
    scale = np.abs(dx_zeta).max()
    assert scale > 1e-3, "a gate on the crab dispersion needs a crab dispersion"
    for point, dxz, dpxz in zip(points, dx_zeta, dpx_zeta, strict=True):
        assert point.crab_dispersion[0] == pytest.approx(dxz, abs=CRAB_ATOL)
        assert point.crab_dispersion[1] == pytest.approx(dpxz, abs=0.1 * CRAB_ATOL)


def test_the_dynamic_dispersion_matches_xtrack_along_the_ring(bendy_pair) -> None:
    """``dx``/``dpx`` element by element -- O1 compared this at one point only.

    A free extra arbiter: the same longitudinal columns of ``W`` that give the crab
    dispersion give this, one index over, so agreeing here and disagreeing there would
    localise a fault to the projection rather than to the transport.
    """
    lattice, line = bendy_pair
    tw = line.twiss()
    points = propagate_normal_form(lattice, closed_normal_form(lattice, method="6d"))
    dx, dpx = _rows(tw, "dx", "dpx")
    for point, d, dp in zip(points, dx, dpx, strict=True):
        assert point.dispersion[0] == pytest.approx(d, abs=DISPERSION_ATOL)
        assert point.dispersion[1] == pytest.approx(dp, abs=0.1 * DISPERSION_ATOL)
