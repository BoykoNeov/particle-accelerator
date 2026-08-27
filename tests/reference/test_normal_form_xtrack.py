r"""Cross-check the linear normal form (O1) against xtrack's ``W_matrix``.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**Why this comparison is the sharp one.** ``M = W R W^-1`` and symplecticity together
leave one free rotation angle per plane (the analytic file demonstrates that with a
deliberately mis-phased ``W`` that passes both). The analytic gates pin those angles
against Stage 1's ``beta``/``alpha``; this file checks the *whole matrix*, entry by entry,
against an independent implementation of the same decomposition — including the entries no
accsim function reads.

**The pre-commitment, written into ``docs/ROADMAP.md`` -> O1 before this file was run.**
The two matrices agree to ``1e-12`` **absolute** on a non-radiating ring. Absolute, not
relative: ``xtrack/linear_normal_form.py`` ends with ``W[abs(W) < 1e-14] = 0``, so a
relative comparison against an entry xtrack has zeroed is a spurious failure with no
physics in it.

**It held on the transverse block and was missed on the longitudinal one — and the miss
is the more useful half.** The ``(x, px, y, py)`` block lands at ``9e-16``, four orders
inside the promise. The longitudinal columns floor at ``2.6e-11``, and the entire excess
is **one entry of xtrack's own one-turn matrix**: it obtains ``R56`` by symmetric finite
difference of its exact drift map, where ``zeta(delta)`` is both curved (an ``h^2``
truncation) and a difference of two nearly-equal path lengths (a cancellation round-off
going as ``1/h``). accsim's ``R56`` is that function's exact derivative, ``L/gamma0^2``,
derived symbolically back in Stage 0. The attribution is *gated*, not asserted: the
residual traces a U in the reference's step size with a minimum at ``ddelta = 1e-5``, and
its size tracks ``|R56_accsim - R56_xtrack|`` one for one at every step. A model
disagreement has no such minimum.

**Two conventions checked rather than assumed.**

  * xtrack writes ``W`` in ``(x, px, y, py, zeta, pzeta)`` where accsim's one-turn matrix
    is in ``delta``. The two linear maps coincide because ``dpzeta/ddelta = 1`` *exactly*
    at ``delta = 0`` (``pzeta = (E - E0)/(beta0^2 E0)`` and ``dE/ddelta = beta0 P0``), so
    no ``beta0^2`` enters anywhere. The analytic file asserts the identity; this file is
    where it would show up if it were wrong.
  * The mode labelling. xtrack's ``sort_modes`` tie-breaks on ``|v[5]|`` then ``|v[2]|``;
    accsim assigns modes to planes by maximising the total per-plane weight. The rules
    agree away from a coupling resonance, which is where the rings here sit.

**The entry-by-entry rings are deliberately bend-free.** xtrack builds its ``R`` matrix by
finite-differencing its *tracked* map, so on a ring with bends the two codes' one-turn
matrices differ by the bend model itself (the residual axis L and B2 already own) and the
comparison would be measuring that instead of ``W``. On a drift/quad ring xtrack's
tracking is exactly linear and the finite difference is exact.
:func:`test_dispersion_matches_xtrack_on_a_bendy_ring` then does use a bendy ring — for
the one quantity that needs dispersion to exist at all — at a tolerance that says so.
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
    closed_twiss,
    normal_form,
    synchrotron_tune,
)

MASS0 = 0.51099895069e6  # electron, eV
GAMMA0 = 5.0
KQ, N_CELLS, LQ, LDRIFT = 1.2, 4, 0.3, 0.7
#: Small enough that the bucket is stable on an 8 m ring at ``gamma0 = 5`` (the ring is
#: bend-free, so it is *below* transition and the stable stationary phase is ``0``).
VOLTAGE, HARMONIC = 1.0e5, 8

#: The pre-committed agreement between the two ``W`` matrices, entry by entry. It holds on
#: the transverse block (measured ``9e-16``); the longitudinal columns cannot reach it —
#: see :func:`test_the_6d_residual_is_the_references_own_r56`.
W_ATOL = 1e-12

#: What the longitudinal columns *can* reach, set by the reference's own finite-difference
#: error in ``R56`` and by nothing on accsim's side. Measured ``2.6e-11``.
LONGITUDINAL_ATOL = 1e-10

#: xtrack's default ``ddelta`` step is ``1e-6``, which for this ring sits on the
#: cancellation side of the ``R56`` U-curve; ``1e-5`` is its minimum. Chosen by scanning
#: the step, not by scanning the tolerance — the scan is itself a test below.
_BEST_STEPS = {
    "dx": 1e-6,
    "dpx": 1e-7,
    "dy": 1e-6,
    "dpy": 1e-7,
    "dzeta": 1e-6,
    "ddelta": 1e-5,
}


def _cells():
    """The same bend-free FODO ring in both codes."""
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


@pytest.fixture(scope="module")
def pair():
    """``(accsim lattice with cavity, xtrack line with cavity)`` — one JIT build.

    Every ``xt.Line`` compiles a fresh C kernel at ~12 s (CONVENTIONS.md -> *Test-suite
    cost*), so the 4D and 6D comparisons share one line and one lattice.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    acc, xtk = _cells()
    plain = Lattice(acc, ref)
    cavity = RFCavity.from_harmonic(VOLTAGE, HARMONIC, plain.length, ref, phi_s=0.0)
    lattice = Lattice([*acc, cavity], ref)

    line = xt.Line(elements=[*xtk, xt.Cavity(voltage=VOLTAGE, frequency=cavity.frequency)])
    # q0 = +1 is accsim's default charge, so the two RF phase conventions coincide and
    # ``phi_s = 0`` maps straight over (see rfcavity.py: for q < 0 they are negatives).
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1.0, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return lattice, line


def test_w_matrix_matches_xtrack_in_4d(pair) -> None:
    """The transverse block of ``W``, entry by entry, against xtrack's 4D twiss.

    xtrack's ``W_matrix`` is 6x6 even in ``4d`` (it splices in a dummy longitudinal
    rotation), so the comparison is over the ``(x, px, y, py)`` block it actually
    normalised.
    """
    lattice, line = pair
    tw = line.twiss(method="4d")
    mine = normal_form(lattice.one_turn_matrix(), method="4d").w
    theirs = np.asarray(tw.W_matrix[0])[:4, :4]
    assert np.allclose(mine, theirs, atol=W_ATOL, rtol=0.0), np.abs(mine - theirs).max()


def test_w_matrix_matches_xtrack_in_6d(pair) -> None:
    """The full 6x6, including the two rows written in ``pzeta`` rather than ``delta``.

    **The pre-commitment held on the transverse block and was missed on the longitudinal
    one, and the miss has a single named owner on the reference's side.** The
    ``(x, px, y, py)`` block agrees at ``9e-16`` — four orders inside the promised
    ``1e-12`` — while the longitudinal columns floor at ``3e-11``. The whole difference is
    one entry of xtrack's one-turn matrix, ``R56``: it computes it by symmetric finite
    difference of its *exact* drift map, whose ``zeta(delta)`` is curved, so the entry
    carries an ``h^2`` truncation above ``h ~ 1e-5`` and a cancellation round-off below it
    (``zeta`` is a difference of two nearly-equal path lengths). accsim's ``R56`` is that
    same function's exact derivative ``L/gamma0^2``, derived symbolically in
    ``tests/analytic/test_drift.py``. :func:`test_the_6d_residual_is_the_references_own_r56`
    gates that attribution rather than asserting it.
    """
    lattice, line = pair
    tw = line.twiss(steps_R_matrix=_BEST_STEPS)
    mine = normal_form(lattice.one_turn_matrix(), method="6d").w
    theirs = np.asarray(tw.W_matrix[0])
    assert np.allclose(mine[:4, :4], theirs[:4, :4], atol=W_ATOL, rtol=0.0)
    assert np.abs(mine - theirs).max() < LONGITUDINAL_ATOL


def test_the_6d_residual_is_the_references_own_r56(pair) -> None:
    """The longitudinal residual has a *minimum* in the reference's step size.

    A model disagreement would not: it would sit there while the step moved. This one
    traces a U — ``h^2`` truncation on one side, cancellation on the other — and its size
    tracks ``|R56_accsim - R56_xtrack|`` one for one at every step, which is what pins the
    residual on one entry of the reference's numerical differentiation rather than on
    either code's physics.
    """
    lattice, line = pair
    m6 = lattice.one_turn_matrix()
    mine = normal_form(m6, method="6d").w
    residual, r56_gap = {}, {}
    for ddelta in (1e-6, 1e-5, 1e-4):
        tw = line.twiss(steps_R_matrix=dict(_BEST_STEPS, ddelta=ddelta))
        residual[ddelta] = float(np.abs(mine - np.asarray(tw.W_matrix[0])).max())
        r56_gap[ddelta] = abs(float(np.asarray(tw.R_matrix)[4, 5]) - m6[4, 5])
    # a genuine U: the middle step is the best, by more than an order each way
    assert residual[1e-5] < residual[1e-6] / 10.0
    assert residual[1e-5] < residual[1e-4] / 10.0
    # ...and the residual *is* that one matrix entry, transported through W. Asserted as a
    # proportionality between steps rather than against a fixed factor: the factor is the
    # size of W's longitudinal columns and so belongs to this ring, while "the two move
    # together over two orders of magnitude" belongs to the claim.
    steps = sorted(residual)
    for a, b in zip(steps[:-1], steps[1:], strict=True):
        assert residual[a] / residual[b] == pytest.approx(r56_gap[a] / r56_gap[b], rel=0.15)
    # the transport factor is O(1) on any ring, which is the ring-independent half
    for ddelta, value in residual.items():
        assert 0.3 < value / r56_gap[ddelta] < 3.0


def test_mode_tunes_match_xtrack(pair) -> None:
    """All three eigen-tunes, and the synchrotron one is accsim's own ``Q_s``."""
    lattice, line = pair
    tw = line.twiss()
    nf = normal_form(lattice.one_turn_matrix(), method="6d")
    assert nf.tunes[0] == pytest.approx(tw.qx % 1.0, abs=1e-10)
    assert nf.tunes[1] == pytest.approx(tw.qy % 1.0, abs=1e-10)
    # xtrack reports the longitudinal mode as ``qs``, always the distance to the nearer
    # integer; accsim's normal form keeps the signed rotation, so compare the fraction.
    assert min(nf.tunes[2], 1.0 - nf.tunes[2]) == pytest.approx(tw.qs, abs=1e-10)
    assert min(nf.tunes[2], 1.0 - nf.tunes[2]) == pytest.approx(synchrotron_tune(lattice), rel=1e-6)


def test_mode_betas_match_xtracks_twiss(pair) -> None:
    """``betx``/``bety`` are entries of ``W``, in xtrack as here — so they must agree."""
    lattice, line = pair
    tw = line.twiss()
    nf = normal_form(lattice.one_turn_matrix(), method="6d")
    assert nf.mode_beta[0] == pytest.approx(float(tw.betx[0]), rel=1e-10)
    assert nf.mode_beta[1] == pytest.approx(float(tw.bety[0]), rel=1e-10)
    assert nf.mode_alpha[0] == pytest.approx(float(tw.alfx[0]), abs=1e-10)
    assert nf.mode_alpha[1] == pytest.approx(float(tw.alfy[0]), abs=1e-10)


def test_bend_free_6d_optics_is_the_4d_optics(pair) -> None:
    """No dispersion, so nothing couples the momentum oscillation into ``x``.

    The 6D-versus-4D departure the analytic file measures is a *dispersive* effect; on
    this ring it is absent in both codes, which is what makes the entry-by-entry
    comparison above a clean test of ``W`` rather than of the bend model.
    """
    lattice, line = pair
    nf6 = normal_form(lattice.one_turn_matrix(), method="6d")
    assert nf6.mode_beta[0] == pytest.approx(closed_twiss(lattice).beta_x, rel=1e-12)
    assert abs(float(line.twiss().dx[0])) < 1e-12


def test_dispersion_matches_xtrack_on_a_bendy_ring() -> None:
    """The dynamic dispersion read off ``W`` is what xtrack reports as ``dx``.

    Needs a ring with bends, so the two codes' one-turn matrices differ by the bend model
    itself — the residual axis L owns — and the tolerance is loose *and stated* rather
    than tuned. What is being checked is that accsim's :attr:`NormalForm.dispersion`
    means the same thing xtrack's ``dx`` does, on a ring where it is 24% away from the
    matched :class:`Twiss` dispersion. The tight number is the bend-free comparison above.
    """
    from test_closed_orbit_6d import ring  # noqa: PLC0415
    from test_closed_orbit_6d_xtrack import _line  # noqa: PLC0415

    lattice, _ = ring()
    line = _line()
    line["cav"].frequency = lattice.elements[-1].frequency
    tw = line.twiss()
    nf = normal_form(lattice.one_turn_matrix(), method="6d")
    assert nf.dispersion[0] == pytest.approx(float(tw.dx[0]), rel=2e-3)
    assert nf.dispersion[1] == pytest.approx(float(tw.dpx[0]), abs=2e-3)
    # ...and it is genuinely not the matched dispersion, in xtrack's number as in ours.
    assert nf.dispersion[0] / closed_twiss(lattice).disp_x == pytest.approx(1.239, abs=0.01)


def test_the_phase_convention_is_xtracks(pair) -> None:
    """Both codes put each mode's position component on the real axis: ``W[2i, 2i+1] = 0``.

    Stated as its own gate because it is the single choice the structural checks cannot
    see, and because it is the one the analytic file justifies against Stage 1 rather
    than against this file.
    """
    lattice, line = pair
    theirs = np.asarray(line.twiss().W_matrix[0])
    mine = normal_form(lattice.one_turn_matrix(), method="6d").w
    for i in range(3):
        assert abs(mine[2 * i, 2 * i + 1]) < 1e-14
        assert abs(theirs[2 * i, 2 * i + 1]) < 1e-14
        assert mine[2 * i, 2 * i] > 0.0
        assert theirs[2 * i, 2 * i] > 0.0


def test_normalised_coordinates_agree_with_xtracks(pair) -> None:
    """``W^-1 x`` for a real particle, against xtrack's own normalisation of the same one.

    xtrack divides by ``sqrt(nemitt/(beta0 gamma0))`` per mode; accsim does not (the
    emittance is deliberately not part of the matrix under test), so the comparison
    multiplies it back in. Anything wrong with a *column* of ``W`` shows up here as a
    wrong normalised coordinate even where the matrix comparison is dominated by other
    entries.
    """
    lattice, line = pair
    from accsim import to_normalized  # noqa: PLC0415

    nemitt = 1e-6
    state = np.array([1.3e-4, -2.1e-5, 0.7e-4, 1.1e-5, 1.0e-3, 2.0e-4])
    particles = line.build_particles(
        x=state[0], px=state[1], y=state[2], py=state[3], zeta=state[4], delta=state[5]
    )
    tw = line.twiss()
    norm = tw.get_normalized_coordinates(particles, nemitt_x=nemitt, nemitt_y=nemitt)
    beta0, gamma0 = float(line.particle_ref.beta0[0]), float(line.particle_ref.gamma0[0])
    scale = math.sqrt(nemitt / (beta0 * gamma0))
    mine = to_normalized(normal_form(lattice.one_turn_matrix(), method="6d"), state)
    assert float(norm.x_norm[0]) * scale == pytest.approx(mine[0], abs=1e-12)
    assert float(norm.px_norm[0]) * scale == pytest.approx(mine[1], abs=1e-12)
    assert float(norm.y_norm[0]) * scale == pytest.approx(mine[2], abs=1e-12)
    assert float(norm.py_norm[0]) * scale == pytest.approx(mine[3], abs=1e-12)
